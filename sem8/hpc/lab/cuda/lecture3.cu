#include <cassert>
#include <iostream>
#include <numeric>
#include <random>
#include <tuple>
#include <vector>

#include <cuda.h>

// {{{
template <typename T> class cuda_array
{
  public:
    cuda_array(size_t size) : _size(size)
    {
        cudaMalloc(&_data, size * sizeof(T));
    }

    cuda_array(const T *h_data, size_t size) : _size(size)
    {
        cudaMalloc(&_data, size * sizeof(T));
        cudaMemcpy(_data, h_data, size * sizeof(T), cudaMemcpyHostToDevice);
    }

    ~cuda_array() { cudaFree(_data); }

    operator T *() { return _data; }

  private:
    T *_data;
    size_t _size;
};

class cuda_timer
{
  public:
    cuda_timer()
    {
        cudaEventCreate(&_start);
        cudaEventCreate(&_stop);
    }

    void start() { cudaEventRecord(_start, 0); }

    float stop()
    {
        cudaEventRecord(_stop, 0);
        cudaEventSynchronize(_stop);

        float ms;
        cudaEventElapsedTime(&ms, _start, _stop);

        _times.emplace_back(ms);

        return ms;
    }

    float mean() const
    {
        return std::accumulate(_times.begin(), _times.end(), 0.0) /
               _times.size();
    }

    float std() const
    {
        float mean = this->mean();
        float var = 0;
        for (const auto &ms : _times)
            var += (ms - mean) * (ms - mean);
        var /= _times.size();

        return sqrt(var);
    }

    std::vector<float> get_times() { return _times; }
    void clear() { _times.clear(); }

    void print(const std::string &name)
    {
        float mean = this->mean();
        float std = this->std();
        std::cout << name << " | " << format_float(mean) << "+/-"
                  << format_float(std) << " | " << " -- "
                  << " |" << std::endl;
    }

    void print(const std::string &name, const cuda_timer &reference)
    {
        float mean = this->mean();
        float std = this->std();
        float mean_ref = reference.mean();
        float std_ref = reference.std();
        std::cout << name << " | " << format_float(mean) << "+/-"
                  << format_float(std) << " | " << format_float(mean_ref / mean)
                  << "+/-"
                  << format_float(
                         sqrt(pow(mean * std_ref, 2) + pow(std * mean, 2)) /
                         pow(mean, 2))
                  << " | " << std::endl;
    }

    ~cuda_timer()
    {
        cudaEventDestroy(_start);
        cudaEventDestroy(_stop);
    }

  private:
    cudaEvent_t _start;
    cudaEvent_t _stop;
    std::vector<float> _times;

    std::string format_float(float ms) const
    {
        std::string str = "";
        str += std::to_string(int(ms)) + ".";
        ms -= int(ms);
        if (ms == 0)
            return str + "0";
        while ((int)ms == 0)
            ms *= 10;
        return str + std::to_string(static_cast<int>(std::round(ms * 10)));
    }
};

template <typename... Args> struct cuda_bench {
    std::string name;
    void (*f)(Args...);
    int runs;
    std::tuple<Args...> args;

    cuda_bench(std::string name, void (*f)(Args...), int runs = 5)
        : name(name), f(f), runs(runs)
    {
    }

    void set_args(Args... args) { this->args = std::tuple<Args...>(args...); }

    template <typename... Bench> void bench(Bench... args)
    {
        cuda_timer reference;
        for (int i = 0; i < runs; i++) {
            reference.start();
            std::apply(f, this->args);
            reference.stop();
        }
        reference.print(name);

        for (auto &b : {args...}) {
            cuda_timer timer;
            for (int i = 0; i < b.runs; i++) {
                timer.start();
                std::apply(b.f, b.args);
                timer.stop();
            }
            timer.print(b.name, reference);
        }
    }
};
// }}}

// reduce {{{
template <typename T> __device__ void gpu_reduce_warp(volatile T *t, int tx)
{
#pragma unroll
    for (unsigned s = 32; s > 0; s >>= 1)
        t[tx] += t[tx + s];
}

template <typename T>
__global__ void gpu_reduce_blocks(size_t N, T const *x, T *y)
{
    unsigned tx = threadIdx.x;
    unsigned bx = blockIdx.x;
    unsigned i = bx * blockDim.x * 2 + tx;
    unsigned grid_size = blockDim.x * 2 * gridDim.x;

    extern __shared__ T t[];

    t[tx] = 0;
    if (i < N)
        t[tx] += x[i];
    if (i + blockDim.x < N)
        t[tx] += x[i + blockDim.x];
    __syncthreads();

    for (unsigned s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tx < s)
            t[tx] += t[tx + s];
        __syncthreads();
    }

    if (tx < 32)
        gpu_reduce_warp<T>(t, tx);

    if (tx == 0)
        y[bx] = t[0];
}
template <typename T> void gpu_reduce(size_t N, const T *x, T *y)
{
    int block_size = 1024;
    int grid_size = ((N + block_size - 1) / block_size) / 2;
    cuda_array<int> d_x(N), d_y(grid_size);

    cudaMemcpy(d_x, x, N * sizeof(T), cudaMemcpyDeviceToDevice);
    for (int M = N; M;) {
        gpu_reduce_blocks<int>
            <<<grid_size, block_size, block_size * sizeof(T)>>>(M, d_x, d_y);
        if (grid_size <= 1)
            break;
        cudaMemcpy(d_x, d_y, grid_size * sizeof(int), cudaMemcpyDeviceToDevice);
        M = grid_size;
        grid_size = (M + block_size - 1) / block_size;
    }
    cudaMemcpy(y, d_y, sizeof(T), cudaMemcpyDeviceToDevice);
}

template <typename T> void cpu_reduce(size_t N, const T *x, T *y)
{
    T res = 0;
    for (size_t i = 0; i < N; i++)
        res += x[i];
    *y = res;
}
// }}}
// scan {{{
template <typename T> __global__ void gpu_scan_add(size_t N, const T *x, T *y)
{
    size_t tx = threadIdx.x;
    size_t bx = blockIdx.x;
    size_t i = bx * blockDim.x + tx;

    if (i < N)
        y[i] += x[bx];
}

template <typename T>
__global__ void gpu_scan_blocks(size_t N, const T *x, T *y, T *b)
{
    size_t tx = threadIdx.x;
    size_t bx = blockIdx.x;
    size_t i = bx * blockDim.x + tx;

    extern __shared__ T t[];

    t[tx] = i < N ? x[i] : 0;
    __syncthreads();

    int d = 0, k = 2;
    for (; k <= blockDim.x; d++, k <<= 1) {
        if (tx < blockDim.x >> (d + 1)) {
            int ai = tx * k + k - 1;
            int bi = tx * k + (k >> 1) - 1;
            t[ai] += t[bi];
        }
        __syncthreads();
    }

    if (tx == blockDim.x - 1) {
        b[bx] = t[tx];
        t[tx] = 0;
    }

    for (d = d - 1, k = k >> 1; d >= 0; d--, k >>= 1) {
        if (tx < blockDim.x >> (d + 1)) {
            int ai = tx * k + k - 1;
            int bi = tx * k + (k >> 1) - 1;
            int tmp = t[bi];
            t[bi] = t[ai];
            t[ai] += tmp;
        }
        __syncthreads();
    }

    if (i < N)
        y[i] = t[tx];
}

template <typename T> void gpu_scan(size_t N, const T *x, T *y)
{
    int block_size = 1024;
    int grid_size = (N + block_size - 1) / block_size;
    cuda_array<T> block_sum(grid_size), block_pref(grid_size);
    gpu_scan_blocks<T><<<grid_size, block_size, 2 * block_size * sizeof(T)>>>(
        N, x, y, block_sum);
    cudaError(cudaDeviceSynchronize());
    if (grid_size > 1) {
        gpu_scan<T>(grid_size, block_sum, block_pref);
        gpu_scan_add<T><<<grid_size, block_size>>>(N, block_pref, y);
        cudaDeviceSynchronize();
    }
}

template <typename T> void cpu_scan(size_t N, const T *x, T *y)
{
    y[0] = 0;
    for (size_t i = 1; i < N; i++)
        y[i] = y[i - 1] + x[i - 1];
}
// }}}

auto random_vector(int N)
{
    std::vector<int> v(N);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, 10);
    for (int i = 0; i < N; i++)
        v[i] = dis(gen);

    return v;
}

int main(void)
{
    int N = 1 << 22;
    auto v = random_vector(N);
    int *h_x = v.data(), h_y[N];
    cuda_array<int> d_x(h_x, N), d_y(N);

    {
        cuda_bench b_cpu("CPU reduce", cpu_reduce<int>, 5);
        cuda_bench b_gpu("GPU reduce", gpu_reduce<int>, 20);
        b_cpu.set_args(N, h_x, h_y);
        b_gpu.set_args(N, d_x, d_y);
        b_cpu.bench(b_gpu);

        int cpu_res[1], gpu_res[1];
        cudaMemcpy(cpu_res, h_y, 1 * sizeof(int), cudaMemcpyHostToHost);
        cudaMemcpy(gpu_res, d_y, 1 * sizeof(int), cudaMemcpyDeviceToHost);
        assert(!memcmp(cpu_res, gpu_res, 1 * sizeof(int)));
    }

    {
        cuda_bench b_cpu("CPU scan", cpu_scan<int>, 5);
        cuda_bench b_gpu("GPU scan", gpu_scan<int>, 20);
        b_cpu.set_args(N, h_x, h_y);
        b_gpu.set_args(N, d_x, d_y);
        b_cpu.bench(b_gpu);

        int cpu_res[N], gpu_res[N];
        cudaMemcpy(cpu_res, h_y, N * sizeof(int), cudaMemcpyHostToHost);
        cudaMemcpy(gpu_res, d_y, N * sizeof(int), cudaMemcpyDeviceToHost);
        assert(!memcmp(cpu_res, gpu_res, N * sizeof(int)));
    }

    return 0;
}
