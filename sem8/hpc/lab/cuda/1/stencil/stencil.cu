#include <cuda.h>
#include <iostream>
#include <numeric>
#include <vector>

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

template <typename... Args> class cuda_kernel
{
  public:
    cuda_kernel(void (*kernel)(Args...), dim3 grid, dim3 block, size_t shm = 0)
        : _kernel(kernel), _grid(grid), _block(block), _shm(shm)
    {
    }

    void operator()(Args... args)
    {
        if (_shm > 0)
            _kernel<<<_grid, _block, _shm>>>(args...);
        else
            _kernel<<<_grid, _block>>>(args...);
    }

  private:
    void (*_kernel)(Args...);
    dim3 _grid;
    dim3 _block;
    size_t _shm;
    // TODO: add stream
};

#include <random>
#include <stdio.h>
#include <time.h>

static void handleError(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

#define cudaCheck(err) (handleError(err, __FILE__, __LINE__))

template <int N, int R> __global__ void stencil_1d_shared(int *x, int *y)
{
    extern __shared__ int s[];

    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int stride = gridDim.x * blockDim.x;

    int g = tx + bx * blockDim.x;

    int is_right = tx > blockDim.x / 2;
    int offset = R + blockDim.x;
    int my_share = R + (blockDim.x / 2 - 1) / (blockDim.x / 2);

    for (; g < N + blockDim.x; g += stride) {
        // [R..R+blockDim.x-1]
        if (0 <= g && g < N)
            s[tx + R] = x[g];
        else
            s[tx + R] = 0;

        // [0..R-1] [R+blockDim.x..R+blockDim.x+R-1]
        for (int i = 0; i < my_share; i++) {
            int j = offset * is_right + i * (blockDim.x / 2);
            int ix = j + tx;
            int is = j + g - R;
            if (ix < 2 * R + blockDim.x) {
                if (0 <= is && is < N)
                    s[ix] = x[is];
                else
                    s[ix] = 0;
            }
        }

        __syncthreads();

        int sum = 0;
        for (int i = 0; i <= 2 * R; i++)
            sum += s[i + tx];
        y[g] = sum;

        __syncthreads();
    }
}

template <int N, int R> __global__ void stencil_1d(int *x, int *y)
{
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int stride = gridDim.x * blockDim.x;

    int g = bx * blockDim.x + tx;
    for (; g < N; g += stride) {
        int sum = 0;
        int l = g - R;
        int r = g + R;
        if (l < 0)
            l = 0;
        if (r > N)
            r = N;
        for (int i = l; i < r; i++)
            sum += x[i];
        y[g] = sum;
    }
}

template <int N, int R> void cpu_stencil_1d(int *x, int *y)
{
    for (int i = 0; i < N; i++) {
        int sum = 0;
        int l = std::max(i - R, 0);
        int r = std::min(i + R, N - 1);
        for (int j = l; j <= r; j++)
            sum += x[j];
        y[i] = sum;
    }
}

template <int BLOCK_SIZE, int N, int R> bool check(void)
{
    constexpr size_t size = N * sizeof(int);
    // PUT YOUR CODE HERE - INPUT AND OUTPUT ARRAYS
    int *h_x, *h_y;

    if (!(h_x = (int *)malloc(size)) || !(h_y = (int *)malloc(size)))
        return false;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, 10);
    for (int i = 0; i < N; i++)
        h_x[i] = dis(gen);

    cuda_timer t;
    cuda_kernel stencil(stencil_1d<N, R>, 65536 / BLOCK_SIZE, BLOCK_SIZE,
                        (BLOCK_SIZE + 2 * R) * sizeof(int));
    for (int i = 0; i < 10; i++) {
        t.start();

        // PUT YOUR CODE HERE - DEVICE MEMORY ALLOCATION
        cuda_array<int> d_x(h_x, N), d_y(N);

        // PUT YOUR CODE HERE - KERNEL EXECUTION
        stencil(d_x, d_y);
        cudaCheck(cudaDeviceSynchronize());
        cudaCheck(cudaPeekAtLastError());

        // PUT YOUR CODE HERE - COPY RESULT FROM DEVICE TO HOST
        cudaCheck(cudaMemcpy(h_y, d_y, size, cudaMemcpyDeviceToHost));

        t.stop();
    }
    t.print("GPU dumb");

    struct timespec cpu_start, cpu_stop;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu_start);

    cpu_stencil_1d<N, R>(h_x, h_y);

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu_stop);
    double result = (cpu_stop.tv_sec - cpu_start.tv_sec) * 1e3 +
                    (cpu_stop.tv_nsec - cpu_start.tv_nsec) / 1e6;
    printf("CPU execution time:  %3.1f ms\n", result);

    free(h_x);
    free(h_y);

    return true;
}

int main()
{
    check<1024, 1000, 3>();
    check<1024, 1000, 30>();
    check<1024, 1000, 300>();
    check<1024, 1000, 3000>();

    check<1024, 1000000, 3>();
    check<1024, 1000000, 30>();
    check<1024, 1000000, 300>();
    check<1024, 1000000, 3000>();

#if 0
    check<1024, 1000000000, 3>();
    check<1024, 1000000000, 30>();
    check<1024, 1000000000, 300>();
    check<1024, 1000000000, 3000>();
#endif

    return 0;
}
