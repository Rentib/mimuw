#include <cassert>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <tuple>
#include <vector>

// errors {{{
#ifndef __ERRORS_H__
#define __ERRORS_H__
#include <stdio.h>

static void HandleError(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

#endif // __ERRORS_H__

// }}}
// utils {{{
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

#define TILE_DIM   32
#define BLOCK_ROWS 8

constexpr int nx = 8192;
constexpr int ny = 8192;

__global__ void transpose_v1(float *go_data, const float *gi_data)
{
    // (x, y) are coordinates inside the global matrix
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockDim.x * blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < nx && y + j < ny)
            go_data[x * width + (y + j)] = gi_data[(y + j) * width + x];
    }
}

int main()
{
    const int mem_size = nx * ny * sizeof(float);

    dim3 dimGrid(nx / TILE_DIM, ny / TILE_DIM, 1);
    // Each block is responsible for transposing one TILE_DIM x TILE_DIM
    // submatrix
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);

    if ((nx % TILE_DIM) | (ny % TILE_DIM)) {
        printf("nx and ny must be a multiple of TILE_DIM\n");
        return 1;
    }

    if (TILE_DIM % BLOCK_ROWS) {
        printf("TILE_DIM must be a multiple of BLOCK_ROWS\n");
        return 1;
    }

    float *host_input = (float *)malloc(mem_size),
          *host_correct = (float *)malloc(mem_size),
          *host_output = (float *)malloc(mem_size);
    float *dev_input, *dev_output;

    HANDLE_ERROR(cudaMalloc(&dev_input, mem_size));
    HANDLE_ERROR(cudaMalloc(&dev_output, mem_size));

    for (int j = 0; j < ny; j++)
        for (int i = 0; i < nx; i++)
            host_input[j * nx + i] = host_correct[i * nx + j] =
                (j + 1) * nx + (i + 1);

    for (int j = 0; j < ny; j++)
        for (int i = 0; i < nx; i++)
            assert(host_input[j * nx + i] != 0);

    HANDLE_ERROR(
        cudaMemcpy(dev_input, host_input, mem_size, cudaMemcpyHostToDevice));

    cuda_timer timer;
    timer.start();
    transpose_v1<<<dimGrid, dimBlock>>>(dev_output, dev_input);
    printf("error: %s\n", cudaGetErrorString(cudaGetLastError()));
    timer.stop();

    timer.print("v1");

    HANDLE_ERROR(
        cudaMemcpy(host_output, dev_output, mem_size, cudaMemcpyDeviceToHost));

    for (int j = 0; j < ny; j++)
        for (int i = 0; i < nx; i++)
            if (host_output[j * nx + i] != host_correct[j * nx + i]) {
                printf("Wrong value at (%d, %d), got %f, expected %f\n", i, j,
                       host_output[j * nx + i], host_correct[j * nx + i]);
                return 1;
            }
}
