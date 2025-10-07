#include <assert.h>
#include <iostream>
#include <numeric>
#include <stdio.h>
#include <tuple>
#include <vector>

// helpers {{{
#ifndef __HELPERS_H__
#define __HELPERS_H__

static void HandleError(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

#define HANDLE_NULL(a)                                                         \
    {                                                                          \
        if (a == NULL) {                                                       \
            printf("Host memory failed in %s at line %d\n", __FILE__,          \
                   __LINE__);                                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }

template <typename T> void swap(T &a, T &b)
{
    T t = a;
    a = b;
    b = t;
}

void *big_random_block(int size)
{
    unsigned char *data = (unsigned char *)malloc(size);
    HANDLE_NULL(data);
    for (int i = 0; i < size; i++)
        data[i] = rand();

    return data;
}

int *big_random_block_int(int size)
{
    int *data = (int *)malloc(size * sizeof(int));
    HANDLE_NULL(data);
    for (int i = 0; i < size; i++)
        data[i] = rand();

    return data;
}

// a place for common kernels - starts here

__device__ unsigned char value(float n1, float n2, int hue)
{
    if (hue > 360)
        hue -= 360;
    else if (hue < 0)
        hue += 360;

    if (hue < 60)
        return (unsigned char)(255 * (n1 + (n2 - n1) * hue / 60));
    if (hue < 180)
        return (unsigned char)(255 * n2);
    if (hue < 240)
        return (unsigned char)(255 * (n1 + (n2 - n1) * (240 - hue) / 60));
    return (unsigned char)(255 * n1);
}

__global__ void float_to_color(unsigned char *optr, const float *outSrc)
{
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float l = outSrc[offset];
    float s = 1;
    int h = (180 + (int)(360.0f * outSrc[offset])) % 360;
    float m1, m2;

    if (l <= 0.5f)
        m2 = l * (1 + s);
    else
        m2 = l + s - l * s;
    m1 = 2 * l - m2;

    optr[offset * 4 + 0] = value(m1, m2, h + 120);
    optr[offset * 4 + 1] = value(m1, m2, h);
    optr[offset * 4 + 2] = value(m1, m2, h - 120);
    optr[offset * 4 + 3] = 255;
}

__global__ void float_to_color(uchar4 *optr, const float *outSrc)
{
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float l = outSrc[offset];
    float s = 1;
    int h = (180 + (int)(360.0f * outSrc[offset])) % 360;
    float m1, m2;

    if (l <= 0.5f)
        m2 = l * (1 + s);
    else
        m2 = l + s - l * s;
    m1 = 2 * l - m2;

    optr[offset].x = value(m1, m2, h + 120);
    optr[offset].y = value(m1, m2, h);
    optr[offset].z = value(m1, m2, h - 120);
    optr[offset].w = 255;
}

#if _WIN32
// Windows threads.
#include <windows.h>

typedef HANDLE CUTThread;
typedef unsigned(WINAPI *CUT_THREADROUTINE)(void *);

#define CUT_THREADPROC unsigned WINAPI
#define CUT_THREADEND  return 0

#else
// POSIX threads.
#include <pthread.h>

typedef pthread_t CUTThread;
typedef void *(*CUT_THREADROUTINE)(void *);

#define CUT_THREADPROC void
#define CUT_THREADEND
#endif

// Create thread.
CUTThread start_thread(CUT_THREADROUTINE, void *data);

// Wait for thread to finish.
void end_thread(CUTThread thread);

// Destroy thread.
void destroy_thread(CUTThread thread);

// Wait for multiple threads.
void wait_for_threads(const CUTThread *threads, int num);

#if _WIN32
// Create thread
CUTThread start_thread(CUT_THREADROUTINE func, void *data)
{
    return CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)func, data, 0, NULL);
}

// Wait for thread to finish
void end_thread(CUTThread thread)
{
    WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
}

// Destroy thread
void destroy_thread(CUTThread thread)
{
    TerminateThread(thread, 0);
    CloseHandle(thread);
}

// Wait for multiple threads
void wait_for_threads(const CUTThread *threads, int num)
{
    WaitForMultipleObjects(num, threads, true, INFINITE);

    for (int i = 0; i < num; i++)
        CloseHandle(threads[i]);
}

#else
// Create thread
CUTThread start_thread(CUT_THREADROUTINE func, void *data)
{
    pthread_t thread;
    pthread_create(&thread, NULL, func, data);
    return thread;
}

// Wait for thread to finish
void end_thread(CUTThread thread) { pthread_join(thread, NULL); }

// Destroy thread
void destroy_thread(CUTThread thread) { pthread_cancel(thread); }

// Wait for multiple threads
void wait_for_threads(const CUTThread *threads, int num)
{
    for (int i = 0; i < num; i++)
        end_thread(threads[i]);
}

#endif

#endif // __HELPERS_H__
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

#define N              (1024 * 1024)
#define FULL_DATA_SIZE (N * 20)

__global__ void kernel_v1(int *a, int *b, int *c)
{
    int tid = threadIdx.x + blockIdx.x + blockDim.x;
    if (tid < N) {
        int tid1 = (tid + 1) % 256;
        int tid2 = (tid + 2) % 256;
        float aSum = (a[tid] + a[tid1] + a[tid2]) / 3.0f;
        float bSum = (b[tid] + b[tid1] + b[tid2]) / 3.0f;
        c[tid] = (aSum + bSum) / 2;
    }
}

constexpr bool enable_v1 = true;
constexpr bool enable_v2 = true;

void run_v1(int *h_a, int *h_b, int *h_c, cuda_timer &timer)
{
    cuda_array<int> d_a(N), d_b(N), d_c(N);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    timer.start();
    for (int i = 0; i < FULL_DATA_SIZE; i += N) {
        HANDLE_ERROR(cudaMemcpyAsync(d_a, h_a + i, N * sizeof(int),
                                     cudaMemcpyHostToDevice, stream));
        HANDLE_ERROR(cudaMemcpyAsync(d_b, h_b + i, N * sizeof(int),
                                     cudaMemcpyHostToDevice, stream));

        kernel_v1<<<N / 256, 256, 0, stream>>>(d_a, d_b, d_c);

        HANDLE_ERROR(cudaMemcpyAsync(h_c + i, d_c, N * sizeof(int),
                                     cudaMemcpyDeviceToHost, stream));
    }

    HANDLE_ERROR(cudaStreamSynchronize(stream));

    timer.stop();
    HANDLE_ERROR(cudaStreamDestroy(stream));
}

void run_v2(int *h_a, int *h_b, int *h_c, cuda_timer &timer)
{
    cuda_array<int> d_a[2] = {cuda_array<int>(N), cuda_array<int>(N)};
    cuda_array<int> d_b[2] = {cuda_array<int>(N), cuda_array<int>(N)};
    cuda_array<int> d_c[2] = {cuda_array<int>(N), cuda_array<int>(N)};

    cudaStream_t stream[2];
    cudaStreamCreate(stream + 0);
    cudaStreamCreate(stream + 1);

    timer.start();
    for (int i = 0, s = 0; i < FULL_DATA_SIZE; i += N, s ^= 1) {
        HANDLE_ERROR(cudaMemcpyAsync(d_a[s], h_a + i, N * sizeof(int),
                                     cudaMemcpyHostToDevice, stream[s]));
        HANDLE_ERROR(cudaMemcpyAsync(d_b[s], h_b + i, N * sizeof(int),
                                     cudaMemcpyHostToDevice, stream[s]));

        kernel_v1<<<N / 256, 256, 0, stream[s]>>>(d_a[s], d_b[s], d_c[s]);

        HANDLE_ERROR(cudaMemcpyAsync(h_c + i, d_c[s], N * sizeof(int),
                                     cudaMemcpyDeviceToHost, stream[s]));
    }

    HANDLE_ERROR(cudaStreamSynchronize(stream[0]));
    HANDLE_ERROR(cudaStreamSynchronize(stream[1]));

    timer.stop();
    HANDLE_ERROR(cudaStreamDestroy(stream[0]));
    HANDLE_ERROR(cudaStreamDestroy(stream[1]));
}

int main(void)
{
    cuda_timer timer_v1, timer_v2;
    int *a, *b, *c1, *c2;

    HANDLE_ERROR(cudaHostAlloc((void **)&a, FULL_DATA_SIZE * sizeof(int),
                               cudaHostAllocDefault));
    HANDLE_ERROR(cudaHostAlloc((void **)&b, FULL_DATA_SIZE * sizeof(int),
                               cudaHostAllocDefault));
    HANDLE_ERROR(cudaHostAlloc((void **)&c1, FULL_DATA_SIZE * sizeof(int),
                               cudaHostAllocDefault));
    HANDLE_ERROR(cudaHostAlloc((void **)&c2, FULL_DATA_SIZE * sizeof(int),
                               cudaHostAllocDefault));

    for (int i = 0; i < FULL_DATA_SIZE; i++) {
        a[i] = rand();
        b[i] = rand();
    }

    if (enable_v1)
        run_v1(a, b, c1, timer_v1);
    if (enable_v2)
        run_v2(a, b, c2, timer_v2);

    if (enable_v1 && enable_v2) {
        assert(!memcmp(c1, c2, FULL_DATA_SIZE * sizeof(int)));
        timer_v1.print("v1");
        timer_v2.print("v2", timer_v1);
    } else if (enable_v1) {
        timer_v1.print("v1");
    } else if (enable_v2) {
        timer_v2.print("v2");
    }

    HANDLE_ERROR(cudaFreeHost(a));
    HANDLE_ERROR(cudaFreeHost(b));
    HANDLE_ERROR(cudaFreeHost(c1));
    HANDLE_ERROR(cudaFreeHost(c2));

    return 0;
}
