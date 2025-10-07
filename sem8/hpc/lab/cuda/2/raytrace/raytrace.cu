#include <cassert>
#include <iostream>
#include <numeric>
#include <tuple>
#include <vector>

// bitmap {{{
#ifndef __CPU_BITMAP_H__
#define __CPU_BITMAP_H__

#include <fstream>

struct CPUBitmap {
    unsigned char *pixels;
    int x, y;
    void *dataBlock;
    void (*bitmapExit)(void *);

    CPUBitmap(int width, int height, void *d = NULL)
    {
        pixels = new unsigned char[width * height * 4];
        x = width;
        y = height;
        dataBlock = d;
    }

    ~CPUBitmap() { delete[] pixels; }

    unsigned char *get_ptr(void) const { return pixels; }
    long image_size(void) const { return x * y * 4; }

    // dumps bitmap in PPM (no alpha channel)
    void dump_ppm(const char *fname)
    {
        std::ofstream out(fname);
        out << "P3" << std::endl;
        out << x << " " << y << " 255" << std::endl;
        for (int i = 0; i < x; i++) {
            for (int j = 0; j < y; j++)
                out << (int)pixels[4 * (i * y + j) + 0] << " "
                    << (int)pixels[4 * (i * y + j) + 1] << " "
                    << (int)pixels[4 * (i * y + j) + 2] << " ";
            out << std::endl;
        }
    }
};

#endif // __CPU_BITMAP_H__
// }}}
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

#define DIM     (1 << 10)
#define rnd(x)  (x * rand() / RAND_MAX)
#define INF     2e10f
#define SPHERES 1000

struct Sphere {
    float red, green, blue;
    float radius;
    float x, y, z;

    __device__ float hit(float bitmapX, float bitmapY, float *colorFalloff)
    {
        float distX = bitmapX - x;
        float distY = bitmapY - y;

        if (distX * distX + distY * distY < radius * radius) {
            float distZ =
                sqrtf(radius * radius - distX * distX - distY * distY);
            *colorFalloff = distZ / sqrtf(radius * radius);
            return distZ + z;
        }

        return -INF;
    }
};

__global__ void kernel_v1(Sphere *spheres, unsigned char *bitmap)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float bitmapX = (x - DIM / 2);
    float bitmapY = (y - DIM / 2);

    float red = 0, green = 0, blue = 0;
    float maxDepth = -INF;

    for (int i = 0; i < SPHERES; i++) {
        float colorFalloff;
        float depth = spheres[i].hit(bitmapX, bitmapY, &colorFalloff);

        if (depth > maxDepth) {
            red = spheres[i].red * colorFalloff;
            green = spheres[i].green * colorFalloff;
            blue = spheres[i].blue * colorFalloff;
            maxDepth = depth;
        }
    }

    bitmap[offset * 4 + 0] = (int)(red * 255);
    bitmap[offset * 4 + 1] = (int)(green * 255);
    bitmap[offset * 4 + 2] = (int)(blue * 255);
    bitmap[offset * 4 + 3] = 255;
}

__global__ void kernel_v2(Sphere *g_spheres, unsigned char *bitmap)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float bitmapX = (x - DIM / 2);
    float bitmapY = (y - DIM / 2);

    float red = 0, green = 0, blue = 0;
    float maxDepth = -INF;

    __shared__ Sphere spheres[SPHERES];
    int n_threads = blockDim.x * blockDim.y;
    int spt = (SPHERES + n_threads - 1) / n_threads;
    for (int j = 0; j < spt; j++) {
        int i = (threadIdx.x + threadIdx.y * blockDim.x) * spt + j;
        if (i < SPHERES)
            spheres[i] = g_spheres[i];
    }
    __syncthreads();

    for (int i = 0; i < SPHERES; i++) {
        float colorFalloff;
        float depth = spheres[i].hit(bitmapX, bitmapY, &colorFalloff);

        if (depth > maxDepth) {
            red = spheres[i].red * colorFalloff;
            green = spheres[i].green * colorFalloff;
            blue = spheres[i].blue * colorFalloff;
            maxDepth = depth;
        }
    }

    bitmap[offset * 4 + 0] = (int)(red * 255);
    bitmap[offset * 4 + 1] = (int)(green * 255);
    bitmap[offset * 4 + 2] = (int)(blue * 255);
    bitmap[offset * 4 + 3] = 255;
}

struct DataBlock {
    unsigned char *hostBitmap;
    Sphere *spheres;
};

void v1_launcher(Sphere *spheres, unsigned char *bitmap)
{
    dim3 grid_size(DIM / 16, DIM / 16);
    dim3 block_size(16, 16);
    kernel_v1<<<grid_size, block_size>>>(spheres, bitmap);
}

void v2_launcher(Sphere *spheres, unsigned char *bitmap)
{
    dim3 grid_size(DIM / 16, DIM / 16);
    dim3 block_size(16, 16);
    kernel_v2<<<grid_size, block_size>>>(spheres, bitmap);
}

int main(void)
{
    DataBlock data;

    CPUBitmap bitmap_v1(DIM, DIM, &data), bitmap_v2(DIM, DIM, &data);
    unsigned char *devBitmap;
    Sphere *devSpheres;

    HANDLE_ERROR(cudaMalloc((void **)&devBitmap, bitmap_v1.image_size()));
    HANDLE_ERROR(cudaMalloc((void **)&devSpheres, sizeof(Sphere) * SPHERES));

    Sphere *hostSpheres = (Sphere *)malloc(sizeof(Sphere) * SPHERES);

    for (int i = 0; i < SPHERES; i++) {
        hostSpheres[i].red = rnd(1.0f);
        hostSpheres[i].green = rnd(1.0f);
        hostSpheres[i].blue = rnd(1.0f);
        hostSpheres[i].x = rnd(1000.0f) - 500;
        hostSpheres[i].y = rnd(1000.0f) - 500;
        hostSpheres[i].z = rnd(1000.0f) - 500;
        hostSpheres[i].radius = rnd(100.0f) + 20;
    }

    HANDLE_ERROR(cudaMemcpy(devSpheres, hostSpheres, sizeof(Sphere) * SPHERES,
                            cudaMemcpyHostToDevice));
    free(hostSpheres);

    {
        dim3 grid_size(DIM / 16, DIM / 16);
        dim3 block_size(16, 16);
        kernel_v1<<<grid_size, block_size>>>(devSpheres, devBitmap);
        HANDLE_ERROR(cudaMemcpy(bitmap_v1.get_ptr(), devBitmap,
                                bitmap_v1.image_size(),
                                cudaMemcpyDeviceToHost));
    }

    {
        dim3 grid_size(DIM / 16, DIM / 16);
        dim3 block_size(16, 16);
        kernel_v2<<<grid_size, block_size>>>(devSpheres, devBitmap);
        HANDLE_ERROR(cudaMemcpy(bitmap_v2.get_ptr(), devBitmap,
                                bitmap_v2.image_size(),
                                cudaMemcpyDeviceToHost));
    }

    assert(!memcmp(bitmap_v1.get_ptr(), bitmap_v2.get_ptr(),
                   bitmap_v1.image_size()));

    {
        cuda_bench b_v1("v1", v1_launcher, 50);
        cuda_bench b_v2("v2", v2_launcher, 50);
        b_v1.set_args(devSpheres, devBitmap);
        b_v2.set_args(devSpheres, devBitmap);
        b_v1.bench(b_v2);
    }

    bitmap_v2.dump_ppm("image.ppm");

    cudaFree(devBitmap);
    cudaFree(devSpheres);
}
