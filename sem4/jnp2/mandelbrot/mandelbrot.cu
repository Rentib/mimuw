#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <vector>

using namespace std;

void die(const char *fmt, ...)
{
  va_list ap;

  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  fputc('\n', stderr);

  exit(1);
}

void computeMandelbrotCPU(int *image, int width, int height, int iterations,
                          double x0, double y0, double x1, double y1)
{
  double dx = (x1 - x0) / width;
  double dy = (y1 - y0) / height;
  int row, column, iteration = 0;
  double zx, zy, tmpx, x, y;

  for (int pixel = 0; pixel < width * height; ++pixel, iteration = 0) {
    row = pixel / width;
    column = pixel % width;

    x = column * dx + x0, y = row * dy + y0;
    zx = 0, zy = 0;

    while (++iteration < iterations && zx * zx + zy * zy < 4.0) {
      tmpx = zx * zx - zy * zy + x;
      zy = 2 * zx * zy + y;
      zx = tmpx;
    }

    image[pixel] = iteration;
  }
}

__global__ void computeMandelbrot(int *image, int width, int height,
                                  int iterations, double x0, double y0,
                                  double x1, double y1)
{
  int pixel = blockDim.x * blockIdx.x + threadIdx.x;
  if (pixel >= width * height) return;

  double dx = (x1 - x0) / (double)width;
  double dy = (y1 - y0) / (double)height;

  int row = pixel / width, column = pixel % width, iteration = 0;

  double x = column * dx + x0, y = row * dy + y0;
  double zx = 0, zy = 0, tmpx;

  while (++iteration < iterations && zx * zx + zy * zy < 4.0) {
    tmpx = zx * zx - zy * zy + x;
    zy = 2 * zx * zy + y;
    zx = tmpx;
  }

  image[pixel] = iteration;
}

__global__ void computeMandelbrot2d(int *image, int width, int height,
                                    int iterations, double x0, double y0,
                                    double x1, double y1)
{
  int pixel = (gridDim.x * blockIdx.y + blockIdx.x) * blockDim.x * blockDim.y +
              threadIdx.y * blockDim.x + threadIdx.x;
  if (pixel >= width * height) return;

  double dx = (x1 - x0) / (double)width;
  double dy = (y1 - y0) / (double)height;

  int row = pixel / width, column = pixel % width, iteration = 0;

  double x = column * dx + x0, y = row * dy + y0;
  double zx = 0, zy = 0, tmpx;

  while (++iteration < iterations && zx * zx + zy * zy < 4.0) {
    tmpx = zx * zx - zy * zy + x;
    zy = 2 * zx * zy + y;
    zx = tmpx;
  }

  image[pixel] = iteration;
}

vector<double> times_cpu;
vector<pair<int, vector<double>>> times_1d;
vector<pair<pair<int, int>, vector<double>>> times_2d;

void print_results(string &&name, vector<double> results)
{
  sort(results.begin(), results.end());
  double median = results[results.size() / 2];
  double minimum = results.front();
  double avg = accumulate(results.begin(), results.end(), 0.0) / results.size();
  double var = accumulate(results.begin(), results.end(), 0.0,
                          [&](double acc, double x) {
                            return acc + (x - avg) * (x - avg);
                          }) /
               (results.size() - 1);
  double avgstd = sqrt(var) / sqrt(results.size());

  if (avgstd != 0) {
    double d = 1;
    while (abs(avgstd) < 10) {
      avgstd *= 10;
      d *= 10;
    }
    avgstd = ceil(avgstd) / d;
  }

  for (int i = 0; i < results.size(); i++)
    results[i] = times_cpu[i] / results[i];

  double avg_speedup =
      accumulate(results.begin(), results.end(), 0.0) / results.size();
  double var_speedup =
      accumulate(results.begin(), results.end(), 0.0,
                 [&](double acc, double x) {
                   return acc + (x - avg_speedup) * (x - avg_speedup);
                 }) /
      (results.size() - 1);
  double avgstd_speedup = sqrt(var_speedup) / sqrt(results.size());

  if (avgstd_speedup != 0) {
    double d = 1;
    while (abs(avgstd_speedup) < 10) {
      avgstd_speedup *= 10;
      d *= 10;
    }
    avgstd_speedup = ceil(avgstd_speedup) / d;
  }

  cout << "| " << name << " | " << median << " | " << minimum << " | " << avg
       << "+/-" << avgstd << " | " << avg_speedup << "+/-" << avgstd_speedup
       << " |" << endl;
}

int main(int argc, char *argv[])
{
  if (argc < 8) die("Usage: %s x0 y0 x1 y1 width height iterations", argv[0]);

  // setup

  double x0 = atof(argv[1]);
  double y0 = atof(argv[2]);
  double x1 = atof(argv[3]);
  double y1 = atof(argv[4]);

  int width = atoi(argv[5]);
  int height = atoi(argv[6]);
  int iterations = atoi(argv[7]);

  if (x0 >= x1 || y0 >= y1) die("Invalid range");
  if (width <= 0 || height <= 0) die("Invalid image size");
  if (iterations <= 0) die("Invalid iterations");

  int *h_image = (int *)malloc(width * height * sizeof(int)), *d_image;
  if (!h_image) die("buy more ram lol");

  cudaError_t status;
  status = cudaMalloc((void **)&d_image, width * height * sizeof(int));
  if (status != cudaSuccess) die("%s", cudaGetErrorString(status));

  // measurments

  // clang-format off
  auto measure_cpu = [&]() -> double {
    auto start = chrono::high_resolution_clock::now();
    computeMandelbrotCPU(h_image, width / 10, height / 10, iterations, x0, y0, x1, y1);
    auto end = chrono::high_resolution_clock::now();
    return chrono::duration_cast<chrono::milliseconds>(end - start).count() * 100;
  };
  // clang-format on

  auto measure_1d = [&](int dim) -> double {
    dim3 block_size(dim);
    dim3 grid_size((width * height + dim - 1) / dim);
    auto start = chrono::high_resolution_clock::now();
    computeMandelbrot<<<grid_size, block_size>>>(d_image, width, height,
                                                 iterations, x0, y0, x1, y1);
    status = cudaDeviceSynchronize();
    if (status != cudaSuccess) die("%s", cudaGetErrorString(status));
    auto end = chrono::high_resolution_clock::now();
    return chrono::duration_cast<chrono::milliseconds>(end - start).count();
  };

  auto measure_2d = [&](pair<int, int> dim) -> double {
    dim3 block_size(dim.first, dim.second);
    dim3 grid_size((width + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y);
    auto start = chrono::high_resolution_clock::now();
    computeMandelbrot2d<<<grid_size, block_size>>>(d_image, width, height,
                                                   iterations, x0, y0, x1, y1);
    status = cudaDeviceSynchronize();
    if (status != cudaSuccess) die("%s", cudaGetErrorString(status));
    auto end = chrono::high_resolution_clock::now();
    return chrono::duration_cast<chrono::milliseconds>(end - start).count();
  };

  for (int x = 32; x <= 1024; x *= 2)
    times_1d.emplace_back(x, vector<double>{});

  for (int x = 256, y = 1; y <= 256; x /= 2, y *= 2)
    times_2d.emplace_back(make_pair(x, y), vector<double>{});
  for (int x = 1024, y = 1; y <= 1024; x /= 2, y *= 2)
    times_2d.emplace_back(make_pair(x, y), vector<double>{});

  times_2d.emplace_back(make_pair(32, 32), vector<double>{});
  times_2d.emplace_back(make_pair(16, 16), vector<double>{});
  times_2d.emplace_back(make_pair(8, 8), vector<double>{});
  times_2d.emplace_back(make_pair(32, 16), vector<double>{});
  times_2d.emplace_back(make_pair(64, 8), vector<double>{});
  times_2d.emplace_back(make_pair(8, 64), vector<double>{});
  times_2d.emplace_back(make_pair(16, 32), vector<double>{});

  int runs = 17;

  // accept everything within 1% of the median
  for (int i = 0; i < runs; i++) times_cpu.emplace_back(measure_cpu());
  sort(times_cpu.begin(), times_cpu.end());
  pair<double, double> acceptance = {times_cpu[times_cpu.size() / 2] * 0.99,
                                     times_cpu[times_cpu.size() / 2] * 1.01};
  times_cpu.clear();

  while (runs) {
    auto time = measure_cpu();
    if (time < acceptance.first || time > acceptance.second) continue;
    times_cpu.emplace_back(time);
    for (auto &i : times_1d) i.second.emplace_back(measure_1d(i.first));
    for (auto &i : times_2d) i.second.emplace_back(measure_2d(i.first));
    runs--;
  }

  // present results

  print_results(string("CPU"), times_cpu);
  for (auto &i : times_1d) print_results(to_string(i.first), i.second);
  for (auto &i : times_2d)
    print_results(to_string(i.first.first) + "x" + to_string(i.first.second),
                  i.second);

  // cleanup

  status = cudaMemcpy(h_image, d_image, width * height * sizeof(int),
                      cudaMemcpyDeviceToHost);
  if (status != cudaSuccess) die("%s", cudaGetErrorString(status));
  status = cudaFree(d_image);
  if (status != cudaSuccess) die("%s", cudaGetErrorString(status));

#ifndef IMAGE
#define IMAGE 0
#endif
#if IMAGE
  FILE *fp;
  if (!(fp = fopen("mandelbrot.ppm", "wb"))) die("Cannot open file");

  fprintf(fp, "P6\n%d %d\n255\n", width, height);
  for (int i = 0; i < width * height; ++i) {
    uint8_t color[3] = {
        (uint8_t)((double)h_image[i] / (double)iterations * 255.0),
        (uint8_t)((double)h_image[i] / (double)iterations * 255.0),
        (uint8_t)((double)h_image[i] / (double)iterations * 255.0),
    };
    fwrite(color, 3, 1, fp);
  }

  fclose(fp);
#endif

  free(h_image);
  return 0;
}
