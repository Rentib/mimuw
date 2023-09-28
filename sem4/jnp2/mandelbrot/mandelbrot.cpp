#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <chrono>
#include <complex>
#include <iostream>
#include <numeric>
#include <vector>

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

  for (int pixel = 0; pixel < width * height; ++pixel, iteration = 0) {
    row = pixel / width;
    column = pixel % width;

    std::complex<double> c(x0 + column * dx, y0 + row * dy);
    std::complex<double> z(0, 0);

    while (++iteration < iterations && norm(z) < 4.0) z = z * z + c;

    image[pixel] = iteration - 1;
  }
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

  double reference;

  std::chrono::time_point<std::chrono::system_clock> start, end;
  std::vector<double> elapsed_times = {};

  int runs = 43;  // cool prime number

  auto print_results = [&](const std::string &name) {
    std::sort(elapsed_times.begin(), elapsed_times.end());

    auto median = elapsed_times[elapsed_times.size() / 2];
    auto minimum = elapsed_times.front();
    auto mean =
        std::accumulate(elapsed_times.begin(), elapsed_times.end(), 0.0) /
        elapsed_times.size();
    auto variance = std::accumulate(
        elapsed_times.begin(), elapsed_times.end(), 0.0,
        [&](double a, double b) { return a + (b - mean) * (b - mean); });
    auto stddev = std::sqrt(variance / elapsed_times.size());
    auto avgstddev = stddev / std::sqrt(elapsed_times.size());

    // precision
    double d = 1;
    while (avgstddev < 10) {
      avgstddev *= 10;
      d *= 10;
    }
    avgstddev = std::ceil(avgstddev) / d;

    std::cout << "| " << name << " | " << median << " | " << minimum << " | "
              << mean << " +/- " << avgstddev << " | " << reference / mean
              << " |\n";

    elapsed_times.clear();
  };

  // computations

  // cpu

  for (int i = 0; i < runs; ++i) {
    start = std::chrono::system_clock::now();
    computeMandelbrotCPU(h_image, width / 10, height / 10, iterations, x0, y0,
                         x1, y1);
    end = std::chrono::system_clock::now();
    elapsed_times.emplace_back(
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count() *
        100.0);
  }

  reference = std::accumulate(elapsed_times.begin(), elapsed_times.end(), 0.0) /
              elapsed_times.size();

  print_results("CPU");

  // cleanup

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
