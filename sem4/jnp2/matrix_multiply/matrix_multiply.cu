#include <cuda.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace std;
using namespace std::chrono;

using real = float;

template <bool is_host, size_t N>
real *matrix_create(int seed)
{
  uniform_real_distribution<real> dist(0.0, 1.0);
  mt19937 re(seed);
  real *matrix;
  if (is_host) {
    matrix = new real[N * N];
  } else {
    cudaMalloc(&matrix, N * N * sizeof(real));
  }
  for (int i = 0; i < N * N; ++i) matrix[i] = dist(re);
  return matrix;
}

template <bool is_host>
void matrix_destroy(real *matrix)
{
  if (is_host) {
    delete[] matrix;
  } else {
    cudaFree(matrix);
  }
}

template <size_t N>
void matrix_transpose(real *matrix)
{
  for (size_t x = 0; x < N; ++x) {
    for (size_t y = 0; y < x; ++y) {
      swap(matrix[x * N + y], matrix[y * N + x]);
    }
  }
}

template <size_t N>
void matrix_multiply_cpu(real *A, real *B, real *C)
{
  for (size_t x = 0; x < N; ++x) {
    for (size_t y = 0; y < N; ++y) {
      real sum = 0.0;
      for (size_t k = 0; k < N; ++k) sum += A[x * N + k] * B[k * N + y];
      C[x * N + y] = sum;
    }
  }
}

template <size_t N>
__global__ void matrix_multiply_gpu1(real *A, real *B, real *C)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= N || j >= N) return;
  real sum = 0.0;
  for (size_t k = 0; k < N; ++k) sum += A[i * N + k] * B[k * N + j];
  C[i * N + j] = sum;
}

template <size_t N, size_t size1, bool isv3>
__global__ void matrix_multiply_gpu23(real *A, real *B, real *C)
{
  constexpr size_t size2 = isv3 ? size1 + 1 : size1;
  __shared__ real res[size1 * size2];
  __shared__ real row[size1];

  size_t tx = threadIdx.x;
  size_t bx = blockIdx.x * size1;
  size_t by = blockIdx.y * size1;

  for (size_t i = 0; i < size1; ++i) res[i * size2 + tx] = 0;
  res[tx * size2 + tx] = 1;
  __syncthreads();

  size_t x = bx + tx;
  size_t y = by + tx;
  for (size_t k = 0; k < N; ++k) {
    row[tx] = x < N ? B[k * N + x] : 0;
    __syncthreads();

    real col = y < N ? A[y * N + k] : 0;
    __syncthreads();

    for (size_t i = 0; i < size1; ++i) res[tx * size2 + i] += col * row[i];
    __syncthreads();
  }
  __syncthreads();

  for (int i = 0; i < size1; ++i) {
    if (y < N && bx + i < N) C[y * N + bx + i] = res[tx * size2 + i];
  }
  __syncthreads();
}

template <size_t N, size_t size>
__global__ void matrix_multiply_gpu4(real *A, real *B, real *C)
{
  __shared__ real res[size * size];

  size_t tx = threadIdx.x;
  size_t bx = blockIdx.x * size;
  size_t by = blockIdx.y * size;

  for (size_t i = 0; i < size; ++i) res[i * size + tx] = 0;

  size_t x = bx + tx;
  size_t y = by + tx;
  for (size_t k = 0; k < N; ++k) {
    real row = x < N ? B[k * N + x] : 0;
    __syncthreads();

    real col = y < N ? A[y * N + k] : 0;
    __syncthreads();

    for (size_t i = 0; i < size; ++i)
      res[tx * size + i] += col * __shfl_sync(0xFFFFFFFF, row, i);
    __syncthreads();
  }

  for (size_t i = 0; i < size; ++i) {
    if (y < N && bx + i < N) C[y * N + bx + i] = res[tx * size + i];
  }
  __syncthreads();
}

template <size_t N, size_t size>
__global__ void __launch_bounds__(32)
    matrix_multiply_gpu5(real *A, real *B, real *C)
{
  real res[size] = {0};
  size_t tx = threadIdx.x;
  size_t bx = blockIdx.x * size;
  size_t by = blockIdx.y * size;

  size_t x = bx + tx;
  size_t y = by + tx;
  for (size_t k = 0; k < N; ++k) {
    real row = x < N ? B[k * N + x] : 0;
    __syncthreads();

    real col = y < N ? A[y * N + k] : 0;
    __syncthreads();

#pragma unroll
    for (size_t i = 0; i < size; ++i)
      res[i] += col * __shfl_sync(0xffffffff, row, i);
  }

#pragma unroll
  for (size_t i = 0; i < size; ++i) {
    if (y < N && bx + i < N) C[y * N + bx + i] = res[i];
  }
  __syncthreads();
}

int main()
{
  constexpr int N = 1000;
  constexpr int runs = 11;

  int seed = 17;

  real *h_A, *h_B, *h_C;
  real *d_A, *d_B, *d_C;

  h_A = matrix_create<false, N>(seed);
  h_B = matrix_create<false, N>(seed);
  h_C = matrix_create<false, N>(seed);
  d_A = matrix_create<true, N>(seed);
  d_B = matrix_create<true, N>(seed);
  d_C = matrix_create<true, N>(seed);

  cudaMemcpy(d_A, h_A, N * N * sizeof(real), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, N * N * sizeof(real), cudaMemcpyHostToDevice);

  pair<double, double> control_cpu;
  {
    vector<double> times;
    for (int i = 0; i < runs; ++i) {
      auto start = chrono::high_resolution_clock::now();
      matrix_multiply_cpu<N>(h_A, h_B, h_C);
      auto end = chrono::high_resolution_clock::now();
      times.push_back(chrono::duration<double>(end - start).count());
    }
    sort(times.begin(), times.end());
    control_cpu = {times[runs / 2] * 0.99, times[runs / 2] * 1.01};
  }

  vector<string> names = {
      "| CPU       |    -- | ", "| Kernel #1 |   8x8 | ",
      "| Kernel #1 | 16x16 | ", "| Kernel #1 | 32x32 | ",
      "| Kernel #2 |  32x1 | ", "| Kernel #2 |  64x1 | ",
      "| Kernel #2 |  96x1 | ", "| Kernel #2 | 128x1 | ",
      "| Kernel #3 |  32x1 | ", "| Kernel #3 |  64x1 | ",
      "| Kernel #4 |  32x1 | ", "| Kernel #5 |  32x1 | ",
  };

  map<string, vector<double>> times;
  map<string, double> means;
  map<string, double> stdevs;

  for (int i = 0; i < runs; ++i) {
    {
      auto start = chrono::high_resolution_clock::now();
      matrix_multiply_cpu<N>(h_A, h_B, h_C);
      auto end = chrono::high_resolution_clock::now();
      double t = chrono::duration<double, milli>(end - start).count();
      if (t < control_cpu.first || t > control_cpu.second) {
        cerr << "CPU time out of range: " << t << "ms" << endl;
        --i;
        continue;
      }
      times["| CPU       |    -- | "].push_back(t);
    }

    {
      auto start = chrono::high_resolution_clock::now();
      dim3 block_size(8, 8);
      dim3 grid_size((N + block_size.x - 1) / block_size.x,
                     (N + block_size.y - 1) / block_size.y);
      matrix_multiply_gpu1<N><<<grid_size, block_size>>>(d_A, d_B, d_C);
      cudaDeviceSynchronize();
      auto end = chrono::high_resolution_clock::now();
      double t = chrono::duration<double, milli>(end - start).count();
      times["| Kernel #1 |   8x8 | "].push_back(t);
    }

    {
      auto start = chrono::high_resolution_clock::now();
      dim3 block_size(16, 16);
      dim3 grid_size((N + block_size.x - 1) / block_size.x,
                     (N + block_size.y - 1) / block_size.y);
      matrix_multiply_gpu1<N><<<grid_size, block_size>>>(d_A, d_B, d_C);
      cudaDeviceSynchronize();
      auto end = chrono::high_resolution_clock::now();
      double t = chrono::duration<double, milli>(end - start).count();
      times["| Kernel #1 | 16x16 | "].push_back(t);
    }

    {
      auto start = chrono::high_resolution_clock::now();
      dim3 block_size(32, 32);
      dim3 grid_size((N + block_size.x - 1) / block_size.x,
                     (N + block_size.y - 1) / block_size.y);
      matrix_multiply_gpu1<N><<<grid_size, block_size>>>(d_A, d_B, d_C);
      cudaDeviceSynchronize();
      auto end = chrono::high_resolution_clock::now();
      double t = chrono::duration<double, milli>(end - start).count();
      times["| Kernel #1 | 32x32 | "].push_back(t);
    }

    {
      auto start = chrono::high_resolution_clock::now();
      dim3 block_size(32, 1);
      dim3 grid_size((N + block_size.x - 1) / block_size.x,
                     (N + block_size.y - 1) / block_size.y);
      matrix_multiply_gpu23<N, 32, false>
          <<<grid_size, block_size>>>(d_A, d_B, d_C);
      cudaDeviceSynchronize();
      auto end = chrono::high_resolution_clock::now();
      double t = chrono::duration<double, milli>(end - start).count();
      times["| Kernel #2 |  32x1 | "].push_back(t);
    }

    {
      auto start = chrono::high_resolution_clock::now();
      dim3 block_size(64, 1);
      dim3 grid_size((N + block_size.x - 1) / block_size.x,
                     (N + block_size.y - 1) / block_size.y);
      matrix_multiply_gpu23<N, 64, false>
          <<<grid_size, block_size>>>(d_A, d_B, d_C);
      cudaDeviceSynchronize();
      auto end = chrono::high_resolution_clock::now();
      double t = chrono::duration<double, milli>(end - start).count();
      times["| Kernel #2 |  64x1 | "].push_back(t);
    }

    {
      auto start = chrono::high_resolution_clock::now();
      dim3 block_size(96, 1);
      dim3 grid_size((N + block_size.x - 1) / block_size.x,
                     (N + block_size.y - 1) / block_size.y);
      matrix_multiply_gpu23<N, 96, false>
          <<<grid_size, block_size>>>(d_A, d_B, d_C);
      cudaDeviceSynchronize();
      auto end = chrono::high_resolution_clock::now();
      double t = chrono::duration<double, milli>(end - start).count();
      times["| Kernel #2 |  64x1 | "].push_back(t);
    }

#if 0
    {
      auto start = chrono::high_resolution_clock::now();
      dim3 block_size(128, 1);
      dim3 grid_size((N + block_size.x - 1) / block_size.x,
                     (N + block_size.y - 1) / block_size.y);
      matrix_multiply_gpu23<N, 128, false>
          <<<grid_size, block_size>>>(d_A, d_B, d_C);
      auto end = chrono::high_resolution_clock::now();
      double t = chrono::duration<double, milli>(end - start).count();
      times["| Kernel #2 |  64x1 | "].push_back(t);
    }
#endif

    {
      auto start = chrono::high_resolution_clock::now();
      dim3 block_size(32, 1);
      dim3 grid_size((N + block_size.x - 1) / block_size.x,
                     (N + block_size.y - 1) / block_size.y);
      matrix_multiply_gpu23<N, 32, true>
          <<<grid_size, block_size>>>(d_A, d_B, d_C);
      cudaDeviceSynchronize();
      auto end = chrono::high_resolution_clock::now();
      double t = chrono::duration<double, milli>(end - start).count();
      times["| Kernel #3 | 32x32 | "].push_back(t);
    }

    {
      auto start = chrono::high_resolution_clock::now();
      dim3 block_size(64, 1);
      dim3 grid_size((N + block_size.x - 1) / block_size.x,
                     (N + block_size.y - 1) / block_size.y);
      matrix_multiply_gpu23<N, 64, true>
          <<<grid_size, block_size>>>(d_A, d_B, d_C);
      cudaDeviceSynchronize();
      auto end = chrono::high_resolution_clock::now();
      double t = chrono::duration<double, milli>(end - start).count();
      times["| Kernel #3 | 64x64 | "].push_back(t);
    }

    {
      auto start = chrono::high_resolution_clock::now();
      dim3 block_size(32, 1);
      dim3 grid_size((N + block_size.x - 1) / block_size.x, 1);
      matrix_multiply_gpu4<N, 32><<<grid_size, block_size>>>(d_A, d_B, d_C);
      cudaDeviceSynchronize();
      auto end = chrono::high_resolution_clock::now();
      double t = chrono::duration<double, milli>(end - start).count();
      times["| Kernel #4 | 32x32 | "].push_back(t);
    }

    {
      auto start = chrono::high_resolution_clock::now();
      dim3 block_size(32, 1);
      dim3 grid_size((N + block_size.x - 1) / block_size.x, 1);
      matrix_multiply_gpu5<N, 32><<<grid_size, block_size>>>(d_A, d_B, d_C);
      cudaDeviceSynchronize();
      auto end = chrono::high_resolution_clock::now();
      double t = chrono::duration<double, milli>(end - start).count();
      times["| Kernel #5 | 32x32 | "].push_back(t);
    }
  }

  for (const auto &p : times) {
    double mean = 0;
    for (const auto &d : p.second) mean += d;
    mean /= p.second.size();
    double var = 0;
    for (const auto &d : p.second) var += (d - mean) * (d - mean);
    var /= p.second.size() - 1;
    double stdev = sqrt(var);
    means[p.first] = mean;
    stdevs[p.first] = stdev;
  }

  string &cpu_name = names[0];
  cout << cpu_name << " | " << means[cpu_name] << "+/-" << stdevs[cpu_name]
       << " | -- |" << endl;
  for (const auto &s : names) {
    if (s == names[0]) continue;
    cout << s << " | " << means[s] << "+/-" << stdevs[s] << " | "
         << means[cpu_name] / means[s] << "+/-"
         << sqrt(pow(means[s] * stdevs[cpu_name], 2) +
                 pow(stdevs[s] * means[s], 2)) /
                pow(means[s], 2)
         << " |" << endl;
  }

  matrix_destroy<false>(h_A);
  matrix_destroy<false>(h_B);
  matrix_destroy<false>(h_C);
  matrix_destroy<true>(d_A);
  matrix_destroy<true>(d_B);
  matrix_destroy<true>(d_C);

  return 0;
}
