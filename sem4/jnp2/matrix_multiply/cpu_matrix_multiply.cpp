#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include <chrono>
#include <iostream>

void matrix_transpose(float *A, float *At, int N);
void matrix_multiply(float *A, float *B, float *C, int N);

void matrix_transpose(float *A, float *At, int N)
{
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      At[j * N + i] = A[i * N + j];
    }
  }
}

void matrix_multiply(float *A, float *B, float *C, int N)
{
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      C[i * N + j] = 0;
      for (int k = 0; k < N; ++k) {
        C[i * N + j] += A[i * N + k] * B[k * N + j];
      }
    }
  }
}

bool matrix_equal(float *A, float *B, int N)
{
  for (int i = 0; i < N * N; ++i)
    if (A[i] != B[i]) return false;
  return true;
}

void die(const char *fmt, ...)
{
  va_list ap;
  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  fputc('\n', stderr);
  exit(1);
}

int main(int argc, char **argv)
{
  int N;
  float *A, *At;
  float *B, *Bt;
  float *C;

  if (argc != 2) die("Usage: %s N\n", argv[0]);

  N = atoi(argv[1]);

  A = (float *)malloc(N * N * sizeof(float));
  At = (float *)malloc(N * N * sizeof(float));
  B = (float *)malloc(N * N * sizeof(float));
  Bt = (float *)malloc(N * N * sizeof(float));
  C = (float *)malloc(N * N * sizeof(float));

  srand(777);

  for (int i = 0; i < N * N; ++i) {
    A[i] = rand() * 1.0 / RAND_MAX;
    B[i] = rand() * 1.0 / RAND_MAX;
  }

  auto time1 = std::chrono::high_resolution_clock::now();
  matrix_transpose(B, Bt, N);
  auto time2 = std::chrono::high_resolution_clock::now();
  matrix_multiply(A, Bt, C, N);
  auto time3 = std::chrono::high_resolution_clock::now();

  auto transpose_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time1)
          .count();
  auto multiply_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(time3 - time2)
          .count();

  std::cout << "Transpose: " << transpose_time << " ms" << std::endl;
  std::cout << "Multiply:  " << multiply_time << " ms" << std::endl;

  free(A);
  free(At);
  free(B);
  free(Bt);
  free(C);

  return 0;
}
