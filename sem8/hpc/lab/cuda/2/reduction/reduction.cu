#include <stdio.h>
#include "common/errors.h"


const int N = 30 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = min(32, (N + threadsPerBlock - 1) / threadsPerBlock);

__global__ void dot(int *a, int *b, int *c) {

    __shared__ int cache[threadsPerBlock];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    int temp = 0;
    while (tid < N) {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    cache[cacheIndex] = temp;
    
    __syncthreads();

    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i) { 
            cache[cacheIndex] += cache[cacheIndex + i]; 
        }
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0) {
        c[blockIdx.x] = cache[0];
    }
}

int main(void) {
    int *a, *b, c, *partial_c, vfy_c;
    int *dev_a, *dev_b, *dev_partial_c;

    a = new int[N];
    b = new int[N];
    partial_c = new int[blocksPerGrid];

    // Generate some data
    for (int i = 0; i < N; i++) {
	    a[i] = 3 * i + 2;
	    b[i] = 2 * i - 1;
    }

    HANDLE_ERROR(cudaMalloc(&dev_a, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc(&dev_b, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc(&dev_partial_c, blocksPerGrid * sizeof(int)));

    HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));


    dot<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_partial_c);

    HANDLE_ERROR(cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid * sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(dev_partial_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    c = 0;
    for (int i = 0; i < blocksPerGrid; i++) {
        c += partial_c[i];
    }

    // Verify
    vfy_c = 0;
    for (int i = 0; i < N; i++)
	    vfy_c += a[i] * b[i];

    if (c == vfy_c)
	    printf("OK\n");

    delete a;
    delete b;
    delete partial_c;
    return 0;
}       
