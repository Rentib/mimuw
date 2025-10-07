#include <unistd.h>
#include <stdio.h>

#define THREADS 256
#define START 0
#define END 1000

__global__ void sum(int* result) {
    __shared__ int partials[THREADS];

    int start = ((END - START) / blockDim.x) * threadIdx.x;
    int end = start + ((END - START) / blockDim.x) - 1;

    if (threadIdx.x == (THREADS - 1)) {
        end = END;
    }

    partials[threadIdx.x] = 0;

    for (int i = start; i <= end; i++) {
        partials[threadIdx.x] += i;
    }

    int i = blockDim.x / 2;
    while (i != 0) {
        if (threadIdx.x < i) {
            partials[threadIdx.x] += partials[threadIdx.x + i];
        }
        i /= 2;
    }

    if (threadIdx.x == 0) {
        *result = partials[0];
    }
}

int main(void) {
    int result;

    int* gpu_result;
    cudaMalloc((void**) &gpu_result, sizeof(int));

    sum<<<1, THREADS>>>(gpu_result);

    cudaMemcpy(&result, gpu_result, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(gpu_result);

    printf("GPU sum = %d.\n", result);
    
    int sum = 0; 
    for (int i = START; i <= END; i++) {
        sum += i;
    }
    printf("CPU sum = %d.\n", sum);
    
    return 0;
}
