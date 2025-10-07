#include "./common/helpers.h"

#define SIZE (10 * 1024 * 1024)

float cuda_malloc_test(int size, bool up) {
    cudaEvent_t start, stop;
    int *a, *dev_a;
    float elapsedTime;

    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));

    a = (int*)malloc(size * sizeof(*a)); 

    HANDLE_NULL(a);
    HANDLE_ERROR(cudaMalloc((void**)&dev_a, size * sizeof(*dev_a)));

    HANDLE_ERROR(cudaEventRecord(start, 0)); 

    for (int i = 0; i < 100; i++) {
        if (up) {
            HANDLE_ERROR(cudaMemcpy(dev_a, a, size * sizeof(*dev_a), cudaMemcpyHostToDevice));
        } else {
            HANDLE_ERROR(cudaMemcpy(a, dev_a, size * sizeof(*dev_a), cudaMemcpyDeviceToHost));
        }
    }

    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));

    free(a);
    HANDLE_ERROR(cudaFree(dev_a));
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));

    return elapsedTime; 
}

int main(void) {
    float elapsedTime;
    float MB = (float)100 * SIZE * sizeof(int) / 1024 / 1024;

    elapsedTime = cuda_malloc_test(SIZE, true);

    printf("Total time for copy up: %3.1f ms\n", elapsedTime);
    printf("\tMB/s during copy up:  %3.1f\n", MB / (elapsedTime / 1000));

    elapsedTime = cuda_malloc_test(SIZE, false); 
    
    printf("Total time for copy down: %3.1f ms\n", elapsedTime);
    printf("\tMB/s during copy down:  %3.1f\n", MB / (elapsedTime / 1000));

    return 0;
}
