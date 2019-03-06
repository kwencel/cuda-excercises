
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <array>
#include <iostream>
#include "CudaUtils.h"

#define WORK_WIDTH 9
#define WORK_HEIGHT 1
#define BLOCK_WIDTH 3
#define BLOCK_HEIGHT 1
#define WORK_TOTAL WORK_WIDTH * WORK_HEIGHT

dim3 dimBlock(BLOCK_WIDTH, BLOCK_HEIGHT);
dim3 dimGrid(ceil(WORK_WIDTH / (float) dimBlock.x / 2), ceil(WORK_HEIGHT / (float) dimBlock.y));

template <typename T>
__device__ void swap(T* a, T* b) {
    T tmp = *a;
    *a = *b;
    *b = tmp;
}

template <typename T>
__global__ void inverseArray(T* devRes) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < WORK_WIDTH / 2) {
        printf("Thread %d swapping pos %d with %d\n", col, col, WORK_WIDTH - 1 - col);
        swap(&devRes[col], &devRes[WORK_WIDTH - 1 - col]);
    }
}

int main() {
    printf("Kernel will be invoked with: Block(%d,%d), Grid(%d,%d)\n", dimBlock.x, dimBlock.y, dimGrid.x, dimGrid.y);
    std::array<float, WORK_TOTAL> src = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::array<float, WORK_TOTAL> res;
    CudaBuffer<float> devRes(WORK_TOTAL);

    devRes.copyFrom(src);
    inverseArray<float> <<<dimGrid, dimBlock>>> (devRes);
    devRes.copyTo(res);

    // Wait for the kernel to complete and check for errors
    checkCuda(cudaPeekAtLastError());
    checkCuda(cudaDeviceSynchronize());

    // Print the results
    for (int col = 0; col < WORK_WIDTH; ++col) {
        std::cout << res[col] << " ";
    }
    std::cout << std::endl;

    return 0;
}
