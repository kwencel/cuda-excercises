#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <array>
#include <iostream>
#include "CudaUtils.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "util/StringConcat.h"

#define WORK_WIDTH 10
#define WORK_HEIGHT 1
#define BLOCK_WIDTH 3
#define BLOCK_HEIGHT 1
#define WORK_TOTAL WORK_WIDTH * WORK_HEIGHT

dim3 dimBlock(BLOCK_WIDTH, BLOCK_HEIGHT);
dim3 dimGrid(ceil(WORK_WIDTH / (float) dimBlock.x), ceil(WORK_HEIGHT / (float) dimBlock.y));

template <typename T>
__global__ void compute(T* devSrc, T* devRes, size_t length, size_t offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= length) {
        printf("Thread %d returns prematurely\n", idx);
        return;
    }
    devRes[(idx + offset) % length] = devSrc[idx];

    printf("Thread %d moving %d to %d\n", idx, idx, (idx + offset) % length);
}

int main() {
    printf("Kernel will be invoked with: Block(%d,%d), Grid(%d,%d)\n", dimBlock.x, dimBlock.y, dimGrid.x, dimGrid.y);

    std::array<float, WORK_TOTAL> src {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    CudaBuffer<float> devSrc(src);

    std::array<float, WORK_TOTAL> res;
    CudaBuffer<float> devRes(WORK_TOTAL);

    compute<float> <<<dimGrid, dimBlock>>> (devSrc, devRes, WORK_TOTAL, 5);
    devRes.copyTo(res);

    // Wait for the kernel to complete and check for errors
    checkCuda(cudaPeekAtLastError());
    checkCuda(cudaDeviceSynchronize());

    // Print the results
    for (int col = 0; col < WORK_TOTAL; ++col) {
        std::cout << res[col] << std::endl;
    }
}
