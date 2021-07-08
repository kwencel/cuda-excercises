#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <array>
#include <iostream>
#include "CudaUtils.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define WORK_WIDTH 12
#define WORK_HEIGHT 1
#define BLOCK_WIDTH 12
#define BLOCK_HEIGHT 1
#define WORK_TOTAL WORK_WIDTH * WORK_HEIGHT

template <typename T, size_t SIZE>
__global__ void compute(T* devSrc, T* devRes) {
    __shared__ T sharedSrc[SIZE];
    int tid = threadIdx.x;

    if (tid < SIZE) {
        sharedSrc[tid] = devSrc[tid];
    }
    __syncthreads();
    // Shared memory got filled

    T sum = 0;
    if (tid < SIZE) {
        for (size_t i = 0; i <= tid; ++i) {
            sum += sharedSrc[i];
        }
        devRes[tid] = sum;
    }
}

int main() {
    std::array<int, WORK_TOTAL> src {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::array<int, WORK_TOTAL> res;

    runWithProfiler([&]() {
        CudaBuffer<int> devSrc {src};
        CudaBuffer<int> devRes {WORK_TOTAL};

        dim3 dimBlock(BLOCK_WIDTH, BLOCK_HEIGHT);
        dim3 dimGrid(ceil(WORK_WIDTH / (float) dimBlock.x), ceil(WORK_HEIGHT / (float) dimBlock.y));
        printf("Invoking with: Block(%d,%d), Grid(%d,%d)\n", dimBlock.x, dimBlock.y, dimGrid.x, dimGrid.y);
        compute<int, WORK_TOTAL> <<<dimGrid, dimBlock>>> (devSrc, devRes);
        devRes.copyTo(res);
    });

    // Print the results
    for (int col = 0; col < WORK_TOTAL; ++col) {
        std::cout << res[col] << std::endl;
    }
}
