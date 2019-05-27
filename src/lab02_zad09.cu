#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_profiler_api.h"

#include <array>
#include <iostream>
#include "CudaUtils.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define WORK_WIDTH 14
#define WORK_HEIGHT 1
#define BLOCK_WIDTH 7
#define BLOCK_HEIGHT 1
#define WORK_TOTAL WORK_WIDTH * WORK_HEIGHT
#define RESULTS_TOTAL 6

template <typename T, std::size_t SHARED_SIZE>
__global__ void compute(T* devSrc, T* devRes, size_t srcSize) {
    __shared__ T sharedSrc[SHARED_SIZE];
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gtid >= srcSize) {
        return;
    }
    if (threadIdx.x < SHARED_SIZE) {
        sharedSrc[threadIdx.x] = 0;
    }
    devRes[gtid] = 0;

    T value = devSrc[gtid];
    __syncthreads();
    atomicAdd(&sharedSrc[value], 1);

    if (threadIdx.x < SHARED_SIZE) {
        __syncthreads();
        atomicAdd(&devRes[threadIdx.x], sharedSrc[threadIdx.x]);
    }
}

int main() {
    std::array<int, WORK_TOTAL> src {0, 1, 3, 2, 5, 0, 0, 4, 3, 2, 1, 1, 5, 4};
    std::array<int, RESULTS_TOTAL> res;

    runWithProfiler([&]() {
        CudaBuffer<int> devSrc(src);
        CudaBuffer<int> devRes(res.size());
        dim3 dimBlock(BLOCK_WIDTH, BLOCK_HEIGHT);
        dim3 dimGrid(ceil(WORK_WIDTH / (float) dimBlock.x), ceil(WORK_HEIGHT / (float) dimBlock.y));
        printf("Invoking with: Block(%d,%d), Grid(%d,%d)\n", dimBlock.x, dimBlock.y, dimGrid.x, dimGrid.y);
        compute<int, RESULTS_TOTAL> << < dimGrid, dimBlock >> > (devSrc, devRes, WORK_TOTAL);
        devRes.copyTo(res);
    });

    // Print the results
    for (int col = 0; col < res.size(); ++col) {
        std::cout << res[col] << std::endl;
    }
}
