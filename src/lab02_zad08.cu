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


template <typename T>
__global__ void compute(T* devSrc, T* devRes, size_t srcSize) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gtid >= srcSize) {
        return;
    }
    devRes[gtid] = 0;
    T value = devSrc[gtid];
    __syncthreads();
    atomicAdd(&devRes[value], 1);
}

int main() {
    std::array<int, WORK_TOTAL> src {0, 1, 3, 2, 5, 0, 0, 4, 3, 2, 1, 1, 5, 4};
    std::array<int, RESULTS_TOTAL> res;

    cudaProfilerStart();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    CudaBuffer<int> devSrc(src);
    CudaBuffer<int> devRes(WORK_TOTAL);
    dim3 dimBlock(BLOCK_WIDTH, BLOCK_HEIGHT);
    dim3 dimGrid(ceil(WORK_WIDTH / (float) dimBlock.x), ceil(WORK_HEIGHT / (float) dimBlock.y));
    printf("Invoking with: Block(%d,%d), Grid(%d,%d)\n", dimBlock.x, dimBlock.y, dimGrid.x, dimGrid.y);
    compute<int> <<<dimGrid, dimBlock>>> (devSrc, devRes, WORK_TOTAL);
    devRes.copyTo(res);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    // Print kernel execution time
    float elapsedTime;
    checkCuda(cudaEventElapsedTime(&elapsedTime, start, stop));
    // Wait for the kernel to complete and check for errors
    checkCuda(cudaPeekAtLastError());
    checkCuda(cudaDeviceSynchronize());
    cudaProfilerStop();
    printf("Kernel execution finished in %.3f ms\n", elapsedTime);

    // Print the results
    for (int col = 0; col < res.size(); ++col) {
        std::cout << res[col] << std::endl;
    }
}
