#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <array>
#include <iostream>
#include "CudaUtils.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define WORK_WIDTH 16
#define WORK_HEIGHT 1
#define BLOCK_WIDTH 4
#define BLOCK_HEIGHT 1
#define WORK_TOTAL WORK_WIDTH * WORK_HEIGHT

template <typename T, size_t SIZE>
__global__ void zad5inclusiveScan(T* devSrc, T* devRes) {
    __shared__ T sharedSrc[SIZE];
    int tid = threadIdx.x;
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gtid < SIZE) {
        sharedSrc[tid] = devSrc[gtid];
    }
    __syncthreads();
    // Shared memory got filled

    T sum = 0;
    if (gtid < SIZE) {
        for (size_t i = 0; i <= tid; ++i) {
            sum += sharedSrc[i];
        }
        devRes[gtid] = sum;
    }
}

template <typename T, size_t SIZE>
__global__ void zad6inclusiveScan(T* devSrc, T* devRes) {
    __shared__ T sharedSrc[SIZE];
    int tid = threadIdx.x;
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gtid < SIZE) {
        sharedSrc[tid] = devSrc[gtid];
    }
    __syncthreads();
    // Shared memory got filled

    T sum = 0;
    if (gtid < SIZE) {
        for (size_t i = 0; i <= tid; ++i) {
            sum += sharedSrc[i];
        }
        devRes[gtid] = sum;
    }
    if (gtid % WORK_WIDTH == WORK_WIDTH - 1) {
        devSrc[blockIdx.x] = sum;
    }
}

int main() {
    std::array<int, WORK_TOTAL> src {1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6, 4, 5, 6, 7};
    std::array<int, WORK_TOTAL> res;
    CudaBuffer<int> devSrc(src);
    CudaBuffer<int> devRes(WORK_TOTAL);

    dim3 dimBlock(BLOCK_WIDTH, BLOCK_HEIGHT);
    dim3 dimGrid(ceil(WORK_WIDTH / (float) dimBlock.x), ceil(WORK_HEIGHT / (float) dimBlock.y));
    printf("Invoking with: Block(%d,%d), Grid(%d,%d)\n", dimBlock.x, dimBlock.y, dimGrid.x, dimGrid.y);
    zad6inclusiveScan<int, WORK_TOTAL> <<<dimGrid, dimBlock>>> (devSrc, devRes);
    // Wait for the kernel to complete and check for errors
    checkCuda(cudaPeekAtLastError());
    checkCuda(cudaDeviceSynchronize());

    devRes.copyTo(res);
    // Print the results
    for (int col = 0; col < WORK_TOTAL; ++col) {
        std::cout << res[col] << std::endl;
    }
}
