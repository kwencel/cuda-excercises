#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <array>
#include <iostream>
#include "CudaUtils.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define WORK_WIDTH 12
#define WORK_HEIGHT 1
#define BLOCK_WIDTH 4
#define BLOCK_HEIGHT 1
#define WORK_TOTAL WORK_WIDTH * WORK_HEIGHT

#define D 5

template <typename A, typename B>
__device__ auto min(const A& a, const B& b) {
    return a < b ? a : b;
}

template <typename A, typename B>
__device__ auto max(const A& a, const B& b) {
    return a > b ? a : b;
}

template <typename T>
__device__ float avg(const T& data, const size_t size) {
    size_t sum = 0;
    for (int i = 0; i < size; ++i) {
        sum += data[i];
    }
    return __fdiv_ru(sum, size);
}

template <typename T>
__global__ void compute(T* devSrc, T* devRes, size_t length) {
    extern __shared__ T shared[];
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    shared[threadIdx.x] = devSrc[max(0, gtid - (D - 1) / 2)];

    if (threadIdx.x < D - 1) {
        shared[blockDim.x + threadIdx.x] = devSrc[min(blockDim.x, blockDim.x + gtid - (D - 1) / 2)];
    }
    __syncthreads();
    // Shared memory got filled

    devRes[gtid] = avg(shared, blockDim.x + D - 1);
//    printf("Thread %d adding #%d and #%d\n", pos, pos + (offset / 2));
}

int main() {
    std::array<int, WORK_TOTAL> src {0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0};
    std::array<int, WORK_TOTAL> res;

    runWithProfiler([&]() {
        CudaBuffer<int> devSrc {src};
        CudaBuffer<int> devRes {WORK_TOTAL};

        dim3 dimBlock(BLOCK_WIDTH, BLOCK_HEIGHT);
        dim3 dimGrid(ceil(WORK_WIDTH / (float) dimBlock.x), ceil(WORK_HEIGHT / (float) dimBlock.y));
        printf("Invoking with: Block(%d,%d), Grid(%d,%d)\n", dimBlock.x, dimBlock.y, dimGrid.x, dimGrid.y);
        compute<int> <<<dimGrid, dimBlock, (dimBlock.x + D - 1) * sizeof(int)>>> (devSrc, devRes, WORK_TOTAL);
        devRes.copyTo(res);
    });

    // Print the results
    for (int col = 0; col < WORK_TOTAL; ++col) {
        std::cout << res[col] << std::endl;
    }
}
