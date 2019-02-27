
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <iostream>
#include <array>
#include "CudaUtils.h"

#define WORK_WIDTH 4
#define WORK_HEIGHT 4
#define BLOCK_WIDTH 4
#define BLOCK_HEIGHT 4
#define WORK_TOTAL WORK_WIDTH * WORK_HEIGHT

dim3 dimBlock(WORK_WIDTH, WORK_HEIGHT);
dim3 dimGrid(ceil(WORK_WIDTH / (float) dimBlock.x), ceil(WORK_HEIGHT / (float) dimBlock.y));

__global__ void addRowNo(float* devSrc, float* devRes) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < WORK_WIDTH && row < WORK_HEIGHT) {
        devRes[col * WORK_WIDTH + row] = devSrc[col * WORK_HEIGHT + row] + col;
    }
}

int main() {
    std::array<float, WORK_TOTAL> src {0};
    std::array<float, WORK_TOTAL> res;

    CudaBuffer<float> devSrc(WORK_TOTAL);
    CudaBuffer<float> devRes(WORK_TOTAL);
    devSrc.copyFrom(src);

    addRowNo <<<dimGrid, dimBlock>>> (devSrc, devRes);
    devRes.copyTo(res);

    checkCuda(cudaDeviceSynchronize());
    for (int i = 0; i < WORK_TOTAL; ++i) {
        std::cout << res[i] << std::endl;
    }

    return 0;
}
