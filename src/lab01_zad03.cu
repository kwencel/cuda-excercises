
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <array>
#include <iostream>
#include "CudaUtils.h"

#define WORK_WIDTH 10
#define WORK_HEIGHT 10
#define BLOCK_WIDTH 4
#define BLOCK_HEIGHT 4
#define WORK_TOTAL WORK_WIDTH * WORK_HEIGHT

dim3 dimBlock(BLOCK_WIDTH, BLOCK_HEIGHT);
dim3 dimGrid(ceil(WORK_WIDTH / (float) dimBlock.x), ceil(WORK_HEIGHT / (float) dimBlock.y));

__global__ void inveseArray(float* devRes) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < WORK_HEIGHT && row < WORK_WIDTH) {
        devRes[row * WORK_WIDTH + col] = (row + 1) * (col + 1);
    }
}

int main() {
    printf("Kernel will be invoked with: Block(%d,%d), Grid(%d,%d)\n", dimBlock.x, dimBlock.y, dimGrid.x, dimGrid.y);
    std::array<float, WORK_TOTAL> res;

    runWithProfiler([&]() {
        CudaBuffer<float> devRes(WORK_TOTAL);
        inveseArray <<<dimGrid, dimBlock>>> (devRes);
        devRes.copyTo(res);
    });

    // Print the results
    for (int col = 0; col < WORK_WIDTH; ++col) {
        for (int row = 0; row < WORK_HEIGHT; ++row) {
            std::cout << res[row * WORK_WIDTH + col] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
