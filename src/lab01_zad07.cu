#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <array>
#include <iostream>
#include "CudaUtils.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "util/StringConcat.h"

#define WORK_WIDTH 8
#define WORK_HEIGHT 1
#define BLOCK_WIDTH 1
#define BLOCK_HEIGHT 1
#define WORK_TOTAL WORK_WIDTH * WORK_HEIGHT

template <typename T>
__global__ void compute(T* devRes, size_t length, size_t offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pos = offset * idx;
    if (pos >= length) {
        printf("Thread %d returning prematurely\n", idx);
        return;
    }
    if (blockDim.x * gridDim.x < length / offset) {
        printf("Too little threads to process the array");
    }

    devRes[pos] += devRes[pos + (offset / 2)];
    printf("Thread %d adding #%d and #%d\n", pos, pos + (offset / 2));
}

int main() {
    std::array<int, WORK_TOTAL> src {2, 3, 1, 2, 3, 3, 0, 1};
    std::array<int, WORK_TOTAL> res;
    CudaBuffer<int> devRes(src);

    for (int i = 1; i <= log2(WORK_TOTAL); ++i) {
        size_t threads = pow(2, i);
        dim3 dimBlock(BLOCK_WIDTH, BLOCK_HEIGHT);
        dim3 dimGrid(ceil(WORK_WIDTH / (float) threads), ceil(WORK_HEIGHT / (float) dimBlock.y));
        printf("Invoking with: k = %zd, Block(%d,%d), Grid(%d,%d)\n", threads, dimBlock.x, dimBlock.y, dimGrid.x, dimGrid.y);
        compute<int> <<<dimGrid, dimBlock>>> (devRes, WORK_TOTAL, threads);
        // Wait for the kernel to complete and check for errors
        checkCuda(cudaPeekAtLastError());
        checkCuda(cudaDeviceSynchronize());
    }
    devRes.copyTo(res);

    // Print the results
    for (int col = 0; col < WORK_TOTAL; ++col) {
        std::cout << res[col] << std::endl;
    }
}
