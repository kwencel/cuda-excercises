
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

const int N = 1024;
const int blocksize = 16;

__global__ void addMatrix(float *a, float *b, float *c, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int index = i + j * N;
    if (i < N && j < N) {
        c[index] = a[index] + b[index];
    }
}

int main() {
    auto* a = new float[N * N];
    auto* b = new float[N * N];
    auto* c = new float[N * N];

    for (int i = 0; i < N * N; ++i) {
        a[i] = 1.0f;
        b[i] = 3.5f;
    }

    float *ad, *bd, *cd;
    const int size = N * N * sizeof(float);
    cudaMalloc((void**)&ad, size);
    cudaMalloc((void**)&bd, size);
    cudaMalloc((void**)&cd, size);

    //    N / bs + ((N % bs) != 0);

    cudaMemcpy(ad, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(bd, b, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(blocksize, blocksize);
    dim3 dimGrid((N / dimBlock.x) + 1, (N / dimBlock.y) + 1);
    addMatrix <<<dimGrid, dimBlock >>> (ad, bd, cd, N);

    cudaMemcpy(c, cd, size, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    for (int i = 0; i < N * N; i++) {
        std::cout << i << c[i] << std::endl;
    }

    cudaFree(ad);
    cudaFree(bd);
    cudaFree(cd);
    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}
