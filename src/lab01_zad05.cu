#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <array>
#include <iostream>
#include "CudaUtils.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "util/StringConcat.h"

#define WORK_WIDTH 3
#define WORK_HEIGHT 1
#define BLOCK_WIDTH 1
#define BLOCK_HEIGHT 1
#define WORK_TOTAL WORK_WIDTH * WORK_HEIGHT

dim3 dimBlock(BLOCK_WIDTH, BLOCK_HEIGHT);
dim3 dimGrid(ceil(WORK_WIDTH / (float) dimBlock.x), ceil(WORK_HEIGHT / (float) dimBlock.y));


struct QuadraticSource {
    float a;
    float b;
    float c;
};

struct QuadraticResult {
    int resultsAmount;
    float firstRes;
    float secondRes;
    float extrema;

    std::string str() const {
        return util::concat("resultsAmount: ", resultsAmount, "\tfirstRes: ", firstRes, "\tsecondRes: ", secondRes, "\textrema: ", extrema);
    }
};

template <class CharT, class Traits, class T>
typename std::enable_if<std::is_same<CharT, char>::value, std::basic_ostream<CharT, Traits>&>::type
operator << (std::basic_ostream<CharT, Traits>& os, const T& t) {
    return os << t.str();
}

__global__ void compute(QuadraticSource* devSrc, QuadraticResult* devRes, int howMany) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= howMany) {
        return;
    }

    QuadraticSource src = devSrc[idx];
    float a = src.a;
    float b = src.b;
    float c = src.c;
    float delta = powf(b, 2) - (4 * a * c);

    QuadraticResult result;
    if (delta > 0) {
        result.resultsAmount = 2;
        result.firstRes = (-b - sqrtf(delta)) / (2 * a);
        result.secondRes = (-b + sqrtf(delta)) / (2 * a);
    } else if (delta == 0) {
        result.resultsAmount = 1;
        result.firstRes = (-b - sqrtf(delta)) / (2 * a);
        result.secondRes = result.firstRes;
    } else {
        result.resultsAmount = 0;
        result.firstRes = 0;
        result.secondRes = 0;
    }
    result.extrema = -delta / (4 * a);
    devRes[idx] = result;
}

int main() {
    printf("Kernel will be invoked with: Block(%d,%d), Grid(%d,%d)\n", dimBlock.x, dimBlock.y, dimGrid.x, dimGrid.y);

    thrust::host_vector<QuadraticSource> src (std::vector<QuadraticSource> {QuadraticSource{1, 2, -3}, QuadraticSource{4, 5, 6}, QuadraticSource{1, 0, 0}});
    thrust::device_vector<QuadraticSource> devSrc = src;
    thrust::device_vector<QuadraticResult> devRes(devSrc.size());

    compute <<<dimGrid, dimBlock>>> (devSrc.data().get(), devRes.data().get(), devSrc.size());
    thrust::host_vector<QuadraticResult> res = devRes;

    // Wait for the kernel to complete and check for errors
    checkCuda(cudaPeekAtLastError());
    checkCuda(cudaDeviceSynchronize());

    // Print the results
    for (int col = 0; col < WORK_TOTAL; ++col) {
        std::cout << res[col] << std::endl;
    }
}
