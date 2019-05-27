#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <array>
#include <iostream>
#include "CudaUtils.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "util/StringConcat.h"
#include <cstdint>

int main() {

    thrust::host_vector<int> src(std::vector<int> { 10, 25, 4, -2, 15, 35, 27, 99, 1 });
    thrust::device_vector<int> devSrc = src;
    thrust::device_vector<std::size_t> devRes(devSrc.size());

    auto isEven = [] __device__ (auto x) { return x % 2 == 0; };
    thrust::transform(devSrc.begin(), devSrc.end(), devRes.begin(), isEven);
    int count = thrust::reduce(devRes.begin(), devRes.end(), 0, thrust::plus<int>());
    thrust::host_vector<int> res = devRes;

    // Wait for the kernel to complete and check for errors
    checkCuda(cudaPeekAtLastError());
    checkCuda(cudaDeviceSynchronize());

    // Print the results
    for (int col = 0; col < res.size(); ++col) {
        std::cout << res[col] << std::endl;
    }
    std::cout << "Even count is: " << count << std::endl;
}
