#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <array>
#include <iostream>
#include "CudaUtils.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cstdint>

int main() {

    thrust::host_vector<int> src(std::vector<int> { 10, 25, 4, -2, 15, 35, 27, 99, 1 });
    thrust::host_vector<int> res;
    int count;

    runWithProfiler([&]() {
        thrust::device_vector<int> devSrc = src;
        thrust::device_vector<std::size_t> devRes(devSrc.size());

        auto isEven = [] __device__ (auto x) { return x % 2 == 0; };
        thrust::transform(devSrc.begin(), devSrc.end(), devRes.begin(), isEven);
        count = thrust::reduce(devRes.begin(), devRes.end(), 0, thrust::plus<int>());
        res = devRes;
    });

    // Print the results
    for (int col = 0; col < res.size(); ++col) {
        std::cout << res[col] << std::endl;
    }
    std::cout << "Even count is: " << count << std::endl;
}
