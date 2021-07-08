#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <array>
#include <iostream>
#include "CudaUtils.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include "util/StringConcat.h"
#include <cstdint>

int main() {

    thrust::host_vector<int> src(std::vector<int> { 10, 25, 4, -2, 15, 35, 27, 99, 1 });
    thrust::host_vector<int> res;
    int sum;

    runWithProfiler([&]() {
        thrust::device_vector<int> devSrc = src;
        thrust::device_vector <uint8_t> devRes(devSrc.size());

        thrust::transform(devSrc.begin(), devSrc.end(), devRes.begin(), [] __device__(auto v) { return __popc(v); });
        res = devRes;
        sum = thrust::reduce(devRes.begin(), devRes.end(), 0, thrust::plus<int>());
    });

    // Print the results
    for (int col = 0; col < res.size(); ++col) {
        std::cout << res[col] << std::endl;
    }
    std::cout << "Sum is: " << sum << std::endl;
}
