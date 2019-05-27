#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <array>
#include <iostream>
#include "CudaUtils.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <cstdint>

int main() {

//    thrust::device_vector<int> devSrc (std::vector<int> {1, 2, 3, 4, 5, 6, 7, 8, 9, 0});
//    thrust::device_vector<bool> devRes(devStr1.size());
//
//    auto begin = thrust::make_zip_iterator(thrust::make_tuple(devStr1.begin(), devStr2.begin()));
//    auto end = thrust::make_zip_iterator(thrust::make_tuple(devStr1.end(), devStr2.end()));
//
//    thrust::transform(begin, end, devRes.begin(), [] __device__ (auto pair) {
//        return thrust::get<0>(pair) == thrust::get<1>(pair);
//    });
//    auto count = thrust::reduce(devRes.begin(), devRes.end(), 0, [] __device__ (auto v1, auto v2) {
//        return v1 + v2;
//    });
//    thrust::host_vector<int> res = devRes;
//
//    // Wait for the kernel to complete and check for errors
//    checkCuda(cudaPeekAtLastError());
//    checkCuda(cudaDeviceSynchronize());
//
//    // Print the results
//    for (int col = 0; col < res.size(); ++col) {
//        std::cout << res[col] << std::endl;
//    }
//    std::cout << "Same prefix length is: " << std::to_string(count) << std::endl;
}
