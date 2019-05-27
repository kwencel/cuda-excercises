#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <array>
#include <iostream>
#include "CudaUtils.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "util/StringConcat.h"
#include <cstdint>

template <typename T, typename std::enable_if<sizeof(T) == 4>::type* = nullptr>
struct BitsCounter : public thrust::unary_function<T, std::size_t> {
    __host__ __device__ uint8_t operator() (T v) {
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        return ((v + (v >> 4) & 0xF0F0F0F) * 0x1010101) >> 24;
    }
};

int main() {

    thrust::host_vector<int> src(std::vector<int> { 10, 25, 4, -2, 15, 35, 27, 99, 1 });
    thrust::device_vector<int> devSrc = src;
    thrust::device_vector<uint8_t> devRes(devSrc.size());

    thrust::transform(devSrc.begin(), devSrc.end(), devRes.begin(), BitsCounter<int>());
    thrust::host_vector<int> res = devRes;

    // Wait for the kernel to complete and check for errors
    checkCuda(cudaPeekAtLastError());
    checkCuda(cudaDeviceSynchronize());

    // Print the results
    for (int col = 0; col < res.size(); ++col) {
        std::cout << res[col] << std::endl;
    }
}
