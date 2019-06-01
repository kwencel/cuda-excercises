#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <array>
#include <cmath>
#include <vector>
#include <type_traits>
#include <cassert>
#include <set>
#include <algorithm>
#include <string>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <sstream>

#include "device_launch_parameters.h"
#include "CudaUtils.h"
using Type = int;

template <typename Container>
std::string printContainer(Container const& container) {
    if (container.empty()) {
        return "{}";
    }
    std::string result = "{" + std::to_string(*(container.begin()));
    if (container.size() == 1) {
        return result + "}";
    }
    for (auto it = std::next(container.begin()); it != container.end(); ++it) {
        result += "," + std::to_string(*it);
    }
    result += '}';
    return result;
}


template <typename Integral, typename std::enable_if_t<std::is_integral<Integral>::value>* = nullptr>
__host__ __device__ std::size_t factorial(Integral const n) {
    #if __cplusplus >= 201703L
    if constexpr (!std::is_unsigned<Integral>::value) {
    #endif
        assert(n >= 0);
    #if __cplusplus >= 201703L
    }
    #endif
    if (n == 0) {
        return 1;
    }
    std::size_t result = 1;
    for (Integral i = 2; i <= n; ++i) {
        result *= i;
    }
    assert(result > 0 && "Overflow detected!");
    return result;
}

template <typename Integral, typename std::enable_if_t<std::is_integral<Integral>::value>* = nullptr>
__host__ __device__ std::size_t variationsCount(Integral const n, Integral const k) {
    #if __cplusplus >= 201703L
    if constexpr (!std::is_unsigned<Integral>::value) {
    #endif
        assert(n >= 0 && k >= 0);
    #if __cplusplus >= 201703L
    }
    #endif
    assert(n >= k);
    return factorial(n) / factorial(n - k);
}

template <typename T, typename Integral, typename std::enable_if_t<std::is_integral<Integral>::value>* = nullptr>
__host__ __device__ void computeVariation(T const* const input, Integral const n, Integral const k, Integral p, T* const output) {
    #if __cplusplus >= 201703L
    if constexpr (!std::is_unsigned<Integral>::value) {
    #endif
        assert(n >= 0 && k >= 0 && k <= n && p >= 0 && p < variationsCount(n, k));
    #if __cplusplus >= 201703L
    }
    #endif

    bool* removed = new bool[k];
    for (Integral i = 0; i < k; ++i) {
        removed[i] = false;
    }

    for (Integral x = 0; x <= k - 1; ++x) {
        Integral v = variationsCount(n - x - 1, k - x - 1);
        Integral t = p / v;

        for (Integral i = 0; i <= t; ++i) {
            if (removed[i]) {
                ++t;
            }
        }

        output[x] = input[t];
        removed[t] = true;
        p = p % v;
    }

    delete[] removed;
}

template <typename T, typename S>
__host__ __device__ void substitutePattern(GpuData<T, S> const& pattern, GpuData<T, S> const& distinctPattern,
                                           T const * const variation, T* const output) {

    for (S patternIndex = 0; patternIndex < pattern.length; ++patternIndex) {
        T currentPatternSymbol = pattern.data[patternIndex];
        // Find the substitution
        for (S distinctPatternIndex = 0; distinctPatternIndex < distinctPattern.length; ++distinctPatternIndex) {
            T currentDistinctPatternSymbol = distinctPattern.data[distinctPatternIndex];
            if (currentDistinctPatternSymbol == currentPatternSymbol) {
                output[patternIndex] = variation[distinctPatternIndex];
                break;
            }
        }
    }
}

template <typename T>
void distinctValues(std::vector<T>& data) {
    std::sort(data.begin(), data.end());
    auto iter = std::unique(data.begin(), data.end());
    data.resize(std::distance(data.begin(), iter));
}

template <typename T, typename S>
__host__ __device__ bool checkPattern(GpuData<T, S> const& sequence, GpuData<T, S> const& pattern) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;

    T const* sequencePtr = sequence.data;
    T const* patternPtr = pattern.data;
    while (sequencePtr - sequence.data < sequence.length) {
        if (*patternPtr == *sequencePtr) {
            ++patternPtr;
            if (patternPtr - pattern.data == pattern.length) {
                printf("[GTID %d] Matches! %d %d %d %d %d %d %d %d %d %d %d\n", gtid, patternPtr[0], patternPtr[1], patternPtr[2], patternPtr[3], patternPtr[4], patternPtr[5], patternPtr[6], patternPtr[7], patternPtr[8], patternPtr[9], patternPtr[10]);
                return true;
            }
        }
        ++sequencePtr;
    }
//    printf("[GTID %d] Not matches!\n", gtid);
    return false;
}

template <typename T, typename S>
__global__ void compute(GpuData<T, S> const sequence, GpuData<T, S> const distinctSequence, GpuData<T, S> const pattern,
                        GpuData<T, S> const distinctPattern, T* outputVariations, bool* outputFound, S workAmount) {

    int const gtid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gtid >= workAmount) {
        return;
    }

    T* variation = new T[distinctPattern.length];
    // Compute the variation to be checked by this thread
    computeVariation(distinctSequence.data, distinctSequence.length, distinctPattern.length, gtid, variation);
    T* finalPattern = outputVariations + (gtid * pattern.length);
    // Assign computed values to the pattern
    substitutePattern(pattern, distinctPattern, variation, finalPattern);
    outputFound[gtid] = checkPattern(sequence, GpuData<T, S> { finalPattern, pattern.length });
    // If found a match, copy the substituted pattern to the output array
    if (outputFound[gtid]) {
        for (S i = 0; i < pattern.length; ++i) {
            outputVariations[(gtid * pattern.length) + i] = finalPattern[i];
        }
    }
    delete[] variation;
}

template <typename S>
struct FoundMatcher : public thrust::unary_function<S, bool> {
    FoundMatcher(S const patternSize, bool const * const found) : patternSize(patternSize), found(found) { }

    __device__ __host__  bool operator()(S index) {
        return found[index / patternSize];
    }

    S const patternSize;
    bool const * const found;
};

using namespace thrust::placeholders;

template <class DestContainer, class Source>
DestContainer parseTo(Source const& source) {
    using Target = typename DestContainer::value_type;
    std::istringstream is(source);
    return DestContainer(std::istream_iterator<Target>(is), std::istream_iterator<Target>());
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Too few arguments. Usage: " << argv[0] << " <[any/all]> <pattern> <sequence>."
                     "Pattern and sequence should be quoted lists of integers seperated by whitespaces.";
        exit(1);
    }

    std::string mode = argv[1];
    auto pattern = parseTo<std::vector<Type>>(argv[2]);
    CudaBuffer<Type, Type> devPattern(pattern);
    auto sequence = parseTo<std::vector<Type>>(argv[3]);
    CudaBuffer<Type, Type> devSequence(sequence);
    assert(pattern.size() <= sequence.size());

    std::vector<Type> distinctPattern(pattern);
    distinctValues(distinctPattern);
    CudaBuffer<Type, Type> devDistinctPattern(distinctPattern);

    std::vector<Type> distinctSequence(sequence);
    distinctValues(distinctSequence);
    CudaBuffer<Type, Type> devDistinctSequence(distinctSequence);

    Type workAmount = variationsCount(distinctSequence.size(), distinctPattern.size());
    std::cout << "[INFO] Variations to check: " << workAmount << std::endl;

    thrust::device_vector<Type> devOutputVariations(workAmount * pattern.size());
    thrust::device_vector<bool> devOutputFound(workAmount);

    /** ---------------------------- Calculate optimal kernel launch parameters ---------------------------- **/
    int blockSize;   // The launch configurator returned block size
    int minGridSize; // The minimum grid size needed to achieve the maximum occupancy for a full device launch
    int gridSize;    // The actual grid size needed, based on input size
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, compute<Type, Type>, 0, workAmount);

    gridSize = (workAmount + blockSize - 1) / blockSize;
    printf("[CUDA] Optimal block size: %d, grid size: %d\n", blockSize, minGridSize);
    dim3 dimBlock(blockSize, 1);
    dim3 dimGrid(gridSize, 1);
    printf("[CUDA] Invoking with: Block(%d,%d), Grid(%d,%d)\n", dimBlock.x, dimBlock.y, dimGrid.x, dimGrid.y);

    // Calculate theoretical occupancy
    int maxActiveBlocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, compute<Type, Type>, blockSize, 0);
    std::cout << maxActiveBlocks << std::endl;
    int device;
    cudaDeviceProp props;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&props, device);
    float occupancy = (maxActiveBlocks * blockSize / props.warpSize) /(float)(props.maxThreadsPerMultiProcessor / props.warpSize);
    printf("[CUDA] Theoretical occupancy of one SM: %f\n", occupancy);
    printf("[CUDA] Throretical occupancy of whole GPU: %f\n", occupancy * ((float) gridSize / minGridSize));

    if (mode == "all") {
        /** ---------------------------------------- Launch the kernel ---------------------------------------- **/
        runWithProfiler([&] {
            compute<Type, Type> <<<gridSize, blockSize>>> (
                    devSequence, devDistinctSequence, devPattern, devDistinctPattern,
                    devOutputVariations.data().get(), devOutputFound.data().get(), workAmount
            );
        });

        /** ----------------------------------------- Process results ----------------------------------------- **/
        auto variationsCorrect = thrust::count(devOutputFound.begin(), devOutputFound.end(), true);
        thrust::device_vector<Type> devResult(variationsCorrect * pattern.size());
        auto predicateSource = thrust::make_transform_iterator(thrust::counting_iterator<Type>(0),
                                                               FoundMatcher<Type>(pattern.size(), devOutputFound.data().get()));
        thrust::copy_if(devOutputVariations.begin(), devOutputVariations.end(), predicateSource, devResult.begin(), _1 == true);

//        thrust::copy_if(
//                devOutputVariations.begin(),
//                devOutputVariations.end(),
//                thrust::make_permutation_iterator(devOutputFound.begin(),
//                                                  thrust::make_transform_iterator(thrust::counting_iterator<int>(0),
//                                                                                  _1 / pattern.size() == 1)),
//                devResult.begin(), _1 == 1
//        );

        thrust::host_vector<Type> result(devResult);
        std::cout << "Found " << variationsCorrect << " patterns" << std::endl;
        for (std::size_t resultIdx = 0; resultIdx < result.size(); resultIdx += pattern.size()) {
            for (std::size_t i = 0; i < pattern.size(); ++i) {
                std::cout << result[resultIdx + i] << " ";
            }
            std::cout << std::endl;
        }
    } else if (mode == "any") {

    }

}
