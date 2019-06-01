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

#include "device_launch_parameters.h"
#include "CudaUtils.h"

#define WORK_WIDTH 6 // TODO liczba wariacji
#define WORK_HEIGHT 1
#define BLOCK_WIDTH 1
#define BLOCK_HEIGHT 1
#define WORK_TOTAL WORK_WIDTH * WORK_HEIGHT
#define RESULTS_TOTAL 10
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
__host__ __device__ Integral factorial(Integral const n) {
//    if constexpr (!std::is_unsigned<Integral>::value) {
    assert(n >= 0);
//    }
    if (n == 0) {
        return 1;
    }
    Integral result = 1;
    for (Integral i = 2; i <= n; ++i) {
        result *= i;
    }
    return result;
}

template <typename Integral, typename std::enable_if_t<std::is_integral<Integral>::value>* = nullptr>
__host__ __device__ Integral variationsCount(Integral const n, Integral const k) {
//    if constexpr (!std::is_unsigned<Integral>::value) {
        assert(n >= 0 && k >= 0);
//    }
    return factorial(n) / factorial(n - k);
}

template <typename T, typename Integral, typename std::enable_if_t<std::is_integral<Integral>::value>* = nullptr>
__host__ __device__ void computeVariation(T const* const input, Integral const n, Integral const k, Integral p, T* const output) {
    assert(n >= 0 && k >= 0 && k <= n && p >= 0 && p < variationsCount(n, k));

    // TODO possible optimization - use bitset
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
                printf("[GTID %d] Matches!\n", gtid);
                return true;
            }
        }
        ++sequencePtr;
    }
    printf("[GTID %d] Not matches!\n", gtid);
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

int main() {

    std::vector<Type> pattern = { 0,1,1,0 };
    CudaBuffer<Type, Type> devPattern(pattern);

    std::vector<Type> sequence = { 1,2,4,3,5,3,6,2,1 };
    CudaBuffer<Type, Type> devSequence(sequence);

    std::vector<Type> distinctPattern(pattern);
    distinctValues(distinctPattern);
    CudaBuffer<Type, Type> devDistinctPattern(distinctPattern);

    std::vector<Type> distinctSequence(sequence);
    distinctValues(distinctSequence);
    CudaBuffer<Type, Type> devDistinctSequence(distinctSequence);

    Type workAmount = variationsCount(distinctSequence.size(), distinctPattern.size());
    thrust::device_vector<Type> devOutputVariations(workAmount * sizeof(Type));
    thrust::device_vector<bool> devOutputFound(workAmount);

    dim3 dimBlock(workAmount, 1);
    dim3 dimGrid(
            static_cast<int>(ceilf(workAmount / static_cast<float>(dimBlock.x))),
            static_cast<int>(ceilf(1 / static_cast<float>(dimBlock.y)))
    );

    printf("Invoking with: Block(%d,%d), Grid(%d,%d)\n", dimBlock.x, dimBlock.y, dimGrid.x, dimGrid.y);
    runWithProfiler([&]{
        compute<Type, Type><<<dimGrid, dimBlock>>>(
                devSequence, devDistinctSequence, devPattern, devDistinctPattern,
                devOutputVariations.data().get(), devOutputFound.data().get(), workAmount);
    });

    auto variationsAmount = thrust::count(devOutputFound.begin(), devOutputFound.end(), true);
    thrust::device_vector<Type> devResult(variationsAmount * pattern.size());
    auto trans = thrust::make_transform_iterator(thrust::counting_iterator<Type>(0), FoundMatcher<Type>(pattern.size(), devOutputFound.data().get()));
    thrust::copy_if(devOutputVariations.begin(), devOutputVariations.end(), trans, devResult.begin(), _1 == true);
//    thrust::copy_if(devOutputVariations.begin(), devOutputVariations.end(), thrust::make_permutation_iterator(devOutputFound.begin(), thrust::make_transform_iterator(thrust::counting_iterator<int>(0), _1 / pattern.size() == 1)), devResult.begin(), _1 == 1);

    thrust::host_vector<Type> result(devResult);

    std::cout << "PRZED " << printContainer(devOutputVariations) << std::endl;
    std::cout << "PO " << printContainer(devResult) << std::endl;

}
