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

#include "device_launch_parameters.h"
#include "CudaUtils.h"

#define WORK_WIDTH 6 // TODO liczba wariacji
#define WORK_HEIGHT 1
#define BLOCK_WIDTH 1
#define BLOCK_HEIGHT 1
#define WORK_TOTAL WORK_WIDTH * WORK_HEIGHT
#define RESULTS_TOTAL 10


template <typename Integral, typename std::enable_if_t<std::is_integral<Integral>::value>* = nullptr>
__host__ __device__ Integral factorial(Integral const n) {
    assert(n >= 0);
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
    assert(n >= 0 && k >= 0);
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
}

template <typename T>
__host__ __device__ void substituteSequence(GpuData<T> const& pattern, GpuData<T> const& distinctPattern,
                                            T const * const variation, T* const output) {

    for (decltype(pattern.length) patternIndex = 0; patternIndex < pattern.length; ++patternIndex) {
        T currentPatternSymbol = pattern.data[patternIndex];
        // Find the substitution
        for (decltype(pattern.length) distinctPatternIndex = 0; distinctPatternIndex < distinctPattern.length; ++distinctPatternIndex) {
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

template <typename T>
__host__ __device__ void checkPattern(GpuData<T> const& sequence, GpuData<T> const& pattern) {
//    __shared__ char sharedSeq[SEQUENCE_SIZE];
//    int tid = threadIdx.x;
//    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
//    if (gtid < SEQUENCE_SIZE) {
//        sharedSeq[tid] = sequence[gtid];
//    }

    int gtid = blockIdx.x * blockDim.x + threadIdx.x;

    T const* sequencePtr = sequence.data;
    T const* patternPtr = pattern.data;
    while (sequencePtr - sequence.data < sequence.length) {
        if (*patternPtr == *sequencePtr) {
            if (++patternPtr - pattern.data == pattern.length) {
                printf("[GTID %d] Matches!\n", gtid);
                return;
            }
        }
        ++sequencePtr;
    }
    printf("[GTID %d] Not matches!\n", gtid);
}

template <typename T>
__global__ void compute(GpuData<T> const sequence, GpuData<T> const distinctSequence, GpuData<T> const pattern,
                        GpuData<T> const distinctPattern) {

    int const gtid = blockIdx.x * blockDim.x + threadIdx.x;

    T* variation = new T[variationsCount(distinctSequence.length, distinctPattern.length)]; // TODO calculate beforehand
    computeVariation(distinctSequence.data, distinctSequence.length, distinctPattern.length, gtid, variation);
    T* finalPattern = new T[pattern.length];
    substituteSequence(pattern, distinctPattern, variation, finalPattern);
    checkPattern(sequence, GpuData<T> { finalPattern, pattern.length }); // Return something

}

int main() {

//    std::vector<int> sequence = { 0,3,9,8,5,7,3,0,9,4,8,5,0,9,4,8,7,3,2,4,0,9,5,8,3,2,7,5,0,9,2,3,8,7,5,9,3,2,8,5,7,3,7,5,6,3,9,8,5,6,9,8 };
    std::vector<int> pattern = { 0,1,1,0 };
    CudaBuffer<int> devPattern(pattern);

    std::vector<int> sequence = { 1,2,4,3,5,3,6,2,1 };
    CudaBuffer<int> devSequence(sequence);

    std::vector<int> distinctPattern(pattern);
    distinctValues(distinctPattern);
    CudaBuffer<int> devDistinctPattern(distinctPattern);

    std::vector<int> distinctSequence(sequence);
    distinctValues(distinctSequence);
    CudaBuffer<int> devDistinctSequence(distinctSequence);

    int workAmount = variationsCount(distinctSequence.size(), distinctPattern.size());
    dim3 dimBlock(workAmount, 1);
    dim3 dimGrid(
            static_cast<int>(ceilf(workAmount / static_cast<float>(dimBlock.x))),
            static_cast<int>(ceilf(1 / static_cast<float>(dimBlock.y)))
    );

    printf("Invoking with: Block(%d,%d), Grid(%d,%d)\n", dimBlock.x, dimBlock.y, dimGrid.x, dimGrid.y);
    runWithProfiler([&]{
        compute<<<dimGrid, dimBlock>>>(devSequence.getStruct(), devDistinctSequence.getStruct(),
                                       devPattern.getStruct(), devDistinctPattern.getStruct());
    });
    checkCuda(cudaPeekAtLastError());
    checkCuda(cudaDeviceSynchronize());

//    char a[] = { 'a','b','c' };
//    char out[6];
//
//    computeVariation(a, 3, 2, 0, out);
//    std::cout << out[0] << out[1] << std::endl;



    return 0;
}
