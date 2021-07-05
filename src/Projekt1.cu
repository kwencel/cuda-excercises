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
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/execution_policy.h>
#include <sstream>
#include <cooperative_groups.h>

#include "device_launch_parameters.h"
#include "CudaUtils.h"
using Type = int;
using namespace thrust::placeholders;

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

template <class DestContainer, class Source>
DestContainer parseTo(Source const& source) {
    using Target = typename DestContainer::value_type;
    std::istringstream is(source);
    return DestContainer(std::istream_iterator<Target>(is), std::istream_iterator<Target>());
}

template <typename T>
void removeDuplicates(std::vector<T>& data) {
    std::sort(data.begin(), data.end());
    auto iter = std::unique(data.begin(), data.end());
    data.resize(std::distance(data.begin(), iter));
}

template <typename T>
thrust::device_vector<T> removeDuplicates(thrust::device_vector<T> const& input) {
    thrust::device_vector<T> result(input);
    thrust::sort(thrust::device, result.begin(), result.end());
    auto iter = thrust::unique(thrust::device, result.begin(), result.end());
    result.resize(std::distance(result.begin(), iter));
    return result;
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

using Block = dim3;
using Grid = dim3;
struct LaunchParameters {
    int optimalBlockSize;
    int optimalGridSize;
    int workloadGridSize;

    /** Maximum occupancy - all blocks resident on SMs**/
    std::pair<Grid, Block> getOptimal() {
        return { dim3(optimalGridSize), dim3(optimalBlockSize) };
    }

    /** Real occupancy - minimal launch configuration to handle the workload **/
    std::pair<Grid, Block> getReal() {
        return { dim3(workloadGridSize), dim3(optimalBlockSize) };
    }

    /** Real occupancy bounded by the residency requirement. No non-resident blocks will be launched **/
    std::pair<Grid, Block> getRealResident() {
        return { dim3(std::min(optimalGridSize, workloadGridSize)), dim3(optimalBlockSize) };
    }
};

template <typename Kernel>
LaunchParameters calculateOptimalLaunchParameters(Kernel kernel, std::size_t dynamicSMemSize, int blockSizeLimit) {
    int blockSize;   // The launch configurator returned block size
    int minGridSize; // The minimum grid size needed to achieve the maximum occupancy for a full device launch
    int gridSize;    // The actual grid size needed, based on input size
    int maxActiveBlocks;
    int device;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, dynamicSMemSize, blockSizeLimit);
    gridSize = (blockSizeLimit + blockSize - 1) / blockSize;
    printf("[CUDA] Optimal block size: %d, grid size: %d\n", blockSize, minGridSize);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, kernel, blockSize, dynamicSMemSize);
    cudaDeviceProp props;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&props, device);
    float occupancy = (maxActiveBlocks * blockSize / props.warpSize) /(float)(props.maxThreadsPerMultiProcessor / props.warpSize);
    printf("[CUDA] Theoretical occupancy assuming launching with optimal grid %d: %f\n", minGridSize, occupancy);
    printf("[CUDA] Theoretical occupancy assuming launching with neccessary grid %d to handle all the work: %f\n", gridSize, occupancy * ((float) gridSize / minGridSize));

    return LaunchParameters { blockSize, minGridSize, gridSize };
}

__device__ void logLaunchParameters(char const* const kernelName) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gtid == 0) {
        printf("[CUDA] Invoking %s with: Block(%d,%d), Grid(%d,%d)\n", kernelName, blockDim.x, blockDim.y, gridDim.x, gridDim.y);
    }
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
    if (k == 0) {
        return 1;
    }
    std::size_t result = n - k + 1;
    for (std::size_t i = result + 1; i <= n; ++i) {
        result *= i;
    }
    return result;
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

    bool* removed = new bool[n];
    for (Integral i = 0; i < n; ++i) {
        removed[i] = false;
    }

    for (Integral x = 0; x < k; ++x) {
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
                                           T const* const variation, T* const output) {

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

template <typename T, typename S>
__host__ __device__ bool checkPattern(GpuData<T, S> const& sequence, GpuData<T, S> const& pattern) {
    T const* sequencePtr = sequence.data;
    T const* patternPtr = pattern.data;
    while (sequencePtr - sequence.data < sequence.length) {
        if (*patternPtr == *sequencePtr) {
            ++patternPtr;
            if (patternPtr - pattern.data == pattern.length) {
//                printf("[WorkNo %d] Matches! %d %d %d %d %d %d %d\n", workNo, pattern.data[0], pattern.data[1], pattern.data[2], pattern.data[3], pattern.data[4], pattern.data[5], pattern.data[6]);
                return true;
            }
        }
        ++sequencePtr;
    }
//    printf("[WorkNo %d] Not matches!\n", gtid);
    return false;
}

template <typename T, typename S>
__device__ bool compute(GpuData<T, S> const& sequence, GpuData<T, S> const& distinctSequence, GpuData<T, S> const& pattern,
                        GpuData<T, S> const& distinctPattern, S workNo, T* const outputVariations, bool* const outputFound) {

    T* variation = new T[distinctPattern.length];
    // Compute the variation to be checked by this thread
    computeVariation(distinctSequence.data, distinctSequence.length, distinctPattern.length, workNo, variation);
    T* finalPattern = outputVariations + (workNo * pattern.length);
    // Assign computed values to the pattern
    substitutePattern(pattern, distinctPattern, variation, finalPattern);
    delete[] variation;
    bool found = checkPattern(sequence, GpuData<T, S> { finalPattern, pattern.length });
    outputFound[workNo] = found;
    return found;
}


template <typename T, typename S>
__global__ void computeAll(GpuData<T, S> const sequence, GpuData<T, S> const distinctSequence,
                           GpuData<T, S> const pattern, GpuData<T, S> const distinctPattern, S const workAmount,
                           T* const outputVariations, bool* const outputFound) {

    logLaunchParameters("computeAll");
    int workNo = blockIdx.x * blockDim.x + threadIdx.x;
    while (workNo < workAmount) {
        compute<T, S>(sequence, distinctSequence, pattern, distinctPattern, workNo, outputVariations, outputFound);
        workNo += (blockDim.x * gridDim.x);
    }
}

template <typename T, typename S>
__global__ void computeAnyWrapper(GpuData<T, S> const sequence, GpuData<T, S> const distinctSequence, GpuData<T, S> const pattern,
                                  GpuData<T, S> const distinctPattern, S const workAmount, S const iteration,
                                  T* const outputVariations, bool* const outputFound, volatile bool* const anyFound) {

    logLaunchParameters("computeAnyWrapper");
    int const workNo = (blockIdx.x * blockDim.x + threadIdx.x) + (iteration * blockDim.x * gridDim.x);
    if (workNo < workAmount) {
        bool found = compute<T, S>(sequence, distinctSequence, pattern, distinctPattern, workNo,
                                   outputVariations, outputFound);
        if (found) {
            *anyFound = true;
        }
    }
}

template <typename T, typename S>
__global__ void computeAny(GpuData<T, S> const sequence, GpuData<T, S> const distinctSequence, GpuData<T, S> const pattern,
                           GpuData<T, S> const distinctPattern, S const workAmount, int const gridSize, int const blockSize,
                           T* const outputVariations, bool* const outputFound, volatile bool* const anyFound) {

    logLaunchParameters("computeAny");
    assert(blockDim.x == 1 && gridDim.x == 1);
    S iterationNo = 0;
    do {
        computeAnyWrapper<T, S><<<gridSize, blockSize>>>(sequence, distinctSequence, pattern, distinctPattern, workAmount,
                                                         iterationNo, outputVariations, outputFound, anyFound);
        cudaDeviceSynchronize();
        printf("[KERNEL] Checked range [%d, %d]\n", (iterationNo) * blockSize * gridSize, (iterationNo + 1) * blockSize * gridSize - 1);
        if (*anyFound) {
            printf("[KERNEL] Found matching pattern!\n");
            return;
        }
        ++iterationNo;
    } while (iterationNo * blockSize * gridSize < workAmount);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Too few arguments. Usage: " << argv[0] << " <[any/all]> <pattern> <sequence>."
                     "Pattern and sequence should be quoted lists of integers seperated by whitespaces.";
        exit(1);
    }

    std::string const mode(argv[1]);
    auto const devPattern = parseTo<thrust::device_vector<Type>>(argv[2]);
    auto const devSequence = parseTo<thrust::device_vector<Type>>(argv[3]);
    assert(devPattern.size() <= devSequence.size());
    auto const devDistinctPattern = removeDuplicates(devPattern);
    auto const devDistinctSequence = removeDuplicates(devSequence);

    Type workAmount = variationsCount(devDistinctSequence.size(), devDistinctPattern.size());
    std::cout << "[INFO] Variations to check: " << workAmount << std::endl;

    thrust::device_vector<Type> devOutputVariations(workAmount * devPattern.size());
    thrust::device_vector<bool> devOutputFound(workAmount);

    if (mode == "all") {
        auto launchParameters = calculateOptimalLaunchParameters(computeAll<Type, Type>, 0, workAmount).getReal();
        dim3 dimGrid = launchParameters.first;
        dim3 dimBlock = launchParameters.second;
        /** ---------------------------------------- Launch the kernel ---------------------------------------- **/
        runWithProfiler([&] {
            computeAll<Type, Type> <<<dimGrid, dimBlock>>> (
                    devSequence, devDistinctSequence, devPattern, devDistinctPattern, workAmount,
                    devOutputVariations.data().get(), devOutputFound.data().get()
            );
        });
    }
    else if (mode == "any") {
        auto launchParameters = calculateOptimalLaunchParameters(computeAll<Type, Type>, 0, workAmount).getRealResident();
        dim3 dimGrid = launchParameters.first;
        dim3 dimBlock = launchParameters.second;
        CudaBuffer<bool> anyFound(false);
        /** ---------------------------------------- Launch the kernel ---------------------------------------- **/
        runWithProfiler([&] {
            computeAny<Type, Type> <<<1, 1>>> (
                    devSequence, devDistinctSequence, devPattern, devDistinctPattern, workAmount, dimGrid.x, dimBlock.x,
                    devOutputVariations.data().get(), devOutputFound.data().get(), anyFound
            );
        });
        std::cout << "Was any pattern found: " << (anyFound.getValue() ? "YES" : "NO") << std::endl;
    }
    else {
        throw std::runtime_error("Wrong argument - allowed [all/any]");
    }

    /** ------------------------------------------- Process results ------------------------------------------- **/
    auto variationsCorrect = thrust::count(devOutputFound.begin(), devOutputFound.end(), true);
    thrust::device_vector<Type> devResult(variationsCorrect * devPattern.size());
    auto predicateSource = thrust::make_transform_iterator(thrust::counting_iterator<Type>(0),
                                                           FoundMatcher<Type>(devPattern.size(), devOutputFound.data().get()));
    thrust::copy_if(devOutputVariations.begin(), devOutputVariations.end(), predicateSource, devResult.begin(), _1 == true);

    thrust::host_vector<Type> result(devResult);
    std::cout << "Found " << variationsCorrect << " patterns" << std::endl;
    for (std::size_t resultIdx = 0; resultIdx < result.size(); resultIdx += devPattern.size()) {
        for (std::size_t i = 0; i < devPattern.size(); ++i) {
            std::cout << result[resultIdx + i] << " ";
        }
        std::cout << std::endl;
    }
}
