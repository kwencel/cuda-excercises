#ifdef __JETBRAINS_IDE__
    #define __host__
    #define __device__
    #define __shared__
    #define __constant__
    #define __global__

    // This is slightly mental, but gets it to properly index device function calls like __popc and whatever.
    #define __CUDACC__
    #include <device_functions.h>

    // These headers are all implicitly present when you compile CUDA with clang. Clion doesn't know that, so
    // we include them explicitly to make the indexer happy. Doing this when you actually build is, obviously,
    // a terrible idea :D
    #include <__clang_cuda_builtin_vars.h>
    #include <__clang_cuda_intrinsics.h>
    #include <__clang_cuda_math_forward_declares.h>
    #include <__clang_cuda_complex_builtins.h>
    #include <__clang_cuda_cmath.h>
#endif

#include <cstdio>
#include <stdexcept>
#include <functional>
#include <type_traits>
#include <cstdint>

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <iostream>
#include <thrust/device_vector.h>

#define checkCuda(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline cudaError_t cudaAssert(cudaError_t result, const char* file, int line, bool abort = true) {
    if (result != cudaSuccess){
        fprintf(stderr, "CUDA Error: \"%s\" in %s:%d\n", cudaGetErrorString(result), file, line);
        if (abort) {
            exit(result);
        }
    }
    return result;
}

void runWithProfiler(const std::function<void ()>& code) {
    checkCuda(cudaProfilerStart());
    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start));
    checkCuda(cudaEventCreate(&stop));
    checkCuda(cudaEventRecord(start, 0));
    code();
    checkCuda(cudaEventRecord(stop, 0));
    checkCuda(cudaEventSynchronize(stop));
    // Print kernel execution time
    float elapsedTime;
    checkCuda(cudaEventElapsedTime(&elapsedTime, start, stop));
    // Wait for the kernel to complete and check for errors
    checkCuda(cudaPeekAtLastError());
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaProfilerStop());
    printf("Kernel execution finished in %.3f ms\n", elapsedTime);
}

template <typename T, typename S = std::size_t>
struct GpuData {
    T const * const data;
    S const length;

    __host__ GpuData(thrust::device_vector<T> const& vector) : data(vector.data().get()), length(vector.size()) { }

    __device__ __host__ GpuData(T const * const data, S const length) : data(data), length(length) { }

    __device__ __host__ S getLength() const {
        return length;
    }

    __device__ __host__ auto getSize() const {
        return length * sizeof(T);
    }
};

template <typename T, typename S = std::size_t>
class CudaBuffer {
public:
    explicit CudaBuffer(std::size_t length) : length(length) {
        checkCuda(cudaMalloc((void**) &buffer, getSize()));
    }

    // Containers
    template <class Container, typename std::enable_if_t<std::is_same<T, typename std::remove_reference_t<Container>::value_type>::value>* = nullptr>
    explicit CudaBuffer(Container&& container) : CudaBuffer(container.size()) {
        copyFrom(container);
    }

    // Non-containers but types that can be copied by simple memcpy()
    template<typename std::enable_if_t<std::is_trivially_copyable<std::remove_reference_t<T>>::value>* = nullptr>
    explicit CudaBuffer(T&& copyable) : CudaBuffer(1) {
        copyFrom(&copyable, getSize());
    }

    ~CudaBuffer() {
        checkCuda(cudaFree(buffer));
    }

    CudaBuffer(const CudaBuffer& other) = delete;
    CudaBuffer& operator = (const CudaBuffer& other) = delete;
    CudaBuffer(CudaBuffer&& other) = delete;
    CudaBuffer& operator = (CudaBuffer&& other) = delete;

    operator T*() const {
        return getPointer();
    }

    operator GpuData<T, S>() const {
        return GpuData<T, S> { getPointer(), getLength() };
    }

    auto getSize() const {
        return getLength() * sizeof(T);
    }

    T* getPointer() const {
        return buffer;
    }

    S getLength() const {
        return length;
    }

    template<typename std::enable_if_t<std::is_trivially_copyable<std::remove_reference_t<T>>::value>* = nullptr>
    T getValue() {
        T temp;
        copyTo(&temp, sizeof(T));
        return temp;
    }

    void copyTo(void* target, std::size_t size) {
        assert(("Tried to copy to host more than the buffer length", size <= getSize()));
        checkCuda(cudaMemcpy(target, buffer, size, cudaMemcpyDeviceToHost));
    }

    template <typename Container>
    void copyTo(Container& target) {
        copyTo(target.data(), target.size() * sizeof(Container::value_type));
    }

    void copyFrom(const void* source, std::size_t size) {
        assert(("Tried to copy to device more than the buffer length", size <= getSize()));
        checkCuda(cudaMemcpy(buffer, source, size, cudaMemcpyHostToDevice));
    }

    template <typename Container>
    void copyFrom(const Container& source) {
        copyFrom(source.data(), source.size() * sizeof(Container::value_type));
    }

private:
    T* buffer;
    S length;
};
