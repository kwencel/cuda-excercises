#include <cstdio>
#include <stdexcept>

#define checkCuda(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline cudaError_t cudaAssert(cudaError_t result, const char *file, int line, bool abort = true) {
    if (result != cudaSuccess){
        fprintf(stderr, "CUDA Error: \"%s\" in %s:%d\n", cudaGetErrorString(result), file, line);
        if (abort) {
            exit(result);
        }
    }
    return result;
}

template <typename T>
class CudaBuffer {
public:
    CudaBuffer(std::size_t length) : length(length) {
        checkCuda(cudaMalloc((void**) &buffer, getSize()));
    }

    ~CudaBuffer() {
        checkCuda(cudaFree(buffer));
    }

//    T& operator* () {
//        return *buffer;
//    }
//
//    T* operator-> () {
//        return buffer;
//    }

    operator T*() {
        return buffer;
    }

    std::size_t getLength() const {
        return length;
    }

    std::size_t getSize() const {
        return getLength() * sizeof(T);
    }

    void copyTo(void* target, std::size_t size) {
        if (size > getSize()) {
            throw std::runtime_error("Tried to copy to host more than the buffer length");
        }
        checkCuda(cudaMemcpy(target, buffer, size, cudaMemcpyDeviceToHost));
    }

    template <typename Container>
    void copyTo(Container& target) {
        copyTo(target.data(), target.size() * sizeof(Container::value_type));
    }

    void copyFrom(const void* source, std::size_t size) {
        if (size > getSize()) {
            throw std::runtime_error("Tried to copy to device more than the buffer length");
        }
        checkCuda(cudaMemcpy(buffer, source, size, cudaMemcpyHostToDevice));
    }

    template <typename Container>
    void copyFrom(const Container& source) {
        copyFrom(source.data(), source.size() * sizeof(Container::value_type));
    }

private:
    T* buffer;
    std::size_t length;
};
