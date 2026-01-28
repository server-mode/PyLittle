#pragma once
#include <cstddef>
#include <string>

namespace pylittle {

class IBackend {
public:
    virtual ~IBackend() = default;
    virtual const char* name() const = 0;

    virtual void* alloc(size_t bytes) = 0;
    virtual void free(void* ptr) = 0;

    virtual bool memcpy_async(void* dst, const void* src, size_t bytes) = 0;
    virtual bool gemm(const float* A, const float* B, float* C, int M, int N, int K) = 0;
};

} // namespace pylittle
