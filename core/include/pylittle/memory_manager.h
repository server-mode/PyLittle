#pragma once
#include <cstddef>
#include <string>
#include <vector>

namespace pylittle {

class MemoryManager {
public:
    MemoryManager();
    ~MemoryManager();

    // Runtime capability
    bool has_cuda() const;

    void* alloc_device(size_t bytes);
    void free_device(void* ptr);

    void* alloc_host(size_t bytes, bool pinned = false);
    void free_host(void* ptr, bool pinned = false);

    // Native demo API: map a file and optionally prefetch some bytes into RAM
    int map_file(const std::string& path);
    bool prefetch(int handle, size_t offset, size_t bytes);
    void unmap(int handle);

    // Async prefetch/copy stubs (real async requires CUDA backend)
    // Returns false if not supported in current build.
    bool prefetch_async(int handle, size_t offset, size_t bytes, void* /*stream*/);

    // CUDA streams and async H2D copy helpers (no-ops without CUDA)
    void* create_stream();
    void destroy_stream(void* stream);
    bool copy_to_device_async(void* dst_device, const void* src_host, size_t bytes, void* stream);
    bool synchronize_stream(void* stream);

    std::string stats() const;

private:
    struct MapEntry {
        std::string path;
        size_t size = 0;
        size_t prefetched_bytes = 0;
        std::vector<char> buffer; // demo cache buffer
        bool valid = false;
    };
    std::vector<MapEntry> maps_;
};

} // namespace pylittle
