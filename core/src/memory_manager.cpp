#include "pylittle/memory_manager.h"
#include <fstream>
#include <sstream>
#include <cstdlib>

#if defined(_WIN32)
#define NOMINMAX
#endif

#ifdef PYLITTLE_WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace pylittle {

MemoryManager::MemoryManager() {}
MemoryManager::~MemoryManager() {}

bool MemoryManager::has_cuda() const {
#ifdef PYLITTLE_WITH_CUDA
	int n = 0;
	auto st = cudaGetDeviceCount(&n);
	return (st == cudaSuccess) && (n > 0);
#else
	return false;
#endif
}

void* MemoryManager::alloc_device(size_t bytes) {
#ifdef PYLITTLE_WITH_CUDA
	void* d = nullptr;
	if (cudaMalloc(&d, bytes) == cudaSuccess) return d;
	return nullptr;
#else
	return std::malloc(bytes);
#endif
}
void MemoryManager::free_device(void* ptr) {
#ifdef PYLITTLE_WITH_CUDA
	if (ptr) cudaFree(ptr);
#else
	std::free(ptr);
#endif
}

void* MemoryManager::alloc_host(size_t bytes, bool pinned) {
#ifdef PYLITTLE_WITH_CUDA
	if (pinned) {
		void* p = nullptr;
		if (cudaHostAlloc(&p, bytes, cudaHostAllocDefault) == cudaSuccess) return p;
		return nullptr;
	}
#endif
	return std::malloc(bytes);
}
void MemoryManager::free_host(void* ptr, bool pinned) {
#ifdef PYLITTLE_WITH_CUDA
	if (pinned) { if (ptr) cudaFreeHost(ptr); return; }
#endif
	std::free(ptr);
}

int MemoryManager::map_file(const std::string& path) {
	MapEntry entry;
	entry.path = path;
	std::ifstream f(path, std::ios::binary | std::ios::ate);
	if (!f) return -1;
	entry.size = static_cast<size_t>(f.tellg());
	entry.valid = true;
	maps_.push_back(std::move(entry));
	return static_cast<int>(maps_.size() - 1);
}

bool MemoryManager::prefetch(int handle, size_t offset, size_t bytes) {
	if (handle < 0 || static_cast<size_t>(handle) >= maps_.size()) return false;
	auto& e = maps_[static_cast<size_t>(handle)];
	if (!e.valid) return false;
	if (offset >= e.size) return false;
	size_t to_read = (offset + bytes > e.size) ? (e.size - offset) : bytes;
	e.buffer.resize(to_read);
	std::ifstream f(e.path, std::ios::binary);
	if (!f) return false;
	f.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
	f.read(reinterpret_cast<char*>(e.buffer.data()), static_cast<std::streamsize>(to_read));
	e.prefetched_bytes = to_read;
	return true;
}

void MemoryManager::unmap(int handle) {
	if (handle < 0 || static_cast<size_t>(handle) >= maps_.size()) return;
	auto& e = maps_[static_cast<size_t>(handle)];
	e.valid = false;
	e.buffer.clear();
}

bool MemoryManager::prefetch_async(int handle, size_t offset, size_t bytes, void* /*stream*/) {
	// For now, just call synchronous prefetch even when CUDA is present.
	// Future: use async file IO or staged H2D with provided stream.
	return prefetch(handle, offset, bytes);
}

std::string MemoryManager::stats() const {
	size_t total_pref = 0;
	for (const auto& e : maps_) {
		if (e.valid) total_pref += e.prefetched_bytes;
	}
	std::ostringstream oss;
	oss << "maps=" << maps_.size() << ", prefetched=" << total_pref;
	return oss.str();
}

void* MemoryManager::create_stream() {
#ifdef PYLITTLE_WITH_CUDA
	cudaStream_t s = nullptr;
	if (cudaStreamCreate(&s) == cudaSuccess) return reinterpret_cast<void*>(s);
	return nullptr;
#else
	return nullptr;
#endif
}
void MemoryManager::destroy_stream(void* stream) {
#ifdef PYLITTLE_WITH_CUDA
	if (stream) cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream));
#else
	(void)stream;
#endif
}
bool MemoryManager::copy_to_device_async(void* dst_device, const void* src_host, size_t bytes, void* stream) {
#ifdef PYLITTLE_WITH_CUDA
	if (!dst_device || !src_host) return false;
	cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
	auto st = cudaMemcpyAsync(dst_device, src_host, bytes, cudaMemcpyHostToDevice, s);
	return st == cudaSuccess;
#else
	if (!dst_device || !src_host) return false;
	std::memcpy(dst_device, src_host, bytes);
	return true;
#endif
}
bool MemoryManager::synchronize_stream(void* stream) {
#ifdef PYLITTLE_WITH_CUDA
	if (!stream) return true;
	return cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)) == cudaSuccess;
#else
	(void)stream; return true;
#endif
}

} // namespace pylittle
