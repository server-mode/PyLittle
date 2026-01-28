#pragma once
#include <cstddef>
#include <string>
#include <unordered_map>
#include <list>
#include <vector>
#include "pylittle/memory_manager.h"

namespace pylittle {

struct KVSlice {
    int seq_id;
    size_t offset; // token index offset
    size_t size_bytes;
};

class KVPager {
public:
    explicit KVPager(MemoryManager* mm, size_t page_bytes)
        : mm_(mm), page_bytes_(page_bytes) {}

    // Register a new sequence
    void add_sequence(int seq_id);

    // Append KV bytes for a sequence
    void append_kv(int seq_id, const void* data, size_t bytes);

    // Request a sliding window of recent tokens
    // Returns number of bytes currently pinned in GPU for the window
    size_t request_window(int seq_id, size_t recent_tokens);

    // Force eviction of least-recently-used pages to satisfy budgets (simplified)
    void evict_lru(size_t target_free_bytes);

    std::string stats() const;

private:
    struct Page {
        std::vector<char> host; // backing storage
        bool on_gpu = false;
        size_t tokens = 0; // tokens represented by this page (approx)
    };

    MemoryManager* mm_;
    size_t page_bytes_;

    // LRU list of (seq_id,page_idx)
    std::list<std::pair<int,int>> lru_;
    // Map: seq -> pages
    std::unordered_map<int, std::vector<Page>> seq_pages_;
};

} // namespace pylittle
