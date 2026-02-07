#include "pylittle/kv_pager.h"
#include <sstream>
#include <algorithm>

namespace pylittle {

void KVPager::add_sequence(int seq_id) {
    if (seq_pages_.count(seq_id) == 0) {
        seq_pages_[seq_id] = {};
    }
}

void KVPager::append_kv(int seq_id, const void* data, size_t bytes) {
    (void)data;
    // Back-compat: treat bytes as tokens for the old demo API.
    append_kv_bytes(seq_id, bytes, bytes);
}

void KVPager::append_kv_bytes(int seq_id, size_t bytes, size_t tokens) {
    auto& pages = seq_pages_[seq_id];
    if (pages.empty() || pages.back().host_bytes + bytes > page_bytes_) {
        Page p; p.host_bytes = 0; p.on_gpu = false; p.tokens = 0;
        pages.emplace_back(std::move(p));
    }
    auto& last = pages.back();

    size_t to_add = bytes;
    size_t toks = tokens;
    while (to_add > 0) {
        size_t can = std::min(page_bytes_ - last.host_bytes, to_add);
        last.host_bytes += can;
        // Distribute tokens proportionally if bytes span multiple pages.
        size_t tok_can = (to_add > 0) ? (toks * can / to_add) : 0;
        last.tokens += tok_can;
        to_add -= can;
        toks -= tok_can;
        if (to_add > 0) {
            Page p; p.host_bytes = 0; p.on_gpu = false; p.tokens = 0;
            pages.emplace_back(std::move(p));
        }
    }
    // Any remaining tokens (rounding) go to last page.
    if (!pages.empty()) {
        pages.back().tokens += toks;
    }
}

size_t KVPager::request_window(int seq_id, size_t recent_tokens) {
    auto it = seq_pages_.find(seq_id);
    if (it == seq_pages_.end()) return 0;
    auto& pages = it->second;
    // Mark last N pages as on_gpu until tokens >= recent_tokens
    size_t acc = 0;
    for (int i = static_cast<int>(pages.size()) - 1; i >= 0 && acc < recent_tokens; --i) {
        auto& pg = pages[static_cast<size_t>(i)];
        if (!pg.on_gpu) {
            pg.on_gpu = true; // pretend pin to GPU
            lru_.push_front({seq_id, i});
        }
        acc += pg.tokens;
    }
    return acc;
}

void KVPager::evict_lru(size_t target_free_bytes) {
    size_t freed = 0;
    while (!lru_.empty() && freed < target_free_bytes) {
        auto [sid, idx] = lru_.back(); lru_.pop_back();
        auto it = seq_pages_.find(sid);
        if (it == seq_pages_.end()) continue;
        auto& pages = it->second;
        if (idx >= 0 && static_cast<size_t>(idx) < pages.size()) {
            if (pages[static_cast<size_t>(idx)].on_gpu) {
                pages[static_cast<size_t>(idx)].on_gpu = false; // unpin
                freed += pages[static_cast<size_t>(idx)].host_bytes;
            }
        }
    }
}

std::string KVPager::stats() const {
    size_t total = 0, on_gpu = 0, bytes_gpu = 0;
    for (const auto& kv : seq_pages_) {
        for (const auto& pg : kv.second) {
            total += 1;
            if (pg.on_gpu) { on_gpu += 1; bytes_gpu += pg.host_bytes; }
        }
    }
    std::ostringstream oss;
    oss << "pages=" << total << ", on_gpu=" << on_gpu << ", bytes_gpu=" << bytes_gpu << ", lru_len=" << lru_.size();
    return oss.str();
}

} // namespace pylittle
