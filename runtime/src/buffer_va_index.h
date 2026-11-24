#pragma once

// Which registered buffer a console VA falls in (VulkanContext::buffer_at, for vertex and index views and
// WriteBufferImmediate). Ranges are kept ordered by their first VA, so a lookup is a binary search instead of a scan of
// every registered resource. No Vulkan or Windows dependency; not thread-safe (VulkanContext guards it with
// resource_mutex_, like the bindings themselves).
//
// D3D12 lets placed buffers alias, so ranges may overlap. A lookup names the bytes it needs, [va, va + bytes), and
// returns, among the ranges that hold va, the one that starts closest below it (the larger ID on a tie) that also
// holds all of those bytes; when none does, the closest one that holds va. So a view that starts in a small placed
// buffer and runs past its end resolves to a larger buffer that aliases the same memory and holds all of it, instead
// of being cut at the small one's end (found in review of #2823). The walk goes down from the nearest start and stops
// once no range can still reach the VA, or, with a fallback found, once no range can still hold all the bytes; both
// are bounded by `longest_`. Without overlaps the first range it looks at is the answer, or none is.
// `longest_` is the longest range indexed now, not the longest ever: a large buffer that is freed stops lengthening
// later misses. A miss still walks every start within the longest live range below the VA.

#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <utility>

namespace leonida {

template<typename Id>
class BufferVaIndex {
public:
    // Registers [va, va + size). A null VA or an empty range is not indexed (no lookup can land in it).
    void insert(std::uint64_t va, std::uint64_t size, Id id) {
        if (!va || !size) return;
        const auto [it, added] = ranges_.try_emplace({va, id}, size);
        if (!added) {  // the same (va, id) again: its new size replaces the old one
            drop_size(it->second);
            it->second = size;
        }
        sizes_.insert(size);
        longest_ = *sizes_.rbegin();
    }

    // Forgets what insert(va, size, id) registered; nothing when it was not indexed.
    void erase(std::uint64_t va, Id id) {
        const auto it = ranges_.find({va, id});
        if (it == ranges_.end()) return;
        drop_size(it->second);
        ranges_.erase(it);
        longest_ = sizes_.empty() ? 0 : *sizes_.rbegin();
    }

    struct Hit {
        Id id;
        std::uint64_t start;  // the range's first VA
        std::uint64_t size;
    };

    // The range that holds `va`, preferring one that holds [va, va + bytes) (bytes 0 counts as 1); false when none
    // holds va.
    bool find(std::uint64_t va, Hit& out, std::uint64_t bytes = 1) const {
        if (!bytes) bytes = 1;
        bool found = false;  // `out` holds the closest range that holds va, though not all the bytes
        auto it = ranges_.upper_bound({va, max_id()});
        while (it != ranges_.begin()) {
            --it;
            const std::uint64_t start = it->first.first;
            const std::uint64_t offset = va - start;  // start <= va here; it only grows as the walk goes down
            if (offset >= longest_) break;            // no range that starts this low reaches va
            if (found && bytes > longest_ - offset) break;  // nor holds all the bytes
            if (offset >= it->second) continue;
            if (bytes <= it->second - offset) {
                out = {it->first.second, start, it->second};
                return true;
            }
            if (!found) {
                out = {it->first.second, start, it->second};
                found = true;
            }
        }
        return found;
    }

    std::size_t size() const { return ranges_.size(); }
    std::uint64_t longest() const { return longest_; }

private:
    static constexpr Id max_id() { return ~Id(0); }

    void drop_size(std::uint64_t size) { sizes_.erase(sizes_.find(size)); }  // one copy, not every range of that size

    std::map<std::pair<std::uint64_t, Id>, std::uint64_t> ranges_;  // (first VA, id) -> bytes
    std::multiset<std::uint64_t> sizes_;  // the size of every indexed range, so the longest is known after an erase
    std::uint64_t longest_ = 0;  // the longest indexed range, 0 when none is
};

} // namespace leonida
