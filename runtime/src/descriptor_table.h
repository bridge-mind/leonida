#pragma once

// The descriptor hash table for set 0, binding 2 (docs/d3d12_vulkan.md section 5.1 and docs/design.md):
// Shaders look up bindless resource/sampler indices by FNV-1a hashing the 8-dword descriptor key (T# or padded S#)
// and probing up to 8 slots in this table. Miss or fallback yields index 0.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <span>
#include <stdexcept>
#include <vector>

namespace leonida {

// FNV-1a constants for 32-bit dwords
inline constexpr std::uint32_t kFnvOffsetBasis = 2166136261u;
inline constexpr std::uint32_t kFnvPrime = 16777619u;
inline constexpr std::uint32_t kDescriptorProbes = 8;
inline constexpr std::uint32_t kSlotOccupiedMarker = 0xA11CA7EDu;

// Key types:
// T# key is 8 dwords.
// S# key is 4 dwords followed by 4x 0xFFFFFFFF.
using DescriptorKey = std::array<std::uint32_t, 8>;

inline std::uint32_t hash_descriptor_key(const std::uint32_t key[8]) {
    std::uint32_t h = kFnvOffsetBasis;
    for (int i = 0; i < 8; ++i) {
        h = (h ^ key[i]) * kFnvPrime;
    }
    return h;
}

inline std::uint32_t hash_descriptor_key(const DescriptorKey& key) {
    return hash_descriptor_key(key.data());
}

inline DescriptorKey make_tsharp_key(const std::uint32_t tsharp[8]) {
    DescriptorKey k{};
    std::memcpy(k.data(), tsharp, sizeof(DescriptorKey));
    return k;
}

inline DescriptorKey make_ssharp_key(const std::uint32_t ssharp[4]) {
    DescriptorKey k{};
    k[0] = ssharp[0];
    k[1] = ssharp[1];
    k[2] = ssharp[2];
    k[3] = ssharp[3];
    k[4] = 0xFFFFFFFFu;
    k[5] = 0xFFFFFFFFu;
    k[6] = 0xFFFFFFFFu;
    k[7] = 0xFFFFFFFFu;
    return k;
}

// Memory layout of each entry in the table: 10 uint32s = 40 bytes.
struct DescriptorEntry {
    std::uint32_t key[8] = {};
    std::uint32_t index = 0;
    std::uint32_t pad = 0; // 0 when empty, kSlotOccupiedMarker when occupied
};
static_assert(sizeof(DescriptorEntry) == 40);

class DescriptorTable {
public:
    enum class InsertResult { added, unchanged, replaced, added_with_relocation };

    explicit DescriptorTable(std::uint32_t initial_capacity = 16) {
        // Slots use nonnegative int indices internally. Larger powers also wrap the capacity counter.
        if (initial_capacity > max_capacity) throw std::length_error("Descriptor table capacity exceeds slot range");
        std::uint32_t cap = 16;
        while (cap < initial_capacity) cap <<= 1;
        mask_ = cap - 1;
        entries_.resize(cap);
    }

    // Inserts or updates a key -> index mapping.
    // If the key already exists in one of the 8 probe slots, its index is updated.
    // Try one bounded relocation before growing. Load never exceeds 50%.
    bool insert(const std::uint32_t key[8], std::uint32_t index) {
        insert_or_update(key, index);
        return true;
    }

    // Report the change under the same lock so the GPU caller can synchronize writes to live slots.
    InsertResult insert_or_update(const std::uint32_t key[8], std::uint32_t index) {
        std::lock_guard lock(mutex_);
        return insert_locked(key, index);
    }

    bool insert(const DescriptorKey& key, std::uint32_t index) {
        return insert(key.data(), index);
    }

    // Queries the index associated with key. Returns 0 if not found.
    std::uint32_t lookup(const std::uint32_t key[8]) const {
        std::lock_guard lock(mutex_);
        const std::uint32_t h = hash_descriptor_key(key);
        for (std::uint32_t k = 0; k < kDescriptorProbes; ++k) {
            const std::uint32_t slot = (h + k) & mask_;
            const auto& entry = entries_[slot];
            if (entry.pad == kSlotOccupiedMarker && std::memcmp(entry.key, key, sizeof(entry.key)) == 0) {
                return entry.index;
            }
        }
        return 0; // Exceeded 8 probes without a match
    }

    std::uint32_t lookup(const DescriptorKey& key) const {
        return lookup(key.data());
    }

    bool contains(const std::uint32_t key[8]) const {
        std::lock_guard lock(mutex_);
        const std::uint32_t h = hash_descriptor_key(key);
        for (std::uint32_t k = 0; k < kDescriptorProbes; ++k) {
            const std::uint32_t slot = (h + k) & mask_;
            const auto& entry = entries_[slot];
            if (entry.pad == kSlotOccupiedMarker && std::memcmp(entry.key, key, sizeof(entry.key)) == 0) {
                return true;
            }
        }
        return false;
    }

    bool contains(const DescriptorKey& key) const {
        return contains(key.data());
    }

    // Erases a key from the table. Returns true if the key was found and erased, false otherwise.
    // Only the key's own slot changes (it becomes empty): no other entry moves. Every lookup, here and in the
    // shaders (rdna2spirv.py desc_probe), compares all kDescriptorProbes slots and never stops at an empty one, so
    // a hole breaks no probe chain and inserts reuse it. Keeping the other entries in place is what lets the GPU
    // copy of the table take an erase while shaders still probe it (BindlessSet::flush_replaces_table).
    bool erase(const std::uint32_t key[8]) {
        std::lock_guard lock(mutex_);
        return erase_locked(key);
    }

    bool erase(const DescriptorKey& key) {
        return erase(key.data());
    }

    void clear() {
        std::lock_guard lock(mutex_);
        std::fill(entries_.begin(), entries_.end(), DescriptorEntry{});
        occupied_count_ = 0;
        all_changed_ = true;
        changed_slots_.clear();
    }

    std::size_t size() const {
        std::lock_guard lock(mutex_);
        return occupied_count_;
    }

    std::size_t capacity() const {
        std::lock_guard lock(mutex_);
        return entries_.size();
    }

    std::uint32_t mask() const {
        std::lock_guard lock(mutex_);
        return mask_;
    }

    std::size_t size_dwords() const {
        std::lock_guard lock(mutex_);
        return 1 + entries_.size() * 10;
    }

    std::size_t size_bytes() const {
        return size_dwords() * sizeof(std::uint32_t);
    }

    // Exports the table buffer format consumed by SPIR-V set 0, binding 2.
    // Word 0 is mask; followed by entries[mask + 1] of 10 dwords: key[8], index, pad.
    std::vector<std::uint32_t> build_gpu_buffer(bool clear_pad = false) const {
        std::lock_guard lock(mutex_);
        std::vector<std::uint32_t> out(1 + entries_.size() * 10, 0);
        out[0] = mask_;
        for (std::size_t s = 0; s < entries_.size(); ++s) {
            const std::size_t base = 1 + s * 10;
            std::memcpy(&out[base], entries_[s].key, sizeof(entries_[s].key));
            out[base + 8] = entries_[s].index;
            out[base + 9] = clear_pad ? 0 : entries_[s].pad;
        }
        return out;
    }

    // What a shader's copy of the table needs re-uploaded since the last take_changes().
    struct Changes {
        bool all = false;                  // the mask (table grew or was cleared): every slot and the header
        std::vector<std::uint32_t> slots;  // slots whose 10 dwords changed, in no particular order
    };

    Changes take_changes() {
        std::lock_guard lock(mutex_);
        Changes changes;
        changes.all = all_changed_;
        changes.slots = std::move(changed_slots_);
        changed_slots_.clear();
        all_changed_ = false;
        return changes;
    }

    // One slot's 10 dwords (key[8], index, pad) to `dest`; false when the slot is out of range.
    bool read_slot(std::uint32_t slot, std::uint32_t* dest) const { return read_slots(slot, 1, dest); }

    // `count` consecutive slots' 10 dwords each to `dest`, under one lock; false when any is out of range.
    bool read_slots(std::uint32_t first, std::uint32_t count, std::uint32_t* dest) const {
        if (count > 0 && !dest) return false;
        std::lock_guard lock(mutex_);
        if (first > entries_.size() || count > entries_.size() - first) return false;
        for (std::uint32_t i = 0; i < count; ++i) {
            const DescriptorEntry& entry = entries_[first + i];
            std::memcpy(dest + i * 10, entry.key, sizeof(entry.key));
            dest[i * 10 + 8] = entry.index;
            dest[i * 10 + 9] = entry.pad;
        }
        return true;
    }

    // Serializes directly into `dest` (must hold at least size_bytes()) without allocating a temporary vector.
    void write_to_buffer(void* dest, bool clear_pad = false) const {
        if (!dest) return;
        std::lock_guard lock(mutex_);
        auto* out = static_cast<std::uint8_t*>(dest);
        std::memcpy(out, &mask_, sizeof(mask_));
        out += sizeof(mask_);
        for (const auto& entry : entries_) {
            std::uint32_t slot[10];
            std::memcpy(slot, entry.key, sizeof(entry.key));
            slot[8] = entry.index;
            slot[9] = clear_pad ? 0 : entry.pad;
            std::memcpy(out, slot, sizeof(slot));
            out += sizeof(slot);
        }
    }

private:
    void mark(std::uint32_t slot) { changed_slots_.push_back(slot); }

    // All eight slots for the incoming key are occupied. Try moving one resident
    // to the first empty slot beyond that window, within its own eight probes.
    // A later hole is farther from every resident, so failure here is final for a single move.
    // `moved_to` receives the hole the resident moved into, so the caller can mark that slot changed too.
    static int make_probe_slot(std::vector<DescriptorEntry>& entries, std::uint32_t hash,
                               std::uint32_t* moved_to = nullptr) {
        const auto mask = static_cast<std::uint32_t>(entries.size() - 1);
        for (auto offset = kDescriptorProbes; offset < 2 * kDescriptorProbes - 1; ++offset) {
            const auto hole = (hash + offset) & mask;
            if (entries[hole].pad == kSlotOccupiedMarker) continue;
            for (auto k = kDescriptorProbes; k > 0; --k) {
                const auto slot = (hash + k - 1) & mask;
                const auto home = hash_descriptor_key(entries[slot].key);
                if (((hole - home) & mask) < kDescriptorProbes) {
                    entries[hole] = entries[slot];
                    if (moved_to) *moved_to = hole;
                    return static_cast<int>(slot); // caller immediately overwrites the resident
                }
            }
            break;
        }
        return -1;
    }

    InsertResult insert_locked(const std::uint32_t key[8], std::uint32_t index) {
        while (true) {
            const std::uint32_t h = hash_descriptor_key(key);
            int empty_slot = -1;

            for (std::uint32_t k = 0; k < kDescriptorProbes; ++k) {
                const std::uint32_t slot = (h + k) & mask_;
                auto& entry = entries_[slot];
                if (entry.pad == kSlotOccupiedMarker) {
                    if (std::memcmp(entry.key, key, sizeof(entry.key)) == 0) {
                        if (entry.index == index) return InsertResult::unchanged;
                        entry.index = index;
                        mark(slot);
                        return InsertResult::replaced;
                    }
                } else if (empty_slot < 0) {
                    empty_slot = static_cast<int>(slot);
                }
            }

            if ((occupied_count_ + 1) * 2 > entries_.size()) {
                grow_locked();
                continue;
            }

            // Nine distinct keys with the same full hash can never fit the shader's eight probes,
            // regardless of capacity. Refuse before allocating ever larger tables.
            if (empty_slot < 0) {
                std::uint32_t same_hash = 0;
                for (std::uint32_t k = 0; k < kDescriptorProbes; ++k) {
                    const auto& entry = entries_[(h + k) & mask_];
                    if (hash_descriptor_key(entry.key) == h) ++same_hash;
                }
                if (same_hash == kDescriptorProbes)
                    throw std::length_error("Descriptor hash collision exceeds shader probe limit");
            }
            const bool relocate = empty_slot < 0;
            std::uint32_t moved_to = 0;
            if (relocate) empty_slot = make_probe_slot(entries_, h, &moved_to);
            if (empty_slot >= 0) {
                auto& entry = entries_[empty_slot];
                std::memcpy(entry.key, key, sizeof(entry.key));
                entry.index = index;
                entry.pad = kSlotOccupiedMarker;
                ++occupied_count_;
                if (relocate) mark(moved_to);
                mark(std::uint32_t(empty_slot));
                return relocate ? InsertResult::added_with_relocation : InsertResult::added;
            }

            grow_locked();
        }
    }

    bool erase_locked(const std::uint32_t key[8]) {
        const std::uint32_t h = hash_descriptor_key(key);
        std::uint32_t found_slot = 0;
        bool found = false;
        for (std::uint32_t k = 0; k < kDescriptorProbes; ++k) {
            const std::uint32_t slot = (h + k) & mask_;
            const auto& entry = entries_[slot];
            if (entry.pad == kSlotOccupiedMarker && std::memcmp(entry.key, key, sizeof(entry.key)) == 0) {
                found_slot = slot;
                found = true;
                break;
            }
        }
        if (!found) return false;

        // No backward shift: see erase().
        entries_[found_slot] = DescriptorEntry{};
        mark(found_slot);
        --occupied_count_;
        return true;
    }

    void grow_locked() {
        if (entries_.size() >= max_capacity) throw std::length_error("Descriptor table capacity exhausted");
        std::uint32_t new_cap = static_cast<std::uint32_t>(entries_.size() * 2);
        while (true) {
            std::vector<DescriptorEntry> new_entries(new_cap);
            const std::uint32_t new_mask = new_cap - 1;
            bool rehash_ok = true;

            for (const auto& old : entries_) {
                if (old.pad != kSlotOccupiedMarker) continue;
                const std::uint32_t h = hash_descriptor_key(old.key);
                int slot = -1;
                for (std::uint32_t k = 0; k < kDescriptorProbes; ++k) {
                    const std::uint32_t s = (h + k) & new_mask;
                    if (new_entries[s].pad != kSlotOccupiedMarker) {
                        slot = static_cast<int>(s);
                        break;
                    }
                }
                if (slot < 0) slot = make_probe_slot(new_entries, h);
                if (slot < 0) {
                    rehash_ok = false;
                    break;
                }
                new_entries[slot] = old;
            }

            if (rehash_ok) {
                entries_ = std::move(new_entries);
                mask_ = new_mask;
                all_changed_ = true;
                changed_slots_.clear();
                break;
            }
            if (new_cap >= max_capacity) throw std::length_error("Descriptor probe range exhausted");
            new_cap *= 2;
        }
    }

    static constexpr std::uint32_t max_capacity = std::uint32_t{1} << 31;
    mutable std::mutex mutex_;
    std::uint32_t mask_ = 15;
    std::vector<DescriptorEntry> entries_;
    std::size_t occupied_count_ = 0;
    bool all_changed_ = false;                  // clear() or a grow: the mask and every slot changed
    std::vector<std::uint32_t> changed_slots_;  // slots inserted, replaced or moved since take_changes()
};

} // namespace leonida
