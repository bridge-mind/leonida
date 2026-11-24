// The blob ID3D12PipelineState::GetCachedBlob hands out and Create*PipelineState checks in CachedPSO.
//
// The Vulkan layer keeps no driver-compiled pipeline in it (D3D-34 seeds a VkPipelineCache from the build instead);
// the blob only records what the PC runtime checks before accepting a cached PSO, measured with the host runtime on
// an RTX 3060 (driver 32.0.16.1714) and WARP:
//   - the same description: accepted;
//   - another shader, or a root signature with other content: E_INVALIDARG (an equal root signature object is fine;
//     D3D12_PIPELINE_STATE_DESC::Flags is not part of the match);
//   - any byte changed, a truncated blob, or a pointer with CachedBlobSizeInBytes 0: E_INVALIDARG;
//   - a null pointer with a size: ignored, the PSO is created;
//   - a blob from another adapter: D3D12_ERROR_ADAPTER_NOT_FOUND.
// A blob from the same adapter and another driver gives D3D12_ERROR_DRIVER_VERSION_MISMATCH (DirectX docs; not
// measured, it needs a driver change).
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <vector>

namespace leonida::cached_pso {

// The physical device the blob was made on: VkPhysicalDeviceProperties' IDs, driverVersion and pipelineCacheUUID.
struct Identity {
    std::uint32_t vendor = 0;
    std::uint32_t device = 0;
    std::uint32_t driver = 0;
    std::uint8_t cache_uuid[16] = {};
};

// What a blob is checked against: the SHA-256 (hex) of the parts of the description the PC runtime matches.
using Key = std::string;

enum class Check { accepted, invalid, other_adapter, other_driver, other_description };

inline constexpr std::uint32_t magic = 0x4f53504c;  // "LPSO"
inline constexpr std::uint32_t version = 1;
inline constexpr std::size_t key_bytes = 64;
inline constexpr std::size_t blob_bytes = 4 + 4 + 12 + 16 + key_bytes + 4;

inline std::uint32_t fnv1a(const std::uint8_t* data, std::size_t size) {
    std::uint32_t h = 0x811c9dc5u;
    for (std::size_t i = 0; i < size; ++i) h = (h ^ data[i]) * 0x01000193u;
    return h;
}

inline void put32(std::vector<std::uint8_t>& out, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) out.push_back(std::uint8_t(v >> (8 * i)));
}

inline std::uint32_t get32(const std::uint8_t* p) {
    return std::uint32_t(p[0]) | (std::uint32_t(p[1]) << 8) | (std::uint32_t(p[2]) << 16) | (std::uint32_t(p[3]) << 24);
}

// magic, version, vendor, device, driver, pipelineCacheUUID, key, then FNV-1a of everything before it: a changed
// byte anywhere makes the blob invalid, as with the PC runtime.
inline std::vector<std::uint8_t> encode(const Identity& identity, const Key& key) {
    std::vector<std::uint8_t> out;
    out.reserve(blob_bytes);
    put32(out, magic);
    put32(out, version);
    put32(out, identity.vendor);
    put32(out, identity.device);
    put32(out, identity.driver);
    out.insert(out.end(), std::begin(identity.cache_uuid), std::end(identity.cache_uuid));
    std::string padded = key;
    padded.resize(key_bytes, '\0');
    out.insert(out.end(), padded.begin(), padded.end());
    put32(out, fnv1a(out.data(), out.size()));
    return out;
}

inline Check check(const Identity& identity, const Key& key, std::span<const std::uint8_t> blob) {
    if (blob.size() != blob_bytes) return Check::invalid;
    const std::uint8_t* p = blob.data();
    if (get32(p + blob_bytes - 4) != fnv1a(p, blob_bytes - 4)) return Check::invalid;
    if (get32(p) != magic || get32(p + 4) != version) return Check::invalid;
    if (get32(p + 8) != identity.vendor || get32(p + 12) != identity.device) return Check::other_adapter;
    if (get32(p + 16) != identity.driver || std::memcmp(p + 20, identity.cache_uuid, 16) != 0)
        return Check::other_driver;
    std::string padded = key;
    padded.resize(key_bytes, '\0');
    if (std::memcmp(p + 36, padded.data(), key_bytes) != 0) return Check::other_description;
    return Check::accepted;
}

// Builds the bytes a key is hashed from: every field the PC runtime matches, in a fixed order with lengths, so two
// descriptions give the same bytes only when those fields are equal.
class KeyBuilder {
public:
    KeyBuilder& tag(const char* name) { return bytes(name, std::strlen(name)); }
    KeyBuilder& u32(std::uint32_t v) {
        put32(data_, v);
        return *this;
    }
    KeyBuilder& bytes(const void* p, std::size_t n) {
        put32(data_, std::uint32_t(n));
        if (n) data_.insert(data_.end(), static_cast<const std::uint8_t*>(p), static_cast<const std::uint8_t*>(p) + n);
        return *this;
    }
    KeyBuilder& text(const char* s) { return s ? bytes(s, std::strlen(s)) : u32(0xffffffffu); }
    const std::vector<std::uint8_t>& data() const { return data_; }

private:
    std::vector<std::uint8_t> data_;
};

}  // namespace leonida::cached_pso
