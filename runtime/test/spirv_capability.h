// What a SPIR-V module declares, read before the device is created, so a test tool enables only the optional features
// the module needs and reports 77 (unsupported) when the device lacks one, instead of asking for every feature a
// translated shader might use. rdna2spirv.py lowers buffer addresses to 32-bit pairs (SHADER-423), so most memory
// shaders no longer declare Int64; asking for shaderInt64 regardless made the tools fail on devices without it (Intel
// UHD/Arc) before the shader ran (TEST-468).
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace leonida::vk_test {

// Capability numbers from spirv/unified1/spirv.h.
inline constexpr std::uint32_t spirv_capability_float64 = 10;
inline constexpr std::uint32_t spirv_capability_int64 = 11;
inline constexpr std::uint32_t spirv_capability_int64_atomics = 12;

// Whether the module declares capability `cap`. OpCapability instructions (2 words each) come first, right after the
// 5-word header, so the scan stops at the first other instruction. `bytes` may be any length; a partial word is ignored.
inline bool declares_capability(const void* code, std::size_t bytes, std::uint32_t cap) {
    const auto* p = static_cast<const unsigned char*>(code);
    const std::size_t words = bytes / 4;
    auto word = [&](std::size_t i) {
        std::uint32_t w;
        std::memcpy(&w, p + 4 * i, 4);
        return w;
    };
    for (std::size_t i = 5; i + 1 < words && word(i) == (2u << 16 | 17u); i += 2)   // OpCapability
        if (word(i + 1) == cap) return true;
    return false;
}

inline bool declares_capability(const std::vector<std::uint32_t>& spv, std::uint32_t cap) {
    return declares_capability(spv.data(), spv.size() * 4, cap);
}

// 64-bit integer types need shaderInt64: the Int64 capability, or Int64Atomics, which implicitly declares it.
inline bool needs_shader_int64(const void* code, std::size_t bytes) {
    return declares_capability(code, bytes, spirv_capability_int64) ||
           declares_capability(code, bytes, spirv_capability_int64_atomics);
}

inline bool needs_shader_int64(const std::vector<std::uint32_t>& spv) {
    return needs_shader_int64(spv.data(), spv.size() * 4);
}

}  // namespace leonida::vk_test
