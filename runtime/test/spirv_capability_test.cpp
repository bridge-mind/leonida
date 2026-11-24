// CPU-only tests for spirv_capability.h, which the standalone GPU tools use to enable shaderInt64 only for modules
// with 64-bit integers (TEST-468). The modules are hand-built word lists; no Vulkan instance or GPU is needed.
#include "spirv_capability.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <vector>

namespace {

int failures = 0;

void expect(const char* what, bool got, bool want) {
    if (got == want) return;
    std::fprintf(stderr, "FAIL %s: got %d, expected %d\n", what, got, want);
    ++failures;
}

// A 5-word header, one OpCapability per entry of `caps`, then OpMemoryModel Logical GLSL450.
std::vector<std::uint32_t> module(std::initializer_list<std::uint32_t> caps) {
    std::vector<std::uint32_t> w{0x07230203u, 0x00010000u, 0u, 1u, 0u};
    for (std::uint32_t c : caps) {
        w.push_back(2u << 16 | 17u);   // OpCapability
        w.push_back(c);
    }
    w.push_back(3u << 16 | 14u);       // OpMemoryModel
    w.push_back(0u);
    w.push_back(1u);
    return w;
}

}  // namespace

int main() {
    using namespace leonida::vk_test;
    constexpr std::uint32_t shader = 1, physical_storage_buffer = 5347;

    // SHADER-423's output: buffer addresses as 32-bit pairs, PhysicalStorageBufferAddresses but no Int64.
    expect("address pairs need no shaderInt64", needs_shader_int64(module({shader, physical_storage_buffer})), false);
    expect("Int64 needs shaderInt64", needs_shader_int64(module({shader, spirv_capability_int64})), true);
    expect("Int64 after other capabilities", needs_shader_int64(module({shader, physical_storage_buffer, 10, 11})), true);
    expect("Int64Atomics implicitly declares Int64", needs_shader_int64(module({shader, spirv_capability_int64_atomics})),
           true);
    // cs_f64_add (vk_compute_test enables shaderFloat64 for it): Float64 without Int64.
    const std::vector<std::uint32_t> f64 = module({shader, spirv_capability_float64});
    expect("Float64", declares_capability(f64, spirv_capability_float64), true);
    expect("Float64 needs no shaderInt64", needs_shader_int64(f64), false);

    // Only the leading OpCapability run counts: a later word equal to 11 is an operand, not a capability.
    std::vector<std::uint32_t> later = module({shader});
    later.push_back(2u << 16 | 17u);
    later.push_back(spirv_capability_int64);
    expect("a capability after OpMemoryModel is ignored", needs_shader_int64(later), false);

    // A short or empty file (a missing module reads as empty) declares nothing and doesn't read past its end.
    expect("empty", needs_shader_int64(std::vector<std::uint32_t>{}), false);
    const std::vector<std::uint32_t> truncated{0x07230203u, 0x00010000u, 0u, 1u, 0u, 2u << 16 | 17u};
    expect("truncated OpCapability", needs_shader_int64(truncated), false);

    // The byte form the tools use for std::vector<char>, including a trailing partial word.
    const std::vector<std::uint32_t> m = module({shader, spirv_capability_int64});
    std::vector<char> bytes(m.size() * 4 + 3, 0);
    std::memcpy(bytes.data(), m.data(), m.size() * 4);
    expect("bytes with a partial word", needs_shader_int64(bytes.data(), bytes.size()), true);

    if (failures) return 1;
    std::printf("spirv_capability: all checks passed\n");
    return 0;
}
