// Unit tests for the OpFmaKHR width gate in vk_shader_fma.h (no Vulkan instance, no GPU). A module built with
// `rdna2spirv.py --fma khr` needs shaderFmaFloat32 for an f32 OpFmaKHR and shaderFmaFloat64 for one on doubles;
// a module that declares Float64 for other doubles but has only f32 OpFmaKHRs needs only shaderFmaFloat32.
#include "vk_shader_fma.h"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

int failures = 0;

void expect(bool ok, const char* what) {
    if (ok) return;
    std::fprintf(stderr, "FAIL %s\n", what);
    ++failures;
}

std::uint32_t word0(std::uint32_t op, std::uint32_t count) { return count << 16 | op; }

// A minimal module: header, OpCapability FMAKHR (+ Float64), float types, and OpFmaKHRs of the given result types.
// Ids: 1 = f32, 2 = f64, 3 = v2f32, 4 = v2f64; FMA operands are ids 10..12 (never defined: the gate does not care).
std::vector<std::uint32_t> module(bool fma_cap, bool float64_cap, const std::vector<std::uint32_t>& fma_types) {
    std::vector<std::uint32_t> w = {0x07230203, 0x00010500, 0, 100, 0};
    if (fma_cap) w.insert(w.end(), {word0(17, 2), 6030});
    if (float64_cap) w.insert(w.end(), {word0(17, 2), 10});
    w.insert(w.end(), {word0(22, 3), 1, 32, word0(22, 3), 2, 64, word0(23, 4), 3, 1, 2, word0(23, 4), 4, 2, 2});
    std::uint32_t id = 20;
    for (const std::uint32_t t : fma_types) w.insert(w.end(), {word0(4427, 6), t, id++, 10, 11, 12});
    return w;
}

bool accepts(const leonida::vk_test::ShaderFma& fma, const std::vector<std::uint32_t>& m) {
    return fma.accepts(m.data(), m.size() * sizeof(m.front()), "test.spv");
}

}  // namespace

int main() {
    using leonida::vk_test::ShaderFma;
    using leonida::vk_test::spirv_fma_widths;

    bool f32 = false, f64 = false;
    auto m = module(true, true, {1});
    spirv_fma_widths(m.data(), m.size() * 4, f32, f64);
    expect(f32 && !f64, "an f32 OpFmaKHR is width 32");
    m = module(true, true, {2});
    spirv_fma_widths(m.data(), m.size() * 4, f32, f64);
    expect(!f32 && f64, "an f64 OpFmaKHR is width 64");
    m = module(true, true, {4, 3});
    spirv_fma_widths(m.data(), m.size() * 4, f32, f64);
    expect(f32 && f64, "vector results take their component width");

    ShaderFma only32;
    only32.float32 = true;
    ShaderFma only64;
    only64.float64 = true;
    ShaderFma both;
    both.float32 = both.float64 = true;
    const ShaderFma none;

    expect(accepts(none, module(false, true, {})), "a module without FMAKHR needs nothing");
    expect(accepts(only32, module(true, false, {1})), "f32 OpFmaKHR on a shaderFmaFloat32 device");
    expect(accepts(only32, module(true, true, {1})), "Float64 elsewhere + f32 OpFmaKHR needs only shaderFmaFloat32");
    expect(!accepts(only32, module(true, true, {2})), "f64 OpFmaKHR is refused without shaderFmaFloat64");
    expect(!accepts(only32, module(true, true, {1, 2})), "mixed widths need both");
    expect(accepts(only64, module(true, true, {2})), "f64 OpFmaKHR on a shaderFmaFloat64 device");
    expect(!accepts(only64, module(true, false, {1})), "f32 OpFmaKHR is refused without shaderFmaFloat32");
    expect(accepts(both, module(true, true, {1, 2})), "both widths on a device with both");
    expect(!accepts(none, module(true, false, {1})), "no VK_KHR_shader_fma refuses any OpFmaKHR");
    expect(!accepts(only64, module(true, false, {})), "FMAKHR without an OpFmaKHR keeps needing shaderFmaFloat32");
    expect(only32.supported() && only64.supported() && !none.supported(), "supported() is either width");

    if (failures) return 1;
    std::puts("vk_shader_fma: all checks passed");
    return 0;
}
