#version 460
// A pixel shader that does an image atomic, written to the translator's runtime contract (docs/design.md, "User
// data" and "Images"), standing in for a translated one in d3d12_stores_atomics_test. The translator declares the
// storage images of such a shader without NonWritable; this one does the same (no `readonly`), at the binding set 0
// has for the R32ui views the atomic instructions use (binding 4, src/vulkan_bindless.h). Outside a compute shader
// Vulkan allows that only on a device with fragmentStoresAndAtomics (VUID-RuntimeSpirv-NonWritable-06340).
// The test creates the pipeline and never draws with it: index 0 is the set's fallback view.
#extension GL_EXT_nonuniform_qualifier : require

layout(location = 0) in vec4 color;
layout(location = 0) out vec4 target;

layout(push_constant) uniform UserData {
    uint s[32];
    uvec2 heap_base;
    uvec2 code_base;
} pc;

layout(set = 0, binding = 4, r32ui) uniform uimage2D counters[];

void main() {
    const uint before = imageAtomicMax(counters[0], ivec2(gl_FragCoord.xy), pc.s[0]);
    target = vec4(color.rgb, float(before & 1u));
}
