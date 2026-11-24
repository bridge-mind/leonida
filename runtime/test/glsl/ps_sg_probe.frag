#version 460
// A pixel shader that outputs the subgroup size the driver ran it with, (0x40000000 | gl_SubgroupSize, 0x40000000 |
// gl_SubgroupInvocationID, ...) as float bit patterns (the or keeps them normal floats), for d3d12_subgroup_test.
#extension GL_KHR_shader_subgroup_basic : require

layout(location = 0) out vec4 target;

void main() {
    // glslang marks the subgroup built-ins RelaxedPrecision, and the Intel driver then runs the or in 16 bits and drops
    // the 0x40000000 tag. highp copies keep the full-width or.
    highp uint size = gl_SubgroupSize;
    highp uint lane = gl_SubgroupInvocationID;
    target = uintBitsToFloat(uvec4(0x40000000u | size, 0x40000000u | lane, 0x40000000u, 0x40000000u));
}
