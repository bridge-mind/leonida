#version 460
// A pixel shader written to the translator's runtime contract, standing in for a translated one in
// d3d12_draw_test. s2:s3 is a root CBV: the console VA of three floats (a tint), read at heap base + VA. The output
// is (red * tint.x, tint.y, blue * tint.z, 1).
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference_uvec2 : require

layout(buffer_reference, std430, buffer_reference_align = 4) buffer Floats {
    float f[];
};

layout(location = 0) in vec4 color;
layout(location = 0) out vec4 target;

layout(push_constant) uniform UserData {
    uint s[32];
    uvec2 heap_base;
    uvec2 code_base;
} pc;

uvec2 add64(uvec2 a, uvec2 b) {
    uint carry;
    const uint low = uaddCarry(a.x, b.x, carry);
    return uvec2(low, a.y + b.y + carry);
}

void main() {
    Floats tint = Floats(add64(pc.heap_base, uvec2(pc.s[2], pc.s[3])));
    target = vec4(color.r * tint.f[0], tint.f[1], color.b * tint.f[2], 1.0);
}
