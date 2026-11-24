#version 460
// A full-screen triangle at a chosen depth, written to the translator's runtime contract (docs/design.md, "User
// data"), for d3d12_depth24_draw_test: s0 is the depth at the centre and s1 how much it changes per unit of clip-space
// x, so s1 = 0 gives a flat triangle and anything else a constant slope across the target.

layout(push_constant) uniform UserData {
    uint s[32];
    uvec2 heap_base;
    uvec2 code_base;
} pc;

void main() {
    const vec2 corner = vec2(float(gl_VertexIndex & 2), float((gl_VertexIndex << 1) & 2));
    const vec2 xy = corner * 2.0 - 1.0;
    gl_Position = vec4(xy, uintBitsToFloat(pc.s[0]) + uintBitsToFloat(pc.s[1]) * xy.x, 1.0);
}
