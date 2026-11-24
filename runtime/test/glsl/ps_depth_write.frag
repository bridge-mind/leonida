#version 460
// A pixel shader that writes its own depth (SV_Depth), for d3d12_depth24_draw_test: s2 is the depth it writes. On a
// D24 target the value is stored quantised to 24 bits, and D3D applies no depth bias to it.

layout(push_constant) uniform UserData {
    uint s[32];
    uvec2 heap_base;
    uvec2 code_base;
} pc;

void main() {
    gl_FragDepth = uintBitsToFloat(pc.s[2]);
}
