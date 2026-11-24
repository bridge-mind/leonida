#version 460
// A vertex shader written to the translator's runtime contract (docs/design.md, "Vertex fetch" and "User data"),
// standing in for a translated one in d3d12_draw_test. Inputs sit at location = D3D input register: COLOR is
// register 0 and POSITION register 1 here, the reverse of the input layout's order, so the test shows that the
// layout is matched by semantic. s0, s1 (root constants) move the vertex in clip space.

layout(location = 0) in vec4 color;     // COLOR0, input register 0
layout(location = 1) in vec4 position;  // POSITION0, input register 1

layout(location = 0) out vec4 out_color;

layout(push_constant) uniform UserData {
    uint s[32];
    uvec2 heap_base;
    uvec2 code_base;
} pc;

void main() {
    const vec2 offset = vec2(uintBitsToFloat(pc.s[0]), uintBitsToFloat(pc.s[1]));
    gl_Position = vec4(position.xy + offset, 0.0, 1.0);
    out_color = color;
}
