#version 460
// Integer-input counterpart of vs_draw.vert: COLOR uses a signed-integer vertex format, so its shader input is
// ivec4. Positions and outputs are the same, so the draw path can be checked with the same PS.

layout(location = 0) in ivec4 color;
layout(location = 1) in vec4 position;

layout(location = 0) out vec4 out_color;

layout(push_constant) uniform UserData {
    uint s[32];
    uvec2 heap_base;
    uvec2 code_base;
} pc;

void main() {
    const vec2 offset = vec2(uintBitsToFloat(pc.s[0]), uintBitsToFloat(pc.s[1]));
    gl_Position = vec4(position.xy + offset, 0.0, 1.0);
    out_color = vec4(color);
}
