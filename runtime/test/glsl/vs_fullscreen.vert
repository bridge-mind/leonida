#version 460
// A vertex shader with no vertex inputs, written to the translator's runtime contract (docs/design.md, "User data"),
// standing in for a translated full-screen pass in d3d12_draw_test: its pipeline has no vertex bindings. Vertices 0,
// 1 and 2 make one triangle that covers the whole viewport, and the colour is yellow.

layout(location = 0) out vec4 out_color;

layout(push_constant) uniform UserData {
    uint s[32];
    uvec2 heap_base;
    uvec2 code_base;
} pc;

void main() {
    // (-1,-1), (-1,3), (3,-1): clockwise in D3D screen space, so front-facing with the default winding.
    const vec2 corner = vec2(float(gl_VertexIndex & 2), float((gl_VertexIndex << 1) & 2));
    gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
    out_color = vec4(1.0, 1.0, 0.0, 1.0);
}
