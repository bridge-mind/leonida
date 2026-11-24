#version 460
// A vertex shader with no vertex inputs, written to the translator's runtime contract (docs/design.md, "User data"),
// standing in for a translated one in d3d12_depth_resolve_test: vertices 0, 1 and 2 make one triangle at depth 0.5
// whose long edge runs diagonally across the viewport, so the pixels it crosses have some samples covered and some
// not. Its HLSL twin for the host runtime is in the test.

layout(push_constant) uniform UserData {
    uint s[32];
    uvec2 heap_base;
    uvec2 code_base;
} pc;

void main() {
    // (-1,-1), (-1,1), (1,1) in D3D clip space: the upper-left half of the viewport, cut by the diagonal from the
    // bottom-left corner to the top-right one.
    const vec2 corners[3] = vec2[](vec2(-1.0, -1.0), vec2(-1.0, 1.0), vec2(1.0, 1.0));
    gl_Position = vec4(corners[gl_VertexIndex], 0.5, 1.0);
}
