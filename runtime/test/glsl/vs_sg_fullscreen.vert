#version 460
// One triangle that covers the whole viewport, from gl_VertexIndex alone, for d3d12_subgroup_test.
void main() {
    const vec2 p = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
