#version 460
// Fullscreen triangle for d3d12_indirect_rt_test (ExecuteIndirect with CountBuffer).
void main() {
    const vec2 pos[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    gl_Position = vec4(pos[gl_VertexIndex], 0.0, 1.0);
}
