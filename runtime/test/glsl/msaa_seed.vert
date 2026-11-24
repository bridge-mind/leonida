#version 450

// Four instances cover this layer. The fragment shader writes each one to its own sample.
layout(location = 0) flat out int tag;
void main() {
    const vec2 vertices[3] = vec2[](vec2(-1, -1), vec2(3, -1), vec2(-1, 3));
    gl_Position = vec4(vertices[gl_VertexIndex], 0, 1);
    tag = gl_InstanceIndex;
}
