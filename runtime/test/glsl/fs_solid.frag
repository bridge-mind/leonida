#version 460
// Solid colour fragment shader for d3d12_indirect_rt_test.
layout(location = 0) out vec4 out_color;
void main() {
    out_color = vec4(32.0 / 255.0, 160.0 / 255.0, 240.0 / 255.0, 1.0);
}
