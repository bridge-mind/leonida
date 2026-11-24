#version 450

layout(location = 0) flat in int tag;
layout(location = 0) out vec4 color;
void main() {
    gl_SampleMask[0] = 1 << (tag % 4);
    color = vec4(floor(gl_FragCoord.xy), tag + 1, 255) / 255.0;
}
