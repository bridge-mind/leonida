#version 460
// A pixel shader with no outputs, for d3d12_depth24_draw_test: it stands in for a translated shader in a pipeline
// with no render target, so the layer's D24 depth edit (src/depth24_fragment.h) runs on a shader it did not build.

layout(push_constant) uniform UserData {
    uint s[32];
    uvec2 heap_base;
    uvec2 code_base;
} pc;

void main() {}
