#version 460
// The UINT twin of ps_draw.frag, for the logic-op pass of d3d12_draw_test: D3D12 allows logic ops only on UINT
// render targets, so the pixel shader writes integers. The output is the vertex color as 0..255 (no tint).
layout(location = 0) in vec4 color;
layout(location = 0) out uvec4 target;

void main() {
    target = uvec4(round(color * 255.0));
}
