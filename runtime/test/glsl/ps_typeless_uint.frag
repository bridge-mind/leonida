#version 460
// A pixel shader standing in for a translated one in d3d12_typeless_families_test (D3D-275): an integer constant
// every UINT view down to R10G10B10A2_UINT holds (7 is 0x00000007 in R32_UINT, not a float's bits). The host draw's
// HLSL writes the same.
layout(location = 0) out uvec4 target;

void main() {
    target = uvec4(7u, 300u, 1000u, 3u);
}
