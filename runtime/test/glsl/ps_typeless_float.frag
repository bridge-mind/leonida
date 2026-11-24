#version 460
// A pixel shader standing in for a translated one in d3d12_typeless_families_test (D3D-275): a constant that FLOAT
// keeps, and UNORM views clamp (-2 to 0, 1.5 to 1) and round (0.25). The host draw's HLSL writes the same.
layout(location = 0) out vec4 target;

void main() {
    target = vec4(0.5, -2.0, 1.5, 0.25);
}
