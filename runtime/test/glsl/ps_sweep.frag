#version 460
// A pixel shader standing in for a translated one in d3d12_state_sweep_test: one constant colour. The host draw's
// HLSL writes the same.
layout(location = 0) out vec4 target;

void main() {
    target = vec4(0.75, 0.5, 0.25, 0.5);
}
