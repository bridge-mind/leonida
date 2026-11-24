#version 450
#extension GL_NV_conservative_raster_underestimation : require
// Explicit synthetic input contract; not an inferred Xbox VGPR mapping.
layout(location=0) out vec4 color;
void main() {
    uint inner = gl_FragFullyCoveredNV ? 1u : 0u;
    color = vec4(1.0, (inner & 1u) != 0u ? 1.0 : 0.0, inner == 0u ? 1.0 : 0.0, 1.0);
}
