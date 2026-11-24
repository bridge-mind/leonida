#version 460
// A geometry shader with the interface of a translated one (docs/design.md, "Geometry shaders": triangles in, a
// triangle strip out, inputs and outputs at the signature's locations), for d3d12_geometry_test. It emits every
// input triangle twice: as it is, and moved right by one in clip space, so a quad drawn in the top-left quadrant
// also covers the top-right one. The colour at location 0 (vs_draw.vert's output) passes through.

layout(triangles) in;
layout(triangle_strip, max_vertices = 6) out;

layout(location = 0) in vec4 in_color[];
layout(location = 0) out vec4 out_color;

void main() {
    for (int copy = 0; copy < 2; ++copy) {
        for (int i = 0; i < 3; ++i) {
            gl_Position = gl_in[i].gl_Position + vec4(float(copy), 0.0, 0.0, 0.0);
            out_color = in_color[i];
            EmitVertex();
        }
        EndPrimitive();
    }
}
