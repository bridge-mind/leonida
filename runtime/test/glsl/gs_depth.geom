#version 460
// A pass-through geometry shader with the interface of a translated one (docs/design.md, "Geometry shaders":
// triangles in, a triangle strip out), for d3d12_depth24_draw_test: a VS + GS pipeline without a pixel shader still
// gets the built-in D24 depth stage. It reads no input but the position and writes none but gl_Position.

layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

void main() {
    for (int i = 0; i < 3; ++i) {
        gl_Position = gl_in[i].gl_Position;
        EmitVertex();
    }
    EndPrimitive();
}
