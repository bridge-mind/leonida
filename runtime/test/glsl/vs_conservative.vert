#version 450
// Synthetic screen-space coverage fixture; no game shader or user data.
layout(location=0) in vec2 position;
void main() { gl_Position = vec4(position, 0.5, 1.0); }
