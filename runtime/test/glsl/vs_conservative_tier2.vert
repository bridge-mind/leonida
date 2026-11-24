#version 450
layout(location=0) in vec3 position;
layout(location=1) in float marker;
layout(location=0) out float out_marker;
void main() { gl_Position = vec4(position, 1.0); out_marker = marker; }
