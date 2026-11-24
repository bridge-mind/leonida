#version 450
layout(location=0) in float marker;
layout(location=0) out vec4 color;
void main() { color = vec4(marker, gl_FragCoord.z, gl_FrontFacing ? 1.0 : 0.0, 1.0); }
