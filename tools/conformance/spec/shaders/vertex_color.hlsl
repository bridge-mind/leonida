// Clip-space position with a per-vertex colour.
// Used by the depth-order and indexed scenes: the two quads/triangles differ in
// depth or index order, and the colour comes from the vertex data so a wrong draw
// is visible as the wrong colour rather than the wrong shade.

struct VsIn
{
    float3 position : POSITION;  // already in clip space: xy in -1..1, z in 0..1
    float4 color    : COLOR;
};

struct VsOut
{
    float4 position : SV_Position;
    float4 color    : COLOR;
};

VsOut VSMain(VsIn input)
{
    VsOut output;
    output.position = float4(input.position, 1.0);
    output.color = input.color;
    return output;
}

float4 PSMain(VsOut input) : SV_Target
{
    return input.color;
}
