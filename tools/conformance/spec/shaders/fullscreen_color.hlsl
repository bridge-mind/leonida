// Fullscreen triangle, one colour from a constant buffer.
// Used by the clear, fullscreen, scissor, blend and constant-buffer scenes: they
// differ only in pipeline state and the value in the CB, so one shader pair covers
// them and a failure points at state, not at shader compilation.

cbuffer SceneConstants : register(b0)
{
    float4 tint;  // straight (non-premultiplied) RGBA in 0..1
};

struct VsOut
{
    float4 position : SV_Position;
};

VsOut VSMain(uint vertexId : SV_VertexID)
{
    // One oversized triangle covering clip space; no vertex buffer needed.
    //   id 0 -> (-1,-1)   id 1 -> (-1, 3)   id 2 -> (3,-1)
    float2 corners[3] =
    {
        float2(-1.0, -1.0),
        float2(-1.0,  3.0),
        float2( 3.0, -1.0),
    };

    VsOut output;
    output.position = float4(corners[vertexId], 0.0, 1.0);
    return output;
}

float4 PSMain(VsOut input) : SV_Target
{
    return tint;
}
