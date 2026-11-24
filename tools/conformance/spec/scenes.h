// Conformance scene contract — the single source of truth shared by the native D3D12
// reference renderer, the Vulkan-shim renderer, and the image diff tool.
//
// Every scene is deterministic and analytically checkable: it names the exact pixels
// that must come out, so a mismatch is a bug with a coordinate, not a vague "the
// picture looks off". Where a value could be read two ways, this header states the
// reading; if an implementer has to guess, that is a defect in this header.
//
// Conventions, stated once and assumed everywhere below:
//   * clip space is D3D style: x and y in -1..1, z in 0..1, w = 1.0. Depth == z.
//     There is no GL-style (z+1)/2 remap anywhere in this suite.
//   * image row 0 is the TOP of the image and corresponds to NDC y = +1.
//     Byte order in memory is R, G, B, A.
//   * the render target is R8G8B8A8_UNORM, so writing v in 0..1 stores round(v * 255)
//     and reading gives the same 8-bit value back. No sRGB conversion, no dithering.
#pragma once

#include <cstdint>

namespace conform {

constexpr int kSceneWidth = 256;
constexpr int kSceneHeight = 256;

// Clear colour, in 8-bit UNORM, shared by every scene.
constexpr uint8_t kClearR = 10;
constexpr uint8_t kClearG = 20;
constexpr uint8_t kClearB = 30;
constexpr uint8_t kClearA = 255;

// Pixel value that the clear colour writes, restated for checks and comments.
inline constexpr uint8_t kClearBytes[4] = {kClearR, kClearG, kClearB, kClearA};

// ---------------------------------------------------------------------------------
// Baseline pipeline state. Every scene starts from this and only the listed scene
// differences change. A zero-initialised D3D12_RASTERIZER_DESC is invalid, and a zero
// sample mask draws nothing, so this table is not optional documentation: it is the
// state an implementer must set.
// ---------------------------------------------------------------------------------
//
//   topology            D3D12_PRIMITIVE_TOPOLOGY_TRIANGLELIST (never a strip: a strip
//                       would bridge the depth-order quads with phantom triangles)
//   cull mode           D3D12_CULL_MODE_NONE, fill SOLID, FrontCounterClockwise FALSE
//   sample count        1, sample mask 0xFFFFFFFF, no MSAA
//   render target       R8G8B8A8_UNORM, RenderTargetWriteMask = ALL (0x0F)
//   viewport            (0, 0, 256, 256), depth 0..1
//   scissor             the full 256x256 target unless the scene says otherwise;
//                       scissor is always enabled in D3D12, so it must be set explicitly
//   root signature      one root CBV at b0, shader visibility ALL, space 0,
//                       flags = ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT (vertex-colour scenes
//                       also bind a vertex buffer and use the same signature unused)
//   constant buffer     a single float4 at byte offset 0, set to tint/255.0f
//   depth               only the depth_order scene binds a depth target, and it binds
//                       D32_FLOAT, clears to 1.0, DepthFunc LESS, DepthWriteMask ALL,
//                       stencil disabled. Every other scene binds no DSV.
//   vertex format       ClipVertex as declared below, stride sizeof(ClipVertex)
//   index format        R16_UINT for every indexed draw
//
// Each scene clears the render target to the clear colour before it draws, so state
// left behind by a previous scene can never explain a pass.

// A vertex in clip space with a straight (non-premultiplied) colour.
// w is 1.0 everywhere, so z is the depth value written to SV_Position.
struct ClipVertex {
    float x;
    float y;
    float z;
    uint8_t r;
    uint8_t g;
    uint8_t b;
    uint8_t a;
};

// depth_order: the NEAR green quad is drawn FIRST, the FAR red quad SECOND.
//
// This order is deliberate and load-bearing. At the overlap, the red quad is drawn
// after the green one, so the only reason green survives is that the depth test rejects
// red (0.75 vs 0.25, LESS). A shim that ignores the depth target paints red over green
// and fails the check at (128,128).
//
// Green spans x -0.95..0.15, red spans x -0.95..0.95, so the right-hand side of the
// image shows red with nothing over it.
inline constexpr ClipVertex kDepthOrderVertices[] = {
    // near quad: green, z = 0.25, drawn first
    {-0.95f, -0.95f, 0.25f, 40, 160, 90, 255},
    {-0.95f,  0.95f, 0.25f, 40, 160, 90, 255},
    { 0.15f,  0.95f, 0.25f, 40, 160, 90, 255},
    { 0.15f, -0.95f, 0.25f, 40, 160, 90, 255},
    // far quad: red, z = 0.75, drawn second
    {-0.95f, -0.95f, 0.75f, 220, 40, 40, 255},
    {-0.95f,  0.95f, 0.75f, 220, 40, 40, 255},
    { 0.95f,  0.95f, 0.75f, 220, 40, 40, 255},
    { 0.95f, -0.95f, 0.75f, 220, 40, 40, 255},
};
// Two triangles per quad, both wound counter-clockwise in NDC, which is irrelevant
// while culling is NONE but stated so the geometry has one meaning.
inline constexpr uint16_t kDepthOrderIndices[] = {
    0, 1, 2, 0, 2, 3,  // near green quad
    4, 5, 6, 4, 6, 7,  // far red quad
};

// indexed: six vertices stored in an order that is NOT the draw order, addressed by a
// non-identity index list. A program that ignores the index buffer and draws the vertex
// buffer in order produces the wrong triangles, which is the failure this scene exists
// to catch.
//
// Vertex layout in memory (deliberately shuffled):
//   slot 0 = left-bottom, slot 1 = right-top, slot 2 = left-top,
//   slot 3 = right-bottom, slot 4 = right-top, slot 5 = left-bottom
inline constexpr ClipVertex kIndexedVertices[] = {
    {-0.60f, -0.60f, 0.50f, 240, 200, 40, 255},  // slot 0: left triangle, bottom left
    { 0.60f, -0.60f, 0.50f, 40, 120, 240, 255},  // slot 1: right triangle, bottom right
    {-0.60f,  0.60f, 0.50f, 240, 200, 40, 255},  // slot 2: left triangle, top left
    { 0.10f,  0.60f, 0.50f, 40, 120, 240, 255},  // slot 3: right triangle, top left
    { 0.10f, -0.60f, 0.50f, 40, 120, 240, 255},  // slot 4: right triangle, bottom left
    {-0.10f, -0.60f, 0.50f, 240, 200, 40, 255},  // slot 5: left triangle, bottom right
};
inline constexpr uint16_t kIndexedIndices[] = {
    0, 2, 5,  // left triangle  (slots 0, 2, 5)
    1, 3, 4,  // right triangle (slots 1, 3, 4)
};

// Scissor rectangle for the scissor scene: D3D12_RECT{left, top, right, bottom} with
// right and bottom EXCLUSIVE. Deliberately asymmetric in both axes so that reading the
// rectangle as (x, y, width, height) - the usual mistake - fails the checks.
constexpr int kScissorLeft = 48;
constexpr int kScissorTop = 32;
constexpr int kScissorRight = 176;
constexpr int kScissorBottom = 96;

struct PixelCheck {
    int x;
    int y;
    uint8_t r;
    uint8_t g;
    uint8_t b;
    uint8_t a;
    const char* why;  // what this pixel proves, phrased as a claim
};

enum class SceneKind {
    Clear,       // clear only, no draw
    Fullscreen,  // one oversized triangle, colour from the constant buffer at b0
    DepthOrder,  // kDepthOrderVertices / kDepthOrderIndices, depth target bound
    Scissor,     // oversized triangle restricted to the scissor rectangle
    Blend,       // oversized triangle, source alpha 0.5, SRC_ALPHA / INV_SRC_ALPHA
    Indexed,     // kIndexedVertices / kIndexedIndices
};

struct Scene {
    const char* name;    // also the PNG name, without extension
    SceneKind kind;
    const char* proves;  // one sentence: the failure this scene would catch

    // Constant-buffer value for the scenes that read it, in 8-bit UNORM.
    // Each scene uses a distinct colour so that state leaking from a previous scene
    // changes the image instead of hiding behind an identical tint.
    uint8_t tintR;
    uint8_t tintG;
    uint8_t tintB;
    uint8_t tintA;

    int checkCount;
    PixelCheck checks[5];
};

// Blend arithmetic, stated so there is nothing to guess:
//   source colour (200, 30, 60) with source alpha 0.5, over the clear colour:
//   colour: SRC_ALPHA, INV_SRC_ALPHA, ADD
//   alpha:  ONE,       INV_SRC_ALPHA, ADD   -> 1.0*0.5 + 0.5*1.0 = 1.0 -> 255
//   0.5*200 + 0.5*10 = 105, 0.5*30 + 0.5*20 = 25, 0.5*60 + 0.5*30 = 45
// With the source alpha quantised to either 127 or 128 the result is still 105/25/45,
// so no tolerance is needed.
inline constexpr Scene kScenes[] = {
    {
        "clear", SceneKind::Clear,
        "render target creation, the clear value and the readback path are wired up",
        10, 20, 30, 255,   // tint unused
        3,
        {
            {0, 0, kClearR, kClearG, kClearB, kClearA, "top-left is the clear colour"},
            {128, 128, kClearR, kClearG, kClearB, kClearA, "centre is the clear colour"},
            {255, 255, kClearR, kClearG, kClearB, kClearA, "bottom-right is the clear colour"},
        },
    },
    {
        "fullscreen", SceneKind::Fullscreen,
        "vertex shader runs, a pipeline binds, and one triangle covers every pixel",
        200, 30, 60, 255,
        3,
        {
            {128, 128, 200, 30, 60, 255, "centre is the constant-buffer colour"},
            {8, 8, 200, 30, 60, 255, "top-left corner is covered: the triangle is oversized"},
            {247, 247, 200, 30, 60, 255, "bottom-right corner is covered too"},
        },
    },
    {
        "depth_order", SceneKind::DepthOrder,
        "the depth test rejects a later draw that is behind: without it this scene is red",
        0, 0, 0, 255,      // tint unused
        2,
        {
            {128, 128, 40, 160, 90, 255, "overlap keeps the near quad drawn first (green, z=0.25) although red (z=0.75) was drawn after it"},
            {224, 128, 220, 40, 40, 255, "uncovered right side shows the far quad (red)"},
        },
    },
    {
        "scissor", SceneKind::Scissor,
        "scissor state is honoured: draws are clipped at the rectangle, not merely darkened",
        70, 140, 210, 255,
        5,
        {
            {112, 64, 70, 140, 210, 255, "inside the rectangle the draw lands"},
            {40, 64, kClearR, kClearG, kClearB, kClearA, "left of the rectangle stays clear"},
            {180, 64, kClearR, kClearG, kClearB, kClearA, "right of the rectangle stays clear"},
            {112, 24, kClearR, kClearG, kClearB, kClearA, "above the rectangle stays clear"},
            {112, 104, kClearR, kClearG, kClearB, kClearA, "below the rectangle stays clear"},
        },
    },
    {
        "blend", SceneKind::Blend,
        "blend state is applied: 50% source over the clear colour, not a replacement",
        200, 30, 60, 128,  // 128/255 ~= 0.5, which rounds to the same result as 127
        2,
        {
            {128, 128, 105, 25, 45, 255, "0.5*(200,30,60) + 0.5*(10,20,30) = (105,25,45)"},
            {8, 8, 105, 25, 45, 255, "blending applies everywhere the triangle covers"},
        },
    },
    {
        "indexed", SceneKind::Indexed,
        "the index buffer is read: a program that ignores it draws different triangles",
        0, 0, 0, 255,      // tint unused
        3,
        {
            {80, 200, 240, 200, 40, 255, "left triangle, from index triple 0,2,5"},
            {176, 200, 40, 120, 240, 255, "right triangle, from index triple 1,3,4"},
            {128, 240, kClearR, kClearG, kClearB, kClearA, "the gap between the two triangles stays clear"},
        },
    },
};

inline constexpr int kSceneCount = static_cast<int>(sizeof(kScenes) / sizeof(kScenes[0]));

}  // namespace conform
