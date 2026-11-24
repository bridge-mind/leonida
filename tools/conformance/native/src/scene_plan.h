#pragma once

#include <d3d12.h>

#include "scenes.h"

// Which of the three pipeline states the draw needs. The PSO carries blend and depth
// state, so this choice fixes both at once; the scissor rect and the depth-target binding
// are per-scene state, because D3D12 always applies the scissor and allows a null DSV.
enum class PipelineKind {
    Fullscreen,  // oversized triangle, no depth, no blend
    Blend,       // identical but with the contract's blend state
    VertexColor, // vertex buffer + 16-bit indices, depth LESS with write
};

enum class GeometryKind {
    None,              // clear only
    FullscreenTriangle, // SV_VertexID, three vertices, no buffers
    DepthOrderQuads,   // conform::kDepthOrderVertices / kDepthOrderIndices
    IndexedTriangles,  // conform::kIndexedVertices / kIndexedIndices
};

// Everything one scene needs bound, resolved from the contract before any GPU call so
// the render loop contains no per-scene branching.
struct DrawPlan {
    PipelineKind pipeline = PipelineKind::Fullscreen;
    GeometryKind geometry = GeometryKind::None;
    int tint_slot = 0;               // which 256-byte slot holds this scene's float4
    bool binds_depth_target = false; // true only for depth_order, per the baseline state
    D3D12_RECT scissor{};
};

// Maps a contract scene to pipeline, geometry, constant-buffer slot, depth binding and
// scissor. scene_index is the scene's position in conform::kScenes and selects its tint.
// Pure function: reads only scenes.h values and touches no device state.
DrawPlan MakeDrawPlan(const conform::Scene& scene, int scene_index);
