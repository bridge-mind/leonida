#include "scene_plan.h"

#include <cstdio>
#include <cstdlib>

DrawPlan MakeDrawPlan(const conform::Scene& scene, int scene_index) {
    DrawPlan plan{};
    plan.tint_slot = scene_index;
    // The baseline scissor is the full target; only the scissor scene narrows it.
    plan.scissor = {0, 0, conform::kSceneWidth, conform::kSceneHeight};

    switch (scene.kind) {
        case conform::SceneKind::Clear:
            // No draw at all: the contract tests the clear and the readback path.
            plan.geometry = GeometryKind::None;
            break;
        case conform::SceneKind::Fullscreen:
            plan.geometry = GeometryKind::FullscreenTriangle;
            break;
        case conform::SceneKind::DepthOrder:
            plan.pipeline = PipelineKind::VertexColor;
            plan.geometry = GeometryKind::DepthOrderQuads;
            // The only scene that binds a depth target; every other scene passes a null
            // DSV, which is what the contract's baseline table says.
            plan.binds_depth_target = true;
            break;
        case conform::SceneKind::Scissor:
            // Same pipeline as fullscreen. D3D12 applies the scissor test whenever a rect
            // is set, with no enable flag in the pipeline state, so this scene proves
            // RSSetScissorRects. Right and bottom are exclusive.
            plan.geometry = GeometryKind::FullscreenTriangle;
            plan.scissor = {conform::kScissorLeft, conform::kScissorTop, conform::kScissorRight,
                            conform::kScissorBottom};
            break;
        case conform::SceneKind::Blend:
            plan.pipeline = PipelineKind::Blend;
            plan.geometry = GeometryKind::FullscreenTriangle;
            break;
        case conform::SceneKind::Indexed:
            plan.pipeline = PipelineKind::VertexColor;
            plan.geometry = GeometryKind::IndexedTriangles;
            break;
        default:
            // Unreachable while every SceneKind has a case above. If scenes.h gains a kind
            // and this switch is forgotten, /we4062 turns the missing case into a build
            // error; this line is the belt for a compiler that does not implement it.
            std::fprintf(stderr, "internal: scene kind %d has no draw plan\n",
                         static_cast<int>(scene.kind));
            std::abort();
    }
    return plan;
}
