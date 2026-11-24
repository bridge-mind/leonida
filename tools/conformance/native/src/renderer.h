#pragma once

#include <d3d12.h>

#include <filesystem>

#include "gpu.h"
#include "pixel_surface.h"
#include "scene_plan.h"

// A D3D12 renderer fixed to the contract's 256x256 R8G8B8A8_UNORM target with a
// D32_FLOAT depth buffer, no swapchain. Owns every GPU object it needs; the only state
// that lives outside is the device and queue in GpuContext.
//
// It does not create a window, pump messages, present, or touch D:\conform\spec; the
// shaders are loaded from disk as compiled by the build.
struct Renderer {
    // Descriptor heaps. One RTV and one DSV is all the contract needs.
    ComPtr<ID3D12DescriptorHeap> render_target_view_heap;
    ComPtr<ID3D12DescriptorHeap> depth_stencil_view_heap;
    D3D12_CPU_DESCRIPTOR_HANDLE color_render_target_view{};
    D3D12_CPU_DESCRIPTOR_HANDLE depth_stencil_view{};

    // Render and readback targets.
    ComPtr<ID3D12Resource> color_target;
    ComPtr<ID3D12Resource> depth_target;
    ComPtr<ID3D12Resource> readback_buffer;

    // Geometry, one upload buffer per array in scenes.h.
    ComPtr<ID3D12Resource> depth_order_vertex_buffer;
    ComPtr<ID3D12Resource> depth_order_index_buffer;
    ComPtr<ID3D12Resource> indexed_vertex_buffer;
    ComPtr<ID3D12Resource> indexed_index_buffer;
    D3D12_VERTEX_BUFFER_VIEW depth_order_vertex_view{};
    D3D12_INDEX_BUFFER_VIEW depth_order_index_view{};
    D3D12_VERTEX_BUFFER_VIEW indexed_vertex_view{};
    D3D12_INDEX_BUFFER_VIEW indexed_index_view{};

    // One constant-buffer slot per scene: the contract now gives each scene its own
    // tint, and a root CBV must sit on a 256-byte boundary. Filled once before any draw,
    // so no CPU write can race a GPU read.
    ComPtr<ID3D12Resource> tint_constants;

    // Pipelines and the one root signature they share.
    ComPtr<ID3D12RootSignature> root_signature;
    ComPtr<ID3D12PipelineState> fullscreen_pipeline;
    ComPtr<ID3D12PipelineState> blend_pipeline;
    ComPtr<ID3D12PipelineState> vertex_color_pipeline;

    // Command recording objects. One list is enough because every scene waits for the
    // GPU before the next reset.
    ComPtr<ID3D12CommandAllocator> command_allocator;
    ComPtr<ID3D12GraphicsCommandList> command_list;

    // Fixed state shared by every scene.
    D3D12_VIEWPORT viewport{};

    // Creates every resource, pipeline and command object. Does not render anything and
    // does not verify output; RenderScene does both of those.
    bool Initialize(GpuContext& gpu, const std::filesystem::path& shader_directory);

    // Renders one scene, waits for the GPU, and fills surface_out with the readback
    // pixels. Does not compare them against the contract; VerifyChecks does that.
    bool RenderScene(GpuContext& gpu, const DrawPlan& plan, PixelSurface& surface_out);
};
