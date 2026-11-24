// D3D12 COM objects over the Vulkan backend (src/vulkan_backend.h): command allocator, graphics command list,
// command queue and fence. Their vtables are the Windows SDK's (ID3D12GraphicsCommandList8, ID3D12CommandQueue,
// ID3D12CommandAllocator, ID3D12Fence1); every method not implemented here is a generated stub that reports its
// name once (tools/gen_d3d12_stubs.py). The Xbox-specific slots measured for the device do not apply to these
// interfaces as far as is known; docs/d3d12_vulkan.md section 3.1 lists what is still unverified.
//
// Contracts with the rest of the layer (the device proxy that will hand these objects out):
// - an ID3D12Resource's leonida::ResourceId is its pointer value; the device registers it with the VulkanContext;
// - so is an ID3D12QueryHeap's leonida::QueryHeapId, registered with its VkQueryPool;
// - the CPU handle of an RTV or DSV points at a leonida::TargetView in the device's descriptor heap.
#pragma once

#include <windows.h>
#include <d3d12.h>
#include <memory>
#include <vector>

#include "../src/gfx10_descriptors.h"
#include "../src/vulkan_backend.h"

#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "../src/root_signature.h"
#include "../src/subgroup_size.h"

namespace leonida {
class VulkanPresenter;
struct PresentPlanes;
}

namespace rt::d3d12 {

// How a queue answers GetClockCalibration. With `calibrated` (vkGetCalibratedTimestampsKHR or its EXT alias, on a
// device whose time domains include DEVICE and QUERY_PERFORMANCE_COUNTER) both clocks are read in one call. Without
// it, the queue writes a timestamp query and reads QueryPerformanceCounter around it; that needs `get_device_proc`
// and a queue family with timestamp_valid_bits > 0. With neither, GetClockCalibration fails.
struct QueueClock {
    PFN_vkGetCalibratedTimestampsKHR calibrated = nullptr;
    PFN_vkGetDeviceProcAddr get_device_proc = nullptr;
    std::uint32_t timestamp_valid_bits = 0;
};

// `device` (optional) is what the object's GetDevice returns; the object holds a reference to it.
HRESULT create_command_queue(leonida::VulkanContext& context, VkQueue queue, std::uint32_t queue_family,
                             const D3D12_COMMAND_QUEUE_DESC& desc, REFIID riid, void** out,
                             IUnknown* device = nullptr, const QueueClock& clock = {},
                             std::shared_ptr<void> queue_lease = {});
HRESULT create_command_allocator(D3D12_COMMAND_LIST_TYPE type, REFIID riid, void** out, IUnknown* device = nullptr);
HRESULT create_command_list(D3D12_COMMAND_LIST_TYPE type, ID3D12CommandAllocator* allocator, REFIID riid,
                            void** out, IUnknown* device = nullptr, ID3D12PipelineState* initial_state = nullptr);
HRESULT create_fence(leonida::VulkanContext& context, UINT64 initial_value, D3D12_FENCE_FLAGS flags, REFIID riid,
                     void** out, IUnknown* device = nullptr);
HRESULT create_command_signature(leonida::IndirectKind kind, UINT byte_stride, REFIID riid, void** out,
                                 IUnknown* device = nullptr);
bool command_signature_info(ID3D12CommandSignature* signature, leonida::IndirectKind& kind, UINT& byte_stride);

// ID3D12Device1::SetEventOnMultipleFenceCompletion on fences from create_fence: sets `event` once every fence
// (D3D12_MULTIPLE_FENCE_WAIT_FLAG_ALL) or any fence (_ANY) has reached its value; a null event blocks until then.
HRESULT set_event_on_multiple_fence_completion(leonida::VulkanContext& context, ID3D12Fence* const* fences,
                                               const UINT64* values, UINT count,
                                               D3D12_MULTIPLE_FENCE_WAIT_FLAGS flags, HANDLE event);

std::shared_ptr<const leonida::TileHeapAllocation> tile_heap(ID3D12Heap* heap, const leonida::VulkanContext& context,
                                                           bool image, std::uint32_t memory_types);
bool is_sparse_image(ID3D12Resource* resource);

// The Vulkan objects a D3D12 device runs on. Vulkan 1.3 with dynamicRendering, synchronization2, timelineSemaphore
// and (when separate_depth_stencil_layouts is set) separateDepthStencilLayouts enabled.
struct VulkanDeviceInfo {
    PFN_vkGetInstanceProcAddr get_instance_proc = nullptr;
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;  // graphics queue used by staging and legacy single-queue callers
    std::uint32_t queue_family = 0;
    // Distinct handles from queue_family, when the creator requested more than one queue.
    // Empty for caller-owned devices that only supplied queue above.
    std::vector<VkQueue> command_queues;
    std::uint32_t compute_family = UINT32_MAX;
    std::uint32_t copy_family = UINT32_MAX;
    std::vector<VkQueue> compute_queues;
    std::vector<VkQueue> copy_queues;
    bool separate_depth_stencil_layouts = true;
    std::uint32_t max_push_constants_size = 0;
    bool push_constants_need_fallback = false;
    // maxBufferSize and maxMemoryAllocationSize, which cap the GPU address arena (src/arena_size.h). 0: read from
    // the physical device when the D3D12 device is created; a test sets smaller ones to simulate a device.
    VkDeviceSize max_buffer_size = 0;
    VkDeviceSize max_memory_allocation_size = 0;
    // The subgroup sizes the device offers and whether subgroupSizeControl / computeFullSubgroups are enabled on it
    // (src/subgroup_size.h). Default: nothing, so pipelines request no size.
    leonida::SubgroupSizeCaps subgroup;
    // Enabled: runtimeDescriptorArray, the dynamic indexing of sampled- and storage-image arrays, and update after
    // bind for sampled images, storage images and storage buffers. The device then keeps the bindless set 0 of
    // the translated shaders (docs section 3.4), and texture views and samplers are put in it as they are
    // created. bindless_partially_bound: descriptorBindingPartiallyBound is enabled as well.
    bool bindless = false;
    bool bindless_partially_bound = false;
    // samplerAnisotropy, samplerMirrorClampToEdge and samplerFilterMinmax enabled; without them such samplers are
    // approximated.
    bool sampler_anisotropy = false;
    bool sampler_mirror_clamp_to_edge = false;
    bool sampler_filter_minmax = false;
    // VK_KHR_calibrated_timestamps or VK_EXT_calibrated_timestamps is enabled on `device` (QueueClock).
    bool calibrated_timestamps = false;
    // VK_EXT_conditional_rendering (with its conditionalRendering feature) is enabled: SetPredication works, and
    // buffers are created with VK_BUFFER_USAGE_CONDITIONAL_RENDERING_BIT_EXT. Without it a predicated list is
    // rejected at ExecuteCommandLists.
    bool conditional_rendering = false;
    bool storage_buffer_8bit = false; // storageBuffer8BitAccess enabled: byte-granular predicated buffer copies.
    // VK_EXT_conservative_rasterization is enabled on the logical device. Tier 1/2 depend on the physical
    // precision/overestimation properties; Tier 2 also requires post-snap degenerate triangles.
    bool conservative_rasterization = false;
    // Opt in only for shader libraries whose InnerCoverage input contract uses FullyCoveredEXT. The device
    // still checks physical Tier 2 and fullyCoveredFragmentShaderInputVariable properties. The standalone
    // device leaves this false until the Xbox PS input mapping is confirmed (synthetic GLSL is not that ABI).
    bool inner_coverage_abi = false;
    // VK_EXT_host_image_copy and its hostImageCopy feature are enabled on `device`: WriteToSubresource and
    // ReadFromSubresource on textures in CPU-visible heaps copy on the CPU instead of through a staging submit.
    bool host_image_copy = false;
    // VK_KHR_shader_fma and its shaderFmaFloat32 feature are enabled on `device`: translated shaders built with
    // `rdna2spirv.py --fma khr` (OpFmaKHR, capability FMAKHR in the index) can be created. Without it only the
    // default `--fma glsl` shaders can.
    bool shader_fma = false;
    bool shader_subgroup_clock = false;
    bool shader_device_clock = false;
    // The caller enabled textureCompressionBC on this logical device.
    bool texture_compression_bc = false;
    // Optional device features the caller enabled. Without occlusionQueryPrecise, D3D12_QUERY_TYPE_OCCLUSION counts
    // are only zero or non-zero; without pipelineStatisticsQuery, pipeline-statistics query heaps are refused.
    bool occlusion_query_precise = false;
    bool pipeline_statistics_query = false;
    bool presentation = false;  // VK_KHR_surface + VK_KHR_win32_surface on the instance, VK_KHR_swapchain on the device
    // VK_KHR_swapchain_maintenance1 (or VK_EXT_swapchain_maintenance1) and its swapchainMaintenance1 feature are
    // enabled on the device, with VK_KHR_get_surface_capabilities2 and the matching *_surface_maintenance1 on the
    // instance (only meaningful with `presentation`): a vsync swapchain can switch to IMMEDIATE or FIFO_RELAXED per
    // present, which PresentX's ImmediateThresholdPercent uses.
    bool swapchain_maintenance1 = false;
    // VK_EXT_swapchain_colorspace is enabled on the instance (with `presentation`): the surface can list its HDR10
    // (ST.2084) formats, which a PresentX plane in HDR10 uses. Without it HDR10 planes are tone mapped to SDR.
    bool swapchain_colorspace = false;
    // VK_EXT_image_view_min_lod and its minLod feature are enabled on the logical device.
    bool image_view_min_lod = false;
    // VK_KHR_vertex_attribute_divisor or VK_EXT_vertex_attribute_divisor with vertexAttributeInstanceRateDivisor.
    bool vertex_attribute_divisor = false;
    // Pixel-shader extensions translated shaders can declare, each enabled with its feature where the GPU has it:
    // VK_KHR_fragment_shader_barycentric (BaryCoordKHR / PerVertexKHR reads), VK_EXT_fragment_shader_interlock
    // (fragmentShaderPixelInterlock: POPS shaders run in PixelInterlockOrderedEXT) and VK_KHR_maintenance8 (texel
    // offsets outside gathers: the index's needs_maintenance8).
    bool fragment_shader_barycentric = false;
    bool fragment_shader_interlock = false;
    bool maintenance8 = false;
    // The instance enabled VK_EXT_debug_utils: SetMarker/BeginEvent/EndEvent become debug labels. Otherwise they
    // do nothing.
    bool debug_utils = false;
    // Vulkan 1.2 bufferDeviceAddress is enabled: buffers and shader-visible heaps then live in the GPU address
    // arena and have console VAs (docs section 3.3), and GetGPUVirtualAddress returns those addresses; ray tracing
    // needs it too. Without it, GetGPUVirtualAddress returns 0 and pipelines cannot be created.
    bool buffer_device_address = false;
    // Optional VK_KHR_acceleration_structure + VK_KHR_deferred_host_operations + VK_KHR_ray_query are enabled:
    // GetRaytracingAccelerationStructurePrebuildInfo and BuildRaytracingAccelerationStructure (BLAS + TLAS) work.
    bool ray_tracing = false;
    // BuildRaytracingAccelerationStructure writes the console's XDXR BVH4 records into the destination buffer
    // (src/xdxr_build.h), the layout the game's translated traversal shaders read, instead of building a
    // VkAccelerationStructureKHR; needs buffer_device_address and the GPU address arena. The standalone device sets it
    // unless LEONIDA_RT_AS=khr (docs/d3d12_vulkan.md 4.8); with it set, ray_tracing is not enabled.
    bool xdxr_acceleration_structures = false;
    // VK_EXT_mesh_shader with its meshShader feature is enabled on `device` (taskShader is not needed: the game's
    // shaders have no amplification stage). Mesh-shader PSOs and DispatchMesh work, and CheckFeatureSupport reports
    // D3D12_MESH_SHADER_TIER_1; without it mesh PSOs are refused and the tier is NOT_SUPPORTED.
    bool mesh_shader = false;
    // The core features enabled on `device`. Graphics PSOs that need independentBlend, fillModeNonSolid,
    // depthBiasClamp or dualSrcBlend are refused unless it is set here.
    VkPhysicalDeviceFeatures features{};
    // Vulkan 1.2 shaderOutputLayer / shaderOutputViewportIndex enabled on `device`: a translated vertex shader that
    // writes the layer or viewport declares ShaderLayer / ShaderViewportIndex (the index's `features`).
    bool shader_output_layer = false;
    bool shader_output_viewport_index = false;
    // The translator's output folder for pipeline creation. An empty value uses LEONIDA_SHADERS or the packaged path.
    std::string shader_dir;
    // VK_EXT_custom_border_color (with customBorderColors and customBorderColorWithoutFormat) is enabled on
    // `device`: samplers with custom border colours pass them through to Vulkan. Without it, custom border
    // colours fall back to transparent black.
    bool custom_border_color = false;
    bool custom_border_color_without_format = false;
};

// A Vulkan-backed ID3D12Device (PC ABI, up to ID3D12Device11) on caller-owned Vulkan objects, which must outlive
// the device and every object it created. Wrap it in rt::d3d12_xs::create_device_proxy for the Xbox ABI.
HRESULT create_vulkan_device(const VulkanDeviceInfo& info, REFIID riid, void** out);

// The same, on a Vulkan instance and device of its own (destroyed with the D3D12 device). Picks the first discrete
// GPU with the features above. LEONIDA_VK_VALIDATION=1 enables VK_LAYER_KHRONOS_validation.
//
// The queue test settings (LEONIDA_VK_MAX_QUEUES_PER_FAMILY, LEONIDA_VK_GRAPHICS_FAMILY_ONLY; docs/testing.md) give
// the device fewer Vulkan queues than the GPU has, so the tests run the path where D3D12 queues share one. Only the
// tests apply them: the game's device (d3d12_xs, LEONIDA_D3D12=vulkan) passes `ignore`, so a variable left set does
// not serialize the game's queues; it says once that it ignores them.
enum class QueueTestSettings { apply, ignore };
HRESULT create_vulkan_device_standalone(REFIID riid, void** out,
                                        QueueTestSettings queue_test_settings = QueueTestSettings::apply);

// Tells a device from create_vulkan_device* that `outer` (the Xbox-ABI proxy that owns it) is what its children's
// GetDevice must return. No reference is taken: the outer object owns the device. `outer` may be the proxy's
// IDXGIDevice1 pointer (what create_device_proxy returns for that IID); the proxy itself becomes the outer. For a
// proxy it also makes the proxy's IDXGIDevice1::GetAdapter return this layer's adapter (d3d12_xs_device.h).
void set_device_outer(ID3D12Device* device, IUnknown* outer);

// The VkImage behind a texture created by such a device, or VK_NULL_HANDLE. For tests and the swapchain.
VkImage vulkan_image(ID3D12Resource* resource);
// The VkBuffer behind a buffer created by such a device, or VK_NULL_HANDLE. For tests.
VkBuffer vulkan_buffer(ID3D12Resource* resource);
// What a device from create_vulkan_device* was created with (enabled features included), or null. For tests.
const VulkanDeviceInfo* vulkan_device_info(ID3D12Device* device);
// True when `device` has VK_KHR_acceleration_structure + VK_KHR_ray_query enabled.
bool device_supports_ray_tracing(ID3D12Device* device);
// The VkAccelerationStructureKHR registered for `va` on `device`, or VK_NULL_HANDLE. For tests.
VkAccelerationStructureKHR vulkan_acceleration_structure(ID3D12Device* device, D3D12_GPU_VIRTUAL_ADDRESS va);

// The property flags of the Vulkan memory type behind a resource created by such a device, or 0. For tests.
VkMemoryPropertyFlags vulkan_memory_flags(ID3D12Resource* resource);

// Whether the requested colour format has an integer numeric class.
bool color_resolve_format_is_integer(DXGI_FORMAT format);

// Whether both resources use the requested typed, non-integer colour format for a regional average resolve.
bool color_resolve_format_matches(DXGI_FORMAT format, ID3D12Resource* dst, ID3D12Resource* src);

// Whether a resolve of one depth or stencil plane can be replayed: both resources are depth-stencil textures of
// `format` (a depth format, or a plane's shader-resource format such as R24_UNORM_X8_TYPELESS), the source
// multisampled and the destination not, the same plane on both sides, MIN or MAX (or AVERAGE of the depth plane by
// its own format), the destination offset the source rectangle's corner (or 0, 0 without one), and the Vulkan device
// has the mode.
bool depth_stencil_resolve_supported(DXGI_FORMAT format, ID3D12Resource* dst, UINT dst_subresource, UINT x, UINT y,
                                     ID3D12Resource* src, UINT src_subresource, const D3D12_RECT* rect,
                                     D3D12_RESOLVE_MODE mode);

// The maxPushConstantsSize in bytes of the Vulkan physical device behind `device`, or 0 if not a Vulkan device.
std::uint32_t device_max_push_constants_size(ID3D12Device* device);

// True when `device`'s Vulkan physical device has maxPushConstantsSize < 144 bytes, requiring
// the uniform-buffer fallback for the 16-byte tail (heap/code base).
bool device_push_constants_need_fallback(ID3D12Device* device);

// True when `device` puts upload-heap buffers in DEVICE_LOCAL | HOST_VISIBLE memory: its GPU has Resizable BAR and
// LEONIDA_UPLOAD_SYSMEM is not set (D3D-96).
bool device_upload_in_vram(ID3D12Device* device);

// True when the device can map reserved buffers onto its D3D12 heaps.
bool device_sparse_buffer_tiles(ID3D12Device* device);

// The size in bytes of the device's GPU address arena, or 0 without one. For tests.
std::uint64_t device_arena_size(ID3D12Device* device);

// The bindless set 0 a device with VulkanDeviceInfo::bindless keeps (vulkan_bindless.h), and its layout: what a
// pipeline whose shader declares set-0 bindings is created with and binds. Null without such a set. The set is
// brought up to date at every ExecuteCommandLists.
VkDescriptorSet bindless_set(ID3D12Device* device);
// What the bindless set did without waiting (D3D-314): table generations made because a flush grew the table or
// changed live entries, the retired ones not freed yet, and vkDeviceWaitIdle calls made for it (always 0 unless
// LEONIDA_BINDLESS_IDLE_WAIT=1 restores the old wait, to compare frame times). All 0 without a bindless set.
struct BindlessStats {
    std::uint64_t generations_made = 0;
    std::uint64_t retired_pending = 0;
    std::uint64_t device_idle_waits = 0;
};
BindlessStats bindless_stats(ID3D12Device* device);
VkDescriptorSetLayout bindless_set_layout(ID3D12Device* device);
// The bindless index the descriptor hash table holds for the 8 key dwords of a T# (or the 4 of an S# followed by
// 0xFFFFFFFF x4), 0 (the fallback) when it has none or `device` has no such set. For tests.
std::uint32_t bindless_hash_lookup(ID3D12Device* device, const std::uint32_t (&key)[8]);
// The free indices of that set's 2D image and sampler arrays: how many more distinct 2D T#s and S#s
// CreateShaderResourceView and CreateSampler can register before they log that the array is full. Zeros when
// `device` has no such set. For tests.
struct BindlessRoom {
    std::uint32_t images_2d = 0;
    std::uint32_t samplers = 0;
};
BindlessRoom bindless_room(ID3D12Device* device);
// Gives each S# a sampler in that set and a key in its hash table, as CreateSampler does, without a heap slot: for
// the literal S#s a shader builds from immediates (`literal_samplers` in the shader index). An S# that has them
// already is left as it is. E_INVALIDARG when `device` is not such a device; E_FAIL when one of them could not be
// registered (no bindless set, or its sampler array is full), the others are registered all the same.
HRESULT register_samplers(ID3D12Device* device, std::span<const leonida::gfx10::SamplerWords> samplers);
// True when GetClockCalibration on `queue` reads calibrated timestamps, false when it submits a timestamp query
// (or `queue` is not one of this module's queues). For tests.
bool queue_uses_calibrated_timestamps(ID3D12CommandQueue* queue);

// Creates a Vulkan-backed IDXGIFactory4 bound to `device`'s VulkanContext, instance, physical device, and queue.
// Its CreateSwapChainForHwnd registers swapchain backbuffers directly with the device's VulkanContext.
HRESULT create_dxgi_factory(ID3D12Device* device, REFIID riid, void** out);

// SDRWhiteLevel of DISPLAYCONFIG_SDR_WHITE_LEVEL (1000 = 80 nits) to nits.
float sdr_white_nits_from_level(unsigned long level);

// Puts back buffers of a device from create_vulkan_device* into a window (leonida::VulkanPresenter), for the
// presentation front end the game uses (DXGI or PresentX, still open). Holds a reference to the device.
class WindowPresenter {
public:
    WindowPresenter(IUnknown* device, std::unique_ptr<leonida::VulkanPresenter> presenter);
    ~WindowPresenter();
    WindowPresenter(const WindowPresenter&) = delete;
    WindowPresenter& operator=(const WindowPresenter&) = delete;

    // `back_buffer` is a colour texture in D3D12's PRESENT state. sync_interval 0 presents without vsync.
    // S_OK also covers a skipped present while the window is minimised.
    // `src_rect` (when not null) limits the blit to that part of the texture, which is stretched to the window.
    // `tear` (with sync_interval > 0): present in the tearing mode when the swapchain has one (tear_supported()), so a
    // frame that missed its vblank is shown at once; last_present_tore() tells whether that happened.
    HRESULT present(ID3D12Resource* back_buffer, UINT sync_interval, const D3D12_RECT* src_rect = nullptr,
                    bool tear = false);
    // A black frame, nothing read (PresentX with no planes).
    HRESULT present_blank(UINT sync_interval);
    // One or two planes composed (VulkanPresenter::present_planes); `planes` names the textures by ResourceId, so the
    // caller fills it with resource_id(). DXGI_ERROR_UNSUPPORTED for a source the compose pass cannot read.
    HRESULT present_planes(const leonida::PresentPlanes& planes, UINT sync_interval, bool tear = false);
    // The swapchain holds HDR10 (ST.2084) values; and a test hook (VulkanPresenter::force_hdr_encoding).
    bool hdr_active() const;
    void force_hdr_encoding(bool on);
    std::uint64_t composed_presents() const;  // presents that went through the compose pass
    // Nits of SDR white the user set in Windows for the output the window is on (DISPLAYCONFIG_SDR_WHITE_LEVEL), or 0 when
    // it cannot be read (no window, no display config, not Windows). Only meaningful while hdr_active().
    float display_sdr_white_nits() const;
    void set_window(HWND window) { window_ = window; }
    bool tear_supported() const;
    bool last_present_tore() const;
    std::uint64_t tear_presents() const;
    // Screenshot support for tests: every presented frame is also copied to host memory (call before presenting).
    bool enable_capture();
    // See VulkanPresenter::set_abort: polled while a present waits for a swapchain image.
    void set_abort(std::function<bool()> abort);
    // See VulkanPresenter::read_back: false after a present skipped while the window is minimised, too.
    bool read_back(std::vector<std::uint8_t>& pixels, VkExtent2D* extent, VkFormat* format);
    VkExtent2D extent() const;
    std::uint64_t presented() const;

private:
    IUnknown* device_;
    HWND window_ = nullptr;
    std::unique_ptr<leonida::VulkanPresenter> presenter_;
};

// Whether `device` (from create_vulkan_device*) was created with VK_EXT_swapchain_colorspace on its instance.
bool device_has_swapchain_colorspace(ID3D12Device* device);

// Null when `device` is not a Vulkan device created with presentation, or the window cannot be presented to.
std::unique_ptr<WindowPresenter> create_window_presenter(ID3D12Device* device, HWND window);

// Where each root parameter's arguments go in the 32 user-data dwords the translated shaders read (docs section
// 3.5). Built when a Vulkan device creates the root signature. A signature can be bound at either bind point, so it
// carries both layouts:
//   - graphics: rt::rootsig::pack() (from s0, two dwords per table; D3D-448 keeps it until per-stage user-data windows
//     exist, because a mesh shader (s16) and a pixel shader (s8) share one push-constant block);
//   - compute: rt::rootsig::pack_stage(Stage::Compute), the measured console layout (from s4, one dword per table).
// A layout that refuses the signature is marked unusable; binding the signature there invalidates the list.
struct RootArgumentLayout {
    struct Parameter {
        rt::rootsig::ParamType type;
        std::uint32_t user_offset;     // graphics: first dword; unused for a parameter with no dwords
        std::uint32_t constant_count;  // constants only
        std::uint32_t compute_offset = rt::rootsig::kNoSgpr;  // compute: first user SGPR
    };
    std::vector<Parameter> parameters;
    std::uint32_t flags = 0;  // D3D12_ROOT_SIGNATURE_FLAGS from the serialized blob
    bool graphics_ok = true;
    bool compute_ok = true;
};

// For the command list: the argument layout of a root signature from a Vulkan device, and the id it records for a
// compute pipeline state (0: not a compute PSO of a Vulkan device).
const RootArgumentLayout* root_signature_layout(ID3D12RootSignature* signature);
std::uintptr_t compute_pipeline_id(ID3D12PipelineState* pipeline);
std::uintptr_t graphics_pipeline_id(ID3D12PipelineState* pipeline);
// What each stage of a Vulkan device's PSO asked for (src/subgroup_size.h), in creation order; null for another PSO.
const std::vector<leonida::SubgroupRequest>* pipeline_subgroup_requests(ID3D12PipelineState* pipeline);

// Unwraps an Xbox-ABI heap proxy if needed and inspects one of this module's descriptor heaps; returns its owning
// device identity pointer, or nullptr when `heap` is null or belongs to another implementation.
IUnknown* descriptor_heap_info(ID3D12DescriptorHeap* heap, D3D12_DESCRIPTOR_HEAP_DESC& desc,
                               D3D12_GPU_VIRTUAL_ADDRESS& gpu_base);

// Reads the shader-visible descriptor heap GPU base addresses currently bound on `list` (0 when unbound).
// Returns false when `list` is not one of this module's command lists.
bool bound_descriptor_heap_bases(ID3D12GraphicsCommandList* list, D3D12_GPU_VIRTUAL_ADDRESS& cbv_srv_uav,
                                 D3D12_GPU_VIRTUAL_ADDRESS& sampler);

// The size of every descriptor slot (GetDescriptorHandleIncrementSize), for handle ranges.
inline constexpr UINT descriptor_size = 32;

// ExecuteCommandLists returns void; this is what the last call on the queue did.
leonida::ReplayResult last_execute_result(ID3D12CommandQueue* queue);

// True when the current thread or process token has SeIncreaseBasePriorityPrivilege enabled (or the test override
// forces it), which D3D12 requires for D3D12_COMMAND_QUEUE_PRIORITY_GLOBAL_REALTIME.
bool has_global_realtime_privilege();
// Test override for has_global_realtime_privilege(): -1 queries the token (default), 0 forces denied, 1 forces granted.
void set_global_realtime_privilege_override_for_test(int state);

// The front/back stencil references last recorded by ID3D12GraphicsCommandList8::OMSetFrontAndBackStencilRef on
// `list`. The values are kept for the future pipeline path, which is what would apply them. False when `list` is
// not one of this module's command lists.
bool front_back_stencil_ref(ID3D12GraphicsCommandList8* list, UINT& front, UINT& back);

// A copy of the commands `list` has recorded so far, for tests of what a bundle puts into its parent. False when
// `list` is not one of this module's command lists.
bool recorded_commands(ID3D12GraphicsCommandList* list, std::vector<leonida::Command>& commands);

inline leonida::ResourceId resource_id(const ID3D12Resource* resource) {
    return reinterpret_cast<leonida::ResourceId>(resource);
}

// A query heap's ID in the command-list IR and the VulkanContext: its pointer value, as for resources.
inline leonida::QueryHeapId query_heap_id(const ID3D12QueryHeap* heap) {
    return reinterpret_cast<leonida::QueryHeapId>(heap);
}

}  // namespace rt::d3d12
