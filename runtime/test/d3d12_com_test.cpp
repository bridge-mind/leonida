// End to end through the D3D12 COM surface: an ID3D12GraphicsCommandList7 records clears (whole and rectangles),
// legacy and enhanced barriers; an ID3D12CommandQueue executes it on Vulkan behind a queue Wait on a fence the CPU
// signals later; an ID3D12Fence1 event reports completion; the images are read back. No shaders or game data.
// Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation; any validation error fails the test.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_trace.h"
#include "../src/gfx10_descriptors.h"

#include <array>
#include <cstdint>
#include <cstring>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() {
        if (p) p->Release();
    }
    T* operator->() const { return p; }
    void** out() { return reinterpret_cast<void**>(&p); }
};

}  // namespace

static_assert(std::is_move_constructible_v<leonida::Draw>);
static_assert(std::is_move_assignable_v<leonida::Draw>);
static_assert(sizeof(leonida::DrawViews::render_targets) == sizeof(leonida::TargetView) * 8);
static_assert(sizeof(leonida::DrawViews::vertex_buffers) == sizeof(leonida::VertexBufferView) * 32);
// The views stay out of the variant: a recorded command is sized by its largest member, Draw included.
static_assert(sizeof(leonida::Draw) < sizeof(leonida::DrawViews));

int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--trace-d3d12") == 0) rt::d3d12_trace::set_enabled(true);
    else if (argc != 1) return 2;
    using namespace leonida;
    vktest::Device d;
    const auto device_status = d.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) return 77;   // no Vulkan GPU: ctest skips
    if (device_status != vktest::DeviceStatus::ready) return 1;
    VulkanContext context(d.device_proc, d.device, true);
    EXPECT(context.loaded());

    constexpr std::uint32_t size = 4, texels = size * size;
    const VkImage rt = d.image(VK_FORMAT_R8G8B8A8_UNORM, size, size, 1, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    const VkImage ds = d.image(VK_FORMAT_D32_SFLOAT_S8_UINT, size, size, 1, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
    void* mapped = nullptr;
    constexpr VkDeviceSize depth_offset = texels * 4, stencil_offset = depth_offset + texels * 4;
    const VkBuffer buffer = d.buffer(stencil_offset + texels, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &mapped);
    EXPECT(rt && ds && buffer && d.depth_format_usable(VK_FORMAT_D32_SFLOAT_S8_UINT));
    EXPECT(d.run([&](VkCommandBuffer cb) {
        d.initial_layout(cb, rt, VK_IMAGE_ASPECT_COLOR_BIT, 1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        d.initial_layout(cb, ds, depth_stencil_aspects, 1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        return true;
    }));

    // Stand-ins for the device: resources are identified by their ID3D12Resource pointer, which the layer never
    // dereferences, and RTV/DSV handles point at TargetViews.
    auto* rt_resource = reinterpret_cast<ID3D12Resource*>(std::uintptr_t{0x1000});
    auto* ds_resource = reinterpret_cast<ID3D12Resource*>(std::uintptr_t{0x2000});
    ResourceBinding rt_binding{};
    rt_binding.image = rt;
    rt_binding.format = VK_FORMAT_R8G8B8A8_UNORM;
    rt_binding.width = rt_binding.height = size;
    ResourceBinding ds_binding = rt_binding;
    ds_binding.image = ds;
    ds_binding.format = VK_FORMAT_D32_SFLOAT_S8_UINT;
    ds_binding.aspects = depth_stencil_aspects;
    context.add_resource(rt::d3d12::resource_id(rt_resource), rt_binding, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    context.add_resource(rt::d3d12::resource_id(ds_resource), ds_binding,
                         VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    const TargetView rtv{rt::d3d12::resource_id(rt_resource), 0, 0, 1};
    const TargetView dsv{rt::d3d12::resource_id(ds_resource), 0, 0, 1};
    const D3D12_CPU_DESCRIPTOR_HANDLE rtv_handle{reinterpret_cast<SIZE_T>(&rtv)};
    const D3D12_CPU_DESCRIPTOR_HANDLE dsv_handle{reinterpret_cast<SIZE_T>(&dsv)};

    {
        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        Ref<ID3D12CommandQueue> queue;
        Ref<ID3D12CommandAllocator> allocator;
        Ref<ID3D12GraphicsCommandList7> list;
        Ref<ID3D12Fence1> done, gate;
        EXPECT(SUCCEEDED(rt::d3d12::create_command_queue(context, d.queue, d.family, queue_desc,
                                                          IID_PPV_ARGS(&queue.p))));
        EXPECT(SUCCEEDED(rt::d3d12::create_command_allocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
        EXPECT(SUCCEEDED(rt::d3d12::create_command_list(D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p,
                                                         IID_PPV_ARGS(&list.p))));
        EXPECT(SUCCEEDED(rt::d3d12::create_fence(context, 0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&done.p))));
        EXPECT(SUCCEEDED(rt::d3d12::create_fence(context, 0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&gate.p))));

        // COM identity: the public chain answers, an unknown IID is refused (and logged).
        {
            Ref<ID3D12GraphicsCommandList> base;
            Ref<ID3D12Device> device;
            EXPECT(SUCCEEDED(list->QueryInterface(IID_PPV_ARGS(&base.p))) && base.p == list.p);
            EXPECT(list->QueryInterface(IID_PPV_ARGS(&device.p)) == E_NOINTERFACE && !device.p);
        }
        EXPECT(queue->GetDesc().Type == D3D12_COMMAND_LIST_TYPE_DIRECT);
        EXPECT(list->GetType() == D3D12_COMMAND_LIST_TYPE_DIRECT);

        // ExecuteBundle: a closed bundle's state calls play on the parent, so what the bundle set stays there.
        Ref<ID3D12GraphicsCommandList8> list8;
        EXPECT(SUCCEEDED(list->QueryInterface(IID_PPV_ARGS(&list8.p))));
        list8->OMSetFrontAndBackStencilRef(1, 2);
        Ref<ID3D12CommandAllocator> bundle_allocator;
        Ref<ID3D12GraphicsCommandList8> bundle;
        EXPECT(SUCCEEDED(rt::d3d12::create_command_allocator(D3D12_COMMAND_LIST_TYPE_BUNDLE,
                                                              IID_PPV_ARGS(&bundle_allocator.p))));
        EXPECT(SUCCEEDED(rt::d3d12::create_command_list(D3D12_COMMAND_LIST_TYPE_BUNDLE, bundle_allocator.p,
                                                         IID_PPV_ARGS(&bundle.p))));
        EXPECT(bundle->GetType() == D3D12_COMMAND_LIST_TYPE_BUNDLE);

        // A command list and its allocator must have the same D3D12 type.
        Ref<ID3D12CommandAllocator> wrong_allocator;
        Ref<ID3D12GraphicsCommandList8> wrong_bundle;
        EXPECT(SUCCEEDED(rt::d3d12::create_command_allocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                              IID_PPV_ARGS(&wrong_allocator.p))));
        EXPECT(rt::d3d12::create_command_list(D3D12_COMMAND_LIST_TYPE_BUNDLE, wrong_allocator.p,
                                               IID_PPV_ARGS(&wrong_bundle.p)) == E_INVALIDARG);
        EXPECT(!wrong_bundle.p);
        bundle->OMSetFrontAndBackStencilRef(7, 9);
        EXPECT(bundle->Close() == S_OK);
        list8->ExecuteBundle(bundle.p);
        UINT front = 0, back = 0;
        EXPECT(rt::d3d12::front_back_stencil_ref(list8.p, front, back) && front == 7 && back == 9);

        // An untouched bundle inherits the parent's current sticky state; its default values must not overwrite it.
        Ref<ID3D12CommandAllocator> untouched_allocator;
        Ref<ID3D12GraphicsCommandList8> untouched_bundle;
        EXPECT(SUCCEEDED(rt::d3d12::create_command_allocator(D3D12_COMMAND_LIST_TYPE_BUNDLE,
                                                              IID_PPV_ARGS(&untouched_allocator.p))));
        EXPECT(SUCCEEDED(rt::d3d12::create_command_list(D3D12_COMMAND_LIST_TYPE_BUNDLE, untouched_allocator.p,
                                                         IID_PPV_ARGS(&untouched_bundle.p))));
        EXPECT(untouched_bundle->Close() == S_OK);
        list8->ExecuteBundle(untouched_bundle.p);
        EXPECT(rt::d3d12::front_back_stencil_ref(list8.p, front, back) && front == 7 && back == 9);
        list8->ExecuteBundle(bundle.p);  // the same closed bundle can be executed repeatedly

        ID3D12CommandList* bundle_submission[] = {bundle.p};
        queue->ExecuteCommandLists(1, bundle_submission);  // bundles are executed only inside direct lists
        EXPECT(rt::d3d12::last_execute_result(queue.p) == ReplayResult::unsupported);

        // Runtime validation failures are deferred to Close, matching D3D12.
        Ref<ID3D12CommandAllocator> bad_parent_allocator, direct_callee_allocator;
        Ref<ID3D12GraphicsCommandList8> bad_parent, direct_callee;
        EXPECT(SUCCEEDED(rt::d3d12::create_command_allocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                              IID_PPV_ARGS(&bad_parent_allocator.p))));
        EXPECT(SUCCEEDED(rt::d3d12::create_command_list(D3D12_COMMAND_LIST_TYPE_DIRECT, bad_parent_allocator.p,
                                                         IID_PPV_ARGS(&bad_parent.p))));
        EXPECT(SUCCEEDED(rt::d3d12::create_command_allocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                              IID_PPV_ARGS(&direct_callee_allocator.p))));
        EXPECT(SUCCEEDED(rt::d3d12::create_command_list(D3D12_COMMAND_LIST_TYPE_DIRECT, direct_callee_allocator.p,
                                                         IID_PPV_ARGS(&direct_callee.p))));
        EXPECT(direct_callee->Close() == S_OK);
        bad_parent->ExecuteBundle(direct_callee.p);  // callee must be a bundle
        EXPECT(bad_parent->Close() == E_INVALIDARG);
        EXPECT(direct_callee->Reset(bundle_allocator.p, nullptr) == E_INVALIDARG);
        EXPECT(SUCCEEDED(direct_callee->Reset(direct_callee_allocator.p, nullptr)));
        EXPECT(direct_callee->Close() == S_OK);

        Ref<ID3D12CommandAllocator> open_parent_allocator, open_bundle_allocator;
        Ref<ID3D12GraphicsCommandList8> open_parent, open_bundle;
        EXPECT(SUCCEEDED(rt::d3d12::create_command_allocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                              IID_PPV_ARGS(&open_parent_allocator.p))));
        EXPECT(SUCCEEDED(rt::d3d12::create_command_list(D3D12_COMMAND_LIST_TYPE_DIRECT, open_parent_allocator.p,
                                                         IID_PPV_ARGS(&open_parent.p))));
        EXPECT(SUCCEEDED(rt::d3d12::create_command_allocator(D3D12_COMMAND_LIST_TYPE_BUNDLE,
                                                              IID_PPV_ARGS(&open_bundle_allocator.p))));
        EXPECT(SUCCEEDED(rt::d3d12::create_command_list(D3D12_COMMAND_LIST_TYPE_BUNDLE, open_bundle_allocator.p,
                                                         IID_PPV_ARGS(&open_bundle.p))));
        open_parent->ExecuteBundle(open_bundle.p);  // callee must already be closed
        EXPECT(open_parent->Close() == E_INVALIDARG);
        EXPECT(open_bundle->Close() == S_OK);

        Ref<ID3D12CommandAllocator> bundle_parent_allocator;
        Ref<ID3D12GraphicsCommandList8> bundle_parent;
        EXPECT(SUCCEEDED(rt::d3d12::create_command_allocator(D3D12_COMMAND_LIST_TYPE_BUNDLE,
                                                              IID_PPV_ARGS(&bundle_parent_allocator.p))));
        EXPECT(SUCCEEDED(rt::d3d12::create_command_list(D3D12_COMMAND_LIST_TYPE_BUNDLE, bundle_parent_allocator.p,
                                                         IID_PPV_ARGS(&bundle_parent.p))));
        bundle_parent->ExecuteBundle(bundle.p);  // caller must be a direct list
        EXPECT(bundle_parent->Close() == E_INVALIDARG);

        // Clear calls are forbidden in bundles; a failed bundle also invalidates its caller.
        Ref<ID3D12CommandAllocator> invalid_bundle_allocator, invalid_caller_allocator;
        Ref<ID3D12GraphicsCommandList8> invalid_bundle, invalid_caller;
        EXPECT(SUCCEEDED(rt::d3d12::create_command_allocator(D3D12_COMMAND_LIST_TYPE_BUNDLE,
                                                              IID_PPV_ARGS(&invalid_bundle_allocator.p))));
        EXPECT(SUCCEEDED(rt::d3d12::create_command_list(D3D12_COMMAND_LIST_TYPE_BUNDLE, invalid_bundle_allocator.p,
                                                         IID_PPV_ARGS(&invalid_bundle.p))));
        const FLOAT forbidden_clear[4] = {0, 0, 0, 1};
        invalid_bundle->ClearRenderTargetView(rtv_handle, forbidden_clear, 0, nullptr);
        EXPECT(invalid_bundle->Close() == E_INVALIDARG);
        EXPECT(SUCCEEDED(rt::d3d12::create_command_allocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                              IID_PPV_ARGS(&invalid_caller_allocator.p))));
        EXPECT(SUCCEEDED(rt::d3d12::create_command_list(D3D12_COMMAND_LIST_TYPE_DIRECT, invalid_caller_allocator.p,
                                                         IID_PPV_ARGS(&invalid_caller.p))));
        invalid_caller->ExecuteBundle(invalid_bundle.p);
        EXPECT(invalid_caller->Close() == E_INVALIDARG);
        constexpr GUID kTestData{0x0badf00d, 0x1, 0x2, {3, 4, 5, 6, 7, 8, 9, 10}};
        const std::uint32_t tag = 0x1234abcd;
        std::uint32_t read = 0;
        UINT read_size = sizeof(read);
        EXPECT(SUCCEEDED(queue->SetPrivateData(kTestData, sizeof(tag), &tag)));
        EXPECT(SUCCEEDED(queue->GetPrivateData(kTestData, &read_size, &read)) && read == tag);

        // An open list is rejected by ExecuteCommandLists and nothing is submitted.
        ID3D12CommandList* lists[] = {list.p};
        queue->ExecuteCommandLists(1, lists);
        EXPECT(rt::d3d12::last_execute_result(queue.p) == ReplayResult::list_open);

        const FLOAT red[4] = {1, 0, 0, 1}, blue[4] = {0, 0, 1, 1};
        const D3D12_RECT square{1, 1, 3, 3};
        list->ClearRenderTargetView(rtv_handle, red, 0, nullptr);
        list->ClearRenderTargetView(rtv_handle, blue, 1, &square);
        list->ClearDepthStencilView(dsv_handle, D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 0.5f, 0x33, 0,
                                    nullptr);
        D3D12_RESOURCE_BARRIER transition{};
        transition.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        transition.Transition.pResource = rt_resource;
        transition.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        transition.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
        transition.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
        list->ResourceBarrier(1, &transition);
        D3D12_TEXTURE_BARRIER texture{};
        texture.SyncBefore = D3D12_BARRIER_SYNC_DEPTH_STENCIL;
        texture.SyncAfter = D3D12_BARRIER_SYNC_COPY;
        texture.AccessBefore = D3D12_BARRIER_ACCESS_DEPTH_STENCIL_WRITE;
        texture.AccessAfter = D3D12_BARRIER_ACCESS_COPY_SOURCE;
        texture.LayoutBefore = D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_WRITE;
        texture.LayoutAfter = D3D12_BARRIER_LAYOUT_COPY_SOURCE;
        texture.pResource = ds_resource;
        texture.Subresources = {0xffffffff, 0, 0, 0, 0, 0};
        D3D12_BARRIER_GROUP group{};
        group.Type = D3D12_BARRIER_TYPE_TEXTURE;
        group.NumBarriers = 1;
        group.pTextureBarriers = &texture;

        // Each currently implemented bundle-forbidden method independently invalidates the bundle.
        const FLOAT bundle_clear_color[4] = {0, 1, 0, 1};
        auto invalid_bundle_call = [&](auto record) {
            Ref<ID3D12CommandAllocator> test_allocator;
            Ref<ID3D12GraphicsCommandList8> test_bundle;
            if (FAILED(rt::d3d12::create_command_allocator(D3D12_COMMAND_LIST_TYPE_BUNDLE,
                                                           IID_PPV_ARGS(&test_allocator.p))))
                return false;
            if (FAILED(rt::d3d12::create_command_list(D3D12_COMMAND_LIST_TYPE_BUNDLE, test_allocator.p,
                                                      IID_PPV_ARGS(&test_bundle.p))))
                return false;
            record(test_bundle.p);
            return test_bundle->Close() == E_INVALIDARG;
        };
        EXPECT(invalid_bundle_call([&](auto* b) { b->ResourceBarrier(1, &transition); }));
        EXPECT(invalid_bundle_call([&](auto* b) { b->Barrier(1, &group); }));
        EXPECT(invalid_bundle_call([&](auto* b) { b->CopyBufferRegion(rt_resource, 0, ds_resource, 0, 16); }));
        D3D12_TEXTURE_COPY_LOCATION copy_dst{};
        copy_dst.pResource = rt_resource;
        copy_dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        D3D12_TEXTURE_COPY_LOCATION copy_src{};
        copy_src.pResource = ds_resource;
        copy_src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        EXPECT(invalid_bundle_call([&](auto* b) { b->CopyTextureRegion(&copy_dst, 0, 0, 0, &copy_src, nullptr); }));
        EXPECT(invalid_bundle_call([&](auto* b) { b->CopyResource(rt_resource, ds_resource); }));
        EXPECT(invalid_bundle_call([&](auto* b) { b->ResolveSubresource(rt_resource, 0, ds_resource, 0,
                                                                          DXGI_FORMAT_R8G8B8A8_UNORM); }));
        EXPECT(invalid_bundle_call([&](auto* b) { b->ResolveSubresourceRegion(rt_resource, 0, 0, 0, ds_resource,
                                                                               0, nullptr,
                                                                               DXGI_FORMAT_R8G8B8A8_UNORM,
                                                                               D3D12_RESOLVE_MODE_AVERAGE); }));
        EXPECT(invalid_bundle_call([&](auto* b) { b->ClearRenderTargetView(rtv_handle, bundle_clear_color, 0, nullptr); }));
        EXPECT(invalid_bundle_call([&](auto* b) { b->ClearDepthStencilView(dsv_handle, D3D12_CLEAR_FLAG_DEPTH,
                                                                            0.5f, 0, 0, nullptr); }));
        // A valid texture UAV descriptor, which a direct list would accept: only the bundle refuses the clear.
        const UINT uav_uint[4] = {};
        const FLOAT uav_float[4] = {};
        leonida::gfx10::Image uav_image{};
        uav_image.format = leonida::gfx10::format_from_vulkan(VK_FORMAT_R8G8B8A8_UNORM);
        uav_image.width = uav_image.height = 4;
        const auto uav_words = leonida::gfx10::encode(uav_image);
        const D3D12_CPU_DESCRIPTOR_HANDLE uav_view{reinterpret_cast<SIZE_T>(uav_words.data())};
        EXPECT(invalid_bundle_call([&](auto* b) {
            b->ClearUnorderedAccessViewUint({}, uav_view, rt_resource, uav_uint, 0, nullptr); }));
        EXPECT(invalid_bundle_call([&](auto* b) {
            b->ClearUnorderedAccessViewFloat({}, uav_view, rt_resource, uav_float, 0, nullptr); }));
        EXPECT(invalid_bundle_call([&](auto* b) { b->SetPredication(nullptr, 0, D3D12_PREDICATION_OP_EQUAL_ZERO); }));
        // Measured on the PC runtime (default adapter and WARP, d3d12_bundle_forbidden_test): these fail a bundle's Close.
        const D3D12_VIEWPORT bundle_viewport{0, 0, 4, 4, 0, 1};
        const D3D12_RECT bundle_scissor{0, 0, 4, 4};
        EXPECT(invalid_bundle_call([&](auto* b) { b->RSSetViewports(1, &bundle_viewport); }));
        EXPECT(invalid_bundle_call([&](auto* b) { b->RSSetScissorRects(1, &bundle_scissor); }));
        EXPECT(invalid_bundle_call([&](auto* b) { b->OMSetRenderTargets(1, &rtv_handle, FALSE, nullptr); }));
        EXPECT(invalid_bundle_call([&](auto* b) { b->ClearState(nullptr); }));
        // D3D-26: DiscardResource and the atomic copies are not allowed in a bundle; WriteBufferImmediate is
        // ("Recording command lists and bundles", the bundle restrictions and the API restrictions table).
        EXPECT(invalid_bundle_call([&](auto* b) { b->DiscardResource(rt_resource, nullptr); }));
        ID3D12Resource* const bundle_dependents[] = {ds_resource};
        const D3D12_SUBRESOURCE_RANGE_UINT64 bundle_ranges[] = {{0, {0, 8}}};
        EXPECT(invalid_bundle_call([&](auto* b) {
            b->AtomicCopyBufferUINT(rt_resource, 0, ds_resource, 0, 1, bundle_dependents, bundle_ranges); }));
        EXPECT(invalid_bundle_call([&](auto* b) {
            b->AtomicCopyBufferUINT64(rt_resource, 0, ds_resource, 0, 1, bundle_dependents, bundle_ranges); }));
        EXPECT(!invalid_bundle_call([&](auto* b) {
            const D3D12_WRITEBUFFERIMMEDIATE_PARAMETER write{0x1000, 1};
            b->WriteBufferImmediate(1, &write, nullptr); }));

        list->Barrier(1, &group);
        // Not implemented: reported once by its generated stub, recording continues.
        list->SetProtectedResourceSession(nullptr);
        EXPECT(list->Close() == S_OK);
        EXPECT(list->Close() == E_FAIL);  // already closed

        // The queue waits for `gate`, which only the CPU signals, so the work cannot have run before that.
        EXPECT(SUCCEEDED(queue->Wait(gate.p, 1)));
        queue->ExecuteCommandLists(1, lists);
        EXPECT(rt::d3d12::last_execute_result(queue.p) == ReplayResult::success);
        EXPECT(SUCCEEDED(queue->Signal(done.p, 1)));
        Sleep(50);
        EXPECT(done->GetCompletedValue() == 0);
        const HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        EXPECT(event);
        EXPECT(SUCCEEDED(done->SetEventOnCompletion(1, event)));
        EXPECT(WaitForSingleObject(event, 0) == WAIT_TIMEOUT);
        EXPECT(SUCCEEDED(gate->Signal(1)));
        EXPECT(WaitForSingleObject(event, 5000) == WAIT_OBJECT_0);
        CloseHandle(event);
        EXPECT(done->GetCompletedValue() == 1);
        EXPECT(SUCCEEDED(done->Signal(1)));  // not above the current value: reported and dropped
        EXPECT(done->GetCompletedValue() == 1);

        EXPECT(context.layout(rt::d3d12::resource_id(rt_resource), 0) == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        EXPECT(context.layout(rt::d3d12::resource_id(ds_resource), 0) == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        EXPECT(context.layout(rt::d3d12::resource_id(ds_resource), 1) == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

        // A depth outside [0, 1] makes Close report the invalid call, as D3D12 does.
        EXPECT(SUCCEEDED(allocator->Reset()));
        EXPECT(SUCCEEDED(list->Reset(allocator.p, nullptr)));
        list->ClearDepthStencilView(dsv_handle, D3D12_CLEAR_FLAG_DEPTH, 2.0f, 0, 0, nullptr);
        EXPECT(list->Close() == E_INVALIDARG);

        // State-setting calls on a closed direct list or bundle are ignored and do not mutate recorded state.
        EXPECT(SUCCEEDED(allocator->Reset()));
        EXPECT(SUCCEEDED(list->Reset(allocator.p, nullptr)));
        list->OMSetStencilRef(5);
        EXPECT(list->Close() == S_OK);
        EXPECT(rt::d3d12::front_back_stencil_ref(list8.p, front, back) && front == 5 && back == 5);
        const FLOAT post_close_factor[4] = {0.1f, 0.2f, 0.3f, 0.4f};
        list->OMSetStencilRef(99);
        list->OMSetBlendFactor(post_close_factor);
        list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        list->IASetIndexBuffer(nullptr);
        list->SetPipelineState(nullptr);
        list->ClearState(nullptr);
        list->ExecuteBundle(bundle.p);
        EXPECT(rt::d3d12::front_back_stencil_ref(list8.p, front, back) && front == 5 && back == 5);
        EXPECT(list->Close() == E_FAIL);

        // Post-close state calls on a closed bundle do not append calls or invalidate the bundle.
        bundle->OMSetStencilRef(99);
        bundle->OMSetBlendFactor(post_close_factor);
        bundle->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_LINELIST);
        bundle->IASetIndexBuffer(nullptr);
        bundle->SetPipelineState(nullptr);
        bundle->ClearState(nullptr);
        bundle->ExecuteBundle(untouched_bundle.p);
        EXPECT(SUCCEEDED(allocator->Reset()));
        EXPECT(SUCCEEDED(list->Reset(allocator.p, nullptr)));
        list->ExecuteBundle(bundle.p);
        EXPECT(rt::d3d12::front_back_stencil_ref(list8.p, front, back) && front == 7 && back == 9);
        EXPECT(list->Close() == S_OK);
    }

    EXPECT(d.run([&](VkCommandBuffer cb) {
        d.read_back(cb, rt, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT, size, size, 1, buffer, 0);
        d.read_back(cb, ds, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT, size, size, 1, buffer,
                    depth_offset);
        d.read_back(cb, ds, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_ASPECT_STENCIL_BIT, size, size, 1, buffer,
                    stencil_offset);
        return true;
    }));
    const auto* data = static_cast<const std::uint8_t*>(mapped);
    unsigned mismatches = 0;
    for (std::uint32_t row = 0; row < size; ++row) {
        for (std::uint32_t col = 0; col < size; ++col) {
            const std::uint32_t i = row * size + col;
            const bool inside = col >= 1 && col < 3 && row >= 1 && row < 3;
            const std::array<std::uint8_t, 4> want = inside ? std::array<std::uint8_t, 4>{0, 0, 255, 255}
                                                            : std::array<std::uint8_t, 4>{255, 0, 0, 255};
            float depth;
            std::memcpy(&depth, data + depth_offset + i * 4, 4);
            if (!std::equal(want.begin(), want.end(), data + i * 4) || depth != 0.5f || data[stencil_offset + i] != 0x33) {
                if (mismatches++ < 4)
                    std::fprintf(stderr, "texel %u: colour %u,%u,%u,%u depth %f stencil 0x%02x\n", i,
                                 unsigned(data[i * 4]), unsigned(data[i * 4 + 1]), unsigned(data[i * 4 + 2]),
                                 unsigned(data[i * 4 + 3]), depth, unsigned(data[stencil_offset + i]));
            }
        }
    }
    EXPECT(!mismatches);
    if (d.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
        return 1;
    }
    std::printf("COM list/queue/fence: %u colour, depth and stencil texels matched; queue wait held the GPU\n", texels);
    return 0;
}
