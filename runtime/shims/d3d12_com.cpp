#include "d3d12_com.h"

#include "d3d12_stubs.h"  // generated
#include "d3d12_com_object.h"
#include "d3d12_fence_waiter.h"
#include "d3d12_trace.h"
#include "d3d12_xs_device.h"
#include "../src/gfx10_descriptors.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <condition_variable>
#include <functional>
#include <memory>
#include <optional>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace rt::d3d12 {

struct TileGrid {
    std::uint64_t columns = 0;
    std::uint64_t rows = 0;
};

static TileGrid tile_grid(ID3D12Resource* resource) {
    if (!resource) return {};
    const auto desc = resource->GetDesc();
    if (desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER && desc.Width)
        return {1 + (desc.Width - 1) / leonida::tile_size_in_bytes, 1};
    if (!is_sparse_image(resource)) return {};
    constexpr auto shape = leonida::element_tile_shape(false, 4);
    return {1 + (desc.Width - 1) / shape.width, 1 + (desc.Height - 1) / shape.height};
}

static bool append_tile_regions(const D3D12_TILED_RESOURCE_COORDINATE* start, const D3D12_TILE_REGION_SIZE* size,
                                TileGrid grid, std::vector<leonida::TileRegion>& regions) {
    const auto x = start ? start->X : 0;
    const auto y = start ? start->Y : 0;
    if ((start && (start->Z || start->Subresource)) || x >= grid.columns || y >= grid.rows) return false;
    const auto first = y * grid.columns + x;
    if (size && size->UseBox) {
        if (size->Depth != 1 || !size->Width || !size->Height ||
            size->Width > grid.columns - x || size->Height > grid.rows - y) return false;
        for (UINT row = 0; row < size->Height; ++row)
            regions.push_back({first + row * grid.columns, size->Width});
        return true;
    }
    const auto tiles = grid.columns * grid.rows;
    const auto count = size ? size->NumTiles : (start ? 1 : tiles);
    if (!count || count > tiles - first) return false;
    regions.push_back({first, count});
    return true;
}

void stub(const char* method) {
    // Generated API stubs pass a bare method name; explanatory diagnostics are not separate calls.
    if (!std::strchr(method, '(')) d3d12_trace::record(method, "unsupported");
    static std::mutex mutex;
    static std::set<std::string> seen;
    std::lock_guard lock(mutex);
    if (seen.insert(method).second) std::fprintf(stderr, "[rt] d3d12: unimplemented %s\n", method);
}

namespace {

// Reports a deliberate simplification once (unlike stub, which reports a method not implemented).
void note_once(const char* what) {
    static std::mutex mutex;
    static std::set<std::string> seen;
    std::lock_guard lock(mutex);
    if (seen.insert(what).second) std::fprintf(stderr, "[rt] d3d12: %s\n", what);
}

const char* name(leonida::ReplayResult result) {
    switch (result) {
    case leonida::ReplayResult::success: return "success";
    case leonida::ReplayResult::list_open: return "a list is not closed";
    case leonida::ReplayResult::unresolved_resource: return "unknown resource";
    case leonida::ReplayResult::invalid_subresource: return "invalid subresource";
    case leonida::ReplayResult::invalid_clear: return "invalid clear";
    case leonida::ReplayResult::invalid_copy: return "invalid copy or resolve";
    case leonida::ReplayResult::invalid_query: return "invalid query";
    case leonida::ReplayResult::invalid_indirect_arguments: return "invalid indirect argument or count buffer range";
    case leonida::ReplayResult::invalid_draw: return "draw kind does not match the pipeline (mesh or vertex)";
    case leonida::ReplayResult::unsupported: return "unsupported";
    }
    return "?";
}

// The label of a SetMarker/BeginEvent call, as UTF-8. Metadata says what the data is (WinPixEventRuntime's
// pix3_win.h: PIX_EVENT_UNICODE_VERSION 0, a UTF-16 string; PIX_EVENT_ANSI_VERSION 1, an ANSI string;
// PIX_EVENT_PIX3BLOB_VERSION 2, PIX's own encoded event). Strings end at their terminator or at `size` bytes. The
// PIX3 blob is not decoded (its console form is unmeasured, docs/d3d12_vulkan.md section 4.10); it gets a fixed
// label, so the event still nests.
std::string event_label(UINT metadata, const void* data, UINT size) {
    if (!data || !size) return {};
    std::wstring wide;
    if (metadata == 0) {
        const auto* text = static_cast<const wchar_t*>(data);
        const std::size_t limit = size / sizeof(wchar_t);
        std::size_t n = 0;
        while (n < limit && text[n]) ++n;
        wide.assign(text, n);
    } else if (metadata == 1) {
        const auto* text = static_cast<const char*>(data);
        std::size_t n = 0;
        while (n < size && text[n]) ++n;
        if (!n) return {};
        const int count = MultiByteToWideChar(CP_ACP, 0, text, int(n), nullptr, 0);
        wide.resize(std::size_t(std::max(count, 0)));
        if (count > 0) MultiByteToWideChar(CP_ACP, 0, text, int(n), wide.data(), count);
    } else {
        if (metadata != 2) stub("SetMarker/BeginEvent with unknown metadata");
        return "PIX event";
    }
    if (wide.empty()) return {};
    const int count = WideCharToMultiByte(CP_UTF8, 0, wide.data(), int(wide.size()), nullptr, 0, nullptr, nullptr);
    std::string out(std::size_t(std::max(count, 0)), '\0');
    if (count > 0) WideCharToMultiByte(CP_UTF8, 0, wide.data(), int(wide.size()), out.data(), count, nullptr, nullptr);
    return out;
}

class CommandSignature final
    : public Child<CommandSignature, ID3D12CommandSignatureStubs, ID3D12Object, ID3D12DeviceChild, ID3D12Pageable,
                   ID3D12CommandSignature> {
public:
    // {9a4e1c2b-55d8-4f7a-b012-3c8e7a1f6d90}
    static constexpr GUID kTag{0x9a4e1c2b, 0x55d8, 0x4f7a, {0xb0, 0x12, 0x3c, 0x8e, 0x7a, 0x1f, 0x6d, 0x90}};

    CommandSignature(leonida::IndirectKind kind, UINT byte_stride) : kind_(kind), stride_(byte_stride) {}

    leonida::IndirectKind kind() const { return kind_; }
    UINT stride() const { return stride_; }
    static CommandSignature* from(IUnknown* object) { return Child::from<CommandSignature>(object); }

private:
    leonida::IndirectKind kind_;
    UINT stride_;
};

class CommandAllocator final
    : public Child<CommandAllocator, ID3D12CommandAllocatorStubs, ID3D12Object, ID3D12DeviceChild, ID3D12Pageable,
                    ID3D12CommandAllocator> {
public:
    // {7d0e6b51-34a2-4f0c-9a61-0c5b2e8f4a10}, public in d3d12_xs_device.h: the device proxy's slot 9 wraps an allocator
    // in an Xbox-ABI allocator proxy only when it answers this tag, because from() below resolves such a proxy.
    static constexpr GUID kTag = d3d12_xs::kLeonidaCommandAllocatorTag;
    explicit CommandAllocator(D3D12_COMMAND_LIST_TYPE type) : type_(type) {}
    // Lists own their recorded commands, so there is nothing to free. As on the PC runtime (D3D-51, measured by
    // d3d12_vulkan_device_test): E_FAIL while a list is recording into this allocator, and S_OK while its lists are
    // still executing on the GPU (keeping that safe is the application's job on D3D12 too).
    HRESULT STDMETHODCALLTYPE Reset() override { return recording_ ? E_FAIL : S_OK; }
    D3D12_COMMAND_LIST_TYPE type() const { return type_; }
    // The one list that is recording into this allocator, or null. D3D12 allows one at a time.
    const void* recording() const { return recording_; }
    void set_recording(const void* list) { recording_ = list; }
    // An allocator proxy (the title's allocator from device slot 9) stands for the allocator behind it, so every
    // list method that takes an allocator (CreateCommandList, Reset) resolves it here.
    static CommandAllocator* from(IUnknown* object) {
        if (ID3D12CommandAllocator* real = d3d12_xs::allocator_proxy_real_allocator(object)) object = real;
        return Child::from<CommandAllocator>(object);
    }

private:
    D3D12_COMMAND_LIST_TYPE type_;
    const void* recording_ = nullptr;
};

class GraphicsCommandList final
    : public Child<GraphicsCommandList, ID3D12GraphicsCommandList8Stubs, ID3D12Object, ID3D12DeviceChild,
                    ID3D12CommandList, ID3D12GraphicsCommandList, ID3D12GraphicsCommandList1,
                    ID3D12GraphicsCommandList2, ID3D12GraphicsCommandList3, ID3D12GraphicsCommandList4,
                    ID3D12GraphicsCommandList5, ID3D12GraphicsCommandList6, ID3D12GraphicsCommandList7,
                    ID3D12GraphicsCommandList8> {
public:
    // {2f8a7c33-91d4-4b6e-8e05-6d3f1a9b2c47}
    static constexpr GUID kTag{0x2f8a7c33, 0x91d4, 0x4b6e, {0x8e, 0x05, 0x6d, 0x3f, 0x1a, 0x9b, 0x2c, 0x47}};
    explicit GraphicsCommandList(D3D12_COMMAND_LIST_TYPE type) : type_(type) {}
    ~GraphicsCommandList() override {
        stop_recording();
        release_held();
    }

    // While open, the list holds a reference to the allocator it records into and marks it busy.
    void record_into(CommandAllocator* allocator) {
        allocator->AddRef();
        allocator->set_recording(this);
        allocator_ = allocator;
    }

    const leonida::CommandList& commands() const { return list_; }
    // The pipeline states and root signatures the recording names (it holds a reference to each until Reset).
    const std::vector<IUnknown*>& pinned() const { return held_; }
    UINT front_stencil_ref() const { return front_stencil_ref_; }
    UINT back_stencil_ref() const { return back_stencil_ref_; }
    D3D12_GPU_VIRTUAL_ADDRESS cbv_srv_uav_base() const { return cbv_srv_uav_base_; }
    D3D12_GPU_VIRTUAL_ADDRESS sampler_base() const { return sampler_base_; }

    D3D12_COMMAND_LIST_TYPE STDMETHODCALLTYPE GetType() override {
        d3d12_trace::record("ID3D12CommandList::GetType");
        return type_;
    }
    D3D12_COMMAND_LIST_TYPE type() const { return type_; }
    bool invalid() const { return invalid_; }

    // E_INVALIDARG when a call during recording was invalid, or a query was begun and not ended, as D3D12's Close
    // reports it.
    HRESULT STDMETHODCALLTYPE Close() override {
        constexpr auto method = "ID3D12GraphicsCommandList::Close";
        if (list_.queries_open()) invalid_ = true;
        if (!list_.close()) return d3d12_trace::result(method, E_FAIL);
        stop_recording();
        return d3d12_trace::result(method, invalid_ ? E_INVALIDARG : S_OK);
    }

    HRESULT STDMETHODCALLTYPE Reset(ID3D12CommandAllocator* allocator, ID3D12PipelineState* state) override {
        constexpr auto method = "ID3D12GraphicsCommandList::Reset";
        CommandAllocator* command_allocator = CommandAllocator::from(allocator);
        if (!command_allocator || command_allocator->type() != type_)
            return d3d12_trace::result(method, E_INVALIDARG);
        // Another list recording into the allocator: E_INVALIDARG, as on the PC runtime (D3D-51).
        if (command_allocator->recording() && command_allocator->recording() != this)
            return d3d12_trace::result(method, E_INVALIDARG);
        if (!list_.reset()) return d3d12_trace::result(method, E_FAIL);  // D3D12 resets only closed lists
        record_into(command_allocator);
        invalid_ = false;
        calls_.clear();
        release_held();  // what the old recording used: it was replayed at ExecuteCommandLists
        reset_state(state);  // Reset returns every list state to its default
        if (type_ == D3D12_COMMAND_LIST_TYPE_BUNDLE) set_initial_pipeline(state);  // reset_state holds it
        return d3d12_trace::result(method, S_OK);
    }

    // CreateCommandList's pipeline state. As SetPipelineState on a direct list; on a bundle it is the first deferred
    // call, which the application did not make, so it is not traced.
    void set_initial_pipeline(ID3D12PipelineState* state) {
        if (!state) return;
        if (type_ != D3D12_COMMAND_LIST_TYPE_BUNDLE) return SetPipelineState(state);
        defer([=](GraphicsCommandList& l) { l.set_pipeline(state); });
        hold(state);
    }

    void STDMETHODCALLTYPE ExecuteBundle(ID3D12GraphicsCommandList* command_list) override {
        d3d12_trace::record("ID3D12GraphicsCommandList::ExecuteBundle");
        if (list_.closed()) return;
        if (type_ != D3D12_COMMAND_LIST_TYPE_DIRECT) return check(false);
        GraphicsCommandList* bundle = from(command_list);
        if (!bundle || bundle->type_ != D3D12_COMMAND_LIST_TYPE_BUNDLE || bundle->invalid_ ||
            !bundle->list_.closed())
            return check(false);
        // The bundle's state calls and draws play on this list through the members the application would have
        // called, so a draw sees this list's state plus what the bundle set, and what the bundle set stays here
        // afterwards (measured on the PC runtime, docs/d3d12_vulkan.md section 3.2). Commands that need no state
        // were recorded in the bundle's own list and are copied between them, in order.
        std::size_t copied = 0;
        for (const BundleCall& call : bundle->calls_) {
            if (!list_.execute_bundle(bundle->list_, copied, call.commands)) return check(false);
            copied = call.commands;
            call.run(*this);
        }
        check(list_.execute_bundle(bundle->list_, copied));
    }

    // Root signatures, pipeline state and root arguments. The arguments are packed into the 32 user-data dwords
    // the translated shaders read (root_signature.h; docs section 3.5); a draw or dispatch records a copy.
    void STDMETHODCALLTYPE SetComputeRootSignature(ID3D12RootSignature* signature) override {
        d3d12_trace::record("ID3D12GraphicsCommandList::SetComputeRootSignature");
        bind_root(compute_, signature);
    }
    void STDMETHODCALLTYPE SetGraphicsRootSignature(ID3D12RootSignature* signature) override {
        d3d12_trace::record("ID3D12GraphicsCommandList::SetGraphicsRootSignature");
        bind_root(graphics_, signature);
    }

    void STDMETHODCALLTYPE SetPipelineState(ID3D12PipelineState* state) override {
        d3d12_trace::record("ID3D12GraphicsCommandList::SetPipelineState");
        if (list_.closed()) return;
        if (defer([=](GraphicsCommandList& l) { l.set_pipeline(state); })) return hold(state);
        set_pipeline(state);
    }

    // Validates shader-visible descriptor heaps and tracks bound heap identities/bases (docs/d3d12_vulkan.md §3.2/§3.4).
    void STDMETHODCALLTYPE SetDescriptorHeaps(UINT num, ID3D12DescriptorHeap* const* heaps) override {
        d3d12_trace::record("ID3D12GraphicsCommandList::SetDescriptorHeaps");
        if (type_ == D3D12_COMMAND_LIST_TYPE_COPY) return check(false);
        if (num > 2 || (num && !heaps)) return check(false);

        ID3D12DescriptorHeap* cbv = nullptr;
        ID3D12DescriptorHeap* smp = nullptr;
        D3D12_GPU_VIRTUAL_ADDRESS cbv_base = 0;
        D3D12_GPU_VIRTUAL_ADDRESS smp_base = 0;
        IUnknown* const owner = this->attached_device();
        for (UINT i = 0; i < num; ++i) {
            ID3D12DescriptorHeap* raw = heaps[i];
            if (!raw) return check(false);
            if (ID3D12DescriptorHeap* real = d3d12_xs::heap_proxy_real_heap(raw)) raw = real;
            D3D12_DESCRIPTOR_HEAP_DESC desc{};
            D3D12_GPU_VIRTUAL_ADDRESS base = 0;
            IUnknown* const heap_owner = descriptor_heap_info(raw, desc, base);
            if (!heap_owner || (owner && heap_owner != owner)) return check(false);
            if (!(desc.Flags & D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE)) return check(false);
            if (desc.Type == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) {
                if (cbv) return check(false);
                cbv = raw;
                cbv_base = base;
            } else if (desc.Type == D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER) {
                if (smp) return check(false);
                smp = raw;
                smp_base = base;
            } else {
                return check(false);
            }
        }
        hold(cbv);
        hold(smp);
        if (defer([=](GraphicsCommandList& l) { l.verify_bundle_descriptor_heaps(cbv, smp); })) return;
        set_descriptor_heaps(cbv, cbv_base, smp, smp_base);
    }

    void STDMETHODCALLTYPE SetComputeRoot32BitConstant(UINT index, UINT value, UINT offset) override {
        constants(compute_, index, 1, &value, offset);
    }
    void STDMETHODCALLTYPE SetComputeRoot32BitConstants(UINT index, UINT count, const void* data, UINT offset) override {
        constants(compute_, index, count, data, offset);
    }
    void STDMETHODCALLTYPE SetComputeRootConstantBufferView(UINT index, D3D12_GPU_VIRTUAL_ADDRESS va) override {
        address(compute_, index, va, rt::rootsig::ParamType::Descriptor);
    }
    void STDMETHODCALLTYPE SetComputeRootShaderResourceView(UINT index, D3D12_GPU_VIRTUAL_ADDRESS va) override {
        address(compute_, index, va, rt::rootsig::ParamType::Descriptor);
    }
    void STDMETHODCALLTYPE SetComputeRootUnorderedAccessView(UINT index, D3D12_GPU_VIRTUAL_ADDRESS va) override {
        address(compute_, index, va, rt::rootsig::ParamType::Descriptor);
    }
    void STDMETHODCALLTYPE SetComputeRootDescriptorTable(UINT index, D3D12_GPU_DESCRIPTOR_HANDLE table) override {
        address(compute_, index, table.ptr, rt::rootsig::ParamType::Table);
    }
    void STDMETHODCALLTYPE SetGraphicsRoot32BitConstant(UINT index, UINT value, UINT offset) override {
        constants(graphics_, index, 1, &value, offset);
    }
    void STDMETHODCALLTYPE SetGraphicsRoot32BitConstants(UINT index, UINT count, const void* data, UINT offset) override {
        constants(graphics_, index, count, data, offset);
    }
    void STDMETHODCALLTYPE SetGraphicsRootConstantBufferView(UINT index, D3D12_GPU_VIRTUAL_ADDRESS va) override {
        address(graphics_, index, va, rt::rootsig::ParamType::Descriptor);
    }
    void STDMETHODCALLTYPE SetGraphicsRootShaderResourceView(UINT index, D3D12_GPU_VIRTUAL_ADDRESS va) override {
        address(graphics_, index, va, rt::rootsig::ParamType::Descriptor);
    }
    void STDMETHODCALLTYPE SetGraphicsRootUnorderedAccessView(UINT index, D3D12_GPU_VIRTUAL_ADDRESS va) override {
        address(graphics_, index, va, rt::rootsig::ParamType::Descriptor);
    }
    void STDMETHODCALLTYPE SetGraphicsRootDescriptorTable(UINT index, D3D12_GPU_DESCRIPTOR_HANDLE table) override {
        d3d12_trace::record("ID3D12GraphicsCommandList::SetGraphicsRootDescriptorTable");
        address(graphics_, index, table.ptr, rt::rootsig::ParamType::Table);
    }

    void STDMETHODCALLTYPE Dispatch(UINT x, UINT y, UINT z) override {
        d3d12_trace::record("ID3D12GraphicsCommandList::Dispatch");
        if (defer([=](GraphicsCommandList& l) { l.record_dispatch(x, y, z); }, true)) return;
        record_dispatch(x, y, z);
    }

    // ClearState returns every piece of list state to its default, as Reset does (stencil references and predication
    // included).
    void STDMETHODCALLTYPE ClearState(ID3D12PipelineState* state) override {
        if (list_.closed()) return;
        if (type_ == D3D12_COMMAND_LIST_TYPE_BUNDLE) return check(false);
        if (!list_.closed()) check(list_.set_predication({0, 0, 0}));
        reset_state(state);
    }

    // Graphics state. Draws record a copy of it (command_list.h, Draw).
    void STDMETHODCALLTYPE IASetPrimitiveTopology(D3D12_PRIMITIVE_TOPOLOGY topology) override {
        if (list_.closed()) return;
        if (defer([=](GraphicsCommandList& l) { l.topology_ = u(topology); })) return;
        topology_ = u(topology);
    }

    void STDMETHODCALLTYPE IASetVertexBuffers(UINT start, UINT count, const D3D12_VERTEX_BUFFER_VIEW* views) override {
        if (std::uint64_t(start) + count > vertex_buffers_.size()) return check(false);
        if (type_ == D3D12_COMMAND_LIST_TYPE_BUNDLE) {
            std::vector<D3D12_VERTEX_BUFFER_VIEW> copy;
            if (views) copy.assign(views, views + count);
            defer([=, copy = std::move(copy), bound = views != nullptr](GraphicsCommandList& l) {
                l.set_vertex_buffers(start, count, bound ? copy.data() : nullptr);
            });
            return;
        }
        set_vertex_buffers(start, count, views);
    }

    void STDMETHODCALLTYPE IASetIndexBuffer(const D3D12_INDEX_BUFFER_VIEW* view) override {
        if (list_.closed()) return;
        const std::optional<D3D12_INDEX_BUFFER_VIEW> copy = view ? std::optional(*view) : std::nullopt;
        if (defer([=](GraphicsCommandList& l) { l.set_index_buffer(copy ? &*copy : nullptr); })) return;
        set_index_buffer(view);
    }

    // A count of 0 unbinds the viewports; draws then rasterize nothing (record_draw).
    void STDMETHODCALLTYPE RSSetViewports(UINT count, const D3D12_VIEWPORT* viewports) override {
        if (type_ == D3D12_COMMAND_LIST_TYPE_BUNDLE) return check(false);
        if (!count) {
            viewport_ = {};
            return;
        }
        if (!viewports || count > D3D12_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE) return check(false);
        if (count > 1) stub("ID3D12GraphicsCommandList::RSSetViewports (more than one viewport; the first is used)");
        const D3D12_VIEWPORT& v = viewports[0];
        viewport_ = {v.TopLeftX, v.TopLeftY, v.Width, v.Height, v.MinDepth, v.MaxDepth};
    }

    void STDMETHODCALLTYPE RSSetScissorRects(UINT count, const D3D12_RECT* rects) override {
        if (type_ == D3D12_COMMAND_LIST_TYPE_BUNDLE) return check(false);
        // A zero-count call disables all previously bound rectangles; it is not a no-op.
        // With no scissor bound, the rasterizer clips every fragment, as on native D3D12.
        if (!count) {
            scissor_ = {};
            return;
        }
        if (!rects) return;
        if (count > 1) stub("ID3D12GraphicsCommandList::RSSetScissorRects (more than one rectangle; the first is used)");
        scissor_ = {rects[0].left, rects[0].top, rects[0].right, rects[0].bottom};
    }

    // D3D12 reads render-target and depth descriptors here, so their views are copied now.
    void STDMETHODCALLTYPE OMSetRenderTargets(UINT count, const D3D12_CPU_DESCRIPTOR_HANDLE* targets, BOOL single_range,
                                              const D3D12_CPU_DESCRIPTOR_HANDLE* depth) override {
        if (type_ == D3D12_COMMAND_LIST_TYPE_BUNDLE) return check(false);
        if (count > D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT) return check(false);
        render_targets_.fill({});
        render_target_count_ = count;
        draw_views_stale_ = true;
        for (UINT i = 0; i < count && targets; ++i) {
            const SIZE_T handle = single_range ? targets[0].ptr + SIZE_T(i) * descriptor_size : targets[i].ptr;
            if (handle) render_targets_[i] = *reinterpret_cast<const leonida::TargetView*>(handle);
        }
        depth_ = depth && depth->ptr ? *reinterpret_cast<const leonida::TargetView*>(depth->ptr) : leonida::TargetView{};
    }

    void STDMETHODCALLTYPE OMSetBlendFactor(const FLOAT factor[4]) override {
        if (list_.closed()) return;
        const std::array<float, 4> value = factor ? std::array<float, 4>{factor[0], factor[1], factor[2], factor[3]}
                                                  : std::array<float, 4>{1, 1, 1, 1};  // null means { 1, 1, 1, 1 }
        if (defer([=](GraphicsCommandList& l) { l.blend_factor_ = value; })) return;
        blend_factor_ = value;
    }

    void STDMETHODCALLTYPE OMSetStencilRef(UINT reference) override {
        if (list_.closed()) return;
        if (defer([=](GraphicsCommandList& l) { l.front_stencil_ref_ = l.back_stencil_ref_ = reference; })) return;
        front_stencil_ref_ = back_stencil_ref_ = reference;
    }

    void STDMETHODCALLTYPE DrawInstanced(UINT vertices, UINT instances, UINT first_vertex, UINT first_instance) override {
        d3d12_trace::record("ID3D12GraphicsCommandList::DrawInstanced");
        if (defer([=](GraphicsCommandList& l) { l.record_draw(false, vertices, instances, first_vertex, 0, first_instance); },
                  true))
            return;
        record_draw(false, vertices, instances, first_vertex, 0, first_instance);
    }

    void STDMETHODCALLTYPE DrawIndexedInstanced(UINT indices, UINT instances, UINT first_index, INT base_vertex,
                                                UINT first_instance) override {
        d3d12_trace::record("ID3D12GraphicsCommandList::DrawIndexedInstanced");
        if (defer([=](GraphicsCommandList& l) {
                l.record_draw(true, indices, instances, first_index, base_vertex, first_instance);
            }, true))
            return;
        record_draw(true, indices, instances, first_index, base_vertex, first_instance);
    }

    // ID3D12GraphicsCommandList6: a mesh draw. The bound graphics pipeline must have a mesh shader (checked at
    // execution, where the pipeline is resolved). D3D12 caps each count at 65535 and their product at 2^22
    // (D3D12_MS_DISPATCH_MAX_THREAD_GROUPS_PER_GRID); a larger grid is a recording error.
    void STDMETHODCALLTYPE DispatchMesh(UINT x, UINT y, UINT z) override {
        d3d12_trace::record("ID3D12GraphicsCommandList6::DispatchMesh");
        if (x > 65535 || y > 65535 || z > 65535 || std::uint64_t(x) * y * z > (1u << 22)) return check(false);
        if (!graphics_pipeline_ || !graphics_.layout) return check(false);
        if (!(viewport_.width > 0) || !(viewport_.height > 0)) return;  // as record_draw: nothing is rasterized
        leonida::Draw draw{graphics_pipeline_, graphics_.data, draw_views(), depth_, {}, 0, viewport_,
                           scissor_, blend_factor_, front_stencil_ref_, back_stencil_ref_, false, x, y, z, 0, 0};
        draw.mesh = true;
        check(list_.draw(std::move(draw)));
    }

    void STDMETHODCALLTYPE ResourceBarrier(UINT count, const D3D12_RESOURCE_BARRIER* barriers) override {
        d3d12_trace::record("ID3D12GraphicsCommandList::ResourceBarrier");
        if (type_ == D3D12_COMMAND_LIST_TYPE_BUNDLE) return check(false);
        for (UINT i = 0; i < count; ++i) {
            const auto& b = barriers[i];
            switch (b.Type) {
            case D3D12_RESOURCE_BARRIER_TYPE_TRANSITION:
                check(list_.transition({resource_id(b.Transition.pResource), b.Transition.Subresource,
                                        static_cast<std::uint32_t>(b.Transition.StateBefore),
                                        static_cast<std::uint32_t>(b.Transition.StateAfter)},
                                       static_cast<std::uint32_t>(b.Flags)));
                break;
            case D3D12_RESOURCE_BARRIER_TYPE_ALIASING:
                check(!b.Flags && list_.aliasing_barrier({resource_id(b.Aliasing.pResourceBefore),
                                                          resource_id(b.Aliasing.pResourceAfter)}));
                break;
            case D3D12_RESOURCE_BARRIER_TYPE_UAV:
                check(!b.Flags && list_.uav_barrier({resource_id(b.UAV.pResource)}));
                break;
            default:
                check(false);
            }
        }
    }

    void STDMETHODCALLTYPE Barrier(UINT32 groups, const D3D12_BARRIER_GROUP* group) override {
        d3d12_trace::record("ID3D12GraphicsCommandList7::Barrier");
        if (type_ == D3D12_COMMAND_LIST_TYPE_BUNDLE) return check(false);
        for (UINT32 g = 0; g < groups; ++g) {
            for (UINT32 i = 0; i < group[g].NumBarriers; ++i) {
                switch (group[g].Type) {
                case D3D12_BARRIER_TYPE_GLOBAL: {
                    const auto& b = group[g].pGlobalBarriers[i];
                    check(list_.barrier(leonida::GlobalBarrier{u(b.SyncBefore), u(b.SyncAfter), u(b.AccessBefore),
                                                               u(b.AccessAfter)}));
                    break;
                }
                case D3D12_BARRIER_TYPE_BUFFER: {
                    const auto& b = group[g].pBufferBarriers[i];
                    check(list_.barrier(leonida::BufferBarrier{u(b.SyncBefore), u(b.SyncAfter), u(b.AccessBefore),
                                                               u(b.AccessAfter), resource_id(b.pResource)}));
                    break;
                }
                case D3D12_BARRIER_TYPE_TEXTURE: {
                    const auto& b = group[g].pTextureBarriers[i];
                    const auto& r = b.Subresources;
                    check(list_.barrier(leonida::TextureBarrier{
                        u(b.SyncBefore), u(b.SyncAfter), u(b.AccessBefore), u(b.AccessAfter), u(b.LayoutBefore),
                        u(b.LayoutAfter), resource_id(b.pResource),
                        {r.IndexOrFirstMipLevel, r.NumMipLevels, r.FirstArraySlice, r.NumArraySlices, r.FirstPlane,
                         r.NumPlanes},
                        u(b.Flags)}));
                    break;
                }
                default:
                    check(false);
                }
            }
        }
    }

    void STDMETHODCALLTYPE CopyBufferRegion(ID3D12Resource* dst, UINT64 dst_offset, ID3D12Resource* src,
                                            UINT64 src_offset, UINT64 bytes) override {
        d3d12_trace::record("ID3D12GraphicsCommandList::CopyBufferRegion");
        if (type_ == D3D12_COMMAND_LIST_TYPE_BUNDLE) return check(false);
        check(list_.copy_buffer({resource_id(dst), dst_offset, resource_id(src), src_offset, bytes}));
    }

    void STDMETHODCALLTYPE CopyTiles(ID3D12Resource* tiled, const D3D12_TILED_RESOURCE_COORDINATE* start,
                                     const D3D12_TILE_REGION_SIZE* size, ID3D12Resource* linear,
                                     UINT64 linear_offset, D3D12_TILE_COPY_FLAGS flags) override {
        d3d12_trace::record("ID3D12GraphicsCommandList::CopyTiles");
        const auto direction = flags & ~D3D12_TILE_COPY_FLAG_NO_HAZARD;
        const bool to_linear = direction == D3D12_TILE_COPY_FLAG_SWIZZLED_TILED_RESOURCE_TO_LINEAR_BUFFER;
        const auto grid = tile_grid(tiled);
        std::vector<leonida::TileRegion> regions;
        if (!start || !size || !linear || linear == tiled ||
            !append_tile_regions(start, size, grid, regions) ||
            (!to_linear && direction != D3D12_TILE_COPY_FLAG_LINEAR_BUFFER_TO_SWIZZLED_TILED_RESOURCE))
            return check(false);
        const auto tiled_id = resource_id(tiled), linear_id = resource_id(linear);
        if (is_sparse_image(tiled)) {
            const auto buffer = linear->GetDesc();
            const auto desc = tiled->GetDesc();
            const auto count = size->UseBox ? std::uint64_t(size->Width) * size->Height : size->NumTiles;
            if (type_ == D3D12_COMMAND_LIST_TYPE_BUNDLE || buffer.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER ||
                linear_offset > buffer.Width || count > (buffer.Width - linear_offset) / leonida::tile_size_in_bytes)
                return check(false);
            constexpr auto shape = leonida::element_tile_shape(false, 4);
            const leonida::CopyLocation image{tiled_id, false, 0, 0, 0, 0, 0, 0};
            for (const auto& region : regions) {
                for (std::uint64_t tile = region.first; tile < region.first + region.count; ++tile) {
                    const auto x = std::uint32_t(tile % grid.columns) * shape.width;
                    const auto y = std::uint32_t(tile / grid.columns) * shape.height;
                    const auto width = std::min(shape.width, std::uint32_t(desc.Width) - x);
                    const auto height = std::min(shape.height, desc.Height - y);
                    const leonida::CopyLocation footprint{
                        linear_id, true, 0, linear_offset, shape.width, shape.height, 1, shape.width * 4};
                    auto copy = to_linear
                        ? leonida::CopyTexture{footprint, 0, 0, 0, image, true, {x, y, 0, x + width, y + height, 1}}
                        : leonida::CopyTexture{image, x, y, 0, footprint, true, {0, 0, 0, width, height, 1}};
                    copy.tile_footprint = true;
                    check(list_.copy_texture(copy));
                    linear_offset += leonida::tile_size_in_bytes;
                }
            }
            return;
        }
        const auto& region = regions.front();
        const auto tiled_offset = region.first * leonida::tile_size_in_bytes;
        const auto bytes = region.count * leonida::tile_size_in_bytes;
        check(list_.copy_buffer(to_linear
            ? leonida::CopyBuffer{linear_id, linear_offset, tiled_id, tiled_offset, bytes, tiled_id}
            : leonida::CopyBuffer{tiled_id, tiled_offset, linear_id, linear_offset, bytes, tiled_id}));
    }

    void STDMETHODCALLTYPE CopyTextureRegion(const D3D12_TEXTURE_COPY_LOCATION* dst, UINT x, UINT y, UINT z,
                                             const D3D12_TEXTURE_COPY_LOCATION* src, const D3D12_BOX* box) override {
        d3d12_trace::record("ID3D12GraphicsCommandList::CopyTextureRegion");
        if (type_ == D3D12_COMMAND_LIST_TYPE_BUNDLE) return check(false);
        if (!dst || !src) return check(false);
        leonida::CopyTexture copy{location(*dst), x, y, z, location(*src), box != nullptr, {}};
        if (box) copy.box = {box->left, box->top, box->front, box->right, box->bottom, box->back};
        check(list_.copy_texture(copy));
    }

    void STDMETHODCALLTYPE CopyResource(ID3D12Resource* dst, ID3D12Resource* src) override {
        d3d12_trace::record("ID3D12GraphicsCommandList::CopyResource");
        if (type_ == D3D12_COMMAND_LIST_TYPE_BUNDLE) return check(false);
        if (dst && dst == src) {
            // A self-copy on a DEFAULT heap is a no-op, but an UPLOAD/READBACK resource still cannot
            // be both source and destination: native D3D12 rejects it at Close. Check this before
            // the no-op is omitted from the recorded command list.
            D3D12_HEAP_PROPERTIES properties{};
            if (FAILED(dst->GetHeapProperties(&properties, nullptr)) ||
                properties.Type != D3D12_HEAP_TYPE_DEFAULT) return check(false);
        }
        check(list_.copy_resource({resource_id(dst), resource_id(src)}));
    }

    // The format picks the member of a typeless family to resolve as; the textures' own format is used.
    void STDMETHODCALLTYPE ResolveSubresource(ID3D12Resource* dst, UINT dst_subresource, ID3D12Resource* src,
                                              UINT src_subresource, DXGI_FORMAT format) override {
        d3d12_trace::record("ID3D12GraphicsCommandList::ResolveSubresource");
        if (type_ == D3D12_COMMAND_LIST_TYPE_BUNDLE) return check(false);
        const auto integer_color = [](ID3D12Resource* resource) {
            if (!resource) return false;
            const auto desc = resource->GetDesc();
            return !(desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL) &&
                   color_resolve_format_is_integer(desc.Format);
        };
        if (color_resolve_format_is_integer(format) || integer_color(dst) || integer_color(src)) {
            note_once("ResolveSubresource refused: integer colour formats cannot be resolved");
            return check(false);
        }
        check(list_.resolve_subresource({resource_id(dst), dst_subresource, resource_id(src), src_subresource}));
    }

    void STDMETHODCALLTYPE ResolveSubresourceRegion(ID3D12Resource* dst, UINT dst_subresource, UINT x, UINT y,
                                                     ID3D12Resource* src, UINT src_subresource, D3D12_RECT* rect,
                                                     DXGI_FORMAT format, D3D12_RESOLVE_MODE mode) override {
        d3d12_trace::record("ID3D12GraphicsCommandList1::ResolveSubresourceRegion");
        if (type_ != D3D12_COMMAND_LIST_TYPE_DIRECT) return check(false);
        leonida::ResolveRegion region{x, y, std::nullopt};
        if (mode != D3D12_RESOLVE_MODE_AVERAGE || !color_resolve_format_matches(format, dst, src)) {
            if (!depth_stencil_resolve_supported(format, dst, dst_subresource, x, y, src, src_subresource, rect,
                                                 mode)) {
                note_once("ResolveSubresourceRegion refused: neither an average colour resolve nor a depth or "
                          "stencil one the device can replay");
                return check(false);
            }
            region.mode = mode == D3D12_RESOLVE_MODE_MIN   ? leonida::d3d12_resolve_mode::minimum
                        : mode == D3D12_RESOLVE_MODE_MAX   ? leonida::d3d12_resolve_mode::maximum
                                                           : leonida::d3d12_resolve_mode::average;
        }
        if (rect) region.src_rect = leonida::Rect{rect->left, rect->top, rect->right, rect->bottom};
        check(list_.resolve_subresource({resource_id(dst), dst_subresource, resource_id(src), src_subresource, region}));
    }

    // A null buffer turns predication off (docs/d3d12_vulkan.md section 4.7). Bundles and copy lists may not set it.
    // Recording-time checks match live Windows 11 WARP and RTX 5090 D3D12: null buffer requires offset 0 and
    // EQUAL_ZERO; a non-null resource must be a non-READBACK buffer with an 8-byte aligned in-bounds range.
    void STDMETHODCALLTYPE SetPredication(ID3D12Resource* buffer, UINT64 offset, D3D12_PREDICATION_OP op) override {
        d3d12_trace::record("ID3D12GraphicsCommandList::SetPredication");
        if (type_ == D3D12_COMMAND_LIST_TYPE_BUNDLE || type_ == D3D12_COMMAND_LIST_TYPE_COPY) return check(false);
        if (!buffer) {
            if (offset != 0 || op != D3D12_PREDICATION_OP_EQUAL_ZERO) return check(false);
            return check(list_.set_predication({0, 0, 0}));
        }
        const auto desc = buffer->GetDesc();
        if (desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER) return check(false);
        if (offset % leonida::predication_alignment != 0 || offset > desc.Width ||
            desc.Width - offset < leonida::predication_alignment)
            return check(false);
        D3D12_HEAP_PROPERTIES heap{};
        if (SUCCEEDED(buffer->GetHeapProperties(&heap, nullptr)) && heap.Type == D3D12_HEAP_TYPE_READBACK)
            return check(false);
        check(list_.set_predication({resource_id(buffer), offset, u(op)}));
    }

    void STDMETHODCALLTYPE ClearRenderTargetView(D3D12_CPU_DESCRIPTOR_HANDLE view, const FLOAT color[4], UINT count,
                                                 const D3D12_RECT* rects) override {
        d3d12_trace::record("ID3D12GraphicsCommandList::ClearRenderTargetView");
        if (type_ == D3D12_COMMAND_LIST_TYPE_BUNDLE) return check(false);
        if (!view.ptr || !color) return check(false);
        check(list_.clear_render_target(*reinterpret_cast<const leonida::TargetView*>(view.ptr),
                                        {color[0], color[1], color[2], color[3]}, to_rects(count, rects)));
    }

    void STDMETHODCALLTYPE ClearDepthStencilView(D3D12_CPU_DESCRIPTOR_HANDLE view, D3D12_CLEAR_FLAGS flags,
                                                 FLOAT depth, UINT8 stencil, UINT count,
                                                 const D3D12_RECT* rects) override {
        d3d12_trace::record("ID3D12GraphicsCommandList::ClearDepthStencilView");
        if (type_ == D3D12_COMMAND_LIST_TYPE_BUNDLE) return check(false);
        if (!view.ptr) return check(false);
        check(list_.clear_depth_stencil(*reinterpret_cast<const leonida::TargetView*>(view.ptr), u(flags), depth,
                                        stencil, to_rects(count, rects)));
    }

    // A null region discards the whole resource. Not allowed in a bundle.
    void STDMETHODCALLTYPE DiscardResource(ID3D12Resource* resource, const D3D12_DISCARD_REGION* region) override {
        if (type_ == D3D12_COMMAND_LIST_TYPE_BUNDLE) return check(false);
        if (!region) return check(list_.discard({resource_id(resource), 0, 0, false, true}));
        if (region->NumRects && !region->pRects) return check(false);
        check(list_.discard({resource_id(resource), region->FirstSubresource, region->NumSubresources,
                             region->NumRects != 0}));
    }

    // GCL2. A null pModes means DEFAULT for every write. The address is resolved when the list is executed.
    // Allowed in a bundle (the API restrictions table of "Recording command lists and bundles").
    void STDMETHODCALLTYPE WriteBufferImmediate(UINT count, const D3D12_WRITEBUFFERIMMEDIATE_PARAMETER* params,
                                                const D3D12_WRITEBUFFERIMMEDIATE_MODE* modes) override {
        if (count && !params) return check(false);
        for (UINT i = 0; i < count; ++i)
            check(list_.write_immediate({params[i].Dest, params[i].Value, modes ? u(modes[i]) : 0}));
    }

    // GCL1. Not allowed in a bundle, as no copy is.
    void STDMETHODCALLTYPE AtomicCopyBufferUINT(ID3D12Resource* dst, UINT64 dst_offset, ID3D12Resource* src,
                                                UINT64 src_offset, UINT count, ID3D12Resource* const* resources,
                                                const D3D12_SUBRESOURCE_RANGE_UINT64* ranges) override {
        atomic_copy(4, dst, dst_offset, src, src_offset, count, resources, ranges);
    }
    void STDMETHODCALLTYPE AtomicCopyBufferUINT64(ID3D12Resource* dst, UINT64 dst_offset, ID3D12Resource* src,
                                                  UINT64 src_offset, UINT count, ID3D12Resource* const* resources,
                                                  const D3D12_SUBRESOURCE_RANGE_UINT64* ranges) override {
        atomic_copy(8, dst, dst_offset, src, src_offset, count, resources, ranges);
    }

    // The CPU handle's slot holds the view as the GFX10 descriptor a shader would load (docs/d3d12_vulkan.md
    // section 3.4): a T# for a texture, a V# for a buffer. It is read now, because the slot may be rewritten before
    // the list runs. The GPU handle names the same view in a shader-visible heap and is not needed.
    void STDMETHODCALLTYPE ClearUnorderedAccessViewUint(D3D12_GPU_DESCRIPTOR_HANDLE, D3D12_CPU_DESCRIPTOR_HANDLE view,
                                                        ID3D12Resource* resource, const UINT values[4], UINT count,
                                                        const D3D12_RECT* rects) override {
        if (!values) return check(false);
        clear_unordered_access(view, resource, false, {values[0], values[1], values[2], values[3]}, count, rects);
    }

    void STDMETHODCALLTYPE ClearUnorderedAccessViewFloat(D3D12_GPU_DESCRIPTOR_HANDLE, D3D12_CPU_DESCRIPTOR_HANDLE view,
                                                         ID3D12Resource* resource, const FLOAT values[4], UINT count,
                                                         const D3D12_RECT* rects) override {
        if (!values) return check(false);
        clear_unordered_access(view, resource, true,
                               {std::bit_cast<std::uint32_t>(values[0]), std::bit_cast<std::uint32_t>(values[1]),
                                std::bit_cast<std::uint32_t>(values[2]), std::bit_cast<std::uint32_t>(values[3])},
                               count, rects);
    }

    // As the host's Close reports it: compute and copy lists hold timestamps only (not even pipeline statistics on
    // compute), and bundles no queries at all.
    bool query_allowed(D3D12_QUERY_TYPE type) const {
        if (type_ == D3D12_COMMAND_LIST_TYPE_DIRECT) return true;
        return type_ != D3D12_COMMAND_LIST_TYPE_BUNDLE && type == D3D12_QUERY_TYPE_TIMESTAMP;
    }

    // Queries (docs/d3d12_vulkan.md section 4.6). Heap types, index ranges and the destination buffer are checked
    // at ExecuteCommandLists, when the heap and the buffer are known.
    void STDMETHODCALLTYPE BeginQuery(ID3D12QueryHeap* heap, D3D12_QUERY_TYPE type, UINT index) override {
        if (!query_allowed(type)) return check(false);
        check(list_.begin_query({query_heap_id(heap), u(type), index}));
    }

    void STDMETHODCALLTYPE EndQuery(ID3D12QueryHeap* heap, D3D12_QUERY_TYPE type, UINT index) override {
        if (!query_allowed(type)) return check(false);
        check(list_.end_query({query_heap_id(heap), u(type), index}));
    }

    void STDMETHODCALLTYPE ResolveQueryData(ID3D12QueryHeap* heap, D3D12_QUERY_TYPE type, UINT first, UINT count,
                                            ID3D12Resource* dst, UINT64 offset) override {
        if (!query_allowed(type)) return check(false);
        check(list_.resolve_queries({query_heap_id(heap), u(type), first, count, resource_id(dst), offset}));
    }

    // GCL8. Each draw applies the two faces with vkCmdSetStencilReference(VK_STENCIL_FACE_FRONT_BIT / BACK_BIT).
    void STDMETHODCALLTYPE OMSetFrontAndBackStencilRef(UINT front, UINT back) override {
        d3d12_trace::record("ID3D12GraphicsCommandList8::OMSetFrontAndBackStencilRef");
        if (list_.closed()) return check(false);
        if (defer([=](GraphicsCommandList& l) {
                l.front_stencil_ref_ = front;
                l.back_stencil_ref_ = back;
            }))
            return;
        front_stencil_ref_ = front;
        back_stencil_ref_ = back;
    }

    // Debug markers, recorded in order with the other commands and replayed as VK_EXT_debug_utils labels. The trace
    // names the call only, never the label.
    void STDMETHODCALLTYPE SetMarker(UINT metadata, const void* data, UINT size) override {
        d3d12_trace::record("ID3D12GraphicsCommandList::SetMarker");
        check(list_.marker(leonida::DebugMarker::Kind::set, event_label(metadata, data, size)));
    }
    void STDMETHODCALLTYPE BeginEvent(UINT metadata, const void* data, UINT size) override {
        d3d12_trace::record("ID3D12GraphicsCommandList::BeginEvent");
        check(list_.marker(leonida::DebugMarker::Kind::begin, event_label(metadata, data, size)));
    }
    void STDMETHODCALLTYPE EndEvent() override {
        d3d12_trace::record("ID3D12GraphicsCommandList::EndEvent");
        check(list_.marker(leonida::DebugMarker::Kind::end));
    }

    void STDMETHODCALLTYPE ExecuteIndirect(ID3D12CommandSignature* signature, UINT max_command_count,
                                           ID3D12Resource* argument_buffer, UINT64 argument_buffer_offset,
                                           ID3D12Resource* count_buffer, UINT64 count_buffer_offset) override {
        if (type_ == D3D12_COMMAND_LIST_TYPE_BUNDLE) return check(false);
        leonida::IndirectKind kind = leonida::IndirectKind::draw;
        UINT stride = 0;
        if (!command_signature_info(signature, kind, stride) || !argument_buffer)
            return check(false);
        const auto arg_desc = argument_buffer->GetDesc();
        if (arg_desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER || argument_buffer_offset % 4 != 0)
            return check(false);
        if (count_buffer) {
            const auto count_desc = count_buffer->GetDesc();
            if (count_desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER || count_buffer_offset % 4 != 0 ||
                count_desc.Width < sizeof(UINT) || count_buffer_offset > count_desc.Width - sizeof(UINT))
                return check(false);
        } else if (count_buffer_offset != 0) {
            return check(false);
        }
        if (max_command_count) {
            const UINT64 record_size =
                kind == leonida::IndirectKind::draw ? sizeof(D3D12_DRAW_ARGUMENTS)
                : kind == leonida::IndirectKind::draw_indexed ? sizeof(D3D12_DRAW_INDEXED_ARGUMENTS)
                : kind == leonida::IndirectKind::dispatch ? sizeof(D3D12_DISPATCH_ARGUMENTS)
                : 0;
            if (!record_size || stride < record_size || stride % 4 != 0)
                return check(false);
            const UINT64 required = UINT64(max_command_count - 1) * stride + record_size;
            if (argument_buffer_offset > arg_desc.Width || arg_desc.Width - argument_buffer_offset < required)
                return check(false);
        }
        leonida::ExecuteIndirect cmd{};
        cmd.kind = kind;
        cmd.stride = stride;
        cmd.max_count = max_command_count;
        cmd.argument_buffer = resource_id(argument_buffer);
        cmd.argument_offset = argument_buffer_offset;
        cmd.count_buffer = count_buffer ? resource_id(count_buffer) : 0;
        cmd.count_offset = count_buffer ? count_buffer_offset : 0;
        if (kind == leonida::IndirectKind::dispatch && max_command_count) {  // the pipeline and root arguments a Dispatch would record
            if (!compute_pipeline_ || !compute_.layout) return check(false);
            cmd.pipeline = compute_pipeline_;
            cmd.user_data = compute_.data;
        }
        check(list_.execute_indirect(cmd));
    }

    void STDMETHODCALLTYPE BuildRaytracingAccelerationStructure(
        const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC* desc, UINT postbuild_count,
        const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC* postbuild) override {
        if (type_ == D3D12_COMMAND_LIST_TYPE_BUNDLE || !desc) return check(false);
        if (postbuild_count && !postbuild) return check(false);
        for (UINT i = 0; i < postbuild_count; ++i) {
            if (!valid_postbuild_info_type(postbuild[i].InfoType)) return check(false);
        }
        const auto& in = desc->Inputs;
        if (in.Type != D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL &&
            in.Type != D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL)
            return check(false);
        if (in.DescsLayout != D3D12_ELEMENTS_LAYOUT_ARRAY &&
            in.DescsLayout != D3D12_ELEMENTS_LAYOUT_ARRAY_OF_POINTERS)
            return check(false);
        leonida::BuildAccelerationStructure cmd{};
        cmd.is_tlas = (in.Type == D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL);
        cmd.update = (in.Flags & D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE) != 0;
        if (in.Flags & D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE)
            cmd.build_flags |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
        if (in.Flags & D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION)
            cmd.build_flags |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;
        if (in.Flags & D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE)
            cmd.build_flags |= VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        if (in.Flags & D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD)
            cmd.build_flags |= VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;
        if (in.Flags & D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_MINIMIZE_MEMORY)
            cmd.build_flags |= VK_BUILD_ACCELERATION_STRUCTURE_LOW_MEMORY_BIT_KHR;   // XDXR: BOX16 nodes (BLAS)
        cmd.dst_va = desc->DestAccelerationStructureData;
        cmd.src_va = desc->SourceAccelerationStructureData;
        cmd.scratch_va = desc->ScratchAccelerationStructureData;
        if (cmd.is_tlas) {
            cmd.instances_va = in.InstanceDescs;
            cmd.instance_count = in.NumDescs;
            cmd.instance_pointers = in.DescsLayout == D3D12_ELEMENTS_LAYOUT_ARRAY_OF_POINTERS;
        } else {
            for (UINT i = 0; i < in.NumDescs; ++i) {
                const D3D12_RAYTRACING_GEOMETRY_DESC* gd =
                    (in.DescsLayout == D3D12_ELEMENTS_LAYOUT_ARRAY_OF_POINTERS)
                        ? (in.ppGeometryDescs ? in.ppGeometryDescs[i] : nullptr)
                        : (in.pGeometryDescs ? &in.pGeometryDescs[i] : nullptr);
                if (!gd || gd->Type != D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES) return check(false);
                leonida::RtTriangleGeometry t{};
                t.vertex_va = gd->Triangles.VertexBuffer.StartAddress;
                t.vertex_count = gd->Triangles.VertexCount;
                t.vertex_stride = static_cast<std::uint32_t>(gd->Triangles.VertexBuffer.StrideInBytes);
                // The vertex formats D3D12 allows for BLAS builds; any other stays VK_FORMAT_UNDEFINED and the
                // build is refused at replay instead of reading the wrong layout.
                switch (gd->Triangles.VertexFormat) {
                case DXGI_FORMAT_R32G32B32_FLOAT: t.vertex_format = VK_FORMAT_R32G32B32_SFLOAT; break;
                case DXGI_FORMAT_R32G32_FLOAT: t.vertex_format = VK_FORMAT_R32G32_SFLOAT; break;
                case DXGI_FORMAT_R16G16B16A16_FLOAT: t.vertex_format = VK_FORMAT_R16G16B16A16_SFLOAT; break;
                case DXGI_FORMAT_R16G16_FLOAT: t.vertex_format = VK_FORMAT_R16G16_SFLOAT; break;
                case DXGI_FORMAT_R16G16B16A16_SNORM: t.vertex_format = VK_FORMAT_R16G16B16A16_SNORM; break;
                case DXGI_FORMAT_R16G16_SNORM: t.vertex_format = VK_FORMAT_R16G16_SNORM; break;
                default: t.vertex_format = VK_FORMAT_UNDEFINED; break;
                }
                t.index_va = gd->Triangles.IndexBuffer;
                t.index_count = gd->Triangles.IndexCount;
                if (gd->Triangles.IndexFormat == DXGI_FORMAT_R16_UINT)
                    t.index_type = 1;
                else if (gd->Triangles.IndexFormat == DXGI_FORMAT_R32_UINT)
                    t.index_type = 2;
                t.transform_va = gd->Triangles.Transform3x4;
                t.opaque = (gd->Flags & D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE) != 0;
                t.flags = static_cast<std::uint32_t>(gd->Flags);
                cmd.triangles.push_back(t);
            }
        }
        const std::uint64_t built = cmd.dst_va;
        const bool recorded = list_.build_acceleration_structure(std::move(cmd));
        check(recorded);
        if (!recorded) return;
        // The postbuild values of the structure just built, after it (D3D12 orders them after the build).
        for (UINT i = 0; postbuild && i < postbuild_count; ++i) {
            leonida::AccelerationStructurePostbuild info{};
            info.dest_va = postbuild[i].DestBuffer;
            info.type = static_cast<std::uint32_t>(postbuild[i].InfoType);
            info.sources = {built};
            const bool ok = list_.acceleration_structure_postbuild(std::move(info));
            check(ok);
            if (!ok) return;
        }
    }

    void STDMETHODCALLTYPE EmitRaytracingAccelerationStructurePostbuildInfo(
        const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC* desc, UINT count,
        const D3D12_GPU_VIRTUAL_ADDRESS* sources) override {
        if (type_ == D3D12_COMMAND_LIST_TYPE_BUNDLE || !desc || !count || !sources) return check(false);
        if (!valid_postbuild_info_type(desc->InfoType)) return check(false);
        leonida::AccelerationStructurePostbuild info{};
        info.dest_va = desc->DestBuffer;
        info.type = static_cast<std::uint32_t>(desc->InfoType);
        info.sources.assign(sources, sources + count);
        check(list_.acceleration_structure_postbuild(std::move(info)));
    }

    void STDMETHODCALLTYPE CopyRaytracingAccelerationStructure(D3D12_GPU_VIRTUAL_ADDRESS dest,
                                                               D3D12_GPU_VIRTUAL_ADDRESS source,
                                                               D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE mode) override {
        if (type_ == D3D12_COMMAND_LIST_TYPE_BUNDLE) return check(false);
        if (!valid_copy_mode(mode)) return check(false);
        check(list_.copy_acceleration_structure({dest, source, static_cast<std::uint32_t>(mode)}));
    }

    static GraphicsCommandList* from(IUnknown* object) { return Child::from<GraphicsCommandList>(object); }

private:
    static bool valid_postbuild_info_type(D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_TYPE type) {
        switch (type) {
        case D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE:
        case D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_TOOLS_VISUALIZATION:
        case D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_SERIALIZATION:
        case D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_CURRENT_SIZE:
            return true;
        default:
            return false;
        }
    }

    static bool valid_copy_mode(D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE mode) {
        switch (mode) {
        case D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_CLONE:
        case D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_COMPACT:
        case D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_VISUALIZATION_DECODE_FOR_TOOLS:
        case D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_SERIALIZE:
        case D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_DESERIALIZE:
            return true;
        default:
            return false;
        }
    }
    template<class E>
    static std::uint32_t u(E value) {
        return static_cast<std::uint32_t>(value);
    }

    static leonida::CopyLocation location(const D3D12_TEXTURE_COPY_LOCATION& l) {
        if (l.Type == D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT) {
            const auto& f = l.PlacedFootprint;
            return {resource_id(l.pResource), true, 0, f.Offset, f.Footprint.Width, f.Footprint.Height,
                    f.Footprint.Depth, f.Footprint.RowPitch};
        }
        return {resource_id(l.pResource), false, l.SubresourceIndex, 0, 0, 0, 0, 0};
    }

    static std::vector<leonida::Rect> to_rects(UINT count, const D3D12_RECT* rects) {
        std::vector<leonida::Rect> out;
        for (UINT i = 0; rects && i < count; ++i)
            out.push_back({rects[i].left, rects[i].top, rects[i].right, rects[i].bottom});
        return out;
    }

    void check(bool ok) {
        if (!ok) invalid_ = true;
    }

    void stop_recording() {
        if (!allocator_) return;
        if (allocator_->recording() == this) allocator_->set_recording(nullptr);
        allocator_->Release();
        allocator_ = nullptr;
    }

    static UINT subresource_count(const D3D12_RESOURCE_DESC& desc) {
        if (desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) return 1;
        const UINT mips = desc.MipLevels ? desc.MipLevels : 1u;
        const UINT layers = desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D
                                ? 1u
                                : (desc.DepthOrArraySize ? desc.DepthOrArraySize : 1u);
        UINT planes = 1;
        switch (desc.Format) {
        case DXGI_FORMAT_D24_UNORM_S8_UINT:
        case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
        case DXGI_FORMAT_R24G8_TYPELESS:
        case DXGI_FORMAT_R32G8X24_TYPELESS:
        case DXGI_FORMAT_R24_UNORM_X8_TYPELESS:
        case DXGI_FORMAT_X24_TYPELESS_G8_UINT:
        case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS:
        case DXGI_FORMAT_X32_TYPELESS_G8X24_UINT:
        case DXGI_FORMAT_NV12:
        case DXGI_FORMAT_P010:
        case DXGI_FORMAT_P016:
            planes = 2;
            break;
        default:
            break;
        }
        return mips * layers * planes;
    }

    void atomic_copy(std::uint32_t bytes, ID3D12Resource* dst, UINT64 dst_offset, ID3D12Resource* src,
                     UINT64 src_offset, UINT count, ID3D12Resource* const* resources,
                     const D3D12_SUBRESOURCE_RANGE_UINT64* ranges) {
        if (type_ == D3D12_COMMAND_LIST_TYPE_BUNDLE || !dst || !src || (count && (!resources || !ranges)))
            return check(false);
        const auto dst_desc = dst->GetDesc();
        const auto src_desc = src->GetDesc();
        if (dst_desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER ||
            src_desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER)
            return check(false);
        if (dst_offset % bytes != 0 || src_offset % bytes != 0 || dst_offset > dst_desc.Width ||
            dst_desc.Width - dst_offset < bytes || src_offset > src_desc.Width || src_desc.Width - src_offset < bytes)
            return check(false);
        leonida::AtomicCopyBuffer copy{resource_id(dst), dst_offset, resource_id(src), src_offset, bytes, {}};
        copy.dependencies.reserve(count);
        for (UINT i = 0; i < count; ++i) {
            ID3D12Resource* dep = resources[i];
            if (!dep) return check(false);
            const auto dep_desc = dep->GetDesc();
            const auto& r = ranges[i];
            UINT64 begin = r.Range.Begin;
            UINT64 end = r.Range.End;
            if (dep_desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) {
                if (r.Subresource != 0) return check(false);
                if (begin >= end || (end != UINT64_MAX && end > dep_desc.Width))
                    return check(false);
            } else {
                const UINT total_subresources = subresource_count(dep_desc);
                if (r.Subresource != D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES && r.Subresource >= total_subresources)
                    return check(false);
                if (begin != 0 || end != UINT64_MAX) return check(false);
            }
            copy.dependencies.push_back({resource_id(dep), r.Subresource, begin, end});
        }
        check(list_.atomic_copy(copy));
    }

    void clear_unordered_access(D3D12_CPU_DESCRIPTOR_HANDLE view, ID3D12Resource* resource, bool is_float,
                                const std::array<std::uint32_t, 4>& values, UINT count, const D3D12_RECT* rects) {
        if (type_ == D3D12_COMMAND_LIST_TYPE_BUNDLE) return check(false);  // no clear is allowed in a bundle
        leonida::UavView target{};
        if (!view.ptr || !resource || !unordered_access_view(view, resource, target)) return check(false);
        check(list_.clear_unordered_access(target, is_float, values, to_rects(count, rects)));
    }

    // The Vulkan format a descriptor's FORMAT and DST_SEL name: a UAV has no component mapping of its own, so its
    // DST_SEL is the one its format reads R, G, B, A with (gfx10::format_swizzle), which tells B8G8R8A8 from
    // R8G8B8A8. VK_FORMAT_UNDEFINED: not a format the table has.
    static VkFormat descriptor_format(std::uint32_t code, const leonida::gfx10::Swizzle& swizzle) {
        for (const auto& row : leonida::gfx10::format_table)
            if (row.gfx10 == code && row.channels == swizzle) return row.vulkan;
        return leonida::gfx10::vulkan_format(code);
    }

    // A V#: stride 0 and no format is a raw buffer of NUM_RECORDS bytes, a stride without a format a structured one
    // of NUM_RECORDS records, and a format a typed one of NUM_RECORDS elements. Its address is a GPU VA, so the
    // view starts at that minus the resource's own.
    static bool unordered_access_view(D3D12_CPU_DESCRIPTOR_HANDLE handle, ID3D12Resource* resource,
                                      leonida::UavView& out) {
        namespace g = leonida::gfx10;
        g::ImageWords words{};
        std::memcpy(words.data(), reinterpret_cast<const void*>(handle.ptr), sizeof(words));
        out.resource = resource_id(resource);
        if (g::is_image(words)) {
            g::Image image{};
            if (!g::decode(words, image)) return false;
            const bool volume = image.type == g::ImageType::image_3d;
            out.buffer = false;
            out.format = std::uint32_t(descriptor_format(image.format, image.swizzle));
            out.mip = image.first_mip;
            out.first_layer = volume ? 0 : image.first_layer;  // a 3D view clears every slice of its mip
            out.layer_count = volume ? 1 : image.layer_count;
            return out.format != VK_FORMAT_UNDEFINED;
        }
        g::BufferWords buffer_words{};
        std::memcpy(buffer_words.data(), words.data(), sizeof(buffer_words));
        g::Buffer buffer{};
        if (!g::decode(buffer_words, buffer)) return false;
        const UINT64 base = resource->GetGPUVirtualAddress();
        if (buffer.address < base) return false;
        out.buffer = true;
        out.offset = buffer.address - base;
        if (buffer.format) {
            const VkFormat format = descriptor_format(buffer.format, buffer.swizzle);
            const auto* layout = leonida::uav_clear::layout(format);
            if (!layout) return false;  // not a typed-UAV format
            out.format = std::uint32_t(format);
            out.element_size = layout->bytes;
            out.size = std::uint64_t(buffer.num_records) * layout->bytes;
        } else if (buffer.stride) {
            out.element_size = buffer.stride;
            out.size = std::uint64_t(buffer.num_records) * buffer.stride;
        } else {
            out.element_size = 4;
            out.size = buffer.num_records;
        }
        return true;
    }

    struct RootState {
        const RootArgumentLayout* layout = nullptr;
        std::array<std::uint32_t, rt::rootsig::kMaxUserSgprs> data{};
    };

    // D3D12 leaves root arguments undefined after a root-signature change; they start at zero here. Setting the
    // signature that is already bound changes nothing: "existing root signature bindings do not become stale"
    // (D3D12, "Using a Root Signature"), and engines rebind routinely.
    void bind_root(RootState& state, ID3D12RootSignature* signature) {
        if (defer([=, compute = &state == &compute_](GraphicsCommandList& l) {
                l.bind_root(compute ? l.compute_ : l.graphics_, signature);
            }))
            return hold(signature);  // the bundle keeps it for as long as it can be executed
        const RootArgumentLayout* layout = root_signature_layout(signature);
        if (layout && layout == state.layout) return;
        hold(signature);  // state.layout points into it
        state.layout = layout;
        state.data = {};
        if (signature && !state.layout) stub("ID3D12GraphicsCommandList::Set*RootSignature(foreign root signature)");
        // CreateRootSignature accepted it for the other bind point only (its log line says why).
        if (layout && !(is_compute(state) ? layout->compute_ok : layout->graphics_ok)) {
            stub(is_compute(state) ? "SetComputeRootSignature(signature that does not fit the compute layout)"
                                   : "SetGraphicsRootSignature(signature that does not fit the graphics layout)");
            check(false);
        }
    }

    bool is_compute(const RootState& state) const { return &state == &compute_; }

    // The parameter's first user-data dword at this bind point (RootArgumentLayout).
    std::uint32_t first_dword(const RootState& state, const RootArgumentLayout::Parameter& p) const {
        return is_compute(state) ? p.compute_offset : p.user_offset;
    }

    // The parameter, if it exists and has the expected type; null after recording an invalid call.
    const RootArgumentLayout::Parameter* parameter(RootState& state, UINT index, rt::rootsig::ParamType type) {
        if (!state.layout || index >= state.layout->parameters.size() ||
            state.layout->parameters[index].type != type) {
            check(false);
            return nullptr;
        }
        return &state.layout->parameters[index];
    }

    // Dwords beyond the 32 user registers are what the console spills to memory; that layout is still open.
    void write(RootState& state, std::uint32_t dword, std::uint32_t value) {
        if (dword < rt::rootsig::kMaxUserSgprs)
            state.data[dword] = value;
        else
            stub("root arguments beyond 32 user-data dwords (spill not modelled)");
    }

    void constants(RootState& state, UINT index, UINT count, const void* data, UINT offset) {
        if (type_ == D3D12_COMMAND_LIST_TYPE_BUNDLE) {
            // The layout is the executing list's, so the range is checked when it runs; no signature takes more
            // than D3D12_MAX_ROOT_COST dwords, which bounds what is copied from the caller's memory.
            if (count > D3D12_MAX_ROOT_COST) return check(false);
            const auto* words = static_cast<const std::uint32_t*>(data);
            std::vector<std::uint32_t> copy;
            if (data) copy.assign(words, words + count);
            // A count of 0 with a non-null pointer is a valid call; an empty vector's data() may be null.
            static constexpr std::uint32_t none = 0;
            defer([=, compute = &state == &compute_, copy = std::move(copy),
                   given = data != nullptr](GraphicsCommandList& l) {
                l.constants(compute ? l.compute_ : l.graphics_, index, count,
                            !given ? nullptr : copy.empty() ? &none : copy.data(), offset);
            });
            return;
        }
        const RootArgumentLayout::Parameter* p = parameter(state, index, rt::rootsig::ParamType::Constants);
        if (!p) return;
        if (!data || std::uint64_t(offset) + count > p->constant_count) return check(false);
        const auto* values = static_cast<const std::uint32_t*>(data);
        const std::uint32_t first = first_dword(state, *p);
        for (UINT i = 0; i < count; ++i) write(state, first + offset + i, values[i]);
    }

    void address(RootState& state, UINT index, std::uint64_t va, rt::rootsig::ParamType type) {
        if (defer([=, compute = &state == &compute_](GraphicsCommandList& l) {
                l.address(compute ? l.compute_ : l.graphics_, index, va, type);
            }))
            return;
        const RootArgumentLayout::Parameter* p = parameter(state, index, type);
        if (!p) return;
        const std::uint32_t first = first_dword(state, *p);
        write(state, first, std::uint32_t(va));
        // In the compute layout a descriptor table is one dword, the 32-bit VA of its first descriptor (the heaps
        // live below 4 GiB); a root view keeps both halves. The graphics layout gives every address two dwords.
        if (!(is_compute(state) && type == rt::rootsig::ParamType::Table)) write(state, first + 1, std::uint32_t(va >> 32));
    }

    D3D12_COMMAND_LIST_TYPE type_;
    CommandAllocator* allocator_ = nullptr;  // referenced while the list is open
    leonida::CommandList list_;
    bool invalid_ = false;
    UINT front_stencil_ref_ = 0;
    UINT back_stencil_ref_ = 0;
    RootState compute_;
    RootState graphics_;
    std::uintptr_t compute_pipeline_ = 0;
    std::uintptr_t graphics_pipeline_ = 0;
    ID3D12DescriptorHeap* cbv_srv_uav_heap_ = nullptr;
    ID3D12DescriptorHeap* sampler_heap_ = nullptr;
    D3D12_GPU_VIRTUAL_ADDRESS cbv_srv_uav_base_ = 0;
    D3D12_GPU_VIRTUAL_ADDRESS sampler_base_ = 0;
    std::uint32_t topology_ = 0;  // D3D_PRIMITIVE_TOPOLOGY_UNDEFINED
    std::array<leonida::VertexBufferView, D3D12_IA_VERTEX_INPUT_RESOURCE_SLOT_COUNT> vertex_buffers_{};
    leonida::IndexBufferView index_{};
    leonida::Viewport viewport_{};
    leonida::Rect scissor_{};
    std::array<leonida::TargetView, D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT> render_targets_{};
    UINT render_target_count_ = 0;
    // One block shared by every draw that keeps the same targets and buffers, rebuilt when either changes, so a
    // run of draws does not allocate per draw.
    std::shared_ptr<const leonida::DrawViews> draw_views_;
    bool draw_views_stale_ = true;
    leonida::TargetView depth_{};
    std::array<float, 4> blend_factor_{1, 1, 1, 1};

    // The root signatures and pipeline states this recording uses. D3D12 lets an application release them right
    // after Set*RootSignature / SetPipelineState; the list keeps them until Reset or its own release, so the root
    // layout it reads and the pipeline ExecuteCommandLists looks up still exist. (Keeping them alive while the GPU
    // runs the submission stays the application's job, as in D3D12.)
    std::vector<IUnknown*> held_;

    // A bundle keeps its state calls and draws instead of applying them: it has no render targets, viewport or root
    // signature of its own to resolve them against, and what it sets belongs to the list that executes it.
    // `commands` is how much of the bundle's own list was recorded before the call, so ExecuteBundle can replay the
    // calls and copy the state-free commands (markers, immediate writes) in their original order. Returns whether the
    // call was kept; on any other list it is false and the caller does the work.
    struct BundleCall {
        std::size_t commands;
        std::function<void(GraphicsCommandList&)> run;
    };
    std::vector<BundleCall> calls_;

    // Every deferred call replays through a private member that does not trace: the trace line is written once, when
    // the application makes the call, and not again at each ExecuteBundle. After Close nothing is appended, since
    // another list may be executing the bundle: a state call is ignored, and a draw or dispatch (`command`) makes
    // the list invalid.
    bool defer(std::function<void(GraphicsCommandList&)> run, bool command = false) {
        if (type_ != D3D12_COMMAND_LIST_TYPE_BUNDLE) return false;
        if (list_.closed()) {
            if (command) check(false);
            return true;
        }
        calls_.push_back({list_.commands().size(), std::move(run)});
        return true;
    }

    void hold(IUnknown* object) {
        if (!object || std::find(held_.begin(), held_.end(), object) != held_.end()) return;
        object->AddRef();
        held_.push_back(object);
    }

    void release_held() {
        for (IUnknown* object : held_) object->Release();
        held_.clear();
    }

    void set_pipeline(ID3D12PipelineState* state) {
        hold(state);
        compute_pipeline_ = compute_pipeline_id(state);
        graphics_pipeline_ = graphics_pipeline_id(state);
        if (state && !compute_pipeline_ && !graphics_pipeline_)
            stub("ID3D12GraphicsCommandList::SetPipelineState(pipeline from another device)");
    }

    void set_descriptor_heaps(ID3D12DescriptorHeap* cbv, D3D12_GPU_VIRTUAL_ADDRESS cbv_base,
                              ID3D12DescriptorHeap* smp, D3D12_GPU_VIRTUAL_ADDRESS smp_base) {
        cbv_srv_uav_heap_ = cbv;
        cbv_srv_uav_base_ = cbv_base;
        sampler_heap_ = smp;
        sampler_base_ = smp_base;
    }

    // D3D12 bundle rule: every non-null heap bound by a bundle's SetDescriptorHeaps must match the parent list's
    // currently bound heap of that type, without clearing any omitted heap slot on the parent.
    void verify_bundle_descriptor_heaps(ID3D12DescriptorHeap* cbv, ID3D12DescriptorHeap* smp) {
        if ((cbv && cbv != cbv_srv_uav_heap_) || (smp && smp != sampler_heap_)) check(false);
    }

    void note_direct_heap_indexing(const RootArgumentLayout* layout) const {
        if (!layout) return;
        if (layout->flags & D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED) {
            if (!cbv_srv_uav_heap_)
                note_once("draw/dispatch uses CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED with no CBV/SRV/UAV heap bound");
            else
                note_once("CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED heap base tracked on list (SGPR slot open, section 3.4)");
        }
        if (layout->flags & D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED) {
            if (!sampler_heap_)
                note_once("draw/dispatch uses SAMPLER_HEAP_DIRECTLY_INDEXED with no sampler heap bound");
            else
                note_once("SAMPLER_HEAP_DIRECTLY_INDEXED heap base tracked on list (SGPR slot open, section 3.4)");
        }
    }

    void reset_state(ID3D12PipelineState* state) {
        front_stencil_ref_ = back_stencil_ref_ = 0;
        compute_ = {};
        graphics_ = {};
        cbv_srv_uav_heap_ = nullptr;
        sampler_heap_ = nullptr;
        cbv_srv_uav_base_ = 0;
        sampler_base_ = 0;
        topology_ = 0;
        vertex_buffers_ = {};
        index_ = {};
        viewport_ = {};
        scissor_ = {};
        render_targets_.fill({});
        render_target_count_ = 0;
        draw_views_stale_ = true;
        depth_ = {};
        blend_factor_ = {1, 1, 1, 1};
        set_pipeline(state);
    }

    void set_vertex_buffers(UINT start, UINT count, const D3D12_VERTEX_BUFFER_VIEW* views) {
        if (count) draw_views_stale_ = true;
        for (UINT i = 0; i < count; ++i)
            vertex_buffers_[start + i] = views ? leonida::VertexBufferView{views[i].BufferLocation, views[i].SizeInBytes,
                                                                          views[i].StrideInBytes}
                                               : leonida::VertexBufferView{};
    }

    void set_index_buffer(const D3D12_INDEX_BUFFER_VIEW* view) {
        index_ = view ? leonida::IndexBufferView{view->BufferLocation, view->SizeInBytes, u(view->Format)}
                      : leonida::IndexBufferView{};
    }

    void record_dispatch(UINT x, UINT y, UINT z) {
        if (!compute_pipeline_ || !compute_.layout) return check(false);
        note_direct_heap_indexing(compute_.layout);
        check(list_.dispatch({compute_pipeline_, compute_.data, x, y, z}));
    }

    // The views block a draw records. Rebuilt only when the render targets or the bound vertex buffers change.
    const std::shared_ptr<const leonida::DrawViews>& draw_views() {
        if (draw_views_stale_ || !draw_views_) {
            auto views = std::make_shared<leonida::DrawViews>();
            views->render_targets = render_targets_;
            views->vertex_buffers = vertex_buffers_;
            views->render_target_count = render_target_count_;
            draw_views_ = std::move(views);
            draw_views_stale_ = false;
        }
        return draw_views_;
    }

    // A draw needs a graphics pipeline, a graphics root signature and a topology.
    void record_draw(bool indexed, UINT count, UINT instances, UINT first, INT base_vertex, UINT first_instance) {
        if (!graphics_pipeline_ || !graphics_.layout || !topology_) return check(false);
        note_direct_heap_indexing(graphics_.layout);
        // No viewport (never set, or unbound with RSSetViewports(0)) or an empty one: D3D12 rasterizes nothing,
        // and Vulkan forbids a 0x0 viewport, so the draw is not recorded.
        if (!(viewport_.width > 0) || !(viewport_.height > 0)) return;
        leonida::Draw draw{graphics_pipeline_, graphics_.data, draw_views(), depth_, index_, topology_, viewport_,
                           scissor_, blend_factor_, front_stencil_ref_, back_stencil_ref_, indexed, count, instances,
                           first, base_vertex, first_instance};
        check(list_.draw(std::move(draw)));
    }
};

class Fence final
    : public Child<Fence, ID3D12Fence1Stubs, ID3D12Object, ID3D12DeviceChild, ID3D12Pageable, ID3D12Fence,
                    ID3D12Fence1> {
public:
    // {b4c39e02-6f71-4d8a-b25e-3e9d0f6c1a88}
    static constexpr GUID kTag{0xb4c39e02, 0x6f71, 0x4d8a, {0xb2, 0x5e, 0x3e, 0x9d, 0x0f, 0x6c, 0x1a, 0x88}};

    Fence(leonida::VulkanContext& context, UINT64 initial, D3D12_FENCE_FLAGS flags)
        : context_(context), fence_(context, initial), flags_(flags), highest_(initial) {}

    // As in D3D12, releasing the fence completes the waits still pending on it (ALL counts it as one fence).
    ~Fence() override { context_.cancel_fence_waits(fence_.semaphore()); }

    bool ok() const { return fence_.semaphore() != VK_NULL_HANDLE; }
    VkSemaphore semaphore() const { return fence_.semaphore(); }

    // A Vulkan timeline only increases; D3D12 allows setting a fence lower. Such a signal is reported and dropped.
    // A queue's claim also records the value as one the GPU will signal, which a later CPU signal must not overtake.
    bool claim(UINT64 value, bool queue = false) {
        std::lock_guard lock(claim_mutex_);
        return claim_locked(value, queue);
    }

    UINT64 STDMETHODCALLTYPE GetCompletedValue() override {
        return context_.is_device_lost() ? UINT64_MAX : fence_.completed();
    }
    D3D12_FENCE_FLAGS STDMETHODCALLTYPE GetCreationFlags() override { return flags_; }

    // vkSignalSemaphore must stay below every signal still pending on the GPU
    // (VUID-VkSemaphoreSignalInfo-value-03259). A CPU signal above a queue signal the GPU has not reached yet, or
    // made while an earlier one is held, is held: the fence dispatcher applies it once the GPU passes that queue
    // value, in the order the signals were made. Until then GetCompletedValue reads the GPU's progress, where D3D12
    // would already read `value`.
    HRESULT STDMETHODCALLTYPE Signal(UINT64 value) override {
        std::lock_guard lock(claim_mutex_);
        if (!claim_locked(value, false)) return S_OK;
        const UINT64 queued = queued_;
        if (held_->load() == 0 && fence_.completed() >= queued) return fence_.signal(value) ? S_OK : E_FAIL;
        held_->fetch_add(1);
        // The callback must not touch the fence, which may be released first. Released (cancel) or device lost,
        // the queue value may not be reached, and signalling then would break the same rule: the signal is dropped.
        leonida::VulkanContext* context = &context_;
        const VkSemaphore semaphore = fence_.semaphore();
        context_.enqueue_fence_callback(semaphore, queued, [context, semaphore, queued, value, held = held_] {
            std::uint64_t now = 0;
            bool signaled = false;
            if (context->vk.vkGetSemaphoreCounterValue(context->device(), semaphore, &now) == VK_SUCCESS &&
                now >= queued && now < value) {
                VkSemaphoreSignalInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
                info.semaphore = semaphore;
                info.value = value;
                signaled = context->vk.vkSignalSemaphore(context->device(), &info) == VK_SUCCESS;
            }
            held->fetch_sub(1);
            if (signaled) {
                context->holds().signaled(semaphore, value);
                context->holds().pump_ready();
            }
        });
        return S_OK;
    }

    // A null event blocks until the value is reached, as D3D12 does.
    HRESULT STDMETHODCALLTYPE SetEventOnCompletion(UINT64 value, HANDLE event) override {
        if (context_.is_device_lost() || fence_.completed() >= value) {
            if (event) SetEvent(event);
            return S_OK;
        }
        if (!event) return fence_.wait(value, UINT64_MAX) ? S_OK : E_FAIL;
        context_.enqueue_fence_wait(fence_.semaphore(), value, event);
        return S_OK;
    }

    // Counts `waiter` down once this fence reaches `value`: now if it has, else via the centralized dispatcher.
    void notify(UINT64 value, std::shared_ptr<FenceWaiter> waiter) {
        if (context_.is_device_lost() || fence_.completed() >= value) {
            waiter->hit();
            return;
        }
        context_.enqueue_fence_callback(fence_.semaphore(), value, [w = std::move(waiter)] { w->hit(); });
    }

    static Fence* from(IUnknown* object) { return Child::from<Fence>(object); }

private:
    bool claim_locked(UINT64 value, bool queue) {
        if (value <= highest_) {
            std::fprintf(stderr, "[rt] d3d12: fence signal %llu is not above %llu; Vulkan timelines cannot go back\n",
                         static_cast<unsigned long long>(value), static_cast<unsigned long long>(highest_));
            return false;
        }
        highest_ = value;
        if (queue) queued_ = value;
        return true;
    }

    leonida::VulkanContext& context_;
    leonida::VulkanFence fence_;
    D3D12_FENCE_FLAGS flags_;
    std::mutex claim_mutex_;
    UINT64 highest_;     // the highest value claimed, from either side
    UINT64 queued_ = 0;  // the highest value a queue will signal
    std::shared_ptr<std::atomic<int>> held_ = std::make_shared<std::atomic<int>>(0);  // CPU signals not applied yet
};

class CommandQueue final
    : public Child<CommandQueue, ID3D12CommandQueueStubs, ID3D12Object, ID3D12DeviceChild, ID3D12Pageable,
                    ID3D12CommandQueue> {
public:
    // {e1a85d76-0b3f-4c29-a7d4-58f2c6e0b913}
    static constexpr GUID kTag{0xe1a85d76, 0x0b3f, 0x4c29, {0xa7, 0xd4, 0x58, 0xf2, 0xc6, 0xe0, 0xb9, 0x13}};

    // What a held batch (queue_holds.h) keeps alive until it has run: the pipeline states and root signatures its
    // lists name, which the application may release and Reset away as soon as ExecuteCommandLists returns. Resources
    // are not held (a D3D12 application keeps them until the fence it signals; the docs, section 4.4, list this).
    static std::shared_ptr<void> pin_lists(const std::vector<const GraphicsCommandList*>& lists) {
        auto objects = std::shared_ptr<std::vector<IUnknown*>>(new std::vector<IUnknown*>, [](auto* held) {
            for (IUnknown* object : *held) object->Release();
            delete held;
        });
        for (const GraphicsCommandList* list : lists)
            for (IUnknown* object : list->pinned()) {
                object->AddRef();
                objects->push_back(object);
            }
        return objects;
    }

    // A D3D12 queue holds the fences it waits on and signals until the work has been submitted.
    static leonida::VulkanQueue::PinFn fence_pin(ID3D12Fence* fence) {
        return [fence]() -> std::shared_ptr<void> {
            fence->AddRef();
            return std::shared_ptr<void>(fence, [](void* held) { static_cast<ID3D12Fence*>(held)->Release(); });
        };
    }

    static leonida::VulkanQueue::QueueKind queue_kind_for(D3D12_COMMAND_LIST_TYPE type) {
        if (type == D3D12_COMMAND_LIST_TYPE_COPY) return leonida::VulkanQueue::QueueKind::copy;
        if (type == D3D12_COMMAND_LIST_TYPE_COMPUTE) return leonida::VulkanQueue::QueueKind::compute;
        return leonida::VulkanQueue::QueueKind::direct;
    }

    CommandQueue(leonida::VulkanContext& context, VkQueue queue, std::uint32_t family,
                 const D3D12_COMMAND_QUEUE_DESC& desc, const QueueClock& clock, std::shared_ptr<void> queue_lease)
        : desc_(desc), context_(context), queue_lease_(std::move(queue_lease)),
          queue_(context, queue, family, queue_kind_for(desc.Type)),
          vk_queue_(queue), family_(family), clock_(clock) {}

    ~CommandQueue() override {
        if (!stamp_.pool) return;
        const VkDevice device = context_.device();
        stamp_.destroy_query_pool(device, stamp_.queries, nullptr);
        context_.vk.vkDestroyCommandPool(device, stamp_.pool, nullptr);
    }

    bool ok() const { return queue_.ok(); }
    leonida::ReplayResult last_result() {
        // A batch the queue held fails when it is released, after the call returned: the failure is the result.
        if (const auto held = queue_.take_held_failure(); held != leonida::ReplayResult::success) last_result_ = held;
        return last_result_;
    }

    D3D12_COMMAND_QUEUE_DESC STDMETHODCALLTYPE GetDesc() override {
        d3d12_trace::record("ID3D12CommandQueue::GetDesc");
        D3D12_COMMAND_QUEUE_DESC desc = desc_;
        if (!desc.NodeMask) desc.NodeMask = 1;  // node mask 0 reads back as the one node, as on the host
        return desc;
    }

    void STDMETHODCALLTYPE ExecuteCommandLists(UINT count, ID3D12CommandList* const* lists) override {
        constexpr auto method = "ID3D12CommandQueue::ExecuteCommandLists";
        std::lock_guard lock(mutex_);
        if (count > 0 && !lists) {
            note_once("ExecuteCommandLists: null command list array with non-zero count");
            last_result_ = leonida::ReplayResult::unsupported;
            d3d12_trace::record(method, "failed");
            return;
        }
        std::vector<const leonida::CommandList*> recorded;
        std::vector<const GraphicsCommandList*> graphics;
        for (UINT i = 0; i < count; ++i) {
            const GraphicsCommandList* list = GraphicsCommandList::from(lists[i]);
            if (!list || list->type() == D3D12_COMMAND_LIST_TYPE_BUNDLE) {
                std::fprintf(stderr, "[rt] d3d12: ExecuteCommandLists: list %u is not a direct command list\n", i);
                last_result_ = leonida::ReplayResult::unsupported;
                d3d12_trace::record(method, "failed");
                return;
            }
            // A list runs only on a queue of its own type (the debug layer's EXECUTECOMMANDLISTS_COMMANDLISTMISMATCH,
            // message 836): a DIRECT list's draws would otherwise be recorded for a COPY queue's transfer-only
            // Vulkan family. The whole batch is refused, as for the other checks here.
            if (list->type() != desc_.Type) {
                std::fprintf(stderr, "[rt] d3d12: ExecuteCommandLists: list %u has type %d, the queue type %d\n", i,
                             int(list->type()), int(desc_.Type));
                last_result_ = leonida::ReplayResult::unsupported;
                d3d12_trace::record(method, "failed");
                return;
            }
            // A list whose recording failed (Close returned E_INVALIDARG) must not replay its earlier commands.
            if (list->invalid()) {
                std::fprintf(stderr, "[rt] d3d12: ExecuteCommandLists: list %u is invalid; refusing batch\n", i);
                last_result_ = leonida::ReplayResult::unsupported;
                d3d12_trace::record(method, "failed");
                return;
            }
            if (!timestamps_fit(*list)) {
                last_result_ = leonida::ReplayResult::invalid_query;
                d3d12_trace::record(method, "failed");
                return;
            }
            recorded.push_back(&list->commands());
            graphics.push_back(list);
        }
        last_result_ = queue_.execute(recorded, [&graphics] { return pin_lists(graphics); });
        d3d12_trace::record(method, last_result_ == leonida::ReplayResult::success ? "ok" : "failed");
        if (last_result_ != leonida::ReplayResult::success)
            std::fprintf(stderr, "[rt] d3d12: ExecuteCommandLists failed: %s; nothing was submitted\n",
                         name(last_result_));
    }

    // A copy list's timestamps go to COPY_QUEUE_TIMESTAMP heaps and every other list's to TIMESTAMP heaps.
    bool timestamps_fit(const GraphicsCommandList& list) {
        namespace h = leonida::d3d12_query_heap;
        const auto other = list.type() == D3D12_COMMAND_LIST_TYPE_COPY ? h::timestamp : h::copy_queue_timestamp;
        for (const auto& command : list.commands().commands()) {
            const auto range = leonida::query_range(command);
            if (range.count && range.type == leonida::d3d12_query::timestamp &&
                context_.query_heap_type(range.heap) == other)
                return false;
        }
        return true;
    }

    void STDMETHODCALLTYPE UpdateTileMappings(ID3D12Resource* resource, UINT region_count,
                                              const D3D12_TILED_RESOURCE_COORDINATE* starts,
                                              const D3D12_TILE_REGION_SIZE* sizes, ID3D12Heap* heap, UINT range_count,
                                              const D3D12_TILE_RANGE_FLAGS* flags, const UINT* offsets,
                                              const UINT* counts, D3D12_TILE_MAPPING_FLAGS mapping_flags) override {
        constexpr auto method = "ID3D12CommandQueue::UpdateTileMappings";
        if (!update_tiles(resource, region_count, starts, sizes, heap, range_count, flags, offsets, counts,
                          mapping_flags)) {
            d3d12_trace::record(method, "refused");
            note_once("ID3D12CommandQueue::UpdateTileMappings: invalid mapping or sparse residency unavailable");
        } else {
            d3d12_trace::record(method, "ok");
        }
    }

    void STDMETHODCALLTYPE CopyTileMappings(ID3D12Resource* dst, const D3D12_TILED_RESOURCE_COORDINATE* dst_start,
                                            ID3D12Resource* src, const D3D12_TILED_RESOURCE_COORDINATE* src_start,
                                            const D3D12_TILE_REGION_SIZE* size, D3D12_TILE_MAPPING_FLAGS flags) override {
        constexpr auto method = "ID3D12CommandQueue::CopyTileMappings";
        std::lock_guard lock(mutex_);
        std::vector<leonida::TileRegion> from, to;
        if ((flags & ~D3D12_TILE_MAPPING_FLAG_NO_HAZARD) || !dst_start || !src_start || !size ||
            !append_tile_regions(src_start, size, tile_grid(src), from) ||
            !append_tile_regions(dst_start, size, tile_grid(dst), to) ||
            !queue_.copy_tile_mappings(resource_id(dst), to, resource_id(src), from)) {
            d3d12_trace::record(method, "refused");
            note_once("ID3D12CommandQueue::CopyTileMappings: invalid mapping or sparse residency unavailable");
        } else {
            d3d12_trace::record(method, "ok");
        }
    }

    HRESULT STDMETHODCALLTYPE Signal(ID3D12Fence* fence, UINT64 value) override {
        constexpr auto method = "ID3D12CommandQueue::Signal";
        Fence* f = Fence::from(fence);
        if (!f) return d3d12_trace::result(method, E_INVALIDARG);
        std::lock_guard lock(mutex_);
        if (!f->claim(value, true)) return d3d12_trace::result(method, S_OK);
        return d3d12_trace::result(method, queue_.signal(f->semaphore(), value, fence_pin(fence)) ? S_OK : E_FAIL);
    }

    HRESULT STDMETHODCALLTYPE Wait(ID3D12Fence* fence, UINT64 value) override {
        constexpr auto method = "ID3D12CommandQueue::Wait";
        Fence* f = Fence::from(fence);
        if (!f) return d3d12_trace::result(method, E_INVALIDARG);
        std::lock_guard lock(mutex_);
        queue_.wait(f->semaphore(), value, fence_pin(fence));
        return d3d12_trace::result(method, S_OK);
    }

    // Ticks per second of the timestamps EndQuery writes: one tick is timestampPeriod nanoseconds. Every D3D12
    // queue of the device shares one VkQueue, so they share one frequency. E_FAIL when that queue has no
    // timestamps (timestampValidBits 0), as D3D12 reports for a queue that does not support them.
    HRESULT STDMETHODCALLTYPE GetTimestampFrequency(UINT64* frequency) override {
        if (!frequency) return E_INVALIDARG;
        const double period = context_.timestamp_period();
        if (!(period > 0)) return E_FAIL;
        *frequency = static_cast<UINT64>(1e9 / period + 0.5);
        return S_OK;
    }

    // Queue labels, in order with the submissions around them.
    void STDMETHODCALLTYPE SetMarker(UINT metadata, const void* data, UINT size) override {
        d3d12_trace::record("ID3D12CommandQueue::SetMarker");
        marker(leonida::DebugMarker::Kind::set, metadata, data, size);
    }
    void STDMETHODCALLTYPE BeginEvent(UINT metadata, const void* data, UINT size) override {
        d3d12_trace::record("ID3D12CommandQueue::BeginEvent");
        marker(leonida::DebugMarker::Kind::begin, metadata, data, size);
    }
    void STDMETHODCALLTYPE EndEvent() override {
        d3d12_trace::record("ID3D12CommandQueue::EndEvent");
        marker(leonida::DebugMarker::Kind::end, 0, nullptr, 0);
    }

    // The GPU timestamp is in device ticks (what GetTimestampFrequency counts: 1e9 / timestampPeriod per second)
    // and the CPU one is QueryPerformanceCounter's, both read at about the same moment.
    HRESULT STDMETHODCALLTYPE GetClockCalibration(UINT64* gpu, UINT64* cpu) override {
        if (!gpu || !cpu) return E_INVALIDARG;
        if (clock_.calibrated) {
            VkCalibratedTimestampInfoKHR domains[2]{{VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR},
                                                    {VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR}};
            domains[0].timeDomain = VK_TIME_DOMAIN_DEVICE_KHR;
            domains[1].timeDomain = VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_KHR;
            std::uint64_t stamps[2]{}, deviation = 0;
            LARGE_INTEGER before{}, after{};
            QueryPerformanceCounter(&before);
            if (clock_.calibrated(context_.device(), 2, domains, stamps, &deviation) == VK_SUCCESS) {
                QueryPerformanceCounter(&after);
                // Wine can expose a host monotonic clock that differs from its Win32 QPC clock.
                // Keep the GPU and CPU samples paired by taking a fresh GPU sample on fallback.
                if (stamps[1] >= UINT64(before.QuadPart) && stamps[1] <= UINT64(after.QuadPart)) {
                    *gpu = stamps[0];
                    *cpu = stamps[1];
                    return S_OK;
                }
            }
        }
        return submitted_calibration(*gpu, *cpu);
    }

    bool uses_calibrated_timestamps() const { return clock_.calibrated != nullptr; }

    static CommandQueue* from(IUnknown* object) { return Child::from<CommandQueue>(object); }

private:
    // The fallback without calibrated timestamps: wait for the queue to go idle, then submit a timestamp write and
    // wait for it, with QueryPerformanceCounter read just before the submission and just after it completes. The
    // GPU timestamp falls between the two; the midpoint is the answer, off by at most half the round trip (tens of
    // microseconds). It stalls the queue, which is acceptable for a call profilers make once per capture.
    HRESULT submitted_calibration(UINT64& gpu, UINT64& cpu) {
        std::lock_guard lock(mutex_);
        if (!clock_.timestamp_valid_bits || !clock_.get_device_proc || !create_stamp()) return E_FAIL;
        auto& vk = context_.vk;
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vk.vkBeginCommandBuffer(stamp_.commands, &begin) != VK_SUCCESS) return E_FAIL;
        stamp_.reset_query_pool(stamp_.commands, stamp_.queries, 0, 1);
        stamp_.write_timestamp(stamp_.commands, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, stamp_.queries, 0);
        if (vk.vkEndCommandBuffer(stamp_.commands) != VK_SUCCESS) return E_FAIL;
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &stamp_.commands;
        LARGE_INTEGER before{}, after{};
        {
            std::lock_guard submit_lock(context_.submit_mutex());
            if (vk.vkQueueWaitIdle(vk_queue_) != VK_SUCCESS) return E_FAIL;
            QueryPerformanceCounter(&before);
            if (vk.vkQueueSubmit(vk_queue_, 1, &submit, VK_NULL_HANDLE) != VK_SUCCESS ||
                vk.vkQueueWaitIdle(vk_queue_) != VK_SUCCESS)
                return E_FAIL;
            QueryPerformanceCounter(&after);
        }
        std::uint64_t stamp = 0;
        if (stamp_.get_query_pool_results(context_.device(), stamp_.queries, 0, 1, sizeof(stamp), &stamp,
                                          sizeof(stamp), VK_QUERY_RESULT_64_BIT) != VK_SUCCESS)
            return E_FAIL;
        if (clock_.timestamp_valid_bits < 64) stamp &= (std::uint64_t(1) << clock_.timestamp_valid_bits) - 1;
        gpu = stamp;
        cpu = UINT64(before.QuadPart) + UINT64(after.QuadPart - before.QuadPart) / 2;
        return S_OK;
    }

    // The fallback's command buffer and one-query timestamp pool, made on first use. Called with mutex_ held.
    bool create_stamp() {
        if (stamp_.pool) return true;
        const VkDevice device = context_.device();
        const auto get = clock_.get_device_proc;
        stamp_.create_query_pool = reinterpret_cast<PFN_vkCreateQueryPool>(get(device, "vkCreateQueryPool"));
        stamp_.destroy_query_pool = reinterpret_cast<PFN_vkDestroyQueryPool>(get(device, "vkDestroyQueryPool"));
        stamp_.reset_query_pool = reinterpret_cast<PFN_vkCmdResetQueryPool>(get(device, "vkCmdResetQueryPool"));
        stamp_.write_timestamp = reinterpret_cast<PFN_vkCmdWriteTimestamp>(get(device, "vkCmdWriteTimestamp"));
        stamp_.get_query_pool_results =
            reinterpret_cast<PFN_vkGetQueryPoolResults>(get(device, "vkGetQueryPoolResults"));
        if (!stamp_.create_query_pool || !stamp_.destroy_query_pool || !stamp_.reset_query_pool ||
            !stamp_.write_timestamp || !stamp_.get_query_pool_results)
            return false;
        auto& vk = context_.vk;
        VkQueryPoolCreateInfo queries{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        queries.queryType = VK_QUERY_TYPE_TIMESTAMP;
        queries.queryCount = 1;
        if (stamp_.create_query_pool(device, &queries, nullptr, &stamp_.queries) != VK_SUCCESS) return false;
        VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool.queueFamilyIndex = family_;
        VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocate.commandBufferCount = 1;
        if (vk.vkCreateCommandPool(device, &pool, nullptr, &stamp_.pool) == VK_SUCCESS) {
            allocate.commandPool = stamp_.pool;
            if (vk.vkAllocateCommandBuffers(device, &allocate, &stamp_.commands) == VK_SUCCESS) return true;
            vk.vkDestroyCommandPool(device, stamp_.pool, nullptr);
            stamp_.pool = VK_NULL_HANDLE;
        }
        stamp_.destroy_query_pool(device, stamp_.queries, nullptr);
        stamp_.queries = VK_NULL_HANDLE;
        return false;
    }

    void marker(leonida::DebugMarker::Kind kind, UINT metadata, const void* data, UINT size) {
        const std::string label = event_label(metadata, data, size);
        std::lock_guard lock(mutex_);
        queue_.marker(kind, label.c_str());
    }

    bool update_tiles(ID3D12Resource* resource, UINT region_count, const D3D12_TILED_RESOURCE_COORDINATE* starts,
                      const D3D12_TILE_REGION_SIZE* sizes, ID3D12Heap* heap, UINT range_count,
                      const D3D12_TILE_RANGE_FLAGS* flags, const UINT* offsets, const UINT* counts,
                      D3D12_TILE_MAPPING_FLAGS mapping_flags) {
        if (!region_count || !range_count || (!starts && region_count != 1) ||
            (mapping_flags & ~D3D12_TILE_MAPPING_FLAG_NO_HAZARD)) return false;
        const auto grid = tile_grid(resource);
        std::vector<leonida::TileRegion> regions;
        regions.reserve(region_count);
        for (UINT i = 0; i < region_count; ++i) {
            if (!append_tile_regions(starts ? starts + i : nullptr, sizes ? sizes + i : nullptr, grid, regions))
                return false;
        }
        std::uint64_t total = 0;
        for (const auto& region : regions) {
            if (region.count > UINT64_MAX - total) return false;
            total += region.count;
        }

        std::vector<leonida::TileRange> ranges;
        ranges.reserve(range_count);
        std::uint64_t remaining = total, heap_tiles_required = 0;
        for (UINT i = 0; i < range_count; ++i) {
            const auto count = counts ? counts[i] : (range_count == 1 ? total : 1);
            if (!count) continue;
            if (count > remaining) return false;
            remaining -= count;
            using leonida::TileRangeMode;
            const auto flag = flags ? flags[i] : D3D12_TILE_RANGE_FLAG_NONE;
            switch (flag) {
            case D3D12_TILE_RANGE_FLAG_NONE:
            case D3D12_TILE_RANGE_FLAG_REUSE_SINGLE_TILE: {
                if (!offsets) return false;
                const auto used = flag == D3D12_TILE_RANGE_FLAG_NONE ? count : 1;
                if (used > UINT64_MAX - offsets[i]) return false;
                heap_tiles_required = std::max(heap_tiles_required, offsets[i] + used);
                ranges.push_back({flag == D3D12_TILE_RANGE_FLAG_NONE ? TileRangeMode::sequential : TileRangeMode::reuse,
                                  offsets[i], count});
                break;
            }
            case D3D12_TILE_RANGE_FLAG_NULL:
                ranges.push_back({TileRangeMode::unmap, 0, count});
                break;
            case D3D12_TILE_RANGE_FLAG_SKIP:
                ranges.push_back({TileRangeMode::skip, 0, count});
                break;
            default:
                return false;
            }
        }
        if (remaining || (heap_tiles_required &&
            (!heap || heap_tiles_required > heap->GetDesc().SizeInBytes / leonida::tile_size_in_bytes))) return false;
        std::lock_guard lock(mutex_);
        return queue_.update_tiles(resource_id(resource), regions, ranges,
                                   [&](bool image, std::uint32_t types) {
                                       return heap_tiles_required ? tile_heap(heap, context_, image, types) : nullptr;
                                   });
    }

    D3D12_COMMAND_QUEUE_DESC desc_;
    leonida::VulkanContext& context_;
    std::mutex mutex_;
    // Destroyed after queue_ has waited idle, so the handle cannot be reused prematurely.
    std::shared_ptr<void> queue_lease_;
    leonida::VulkanQueue queue_;
    leonida::ReplayResult last_result_ = leonida::ReplayResult::success;
    VkQueue vk_queue_;
    std::uint32_t family_;
    QueueClock clock_;
    struct {
        PFN_vkCreateQueryPool create_query_pool = nullptr;
        PFN_vkDestroyQueryPool destroy_query_pool = nullptr;
        PFN_vkCmdResetQueryPool reset_query_pool = nullptr;
        PFN_vkCmdWriteTimestamp write_timestamp = nullptr;
        PFN_vkGetQueryPoolResults get_query_pool_results = nullptr;
        VkQueryPool queries = VK_NULL_HANDLE;
        VkCommandPool pool = VK_NULL_HANDLE;
        VkCommandBuffer commands = VK_NULL_HANDLE;
    } stamp_;
};

}  // namespace

HRESULT create_command_queue(leonida::VulkanContext& context, VkQueue queue, std::uint32_t queue_family,
                             const D3D12_COMMAND_QUEUE_DESC& desc, REFIID riid, void** out, IUnknown* device,
                             const QueueClock& clock, std::shared_ptr<void> queue_lease) {
    if (!out) return E_POINTER;
    *out = nullptr;
    auto* object = new CommandQueue(context, queue, queue_family, desc, clock, std::move(queue_lease));
    if (!object->ok()) {
        object->Release();
        return E_FAIL;
    }
    object->attach_device(device);
    return hand_out(object, riid, out);
}

HRESULT create_command_allocator(D3D12_COMMAND_LIST_TYPE type, REFIID riid, void** out, IUnknown* device) {
    // The host runtime's order (d3d12_vulkan_device_test, compare_invalid_arguments "command allocator:"): a type
    // past VIDEO_ENCODE is E_INVALIDARG, then a null output is E_UNEXPECTED.
    if (out) *out = nullptr;
    if (UINT(type) > UINT(D3D12_COMMAND_LIST_TYPE_VIDEO_ENCODE)) return E_INVALIDARG;
    if (!out) return E_UNEXPECTED;
    auto* object = new CommandAllocator(type);
    object->attach_device(device);
    return hand_out(object, riid, out);
}

HRESULT create_command_list(D3D12_COMMAND_LIST_TYPE type, ID3D12CommandAllocator* allocator, REFIID riid,
                            void** out, IUnknown* device, ID3D12PipelineState* initial_state) {
    if (out) *out = nullptr;
    CommandAllocator* command_allocator = CommandAllocator::from(allocator);
    if (!command_allocator || command_allocator->type() != type) return E_INVALIDARG;
    // A list is created open, so the allocator must be free: E_INVALIDARG as on the PC runtime (D3D-51).
    if (command_allocator->recording()) return E_INVALIDARG;
    // A null output only after the arguments, as the host runtime answers ("command list:" cases).
    if (!out) return E_UNEXPECTED;
    if (type != D3D12_COMMAND_LIST_TYPE_DIRECT && type != D3D12_COMMAND_LIST_TYPE_BUNDLE &&
        type != D3D12_COMMAND_LIST_TYPE_COMPUTE && type != D3D12_COMMAND_LIST_TYPE_COPY)
        stub("create_command_list(non-direct type)");
    auto* object = new GraphicsCommandList(type);  // created open, as D3D12 does
    object->record_into(command_allocator);
    object->attach_device(device);
    const HRESULT hr = hand_out(object, riid, out);  // releases the creation reference, so a failure frees the list
    // The initial pipeline state only once the list exists for the caller, as SetPipelineState on it would be.
    if (SUCCEEDED(hr) && initial_state)
        if (GraphicsCommandList* list = GraphicsCommandList::from(static_cast<IUnknown*>(*out)))
            list->set_initial_pipeline(initial_state);
    return hr;
}

HRESULT create_fence(leonida::VulkanContext& context, UINT64 initial_value, D3D12_FENCE_FLAGS flags, REFIID riid,
                     void** out, IUnknown* device) {
    if (!out) return E_POINTER;
    *out = nullptr;
    context.set_signal_event_fn([](void* h) { SetEvent(static_cast<HANDLE>(h)); });
    auto* object = new Fence(context, initial_value, flags);
    if (!object->ok()) {
        object->Release();
        return E_FAIL;
    }
    object->attach_device(device);
    return hand_out(object, riid, out);
}

HRESULT set_event_on_multiple_fence_completion(leonida::VulkanContext& context, ID3D12Fence* const* fences,
                                               const UINT64* values, UINT count,
                                               D3D12_MULTIPLE_FENCE_WAIT_FLAGS flags, HANDLE event) {
    if (flags != D3D12_MULTIPLE_FENCE_WAIT_FLAG_ALL && flags != D3D12_MULTIPLE_FENCE_WAIT_FLAG_ANY)
        return E_INVALIDARG;
    if (!count || !fences || !values) return E_INVALIDARG;  // no fences is refused, as on the host runtime
    std::vector<Fence*> objects(count);
    for (UINT i = 0; i < count; ++i)
        if (!(objects[i] = Fence::from(fences[i]))) return E_INVALIDARG;
    const bool any = flags == D3D12_MULTIPLE_FENCE_WAIT_FLAG_ANY;
    if (!event) {
        // One vkWaitSemaphores call: VK_SEMAPHORE_WAIT_ANY_BIT is the ANY flag, no flag is ALL.
        std::vector<VkSemaphore> semaphores(count);
        for (UINT i = 0; i < count; ++i) semaphores[i] = objects[i]->semaphore();
        VkSemaphoreWaitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
        wait.flags = any ? VK_SEMAPHORE_WAIT_ANY_BIT : 0;
        wait.semaphoreCount = count;
        wait.pSemaphores = semaphores.data();
        wait.pValues = values;
        const VkResult result = context.vk.vkWaitSemaphores(context.device(), &wait, UINT64_MAX);
        // Unlike a single fence's blocking SetEventOnCompletion, this path invokes Vulkan directly.
        // A lost device must wake its other registered fence waiters and reject later submissions.
        if (result == VK_ERROR_DEVICE_LOST) context.mark_device_lost();
        return result == VK_SUCCESS ? S_OK : E_FAIL;
    }
    // Each fence's own waiter thread counts the shared waiter down; no thread of its own.
    const auto waiter = std::make_shared<FenceWaiter>(event, any ? 1u : count);
    for (UINT i = 0; i < count && !waiter->done(); ++i) objects[i]->notify(values[i], waiter);
    return S_OK;
}

bool queue_uses_calibrated_timestamps(ID3D12CommandQueue* queue) {
    if (ID3D12CommandQueue* real = d3d12_xs::queue_proxy_real_queue(queue)) queue = real;  // the Xbox-ABI proxy
    const CommandQueue* q = CommandQueue::from(queue);
    return q && q->uses_calibrated_timestamps();
}

HRESULT create_command_signature(leonida::IndirectKind kind, UINT byte_stride, REFIID riid, void** out,
                                 IUnknown* device) {
    if (!out) return E_POINTER;
    *out = nullptr;
    auto* sig = new CommandSignature(kind, byte_stride);
    sig->attach_device(device);
    return hand_out(sig, riid, out);
}

bool command_signature_info(ID3D12CommandSignature* signature, leonida::IndirectKind& kind, UINT& byte_stride) {
    const CommandSignature* s = CommandSignature::from(signature);
    if (!s) return false;
    kind = s->kind();
    byte_stride = s->stride();
    return true;
}

leonida::ReplayResult last_execute_result(ID3D12CommandQueue* queue) {
    if (ID3D12CommandQueue* real = d3d12_xs::queue_proxy_real_queue(queue)) queue = real;  // the Xbox-ABI proxy
    CommandQueue* q = CommandQueue::from(queue);
    return q ? q->last_result() : leonida::ReplayResult::unsupported;
}

bool front_back_stencil_ref(ID3D12GraphicsCommandList8* list, UINT& front, UINT& back) {
    const GraphicsCommandList* l = GraphicsCommandList::from(list);
    if (!l) return false;
    front = l->front_stencil_ref();
    back = l->back_stencil_ref();
    return true;
}

bool recorded_commands(ID3D12GraphicsCommandList* list, std::vector<leonida::Command>& commands) {
    const GraphicsCommandList* l = GraphicsCommandList::from(list);
    if (!l) return false;
    const auto recorded = l->commands().commands();
    commands.assign(recorded.begin(), recorded.end());
    return true;
}

bool bound_descriptor_heap_bases(ID3D12GraphicsCommandList* list, D3D12_GPU_VIRTUAL_ADDRESS& cbv_srv_uav,
                                 D3D12_GPU_VIRTUAL_ADDRESS& sampler) {
    const GraphicsCommandList* l = GraphicsCommandList::from(list);
    if (!l) return false;
    cbv_srv_uav = l->cbv_srv_uav_base();
    sampler = l->sampler_base();
    return true;
}

}  // namespace rt::d3d12
