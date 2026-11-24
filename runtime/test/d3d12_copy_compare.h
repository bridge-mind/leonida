// D3D-46: copies compared with the host PC runtime (or WARP). Included by d3d12_vulkan_device_test.cpp.
// The same scenario runs on the layer's device and on a host D3D12 device: textures are filled from an upload
// buffer with deterministic bytes, copied with CopyResource / CopyTextureRegion (whole subresources, boxes,
// destination offsets, other array slices, placed footprints in both directions) and CopyBufferRegion, read back
// into a readback buffer prefilled with 0xcd, and the bytes are compared. Only the texels a footprint row holds are
// compared (row padding is the driver's); everything else, including the 0xcd around a partial copy, must match.
// Layouts come from the host's GetCopyableFootprints for both sides (those are compared separately).
// LEONIDA_HOST_ADAPTER picks the host side: "warp", or part of an adapter name; unset is the layer's own adapter when
// the test has set layer_adapter(), else the default adapter.
#pragma once
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "../shims/d3d12_com.h"
#include "host_adapter.h"

namespace copycmp {

template<class T>
struct Ptr {
    T* p = nullptr;
    Ptr() = default;
    Ptr(const Ptr&) = delete;
    ~Ptr() { reset(); }
    void reset() {
        if (p) p->Release();
        p = nullptr;
    }
    T* operator->() const { return p; }
};

// True under Wine, where there is no host to compare with: the comparisons take Microsoft's d3d12.dll as the
// reference, and Wine's (vkd3d) lays out depth-stencil formats in one plane, rejects R9G9B9E5 and BC footprints,
// accepts descriptions Microsoft's runtime refuses and answers shader model 5.1. Its device also shares the process's
// Vulkan with ours (winevulkan asserted in vkBindBufferMemory). Says so once.
inline bool host_is_wine() {
    static const bool wine = GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "wine_get_version") != nullptr;
    static bool said = false;
    if (wine && !said) std::printf("host D3D12 is Wine's, not Microsoft's: comparisons with the host are skipped\n");
    said = true;
    return wine;
}

// An explicit adapter LUID takes precedence over LEONIDA_HOST_ADAPTER for same-adapter comparisons.
inline bool try_create_host(Ptr<ID3D12Device>& host, std::string& name, const LUID* luid = nullptr) {
    if (host_is_wine()) return false;
    const HMODULE d3d12 = LoadLibraryExA("d3d12.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    const HMODULE dxgi = LoadLibraryExA("dxgi.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!d3d12 || !dxgi) return false;
    const auto create = reinterpret_cast<PFN_D3D12_CREATE_DEVICE>(
        reinterpret_cast<void*>(GetProcAddress(d3d12, "D3D12CreateDevice")));
    using CreateFactory = HRESULT(WINAPI*)(REFIID, void**);
    const auto create_factory =
        reinterpret_cast<CreateFactory>(reinterpret_cast<void*>(GetProcAddress(dxgi, "CreateDXGIFactory1")));
    if (!create || !create_factory) return false;
    Ptr<IDXGIFactory4> factory;
    if (FAILED(create_factory(IID_PPV_ARGS(&factory.p)))) return false;
    // The variable and the name match are host_adapter.h's, shared with the tests that create their host devices
    // themselves.
    const std::string needle = hostadapter::wanted();
    Ptr<IDXGIAdapter1> adapter;
    if (luid) {
        if (FAILED(factory->EnumAdapterByLuid(*luid, IID_PPV_ARGS(&adapter.p)))) return false;
    } else if (needle == "warp") {
        if (FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter.p)))) return false;
    } else if (!needle.empty()) {
        adapter.p = hostadapter::named(factory.p, needle);
        if (!adapter.p) return false;
    }
    if (FAILED(create(adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p)))) return false;
    Ptr<IDXGIAdapter1> used;
    if (adapter.p) {
        used.p = adapter.p;
        used.p->AddRef();
    } else {
        Ptr<IDXGIAdapter> base;
        if (SUCCEEDED(factory->EnumAdapterByLuid(host->GetAdapterLuid(), IID_PPV_ARGS(&base.p))))
            base->QueryInterface(IID_PPV_ARGS(&used.p));
    }
    name = "unknown adapter";
    if (used.p) {
        DXGI_ADAPTER_DESC1 desc{};
        used->GetDesc1(&desc);
        name.clear();
        for (const wchar_t* c = desc.Description; *c; ++c) name += char(*c < 128 ? *c : '?');
    }
    return true;
}

// The adapter behind the layer's Vulkan device (VkPhysicalDeviceIDProperties::deviceLUID), for a test that knows it.
// On a PC with two GPUs D3D12's default adapter can be the other one, whose driver accepts other formats (an AMD
// layer device accepts an R9G9B9E5 render target that the NVIDIA host refuses), so the comparison would be between
// two GPUs instead of between the layer and the host runtime.
inline std::optional<LUID>& layer_adapter() {
    static std::optional<LUID> luid;
    return luid;
}

// A host device, or false when there is none (the comparisons are then skipped). Without LEONIDA_HOST_ADAPTER it is
// on the layer's own adapter when that is set and has a D3D12 device, else on the default adapter. When
// LEONIDA_HOST_ADAPTER names an adapter that cannot be had, the run fails: a typo must not turn the comparisons into a
// silent pass.
inline bool create_host(Ptr<ID3D12Device>& host, std::string& name) {
    const std::string needle = hostadapter::wanted();
    const bool named = !needle.empty();
    if (!named && layer_adapter() && try_create_host(host, name, &*layer_adapter())) return true;
    if (try_create_host(host, name)) return true;
    if (named) {
        std::fprintf(stderr, "LEONIDA_HOST_ADAPTER=%s: no such adapter, or no D3D12 device on it\n", needle.c_str());
        std::exit(1);
    }
    return false;
}

struct Region {
    std::string label;
    size_t offset = 0, pitch = 0, row_bytes = 0, rows = 0, depth = 1;
    bool x8 = false;  // D24 depth plane: every fourth byte is the unspecified X8 (the host writes 0, Vulkan leaves it)
};

struct Outcome {
    std::vector<std::uint8_t> bytes;
    std::vector<Region> regions;
    std::string error;
};

struct TexCase {
    const char* name;
    D3D12_RESOURCE_DIMENSION dim;
    DXGI_FORMAT format;
    UINT width, height, depth_or_layers;
    UINT16 mips;
    bool floaty;  // bytes stay small so floats are finite
    bool ds;      // depth-stencil: whole-subresource copies only
    bool bc;
};

// One device with a direct queue, an allocator, a list and a fence.
struct Run {
    ID3D12Device* device = nullptr;
    bool ours = false;
    Ptr<ID3D12CommandQueue> queue;
    Ptr<ID3D12CommandAllocator> allocator;
    Ptr<ID3D12GraphicsCommandList> list;
    Ptr<ID3D12Fence> fence;
    UINT64 value = 0;
    bool refused = false;  // the last submit failed at Close or in the layer's replay: the list was rejected

    bool init(ID3D12Device* d, bool is_ours) {
        device = d;
        ours = is_ours;
        D3D12_COMMAND_QUEUE_DESC qd{};
        qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        const auto check_setup = [is_ours](const char* call, HRESULT hr) {
            if (SUCCEEDED(hr)) return true;
            std::fprintf(stderr, "copy setup: %s %s failed (HRESULT 0x%08x)\n",
                         is_ours ? "Vulkan" : "host", call, static_cast<unsigned>(hr));
            return false;
        };
        return check_setup("CreateCommandQueue", d->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue.p))) &&
               check_setup("CreateCommandAllocator",
                           d->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))) &&
               check_setup("CreateCommandList", d->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p,
                                                                     nullptr, IID_PPV_ARGS(&list.p))) &&
               check_setup("CreateFence", d->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p)));
    }
    // Starts another scenario on the same queue: the list and allocator are reset (a list that failed at Close can be).
    bool used = false;
    void begin() {
        if (used && refused) {  // a rejected list is dropped with its allocator rather than reset
            list.reset();
            allocator.reset();
            device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p));
            device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr, IID_PPV_ARGS(&list.p));
        } else if (used) {
            allocator->Reset();
            list->Reset(allocator.p, nullptr);
        }
        refused = false;
        used = true;
    }
    bool buffer(D3D12_HEAP_TYPE heap,UINT64 bytes, D3D12_RESOURCE_STATES state, Ptr<ID3D12Resource>& out) {
        D3D12_HEAP_PROPERTIES hp{};
        hp.Type = heap;
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Width = bytes;
        desc.Height = desc.DepthOrArraySize = desc.MipLevels = 1;
        desc.SampleDesc.Count = 1;
        desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        return SUCCEEDED(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                                         IID_PPV_ARGS(&out.p)));
    }
    bool texture(const D3D12_RESOURCE_DESC& desc, D3D12_RESOURCE_STATES state, Ptr<ID3D12Resource>& out) {
        D3D12_HEAP_PROPERTIES hp{};
        hp.Type = D3D12_HEAP_TYPE_DEFAULT;
        return SUCCEEDED(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                                         IID_PPV_ARGS(&out.p)));
    }
    void barrier(ID3D12Resource* r, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
        D3D12_RESOURCE_BARRIER b{};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition.pResource = r;
        b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        b.Transition.StateBefore = before;
        b.Transition.StateAfter = after;
        list->ResourceBarrier(1, &b);
    }
    // Close, execute, wait. Returns an error text, or an empty string.
    std::string submit() {
        if (const HRESULT hr = list->Close(); FAILED(hr)) { refused = true; return "Close failed 0x" + hex(hr); }
        ID3D12CommandList* lists[] = {list.p};
        queue->ExecuteCommandLists(1, lists);
        if (ours) {
            const auto result = rt::d3d12::last_execute_result(queue.p);
            if (result != leonida::ReplayResult::success) {
                refused = true;
                return "the layer refused the list (ReplayResult " + std::to_string(int(result)) + ")";
            }
        }
        if (FAILED(queue->Signal(fence.p, ++value)) || FAILED(fence->SetEventOnCompletion(value, nullptr)))
            return "fence wait failed";
        if (fence->GetCompletedValue() < value) return "fence did not complete";
        if (const HRESULT hr = device->GetDeviceRemovedReason(); FAILED(hr)) return "device removed 0x" + hex(hr);
        return {};
    }
    static std::string hex(HRESULT hr) {
        char text[16];
        std::snprintf(text, sizeof(text), "%08x", static_cast<unsigned>(hr));
        return text;
    }
};

inline std::uint8_t fill_byte(unsigned seed, size_t index, bool floaty) {
    std::uint32_t x = std::uint32_t(index) * 2654435761u ^ seed * 0x9E3779B1u;
    x ^= x >> 15;
    x *= 0x85ebca6bu;
    x ^= x >> 13;
    x *= 0xc2b2ae35u;
    x ^= x >> 16;
    return std::uint8_t(x >> 8) & (floaty ? 0x3f : 0xff);
}

using Placed = D3D12_PLACED_SUBRESOURCE_FOOTPRINT;

struct Layout {
    std::vector<Placed> placed;
    std::vector<UINT> rows;
    std::vector<UINT64> row_bytes;
    UINT64 total = 0;
};

inline D3D12_TEXTURE_COPY_LOCATION sub_loc(ID3D12Resource* r, UINT index) {
    D3D12_TEXTURE_COPY_LOCATION l{};
    l.pResource = r;
    l.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    l.SubresourceIndex = index;
    return l;
}

inline D3D12_TEXTURE_COPY_LOCATION foot_loc(ID3D12Resource* r, const Placed& p) {
    D3D12_TEXTURE_COPY_LOCATION l{};
    l.pResource = r;
    l.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    l.PlacedFootprint = p;
    return l;
}

// One texture case on one device. `host` supplies the layouts.
inline void run_texture(Run& run, ID3D12Device* host, const TexCase& tc, Outcome& out) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = tc.dim;
    desc.Width = tc.width;
    desc.Height = tc.height;
    desc.DepthOrArraySize = UINT16(tc.depth_or_layers);
    desc.MipLevels = tc.mips;
    desc.Format = tc.format;
    desc.SampleDesc.Count = 1;
    if (tc.ds) desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    const bool volume = tc.dim == D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    const UINT layers = volume ? 1 : tc.depth_or_layers;
    D3D12_FEATURE_DATA_FORMAT_INFO info{tc.format, 1};
    host->CheckFeatureSupport(D3D12_FEATURE_FORMAT_INFO, &info, sizeof(info));
    const UINT n = UINT(tc.mips) * layers * info.PlaneCount;
    const UINT bw = tc.bc ? 4 : 1;

    // Sets: 0 A (source), 1 B, 2 C, 3 D (prefilled), then readbacks R, B, C, D, A, P.
    constexpr int sets = 6;
    Layout layout[sets];
    UINT64 total_one = 0;
    host->GetCopyableFootprints(&desc, 0, n, 0, nullptr, nullptr, nullptr, &total_one);
    const UINT64 stride = (total_one + 511) / 512 * 512;
    for (int k = 0; k < sets; ++k) {
        Layout& l = layout[k];
        l.placed.resize(n);
        l.rows.resize(n);
        l.row_bytes.resize(n);
        host->GetCopyableFootprints(&desc, 0, n, stride * k, l.placed.data(), l.rows.data(), l.row_bytes.data(), &l.total);
    }

    Ptr<ID3D12Resource> up, rb, A, B, C, D, R;
    constexpr auto copy_dest = D3D12_RESOURCE_STATE_COPY_DEST, copy_source = D3D12_RESOURCE_STATE_COPY_SOURCE;
    if (!run.buffer(D3D12_HEAP_TYPE_UPLOAD, stride * 4, D3D12_RESOURCE_STATE_GENERIC_READ, up) ||
        !run.buffer(D3D12_HEAP_TYPE_READBACK, stride * sets, copy_dest, rb) ||
        !run.texture(desc, copy_dest, A) || !run.texture(desc, copy_dest, B) || !run.texture(desc, copy_dest, C) ||
        !run.texture(desc, copy_dest, D) || !run.texture(desc, copy_dest, R)) {
        out.error = "resource creation failed";
        return;
    }
    std::uint8_t *upm = nullptr, *rbm = nullptr;
    if (FAILED(up->Map(0, nullptr, reinterpret_cast<void**>(&upm))) ||
        FAILED(rb->Map(0, nullptr, reinterpret_cast<void**>(&rbm)))) {
        out.error = "Map failed";
        return;
    }
    std::memset(upm, 0, size_t(stride) * 4);
    std::memset(rbm, 0xcd, size_t(stride) * sets);
    for (int k = 0; k < 4; ++k)
        for (UINT i = 0; i < n; ++i) {
            const Placed& p = layout[k].placed[i];
            for (UINT z = 0; z < p.Footprint.Depth; ++z)
                for (UINT row = 0; row < layout[k].rows[i]; ++row)
                    for (UINT64 col = 0; col < layout[k].row_bytes[i]; ++col) {
                        const size_t at = size_t(p.Offset) + (size_t(z) * layout[k].rows[i] + row) * p.Footprint.RowPitch + size_t(col);
                        // The X8 byte of D24's depth plane is unspecified; the host stores 0 there, so upload 0.
                        const bool x8 = tc.format == DXGI_FORMAT_D24_UNORM_S8_UINT && i < UINT(tc.mips) * layers && col % 4 == 3;
                        std::uint8_t value = x8 ? 0 : fill_byte(unsigned(k + 1) * 7919u + i, (size_t(z) * 4099 + row) * 131 + size_t(col), tc.floaty);
                        // Float depth: keep every value a normal float (hardware flushes denormal depth on copies).
                        if (tc.ds && tc.floaty && i < UINT(tc.mips) * layers && col % 4 == 3) value |= 0x20;
                        upm[at] = value;
                    }
        }

    // Mip dimensions and the inner box of a subresource.
    struct Inner { UINT ix, iy, iz, wi, hi, di; bool inset; };
    const auto inner = [&](UINT sub) {
        const UINT mip = sub % tc.mips;
        const UINT w = std::max(1u, tc.width >> mip), h = std::max(1u, tc.height >> mip);
        const UINT d = volume ? std::max(1u, tc.depth_or_layers >> mip) : 1u;
        Inner r{0, 0, 0, w, h, d, false};
        if (tc.ds) return r;
        const UINT wb = (w + bw - 1) / bw, hb = (h + bw - 1) / bw;
        if (wb >= 3) { r.ix = bw; r.wi = (wb - 2) * bw; r.inset = true; }
        if (hb >= 3) { r.iy = bw; r.hi = (hb - 2) * bw; r.inset = true; }
        if (r.inset && tc.bc) {  // a block-compressed box is whole blocks on every axis
            if (wb < 3) r.wi = wb * bw;
            if (hb < 3) r.hi = hb * bw;
        }
        if (d >= 3) { r.iz = 1; r.di = d - 2; r.inset = true; }
        return r;
    };
    const auto swapped = [&](UINT sub) {  // the same mip and plane in the next array slice
        const UINT mip = sub % tc.mips, slice = (sub / tc.mips) % layers, plane = sub / (tc.mips * layers);
        return mip + ((slice + 1) % layers) * tc.mips + plane * tc.mips * layers;
    };

    // Uploads, whole.
    Ptr<ID3D12Resource>* targets[4] = {&A, &B, &C, &D};
    for (int k = 0; k < 4; ++k)
        for (UINT i = 0; i < n; ++i) {
            const auto dst = sub_loc(targets[k]->p, i);
            const auto src = foot_loc(up.p, layout[k].placed[i]);
            run.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        }
    run.barrier(A.p, copy_dest, copy_source);
    // The partial copies below write the textures the uploads just wrote: a round trip through COPY_SOURCE orders them.
    for (auto* t : {&B, &C, &D}) {
        run.barrier(t->p, copy_dest, copy_source);
        run.barrier(t->p, copy_source, copy_dest);
    }
    run.list->CopyResource(R.p, A.p);
    for (UINT i = 0; i < n; ++i) {
        const Inner in = inner(i);
        const D3D12_BOX inner_box{in.ix, in.iy, in.iz, in.ix + in.wi, in.iy + in.hi, in.iz + in.di};
        const D3D12_BOX origin_box{0, 0, 0, in.wi, in.hi, in.di};
        const D3D12_BOX* box = tc.ds || !in.inset ? nullptr : &inner_box;
        const auto from = sub_loc(A.p, i);
        {  // B: the inner box to the origin
            const auto dst = sub_loc(B.p, i);
            run.list->CopyTextureRegion(&dst, 0, 0, 0, &from, box);
        }
        {  // C: the origin box into another slice at the inset
            const auto dst = sub_loc(C.p, swapped(i));
            run.list->CopyTextureRegion(&dst, in.ix, in.iy, in.iz, &from, tc.ds || !in.inset ? nullptr : &origin_box);
        }
        {  // D: the inner box of the uploaded footprint to the inset
            const auto dst = sub_loc(D.p, i);
            const auto src = foot_loc(up.p, layout[0].placed[i]);
            run.list->CopyTextureRegion(&dst, in.ix, in.iy, in.iz, &src, box);
        }
        {  // P: the inner box of A into a footprint, at the inset
            const auto dst = foot_loc(rb.p, layout[5].placed[i]);
            run.list->CopyTextureRegion(&dst, in.ix, in.iy, in.iz, &from, box);
        }
    }
    for (auto* t : {&B, &C, &D, &R}) run.barrier(t->p, copy_dest, copy_source);
    Ptr<ID3D12Resource>* readers[5] = {&R, &B, &C, &D, &A};
    for (int k = 0; k < 5; ++k)
        for (UINT i = 0; i < n; ++i) {
            const auto dst = foot_loc(rb.p, layout[k].placed[i]);
            const auto src = sub_loc(readers[k]->p, i);
            run.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        }
    out.error = run.submit();
    if (!out.error.empty()) return;
    out.bytes.assign(rbm, rbm + size_t(stride) * sets);
    static const char* names[sets] = {"R (CopyResource)", "B (box to origin)", "C (box to other slice)",
                                      "D (footprint box to texture)", "A (source)", "P (texture box to footprint)"};
    for (int k = 0; k < sets; ++k)
        for (UINT i = 0; i < n; ++i) {
            const Placed& p = layout[k].placed[i];
            const UINT mip = i % tc.mips, slice = (i / tc.mips) % layers, plane = i / (tc.mips * layers);
            char label[96];
            std::snprintf(label, sizeof(label), "%s sub %u (mip %u slice %u plane %u)", names[k], i, mip, slice, plane);
            out.regions.push_back({label, size_t(p.Offset), p.Footprint.RowPitch, size_t(layout[k].row_bytes[i]),
                                   layout[k].rows[i], p.Footprint.Depth, tc.format == DXGI_FORMAT_D24_UNORM_S8_UINT && plane == 0});
        }
    up->Unmap(0, nullptr);
    rb->Unmap(0, nullptr);
}

// Buffer copies: odd offsets and sizes, CopyResource of a buffer, an upload straight into a readback buffer.
inline void run_buffers(Run& run, Outcome& out) {
    constexpr UINT64 kb = 4096;
    constexpr auto copy_dest = D3D12_RESOURCE_STATE_COPY_DEST, copy_source = D3D12_RESOURCE_STATE_COPY_SOURCE;
    Ptr<ID3D12Resource> up, rb, A, B, C, D, E;
    if (!run.buffer(D3D12_HEAP_TYPE_UPLOAD, kb * 3, D3D12_RESOURCE_STATE_GENERIC_READ, up) ||
        !run.buffer(D3D12_HEAP_TYPE_READBACK, kb * 4 + 256, copy_dest, rb) ||
        !run.buffer(D3D12_HEAP_TYPE_DEFAULT, kb, copy_dest, A) || !run.buffer(D3D12_HEAP_TYPE_DEFAULT, kb, copy_dest, B) ||
        !run.buffer(D3D12_HEAP_TYPE_DEFAULT, kb, copy_dest, C) || !run.buffer(D3D12_HEAP_TYPE_DEFAULT, 4093, copy_dest, D) ||
        !run.buffer(D3D12_HEAP_TYPE_DEFAULT, 4093, copy_dest, E)) {
        out.error = "resource creation failed";
        return;
    }
    std::uint8_t *upm = nullptr, *rbm = nullptr;
    if (FAILED(up->Map(0, nullptr, reinterpret_cast<void**>(&upm))) || FAILED(rb->Map(0, nullptr, reinterpret_cast<void**>(&rbm)))) {
        out.error = "Map failed";
        return;
    }
    for (size_t i = 0; i < kb * 3; ++i) upm[i] = fill_byte(unsigned(i / kb) + 1, i % kb, false);
    std::memset(rbm, 0xcd, size_t(kb) * 4 + 256);
    auto& l = *run.list.p;
    l.CopyBufferRegion(A.p, 0, up.p, 0, kb);
    l.CopyBufferRegion(B.p, 0, up.p, kb, kb);
    l.CopyBufferRegion(E.p, 0, up.p, 2 * kb, 4093);
    run.barrier(A.p, copy_dest, copy_source);
    run.barrier(B.p, copy_dest, copy_source);  // orders B's upload before the partial copies into it
    run.barrier(B.p, copy_source, copy_dest);
    l.CopyBufferRegion(B.p, 1, A.p, 3, 1);
    l.CopyBufferRegion(B.p, 7, A.p, 100, 5);
    l.CopyBufferRegion(B.p, 64, A.p, 0, 256);
    l.CopyBufferRegion(B.p, 1000, A.p, 513, 777);
    l.CopyBufferRegion(B.p, 4095, A.p, 4095, 1);
    l.CopyResource(C.p, A.p);
    run.barrier(E.p, copy_dest, copy_source);
    l.CopyResource(D.p, E.p);
    l.CopyBufferRegion(rb.p, kb * 4 + 3, up.p, 11, 17);
    for (auto* r : {&B, &C, &D}) run.barrier(r->p, copy_dest, copy_source);
    l.CopyBufferRegion(rb.p, 0, B.p, 0, kb);
    l.CopyBufferRegion(rb.p, kb, C.p, 0, kb);
    l.CopyBufferRegion(rb.p, 2 * kb, D.p, 0, 4093);
    l.CopyBufferRegion(rb.p, 3 * kb, A.p, 0, kb);
    out.error = run.submit();
    if (!out.error.empty()) return;
    out.bytes.assign(rbm, rbm + size_t(kb) * 4 + 256);
    const std::pair<const char*, size_t> parts[] = {{"B (partial copies)", kb}, {"C (CopyResource)", kb},
                                                    {"D (CopyResource, 4093 bytes)", kb}, {"A (source)", kb},
                                                    {"upload to readback, 17 bytes at 3", 256}};
    size_t at = 0;
    for (const auto& [label, size] : parts) {
        out.regions.push_back({label, at, size, size, 1, 1, false});
        at += size;
    }
    up->Unmap(0, nullptr);
    rb->Unmap(0, nullptr);
}

// Texture to texture between different mips of one 24x24 texture (mips 24, 12, 6, 3, 1): sizes that are not block
// multiples, boxes that run to a padded edge, and boxes that do not fit the destination (which must be refused).
struct MixCase {
    const char* name;
    UINT src_mip, dst_mip;
    bool has_box;
    D3D12_BOX box;
    UINT dx, dy;
};

// Copies one subresource of A into one of D (both prefilled), reads D back. `out.error` is set when the list was
// rejected (`run.refused`).
inline void run_mixed(Run& run, ID3D12Device* host, DXGI_FORMAT format, const MixCase& mc, Outcome& out) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = desc.Height = 24;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 5;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    constexpr UINT n = 5;
    Layout layout[3];
    UINT64 total_one = 0;
    host->GetCopyableFootprints(&desc, 0, n, 0, nullptr, nullptr, nullptr, &total_one);
    const UINT64 stride = (total_one + 511) / 512 * 512;
    for (int k = 0; k < 3; ++k) {
        layout[k].placed.resize(n);
        layout[k].rows.resize(n);
        layout[k].row_bytes.resize(n);
        host->GetCopyableFootprints(&desc, 0, n, k == 2 ? 0 : stride * k, layout[k].placed.data(), layout[k].rows.data(),
                                    layout[k].row_bytes.data(), &layout[k].total);
    }
    constexpr auto copy_dest = D3D12_RESOURCE_STATE_COPY_DEST, copy_source = D3D12_RESOURCE_STATE_COPY_SOURCE;
    Ptr<ID3D12Resource> up, rb, A, D;
    if (!run.buffer(D3D12_HEAP_TYPE_UPLOAD, stride * 2, D3D12_RESOURCE_STATE_GENERIC_READ, up) ||
        !run.buffer(D3D12_HEAP_TYPE_READBACK, stride, copy_dest, rb) || !run.texture(desc, copy_dest, A) ||
        !run.texture(desc, copy_dest, D)) {
        out.error = "resource creation failed";
        return;
    }
    std::uint8_t *upm = nullptr, *rbm = nullptr;
    if (FAILED(up->Map(0, nullptr, reinterpret_cast<void**>(&upm))) || FAILED(rb->Map(0, nullptr, reinterpret_cast<void**>(&rbm)))) {
        out.error = "Map failed";
        return;
    }
    std::memset(upm, 0, size_t(stride) * 2);
    std::memset(rbm, 0xcd, size_t(stride));
    for (int k = 0; k < 2; ++k)
        for (UINT i = 0; i < n; ++i) {
            const Placed& p = layout[k].placed[i];
            for (UINT row = 0; row < layout[k].rows[i]; ++row)
                for (UINT64 col = 0; col < layout[k].row_bytes[i]; ++col)
                    upm[size_t(p.Offset) + size_t(row) * p.Footprint.RowPitch + size_t(col)] =
                        fill_byte(unsigned(k + 11) * 104729u + i, size_t(row) * 131 + size_t(col), false);
            const auto dst = sub_loc(k ? D.p : A.p, i);
            const auto src = foot_loc(up.p, p);
            run.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        }
    run.barrier(A.p, copy_dest, copy_source);
    run.barrier(D.p, copy_dest, copy_source);  // orders the upload before the partial copy
    run.barrier(D.p, copy_source, copy_dest);
    {
        const auto dst = sub_loc(D.p, mc.dst_mip), src = sub_loc(A.p, mc.src_mip);
        run.list->CopyTextureRegion(&dst, mc.dx, mc.dy, 0, &src, mc.has_box ? &mc.box : nullptr);
    }
    run.barrier(D.p, copy_dest, copy_source);
    for (UINT i = 0; i < n; ++i) {
        const auto dst = foot_loc(rb.p, layout[2].placed[i]);
        const auto src = sub_loc(D.p, i);
        run.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }
    out.error = run.submit();
    if (!out.error.empty()) return;
    out.bytes.assign(rbm, rbm + size_t(stride));
    for (UINT i = 0; i < n; ++i) {
        const Placed& p = layout[2].placed[i];
        char label[64];
        std::snprintf(label, sizeof(label), "D sub %u (mip %u)", i, i);
        out.regions.push_back({label, size_t(p.Offset), p.Footprint.RowPitch, size_t(layout[2].row_bytes[i]),
                               layout[2].rows[i], p.Footprint.Depth, false});
    }
    up->Unmap(0, nullptr);
    rb->Unmap(0, nullptr);
}

// Compares two outcomes; returns 1 on a mismatch (and says what and where), else 0.
inline int compare(const char* name, const Outcome& ours, const Outcome& host) {
    if (!host.error.empty()) {
        std::fprintf(stderr, "copy case %s: the HOST run failed (%s); fix the case\n", name, host.error.c_str());
        return 1;
    }
    if (!ours.error.empty()) {
        std::fprintf(stderr, "copy case %s: layer run failed: %s\n", name, ours.error.c_str());
        return 1;
    }
    int bad_regions = 0;
    for (const Region& r : host.regions) {
        size_t differing = 0, first = 0, padding = 0, first_row = 0, first_col = 0;
        for (size_t z = 0; z < r.depth; ++z)
            for (size_t row = 0; row < r.rows; ++row) {
                const size_t base = r.offset + (z * r.rows + row) * r.pitch;
                for (size_t col = 0; col < r.pitch; ++col) {
                    if (ours.bytes[base + col] == host.bytes[base + col]) continue;
                    if (r.x8 && col % 4 == 3 && col < r.row_bytes) continue;
                    if (col >= r.row_bytes) { ++padding; continue; }
                    if (!differing++) { first = base + col; first_row = z * r.rows + row; first_col = col; }
                }
            }
        if (differing && bad_regions++ < 3)
            std::fprintf(stderr, "copy case %s: %s: %zu bytes differ, first at row %zu col %zu: ours %02x host %02x\n",
                         name, r.label.c_str(), differing, first_row, first_col, ours.bytes[first], host.bytes[first]);
        if (padding && !differing)
            std::printf("copy case %s: %s: %zu row-padding bytes differ (not compared)\n", name, r.label.c_str(), padding);
    }
    if (bad_regions > 3) std::fprintf(stderr, "copy case %s: %d regions differ in all\n", name, bad_regions);
    return bad_regions ? 1 : 0;
}

// Runs every case; returns the number of cases that differ, or -1 with no host device.
inline int compare_copies(ID3D12Device* ours, bool bc_supported) {
    Ptr<ID3D12Device> host;
    std::string adapter;
    if (!create_host(host, adapter)) return -1;
    std::printf("copy comparison: host side is %s\n", adapter.c_str());
    using D = D3D12_RESOURCE_DIMENSION;
    constexpr D t2 = D3D12_RESOURCE_DIMENSION_TEXTURE2D, t3 = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    std::vector<TexCase> cases;
    const auto color = [&](const char* name, DXGI_FORMAT f, bool floaty = false) {
        cases.push_back({name, t2, f, 36, 20, 3, 6, floaty, false, false});
    };
    color("R8_UNORM", DXGI_FORMAT_R8_UNORM);
    color("R8G8_SNORM", DXGI_FORMAT_R8G8_SNORM);
    color("R8G8B8A8_UNORM", DXGI_FORMAT_R8G8B8A8_UNORM);
    color("R8G8B8A8_UNORM_SRGB", DXGI_FORMAT_R8G8B8A8_UNORM_SRGB);
    color("R8G8B8A8_TYPELESS", DXGI_FORMAT_R8G8B8A8_TYPELESS);
    color("B8G8R8A8_UNORM", DXGI_FORMAT_B8G8R8A8_UNORM);
    color("R16_UNORM", DXGI_FORMAT_R16_UNORM);
    color("R16G16_FLOAT", DXGI_FORMAT_R16G16_FLOAT, true);
    color("R16G16B16A16_FLOAT", DXGI_FORMAT_R16G16B16A16_FLOAT, true);
    color("R32_UINT", DXGI_FORMAT_R32_UINT);
    color("R32G32_UINT", DXGI_FORMAT_R32G32_UINT);
    color("R32G32B32A32_FLOAT", DXGI_FORMAT_R32G32B32A32_FLOAT, true);
    color("R10G10B10A2_UNORM", DXGI_FORMAT_R10G10B10A2_UNORM);
    color("R11G11B10_FLOAT", DXGI_FORMAT_R11G11B10_FLOAT, true);
    color("R9G9B9E5_SHAREDEXP", DXGI_FORMAT_R9G9B9E5_SHAREDEXP);
    color("B5G6R5_UNORM", DXGI_FORMAT_B5G6R5_UNORM);
    color("B5G5R5A1_UNORM", DXGI_FORMAT_B5G5R5A1_UNORM);
    cases.push_back({"3D R8G8B8A8_UNORM", t3, DXGI_FORMAT_R8G8B8A8_UNORM, 20, 12, 9, 4, false, false, false});
    cases.push_back({"3D R32_FLOAT", t3, DXGI_FORMAT_R32_FLOAT, 9, 5, 6, 4, true, false, false});
    cases.push_back({"single mip, single slice R8G8B8A8_UNORM", t2, DXGI_FORMAT_R8G8B8A8_UNORM, 16, 16, 1, 1, false, false, false});
    const auto depth = [&](const char* name, DXGI_FORMAT f, bool floaty) {
        cases.push_back({name, t2, f, 20, 12, 2, 3, floaty, true, false});
    };
    depth("D16_UNORM", DXGI_FORMAT_D16_UNORM, false);
    depth("D32_FLOAT", DXGI_FORMAT_D32_FLOAT, true);
    depth("D24_UNORM_S8_UINT", DXGI_FORMAT_D24_UNORM_S8_UINT, false);
    // Without a native D24 the layer stores D32_SFLOAT_S8_UINT and converts the depth plane of an upload and a readback
    // (depth24.h); the random 24-bit values here must come back exactly, as on the host.
    depth("D32_FLOAT_S8X24_UINT", DXGI_FORMAT_D32_FLOAT_S8X24_UINT, true);
    if (bc_supported) {
        const auto bc = [&](const char* name, DXGI_FORMAT f) {
            cases.push_back({name, t2, f, 36, 20, 3, 6, false, false, true});
        };
        bc("BC1_UNORM", DXGI_FORMAT_BC1_UNORM);
        bc("BC1_UNORM_SRGB", DXGI_FORMAT_BC1_UNORM_SRGB);
        bc("BC2_UNORM", DXGI_FORMAT_BC2_UNORM);
        bc("BC3_UNORM", DXGI_FORMAT_BC3_UNORM);
        bc("BC4_UNORM", DXGI_FORMAT_BC4_UNORM);
        bc("BC4_SNORM", DXGI_FORMAT_BC4_SNORM);
        bc("BC5_UNORM", DXGI_FORMAT_BC5_UNORM);
        bc("BC5_SNORM", DXGI_FORMAT_BC5_SNORM);
        bc("BC6H_UF16", DXGI_FORMAT_BC6H_UF16);
        bc("BC6H_SF16", DXGI_FORMAT_BC6H_SF16);
        bc("BC7_UNORM", DXGI_FORMAT_BC7_UNORM);
        bc("BC7_UNORM_SRGB", DXGI_FORMAT_BC7_UNORM_SRGB);
        cases.push_back({"3D BC1_UNORM", t3, DXGI_FORMAT_BC1_UNORM, 16, 8, 6, 3, false, false, true});
        cases.push_back({"BC7_UNORM, 4x4 (one block)", t2, DXGI_FORMAT_BC7_UNORM, 4, 4, 2, 1, false, false, true});
    } else {
        std::printf("copy comparison: no textureCompressionBC, BC cases skipped\n");
    }

    int bad = 0, total = 0;
    const auto both = [&](const char* name, const std::function<void(Run&, Outcome&)>& scenario) {
        ++total;
        Outcome a, b;
        Run ours_run, host_run;
        if (!ours_run.init(ours, true) || !host_run.init(host.p, false)) {
            std::fprintf(stderr, "copy case %s: could not create the queue and list\n", name);
            ++bad;
            return;
        }
        scenario(ours_run, a);
        scenario(host_run, b);
        const int r = compare(name, a, b);
        bad += r;
        if (!r) std::printf("copy case %s: matches the host (%zu regions)\n", name, b.regions.size());
    };
    for (const TexCase& tc : cases) both(tc.name, [&](Run& r, Outcome& o) { run_texture(r, host.p, tc, o); });
    both("buffers", [&](Run& r, Outcome& o) { run_buffers(r, o); });

    // Different mips of one texture. Both sides must refuse what does not fit; what the host copies the layer must
    // copy the same. A block-compressed copy whose mip edge falls inside a block on one side only is staged through
    // transfer scratch by the layer (vulkan_replay.h, prepare_texture_copy).
    constexpr D3D12_BOX no_box{};
    const MixCase mixes[] = {
        {"mip 0 box 12x12 into mip 1", 0, 1, true, {0, 0, 0, 12, 12, 1}, 0, 0},
        {"mip 1 into mip 0 at 8,8", 1, 0, false, no_box, 8, 8},
        {"mip 1 box 4..12 into mip 0", 1, 0, true, {4, 4, 0, 12, 12, 1}, 0, 0},
        {"mip 0 box 8x8 into the 6x6 mip 2", 0, 2, true, {0, 0, 0, 8, 8, 1}, 0, 0},
        {"mip 2 (6x6) into mip 0 at 4,4", 2, 0, false, no_box, 4, 4},
        {"mip 2 (6x6) into mip 1", 2, 1, false, no_box, 0, 0},
        {"mip 0 box 4x4 into the 3x3 mip 3", 0, 3, true, {0, 0, 0, 4, 4, 1}, 0, 0},
        // The source's mip edge inside the block: the box 4..8 of the 6x6 mip 2 (2x2 texels) into the 3x3 mip 3.
        {"mip 2 (6x6) box 4..8 into the 3x3 mip 3", 2, 3, true, {4, 4, 0, 8, 8, 1}, 0, 0},
        // Both mip edges inside one block, at an offset: the 3x3 mip 3 into the 6x6 mip 2 at 4,4 (2x2 texels land).
        {"mip 3 (3x3) into mip 2 at 4,4", 3, 2, false, no_box, 4, 4},
        {"mip 0 box 16x16 into the 3x3 mip 3 (does not fit)", 0, 3, true, {0, 0, 0, 16, 16, 1}, 0, 0},
        {"mip 0 box 16x16 into the 6x6 mip 2 (does not fit)", 0, 2, true, {0, 0, 0, 16, 16, 1}, 0, 0},
        {"mip 1 into mip 0 at 16,16 (does not fit)", 1, 0, false, no_box, 16, 16},
    };
    Run ours_run, host_run;  // one queue each for every mixed-mip case
    const bool ours_ok = ours_run.init(ours, true), host_ok = host_run.init(host.p, false);
    if (!ours_ok || !host_ok) {
        std::fprintf(stderr, "copy cases with mixed mips: could not create the queue and list on %s\n",
                     ours_ok ? "the host" : "the layer");
        return bad + 1;
    }
    for (const DXGI_FORMAT format : {DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_BC1_UNORM, DXGI_FORMAT_BC7_UNORM}) {
        const bool bc = format != DXGI_FORMAT_R8G8B8A8_UNORM;
        if (bc && !bc_supported) continue;
        for (const MixCase& mc : mixes) {
            ++total;
            const std::string name = std::string(bc ? "BC " : "RGBA8 ") + mc.name;
            Outcome a, b;
            ours_run.begin();
            host_run.begin();
            run_mixed(ours_run, host.p, format, mc, a);
            run_mixed(host_run, host.p, format, mc, b);
            if (!host_run.refused && !b.error.empty()) {
                std::fprintf(stderr, "copy case %s: the HOST run failed (%s); fix the case\n", name.c_str(), b.error.c_str());
                ++bad;
            } else if (host_run.refused != ours_run.refused) {
                std::fprintf(stderr, "copy case %s: the host %s the copy, the layer %s it (%s)\n", name.c_str(),
                             host_run.refused ? "refuses" : "does", ours_run.refused ? "refuses" : "does",
                             ours_run.refused ? a.error.c_str() : "accepted");
                ++bad;
            } else if (!host_run.refused) {
                const int r = compare(name.c_str(), a, b);
                bad += r;
                if (!r) std::printf("copy case %s: matches the host\n", name.c_str());
            } else {
                std::printf("copy case %s: refused by both (host: %s; layer: %s)\n", name.c_str(), b.error.c_str(),
                            a.error.c_str());
            }
        }
    }
    std::printf("copy comparison: %d of %d cases differ from the host\n", bad, total);
    return bad;
}

}  // namespace copycmp
