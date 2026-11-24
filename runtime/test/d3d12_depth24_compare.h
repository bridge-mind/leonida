// D24_UNORM_S8_UINT against the host PC runtime (or WARP), byte for byte. Included by
// d3d12_vulkan_device_test.cpp after d3d12_copy_compare.h, whose Run and host selection (LEONIDA_HOST_ADAPTER) it uses.
//
// On a GPU without D24 the layer stores D32_SFLOAT_S8_UINT (depth24.h): a clear must leave the 24-bit value D3D's
// quantisation gives, and a footprint copy converts. The copy paths are compared with random 24-bit words in
// d3d12_copy_compare.h; this adds the clears (whole view, and a rectangle per value) with values on and between the
// 24-bit steps, a stencil value, and the depth and stencil planes read back through a footprint.
#pragma once
#include "d3d12_copy_compare.h"
#include "../src/depth24.h"
#include "../src/gfx10_descriptors.h"

#include <cmath>

namespace copycmp {

// Clear values: exact steps, ties (0.5 * 16777215 is a half), steps of the float grid around them, and the ends.
inline std::vector<float> depth24_clear_values() {
    std::vector<float> v = {0.0f, 1.0f, 0.5f, 0.25f, 0.75f, 1.0f / 3.0f, 2.0f / 3.0f, 0.1f, 0.9f, 0.123456789f,
                            1e-8f, 3e-8f, 6e-8f, 1.0e-7f, 1e-3f, 0.999999f, 0.99999994f, 0.9999999f, 0.5000001f,
                            0.49999997f, 8388607.0f / 16777215.0f, 8388608.0f / 16777215.0f, 1.0f / 16777215.0f,
                            0.5f / 16777215.0f, 1.5f / 16777215.0f, 2.5f / 16777215.0f, 16777214.5f / 16777215.0f,
                            16777214.0f / 16777215.0f, 5.9604645e-8f, 8.940697e-8f, 2.9802322e-8f, 1.0f / 65535.0f};
    // 1/64 steps, uniform random values, values at every scale (a random exponent), and the floats around random
    // half steps ((k + 0.5) / 16777215, where the rounding rule shows) and one step of the float grid either side.
    for (int i = 1; i < 64; ++i) v.push_back(float(i) / 64.0f + 1e-9f);
    std::uint32_t x = 12345;
    const auto next = [&] { return x = x * 1664525u + 1013904223u; };
    for (int i = 0; i < 1024; ++i) v.push_back(float(next() >> 8) / 16777216.0f);
    for (int i = 0; i < 1024; ++i) v.push_back(std::ldexp(1.0f + float(next() >> 9) / 8388608.0f, -int(next() >> 27) - 1));
    for (int i = 0; i < 1024; ++i) {
        const float half = float((double(next() >> 8) + 0.5) / 16777215.0);
        v.push_back(half);
        v.push_back(std::nextafter(half, 0.0f));
        v.push_back(std::nextafter(half, 1.0f));
    }
    return v;
}

// One scenario on one device: every value cleared into a 4x4 texture (a whole-view clear) and into a column of a wide
// one (a rectangle clear), the planes of both read back. out.bytes holds them, regions say where.
inline void run_depth24_clears(Run& run, ID3D12Device* host, const std::vector<float>& values, Outcome& out,
                               DXGI_FORMAT format = DXGI_FORMAT_D24_UNORM_S8_UINT, bool d24_native = true) {
    const UINT count = UINT(values.size());
    const UINT whole = std::min<UINT>(count, 160);  // whole-view clears, one texture each; the wide one takes all
    D3D12_RESOURCE_DESC tiny{};
    tiny.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    tiny.Width = 4;
    tiny.Height = 4;
    tiny.DepthOrArraySize = 1;
    tiny.MipLevels = 1;
    tiny.Format = format;
    tiny.SampleDesc.Count = 1;
    tiny.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    D3D12_RESOURCE_DESC wide = tiny;
    wide.Width = count;

    // Footprints come from the host: the layer's must be the same, and are compared elsewhere.
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT tiny_foot[2], wide_foot[2];
    UINT tiny_rows[2], wide_rows[2];
    UINT64 tiny_row_bytes[2], wide_row_bytes[2], tiny_total = 0, wide_total = 0;
    host->GetCopyableFootprints(&tiny, 0, 2, 0, tiny_foot, tiny_rows, tiny_row_bytes, &tiny_total);
    host->GetCopyableFootprints(&wide, 0, 2, 0, wide_foot, wide_rows, wide_row_bytes, &wide_total);
    const UINT64 tiny_stride = (tiny_total + 511) / 512 * 512, wide_stride = (wide_total + 511) / 512 * 512;
    const UINT64 wide_base = tiny_stride * whole;
    const UINT64 size = wide_base + wide_stride;

    Ptr<ID3D12Resource> rb, wide_tex;
    std::vector<Ptr<ID3D12Resource>> tex(whole);
    Ptr<ID3D12DescriptorHeap> heap;
    D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, whole + 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    if (!run.buffer(D3D12_HEAP_TYPE_READBACK, size, D3D12_RESOURCE_STATE_COPY_DEST, rb) ||
        !run.texture(wide, D3D12_RESOURCE_STATE_DEPTH_WRITE, wide_tex) ||
        FAILED(run.device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap.p)))) {
        out.error = "resource creation failed";
        return;
    }
    for (UINT k = 0; k < whole; ++k)
        if (!run.texture(tiny, D3D12_RESOURCE_STATE_DEPTH_WRITE, tex[k])) {
            out.error = "resource creation failed";
            return;
        }
    std::uint8_t* mapped = nullptr;
    if (FAILED(rb->Map(0, nullptr, reinterpret_cast<void**>(&mapped)))) {
        out.error = "Map failed";
        return;
    }
    std::memset(mapped, 0xcd, size_t(size));

    const UINT step = run.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
    // The layer's proxy heap answers in the Xbox shape (slot 9 returns the 8-byte handle in RAX); the host's is the PC's.
    D3D12_CPU_DESCRIPTOR_HANDLE start{};
    if (run.ours) {
        using XboxHandle = std::uint64_t(WINAPI*)(void*);
        start.ptr = SIZE_T(reinterpret_cast<XboxHandle>((*reinterpret_cast<void***>(heap.p))[9])(heap.p));
    } else {
        start = heap->GetCPUDescriptorHandleForHeapStart();
    }
    const auto handle = [&](UINT i) {
        D3D12_CPU_DESCRIPTOR_HANDLE h = start;
        h.ptr += size_t(i) * step;
        return h;
    };
    D3D12_DEPTH_STENCIL_VIEW_DESC dsv{};
    dsv.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
    dsv.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
    for (UINT k = 0; k < whole; ++k) {
        run.device->CreateDepthStencilView(tex[k].p, &dsv, handle(k));
    }
    run.device->CreateDepthStencilView(wide_tex.p, &dsv, handle(whole));

    // Shader-resource views of a typeless D24 texture: R24_UNORM_X8 reads the depth plane, X24_TYPELESS_G8_UINT the
    // stencil plane. The layer writes a T# into the slot (the host's is opaque and no shader runs here), so what can be
    // checked is that it is the one the console's own descriptor would be.
    if (run.ours && format == DXGI_FORMAT_R24G8_TYPELESS) {
        Ptr<ID3D12DescriptorHeap> srv_heap;
        const D3D12_DESCRIPTOR_HEAP_DESC srv_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 2, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        if (FAILED(run.device->CreateDescriptorHeap(&srv_desc, IID_PPV_ARGS(&srv_heap.p)))) {
            out.error = "SRV heap creation failed";
            return;
        }
        using XboxHandle = std::uint64_t(WINAPI*)(void*);
        const SIZE_T base = SIZE_T(reinterpret_cast<XboxHandle>((*reinterpret_cast<void***>(srv_heap.p))[9])(srv_heap.p));
        const UINT srv_step = run.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        const DXGI_FORMAT srv_formats[2] = {DXGI_FORMAT_R24_UNORM_X8_TYPELESS, DXGI_FORMAT_X24_TYPELESS_G8_UINT};
        for (UINT plane = 0; plane < 2; ++plane) {
            D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
            srv.Format = srv_formats[plane];
            srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            srv.Texture2D.MipLevels = 1;
            srv.Texture2D.PlaneSlice = plane;
            const D3D12_CPU_DESCRIPTOR_HANDLE slot{base + size_t(plane) * srv_step};
            run.device->CreateShaderResourceView(tex[0].p, &srv, slot);
            leonida::gfx10::ImageWords image_words{};
            std::memcpy(image_words.data(), reinterpret_cast<const void*>(slot.ptr), sizeof(image_words));
            leonida::gfx10::Image decoded{};
            // The depth plane's T# code follows the texture's backing format (D3D-75): X24_8_32_FLOAT where the GPU has no
            // D24 and the layer stores D32_SFLOAT_S8_UINT, 8_24_UNORM where it has one; the stencil plane is 8_UINT.
            const std::uint32_t want = plane ? leonida::gfx10::stencil_format
                                             : leonida::gfx10::format_from_vulkan(d24_native ? VK_FORMAT_D24_UNORM_S8_UINT
                                                                                             : VK_FORMAT_D32_SFLOAT_S8_UINT);
            if (!leonida::gfx10::is_image(image_words) || !leonida::gfx10::decode(image_words, decoded) || decoded.format != want || !want) {
                out.error = std::string("the shader-resource view of ") + (plane ? "X24_TYPELESS_G8_UINT" : "R24_UNORM_X8_TYPELESS") +
                            " did not write the expected T#";
                return;
            }
        }
    }

    // Whole-view clears, one per texture; the wide texture first gets a stencil-only clear, then a column per value.
    for (UINT k = 0; k < whole; ++k)
        run.list->ClearDepthStencilView(handle(k), D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, values[k],
                                        UINT8(0x40 + k % 64), 0, nullptr);
    run.list->ClearDepthStencilView(handle(whole), D3D12_CLEAR_FLAG_STENCIL, 0.0f, 0x11, 0, nullptr);
    for (UINT k = 0; k < count; ++k) {
        const D3D12_RECT column{LONG(k), 0, LONG(k + 1), 4};
        run.list->ClearDepthStencilView(handle(whole), D3D12_CLEAR_FLAG_DEPTH, values[k], 0, 1, &column);
    }
    for (UINT k = 0; k < whole; ++k) run.barrier(tex[k].p, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_COPY_SOURCE);
    run.barrier(wide_tex.p, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_COPY_SOURCE);

    const auto copy_out = [&](ID3D12Resource* texture, const D3D12_PLACED_SUBRESOURCE_FOOTPRINT* foot, UINT64 base,
                              const UINT* rows, const UINT64* row_bytes, const char* what, UINT index) {
        for (UINT plane = 0; plane < 2; ++plane) {
            D3D12_PLACED_SUBRESOURCE_FOOTPRINT at = foot[plane];
            at.Offset += base;
            const auto dst = foot_loc(rb.p, at);
            const auto src = sub_loc(texture, plane);
            run.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            char label[96];
            std::snprintf(label, sizeof(label), "%s %u (value %.9g) %s plane", what, index,
                          index < values.size() ? double(values[index]) : 0.0, plane ? "stencil" : "depth");
            out.regions.push_back({label, size_t(at.Offset), at.Footprint.RowPitch, size_t(row_bytes[plane]), rows[plane],
                                   1, plane == 0});
        }
    };
    for (UINT k = 0; k < whole; ++k) copy_out(tex[k].p, tiny_foot, tiny_stride * k, tiny_rows, tiny_row_bytes, "whole-view clear", k);
    copy_out(wide_tex.p, wide_foot, wide_base, wide_rows, wide_row_bytes, "rectangle clears", 0);
    out.error = run.submit();
    out.bytes.assign(mapped, mapped + size_t(size));
    rb->Unmap(0, nullptr);
}

// An upload of random 24-bit words and stencil bytes into a D24 texture and a readback of both planes, optionally with
// a predicate that skips the copies (a zero word, EQUAL_ZERO). D3D12 would skip them; the layer runs copies
// unpredicated (docs section 4.7), and the conversion around its D24 copies must run too, so the layer's
// predicated run must read back what the host's unpredicated run does.
inline void run_depth24_predicated(Run& run, ID3D12Device* host, bool predicated, Outcome& out) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = 20;
    desc.Height = 12;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
    desc.SampleDesc.Count = 1;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT foot[2];
    UINT rows[2];
    UINT64 row_bytes[2], total = 0;
    host->GetCopyableFootprints(&desc, 0, 2, 0, foot, rows, row_bytes, &total);
    const UINT64 stride = (total + 511) / 512 * 512;
    Ptr<ID3D12Resource> up, rb, tex, predicate;
    if (!run.buffer(D3D12_HEAP_TYPE_UPLOAD, stride, D3D12_RESOURCE_STATE_GENERIC_READ, up) ||
        !run.buffer(D3D12_HEAP_TYPE_READBACK, stride, D3D12_RESOURCE_STATE_COPY_DEST, rb) ||
        !run.buffer(D3D12_HEAP_TYPE_UPLOAD, 256, D3D12_RESOURCE_STATE_GENERIC_READ, predicate) ||
        !run.texture(desc, D3D12_RESOURCE_STATE_COPY_DEST, tex)) {
        out.error = "resource creation failed";
        return;
    }
    std::uint8_t *upm = nullptr, *rbm = nullptr, *pm = nullptr;
    if (FAILED(up->Map(0, nullptr, reinterpret_cast<void**>(&upm))) || FAILED(rb->Map(0, nullptr, reinterpret_cast<void**>(&rbm))) ||
        FAILED(predicate->Map(0, nullptr, reinterpret_cast<void**>(&pm)))) {
        out.error = "Map failed";
        return;
    }
    std::memset(pm, 0, 256);
    std::memset(upm, 0, size_t(stride));
    std::memset(rbm, 0xcd, size_t(stride));
    for (UINT plane = 0; plane < 2; ++plane)
        for (UINT row = 0; row < rows[plane]; ++row)
            for (UINT64 col = 0; col < row_bytes[plane]; ++col) {
                const bool x8 = plane == 0 && col % 4 == 3;
                upm[size_t(foot[plane].Offset) + row * foot[plane].Footprint.RowPitch + size_t(col)] =
                    x8 ? 0 : fill_byte(77u + plane, row * 131 + size_t(col), false);
            }
    if (predicated) run.list->SetPredication(predicate.p, 0, D3D12_PREDICATION_OP_EQUAL_ZERO);
    for (UINT plane = 0; plane < 2; ++plane) {
        const auto dst = sub_loc(tex.p, plane), src = foot_loc(up.p, foot[plane]);
        run.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }
    run.barrier(tex.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
    for (UINT plane = 0; plane < 2; ++plane) {
        const auto dst = foot_loc(rb.p, foot[plane]), src = sub_loc(tex.p, plane);
        run.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        out.regions.push_back({plane ? "stencil plane" : "depth plane", size_t(foot[plane].Offset),
                               foot[plane].Footprint.RowPitch, size_t(row_bytes[plane]), rows[plane], 1, plane == 0});
    }
    if (predicated) run.list->SetPredication(nullptr, 0, D3D12_PREDICATION_OP_EQUAL_ZERO);
    out.error = run.submit();
    out.bytes.assign(rbm, rbm + size_t(stride));
    rb->Unmap(0, nullptr);
}

// How many clear values read back (depth plane or stencil plane) as something other than the reference: the 24-bit
// word of depth24_quantize for the depth, the stencil value the scenario cleared. `o.regions` is the order
// run_depth24_clears made: depth and stencil of each whole-view texture, then the wide texture's.
inline size_t depth24_reference_deviations(const Outcome& o, const std::vector<float>& values, size_t* first = nullptr) {
    const size_t whole = std::min<size_t>(values.size(), 160);
    const auto word = [&](const Region& r, size_t row, size_t col) {
        std::uint32_t w = 0;
        std::memcpy(&w, &o.bytes[r.offset + row * r.pitch + col * 4], 4);
        return w & 0xffffffu;  // the X8 byte is unspecified
    };
    size_t bad = 0;
    const auto note = [&](size_t k) {
        if (!bad++ && first) *first = k;
    };
    for (size_t k = 0; k < values.size(); ++k) {
        const std::uint32_t want = leonida::depth24_quantize(values[k]);
        bool ok = true;
        if (k < whole) {
            const Region &d = o.regions[2 * k], &s = o.regions[2 * k + 1];
            for (size_t row = 0; row < 4; ++row)
                for (size_t col = 0; col < 4; ++col)
                    ok = ok && word(d, row, col) == want && o.bytes[s.offset + row * s.pitch + col] == std::uint8_t(0x40 + k % 64);
        }
        const Region &wd = o.regions[2 * whole], &ws = o.regions[2 * whole + 1];
        for (size_t row = 0; row < 4; ++row)
            ok = ok && word(wd, row, k) == want && o.bytes[ws.offset + row * ws.pitch + k] == 0x11;
        if (!ok) note(k);
    }
    return bad;
}

// The clears of one format on the layer and on the host. The layer must read back what the reference says (the
// spec's FLOAT -> UNORM as WARP and AMD round it, depth24.h). The host is compared byte for byte when it agrees with
// the reference; where it does not (NVIDIA's D3D12 driver rounds some ties the other way) that is logged as a host
// quirk and not a failure. Returns the failures.
inline int compare_depth24_clears(const char* name, const Outcome& a, const Outcome& b, const std::vector<float>& values) {
    if (!a.error.empty() || !b.error.empty()) return compare(name, a, b);  // prints which side failed
    size_t first_ours = 0, first_host = 0;
    const size_t ours = depth24_reference_deviations(a, values, &first_ours);
    const size_t host = depth24_reference_deviations(b, values, &first_host);
    if (ours) {
        std::fprintf(stderr, "copy case %s: the layer reads back %zu of %zu clear values differently from the reference "
                     "(first: value %.9g)\n", name, ours, values.size(), double(values[first_ours]));
        return 1;
    }
    if (host) {
        std::printf("copy case %s: host quirk: the host reads back %zu of %zu clear values differently from the "
                    "reference (first: value %.9g, a tie it rounds the other way); the layer matches the reference\n",
                    name, host, values.size(), double(values[first_host]));
        return 0;
    }
    return compare(name, a, b);
}

// Returns the number of differences, or -1 with no host device.
inline int compare_depth24(ID3D12Device* ours, bool d24_native) {
    Ptr<ID3D12Device> host;
    std::string adapter;
    if (!create_host(host, adapter)) return -1;
    const std::vector<float> values = depth24_clear_values();
    Outcome a, b;
    Run ours_run, host_run;
    if (!ours_run.init(ours, true) || !host_run.init(host.p, false)) {
        std::fprintf(stderr, "depth24 clears: could not create the queue and list\n");
        return 1;
    }
    run_depth24_clears(ours_run, host.p, values, a);
    run_depth24_clears(host_run, host.p, values, b);
    int r = compare_depth24_clears("D24 clears", a, b, values);
    if (!r) std::printf("depth24 clears: %zu values, whole-view and rectangle clears match the host (%zu regions)\n",
                        values.size(), b.regions.size());

    // R24G8_TYPELESS resource with a typed D24 DSV (and, on the layer, shader-resource views of both planes).
    const std::vector<float> few(values.begin(), values.begin() + 200);
    Outcome ta, tb;
    Run ours_typeless, host_typeless;
    if (!ours_typeless.init(ours, true) || !host_typeless.init(host.p, false)) return r + 1;
    run_depth24_clears(ours_typeless, host.p, few, ta, DXGI_FORMAT_R24G8_TYPELESS, d24_native);
    run_depth24_clears(host_typeless, host.p, few, tb, DXGI_FORMAT_R24G8_TYPELESS);
    const int typeless = compare_depth24_clears("R24G8_TYPELESS clears", ta, tb, few);
    if (!typeless) std::printf("depth24 typeless: R24G8_TYPELESS with a D24 DSV, %zu values, match the host\n", few.size());
    r += typeless;

    // Copies under a predicate that would skip them (the layer does not predicate copies; the host run does not set one).
    Outcome pa, pb, plain;
    Run ours_predicated, ours_plain, host_run2;
    if (!ours_predicated.init(ours, true) || !ours_plain.init(ours, true) || !host_run2.init(host.p, false)) return r + 1;
    run_depth24_predicated(host_run2, host.p, false, pb);
    run_depth24_predicated(ours_plain, host.p, false, plain);
    r += compare("D24 copies", plain, pb);
    run_depth24_predicated(ours_predicated, host.p, true, pa);
    if (!pa.error.empty() && pa.error.find("ReplayResult") != std::string::npos) {
        std::printf("depth24 predicated copies: the device has no conditional rendering, skipped (%s)\n", pa.error.c_str());
    } else {
        const int predicated = compare("D24 copies under a false predicate", pa, pb);
        if (!predicated) std::printf("depth24 predicated copies: upload and readback under a skipping predicate match the host's unpredicated run\n");
        r += predicated;
    }
    return r;
}

}  // namespace copycmp
