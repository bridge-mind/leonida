// ID3D12Resource::WriteToSubresource / ReadFromSubresource on the Vulkan-backed device, compared with the host PC
// runtime. The same script runs on both devices: textures in CPU-visible custom heaps (write-back and
// write-combine), a volume, an array, BC1 blocks, a buffer, an upload-heap buffer, a default-heap texture and a
// multisampled one, with whole and boxed copies, pitches too small or zero, boxes outside or empty or cutting a
// block, subresources past the last, and null pointers. Every HRESULT is recorded, and every read records a hash of
// its whole destination buffer (so bytes written outside the region count too). The two records must be the same,
// step by step, and the same as Microsoft's answers recorded in d3d12_subresource_copy_answers.h, which is what
// Wine and Proton check (their host D3D12 is vkd3d). A step the host runtime faults on is reported and not compared.
// No shaders or game data. Skips (77) without a Vulkan GPU. Set LEONIDA_D3D49_VERBOSE=1 to print every step, and
// LEONIDA_D3D49_DUMP=1 to print the host's record in the answers file's format (Windows only).
#include "vk_test_device.h"
#include "host_wine.h"
#include "d3d12_subresource_copy_answers.h"
#include "../shims/d3d12_com.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { reset(); }
    void reset() {
        if (p) p->Release();
        p = nullptr;
    }
    T* operator->() const { return p; }
};

constexpr std::uint64_t kFaulted = 0xFA017EDull;  // the call raised an exception
constexpr std::uint64_t kMissing = 0x5C1F1Dull;   // the resource couldn't be created on this device
constexpr std::size_t kBytes = 4096;

struct Resources {
    ID3D12Resource* tex = nullptr;     // 8x8 RGBA8, 2 mips, custom heap write-back L0
    ID3D12Resource* vol = nullptr;     // 4x4x4 RGBA8 volume, same heap
    ID3D12Resource* arr = nullptr;     // 4x4 RGBA8, 2 layers
    ID3D12Resource* bc1 = nullptr;     // 8x8 BC1
    ID3D12Resource* wc = nullptr;      // 4x4 RGBA8, custom heap write-combine L0
    ID3D12Resource* buffer = nullptr;  // 4096-byte buffer, custom heap write-back L0
    ID3D12Resource* upload = nullptr;  // 4096-byte buffer, upload heap
    ID3D12Resource* gpu = nullptr;     // 4x4 RGBA8, default heap
    ID3D12Resource* msaa = nullptr;    // 4x4 RGBA8 render target, 4 samples, default heap
    std::uint8_t src[kBytes];
    std::uint8_t dst[kBytes];
};

using Call = HRESULT (*)(Resources&);

struct Step {
    const char* name;
    ID3D12Resource* Resources::*resource;
    Call call;
    bool read;  // also record a hash of r.dst
};

HRESULT guarded(Call call, Resources* r) {
    __try {
        return call(*r);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return HRESULT(kFaulted);
    }
}

D3D12_BOX box(UINT left, UINT top, UINT front, UINT right, UINT bottom, UINT back) {
    return D3D12_BOX{left, top, front, right, bottom, back};
}

HRESULT write(ID3D12Resource* t, UINT sub, const D3D12_BOX* b, Resources& r, UINT row, UINT depth) {
    return t->WriteToSubresource(sub, b, r.src, row, depth);
}
HRESULT read(ID3D12Resource* t, UINT sub, const D3D12_BOX* b, Resources& r, UINT row, UINT depth) {
    std::memset(r.dst, 0xcd, kBytes);
    return t->ReadFromSubresource(r.dst, row, depth, sub, b);
}

#define W(res, sub, B, row, depth) [](Resources& r) -> HRESULT { const D3D12_BOX b = B; return write(r.res, sub, &b, r, row, depth); }
#define WN(res, sub, row, depth) [](Resources& r) -> HRESULT { return write(r.res, sub, nullptr, r, row, depth); }
#define R(res, sub, B, row, depth) [](Resources& r) -> HRESULT { const D3D12_BOX b = B; return read(r.res, sub, &b, r, row, depth); }
#define RN(res, sub, row, depth) [](Resources& r) -> HRESULT { return read(r.res, sub, nullptr, r, row, depth); }

const Step kSteps[] = {
    // Before any Map: D3D12 asks for the subresource to be mapped (with a null pointer) first.
    {"tex: write before Map", &Resources::tex, WN(tex, 0, 32, 0), false},
    {"tex: read before Map", &Resources::tex, RN(tex, 0, 32, 0), true},
    {"buffer: write before Map", &Resources::buffer, WN(buffer, 0, 64, 0), false},
    // Map(sub, nullptr) on every texture subresource the script uses comes next (in run()).
    {"tex: write mip 0 whole", &Resources::tex, WN(tex, 0, 32, 0), false},
    {"tex: write mip 1 whole", &Resources::tex, WN(tex, 1, 16, 0), false},
    {"tex: write subresource 2 (past the last)", &Resources::tex, WN(tex, 2, 32, 0), false},
    {"tex: write subresource 0xffffffff", &Resources::tex, WN(tex, 0xffffffffu, 32, 0), false},
    {"tex: write a 5x4 box at a 100-byte pitch", &Resources::tex, W(tex, 0, box(2, 1, 0, 7, 5, 1), 100, 0), false},
    {"tex: write a box past the right edge", &Resources::tex, W(tex, 0, box(0, 0, 0, 9, 1, 1), 100, 0), false},
    {"tex: write a box past the bottom edge", &Resources::tex, W(tex, 0, box(0, 0, 0, 1, 9, 1), 100, 0), false},
    {"tex: write a box with back 2", &Resources::tex, W(tex, 0, box(0, 0, 0, 1, 1, 2), 100, 0), false},
    {"tex: write a box with front 1, back 2", &Resources::tex, W(tex, 0, box(0, 0, 1, 1, 1, 2), 100, 0), false},
    {"tex: write an empty box (left == right)", &Resources::tex, W(tex, 0, box(2, 2, 0, 2, 3, 1), 100, 0), false},
    {"tex: write an empty box (top == bottom)", &Resources::tex, W(tex, 0, box(2, 2, 0, 3, 2, 1), 100, 0), false},
    {"tex: write an empty box (front == back)", &Resources::tex, W(tex, 0, box(2, 2, 0, 3, 3, 0), 100, 0), false},
    {"tex: write an inverted box", &Resources::tex, W(tex, 0, box(5, 0, 0, 3, 1, 1), 100, 0), false},
    {"tex: write an inverted box starting past the right edge", &Resources::tex,
     W(tex, 0, box(9, 0, 0, 3, 1, 1), 100, 0), false},
    {"tex: write an empty box at the right edge", &Resources::tex, W(tex, 0, box(8, 0, 0, 8, 1, 1), 100, 0), false},
    {"tex: write an empty box past the right edge", &Resources::tex, W(tex, 0, box(9, 0, 0, 9, 1, 1), 100, 0), false},
    {"tex: write an empty box past the bottom edge", &Resources::tex, W(tex, 0, box(0, 9, 0, 1, 9, 1), 100, 0), false},
    {"tex: write an empty box in subresource 2", &Resources::tex, W(tex, 2, box(0, 0, 0, 0, 1, 1), 100, 0), false},
    {"tex: read mip 0 after the empty and inverted boxes", &Resources::tex, RN(tex, 0, 32, 0), true},
    {"tex: write a box of mip 1 past its edge", &Resources::tex, W(tex, 1, box(0, 0, 0, 5, 1, 1), 100, 0), false},
    {"tex: write null data", &Resources::tex, [](Resources& r) -> HRESULT {
         return r.tex->WriteToSubresource(0, nullptr, nullptr, 32, 0);
     }, false},
    {"tex: write rows 16 apart (32-byte rows)", &Resources::tex, WN(tex, 0, 16, 0), false},
    {"tex: write rows 31 apart", &Resources::tex, WN(tex, 0, 31, 0), false},
    {"tex: write rows 0 apart", &Resources::tex, WN(tex, 0, 0, 0), false},
    {"tex: write one row, pitch 0", &Resources::tex, W(tex, 0, box(0, 3, 0, 8, 4, 1), 0, 0), false},
    {"tex: write one row, pitch 4", &Resources::tex, W(tex, 0, box(0, 3, 0, 8, 4, 1), 4, 0), false},
    {"tex: write one texel, pitch 0", &Resources::tex, W(tex, 0, box(7, 7, 0, 8, 8, 1), 0, 0), false},
    {"tex: write mip 0, depth pitch 1", &Resources::tex, WN(tex, 0, 32, 1), false},
    {"tex: write mip 0 at a 48-byte pitch", &Resources::tex, WN(tex, 0, 48, 0), false},
    {"tex: read null data", &Resources::tex, [](Resources& r) -> HRESULT {
         return r.tex->ReadFromSubresource(nullptr, 32, 0, 0, nullptr);
     }, false},
    {"tex: read subresource 2", &Resources::tex, RN(tex, 2, 32, 0), true},
    {"tex: read a box past the right edge", &Resources::tex, R(tex, 0, box(0, 0, 0, 9, 1, 1), 64, 0), true},
    {"tex: read an empty box", &Resources::tex, R(tex, 0, box(2, 2, 0, 2, 3, 1), 64, 0), true},
    {"tex: read rows 16 apart [overlap]", &Resources::tex, RN(tex, 0, 16, 0), true},
    {"tex: read rows 0 apart [overlap]", &Resources::tex, RN(tex, 0, 0, 0), true},
    {"tex: read mip 0 whole at a 40-byte pitch", &Resources::tex, RN(tex, 0, 40, 0), true},
    {"tex: read mip 1 whole", &Resources::tex, RN(tex, 1, 16, 0), true},
    {"tex: read a 2x3 box of mip 1", &Resources::tex, R(tex, 1, box(1, 1, 0, 3, 4, 1), 8, 0), true},
    {"tex: read one row, pitch 0", &Resources::tex, R(tex, 0, box(0, 5, 0, 8, 6, 1), 0, 0), true},

    {"vol: write whole, rows 16 and slices 64 apart", &Resources::vol, WN(vol, 0, 16, 64), false},
    {"vol: write a 3x4x3 box, rows 12 and slices 64 apart", &Resources::vol,
     W(vol, 0, box(1, 0, 1, 4, 4, 4), 12, 64), false},
    {"vol: write slices 32 apart (64-byte slices)", &Resources::vol, WN(vol, 0, 16, 32), false},
    {"vol: write slices 0 apart", &Resources::vol, WN(vol, 0, 16, 0), false},
    {"vol: write one slice, depth pitch 0", &Resources::vol, W(vol, 0, box(0, 0, 2, 4, 4, 3), 16, 0), false},
    {"vol: write a box with back 5", &Resources::vol, W(vol, 0, box(0, 0, 0, 4, 4, 5), 16, 64), false},
    {"vol: read whole, rows 20 and slices 96 apart", &Resources::vol, RN(vol, 0, 20, 96), true},
    {"vol: read slices 0 apart [overlap]", &Resources::vol, RN(vol, 0, 16, 0), true},

    {"arr: write layer 1", &Resources::arr, WN(arr, 1, 16, 0), false},
    {"arr: write subresource 2 (past the last)", &Resources::arr, WN(arr, 2, 16, 0), false},
    {"arr: write layer 0 with back 2", &Resources::arr, W(arr, 0, box(0, 0, 0, 4, 4, 2), 16, 64), false},
    {"arr: read layer 1", &Resources::arr, RN(arr, 1, 16, 0), true},
    {"arr: read layer 0", &Resources::arr, RN(arr, 0, 16, 0), true},

    {"bc1: write whole (2x2 blocks, rows 16 apart)", &Resources::bc1, WN(bc1, 0, 16, 0), false},
    {"bc1: write the bottom-right block", &Resources::bc1, W(bc1, 0, box(4, 4, 0, 8, 8, 1), 8, 0), false},
    {"bc1: write a box starting inside a block", &Resources::bc1, W(bc1, 0, box(2, 0, 0, 8, 4, 1), 16, 0), false},
    {"bc1: write a box ending inside a block", &Resources::bc1, W(bc1, 0, box(0, 0, 0, 6, 4, 1), 16, 0), false},
    {"bc1: write a box ending inside a block row", &Resources::bc1, W(bc1, 0, box(0, 0, 0, 4, 2, 1), 16, 0), false},
    {"bc1: write an empty box inside a block", &Resources::bc1, W(bc1, 0, box(2, 0, 0, 2, 4, 1), 16, 0), false},
    {"bc1: write an inverted box cutting blocks", &Resources::bc1, W(bc1, 0, box(6, 0, 0, 2, 4, 1), 16, 0), false},
    {"bc1: read whole after the empty boxes", &Resources::bc1, RN(bc1, 0, 32, 0), true},
    {"bc1: write rows 8 apart (16-byte block rows)", &Resources::bc1, WN(bc1, 0, 8, 0), false},
    {"bc1: write a block row, pitch 0", &Resources::bc1, W(bc1, 0, box(0, 4, 0, 8, 8, 1), 0, 0), false},
    {"bc1: read whole, rows 32 apart", &Resources::bc1, RN(bc1, 0, 32, 0), true},

    {"wc: write whole", &Resources::wc, WN(wc, 0, 16, 0), false},
    {"wc: read whole", &Resources::wc, RN(wc, 0, 16, 0), true},

    {"buffer: write", &Resources::buffer, WN(buffer, 0, 64, 0), false},
    {"buffer: write a 16-byte box", &Resources::buffer, W(buffer, 0, box(0, 0, 0, 16, 1, 1), 16, 0), false},
    {"buffer: read", &Resources::buffer, RN(buffer, 0, 64, 0), true},
    {"upload: write", &Resources::upload, WN(upload, 0, 64, 0), false},
    {"upload: read", &Resources::upload, RN(upload, 0, 64, 0), true},

    {"gpu: write (default heap)", &Resources::gpu, WN(gpu, 0, 16, 0), false},
    {"gpu: read (default heap)", &Resources::gpu, RN(gpu, 0, 16, 0), true},
    {"msaa: write (4 samples)", &Resources::msaa, WN(msaa, 0, 16, 0), false},
    {"msaa: read (4 samples)", &Resources::msaa, RN(msaa, 0, 16, 0), true},
};
constexpr std::size_t kBeforeMap = 3;  // the first steps run before Map

#undef W
#undef WN
#undef R
#undef RN

std::uint64_t fnv(const std::uint8_t* p, std::size_t n) {
    std::uint64_t h = 1469598103934665603ull;
    for (std::size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 1099511628211ull;
    return h;
}

D3D12_RESOURCE_DESC texture(UINT width, UINT height, UINT16 depth, UINT16 mips, DXGI_FORMAT format,
                            D3D12_RESOURCE_DIMENSION dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = dimension;
    d.Width = width;
    d.Height = height;
    d.DepthOrArraySize = depth;
    d.MipLevels = mips;
    d.Format = format;
    d.SampleDesc.Count = 1;
    return d;
}

D3D12_RESOURCE_DESC buffer(UINT64 bytes) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width = bytes;
    d.Height = 1;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return d;
}

struct Record {
    std::vector<std::string> names;
    std::vector<std::uint64_t> values;
    void add(const std::string& name, std::uint64_t value) {
        names.push_back(name);
        values.push_back(value);
    }
};

// Creates the resources on `device`, runs the script and records it. A resource the device refuses is recorded
// as missing, and its steps too.
bool run(ID3D12Device* device, bool bc, Record& out) {
    const D3D12_HEAP_PROPERTIES write_back{D3D12_HEAP_TYPE_CUSTOM, D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
                                           D3D12_MEMORY_POOL_L0};
    const D3D12_HEAP_PROPERTIES write_combine{D3D12_HEAP_TYPE_CUSTOM, D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE,
                                              D3D12_MEMORY_POOL_L0};
    const D3D12_HEAP_PROPERTIES upload{D3D12_HEAP_TYPE_UPLOAD};
    const D3D12_HEAP_PROPERTIES gpu{D3D12_HEAP_TYPE_DEFAULT};
    static Resources r;  // too big for the stack frame __try lives next to
    std::memset(&r, 0, sizeof(r));
    for (std::size_t i = 0; i < kBytes; ++i) r.src[i] = std::uint8_t(i * 7 + (i >> 8) * 13 + 1);
    const auto create = [&](const char* what, const D3D12_HEAP_PROPERTIES& heap, const D3D12_RESOURCE_DESC& desc,
                            D3D12_RESOURCE_STATES state, ID3D12Resource*& to) {
        const HRESULT hr = device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                                           IID_PPV_ARGS(&to));
        out.add(std::string("create ") + what, std::uint64_t(std::uint32_t(hr)));
        if (FAILED(hr)) to = nullptr;
    };
    create("tex", write_back, texture(8, 8, 1, 2, DXGI_FORMAT_R8G8B8A8_UNORM), D3D12_RESOURCE_STATE_COMMON, r.tex);
    create("vol", write_back, texture(4, 4, 4, 1, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_DIMENSION_TEXTURE3D),
           D3D12_RESOURCE_STATE_COMMON, r.vol);
    create("arr", write_back, texture(4, 4, 2, 1, DXGI_FORMAT_R8G8B8A8_UNORM), D3D12_RESOURCE_STATE_COMMON, r.arr);
    if (bc) create("bc1", write_back, texture(8, 8, 1, 1, DXGI_FORMAT_BC1_UNORM), D3D12_RESOURCE_STATE_COMMON, r.bc1);
    create("wc", write_combine, texture(4, 4, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM), D3D12_RESOURCE_STATE_COMMON, r.wc);
    create("buffer", write_back, buffer(kBytes), D3D12_RESOURCE_STATE_COMMON, r.buffer);
    create("upload", upload, buffer(kBytes), D3D12_RESOURCE_STATE_GENERIC_READ, r.upload);
    create("gpu", gpu, texture(4, 4, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM), D3D12_RESOURCE_STATE_COMMON, r.gpu);
    auto ms = texture(4, 4, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM);
    ms.SampleDesc.Count = 4;
    ms.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    create("msaa", gpu, ms, D3D12_RESOURCE_STATE_RENDER_TARGET, r.msaa);

    const auto step = [&](const Step& s) {
        if (!(r.*s.resource)) {
            out.add(s.name, kMissing);
            if (s.read) out.add(std::string(s.name) + " (bytes)", kMissing);
            return;
        }
        const HRESULT hr = guarded(s.call, &r);
        out.add(s.name, std::uint64_t(std::uint32_t(hr)));
        if (s.read) out.add(std::string(s.name) + " (bytes)", hr == HRESULT(kFaulted) ? kFaulted : fnv(r.dst, kBytes));
    };
    for (std::size_t i = 0; i < kBeforeMap; ++i) step(kSteps[i]);
    // Map with a null pointer, as D3D12 asks before a CPU copy: every texture subresource in a CPU-visible heap.
    const struct { const char* name; ID3D12Resource* res; UINT subs; } maps[] = {
        {"tex", r.tex, 2}, {"vol", r.vol, 1},    {"arr", r.arr, 2},      {"bc1", r.bc1, 1},
        {"wc", r.wc, 1},   {"buffer", r.buffer, 1}, {"upload", r.upload, 1}};
    for (const auto& m : maps)
        for (UINT s = 0; m.res && s < m.subs; ++s)
            out.add(std::string("Map ") + m.name + " " + std::to_string(s),
                    std::uint64_t(std::uint32_t(m.res->Map(s, nullptr, nullptr))));
    for (std::size_t i = kBeforeMap; i < sizeof(kSteps) / sizeof(kSteps[0]); ++i) step(kSteps[i]);
    for (const auto& m : maps)
        for (UINT s = 0; m.res && s < m.subs; ++s) m.res->Unmap(s, nullptr);
    for (ID3D12Resource* res : {r.tex, r.vol, r.arr, r.bc1, r.wc, r.buffer, r.upload, r.gpu, r.msaa})
        if (res) res->Release();
    return true;
}

// Compares the layer's record with a reference one (the host runtime's, or Microsoft's recorded answers).
int compare(const char* path, const char* against, const Record& ours, const Record& ref, bool verbose) {
    if (ref.names != ours.names) {
        std::fprintf(stderr, "CPU copies (%s): the steps differ from %s's\n", path, against);
        return 1;
    }
    int mismatches = 0, skipped = 0;
    const auto show = [&](std::size_t i, const char* note) {
        std::printf("CPU copies (%s) %s: ours 0x%llx, %s 0x%llx%s\n", path, ours.names[i].c_str(),
                    static_cast<unsigned long long>(ours.values[i]), against,
                    static_cast<unsigned long long>(ref.values[i]), note);
    };
    for (std::size_t i = 0; i < ours.values.size(); ++i) {
        if (ref.values[i] == kFaulted) {  // null data: the host faults
            if (verbose) show(i, " (the host faults; not compared)");
            ++skipped;
            continue;
        }
        // A default-heap texture: the host refuses CPU copies there, the layer makes them and logs it
        // (docs/d3d12_vulkan.md), on purpose.
        if (ours.names[i].find("(default heap)") != std::string::npos) {
            if (verbose) show(i, " (the layer copies on purpose; not compared)");
            ++skipped;
            continue;
        }
        // A read whose rows or slices overlap in the destination: which bytes win is the order the driver
        // copies texels in (tiled on an RTX 3060, not row by row), so only its HRESULT is compared.
        if (ours.names[i].find("[overlap] (bytes)") != std::string::npos) {
            if (verbose) show(i, " (the driver's copy order; not compared)");
            ++skipped;
            continue;
        }
        const bool same = ours.values[i] == ref.values[i];
        if (verbose || !same) show(i, same ? "" : "  MISMATCH");
        mismatches += !same;
    }
    std::printf("CPU copies (%s): %zu steps compared with %s (%d more not compared), %d mismatches\n", path,
                ours.values.size() - skipped, against, skipped, mismatches);
    return mismatches;
}

}  // namespace

int main() {
    vktest::Device d;
    const auto device_status = d.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) return 77;  // no Vulkan GPU: ctest skips
    if (device_status != vktest::DeviceStatus::ready) return 1;
    const bool verbose = GetEnvironmentVariableA("LEONIDA_D3D49_VERBOSE", nullptr, 0) > 0;
    const bool dump = GetEnvironmentVariableA("LEONIDA_D3D49_DUMP", nullptr, 0) > 0;

    const HMODULE d3d12 = LoadLibraryExA("d3d12.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    const auto create_host = d3d12 ? reinterpret_cast<PFN_D3D12_CREATE_DEVICE>(
                                         reinterpret_cast<void*>(GetProcAddress(d3d12, "D3D12CreateDevice")))
                                   : nullptr;
    Ref<ID3D12Device> host;
    Record theirs;
    if (create_host && !hostwine::host_is_wine() &&
        SUCCEEDED(create_host(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p))))
        EXPECT(run(host.p, d.bc, theirs));
    // Without BC formats there is no BC1 texture: no "create bc1" or "Map bc1" record, and its steps are missing.
    Record recorded;
    for (const SubresourceCopyAnswer& a : kSubresourceCopyAnswers) {
        const std::string name = a.name;
        if (d.bc || !a.bc) recorded.add(name, a.value);
        else if (name.rfind("bc1:", 0) == 0) recorded.add(name, kMissing);
    }

    // Once per copy path: on the host with VK_EXT_host_image_copy where the GPU has it, and through a staging submit
    // (a second D3D12 device on the same Vulkan device, told the extension is off).
    for (const bool host_copy : {true, false}) {
        const char* path = host_copy ? "host image copy" : "staging copy";
        if (host_copy && !d.host_image_copy) {
            std::printf("CPU copies (%s): no VK_EXT_host_image_copy, skipped\n", path);
            continue;
        }
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = d.instance_proc;
        info.instance = d.instance;
        info.physical = d.physical;
        info.device = d.device;
        info.queue = d.queue;
        info.queue_family = d.family;
        info.texture_compression_bc = d.bc;
        info.host_image_copy = host_copy;
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p))));
        Record ours;
        EXPECT(run(device.p, d.bc, ours));
        if (dump) {
            if (!host.p) {
                std::fprintf(stderr, "LEONIDA_D3D49_DUMP prints Microsoft's runtime's record: run it on Windows\n");
                return 1;
            }
            for (std::size_t i = 0; i < theirs.values.size(); ++i)
                std::printf("    {\"%s\", 0x%llxull, %s},\n", theirs.names[i].c_str(),
                            static_cast<unsigned long long>(theirs.values[i]),
                            theirs.names[i].find("bc1") != std::string::npos ? "true" : "false");
            return 0;
        }
        if (host.p) EXPECT(compare(path, "the host", ours, theirs, verbose) == 0);
        EXPECT(compare(path, "Microsoft's recorded answers", ours, recorded, verbose) == 0);
    }
    std::printf("d3d12_subresource_copy_test: OK\n");
    return 0;
}
