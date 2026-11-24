// TEST-528: a real boot-movie frame through the whole present path, on pixels.
//
// Decodes two frames of a real .bk2 through the package's own Bink2Scarlett.dll
// (rt::Loader, the bink_decode_test pattern), uploads each BGRA frame to a
// committed ALLOW_DISPLAY texture through the Xbox-ABI proxy (the
// d3d12_m2_textured_test CopyTextureRegion pattern), presents it with
// PresentX through the in-process XboxPresentation (the xbox_present_test
// pattern), and compares the presented pixels with the decoded frame.
//
//   LEONIDA_GAME_DIR=<package root>  (contains Bink2Scarlett.dll)
//   LEONIDA_MOVIE=<file.bk2>         (a carved Bink 2 stream)
//   LEONIDA_MOVIE_FRAME=<frame>      (optional, default 0; a second frame,
//                                     200, is also tested when the movie has it)
//
// With LEONIDA_GAME_DIR or LEONIDA_MOVIE unset the test prints
// "SKIP (no package/movie)" and exits 77 (CTest skip), so CI passes without
// the package. Only movie properties (size, frame count, fps) are reported,
// never the file name. Nothing is written anywhere: every comparison is in
// memory, so no game data can end up in the repo or in %TEMP%.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "../shims/xbox_present.h"

#include "../shims/shims.h"
#include "../src/pe_loader.h"

#include <windows.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

int g_failures = 0;

void check(bool condition, const std::string& what) {
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what.c_str());
    if (!condition) ++g_failures;
}

[[noreturn]] void fatal(const std::string& what) {
    std::fprintf(stderr, "FAIL %s\n", what.c_str());
    std::exit(1);
}

std::string env(const char* name) {
    const char* v = std::getenv(name);
    return v ? v : "";
}

uint32_t read_u32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

struct MovieHeader {
    uint32_t frames = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t fps_num = 0;
    uint32_t fps_den = 0;
};

MovieHeader read_movie_header(const std::filesystem::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) fatal("cannot open LEONIDA_MOVIE");
    uint8_t h[0x2C]{};
    f.read(reinterpret_cast<char*>(h), sizeof(h));
    if (f.gcount() != (std::streamsize)sizeof(h)) fatal("movie is shorter than the 44-byte Bink 2 header");
    if (h[0] != 'K' || h[1] != 'B' || h[2] != '2' || h[3] < 'a' || h[3] > 'z')
        fatal("movie does not start with KB2[a-z]");
    MovieHeader out;
    out.frames = read_u32(h + 0x08);
    out.width = read_u32(h + 0x14);
    out.height = read_u32(h + 0x18);
    out.fps_num = read_u32(h + 0x1C);
    out.fps_den = read_u32(h + 0x20);
    if (!out.frames || !out.width || !out.height || !out.fps_num || !out.fps_den)
        fatal("movie header has a zero frame count, size or fps");
    return out;
}

uint64_t fnv1a(const std::vector<uint8_t>& bytes) {
    uint64_t h = 1469598103934665603ull;
    for (uint8_t b : bytes) {
        h ^= b;
        h *= 1099511628211ull;
    }
    return h;
}

size_t distinct_bytes(const std::vector<uint8_t>& bytes) {
    bool seen[256]{};
    size_t n = 0;
    for (uint8_t b : bytes)
        if (!seen[b]) {
            seen[b] = true;
            ++n;
        }
    return n;
}

struct BinkApi {
    using Open = void* (*)(const char*, uint32_t);
    using Wait = uint32_t (*)(void*);
    using DoFrame = uint32_t (*)(void*);
    using CopyToBuffer = uint32_t (*)(void*, void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
    using NextFrame = void (*)(void*);
    using Goto = void (*)(void*, uint32_t, uint32_t);
    using Close = void (*)(void*);
    using GetError = const char* (*)();

    Open open = nullptr;
    Wait wait = nullptr;
    DoFrame do_frame = nullptr;
    CopyToBuffer copy = nullptr;
    NextFrame next = nullptr;
    Goto go_to = nullptr;
    Close close = nullptr;
    GetError get_error = nullptr;
};

void* symbol(rt::Module& m, const char* name) {
    auto it = m.exports.find(name);
    if (it == m.exports.end()) fatal(std::string("Bink2Scarlett.dll does not export ") + name);
    return it->second;
}

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

LRESULT CALLBACK window_proc(HWND window, UINT message, WPARAM w, LPARAM l) {
    return DefWindowProcW(window, message, w, l);
}

void pump() {
    MSG message;
    while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE)) {
        TranslateMessage(&message);
        DispatchMessage(&message);
    }
}

// The swapchain frame as RGB, whatever its channel order.
struct Frame {
    std::vector<std::uint8_t> bytes;
    VkExtent2D extent{};
    VkFormat format = VK_FORMAT_UNDEFINED;
    bool bgr() const { return format == VK_FORMAT_B8G8R8A8_UNORM || format == VK_FORMAT_B8G8R8A8_SRGB; }
};

struct RGB {
    int r, g, b;
};

RGB presented_at(const Frame& f, UINT x, UINT y) {
    const std::uint8_t* p = &f.bytes[(std::size_t(y) * f.extent.width + x) * 4];
    return f.bgr() ? RGB{p[2], p[1], p[0]} : RGB{p[0], p[1], p[2]};
}

RGB decoded_at(const std::vector<uint8_t>& bgra, UINT w, UINT x, UINT y) {
    const std::uint8_t* p = &bgra[(std::size_t(y) * w + x) * 4];
    return RGB{p[2], p[1], p[0]};
}

// The decoded frame as the presenter's scaling blit samples it at the centre of
// destination texel (x, y). vkCmdBlitImage maps the destination centre through
// normalized coordinates onto the source with CLAMP_TO_EDGE (so a sample outside
// the frame repeats the edge texel) and then filters: linearly when the source
// format reports VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT, the nearest
// texel otherwise, which is the presenter's choice (runtime/src/vulkan_present.h,
// blit_filter). At 1:1 the centre lands on a texel centre and this reduces to
// decoded_at(x, y).
RGB sampled_at(const std::vector<uint8_t>& bgra, UINT W, UINT H, UINT ew, UINT eh, UINT x, UINT y, bool linear) {
    const double sx = ((double)x + 0.5) * double(W) / double(ew) - 0.5;
    const double sy = ((double)y + 0.5) * double(H) / double(eh) - 0.5;
    if (!linear) {
        const int nx = std::clamp((int)std::floor(sx + 0.5), 0, (int)W - 1);
        const int ny = std::clamp((int)std::floor(sy + 0.5), 0, (int)H - 1);
        return decoded_at(bgra, W, (UINT)nx, (UINT)ny);
    }
    const int x0 = (int)std::floor(sx), y0 = (int)std::floor(sy);
    const double fx = sx - x0, fy = sy - y0;
    const auto texel = [&](int tx, int ty) {
        return decoded_at(bgra, W, (UINT)std::clamp(tx, 0, (int)W - 1), (UINT)std::clamp(ty, 0, (int)H - 1));
    };
    const RGB t00 = texel(x0, y0), t10 = texel(x0 + 1, y0);
    const RGB t01 = texel(x0, y0 + 1), t11 = texel(x0 + 1, y0 + 1);
    const auto mix = [&](int a, int b, int c, int d) {
        const double v = (1 - fx) * (1 - fy) * a + fx * (1 - fy) * b + (1 - fx) * fy * c + fx * fy * d;
        return (int)std::lround(v);
    };
    return RGB{mix(t00.r, t10.r, t01.r, t11.r), mix(t00.g, t10.g, t01.g, t11.g),
               mix(t00.b, t10.b, t01.b, t11.b)};
}

D3D12_RESOURCE_DESC buffer_desc(UINT64 bytes) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width = bytes;
    d.Height = 1;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.Format = DXGI_FORMAT_UNKNOWN;
    d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return d;
}

D3D12_RESOURCE_BARRIER transition(ID3D12Resource* resource, D3D12_RESOURCE_STATES before,
                                 D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = resource;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = before;
    b.Transition.StateAfter = after;
    return b;
}

}  // namespace

int main() {
    const std::string game_dir_arg = env("LEONIDA_GAME_DIR");
    const std::string movie_arg = env("LEONIDA_MOVIE");
    if (game_dir_arg.empty() || movie_arg.empty()) {
        std::printf("SKIP (no package/movie)\n");
        return 77;
    }

    const std::filesystem::path game_dir = game_dir_arg;
    const std::filesystem::path movie = movie_arg;
    std::error_code ec;
    if (!std::filesystem::is_directory(game_dir, ec)) fatal("LEONIDA_GAME_DIR is not a directory");
    if (!std::filesystem::is_regular_file(game_dir / "Bink2Scarlett.dll", ec))
        fatal("LEONIDA_GAME_DIR does not contain Bink2Scarlett.dll");
    if (!std::filesystem::is_regular_file(movie, ec)) fatal("LEONIDA_MOVIE is not a file");

    const MovieHeader header = read_movie_header(movie);
    const UINT W = header.width, H = header.height;
    // Properties only, never the file name.
    std::printf("movie: %ux%u %u frames %u/%u fps\n", W, H, header.frames, header.fps_num, header.fps_den);

    uint32_t first = 0;
    const std::string frame_arg = env("LEONIDA_MOVIE_FRAME");
    if (!frame_arg.empty()) {
        char* end = nullptr;
        const unsigned long v = std::strtoul(frame_arg.c_str(), &end, 10);
        if (!end || *end || v >= header.frames) fatal("LEONIDA_MOVIE_FRAME is not a frame in this movie");
        first = (uint32_t)v;
    }
    // A second frame far from the first, when the movie has it.
    uint32_t second = UINT32_MAX;
    for (uint32_t candidate : {200u, header.frames - 1, 1u}) {
        if (candidate < header.frames && candidate != first) {
            second = candidate;
            break;
        }
    }
    const bool has_second = second != UINT32_MAX && second != first;
    std::printf("frames under test: %u%s\n", first, has_second ? (" and " + std::to_string(second)).c_str() : " (only)");

    std::vector<uint8_t> decoded_first((size_t)W * H * 4), decoded_second;
    if (has_second) decoded_second.resize((size_t)W * H * 4);
    uint64_t hash_first = 0, hash_second = 0;

    try {
        rt::Loader loader(game_dir);
        rt::Module& bink = loader.load(game_dir / "Bink2Scarlett.dll");
        std::printf("mapped %s at %p, size %u%s\n", bink.name.c_str(), bink.base, bink.size,
                    bink.relocated ? " (relocated)" : "");

        int shim = 0, host = 0, module = 0, unimplemented = 0;
        for (auto& r : bink.imports) {
            switch (r.binding) {
                case rt::Binding::Shim: ++shim; break;
                case rt::Binding::Host: ++host; break;
                case rt::Binding::Module: ++module; break;
                default: ++unimplemented; break;
            }
            if (rt::lower(r.dll) == "xmem.dll") {
                check(r.binding == rt::Binding::Shim, "xmem.dll!" + r.name + " binds to the shim");
                void* expected = rt::shims::find("xmem.dll", r.name);
                auto* dos = reinterpret_cast<IMAGE_DOS_HEADER*>(bink.base);
                auto* nt = reinterpret_cast<IMAGE_NT_HEADERS64*>(bink.base + dos->e_lfanew);
                auto& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
                bool iat_ok = false;
                if (dir.VirtualAddress) {
                    auto* desc = reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR*>(bink.base + dir.VirtualAddress);
                    for (; desc->Name; ++desc) {
                        const char* dll = reinterpret_cast<const char*>(bink.base + desc->Name);
                        if (rt::lower(dll) != "xmem.dll") continue;
                        auto* lookup = reinterpret_cast<IMAGE_THUNK_DATA64*>(
                            bink.base + (desc->OriginalFirstThunk ? desc->OriginalFirstThunk : desc->FirstThunk));
                        auto* iat = reinterpret_cast<IMAGE_THUNK_DATA64*>(bink.base + desc->FirstThunk);
                        for (; lookup->u1.AddressOfData; ++lookup, ++iat) {
                            if (IMAGE_SNAP_BY_ORDINAL64(lookup->u1.Ordinal)) continue;
                            auto* name = reinterpret_cast<IMAGE_IMPORT_BY_NAME*>(
                                bink.base + (uint32_t)lookup->u1.AddressOfData);
                            if (r.name == name->Name)
                                iat_ok = expected && *static_cast<void**>(reinterpret_cast<void*>(&iat->u1.Function)) ==
                                                         expected;
                        }
                    }
                }
                check(iat_ok, "xmem.dll!" + r.name + " IAT points at rt::shims::find");
            }
            if (r.binding == rt::Binding::Unimplemented &&
                rt::lower(r.dll).find("xaudio") == std::string::npos)
                fatal("non-audio unimplemented import " + r.dll + "!" + r.name);
        }
        std::printf("imports: shim=%d host=%d module=%d unimplemented=%d\n", shim, host, module, unimplemented);

        BinkApi api;
        api.open = reinterpret_cast<BinkApi::Open>(symbol(bink, "BinkOpen"));
        api.wait = reinterpret_cast<BinkApi::Wait>(symbol(bink, "BinkWait"));
        api.do_frame = reinterpret_cast<BinkApi::DoFrame>(symbol(bink, "BinkDoFrame"));
        api.copy = reinterpret_cast<BinkApi::CopyToBuffer>(symbol(bink, "BinkCopyToBuffer"));
        api.next = reinterpret_cast<BinkApi::NextFrame>(symbol(bink, "BinkNextFrame"));
        api.go_to = reinterpret_cast<BinkApi::Goto>(symbol(bink, "BinkGoto"));
        api.close = reinterpret_cast<BinkApi::Close>(symbol(bink, "BinkClose"));
        api.get_error = reinterpret_cast<BinkApi::GetError>(symbol(bink, "BinkGetError"));

        void* handle = api.open(movie.string().c_str(), 0);
        if (!handle) {
            const char* error = api.get_error ? api.get_error() : "";
            std::printf("FAIL BinkOpen returned null: %s\n", error && *error ? error : "(no BinkGetError text)");
            return 1;
        }
        std::printf("PASS BinkOpen returned %p\n", handle);

        const uint32_t pitch = W * 4;
        uint32_t positioned = 0;  // the frame the handle sits on after BinkOpen
        auto decode = [&](uint32_t index, std::vector<uint8_t>& out, const char* label) {
            if (index != positioned) {
                api.go_to(handle, index, 0);
                positioned = index;
            }
            api.wait(handle);
            api.do_frame(handle);
            // This GDK build returns 0 from BinkDoFrame/BinkCopyToBuffer even when it writes a valid frame; the
            // sentinel replacement and the hash below are the proof, not the return values.
            std::fill(out.begin(), out.end(), (uint8_t)0xCD);
            api.copy(handle, out.data(), pitch, H, 0, 0, 3);
            const size_t untouched = (size_t)std::count(out.begin(), out.end(), 0xCD);
            const size_t unique = distinct_bytes(out);
            const uint64_t hash = fnv1a(out);
            std::printf("%s frame %u: untouched=%zu unique=%zu hash=%016llx\n", label, index, untouched, unique,
                        (unsigned long long)hash);
            check(untouched < out.size(), std::string(label) + " BinkCopyToBuffer replaced the sentinel buffer");
            check(unique > 1, std::string(label) + " wrote more than one byte value");
            return hash;
        };

        hash_first = decode(first, decoded_first, "first");
        if (has_second) hash_second = decode(second, decoded_second, "second");
        if (has_second) {
            std::printf("decoded frame %u vs %u: %s\n", first, second,
                        hash_first == hash_second ? "identical" : "different");
            check(hash_first != hash_second, "the two decoded frames differ (the pipeline carries real frames)");
        }
        api.close(handle);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
    if (g_failures) {
        std::printf("RESULT FAIL (decode failed, no present attempted)\n");
        return 1;
    }
    std::printf("decode: ok, the texture round-trip and present come next\n");

    // ---- GPU: the Xbox-ABI proxy device, a movie-sized window, PresentX, capture. ----
    vktest::Device d;
    const auto device_status = d.initialize(true);
    if (device_status == vktest::DeviceStatus::unavailable) {
        std::printf("SKIP: no usable Vulkan presentation device\n");
        return 77;
    }
    EXPECT(device_status == vktest::DeviceStatus::ready);

    WNDCLASSW window_class{};
    window_class.lpfnWndProc = window_proc;
    window_class.hInstance = GetModuleHandleW(nullptr);
    window_class.lpszClassName = L"LeonidaBootMovieTest";
    window_class.hCursor = LoadCursorW(nullptr, reinterpret_cast<LPCWSTR>(IDC_ARROW));
    EXPECT(RegisterClassW(&window_class));
    RECT frame_rect{0, 0, LONG(W), LONG(H)};
    AdjustWindowRect(&frame_rect, WS_OVERLAPPEDWINDOW, FALSE);
    HWND window = CreateWindowExW(0, window_class.lpszClassName, L"Leonida: boot movie on Vulkan", WS_OVERLAPPEDWINDOW,
                                  0, 0, frame_rect.right - frame_rect.left, frame_rect.bottom - frame_rect.top,
                                  nullptr, nullptr, window_class.hInstance, nullptr);
    if (!window) {
        std::fprintf(stderr, "FAIL: cannot create the presentation window (Win32 error %lu)\n", GetLastError());
        return 1;
    }
    ShowWindow(window, SW_SHOWNOACTIVATE);
    UpdateWindow(window);
    pump();
    {
        RECT client{};
        GetClientRect(window, &client);
        std::printf("window client: %ldx%ld (movie %ux%u)\n", client.right - client.left,
                    client.bottom - client.top, W, H);
    }

    Frame presented_first, presented_second;
    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = d.instance_proc;
        info.instance = d.instance;
        info.physical = d.physical;
        info.device = d.device;
        info.queue = d.queue;
        info.queue_family = d.family;
        info.presentation = true;
        info.swapchain_maintenance1 = d.swapchain_maintenance1;
        ID3D12Device* inner = nullptr;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&inner))));
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12_xs::create_device_proxy(inner, __uuidof(ID3D12Device),
                                                            reinterpret_cast<void**>(&device.p))));
        rt::d3d12::set_device_outer(inner, device.p);
        auto presenter = rt::d3d12::create_window_presenter(inner, window);
        EXPECT(presenter);
        EXPECT(presenter->enable_capture());

        leonida::ManualFrameClock clock;
        auto owner = std::make_unique<rt::d3d12::XboxPresentation>(std::move(presenter), &clock);
        rt::d3d12::XboxPresentation& xbox = *owner;

        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        Ref<ID3D12CommandQueue> queue;
        Ref<ID3D12CommandAllocator> allocator;
        Ref<ID3D12GraphicsCommandList> list;
        Ref<ID3D12Fence> fence;
        EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
        EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
        EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                                   IID_PPV_ARGS(&list.p))));
        EXPECT(SUCCEEDED(list->Close()));
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
        HANDLE fence_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        EXPECT(fence_event != nullptr);

        // The movie frame as a display-capable texture: committed, ALLOW_DISPLAY, in PRESENT.
        const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
        D3D12_RESOURCE_DESC tex_desc{};
        tex_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        tex_desc.Width = W;
        tex_desc.Height = H;
        tex_desc.DepthOrArraySize = 1;
        tex_desc.MipLevels = 1;
        tex_desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;  // the BinkCopyToBuffer byte order above
        tex_desc.SampleDesc.Count = 1;
        tex_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        Ref<ID3D12Resource> texture;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_ALLOW_DISPLAY, &tex_desc,
                                                         D3D12_RESOURCE_STATE_PRESENT, nullptr,
                                                         IID_PPV_ARGS(&texture.p))));

        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
        UINT64 total_bytes = 0;
        device->GetCopyableFootprints(&tex_desc, 0, 1, 0, &footprint, nullptr, nullptr, &total_bytes);
        EXPECT(total_bytes > 0 && footprint.Footprint.RowPitch >= W * 4);
        const UINT row_pitch = footprint.Footprint.RowPitch;

        const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD};
        const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
        const auto upload_desc = buffer_desc(total_bytes);
        const auto readback_desc = buffer_desc(total_bytes);
        Ref<ID3D12Resource> upload, readback;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &upload_desc,
                                                         D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                         IID_PPV_ARGS(&upload.p))));
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &readback_desc,
                                                         D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                         IID_PPV_ARGS(&readback.p))));
        std::uint8_t* up = nullptr;
        EXPECT(SUCCEEDED(upload->Map(0, nullptr, reinterpret_cast<void**>(&up))));

        UINT64 signalled = 0;
        const auto run_list = [&]() -> bool {
            if (FAILED(list->Close())) return false;
            ID3D12CommandList* lists[] = {list.p};
            queue->ExecuteCommandLists(1, lists);
            if (rt::d3d12::last_execute_result(queue.p) != leonida::ReplayResult::success) return false;
            return SUCCEEDED(queue->Signal(fence.p, ++signalled)) &&
                   SUCCEEDED(fence->SetEventOnCompletion(signalled, fence_event)) &&
                   WaitForSingleObject(fence_event, 60'000) == WAIT_OBJECT_0;
        };

        // Uploads `bgra` to the display texture and copies the texture back to the
        // readback buffer in the same list, so a failure here fails the texture
        // round-trip before any present is attempted.
        const auto upload_and_verify = [&](const std::vector<uint8_t>& bgra, const char* label) -> bool {
            for (UINT y = 0; y < H; ++y)
                std::memcpy(up + (size_t)y * row_pitch, bgra.data() + (size_t)y * W * 4, (size_t)W * 4);
            if (FAILED(allocator->Reset()) || FAILED(list->Reset(allocator.p, nullptr))) return false;
            D3D12_RESOURCE_BARRIER b0 = transition(texture.p, D3D12_RESOURCE_STATE_PRESENT,
                                                  D3D12_RESOURCE_STATE_COPY_DEST);
            list->ResourceBarrier(1, &b0);
            D3D12_TEXTURE_COPY_LOCATION dst{};
            dst.pResource = texture.p;
            dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            D3D12_TEXTURE_COPY_LOCATION src{};
            src.pResource = upload.p;
            src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            src.PlacedFootprint = footprint;
            list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            D3D12_RESOURCE_BARRIER b1 = transition(texture.p, D3D12_RESOURCE_STATE_COPY_DEST,
                                                  D3D12_RESOURCE_STATE_COPY_SOURCE);
            list->ResourceBarrier(1, &b1);
            D3D12_TEXTURE_COPY_LOCATION rb{};
            rb.pResource = readback.p;
            rb.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            rb.PlacedFootprint = footprint;
            D3D12_TEXTURE_COPY_LOCATION st{};
            st.pResource = texture.p;
            st.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            list->CopyTextureRegion(&rb, 0, 0, 0, &st, nullptr);
            D3D12_RESOURCE_BARRIER b2 = transition(texture.p, D3D12_RESOURCE_STATE_COPY_SOURCE,
                                                  D3D12_RESOURCE_STATE_PRESENT);
            list->ResourceBarrier(1, &b2);
            if (!run_list()) return false;
            void* mapped = nullptr;
            if (FAILED(readback->Map(0, nullptr, &mapped))) return false;
            uint64_t differing = 0;
            for (UINT y = 0; y < H; ++y)
                if (std::memcmp(static_cast<const uint8_t*>(mapped) + (size_t)y * row_pitch,
                                bgra.data() + (size_t)y * W * 4, (size_t)W * 4) != 0)
                    ++differing;
            readback->Unmap(0, nullptr);
            std::printf("%s texture round-trip: %llu of %u rows differ\n", label,
                        (unsigned long long)differing, (unsigned)H);
            return differing == 0;
        };

        const bool first_round_trip = upload_and_verify(decoded_first, "first");
        check(first_round_trip, "first frame texture round-trip bit-exact");
        if (!first_round_trip) return 1;

        const UINT period = rt::d3d12::xbox_frame_interval_60hz_us;
        EXPECT(SUCCEEDED(xbox.SetFrameIntervalX(nullptr, period, 1, 0)));
        EXPECT(SUCCEEDED(xbox.ScheduleFrameEventX(rt::d3d12::xbox_frame_event_origin, 0, 0)));

        // The presenter filters a scaling blit linearly when the source format
        // reports linear filtering and takes the nearest texel otherwise
        // (runtime/src/vulkan_present.h, blit_filter). Mirror that query so the
        // expected pixels follow the filter the blit actually used.
        VkFormatProperties source_properties{};
        d.instance_fn<PFN_vkGetPhysicalDeviceFormatProperties>("vkGetPhysicalDeviceFormatProperties")(
            d.physical, VK_FORMAT_B8G8R8A8_UNORM, &source_properties);
        const bool scaling_linear =
            (source_properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) != 0;

        // Compares the captured swapchain image with the decoded BGRA frame: every
        // pixel bit-exact at 1:1 (the window is the movie's size, so the plain
        // blit samples texel centres), or within 2 levels of the presenter's
        // filter model at destination texel centres when the swapchain came back
        // at another size (documented, not silent: extent and filter are printed).
        const auto compare_presented = [&](const Frame& frame, const std::vector<uint8_t>& bgra,
                                           const char* label) {
            const UINT ew = frame.extent.width, eh = frame.extent.height;
            const bool one_to_one = (ew == W && eh == H);
            std::printf("%s presented extent: %ux%u (movie %ux%u, swapchain format %d, %s filter)%s\n", label, ew, eh,
                        W, H, int(frame.format), scaling_linear ? "linear" : "nearest",
                        one_to_one ? "" : " (scaled comparison)");
            if (frame.format == VK_FORMAT_B8G8R8A8_SRGB) {
                std::printf("%s: sRGB swapchain, pixel comparison skipped (the blit encodes sRGB and these bytes "
                            "are compared as UNORM)\n", label);
                return;
            }
            uint64_t exact = 0, within1 = 0, within2 = 0;
            int worst = 0;
            UINT bad_x = 0, bad_y = 0;
            RGB bad_got{0, 0, 0}, bad_want{0, 0, 0};
            for (UINT y = 0; y < eh; ++y) {
                for (UINT x = 0; x < ew; ++x) {
                    const RGB want = one_to_one ? decoded_at(bgra, W, x, y)
                                                : sampled_at(bgra, W, H, ew, eh, x, y, scaling_linear);
                    const RGB got = presented_at(frame, x, y);
                    const int err = std::max({std::abs(got.r - want.r), std::abs(got.g - want.g),
                                              std::abs(got.b - want.b)});
                    worst = std::max(worst, err);
                    if (err > 0 && exact++ == 0) {
                        bad_x = x;
                        bad_y = y;
                        bad_got = got;
                        bad_want = want;
                    }
                    within1 += err <= 1;
                    within2 += err <= 2;
                }
            }
            const uint64_t total = (uint64_t)ew * eh;
            std::printf("%s presented vs decoded: exact %llu/%llu, within 1: %llu/%llu, within 2: %llu/%llu, "
                        "worst %d",
                        label, (unsigned long long)(total - exact), (unsigned long long)total,
                        (unsigned long long)within1, (unsigned long long)total, (unsigned long long)within2,
                        (unsigned long long)total, worst);
            if (exact)
                std::printf(" (first at %u,%u: got %d,%d,%d want %d,%d,%d)", bad_x, bad_y, bad_got.r, bad_got.g,
                            bad_got.b, bad_want.r, bad_want.g, bad_want.b);
            std::printf("\n");
            if (one_to_one)
                check(exact == 0, std::string(label) + " presented pixels bit-exact at 1:1");
            else
                check(within2 == total, std::string(label) + " presented pixels within 2 of the filter model");
            // The centre pixel and the four corners of the presented frame,
            // against the same filter model.
            const UINT px[5][2] = {{ew / 2, eh / 2}, {0, 0}, {ew - 1, 0}, {0, eh - 1}, {ew - 1, eh - 1}};
            const char* name[5] = {"centre", "top-left", "top-right", "bottom-left", "bottom-right"};
            for (int i = 0; i < 5; ++i) {
                const RGB got = presented_at(frame, px[i][0], px[i][1]);
                const RGB want = one_to_one ? decoded_at(bgra, W, px[i][0], px[i][1])
                                            : sampled_at(bgra, W, H, ew, eh, px[i][0], px[i][1], scaling_linear);
                const int err = std::max({std::abs(got.r - want.r), std::abs(got.g - want.g),
                                          std::abs(got.b - want.b)});
                std::printf("%s %s (%u,%u): presented %d,%d,%d expected %d,%d,%d\n", label, name[i], px[i][0],
                            px[i][1], got.r, got.g, got.b, want.r, want.g, want.b);
                check(err <= (one_to_one ? 0 : 2), std::string(label) + " " + name[i] + " pixel matches");
            }
        };

        std::uint64_t token = 0;
        const auto present = [&](const char* label, Frame& out) -> bool {
            clock.advance_us(period);
            if (xbox.WaitFrameEventX(rt::d3d12::xbox_frame_event_origin, 0, 0, &token) != S_OK) return false;
            rt::d3d12::XboxPresentPlaneParameters plane{token, 1, &texture.p, nullptr, 0};
            if (xbox.PresentX(1, &plane, nullptr) != S_OK) return false;
            return xbox.presenter().read_back(out.bytes, &out.extent, &out.format);
        };

        EXPECT(present("first", presented_first));
        compare_presented(presented_first, decoded_first, "first");

        if (has_second) {
            const bool second_round_trip = upload_and_verify(decoded_second, "second");
            check(second_round_trip, "second frame texture round-trip bit-exact");
            if (!second_round_trip) return 1;
            EXPECT(present("second", presented_second));
            compare_presented(presented_second, decoded_second, "second");
            if (presented_first.extent.width == presented_second.extent.width &&
                presented_first.extent.height == presented_second.extent.height &&
                presented_first.format == presented_second.format) {
                uint64_t differing = 0;
                for (size_t i = 0; i < presented_first.bytes.size(); ++i)
                    differing += presented_first.bytes[i] != presented_second.bytes[i];
                std::printf("presented frame %u vs %u: %llu of %zu bytes differ\n", first, second,
                            (unsigned long long)differing, presented_first.bytes.size());
                check(differing > 0, "the two presented frames differ (the present carries real frames)");
            } else {
                std::printf("presented frames have different extents/formats, skipping the byte comparison\n");
            }
        }

        upload->Unmap(0, nullptr);
        CloseHandle(fence_event);
    }
    DestroyWindow(window);
    UnregisterClassW(window_class.lpszClassName, window_class.hInstance);

    if (d.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
        return 1;
    }
    std::printf("%s\n", g_failures ? "RESULT FAIL" : "RESULT PASS");
    return g_failures ? 1 : 0;
}
