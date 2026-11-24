// PresentX and the frame events (shims/xbox_present.h) on the Vulkan-backed device, checked on pixels: every case
// presents to a window and reads the swapchain image back (VulkanPresenter capture). The frame events run on a
// fake clock, so the test is fast and the order of tokens and presents is exact.
//
// Cases: 0 planes (blank), 1 plane (solid colour, two colours side by side, a source rectangle, an RGBA and a BGRA
// source, an sRGB source), 2 planes (SDR overlay), plane composition (placement, point/bilinear, premultiplied and
// straight alpha, HDR10 and PQ conversions, refusals) against a CPU model with the tolerances stated at each case, token
// errors (null, unknown, stale, repeated), suspend/resume with re-registration, statistics, and a real-time loop at
// 60 Hz. An HDR10 swapchain is only checked where the surface lists an ST.2084 format (else logged as skipped).
// Needs a window and a Vulkan GPU with presentation.
// Set LEONIDA_VK_VALIDATION=1 for VK_LAYER_KHRONOS_validation; any validation error fails the test, except that
// the NVIDIA layer's known sRGB-swapchain STORAGE check makes the run skip rather than pass or fail.
#include "vk_test_device.h"
#include "xbox_present_readback_size.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "../shims/xbox_present.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <cstdint>
#include <cstdlib>
#include <array>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

using XboxHeapHandleFn = std::uint64_t(WINAPI*)(void* heap);  // heap slot 9: 8-byte handle in RAX

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
        DispatchMessageW(&message);
    }
}

struct Px {
    int r, g, b;
};

constexpr HRESULT invalid_state = HRESULT_FROM_WIN32(ERROR_INVALID_STATE);
constexpr UINT width = 320, height = 240;

// The swapchain frame as RGB, whatever its channel order.
struct Frame {
    std::vector<std::uint8_t> bytes;
    VkExtent2D extent{};
    VkFormat format = VK_FORMAT_UNDEFINED;
    bool bgr() const { return format == VK_FORMAT_B8G8R8A8_UNORM || format == VK_FORMAT_B8G8R8A8_SRGB; }
    bool srgb() const { return format == VK_FORMAT_B8G8R8A8_SRGB || format == VK_FORMAT_R8G8B8A8_SRGB; }
    Px at(UINT x, UINT y) const {
        const std::uint8_t* p = &bytes[(std::size_t(y) * extent.width + x) * 4];
        return bgr() ? Px{p[2], p[1], p[0]} : Px{p[0], p[1], p[2]};
    }
};

bool close_to(Px a, Px b, int tolerance) {
    return std::abs(a.r - b.r) <= tolerance && std::abs(a.g - b.g) <= tolerance && std::abs(a.b - b.b) <= tolerance;
}

// True when every pixel of the rectangle is within the tolerance of `want`.
bool region_is(const Frame& frame, UINT x0, UINT y0, UINT x1, UINT y1, Px want, int tolerance = 1) {
    for (UINT y = y0; y < y1; ++y)
        for (UINT x = x0; x < x1; ++x)
            if (!close_to(frame.at(x, y), want, tolerance)) {
                const Px got = frame.at(x, y);
                std::fprintf(stderr, "pixel %u,%u is %d,%d,%d, expected %d,%d,%d\n", x, y, got.r, got.g, got.b, want.r,
                             want.g, want.b);
                return false;
            }
    return true;
}

// A CPU model of runtime/src/present_compose.comp, in double precision, written from the documented behaviour
// (docs/d3d12_vulkan.md section 4.5), not from the shader's code: planes scaled from a source rectangle to a
// destination rectangle (point: the texel containing the sample; bilinear: four texels clamped inside the source
// rectangle). SDR = piecewise sRGB (or G2.2, `gamma22`) Rec.709, HDR10 = PQ Rec.2020. Every texel is decoded to linear light,
// premultiplied, before it is weighted; the filtered plane is converted to the output's linear working space (HDR10 output:
// Rec.2020 nits with SDR white = `white` nits; SDR output: Rec.709 with SDR white = 1, HDR10 content tone mapped by
// extended Reinhard with its white point at `hdr_peak` nits), plane 1 is blended over plane 0 there and the sum is encoded once.
namespace xbox_present_ref {

struct Tex {
    UINT w = 0, h = 0;
    std::vector<std::array<double, 4>> px;  // 0..1, as stored
};
struct Ref_plane {
    const Tex* tex = nullptr;
    double src[4]{}, dest[4]{};  // texels; output pixels
    bool point = false, hdr = false, straight = false;
};
using V3 = std::array<double, 3>;
bool gamma22 = false;       // SDR EOTF: false piecewise sRGB, true pure 2.2
double hdr_peak = 1000.0;   // nits, the tone map's white point

double pq_encode(double nits) {
    const double m1 = 0.1593017578125, m2 = 78.84375, c1 = 0.8359375, c2 = 18.8515625, c3 = 18.6875;
    const double y = std::pow(std::clamp(nits / 10000.0, 0.0, 1.0), m1);
    return std::pow((c1 + c2 * y) / (1.0 + c3 * y), m2);
}
double pq_decode(double e) {
    const double m1 = 0.1593017578125, m2 = 78.84375, c1 = 0.8359375, c2 = 18.8515625, c3 = 18.6875;
    const double ep = std::pow(std::clamp(e, 0.0, 1.0), 1.0 / m2);
    return 10000.0 * std::pow(std::max(ep - c1, 0.0) / (c2 - c3 * ep), 1.0 / m1);
}
double sdr_decode(double e) {
    e = std::max(e, 0.0);
    if (gamma22) return std::pow(e, 2.2);
    return e <= 0.04045 ? e / 12.92 : std::pow((e + 0.055) / 1.055, 2.4);
}
double sdr_encode(double l) {
    l = std::clamp(l, 0.0, 1.0);
    if (gamma22) return std::pow(l, 1.0 / 2.2);
    return l <= 0.0031308 ? l * 12.92 : 1.055 * std::pow(l, 1.0 / 2.4) - 0.055;
}
// The tone map's luminance curve, in units of SDR white: identity up to the knee (0.8), above it extended Reinhard on the
// excess (Reinhard et al. 2002, eq. 4) with value and slope continuous at the knee and the content peak mapped to 1.0.
double tone_y(double y, double white) {
    const double knee = 0.8, t = 1.0 - knee;
    if (y <= knee) return y;
    const double lw = std::max(hdr_peak / white, 1.0), w = (lw - knee) / t, u = (y - knee) / t;
    return knee + t * u * (1.0 + u / (w * w)) / (1.0 + u);
}
// v in linear Rec.709 with SDR white = 1: colour scaled by Yd / Y.
V3 tone_map(V3 v, double white) {
    const double y = 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2];
    const double s = y > 0.0 ? tone_y(y, white) / y : 1.0;
    return {v[0] * s, v[1] * s, v[2] * s};
}
// A premultiplied linear colour of alpha `a` from the plane's space to the output's working space.
V3 to_working(V3 lin, double a, bool plane_hdr, bool out_hdr, double white) {
    if (plane_hdr == out_hdr) return lin;
    if (!plane_hdr)  // Rec.709 -> Rec.2020, SDR white at `white` nits
        return {(0.627404 * lin[0] + 0.329283 * lin[1] + 0.043313 * lin[2]) * white,
                (0.069097 * lin[0] + 0.919540 * lin[1] + 0.011362 * lin[2]) * white,
                (0.016391 * lin[0] + 0.088013 * lin[1] + 0.895595 * lin[2]) * white};
    if (a <= 0.0) return {0, 0, 0};
    const double r = lin[0] / a, g = lin[1] / a, b = lin[2] / a;
    V3 v{std::max((1.660491 * r - 0.587641 * g - 0.072850 * b) / white, 0.0),
         std::max((-0.124550 * r + 1.132900 * g - 0.008349 * b) / white, 0.0),
         std::max((-0.018151 * r - 0.100579 * g + 1.118730 * b) / white, 0.0)};
    v = tone_map(v, white);
    return {v[0] * a, v[1] * a, v[2] * a};
}
// The texel as linear premultiplied colour and alpha, in the plane's space. `opaque`: plane 0, whose alpha is ignored.
std::array<double, 4> decode_texel(const std::array<double, 4>& t, const Ref_plane& p, bool opaque) {
    const double a = opaque ? 1.0 : std::clamp(t[3], 0.0, 1.0);
    std::array<double, 4> out{0, 0, 0, a};
    for (int c = 0; c < 3; ++c) {
        double e = t[c];
        if (!opaque && !p.straight) e = a > 0.0 ? e / a : 0.0;  // premultiplied data is divided by alpha for the decode
        out[c] = (p.hdr ? pq_decode(e) : sdr_decode(e)) * a;
    }
    return out;
}
std::array<double, 4> sample(const Ref_plane& p, bool opaque, int x, int y) {
    const Tex& t = *p.tex;
    const double u = (x + 0.5 - p.dest[0]) / (p.dest[2] - p.dest[0]), v = (y + 0.5 - p.dest[1]) / (p.dest[3] - p.dest[1]);
    const double sx = p.src[0] + u * (p.src[2] - p.src[0]), sy = p.src[1] + v * (p.src[3] - p.src[1]);
    const auto at = [&](int ix, int iy) { return decode_texel(t.px[std::size_t(iy) * t.w + ix], p, opaque); };
    if (p.point)
        return at(int(std::clamp(std::floor(sx), p.src[0], p.src[2] - 1)), int(std::clamp(std::floor(sy), p.src[1], p.src[3] - 1)));
    const double fx = std::clamp(sx - 0.5, p.src[0], p.src[2] - 1), fy = std::clamp(sy - 0.5, p.src[1], p.src[3] - 1);
    const int x0 = int(std::floor(fx)), y0 = int(std::floor(fy));
    const int x1 = int(std::min(double(x0) + 1, p.src[2] - 1)), y1 = int(std::min(double(y0) + 1, p.src[3] - 1));
    const double wx = fx - x0, wy = fy - y0;
    // Four decoded texels with their bilinear weights: linear light, premultiplied, so a transparent texel contributes no
    // colour of its own.
    const double weight[4] = {(1 - wx) * (1 - wy), wx * (1 - wy), (1 - wx) * wy, wx * wy};
    const std::array<double, 4> texel[4] = {at(x0, y0), at(x1, y0), at(x0, y1), at(x1, y1)};
    std::array<double, 4> out{};
    for (int i = 0; i < 4; ++i)
        for (int c = 0; c < 4; ++c) out[c] += texel[i][c] * weight[i];
    return out;
}
V3 compose(const Ref_plane* planes, int count, bool out_hdr, double white, int x, int y) {
    const auto inside = [&](const Ref_plane& p) {
        return x + 0.5 >= p.dest[0] && x + 0.5 < p.dest[2] && y + 0.5 >= p.dest[1] && y + 0.5 < p.dest[3];
    };
    V3 base{0, 0, 0};
    if (inside(planes[0])) {
        const auto c = sample(planes[0], true, x, y);
        base = to_working({c[0], c[1], c[2]}, 1.0, planes[0].hdr, out_hdr, white);
    }
    if (count > 1 && inside(planes[1])) {
        const auto c = sample(planes[1], false, x, y);
        const V3 over = to_working({c[0], c[1], c[2]}, c[3], planes[1].hdr, out_hdr, white);
        for (int i = 0; i < 3; ++i) base[i] = over[i] + (1 - c[3]) * base[i];
    }
    for (double& v : base) v = out_hdr ? pq_encode(std::max(v, 0.0)) : sdr_encode(v);
    return base;
}

// The whole swapchain frame against the model; prints the first pixel that is off by more than `tolerance` levels.
bool frame_matches(const Frame& frame, const Ref_plane* planes, int count, bool out_hdr, double white, int tolerance) {
    int worst = 0;
    UINT bad = 0, first_x = 0, first_y = 0;
    Px first_got{}, first_want{};
    for (UINT y = 0; y < frame.extent.height; ++y)
        for (UINT x = 0; x < frame.extent.width; ++x) {
            const V3 v = compose(planes, count, out_hdr, white, int(x), int(y));
            const Px want{int(std::lround(std::clamp(v[0], 0.0, 1.0) * 255)), int(std::lround(std::clamp(v[1], 0.0, 1.0) * 255)),
                          int(std::lround(std::clamp(v[2], 0.0, 1.0) * 255))};
            const Px got = frame.at(x, y);
            const int err = std::max({std::abs(got.r - want.r), std::abs(got.g - want.g), std::abs(got.b - want.b)});
            worst = std::max(worst, err);
            if (err > tolerance && bad++ == 0) {
                first_x = x;
                first_y = y;
                first_got = got;
                first_want = want;
            }
        }
    if (bad)
        std::fprintf(stderr, "%u pixels differ from the model by more than %d (worst %d); first at %u,%u is %d,%d,%d, model %d,%d,%d\n", bad,
                     tolerance, worst, first_x, first_y, first_got.r, first_got.g, first_got.b, first_want.r, first_want.g,
                     first_want.b);
    return bad == 0;
}

}  // namespace xbox_present_ref

}  // namespace

int main() {
    {
        // The callback's narrow exception must match the sRGB swapchain VUID, including its follow-ons, and nothing else.
        VkDebugUtilsMessengerCallbackDataEXT data{};
        data.messageIdNumber = 0xc036022f;
        data.pMessage = "vkCreateSwapchainKHR(): VK_IMAGE_USAGE_STORAGE_BIT is not supported";
        EXPECT(vktest::known_srgb_swapchain_storage_error(&data));
        data.pMessage = "vkCreateSwapchainKHR(): imageArrayLayers 1, but the maximum is 0";
        EXPECT(vktest::known_srgb_swapchain_storage_error(&data));
        data.messageIdNumber = 0x12345678;
        data.pMessage = "vkCreateSwapchainKHR(): another validation error";
        EXPECT(!vktest::known_srgb_swapchain_storage_error(&data));
    }
    vktest::Device d;
    if (!d.create(true)) {  // no Vulkan loader, device or presentation extensions: nothing to test (ctest: skipped)
        std::printf("no Vulkan device with presentation: skipped\n");
        return 77;
    }

    WNDCLASSW window_class{};
    window_class.lpfnWndProc = window_proc;
    window_class.hInstance = GetModuleHandleW(nullptr);
    window_class.lpszClassName = L"LeonidaXboxPresentTest";
    window_class.hCursor = LoadCursorW(nullptr, reinterpret_cast<LPCWSTR>(IDC_ARROW));
    EXPECT(RegisterClassW(&window_class));
    RECT frame_rect{0, 0, int(width), int(height)};
    AdjustWindowRect(&frame_rect, WS_OVERLAPPEDWINDOW, FALSE);
    HWND window = CreateWindowExW(0, window_class.lpszClassName, L"Leonida: PresentX on Vulkan", WS_OVERLAPPEDWINDOW,
                                  100, 100, frame_rect.right - frame_rect.left, frame_rect.bottom - frame_rect.top,
                                  nullptr, nullptr, window_class.hInstance, nullptr);
    EXPECT(window);
    ShowWindow(window, SW_SHOWNOACTIVATE);
    UpdateWindow(window);
    pump();

    {
        // The real runtime's instance path (create_vulkan_device_standalone, not the test device): it must enable
        // VK_EXT_swapchain_colorspace when the loader has it, or the HDR10 surface formats can never be listed.
        Ref<ID3D12Device> standalone;
        if (SUCCEEDED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&standalone.p)))) {
            const auto enumerate = reinterpret_cast<PFN_vkEnumerateInstanceExtensionProperties>(
                d.instance_proc(nullptr, "vkEnumerateInstanceExtensionProperties"));
            std::uint32_t count = 0;
            enumerate(nullptr, &count, nullptr);
            std::vector<VkExtensionProperties> available(count);
            enumerate(nullptr, &count, available.data());
            bool loader_has = false;
            for (const auto& e : available) loader_has = loader_has || !std::strcmp(e.extensionName, "VK_EXT_swapchain_colorspace");
            auto standalone_presenter = rt::d3d12::create_window_presenter(standalone.p, window);
            if (standalone_presenter) {
                const bool colorspace_enabled = rt::d3d12::device_has_swapchain_colorspace(standalone.p);
                EXPECT(colorspace_enabled == loader_has);
                // The presenter owns the remaining reference to the device and its Vulkan instance.
                standalone.reset();
                EXPECT(standalone_presenter->enable_capture());
                EXPECT(SUCCEEDED(standalone_presenter->present_blank(1)));
                Frame frame;
                EXPECT(standalone_presenter->read_back(frame.bytes, &frame.extent, &frame.format));
                EXPECT(!frame.bytes.empty());
                std::printf("standalone device (real instance path): VK_EXT_swapchain_colorspace %s on the instance (loader has it: %s)\n",
                            colorspace_enabled ? "enabled" : "not enabled", loader_has ? "yes" : "no");
                std::printf("standalone presenter: frame captured after releasing the caller's device reference\n");
            } else {
                std::printf("standalone device: no window presenter (skipped the instance-extension check)\n");
            }
        }
    }

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
        Ref<ID3D12DescriptorHeap> rtv_heap;
        EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
        EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
        EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                                   IID_PPV_ARGS(&list.p))));
        EXPECT(SUCCEEDED(list->Close()));
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
        constexpr UINT rtv_slots = 24;
        const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, rtv_slots,
                                                   D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&rtv_heap.p))));
        const SIZE_T rtv_start = SIZE_T(reinterpret_cast<XboxHeapHandleFn>(
            (*reinterpret_cast<void***>(rtv_heap.p))[9])(rtv_heap.p));
        const UINT rtv_step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

        // A display-capable back buffer: committed, ALLOW_DISPLAY, in the PRESENT state, with a view in `slot`.
        UINT slot_count = 0;
        struct Source {
            Ref<ID3D12Resource> resource;
            D3D12_CPU_DESCRIPTOR_HANDLE rtv{};
        };
        const auto make_source = [&](DXGI_FORMAT format, Source& out, UINT w = width, UINT h = height) -> bool {
            const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
            D3D12_RESOURCE_DESC desc{};
            desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            desc.Width = w;
            desc.Height = h;
            desc.DepthOrArraySize = 1;
            desc.MipLevels = 1;
            desc.Format = format;
            desc.SampleDesc.Count = 1;
            desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
            if (slot_count >= rtv_slots) return false;
            if (FAILED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_ALLOW_DISPLAY, &desc,
                                                       D3D12_RESOURCE_STATE_PRESENT, nullptr,
                                                       IID_PPV_ARGS(&out.resource.p))))
                return false;
            out.rtv.ptr = rtv_start + SIZE_T(slot_count++) * rtv_step;
            device->CreateRenderTargetView(out.resource.p, nullptr, out.rtv);
            return true;
        };
        UINT64 signalled = 0;
        // Clears the whole resource to `colour`, or the given rectangles; leaves it in PRESENT.
        const auto fill = [&](Source& s, const FLOAT colour[4], UINT rect_count = 0, const D3D12_RECT* rects = nullptr) -> bool {
            if (FAILED(allocator->Reset()) || FAILED(list->Reset(allocator.p, nullptr))) return false;
            D3D12_RESOURCE_BARRIER barrier{};
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barrier.Transition.pResource = s.resource.p;
            barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
            barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
            list->ResourceBarrier(1, &barrier);
            list->ClearRenderTargetView(s.rtv, colour, rect_count, rects);
            barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
            barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
            list->ResourceBarrier(1, &barrier);
            if (FAILED(list->Close())) return false;
            ID3D12CommandList* lists[] = {list.p};
            queue->ExecuteCommandLists(1, lists);
            if (rt::d3d12::last_execute_result(queue.p) != leonida::ReplayResult::success) return false;
            return SUCCEEDED(queue->Signal(fence.p, ++signalled)) &&
                   SUCCEEDED(fence->SetEventOnCompletion(signalled, nullptr));
        };

        // The title's frame loop: wait for the origin, render (the caller did), PresentX with that token.
        const UINT period = rt::d3d12::xbox_frame_interval_60hz_us;
        EXPECT(SUCCEEDED(xbox.SetFrameIntervalX(nullptr, period, 1, 0)));
        EXPECT(SUCCEEDED(xbox.ScheduleFrameEventX(rt::d3d12::xbox_frame_event_origin, 0, 0)));
        const auto next_token = [&](std::uint64_t& token) -> bool {
            clock.advance_us(period);
            return xbox.WaitFrameEventX(rt::d3d12::xbox_frame_event_origin, 0, 0, &token) == S_OK;
        };
        UINT want_w = width, want_h = height;  // the client size the frames are expected at (the resize case changes it)
        const auto grab = [&](Frame& frame) -> bool {
            return xbox.presenter().read_back(frame.bytes, &frame.extent, &frame.format) &&
                   xbox_present_readback_size_matches(frame.bytes.size(), frame.extent.width, frame.extent.height,
                                                     want_w, want_h);
        };

        const FLOAT blue_ish[4] = {0.2f, 0.6f, 0.9f, 1.0f};  // 51, 153, 230
        const FLOAT red[4] = {1, 0, 0, 1}, green[4] = {0, 1, 0, 1}, blue[4] = {0, 0, 1, 1};
        std::uint64_t token = 0;
        Frame frame;
        const auto plane_of = [&](Source& s, std::uint64_t t, const D3D12_RECT* rect = nullptr) {
            return rt::d3d12::XboxPresentPlaneParameters{t, 1, &s.resource.p, rect, 0};
        };

        // 1 plane, solid colour, RGBA8.
        Source rgba;
        EXPECT(make_source(DXGI_FORMAT_R8G8B8A8_UNORM, rgba));
        EXPECT(fill(rgba, blue_ish));
        EXPECT(next_token(token) && token == 1);
        auto plane = plane_of(rgba, token);
        EXPECT(xbox.PresentX(1, &plane, nullptr) == S_OK);
        EXPECT(grab(frame));
        EXPECT(region_is(frame, 0, 0, width, height, {51, 153, 230}));
        std::printf("1 plane, RGBA8 solid: ok (swapchain format %d)\n", int(frame.format));

        // 0 planes: a black frame, and no token is needed.
        EXPECT(xbox.PresentX(0, nullptr, nullptr) == S_OK);
        EXPECT(grab(frame));
        EXPECT(region_is(frame, 0, 0, width, height, {0, 0, 0}, 0));
        // (The blank frame is a frame in flight like any other: with latency 1 the next wait needs it shown first.)
        EXPECT(xbox.pacer().frames_in_flight() == 2);  // the first frame and the blank one
        clock.advance_us(period);
        std::printf("0 planes (blank): ok\n");

        // Two colours side by side: left half red, right half blue; the swapchain shows them where they were.
        Source split;
        EXPECT(make_source(DXGI_FORMAT_R8G8B8A8_UNORM, split));
        const D3D12_RECT left_half{0, 0, LONG(width / 2), LONG(height)}, right_half{LONG(width / 2), 0, LONG(width), LONG(height)};
        EXPECT(fill(split, red, 1, &left_half));
        EXPECT(fill(split, blue, 1, &right_half));
        EXPECT(next_token(token));
        plane = plane_of(split, token);
        EXPECT(xbox.PresentX(1, &plane, nullptr) == S_OK);
        EXPECT(grab(frame));
        EXPECT(region_is(frame, 0, 0, width / 2 - 2, height, {255, 0, 0}, 0));
        EXPECT(region_is(frame, width / 2 + 2, 0, width, height, {0, 0, 255}, 0));
        std::printf("1 plane, two colours (orientation): ok\n");

        // A source rectangle: only the right half (blue) is shown, stretched over the window.
        EXPECT(next_token(token));
        plane = plane_of(split, token, &right_half);
        EXPECT(xbox.PresentX(1, &plane, nullptr) == S_OK);
        EXPECT(grab(frame));
        // (The blit filters bilinearly, so the outermost texel of the rectangle blends with its neighbour outside it: the
        // border pixels are left out.)
        EXPECT(region_is(frame, 2, 2, width - 2, height - 2, {0, 0, 255}, 0));
        // And a quarter in the middle that straddles the edge: the left part shows red, the right part blue.
        const D3D12_RECT middle{LONG(width / 4), LONG(height / 4), LONG(width * 3 / 4), LONG(height * 3 / 4)};
        EXPECT(next_token(token));
        plane = plane_of(split, token, &middle);
        EXPECT(xbox.PresentX(1, &plane, nullptr) == S_OK);
        EXPECT(grab(frame));
        EXPECT(region_is(frame, 4, 4, width / 2 - 4, height - 4, {255, 0, 0}, 0));
        EXPECT(region_is(frame, width / 2 + 4, 4, width - 4, height - 4, {0, 0, 255}, 0));
        // Rectangles that are empty, inverted or outside the texture are refused and present nothing.
        const std::uint64_t before = xbox.presents();
        const D3D12_RECT bad_rects[] = {{10, 10, 10, 20}, {20, 10, 10, 20}, {0, 0, LONG(width + 1), 10}, {-1, 0, 10, 10}};
        for (const D3D12_RECT& bad : bad_rects) {
            clock.advance_us(4 * period);  // let the queued frames show: with latency 1 the wait holds for them
            EXPECT(next_token(token));
            plane = plane_of(split, token, &bad);
            EXPECT(xbox.PresentX(1, &plane, nullptr) == E_INVALIDARG);
        }
        EXPECT(xbox.presents() == before);
        std::printf("source rectangles (valid and invalid): ok\n");

        // A BGRA source with a colour whose channels differ: red must stay red.
        Source bgra;
        EXPECT(make_source(DXGI_FORMAT_B8G8R8A8_UNORM, bgra));
        const FLOAT orange[4] = {1.0f, 0.5f, 0.0f, 1.0f};
        EXPECT(fill(bgra, orange));
        EXPECT(next_token(token));  // (the token from the refused rect presents is still unused, which is fine)
        plane = plane_of(bgra, token);
        EXPECT(xbox.PresentX(1, &plane, nullptr) == S_OK);
        EXPECT(grab(frame));
        EXPECT(region_is(frame, 0, 0, width, height, {255, 128, 0}, 1));
        std::printf("1 plane, BGRA8 source: ok\n");

        // An sRGB source: the clear value is linear, the stored (and shown) bytes are encoded: 0.5 -> 188. The
        // swapchain follows the source's format, so those bytes go through unchanged.
        Source srgb;
        EXPECT(make_source(DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, srgb));
        const FLOAT grey[4] = {0.5f, 0.5f, 0.5f, 1.0f};
        EXPECT(fill(srgb, grey));
        EXPECT(next_token(token));
        plane = plane_of(srgb, token);
        EXPECT(xbox.PresentX(1, &plane, nullptr) == S_OK);
        EXPECT(grab(frame));
        EXPECT(frame.srgb());
        EXPECT(region_is(frame, 0, 0, width, height, {188, 188, 188}, 1));
        // Back to UNORM: the swapchain goes back to a UNORM format.
        EXPECT(next_token(token));
        plane = plane_of(rgba, token);
        EXPECT(xbox.PresentX(1, &plane, nullptr) == S_OK);
        EXPECT(grab(frame));
        EXPECT(!frame.srgb());
        EXPECT(region_is(frame, 0, 0, width, height, {51, 153, 230}));
        std::printf("1 plane, sRGB source (swapchain follows): ok\n");

        // 2 planes sharing a token, both SDR: plane 1 is an overlay (section 4.5), and an opaque one covers plane 0.
        // (The plane composition has its own checks further down.)
        Source second;
        EXPECT(make_source(DXGI_FORMAT_R8G8B8A8_UNORM, second));
        EXPECT(fill(second, green));
        EXPECT(next_token(token));
        rt::d3d12::XboxPresentPlaneParameters two[2] = {plane_of(rgba, token), plane_of(second, token)};
        const std::uint64_t composed_before = xbox.presenter().composed_presents();
        EXPECT(xbox.PresentX(2, two, nullptr) == S_OK);
        EXPECT(xbox.presenter().composed_presents() == composed_before);  // plane 1 without the overlay flag: not composed
        EXPECT(grab(frame));
        EXPECT(region_is(frame, 0, 0, width, height, {51, 153, 230}));  // plane 0 only, by the plain blit
        // Planes with different tokens, or more than two planes, are refused.
        EXPECT(next_token(token));
        two[0] = plane_of(rgba, token);
        two[1] = plane_of(second, token + 1);
        EXPECT(xbox.PresentX(2, two, nullptr) == E_INVALIDARG);
        rt::d3d12::XboxPresentPlaneParameters three[3] = {plane_of(rgba, token), plane_of(second, token), plane_of(second, token)};
        EXPECT(xbox.PresentX(3, three, nullptr) == E_INVALIDARG);
        EXPECT(xbox.PresentX(1, nullptr, nullptr) == E_INVALIDARG);
        {   // ResourceCount above 1 (either plane, either path) is refused, nothing presented, the frame stays
            ID3D12Resource* both[2] = {rgba.resource.p, second.resource.p};
            const std::uint64_t presents_before_count = xbox.presents();
            const auto count_two = [&](int color_space, UINT flags) {
                rt::d3d12::XboxPresentPlaneParameters p{token, 2, both, nullptr, 0};
                p.ColorSpace = color_space;
                p.Flags = flags;
                return p;
            };
            auto one_count = count_two(0, 0);
            EXPECT(xbox.PresentX(1, &one_count, nullptr) == DXGI_ERROR_UNSUPPORTED);  // plain blit path
            one_count = count_two(rt::d3d12::xbox_color_space_g2084_p2020, 0);
            EXPECT(xbox.PresentX(1, &one_count, nullptr) == DXGI_ERROR_UNSUPPORTED);  // compose path
            rt::d3d12::XboxPresentPlaneParameters pair_count[2] = {plane_of(rgba, token), count_two(0, rt::d3d12::xbox_plane_flag_overlay)};
            EXPECT(xbox.PresentX(2, pair_count, nullptr) == DXGI_ERROR_UNSUPPORTED);   // plane 1
            EXPECT(xbox.presents() == presents_before_count);
            std::printf("ResourceCount above 1 refused (plain, compose, plane 1): ok\n");
        }
        std::printf("2 planes (SDR overlay over plane 0) and refused plane sets: ok\n");

        // The parameters struct is accepted (ImmediateThresholdPercent is ignored).
        const rt::d3d12::XboxPresentParameters params{50.0f};
        plane = plane_of(second, token);
        EXPECT(xbox.PresentX(1, &plane, &params) == S_OK);
        EXPECT(grab(frame));
        EXPECT(region_is(frame, 0, 0, width, height, {0, 255, 0}, 0));

        // Tokens: a token that was never handed out is refused and nothing is presented. A null, repeated or stale one
        // is presented anyway (and logged once per kind, see the log lines above): alternate the sources so that
        // each present shows on screen.
        const std::uint64_t presented_before = xbox.presents();
        plane = plane_of(rgba, xbox.pacer().tokens_issued() + 1);
        EXPECT(xbox.PresentX(1, &plane, nullptr) == E_INVALIDARG);
        EXPECT(xbox.presents() == presented_before);
        EXPECT(grab(frame));
        EXPECT(region_is(frame, 0, 0, width, height, {0, 255, 0}, 0));  // still the last frame
        plane = plane_of(rgba, 0);  // null token, as a loading screen might
        EXPECT(xbox.PresentX(1, &plane, nullptr) == S_OK);
        EXPECT(grab(frame));
        EXPECT(region_is(frame, 0, 0, width, height, {51, 153, 230}));
        plane = plane_of(second, token);  // repeated: the token of the last tokened present
        EXPECT(xbox.PresentX(1, &plane, nullptr) == S_OK);
        EXPECT(grab(frame));
        EXPECT(region_is(frame, 0, 0, width, height, {0, 255, 0}, 0));
        plane = plane_of(rgba, 1);  // stale: long gone
        EXPECT(xbox.PresentX(1, &plane, nullptr) == S_OK);
        EXPECT(grab(frame));
        EXPECT(region_is(frame, 0, 0, width, height, {51, 153, 230}));
        EXPECT(xbox.presents() == presented_before + 3);

        // Sources the GPU cannot blit are refused with DXGI_ERROR_UNSUPPORTED and present nothing: multisampled,
        // integer and block-compressed. The last frame stays on screen and the token is still usable afterwards.
        {
            const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
            const auto make_special = [&](DXGI_FORMAT format, UINT samples, D3D12_RESOURCE_FLAGS flags,
                                          Ref<ID3D12Resource>& out) {
                D3D12_RESOURCE_DESC desc{};
                desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
                desc.Width = width;
                desc.Height = height;
                desc.DepthOrArraySize = 1;
                desc.MipLevels = 1;
                desc.Format = format;
                desc.SampleDesc.Count = samples;
                desc.Flags = flags;
                return SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_ALLOW_DISPLAY, &desc,
                                                                 D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                                 IID_PPV_ARGS(&out.p)));
            };
            Ref<ID3D12Resource> msaa, integer, bc;
            EXPECT(make_special(DXGI_FORMAT_R8G8B8A8_UNORM, 4, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, msaa));
            EXPECT(make_special(DXGI_FORMAT_R8G8B8A8_UINT, 1, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, integer));
            EXPECT(make_special(DXGI_FORMAT_BC1_UNORM, 1, D3D12_RESOURCE_FLAG_NONE, bc));
            ID3D12Resource* unsupported[3] = {msaa.p, integer.p, bc.p};
            clock.advance_us(4 * period);  // let the queued frames show: with latency 1 the wait holds for them
            EXPECT(next_token(token));
            const std::uint64_t before_unsupported = xbox.presents();
            for (ID3D12Resource* resource : unsupported) {
                const rt::d3d12::XboxPresentPlaneParameters p{token, 1, &resource, nullptr, 0};
                EXPECT(xbox.PresentX(1, &p, nullptr) == DXGI_ERROR_UNSUPPORTED);
            }
            EXPECT(xbox.presents() == before_unsupported);
            EXPECT(grab(frame));
            EXPECT(region_is(frame, 0, 0, width, height, {51, 153, 230}));
            plane = plane_of(second, token);
            EXPECT(xbox.PresentX(1, &plane, nullptr) == S_OK);  // the same token, now with a supported source
            EXPECT(grab(frame));
            EXPECT(region_is(frame, 0, 0, width, height, {0, 255, 0}, 0));
            std::printf("unsupported sources (MSAA, integer, BC) refused: ok\n");
        }

        // ---- Plane composition (docs section 4.5), against a CPU model of the compose pass (xbox_present_ref below). ----
        // Exact (tolerance 0) for point filtering between SDR planes without alpha; 1 level for bilinear filtering and
        // alpha blending (the pass works in float, stores RGBA16F and the copy rounds to 8 bits); 2 levels where PQ and
        // gamma conversions are involved (GPU pow is not correctly rounded).
        {
            using namespace xbox_present_ref;
            using rt::d3d12::XboxViewRect;
            struct Painted {
                Source source;
                Tex tex;
            };
            const auto make_painted = [&](Painted& p, UINT w, UINT h, DXGI_FORMAT format = DXGI_FORMAT_R8G8B8A8_UNORM) {
                p.tex = Tex{w, h, std::vector<std::array<double, 4>>(std::size_t(w) * h, {0, 0, 0, 0})};
                const FLOAT zero[4] = {0, 0, 0, 0};
                return make_source(format, p.source, w, h) && fill(p.source, zero);
            };
            // Paints a rectangle with 8-bit values (R8G8B8A8) in the texture and in the model.
            const auto paint = [&](Painted& p, D3D12_RECT r, int R, int G, int B, int A) {
                const FLOAT c[4] = {R / 255.0f, G / 255.0f, B / 255.0f, A / 255.0f};
                if (!fill(p.source, c, 1, &r)) return false;
                for (LONG y = r.top; y < r.bottom; ++y)
                    for (LONG x = r.left; x < r.right; ++x)
                        p.tex.px[std::size_t(y) * p.tex.w + x] = {R / 255.0, G / 255.0, B / 255.0, A / 255.0};
                return true;
            };
            // The same for R10G10B10A2 (10-bit codes; alpha 2 bits): the model keeps the quantised value.
            const auto paint10 = [&](Painted& p, D3D12_RECT r, int R, int G, int B) {
                const FLOAT c[4] = {R / 1023.0f, G / 1023.0f, B / 1023.0f, 1.0f};
                if (!fill(p.source, c, 1, &r)) return false;
                for (LONG y = r.top; y < r.bottom; ++y)
                    for (LONG x = r.left; x < r.right; ++x)
                        p.tex.px[std::size_t(y) * p.tex.w + x] = {R / 1023.0, G / 1023.0, B / 1023.0, 1.0};
                return true;
            };
            // R10G10B10A2 with a 2-bit alpha (0..3): stored colour is premultiplied by the caller.
            const auto paint10a = [&](Painted& p, D3D12_RECT r, int R, int G, int B, int A) {
                const FLOAT c[4] = {R / 1023.0f, G / 1023.0f, B / 1023.0f, A / 3.0f};
                if (!fill(p.source, c, 1, &r)) return false;
                for (LONG y = r.top; y < r.bottom; ++y)
                    for (LONG x = r.left; x < r.right; ++x)
                        p.tex.px[std::size_t(y) * p.tex.w + x] = {R / 1023.0, G / 1023.0, B / 1023.0, A / 3.0};
                return true;
            };
            // One PresentX of `count` planes, a fresh token for all; the swapchain frame is read back into `frame`.
            const auto present_planes = [&](rt::d3d12::XboxPresentPlaneParameters* planes_, UINT count_, HRESULT want = S_OK) {
                clock.advance_us(4 * period);
                if (!next_token(token)) return false;
                for (UINT i = 0; i < count_; ++i) planes_[i].Token = token;
                if (xbox.PresentX(count_, planes_, nullptr) != want) return false;
                return want != S_OK || grab(frame);
            };
            const auto parameters = [&](Source& s, const XboxViewRect* place, int filter, int color_space = 0,
                                        UINT flags = 0, const D3D12_RECT* rect = nullptr) {
                rt::d3d12::XboxPresentPlaneParameters p = plane_of(s, 0, rect);
                p.ColorSpace = color_space;
                p.pDestPlacementBase = place;
                p.ScaleFilter = filter;
                p.Flags = flags;
                return p;
            };
            const auto model = [&](const Painted& p, const XboxViewRect* place, bool point, const D3D12_RECT* rect = nullptr,
                                   bool hdr = false, bool straight = false) {
                Ref_plane r;
                r.tex = &p.tex;
                r.src[0] = rect ? rect->left : 0;
                r.src[1] = rect ? rect->top : 0;
                r.src[2] = rect ? rect->right : double(p.tex.w);
                r.src[3] = rect ? rect->bottom : double(p.tex.h);
                const XboxViewRect whole{0, 0, 1, 1};
                const XboxViewRect& v = place ? *place : whole;
                r.dest[0] = v.left * width;
                r.dest[1] = v.top * height;
                r.dest[2] = v.right * width;
                r.dest[3] = v.bottom * height;
                r.point = point;
                r.hdr = hdr;
                r.straight = straight;
                return r;
            };
            using rt::d3d12::xbox_scale_filter_linear;
            using rt::d3d12::xbox_scale_filter_point;
            const std::uint64_t composed_start = xbox.presenter().composed_presents();
            std::uint64_t composed_expected = composed_start;
            const auto composed_ok = [&] { return xbox.presenter().composed_presents() == ++composed_expected; };
            // Fixed inputs whatever the machine says: SDR white 203 nits (not the Windows slider), piecewise sRGB (not
            // LEONIDA_SDR_EOTF), peak 1000 nits. The query itself is checked on its own below.
            xbox.set_sdr_white_nits(203.0f);
            xbox.set_sdr_eotf(leonida::PresentPlanes::SdrEotf::srgb);
            xbox.set_hdr_peak_nits(1000.0f);
            xbox_present_ref::gamma22 = false;
            xbox_present_ref::hdr_peak = 1000.0;
            {
                EXPECT(rt::d3d12::sdr_white_nits_from_level(1000) == 80.0f && rt::d3d12::sdr_white_nits_from_level(2500) == 200.0f &&
                       rt::d3d12::sdr_white_nits_from_level(6000) == 480.0f);
                // The query's cost: why PresentX does not run it (the layer caches it from a background thread).
                float windows_white = 0.0f;
                double total_us = 0, worst_us = 0;
                const int queries = 50;
                for (int q = 0; q < queries; ++q) {
                    const auto t0 = std::chrono::steady_clock::now();
                    windows_white = xbox.presenter().display_sdr_white_nits();  // 0 when Windows has no answer
                    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
                    total_us += us;
                    worst_us = std::max(worst_us, us);
                }
                std::printf("SDR white level query: %.0f us mean, %.0f us worst over %d calls\n", total_us / queries, worst_us, queries);
                EXPECT(windows_white == 0.0f || (windows_white >= 80.0f && windows_white <= 480.0f));
                std::printf("SDR white level of this window's output from Windows: %s%.1f nits (hdr_active %d)\n",
                            windows_white > 0 ? "" : "unavailable, ", windows_white, int(xbox.presenter().hdr_active()));
            }

            // A: one plane, scaled 2x to the middle of the output, point and linear; the rest of the output is black.
            Painted quad;
            EXPECT(make_painted(quad, 80, 60));
            EXPECT(paint(quad, {0, 0, 40, 30}, 255, 0, 0, 255) && paint(quad, {40, 0, 80, 30}, 0, 255, 0, 255));
            EXPECT(paint(quad, {0, 30, 40, 60}, 0, 0, 255, 255) && paint(quad, {40, 30, 80, 60}, 255, 255, 255, 255));
            const XboxViewRect middle_half{0.25f, 0.25f, 0.75f, 0.75f};
            auto one = parameters(quad.source, &middle_half, xbox_scale_filter_point);
            EXPECT(present_planes(&one, 1) && composed_ok());
            EXPECT(frame.format == VK_FORMAT_B8G8R8A8_UNORM || frame.format == VK_FORMAT_R8G8B8A8_UNORM);
            Ref_plane ref0 = model(quad, &middle_half, true);
            EXPECT(frame_matches(frame, &ref0, 1, false, 203.0, 0));
            std::printf("compose: 1 plane, 80x60 -> dest placement 160x120, point: exact\n");
            one = parameters(quad.source, &middle_half, xbox_scale_filter_linear);
            EXPECT(present_planes(&one, 1) && composed_ok());
            ref0 = model(quad, &middle_half, false);
            EXPECT(frame_matches(frame, &ref0, 1, false, 203.0, 1));
            std::printf("compose: same, bilinear: within 1 level of the model\n");
            // A source rectangle (the right half) onto the whole output, point: no placement.
            const D3D12_RECT right_half_of_quad{40, 0, 80, 60};
            one = parameters(quad.source, nullptr, xbox_scale_filter_point, 0, 0, &right_half_of_quad);
            EXPECT(present_planes(&one, 1) && composed_ok());
            ref0 = model(quad, nullptr, true, &right_half_of_quad);
            EXPECT(frame_matches(frame, &ref0, 1, false, 203.0, 0));
            // Bilinear with a source rectangle must not read outside it: the right half shows only green over white.
            one = parameters(quad.source, nullptr, xbox_scale_filter_linear, 0, 0, &right_half_of_quad);
            EXPECT(present_planes(&one, 1) && composed_ok());
            ref0 = model(quad, nullptr, false, &right_half_of_quad);
            EXPECT(frame_matches(frame, &ref0, 1, false, 203.0, 1));
            std::printf("compose: source rectangle, point and bilinear (clamped inside it): ok\n");

            // The compose pass with point filtering at 1:1 gives the very bytes of the plain blit.
            plane = plane_of(split, 0);
            clock.advance_us(4 * period);
            EXPECT(next_token(token));
            plane.Token = token;
            const std::uint64_t blits = xbox.presenter().composed_presents();
            EXPECT(xbox.PresentX(1, &plane, nullptr) == S_OK && xbox.presenter().composed_presents() == blits);
            Frame blit_frame;
            EXPECT(grab(blit_frame));
            one = parameters(split, nullptr, xbox_scale_filter_point);
            EXPECT(present_planes(&one, 1) && composed_ok());
            EXPECT(frame.bytes == blit_frame.bytes);
            std::printf("compose at 1:1 equals the plain blit byte for byte: ok\n");

            // An sRGB source (0.5 linear is byte 188): the pass encodes it again, so the bytes come out as stored.
            one = parameters(srgb, nullptr, xbox_scale_filter_point);
            EXPECT(present_planes(&one, 1) && composed_ok());
            EXPECT(!frame.srgb());
            EXPECT(region_is(frame, 0, 0, width, height, {188, 188, 188}, 1));
            std::printf("compose: sRGB source, UNORM swapchain: ok\n");

            // B: plane 1 over plane 0. Plane 0 is a solid; plane 1 (80x60, scaled 2x into the middle) has a half
            // transparent red (premultiplied: 128,0,0 alpha 128), nothing, opaque green and 25% blue (0,0,64, alpha 64).
            Painted solid;
            EXPECT(make_painted(solid, width, height));
            EXPECT(paint(solid, {0, 0, LONG(width), LONG(height)}, 51, 153, 230, 255));
            Painted ui;
            EXPECT(make_painted(ui, 80, 60));
            EXPECT(paint(ui, {0, 0, 40, 30}, 128, 0, 0, 128) && paint(ui, {0, 30, 40, 60}, 0, 255, 0, 255));
            EXPECT(paint(ui, {40, 30, 80, 60}, 0, 0, 64, 64));
            rt::d3d12::XboxPresentPlaneParameters pair[2];
            pair[0] = parameters(solid.source, nullptr, xbox_scale_filter_point);
            pair[1] = parameters(ui.source, &middle_half, xbox_scale_filter_point, 0, rt::d3d12::xbox_plane_flag_overlay);
            EXPECT(present_planes(pair, 2) && composed_ok());
            Ref_plane refs[2] = {model(solid, nullptr, true), model(ui, &middle_half, true)};
            EXPECT(frame_matches(frame, refs, 2, false, 203.0, 1));
            std::printf("compose: 2 planes, premultiplied alpha overlay, point: within 1 level of the model\n");
            pair[1] = parameters(ui.source, &middle_half, xbox_scale_filter_point, 0, rt::d3d12::xbox_plane_flag_overlay | rt::d3d12::xbox_plane_flag_straight_alpha);
            EXPECT(present_planes(pair, 2) && composed_ok());
            refs[1] = model(ui, &middle_half, true, nullptr, false, true);
            EXPECT(frame_matches(frame, refs, 2, false, 203.0, 1));
            std::printf("compose: 2 planes, straight alpha overlay: within 1 level of the model\n");
            pair[1] = parameters(ui.source, &middle_half, xbox_scale_filter_linear, 0, rt::d3d12::xbox_plane_flag_overlay);
            EXPECT(present_planes(pair, 2) && composed_ok());
            refs[1] = model(ui, &middle_half, false);
            EXPECT(frame_matches(frame, refs, 2, false, 203.0, 1));
            std::printf("compose: 2 planes, bilinear overlay: within 1 level of the model\n");
            // Straight alpha with bilinear filtering across a transparent edge: opaque white on the left, transparent black on the
            // right. Colour is filtered premultiplied, so the edge is white at partial alpha, not grey (filtering the
            // unassociated colour would also quarter the colour and quarter it again by the alpha). At output x=160 the sample
            // is 0.75 of the way from the white texel to the transparent one: alpha 0.25, colour white, blended in linear light.
            Painted edge;
            EXPECT(make_painted(edge, 80, 60));
            EXPECT(paint(edge, {0, 0, 40, 60}, 255, 255, 255, 255) && paint(edge, {40, 0, 80, 60}, 0, 0, 0, 0));
            pair[1] = parameters(edge.source, &middle_half, xbox_scale_filter_linear, 0,
                                 rt::d3d12::xbox_plane_flag_overlay | rt::d3d12::xbox_plane_flag_straight_alpha);
            EXPECT(present_planes(pair, 2) && composed_ok());
            refs[1] = model(edge, &middle_half, false, nullptr, false, true);
            EXPECT(frame_matches(frame, refs, 2, false, 203.0, 1));
            {
                const Px edge_px = frame.at(160, 120);
                // Linear light: red 0.25 + 0.75 * lin(51/255), blue 0.25 + 0.75 * lin(230/255), then encoded (piecewise sRGB:
                // about 143 and 237); the unfiltered-colour result would be near 54 and blending the encoded values would give 102.
                const auto expect_level = [](double base255) {
                    return int(std::lround(255 * xbox_present_ref::sdr_encode(0.25 + 0.75 * xbox_present_ref::sdr_decode(base255 / 255))));
                };
                EXPECT(std::abs(edge_px.r - expect_level(51)) <= 1 && std::abs(edge_px.b - expect_level(230)) <= 1);
                std::printf("compose: edge pixel %d,%d,%d (linear-light model %d,%d)\n", edge_px.r, edge_px.g, edge_px.b, expect_level(51), expect_level(230));
            }
            std::printf("compose: straight alpha, bilinear across a transparent edge (premultiplied filtering): ok\n");

            // C: HDR10 (PQ, Rec.2020) planes. Plane 0 is 80x60 R10G10B10A2 (three bands: SDR white level, a bright
            // colour above it, and a dim colour), shown 4x on the whole output.
            Painted hdr;
            EXPECT(make_painted(hdr, 80, 60, DXGI_FORMAT_R10G10B10A2_UNORM));
            const auto code = [](double nits) { return int(std::lround(pq_encode(nits) * 1023.0)); };
            EXPECT(paint10(hdr, {0, 0, 27, 60}, code(203), code(203), code(203)));
            EXPECT(paint10(hdr, {27, 0, 54, 60}, code(600), code(150), code(20)));
            EXPECT(paint10(hdr, {54, 0, 80, 60}, code(40), code(80), code(120)));
            one = parameters(hdr.source, nullptr, xbox_scale_filter_point, rt::d3d12::xbox_color_space_g2084_p2020);
            EXPECT(present_planes(&one, 1) && composed_ok());
            Ref_plane href = model(hdr, nullptr, true, nullptr, true);
            const bool surface_hdr = xbox.presenter().hdr_active();
            if (!surface_hdr) {
                EXPECT(frame_matches(frame, &href, 1, false, 203.0, 2));
                std::printf("compose: HDR10 plane to an SDR swapchain (tone mapped to SDR, white = 203 nits, peak 1000): within 2 levels of the model\n");
            } else {
                // A real HDR10 swapchain: the 10-bit codes come out as they went in (plane 0, point, 1:1 PQ).
                EXPECT(frame.format == VK_FORMAT_A2B10G10R10_UNORM_PACK32 || frame.format == VK_FORMAT_A2R10G10B10_UNORM_PACK32);
                bool ok = true;
                for (UINT y = 0; y < height && ok; y += 7)
                    for (UINT x = 0; x < width && ok; x += 5) {
                        std::uint32_t word;
                        std::memcpy(&word, &frame.bytes[(std::size_t(y) * frame.extent.width + x) * 4], 4);
                        const bool abgr = frame.format == VK_FORMAT_A2B10G10R10_UNORM_PACK32;
                        const int r = int(abgr ? (word & 1023) : ((word >> 20) & 1023));
                        const auto want = hdr.tex.px[std::size_t(y / 4) * 80 + x / 4];
                        ok = std::abs(r - int(std::lround(want[0] * 1023.0))) <= 1;
                    }
                EXPECT(ok);
                std::printf("compose: HDR10 plane to an HDR10 (ST.2084) swapchain: codes within 1 of the source: ok\n");
            }
            // Same plane, conversions to the PQ encoding of an 8-bit swapchain (test hook): an SDR overlay over it.
            xbox.presenter().force_hdr_encoding(true);
            pair[0] = parameters(hdr.source, nullptr, xbox_scale_filter_point, rt::d3d12::xbox_color_space_g2084_p2020);
            pair[1] = parameters(ui.source, &middle_half, xbox_scale_filter_point, 0, rt::d3d12::xbox_plane_flag_overlay);
            EXPECT(present_planes(pair, 2) && composed_ok());
            Ref_plane hrefs[2] = {model(hdr, nullptr, true, nullptr, true), model(ui, &middle_half, true)};
            if (!surface_hdr) {
                EXPECT(frame_matches(frame, hrefs, 2, true, 203.0, 2));
                std::printf("compose: HDR10 plane + SDR overlay in PQ encoding (8-bit, forced): within 2 levels of the model\n");
            }
            // SDR planes into PQ: the SDR white of the quad's white corner goes to 203 nits.
            pair[0] = parameters(quad.source, nullptr, xbox_scale_filter_point);
            pair[1] = parameters(ui.source, &middle_half, xbox_scale_filter_point, 0, rt::d3d12::xbox_plane_flag_overlay);
            EXPECT(present_planes(pair, 2) && composed_ok());
            Ref_plane srefs[2] = {model(quad, nullptr, true), model(ui, &middle_half, true)};
            if (!surface_hdr) {
                EXPECT(frame_matches(frame, srefs, 2, true, 203.0, 2));
                std::printf("compose: SDR planes converted to PQ (forced): within 2 levels of the model\n");
            }
            xbox.presenter().force_hdr_encoding(false);
            // The HDR plane reaches the SDR swapchain again after an HDR request on a surface that has none.
            if (!surface_hdr) {
                xbox.set_sdr_white_nits(100.0f);
                one = parameters(hdr.source, nullptr, xbox_scale_filter_point, rt::d3d12::xbox_color_space_g2084_p2020);
                EXPECT(present_planes(&one, 1) && composed_ok());
                EXPECT(frame_matches(frame, &href, 1, false, 100.0, 2));
                xbox.set_sdr_white_nits(203.0f);
                std::printf("compose: SDR white level 100 nits applies: ok\n");
            } else {
                std::printf("compose: the surface has HDR10: the forced-encoding checks are skipped on it\n");
            }
            if (!surface_hdr)
                std::printf("compose: HDR10 swapchain: this surface/display lists no ST.2084 format, so only the SDR tone map and "
                            "the PQ encoder were run (HDR10 swapchain case skipped)\n");

            // C2: linear-light blending and the SDR EOTF option. A 50% white overlay (premultiplied 128,128,128 alpha 128) over
            // black is half the light: byte 188 with piecewise sRGB, 186 with gamma 2.2 (blending the encoded values would give
            // 128), checked against the model and against these numbers.
            {
                Painted black, half_white;
                EXPECT(make_painted(black, width, height));
                EXPECT(paint(black, {0, 0, LONG(width), LONG(height)}, 0, 0, 0, 255));
                EXPECT(make_painted(half_white, 8, 8));
                EXPECT(paint(half_white, {0, 0, 8, 8}, 128, 128, 128, 128));
                pair[0] = parameters(black.source, nullptr, xbox_scale_filter_point);
                pair[1] = parameters(half_white.source, nullptr, xbox_scale_filter_point, 0, rt::d3d12::xbox_plane_flag_overlay);
                Ref_plane lrefs[2] = {model(black, nullptr, true), model(half_white, nullptr, true)};
                const double alpha = 128.0 / 255.0;
                for (const bool g22 : {false, true}) {
                    xbox.set_sdr_eotf(g22 ? leonida::PresentPlanes::SdrEotf::gamma22 : leonida::PresentPlanes::SdrEotf::srgb);
                    xbox_present_ref::gamma22 = g22;
                    EXPECT(present_planes(pair, 2) && composed_ok());
                    EXPECT(frame_matches(frame, lrefs, 2, false, 203.0, 1));
                    // the overlay's white, as light (premultiplied 128/255 over alpha 0.502 is white): 0.502 of full white
                    const int want = int(std::lround(255 * xbox_present_ref::sdr_encode(alpha)));
                    EXPECT(region_is(frame, 0, 0, width, height, {want, want, want}, 1));
                    // independent of the model's formulas (hand calculation: 0.50196 as light is 187.8 and 186.4 in bytes)
                    EXPECT(region_is(frame, 0, 0, width, height, g22 ? Px{186, 186, 186} : Px{188, 188, 188}, 1));
                    std::printf("compose: 50%% white over black, %s: byte %d (model %d, encoded-space blend would be 128)\n",
                                g22 ? "gamma 2.2" : "piecewise sRGB", frame.at(10, 10).r, want);
                    // an SDR plane 0 alone through the same EOTF: bytes come out as they went in
                    one = parameters(quad.source, nullptr, xbox_scale_filter_point);
                    EXPECT(present_planes(&one, 1) && composed_ok());
                    Ref_plane q = model(quad, nullptr, true);
                    EXPECT(frame_matches(frame, &q, 1, false, 203.0, 0));
                }
                xbox.set_sdr_eotf(leonida::PresentPlanes::SdrEotf::srgb);
                xbox_present_ref::gamma22 = false;
                std::printf("compose: SDR EOTF sRGB and gamma 2.2: linear blend matches the model, single planes unchanged: ok\n");
            }

            // C3: HDR10 -> SDR output is tone mapped, not clipped: identity on luminance up to 0.8 of SDR white (162 nits), then
            // a Reinhard shoulder that maps the content peak to 1.0. Eight columns of grey (50, 100, 150, 203, 300, 500, 750, 1000
            // nits). The expected bytes are hand calculations (sRGB encode of the knee curve), not the model's code; 10-bit PQ
            // codes make them good to 2.
            {
                // the curve itself: monotonic, identity below the knee, continuous in slope, peak -> 1
                for (const double white : {203.0, 100.0}) {
                    double previous = -1;
                    for (double y = 0; y <= 20; y += 0.01) {
                        const double d_y = xbox_present_ref::tone_y(y, white);
                        EXPECT(d_y >= previous);
                        EXPECT(y > 0.8 || d_y == y);
                        previous = d_y;
                    }
                    const double lw = xbox_present_ref::hdr_peak / white;
                    EXPECT(std::abs(xbox_present_ref::tone_y(std::max(lw, 1.0), white) - 1.0) < 1e-12);
                    const double e = 1e-6;
                    EXPECT(std::abs((xbox_present_ref::tone_y(0.8 + e, white) - 0.8) / e - 1.0) < 1e-3);
                }
            }
            if (!surface_hdr) {
                Painted ramp;
                EXPECT(make_painted(ramp, 80, 60, DXGI_FORMAT_R10G10B10A2_UNORM));
                const double ramp_nits[8] = {50, 100, 150, 203, 300, 500, 750, 1000};
                const int want_1000[8] = {136, 186, 223, 243, 250, 253, 254, 255};
                const int want_600[8] = {136, 186, 223, 244, 250, 254, 255, 255};
                for (int i = 0; i < 8; ++i)
                    EXPECT(paint10(ramp, {10 * i, 0, 10 * i + 10, 60}, code(ramp_nits[i]), code(ramp_nits[i]), code(ramp_nits[i])));
                for (const float peak_nits : {1000.0f, 600.0f}) {
                    xbox.set_hdr_peak_nits(peak_nits);
                    xbox_present_ref::hdr_peak = peak_nits;
                    one = parameters(ramp.source, nullptr, xbox_scale_filter_point, rt::d3d12::xbox_color_space_g2084_p2020);
                    EXPECT(present_planes(&one, 1) && composed_ok());
                    Ref_plane rref = model(ramp, nullptr, true, nullptr, true);
                    EXPECT(frame_matches(frame, &rref, 1, false, 203.0, 2));
                    const int* want = peak_nits == 1000.0f ? want_1000 : want_600;
                    int previous = -1;
                    bool ok = true;
                    for (int i = 0; i < 8; ++i) {
                        const Px got = frame.at(UINT(i * 40 + 20), 10);
                        ok = ok && std::abs(got.r - want[i]) <= 2 && got.r >= previous;
                        previous = got.r;
                    }
                    EXPECT(ok);
                    // SDR-range content (100 and 150 nits are below the knee) is not touched by the tone map
                    EXPECT(std::abs(frame.at(60, 10).r - 186) <= 2 && std::abs(frame.at(100, 10).r - 223) <= 2);
                    std::printf("compose: HDR10 ramp tone mapped onto SDR, peak %.0f nits: bytes", peak_nits);
                    for (int i = 0; i < 8; ++i) std::printf(" %d", frame.at(UINT(i * 40 + 20), 10).r);
                    std::printf(" (hand calculation:");
                    for (int i = 0; i < 8; ++i) std::printf(" %d", want[i]);
                    std::printf(")\n");
                }
                xbox.set_hdr_peak_nits(1000.0f);
                xbox_present_ref::hdr_peak = 1000.0;
            }

            // C4: an HDR10 plane 1 over an SDR plane 0 (the reverse of the case above), on a forced HDR10 encoding and on the SDR
            // output. Plane 1 is premultiplied grey of 400 nits at alpha 2/3 (stored code = PQ(400) * 2/3), plane 0 black. As
            // light the result is 266.7 nits (PQ byte 155, hand calculation) on the HDR10 output; on the SDR output 400 nits is
            // 1.97 of SDR white, shoulder-mapped to 0.973, times 2/3, sRGB encoded: byte 211.
            if (!surface_hdr) {
                Painted black4, hdr_ui;
                EXPECT(make_painted(black4, width, height));
                EXPECT(paint(black4, {0, 0, LONG(width), LONG(height)}, 0, 0, 0, 255));
                EXPECT(make_painted(hdr_ui, 16, 16, DXGI_FORMAT_R10G10B10A2_UNORM));
                const int stored = int(std::lround(code(400) * 2.0 / 3.0));
                EXPECT(paint10a(hdr_ui, {0, 0, 16, 16}, stored, stored, stored, 2));
                pair[0] = parameters(black4.source, nullptr, xbox_scale_filter_point);
                pair[1] = parameters(hdr_ui.source, nullptr, xbox_scale_filter_point, rt::d3d12::xbox_color_space_g2084_p2020,
                                     rt::d3d12::xbox_plane_flag_overlay);
                Ref_plane urefs[2] = {model(black4, nullptr, true), model(hdr_ui, nullptr, true, nullptr, true)};
                xbox.presenter().force_hdr_encoding(true);
                EXPECT(present_planes(pair, 2) && composed_ok());
                EXPECT(frame_matches(frame, urefs, 2, true, 203.0, 2));
                const int hdr_byte = int(frame.at(5, 5).r);
                EXPECT(region_is(frame, 0, 0, width, height, {155, 155, 155}, 3));
                xbox.presenter().force_hdr_encoding(false);
                EXPECT(present_planes(pair, 2) && composed_ok());
                EXPECT(frame_matches(frame, urefs, 2, false, 203.0, 2));
                EXPECT(region_is(frame, 0, 0, width, height, {211, 211, 211}, 3));
                std::printf("compose: HDR10 plane 1 (premultiplied, alpha 2/3) over SDR plane 0: HDR10 output byte %d (hand 155), SDR output byte %d (hand 211): ok\n",
                            hdr_byte, int(frame.at(5, 5).r));
            }

            // D: an SDR plane 1 next to an HDR10 plane 0 that is not flagged an overlay is the GameDVR copy: not shown.
            pair[0] = parameters(hdr.source, nullptr, xbox_scale_filter_point, rt::d3d12::xbox_color_space_g2084_p2020);
            pair[1] = parameters(second, nullptr, xbox_scale_filter_point, 0);
            EXPECT(present_planes(pair, 2) && composed_ok());
            if (!surface_hdr) EXPECT(frame_matches(frame, &href, 1, false, 203.0, 2));
            std::printf("compose: HDR10 plane 0 + unflagged SDR plane 1 (GameDVR copy) shows plane 0 only: ok\n");

            // E: a placement that is not a sub-rectangle of the output (empty, inverted, outside, NaN), an unknown filter and
            // extended descriptors are ignored (logged once): the plane is shown over the whole output, point filtered.
            const float nan = std::numeric_limits<float>::quiet_NaN();
            const XboxViewRect bad_places[] = {{0.5f, 0, 0.5f, 1}, {-0.1f, 0, 1, 1}, {0, 0, 1.5f, 1}, {0, 0.75f, 1, 0.25f}, {nan, 0, 1, 1}};
            for (const XboxViewRect& bad : bad_places) {
                one = parameters(quad.source, &bad, xbox_scale_filter_point);
                EXPECT(present_planes(&one, 1) && composed_ok());
                Ref_plane whole = model(quad, nullptr, true);
                EXPECT(frame_matches(frame, &whole, 1, false, 203.0, 0));
            }
            one = parameters(quad.source, nullptr, 77);  // unknown filter: default (bilinear), plain blit path
            one.ExtendedDescCount = 2;
            EXPECT(present_planes(&one, 1));
            std::printf("compose: implausible placement, unknown filter, extended descriptors ignored (shown, not refused): ok\n");
            // Sources the pass cannot read (depth, multisampled) are refused: nothing is presented, the token still works.
            const std::uint64_t presents_before = xbox.presents();
            {
                const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
                const auto make_special = [&](DXGI_FORMAT format, UINT samples, D3D12_RESOURCE_FLAGS flags, Ref<ID3D12Resource>& out) {
                    D3D12_RESOURCE_DESC desc{};
                    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
                    desc.Width = width;
                    desc.Height = height;
                    desc.DepthOrArraySize = 1;
                    desc.MipLevels = 1;
                    desc.Format = format;
                    desc.SampleDesc.Count = samples;
                    desc.Flags = flags;
                    return SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_ALLOW_DISPLAY, &desc,
                                                                     D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&out.p)));
                };
                Ref<ID3D12Resource> depth_plane, msaa_plane;
                EXPECT(make_special(DXGI_FORMAT_D32_FLOAT, 1, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL | D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE, depth_plane));
                EXPECT(make_special(DXGI_FORMAT_R8G8B8A8_UNORM, 4, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, msaa_plane));
                ID3D12Resource* cannot[2] = {depth_plane.p, msaa_plane.p};
                for (ID3D12Resource* resource : cannot) {
                    rt::d3d12::XboxPresentPlaneParameters p{0, 1, &resource, nullptr, 0};
                    p.ScaleFilter = xbox_scale_filter_point;
                    EXPECT(present_planes(&p, 1, DXGI_ERROR_UNSUPPORTED));
                }
            }
            EXPECT(xbox.presents() == presents_before);
            EXPECT(xbox.presenter().composed_presents() == composed_expected);
            one = parameters(quad.source, &middle_half, xbox_scale_filter_point);
            EXPECT(present_planes(&one, 1) && composed_ok());  // still working after the refusals
            std::printf("compose: bad placements, depth and multisampled sources refused: ok\n");

            // F: the window changes size between composed presents (the swapchain is recreated, possibly in the middle of a
            // present after an out-of-date acquire): the compose target follows, pixels match the model at the new size.
            const UINT sizes[][2] = {{400, 300}, {200, 150}, {width, height}};
            for (const auto& size : sizes) {
                RECT outer{0, 0, LONG(size[0]), LONG(size[1])};
                AdjustWindowRect(&outer, WS_OVERLAPPEDWINDOW, FALSE);
                SetWindowPos(window, nullptr, 0, 0, outer.right - outer.left, outer.bottom - outer.top,
                             SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE);
                pump();
                want_w = want_h = 0;  // the first present after a resize may still be at the old size (suboptimal swapchain)
                for (int round = 0; round < 3; ++round) {
                    one = parameters(quad.source, &middle_half, xbox_scale_filter_point);
                    // (A present the driver reports out of date right after the resize is not counted as shown, so the counter
                    // is not checked here; the frame is still composed and read back.)
                    EXPECT(present_planes(&one, 1));
                    // Whatever size the frame has, it must be the composition at that size (no stale or mis-sized target).
                    Ref_plane resized = model(quad, &middle_half, true);
                    resized.dest[0] = middle_half.left * double(frame.extent.width);
                    resized.dest[1] = middle_half.top * double(frame.extent.height);
                    resized.dest[2] = middle_half.right * double(frame.extent.width);
                    resized.dest[3] = middle_half.bottom * double(frame.extent.height);
                    EXPECT(frame_matches(frame, &resized, 1, false, 203.0, 0));
                    if (round == 2) EXPECT(frame.extent.width == size[0] && frame.extent.height == size[1]);
                }
            }
            want_w = width;
            want_h = height;
            composed_expected = xbox.presenter().composed_presents();
            std::printf("compose: window resized 320x240 -> 400x300 -> 200x150 -> 320x240 between composed presents: ok\n");
        }

        // Skipping a frame (waiting twice, presenting the second token) is fine.
        std::uint64_t skipped = 0;
        clock.advance_us(4 * period);
        EXPECT(next_token(skipped) && next_token(token) && token == skipped + 1);
        plane = plane_of(rgba, token);
        EXPECT(xbox.PresentX(1, &plane, nullptr) == S_OK);
        std::printf("token validation: ok\n");

        // Statistics for the frame just presented: event and present now, display after the next vblank.
        UINT count = 3;
        leonida::FrameStatistic stats[3];
        EXPECT(xbox.GetFrameStatisticsX(token, leonida::stat_frame_event | leonida::stat_present | leonida::stat_display,
                                        &count, stats) == S_FALSE);
        EXPECT(count == 2 && stats[0].type == leonida::stat_frame_event && stats[1].type == leonida::stat_present);
        clock.advance_us(period);
        count = 3;
        EXPECT(xbox.GetFrameStatisticsX(token, leonida::stat_frame_event | leonida::stat_present | leonida::stat_display,
                                        &count, stats) == S_OK);
        EXPECT(count == 3 && stats[2].type == leonida::stat_display && stats[2].token == token);
        count = 3;
        EXPECT(xbox.GetFrameStatisticsX(99999, leonida::stat_display, &count, stats) == S_FALSE && count == 0);
        std::printf("frame statistics: ok\n");

        // Suspend: presents are refused and the registration is gone; after resume it has to be redone.
        EXPECT(next_token(token));
        EXPECT(xbox.SuspendX(0) == S_OK);
        plane = plane_of(rgba, token);
        EXPECT(xbox.PresentX(1, &plane, nullptr) == invalid_state);
        EXPECT(xbox.PresentX(0, nullptr, nullptr) == invalid_state);
        EXPECT(xbox.WaitFrameEventX(rt::d3d12::xbox_frame_event_origin, 0, 0, &token) == invalid_state);
        EXPECT(xbox.ResumeX() == S_OK);
        EXPECT(xbox.WaitFrameEventX(rt::d3d12::xbox_frame_event_origin, 0, 0, &token) == invalid_state);  // not re-registered
        EXPECT(xbox.PresentX(0, nullptr, nullptr) == S_OK);  // a blank present needs no registration
        EXPECT(SUCCEEDED(xbox.SetFrameIntervalX(nullptr, period, 1, 0)));
        EXPECT(SUCCEEDED(xbox.ScheduleFrameEventX(rt::d3d12::xbox_frame_event_origin, 0, 0)));
        const std::uint64_t before_resume = xbox.pacer().tokens_issued();
        EXPECT(next_token(token) && token == before_resume + 1);
        plane = plane_of(rgba, token);
        EXPECT(xbox.PresentX(1, &plane, nullptr) == S_OK);
        EXPECT(grab(frame));
        EXPECT(region_is(frame, 0, 0, width, height, {51, 153, 230}));
        std::printf("suspend/resume with re-registration: ok\n");

        // ImmediateThresholdPercent: a frame that missed its vblank by less than the threshold is shown at once (mid-scan,
        // IMMEDIATE or FIFO_RELAXED), else it waits; the decision is the pacer's lateness on the fake clock, and the
        // display statistic and immediate_presents() tell which way it went. Without VK_*_swapchain_maintenance1 (or
        // an IMMEDIATE/FIFO_RELAXED mode the swapchain can switch to) the frame always waits.
        {
            const bool can_tear = xbox.presenter().tear_supported();
            std::printf("tearing present mode available: %s (device swapchain_maintenance1: %s)\n",
                        can_tear ? "yes" : "no", d.swapchain_maintenance1 ? "yes" : "no");
            EXPECT(d.swapchain_maintenance1 || !can_tear);
            clock.advance_us(4 * period);  // everything in flight is shown
            EXPECT(SUCCEEDED(xbox.SetFrameIntervalX(nullptr, period, 1, 0)));
            EXPECT(SUCCEEDED(xbox.ScheduleFrameEventX(rt::d3d12::xbox_frame_event_origin, 0, 0)));
            EXPECT(xbox.pacer().frames_in_flight() == 0);
            const std::uint64_t grid_epoch = clock.now_us();  // vblanks are at grid_epoch + n * period
            const std::uint64_t tears_before = xbox.immediate_presents();
            std::uint64_t expected_tears = tears_before;
            // One frame: wait for its origin, spend `late_us` after the vblank it was due at, present with `threshold`.
            // Checks the pixels, whether it was shown at once and the display statistic.
            const auto frame_with = [&](std::uint64_t late_us, float threshold, bool pass_params, bool expect_tear) -> bool {
                std::uint64_t t = 0;
                const std::uint64_t now = clock.now_us();                 // to the next vblank: origin n
                clock.advance_us(grid_epoch + ((now - grid_epoch) / period + 1) * period - now);
                if (xbox.WaitFrameEventX(rt::d3d12::xbox_frame_event_origin, 0, 0, &t) != S_OK) return false;
                clock.advance_us(period + late_us);                       // due at vblank n + 1; now = due + late_us
                rt::d3d12::XboxPresentParameters params{threshold};
                auto p = plane_of(rgba, t);
                if (xbox.PresentX(1, &p, pass_params ? &params : nullptr) != S_OK) return false;
                if (!grab(frame) || !region_is(frame, 0, 0, width, height, {51, 153, 230})) return false;
                if (expect_tear) ++expected_tears;
                if (xbox.immediate_presents() != expected_tears) {
                    std::fprintf(stderr, "late %llu us, threshold %g: immediate_presents %llu, expected %llu\n",
                                 static_cast<unsigned long long>(late_us), double(threshold),
                                 static_cast<unsigned long long>(xbox.immediate_presents()),
                                 static_cast<unsigned long long>(expected_tears));
                    return false;
                }
                // The display statistic: at once (display time = present time, scan position = lateness) or at a vblank.
                clock.advance_us(period);  // past the next vblank: shown either way
                leonida::FrameStatistic stats[2];
                UINT count = 2;
                if (xbox.GetFrameStatisticsX(t, leonida::stat_present | leonida::stat_display, &count, stats) != S_OK ||
                    count != 2)
                    return false;
                const bool at_once = stats[1].time_us == stats[0].time_us;
                if (at_once != expect_tear) {
                    std::fprintf(stderr, "late %llu us, threshold %g: shown at once = %d, expected %d\n",
                                 static_cast<unsigned long long>(late_us), double(threshold), int(at_once), int(expect_tear));
                    return false;
                }
                const float want_percent = expect_tear ? 100.0f * float(late_us) / float(period) : 0.0f;
                return std::fabs(stats[1].percent_scanned - want_percent) < 0.01f;
            };
            const bool t = can_tear;
            // The period is 16667 us: 8000 us is 48% of it, 2000 us 12%, 1000 us 6%.
            EXPECT(frame_with(8000, 60.0f, true, t));      // 48% scanned, threshold 60%: shown at once
            EXPECT(frame_with(8000, 49.0f, true, t));      // just inside the threshold
            EXPECT(frame_with(8000, 47.0f, true, false));  // the scan is past 47%: waits for the vblank
            EXPECT(frame_with(8000, 0.0f, true, false));   // threshold 0 disables it
            EXPECT(frame_with(8000, 0.0f, false, false));  // no parameters: the same
            EXPECT(frame_with(8000, -10.0f, true, false));
            EXPECT(frame_with(0, 100.0f, true, false));    // on time: nothing to tear
            EXPECT(frame_with(1000, 100.0f, true, t));     // 100: a late flip anywhere in the scan
            EXPECT(frame_with(15000, 100.0f, true, t));
            EXPECT(frame_with(2000, 10.0f, true, false));  // 12% scanned, threshold 10
            EXPECT(frame_with(2000, 13.0f, true, t));
            EXPECT(xbox.immediate_presents() == expected_tears);
            std::printf("ImmediateThresholdPercent (%s): ok, %llu of 11 frames shown at once\n",
                        can_tear ? "tearing present mode in use" : "unsupported here, frames wait",
                        static_cast<unsigned long long>(xbox.immediate_presents() - tears_before));
            if (can_tear) EXPECT(xbox.presenter().tear_presents() == xbox.immediate_presents() - tears_before);
        }

        owner.reset();  // one swapchain per window: the first presenter goes
        // Kill switch: LEONIDA_PRESENT_NO_TEAR=1 keeps the swapchain FIFO-only, so a late frame with a threshold waits.
        {
            _putenv_s("LEONIDA_PRESENT_NO_TEAR", "1");
            auto no_tear = rt::d3d12::create_window_presenter(inner, window);
            _putenv_s("LEONIDA_PRESENT_NO_TEAR", "");
            EXPECT(no_tear && no_tear->enable_capture());
            EXPECT(!no_tear->tear_supported());
            leonida::ManualFrameClock fake;
            rt::d3d12::XboxPresentation fifo_only(std::move(no_tear), &fake);
            EXPECT(SUCCEEDED(fifo_only.SetFrameIntervalX(nullptr, period, 1, 0)));
            EXPECT(SUCCEEDED(fifo_only.ScheduleFrameEventX(rt::d3d12::xbox_frame_event_origin, 0, 0)));
            fake.advance_us(period);
            std::uint64_t t = 0;
            EXPECT(fifo_only.WaitFrameEventX(rt::d3d12::xbox_frame_event_origin, 0, 0, &t) == S_OK);
            fake.advance_us(period + 8000);  // half an interval late
            rt::d3d12::XboxPresentParameters fifo_params{100.0f};
            auto p = plane_of(rgba, t);
            EXPECT(fifo_only.PresentX(1, &p, &fifo_params) == S_OK);
            EXPECT(fifo_only.immediate_presents() == 0 && fifo_only.pacer().frames_in_flight() == 1);
            EXPECT(fifo_only.presenter().read_back(frame.bytes, &frame.extent, &frame.format));
            EXPECT(region_is(frame, 0, 0, width, height, {51, 153, 230}));
            std::printf("no tearing mode (LEONIDA_PRESENT_NO_TEAR): late frame waits for the vblank: ok\n");
        }
        // The device a title holds is the Xbox-ABI proxy, not `inner`: the presenter comes from it all the same
        // (D3D-414: it used to be null, with nothing in the log).
        {
            auto from_proxy = rt::d3d12::create_window_presenter(device.p, window);
            EXPECT(from_proxy && from_proxy->enable_capture());
            EXPECT(SUCCEEDED(from_proxy->present_blank(1)));
            std::printf("presenter from the Xbox device proxy: created, a blank frame presented: ok\n");
        }
        // Real time, 60 Hz: 30 frames cycling through three colours; tokens count up one per wait, each frame shows
        // the colour that was presented, and the loop takes about 30 vblanks.
        auto presenter2 = rt::d3d12::create_window_presenter(inner, window);
        EXPECT(presenter2 && presenter2->enable_capture());
        rt::d3d12::XboxPresentation real(std::move(presenter2));  // the real clock
        Source colours[3];
        const FLOAT* wanted[3] = {red, green, blue};
        const Px expect_rgb[3] = {{255, 0, 0}, {0, 255, 0}, {0, 0, 255}};
        for (int i = 0; i < 3; ++i) EXPECT(make_source(DXGI_FORMAT_R8G8B8A8_UNORM, colours[i]) && fill(colours[i], wanted[i]));
        EXPECT(SUCCEEDED(real.SetFrameIntervalX(nullptr, period, 1, 0)));
        EXPECT(SUCCEEDED(real.ScheduleFrameEventX(rt::d3d12::xbox_frame_event_origin, 0, 0)));
        constexpr int frames = 30;
        const auto start = std::chrono::steady_clock::now();
        std::uint64_t last = 0;
        // One real-time frame: wait for the origin, present colour `c`. True with `shown` false when the present was
        // skipped because the window is minimised (PresentX still S_OK, nothing on screen, nothing to read back).
        const auto real_frame = [&](int c, bool& shown) -> bool {
            pump();
            std::uint64_t t = 0;
            if (real.WaitFrameEventX(rt::d3d12::xbox_frame_event_origin, 1000, 0, &t) != S_OK || t != last + 1) {
                std::fprintf(stderr, "real time: token %llu after %llu\n", static_cast<unsigned long long>(t),
                             static_cast<unsigned long long>(last));
                return false;
            }
            last = t;
            auto p = plane_of(colours[c], t);
            if (real.PresentX(1, &p, nullptr) != S_OK) return false;
            shown = real.presenter().read_back(frame.bytes, &frame.extent, &frame.format);
            if (!shown) return IsIconic(window) != FALSE;  // skipped only while minimised
            const Px got = frame.at(width / 2, height / 2);
            if (close_to(got, expect_rgb[c], 0)) return true;
            std::fprintf(stderr, "real time: centre %d,%d,%d, presented %d,%d,%d\n", got.r, got.g, got.b, expect_rgb[c].r,
                         expect_rgb[c].g, expect_rgb[c].b);
            return false;
        };
        // Someone at the desktop can minimise the window during the loop (Win+D in the middle of a CTest run): those
        // presents are skipped, and the window is restored so the rest of the loop is seen again.
        int unseen = 0;
        for (int i = 0; i < frames; ++i) {
            bool shown = false;
            EXPECT(real_frame(i % 3, shown));
            if (!shown) {
                ++unseen;
                ShowWindow(window, SW_RESTORE);
            }
        }
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
        std::printf("real time: %d frames at 60 Hz in %lld ms (ideal %d), %d skipped while minimised\n", frames,
                    static_cast<long long>(ms), frames * int(period) / 1000, unseen);
        EXPECT(ms >= (frames - 1) * int(period) / 1000 * 9 / 10 && ms < 5000);
        EXPECT(unseen < frames);

        // Minimised on purpose: the surface is 0x0 (VK_KHR_win32_surface), so PresentX succeeds without showing the
        // frame, and read_back must say so rather than hand out the frame before it (TEST-378: the real-time loop
        // above once compared that older frame). Each colour differs from the one shown last, so a stale frame fails.
        // Restored, the next frame is shown and read back again.
        {
            ShowWindow(window, SW_MINIMIZE);
            int skipped_presents = 0;
            for (int i = 0; i < 3; ++i) {  // the first present can still reach the old swapchain (then out of date)
                bool shown = true;
                EXPECT(real_frame(i, shown));
                if (!shown) ++skipped_presents;
            }
            EXPECT(skipped_presents > 0);
            ShowWindow(window, SW_RESTORE);
            bool shown = false;
            for (int i = 0; i < 3 && !shown; ++i) EXPECT(real_frame((i + 1) % 3, shown));  // stale, then recreated
            EXPECT(shown);
            std::printf("minimised: %d of 3 presents skipped and not read back; restored: shown again\n", skipped_presents);
        }
        EXPECT(real.PresentX(0, nullptr, nullptr) == S_OK);  // the shutdown blank
    }
    DestroyWindow(window);
    UnregisterClassW(window_class.lpszClassName, window_class.hInstance);

    const unsigned validation_errors = d.validation_errors();
    const unsigned known_errors = d.known_validation_errors();
    if (validation_errors != known_errors) {
        std::fprintf(stderr, "%u validation errors (%u known sRGB swapchain checks)\n", validation_errors,
                     known_errors);
        return 1;
    }
    if (known_errors) {
        std::fprintf(stderr, "%u known sRGB swapchain STORAGE validation errors; skipped\n", known_errors);
        return 77;
    }
    std::printf("xbox_present_test: all cases pass\n");
    return 0;
}
