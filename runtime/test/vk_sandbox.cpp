// Real-shader sandbox: draws a tools/sandbox_scene.py scene through two translated game shaders.
//
//   vk_sandbox <scene.bin> <vs.spv> <ps.spv> --out <image.ppm> [--size W H] [--cam x y z yaw pitch] [--fov deg]
//   vk_sandbox <scene.bin> <vs.spv> <ps.spv> --window [--size W H] [--cam ...] [--mode fly|walk|drive]   (Windows only)
//   --window --frames N --out <image.ppm>: test run, the window shows without activation and quits after N frames
//
// The pair (chosen by tools/sandbox.py) is a transform VS with POSITION/NORMAL/TEXCOORD inputs and a textured PS, both reading user SGPR 8:
// a 32-bit pointer (high dword forced to 0) to one table of descriptors in the heap, laid out as the shaders load it:
//   +0x00 V# of the object constants: world rows 0..2 at 0x00/0x10/0x20, translation at 0x30
//   +0x20 T# of the albedo texture (8 dwords)
//   +0x40 V# of the view constants: light direction at 0xa0, view-projection rows at 0x160 (row-vector convention)
//   +0x60 V# of the colour constants: rgba at 0x00
// Every draw gets its own table and object constants; view and colour constants are shared. The PS samples with the
// S# literal in its own code, so the sampler is built from those words (gfx10::decode). Nothing here is game data:
// the scene file and the shaders come from the user's work folder.
//
// Window keys (only while the window has focus): WASD move, mouse with the right button held looks, Q/E down/up,
// Shift faster, 1 fly, 2 walk (eye 1.7 m over the ground under the camera), 3 drive (eye 1.2 m, accelerates),
// P writes sandbox_shot.ppm, Esc quits.
#define VK_NO_PROTOTYPES
#ifdef _WIN32
#define VK_USE_PLATFORM_WIN32_KHR
#include <windows.h>
#endif
#include "vulkan_loader.h"
#include "vk_device_pick.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "../src/descriptor_table.h"
#include "../src/gfx10_descriptors.h"

static PFN_vkGetInstanceProcAddr gipa;
#define CHECK(x)                                                                     \
    do {                                                                             \
        VkResult r_ = (x);                                                           \
        if (r_ != VK_SUCCESS) {                                                      \
            std::fprintf(stderr, "FAILED %s -> %d (line %d)\n", #x, r_, __LINE__);   \
            return 1;                                                                \
        }                                                                            \
    } while (0)

namespace {

struct Texture { uint32_t w, h; std::vector<uint8_t> rgba; };
struct Mesh { uint32_t first_vertex, vertex_count, first_index, index_count; };
struct Draw { uint32_t mesh; int32_t texture; float rows[4][3]; };
struct Scene {
    std::vector<Texture> textures;
    std::vector<Mesh> meshes;
    std::vector<Draw> draws;
    std::vector<float> vertices;     // 8 floats each
    std::vector<uint32_t> indices;
    float center[3];
};

bool load_scene(const char* path, Scene& s) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    const uint64_t file_size = f ? uint64_t(f.tellg()) : 0;
    f.seekg(0);
    char magic[4];
    uint32_t hdr[4];
    if (!f.read(magic, 4) || std::memcmp(magic, "LSB1", 4) || !f.read(reinterpret_cast<char*>(hdr), 16) ||
        hdr[0] != 1 || !f.read(reinterpret_cast<char*>(s.center), 12))
        return false;
    for (uint32_t i = 0; i < hdr[1]; ++i) {
        Texture t;
        f.read(reinterpret_cast<char*>(&t.w), 4).read(reinterpret_cast<char*>(&t.h), 4);
        if (!f || !t.w || !t.h || t.w > 16384 || t.h > 16384 || uint64_t(t.w) * t.h * 4 > file_size) return false;
        t.rgba.resize(size_t(t.w) * t.h * 4);
        f.read(reinterpret_cast<char*>(t.rgba.data()), std::streamsize(t.rgba.size()));
        s.textures.push_back(std::move(t));
    }
    for (uint32_t i = 0; i < hdr[2]; ++i) {
        uint32_t vc = 0, ic = 0;
        f.read(reinterpret_cast<char*>(&vc), 4).read(reinterpret_cast<char*>(&ic), 4);
        if (!f || uint64_t(vc) * 32 + uint64_t(ic) * 4 > file_size) return false;
        Mesh m{uint32_t(s.vertices.size() / 8), vc, uint32_t(s.indices.size()), ic};
        s.vertices.resize(s.vertices.size() + size_t(vc) * 8);
        s.indices.resize(s.indices.size() + ic);
        f.read(reinterpret_cast<char*>(s.vertices.data() + size_t(m.first_vertex) * 8), std::streamsize(vc) * 32);
        f.read(reinterpret_cast<char*>(s.indices.data() + m.first_index), std::streamsize(ic) * 4);
        if (!f) return false;
        for (uint32_t k = 0; k < ic; ++k)
            if (s.indices[m.first_index + k] >= vc) return false;
        s.meshes.push_back(m);
    }
    for (uint32_t i = 0; i < hdr[3]; ++i) {
        Draw d;
        f.read(reinterpret_cast<char*>(&d), sizeof(d));
        if (!f || d.mesh >= s.meshes.size() || d.texture >= int32_t(s.textures.size())) return false;
        s.draws.push_back(d);
    }
    return bool(f);
}

std::vector<uint32_t> read_spv(const char* p) {
    std::ifstream f(p, std::ios::binary);
    std::vector<char> b((std::istreambuf_iterator<char>(f)), {});
    std::vector<uint32_t> w(b.size() / 4);
    std::memcpy(w.data(), b.data(), w.size() * 4);
    return w;
}

struct Camera {
    float pos[3], yaw = 0, pitch = 0, fov = 70, speed = 0;
    int mode = 0;   // 0 fly, 1 walk, 2 drive
    void forward(float f[3]) const {
        f[0] = std::cos(pitch) * std::cos(yaw); f[1] = std::cos(pitch) * std::sin(yaw); f[2] = std::sin(pitch);
    }
};

// clip = wx*row0 + wy*row1 + wz*row2 + row3 (D3D clip space: y up, z 0..1), stored as 16 floats.
void view_projection(const Camera& c, float aspect, float out[16]) {
    float f[3]; c.forward(f);
    float r[3] = {f[1], -f[0], 0};
    const float rl = std::sqrt(r[0] * r[0] + r[1] * r[1]);
    if (rl < 1e-6f) { r[0] = std::sin(c.yaw); r[1] = -std::cos(c.yaw); } else { r[0] /= rl; r[1] /= rl; }
    const float u[3] = {r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]};
    const float n = 0.1f, fa = 5000.f, sy = 1.f / std::tan(c.fov * 3.14159265f / 360.f), sx = sy / aspect;
    const float A = fa / (fa - n), B = -n * fa / (fa - n);
    // view = (dot(w - p, r), dot(w - p, u), dot(w - p, f)); clip = (vx*sx, vy*sy, vz*A + B, vz)
    const float* axes[3] = {r, u, f};
    const float scale[4] = {sx, sy, A, 1};
    for (int k = 0; k < 4; ++k) {
        const float* a = axes[k < 3 ? k : 2];
        for (int j = 0; j < 3; ++j) out[j * 4 + k] = a[j] * scale[k];
        out[12 + k] = -(a[0] * c.pos[0] + a[1] * c.pos[1] + a[2] * c.pos[2]) * scale[k] + (k == 2 ? B : 0);
    }
}

// Ground under a point: highest scene triangle below z + 1 at (x, y), from a 4 m grid built once.
struct Ground {
    float cell = 4, x0 = 0, y0 = 0;
    int nx = 0, ny = 0;
    std::vector<std::vector<float>> tris;   // per cell: 9 floats a triangle
    void build(const Scene& s) {
        float lo[2] = {1e30f, 1e30f}, hi[2] = {-1e30f, -1e30f};
        std::vector<float> all;
        for (const Draw& d : s.draws) {
            const Mesh& m = s.meshes[d.mesh];
            for (uint32_t i = 0; i + 2 < m.index_count; i += 3)
                for (int k = 0; k < 3; ++k) {
                    const float* v = &s.vertices[size_t(m.first_vertex + s.indices[m.first_index + i + k]) * 8];
                    for (int j = 0; j < 3; ++j) {
                        const float w = v[0] * d.rows[0][j] + v[1] * d.rows[1][j] + v[2] * d.rows[2][j] + d.rows[3][j];
                        all.push_back(w);
                        if (j < 2) { lo[j] = std::min(lo[j], w); hi[j] = std::max(hi[j], w); }
                    }
                }
        }
        if (all.empty()) return;
        x0 = lo[0]; y0 = lo[1];
        nx = std::min(4096, int((hi[0] - lo[0]) / cell) + 1); ny = std::min(4096, int((hi[1] - lo[1]) / cell) + 1);
        tris.assign(size_t(nx) * ny, {});
        for (size_t t = 0; t < all.size(); t += 9) {
            const float* p = &all[t];
            const float ax = std::min({p[0], p[3], p[6]}), bx = std::max({p[0], p[3], p[6]});
            const float ay = std::min({p[1], p[4], p[7]}), by = std::max({p[1], p[4], p[7]});
            if (bx - ax > 64 || by - ay > 64) continue;   // skip huge sheets (skydomes, water planes)
            for (int cy = std::max(0, int(std::floor((ay - y0) / cell))); cy <= std::min(ny - 1, int(std::floor((by - y0) / cell))); ++cy)
                for (int cx = std::max(0, int(std::floor((ax - x0) / cell))); cx <= std::min(nx - 1, int(std::floor((bx - x0) / cell))); ++cx)
                    tris[size_t(cy) * nx + cx].insert(tris[size_t(cy) * nx + cx].end(), p, p + 9);
        }
    }
    bool height(float x, float y, float below, float& out) const {
        const int cx = int(std::floor((x - x0) / cell)), cy = int(std::floor((y - y0) / cell));
        if (cx < 0 || cy < 0 || cx >= nx || cy >= ny) return false;
        bool hit = false;
        const auto& v = tris[size_t(cy) * nx + cx];
        for (size_t t = 0; t < v.size(); t += 9) {
            const float* p = &v[t];
            const float d = (p[3] - p[0]) * (p[7] - p[1]) - (p[6] - p[0]) * (p[4] - p[1]);
            if (std::fabs(d) < 1e-9f) continue;
            const float a = ((p[3] - x) * (p[7] - y) - (p[6] - x) * (p[4] - y)) / d;
            const float b = ((p[6] - x) * (p[1] - y) - (p[0] - x) * (p[7] - y)) / d;
            const float c = 1 - a - b;
            if (a < 0 || b < 0 || c < 0) continue;
            const float z = a * p[2] + b * p[5] + c * p[8];
            if (z <= below && (!hit || z > out)) { out = z; hit = true; }
        }
        return hit;
    }
};

#ifdef _WIN32
bool g_quit = false, g_look = false;
int g_mx = 0, g_my = 0, g_dx = 0, g_dy = 0;
LRESULT CALLBACK wndproc(HWND h, UINT m, WPARAM w, LPARAM l) {
    switch (m) {
    case WM_CLOSE: g_quit = true; return 0;
    case WM_RBUTTONDOWN: g_look = true; g_mx = short(LOWORD(l)); g_my = short(HIWORD(l)); SetCapture(h); return 0;
    case WM_RBUTTONUP: g_look = false; ReleaseCapture(); return 0;
    case WM_MOUSEMOVE:
        if (g_look) { const int x = short(LOWORD(l)), y = short(HIWORD(l)); g_dx += x - g_mx; g_dy += y - g_my; g_mx = x; g_my = y; }
        return 0;
    }
    return DefWindowProcW(h, m, w, l);
}
#endif

bool write_ppm(const char* path, const uint8_t* bgra_or_rgba, uint32_t w, uint32_t h, bool bgra) {
    std::ofstream o(path, std::ios::binary);
    o << "P6\n" << w << " " << h << "\n255\n";
    for (size_t i = 0; i < size_t(w) * h; ++i) {
        const uint8_t* p = bgra_or_rgba + i * 4;
        const char px[3] = {char(p[bgra ? 2 : 0]), char(p[1]), char(p[bgra ? 0 : 2])};
        o.write(px, 3);
    }
    return bool(o);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: vk_sandbox scene.bin vs.spv ps.spv (--out image.ppm | --window) [--size W H] "
                             "[--cam x y z yaw pitch] [--fov deg]\n");
        return 2;
    }
    Scene scene;
    if (!load_scene(argv[1], scene)) { std::fprintf(stderr, "bad scene file %s\n", argv[1]); return 2; }
    uint32_t W = 1280, H = 720;
    const char* out_path = nullptr;
    bool window = false;
    int frames = 0;   // --frames N: window test run, shown without activation, quits after N frames
    Camera cam{};
    cam.pos[0] = scene.center[0] - 40; cam.pos[1] = scene.center[1] - 40; cam.pos[2] = scene.center[2] + 25;
    cam.yaw = 0.785398f; cam.pitch = -0.35f;
    for (int i = 4; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
        else if (!std::strcmp(argv[i], "--window")) window = true;
        else if (!std::strcmp(argv[i], "--frames") && i + 1 < argc) frames = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--mode") && i + 1 < argc) {
            const char* m = argv[++i];
            cam.mode = !std::strcmp(m, "walk") ? 1 : !std::strcmp(m, "drive") ? 2 : 0;
        }
        else if (!std::strcmp(argv[i], "--size") && i + 2 < argc) { W = std::atoi(argv[++i]); H = std::atoi(argv[++i]); }
        else if (!std::strcmp(argv[i], "--fov") && i + 1 < argc) cam.fov = float(std::atof(argv[++i]));
        else if (!std::strcmp(argv[i], "--cam") && i + 5 < argc) {
            for (int k = 0; k < 3; ++k) cam.pos[k] = float(std::atof(argv[++i]));
            cam.yaw = float(std::atof(argv[++i])) * 3.14159265f / 180; cam.pitch = float(std::atof(argv[++i])) * 3.14159265f / 180;
        } else { std::fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
    }
#ifndef _WIN32
    if (window) { std::fprintf(stderr, "--window needs Windows\n"); return 2; }
#endif
    if (window && frames && !out_path) { std::fprintf(stderr, "--frames needs --out\n"); return 2; }
    if (!out_path && !window) { std::fprintf(stderr, "give --out or --window\n"); return 2; }
    std::fprintf(stderr, "scene: %zu draws, %zu meshes, %zu textures, %zu triangles\n", scene.draws.size(),
                 scene.meshes.size(), scene.textures.size(), scene.indices.size() / 3);

    gipa = load_vulkan();
    if (!gipa) { std::fprintf(stderr, "the Vulkan loader was not found\n"); return 77; }
    VkInstance inst = VK_NULL_HANDLE;
#define LI(n) auto n = reinterpret_cast<PFN_##n>(gipa(inst, #n))
    LI(vkCreateInstance);
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion = VK_API_VERSION_1_2;
    const char* inst_ext[2] = {"VK_KHR_surface", "VK_KHR_win32_surface"};
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo = &app;
    if (window) { ici.enabledExtensionCount = 2; ici.ppEnabledExtensionNames = inst_ext; }
    if (vkCreateInstance(&ici, nullptr, &inst) != VK_SUCCESS) { std::fprintf(stderr, "unsupported: no Vulkan instance\n"); return 77; }
    LI(vkEnumeratePhysicalDevices); LI(vkGetPhysicalDeviceProperties); LI(vkGetPhysicalDeviceQueueFamilyProperties);
    LI(vkGetPhysicalDeviceMemoryProperties); LI(vkCreateDevice); LI(vkGetDeviceProcAddr);
    uint32_t n = 0;
    CHECK(vkEnumeratePhysicalDevices(inst, &n, nullptr));
    if (!n) { std::fprintf(stderr, "unsupported: no Vulkan physical device\n"); return 77; }
    std::vector<VkPhysicalDevice> pds(n);
    CHECK(vkEnumeratePhysicalDevices(inst, &n, pds.data()));
    VkPhysicalDevice pd = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties props{};
    int best = -1;
    for (auto p : pds) {
        VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties(p, &pp);
        if (int r = leonida::vk_test::device_rank(pp.deviceType, pp.vendorID); r > best) { pd = p; props = pp; best = r; }
    }
    constexpr uint32_t push_words = 36, push_bytes = push_words * 4;
    if (props.limits.maxPushConstantsSize < push_bytes) { std::fprintf(stderr, "unsupported: push constants\n"); return 77; }
    std::fprintf(stderr, "device: %s\n", props.deviceName);
    uint32_t qn = 0; vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qf(qn); vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, qf.data());
    uint32_t qi = 0;
    while (qi < qn && !(qf[qi].queueFlags & VK_QUEUE_GRAPHICS_BIT)) ++qi;
    if (qi == qn) { std::fprintf(stderr, "unsupported: no graphics queue\n"); return 77; }

#ifdef _WIN32
    HWND hwnd = nullptr;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    if (window) {
        WNDCLASSW wc{}; wc.lpfnWndProc = wndproc; wc.hInstance = GetModuleHandleW(nullptr); wc.lpszClassName = L"vk_sandbox";
        wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
        RegisterClassW(&wc);
        RECT rc{0, 0, LONG(W), LONG(H)}; AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, FALSE);
        hwnd = CreateWindowExW(0, L"vk_sandbox", L"Leonida real-shader sandbox", WS_OVERLAPPEDWINDOW, CW_USEDEFAULT,
                               CW_USEDEFAULT, rc.right - rc.left, rc.bottom - rc.top, nullptr, nullptr, wc.hInstance, nullptr);
        ShowWindow(hwnd, frames ? SW_SHOWNOACTIVATE : SW_SHOWNORMAL);
        LI(vkCreateWin32SurfaceKHR);
        VkWin32SurfaceCreateInfoKHR sci{VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR};
        sci.hinstance = wc.hInstance; sci.hwnd = hwnd;
        CHECK(vkCreateWin32SurfaceKHR(inst, &sci, nullptr, &surface));
    }
#endif

    float prio = 1;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = qi; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    f12.bufferDeviceAddress = VK_TRUE; f12.runtimeDescriptorArray = VK_TRUE; f12.descriptorIndexing = VK_TRUE;
    f12.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2}; f2.pNext = &f12;
    f2.features.shaderInt64 = VK_TRUE;
    const char* dev_ext[1] = {"VK_KHR_swapchain"};
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &f2; dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    if (window) { dci.enabledExtensionCount = 1; dci.ppEnabledExtensionNames = dev_ext; }
    VkDevice dev; CHECK(vkCreateDevice(pd, &dci, nullptr, &dev));
#define LD(n) auto n = reinterpret_cast<PFN_##n>(vkGetDeviceProcAddr(dev, #n))
    LD(vkGetDeviceQueue); LD(vkCreateImage); LD(vkGetImageMemoryRequirements); LD(vkAllocateMemory);
    LD(vkBindImageMemory); LD(vkCreateImageView); LD(vkCreateRenderPass); LD(vkCreateFramebuffer);
    LD(vkCreateShaderModule); LD(vkCreatePipelineLayout); LD(vkCreateGraphicsPipelines); LD(vkCreateCommandPool);
    LD(vkAllocateCommandBuffers); LD(vkBeginCommandBuffer); LD(vkCmdBeginRenderPass); LD(vkCmdBindPipeline);
    LD(vkCmdDrawIndexed); LD(vkCmdEndRenderPass); LD(vkCmdCopyImageToBuffer); LD(vkCmdCopyBufferToImage);
    LD(vkEndCommandBuffer); LD(vkQueueSubmit); LD(vkQueueWaitIdle); LD(vkCreateBuffer); LD(vkResetCommandBuffer);
    LD(vkGetBufferMemoryRequirements); LD(vkBindBufferMemory); LD(vkMapMemory); LD(vkGetBufferDeviceAddress);
    LD(vkCreateSampler); LD(vkCreateDescriptorSetLayout); LD(vkCreateDescriptorPool); LD(vkAllocateDescriptorSets);
    LD(vkUpdateDescriptorSets); LD(vkCmdBindDescriptorSets); LD(vkCmdPushConstants); LD(vkCmdPipelineBarrier);
    LD(vkCmdBindVertexBuffers); LD(vkCmdBindIndexBuffer); LD(vkCmdBlitImage);
    VkQueue q; vkGetDeviceQueue(dev, qi, 0, &q);
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    auto memtype = [&](uint32_t bits, VkMemoryPropertyFlags want) {
        for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
            if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
        return 0u;
    };
    auto make_buffer = [&](VkDeviceSize size, VkBufferUsageFlags usage, VkBuffer* b, void** map) {
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bi.size = size; bi.usage = usage;
        VkResult r = vkCreateBuffer(dev, &bi, nullptr, b); if (r) return r;
        VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, *b, &mr);
        VkMemoryAllocateFlagsInfo fi{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO}; fi.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.pNext = &fi; ai.allocationSize = mr.size;
        ai.memoryTypeIndex = memtype(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VkDeviceMemory m; r = vkAllocateMemory(dev, &ai, nullptr, &m); if (r) return r;
        r = vkBindBufferMemory(dev, *b, m, 0); if (r) return r;
        return vkMapMemory(dev, m, 0, size, 0, map);
    };
    auto make_image = [&](uint32_t w, uint32_t h, VkFormat fmt, VkImageUsageFlags usage, VkImageAspectFlags aspect,
                          VkImage* img, VkImageView* view) {
        VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ii.imageType = VK_IMAGE_TYPE_2D; ii.format = fmt; ii.extent = {w, h, 1}; ii.mipLevels = 1; ii.arrayLayers = 1;
        ii.samples = VK_SAMPLE_COUNT_1_BIT; ii.tiling = VK_IMAGE_TILING_OPTIMAL; ii.usage = usage;
        VkResult r = vkCreateImage(dev, &ii, nullptr, img); if (r) return r;
        VkMemoryRequirements mr; vkGetImageMemoryRequirements(dev, *img, &mr);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize = mr.size;
        ai.memoryTypeIndex = memtype(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VkDeviceMemory m; r = vkAllocateMemory(dev, &ai, nullptr, &m); if (r) return r;
        r = vkBindImageMemory(dev, *img, m, 0); if (r) return r;
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = *img; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = fmt; vi.subresourceRange = {aspect, 0, 1, 0, 1};
        return vkCreateImageView(dev, &vi, nullptr, view);
    };

    // Heap: the Xbox GPU VA window. 0x1000 view constants, 0x1200 colour constants, then 0x80 per draw: the
    // descriptor table at +0 (0x70 bytes used) and nothing else; object constants follow all tables.
    const uint32_t view_va = 0x1000, colour_va = 0x1200, table_va = 0x2000;
    const uint32_t ndraw = uint32_t(scene.draws.size());
    const uint32_t object_va = table_va + ndraw * 0x80;
    const VkDeviceSize heap_bytes = object_va + VkDeviceSize(ndraw) * 0x40 + 0x1000;
    VkBuffer heap; void* heapp;
    CHECK(make_buffer(heap_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, &heap, &heapp));
    std::memset(heapp, 0, size_t(heap_bytes));
    auto heap_u32 = [&](uint32_t va) { return reinterpret_cast<uint32_t*>(static_cast<char*>(heapp) + va); };
    auto heap_f32 = [&](uint32_t va) { return reinterpret_cast<float*>(static_cast<char*>(heapp) + va); };
    VkBufferDeviceAddressInfo bdai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO}; bdai.buffer = heap;
    const VkDeviceAddress heap_base = vkGetBufferDeviceAddress(dev, &bdai);
    // A raw V#: base in word0/word1[15:0], stride 0, NUM_RECORDS bytes, OOB_SELECT 3 (raw byte range).
    auto vsharp = [&](uint32_t at, uint32_t base, uint32_t bytes) {
        uint32_t* p = heap_u32(at); p[0] = base; p[1] = 0; p[2] = bytes; p[3] = 3u << 28;
    };
    // T# keys: unique words per texture with the RGBA8 UNORM format in word 1 (the numeric type the shader reads).
    const uint32_t fmt_word = leonida::gfx10::format_from_vulkan(VK_FORMAT_R8G8B8A8_UNORM) << 20;
    auto tsharp = [&](uint32_t image) {
        std::array<uint32_t, 8> t{0x5A000000u | image, fmt_word, 0x7FF7FFu, 0, 0, 0, 0, 0};
        return t;
    };
    const uint32_t nimg = uint32_t(scene.textures.size()) + 1;   // image 0: 1x1 white for untextured draws
    leonida::DescriptorTable table(2 * (nimg + 2));
    for (uint32_t i = 0; i < nimg; ++i) table.insert(tsharp(i).data(), i);
    uint32_t ssharp[8] = {1024, 12582912, 418381824, 0, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu};   // the PS's literal S#
    table.insert(ssharp, 0);
    for (uint32_t d = 0; d < ndraw; ++d) {
        const Draw& dr = scene.draws[d];
        const uint32_t t = table_va + d * 0x80, o = object_va + d * 0x40;
        vsharp(t + 0x00, o, 0x40);
        const auto ts = tsharp(dr.texture < 0 ? 0 : uint32_t(dr.texture) + 1);
        std::memcpy(heap_u32(t + 0x20), ts.data(), 32);
        vsharp(t + 0x40, view_va, 0x1a0);
        vsharp(t + 0x60, colour_va, 0x10);
        float* ob = heap_f32(o);
        for (int r = 0; r < 4; ++r) for (int j = 0; j < 3; ++j) ob[r * 4 + j] = dr.rows[r][j];
    }
    float* colour = heap_f32(colour_va); colour[0] = colour[1] = colour[2] = colour[3] = 1;
    float* light = heap_f32(view_va + 0xa0);
    { const float l[3] = {-0.55f, -0.45f, 0.7f}; const float ln = std::sqrt(l[0] * l[0] + l[1] * l[1] + l[2] * l[2]);
      for (int k = 0; k < 3; ++k) light[k] = l[k] / ln; }
    const auto gpu_table = table.build_gpu_buffer(true);
    VkBuffer tb; void* tbp;
    CHECK(make_buffer(gpu_table.size() * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &tb, &tbp));
    std::memcpy(tbp, gpu_table.data(), gpu_table.size() * 4);

    // Geometry.
    VkBuffer vb, ib; void* vbp; void* ibp;
    CHECK(make_buffer(std::max<size_t>(32, scene.vertices.size() * 4), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, &vb, &vbp));
    CHECK(make_buffer(std::max<size_t>(4, scene.indices.size() * 4), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, &ib, &ibp));
    std::memcpy(vbp, scene.vertices.data(), scene.vertices.size() * 4);
    std::memcpy(ibp, scene.indices.data(), scene.indices.size() * 4);

    // Textures.
    std::vector<VkImage> timg(nimg); std::vector<VkImageView> tview(nimg);
    VkDeviceSize up_bytes = 4;
    for (const auto& t : scene.textures) up_bytes += t.rgba.size();
    VkBuffer up; void* upp;
    CHECK(make_buffer(up_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &up, &upp));
    std::vector<VkDeviceSize> up_off(nimg);
    { const uint32_t white = 0xFFFFFFFFu; std::memcpy(upp, &white, 4); VkDeviceSize at = 4;
      for (uint32_t i = 1; i < nimg; ++i) { up_off[i] = at; std::memcpy(static_cast<char*>(upp) + at, scene.textures[i - 1].rgba.data(), scene.textures[i - 1].rgba.size()); at += scene.textures[i - 1].rgba.size(); } }
    for (uint32_t i = 0; i < nimg; ++i)
        CHECK(make_image(i ? scene.textures[i - 1].w : 1, i ? scene.textures[i - 1].h : 1, VK_FORMAT_R8G8B8A8_UNORM,
                         VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT, &timg[i], &tview[i]));
    VkSamplerCreateInfo sci = leonida::gfx10::vulkan_sampler(leonida::gfx10::decode(
        leonida::gfx10::SamplerWords{ssharp[0], ssharp[1], ssharp[2], ssharp[3]}));
    sci.maxLod = 0;   // the scene stores mip 0 only
    VkSampler smp; CHECK(vkCreateSampler(dev, &sci, nullptr, &smp));

    // Targets: RGBA8 colour + D32 depth.
    const VkFormat cfmt = VK_FORMAT_R8G8B8A8_UNORM, dfmt = VK_FORMAT_D32_SFLOAT;
    VkImage rt, dt; VkImageView rtv, dtv;
    CHECK(make_image(W, H, cfmt, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_COLOR_BIT, &rt, &rtv));
    CHECK(make_image(W, H, dfmt, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, &dt, &dtv));
    VkAttachmentDescription ad[2]{};
    ad[0].format = cfmt; ad[0].samples = VK_SAMPLE_COUNT_1_BIT; ad[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    ad[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE; ad[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    ad[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    ad[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; ad[0].finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    ad[1] = ad[0]; ad[1].format = dfmt; ad[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    ad[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    VkAttachmentReference car{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}, dar{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sd{}; sd.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sd.colorAttachmentCount = 1; sd.pColorAttachments = &car; sd.pDepthStencilAttachment = &dar;
    VkSubpassDependency dep{0, VK_SUBPASS_EXTERNAL, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT};
    VkRenderPassCreateInfo rpi{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rpi.attachmentCount = 2; rpi.pAttachments = ad; rpi.subpassCount = 1; rpi.pSubpasses = &sd;
    rpi.dependencyCount = 1; rpi.pDependencies = &dep;
    VkRenderPass rp; CHECK(vkCreateRenderPass(dev, &rpi, nullptr, &rp));
    VkImageView fbv[2] = {rtv, dtv};
    VkFramebufferCreateInfo fbi{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fbi.renderPass = rp; fbi.attachmentCount = 2; fbi.pAttachments = fbv; fbi.width = W; fbi.height = H; fbi.layers = 1;
    VkFramebuffer fb; CHECK(vkCreateFramebuffer(dev, &fbi, nullptr, &fb));

    // Descriptor set: binding 0 images[], 1 samplers[], 2 the descriptor hash table.
    VkDescriptorSetLayoutBinding b[3]{};
    b[0] = {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, nimg, VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_VERTEX_BIT, nullptr};
    b[1] = {1, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_VERTEX_BIT, nullptr};
    b[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_VERTEX_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; dli.bindingCount = 3; dli.pBindings = b;
    VkDescriptorSetLayout dsl; CHECK(vkCreateDescriptorSetLayout(dev, &dli, nullptr, &dsl));
    VkDescriptorPoolSize ps[3]{{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, nimg}, {VK_DESCRIPTOR_TYPE_SAMPLER, 1}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; dpi.maxSets = 1; dpi.poolSizeCount = 3; dpi.pPoolSizes = ps;
    VkDescriptorPool dp; CHECK(vkCreateDescriptorPool(dev, &dpi, nullptr, &dp));
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; dai.descriptorPool = dp; dai.descriptorSetCount = 1; dai.pSetLayouts = &dsl;
    VkDescriptorSet ds; CHECK(vkAllocateDescriptorSets(dev, &dai, &ds));
    std::vector<VkDescriptorImageInfo> iinfo(nimg);
    for (uint32_t i = 0; i < nimg; ++i) iinfo[i] = {VK_NULL_HANDLE, tview[i], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo sinfo{smp};
    VkDescriptorBufferInfo tinfo{tb, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet wr[3]{};
    for (auto& w : wr) { w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w.dstSet = ds; }
    wr[0].dstBinding = 0; wr[0].descriptorCount = nimg; wr[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; wr[0].pImageInfo = iinfo.data();
    wr[1].dstBinding = 1; wr[1].descriptorCount = 1; wr[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER; wr[1].pImageInfo = &sinfo;
    wr[2].dstBinding = 2; wr[2].descriptorCount = 1; wr[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wr[2].pBufferInfo = &tinfo;
    vkUpdateDescriptorSets(dev, 3, wr, 0, nullptr);

    // Pipeline: the translated VS and PS, vertices (pos, normal, uv) at locations 0..2, depth test, no culling.
    auto module = [&](const char* p, VkShaderModule* m) {
        auto code = read_spv(p);
        if (code.empty()) { std::fprintf(stderr, "cannot read %s\n", p); return VK_ERROR_INITIALIZATION_FAILED; }
        VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; smi.codeSize = code.size() * 4; smi.pCode = code.data();
        return vkCreateShaderModule(dev, &smi, nullptr, m);
    };
    VkShaderModule vs, fs; CHECK(module(argv[2], &vs)); CHECK(module(argv[3], &fs));
    VkPipelineShaderStageCreateInfo st[2]{{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}, {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}};
    st[0].stage = VK_SHADER_STAGE_VERTEX_BIT; st[0].module = vs; st[0].pName = "main";
    st[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; st[1].module = fs; st[1].pName = "main";
    VkPushConstantRange pcr{VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_VERTEX_BIT, 0, push_bytes};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1; pli.pSetLayouts = &dsl; pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
    VkPipelineLayout pl; CHECK(vkCreatePipelineLayout(dev, &pli, nullptr, &pl));
    VkVertexInputBindingDescription vbd{0, 32, VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription vad[3]{{0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0}, {1, 0, VK_FORMAT_R32G32B32_SFLOAT, 12},
                                             {2, 0, VK_FORMAT_R32G32_SFLOAT, 24}};
    VkPipelineVertexInputStateCreateInfo vin{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vin.vertexBindingDescriptionCount = 1; vin.pVertexBindingDescriptions = &vbd;
    vin.vertexAttributeDescriptionCount = 3; vin.pVertexAttributeDescriptions = vad;
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO}; ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    // D3D clip space has y up: a negative-height viewport (Vulkan 1.1) keeps the game's matrices unchanged.
    VkViewport vp{0, float(H), float(W), -float(H), 0, 1}; VkRect2D sc{{0, 0}, {W, H}};
    VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vps.viewportCount = 1; vps.pViewports = &vp; vps.scissorCount = 1; vps.pScissors = &sc;
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO}; rs.cullMode = VK_CULL_MODE_NONE; rs.lineWidth = 1;
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO}; ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo dss{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    dss.depthTestEnable = VK_TRUE; dss.depthWriteEnable = VK_TRUE; dss.depthCompareOp = VK_COMPARE_OP_LESS;
    VkPipelineColorBlendAttachmentState cba{}; cba.colorWriteMask = 0xF;
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO}; cb.attachmentCount = 1; cb.pAttachments = &cba;
    VkGraphicsPipelineCreateInfo gpi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gpi.stageCount = 2; gpi.pStages = st; gpi.pVertexInputState = &vin; gpi.pInputAssemblyState = &ia; gpi.pViewportState = &vps;
    gpi.pRasterizationState = &rs; gpi.pMultisampleState = &ms; gpi.pDepthStencilState = &dss; gpi.pColorBlendState = &cb;
    gpi.layout = pl; gpi.renderPass = rp;
    VkPipeline pipe; CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpi, nullptr, &pipe));

    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; cpi.queueFamilyIndex = qi;
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool pool; CHECK(vkCreateCommandPool(dev, &cpi, nullptr, &pool));
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; cai.commandPool = pool; cai.commandBufferCount = 1;
    VkCommandBuffer cmd; CHECK(vkAllocateCommandBuffers(dev, &cai, &cmd));
    auto barrier = [&](VkImage img, VkImageLayout from, VkImageLayout to, VkAccessFlags sa, VkAccessFlags da,
                       VkPipelineStageFlags ss, VkPipelineStageFlags dst) {
        VkImageMemoryBarrier ib{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        ib.oldLayout = from; ib.newLayout = to; ib.image = img; ib.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        ib.srcAccessMask = sa; ib.dstAccessMask = da; ib.srcQueueFamilyIndex = ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        vkCmdPipelineBarrier(cmd, ss, dst, 0, 0, nullptr, 0, nullptr, 1, &ib);
    };
    auto submit = [&]() { VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
                          VkResult r = vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE); return r ? r : vkQueueWaitIdle(q); };
    {   // Upload the textures once.
        VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; CHECK(vkBeginCommandBuffer(cmd, &cbi));
        for (uint32_t i = 0; i < nimg; ++i) {
            barrier(timg[i], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
            VkBufferImageCopy c{}; c.bufferOffset = up_off[i]; c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            c.imageExtent = {i ? scene.textures[i - 1].w : 1, i ? scene.textures[i - 1].h : 1, 1};
            vkCmdCopyBufferToImage(cmd, up, timg[i], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
            barrier(timg[i], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
        }
        CHECK(vkEndCommandBuffer(cmd)); CHECK(submit());
    }
    // Records the scene into rt (left in TRANSFER_SRC_OPTIMAL). The caller ends and submits.
    auto record = [&]() {
        view_projection(cam, float(W) / float(H), heap_f32(view_va + 0x160));
        VkClearValue clear[2]{}; clear[0].color = {{0.55f, 0.68f, 0.85f, 1}}; clear[1].depthStencil = {1, 0};
        VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        rbi.renderPass = rp; rbi.framebuffer = fb; rbi.renderArea = sc; rbi.clearValueCount = 2; rbi.pClearValues = clear;
        vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pl, 0, 1, &ds, 0, nullptr);
        VkDeviceSize zero = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &zero);
        vkCmdBindIndexBuffer(cmd, ib, 0, VK_INDEX_TYPE_UINT32);
        uint32_t pc[push_words] = {};
        pc[32] = uint32_t(heap_base); pc[33] = uint32_t(heap_base >> 32);
        for (uint32_t d = 0; d < ndraw; ++d) {
            const Mesh& m = scene.meshes[scene.draws[d].mesh];
            pc[8] = table_va + d * 0x80;   // s8: the descriptor table; s9 (the PS's M0) stays 0
            vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_VERTEX_BIT, 0, push_bytes, pc);
            vkCmdDrawIndexed(cmd, m.index_count, 1, m.first_index, int32_t(m.first_vertex), 0);
        }
        vkCmdEndRenderPass(cmd);
    };
    VkBuffer rb; void* rbp;
    CHECK(make_buffer(VkDeviceSize(W) * H * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &rb, &rbp));
    auto read_back = [&]() {
        VkBufferImageCopy rc{}; rc.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; rc.imageExtent = {W, H, 1};
        vkCmdCopyImageToBuffer(cmd, rt, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb, 1, &rc);
    };

    if (!window) {
        VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; CHECK(vkBeginCommandBuffer(cmd, &cbi));
        record(); read_back();
        CHECK(vkEndCommandBuffer(cmd)); CHECK(submit());
        if (!write_ppm(out_path, static_cast<const uint8_t*>(rbp), W, H, false)) { std::fprintf(stderr, "cannot write %s\n", out_path); return 1; }
        // Coverage: pixels that differ from the clear colour were written by the translated PS.
        const uint8_t* px = static_cast<const uint8_t*>(rbp);
        const uint8_t clear_px[3] = {uint8_t(0.55f * 255 + 0.5f), uint8_t(0.68f * 255 + 0.5f), uint8_t(0.85f * 255 + 0.5f)};
        size_t covered = 0;
        for (size_t i = 0; i < size_t(W) * H; ++i)
            covered += std::abs(px[i * 4] - clear_px[0]) > 1 || std::abs(px[i * 4 + 1] - clear_px[1]) > 1 || std::abs(px[i * 4 + 2] - clear_px[2]) > 1;
        std::printf("vk_sandbox: %u draws, %zu of %u pixels covered -> %s\n", ndraw, covered, W * H, out_path);
        return 0;
    }

#ifdef _WIN32
    LI(vkGetPhysicalDeviceSurfaceCapabilitiesKHR); LI(vkGetPhysicalDeviceSurfaceFormatsKHR); LI(vkGetPhysicalDeviceSurfaceSupportKHR);
    LD(vkCreateSwapchainKHR); LD(vkDestroySwapchainKHR); LD(vkGetSwapchainImagesKHR); LD(vkAcquireNextImageKHR);
    LD(vkQueuePresentKHR); LD(vkCreateSemaphore);
    VkBool32 can_present = VK_FALSE; vkGetPhysicalDeviceSurfaceSupportKHR(pd, qi, surface, &can_present);
    if (!can_present) { std::fprintf(stderr, "the graphics queue cannot present\n"); return 1; }
    uint32_t nf = 0; vkGetPhysicalDeviceSurfaceFormatsKHR(pd, surface, &nf, nullptr);
    std::vector<VkSurfaceFormatKHR> sfs(nf); vkGetPhysicalDeviceSurfaceFormatsKHR(pd, surface, &nf, sfs.data());
    if (sfs.empty()) { std::fprintf(stderr, "the surface has no formats\n"); return 1; }
    VkSurfaceFormatKHR sf = sfs[0];   // a UNORM format keeps the blit byte-exact with the headless render
    for (const VkSurfaceFormatKHR& c : sfs)
        if (c.format == VK_FORMAT_B8G8R8A8_UNORM || c.format == VK_FORMAT_R8G8B8A8_UNORM) { sf = c; break; }
    VkSwapchainCreateInfoKHR swi{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    VkSwapchainKHR swap = VK_NULL_HANDLE;
    std::vector<VkImage> simg;
    // (Re)creates the swapchain for the window's current size; false while it is minimised (extent 0).
    auto make_swap = [&]() -> int {
        VkSurfaceCapabilitiesKHR caps;
        if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(pd, surface, &caps) != VK_SUCCESS) return -1;
        if (!(caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT)) return -1;
        const VkExtent2D ext = caps.currentExtent.width != 0xFFFFFFFFu ? caps.currentExtent : VkExtent2D{W, H};
        if (!ext.width || !ext.height) return 0;
        vkQueueWaitIdle(q);
        swi.surface = surface; swi.minImageCount = std::max(2u, caps.minImageCount);
        if (caps.maxImageCount) swi.minImageCount = std::min(swi.minImageCount, caps.maxImageCount);
        swi.imageFormat = sf.format; swi.imageColorSpace = sf.colorSpace; swi.imageExtent = ext;
        swi.imageArrayLayers = 1; swi.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT; swi.preTransform = caps.currentTransform;
        swi.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR; swi.presentMode = VK_PRESENT_MODE_FIFO_KHR; swi.clipped = VK_TRUE;
        swi.oldSwapchain = swap;
        VkSwapchainKHR next;
        if (vkCreateSwapchainKHR(dev, &swi, nullptr, &next) != VK_SUCCESS) return -1;
        if (swap) vkDestroySwapchainKHR(dev, swap, nullptr);
        swap = next;
        uint32_t ns = 0; vkGetSwapchainImagesKHR(dev, swap, &ns, nullptr);
        simg.resize(ns); vkGetSwapchainImagesKHR(dev, swap, &ns, simg.data());
        return 1;
    };
    if (make_swap() < 0) { std::fprintf(stderr, "cannot create a swapchain (TRANSFER_DST unsupported?)\n"); return 1; }
    VkSemaphoreCreateInfo semi{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkSemaphore acquired; CHECK(vkCreateSemaphore(dev, &semi, nullptr, &acquired));
    Ground ground; ground.build(scene);
    auto last = std::chrono::steady_clock::now();
    bool shot_held = false;
    int frame = 0;
    while (!g_quit) {
        MSG msg;
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) { TranslateMessage(&msg); DispatchMessageW(&msg); }
        const auto now = std::chrono::steady_clock::now();
        const float dt = std::min(0.1f, std::chrono::duration<float>(now - last).count()); last = now;
        const bool focus = GetForegroundWindow() == hwnd;
        auto key = [&](int vk) { return focus && (GetAsyncKeyState(vk) & 0x8000); };
        if (key(VK_ESCAPE)) break;
        if (key('1')) cam.mode = 0; if (key('2')) cam.mode = 1; if (key('3')) cam.mode = 2;
        cam.yaw -= g_dx * 0.004f; cam.pitch = std::clamp(cam.pitch - g_dy * 0.004f, -1.5f, 1.5f); g_dx = g_dy = 0;
        float f[3]; cam.forward(f);
        const float r[2] = {std::sin(cam.yaw), -std::cos(cam.yaw)};
        const float fwd = float(key('W')) - float(key('S')), side = float(key('D')) - float(key('A'));
        if (cam.mode == 0) {
            const float v = (key(VK_SHIFT) ? 60.f : 12.f) * dt;
            for (int k = 0; k < 3; ++k) cam.pos[k] += f[k] * fwd * v;
            cam.pos[0] += r[0] * side * v; cam.pos[1] += r[1] * side * v;
            cam.pos[2] += (float(key('E')) - float(key('Q'))) * v;
        } else {
            const float h = std::hypot(f[0], f[1]) > 1e-4f ? std::hypot(f[0], f[1]) : 1;
            if (cam.mode == 1) cam.speed = fwd * (key(VK_SHIFT) ? 7.f : 2.5f);
            else { cam.speed += fwd * 12.f * dt; cam.speed *= std::pow(0.6f, dt); cam.yaw -= side * dt * std::clamp(cam.speed / 10.f, -1.f, 1.f); }
            cam.pos[0] += f[0] / h * cam.speed * dt; cam.pos[1] += f[1] / h * cam.speed * dt;
            if (cam.mode == 1) { cam.pos[0] += r[0] * side * 2.5f * dt; cam.pos[1] += r[1] * side * 2.5f * dt; }
            const float eye = cam.mode == 1 ? 1.7f : 1.2f;
            float gz;
            if (ground.height(cam.pos[0], cam.pos[1], cam.pos[2] - eye + 0.6f, gz)) cam.pos[2] = gz + eye;
        }
        uint32_t idx = 0;
        if (!swap) {   // minimised at start or since the last resize
            const int made = make_swap();
            if (made < 0) { std::fprintf(stderr, "cannot recreate the swapchain\n"); return 1; }
            if (!made) { Sleep(50); continue; }
        }
        const VkResult acq = vkAcquireNextImageKHR(dev, swap, UINT64_MAX, acquired, VK_NULL_HANDLE, &idx);
        if (acq == VK_ERROR_OUT_OF_DATE_KHR) {
            const int made = make_swap();
            if (made < 0) { std::fprintf(stderr, "cannot recreate the swapchain\n"); return 1; }
            if (!made) { vkDestroySwapchainKHR(dev, swap, nullptr); swap = VK_NULL_HANDLE; Sleep(50); }
            continue;
        }
        if (acq != VK_SUCCESS && acq != VK_SUBOPTIMAL_KHR) { std::fprintf(stderr, "vkAcquireNextImageKHR failed (%d)\n", int(acq)); return 1; }
        CHECK(vkResetCommandBuffer(cmd, 0));
        VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; CHECK(vkBeginCommandBuffer(cmd, &cbi));
        record();
        const bool last_frame = frames && ++frame >= frames;
        const bool shot = (key('P') && !shot_held) || last_frame; shot_held = key('P');
        if (shot) read_back();
        barrier(simg[idx], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkImageBlit bl{}; bl.srcSubresource = bl.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        bl.srcOffsets[1] = {int32_t(W), int32_t(H), 1};
        bl.dstOffsets[1] = {int32_t(swi.imageExtent.width), int32_t(swi.imageExtent.height), 1};
        vkCmdBlitImage(cmd, rt, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, simg[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bl, VK_FILTER_LINEAR);
        barrier(simg[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_ACCESS_TRANSFER_WRITE_BIT, 0,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
        CHECK(vkEndCommandBuffer(cmd));
        VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.waitSemaphoreCount = 1; si.pWaitSemaphores = &acquired;
        si.pWaitDstStageMask = &wait_stage; si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
        CHECK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE)); CHECK(vkQueueWaitIdle(q));
        if (shot) {
            const char* path = last_frame ? out_path : "sandbox_shot.ppm";
            if (!write_ppm(path, static_cast<const uint8_t*>(rbp), W, H, false)) {
                std::fprintf(stderr, "cannot write %s\n", path);
                if (last_frame) return 1;
            }
        }
        VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR}; pi.swapchainCount = 1; pi.pSwapchains = &swap; pi.pImageIndices = &idx;
        const VkResult pres = vkQueuePresentKHR(q, &pi);
        if ((pres == VK_ERROR_OUT_OF_DATE_KHR || pres == VK_SUBOPTIMAL_KHR || acq == VK_SUBOPTIMAL_KHR) && !last_frame) {
            const int made = make_swap();
            if (made < 0) { std::fprintf(stderr, "cannot recreate the swapchain\n"); return 1; }
            if (!made) { vkDestroySwapchainKHR(dev, swap, nullptr); swap = VK_NULL_HANDLE; }
        }
        char title[160];
        std::snprintf(title, sizeof title, "Leonida real-shader sandbox  [%s]  %.0f %.0f %.0f  %.0f fps", cam.mode == 0 ? "fly" : cam.mode == 1 ? "walk" : "drive",
                      cam.pos[0], cam.pos[1], cam.pos[2], 1.f / std::max(dt, 1e-4f));
        SetWindowTextA(hwnd, title);
        if (last_frame) {
            std::printf("vk_sandbox: window ran %d frames -> %s, camera %.2f %.2f %.2f\n", frame, out_path,
                        cam.pos[0], cam.pos[1], cam.pos[2]);
            break;
        }
    }
    vkQueueWaitIdle(q);
#endif
    return 0;
}
