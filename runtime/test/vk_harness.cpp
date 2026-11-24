// Headless Vulkan harness: draws one triangle (3 vertices, no vertex buffers) with a given VS + FS into a
// W x H RGBA32F image and writes the raw floats to a file. Used to check translated shaders on the real GPU.
//
//   vk_harness <vs.spv> <fs.spv> <out.raw> [width height [clip-test-s0 clip-test-s1]]
//   vk_harness <ms.spv> <fs.spv> <out.raw> --mesh-group-readback X Y Z
// The latter captures one full-screen triangle per mesh workgroup, in Vulkan rasterization order.
//
// Push constants follow the contract in docs/design.md: s0..s31, then the heap base lo/hi. They are zero unless
//   VKH_SGPRS=w0,w1,...          sets s0, s1, ... (decimal or 0x hex)
//   VKH_GS_PRIMITIVES=1          reports "gs_primitives=N" on stderr for the VKH_GS draw (77 if unsupported)
//   VKH_HEAP=addr=word,...       allocates a zeroed 64 KiB heap (Xbox VA 0 = its device address), writes each word at
//                                its address and passes the heap base, so a V# in VKH_SGPRS can point into it.
//   VKH_HEAP_DUMP=<file>         after the draw, writes the 64 KiB heap (needs VKH_HEAP, which may be empty-valued as
//                                "0=0"), so a test can read what the pixel shader stored or atomically added.
//   VKH_DRAW_MESH=<file> + VKH_VERTEX_LAYOUT=0:3f,1:2f
//                                draws an indexed mesh instead of the 3-vertex triangle. The file is u32
//                                vertex_count, u32 index_count, vertex_count*stride bytes, index_count u32
//                                indices; the layout lists location:float-components, one item per attribute.
//   VKH_BINARY_OUT=<file>        writes the pipeline's VK_KHR_pipeline_binary data after building it
//   VKH_BINARY_IN=<file>         builds the pipeline from such a file instead (tools/ps_oracle.py patches the pixel
//                                shader in it); both exit 77 without VK_KHR_pipeline_binary
//   VKH_COLOR_FORMAT=<VkFormat>:<bytes>  the color target's format and texel size (default RGBA32F, 16); out.raw
//                                holds the texels as stored; 77 when the format is no color attachment
//   VKH_SUBGROUP_SIZE=32|64      requires that subgroup size of the fragment shader (or of the compute or geometry
//                                stage under test); stderr says "fragment subgroup size: N" when it took effect and
//                                "... not available" when the device cannot require it (exit code stays 0).
//   VKH_TEX_RGBA=<file>          binds a 2D RGBA8 texture (u32 width, u32 height, width*height*4 bytes) at bindless
//                                set 0 image index 1 (with 1x1 magenta at index 0), writes a T# at VKH_TSHARP_VA
//                                (default 0x100) in the 64 KiB heap, and registers VKH_SSHARP (4 comma-separated
//                                dwords, default 0x92,0xc00000,0x14500000,0) at sampler index 1.
//
#define VK_NO_PROTOTYPES
#include "vulkan_loader.h"
#include "vk_device_pick.h"
#include "vk_shader_fma.h"
#include "vk_pc_tail.h"
#include "../src/descriptor_table.h"
#include "../src/gfx10_descriptors.h"
#include "spirv_capability.h"

#include <cctype>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <utility>
#include <vector>

static PFN_vkGetInstanceProcAddr gipa;
#define LOADI(name) auto name = reinterpret_cast<PFN_##name>(gipa(inst, #name))
#define CHECK(x)                                                                 \
    do {                                                                         \
        VkResult r_ = (x);                                                       \
        if (r_ != VK_SUCCESS) {                                                  \
            std::fprintf(stderr, "FAILED %s -> %d (line %d)\n", #x, r_, __LINE__); \
            return 1;                                                            \
        }                                                                        \
    } while (0)

static std::vector<uint32_t> read_spv(const char* path) {
    std::ifstream f(path, std::ios::binary);
    std::vector<char> b((std::istreambuf_iterator<char>(f)), {});
    std::vector<uint32_t> w(b.size() / 4);
    std::memcpy(w.data(), b.data(), w.size() * 4);
    return w;
}

static bool parse_u32(const char*& s, uint32_t& v) {
    const bool hex = s[0] == '0' && (s[1] == 'x' || s[1] == 'X');
    const char* digits = hex ? s + 2 : s;
    if (!(hex ? std::isxdigit(uint8_t(*digits)) : std::isdigit(uint8_t(*digits)))) return false;   // no sign or space
    char* end;
    unsigned long long x = std::strtoull(digits, &end, hex ? 16 : 10);
    if (x > 0xFFFFFFFFull) return false;
    v = uint32_t(x); s = end;
    return true;
}

// A list separator must be followed by another item: "1,2," is rejected.
static bool next_item(const char*& s) {
    if (!*s) return true;
    return *s++ == ',' && *s;
}

static bool parse_sgprs(const char* s, uint32_t* sgprs) {
    for (uint32_t i = 0; *s; i++) {
        if (i == 32 || !parse_u32(s, sgprs[i]) || !next_item(s)) return false;
    }
    return true;
}

static bool parse_heap(const char* s, std::vector<std::pair<uint32_t, uint32_t>>& words, uint32_t heap_bytes) {
    while (*s) {
        uint32_t addr, word;
        if (!parse_u32(s, addr) || *s++ != '=' || !parse_u32(s, word) || !next_item(s)) return false;
        if (addr % 4 || addr > heap_bytes - 4) return false;
        words.emplace_back(addr, word);
    }
    return true;
}

// VKH_VERTEX_LAYOUT=0:3f,1:2f: one attribute per item, float components only. Location 0 is the first float3.
struct VertexAttr {
    uint32_t location, components, offset;
    VkFormat format;
};

static bool parse_vertex_layout(const char* s, VertexAttr* attrs, uint32_t& count, uint32_t& stride) {
    count = 0; stride = 0;
    while (*s) {
        uint32_t loc, comps;
        if (count == 8 || !parse_u32(s, loc) || *s++ != ':' || !parse_u32(s, comps) ||
            comps < 1 || comps > 4 || *s++ != 'f' || !next_item(s))
            return false;
        static const VkFormat formats[4] = {VK_FORMAT_R32_SFLOAT, VK_FORMAT_R32G32_SFLOAT,
                                            VK_FORMAT_R32G32B32_SFLOAT, VK_FORMAT_R32G32B32A32_SFLOAT};
        attrs[count++] = {loc, comps, stride, formats[comps - 1]};
        stride += comps * 4;
    }
    return count > 0;
}

// VKH_DRAW_MESH file: u32 vertex_count, u32 index_count, vertex_count*stride bytes, index_count u32 indices.
static bool read_mesh(const char* path, uint32_t stride, std::vector<uint8_t>& vertices,
                      std::vector<uint32_t>& indices) {
    std::ifstream f(path, std::ios::binary);
    uint32_t vcount = 0, icount = 0;
    f.read(reinterpret_cast<char*>(&vcount), 4);
    f.read(reinterpret_cast<char*>(&icount), 4);
    if (!f || vcount == 0 || icount == 0 || vcount > 10000000u || icount > 30000000u) return false;
    vertices.resize(size_t(vcount) * stride);
    indices.resize(icount);
    f.read(reinterpret_cast<char*>(vertices.data()), std::streamsize(vertices.size()));
    f.read(reinterpret_cast<char*>(indices.data()), std::streamsize(indices.size() * 4));
    return bool(f);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: vk_harness vs.spv fs.spv out.raw [w h [clip-test-s0 clip-test-s1]]\n"
                             "       vk_harness ms.spv fs.spv out.raw --mesh-group-readback X Y Z\n");
        return 2;
    }
    const bool group_readback = argc > 4 && !std::strcmp(argv[4], "--mesh-group-readback");
    const bool clip_shifts = !group_readback && argc == 8;
    uint32_t groups[3] = {1, 1, 1};
    uint32_t W = 64, H = 64;
    if (group_readback) {
        if (argc != 8) {
            std::fprintf(stderr, "--mesh-group-readback requires X Y Z\n");
            return 2;
        }
        W = H = 1;
        for (unsigned i = 0; i < 3; ++i) {
            char* end;
            long n = std::strtol(argv[5 + i], &end, 10);
            if (*end || n < 1 || n > 256 / W) {
                std::fprintf(stderr, "mesh grid must contain 1..256 groups (8-bit stencil)\n");
                return 2;
            }
            groups[i] = uint32_t(n);
            W *= groups[i];
        }
    } else if (argc > 5) {
        W = atoi(argv[4]); H = atoi(argv[5]);
    }
    // Push constants: s0..s31, heap base (32..33), code base (34..35), --mesh-compute output address (36..37).
    uint32_t pc[38] = {};
    const char* draw_mesh_path = std::getenv("VKH_DRAW_MESH");
    if (draw_mesh_path && !*draw_mesh_path) draw_mesh_path = nullptr;
    VertexAttr vertex_attrs[8];
    uint32_t vertex_attr_count = 0, vertex_stride = 0;
    if (draw_mesh_path) {
        const char* layout = std::getenv("VKH_VERTEX_LAYOUT");
        if (!layout || !parse_vertex_layout(layout, vertex_attrs, vertex_attr_count, vertex_stride)) {
            std::fprintf(stderr, "VKH_VERTEX_LAYOUT: expected locations like 0:3f,1:2f with VKH_DRAW_MESH\n");
            return 2;
        }
    }
    const char* sgprs_env = std::getenv("VKH_SGPRS");
    if (sgprs_env && !parse_sgprs(sgprs_env, pc)) {
        std::fprintf(stderr, "VKH_SGPRS: expected up to 32 comma-separated words, got \"%s\"\n", sgprs_env);
        return 2;
    }
    const uint32_t heap_bytes = 64 * 1024;
    const char* heap_env = std::getenv("VKH_HEAP");
    if (heap_env && !*heap_env) heap_env = nullptr;
    std::vector<std::pair<uint32_t, uint32_t>> heap_words;
    if (heap_env && !parse_heap(heap_env, heap_words, heap_bytes)) {
        std::fprintf(stderr, "VKH_HEAP: expected addr=word pairs, 4-byte aligned below 0x%x, got \"%s\"\n",
                     heap_bytes, heap_env);
        return 2;
    }
    const char* tex_rgba_path = std::getenv("VKH_TEX_RGBA");
    if (tex_rgba_path && !*tex_rgba_path) tex_rgba_path = nullptr;
    uint32_t tsharp_va = 0x100;
    uint32_t ssharp[8] = {0x92, 0xc00000, 0x14500000, 0,
                          0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu};
    uint32_t tex_w = 0, tex_h = 0;
    std::vector<uint8_t> tex_pixels;
    if (tex_rgba_path) {
        if (const char* va_env = std::getenv("VKH_TSHARP_VA"); va_env && *va_env) {
            const char* p = va_env;
            if (!parse_u32(p, tsharp_va) || *p || (tsharp_va & 3u) || tsharp_va > heap_bytes - 32) {
                std::fprintf(stderr, "VKH_TSHARP_VA: expected 4-byte aligned offset at most 0x%x, got \"%s\"\n",
                             heap_bytes - 32, va_env);
                return 2;
            }
        }
        if (const char* ss_env = std::getenv("VKH_SSHARP"); ss_env && *ss_env) {
            const char* p = ss_env;
            for (int i = 0; i < 4; ++i) {
                if (i && *p++ != ',') {
                    std::fprintf(stderr, "VKH_SSHARP: expected 4 comma-separated dwords, got \"%s\"\n", ss_env);
                    return 2;
                }
                if (!parse_u32(p, ssharp[i])) {
                    std::fprintf(stderr, "VKH_SSHARP: expected 4 comma-separated dwords, got \"%s\"\n", ss_env);
                    return 2;
                }
            }
            if (*p) {
                std::fprintf(stderr, "VKH_SSHARP: expected 4 comma-separated dwords, got \"%s\"\n", ss_env);
                return 2;
            }
        }
        std::ifstream tf(tex_rgba_path, std::ios::binary);
        uint32_t hdr[2] = {};
        if (!tf.read(reinterpret_cast<char*>(hdr), 8) || hdr[0] == 0 || hdr[1] == 0 ||
            uint64_t(hdr[0]) * uint64_t(hdr[1]) > (1ull << 26)) {
            std::fprintf(stderr, "VKH_TEX_RGBA: invalid header in %s\n", tex_rgba_path);
            return 1;
        }
        tex_w = hdr[0];
        tex_h = hdr[1];
        tex_pixels.resize(size_t(tex_w) * size_t(tex_h) * 4);
        if (!tf.read(reinterpret_cast<char*>(tex_pixels.data()), std::streamsize(tex_pixels.size()))) {
            std::fprintf(stderr, "VKH_TEX_RGBA: truncated pixel data in %s\n", tex_rgba_path);
            return 1;
        }
    }
    const bool need_heap = heap_env != nullptr || tex_rgba_path != nullptr;
    gipa = load_vulkan();
    if (!gipa) return 1;
    VkInstance inst = VK_NULL_HANDLE;
    LOADI(vkCreateInstance);
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    CHECK(vkCreateInstance(&ici, nullptr, &inst));
    LOADI(vkEnumeratePhysicalDevices); LOADI(vkGetPhysicalDeviceProperties);
    LOADI(vkGetPhysicalDeviceProperties2); LOADI(vkGetPhysicalDeviceFeatures2);
    LOADI(vkGetPhysicalDeviceQueueFamilyProperties); LOADI(vkGetPhysicalDeviceMemoryProperties);
    LOADI(vkCreateDevice); LOADI(vkGetDeviceProcAddr);
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> pds(n);
    vkEnumeratePhysicalDevices(inst, &n, pds.data());
    std::vector<VkPhysicalDeviceProperties> candidates(pds.size());
    for (std::size_t i = 0; i < pds.size(); ++i) vkGetPhysicalDeviceProperties(pds[i], &candidates[i]);
    const std::size_t chosen = leonida::vk_test::pick_device(candidates);
    VkPhysicalDevice pd = chosen < pds.size() ? pds[chosen] : VK_NULL_HANDLE;
    if (!pd) {
        leonida::vk_test::report_no_device();
        return 1;
    }
    VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(pd, &props);
    VkPhysicalDeviceSubgroupSizeControlProperties sgProps{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
    VkPhysicalDeviceDriverProperties driver{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
    if (props.apiVersion >= VK_API_VERSION_1_2) sgProps.pNext = &driver;
    VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    props2.pNext = &sgProps;
    vkGetPhysicalDeviceProperties2(pd, &props2);
    std::fprintf(stderr, "device: %s (subgroup %u..%u)\n", props.deviceName, sgProps.minSubgroupSize, sgProps.maxSubgroupSize);
    if (driver.driverName[0]) std::fprintf(stderr, "driver: %s\n", driver.driverName);
    uint32_t qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qf(qn);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, qf.data());
    uint32_t qi = 0;
    while (qi < qn && !(qf[qi].queueFlags & VK_QUEUE_GRAPHICS_BIT)) qi++;
    float prio = 1;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = qi; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    // Optional features some translated shaders need (enabled only when the device has them):
    // raw barycentric reads (BaryCoordKHR), POPS ordering (pixel interlock), non-constant texel offsets (maintenance8).
    auto vkEnumerateDeviceExtensionProperties = reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(
        gipa(inst, "vkEnumerateDeviceExtensionProperties"));
    uint32_t en = 0;
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &en, nullptr);
    std::vector<VkExtensionProperties> exts(en);
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &en, exts.data());
    auto has = [&](const char* e) {
        for (auto& x : exts) if (!std::strcmp(x.extensionName, e)) return true;
        return false;
    };
    // A translated pixel shader that reads BaryCoordKHR aborts the driver when the device lacks barycentrics
    // (llvmpipe and other software devices), so the suite reports it as unsupported instead. 77 is the existing
    // "this device cannot run the test" code (runtime/test/shaders/check.py, scripts/run_gpu_tests.py).
    const std::vector<uint32_t> fs_code = read_spv(argv[2]);
    bool needs_barycentric = false, needs_float64 = false;
    for (std::size_t i = 5; i + 1 < fs_code.size();) {   // instructions follow the 5-word SPIR-V header
        const uint32_t words = fs_code[i] >> 16, op = fs_code[i] & 0xffffu;
        if (!words) break;
        if (op == 17 && fs_code[i + 1] == 5284)          // OpCapability FragmentBarycentricKHR
            needs_barycentric = true;
        if (op == 17 && fs_code[i + 1] == 10)            // OpCapability Float64 (v_add_f64 and friends)
            needs_float64 = true;
        i += words;
    }
    if (needs_barycentric && !has(VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME)) {
        std::fprintf(stderr, "unsupported: %s reads BaryCoordKHR and the device has no %s\n",
                     argv[2], VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME);
        return 77;
    }
    // OpFmaKHR (rdna2spirv.py --fma khr) needs VK_KHR_shader_fma; a module of ANY stage that needs it is refused
    // (argv[1] is the vertex shader, or the mesh shader run as compute under VKH_MESH_COMPUTE).
    leonida::vk_test::ShaderFma fma;
    fma.query(gipa, inst, pd);
    if (!fma.accepts(fs_code.data(), fs_code.size() * 4, argv[2])) return 77;
    if (!fma.accepts_file(argv[1])) return 77;
    if (!fma.accepts_file(std::getenv("VKH_MESH_COMPUTE"))) return 77;
    std::vector<const char*> enable;
    void* chain = nullptr;
    if (fma.supported()) {
        enable.push_back(fma.extension());
#ifdef VK_KHR_SHADER_FMA_EXTENSION_NAME  // older Vulkan headers have no VkPhysicalDeviceShaderFmaFeaturesKHR
        fma.features.pNext = chain; chain = &fma.features;
#endif
    }
    VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR bary{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR};
    VkPhysicalDeviceFragmentShaderInterlockFeaturesEXT lock{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_INTERLOCK_FEATURES_EXT};
    if (has(VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME)) {
        enable.push_back(VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME);
        bary.fragmentShaderBarycentric = VK_TRUE; bary.pNext = chain; chain = &bary;
    }
    if (has(VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME)) {
        enable.push_back(VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME);
        lock.fragmentShaderPixelInterlock = VK_TRUE; lock.pNext = chain; chain = &lock;
    }
    // VKH_MESH_COMPUTE=<vs.spv>: the first shader is a mesh shader translated with rdna2spirv.py --mesh-compute.
    // It is dispatched as compute over the mesh grid (1x1x1, or --mesh-group-readback's), writing its slots to an
    // output buffer (VKH_MESH_SLOT bytes per workgroup, address in push-constant dwords 36..37). Then
    // vs.spv (tools/mesh_compute_vs.py) draws groups * VKH_MESH_PRIMS * 3 vertices with the fragment shader. No
    // VK_EXT_mesh_shader is needed.
    const char* mesh_vs_path = std::getenv("VKH_MESH_COMPUTE");
    if (mesh_vs_path && !*mesh_vs_path) mesh_vs_path = nullptr;
    const bool mesh_compute = mesh_vs_path != nullptr;
    uint64_t mesh_slot = 0;
    uint32_t mesh_prims = 0;
    if (mesh_compute) {
        const char* slot_env = std::getenv("VKH_MESH_SLOT");
        const char* prims_env = std::getenv("VKH_MESH_PRIMS");
        mesh_slot = slot_env ? std::strtoull(slot_env, nullptr, 0) : 0;
        mesh_prims = prims_env ? uint32_t(std::strtoul(prims_env, nullptr, 0)) : 0;
        if (!mesh_slot || !mesh_prims) {
            std::fprintf(stderr, "VKH_MESH_COMPUTE needs VKH_MESH_SLOT and VKH_MESH_PRIMS (index.json mesh_compute)\n");
            return 2;
        }
    }
    // Translated shaders declare a 144-byte push-constant block (user SGPRs, heap base, code base; see VKH_SGPRS /
    // VKH_HEAP); --mesh-compute modules and their vertex shaders add the output address (152 bytes). A device with
    // fewer bytes (128, the Vulkan minimum, as on some Intel Arc drivers and AMD's Windows driver for RDNA2 iGPUs)
    // cannot take that range: declaring it anyway is invalid, and there a shader reading past the limit crashed or
    // hung the device. 77, as vk_texture_test and vk_bindless_test: a device limit, not a wrong result (D3D-14).
    // LEONIDA_PC_TAIL_UBO=1 (rdna2spirv.py --pc-tail-ubo): s0..s31 only, 128 bytes, which every device has; the
    // tail is a uniform block at set 1 (vk_pc_tail.h). tools/mesh_compute_vs.py's vertex shader still reads the
    // --mesh-compute output address from push constants, so that combination isn't run.
    const bool pc_tail = leonida::vk_test::pc_tail_requested();
    if (pc_tail && mesh_compute) {
        std::fprintf(stderr, "unsupported: --mesh-compute's vertex shader reads the output address from push "
                             "constants, not from the --pc-tail-ubo block\n");
        return 77;
    }
    const uint32_t pc_bytes = pc_tail ? 128u : mesh_compute ? 152u : 144u;
    if (props.limits.maxPushConstantsSize < pc_bytes) {
        std::fprintf(stderr, "unsupported: need %u push constant bytes, device has %u\n", pc_bytes,
                     props.limits.maxPushConstantsSize);
        return 77;
    }
    // VKH_MESH=1: the first shader is a mesh shader (VK_EXT_mesh_shader), drawn with vkCmdDrawMeshTasksEXT(1,1,1).
    const bool mesh = !mesh_compute && (group_readback || std::getenv("VKH_MESH") != nullptr);
    VkPhysicalDeviceMeshShaderFeaturesEXT meshf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
    if (mesh) {
        // Query the feature, not just the extension. 77: the device cannot run this test (as vk_compute_test and
        // vk_store_test), so the runners report SKIP.
        VkPhysicalDeviceFeatures2 mesh_query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        mesh_query.pNext = &meshf;
        vkGetPhysicalDeviceFeatures2(pd, &mesh_query);
        if (!has(VK_EXT_MESH_SHADER_EXTENSION_NAME) || !meshf.meshShader) {
            std::fprintf(stderr, "unsupported: no VK_EXT_mesh_shader\n");
            return 77;
        }
        enable.push_back(VK_EXT_MESH_SHADER_EXTENSION_NAME);
        meshf = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
        meshf.meshShader = VK_TRUE; meshf.pNext = chain; chain = &meshf;
    }
#ifdef VK_KHR_MAINTENANCE_8_EXTENSION_NAME
    VkPhysicalDeviceMaintenance8FeaturesKHR m8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_8_FEATURES_KHR};
    if (has(VK_KHR_MAINTENANCE_8_EXTENSION_NAME)) {
        enable.push_back(VK_KHR_MAINTENANCE_8_EXTENSION_NAME);
        m8.maintenance8 = VK_TRUE; m8.pNext = chain; chain = &m8;
    }
#endif
    VkPhysicalDeviceVulkan11Features f11q{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    VkPhysicalDeviceVulkan12Features f12q{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    f12q.pNext = &f11q;
    VkPhysicalDeviceVulkan13Features f13q{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    f13q.pNext = &f12q;
    VkPhysicalDeviceFeatures2 f2q{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2q.pNext = &f13q;
    vkGetPhysicalDeviceFeatures2(pd, &f2q);
    // Translated vertex shaders read BaseVertex / BaseInstance (DrawParameters) to give v0 / v3 the hardware's
    // zero-based ids.
    VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    if (f11q.shaderDrawParameters) {
        f11.shaderDrawParameters = VK_TRUE;
        f11.pNext = chain;
        chain = &f11;
    }
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    if (f13q.subgroupSizeControl) {
        f13.subgroupSizeControl = VK_TRUE;
        f13.pNext = chain;
        chain = &f13;
    }
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    if (need_heap || mesh_compute) {
        // 77 is the "this device cannot run the test" code that check.py and run_gpu_tests.py report as SKIP.
        if (!f12q.bufferDeviceAddress) {
            std::fprintf(stderr, "unsupported: %s needs bufferDeviceAddress\n",
                         tex_rgba_path ? "VKH_TEX_RGBA" : heap_env ? "VKH_HEAP" : "VKH_MESH_COMPUTE");
            return 77;
        }
        f12.bufferDeviceAddress = VK_TRUE;
        if (tex_rgba_path) {
            if (!f12q.runtimeDescriptorArray || !f12q.descriptorIndexing) {
                std::fprintf(stderr, "unsupported: VKH_TEX_RGBA needs runtimeDescriptorArray and descriptorIndexing\n");
                return 77;
            }
            f12.runtimeDescriptorArray = VK_TRUE;
            f12.descriptorIndexing = VK_TRUE;
        }
        f12.pNext = chain;
        chain = &f12;
    }
    // VKH_BINARY_OUT=<file> / VKH_BINARY_IN=<file>: VK_KHR_pipeline_binary. OUT writes the pipeline's binaries after
    // it is built; IN builds the pipeline from such a file instead of compiling the shaders (tools/ps_oracle.py
    // patches the pixel shader's code in it). File: u32 count, then per binary u32 key size, 32 key bytes, u64 data
    // size, the data.
    const char* bin_out = std::getenv("VKH_BINARY_OUT");
    const char* bin_in = std::getenv("VKH_BINARY_IN");
    VkPhysicalDevicePipelineBinaryFeaturesKHR pbf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_BINARY_FEATURES_KHR};
    VkPhysicalDeviceMaintenance5FeaturesKHR m5{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES_KHR};
    if (bin_out || bin_in) {
        VkPhysicalDeviceFeatures2 q{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        VkPhysicalDevicePipelineBinaryFeaturesKHR pbq{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_BINARY_FEATURES_KHR};
        q.pNext = &pbq;
        vkGetPhysicalDeviceFeatures2(pd, &q);
        if (!has(VK_KHR_PIPELINE_BINARY_EXTENSION_NAME) || !has(VK_KHR_MAINTENANCE_5_EXTENSION_NAME) ||
            !pbq.pipelineBinaries) {
            std::fprintf(stderr, "unsupported: no VK_KHR_pipeline_binary\n");
            return 77;
        }
        enable.push_back(VK_KHR_PIPELINE_BINARY_EXTENSION_NAME);
        enable.push_back(VK_KHR_MAINTENANCE_5_EXTENSION_NAME);
        pbf.pipelineBinaries = VK_TRUE; pbf.pNext = chain; chain = &pbf;
        m5.maintenance5 = VK_TRUE; m5.pNext = chain; chain = &m5;
    }
    for (auto e : enable) std::fprintf(stderr, "enabled %s\n", e);
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = uint32_t(enable.size()); dci.ppEnabledExtensionNames = enable.data();
    dci.pNext = chain;
    // VKH_GS=<geometry.spv>: insert a geometry stage between the vertex and fragment shaders. VKH_SUBGROUP_SIZE then
    // applies to it, unless VKH_GS_HELPER=1 says the GS only feeds the pixel shader under test.
    const char* gs_path = std::getenv("VKH_GS");
    const char* count_opt = std::getenv("VKH_GS_PRIMITIVES");
    const bool count_gs = count_opt && std::strcmp(count_opt, "1") == 0;
    if (count_gs && (!gs_path || !*gs_path || mesh || mesh_compute)) {
        std::fprintf(stderr, "VKH_GS_PRIMITIVES=1 requires a VS/GS/FS draw with VKH_GS\n");
        return 1;
    }
    if (gs_path && !f2q.features.geometryShader) {   // MoltenVK: Metal has no geometry stage
        std::fprintf(stderr, "unsupported: no geometryShader\n");
        return 77;
    }
    VkPhysicalDeviceFeatures feats{};
    if (count_gs && !f2q.features.pipelineStatisticsQuery) {
        std::fprintf(stderr, "unsupported: no pipelineStatisticsQuery\n");
        return 77;
    }
    feats.pipelineStatisticsQuery = count_gs;
    if (clip_shifts && !f2q.features.shaderClipDistance) {
        std::fprintf(stderr, "unsupported: shaderClipDistance\n");
        return 77;
    }
    feats.geometryShader = gs_path ? VK_TRUE : VK_FALSE;
    feats.shaderClipDistance = f2q.features.shaderClipDistance;
    feats.shaderCullDistance = f2q.features.shaderCullDistance;
    if (needs_float64) {
        if (!f2q.features.shaderFloat64) {
            std::fprintf(stderr, "unsupported: %s uses Float64 and the device has no shaderFloat64\n", argv[2]);
            return 77;
        }
        feats.shaderFloat64 = VK_TRUE;
    }
    // shaderInt64 only when a stage has 64-bit integers. Heap and --mesh-compute addresses are 64-bit, but rdna2spirv.py
    // builds them from 32-bit pairs unless the shader also does 64-bit arithmetic (SHADER-423), so the module decides,
    // not VKH_HEAP (TEST-468).
    bool needs_int64 = leonida::vk_test::needs_shader_int64(fs_code);
    for (const char* p : std::initializer_list<const char*>{argv[1], mesh_vs_path, gs_path})
        if (p && !needs_int64) needs_int64 = leonida::vk_test::needs_shader_int64(read_spv(p));
    if (needs_int64) {
        if (!f2q.features.shaderInt64) {
            std::fprintf(stderr, "unsupported: a stage declares Int64 and the device has no shaderInt64\n");
            return 77;
        }
        feats.shaderInt64 = VK_TRUE;
    }
    if (tex_rgba_path) {
        if (!f2q.features.shaderSampledImageArrayDynamicIndexing) {
            std::fprintf(stderr, "unsupported: VKH_TEX_RGBA needs shaderSampledImageArrayDynamicIndexing\n");
            return 77;
        }
        feats.shaderSampledImageArrayDynamicIndexing = VK_TRUE;
    }
    dci.pEnabledFeatures = &feats;
    VkDevice dev;
    CHECK(vkCreateDevice(pd, &dci, nullptr, &dev));
#define LOADD(name) auto name = reinterpret_cast<PFN_##name>(vkGetDeviceProcAddr(dev, #name))
    LOADD(vkGetDeviceQueue); LOADD(vkCreateImage); LOADD(vkGetImageMemoryRequirements); LOADD(vkAllocateMemory);
    LOADD(vkBindImageMemory); LOADD(vkCreateImageView); LOADD(vkCreateRenderPass); LOADD(vkCreateFramebuffer);
    LOADD(vkCreateShaderModule); LOADD(vkCreatePipelineLayout); LOADD(vkCreateGraphicsPipelines);
    LOADD(vkCreateCommandPool); LOADD(vkAllocateCommandBuffers); LOADD(vkBeginCommandBuffer);
    LOADD(vkCmdBeginRenderPass); LOADD(vkCmdBindPipeline); LOADD(vkCmdDraw); LOADD(vkCmdEndRenderPass);
    LOADD(vkCmdBindVertexBuffers); LOADD(vkCmdBindIndexBuffer); LOADD(vkCmdDrawIndexed);
    LOADD(vkCmdPipelineBarrier);
    LOADD(vkCmdCopyImageToBuffer); LOADD(vkEndCommandBuffer); LOADD(vkQueueSubmit); LOADD(vkQueueWaitIdle);
    LOADD(vkCreateBuffer); LOADD(vkGetBufferMemoryRequirements); LOADD(vkBindBufferMemory); LOADD(vkMapMemory);
    VkQueue q; vkGetDeviceQueue(dev, qi, 0, &q);
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    auto memtype = [&](uint32_t bits, VkMemoryPropertyFlags want) {
        for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
            if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
        return 0u;
    };
    auto attachment = [&](VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect,
                          VkImage& image, VkImageView& view) -> int {
        VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ii.imageType = VK_IMAGE_TYPE_2D; ii.format = format; ii.extent = {W, H, 1}; ii.mipLevels = 1; ii.arrayLayers = 1;
        ii.samples = VK_SAMPLE_COUNT_1_BIT; ii.tiling = VK_IMAGE_TILING_OPTIMAL; ii.usage = usage;
        CHECK(vkCreateImage(dev, &ii, nullptr, &image));
        VkMemoryRequirements mr; vkGetImageMemoryRequirements(dev, image, &mr);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = mr.size; mai.memoryTypeIndex = memtype(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VkDeviceMemory mem; CHECK(vkAllocateMemory(dev, &mai, nullptr, &mem));
        CHECK(vkBindImageMemory(dev, image, mem, 0));
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = image; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = format;
        vi.subresourceRange = {aspect, 0, 1, 0, 1};
        CHECK(vkCreateImageView(dev, &vi, nullptr, &view));
        return 0;
    };
    // VKH_COLOR_FORMAT=<VkFormat number>:<bytes per texel>: the color target's format (default RGBA32F, 16 bytes);
    // out.raw then holds the target's texels as they are (tools/ps_oracle.py --format).
    VkFormat fmt = VK_FORMAT_R32G32B32A32_SFLOAT;
    uint32_t texel_bytes = 16;
    if (const char* cf = std::getenv("VKH_COLOR_FORMAT"); cf && *cf) {
        char* end = nullptr;
        fmt = VkFormat(std::strtoul(cf, &end, 0));
        texel_bytes = (end && *end == ':') ? uint32_t(std::strtoul(end + 1, nullptr, 0)) : 0;
        if (!texel_bytes || texel_bytes > 16) { std::fprintf(stderr, "VKH_COLOR_FORMAT: want <format>:<bytes>\n"); return 2; }
        auto get_fp = reinterpret_cast<PFN_vkGetPhysicalDeviceFormatProperties>(
            gipa(inst, "vkGetPhysicalDeviceFormatProperties"));
        VkFormatProperties cp; get_fp(pd, fmt, &cp);
        if (!(cp.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT)) {
            std::fprintf(stderr, "unsupported: format %u is not a color attachment\n", unsigned(fmt));
            return 77;
        }
    }
    VkImage img;
    VkImageView views[2]{};
    if (attachment(fmt, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                   VK_IMAGE_ASPECT_COLOR_BIT, img, views[0])) return 1;
    VkAttachmentDescription attachments[2]{};
    auto& ad = attachments[0];
    ad.format = fmt; ad.samples = VK_SAMPLE_COUNT_1_BIT; ad.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    ad.storeOp = VK_ATTACHMENT_STORE_OP_STORE; ad.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    ad.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE; ad.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ad.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    VkAttachmentReference ar{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sd{}; sd.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sd.colorAttachmentCount = 1; sd.pColorAttachments = &ar;
    VkAttachmentReference stencil_ref{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    if (group_readback) {
        LOADI(vkGetPhysicalDeviceFormatProperties);
        auto& ds = attachments[1];
        for (auto format : {VK_FORMAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D32_SFLOAT_S8_UINT}) {
            VkFormatProperties fp;
            vkGetPhysicalDeviceFormatProperties(pd, format, &fp);
            if (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) {
                ds.format = format;
                break;
            }
        }
        if (!ds.format) { std::fprintf(stderr, "no stencil attachment format\n"); return 1; }
        VkImage stencil;
        VkImageAspectFlags aspects = VK_IMAGE_ASPECT_STENCIL_BIT;
        if (ds.format != VK_FORMAT_S8_UINT) aspects |= VK_IMAGE_ASPECT_DEPTH_BIT;
        if (attachment(ds.format, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, aspects, stencil, views[1])) return 1;
        ds.samples = VK_SAMPLE_COUNT_1_BIT;
        ds.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; ds.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        ds.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; ds.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        ds.finalLayout = stencil_ref.layout;
        sd.pDepthStencilAttachment = &stencil_ref;
    }
    VkRenderPassCreateInfo rpi{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rpi.attachmentCount = group_readback ? 2 : 1; rpi.pAttachments = attachments; rpi.subpassCount = 1; rpi.pSubpasses = &sd;
    VkSubpassDependency copy_dep{0, VK_SUBPASS_EXTERNAL, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT};
    rpi.dependencyCount = 1; rpi.pDependencies = &copy_dep;
    VkRenderPass rp; CHECK(vkCreateRenderPass(dev, &rpi, nullptr, &rp));
    VkFramebufferCreateInfo fbi{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fbi.renderPass = rp; fbi.attachmentCount = rpi.attachmentCount; fbi.pAttachments = views; fbi.width = W; fbi.height = H; fbi.layers = 1;
    VkFramebuffer fb; CHECK(vkCreateFramebuffer(dev, &fbi, nullptr, &fb));
    bool tail_block = false;   // some stage reads the --pc-tail-ubo block (declares set 1)
    auto module = [&](const char* p, VkShaderModule* m) {
        auto code = read_spv(p);
        if (pc_tail && leonida::vk_test::spirv_declares_set(code.data(), code.size() * 4, leonida::vk_test::pc_tail_set))
            tail_block = true;
        VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smi.codeSize = code.size() * 4; smi.pCode = code.data();
        return vkCreateShaderModule(dev, &smi, nullptr, m);
    };
    VkShaderModule vs, fs, csm = VK_NULL_HANDLE;
    CHECK(module(mesh_compute ? mesh_vs_path : argv[1], &vs)); CHECK(module(argv[2], &fs));
    if (mesh_compute) CHECK(module(argv[1], &csm));
    VkPipelineShaderStageCreateInfo stages[3]{{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
                                              {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
                                              {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}};
    stages[0].stage = mesh ? VK_SHADER_STAGE_MESH_BIT_EXT : VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vs; stages[0].pName = "main";
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module = fs; stages[1].pName = "main";
    uint32_t nstages = 2;
    if (gs_path) {
        VkShaderModule gsm;
        CHECK(module(gs_path, &gsm));
        stages[2].stage = VK_SHADER_STAGE_GEOMETRY_BIT; stages[2].module = gsm; stages[2].pName = "main";
        nstages = 3;
    }
    const char* sg_env = std::getenv("VKH_SUBGROUP_SIZE");
    const uint32_t req_sg = sg_env ? uint32_t(atoi(sg_env)) : 0;
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo reqSg{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
    reqSg.requiredSubgroupSize = req_sg;
    if (req_sg > 0 && f13q.subgroupSizeControl &&
        req_sg >= sgProps.minSubgroupSize && req_sg <= sgProps.maxSubgroupSize) {
        // VKH_GS_HELPER=1: the geometry shader only feeds the pixel shader under test (the --bary-fallback
        // pass-through), so the size still goes to the fragment stage.
        const bool gs_tested = gs_path && !std::getenv("VKH_GS_HELPER");
        if (mesh && (sgProps.requiredSubgroupSizeStages & VK_SHADER_STAGE_MESH_BIT_EXT))
            stages[0].pNext = &reqSg;
        else if (gs_tested && (sgProps.requiredSubgroupSizeStages & VK_SHADER_STAGE_GEOMETRY_BIT))
            stages[2].pNext = &reqSg;
        else if (!mesh && !gs_tested && (sgProps.requiredSubgroupSizeStages & VK_SHADER_STAGE_FRAGMENT_BIT))
            stages[1].pNext = &reqSg;
        // A device with one subgroup size runs every stage at it, required or not (AMD GCN: 64, with only COMPUTE
        // in requiredSubgroupSizeStages), so the fragment stage of a plain draw already has the size asked for.
        const bool fixed = !mesh && !gs_tested && sgProps.minSubgroupSize == sgProps.maxSubgroupSize;
        if (stages[1].pNext == &reqSg || fixed) std::fprintf(stderr, "fragment subgroup size: %u\n", req_sg);
    } else if (req_sg > 0) {
        std::fprintf(stderr, "fragment subgroup size: %u not available\n", req_sg);
    }
    VkPipelineVertexInputStateCreateInfo vin{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkVertexInputBindingDescription vbind{0, vertex_stride, VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription vattr[8]{};
    if (draw_mesh_path) {
        for (uint32_t i = 0; i < vertex_attr_count; ++i)
            vattr[i] = {vertex_attrs[i].location, 0, vertex_attrs[i].format, vertex_attrs[i].offset};
        vin.vertexBindingDescriptionCount = 1;
        vin.pVertexBindingDescriptions = &vbind;
        vin.vertexAttributeDescriptionCount = vertex_attr_count;
        vin.pVertexAttributeDescriptions = vattr;
    }
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkViewport vp{0, 0, float(W), float(H), 0, 1}; VkRect2D sc{{0, 0}, {W, H}};
    VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vps.viewportCount = 1; vps.pViewports = &vp; vps.scissorCount = 1; vps.pScissors = &sc;
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.cullMode = VK_CULL_MODE_NONE; rs.lineWidth = 1;
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState cba{}; cba.colorWriteMask = 0xF;
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1; cb.pAttachments = &cba;
    // The push-constant block (pc_bytes, checked against the device limit above).
    const VkShaderStageFlags pcs = (mesh ? VK_SHADER_STAGE_MESH_BIT_EXT : VK_SHADER_STAGE_VERTEX_BIT) |
                                   VK_SHADER_STAGE_FRAGMENT_BIT | (gs_path ? VK_SHADER_STAGE_GEOMETRY_BIT : 0) |
                                   (mesh_compute ? VK_SHADER_STAGE_COMPUTE_BIT : 0);
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkDescriptorSet ds = VK_NULL_HANDLE;
    VkImage tex_fallback_img = VK_NULL_HANDLE, tex_img = VK_NULL_HANDLE;
    VkBuffer tex_staging = VK_NULL_HANDLE;
    if (tex_rgba_path) {
        LOADD(vkCreateDescriptorSetLayout);
        VkDescriptorSetLayoutBinding dslb[3]{};
        dslb[0] = {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 2, VK_SHADER_STAGE_ALL, nullptr};
        dslb[1] = {1, VK_DESCRIPTOR_TYPE_SAMPLER, 2, VK_SHADER_STAGE_ALL, nullptr};
        dslb[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL, nullptr};
        VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dslci.bindingCount = 3;
        dslci.pBindings = dslb;
        CHECK(vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsl));
    }
    VkPushConstantRange pcr{pcs, 0, pc_bytes};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    if (dsl) {
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &dsl;
    }
    pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
    // --pc-tail-ubo: the tail block at set 1 (set 0 stays empty here), one slot for the one draw.
    leonida::vk_test::PcTail tail;
    if (tail_block && !tail.create(vkGetDeviceProcAddr, dev, mp, props.limits.minUniformBufferOffsetAlignment, pcs, 1)) {
        std::fprintf(stderr, "FAILED creating the --pc-tail-ubo block\n");
        return 1;
    }
    const VkDescriptorSetLayout tail_layouts[] = {tail.empty_layout, tail.set_layout};
    if (tail_block) { pli.setLayoutCount = 2; pli.pSetLayouts = tail_layouts; }
    VkPipelineLayout pl; CHECK(vkCreatePipelineLayout(dev, &pli, nullptr, &pl));
    VkGraphicsPipelineCreateInfo gpi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gpi.stageCount = nstages; gpi.pStages = stages;
    gpi.pVertexInputState = mesh ? nullptr : &vin; gpi.pInputAssemblyState = mesh ? nullptr : &ia;
    gpi.pViewportState = &vps; gpi.pRasterizationState = &rs; gpi.pMultisampleState = &ms; gpi.pColorBlendState = &cb;
    VkPipelineDepthStencilStateCreateInfo ds_state{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    if (group_readback) {
        // Pixel i starts at stencil i. Every triangle decrements it, but only the i-th sees zero and writes color.
        ds_state.stencilTestEnable = VK_TRUE;
        auto& face = ds_state.front;
        face.compareOp = VK_COMPARE_OP_EQUAL;
        face.compareMask = face.writeMask = 0xff;
        face.failOp = face.passOp = VK_STENCIL_OP_DECREMENT_AND_WRAP;
        ds_state.back = face;
        gpi.pDepthStencilState = &ds_state;
    }
    gpi.layout = pl; gpi.renderPass = rp;
    VkPipelineCreateFlags2CreateInfoKHR capture{VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO_KHR};
    std::vector<VkPipelineBinaryKHR> in_bins;
    VkPipelineBinaryInfoKHR bin_info{VK_STRUCTURE_TYPE_PIPELINE_BINARY_INFO_KHR};
    if (bin_out) {
        capture.flags = VK_PIPELINE_CREATE_2_CAPTURE_DATA_BIT_KHR;
        gpi.pNext = &capture;
    }
    if (bin_in) {
        auto vkCreatePipelineBinariesKHR = reinterpret_cast<PFN_vkCreatePipelineBinariesKHR>(
            vkGetDeviceProcAddr(dev, "vkCreatePipelineBinariesKHR"));
        std::ifstream f(bin_in, std::ios::binary);
        std::vector<char> all((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        size_t at = 0;
        auto take = [&](void* p, size_t n) {
            if (at + n > all.size()) { std::fprintf(stderr, "%s: truncated\n", bin_in); std::exit(2); }
            std::memcpy(p, all.data() + at, n); at += n;
        };
        uint32_t count = 0; take(&count, 4);
        std::vector<VkPipelineBinaryKeyKHR> keys(count, {VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR});
        std::vector<VkPipelineBinaryDataKHR> datas(count);
        std::vector<std::vector<char>> blobs(count);
        for (uint32_t i = 0; i < count; i++) {
            take(&keys[i].keySize, 4); take(keys[i].key, VK_MAX_PIPELINE_BINARY_KEY_SIZE_KHR);
            uint64_t n = 0; take(&n, 8);
            blobs[i].resize(size_t(n)); take(blobs[i].data(), size_t(n));
            datas[i].dataSize = size_t(n); datas[i].pData = blobs[i].data();
        }
        VkPipelineBinaryKeysAndDataKHR kd{count, keys.data(), datas.data()};
        VkPipelineBinaryCreateInfoKHR bci{VK_STRUCTURE_TYPE_PIPELINE_BINARY_CREATE_INFO_KHR};
        bci.pKeysAndDataInfo = &kd;
        in_bins.resize(count);
        VkPipelineBinaryHandlesInfoKHR hi{VK_STRUCTURE_TYPE_PIPELINE_BINARY_HANDLES_INFO_KHR};
        hi.pipelineBinaryCount = count; hi.pPipelineBinaries = in_bins.data();
        CHECK(vkCreatePipelineBinariesKHR(dev, &bci, nullptr, &hi));
        bin_info.binaryCount = count; bin_info.pPipelineBinaries = in_bins.data();
        gpi.pNext = &bin_info;
        std::fprintf(stderr, "pipeline from %u binaries in %s\n", count, bin_in);
    }
    VkPipeline pipe; CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpi, nullptr, &pipe));
    if (bin_out) {
        auto vkCreatePipelineBinariesKHR = reinterpret_cast<PFN_vkCreatePipelineBinariesKHR>(
            vkGetDeviceProcAddr(dev, "vkCreatePipelineBinariesKHR"));
        auto vkGetPipelineBinaryDataKHR = reinterpret_cast<PFN_vkGetPipelineBinaryDataKHR>(
            vkGetDeviceProcAddr(dev, "vkGetPipelineBinaryDataKHR"));
        VkPipelineBinaryCreateInfoKHR bci{VK_STRUCTURE_TYPE_PIPELINE_BINARY_CREATE_INFO_KHR};
        bci.pipeline = pipe;
        VkPipelineBinaryHandlesInfoKHR hi{VK_STRUCTURE_TYPE_PIPELINE_BINARY_HANDLES_INFO_KHR};
        CHECK(vkCreatePipelineBinariesKHR(dev, &bci, nullptr, &hi));
        std::vector<VkPipelineBinaryKHR> bins(hi.pipelineBinaryCount);
        hi.pPipelineBinaries = bins.data();
        CHECK(vkCreatePipelineBinariesKHR(dev, &bci, nullptr, &hi));
        std::ofstream f(bin_out, std::ios::binary);
        uint32_t count = uint32_t(bins.size());
        f.write(reinterpret_cast<const char*>(&count), 4);
        for (auto b : bins) {
            VkPipelineBinaryDataInfoKHR di{VK_STRUCTURE_TYPE_PIPELINE_BINARY_DATA_INFO_KHR};
            di.pipelineBinary = b;
            VkPipelineBinaryKeyKHR key{VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR};
            size_t n = 0;
            CHECK(vkGetPipelineBinaryDataKHR(dev, &di, &key, &n, nullptr));
            std::vector<char> data(n);
            CHECK(vkGetPipelineBinaryDataKHR(dev, &di, &key, &n, data.data()));
            uint64_t n64 = n;
            f.write(reinterpret_cast<const char*>(&key.keySize), 4);
            f.write(reinterpret_cast<const char*>(key.key), VK_MAX_PIPELINE_BINARY_KEY_SIZE_KHR);
            f.write(reinterpret_cast<const char*>(&n64), 8);
            f.write(data.data(), std::streamsize(n));
        }
        f.close();
        if (!f) {
            std::fprintf(stderr, "failed to write pipeline binaries to %s\n", bin_out);
            return 1;
        }
        std::fprintf(stderr, "wrote %u pipeline binaries to %s\n", count, bin_out);
    }
    VkPipeline cpipe = VK_NULL_HANDLE;
    if (mesh_compute) {
        VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cpi.stage.module = csm; cpi.stage.pName = "main";
        // VKH_SUBGROUP_SIZE goes to the mesh shader under test, now a compute stage.
        if (req_sg > 0 && f13q.subgroupSizeControl && req_sg >= sgProps.minSubgroupSize &&
            req_sg <= sgProps.maxSubgroupSize && (sgProps.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT))
            cpi.stage.pNext = &reqSg;
        cpi.layout = pl;
        LOADD(vkCreateComputePipelines);
        CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &cpipe));
    }
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = VkDeviceSize(W) * H * texel_bytes; bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkBuffer buf; CHECK(vkCreateBuffer(dev, &bi, nullptr, &buf));
    VkMemoryRequirements br; vkGetBufferMemoryRequirements(dev, buf, &br);
    VkMemoryAllocateInfo bai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    bai.allocationSize = br.size;
    bai.memoryTypeIndex = memtype(br.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkDeviceMemory bmem; CHECK(vkAllocateMemory(dev, &bai, nullptr, &bmem));
    CHECK(vkBindBufferMemory(dev, buf, bmem, 0));
    std::vector<uint8_t> mesh_vertices;
    std::vector<uint32_t> mesh_indices;
    VkBuffer vbo = VK_NULL_HANDLE, ibo = VK_NULL_HANDLE;
    VkDeviceMemory vbo_mem = VK_NULL_HANDLE, ibo_mem = VK_NULL_HANDLE;
    if (draw_mesh_path) {
        if (!read_mesh(draw_mesh_path, vertex_stride, mesh_vertices, mesh_indices)) {
            std::fprintf(stderr, "VKH_DRAW_MESH: cannot read %s\n", draw_mesh_path);
            return 1;
        }
        auto host_buffer = [&](VkDeviceSize bytes, VkBufferUsageFlags usage, VkBuffer& b,
                               VkDeviceMemory& m, void*& p) -> bool {
            VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            ci.size = bytes; ci.usage = usage;
            if (vkCreateBuffer(dev, &ci, nullptr, &b) != VK_SUCCESS) return false;
            VkMemoryRequirements rq; vkGetBufferMemoryRequirements(dev, b, &rq);
            VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            ai.allocationSize = rq.size;
            ai.memoryTypeIndex = memtype(rq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            return vkAllocateMemory(dev, &ai, nullptr, &m) == VK_SUCCESS &&
                   vkBindBufferMemory(dev, b, m, 0) == VK_SUCCESS &&
                   vkMapMemory(dev, m, 0, bytes, 0, &p) == VK_SUCCESS;
        };
        void* vmap = nullptr;
        void* imap = nullptr;
        if (!host_buffer(VkDeviceSize(mesh_vertices.size()), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vbo, vbo_mem, vmap) ||
            !host_buffer(VkDeviceSize(mesh_indices.size() * 4), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, ibo, ibo_mem, imap)) {
            std::fprintf(stderr, "VKH_DRAW_MESH: cannot create the vertex or index buffer\n");
            return 1;
        }
        std::memcpy(vmap, mesh_vertices.data(), mesh_vertices.size());
        std::memcpy(imap, mesh_indices.data(), mesh_indices.size() * 4);
    }
    void* heap_map = nullptr;
    uint32_t tsharp[8] = {0x11111111, leonida::gfx10::format_from_vulkan(VK_FORMAT_R8G8B8A8_UNORM) << 20, 0x33333333,
                          0x44444444, 0x55555555, 0x66666666, 0x77777777, 0x88888888};
    if (need_heap) {
        LOADD(vkGetBufferDeviceAddress);
        VkBufferCreateInfo hbi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        hbi.size = heap_bytes;
        hbi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        VkBuffer heap; CHECK(vkCreateBuffer(dev, &hbi, nullptr, &heap));
        VkMemoryRequirements hr; vkGetBufferMemoryRequirements(dev, heap, &hr);
        VkMemoryAllocateFlagsInfo haf{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
        haf.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        VkMemoryAllocateInfo hai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        hai.pNext = &haf; hai.allocationSize = hr.size;
        hai.memoryTypeIndex = memtype(hr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VkDeviceMemory hmem; CHECK(vkAllocateMemory(dev, &hai, nullptr, &hmem));
        CHECK(vkBindBufferMemory(dev, heap, hmem, 0));
        void* hp; CHECK(vkMapMemory(dev, hmem, 0, heap_bytes, 0, &hp));
        heap_map = hp;
        std::memset(hp, 0, heap_bytes);
        if (tex_rgba_path) std::memcpy(static_cast<char*>(hp) + tsharp_va, tsharp, 32);
        for (auto [addr, word] : heap_words) std::memcpy(static_cast<char*>(hp) + addr, &word, 4);
        VkBufferDeviceAddressInfo hda{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        hda.buffer = heap;
        const uint64_t base = vkGetBufferDeviceAddress(dev, &hda);
        pc[32] = uint32_t(base); pc[33] = uint32_t(base >> 32);
    }
    if (tex_rgba_path) {
        LOADD(vkCreateSampler); LOADD(vkCreateDescriptorPool); LOADD(vkAllocateDescriptorSets);
        LOADD(vkUpdateDescriptorSets);
        leonida::DescriptorTable desc_table(16);
        desc_table.insert(tsharp, 1);
        desc_table.insert(ssharp, 1);
        const auto table = desc_table.build_gpu_buffer(true);
        VkBuffer tb = VK_NULL_HANDLE;
        VkDeviceMemory tbm = VK_NULL_HANDLE;
        void* tbp = nullptr;
        {
            VkBufferCreateInfo tbi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            tbi.size = table.size() * 4; tbi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            CHECK(vkCreateBuffer(dev, &tbi, nullptr, &tb));
            VkMemoryRequirements tmr; vkGetBufferMemoryRequirements(dev, tb, &tmr);
            VkMemoryAllocateInfo tai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            tai.allocationSize = tmr.size;
            tai.memoryTypeIndex = memtype(tmr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            CHECK(vkAllocateMemory(dev, &tai, nullptr, &tbm));
            CHECK(vkBindBufferMemory(dev, tb, tbm, 0));
            CHECK(vkMapMemory(dev, tbm, 0, tbi.size, 0, &tbp));
            std::memcpy(tbp, table.data(), table.size() * 4);
        }
        auto make_sampled_tex = [&](uint32_t tw, uint32_t th, VkImage& out_img, VkImageView& out_view) -> int {
            VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            ii.imageType = VK_IMAGE_TYPE_2D; ii.format = VK_FORMAT_R8G8B8A8_UNORM; ii.extent = {tw, th, 1};
            ii.mipLevels = 1; ii.arrayLayers = 1; ii.samples = VK_SAMPLE_COUNT_1_BIT;
            ii.tiling = VK_IMAGE_TILING_OPTIMAL;
            ii.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            CHECK(vkCreateImage(dev, &ii, nullptr, &out_img));
            VkMemoryRequirements mr; vkGetImageMemoryRequirements(dev, out_img, &mr);
            VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            mai.allocationSize = mr.size;
            mai.memoryTypeIndex = memtype(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            VkDeviceMemory mem; CHECK(vkAllocateMemory(dev, &mai, nullptr, &mem));
            CHECK(vkBindImageMemory(dev, out_img, mem, 0));
            VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vi.image = out_img; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = VK_FORMAT_R8G8B8A8_UNORM;
            vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            CHECK(vkCreateImageView(dev, &vi, nullptr, &out_view));
            return 0;
        };
        VkImageView tex_views[2]{};
        if (make_sampled_tex(1, 1, tex_fallback_img, tex_views[0]) ||
            make_sampled_tex(tex_w, tex_h, tex_img, tex_views[1])) return 1;
        {
            const VkDeviceSize stage_bytes = 4 + tex_pixels.size();
            VkBufferCreateInfo sbi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            sbi.size = stage_bytes; sbi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            CHECK(vkCreateBuffer(dev, &sbi, nullptr, &tex_staging));
            VkMemoryRequirements smr; vkGetBufferMemoryRequirements(dev, tex_staging, &smr);
            VkMemoryAllocateInfo sai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            sai.allocationSize = smr.size;
            sai.memoryTypeIndex = memtype(smr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            VkDeviceMemory smem; CHECK(vkAllocateMemory(dev, &sai, nullptr, &smem));
            CHECK(vkBindBufferMemory(dev, tex_staging, smem, 0));
            void* sp = nullptr; CHECK(vkMapMemory(dev, smem, 0, stage_bytes, 0, &sp));
            const uint32_t magenta = 0xFFFF00FFu;
            std::memcpy(sp, &magenta, 4);
            std::memcpy(static_cast<char*>(sp) + 4, tex_pixels.data(), tex_pixels.size());
        }
        VkSampler samplers[2]{};
        VkSamplerCreateInfo def_sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        def_sci.magFilter = VK_FILTER_NEAREST; def_sci.minFilter = VK_FILTER_NEAREST;
        CHECK(vkCreateSampler(dev, &def_sci, nullptr, &samplers[0]));
        const leonida::gfx10::SamplerWords sw{ssharp[0], ssharp[1], ssharp[2], ssharp[3]};
        VkSamplerCreateInfo sci = leonida::gfx10::vulkan_sampler(leonida::gfx10::decode(sw));
        sci.anisotropyEnable = VK_FALSE;
        CHECK(vkCreateSampler(dev, &sci, nullptr, &samplers[1]));
        VkDescriptorPoolSize dps[3] = {
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 2},
            {VK_DESCRIPTOR_TYPE_SAMPLER, 2},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
        };
        VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpci.maxSets = 1; dpci.poolSizeCount = 3; dpci.pPoolSizes = dps;
        VkDescriptorPool dpool; CHECK(vkCreateDescriptorPool(dev, &dpci, nullptr, &dpool));
        VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dsai.descriptorPool = dpool; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &dsl;
        CHECK(vkAllocateDescriptorSets(dev, &dsai, &ds));
        VkDescriptorImageInfo img_infos[2] = {
            {VK_NULL_HANDLE, tex_views[0], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
            {VK_NULL_HANDLE, tex_views[1], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        };
        VkDescriptorImageInfo smp_infos[2] = {
            {samplers[0], VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED},
            {samplers[1], VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED},
        };
        VkDescriptorBufferInfo tbl_info{tb, 0, VK_WHOLE_SIZE};
        VkWriteDescriptorSet writes[3]{};
        writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, 0, 0, 2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, img_infos};
        writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, 1, 0, 2, VK_DESCRIPTOR_TYPE_SAMPLER, smp_infos};
        writes[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &tbl_info};
        vkUpdateDescriptorSets(dev, 3, writes, 0, nullptr);
    }
    const uint64_t mesh_groups = uint64_t(groups[0]) * groups[1] * groups[2];
    if (mesh_compute) {
        LOADD(vkGetBufferDeviceAddress);
        VkBufferCreateInfo obi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        obi.size = mesh_groups * mesh_slot;
        obi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        VkBuffer obuf; CHECK(vkCreateBuffer(dev, &obi, nullptr, &obuf));
        VkMemoryRequirements orq; vkGetBufferMemoryRequirements(dev, obuf, &orq);
        VkMemoryAllocateFlagsInfo oaf{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
        oaf.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        VkMemoryAllocateInfo oai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        oai.pNext = &oaf; oai.allocationSize = orq.size;
        oai.memoryTypeIndex = memtype(orq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VkDeviceMemory omem; CHECK(vkAllocateMemory(dev, &oai, nullptr, &omem));
        CHECK(vkBindBufferMemory(dev, obuf, omem, 0));
        void* op; CHECK(vkMapMemory(dev, omem, 0, obi.size, 0, &op));
        std::memset(op, 0, size_t(obi.size));   // a workgroup that never writes its header draws nothing
        VkBufferDeviceAddressInfo oda{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        oda.buffer = obuf;
        const uint64_t oaddr = vkGetBufferDeviceAddress(dev, &oda);
        pc[36] = uint32_t(oaddr); pc[37] = uint32_t(oaddr >> 32);
        std::fprintf(stderr, "mesh compute: %llu workgroups x %llu bytes, %u primitives each\n",
                     (unsigned long long)mesh_groups, (unsigned long long)mesh_slot, mesh_prims);
    }
    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; cpi.queueFamilyIndex = qi;
    VkCommandPool pool; CHECK(vkCreateCommandPool(dev, &cpi, nullptr, &pool));
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = pool; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
    VkCommandBuffer cmd; CHECK(vkAllocateCommandBuffers(dev, &cai, &cmd));
    VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    CHECK(vkBeginCommandBuffer(cmd, &cbi));
    VkQueryPool gs_query = VK_NULL_HANDLE;
    if (count_gs) {
        LOADD(vkCreateQueryPool);
        LOADD(vkCmdResetQueryPool);
        VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        info.queryType = VK_QUERY_TYPE_PIPELINE_STATISTICS;
        info.queryCount = 1;
        info.pipelineStatistics = VK_QUERY_PIPELINE_STATISTIC_GEOMETRY_SHADER_PRIMITIVES_BIT;
        CHECK(vkCreateQueryPool(dev, &info, nullptr, &gs_query));
        vkCmdResetQueryPool(cmd, gs_query, 0, 1);
    }
    VkClearValue clears[2]{};
    for (float& c : clears[0].color.float32) c = -1;   // marks untouched pixels
    if (clip_shifts) {
        const float shifts[] = {float(std::atof(argv[6])), float(std::atof(argv[7]))};
        std::memcpy(pc, shifts, sizeof(shifts));
    }
    auto push = reinterpret_cast<PFN_vkCmdPushConstants>(vkGetDeviceProcAddr(dev, "vkCmdPushConstants"));
    if (tex_rgba_path) {
        LOADD(vkCmdPipelineBarrier); LOADD(vkCmdCopyBufferToImage);
        VkImageMemoryBarrier to_dst[2]{};
        for (int i = 0; i < 2; ++i) {
            to_dst[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            to_dst[i].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            to_dst[i].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            to_dst[i].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            to_dst[i].image = i == 0 ? tex_fallback_img : tex_img;
            to_dst[i].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        }
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                             0, nullptr, 0, nullptr, 2, to_dst);
        VkBufferImageCopy c0{};
        c0.bufferOffset = 0; c0.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; c0.imageExtent = {1, 1, 1};
        vkCmdCopyBufferToImage(cmd, tex_staging, tex_fallback_img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c0);
        VkBufferImageCopy c1{};
        c1.bufferOffset = 4; c1.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; c1.imageExtent = {tex_w, tex_h, 1};
        vkCmdCopyBufferToImage(cmd, tex_staging, tex_img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c1);
        VkImageMemoryBarrier to_ro[2]{};
        for (int i = 0; i < 2; ++i) {
            to_ro[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            to_ro[i].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            to_ro[i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            to_ro[i].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            to_ro[i].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            to_ro[i].image = i == 0 ? tex_fallback_img : tex_img;
            to_ro[i].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        }
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                             0, nullptr, 0, nullptr, 2, to_ro);
    }
    if (mesh_compute) {
        // The mesh shader as compute, then make its stores visible to the vertex shader's reads.
        LOADD(vkCmdDispatch);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cpipe);
        push(cmd, pl, pcs, 0, pc_bytes, pc);
        vkCmdDispatch(cmd, groups[0], groups[1], groups[2]);
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, 0,
                             1, &mb, 0, nullptr, 0, nullptr);
    }
    VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rbi.renderPass = rp; rbi.framebuffer = fb; rbi.renderArea = sc; rbi.clearValueCount = rpi.attachmentCount; rbi.pClearValues = clears;
    vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
    if (group_readback) {
        LOADD(vkCmdClearAttachments);
        VkClearAttachment seed{VK_IMAGE_ASPECT_STENCIL_BIT};
        for (uint32_t i = 1; i < W; ++i) {
            seed.clearValue.depthStencil.stencil = i;
            VkClearRect pixel{{{int32_t(i), 0}, {1, 1}}, 0, 1};
            vkCmdClearAttachments(cmd, 1, &seed, 1, &pixel);
        }
        std::fprintf(stderr, "mesh workgroups: %u x %u x %u\n", groups[0], groups[1], groups[2]);
    }
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    if (ds) {
        LOADD(vkCmdBindDescriptorSets);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pl, 0, 1, &ds, 0, nullptr);
    }
    push(cmd, pl, pcs, 0, pc_bytes, pc);
    if (tail_block) {
        LOADD(vkCmdBindDescriptorSets);
        tail.write(0, pc, 4);           // heap base (VKH_HEAP) and code base, as pc[32..35] are without the flag
        const uint32_t offset = tail.offset(0);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pl, leonida::vk_test::pc_tail_set, 1, &tail.set,
                                1, &offset);
    }
    // VKH_VERTICES=<n>: draw n vertices instead of one triangle, so a vertex shader fills whole waves (TEST-251).
    const char* vertices_env = std::getenv("VKH_VERTICES");
    if (count_gs) {
        LOADD(vkCmdBeginQuery);
        vkCmdBeginQuery(cmd, gs_query, 0, 0);
    }
    if (draw_mesh_path) {
        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &vbo, &offset);
        vkCmdBindIndexBuffer(cmd, ibo, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, uint32_t(mesh_indices.size()), 1, 0, 0, 0);
    } else if (const unsigned vertices = vertices_env ? unsigned(std::strtoul(vertices_env, nullptr, 10)) : 0;
               vertices && !mesh)
        vkCmdDraw(cmd, vertices, 1, 0, 0);
    else if (mesh_compute)
        vkCmdDraw(cmd, uint32_t(mesh_groups * mesh_prims * 3), 1, 0, 0);
    else if (mesh)
        reinterpret_cast<PFN_vkCmdDrawMeshTasksEXT>(vkGetDeviceProcAddr(dev, "vkCmdDrawMeshTasksEXT"))(cmd, groups[0], groups[1], groups[2]);
    else {
        // VKH_DRAW=<firstVertex>,<firstInstance>: draw the triangle with non-zero bases (vs_base_ids).
        unsigned first_vertex = 0, first_instance = 0;
        if (const char* d = std::getenv("VKH_DRAW")) std::sscanf(d, "%u,%u", &first_vertex, &first_instance);
        vkCmdDraw(cmd, 3, 1, first_vertex, first_instance);
    }
    if (count_gs) {
        LOADD(vkCmdEndQuery);
        vkCmdEndQuery(cmd, gs_query, 0);
    }
    vkCmdEndRenderPass(cmd);
    VkBufferImageCopy reg{}; reg.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; reg.imageExtent = {W, H, 1};
    vkCmdCopyImageToBuffer(cmd, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buf, 1, &reg);
    // Make the image copy and any shader writes to the heap visible to the host after the queue wait.
    VkMemoryBarrier readback{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    readback.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    readback.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0,
                         1, &readback, 0, nullptr, 0, nullptr);
    CHECK(vkEndCommandBuffer(cmd));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    CHECK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE));
    CHECK(vkQueueWaitIdle(q));
    if (count_gs) {
        LOADD(vkGetQueryPoolResults);
        LOADD(vkDestroyQueryPool);
        uint64_t primitives = 0;
        CHECK(vkGetQueryPoolResults(dev, gs_query, 0, 1, sizeof(primitives), &primitives,
                                    sizeof(primitives), VK_QUERY_RESULT_64_BIT));
        vkDestroyQueryPool(dev, gs_query, nullptr);
        std::fprintf(stderr, "gs_primitives=%llu\n", static_cast<unsigned long long>(primitives));
    }
    void* p; CHECK(vkMapMemory(dev, bmem, 0, bi.size, 0, &p));
    std::ofstream output(argv[3], std::ios::binary);
    output.write(static_cast<const char*>(p), std::streamsize(bi.size));
    output.close();
    if (!output) { std::fprintf(stderr, "failed to write %s\n", argv[3]); return 1; }
    if (const char* dump = std::getenv("VKH_HEAP_DUMP"); dump && *dump && heap_map) {
        std::ofstream hf(dump, std::ios::binary);
        hf.write(static_cast<const char*>(heap_map), heap_bytes);
        hf.close();
        if (!hf) { std::fprintf(stderr, "failed to write %s\n", dump); return 1; }
    }
    std::fprintf(stderr, "ok %ux%u -> %s\n", W, H, argv[3]);
    return 0;
}
