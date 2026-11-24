// Real GPU test for:
//   1) ExecuteIndirect with CountBuffer (ID3D12Device::CreateCommandSignature +
//      ID3D12GraphicsCommandList::ExecuteIndirect -> vkCmdDrawIndirectCount, Vulkan 1.2 core).
//   2) Hardware Ray Tracing / BVH (ID3D12Device5::GetRaytracingAccelerationStructurePrebuildInfo +
//      ID3D12GraphicsCommandList4::BuildRaytracingAccelerationStructure BLAS + TLAS + RayQueryKHR
//      hit vs. miss; clean SKIP on GPUs without VK_KHR_acceleration_structure / VK_KHR_ray_query).
// Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation (Core + Synchronization Validation).

#include "../shims/d3d12_com.h"
#include "vulkan_loader.h"
#include "../src/vulkan_device_filter.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

namespace {

int fail(const char* what) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    return 1;
}

unsigned& validation_errors() {
    static unsigned count = 0;
    return count;
}

VKAPI_ATTR VkBool32 VKAPI_CALL on_validation(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                             VkDebugUtilsMessageTypeFlagsEXT types,
                                             const VkDebugUtilsMessengerCallbackDataEXT* data, void*) {
    if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) &&
        (types & VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT))
        ++validation_errors();
    std::fprintf(stderr, "validation: %s\n", data && data->pMessage ? data->pMessage : "(no message)");
    return VK_FALSE;
}

bool read_spv(const char* path, std::vector<std::uint32_t>& words) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    const auto sz = f.tellg();
    if (sz <= 0 || (sz % 4) != 0) return false;
    f.seekg(0);
    words.resize(static_cast<std::size_t>(sz) / 4);
    return static_cast<bool>(f.read(reinterpret_cast<char*>(words.data()), sz));
}

bool wait_fence(ID3D12Fence* fence, UINT64 value) {
    HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!ev) return false;
    if (FAILED(fence->SetEventOnCompletion(value, ev))) {
        CloseHandle(ev);
        return false;
    }
    const DWORD w = WaitForSingleObject(ev, 5000);
    CloseHandle(ev);
    return w == WAIT_OBJECT_0;
}

D3D12_RESOURCE_DESC make_buf_desc(UINT64 bytes, D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width = bytes;
    d.Height = 1;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    d.Flags = flags;
    return d;
}

}  // namespace

int main() {
    const PFN_vkGetInstanceProcAddr gipa = load_vulkan();
    if (!gipa) {
        std::printf("SKIP: vulkan-1.dll not available\n");
        return 77;  // ctest skips, as in the other GPU tests
    }
    const auto create_inst = reinterpret_cast<PFN_vkCreateInstance>(gipa(nullptr, "vkCreateInstance"));
    const auto enum_layers =
        reinterpret_cast<PFN_vkEnumerateInstanceLayerProperties>(gipa(nullptr, "vkEnumerateInstanceLayerProperties"));
    if (!create_inst) return fail("vkCreateInstance missing");

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
    const char* env_val = std::getenv("LEONIDA_VK_VALIDATION");
#ifdef _MSC_VER
#pragma warning(pop)
#endif
    const bool want_val = env_val && std::strcmp(env_val, "0") != 0;
    const char* val_layer = "VK_LAYER_KHRONOS_validation";
    const char* dbg_ext = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
    bool val_active = false;
    if (want_val && enum_layers) {
        std::uint32_t cnt = 0;
        enum_layers(&cnt, nullptr);
        std::vector<VkLayerProperties> layers(cnt);
        enum_layers(&cnt, layers.data());
        for (const auto& l : layers) {
            if (!std::strcmp(l.layerName, val_layer)) val_active = true;
        }
        if (!val_active) return fail("LEONIDA_VK_VALIDATION is set but VK_LAYER_KHRONOS_validation is not installed");
    }

    VkDebugUtilsMessengerCreateInfoEXT dbg{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
    dbg.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    dbg.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
    dbg.pfnUserCallback = on_validation;

    const VkValidationFeatureEnableEXT sync_val = VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT;
    VkValidationFeaturesEXT val_feats{VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT, &dbg};
    val_feats.enabledValidationFeatureCount = 1;
    val_feats.pEnabledValidationFeatures = &sync_val;

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    if (val_active) {
        ici.pNext = &val_feats;
        ici.enabledLayerCount = 1;
        ici.ppEnabledLayerNames = &val_layer;
        ici.enabledExtensionCount = 1;
        ici.ppEnabledExtensionNames = &dbg_ext;
    }

    VkInstance instance = VK_NULL_HANDLE;
    const VkResult inst_result = create_inst(&ici, nullptr, &instance);
    if (inst_result == VK_ERROR_INCOMPATIBLE_DRIVER) {
        // The loader is present but there is no ICD (CI runners, headless machines): ctest skips.
        std::printf("SKIP: no Vulkan driver (vkCreateInstance returned VK_ERROR_INCOMPATIBLE_DRIVER)\n");
        return 77;
    }
    if (inst_result != VK_SUCCESS) return fail("vkCreateInstance");
    const auto destroy_inst = reinterpret_cast<PFN_vkDestroyInstance>(gipa(instance, "vkDestroyInstance"));
    const auto gdpa = reinterpret_cast<PFN_vkGetDeviceProcAddr>(gipa(instance, "vkGetDeviceProcAddr"));
    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    PFN_vkDestroyDebugUtilsMessengerEXT destroy_dbg = nullptr;
    if (val_active) {
        const auto create_dbg =
            reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(gipa(instance, "vkCreateDebugUtilsMessengerEXT"));
        destroy_dbg =
            reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(gipa(instance, "vkDestroyDebugUtilsMessengerEXT"));
        if (create_dbg) create_dbg(instance, &dbg, nullptr, &messenger);
    }

    const auto enum_devs = reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(gipa(instance, "vkEnumeratePhysicalDevices"));
    const auto get_props = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(gipa(instance, "vkGetPhysicalDeviceProperties"));
    const auto get_feats2 =
        reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(gipa(instance, "vkGetPhysicalDeviceFeatures2"));
    const auto enum_dev_exts = reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(
        gipa(instance, "vkEnumerateDeviceExtensionProperties"));
    const auto get_qf = reinterpret_cast<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
        gipa(instance, "vkGetPhysicalDeviceQueueFamilyProperties"));
    const auto create_dev = reinterpret_cast<PFN_vkCreateDevice>(gipa(instance, "vkCreateDevice"));

    std::uint32_t pcnt = 0;
    enum_devs(instance, &pcnt, nullptr);
    std::vector<VkPhysicalDevice> physicals(pcnt);
    enum_devs(instance, &pcnt, physicals.data());

    VkPhysicalDevice physical = VK_NULL_HANDLE;
    std::uint32_t family = 0;
    VkPhysicalDeviceProperties chosen{};
    VkPhysicalDeviceVulkan12Features chosen12{};
    int best_rank = -1;
    for (auto cand : physicals) {
        VkPhysicalDeviceProperties props{};
        get_props(cand, &props);
        if (props.apiVersion < VK_API_VERSION_1_3 || !leonida::vulkan_device_wanted(props.deviceName)) continue;
        VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f13};
        VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f12};
        get_feats2(cand, &f2);
        if (!f13.dynamicRendering || !f13.synchronization2 || !f12.timelineSemaphore ||
            !f12.separateDepthStencilLayouts)
            continue;
        std::uint32_t fcnt = 0;
        get_qf(cand, &fcnt, nullptr);
        std::vector<VkQueueFamilyProperties> fams(fcnt);
        get_qf(cand, &fcnt, fams.data());
        for (std::uint32_t i = 0; i < fcnt; ++i) {
            if (!(fams[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) continue;
            // the runtime's pick: the first usable device of the highest rank (native driver over Dozen, #915)
            if (const int rank = leonida::vulkan_device_rank(static_cast<int>(props.deviceType), props.vendorID);
                rank > best_rank) {
                best_rank = rank;
                physical = cand;
                family = i;
                chosen = props;
                chosen12 = f12;
            }
            break;
        }
    }
    if (!physical) {
        const std::string filter = leonida::vulkan_device_filter();
        if (filter.empty()) std::printf("SKIP: no Vulkan 1.3 GPU found\n");
        else std::fprintf(stderr, "LEONIDA_VK_DEVICE=%s: no Vulkan 1.3 GPU matches\n", filter.c_str());
        if (messenger && destroy_dbg) destroy_dbg(instance, messenger, nullptr);
        destroy_inst(instance, nullptr);
        if (!filter.empty()) return 1;  // a filter matching nothing fails, never skips
        return 77;  // ctest skips, as in the other GPU tests
    }

    bool rt_supported = false;
    {
        std::uint32_t ec = 0;
        enum_dev_exts(physical, nullptr, &ec, nullptr);
        std::vector<VkExtensionProperties> exts(ec);
        enum_dev_exts(physical, nullptr, &ec, exts.data());
        bool has_as = false, has_dho = false, has_rq = false;
        for (const auto& e : exts) {
            if (!std::strcmp(e.extensionName, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME)) has_as = true;
            if (!std::strcmp(e.extensionName, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME)) has_dho = true;
            if (!std::strcmp(e.extensionName, VK_KHR_RAY_QUERY_EXTENSION_NAME)) has_rq = true;
        }
        if (has_as && has_dho && has_rq && chosen12.bufferDeviceAddress) {
            VkPhysicalDeviceRayQueryFeaturesKHR rq{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
            VkPhysicalDeviceAccelerationStructureFeaturesKHR as{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR, &rq};
            VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &as};
            get_feats2(physical, &f2);
            rt_supported = (as.accelerationStructure == VK_TRUE && rq.rayQuery == VK_TRUE);
        }
    }

    VkPhysicalDeviceRayQueryFeaturesKHR en_rq{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
    en_rq.rayQuery = VK_TRUE;
    VkPhysicalDeviceAccelerationStructureFeaturesKHR en_as{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR, &en_rq};
    en_as.accelerationStructure = VK_TRUE;

    VkPhysicalDeviceVulkan13Features en13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
                                          rt_supported ? &en_as : nullptr};
    en13.dynamicRendering = VK_TRUE;
    en13.synchronization2 = VK_TRUE;
    VkPhysicalDeviceVulkan12Features en12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &en13};
    en12.timelineSemaphore = VK_TRUE;
    en12.separateDepthStencilLayouts = VK_TRUE;
    en12.drawIndirectCount = chosen12.drawIndirectCount;
    en12.bufferDeviceAddress = chosen12.bufferDeviceAddress;

    const float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;

    std::vector<const char*> dev_exts;
    if (rt_supported) {
        dev_exts.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
        dev_exts.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
        dev_exts.push_back(VK_KHR_RAY_QUERY_EXTENSION_NAME);
    }
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &en12};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = static_cast<std::uint32_t>(dev_exts.size());
    dci.ppEnabledExtensionNames = dev_exts.empty() ? nullptr : dev_exts.data();

    VkDevice vk_dev = VK_NULL_HANDLE;
    if (create_dev(physical, &dci, nullptr, &vk_dev) != VK_SUCCESS) return fail("vkCreateDevice");

#define DF(name) [[maybe_unused]] const auto name = reinterpret_cast<PFN_##name>(gdpa(vk_dev, #name))
    DF(vkDestroyDevice);
    DF(vkDeviceWaitIdle);
    DF(vkGetDeviceQueue);
    DF(vkCreateCommandPool);
    DF(vkDestroyCommandPool);
    DF(vkAllocateCommandBuffers);
    DF(vkBeginCommandBuffer);
    DF(vkEndCommandBuffer);
    DF(vkQueueSubmit);
    DF(vkQueueWaitIdle);
    DF(vkCreateShaderModule);
    DF(vkDestroyShaderModule);
    DF(vkCreatePipelineLayout);
    DF(vkDestroyPipelineLayout);
    DF(vkCreateGraphicsPipelines);
    DF(vkCreateComputePipelines);
    DF(vkDestroyPipeline);
    DF(vkCreateDescriptorSetLayout);
    DF(vkDestroyDescriptorSetLayout);
    DF(vkCreateDescriptorPool);
    DF(vkDestroyDescriptorPool);
    DF(vkAllocateDescriptorSets);
    DF(vkUpdateDescriptorSets);
    DF(vkCreateImageView);
    DF(vkDestroyImageView);
    DF(vkCmdBeginRendering);
    DF(vkCmdEndRendering);
    DF(vkCmdBindPipeline);
    DF(vkCmdBindDescriptorSets);
    DF(vkCmdSetViewport);
    DF(vkCmdSetScissor);
    DF(vkCmdDispatch);
    DF(vkCmdPipelineBarrier);
    DF(vkCmdDrawIndirectCount);
    DF(vkCmdCopyImageToBuffer);
#undef DF

    VkQueue vk_queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(vk_dev, family, 0, &vk_queue);

    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = gipa;
    info.instance = instance;
    info.physical = physical;
    info.device = vk_dev;
    info.queue = vk_queue;
    info.queue_family = family;
    info.separate_depth_stencil_layouts = true;
    info.buffer_device_address = (chosen12.bufferDeviceAddress == VK_TRUE);
    info.ray_tracing = rt_supported;

    ID3D12Device5* dev = nullptr;
    if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&dev))) || !dev) return fail("create_vulkan_device");

    D3D12_COMMAND_QUEUE_DESC qd{D3D12_COMMAND_LIST_TYPE_DIRECT, 0, D3D12_COMMAND_QUEUE_FLAG_NONE, 0};
    ID3D12CommandQueue* queue = nullptr;
    ID3D12CommandAllocator* alloc = nullptr;
    ID3D12GraphicsCommandList4* cl = nullptr;
    ID3D12Fence* fence = nullptr;
    if (FAILED(dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue))) ||
        FAILED(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc))) ||
        FAILED(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, nullptr, IID_PPV_ARGS(&cl))) ||
        FAILED(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence))))
        return fail("queue/alloc/cl/fence");
    cl->Close();

    // -------------------------------------------------------------------------
    // Part 1: ExecuteIndirect with CountBuffer (vkCmdDrawIndirectCount)
    // -------------------------------------------------------------------------
    {
        D3D12_INDIRECT_ARGUMENT_DESC arg_desc{D3D12_INDIRECT_ARGUMENT_TYPE_DRAW};
        D3D12_COMMAND_SIGNATURE_DESC sig_desc{sizeof(D3D12_DRAW_ARGUMENTS), 1, &arg_desc, 0};
        ID3D12CommandSignature* cmd_sig = nullptr;
        if (FAILED(dev->CreateCommandSignature(&sig_desc, nullptr, IID_PPV_ARGS(&cmd_sig))) || !cmd_sig)
            return fail("CreateCommandSignature");

        // Verify COM command-list recording of ExecuteIndirect with CountBuffer.
        D3D12_HEAP_PROPERTIES up_hp{D3D12_HEAP_TYPE_UPLOAD};
        constexpr VkDeviceSize buffer_size = 256, argument_base = 64, count_base = 128;
        auto b256 = make_buf_desc(buffer_size);
        ID3D12Resource* arg_buf = nullptr;
        ID3D12Resource* count_buf = nullptr;
        if (FAILED(dev->CreateCommittedResource(&up_hp, D3D12_HEAP_FLAG_NONE, &b256, D3D12_RESOURCE_STATE_GENERIC_READ,
                                                nullptr, IID_PPV_ARGS(&arg_buf))) ||
            FAILED(dev->CreateCommittedResource(&up_hp, D3D12_HEAP_FLAG_NONE, &b256, D3D12_RESOURCE_STATE_GENERIC_READ,
                                                nullptr, IID_PPV_ARGS(&count_buf))))
            return fail("CreateCommittedResource(indirect bufs)");

        void* p = nullptr;
        if (FAILED(arg_buf->Map(0, nullptr, &p))) return fail("Map(indirect arguments)");
        std::memset(p, 0, buffer_size);
        auto* draws = reinterpret_cast<D3D12_DRAW_ARGUMENTS*>(static_cast<std::uint8_t*>(p) + argument_base);
        draws[0] = {3, 1, 0, 0};
        draws[1] = {3, 1, 0, 0};
        arg_buf->Unmap(0, nullptr);

        if (FAILED(count_buf->Map(0, nullptr, &p))) return fail("Map(indirect count)");
        // Counts at unadjusted offsets are deliberately reversed.
        auto* unadjusted = static_cast<std::uint32_t*>(p);
        unadjusted[0] = 1u;
        unadjusted[1] = 0u;
        auto* counts = reinterpret_cast<std::uint32_t*>(static_cast<std::uint8_t*>(p) + count_base);
        counts[0] = 0u;  // draw 0 triangles
        counts[1] = 1u;  // draw 1 fullscreen triangle
        count_buf->Unmap(0, nullptr);

        alloc->Reset();
        cl->Reset(alloc, nullptr);
        cl->ExecuteIndirect(cmd_sig, 2, arg_buf, argument_base, count_buf, count_base + sizeof(std::uint32_t));
        if (FAILED(cl->Close())) return fail("ExecuteIndirect COM record Close");

#if defined(LEONIDA_TEST_VS_TRI_SPV) && defined(LEONIDA_TEST_FS_SOLID_SPV)
        if (chosen12.drawIndirectCount) {
            std::vector<std::uint32_t> vs_words, fs_words;
            if (!read_spv(LEONIDA_TEST_VS_TRI_SPV, vs_words) || !read_spv(LEONIDA_TEST_FS_SOLID_SPV, fs_words))
                return fail("read vs_tri/fs_solid spv");

            constexpr UINT kW = 16, kH = 16;
            D3D12_HEAP_PROPERTIES def_hp{D3D12_HEAP_TYPE_DEFAULT};
            D3D12_RESOURCE_DESC rt_desc{};
            rt_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            rt_desc.Width = kW;
            rt_desc.Height = kH;
            rt_desc.DepthOrArraySize = 1;
            rt_desc.MipLevels = 1;
            rt_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            rt_desc.SampleDesc.Count = 1;
            rt_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
            ID3D12Resource* rt = nullptr;
            if (FAILED(dev->CreateCommittedResource(&def_hp, D3D12_HEAP_FLAG_NONE, &rt_desc,
                                                    D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&rt))))
                return fail("CreateCommittedResource(rt)");

            D3D12_HEAP_PROPERTIES rb_hp{D3D12_HEAP_TYPE_READBACK};
            auto rb_desc = make_buf_desc(kW * kH * 4u);
            ID3D12Resource* rb = nullptr;
            if (FAILED(dev->CreateCommittedResource(&rb_hp, D3D12_HEAP_FLAG_NONE, &rb_desc,
                                                    D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&rb))))
                return fail("CreateCommittedResource(rb)");

            VkShaderModule vs_mod = VK_NULL_HANDLE, fs_mod = VK_NULL_HANDLE;
            VkShaderModuleCreateInfo vci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0, vs_words.size() * 4,
                                         vs_words.data()};
            VkShaderModuleCreateInfo fci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0, fs_words.size() * 4,
                                         fs_words.data()};
            if (vkCreateShaderModule(vk_dev, &vci, nullptr, &vs_mod) != VK_SUCCESS ||
                vkCreateShaderModule(vk_dev, &fci, nullptr, &fs_mod) != VK_SUCCESS)
                return fail("vkCreateShaderModule(vs/fs)");

            VkPipelineLayout pl = VK_NULL_HANDLE;
            VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            if (vkCreatePipelineLayout(vk_dev, &pli, nullptr, &pl) != VK_SUCCESS) return fail("vkCreatePipelineLayout");

            VkPipelineShaderStageCreateInfo stages[2]{};
            stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
            stages[0].module = vs_mod;
            stages[0].pName = "main";
            stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
            stages[1].module = fs_mod;
            stages[1].pName = "main";

            VkPipelineVertexInputStateCreateInfo vis{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
            VkPipelineInputAssemblyStateCreateInfo ias{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
            ias.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
            vps.viewportCount = 1;
            vps.scissorCount = 1;
            VkPipelineRasterizationStateCreateInfo ras{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
            ras.polygonMode = VK_POLYGON_MODE_FILL;
            ras.cullMode = VK_CULL_MODE_NONE;
            ras.lineWidth = 1.0f;
            VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
            ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
            VkPipelineColorBlendAttachmentState cba{};
            cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
                                 VK_COLOR_COMPONENT_A_BIT;
            VkPipelineColorBlendStateCreateInfo cbs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
            cbs.attachmentCount = 1;
            cbs.pAttachments = &cba;
            const VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
            VkPipelineDynamicStateCreateInfo dys{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
            dys.dynamicStateCount = 2;
            dys.pDynamicStates = dyn;
            VkFormat color_fmt = VK_FORMAT_R8G8B8A8_UNORM;
            VkPipelineRenderingCreateInfo rci{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
            rci.colorAttachmentCount = 1;
            rci.pColorAttachmentFormats = &color_fmt;

            VkGraphicsPipelineCreateInfo gpci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &rci};
            gpci.stageCount = 2;
            gpci.pStages = stages;
            gpci.pVertexInputState = &vis;
            gpci.pInputAssemblyState = &ias;
            gpci.pViewportState = &vps;
            gpci.pRasterizationState = &ras;
            gpci.pMultisampleState = &ms;
            gpci.pColorBlendState = &cbs;
            gpci.pDynamicState = &dys;
            gpci.layout = pl;
            VkPipeline pso = VK_NULL_HANDLE;
            if (vkCreateGraphicsPipelines(vk_dev, VK_NULL_HANDLE, 1, &gpci, nullptr, &pso) != VK_SUCCESS)
                return fail("vkCreateGraphicsPipelines");
            vkDestroyShaderModule(vk_dev, vs_mod, nullptr);
            vkDestroyShaderModule(vk_dev, fs_mod, nullptr);

            VkImageView rt_view = VK_NULL_HANDLE;
            VkImageViewCreateInfo vinfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vinfo.image = rt::d3d12::vulkan_image(rt);
            vinfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vinfo.format = color_fmt;
            vinfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCreateImageView(vk_dev, &vinfo, nullptr, &rt_view);

            VkCommandPool pool = VK_NULL_HANDLE;
            VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
            pci.queueFamilyIndex = family;
            pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            vkCreateCommandPool(vk_dev, &pci, nullptr, &pool);
            VkCommandBuffer cb = VK_NULL_HANDLE;
            VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            cbai.commandPool = pool;
            cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cbai.commandBufferCount = 1;
            vkAllocateCommandBuffers(vk_dev, &cbai, &cb);

            leonida::ReplayFunctions rf{};
            rf.draw_indirect_count = vkCmdDrawIndirectCount;
            auto run_indirect_count = [&](std::uint64_t count_offset, std::uint8_t exp_r, std::uint8_t exp_g,
                                          std::uint8_t exp_b) -> int {
                leonida::CommandList ir;
                ir.execute_indirect({leonida::IndirectKind::draw, sizeof(D3D12_DRAW_ARGUMENTS), 2u,
                                     rt::d3d12::resource_id(arg_buf), 0, rt::d3d12::resource_id(count_buf),
                                     count_offset});
                ir.close();

                VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
                bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                vkBeginCommandBuffer(cb, &bi);

                VkImageMemoryBarrier b_to_rt{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                b_to_rt.srcAccessMask = 0;
                b_to_rt.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
                b_to_rt.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                b_to_rt.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                b_to_rt.image = rt::d3d12::vulkan_image(rt);
                b_to_rt.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                     VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, nullptr, 0, nullptr, 1,
                                     &b_to_rt);

                VkRenderingAttachmentInfo att{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
                att.imageView = rt_view;
                att.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                att.clearValue.color.float32[3] = 1.0f;
                VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
                ri.renderArea = {{0, 0}, {kW, kH}};
                ri.layerCount = 1;
                ri.colorAttachmentCount = 1;
                ri.pColorAttachments = &att;
                vkCmdBeginRendering(cb, &ri);
                VkViewport vp{0, 0, float(kW), float(kH), 0, 1};
                VkRect2D sc{{0, 0}, {kW, kH}};
                vkCmdSetViewport(cb, 0, 1, &vp);
                vkCmdSetScissor(cb, 0, 1, &sc);
                vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pso);

                const auto resolver = [&](leonida::ResourceId id) -> const leonida::ResourceBinding* {
                    // Treat the two upload buffers as backing stores for slices that start after a prefix.
                    static leonida::ResourceBinding arg_b{}, cnt_b{};
                    arg_b.buffer = rt::d3d12::vulkan_buffer(arg_buf);
                    arg_b.buffer_offset = argument_base;
                    arg_b.size = buffer_size - argument_base;
                    cnt_b.buffer = rt::d3d12::vulkan_buffer(count_buf);
                    cnt_b.buffer_offset = count_base;
                    cnt_b.size = buffer_size - count_base;
                    if (id == rt::d3d12::resource_id(arg_buf)) return &arg_b;
                    if (id == rt::d3d12::resource_id(count_buf)) return &cnt_b;
                    return nullptr;
                };
                if (leonida::replay(ir, cb, resolver, rf) != leonida::ReplayResult::success)
                    return fail("leonida::replay(ExecuteIndirect CountBuffer)");

                vkCmdEndRendering(cb);

                VkImageMemoryBarrier b_to_src{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                b_to_src.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
                b_to_src.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                b_to_src.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                b_to_src.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                b_to_src.image = rt::d3d12::vulkan_image(rt);
                b_to_src.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                     0, 0, nullptr, 0, nullptr, 1, &b_to_src);

                VkBufferImageCopy bic{};
                bic.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                bic.imageExtent = {kW, kH, 1};
                vkCmdCopyImageToBuffer(cb, rt::d3d12::vulkan_image(rt), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                       rt::d3d12::vulkan_buffer(rb), 1, &bic);

                VkMemoryBarrier host_b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                host_b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                host_b.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
                vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host_b, 0,
                                     nullptr, 0, nullptr);
                vkEndCommandBuffer(cb);

                VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
                si.commandBufferCount = 1;
                si.pCommandBuffers = &cb;
                vkQueueSubmit(vk_queue, 1, &si, VK_NULL_HANDLE);
                vkQueueWaitIdle(vk_queue);

                void* mp = nullptr;
                rb->Map(0, nullptr, &mp);
                const auto* px = static_cast<const std::uint8_t*>(mp);
                const bool ok = std::abs(int(px[0]) - int(exp_r)) <= 1 && std::abs(int(px[1]) - int(exp_g)) <= 1 &&
                                std::abs(int(px[2]) - int(exp_b)) <= 1 && px[3] == 255;
                rb->Unmap(0, nullptr);
                if (!ok) return fail("ExecuteIndirect CountBuffer pixel mismatch");
                return 0;
            };

            if (run_indirect_count(0, 0, 0, 0) != 0) return 1;        // count = 0 -> black
            if (run_indirect_count(4, 32, 160, 240) != 0) return 1;   // count = 1 -> triangle

            vkDestroyCommandPool(vk_dev, pool, nullptr);
            vkDestroyImageView(vk_dev, rt_view, nullptr);
            vkDestroyPipeline(vk_dev, pso, nullptr);
            vkDestroyPipelineLayout(vk_dev, pl, nullptr);
            rb->Release();
            rt->Release();
            std::printf("  [PASS] ExecuteIndirect with suballocated argument and count buffers "
                        "(count=0 skipped, count=1 rendered)\n");
        }
#endif
        count_buf->Release();
        arg_buf->Release();
        cmd_sig->Release();
    }

    // -------------------------------------------------------------------------
    // Part 2: Hardware Ray Tracing / BVH (BLAS + TLAS + RayQueryKHR)
    // -------------------------------------------------------------------------
    if (!rt::d3d12::device_supports_ray_tracing(dev)) {
        std::printf("  [SKIP] VK_KHR_acceleration_structure / VK_KHR_ray_query not supported on %s\n", chosen.deviceName);
    } else {
        D3D12_HEAP_PROPERTIES up_hp{D3D12_HEAP_TYPE_UPLOAD};
        D3D12_HEAP_PROPERTIES def_hp{D3D12_HEAP_TYPE_DEFAULT};
        D3D12_HEAP_PROPERTIES rb_hp{D3D12_HEAP_TYPE_READBACK};

        auto vb_desc = make_buf_desc(256);
        ID3D12Resource* vb = nullptr;
        if (FAILED(dev->CreateCommittedResource(&up_hp, D3D12_HEAP_FLAG_NONE, &vb_desc, D3D12_RESOURCE_STATE_GENERIC_READ,
                                                nullptr, IID_PPV_ARGS(&vb))))
            return fail("CreateCommittedResource(vb)");
        void* p = nullptr;
        vb->Map(0, nullptr, &p);
        const float verts[9] = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
        std::memcpy(p, verts, sizeof(verts));
        vb->Unmap(0, nullptr);

        D3D12_RAYTRACING_GEOMETRY_DESC geom{};
        geom.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
        geom.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
        geom.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
        geom.Triangles.VertexCount = 3;
        geom.Triangles.VertexBuffer.StartAddress = vb->GetGPUVirtualAddress();
        geom.Triangles.VertexBuffer.StrideInBytes = 12;

        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS blas_in{};
        blas_in.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
        blas_in.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
        blas_in.NumDescs = 1;
        blas_in.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
        blas_in.pGeometryDescs = &geom;

        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO blas_info{};
        dev->GetRaytracingAccelerationStructurePrebuildInfo(&blas_in, &blas_info);
        if (blas_info.ResultDataMaxSizeInBytes == 0 || blas_info.ScratchDataSizeInBytes == 0)
            return fail("GetRaytracingAccelerationStructurePrebuildInfo(BLAS)");

        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS tlas_in{};
        tlas_in.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
        tlas_in.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
        tlas_in.NumDescs = 1;
        tlas_in.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;

        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO tlas_info{};
        dev->GetRaytracingAccelerationStructurePrebuildInfo(&tlas_in, &tlas_info);
        if (tlas_info.ResultDataMaxSizeInBytes == 0 || tlas_info.ScratchDataSizeInBytes == 0)
            return fail("GetRaytracingAccelerationStructurePrebuildInfo(TLAS)");

        const auto align256 = [](UINT64 v) -> UINT64 { return (v + 255u) & ~UINT64(255u); };
        auto blas_desc = make_buf_desc(align256(blas_info.ResultDataMaxSizeInBytes),
                                       D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        auto tlas_desc = make_buf_desc(align256(tlas_info.ResultDataMaxSizeInBytes),
                                       D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        auto scratch_desc = make_buf_desc(
            align256(std::max(blas_info.ScratchDataSizeInBytes, tlas_info.ScratchDataSizeInBytes)) + 256u,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        auto inst_desc_buf = make_buf_desc(256);

        ID3D12Resource* blas_buf = nullptr;
        ID3D12Resource* tlas_buf = nullptr;
        ID3D12Resource* tlas_ptr_buf = nullptr;
        ID3D12Resource* scratch_buf = nullptr;
        ID3D12Resource* inst_buf = nullptr;
        if (FAILED(dev->CreateCommittedResource(&def_hp, D3D12_HEAP_FLAG_NONE, &blas_desc,
                                                D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, nullptr,
                                                IID_PPV_ARGS(&blas_buf))) ||
            FAILED(dev->CreateCommittedResource(&def_hp, D3D12_HEAP_FLAG_NONE, &tlas_desc,
                                                D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, nullptr,
                                                IID_PPV_ARGS(&tlas_buf))) ||
            FAILED(dev->CreateCommittedResource(&def_hp, D3D12_HEAP_FLAG_NONE, &tlas_desc,
                                                D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, nullptr,
                                                IID_PPV_ARGS(&tlas_ptr_buf))) ||
            FAILED(dev->CreateCommittedResource(&def_hp, D3D12_HEAP_FLAG_NONE, &scratch_desc,
                                                D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                                IID_PPV_ARGS(&scratch_buf))) ||
            FAILED(dev->CreateCommittedResource(&up_hp, D3D12_HEAP_FLAG_NONE, &inst_desc_buf,
                                                D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&inst_buf))))
            return fail("CreateCommittedResource(RT buffers)");

        inst_buf->Map(0, nullptr, &p);
        D3D12_RAYTRACING_INSTANCE_DESC inst{};
        inst.Transform[0][0] = 1.0f;
        inst.Transform[1][1] = 1.0f;
        inst.Transform[2][2] = 1.0f;
        inst.InstanceMask = 0xFF;
        inst.Flags = D3D12_RAYTRACING_INSTANCE_FLAG_TRIANGLE_CULL_DISABLE;
        inst.AccelerationStructure = blas_buf->GetGPUVirtualAddress();
        std::memcpy(p, &inst, sizeof(inst));
        inst_buf->Unmap(0, nullptr);

        tlas_in.InstanceDescs = inst_buf->GetGPUVirtualAddress();

        // The same instance desc, named through an ARRAY_OF_POINTERS list. The list holds a console VA; the KHR
        // path must stage a copy whose entry is a Vulkan device address before building.
        auto inst_ptr_desc = make_buf_desc(256);
        ID3D12Resource* inst_ptr_buf = nullptr;
        if (FAILED(dev->CreateCommittedResource(&up_hp, D3D12_HEAP_FLAG_NONE, &inst_ptr_desc,
                                                D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                IID_PPV_ARGS(&inst_ptr_buf))))
            return fail("CreateCommittedResource(instance pointer list)");
        const UINT64 inst_va = inst_buf->GetGPUVirtualAddress();
        inst_ptr_buf->Map(0, nullptr, &p);
        std::memcpy(p, &inst_va, sizeof(inst_va));
        inst_ptr_buf->Unmap(0, nullptr);

        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS tlas_ptr_in = tlas_in;
        tlas_ptr_in.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY_OF_POINTERS;
        tlas_ptr_in.InstanceDescs = inst_ptr_buf->GetGPUVirtualAddress();
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO tlas_ptr_info{};
        dev->GetRaytracingAccelerationStructurePrebuildInfo(&tlas_ptr_in, &tlas_ptr_info);
        if (tlas_ptr_info.ResultDataMaxSizeInBytes != tlas_info.ResultDataMaxSizeInBytes ||
            tlas_ptr_info.ScratchDataSizeInBytes != tlas_info.ScratchDataSizeInBytes)
            return fail("prebuild info differs between ARRAY and ARRAY_OF_POINTERS");

        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC blas_build{};
        blas_build.DestAccelerationStructureData = blas_buf->GetGPUVirtualAddress();
        blas_build.Inputs = blas_in;
        blas_build.ScratchAccelerationStructureData = scratch_buf->GetGPUVirtualAddress();

        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC tlas_build{};
        tlas_build.DestAccelerationStructureData = tlas_buf->GetGPUVirtualAddress();
        tlas_build.Inputs = tlas_in;
        tlas_build.ScratchAccelerationStructureData = scratch_buf->GetGPUVirtualAddress();

        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC tlas_ptr_build{};
        tlas_ptr_build.DestAccelerationStructureData = tlas_ptr_buf->GetGPUVirtualAddress();
        tlas_ptr_build.Inputs = tlas_ptr_in;
        tlas_ptr_build.ScratchAccelerationStructureData = scratch_buf->GetGPUVirtualAddress();

        D3D12_RESOURCE_BARRIER uav_b{};
        uav_b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;

        alloc->Reset();
        cl->Reset(alloc, nullptr);
        cl->BuildRaytracingAccelerationStructure(&blas_build, 0, nullptr);
        cl->ResourceBarrier(1, &uav_b);
        cl->BuildRaytracingAccelerationStructure(&tlas_build, 0, nullptr);
        cl->ResourceBarrier(1, &uav_b);
        cl->BuildRaytracingAccelerationStructure(&tlas_ptr_build, 0, nullptr);
        cl->ResourceBarrier(1, &uav_b);
        if (FAILED(cl->Close())) return fail("RT build Close");

        ID3D12CommandList* lists[1] = {cl};
        queue->ExecuteCommandLists(1, lists);
        queue->Signal(fence, 1);
        if (!wait_fence(fence, 1)) return fail("wait_fence(RT build)");

        VkAccelerationStructureKHR tlas_vk =
            rt::d3d12::vulkan_acceleration_structure(dev, tlas_buf->GetGPUVirtualAddress());
        VkAccelerationStructureKHR tlas_ptr_vk =
            rt::d3d12::vulkan_acceleration_structure(dev, tlas_ptr_buf->GetGPUVirtualAddress());
        if (!tlas_vk || !tlas_ptr_vk) return fail("vulkan_acceleration_structure(TLAS)");

#if defined(LEONIDA_TEST_CS_RAYQUERY_SPV)
        std::vector<std::uint32_t> rq_words;
        if (!read_spv(LEONIDA_TEST_CS_RAYQUERY_SPV, rq_words)) return fail("read cs_rayquery.spv");

        auto out_desc = make_buf_desc(64);
        ID3D12Resource* out_buf = nullptr;
        if (FAILED(dev->CreateCommittedResource(&rb_hp, D3D12_HEAP_FLAG_NONE, &out_desc, D3D12_RESOURCE_STATE_COPY_DEST,
                                                nullptr, IID_PPV_ARGS(&out_buf))))
            return fail("CreateCommittedResource(rayquery out)");

        VkDescriptorSetLayoutBinding bindings[2]{};
        bindings[0] = {0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo dsli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dsli.bindingCount = 2;
        dsli.pBindings = bindings;
        VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
        vkCreateDescriptorSetLayout(vk_dev, &dsli, nullptr, &dsl);

        VkDescriptorPoolSize pool_sizes[2] = {
            {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
        };
        VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpi.maxSets = 1;
        dpi.poolSizeCount = 2;
        dpi.pPoolSizes = pool_sizes;
        VkDescriptorPool dpool = VK_NULL_HANDLE;
        vkCreateDescriptorPool(vk_dev, &dpi, nullptr, &dpool);

        VkDescriptorSet dset = VK_NULL_HANDLE;
        VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dsai.descriptorPool = dpool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &dsl;
        vkAllocateDescriptorSets(vk_dev, &dsai, &dset);

        VkWriteDescriptorSetAccelerationStructureKHR as_write{
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
        as_write.accelerationStructureCount = 1;
        as_write.pAccelerationStructures = &tlas_vk;
        VkDescriptorBufferInfo out_bi{rt::d3d12::vulkan_buffer(out_buf), 0, 64};

        VkWriteDescriptorSet writes[2]{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].pNext = &as_write;
        writes[0].dstSet = dset;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = dset;
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].pBufferInfo = &out_bi;
        vkUpdateDescriptorSets(vk_dev, 2, writes, 0, nullptr);

        VkShaderModule cs_mod = VK_NULL_HANDLE;
        VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0, rq_words.size() * 4,
                                     rq_words.data()};
        if (vkCreateShaderModule(vk_dev, &smi, nullptr, &cs_mod) != VK_SUCCESS) return fail("vkCreateShaderModule(rq)");
        VkPipelineLayout pl = VK_NULL_HANDLE;
        VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &dsl;
        vkCreatePipelineLayout(vk_dev, &pli, nullptr, &pl);
        VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpi.stage.module = cs_mod;
        cpi.stage.pName = "main";
        cpi.layout = pl;
        VkPipeline pso = VK_NULL_HANDLE;
        if (vkCreateComputePipelines(vk_dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &pso) != VK_SUCCESS)
            return fail("vkCreateComputePipelines(rq)");
        vkDestroyShaderModule(vk_dev, cs_mod, nullptr);

        VkCommandPool pool = VK_NULL_HANDLE;
        VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pci.queueFamilyIndex = family;
        vkCreateCommandPool(vk_dev, &pci, nullptr, &pool);
        VkCommandBuffer cb = VK_NULL_HANDLE;
        VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cbai.commandPool = pool;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        vkAllocateCommandBuffers(vk_dev, &cbai, &cb);

        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cb, &bi);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pso);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &dset, 0, nullptr);
        vkCmdDispatch(cb, 1, 1, 1);
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, nullptr,
                             0, nullptr);
        vkEndCommandBuffer(cb);

        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cb;
        vkQueueSubmit(vk_queue, 1, &si, VK_NULL_HANDLE);
        vkQueueWaitIdle(vk_queue);

        out_buf->Map(0, nullptr, &p);
        const auto* res = static_cast<const std::uint32_t*>(p);
        const std::uint32_t hit = res[0];
        const std::uint32_t miss = res[1];
        out_buf->Unmap(0, nullptr);
        if (hit != 1u || miss != 0u) {
            std::fprintf(stderr, "FAIL: RayQueryKHR ARRAY expected hit=1 miss=0, got hit=%u miss=%u\n", hit, miss);
            return 1;
        }

        // The second TLAS was built from an ARRAY_OF_POINTERS list. Run the same query against it: without the
        // address translation the pointer bytes are read as an instance desc with mask 0, and the ray misses.
        as_write.pAccelerationStructures = &tlas_ptr_vk;
        vkUpdateDescriptorSets(vk_dev, 1, writes, 0, nullptr);
        VkCommandBuffer cb2 = VK_NULL_HANDLE;
        vkAllocateCommandBuffers(vk_dev, &cbai, &cb2);
        vkBeginCommandBuffer(cb2, &bi);
        vkCmdBindPipeline(cb2, VK_PIPELINE_BIND_POINT_COMPUTE, pso);
        vkCmdBindDescriptorSets(cb2, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &dset, 0, nullptr);
        vkCmdDispatch(cb2, 1, 1, 1);
        vkCmdPipelineBarrier(cb2, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0,
                             nullptr, 0, nullptr);
        vkEndCommandBuffer(cb2);
        VkSubmitInfo si2{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si2.commandBufferCount = 1;
        si2.pCommandBuffers = &cb2;
        vkQueueSubmit(vk_queue, 1, &si2, VK_NULL_HANDLE);
        vkQueueWaitIdle(vk_queue);

        out_buf->Map(0, nullptr, &p);
        const auto* res2 = static_cast<const std::uint32_t*>(p);
        const std::uint32_t hit2 = res2[0];
        const std::uint32_t miss2 = res2[1];
        out_buf->Unmap(0, nullptr);
        if (hit2 != 1u || miss2 != 0u) {
            std::fprintf(stderr,
                         "FAIL: RayQueryKHR ARRAY_OF_POINTERS expected hit=1 miss=0, got hit=%u miss=%u\n", hit2,
                         miss2);
            return 1;
        }
        std::printf("  [PASS] BLAS + TLAS (ARRAY and ARRAY_OF_POINTERS) + RayQueryKHR (hit=%u, miss=%u)\n", hit, miss);

        vkDestroyCommandPool(vk_dev, pool, nullptr);
        vkDestroyPipeline(vk_dev, pso, nullptr);
        vkDestroyPipelineLayout(vk_dev, pl, nullptr);
        vkDestroyDescriptorPool(vk_dev, dpool, nullptr);
        vkDestroyDescriptorSetLayout(vk_dev, dsl, nullptr);
        out_buf->Release();
#endif
        // A null entry in the pointer list is refused while the list is prepared, before anything is submitted.
        auto bad_ptr_desc = make_buf_desc(256);
        ID3D12Resource* bad_ptr_buf = nullptr;
        if (FAILED(dev->CreateCommittedResource(&up_hp, D3D12_HEAP_FLAG_NONE, &bad_ptr_desc,
                                                D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                IID_PPV_ARGS(&bad_ptr_buf))))
            return fail("CreateCommittedResource(null instance pointer list)");
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS bad_in = tlas_ptr_in;
        bad_in.InstanceDescs = bad_ptr_buf->GetGPUVirtualAddress();
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC bad_build{};
        bad_build.DestAccelerationStructureData = tlas_ptr_buf->GetGPUVirtualAddress();
        bad_build.Inputs = bad_in;
        bad_build.ScratchAccelerationStructureData = scratch_buf->GetGPUVirtualAddress();
        alloc->Reset();
        cl->Reset(alloc, nullptr);
        cl->BuildRaytracingAccelerationStructure(&bad_build, 0, nullptr);
        if (FAILED(cl->Close())) return fail("null pointer build Close");
        queue->ExecuteCommandLists(1, lists);
        if (rt::d3d12::last_execute_result(queue) != leonida::ReplayResult::invalid_copy)
            return fail("null instance pointer must be refused before submission");
        std::printf("  [PASS] ARRAY_OF_POINTERS null entry refused before submission\n");
        bad_ptr_buf->Release();

        // A pointer list placed in a DEFAULT heap is GPU-written, so its mapped arena bytes may still hold an
        // earlier pointer when the CPU prepares a build. Seed those bytes with a valid pointer in one submission,
        // then overwrite them on the GPU in the next and build. The shim must refuse the second list instead of
        // translating the stale bytes into a TLAS built from the wrong instances.
        D3D12_HEAP_DESC placed_heap_desc{};
        placed_heap_desc.SizeInBytes = 64 * 1024;
        placed_heap_desc.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
        placed_heap_desc.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
        ID3D12Heap* placed_heap = nullptr;
        if (FAILED(dev->CreateHeap(&placed_heap_desc, IID_PPV_ARGS(&placed_heap))))
            return fail("CreateHeap(DEFAULT pointer list heap)");
        auto stale_ptr_desc = make_buf_desc(256);
        ID3D12Resource* stale_ptr_buf = nullptr;
        if (FAILED(dev->CreatePlacedResource(placed_heap, 0, &stale_ptr_desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                             IID_PPV_ARGS(&stale_ptr_buf))))
            return fail("CreatePlacedResource(DEFAULT pointer list)");
        // The second submission's GPU overwrite. Its bytes are still the seeded pointer when prepare() runs.
        auto zero_ptr_desc = make_buf_desc(256);
        ID3D12Resource* zero_ptr_buf = nullptr;
        if (FAILED(dev->CreateCommittedResource(&up_hp, D3D12_HEAP_FLAG_NONE, &zero_ptr_desc,
                                                D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                IID_PPV_ARGS(&zero_ptr_buf))))
            return fail("CreateCommittedResource(zero pointer list)");

        alloc->Reset();
        cl->Reset(alloc, nullptr);
        cl->CopyBufferRegion(stale_ptr_buf, 0, inst_ptr_buf, 0, sizeof(UINT64));
        if (FAILED(cl->Close())) return fail("seed placed pointer list Close");
        queue->ExecuteCommandLists(1, lists);
        queue->Signal(fence, 2);
        if (!wait_fence(fence, 2)) return fail("wait_fence(seed placed pointer list)");

        alloc->Reset();
        cl->Reset(alloc, nullptr);
        cl->CopyBufferRegion(stale_ptr_buf, 0, zero_ptr_buf, 0, sizeof(UINT64));
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC stale_build{};
        stale_build.DestAccelerationStructureData = tlas_ptr_buf->GetGPUVirtualAddress();
        stale_build.Inputs = tlas_ptr_in;
        stale_build.Inputs.InstanceDescs = stale_ptr_buf->GetGPUVirtualAddress();
        stale_build.ScratchAccelerationStructureData = scratch_buf->GetGPUVirtualAddress();
        cl->BuildRaytracingAccelerationStructure(&stale_build, 0, nullptr);
        if (FAILED(cl->Close())) return fail("stale placed pointer build Close");
        queue->ExecuteCommandLists(1, lists);
        if (rt::d3d12::last_execute_result(queue) != leonida::ReplayResult::unsupported)
            return fail("GPU-written placed pointer list must be refused as unsupported before submission");
        std::printf("  [PASS] ARRAY_OF_POINTERS GPU-written placed pointer list unsupported before submission\n");
        zero_ptr_buf->Release();
        stale_ptr_buf->Release();
        placed_heap->Release();

        inst_ptr_buf->Release();
        inst_buf->Release();
        tlas_ptr_buf->Release();
        scratch_buf->Release();
        tlas_buf->Release();
        blas_buf->Release();
        vb->Release();
    }

    fence->Release();
    cl->Release();
    alloc->Release();
    queue->Release();
    dev->Release();
    vkDeviceWaitIdle(vk_dev);
    vkDestroyDevice(vk_dev, nullptr);
    if (messenger && destroy_dbg) destroy_dbg(instance, messenger, nullptr);
    destroy_inst(instance, nullptr);

    if (validation_errors() != 0) {
        std::fprintf(stderr, "FAIL: %u Vulkan validation errors (Core + Sync)\n", validation_errors());
        return 1;
    }
    std::printf("device: %s (validation %s)\n", chosen.deviceName, val_active ? "on" : "off");
    std::printf("d3d12_indirect_rt_test: PASS (0 validation errors)\n");
    return 0;
}
