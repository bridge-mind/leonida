// Opt-in Khronos validation for the standalone Vulkan test programs. With LEONIDA_VK_VALIDATION set (and not "0") the
// instance enables VK_LAYER_KHRONOS_validation and VK_EXT_debug_utils, every message goes to stderr, and
// validation-type errors are counted so the program can fail on them (exit_status). The layer's own variables still
// apply, e.g. VK_KHRONOS_VALIDATION_GPUAV_ENABLE=1 for GPU-assisted validation of what a shader does at run time.
// Include after vulkan_loader.h (VK_NO_PROTOTYPES): every call goes through the vkGetInstanceProcAddr it loaded.
#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace leonida::vk_test {

class OptInValidation {
public:
    // Before vkCreateInstance: adds the layer, the extension and a messenger that also covers instance creation to
    // ici (keeping what it already enables). False, after a message, when validation was asked for but the layer
    // isn't installed: the caller should fail instead of running unchecked. This object must outlive the call.
    bool prepare(PFN_vkGetInstanceProcAddr gipa, VkInstanceCreateInfo& ici) {
        const char* env = std::getenv("LEONIDA_VK_VALIDATION");
        on_ = env && std::strcmp(env, "0") != 0;
        if (!on_) return true;
        const auto enumerate = reinterpret_cast<PFN_vkEnumerateInstanceLayerProperties>(
            gipa(VK_NULL_HANDLE, "vkEnumerateInstanceLayerProperties"));
        uint32_t count = 0;
        if (enumerate) enumerate(&count, nullptr);
        std::vector<VkLayerProperties> available(count);
        if (count) enumerate(&count, available.data());
        bool found = false;
        for (const auto& properties : available) found = found || !std::strcmp(properties.layerName, layer_);
        if (!found) {
            std::fprintf(stderr, "LEONIDA_VK_VALIDATION is set but %s is not installed\n", layer_);
            return false;
        }
        layers_.assign(ici.ppEnabledLayerNames, ici.ppEnabledLayerNames + ici.enabledLayerCount);
        layers_.push_back(layer_);
        extensions_.assign(ici.ppEnabledExtensionNames, ici.ppEnabledExtensionNames + ici.enabledExtensionCount);
        extensions_.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        info_.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        info_.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
        info_.pfnUserCallback = on_message;
        info_.pNext = ici.pNext;
        ici.pNext = &info_;
        ici.enabledLayerCount = uint32_t(layers_.size());
        ici.ppEnabledLayerNames = layers_.data();
        ici.enabledExtensionCount = uint32_t(extensions_.size());
        ici.ppEnabledExtensionNames = extensions_.data();
        return true;
    }

    // After vkCreateInstance: the messenger for everything that follows. The program may exit without destroying it.
    VkResult attach(PFN_vkGetInstanceProcAddr gipa, VkInstance instance) {
        if (!on_) return VK_SUCCESS;
        const auto create = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            gipa(instance, "vkCreateDebugUtilsMessengerEXT"));
        if (!create) return VK_ERROR_EXTENSION_NOT_PRESENT;
        info_.pNext = nullptr;
        return create(instance, &info_, nullptr, &messenger_);
    }

    bool enabled() const { return on_; }
    static unsigned errors() { return error_count(); }

    // The exit status with validation counted: a run that reported a validation error fails (1) even when its own
    // check passed or it reported itself unsupported (77); any other failure code is kept.
    static int exit_status(int rc) {
        if (!errors()) return rc;
        std::fprintf(stderr, "%u validation errors\n", errors());
        return rc != 0 && rc != 77 ? rc : 1;
    }

private:
    static unsigned& error_count() {
        static unsigned count = 0;
        return count;
    }

    // Only validation-type errors count: a broken third-party layer must not fail the test.
    static VKAPI_ATTR VkBool32 VKAPI_CALL on_message(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                                     VkDebugUtilsMessageTypeFlagsEXT types,
                                                     const VkDebugUtilsMessengerCallbackDataEXT* data, void*) {
        const bool validation = (types & VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT) != 0;
        if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) && validation) ++error_count();
        std::fprintf(stderr, "%s: %s\n", validation ? "validation" : "vulkan (not counted)",
                     data && data->pMessage ? data->pMessage : "(no message)");
        return VK_FALSE;
    }

    bool on_ = false;
    const char* layer_ = "VK_LAYER_KHRONOS_validation";
    std::vector<const char*> layers_, extensions_;
    VkDebugUtilsMessengerCreateInfoEXT info_{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
    VkDebugUtilsMessengerEXT messenger_ = VK_NULL_HANDLE;
};

}  // namespace leonida::vk_test
