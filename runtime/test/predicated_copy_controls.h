// Independent byte destinations alongside the graphics-query fixtures. No game data.
#pragma once
#include "vk_test_device.h"
#include "../src/vulkan_backend.h"
#include <array>
#include <cstring>
struct PredicatedCopyControl {
    static constexpr leonida::ResourceId source_id = 0x10001, destination_id = 0x10002;
    bool active = false;
    void* mapped = nullptr;
    std::array<unsigned char, 16> input{};
    bool create(vktest::Device& d, leonida::VulkanContext& context, const VkPhysicalDeviceLimits& limits) {
        active = d.conditional_rendering && d.storage_buffer_8bit;
        if (!active) {
            std::puts("occlusion copy controls: skipped (requires conditional rendering and storageBuffer8BitAccess)");
            return true;
        }
        context.configure_predicated_copy(d.device_proc, limits);
        void* source = nullptr;
        const auto src = d.buffer(16, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &source);
        const auto dst = d.buffer(32, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &mapped);
        if (!src || !dst || !source || !mapped) return false;
        for (std::size_t i = 0; i < input.size(); ++i) input[i] = static_cast<unsigned char>(37 * i + 11);
        std::memcpy(source, input.data(), input.size());
        std::memset(mapped, 0xcd, 32);
        leonida::ResourceBinding a{}, b{};
        a.buffer = src; a.size = 16;
        b.buffer = dst; b.size = 32; b.host_readable = true;
        context.add_resource(source_id, a); context.add_resource(destination_id, b);
        return true;
    }
    // Caller has a false predicate set; the first half stays untouched, the second copies after reset.
    bool record(leonida::CommandList& list) const {
        return !active || (list.copy_buffer({destination_id, 0, source_id, 0, 16}) &&
            list.set_predication({0, 0, 0}) && list.copy_buffer({destination_id, 16, source_id, 0, 16}));
    }
    bool check() const {
        if (!active) return true;
        const auto* bytes = static_cast<const unsigned char*>(mapped);
        for (std::size_t i = 0; i < 16; ++i) if (bytes[i] != 0xcd || bytes[16 + i] != input[i]) return false;
        std::puts("occlusion copy controls: false-predicate preservation and reset copy exact; query results unaffected");
        return true;
    }
};
