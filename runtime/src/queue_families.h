#pragma once

// Picking Vulkan queue families for D3D12's direct, compute and copy queues. Header-only and free of the rest of
// the backend, so test harnesses can use it too.

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <span>

namespace leonida {

// Queue families for D3D12's three queue types. Direct needs graphics. Compute prefers a family with compute and no
// graphics (async compute), copy one with transfer and neither (a DMA engine); each falls back to a more capable
// family, ending at the graphics one. A dedicated copy family is only used when its image transfer granularity is
// 1x1x1, because D3D12 copies any box.
struct QueueFamilies {
    std::uint32_t direct = 0;
    std::uint32_t compute = 0;
    std::uint32_t copy = 0;
};

inline QueueFamilies pick_queue_families(std::span<const VkQueueFamilyProperties> families, std::uint32_t graphics) {
    QueueFamilies out{graphics, graphics, graphics};
    constexpr VkQueueFlags g = VK_QUEUE_GRAPHICS_BIT, c = VK_QUEUE_COMPUTE_BIT, t = VK_QUEUE_TRANSFER_BIT;
    const auto fine_grained = [](const VkQueueFamilyProperties& f) {
        return f.minImageTransferGranularity.width == 1 && f.minImageTransferGranularity.height == 1 &&
               f.minImageTransferGranularity.depth == 1;
    };
    bool have_compute = false, have_copy = false;
    for (std::uint32_t i = 0; i < families.size(); ++i) {
        const auto& f = families[i];
        if (!f.queueCount) continue;
        if (!have_compute && (f.queueFlags & c) && !(f.queueFlags & g)) {
            out.compute = i;
            have_compute = true;
        }
        if (!have_copy && (f.queueFlags & t) && !(f.queueFlags & (g | c)) && fine_grained(f)) {
            out.copy = i;
            have_copy = true;
        }
    }
    // No DMA family: copies go to the async-compute family, which can transfer too (compute implies transfer).
    if (!have_copy && have_compute && fine_grained(families[out.compute])) out.copy = out.compute;
    return out;
}

// A D3D12 COPY/COMPUTE queue may call UpdateTileMappings at any time. Vulkan binds sparse
// pages on the VkQueue itself, so a family without SPARSE_BINDING cannot back that D3D12
// queue when reserved buffers are offered. Use the graphics family (already checked for
// sparse support by the device) rather than submitting an invalid vkQueueBindSparse.
// Kept separate from the preference-only picker so devices without sparse resources keep
// their dedicated async families unchanged.
inline QueueFamilies sparse_capable_queue_families(std::span<const VkQueueFamilyProperties> families,
                                                  QueueFamilies chosen, bool sparse_buffers) {
    if (!sparse_buffers || chosen.direct >= families.size() ||
        !(families[chosen.direct].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT))
        return chosen;
    if (chosen.compute >= families.size() ||
        !(families[chosen.compute].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT))
        chosen.compute = chosen.direct;
    if (chosen.copy >= families.size() ||
        !(families[chosen.copy].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT))
        chosen.copy = chosen.direct;
    return chosen;
}

// Which VkQueue a new D3D12 queue runs on. `pools` are its candidate VkQueues in order of preference (a COMPUTE or
// COPY queue's own family first, then the graphics family); each entry counts the live D3D12 queues on that VkQueue.
// An unused VkQueue comes first, in pool order, so a GPU with enough queues keeps one VkQueue per live D3D12 queue
// (D3D-97). With none unused, the VkQueue with the fewest D3D12 queues is shared, the earliest on a tie: the D3D12
// queues on it then run one after the other, in the order their batches reach it (docs/d3d12_vulkan.md 4.4).
// `pool` is SIZE_MAX when every pool is empty.
struct QueueSlot {
    std::size_t pool = SIZE_MAX;
    std::size_t index = 0;
};

inline QueueSlot pick_queue_slot(std::initializer_list<std::span<const std::uint32_t>> pools) {
    QueueSlot chosen;  // the first VkQueue with the fewest users: an unused one whenever there is one
    std::uint32_t fewest = UINT32_MAX;
    std::size_t p = 0;
    for (const auto users : pools) {
        for (std::size_t i = 0; i < users.size(); ++i)
            if (users[i] < fewest) {
                fewest = users[i];
                chosen = {p, i};
            }
        ++p;
    }
    return chosen;
}

}  // namespace leonida
