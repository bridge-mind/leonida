// Synthetic GPU lifetime test: gate the queue so no predicate arena can retire before all submissions exist.
#include "vk_test_device.h"
#include "../src/vulkan_backend.h"
#include <array>
#include <cstring>
#include <cstdio>
#define EXPECT(x) do { if (!(x)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); return 1; } } while (0)
namespace {
PFN_vkCmdPipelineBarrier2 real_barrier2 = nullptr;
unsigned host_barriers = 0;
VkPipelineStageFlags2 host_source_stages = 0;
VKAPI_ATTR void VKAPI_CALL observe_barrier2(VkCommandBuffer cb, const VkDependencyInfo* info) {
    for (std::uint32_t i = 0; i < info->memoryBarrierCount; ++i) {
        const auto& barrier = info->pMemoryBarriers[i];
        if (barrier.dstStageMask & VK_PIPELINE_STAGE_2_HOST_BIT) {
            ++host_barriers;
            host_source_stages |= barrier.srcStageMask;
        }
    }
    real_barrier2(cb, info);
}
}
int main() {
    vktest::Device d;
    const auto status = d.initialize();
    if (status == vktest::DeviceStatus::unavailable) {
        std::printf("vk_predicate_capture_test: no Vulkan device, skipped\n");
        return 77;
    }
    if (status != vktest::DeviceStatus::ready) return 1;
    if (!d.conditional_rendering) { std::puts("predicate capture: skipped (no conditional rendering)"); return 0; }
    using namespace leonida;
    VkPhysicalDeviceMemoryProperties memory{};
    d.instance_fn<PFN_vkGetPhysicalDeviceMemoryProperties>("vkGetPhysicalDeviceMemoryProperties")(d.physical, &memory);
    VkPhysicalDeviceProperties properties{};
    d.instance_fn<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties")(d.physical, &properties);
    PredicateReducer limits(d.device_proc, d.device, memory, 16);
    EXPECT(!limits.batch(0) && !limits.batch(2));
    PredicateReducer missing(+[](VkDevice, const char*) -> PFN_vkVoidFunction { return nullptr; }, d.device, memory, 16);
    EXPECT(!missing.batch(1));
    auto no_memory = memory; no_memory.memoryTypeCount = 0;
    PredicateReducer allocation_failure(d.device_proc, d.device, no_memory, 16);
    EXPECT(!allocation_failure.batch(1));
    VulkanContext context(d.device_proc, d.device, true, true);
    context.configure_predication(d.device_proc, memory, properties.limits.maxStorageBufferRange, d.family);
    VulkanQueue queue(context, d.queue, d.family);
    VulkanFence gate(context, 0);
    EXPECT(context.loaded() && queue.ok() && gate.semaphore());
    constexpr unsigned count = 32;
    void* mapped = nullptr;
    const VkBuffer predicate = d.buffer(16, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &mapped);
    EXPECT(predicate && mapped);
    const std::uint64_t values[] = {UINT64_C(1) << 32, 0};
    std::memcpy(mapped, values, sizeof(values));
    const VkImage image = d.image(VK_FORMAT_R8G8B8A8_UINT, 2, 1, count, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    void* pixels = nullptr;
    const VkBuffer readback = d.buffer(count * 8, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &pixels);
    EXPECT(image && readback && pixels);
    ResourceBinding source{}; source.buffer = predicate; source.size = 16;
    ResourceBinding target{}; target.image = image; target.format = VK_FORMAT_R8G8B8A8_UINT;
    target.width = 2; target.height = 1; target.array_layers = count;
    context.add_resource(1, source);
    context.add_resource(2, target, VK_IMAGE_LAYOUT_UNDEFINED);
    // Each list captures high-only -> zero -> high-only with two offsets, all while no prior batch can finish.
    std::array<CommandList, count> lists;
    queue.wait(gate.semaphore(), 1);
    bool submitted = true;
    for (unsigned i = 0; i < count; ++i) {
        auto& list = lists[i];
        submitted = submitted && list.set_predication({1, 0, d3d12_predication::equal_zero});
        submitted = submitted && list.clear_render_target({2, 0, i, 1}, {7, 8, 9, 10});
        submitted = submitted && list.set_predication({1, 8, d3d12_predication::equal_zero});
        submitted = submitted && list.clear_render_target({2, 0, i, 1}, {99, 99, 99, 99});
        submitted = submitted && list.set_predication({1, 0, d3d12_predication::not_equal_zero});
        submitted = submitted && list.clear_render_target({2, 0, i, 1}, {100, 100, 100, 100});
        submitted = submitted && list.set_predication({0, 0, 0});
        const Rect right{1, 0, 2, 1};
        submitted = submitted && list.clear_render_target({2, 0, i, 1}, {float(i), 31, 73, 127}, {&right, 1});
        submitted = submitted && list.close();
        const CommandList* pointer = &list;
        if (!submitted || queue.execute({&pointer, 1}) != ReplayResult::success) { submitted = false; break; }
    }
    // Always release the gate, including on failure, so destructors cannot wait forever on queued work.
    EXPECT(gate.signal(1));
    EXPECT(queue.idle());
    EXPECT(submitted);
    // Completed command-buffer slots must release their old arenas before being reused.
    for (auto& list : lists) {
        const CommandList* pointer = &list;
        EXPECT(queue.execute({&pointer, 1}) == ReplayResult::success);
    }
    EXPECT(queue.idle());
    // Failed submissions must not commit their speculative layout transitions.
    EXPECT(context.layout(2, 0) == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    CommandList failed;
    EXPECT(failed.transition({2, 0, d3d12_state::render_target, d3d12_state::copy_source}) && failed.close());
    const CommandList* failed_pointer = &failed;
    const auto real_submit = context.vk.vkQueueSubmit;
    context.vk.vkQueueSubmit = +[](VkQueue, std::uint32_t, const VkSubmitInfo*, VkFence) -> VkResult {
        return VK_ERROR_OUT_OF_HOST_MEMORY; // no GPU work and no device loss
    };
    const auto failed_result = queue.execute({&failed_pointer, 1});
    context.vk.vkQueueSubmit = real_submit;
    EXPECT(failed_result == ReplayResult::unsupported);
    EXPECT(context.layout(2, 0) == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    // Preserve main's readback-stage collection on every logical queue kind, using real transfer readbacks.
    ResourceBinding output{}; output.buffer = readback; output.size = count * 8; output.host_readable = true;
    context.add_resource(3, output);
    CommandList copy, capture_only, empty;
    EXPECT(copy.copy_buffer({3, 0, 1, 0, 16}) && copy.close());
    EXPECT(capture_only.set_predication({1, 0, d3d12_predication::equal_zero}) && capture_only.close());
    EXPECT(empty.close());
    real_barrier2 = context.vk.vkCmdPipelineBarrier2;
    EXPECT(real_barrier2);
    context.vk.vkCmdPipelineBarrier2 = observe_barrier2;
    for (const auto kind : {VulkanQueue::QueueKind::direct, VulkanQueue::QueueKind::compute, VulkanQueue::QueueKind::copy}) {
        VulkanQueue typed(context, d.queue, d.family, kind);
        EXPECT(typed.ok());
        const CommandList* no_write = &empty;
        host_barriers = 0; host_source_stages = 0;
        EXPECT(typed.execute({&no_write, 1}) == ReplayResult::success);
        EXPECT(host_barriers == 0);
        const CommandList* capture = &capture_only;
        EXPECT(typed.execute({&capture, 1}) == (kind == VulkanQueue::QueueKind::copy
                                                ? ReplayResult::unsupported : ReplayResult::success));
        EXPECT(host_barriers == 0);
        const CommandList* transfer = &copy;
        EXPECT(typed.execute({&transfer, 1}) == ReplayResult::success);
        EXPECT(typed.idle());
        EXPECT(host_barriers == 1 && host_source_stages != 0);
        EXPECT((host_source_stages & ~VulkanQueue::queue_supported_stages(kind)) == 0);
        EXPECT((host_source_stages & ~(VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT)) == 0);
        EXPECT(std::memcmp(pixels, values, sizeof(values)) == 0);
    }
    context.vk.vkCmdPipelineBarrier2 = real_barrier2;
    EXPECT(d.run([&](VkCommandBuffer cb) {
        VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT; barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image; barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, count};
        context.vk.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                      0, 0, nullptr, 0, nullptr, 1, &barrier);
        VkBufferImageCopy region{}; region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, count};
        region.imageExtent = {2, 1, 1};
        context.vk.vkCmdCopyImageToBuffer(cb, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback, 1, &region);
        VkMemoryBarrier host{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        host.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        context.vk.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                                      0, 1, &host, 0, nullptr, 0, nullptr);
        return true;
    }));
    const auto* bytes = static_cast<const unsigned char*>(pixels);
    for (unsigned i = 0; i < count; ++i) {
        EXPECT(bytes[i * 8] == 7 && bytes[i * 8 + 1] == 8 && bytes[i * 8 + 2] == 9 && bytes[i * 8 + 3] == 10);
        EXPECT(bytes[i * 8 + 4] == i && bytes[i * 8 + 5] == 31 && bytes[i * 8 + 6] == 73 && bytes[i * 8 + 7] == 127);
    }
    EXPECT(d.validation_errors() == 0);
    std::puts("predicate capture: 32 gated + 32 reused submissions, 192 snapshots, offset rebinding and reset passed");
    std::puts("predicate capture integration: failed-submit tracker rollback and direct/compute/copy readback stages passed");
}
