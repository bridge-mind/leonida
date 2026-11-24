// Proves that Vulkan synchronization validation reports hazards on this machine when LEONIDA_VK_VALIDATION=1, so a
// "0 validation errors" from the GPU tests means the layer looked, not that it was absent or silent.
//
// vk_test_device.h asks for VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT whenever validation is on.
// This test records two vkCmdCopyBuffer into the same bytes:
//   - without a barrier between them, synchronization validation must report the write-after-write hazard;
//   - with a transfer -> transfer memory barrier, it must report nothing.
// The second half keeps the first honest: a layer that reported every copy would fail it.
//
// Exit codes: 0 pass; 1 the hazard was not reported, the barrier case was, or the device failed;
// 77 (ctest skip) when validation is not requested or there is no Vulkan device. No game files, no runtime code.
#include "vk_test_device.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

// Validation errors reported while recording and submitting the two copies, or ~0u when the copies could not run.
unsigned copy_twice(vktest::Device& d, bool barrier) {
    void* mapped = nullptr;
    const VkBuffer src = d.buffer(4096, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &mapped);
    const VkBuffer dst = d.buffer(4096, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &mapped);
    if (!src || !dst) return ~0u;
    const auto copy = d.fn<PFN_vkCmdCopyBuffer>("vkCmdCopyBuffer");
    const auto pipeline_barrier = d.fn<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier");
    const unsigned before = d.validation_errors();
    const bool ran = d.run([&](VkCommandBuffer cb) {
        const VkBufferCopy region{0, 0, 4096};
        copy(cb, src, dst, 1, &region);
        if (barrier) {
            VkMemoryBarrier memory{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            memory.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            memory.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            pipeline_barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &memory, 0,
                             nullptr, 0, nullptr);
        }
        copy(cb, src, dst, 1, &region);
        return true;
    });
    return ran ? d.validation_errors() - before : ~0u;
}

bool validation_requested() {
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)  // getenv: read once, as vk_test_device.h does
#endif
    const char* env = std::getenv("LEONIDA_VK_VALIDATION");
#ifdef _MSC_VER
#pragma warning(pop)
#endif
    return env && std::strcmp(env, "0") != 0;
}

}  // namespace

int main() {
    if (!validation_requested()) {
        std::printf("SKIP vk_sync_validation_selftest: LEONIDA_VK_VALIDATION is not set, nothing to prove\n");
        return 77;
    }
    vktest::Device d;
    const auto status = d.initialize();  // fails, with a message, when the Khronos layer is not installed
    if (status == vktest::DeviceStatus::unavailable) {
        std::printf("SKIP vk_sync_validation_selftest: no Vulkan device\n");
        return 77;
    }
    if (status != vktest::DeviceStatus::ready || !d.validation) {
        std::fprintf(stderr, "vk_sync_validation_selftest: no validated Vulkan device\n");
        return 1;
    }
    const unsigned hazard = copy_twice(d, false);
    const unsigned clean = copy_twice(d, true);
    if (hazard == ~0u || clean == ~0u) {
        std::fprintf(stderr, "vk_sync_validation_selftest: the copies could not be recorded or submitted\n");
        return 1;
    }
    std::printf("two copies into the same bytes: %u validation error(s) without a barrier, %u with one\n", hazard,
                clean);
    if (hazard == 0) {
        std::fprintf(stderr, "vk_sync_validation_selftest: FAIL, the write-after-write hazard was not reported: "
                             "synchronization validation is not active, so 0 errors elsewhere proves nothing\n");
        return 1;
    }
    if (clean != 0) {
        std::fprintf(stderr, "vk_sync_validation_selftest: FAIL, the synchronized copies were reported too\n");
        return 1;
    }
    std::printf("vk_sync_validation_selftest: PASS, synchronization validation is active\n");
    return 0;
}
