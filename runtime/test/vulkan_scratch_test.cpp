// Unit tests for ScratchBuffers (runtime/src/vulkan_scratch.h). CPU-only, no Vulkan instance and no GPU:
// synthetic device procs exercise alignment, minimum chunk size, failure cleanup, retire/collect/forget, and spare cap.
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include "../src/vulkan_scratch.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <unordered_set>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

using leonida::ScratchBuffers;

namespace {

template <typename Handle>
Handle make_handle(std::uintptr_t id) {
    return reinterpret_cast<Handle>(id);
}

struct FakeVulkan {
    std::uintptr_t next_buffer = 1;
    std::uintptr_t next_memory = 1;
    std::unordered_map<std::uintptr_t, VkDeviceSize> live_buffers;
    std::unordered_set<std::uintptr_t> live_memory;
    std::unordered_map<std::uintptr_t, std::uint64_t> semaphore_values;

    int create_buffer_calls = 0;
    int destroy_buffer_calls = 0;
    int allocate_calls = 0;
    int free_memory_calls = 0;
    int wait_idle_calls = 0;
    int get_counter_calls = 0;

    VkResult create_buffer_result = VK_SUCCESS;
    VkResult bind_result = VK_SUCCESS;
    VkResult counter_result = VK_SUCCESS;
    bool counter_write_on_error = false;
    bool return_null_memory = false;
    const char* omit_proc = nullptr;
};

FakeVulkan* g_vk = nullptr;
const VkDevice k_device = make_handle<VkDevice>(0x100);

VKAPI_ATTR VkResult VKAPI_CALL fake_create_buffer(
    VkDevice, const VkBufferCreateInfo* info, const VkAllocationCallbacks*, VkBuffer* out) {
    ++g_vk->create_buffer_calls;
    if (g_vk->create_buffer_result != VK_SUCCESS) return g_vk->create_buffer_result;
    const std::uintptr_t id = g_vk->next_buffer++;
    g_vk->live_buffers[id] = info->size;
    *out = make_handle<VkBuffer>(id);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL fake_destroy_buffer(VkDevice, VkBuffer buffer, const VkAllocationCallbacks*) {
    ++g_vk->destroy_buffer_calls;
    g_vk->live_buffers.erase(reinterpret_cast<std::uintptr_t>(buffer));
}

VKAPI_ATTR void VKAPI_CALL fake_get_buffer_memory_requirements(
    VkDevice, VkBuffer buffer, VkMemoryRequirements* reqs) {
    const auto it = g_vk->live_buffers.find(reinterpret_cast<std::uintptr_t>(buffer));
    reqs->size = it != g_vk->live_buffers.end() ? it->second : 0;
    reqs->alignment = 256;
    reqs->memoryTypeBits = 1;
}

VKAPI_ATTR VkResult VKAPI_CALL fake_bind_buffer_memory(VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize) {
    return g_vk->bind_result;
}

VKAPI_ATTR void VKAPI_CALL fake_free_memory(VkDevice, VkDeviceMemory memory, const VkAllocationCallbacks*) {
    ++g_vk->free_memory_calls;
    g_vk->live_memory.erase(reinterpret_cast<std::uintptr_t>(memory));
}

VKAPI_ATTR VkResult VKAPI_CALL fake_get_semaphore_counter_value(
    VkDevice, VkSemaphore semaphore, std::uint64_t* value) {
    ++g_vk->get_counter_calls;
    const auto it = g_vk->semaphore_values.find(reinterpret_cast<std::uintptr_t>(semaphore));
    const std::uint64_t current = it != g_vk->semaphore_values.end() ? it->second : 0;
    if (g_vk->counter_result != VK_SUCCESS) {
        if (g_vk->counter_write_on_error) *value = current;
        return g_vk->counter_result;
    }
    *value = current;
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL fake_device_wait_idle(VkDevice) {
    ++g_vk->wait_idle_calls;
    return VK_SUCCESS;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL fake_get_device_proc(VkDevice, const char* name) {
    if (g_vk->omit_proc && std::strcmp(name, g_vk->omit_proc) == 0) return nullptr;
    if (std::strcmp(name, "vkCreateBuffer") == 0) return reinterpret_cast<PFN_vkVoidFunction>(fake_create_buffer);
    if (std::strcmp(name, "vkDestroyBuffer") == 0) return reinterpret_cast<PFN_vkVoidFunction>(fake_destroy_buffer);
    if (std::strcmp(name, "vkGetBufferMemoryRequirements") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(fake_get_buffer_memory_requirements);
    if (std::strcmp(name, "vkBindBufferMemory") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(fake_bind_buffer_memory);
    if (std::strcmp(name, "vkFreeMemory") == 0) return reinterpret_cast<PFN_vkVoidFunction>(fake_free_memory);
    if (std::strcmp(name, "vkGetSemaphoreCounterValue") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(fake_get_semaphore_counter_value);
    if (std::strcmp(name, "vkDeviceWaitIdle") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(fake_device_wait_idle);
    return nullptr;
}

VkDeviceMemory fake_allocate(const VkMemoryRequirements&) {
    ++g_vk->allocate_calls;
    if (g_vk->return_null_memory) return VK_NULL_HANDLE;
    const std::uintptr_t id = g_vk->next_memory++;
    g_vk->live_memory.insert(id);
    return make_handle<VkDeviceMemory>(id);
}

}  // namespace

int main() {
    // Missing any required proc leaves loaded() false, refuses get(), and skips vkDeviceWaitIdle in ~ScratchBuffers().
    for (const char* missing : {"vkCreateBuffer", "vkGetSemaphoreCounterValue", "vkDeviceWaitIdle"}) {
        FakeVulkan vk;
        vk.omit_proc = missing;
        g_vk = &vk;
        {
            ScratchBuffers scratch(fake_get_device_proc, k_device, fake_allocate);
            EXPECT(!scratch.loaded());
            VkBuffer buf = VK_NULL_HANDLE;
            VkDeviceSize off = 123;
            EXPECT(!scratch.get(256, buf, off));
            EXPECT(buf == VK_NULL_HANDLE && off == 123);
        }
        EXPECT(vk.wait_idle_calls == 0 && vk.create_buffer_calls == 0);
    }

    // Zero-byte request fails without allocating; sub-allocations align to 256 bytes inside a 4 MiB minimum chunk.
    {
        FakeVulkan vk;
        g_vk = &vk;
        constexpr VkDeviceSize k_4mib = 4u << 20;
        {
            ScratchBuffers scratch(fake_get_device_proc, k_device, fake_allocate);
            EXPECT(scratch.loaded());

            VkBuffer buf = VK_NULL_HANDLE;
            VkDeviceSize off = 0;
            EXPECT(!scratch.get(0, buf, off));
            EXPECT(vk.create_buffer_calls == 0);

            EXPECT(scratch.get(100, buf, off));
            const VkBuffer chunk1 = buf;
            EXPECT(chunk1 != VK_NULL_HANDLE && off == 0);
            EXPECT(vk.live_buffers[reinterpret_cast<std::uintptr_t>(chunk1)] == k_4mib);

            EXPECT(scratch.get(12, buf, off));
            EXPECT(buf == chunk1 && off == 256);

            EXPECT(scratch.get(256, buf, off));
            EXPECT(buf == chunk1 && off == 512);

            // Exact remaining space in chunk1 (used == 768 -> aligned start 768) fits without a new chunk.
            EXPECT(scratch.get(k_4mib - 768, buf, off));
            EXPECT(buf == chunk1 && off == 768);
            EXPECT(vk.create_buffer_calls == 1 && vk.allocate_calls == 1);

            // One more byte spills into a second 4 MiB chunk.
            EXPECT(scratch.get(1, buf, off));
            EXPECT(buf != chunk1 && off == 0);
            EXPECT(vk.create_buffer_calls == 2 && vk.allocate_calls == 2);

            // Requests larger than 4 MiB allocate a chunk of the requested size.
            constexpr VkDeviceSize k_6mib = 6u << 20;
            EXPECT(scratch.get(k_6mib, buf, off));
            EXPECT(off == 0);
            EXPECT(vk.live_buffers[reinterpret_cast<std::uintptr_t>(buf)] == k_6mib);
            EXPECT(vk.live_buffers.size() == 3 && vk.live_memory.size() == 3);
        }
        EXPECT(vk.wait_idle_calls == 1);
        EXPECT(vk.live_buffers.empty() && vk.live_memory.empty());
    }

    // Allocation and bind failure paths clean up partial buffer/memory handles without leaking.
    {
        FakeVulkan vk;
        g_vk = &vk;
        {
            ScratchBuffers scratch(fake_get_device_proc, k_device, fake_allocate);
            VkBuffer buf = VK_NULL_HANDLE;
            VkDeviceSize off = 0;

            vk.create_buffer_result = VK_ERROR_OUT_OF_DEVICE_MEMORY;
            EXPECT(!scratch.get(256, buf, off));
            EXPECT(vk.allocate_calls == 0 && vk.live_buffers.empty() && vk.live_memory.empty());

            vk.create_buffer_result = VK_SUCCESS;
            vk.return_null_memory = true;
            EXPECT(!scratch.get(256, buf, off));
            EXPECT(vk.destroy_buffer_calls == 1 && vk.live_buffers.empty() && vk.live_memory.empty());

            vk.return_null_memory = false;
            vk.bind_result = VK_ERROR_OUT_OF_DEVICE_MEMORY;
            EXPECT(!scratch.get(256, buf, off));
            EXPECT(vk.destroy_buffer_calls == 2 && vk.free_memory_calls == 1);
            EXPECT(vk.live_buffers.empty() && vk.live_memory.empty());
        }
        {
            ScratchBuffers no_alloc(fake_get_device_proc, k_device, nullptr);
            VkBuffer buf = VK_NULL_HANDLE;
            VkDeviceSize off = 0;
            EXPECT(!no_alloc.get(256, buf, off));
            EXPECT(vk.live_buffers.empty() && vk.live_memory.empty());
        }
    }

    // retire/collect: ticket 0 collects immediately, semaphore errors/incomplete tickets stay retired, and spare caps at 4.
    {
        FakeVulkan vk;
        g_vk = &vk;
        const VkSemaphore sem_a = make_handle<VkSemaphore>(0x201);
        const VkSemaphore sem_b = make_handle<VkSemaphore>(0x202);
        constexpr VkDeviceSize k_5mib = 5u << 20;
        constexpr VkDeviceSize k_6mib = 6u << 20;
        {
            ScratchBuffers scratch(fake_get_device_proc, k_device, fake_allocate);
            VkBuffer buf = VK_NULL_HANDLE;
            VkDeviceSize off = 0;

            // Empty retire() is a no-op; ticket 0 collects without a timeline semaphore.
            scratch.retire(sem_a, 1);
            EXPECT(scratch.get(256, buf, off));
            const VkBuffer reused_4mib = buf;
            scratch.retire(VK_NULL_HANDLE, 0);
            scratch.collect();
            EXPECT(vk.get_counter_calls == 0);
            EXPECT(vk.live_buffers.size() == 1);

            // Reusing from spare_ makes no new vkCreateBuffer or allocate call.
            EXPECT(scratch.get(512, buf, off));
            EXPECT(buf == reused_4mib && off == 0 && vk.create_buffer_calls == 1);

            // Retire with ticket > 0 and null timeline cannot be collected.
            scratch.retire(VK_NULL_HANDLE, 5);
            scratch.collect();
            EXPECT(vk.live_buffers.size() == 1);

            // Non-VK_SUCCESS from vkGetSemaphoreCounterValue keeps the batch retired even if value >= ticket.
            EXPECT(scratch.get(k_6mib, buf, off));
            const VkBuffer big_buf = buf;
            scratch.retire(sem_a, 10);
            vk.semaphore_values[reinterpret_cast<std::uintptr_t>(sem_a)] = 10;
            vk.counter_result = VK_ERROR_DEVICE_LOST;
            vk.counter_write_on_error = true;
            scratch.collect();
            EXPECT(vk.live_buffers.size() == 2);

            // Once counter succeeds and reaches ticket, big_buf enters spare_ and satisfies a 5 MiB request directly.
            vk.counter_result = VK_SUCCESS;
            scratch.collect();
            EXPECT(scratch.get(k_5mib, buf, off));
            EXPECT(buf == big_buf && off == 0 && vk.create_buffer_calls == 2);
            scratch.retire(sem_b, 20);

            // Create and retire 5 distinct 4 MiB chunks; collect() keeps 4 in spare_ and destroys the 5th.
            for (int i = 0; i < 5; ++i) {
                EXPECT(scratch.get(256, buf, off));
                scratch.retire(sem_a, 10 + static_cast<std::uint64_t>(i) + 1);
            }
            vk.semaphore_values[reinterpret_cast<std::uintptr_t>(sem_a)] = 20;
            // sem_b is still at 0 (< 20), so big_buf stays retired while the 5 sem_a chunks are collected.
            scratch.collect();
            EXPECT(vk.destroy_buffer_calls == 1 && vk.free_memory_calls == 1);
            // Live buffers: 1 null-timeline + 1 sem_b (retired) + 4 spare = 6.
            EXPECT(vk.live_buffers.size() == 6);

            // forget(sem_b) frees only sem_b's retired batch and leaves the null-timeline batch and 4 spares intact.
            scratch.forget(sem_b);
            EXPECT(vk.live_buffers.find(reinterpret_cast<std::uintptr_t>(big_buf)) == vk.live_buffers.end());
            EXPECT(vk.live_buffers.size() == 5);
        }
        EXPECT(vk.live_buffers.empty() && vk.live_memory.empty());
    }

    std::printf("vulkan_scratch: all checks passed\n");
    return 0;
}
