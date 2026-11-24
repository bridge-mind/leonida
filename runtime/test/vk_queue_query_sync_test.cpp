// VulkanQueue / VulkanContext query-state synchronization and failed-submit wait preservation regressions
// (runtime/src/vulkan_backend.h):
//   1. VulkanContext::remove_query_pool() must serialize with tracker_mutex_ so an in-flight VulkanQueue::execute()
//      that already unlocked resource_mutex_ cannot suffer a use-after-free in QueryReplay::commit().
//   2. VulkanQueue::execute() must acquire tracker_mutex_ before QueryReplay::plan() so concurrent queues see
//      submission-order query state (no data race on QueryPool::ended / QueryPool::occlusion and no stale zero-fill).
//   3. A transient vkQueueSubmit failure must preserve pending VulkanQueue::wait() timeline semaphores for the retry.
//   4. Concurrent multi-queue query End/Resolve + add/remove_query_pool stress with zero Vulkan validation errors.
// Skips with exit code 77 when no Vulkan 1.3 device is available. No game data.
#include "vk_test_device.h"
#include "../src/vulkan_backend.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string_view>
#include <thread>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

namespace driver {
PFN_vkGetDeviceProcAddr real_get_device = nullptr;
PFN_vkQueueSubmit real_submit = nullptr;
PFN_vkGetSemaphoreCounterValue real_counter = nullptr;
PFN_vkCmdCopyQueryPoolResults real_copy_queries = nullptr;
PFN_vkCmdFillBuffer real_fill_buffer = nullptr;

std::atomic<VkResult> fail_next_submit{VK_SUCCESS};
std::atomic<VkResult> fail_next_counter{VK_SUCCESS};
std::atomic<unsigned> injected_counter_errors{0};
std::thread::id counter_test_thread;
std::atomic<bool> capture_next_waits{false};
std::atomic<std::uint32_t> captured_wait_count{0};
std::atomic<std::uintptr_t> captured_wait_sem{0};
std::atomic<std::uint64_t> captured_wait_val{0};

std::atomic<bool> count_resolves{false};
std::atomic<std::uint32_t> copy_query_calls{0};
std::atomic<std::uint32_t> fill_buffer_calls{0};

// Optional one-shot callback invoked inside vkQueueSubmit right before calling the real driver.
// Returning non-VK_SUCCESS short-circuits the submit with that status.
std::function<VkResult()> on_next_submit;

VKAPI_ATTR VkResult VKAPI_CALL queue_submit(VkQueue queue, std::uint32_t count, const VkSubmitInfo* submits,
                                            VkFence fence) {
    if (on_next_submit) {
        auto cb = std::move(on_next_submit);
        on_next_submit = {};
        const VkResult override_res = cb();
        if (override_res != VK_SUCCESS) return override_res;
    }
    const VkResult armed = fail_next_submit.exchange(VK_SUCCESS);
    if (armed != VK_SUCCESS) return armed;
    if (capture_next_waits.exchange(false) && count > 0 && submits) {
        captured_wait_count.store(submits[0].waitSemaphoreCount);
        if (submits[0].waitSemaphoreCount > 0 && submits[0].pWaitSemaphores) {
            captured_wait_sem.store(reinterpret_cast<std::uintptr_t>(submits[0].pWaitSemaphores[0]));
            const auto* timeline = reinterpret_cast<const VkTimelineSemaphoreSubmitInfo*>(submits[0].pNext);
            if (timeline && timeline->sType == VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO &&
                timeline->waitSemaphoreValueCount > 0 && timeline->pWaitSemaphoreValues) {
                captured_wait_val.store(timeline->pWaitSemaphoreValues[0]);
            }
        }
    }
    return real_submit(queue, count, submits, fence);
}

VKAPI_ATTR VkResult VKAPI_CALL get_counter(VkDevice device, VkSemaphore semaphore, std::uint64_t* value) {
    if (std::this_thread::get_id() == counter_test_thread) {
        const VkResult result = fail_next_counter.exchange(VK_SUCCESS);
        if (result != VK_SUCCESS) {
            *value = 999;  // failure output is undefined, so make accidental reuse detectable
            injected_counter_errors.fetch_add(1);
            return result;
        }
    }
    return real_counter(device, semaphore, value);
}

VKAPI_ATTR void VKAPI_CALL cmd_copy_query_pool_results(VkCommandBuffer cb, VkQueryPool pool, std::uint32_t first,
                                                       std::uint32_t count, VkBuffer dst, VkDeviceSize offset,
                                                       VkDeviceSize stride, VkQueryResultFlags flags) {
    if (count_resolves.load(std::memory_order_relaxed)) copy_query_calls.fetch_add(1, std::memory_order_relaxed);
    real_copy_queries(cb, pool, first, count, dst, offset, stride, flags);
}

VKAPI_ATTR void VKAPI_CALL cmd_fill_buffer(VkCommandBuffer cb, VkBuffer dst, VkDeviceSize offset, VkDeviceSize size,
                                           std::uint32_t data) {
    if (count_resolves.load(std::memory_order_relaxed)) fill_buffer_calls.fetch_add(1, std::memory_order_relaxed);
    real_fill_buffer(cb, dst, offset, size, data);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_device(VkDevice device, const char* name) {
    const std::string_view n = name;
    if (n == "vkQueueSubmit") return reinterpret_cast<PFN_vkVoidFunction>(queue_submit);
    if (n == "vkGetSemaphoreCounterValue") return reinterpret_cast<PFN_vkVoidFunction>(get_counter);
    if (n == "vkCmdCopyQueryPoolResults") return reinterpret_cast<PFN_vkVoidFunction>(cmd_copy_query_pool_results);
    if (n == "vkCmdFillBuffer") return reinterpret_cast<PFN_vkVoidFunction>(cmd_fill_buffer);
    return real_get_device(device, name);
}

void install(PFN_vkGetDeviceProcAddr device_proc, VkDevice device) {
    real_get_device = device_proc;
    real_submit = reinterpret_cast<PFN_vkQueueSubmit>(device_proc(device, "vkQueueSubmit"));
    real_counter = reinterpret_cast<PFN_vkGetSemaphoreCounterValue>(
        device_proc(device, "vkGetSemaphoreCounterValue"));
    real_copy_queries =
        reinterpret_cast<PFN_vkCmdCopyQueryPoolResults>(device_proc(device, "vkCmdCopyQueryPoolResults"));
    real_fill_buffer = reinterpret_cast<PFN_vkCmdFillBuffer>(device_proc(device, "vkCmdFillBuffer"));
}
}  // namespace driver

}  // namespace

int main() {
    using namespace leonida;

    vktest::Device d;
    if (!d.create(false)) {
        std::fprintf(stderr, "vk_queue_query_sync_test: no Vulkan 1.3 device, skipped\n");
        return 77;
    }
    driver::install(d.device_proc, d.device);
    driver::counter_test_thread = std::this_thread::get_id();

    const auto create_query_pool = d.fn<PFN_vkCreateQueryPool>("vkCreateQueryPool");
    const auto destroy_query_pool = d.fn<PFN_vkDestroyQueryPool>("vkDestroyQueryPool");
    EXPECT(create_query_pool && destroy_query_pool);

    auto make_occlusion_pool = [&](std::uint32_t count) -> VkQueryPool {
        VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        info.queryType = VK_QUERY_TYPE_OCCLUSION;
        info.queryCount = count;
        VkQueryPool pool = VK_NULL_HANDLE;
        if (create_query_pool(d.device, &info, nullptr, &pool) != VK_SUCCESS) return VK_NULL_HANDLE;
        return pool;
    };

    const VkQueryPool pool_uaf = make_occlusion_pool(2);
    const VkQueryPool pool_race = make_occlusion_pool(2);
    const VkQueryPool pool_stress = make_occlusion_pool(32);
    EXPECT(pool_uaf && pool_race && pool_stress);
    bool progress_loss_ok = true;  // deferred verdict: release Vulkan query pools even when main is expected-red

    {
        VulkanContext context(driver::get_device, d.device, true);
        EXPECT(context.loaded());
        VulkanQueue queue_a(context, d.queue, d.family);
        VulkanQueue queue_b(context, d.queue, d.family);
        EXPECT(queue_a.ok() && queue_b.ok());

        void* src_ptr = nullptr;
        void* dst_ptr = nullptr;
        void* res_ptr = nullptr;
        const VkBuffer src_buf = d.buffer(64, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &src_ptr);
        const VkBuffer dst_buf = d.buffer(64, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &dst_ptr);
        const VkBuffer res_buf = d.buffer(64, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &res_ptr);
        EXPECT(src_buf && dst_buf && res_buf);

        ResourceBinding src_rb{};
        src_rb.buffer = src_buf;
        src_rb.size = 64;
        ResourceBinding dst_rb{};
        dst_rb.buffer = dst_buf;
        dst_rb.size = 64;
        dst_rb.host_readable = true;
        ResourceBinding res_rb{};
        res_rb.buffer = res_buf;
        res_rb.size = 64;
        res_rb.host_readable = true;
        context.add_resource(1, src_rb);
        context.add_resource(2, dst_rb);
        context.add_resource(3, res_rb);

        // 1. remove_query_pool() must block on tracker_mutex_ while execute() is between resource_lock.unlock() and
        //    QueryReplay::commit(), preventing a use-after-free on the QueryPool* stored in QueryReplay::pending_.
        {
            constexpr QueryHeapId heap_id = 77;
            QueryPool qp{};
            qp.pool = pool_uaf;
            qp.heap_type = d3d12_query_heap::occlusion;
            qp.count = 2;
            qp.ended.assign(2, false);
            context.add_query_pool(heap_id, qp);

            CommandList list;
            EXPECT(list.begin_query({heap_id, d3d12_query::occlusion, 0}));
            EXPECT(list.end_query({heap_id, d3d12_query::occlusion, 0}));
            EXPECT(list.close());
            const CommandList* ptr = &list;

            std::atomic<bool> removed{false};
            bool removed_during_submit = false;
            std::thread remover;
            driver::on_next_submit = [&]() -> VkResult {
                remover = std::thread([&] {
                    context.remove_query_pool(heap_id);
                    removed.store(true, std::memory_order_release);
                });
                std::this_thread::sleep_for(std::chrono::milliseconds(30));
                removed_during_submit = removed.load(std::memory_order_acquire);
                // If remove_query_pool already erased the node, abort the submit so queries.commit() does not
                // corrupt the heap before the EXPECT below reports the failure cleanly.
                return removed_during_submit ? VK_ERROR_OUT_OF_HOST_MEMORY : VK_SUCCESS;
            };
            const ReplayResult exec_res = queue_a.execute({&ptr, 1});
            if (remover.joinable()) remover.join();
            EXPECT(!removed_during_submit);
            EXPECT(exec_res == ReplayResult::success);
            EXPECT(removed.load(std::memory_order_acquire));
            EXPECT(context.query_heap_type(heap_id) == ~0u);
            EXPECT(queue_a.idle());
            std::printf("queue/query sync: remove_query_pool serialized with QueryReplay::commit ok\n");
        }

        // 2. Concurrent queue_b.execute(ResolveQueries) must not run QueryReplay::plan() before queue_a commits
        //    EndQuery; otherwise queue_b sees ended[0] == false and emits vkCmdFillBuffer(0) instead of copying.
        {
            constexpr QueryHeapId heap_id = 88;
            QueryPool qp{};
            qp.pool = pool_race;
            qp.heap_type = d3d12_query_heap::occlusion;
            qp.count = 2;
            qp.ended.assign(2, false);
            context.add_query_pool(heap_id, qp);

            CommandList end_list;
            EXPECT(end_list.begin_query({heap_id, d3d12_query::occlusion, 0}));
            EXPECT(end_list.end_query({heap_id, d3d12_query::occlusion, 0}));
            EXPECT(end_list.close());
            const CommandList* end_ptr = &end_list;

            CommandList resolve_list;
            EXPECT(resolve_list.resolve_queries({heap_id, d3d12_query::occlusion, 0, 1, 3, 0}));
            EXPECT(resolve_list.close());
            const CommandList* resolve_ptr = &resolve_list;

            std::memset(res_ptr, 0xAB, 64);
            driver::copy_query_calls.store(0);
            driver::fill_buffer_calls.store(0);
            driver::count_resolves.store(true);

            std::thread resolver;
            ReplayResult resolve_res = ReplayResult::unsupported;
            driver::on_next_submit = [&]() -> VkResult {
                resolver = std::thread([&] { resolve_res = queue_b.execute({&resolve_ptr, 1}); });
                // Give queue_b ample time to run prepare() and reach QueryReplay::plan() while queue_a holds
                // tracker_mutex_ (and has already unlocked resource_mutex_).
                std::this_thread::sleep_for(std::chrono::milliseconds(40));
                return VK_SUCCESS;
            };

            const ReplayResult end_res = queue_a.execute({&end_ptr, 1});
            if (resolver.joinable()) resolver.join();
            driver::count_resolves.store(false);
            EXPECT(end_res == ReplayResult::success);
            EXPECT(resolve_res == ReplayResult::success);
            EXPECT(queue_a.idle() && queue_b.idle());
            EXPECT(driver::copy_query_calls.load() == 1);
            EXPECT(driver::fill_buffer_calls.load() == 0);
            std::uint64_t resolved = 0xDEADBEEFULL;
            std::memcpy(&resolved, res_ptr, sizeof(resolved));
            EXPECT(resolved == 0);  // empty occlusion query resolves 0 via vkCmdCopyQueryPoolResults
            context.remove_query_pool(heap_id);
            std::printf("queue/query sync: cross-queue QueryReplay::plan serialized with commit ok\n");
        }

        // 3. A transient vkQueueSubmit failure must preserve pending VulkanQueue::wait() semaphores for the retry.
        {
            VulkanFence fence(context, 0);
            EXPECT(fence.semaphore() != VK_NULL_HANDLE);

            const std::uint64_t pattern = 0x1122334455667788ULL;
            std::memcpy(src_ptr, &pattern, sizeof(pattern));
            std::memset(dst_ptr, 0, sizeof(pattern));

            CommandList copy_list;
            EXPECT(copy_list.copy_buffer({2, 0, 1, 0, sizeof(pattern)}));
            EXPECT(copy_list.close());
            const CommandList* ptr = &copy_list;

            // The Signal comes first: a Wait with no submitted Signal holds the batch on the CPU (queue_holds.h) and
            // never reaches vkQueueSubmit, so it could not fail there.
            EXPECT(fence.signal(1));
            queue_a.wait(fence.semaphore(), 1);
            driver::fail_next_submit.store(VK_ERROR_OUT_OF_HOST_MEMORY);
            EXPECT(queue_a.execute({&ptr, 1}) == ReplayResult::unsupported);

            // Retry: the pending wait on (fence, 1) must still be attached to the next vkQueueSubmit.
            driver::captured_wait_count.store(0);
            driver::captured_wait_sem.store(0);
            driver::captured_wait_val.store(0);
            driver::capture_next_waits.store(true);
            EXPECT(queue_a.execute({&ptr, 1}) == ReplayResult::success);
            EXPECT(queue_a.idle());
            EXPECT(driver::captured_wait_count.load() == 1);
            EXPECT(driver::captured_wait_sem.load() == reinterpret_cast<std::uintptr_t>(fence.semaphore()));
            EXPECT(driver::captured_wait_val.load() == 1);

            std::uint64_t got = 0;
            std::memcpy(&got, dst_ptr, sizeof(got));
            EXPECT(got == pattern);
            std::printf("queue/query sync: pending Wait() preserved across failed vkQueueSubmit ok\n");
        }

        // 3b. The same for a batch held behind a Wait with no submitted Signal (queue_holds.h): its submission
        //     fails when the CPU Signal releases it, the failure is reported through take_held_failure(), and its
        //     waits stay for the next submission, as when the call is not held. A Signal queued behind the failed
        //     batch still runs.
        {
            VulkanFence gate(context, 0), after(context, 0);
            EXPECT(gate.semaphore() != VK_NULL_HANDLE && after.semaphore() != VK_NULL_HANDLE);
            const std::uint64_t pattern = 0x0badf00dcafe1234ULL;
            std::memcpy(src_ptr, &pattern, sizeof(pattern));
            std::memset(dst_ptr, 0, sizeof(pattern));
            CommandList copy_list;
            EXPECT(copy_list.copy_buffer({2, 0, 1, 0, sizeof(pattern)}));
            EXPECT(copy_list.close());
            const CommandList* ptr = &copy_list;

            queue_a.wait(gate.semaphore(), 1);
            EXPECT(queue_a.execute({&ptr, 1}) == ReplayResult::success);   // held: the Signal is not submitted yet
            EXPECT(queue_a.signal(after.semaphore(), 1));                  // held behind it
            EXPECT(after.completed() == 0 && queue_a.take_held_failure() == ReplayResult::success);
            driver::fail_next_submit.store(VK_ERROR_OUT_OF_HOST_MEMORY);
            EXPECT(gate.signal(1));                                        // releases the batch on this thread
            EXPECT(queue_a.take_held_failure() == ReplayResult::unsupported);
            EXPECT(queue_a.take_held_failure() == ReplayResult::success);  // reported once
            EXPECT(queue_a.idle() && after.completed() == 1);              // the Signal behind it still signaled

            driver::captured_wait_count.store(0);
            driver::captured_wait_sem.store(0);
            driver::captured_wait_val.store(0);
            driver::capture_next_waits.store(true);
            EXPECT(queue_a.execute({&ptr, 1}) == ReplayResult::success);
            EXPECT(queue_a.idle());
            EXPECT(driver::captured_wait_count.load() == 1);   // the failed batch's wait on (gate, 1) came along
            EXPECT(driver::captured_wait_sem.load() == reinterpret_cast<std::uintptr_t>(gate.semaphore()));
            EXPECT(driver::captured_wait_val.load() == 1);
            std::uint64_t got = 0;
            std::memcpy(&got, dst_ptr, sizeof(got));
            EXPECT(got == pattern);
            std::printf("queue/query sync: failed held batch reported, its wait kept for the retry, Signal behind it ran\n");
        }

        // 3c. A Signal whose vkQueueSubmit fails signals nothing: a queue waiting for it stays held (the value is not
        //     counted as submitted), and the application's retry of the Signal goes through and releases it.
        {
            VulkanFence fence(context, 0);
            EXPECT(fence.semaphore() != VK_NULL_HANDLE);
            const std::uint64_t pattern = 0x5ca1ab1e0ddba11ULL;
            std::memcpy(src_ptr, &pattern, sizeof(pattern));
            std::memset(dst_ptr, 0, sizeof(pattern));
            CommandList copy_list;
            EXPECT(copy_list.copy_buffer({2, 0, 1, 0, sizeof(pattern)}));
            EXPECT(copy_list.close());
            const CommandList* ptr = &copy_list;
            queue_b.wait(fence.semaphore(), 1);
            EXPECT(queue_b.execute({&ptr, 1}) == ReplayResult::success);  // held for the Signal
            driver::fail_next_submit.store(VK_ERROR_OUT_OF_HOST_MEMORY);
            EXPECT(!queue_a.signal(fence.semaphore(), 1));                     // the submit fails
            EXPECT(queue_b.idle() && fence.completed() == 0);
            std::uint64_t got = 1;
            std::memcpy(&got, dst_ptr, sizeof(got));
            EXPECT(got == 0);                                                  // the waiter did not run
            EXPECT(queue_a.signal(fence.semaphore(), 1));                      // the retry signals, and releases it
            EXPECT(queue_b.idle() && queue_a.idle() && fence.completed() == 1);
            std::memcpy(&got, dst_ptr, sizeof(got));
            EXPECT(got == pattern);
            std::printf("queue/query sync: a failed Signal does not release its waiters; the retry does\n");
        }

        // 4. Concurrent multi-queue EndQuery + ResolveQueries + temporary query pool add/remove stress.
        //    Each iteration uses a distinct query slot so no slot is reset while another submission resolves it.
        {
            constexpr QueryHeapId main_heap = 99;
            QueryPool qp{};
            qp.pool = pool_stress;
            qp.heap_type = d3d12_query_heap::occlusion;
            qp.count = 32;
            qp.ended.assign(32, false);
            context.add_query_pool(main_heap, qp);

            CommandList res_list;
            EXPECT(res_list.resolve_queries({main_heap, d3d12_query::occlusion, 0, 1, 3, 0}));
            EXPECT(res_list.close());
            const CommandList* res_ptr_cmd = &res_list;

            std::atomic<bool> stop{false};
            std::atomic<bool> worker_err{false};
            std::thread worker([&] {
                for (std::uint64_t round = 0; !stop.load(std::memory_order_relaxed); ++round) {
                    if (queue_b.execute({&res_ptr_cmd, 1}) != ReplayResult::success || !queue_b.idle()) {
                        worker_err.store(true);
                        break;
                    }
                    const QueryHeapId temp_id = 1000 + (round % 16);
                    context.add_query_pool(temp_id, qp);
                    context.remove_query_pool(temp_id);
                }
            });

            ReplayResult main_err = ReplayResult::success;
            for (std::uint32_t i = 0; i < 32; ++i) {
                CommandList end_list;
                if (!end_list.begin_query({main_heap, d3d12_query::occlusion, i}) ||
                    !end_list.end_query({main_heap, d3d12_query::occlusion, i}) || !end_list.close()) {
                    main_err = ReplayResult::invalid_query;
                    break;
                }
                const CommandList* end_ptr = &end_list;
                main_err = queue_a.execute({&end_ptr, 1});
                if (main_err != ReplayResult::success) break;
            }
            stop.store(true, std::memory_order_relaxed);
            worker.join();
            EXPECT(main_err == ReplayResult::success);
            EXPECT(!worker_err.load());
            EXPECT(queue_a.idle() && queue_b.idle());
            context.remove_query_pool(main_heap);
            std::printf("queue/query sync: concurrent multi-queue query stress ok\n");
        }

        // 5. A progress-semaphore query in acquire() can be the first Vulkan operation to discover
        // device loss. A transient query failure must not poison a healthy queue, while DEVICE_LOST
        // must prevent acquiring/recycling and submitting new command buffers.
        {
            EXPECT(!context.is_device_lost());
            CommandList empty;
            EXPECT(empty.close());
            const CommandList* ptr = &empty;
            EXPECT(queue_a.idle() && queue_b.idle());
            (void)context.completed_retire_ticket();  // drain old submits before faulting queue's own progress query
            driver::fail_next_counter.store(VK_ERROR_OUT_OF_HOST_MEMORY);
            const ReplayResult transient = queue_a.execute({&ptr, 1});
            const bool transient_ok = transient == ReplayResult::success && !context.is_device_lost() &&
                                      driver::injected_counter_errors.load() == 1;

            EXPECT(queue_a.idle());
            (void)context.completed_retire_ticket();
            driver::fail_next_counter.store(VK_ERROR_DEVICE_LOST);
            const ReplayResult after_loss = queue_a.execute({&ptr, 1});
            const bool lost = context.is_device_lost();
            std::printf("queue/query sync: injected nonfatal execute=%d, device-loss execute=%d, "
                        "context_lost=%d, interceptions=%u\n", int(transient), int(after_loss), int(lost),
                        driver::injected_counter_errors.load());
            progress_loss_ok = transient_ok && after_loss == ReplayResult::unsupported && lost &&
                               driver::injected_counter_errors.load() == 2;
            std::printf("queue/query sync: progress counter device loss %s\n",
                        progress_loss_ok ? "marks context and rejects work" : "NOT propagated");
        }
    }

    destroy_query_pool(d.device, pool_uaf, nullptr);
    destroy_query_pool(d.device, pool_race, nullptr);
    destroy_query_pool(d.device, pool_stress, nullptr);

    EXPECT(d.validation_errors() == 0);
    EXPECT(progress_loss_ok);
    std::printf("vk_queue_query_sync_test: ok\n");
    return 0;
}
