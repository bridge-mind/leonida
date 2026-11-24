// Multi-threaded D3D12-on-Vulkan concurrency regression test:
// - Two worker threads submit concurrently on two independent ID3D12CommandQueues (32 iterations each),
//   clearing render targets to iteration-unique colours, transitioning layouts, copying to readback buffers,
//   waiting via ID3D12Fence::SetEventOnCompletion, and verifying every readback texel.
// - A third thread concurrently creates and destroys temporary buffers/textures on the shared ID3D12Device11
//   while both queues are submitting.
// - Verifies ID3D12Device1::SetEventOnMultipleFenceCompletion (ALL and ANY) across both queues' fences.
// Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation; any validation error fails the test.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

namespace {

template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() {
        if (p) p->Release();
    }
    Ref() = default;
    Ref(const Ref&) = delete;
    Ref& operator=(const Ref&) = delete;
    T* operator->() const { return p; }
    void** out() { return reinterpret_cast<void**>(&p); }
};

struct EventHandle {
    HANDLE h = nullptr;
    EventHandle() : h(CreateEventW(nullptr, FALSE, FALSE, nullptr)) {}
    ~EventHandle() {
        if (h) CloseHandle(h);
    }
    EventHandle(const EventHandle&) = delete;
    EventHandle& operator=(const EventHandle&) = delete;
};

int fail(const char* msg, int thread_id = -1, int iter = -1) {
    if (thread_id >= 0)
        std::fprintf(stderr, "FAIL [thread %d iter %d]: %s\n", thread_id, iter, msg);
    else
        std::fprintf(stderr, "FAIL: %s\n", msg);
    return 1;
}

}  // namespace

int main() {
    constexpr int kIterations = 32;
    {
        vktest::Device d;
        const auto status = d.initialize();
        if (status == vktest::DeviceStatus::unavailable) return 77;  // no Vulkan GPU: ctest skips
        if (status != vktest::DeviceStatus::ready) return fail("vktest::Device::initialize");

        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = d.instance_proc;
        info.instance = d.instance;
        info.physical = d.physical;
        info.device = d.device;
        info.queue = d.queue;
        info.queue_family = d.family;
        info.separate_depth_stencil_layouts = true;

        {
            Ref<ID3D12Device1> device;
            if (FAILED(rt::d3d12::create_vulkan_device(info, __uuidof(ID3D12Device1), device.out())) || !device.p)
                return fail("create_vulkan_device");

            constexpr UINT kSize = 4;
        std::atomic<int> failures{0};
        std::atomic<bool> stop_churn{false};
        std::atomic<int> ready{0};

        // Create two independent queues and fences up front so we can also test multi-fence completion afterward.
        D3D12_COMMAND_QUEUE_DESC qd{D3D12_COMMAND_LIST_TYPE_DIRECT, 0, D3D12_COMMAND_QUEUE_FLAG_NONE, 0};
        Ref<ID3D12CommandQueue> queues[2];
        Ref<ID3D12Fence> fences[2];
        UINT64 fence_vals[2] = {0, 0};
        for (int i = 0; i < 2; ++i) {
            if (FAILED(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queues[i].p))) || !queues[i].p)
                return fail("CreateCommandQueue");
            if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fences[i].p))) || !fences[i].p)
                return fail("CreateFence");
    }

    auto queue_worker = [&](int tid) {
        Ref<ID3D12CommandAllocator> alloc;
        Ref<ID3D12GraphicsCommandList> cl;
        Ref<ID3D12DescriptorHeap> rtv_heap;
        Ref<ID3D12Resource> rt;
        Ref<ID3D12Resource> rb;
        EventHandle ev;
        if (!ev.h) {
            failures.fetch_add(1);
            fail("CreateEventW", tid);
            return;
        }
        if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc.p))) ||
            FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.p, nullptr,
                                             IID_PPV_ARGS(&cl.p)))) {
            failures.fetch_add(1);
            fail("CreateCommandAllocator/List", tid);
            return;
        }
        cl->Close();

        D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        if (FAILED(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&rtv_heap.p)))) {
            failures.fetch_add(1);
            fail("CreateDescriptorHeap", tid);
            return;
        }

        D3D12_HEAP_PROPERTIES def_hp{D3D12_HEAP_TYPE_DEFAULT};
        D3D12_RESOURCE_DESC rt_desc{};
        rt_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        rt_desc.Width = kSize;
        rt_desc.Height = kSize;
        rt_desc.DepthOrArraySize = 1;
        rt_desc.MipLevels = 1;
        rt_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        rt_desc.SampleDesc.Count = 1;
        rt_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        if (FAILED(device->CreateCommittedResource(&def_hp, D3D12_HEAP_FLAG_NONE, &rt_desc,
                                                   D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&rt.p)))) {
            failures.fetch_add(1);
            fail("CreateCommittedResource RT", tid);
            return;
        }

        const D3D12_CPU_DESCRIPTOR_HANDLE rtv = rtv_heap->GetCPUDescriptorHandleForHeapStart();
        device->CreateRenderTargetView(rt.p, nullptr, rtv);

        D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
        UINT64 total_bytes = 0;
        device->GetCopyableFootprints(&rt_desc, 0, 1, 0, &fp, nullptr, nullptr, &total_bytes);

        D3D12_HEAP_PROPERTIES rb_hp{D3D12_HEAP_TYPE_READBACK};
        D3D12_RESOURCE_DESC rb_desc{};
        rb_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rb_desc.Width = total_bytes;
        rb_desc.Height = 1;
        rb_desc.DepthOrArraySize = 1;
        rb_desc.MipLevels = 1;
        rb_desc.SampleDesc.Count = 1;
        rb_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        if (FAILED(device->CreateCommittedResource(&rb_hp, D3D12_HEAP_FLAG_NONE, &rb_desc,
                                                   D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&rb.p)))) {
            failures.fetch_add(1);
            fail("CreateCommittedResource RB", tid);
            return;
        }

        // Wait until both queue workers are ready so submissions overlap maximally.
        ready.fetch_add(1);
        while (ready.load() < 2) std::this_thread::yield();

        D3D12_RESOURCE_STATES prev_state = D3D12_RESOURCE_STATE_COMMON;
        for (int it = 0; it < kIterations; ++it) {
            if (FAILED(alloc->Reset()) || FAILED(cl->Reset(alloc.p, nullptr))) {
                failures.fetch_add(1);
                fail("Reset", tid, it);
                return;
            }
            D3D12_RESOURCE_BARRIER b0{};
            b0.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            b0.Transition = {rt.p, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, prev_state,
                             D3D12_RESOURCE_STATE_RENDER_TARGET};
            cl->ResourceBarrier(1, &b0);

            const std::uint8_t exp_r = static_cast<std::uint8_t>((tid * 100 + it * 7) & 0xFF);
            const std::uint8_t exp_g = static_cast<std::uint8_t>((50 + it * 5) & 0xFF);
            const std::uint8_t exp_b = static_cast<std::uint8_t>((200 - tid * 80 - it * 3) & 0xFF);
            const float color[4] = {exp_r / 255.0f, exp_g / 255.0f, exp_b / 255.0f, 1.0f};
            cl->ClearRenderTargetView(rtv, color, 0, nullptr);

            D3D12_RESOURCE_BARRIER b1{};
            b1.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            b1.Transition = {rt.p, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_RENDER_TARGET,
                             D3D12_RESOURCE_STATE_COPY_SOURCE};
            cl->ResourceBarrier(1, &b1);

            D3D12_TEXTURE_COPY_LOCATION src_loc{rt.p, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {0}};
            D3D12_TEXTURE_COPY_LOCATION dst_loc{rb.p, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
            dst_loc.PlacedFootprint = fp;
            cl->CopyTextureRegion(&dst_loc, 0, 0, 0, &src_loc, nullptr);
            cl->Close();
            prev_state = D3D12_RESOURCE_STATE_COPY_SOURCE;

            ID3D12CommandList* lists[1] = {cl.p};
            queues[tid]->ExecuteCommandLists(1, lists);
            const UINT64 target = ++fence_vals[tid];
            // 60 s, not 15 s: under heavy parallel load (two full `ctest -j 6` suites sharing the GPU) a fence can
            // take longer than 15 s to signal; sibling GPU tests already wait 15-60 s (TEST-514 follow-up).
            if (FAILED(queues[tid]->Signal(fences[tid].p, target)) ||
                FAILED(fences[tid]->SetEventOnCompletion(target, ev.h)) ||
                WaitForSingleObject(ev.h, 60000) != WAIT_OBJECT_0) {
                failures.fetch_add(1);
                fail("Signal/SetEventOnCompletion/Wait", tid, it);
                return;
            }

            void* mapped = nullptr;
            if (FAILED(rb->Map(0, nullptr, &mapped)) || !mapped) {
                failures.fetch_add(1);
                fail("Map", tid, it);
                return;
            }
            const auto* base = static_cast<const std::uint8_t*>(mapped);
            bool bad = false;
            for (UINT y = 0; y < kSize && !bad; ++y) {
                const auto* row = base + static_cast<std::size_t>(y) * fp.Footprint.RowPitch;
                for (UINT x = 0; x < kSize; ++x) {
                    const auto* px = row + static_cast<std::size_t>(x) * 4;
                    if (std::abs(int(px[0]) - int(exp_r)) > 1 || std::abs(int(px[1]) - int(exp_g)) > 1 ||
                        std::abs(int(px[2]) - int(exp_b)) > 1 || px[3] != 255) {
                        bad = true;
                        break;
                    }
                }
            }
            rb->Unmap(0, nullptr);
            if (bad) {
                failures.fetch_add(1);
                fail("readback texel mismatch", tid, it);
                return;
            }
        }
    };

    // Third thread creates and releases temporary buffers while the two queues submit concurrently.
    auto resource_churner = [&]() {
        D3D12_HEAP_PROPERTIES up_hp{D3D12_HEAP_TYPE_UPLOAD};
        D3D12_RESOURCE_DESC bd{};
        bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        bd.Width = 4096;
        bd.Height = 1;
        bd.DepthOrArraySize = 1;
        bd.MipLevels = 1;
        bd.SampleDesc.Count = 1;
        bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        while (!stop_churn.load(std::memory_order_acquire)) {
            Ref<ID3D12Resource> tmp;
            if (FAILED(device->CreateCommittedResource(&up_hp, D3D12_HEAP_FLAG_NONE, &bd,
                                                       D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                       IID_PPV_ARGS(&tmp.p)))) {
                failures.fetch_add(1);
                fail("churn CreateCommittedResource", 2);
                return;
            }
        }
    };

    std::thread t0(queue_worker, 0);
    std::thread t1(queue_worker, 1);
    std::thread t2(resource_churner);
    t0.join();
    t1.join();
    stop_churn.store(true, std::memory_order_release);
    t2.join();

    if (failures.load() != 0) return 1;

        // Cross-queue shared-resource handoff: queue 0 clears a shared texture and signals fence 0;
        // queue 1 Wait()s on fence 0, transitions the SAME texture to COPY_SOURCE, copies to readback,
        // and verifies every texel; then repeats with a second colour.
        {
            Ref<ID3D12CommandAllocator> alloc0, alloc1;
            Ref<ID3D12GraphicsCommandList> cl0, cl1;
            Ref<ID3D12DescriptorHeap> shared_rtv_heap;
            Ref<ID3D12Resource> shared_rt, shared_rb;
            EventHandle handoff_ev;
            if (!handoff_ev.h) return fail("CreateEventW (handoff)");
            if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc0.p))) ||
                FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc1.p))) ||
                FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc0.p, nullptr,
                                                 IID_PPV_ARGS(&cl0.p))) ||
                FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc1.p, nullptr,
                                                 IID_PPV_ARGS(&cl1.p))))
                return fail("CreateCommandAllocator/List (handoff)");
            cl0->Close();
            cl1->Close();

            D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
            if (FAILED(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&shared_rtv_heap.p))))
                return fail("CreateDescriptorHeap (handoff)");

            D3D12_HEAP_PROPERTIES def_hp{D3D12_HEAP_TYPE_DEFAULT};
            D3D12_RESOURCE_DESC rt_desc{};
            rt_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            rt_desc.Width = kSize;
            rt_desc.Height = kSize;
            rt_desc.DepthOrArraySize = 1;
            rt_desc.MipLevels = 1;
            rt_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            rt_desc.SampleDesc.Count = 1;
            rt_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
            if (FAILED(device->CreateCommittedResource(&def_hp, D3D12_HEAP_FLAG_NONE, &rt_desc,
                                                       D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                       IID_PPV_ARGS(&shared_rt.p))))
                return fail("CreateCommittedResource shared RT");

            const D3D12_CPU_DESCRIPTOR_HANDLE rtv = shared_rtv_heap->GetCPUDescriptorHandleForHeapStart();
            device->CreateRenderTargetView(shared_rt.p, nullptr, rtv);

            D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
            UINT64 total_bytes = 0;
            device->GetCopyableFootprints(&rt_desc, 0, 1, 0, &fp, nullptr, nullptr, &total_bytes);

            D3D12_HEAP_PROPERTIES rb_hp{D3D12_HEAP_TYPE_READBACK};
            D3D12_RESOURCE_DESC rb_desc{};
            rb_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            rb_desc.Width = total_bytes;
            rb_desc.Height = 1;
            rb_desc.DepthOrArraySize = 1;
            rb_desc.MipLevels = 1;
            rb_desc.SampleDesc.Count = 1;
            rb_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            if (FAILED(device->CreateCommittedResource(&rb_hp, D3D12_HEAP_FLAG_NONE, &rb_desc,
                                                       D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                       IID_PPV_ARGS(&shared_rb.p))))
                return fail("CreateCommittedResource shared RB");

            const std::uint8_t colours[2][3] = {{180, 90, 45}, {30, 210, 120}};
            D3D12_RESOURCE_STATES prev_state = D3D12_RESOURCE_STATE_COMMON;
            for (int round = 0; round < 2; ++round) {
                if (FAILED(alloc0->Reset()) || FAILED(cl0->Reset(alloc0.p, nullptr))) return fail("Reset cl0 (handoff)");
                D3D12_RESOURCE_BARRIER b0{};
                b0.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                b0.Transition = {shared_rt.p, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, prev_state,
                                 D3D12_RESOURCE_STATE_RENDER_TARGET};
                cl0->ResourceBarrier(1, &b0);
                const float col[4] = {colours[round][0] / 255.0f, colours[round][1] / 255.0f,
                                      colours[round][2] / 255.0f, 1.0f};
                cl0->ClearRenderTargetView(rtv, col, 0, nullptr);
                cl0->Close();

                ID3D12CommandList* l0[1] = {cl0.p};
                queues[0]->ExecuteCommandLists(1, l0);
                const UINT64 v0 = ++fence_vals[0];
                if (FAILED(queues[0]->Signal(fences[0].p, v0))) return fail("Signal q0 (handoff)");

                // Queue 1 waits on queue 0's fence, then transitions and copies the same shared texture.
                if (FAILED(queues[1]->Wait(fences[0].p, v0))) return fail("Wait q1 (handoff)");
                if (FAILED(alloc1->Reset()) || FAILED(cl1->Reset(alloc1.p, nullptr))) return fail("Reset cl1 (handoff)");
                D3D12_RESOURCE_BARRIER b1{};
                b1.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                b1.Transition = {shared_rt.p, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                                 D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE};
                cl1->ResourceBarrier(1, &b1);
                D3D12_TEXTURE_COPY_LOCATION src_loc{shared_rt.p, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {0}};
                D3D12_TEXTURE_COPY_LOCATION dst_loc{shared_rb.p, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
                dst_loc.PlacedFootprint = fp;
                cl1->CopyTextureRegion(&dst_loc, 0, 0, 0, &src_loc, nullptr);
                cl1->Close();
                prev_state = D3D12_RESOURCE_STATE_COPY_SOURCE;

                ID3D12CommandList* l1[1] = {cl1.p};
                queues[1]->ExecuteCommandLists(1, l1);
                const UINT64 v1 = ++fence_vals[1];
                if (FAILED(queues[1]->Signal(fences[1].p, v1)) ||
                    FAILED(fences[1]->SetEventOnCompletion(v1, handoff_ev.h)) ||
                    WaitForSingleObject(handoff_ev.h, 60000) != WAIT_OBJECT_0)
                    return fail("Signal/Wait q1 (handoff)");

                void* mapped = nullptr;
                if (FAILED(shared_rb->Map(0, nullptr, &mapped)) || !mapped) return fail("Map (handoff)");
                const auto* base = static_cast<const std::uint8_t*>(mapped);
                bool bad = false;
                for (UINT y = 0; y < kSize && !bad; ++y) {
                    const auto* row = base + static_cast<std::size_t>(y) * fp.Footprint.RowPitch;
                    for (UINT x = 0; x < kSize; ++x) {
                        const auto* px = row + static_cast<std::size_t>(x) * 4;
                        if (std::abs(int(px[0]) - int(colours[round][0])) > 1 ||
                            std::abs(int(px[1]) - int(colours[round][1])) > 1 ||
                            std::abs(int(px[2]) - int(colours[round][2])) > 1 || px[3] != 255) {
                            bad = true;
                            break;
                        }
                    }
                }
                shared_rb->Unmap(0, nullptr);
                if (bad) return fail("shared-texture cross-queue readback mismatch");
                if (FAILED(queues[0]->Wait(fences[1].p, v1))) return fail("Wait q0 (handoff)");
            }
        }

        // Exercise SetEventOnMultipleFenceCompletion (ALL) across both queues' fences after concurrent work.
        EventHandle multi_ev;
        if (!multi_ev.h) return fail("CreateEventW (multi ALL)");
        const UINT64 targets[2] = {++fence_vals[0], ++fence_vals[1]};
        ID3D12Fence* pair[2] = {fences[0].p, fences[1].p};
        if (FAILED(device->SetEventOnMultipleFenceCompletion(pair, targets, 2, D3D12_MULTIPLE_FENCE_WAIT_FLAG_ALL,
                                                             multi_ev.h)))
            return fail("SetEventOnMultipleFenceCompletion (ALL)");
        if (FAILED(queues[0]->Signal(fences[0].p, targets[0])) || FAILED(queues[1]->Signal(fences[1].p, targets[1])))
            return fail("Signal (multi ALL)");
        if (WaitForSingleObject(multi_ev.h, 60000) != WAIT_OBJECT_0) return fail("WaitForSingleObject (multi ALL)");

        // Exercise SetEventOnMultipleFenceCompletion (ANY): signalling only fence 0 satisfies the wait while fence 1 stays below target.
        EventHandle any_ev;
        if (!any_ev.h) return fail("CreateEventW (multi ANY)");
        const UINT64 any_targets[2] = {++fence_vals[0], fence_vals[1] + 10};
        if (FAILED(device->SetEventOnMultipleFenceCompletion(pair, any_targets, 2, D3D12_MULTIPLE_FENCE_WAIT_FLAG_ANY,
                                                             any_ev.h)))
            return fail("SetEventOnMultipleFenceCompletion (ANY)");
        if (FAILED(queues[0]->Signal(fences[0].p, any_targets[0]))) return fail("Signal (multi ANY)");
        if (WaitForSingleObject(any_ev.h, 60000) != WAIT_OBJECT_0) return fail("WaitForSingleObject (multi ANY)");
        }
    }  // vktest::Device::~Device() destroys VkDevice here so any leaked-object VUIDs are counted below.

    if (vktest::validation_error_count() != 0) {
        std::fprintf(stderr, "FAIL: %u Vulkan validation error(s)\n", vktest::validation_error_count());
        return 1;
    }
    std::printf("d3d12_multithread_test: PASS (2 queues x %d concurrent submissions + resource churn + shared-texture "
                "handoff + multi-fence ALL/ANY; 0 validation errors)\n",
                kIterations);
    return 0;
}
