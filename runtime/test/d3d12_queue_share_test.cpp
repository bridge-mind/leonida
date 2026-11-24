// D3D12 queues on shared Vulkan queues (docs/d3d12_vulkan.md section 4.4, "More D3D12 queues than Vulkan queues").
//
// A GPU whose queue family exposes one VkQueue (the Intel UHD 770 on Windows, lavapipe) used to refuse the second
// D3D12 queue with E_OUTOFMEMORY, so a title with a DIRECT and a COMPUTE queue stopped at boot. The layer now puts a
// D3D12 queue on a VkQueue another one already uses when every VkQueue of its families is taken. Cases:
//   many      36 queues (12 per type, more than the VkQueues of any GPU we know: the RTX 5080 has 26), then a buffer
//             copied through all 36 in a chain linked by fences, submitted consumer first (every Wait before its
//             Signal) and then producer first; the readback must be exact.
//   deferred  a CPU Signal that has to wait behind a queue Signal the GPU has not reached, and a queue that waits
//             for the CPU value: the queue must run once the CPU value lands (on one VkQueue or two).
//   release   a fence released right after the GPU passed the queue Signal its deferred CPU Signal waited behind,
//             while another queue holds a batch for the CPU value: the batch runs, and under the validation layer
//             the semaphore is not destroyed under it.
//   stress    4 threads, 9 queues (3 per type), each thread copying a buffer through 3 queues per iteration with
//             a CPU-gated producer; every iteration's readback must be exact.
// With LEONIDA_VK_MAX_QUEUES_PER_FAMILY=1 and LEONIDA_VK_GRAPHICS_FAMILY_ONLY=1 every GPU looks like the UHD 770 (one
// family, one VkQueue) and every D3D12 queue shares that VkQueue, as on lavapipe; with the first alone, same-type
// queues share their family's VkQueue. No game data.
#include "../shims/d3d12_com.h"

#include <wrl/client.h>

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace {

constexpr UINT64 bytes = 1024;
constexpr UINT words = UINT(bytes / 4);
constexpr D3D12_COMMAND_LIST_TYPE types[] = {D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                             D3D12_COMMAND_LIST_TYPE_COPY};

const char* type_name(D3D12_COMMAND_LIST_TYPE type) {
    return type == D3D12_COMMAND_LIST_TYPE_DIRECT ? "DIRECT" : type == D3D12_COMMAND_LIST_TYPE_COMPUTE ? "COMPUTE"
                                                                                                       : "COPY";
}

// The watchdog: a stuck case ends the process with its name instead of a ctest timeout with none.
std::atomic<ULONGLONG> deadline{0};
std::atomic<const char*> current_case{"start"};
std::string case_names[64];
int case_count = 0;
void arm(const std::string& name, ULONGLONG ms) {  // called from the main thread only
    std::string& slot = case_names[case_count++ % 64];
    slot = name;
    current_case = slot.c_str();
    deadline = GetTickCount64() + ms;
}

bool fail(const char* what, int line) {
    std::fprintf(stderr, "case \"%s\": line %d: %s\n", current_case.load(), line, what);
    return false;
}
#define CHECK(c) do { if (!(c)) return fail(#c, __LINE__); } while (0)

std::uint32_t pattern(std::uint32_t seed, std::uint32_t i) {
    std::uint32_t x = seed * 0x9e3779b1u + i * 0x85ebca6bu + 0x1234567u;
    x ^= x >> 15; x *= 0x2c1b3c6du; x ^= x >> 12; x *= 0x297a2d39u; x ^= x >> 15;
    return x | 0x01000000u;  // never zero, so a zero-filled or stale buffer cannot match
}

ComPtr<ID3D12Resource> buffer(ID3D12Device* device, D3D12_HEAP_TYPE heap, D3D12_RESOURCE_STATES state) {
    D3D12_HEAP_PROPERTIES properties{};
    properties.Type = heap;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ComPtr<ID3D12Resource> resource;
    if (FAILED(device->CreateCommittedResource(&properties, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                               IID_PPV_ARGS(&resource))))
        return {};
    return resource;
}

bool fill(ID3D12Resource* upload, std::uint32_t seed) {
    std::uint32_t* mapped = nullptr;
    const D3D12_RANGE none{0, 0};
    CHECK(SUCCEEDED(upload->Map(0, &none, reinterpret_cast<void**>(&mapped))));
    for (UINT i = 0; i < words; ++i) mapped[i] = pattern(seed, i);
    upload->Unmap(0, nullptr);
    return true;
}

bool exact(ID3D12Resource* readback, std::uint32_t seed) {
    std::uint32_t* mapped = nullptr;
    CHECK(SUCCEEDED(readback->Map(0, nullptr, reinterpret_cast<void**>(&mapped))));
    unsigned bad = 0;
    for (UINT i = 0; i < words; ++i)
        if (mapped[i] != pattern(seed, i) && bad++ < 3)
            std::fprintf(stderr, "word %u: got %08x want %08x\n", i, mapped[i], pattern(seed, i));
    const D3D12_RANGE none{0, 0};
    readback->Unmap(0, &none);
    if (bad) std::fprintf(stderr, "%u of %u words differ\n", bad, words);
    return bad == 0 ? true : fail("the readback differs from the data the CPU wrote", __LINE__);
}

D3D12_RESOURCE_BARRIER transition(ID3D12Resource* resource, D3D12_RESOURCE_STATES before,
                                  D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition = {resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before, after};
    return barrier;
}

// Records src -> dst into the open list and closes it. A default-heap buffer (dst_default, src_default) goes from
// COMMON to COPY_SOURCE / COPY_DEST and back; the upload and readback buffers stay in their heap's state.
bool record_copy(ID3D12GraphicsCommandList* list, ID3D12Resource* dst, bool dst_default, ID3D12Resource* src,
                 bool src_default) {
    D3D12_RESOURCE_BARRIER before[2], after[2];
    UINT count = 0;
    if (src_default) {
        before[count] = transition(src, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
        after[count++] = transition(src, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
    }
    if (dst_default) {
        before[count] = transition(dst, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
        after[count++] = transition(dst, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
    }
    if (count) list->ResourceBarrier(count, before);
    list->CopyBufferRegion(dst, 0, src, 0, bytes);
    if (count) list->ResourceBarrier(count, after);
    return SUCCEEDED(list->Close());
}

// One closed copy list and its allocator.
struct CopyList {
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    bool make(ID3D12Device* device, D3D12_COMMAND_LIST_TYPE type, ID3D12Resource* dst, bool dst_default,
              ID3D12Resource* src, bool src_default) {
        allocator.Reset();
        list.Reset();
        return SUCCEEDED(device->CreateCommandAllocator(type, IID_PPV_ARGS(&allocator))) &&
               SUCCEEDED(device->CreateCommandList(0, type, allocator.Get(), nullptr, IID_PPV_ARGS(&list))) &&
               record_copy(list.Get(), dst, dst_default, src, src_default);
    }
    void execute(ID3D12CommandQueue* queue) const {
        ID3D12CommandList* lists[] = {list.Get()};
        queue->ExecuteCommandLists(1, lists);
    }
};

ComPtr<ID3D12Fence> make_fence(ID3D12Device* device) {
    ComPtr<ID3D12Fence> fence;
    if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)))) return {};
    return fence;
}

ComPtr<ID3D12CommandQueue> make_queue(ID3D12Device* device, D3D12_COMMAND_LIST_TYPE type, HRESULT* result = nullptr) {
    D3D12_COMMAND_QUEUE_DESC desc{};
    desc.Type = type;
    ComPtr<ID3D12CommandQueue> queue;
    const HRESULT hr = device->CreateCommandQueue(&desc, IID_PPV_ARGS(&queue));
    if (result) *result = hr;
    return SUCCEEDED(hr) ? queue : nullptr;
}

// Waits for the fence; a fence that never gets there ends the process (a stuck queue keeps destructors blocked).
void wait_for(ID3D12Fence* fence, UINT64 value, DWORD ms = 20000) {
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (event && SUCCEEDED(fence->SetEventOnCompletion(value, event)) &&
        WaitForSingleObject(event, ms) == WAIT_OBJECT_0) {
        CloseHandle(event);
        return;
    }
    std::fprintf(stderr, "case \"%s\": a fence never reached %llu (completed %llu): the queues did not make progress\n",
                 current_case.load(), static_cast<unsigned long long>(value),
                 static_cast<unsigned long long>(fence->GetCompletedValue()));
    std::fflush(stderr);
    std::_Exit(1);
}

// 36 queues, more than the VkQueues of any GPU we know, and a buffer copied through all of them.
bool many_queues(ID3D12Device* device) {
    constexpr int per_type = 12;
    arm("36 queues", 120000);
    std::vector<ComPtr<ID3D12CommandQueue>> queues;
    std::vector<D3D12_COMMAND_LIST_TYPE> queue_types;
    for (int i = 0; i < per_type; ++i)
        for (const auto type : types) {
            HRESULT hr = S_OK;
            auto queue = make_queue(device, type, &hr);
            if (!queue) {
                std::fprintf(stderr, "CreateCommandQueue(%s) number %zu: 0x%08lx\n", type_name(type),
                             queues.size() + 1, static_cast<unsigned long>(hr));
                return fail("a D3D12 queue was refused", __LINE__);
            }
            queues.push_back(queue);
            queue_types.push_back(type);
        }
    std::printf("created %zu D3D12 queues (%d per type)\n", queues.size(), per_type);
    const std::size_t n = queues.size();
    for (const bool consumer_first : {true, false}) {
        arm(std::string("chain through 36 queues, ") + (consumer_first ? "consumer first" : "producer first"), 120000);
        const std::uint32_t seed = consumer_first ? 11 : 12;
        auto upload = buffer(device, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        auto readback = buffer(device, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        CHECK(upload && readback && fill(upload.Get(), seed));
        std::vector<ComPtr<ID3D12Resource>> stage(n - 1);
        for (auto& s : stage) CHECK((s = buffer(device, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON)));
        std::vector<ComPtr<ID3D12Fence>> fences(n);
        for (auto& f : fences) CHECK((f = make_fence(device)));
        // Step k on queue k: Wait(fence k-1), copy stage k-1 -> stage k (from the upload first, to the readback
        // last), Signal(fence k).
        std::vector<CopyList> lists(n);
        for (std::size_t k = 0; k < n; ++k) {
            ID3D12Resource* src = k == 0 ? upload.Get() : stage[k - 1].Get();
            ID3D12Resource* dst = k == n - 1 ? readback.Get() : stage[k].Get();
            CHECK(lists[k].make(device, queue_types[k], dst, k != n - 1, src, k != 0));
        }
        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t k = consumer_first ? n - 1 - i : i;
            if (k) CHECK(SUCCEEDED(queues[k]->Wait(fences[k - 1].Get(), 1)));
            lists[k].execute(queues[k].Get());
            CHECK(SUCCEEDED(queues[k]->Signal(fences[k].Get(), 1)));
        }
        wait_for(fences[n - 1].Get(), 1);
        if (!exact(readback.Get(), seed)) return false;
        std::printf("%s: exact\n", current_case.load());
    }
    return true;
}

// Queue a signals f = 1 behind a gate the CPU opens last; the CPU signals f = 2 before that, which the layer applies
// only once the GPU has passed 1 (a Vulkan timeline must not be set past a pending signal); queue b waits for f = 2
// and copies. b must run once the CPU value lands.
bool deferred_cpu_signal(ID3D12Device* device, D3D12_COMMAND_LIST_TYPE ta, D3D12_COMMAND_LIST_TYPE tb) {
    arm(std::string("deferred CPU Signal, ") + type_name(ta) + " then " + type_name(tb), 30000);
    auto a = make_queue(device, ta);
    auto b = make_queue(device, tb);
    auto gate = make_fence(device), f = make_fence(device), done = make_fence(device);
    auto upload = buffer(device, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    auto readback = buffer(device, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    const std::uint32_t seed = 100 + UINT(ta) * 8 + UINT(tb);
    CopyList copy;
    CHECK(a && b && gate && f && done && upload && readback && fill(upload.Get(), seed));
    CHECK(copy.make(device, tb, readback.Get(), false, upload.Get(), false));
    CHECK(SUCCEEDED(a->Wait(gate.Get(), 1)));
    CHECK(SUCCEEDED(a->Signal(f.Get(), 1)));
    CHECK(SUCCEEDED(f->Signal(2)));
    CHECK(SUCCEEDED(b->Wait(f.Get(), 2)));
    copy.execute(b.Get());
    CHECK(SUCCEEDED(b->Signal(done.Get(), 1)));
    CHECK(SUCCEEDED(gate->Signal(1)));
    wait_for(done.Get(), 1);
    if (!exact(readback.Get(), seed)) return false;
    std::printf("%s: completed, exact\n", current_case.load());
    return true;
}

// A fence released just after the GPU passed queue a's Signal of 1, while the CPU's Signal of 2 is still deferred
// behind it and queue b holds a batch that waits for 2. Either the fence wait thread or the release applies the CPU
// value; the fence reached 2 before it was released, so b's batch must run (as on D3D12, where the CPU Signal lands at
// once), and the semaphore must outlive that batch (VUID-vkDestroySemaphore-semaphore-05149, formerly -01137, under
// the validation layer). Every iteration's readback must hold the copy, never the sentinel the CPU left there.
bool release_while_held(ID3D12Device* device, D3D12_COMMAND_LIST_TYPE type, int iterations) {
    arm(std::string("fence released while a batch waits for its deferred CPU Signal, ") + type_name(type),
        30000 + ULONGLONG(iterations) * 200);
    constexpr std::uint32_t sentinel = 0xdeadbeefu;
    auto a = make_queue(device, type), b = make_queue(device, type);
    auto upload = buffer(device, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    auto readback = buffer(device, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    CHECK(a && b && upload && readback && fill(upload.Get(), 7));
    int dropped = 0;
    for (int i = 0; i < iterations; ++i) {
        std::uint32_t* mapped = nullptr;
        CHECK(SUCCEEDED(readback->Map(0, nullptr, reinterpret_cast<void**>(&mapped))));
        for (UINT w = 0; w < words; ++w) mapped[w] = sentinel;
        readback->Unmap(0, nullptr);
        auto gate = make_fence(device), f = make_fence(device), done = make_fence(device);
        CopyList copy;
        CHECK(gate && f && done && copy.make(device, type, readback.Get(), false, upload.Get(), false));
        CHECK(SUCCEEDED(a->Wait(gate.Get(), 1)));
        CHECK(SUCCEEDED(a->Signal(f.Get(), 1)));
        CHECK(SUCCEEDED(f->Signal(2)));  // deferred: a's Signal of 1 is not on the GPU yet
        CHECK(SUCCEEDED(b->Wait(f.Get(), 2)));
        copy.execute(b.Get());
        CHECK(SUCCEEDED(b->Signal(done.Get(), 1)));
        CHECK(SUCCEEDED(gate->Signal(1)));
        const ULONGLONG give_up = GetTickCount64() + 5000;
        while (f->GetCompletedValue() < 1)
            CHECK(GetTickCount64() < give_up);
        f.Reset();  // the release, as soon as the GPU has passed a's Signal
        wait_for(done.Get(), 1);  // b's Signal comes after its batch
        CHECK(SUCCEEDED(readback->Map(0, nullptr, reinterpret_cast<void**>(&mapped))));
        bool copied = true;
        for (UINT w = 0; w < words; ++w) copied &= mapped[w] == pattern(7, w);
        const D3D12_RANGE none{0, 0};
        readback->Unmap(0, &none);
        dropped += !copied;
    }
    std::printf("%s: %d of %d batches ran\n", current_case.load(), iterations - dropped, iterations);
    CHECK(dropped == 0);
    return true;
}

// 4 threads copy a buffer through 3 of 9 queues (3 per type) per iteration, the producer held until the thread's
// CPU gate opens; submissions are in dependency order, so no thread waits for another.
bool stress(ID3D12Device* device, int iterations) {
    constexpr int threads = 4, per_type = 3, queue_count = 3 * per_type;
    arm("stress: 4 threads on 9 queues", 30000 + ULONGLONG(iterations) * 500);
    ComPtr<ID3D12CommandQueue> queues[queue_count];
    D3D12_COMMAND_LIST_TYPE queue_types[queue_count];
    for (int i = 0; i < queue_count; ++i) {
        queue_types[i] = types[i / per_type];  // same-type queues next to each other: a chain shares VkQueues
        CHECK((queues[i] = make_queue(device, queue_types[i])));
    }
    std::atomic<int> failures{0}, completed{0};
    auto worker = [&](int t) {
        auto upload = buffer(device, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        auto readback = buffer(device, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        auto mid0 = buffer(device, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON);
        auto mid1 = buffer(device, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON);
        ComPtr<ID3D12Fence> gate = make_fence(device);
        ComPtr<ID3D12Fence> fences[3] = {make_fence(device), make_fence(device), make_fence(device)};
        if (!upload || !readback || !mid0 || !mid1 || !gate || !fences[0] || !fences[1] || !fences[2]) {
            ++failures;
            return;
        }
        for (int i = 0; i < iterations && !failures; ++i) {
            const std::uint32_t seed = std::uint32_t(t) * 100003u + std::uint32_t(i);
            const int k = (2 * t + i) % queue_count;
            const int q[3] = {k, (k + 1) % queue_count, (k + 2) % queue_count};
            CopyList lists[3];
            const UINT64 v = UINT64(i) + 1;
            if (!fill(upload.Get(), seed) ||
                !lists[0].make(device, queue_types[q[0]], mid0.Get(), true, upload.Get(), false) ||
                !lists[1].make(device, queue_types[q[1]], mid1.Get(), true, mid0.Get(), true) ||
                !lists[2].make(device, queue_types[q[2]], readback.Get(), false, mid1.Get(), true) ||
                FAILED(queues[q[0]]->Wait(gate.Get(), v))) {
                ++failures;
                return;
            }
            for (int s = 0; s < 3; ++s) {
                if (s && FAILED(queues[q[s]]->Wait(fences[s - 1].Get(), v))) ++failures;
                lists[s].execute(queues[q[s]].Get());
                if (FAILED(queues[q[s]]->Signal(fences[s].Get(), v))) ++failures;
            }
            if (FAILED(gate->Signal(v))) ++failures;
            wait_for(fences[2].Get(), v);
            if (!exact(readback.Get(), seed)) ++failures;
            ++completed;
        }
    };
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t) pool.emplace_back(worker, t);
    for (auto& thread : pool) thread.join();
    std::printf("stress: %d of %d iterations completed exact over %d threads\n", completed.load(),
                threads * iterations, threads);
    CHECK(failures == 0 && completed == threads * iterations);
    return true;
}

constexpr int most_iterations = 1000000;

// Exit code 2 and the usage line, before any device exists: a misspelt case or count must not pass with nothing run.
int usage(const char* problem, const char* value) {
    std::fprintf(stderr, "%s%s\nusage: d3d12_queue_share_test [--case many|deferred|release|stress|all] "
                         "[--iterations N] (N from 1 to %d, default 100)\n",
                 problem, value, most_iterations);
    return 2;
}

bool known_case(const std::string& which) {
    for (const char* name : {"many", "deferred", "release", "stress", "all"})
        if (which == name) return true;
    return false;
}

// The whole argument as a decimal count from 1 to most_iterations, or 0.
int parse_iterations(const char* text) {
    char* end = nullptr;
    errno = 0;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end || errno == ERANGE || value < 1 || value > most_iterations) return 0;
    return int(value);
}

}  // namespace

int main(int argc, char** argv) {
    std::string which = "all";
    int iterations = 100;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--case") && i + 1 < argc) {
            which = argv[++i];
            if (!known_case(which)) return usage("unknown case: ", which.c_str());
        } else if (!std::strcmp(argv[i], "--iterations") && i + 1 < argc) {
            iterations = parse_iterations(argv[++i]);
            if (!iterations) return usage("not a whole number of iterations in range: ", argv[i]);
        } else {
            return usage("unknown argument: ", argv[i]);
        }
    }
    std::thread([] {
        for (;;) {
            Sleep(100);
            const ULONGLONG limit = deadline.load();
            if (limit && GetTickCount64() > limit) {
                std::fprintf(stderr, "HANG in case \"%s\"\n", current_case.load());
                std::fflush(stderr);
                std::_Exit(1);
            }
        }
    }).detach();

    ComPtr<ID3D12Device> device;
    if (FAILED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&device)))) {
        // With LEONIDA_VK_VALIDATION set, no device means no validation (the layer is missing): a failure, not a skip,
        // so a ctest run that asked for validation never passes or skips without it.
        char validation[8] = {};
        size_t validation_length = 0;
        const bool wants_validation = getenv_s(&validation_length, validation, "LEONIDA_VK_VALIDATION") == 0 &&
                                      validation_length > 1 && validation[0] != '0';
        std::puts(wants_validation ? "Vulkan standalone device unavailable with LEONIDA_VK_VALIDATION set (see the [rt] line above)"
                                   : "Vulkan standalone device unavailable");
        return wants_validation ? 1 : 77;
    }
    bool ok = true;
    if (which == "all" || which == "many") ok &= many_queues(device.Get());
    if (which == "all" || which == "deferred")
        for (const auto ta : types)
            for (const auto tb : types) ok &= deferred_cpu_signal(device.Get(), ta, tb);
    if (which == "all" || which == "release")
        for (const auto type : types) ok &= release_while_held(device.Get(), type, iterations);
    if (which == "all" || which == "stress") ok &= stress(device.Get(), iterations);
    deadline = 0;
    std::puts(ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
