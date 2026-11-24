// The first CheckFeatureSupport(D3D12_OPTIONS1) calls on a new device, made by several threads at the same time.
// The device reads the physical device's subgroup facts on the first call; every caller must get the complete
// answer, also the ones that arrive while that first read is in progress. Each round makes a new D3D12 device on the
// same Vulkan device (cheap), so each round has a first use. No shaders or game data.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

namespace {

int fail(const char* what) {
    std::fprintf(stderr, "d3d12_options1_first_use_test: %s\n", what);
    return 1;
}

// The fields by name, so that padding or a field a later SDK adds cannot change the result.
bool same_answer(const D3D12_FEATURE_DATA_D3D12_OPTIONS1& a, const D3D12_FEATURE_DATA_D3D12_OPTIONS1& b) {
    return a.WaveOps == b.WaveOps && a.WaveLaneCountMin == b.WaveLaneCountMin &&
           a.WaveLaneCountMax == b.WaveLaneCountMax && a.TotalLaneCount == b.TotalLaneCount &&
           a.ExpandedComputeResourceStates == b.ExpandedComputeResourceStates &&
           a.Int64ShaderOps == b.Int64ShaderOps;
}

}  // namespace

int main() {
    // The threads poll between rounds, so there are never more of them than cores left beside the main thread.
    const unsigned cores = std::thread::hardware_concurrency();
    const int kThreads = cores >= 9 ? 8 : cores >= 3 ? int(cores) - 1 : 2;
    constexpr int kRounds = 500;

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

    // What one thread gets, from a device of its own: the answer every racing call must equal.
    D3D12_FEATURE_DATA_D3D12_OPTIONS1 expected{};
    {
        ID3D12Device* device = nullptr;
        if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device))) || !device)
            return fail("create_vulkan_device");
        const HRESULT hr = device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &expected, sizeof(expected));
        device->Release();
        if (FAILED(hr)) return fail("CheckFeatureSupport(OPTIONS1) on one thread");
    }
    if (expected.WaveLaneCountMin < 1 || expected.WaveLaneCountMax < expected.WaveLaneCountMin)
        return fail("the single-thread answer has no lane range");

    // The threads live for the whole test and poll between rounds (with a yield, so an idle thread gives its core
    // away), so that they all enter the call inside the few microseconds the first read takes.
    std::atomic<ID3D12Device*> shared{nullptr};
    std::atomic<int> round{0}, finished{0};
    std::atomic<long long> wrong{0}, errors{0};
    std::atomic<bool> stop{false};
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t)
        threads.emplace_back([&] {
            int seen = 0;
            while (!stop.load(std::memory_order_acquire)) {
                if (round.load(std::memory_order_acquire) == seen) {
                    std::this_thread::yield();
                    continue;
                }
                ++seen;
                D3D12_FEATURE_DATA_D3D12_OPTIONS1 got{};
                if (FAILED(shared.load()->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &got, sizeof(got))))
                    ++errors;
                else if (!same_answer(got, expected))
                    ++wrong;
                ++finished;
            }
        });

    bool created = true;
    for (int r = 0; r < kRounds && created; ++r) {
        ID3D12Device* device = nullptr;
        created = SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device))) && device;
        if (!created) break;
        shared.store(device);
        finished.store(0);
        round.fetch_add(1, std::memory_order_release);
        // A thread that does not come back cannot be joined: stop the process with the reason, not a hang.
        const auto give_up = std::chrono::steady_clock::now() + std::chrono::seconds(60);
        while (finished.load() != kThreads) {
            if (std::chrono::steady_clock::now() > give_up) {
                std::fprintf(stderr, "d3d12_options1_first_use_test: round %d: %d of %d threads did not return from "
                                     "CheckFeatureSupport(OPTIONS1) in 60 s\n", r, kThreads - finished.load(), kThreads);
                std::fflush(stderr);
                std::_Exit(1);
            }
            std::this_thread::yield();
        }
        device->Release();
    }
    stop.store(true, std::memory_order_release);
    for (std::thread& thread : threads) thread.join();

    if (!created) return fail("create_vulkan_device in a round");
    std::printf("D3D12_OPTIONS1 first use: %d rounds of %d threads, %lld incomplete answers, %lld failed calls "
                "(expected WaveOps %d, lanes %u..%u)\n",
                kRounds, kThreads, wrong.load(), errors.load(), int(expected.WaveOps), expected.WaveLaneCountMin,
                expected.WaveLaneCountMax);
    if (errors.load()) return fail("a racing CheckFeatureSupport(OPTIONS1) call failed");
    if (wrong.load()) return fail("a racing first call got an answer that differs from the single-thread answer");
    return 0;
}
