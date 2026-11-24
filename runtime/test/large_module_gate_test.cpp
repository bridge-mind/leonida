// vk_pipeline_check's memory budget for large modules (large_module_gate.h), without a Vulkan device.
#include "large_module_gate.h"

#include <atomic>
#include <string>
#ifndef _WIN32
#include <unistd.h>
#endif
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

#define CHECK(condition) do {                                        \
    if (!(condition)) {                                              \
        std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
        return 1;                                                    \
    }                                                                \
} while (0)

using leonida::LargeModuleGate;

static int budget_and_count() {
    LargeModuleGate gate(10, 3);
    CHECK(gate.admits(4));
    gate.enter(4);
    CHECK(gate.admits(4));
    gate.enter(4);
    CHECK(!gate.admits(4));   // 12 > 10
    CHECK(gate.admits(2));    // 10 fits exactly
    gate.enter(2);
    CHECK(!gate.admits(0));   // three in: max_inflight
    gate.leave(4);
    CHECK(gate.admits(4));
    gate.leave(4);
    gate.leave(2);
    CHECK(gate.inflight() == 0);
    return 0;
}

static int oversized_module_still_gets_in_alone() {
    // Before: a module whose estimate exceeds the whole budget would wait forever.
    LargeModuleGate gate(10, 4);
    CHECK(gate.admits(100));
    gate.enter(100);
    CHECK(!gate.admits(1));   // nothing joins it
    gate.leave(100);
    CHECK(gate.admits(1));
    return 0;
}

static int max_inflight_zero_means_one() {
    LargeModuleGate gate(100, 0);
    CHECK(gate.max_inflight() == 1);
    gate.enter(1);
    CHECK(!gate.admits(1));
    gate.leave(1);
    return 0;
}

static int ticket_releases() {
    LargeModuleGate gate(10, 2);
    {
        LargeModuleGate::Ticket a(gate, 6);
        CHECK(gate.inflight() == 1);
        LargeModuleGate::Ticket moved = std::move(a);   // the moved-from ticket must not leave twice
        CHECK(gate.inflight() == 1);
        LargeModuleGate::Ticket b;
        b = LargeModuleGate::Ticket(gate, 4);
        CHECK(gate.inflight() == 2);
        b.release();
        CHECK(gate.inflight() == 1);
    }
    CHECK(gate.inflight() == 0);
    return 0;
}

static int threads_never_exceed_the_budget() {
    constexpr uint64_t kBudget = 10;
    LargeModuleGate gate(kBudget, 3);
    std::atomic<uint64_t> used{0};
    std::atomic<unsigned> in{0};
    std::atomic<bool> over{false};
    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t) {
        threads.emplace_back([&, t] {
            for (int i = 0; i < 200; ++i) {
                const uint64_t est = 1 + uint64_t((t + i) % 5);   // 1..5
                LargeModuleGate::Ticket ticket(gate, est);
                const uint64_t u = used += est;
                const unsigned n = ++in;
                if (n > 3 || (n > 1 && u > kBudget)) over = true;
                std::this_thread::sleep_for(std::chrono::microseconds(50));
                --in;
                used -= est;
            }
        });
    }
    for (auto& th : threads) th.join();
    CHECK(!over);
    CHECK(gate.inflight() == 0);
    return 0;
}

static int estimate_and_memory_query() {
    CHECK(leonida::large_module_estimate(4u << 20) == (uint64_t(4) << 20) * leonida::kBytesPerSpirvByte);
#if defined(__linux__) || defined(_WIN32)
    CHECK(leonida::available_memory_bytes() > 0);   // Linux: MemAvailable; Windows: ullAvailPhys
#endif
    leonida::release_free_memory();   // must be callable anywhere
    return 0;
}

#if defined(__linux__)
#include <filesystem>
#include <fstream>

static void write_file(const std::filesystem::path& p, const char* text) {
    std::filesystem::create_directories(p.parent_path());
    std::ofstream(p) << text;
}

static int cgroup_limits_are_read_up_the_tree() {
    // Before: only MemAvailable was read, so under systemd-run -p MemoryMax=... (or in a container) the budget was
    // the whole machine's and the checker could be killed by its own cgroup.
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / ("lmg_test_" + std::to_string(::getpid()));
    fs::remove_all(root);
    write_file(root / "self", "0::/user.slice/app.scope\n");
    write_file(root / "cg/user.slice/memory.max", "10000\n");
    write_file(root / "cg/user.slice/memory.current", "4000\n");          // room 6000
    write_file(root / "cg/user.slice/app.scope/memory.max", "max\n");     // no limit of its own
    write_file(root / "cg/user.slice/app.scope/memory.current", "3000\n");
    CHECK(leonida::cgroup_memory_room((root / "self").string(), (root / "cg").string()) == 6000);
    write_file(root / "cg/user.slice/app.scope/memory.max", "5000\n");    // room 2000, the tighter one
    CHECK(leonida::cgroup_memory_room((root / "self").string(), (root / "cg").string()) == 2000);
    write_file(root / "cg/user.slice/app.scope/memory.current", "7000\n"); // over its limit: no room
    CHECK(leonida::cgroup_memory_room((root / "self").string(), (root / "cg").string()) == 0);
    // A mount rooted at a constrained cgroup still exposes its budget at the mount root.
    write_file(root / "cg/user.slice/app.scope/memory.max", "max\n");
    write_file(root / "cg/memory.max", "5000\n");
    write_file(root / "cg/memory.current", "3000\n");
    CHECK(leonida::cgroup_memory_room((root / "self").string(), (root / "cg").string()) == 2000);
    write_file(root / "self", "0::/\n");
    CHECK(leonida::cgroup_memory_room((root / "self").string(), (root / "cg").string()) == 2000);
    write_file(root / "cg/memory.max", "max\n");
    CHECK(leonida::cgroup_memory_room((root / "self").string(), (root / "cg").string()) == UINT64_MAX);
    write_file(root / "self", "12:memory:/legacy\n");                      // cgroup v1 only: no limit read
    CHECK(leonida::cgroup_memory_room((root / "self").string(), (root / "cg").string()) == UINT64_MAX);
    CHECK(leonida::cgroup_memory_room((root / "missing").string(), (root / "cg").string()) == UINT64_MAX);
    fs::remove_all(root);
    return 0;
}
#endif

int main() {
#if defined(__linux__)
    if (int r = cgroup_limits_are_read_up_the_tree()) return r;
#endif
    if (int r = budget_and_count()) return r;
    if (int r = oversized_module_still_gets_in_alone()) return r;
    if (int r = max_inflight_zero_means_one()) return r;
    if (int r = ticket_releases()) return r;
    if (int r = threads_never_exceed_the_budget()) return r;
    if (int r = estimate_and_memory_query()) return r;
    std::puts("large_module_gate: ok");
    return 0;
}
