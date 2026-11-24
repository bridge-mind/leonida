// The event behind SetEventOnCompletion and SetEventOnMultipleFenceCompletion (d3d12_com.cpp), on its own so a test
// can check the countdown without a GPU (test/d3d12_fence_waiter_test.cpp).
#pragma once

#include <windows.h>

#include <atomic>

namespace rt::d3d12 {

// An event waiting for `remaining` fence values: 1 for SetEventOnCompletion and for the ANY flag of
// SetEventOnMultipleFenceCompletion, the fence count for ALL. The count is the API's UINT, kept unsigned so every
// count the caller can pass is waited for. Each fence that reaches its value counts down once; the hit that takes it
// from 1 to 0 sets the event, and later hits (ANY) leave it at 0 instead of wrapping.
struct FenceWaiter {
    FenceWaiter(HANDLE e, UINT n) : event(e), remaining(n) {}
    void hit() {
        UINT left = remaining.load();
        while (left && !remaining.compare_exchange_weak(left, left - 1)) {
        }
        if (left == 1) SetEvent(event);
    }
    bool done() const { return remaining.load() == 0; }
    HANDLE event;
    std::atomic<UINT> remaining;
};

}  // namespace rt::d3d12
