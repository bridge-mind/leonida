// The countdown behind the event-backed fence waits (shims/d3d12_fence_waiter.h), without a GPU: the ALL count keeps
// the API's full unsigned range, and hits past zero (ANY) neither wrap nor set the event again.
#include "../shims/d3d12_fence_waiter.h"

#include <climits>
#include <cstdio>
#include <initializer_list>

using rt::d3d12::FenceWaiter;

static int failures = 0;

static void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}

static bool signaled(HANDLE event) { return WaitForSingleObject(event, 0) == WAIT_OBJECT_0; }

int main() {
    HANDLE event = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (!event) return 1;

    // Counts past INT_MAX: an int countdown would start negative, look done and never set the event.
    for (UINT count : {UINT(INT_MAX) + 1u, UINT_MAX}) {
        FenceWaiter waiter(event, count);
        check(!waiter.done(), "a count past INT_MAX is not done before any hit");
        for (int i = 0; i < 1000; ++i) waiter.hit();
        check(!waiter.done() && waiter.remaining.load() == count - 1000, "a large count counts down one per hit");
        check(!signaled(event), "a large count does not set the event early");
        // The last hits, from just above the end.
        waiter.remaining.store(2);
        waiter.hit();
        check(!signaled(event), "the second-to-last hit does not set the event");
        waiter.hit();
        check(waiter.done() && signaled(event), "the last hit sets the event");
        ResetEvent(event);
    }

    // ALL over three fences: only the third hit sets the event.
    {
        FenceWaiter waiter(event, 3);
        waiter.hit();
        waiter.hit();
        check(!waiter.done() && !signaled(event), "ALL waits for every fence");
        waiter.hit();
        check(waiter.done() && signaled(event), "ALL sets the event on the last fence");
        ResetEvent(event);
    }

    // ANY: the first hit sets the event; the other fences' hits stay at zero instead of wrapping to UINT_MAX.
    {
        FenceWaiter waiter(event, 1);
        waiter.hit();
        check(waiter.done() && signaled(event), "ANY sets the event on the first fence");
        ResetEvent(event);
        for (int i = 0; i < 5; ++i) waiter.hit();
        check(waiter.done() && waiter.remaining.load() == 0, "hits past zero do not wrap");
        check(!signaled(event), "hits past zero do not set the event again");
    }

    CloseHandle(event);
    if (failures) return 1;
    std::printf("d3d12_fence_waiter_test: ok\n");
    return 0;
}
