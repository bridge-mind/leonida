#include "pch.h"
#include "XTaskQueue.h"
#include "XTaskQueuePriv.h"

#include <cstdio>

namespace
{
struct TestState
{
    XTaskQueueHandle owner = nullptr;
    XTaskQueueHandle composite = nullptr;
    int monitorCalls = 0;
    bool closeReturned = false;
    bool unexpectedCallback = false;
};

void CALLBACK NoopCallback(void*, bool)
{
}

void CALLBACK CloseCompositeMonitor(
    void* context,
    XTaskQueueHandle queue,
    XTaskQueuePort port)
{
    auto* state = static_cast<TestState*>(context);

    std::printf(
        "TRACE: monitor queue=%p owner=%p port=%u expectedPort=%u\n",
        static_cast<void*>(queue),
        static_cast<void*>(state->owner),
        static_cast<unsigned>(port),
        static_cast<unsigned>(XTaskQueuePort::Work));

    if (port != XTaskQueuePort::Work)
    {
        state->unexpectedCallback = true;
        return;
    }

    ++state->monitorCalls;

    if (state->monitorCalls != 1 || state->composite == nullptr)
    {
        state->unexpectedCallback = true;
        return;
    }

    // B has no extra handles or queued entries holding it alive.
    // Its final handle is closed while AtomicVector::Visit is active.
    XTaskQueueHandle composite = state->composite;
    state->composite = nullptr;

    std::puts("TRACE: monitor A entered");
    std::puts("TRACE: closing final handle of composite B");

    XTaskQueueCloseHandle(composite);

    state->closeReturned = true;
    std::puts("TRACE: composite B close returned");
}

bool Check(bool condition, const char* description)
{
    std::printf("%s %s\n",
        condition ? "PASS" : "FAIL",
        description);
    return condition;
}
}

int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    int failures = 0;
    XTaskQueueHandle owner = nullptr;

    HRESULT hr = XTaskQueueCreate(
        XTaskQueueDispatchMode::Manual,
        XTaskQueueDispatchMode::Manual,
        &owner);

    if (!Check(SUCCEEDED(hr) && owner != nullptr,
               "owner queue A created"))
    {
        return 2;
    }

    XTaskQueuePortHandle workPort = nullptr;
    XTaskQueuePortHandle completionPort = nullptr;

    HRESULT workHr = XTaskQueueGetPort(
        owner, XTaskQueuePort::Work, &workPort);

    HRESULT completionHr = XTaskQueueGetPort(
        owner, XTaskQueuePort::Completion, &completionPort);

    if (!Check(SUCCEEDED(workHr) &&
               SUCCEEDED(completionHr) &&
               workPort != nullptr &&
               completionPort != nullptr,
               "owner ports acquired"))
    {
        XTaskQueueCloseHandle(owner);
        return 2;
    }

    TestState state;
    state.owner = owner;

    hr = XTaskQueueCreateComposite(
        workPort, completionPort, &state.composite);

    if (!Check(SUCCEEDED(hr) && state.composite != nullptr,
               "composite queue B created"))
    {
        XTaskQueueCloseHandle(owner);
        return 2;
    }

    XTaskQueueRegistrationToken token = {};

    hr = XTaskQueueRegisterMonitor(
        owner, &state, CloseCompositeMonitor, &token);

    if (!Check(SUCCEEDED(hr), "monitor registered on A"))
    {
        XTaskQueueCloseHandle(state.composite);
        XTaskQueueCloseHandle(owner);
        return 2;
    }

    // Submit to A, not B: the queued entry cannot keep B alive.
    // On the original implementation this call is expected to hang
    // while B.Detach tries to remove an active AtomicVector reader.
    std::puts("TRACE: submitting callback to A");

    hr = XTaskQueueSubmitCallback(
        owner, XTaskQueuePort::Work, nullptr, NoopCallback);

    if (!Check(SUCCEEDED(hr), "submission returned"))
        ++failures;

    if (!Check(state.monitorCalls == 1,
               "monitor invoked exactly once"))
        ++failures;

    if (!Check(state.closeReturned,
               "final close of B returned inside monitor"))
        ++failures;

    if (!Check(!state.unexpectedCallback,
               "no unexpected monitor invocation"))
        ++failures;

    XTaskQueueUnregisterMonitor(owner, token);

    // Drain the callback submitted to A before releasing its handle.
    while (XTaskQueueDispatch(owner, XTaskQueuePort::Work, 0))
    {
    }

    if (state.composite != nullptr)
    {
        XTaskQueueCloseHandle(state.composite);
        state.composite = nullptr;
    }

    XTaskQueueCloseHandle(owner);

    if (!Check(XTaskQueueUninitialize(1000),
               "all task queue references released"))
        ++failures;

    std::printf("RESULT: %s (%d failures)\n",
        failures == 0 ? "PASS" : "FAIL",
        failures);

    return failures == 0 ? 0 : 1;
}
