#include "pch.h"
#include "XTaskQueue.h"
#include "XTaskQueuePriv.h"

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <thread>

namespace
{
struct State
{
    std::mutex mutex;
    std::condition_variable cv;
    XTaskQueueHandle owner = nullptr;
    XTaskQueueHandle composite = nullptr;
    bool monitorEntered = false;
    bool closeReturned = false;
    bool observedClose = false;
    bool unexpectedMonitor = false;
};

void CALLBACK NoopCallback(void*, bool)
{
}

void CALLBACK Monitor(
    void* context,
    XTaskQueueHandle,
    XTaskQueuePort port)
{
    auto* state = static_cast<State*>(context);

    if (port != XTaskQueuePort::Work)
    {
        state->unexpectedMonitor = true;
        return;
    }

    {
        std::lock_guard<std::mutex> lock(state->mutex);

        if (state->monitorEntered)
        {
            state->unexpectedMonitor = true;
            return;
        }

        state->monitorEntered = true;
    }

    std::puts("TRACE: monitor entered");
    state->cv.notify_all();

    std::unique_lock<std::mutex> lock(state->mutex);

    state->observedClose = state->cv.wait_for(
        lock,
        std::chrono::seconds(4),
        [&] { return state->closeReturned; });

    std::printf(
        "TRACE: close observed before monitor return=%d\n",
        state->observedClose ? 1 : 0);
}

bool Check(bool condition, const char* description)
{
    std::printf(
        "%s %s\n",
        condition ? "PASS" : "FAIL",
        description);

    return condition;
}
}

int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    int failures = 0;
    State state;

    HRESULT hr = XTaskQueueCreate(
        XTaskQueueDispatchMode::Manual,
        XTaskQueueDispatchMode::Manual,
        &state.owner);

    if (!Check(SUCCEEDED(hr) && state.owner != nullptr,
               "owner queue created"))
    {
        return 2;
    }

    XTaskQueuePortHandle work = nullptr;
    XTaskQueuePortHandle completion = nullptr;

    HRESULT workHr = XTaskQueueGetPort(
        state.owner, XTaskQueuePort::Work, &work);
    HRESULT completionHr = XTaskQueueGetPort(
        state.owner, XTaskQueuePort::Completion, &completion);

    if (!Check(SUCCEEDED(workHr) && SUCCEEDED(completionHr) &&
               work != nullptr && completion != nullptr,
               "owner ports acquired"))
    {
        XTaskQueueCloseHandle(state.owner);
        return 2;
    }

    hr = XTaskQueueCreateComposite(
        work, completion, &state.composite);

    if (!Check(SUCCEEDED(hr) && state.composite != nullptr,
               "composite queue created"))
    {
        XTaskQueueCloseHandle(state.owner);
        return 2;
    }

    XTaskQueueRegistrationToken token = {};
    hr = XTaskQueueRegisterMonitor(
        state.owner, &state, Monitor, &token);

    if (!Check(SUCCEEDED(hr), "monitor registered"))
    {
        XTaskQueueCloseHandle(state.composite);
        XTaskQueueCloseHandle(state.owner);
        return 2;
    }

    std::thread closer([&]
    {
        {
            std::unique_lock<std::mutex> lock(state.mutex);

            if (!state.cv.wait_for(
                    lock,
                    std::chrono::seconds(4),
                    [&] { return state.monitorEntered; }))
            {
                return;
            }
        }

        std::puts("TRACE: worker closing composite");
        XTaskQueueCloseHandle(state.composite);

        {
            std::lock_guard<std::mutex> lock(state.mutex);
            state.composite = nullptr;
            state.closeReturned = true;
        }

        std::puts("TRACE: worker close returned");
        state.cv.notify_all();
    });

    hr = XTaskQueueSubmitCallback(
        state.owner,
        XTaskQueuePort::Work,
        nullptr,
        NoopCallback);

    if (!Check(SUCCEEDED(hr), "submission succeeded")) ++failures;

    closer.join();

    if (!Check(state.monitorEntered, "monitor entered")) ++failures;
    if (!Check(state.closeReturned, "worker close returned")) ++failures;
    if (!Check(state.observedClose,
               "close completed while monitor was active")) ++failures;
    if (!Check(!state.unexpectedMonitor,
               "no unexpected monitor invocation")) ++failures;

    XTaskQueueUnregisterMonitor(state.owner, token);

    while (XTaskQueueDispatch(
        state.owner, XTaskQueuePort::Work, 0))
    {
    }

    if (state.composite != nullptr)
    {
        XTaskQueueCloseHandle(state.composite);
    }

    XTaskQueueCloseHandle(state.owner);

    if (!Check(XTaskQueueUninitialize(1000),
               "all task queue references released"))
    {
        ++failures;
    }

    std::printf(
        "RESULT: %s (%d failures)\n",
        failures == 0 ? "PASS" : "FAIL",
        failures);

    return failures == 0 ? 0 : 1;
}
