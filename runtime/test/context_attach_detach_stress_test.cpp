#include "pch.h"
#include "XTaskQueue.h"
#include "XTaskQueuePriv.h"

#include <atomic>
#include <cstdio>
#include <thread>

namespace
{
constexpr int Iterations = 1500;

struct State
{
    std::atomic<int> created{0};
    std::atomic<int> submitted{0};
    std::atomic<int> completed{0};
    std::atomic<int> monitors{0};
    std::atomic<int> failures{0};
    std::atomic<int> canceled{0};
};

void CALLBACK OnCallback(void* context, bool canceled)
{
    auto* state = static_cast<State*>(context);
    if (canceled)
    {
        ++state->canceled;
    }
    ++state->completed;
}

void CALLBACK OnMonitor(
    void* context,
    XTaskQueueHandle,
    XTaskQueuePort port)
{
    auto* state = static_cast<State*>(context);
    if (port != XTaskQueuePort::Work)
    {
        ++state->failures;
    }
    ++state->monitors;
}

bool Check(bool value, const char* message)
{
    std::printf("%s %s\n", value ? "PASS" : "FAIL", message);
    return value;
}
}

int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    State state;
    XTaskQueueHandle owner = nullptr;

    HRESULT hr = XTaskQueueCreate(
        XTaskQueueDispatchMode::Manual,
        XTaskQueueDispatchMode::Manual,
        &owner);

    if (!Check(SUCCEEDED(hr) && owner != nullptr,
               "owner queue created"))
    {
        return 2;
    }

    XTaskQueuePortHandle work = nullptr;
    XTaskQueuePortHandle completion = nullptr;

    HRESULT workHr = XTaskQueueGetPort(
        owner, XTaskQueuePort::Work, &work);

    HRESULT completionHr = XTaskQueueGetPort(
        owner, XTaskQueuePort::Completion, &completion);

    if (!Check(SUCCEEDED(workHr) &&
               SUCCEEDED(completionHr) &&
               work != nullptr && completion != nullptr,
               "owner ports acquired"))
    {
        XTaskQueueCloseHandle(owner);
        return 2;
    }

    XTaskQueueRegistrationToken token = {};

    hr = XTaskQueueRegisterMonitor(
        owner, &state, OnMonitor, &token);

    if (!Check(SUCCEEDED(hr), "monitor registered"))
    {
        XTaskQueueCloseHandle(owner);
        return 2;
    }

    std::atomic<bool> start{false};

    std::thread churn([&]
    {
        while (!start.load(std::memory_order_acquire))
        {
            std::this_thread::yield();
        }

        for (int i = 0; i < Iterations; ++i)
        {
            XTaskQueueHandle composite = nullptr;

            HRESULT result = XTaskQueueCreateComposite(
                work, completion, &composite);

            if (SUCCEEDED(result) && composite != nullptr)
            {
                ++state.created;
                XTaskQueueCloseHandle(composite);
            }
            else
            {
                ++state.failures;
            }

            if ((i & 7) == 0)
            {
                std::this_thread::yield();
            }
        }
    });

    std::thread submitter([&]
    {
        while (!start.load(std::memory_order_acquire))
        {
            std::this_thread::yield();
        }

        for (int i = 0; i < Iterations; ++i)
        {
            HRESULT result = XTaskQueueSubmitCallback(
                owner,
                XTaskQueuePort::Work,
                &state,
                OnCallback);

            if (SUCCEEDED(result))
            {
                ++state.submitted;
                XTaskQueueDispatch(
                    owner, XTaskQueuePort::Work, 0);
            }
            else
            {
                ++state.failures;
            }

            if ((i & 7) == 0)
            {
                std::this_thread::yield();
            }
        }
    });

    std::puts("TRACE: starting concurrent attach/detach and submissions");
    start.store(true, std::memory_order_release);

    churn.join();
    submitter.join();

    while (XTaskQueueDispatch(
        owner, XTaskQueuePort::Work, 0))
    {
    }

    XTaskQueueUnregisterMonitor(owner, token);

    int failures = 0;

    if (!Check(state.created.load() == Iterations,
               "all composites created and closed")) ++failures;

    if (!Check(state.submitted.load() == Iterations,
               "all callbacks submitted")) ++failures;

    if (!Check(state.completed.load() == Iterations,
               "all callbacks executed")) ++failures;

    if (!Check(state.monitors.load() == Iterations,
               "all monitor notifications received")) ++failures;

    if (!Check(state.failures.load() == 0,
               "no API or monitor failures")) ++failures;

    if (!Check(state.canceled.load() == 0,
               "no unexpected cancellation")) ++failures;

    std::printf(
        "TRACE: created=%d submitted=%d completed=%d monitors=%d\n",
        state.created.load(),
        state.submitted.load(),
        state.completed.load(),
        state.monitors.load());

    XTaskQueueCloseHandle(owner);

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
