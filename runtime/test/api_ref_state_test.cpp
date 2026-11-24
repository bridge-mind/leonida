#include <windows.h>
#include <ApiRefState.h>

#include <cstdio>
#include <atomic>
#include <thread>

static int failures = 0;

void Check(bool condition, const char* name)
{
    std::printf("%s %s\n",
        condition ? "PASS" : "FAIL", name);

    if (!condition)
        ++failures;
}

struct RaceContext
{
    ApiRefState state;
    HANDLE rundownEntered = nullptr;
    HANDLE allowRundownExit = nullptr;
};

DWORD WINAPI RundownThread(void* arg)
{
    auto& ctx = *static_cast<RaceContext*>(arg);

    if (ctx.state.AddRef() != 0)
        return 1;

    auto finalRelease = ctx.state.Release();

    if (!finalRelease.startRundown ||
        finalRelease.remaining != 0)
        return 2;

    // Simulate an internal AddRef during RundownObject().
    if (ctx.state.AddRef() != 0)
        return 3;

    SetEvent(ctx.rundownEntered);

    if (WaitForSingleObject(
        ctx.allowRundownExit, 3000) != WAIT_OBJECT_0)
        return 4;

    auto internalRelease = ctx.state.Release();

    if (internalRelease.startRundown ||
        internalRelease.remaining != 0)
        return 5;

    return 0;
}

void CheckPinWinsBeforeFinalRelease()
{
    constexpr int rounds = 200;
    int failuresLocal = 0;

    for (int i = 0; i < rounds; ++i)
    {
        ApiRefState state;
        state.AddRef();

        std::atomic<bool> pinFinished{ false };
        std::atomic<bool> allowPinRelease{ false };

        bool acquired = false;
        ApiRefState::ReleaseResult pinRelease{ 0, false };

        std::thread contender([&]
        {
            acquired = state.TryAddRef();

            pinFinished.store(true, std::memory_order_release);

            if (acquired)
            {
                while (!allowPinRelease.load(
                    std::memory_order_acquire))
                {
                    std::this_thread::yield();
                }

                pinRelease = state.Release();
            }
        });

        while (!pinFinished.load(std::memory_order_acquire))
        {
            std::this_thread::yield();
        }

        const auto ownerRelease = state.Release();

        bool valid =
            acquired &&
            ownerRelease.remaining == 1 &&
            !ownerRelease.startRundown &&
            state.RefCount() == 1 &&
            !state.IsDying();

        allowPinRelease.store(true, std::memory_order_release);

        contender.join();

        valid =
            valid &&
            pinRelease.remaining == 0 &&
            pinRelease.startRundown &&
            state.RefCount() == 0 &&
            state.IsDying() &&
            !state.TryAddRef();

        if (!valid)
        {
            ++failuresLocal;
        }
    }

    Check(failuresLocal == 0,
          "pin acquired before final owner Release stays valid");

    std::printf(
        "forced pin-first rounds: %d, failures: %d\n",
        rounds, failuresLocal);
}
void CheckFinalReleaseRace()
{
    constexpr int rounds = 1000;
    int badRundown = 0;
    int badFinalState = 0;
    int successfulPins = 0;

    for (int i = 0; i < rounds; ++i)
    {
        ApiRefState state;
        state.AddRef();

        std::atomic<bool> ready{ false };
        std::atomic<bool> go{ false };
        std::atomic<int> rundownCount{ 0 };
        std::atomic<bool> pinned{ false };

        std::thread contender([&]
        {
            ready.store(true, std::memory_order_release);

            while (!go.load(std::memory_order_acquire))
                std::this_thread::yield();

            if (state.TryAddRef())
            {
                pinned.store(true, std::memory_order_relaxed);

                if (state.Release().startRundown)
                    rundownCount.fetch_add(1);
            }
        });

        while (!ready.load(std::memory_order_acquire))
            std::this_thread::yield();

        go.store(true, std::memory_order_release);

        if (state.Release().startRundown)
            rundownCount.fetch_add(1);

        contender.join();

        if (pinned.load(std::memory_order_relaxed))
            ++successfulPins;

        if (rundownCount.load() != 1)
            ++badRundown;

        bool unexpectedPin = state.TryAddRef();

        if (unexpectedPin)
            state.Release();

        if (state.RefCount() != 0 ||
            !state.IsDying() ||
            unexpectedPin)
        {
            ++badFinalState;
        }
    }

    Check(badRundown == 0,
          "racing final Release starts rundown exactly once");

    Check(badFinalState == 0,
          "racing final Release preserves terminal state");

    std::printf(
        "race rounds: %d, successful pins: %d, rejected pins: %d\n",
        rounds, successfulPins, rounds - successfulPins);
}
int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    std::puts("api_ref_state_test");

    ApiRefState state;

    Check(!state.TryAddRef(),
          "cannot pin zero-ref object");

    Check(state.AddRef() == 0,
          "initial AddRef");

    Check(state.TryAddRef(),
          "can pin live object");

    Check(state.RefCount() == 2,
          "two references held");

    auto first = state.Release();

    Check(first.remaining == 1 &&
          !first.startRundown,
          "non-final Release");

    auto last = state.Release();

    Check(last.remaining == 0 &&
          last.startRundown,
          "final Release starts rundown");

    Check(state.IsDying(),
          "dying marker is permanent");

    Check(!state.TryAddRef(),
          "cannot pin after final Release");

    RaceContext race;

    race.rundownEntered =
        CreateEventW(nullptr, TRUE, FALSE, nullptr);

    race.allowRundownExit =
        CreateEventW(nullptr, TRUE, FALSE, nullptr);

    if (!race.rundownEntered || !race.allowRundownExit)
        return 2;

    HANDLE thread = CreateThread(
        nullptr, 0, RundownThread, &race, 0, nullptr);

    if (!thread)
        return 3;

    DWORD entered = WaitForSingleObject(
        race.rundownEntered, 3000);

    Check(entered == WAIT_OBJECT_0,
          "rundown thread reached checkpoint");

    if (entered == WAIT_OBJECT_0)
    {
        Check(race.state.IsDying(),
              "dying bit survives internal AddRef");

        Check(race.state.RefCount() == 1,
              "temporary internal reference exists");

        bool rejected = true;

        for (int i = 0; i < 10000; ++i)
        {
            if (race.state.TryAddRef())
            {
                rejected = false;
                race.state.Release();
                break;
            }
        }

        Check(rejected,
              "concurrent pins rejected during rundown");
    }

    SetEvent(race.allowRundownExit);

    DWORD joined = WaitForSingleObject(thread, 3000);

    Check(joined == WAIT_OBJECT_0,
          "rundown thread completed");

    if (joined == WAIT_OBJECT_0)
    {
        DWORD exitCode = 1;
        GetExitCodeThread(thread, &exitCode);

        Check(exitCode == 0,
              "internal Release did not restart rundown");
    }

    CloseHandle(thread);
    CloseHandle(race.rundownEntered);
    CloseHandle(race.allowRundownExit);

    CheckPinWinsBeforeFinalRelease();
    CheckFinalReleaseRace();

    std::printf("RESULT: %s (%d failures)\n",
        failures == 0 ? "PASS" : "FAIL", failures);

    return failures == 0 ? 0 : 1;
}
