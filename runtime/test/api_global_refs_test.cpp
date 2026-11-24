#include <windows.h>
#include <XTaskQueue.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>

// Internal API implemented by the vendored xtaskqueue runtime.
extern "C" bool __stdcall XTaskQueueUninitialize(
    uint32_t timeoutMilliseconds) noexcept;

static int failures = 0;

void Check(bool condition, const char* name)
{
    std::printf("%s %s\n",
        condition ? "PASS" : "FAIL", name);

    if (!condition)
        ++failures;
}

int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::puts("api_global_refs_test");

    XTaskQueueHandle owner = nullptr;
    XTaskQueueHandle duplicate = nullptr;

    HRESULT hr = XTaskQueueCreate(
        XTaskQueueDispatchMode::Manual,
        XTaskQueueDispatchMode::Manual,
        &owner);

    Check(SUCCEEDED(hr) && owner != nullptr,
          "create task queue");

    if (FAILED(hr) || owner == nullptr)
        return 2;

    hr = XTaskQueueDuplicateHandle(owner, &duplicate);

    Check(SUCCEEDED(hr) && duplicate != nullptr,
          "duplicate task queue handle");

    if (FAILED(hr) || duplicate == nullptr)
    {
        XTaskQueueCloseHandle(owner);
        return 3;
    }

    XTaskQueueCloseHandle(owner);

    Check(!XTaskQueueUninitialize(0),
          "uninitialize rejects outstanding handle");

    XTaskQueueCloseHandle(duplicate);

    Check(XTaskQueueUninitialize(1000),
          "uninitialize succeeds after final close");

    owner = nullptr;
    duplicate = nullptr;

    hr = XTaskQueueCreate(
        XTaskQueueDispatchMode::Manual,
        XTaskQueueDispatchMode::Manual,
        &owner);

    Check(SUCCEEDED(hr) && owner != nullptr,
          "create queue for concurrent shutdown");

    if (FAILED(hr) || owner == nullptr)
        return 4;

    hr = XTaskQueueDuplicateHandle(owner, &duplicate);

    Check(SUCCEEDED(hr) && duplicate != nullptr,
          "duplicate handle for concurrent shutdown");

    if (FAILED(hr) || duplicate == nullptr)
    {
        XTaskQueueCloseHandle(owner);
        return 5;
    }

    XTaskQueueCloseHandle(owner);

    std::atomic<bool> start{ false };

    std::thread closer([&]
    {
        while (!start.load(std::memory_order_acquire))
            std::this_thread::yield();

        std::this_thread::sleep_for(
            std::chrono::milliseconds(75));

        XTaskQueueCloseHandle(duplicate);
    });

    start.store(true, std::memory_order_release);

    bool shutdownCompleted = XTaskQueueUninitialize(2000);

    closer.join();

    Check(shutdownCompleted,
          "uninitialize waits for concurrent final close");

    Check(XTaskQueueUninitialize(1000),
          "no outstanding references after shutdown");

    std::printf("RESULT: %s (%d failures)\n",
        failures == 0 ? "PASS" : "FAIL",
        failures);

    return failures == 0 ? 0 : 1;
}
