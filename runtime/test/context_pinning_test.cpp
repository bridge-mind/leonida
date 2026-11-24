#include "pch.h"
#include "XTaskQueuePriv.h"
#include "TaskQueueP.h"
#include "referenced_ptr.h"
#include "TaskQueueImpl.h"

#include <cstdio>


static int failures = 0;

static void Check(bool condition, const char* description)
{
    std::printf("%s %s\n",
        condition ? "PASS" : "FAIL", description);

    if (!condition)
        ++failures;
}

int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    XTaskQueueHandle handle = nullptr;

    HRESULT hr = XTaskQueueCreate(
        XTaskQueueDispatchMode::Manual,
        XTaskQueueDispatchMode::Manual,
        &handle);

    Check(SUCCEEDED(hr) && handle != nullptr,
          "queue created");

    if (FAILED(hr) || handle == nullptr)
        return 2;

    ITaskQueue* owner = GetQueue(handle);

    for (XTaskQueuePort port :
         { XTaskQueuePort::Work, XTaskQueuePort::Completion })
    {
        ITaskQueuePortContext* context = nullptr;
        hr = owner->GetPortContext(port, &context);

        Check(SUCCEEDED(hr) && context != nullptr,
              "context acquired");

        if (FAILED(hr) || context == nullptr)
            continue;

        // Unsupported capabilities must clear the output pointer.
        void* unsupported = reinterpret_cast<void*>(1);
        HRESULT unsupportedHr = context->QueryApi(
            static_cast<ApiId>(0x7F000001), &unsupported);

        Check(unsupportedHr == E_NOINTERFACE &&
              unsupported == nullptr,
              "unsupported capability clears output");

        Check(context->QueryApi(
                  TaskQueueInternal::kPinnableContextApiId,
                  nullptr) == E_POINTER,
              "null output pointer rejected");

        void* queried = nullptr;

        hr = context->QueryApi(
            TaskQueueInternal::kPinnableContextApiId,
            &queried);

        Check(SUCCEEDED(hr) && queried != nullptr,
              "pinning capability available");

        if (SUCCEEDED(hr) && queried != nullptr)
        {
            auto* concrete =
                static_cast<TaskQueuePortContextImpl*>(queried);

            Check(
                static_cast<ITaskQueuePortContext*>(concrete)
                    == context,
                "capability identifies original context");

            Check(concrete->GetQueue() == owner,
                  "context preserves queue identity");

            bool pinned = concrete->TryPin();

            Check(pinned,
                  "live queue accepts temporary pin");

            if (pinned)
                concrete->Release();

            // Balance the reference acquired by QueryApi.
            concrete->Release();
        }

        // Balance GetPortContext.
        context->Release();
    }

    XTaskQueueCloseHandle(handle);

    Check(XTaskQueueUninitialize(1000),
          "all references released after closing queue");

    std::printf("RESULT: %s (%d failures)\n",
        failures == 0 ? "PASS" : "FAIL", failures);

    return failures == 0 ? 0 : 1;
}
