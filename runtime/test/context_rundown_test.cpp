#include "pch.h"
#include "referenced_ptr.h"
#include "XTaskQueuePriv.h"
#include "TaskQueueP.h"
#include "TaskQueueImpl.h"

#include <atomic>
#include <cstdio>
#include <new>
#include <thread>

struct Results
{
    std::atomic<int> rundowns{0};
    std::atomic<int> destructors{0};
    std::atomic<int> rejected{0};
    std::atomic<int> errors{0};
};

class ProbeQueue final : public TaskQueueImpl
{
public:
    explicit ProbeQueue(Results& results) : results_(results) {}

    ~ProbeQueue() override
    {
        ++results_.destructors;
    }

    void SetContext(TaskQueuePortContextImpl* context)
    {
        context_ = context;
    }

protected:
    void RundownObject() override
    {
        ++results_.rundowns;

        auto tryRejected = [this]()
        {
            if (context_->TryPin())
            {
                ++results_.errors;
                context_->Release();
            }
            else
            {
                ++results_.rejected;
            }
        };

        // The final Release has already marked the queue as dying.
        tryRejected();

        // Internal AddRef must not clear the dying state.
        AddRef();
        tryRejected();

        // Both phases execute on a second thread while the
        // rundown stack guarantees the object's lifetime.
        std::atomic<int> phase{0};

        std::thread worker([&]()
        {
            while (phase.load() != 1)
                std::this_thread::yield();

            for (int i = 0; i < 1000; ++i)
                tryRejected();

            phase.store(2);

            while (phase.load() != 3)
                std::this_thread::yield();

            for (int i = 0; i < 1000; ++i)
                tryRejected();

            phase.store(4);
        });

        phase.store(1);

        while (phase.load() != 2)
            std::this_thread::yield();

        // Drop the temporary reference while still in rundown.
        Release();
        phase.store(3);

        while (phase.load() != 4)
            std::this_thread::yield();

        worker.join();
        tryRejected();

        TaskQueueImpl::RundownObject();
    }

private:
    Results& results_;
    TaskQueuePortContextImpl* context_ = nullptr;
};

int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    Results results;
    auto* queue = new (std::nothrow) ProbeQueue(results);

    if (!queue)
        return 2;

    queue->AddRef();

    ITaskQueuePortContext* context = nullptr;
    HRESULT hr = queue->GetPortContext(
        XTaskQueuePort::Work, &context);

    if (FAILED(hr) || !context)
    {
        std::puts("FAIL acquire context");
        queue->Release();
        return 3;
    }

    void* raw = nullptr;
    hr = context->QueryApi(
        TaskQueueInternal::kPinnableContextApiId, &raw);

    if (FAILED(hr) || !raw)
    {
        std::puts("FAIL query pinning capability");
        context->Release();
        queue->Release();
        return 4;
    }

    auto* concrete =
        static_cast<TaskQueuePortContextImpl*>(raw);

    // Release QueryApi and GetPortContext references.
    // The original queue reference keeps the context alive.
    concrete->Release();
    context->Release();

    queue->SetContext(concrete);

    // This triggers RundownObject and destroys the queue.
    // Do not access queue or concrete after this call.
    queue->Release();

    bool passed =
        results.rundowns.load() == 1 &&
        results.destructors.load() == 1 &&
        results.rejected.load() == 2003 &&
        results.errors.load() == 0;

    bool zeroRefs = XTaskQueueUninitialize(1000);

    std::printf(
        "rundowns=%d destructors=%d rejected=%d errors=%d\n",
        results.rundowns.load(),
        results.destructors.load(),
        results.rejected.load(),
        results.errors.load());

    std::printf("uninitialize=%s\n",
        zeroRefs ? "PASS" : "FAIL");

    std::printf("RESULT: %s\n",
        passed && zeroRefs ? "PASS" : "FAIL");

    return passed && zeroRefs ? 0 : 1;
}
