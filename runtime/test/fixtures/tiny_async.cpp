// A mapped caller submits through its imports; the shim calls back into this image twice,
// once on the worker and once on the thread manually dispatching completions.
#include <windows.h>
#include <XAsync.h>

#include <atomic>
#include <cstdio>

extern "C" __declspec(dllimport) HRESULT __stdcall XGameRuntimeInitialize();
extern "C" __declspec(dllimport) void __stdcall XGameRuntimeUninitialize();

namespace {
struct Observation {
    std::atomic<unsigned> work_calls{0};
    std::atomic<unsigned> completions{0};
    std::atomic<DWORD> worker_thread{0};
    std::atomic<DWORD> completion_thread{0};
    std::atomic<HRESULT> result{E_PENDING};
};
Observation observed;
XAsyncBlock operation{}; // Outlives callbacks even if the test times out.

HRESULT CALLBACK work(XAsyncBlock* block) {
    auto* state = static_cast<Observation*>(block->context);
    ++state->work_calls;
    state->worker_thread = GetCurrentThreadId();
    return E_ACCESSDENIED; // Distinguish the operation result from XAsyncRun accepting the request.
}

void CALLBACK completed(XAsyncBlock* block) {
    auto* state = static_cast<Observation*>(block->context);
    state->result = XAsyncGetStatus(block, false);
    state->completion_thread = GetCurrentThreadId();
    ++state->completions;
}

int fail(const char* stage, HRESULT hr) {
    std::printf("tiny_async: FAIL %s (0x%08lx)\n", stage, static_cast<unsigned long>(hr));
    return 1;
}

int run_operation() {
    XTaskQueueHandle queue = nullptr;
    HRESULT hr = XTaskQueueCreate(XTaskQueueDispatchMode::ThreadPool, XTaskQueueDispatchMode::Manual, &queue);
    if (hr != S_OK || !queue) return fail("queue creation", hr);

    operation.queue = queue;
    operation.context = &observed;
    operation.callback = completed;
    hr = XAsyncRun(&operation, work);
    if (hr != S_OK) {
        XTaskQueueCloseHandle(queue);
        return fail("submission", hr);
    }

    const auto start = GetTickCount64();
    while (!observed.completions && GetTickCount64() - start < 5000)
        XTaskQueueDispatch(queue, XTaskQueuePort::Completion, 50);
    // Drain any already queued duplicate, with the same bounded deadline.
    while (GetTickCount64() - start < 5000 && XTaskQueueDispatch(queue, XTaskQueuePort::Completion, 0)) {}
    XTaskQueueCloseHandle(queue);

    const DWORD caller = GetCurrentThreadId();
    std::printf("tiny_async: work=%u completion=%u worker=%lu dispatcher=%lu caller=%lu result=0x%08lx\n",
                observed.work_calls.load(), observed.completions.load(), observed.worker_thread.load(),
                observed.completion_thread.load(), caller, static_cast<unsigned long>(observed.result.load()));
    if (observed.work_calls != 1 || observed.worker_thread == caller ||
        observed.completions != 1 || observed.completion_thread != caller || observed.result != E_ACCESSDENIED)
        return 1;
    return 0;
}
} // namespace

int main() {
    const HRESULT hr = XGameRuntimeInitialize();
    if (hr != S_OK) return fail("runtime initialization", hr);
    const int result = run_operation();
    XGameRuntimeUninitialize();
    if (result != 0) return result;
    std::puts("tiny_async: PASS");
    std::fflush(stdout);
    return 0;
}
