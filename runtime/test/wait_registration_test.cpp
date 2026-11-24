#include "pch.h"
#include "XTaskQueuePriv.h"
#include "TaskQueueP.h"
#include "TaskQueueImpl.h"

#include <cstdlib>
#include <string_view>
#include <barrier>
#ifdef _DEBUG
#include <crtdbg.h>
#endif

namespace
{
constexpr auto work = XTaskQueuePort::Work;
constexpr auto manual = XTaskQueueDispatchMode::Manual;
constexpr auto immediate = XTaskQueueDispatchMode::Immediate;

void Check(bool success, const char* step)
{
    std::fprintf(stderr, "%s %s\n", success ? "PASS" : "FAIL", step);
    if (!success) std::exit(1);
}

struct Waiter
{
    explicit Waiter(const wchar_t* name = nullptr) :
        signal(CreateEventW(nullptr, FALSE, FALSE, name)) {}

    XTaskQueueHandle queue{};
    XTaskQueueRegistrationToken token{};
    HANDLE signal;
    HANDLE published = CreateEvent(nullptr, TRUE, FALSE, nullptr);
    HANDLE returned = CreateEvent(nullptr, TRUE, FALSE, nullptr);
    std::atomic<unsigned> calls{0};
    std::atomic<unsigned> canceled{0};
    bool unregisterInCallback = false;
    bool unregisterInMonitor = false;
    bool closeInCallback = false;
    bool holdCancellation = false;
    HANDLE allowExit = nullptr;
    std::atomic<bool> leavingCallback{false};
    Waiter* unregisterTarget = nullptr;
    Waiter* peer = nullptr;
    std::barrier<>* bothEntered = nullptr;

    void Register()
    {
        Check(signal && published && returned, "events created");
        Check(SUCCEEDED(XTaskQueueRegisterWaiter(queue, work, signal, this,
            [](void* context, bool cancel)
            {
                auto& w = *static_cast<Waiter*>(context);
                ++w.calls;
                if (cancel) ++w.canceled;
                if (w.peer)
                {
                    w.bothEntered->arrive_and_wait();
                    XTaskQueueUnregisterWaiter(w.peer->queue, w.peer->token);
                    SetEvent(w.returned);
                }
                if (w.unregisterInCallback)
                {
                    Check(w.token.token != 0, "public token visible inside callback");
                    XTaskQueueUnregisterWaiter(w.queue, w.token);
                    SetEvent(w.returned);
                    if (w.allowExit)
                    {
                        Check(WaitForSingleObject(w.allowExit, 5000) == WAIT_OBJECT_0, "release running callback");
                        w.leavingCallback = true;
                    }
                }
                if (w.unregisterTarget)
                {
                    auto& target = *w.unregisterTarget;
                    XTaskQueueUnregisterWaiter(target.queue, target.token);
                    Check(target.calls == 1, "reentrant unregister observed physical entry");
                    SetEvent(w.returned);
                }
                if (w.closeInCallback)
                {
                    XTaskQueueCloseHandle(w.queue);
                    SetEvent(w.returned);
                }
            }, &token)), "waiter token published");
    }

    void CloseEvents()
    {
        if (signal) CloseHandle(signal);
        CloseHandle(published);
        CloseHandle(returned);
    }
};

void CALLBACK Monitor(void* context, XTaskQueueHandle, XTaskQueuePort port)
{
    if (port != work) return;
    auto& w = *static_cast<Waiter*>(context);
    if (w.unregisterInMonitor)
    {
        XTaskQueueUnregisterWaiter(w.queue, w.token);
        SetEvent(w.returned);
    }
    SetEvent(w.published);
    if (w.holdCancellation)
        Check(WaitForSingleObject(w.returned, 5000) == WAIT_OBJECT_0, "cancellation monitor released");
}

void Await(HANDLE event, const char* step)
{
    Check(WaitForSingleObject(event, 5000) == WAIT_OBJECT_0, step);
}

template<typename Predicate>
void AwaitCondition(Predicate predicate, const char* step)
{
    const auto deadline = GetTickCount64() + 5000;
    while (!predicate())
    {
        if (GetTickCount64() >= deadline) Check(false, step);
        std::this_thread::yield();
    }
    Check(true, step);
}
}

struct WaitRegistrationTestAccess
{
    static TaskQueuePortImpl& Port(Waiter& waiter)
    {
        XTaskQueuePortHandle handle{};
        Check(SUCCEEDED(XTaskQueueGetPort(waiter.queue, work, &handle)), "local work port");
        return *static_cast<TaskQueuePortImpl*>(handle->m_port);
    }

    static void HoldSnapshot(Waiter& waiter, const wchar_t* eventName)
    {
        auto& port = Port(waiter);
        {
            TaskQueuePortImpl::WaitSnapshot snapshot;
            port.CaptureWaitSnapshot(snapshot);
            Check(snapshot.registrations.count() == 1 && snapshot.epoch, "snapshot includes waiter and epoch");
            std::thread retire([&] { XTaskQueueUnregisterWaiter(waiter.queue, waiter.token); });
            retire.join();
            CloseHandle(waiter.signal);
            waiter.signal = nullptr;
            Check(SetEvent(snapshot.events[1]), "snapshot owns a usable handle after retirement");
            Check(WaitForSingleObject(snapshot.events[1], 0) == WAIT_OBJECT_0, "old snapshot consumed signal");
            port.ProcessWaitCallback(snapshot.registrations[0].get());
        }
        Check(!XTaskQueueDispatch(waiter.queue, work, 0) && waiter.calls == 0, "old snapshot cannot deliver");
        HANDLE remaining = OpenEventW(SYNCHRONIZE, FALSE, eventName);
        Check(!remaining && GetLastError() == ERROR_FILE_NOT_FOUND, "last snapshot released event");
    }

    static void CheckAdmission(Waiter& waiter, bool reentrant)
    {
        auto& port = Port(waiter);
        struct Status : OS::ThreadPoolActionStatus
        {
            HANDLE admitted = CreateEvent(nullptr, TRUE, FALSE, nullptr);
            HANDLE proceed = CreateEvent(nullptr, TRUE, FALSE, nullptr);
            void Complete() override {}
            void MayRunLong() override
            {
                SetEvent(admitted);
                Await(proceed, "allowed to enter user code");
            }
            ~Status() { CloseHandle(admitted); CloseHandle(proceed); }
        } status;
        Check(SetEvent(waiter.signal), "waiter signaled");
        Await(waiter.published, "ready for dispatch");
        TaskQueuePortImpl::WaitSnapshot snapshot;
        port.CaptureWaitSnapshot(snapshot);
        std::thread dispatch([&] { Check(port.DrainOneItem(status), "admitted entry dispatched"); });
        Await(status.admitted, "admitted before indirect call");
        Waiter caller;
        if (reentrant)
        {
            caller.unregisterTarget = &waiter;
            Check(SUCCEEDED(XTaskQueueCreate(manual, immediate, &caller.queue)), "reentrant caller queue");
            caller.Register();
            Check(SetEvent(caller.signal), "reentrant caller signaled");
        }
        std::thread retire([&] {
            if (reentrant)
            {
                Check(XTaskQueueDispatch(caller.queue, work, 5000), "reentrant caller dispatched");
                Await(caller.returned, "reentrant unregister returned");
            }
            else XTaskQueueUnregisterWaiter(waiter.queue, waiter.token);
            Check(waiter.calls == 1, "physical entry preceded unregister return");
        });
        {
            std::unique_lock<std::mutex> lock(port.m_lock);
            Check(port.m_waitFinished.wait_for(lock, std::chrono::seconds(5),
                [&] { return snapshot.registrations[0]->drained; }), "native drainage complete");
            Check(waiter.calls == 0, "admission alone did not invoke user");
        }
        SetEvent(status.proceed);
        dispatch.join();
        retire.join();
        if (reentrant)
        {
            XTaskQueueUnregisterWaiter(caller.queue, caller.token);
            XTaskQueueCloseHandle(caller.queue);
        }
        caller.CloseEvents();
    }

    static void ExternalAfterSelf(Waiter& w)
    {
        auto& port = Port(w);
        TaskQueuePortImpl::WaitSnapshot snapshot;
        port.CaptureWaitSnapshot(snapshot);
        auto* reg = snapshot.registrations[0].get();
        auto registry = reg->publicRegistry;
        w.unregisterInCallback = true;
        w.allowExit = CreateEvent(nullptr, TRUE, FALSE, nullptr);
        Check(w.allowExit != nullptr, "exit barrier created");
        // Publish without a native worker so the reference baseline is stable.
        port.ProcessWaitCallback(reg);
        std::thread dispatch([&] { Check(XTaskQueueDispatch(w.queue, work, 0), "callback dispatched"); });
        Await(w.returned, "self-unregister returned while callback is held");
        Check(registry->Lookup(w.token).PortToken == reg->token, "public token still resolves after self-unregister");
        const auto refs = reg->refs.load();
        std::thread external([&] {
            XTaskQueueUnregisterWaiter(w.queue, w.token);
            Check(w.leavingCallback.load(), "external caller waited for callback");
            std::lock_guard<std::mutex> guard(port.m_lock);
            Check(!reg->running, "dispatch recorded physical return before external caller completed");
        });
        AwaitCondition([&] { return reg->refs.load() > refs; }, "external caller acquired retired registration via public token");
        SetEvent(w.allowExit);
        dispatch.join();
        external.join();
        Check(!port.m_waits.count(reg->token), "port token retired after callback");
        Check(registry->Lookup(w.token).PortToken == 0, "public token retired after callback");
        CloseHandle(w.allowExit);
    }

    static void TwoDispatchers(Waiter& w, bool replace)
    {
        auto& port = Port(w);
        if (!replace) XTaskQueueUnregisterWaiter(w.queue, w.token);
        // Drain the setup wake so both callers can stay in the same infinite wait.
        Check(!XTaskQueueDispatch(w.queue, work, 0), "no initial work");
        std::atomic<unsigned> deliveries{0};
        auto dispatch = [&] {
            if (XTaskQueueDispatch(w.queue, work, INFINITE)) ++deliveries;
        };
        std::thread first(dispatch), second(dispatch);
        auto awaitPair = [&] {
            std::weak_ptr<TaskQueuePortImpl::WaitHandleOwner> result;
            AwaitCondition([&] {
                std::lock_guard<std::mutex> lock(port.m_lock);
                if (port.m_waitEpoch.use_count() != 3) return false;
                result = port.m_waitEpoch;
                return true;
            }, "two real dispatchers own snapshots of the current epoch");
            return result;
        };
        auto epoch = awaitPair();
        // Also verify the signal on the exact kernel object. Merely seeing
        // both threads return could pass because of unrelated work wakes.
        auto oldEpoch = epoch.lock();
        Check(oldEpoch != nullptr, "captured live epoch for both dispatchers");
        auto verifyBroadcast = [&] {
            Check(WaitForSingleObject(oldEpoch->handle, 0) == WAIT_OBJECT_0 &&
                  WaitForSingleObject(oldEpoch->handle, 0) == WAIT_OBJECT_0,
                  "retired epoch stays signaled for late observers");
            oldEpoch.reset();
        };
        Waiter replacement;
        if (replace)
        {
            XTaskQueueUnregisterWaiter(w.queue, w.token);
            auto emptyEpoch = awaitPair();
            verifyBroadcast();
            Check(epoch.expired(), "both dispatchers released snapshots containing original waiter");
            oldEpoch = emptyEpoch.lock();
            replacement.queue = w.queue;
            replacement.Register();
            epoch = awaitPair();
            verifyBroadcast();
            Check(emptyEpoch.expired(), "both dispatchers captured replacement");
            oldEpoch = epoch.lock();
            port.SuspendPort();
            auto suspendedEpoch = awaitPair();
            verifyBroadcast();
            Check(epoch.expired(), "both dispatchers renewed on suspension");
            epoch = suspendedEpoch;
            Check(SetEvent(replacement.signal), "replacement signaled");
            AwaitCondition([&] { return port.m_readyEntries.load() == 1; }, "replacement entry queued while suspended");
            Check(deliveries == 0, "suspended port did not dispatch");
            port.ResumePort();
            AwaitCondition([&] { return deliveries.load() == 1; }, "one dispatcher delivered replacement");
            XTaskQueueUnregisterWaiter(w.queue, replacement.token);
        }
        Check(SUCCEEDED(XTaskQueueTerminate(w.queue, false, w.returned,
            [](void* event) { SetEvent(static_cast<HANDLE>(event)); })), "terminate manual port");
        first.join();
        second.join();
        Await(w.returned, "termination completed");
        if (!replace) verifyBroadcast();
        Check(epoch.expired() && w.calls == 0 && replacement.calls == (replace ? 1u : 0u)
            && deliveries == replacement.calls, "both infinite waits ended with expected deliveries");
        replacement.CloseEvents();
    }
};

int main(int argc, char** argv)
{
    if (argc != 2) return 1;
    std::string_view scenario(argv[1]);
    PROCESS_MITIGATION_STRICT_HANDLE_CHECK_POLICY strict{};
    strict.RaiseExceptionOnInvalidHandleReference = 1;
    strict.HandleExceptionsPermanentlyEnabled = 1;
    Check(SetProcessMitigationPolicy(ProcessStrictHandleCheckPolicy, &strict, sizeof(strict)),
        "strict Win32 handle checks enabled");
    wchar_t eventName[80]{};
    swprintf_s(eventName, L"Local\\LeonidaWaitRegistration-%lu", GetCurrentProcessId());
    Waiter w(eventName);
    w.unregisterInCallback = scenario == "self" || scenario == "presignaled";
    w.unregisterInMonitor = scenario == "monitor";
    w.closeInCallback = scenario == "close";
    w.holdCancellation = scenario == "cancel_unregister";
    Check(SUCCEEDED(XTaskQueueCreate(w.unregisterInCallback || w.closeInCallback ? immediate : manual,
        immediate, &w.queue)), "queue created");

    XTaskQueueRegistrationToken monitor{};
    Check(SUCCEEDED(XTaskQueueRegisterMonitor(w.queue, &w, Monitor, &monitor)), "monitor registered");
    if (scenario == "presignaled") Check(SetEvent(w.signal), "event signaled before registration");
    w.Register();

    if (w.holdCancellation)
    {
        HANDLE terminated = CreateEvent(nullptr, TRUE, FALSE, nullptr);
        Check(terminated != nullptr, "termination event created");
        std::thread cancel([&] {
            Check(SUCCEEDED(XTaskQueueTerminate(w.queue, false, terminated,
                [](void* event) { SetEvent(static_cast<HANDLE>(event)); })), "termination requested");
        });
        Await(w.published, "canceled entry published");
        XTaskQueueUnregisterWaiter(w.queue, w.token);
        SetEvent(w.returned);
        cancel.join();
        Check(!XTaskQueueDispatch(w.queue, work, 0), "suppressed canceled delivery");
        Await(terminated, "termination completed after unregister");
        Check(w.calls == 0, "no late canceled callback");
        CloseHandle(terminated);
    }
    else if (scenario == "held_snapshot") WaitRegistrationTestAccess::HoldSnapshot(w, eventName);
    else if (scenario == "admission" || scenario == "reentrant_admission")
        WaitRegistrationTestAccess::CheckAdmission(w, scenario == "reentrant_admission");
    else if (scenario == "reentrant_vs_external_unregister") WaitRegistrationTestAccess::ExternalAfterSelf(w);
    else if (scenario == "multi_dispatcher_terminate" || scenario == "multi_dispatcher_unregister_replace")
        WaitRegistrationTestAccess::TwoDispatchers(w, scenario == "multi_dispatcher_unregister_replace");
    else if (scenario == "cross")
    {
        Waiter peer;
        Check(SUCCEEDED(XTaskQueueCreate(manual, immediate, &peer.queue)), "second manual port");
        peer.Register();
        std::barrier entered(2);
        w.bothEntered = peer.bothEntered = &entered;
        w.peer = &peer;
        peer.peer = &w;
        Check(SetEvent(w.signal) && SetEvent(peer.signal), "both waiters signaled");
        std::thread dispatcher([&] { Check(XTaskQueueDispatch(peer.queue, work, 5000), "peer dispatched"); });
        Check(XTaskQueueDispatch(w.queue, work, 5000), "owner dispatched");
        dispatcher.join();
        Await(w.returned, "first cross-unregister returned");
        Await(peer.returned, "second cross-unregister returned");
        Check(w.calls == 1 && peer.calls == 1, "single delivery per waiter");
        XTaskQueueCloseHandle(peer.queue);
        peer.CloseEvents();
    }
    else if (scenario == "concurrent")
    {
        Check(SetEvent(w.signal), "waiter signaled");
        Await(w.published, "entry published");
        std::barrier start(2);
        auto retire = [&] {
            start.arrive_and_wait();
            XTaskQueueUnregisterWaiter(w.queue, w.token);
        };
        std::thread other(retire);
        retire();
        other.join();
        CloseHandle(w.signal);
        w.signal = nullptr;
        HANDLE remaining = OpenEventW(SYNCHRONIZE, FALSE, eventName);
        Check(!remaining && GetLastError() == ERROR_FILE_NOT_FOUND, "tombstone released event");
        Check(XTaskQueueIsEmpty(w.queue, work), "tombstone logically empty");
        Check(SUCCEEDED(XTaskQueueTerminate(w.queue, false, w.returned,
            [](void* event) { SetEvent(static_cast<HANDLE>(event)); })), "terminate with tombstone");
        Check(!XTaskQueueDispatch(w.queue, work, 0), "tombstone is not a delivery");
        Await(w.returned, "termination completed");
        Check(w.calls == 0, "suppressed callback stayed suppressed");
    }
    else if (scenario == "snapshot")
    {
        // Exercise both the registration wake and timeout against a live waiter.
        for (unsigned i = 0; i != 128; ++i)
            if (XTaskQueueDispatch(w.queue, work, 0)) return 1;
        Check(w.calls == 0, "polls timed out without delivery");
        XTaskQueueUnregisterWaiter(w.queue, w.token);
        CloseHandle(w.signal);
        w.signal = nullptr;
        HANDLE remaining = OpenEventW(SYNCHRONIZE, FALSE, eventName);
        Check(!remaining && GetLastError() == ERROR_FILE_NOT_FOUND, "last handle to snapshot event released");
    }
    else if (scenario == "composite")
    {
        XTaskQueuePortHandle shared{}, completion{};
        Check(SUCCEEDED(XTaskQueueGetPort(w.queue, work, &shared)) &&
            SUCCEEDED(XTaskQueueGetPort(w.queue, XTaskQueuePort::Completion, &completion)), "ports obtained");
        Waiter sibling;
        Check(SUCCEEDED(XTaskQueueCreateComposite(shared, completion, &sibling.queue)), "composite created");
        sibling.Register();
        Check(SUCCEEDED(XTaskQueueTerminate(sibling.queue, false, nullptr, nullptr)), "composite terminating");
        Check(XTaskQueueDispatch(w.queue, work, 0) && sibling.canceled == 1, "composite waiter canceled");
        Check(SetEvent(w.signal) != FALSE, "owner waiter signaled");
        Check(XTaskQueueDispatch(w.queue, work, 5000) && w.calls == 1 && w.canceled == 0, "owner waiter still active");
        XTaskQueueUnregisterWaiter(w.queue, w.token);
        XTaskQueueCloseHandle(sibling.queue);
        sibling.CloseEvents();
    }
    else
    {
        if (scenario != "parked")
        {
            if (scenario != "presignaled")
                Check(SetEvent(w.signal) != FALSE, "signaled after token publication");
            Await(w.published, "entry published on the work port");
        }
        if (scenario == "terminate" || scenario == "parked")
        {
            Check(SUCCEEDED(XTaskQueueTerminate(w.queue, false, nullptr, nullptr)), "termination initiated");
            Check(XTaskQueueDispatch(w.queue, work, 0), "cancellation dispatched");
            Check(w.calls == 1 && w.canceled == 1, "single canceled callback");
            Check(!XTaskQueueDispatch(w.queue, work, 0), "no second delivery");
        }
        else if (w.closeInCallback)
        {
            Await(w.returned, "last external queue handle closed in callback");
            Check(w.calls == 1, "one callback before rundown");
        }
        else if (w.unregisterInCallback || w.unregisterInMonitor)
        {
            Await(w.returned, "reentrant unregister returned");
            XTaskQueueDispatch(w.queue, work, 0);
            Check(w.calls == (w.unregisterInCallback ? 1u : 0u), "expected user callback count");
        }
        else if (scenario == "unregister" || scenario == "undispatched")
        {
            XTaskQueueUnregisterWaiter(w.queue, w.token);
            CloseHandle(w.signal);
            w.signal = nullptr;
            if (scenario == "unregister") XTaskQueueDispatch(w.queue, work, 0);
            Check(w.calls == 0, "no user callback after unregister");
        }
        else return 1;
    }

    if (!w.closeInCallback)
    {
        XTaskQueueUnregisterMonitor(w.queue, monitor);
        XTaskQueueCloseHandle(w.queue);
    }
    Check(XTaskQueueUninitialize(5000), "all queue and port references released");
    w.CloseEvents();
#ifdef _DEBUG
    Check(_CrtCheckMemory() != 0, "debug CRT heap intact after shutdown");
#endif
    return 0;
}
