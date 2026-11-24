// Copyright (c) Microsoft Corporation
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

#pragma once

#include "ApiRefState.h"
#include "referenced_ptr.h"
#include "AtomicVector.h"
#include <map>
#include "LocklessQueue.h"
#include "StaticArray.h"
#include "ThreadPool.h"
#include "WaitTimer.h"

#ifdef SUSPEND_API
#include "SuspendState.h"
#endif

SYSTEM_HANDLE_DEFINE_HELPERS(XTaskQueueHandle, XSystemHandleType::TaskQueue);

namespace TaskQueueInternal
{
    // Private capability identifier; keep the shared ApiId enum unchanged.
    constexpr ApiId kPinnableContextApiId =
        static_cast<ApiId>(0x50494E31);
    constexpr ApiId kWaitPublicationApiId = static_cast<ApiId>(0x57505432);
}

namespace ApiRefs
{
    void GlobalAddRef();
    void GlobalRelease();

#ifdef HC_UNITTEST_API
    // Unit-test diagnostics: tracks live Api<> objects by type so a failed
    // XTaskQueueUninitialize can report what leaked.
    void TrackLiveObject(ApiId id, int32_t delta) noexcept;
#endif
}

template <ApiId iid, typename TInterface>
class Api : public TInterface
{
public:

#ifdef HC_UNITTEST_API
    Api() noexcept
    {
        ApiRefs::TrackLiveObject(iid, 1);
    }
#endif

    virtual ~Api() 
    {
#ifdef HC_UNITTEST_API
        ApiRefs::TrackLiveObject(iid, -1);
#endif
    }
    
    uint32_t __stdcall AddRef()
    {
        ApiRefs::GlobalAddRef();
        return m_refState.AddRef();
    }

    bool TryAddRef() noexcept
    {
        // Account globally before publishing the local reference.
        ApiRefs::GlobalAddRef();

        if (m_refState.TryAddRef())
        {
            return true;
        }

        ApiRefs::GlobalRelease();
        return false;
    }

    uint32_t __stdcall Release()
    {
        const auto result = m_refState.Release();

        if (result.startRundown)
        {
            // Keep the global reference count nonzero during teardown.
            ApiRefs::GlobalAddRef();
            RundownObject();
            delete this;
            ApiRefs::GlobalRelease();
        }

        ApiRefs::GlobalRelease();
        return result.remaining;
    }

    HRESULT __stdcall QueryApi(ApiId id, void** ptr)
    {
        if (ptr == nullptr)
        {
            return E_POINTER;
        }

        *ptr = QueryApiImpl(id);

        if ((*ptr) != nullptr)
        {
            AddRef();
            return S_OK;
        }

        return E_NOINTERFACE;
    }

protected:

    virtual void* QueryApiImpl(ApiId id) 
    {
        if (id == ApiId::Identity || id == iid)
        {
            return static_cast<TInterface*>(this);
        }
        return nullptr;
    }

    // Called when the object is about to be deleted.
    virtual void RundownObject()
    {
    }

private:
    ApiRefState m_refState;
};

static uint32_t const SUBMIT_CALLBACK_MAX = 32;

class SubmitCallback
{
public:

    SubmitCallback(_In_ XTaskQueueHandle queue);

    HRESULT Register(_In_opt_ void* context, _In_ XTaskQueueMonitorCallback* callback, _Out_ XTaskQueueRegistrationToken* token);
    void Unregister(_In_ XTaskQueueRegistrationToken token);
    void Invoke(_In_ XTaskQueuePort port);

private:

    struct CallbackRegistration
    {
        uint64_t Token;
        void* Context;
        XTaskQueueMonitorCallback* Callback;
    };

    std::atomic<uint64_t> m_nextToken{ 0 };
    DefaultUnnamedMutex m_lock;
    CallbackRegistration m_buffer1[SUBMIT_CALLBACK_MAX];
    CallbackRegistration m_buffer2[SUBMIT_CALLBACK_MAX];
    CallbackRegistration* m_buffers[2]= { m_buffer1, m_buffer2 };
    std::atomic<uint32_t> m_indexAndRef { 0 };
    XTaskQueueHandle m_queue;
};

#define PORT_WAIT_MAX 60
#define PORT_EVENT_MAX (PORT_WAIT_MAX + 2)
#define QUEUE_WAIT_MAX (PORT_WAIT_MAX * 2)

class QueueWaitRegistry
{
public:

    void AddRef() noexcept { ++m_refs; }
    void Release() noexcept { if (--m_refs == 0) delete this; }
    bool Reserve();
    void CancelReservation();
    void Publish(XTaskQueuePort port, XTaskQueueRegistrationToken portToken,
        XTaskQueueRegistrationToken* token, bool retiredByPort);
    struct WaitRegistration
    {
        uint64_t Token;
        uint64_t PortToken;
        XTaskQueuePort Port;
        bool retiredByPort;
    };

    WaitRegistration Lookup(XTaskQueueRegistrationToken token);
    void Remove(XTaskQueueRegistrationToken token);

private:
    std::atomic<uint32_t> m_refs{0};
    std::atomic<uint64_t> m_nextToken{ 0 };
    uint32_t m_reserved = 0;
    StaticArray<WaitRegistration, QUEUE_WAIT_MAX> m_callbacks;
    DefaultUnnamedMutex m_lock;
};

class TaskQueuePortContextImpl;

class TaskQueuePortImpl: public Api<ApiId::TaskQueuePort, ITaskQueuePort>
{
public:

    TaskQueuePortImpl();
    virtual ~TaskQueuePortImpl();

    HRESULT Initialize(
        _In_ XTaskQueueDispatchMode mode);

    XTaskQueuePortHandle __stdcall GetHandle() { return &m_header; }

    HRESULT __stdcall QueueItem(
        _In_ ITaskQueuePortContext* portContext,
        _In_ uint32_t waitMs,
        _In_opt_ void* callbackContext,
        _In_ XTaskQueueCallback* callback);

    HRESULT __stdcall RegisterWaitHandle(
        _In_ ITaskQueuePortContext* portContext,
        _In_ HANDLE waitHandle,
        _In_opt_ void* callbackContext,
        _In_ XTaskQueueCallback* callback,
        _Out_ XTaskQueueRegistrationToken* token);

    // Local capability for publishing the public token before the wait is armed.
    HRESULT RegisterWaitWithToken(ITaskQueuePortContext* portContext, HANDLE waitHandle,
        void* callbackContext, XTaskQueueCallback* callback, XTaskQueueRegistrationToken* token,
        QueueWaitRegistry* registry);

    void __stdcall UnregisterWaitHandle(
        _In_ XTaskQueueRegistrationToken token);

    HRESULT __stdcall PrepareTerminate(
        _In_ ITaskQueuePortContext* portContext,
        _In_ void* callbackContext,
        _In_ XTaskQueueTerminatedCallback* callback,
        _Out_ void** token);

    void __stdcall CancelTermination(
        _In_ void* token);

    void __stdcall Terminate(
        _In_ void* token);

    virtual HRESULT __stdcall Attach(
        _In_ ITaskQueuePortContext* portContext);

    void __stdcall Detach(
        _In_ ITaskQueuePortContext* portContext);

    bool __stdcall Dispatch(
        _In_ ITaskQueuePortContext* portContext,
        _In_ uint32_t timeoutInMs);

    bool __stdcall IsEmpty();

    void __stdcall WaitForUnwind();

    HRESULT __stdcall SuspendTermination(
        _In_ ITaskQueuePortContext* portContext);

    void __stdcall ResumeTermination(
        _In_ ITaskQueuePortContext* portContext);

    void __stdcall SuspendPort();
    void __stdcall ResumePort();

    void __stdcall SubmitPendingCallbacks();

private:

    void* QueryApiImpl(ApiId id) override
    {
        if (id == TaskQueueInternal::kWaitPublicationApiId) return this;
        return Api<ApiId::TaskQueuePort, ITaskQueuePort>::QueryApiImpl(id);
    }

    struct WaitRegistration;

    struct QueueEntry
    {
        ITaskQueuePortContext* portContext;
        void* callbackContext;
        XTaskQueueCallback* callback;
        WaitRegistration* waitRegistration;
        uint64_t enqueueTime;
        uint64_t id;
    };

    struct TerminationEntry
    {
        ITaskQueuePortContext* portContext;
        void* callbackContext;
        XTaskQueueTerminatedCallback* callback;
        std::uint64_t node;
    };

#ifdef _WIN32
    struct WaitHandleOwner
    {
        HANDLE handle = nullptr;
        ~WaitHandleOwner() { if (handle) CloseHandle(handle); }
    };

    struct WaitRegistration
    {
        enum class State { Active, Canceled, Removed };
        std::atomic<uint32_t> refs{0};
        void AddRef() noexcept { ++refs; }
        void Release() noexcept { if (--refs == 0) delete this; }
        ~WaitRegistration();

        uint64_t token = 0;
        referenced_ptr<QueueWaitRegistry> publicRegistry;
        XTaskQueueRegistrationToken publicToken{};
        std::shared_ptr<WaitHandleOwner> handleOwner;
        std::once_flag drainOnce;
        PTP_WAIT threadpoolWait = nullptr;
        TaskQueuePortImpl* port = nullptr;
        TaskQueuePortContextImpl* pinnable = nullptr;
        QueueEntry queueEntry{};

        // Protected by port->m_lock. Detach retires Active registrations
        // before the embedded context can disappear.
        State state = State::Active;
        referenced_ptr<ITaskQueuePortContext> queuedContext;
        bool appended = false;
        bool running = false;
        bool enteredUserCode = false;
        bool drained = false;
        uint64_t cancelNode = 0;
    };
    struct WaitSnapshot
    {
        std::shared_ptr<WaitHandleOwner> epoch;
        StaticArray<HANDLE, PORT_EVENT_MAX> events;
        StaticArray<std::shared_ptr<WaitHandleOwner>, PORT_WAIT_MAX> owners;
        StaticArray<referenced_ptr<WaitRegistration>, PORT_WAIT_MAX> registrations;
    };
    void InvalidateWaitEpochLocked();
    void FinishRetirementLocked(WaitRegistration* reg);
    void CaptureWaitSnapshot(WaitSnapshot& snapshot);
    friend struct WaitRegistrationTestAccess;

    struct WaitCallbackFrame
    {
        explicit WaitCallbackFrame(WaitRegistration* registration);
        ~WaitCallbackFrame();
        WaitRegistration* reg;
        WaitCallbackFrame* next;
        WaitCallbackFrame** link;
        static DefaultUnnamedMutex lock;
        static WaitCallbackFrame* head;
        static bool ConfirmEntry();
    };

    void DrainRetiredWait(const referenced_ptr<WaitRegistration>& reg);
#endif

    XTaskQueuePortObject m_header = { };
    XTaskQueueDispatchMode m_dispatchMode = XTaskQueueDispatchMode::Manual;
    struct AttachedContextRegistration
    {
        ITaskQueuePortContext* context;
        TaskQueuePortContextImpl* pinnable;
    };

    DefaultUnnamedMutex m_attachedContextsLock;
    std::map<uint64_t, AttachedContextRegistration> m_attachedContexts;

    uint64_t m_nextAttachedContextId = 1;

    bool TryAcquireNextAttachedContext(
        uint64_t& cursor,
        uint64_t upperBound,
        ITaskQueuePortContext** result);

    template<typename Callback>
    void VisitAttachedContexts(Callback callback)
    {
        // Keep the port alive even if a callback closes its owning queue.
        if (!TryAddRef())
        {
            return;
        }

        struct PortGuard
        {
            TaskQueuePortImpl* port;
            ~PortGuard() { port->Release(); }
        } portGuard{ this };

        uint64_t upperBound;
        {
            std::lock_guard<std::mutex> lock(m_attachedContextsLock);
            upperBound = m_nextAttachedContextId - 1;
        }

        uint64_t cursor = 0;

        for (;;)
        {
            ITaskQueuePortContext* context = nullptr;

            if (!TryAcquireNextAttachedContext(
                    cursor, upperBound, &context))
            {
                break;
            }

            struct ContextGuard
            {
                ITaskQueuePortContext* context;
                ~ContextGuard() { context->Release(); }
            } contextGuard{ context };

            // Never invoke external code while holding the registry mutex.
            callback(context);
        }

        // Only safely pinnable contexts can be visited.
    }
    std::atomic<uint32_t> m_processingSerializedTbCallback{ 0 };
    std::atomic<uint32_t> m_processingCallback{ 0 };
    DefaultUnnamedConditionVariable m_processingCallbackCv;
    DefaultUnnamedMutex m_lock;
    std::unique_ptr<LocklessQueue<QueueEntry>> m_queueList;
    std::unique_ptr<LocklessQueue<QueueEntry>> m_pendingList;
    std::unique_ptr<LocklessQueue<TerminationEntry*>> m_terminationList;
    std::unique_ptr<LocklessQueue<TerminationEntry*>> m_pendingTerminationList;
    DefaultUnnamedMutex m_terminationLock;
    OS::WaitTimer m_timer;
    OS::ThreadPool m_threadPool;
    std::atomic<uint64_t> m_timerDue = { UINT64_MAX };
    std::atomic<uint64_t> m_nextId = { 0 };
    std::atomic<bool> m_suspended = { false };

#ifdef _WIN32
    std::map<uint64_t, referenced_ptr<WaitRegistration>> m_waits;
    DefaultUnnamedConditionVariable m_waitFinished;
    HANDLE m_queueEvent = nullptr;
    std::shared_ptr<WaitHandleOwner> m_waitEpoch;
    std::atomic<size_t> m_readyEntries{0};
    uint32_t m_activeWaits = 0;
    uint64_t m_nextWaitToken = 0;
#else
    bool m_signaled = false;
    DefaultUnnamedConditionVariableAny m_event;
#endif

    bool ReadyQueueEmpty();
    HRESULT VerifyNotTerminated(_In_ ITaskQueuePortContext* portContext);

    bool IsCallCanceled(_In_ const QueueEntry& entry);

    // Appends the given entry to the active queue.  The entry should already
    // be add-refd.
    bool AppendEntry(
        _In_ const QueueEntry& entry,
        _In_opt_ uint64_t node = 0);

    void CancelPendingEntries(
        _In_ ITaskQueuePortContext* portContext,
        _In_ bool appendToQueue);

    bool DrainOneItem();

    bool DrainOneItem(
        _In_ OS::ThreadPoolActionStatus& status);

    bool Wait(
        _In_ ITaskQueuePortContext* portContext,
        _In_ uint32_t timeout);

    static void EraseQueue(
        _In_opt_ LocklessQueue<QueueEntry>* queue);

    bool ArmTimerIfEarlier(_In_ uint64_t dueTime);

    bool ArmTimerForNextPendingDueTime(
        _In_ uint64_t previousDueTime,
        _In_ uint64_t nextDueTime);

    bool RearmTimerIfDueTimeUnchanged(_In_ uint64_t dueTime);

    // After cancellation removed pending entries, re-point the shared timer at
    // the earliest surviving pending entry instead of leaving it armed for a
    // deadline that no longer has an owner.
    void RearmTimerForEarliestPending();

    void PromoteReadyPendingCallbacks(
        _In_ uint64_t dueTime,
        _In_ uint64_t now);

    void SignalTerminations();
    void ScheduleTermination(_In_ TerminationEntry* term);
    bool TerminationListEmpty();

    void SignalQueue();
    void NotifyItemQueued();

    void ProcessThreadPoolCallback(_In_ OS::ThreadPoolActionStatus& status);

#ifdef _WIN32
    bool PublishWaitEntryLocked(WaitRegistration* reg, uint64_t node);
    void RetireWaitLocked(WaitRegistration* reg, WaitRegistration::State state);
    bool DispatchWaitEntry(const QueueEntry& entry, OS::ThreadPoolActionStatus& status,
        referenced_ptr<ITaskQueuePortContext>& context);
    void ProcessWaitCallback(WaitRegistration* reg);

    static void CALLBACK WaitCallback(
        _In_ PTP_CALLBACK_INSTANCE instance,
        _Inout_opt_ void* context,
        _Inout_ PTP_WAIT wait,
        _In_ TP_WAIT_RESULT waitResult);
#endif
};

class TaskQueueImpl;

class TaskQueuePortContextImpl : public ITaskQueuePortContext
{
public:
    
    TaskQueuePortContextImpl(
        _In_ TaskQueueImpl* queue,
        _In_ XTaskQueuePort type,
        _In_ SubmitCallback* submitCallback);

    uint32_t __stdcall AddRef() override;
    uint32_t __stdcall Release() override;
    bool TryPin() noexcept;
    HRESULT __stdcall QueryApi(_In_ ApiId id, _Out_ void** ptr) override;

    XTaskQueuePort __stdcall GetType() override;
    TaskQueuePortStatus __stdcall GetStatus() override;
    ITaskQueue* __stdcall GetQueue() override;
    ITaskQueuePort* __stdcall GetPort() override;

    bool __stdcall TrySetStatus(
        _In_ TaskQueuePortStatus expectedStatus,
        _In_ TaskQueuePortStatus status) override;

    void __stdcall SetStatus(
        _In_ TaskQueuePortStatus status) override;

    void __stdcall ItemQueued() override;

    bool __stdcall AddSuspend() override;
    bool __stdcall RemoveSuspend() override;

    referenced_ptr<ITaskQueuePort> Port;
    referenced_ptr<ITaskQueue> Source;

private:

    TaskQueueImpl* m_queue = nullptr;
    XTaskQueuePort m_type = XTaskQueuePort::Work;
    SubmitCallback* m_submitCallback = nullptr;
    std::atomic<TaskQueuePortStatus> m_status = { TaskQueuePortStatus::Active };
    std::atomic<uint32_t> m_suspendCount = { 0 };
};

class TaskQueueImpl : public Api<ApiId::TaskQueue, ITaskQueue>
#ifdef SUSPEND_API
    , public ISuspendResumeCallback
#endif
{
public:

    TaskQueueImpl();
    virtual ~TaskQueueImpl();

    HRESULT Initialize(
        _In_ XTaskQueueDispatchMode workMode,
        _In_ XTaskQueueDispatchMode completionMode);

    HRESULT Initialize(
        _In_ XTaskQueuePortHandle workPort,
        _In_ XTaskQueuePortHandle completionPort);
    
    XTaskQueueHandle __stdcall GetHandle() override { return &m_header; }
    XTaskQueueTestHooks* __stdcall GetTestHooks() override { return m_testHooks; }
    void __stdcall SetTestHooks(_In_ XTaskQueueTestHooks* testHooks) override { m_testHooks = testHooks; }

    HRESULT __stdcall GetPortContext(
        _In_ XTaskQueuePort port,
        _Out_ ITaskQueuePortContext** portContext) override;

    HRESULT __stdcall RegisterWaitHandle(
        _In_ XTaskQueuePort port,
        _In_ HANDLE waitHandle,
        _In_opt_ void* callbackContext,
        _In_ XTaskQueueCallback* callback,
        _Out_ XTaskQueueRegistrationToken* token) override;

    void __stdcall UnregisterWaitHandle(
        _In_ XTaskQueueRegistrationToken token) override;

    HRESULT __stdcall RegisterSubmitCallback(
        _In_opt_ void* context,
        _In_ XTaskQueueMonitorCallback* callback,
        _Out_ XTaskQueueRegistrationToken* token) override;

    void __stdcall UnregisterSubmitCallback(
        _In_ XTaskQueueRegistrationToken token) override;

    HRESULT __stdcall Terminate(
        _In_ bool wait, 
        _In_opt_ void* callbackContext, 
        _In_opt_ XTaskQueueTerminatedCallback* callback) override ;

protected:

    void RundownObject() override;

private:

#ifdef SUSPEND_API
    void OnSuspendResume(_In_ bool isSuspended) override;
#endif

    static void CALLBACK OnTerminationCallback(_In_ void* context);

private:

    enum class TerminationLevel
    {
        None,
        Work,
        Completion
    };

    struct TerminationEntry
    {
        TaskQueueImpl* owner;
        TerminationLevel level;
        void* completionPortToken;
        void* context;
        XTaskQueueTerminatedCallback* callback;
    };

    struct TerminationData
    {
        bool terminated;
        DefaultUnnamedMutex lock;
        DefaultUnnamedConditionVariable cv;
    };

    XTaskQueueObject m_header = { };
    SubmitCallback m_callbackSubmitted;
    referenced_ptr<QueueWaitRegistry> m_waitRegistry{new (std::nothrow) QueueWaitRegistry};
    TerminationData m_termination;
    TaskQueuePortContextImpl m_work;
    TaskQueuePortContextImpl m_completion;
    XTaskQueueTestHooks* m_testHooks = nullptr;

#ifdef SUSPEND_API
    SuspendResumeHandler m_suspendHandler;
#endif
};

inline ITaskQueue* GetQueue(XTaskQueueHandle handle)
{
    if (handle->m_signature != TASK_QUEUE_SIGNATURE ||
        handle->m_runtimeIteration != GetCurrentRuntimeIteration())
    {
        SystemHandleAssert(handle);
        ASSERT("Invalid XTaskQueueHandle");
        return nullptr;
    }

    ITaskQueue* queue = handle->m_queue;

    // Only SystemHandleAssert if the handle provided here is
    // not the internal queue handle, which is never tracked
    // by handle tracking. The internal queue handle is given out
    // for the default process queue, when calling back into
    // task queue monitors, and when async operations are created.

    if (handle != queue->GetHandle())
    {
        SystemHandleAssert(handle);
    }

    ASSERT(queue->GetHandle()->m_signature == TASK_QUEUE_SIGNATURE);
    return queue;
}
