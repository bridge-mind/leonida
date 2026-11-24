// Copyright (c) Microsoft Corporation
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

#include "pch.h"
#include "referenced_ptr.h"
#include "XTaskQueuePriv.h"
#include "TaskQueueP.h"
#include "TaskQueueImpl.h"

#if defined(HC_ENABLE_UNNAMED_OBJECT_DIAGNOSTICS)
namespace
{
struct UnnamedObjectDiagnosticCounters
{
    std::atomic<uint64_t> constructed[4]{};
    std::atomic<uint64_t> destroyed[4]{};
};

UnnamedObjectDiagnosticCounters& DiagnosticCounters() noexcept
{
    static UnnamedObjectDiagnosticCounters counters;
    return counters;
}

size_t DiagnosticTypeIndex(UnnamedObjectDiagnostics::Type type) noexcept
{
    return static_cast<size_t>(type);
}
}

namespace UnnamedObjectDiagnostics
{
void RecordConstruction(Type type) noexcept
{
    ++DiagnosticCounters().constructed[DiagnosticTypeIndex(type)];
}

void RecordDestruction(Type type) noexcept
{
    ++DiagnosticCounters().destroyed[DiagnosticTypeIndex(type)];
}
}

STDAPI XTaskQueueGetSyncObjectDiagnosticSnapshot(
    _Out_ XTaskQueueSyncObjectSnapshot* snapshot
    ) noexcept
{
    RETURN_HR_IF(E_POINTER, snapshot == nullptr);

    auto loadCounts = [](UnnamedObjectDiagnostics::Type type)
    {
        const size_t index = DiagnosticTypeIndex(type);
        auto& counters = DiagnosticCounters();
        const uint64_t destroyed = counters.destroyed[index].load();
        const uint64_t constructed = counters.constructed[index].load();
        return XTaskQueueSyncObjectCounts
        {
            constructed,
            destroyed,
            constructed - destroyed
        };
    };

    snapshot->mutex = loadCounts(UnnamedObjectDiagnostics::Type::Mutex);
    snapshot->recursiveMutex = loadCounts(UnnamedObjectDiagnostics::Type::RecursiveMutex);
    snapshot->conditionVariable = loadCounts(UnnamedObjectDiagnostics::Type::ConditionVariable);
    snapshot->conditionVariableAny = loadCounts(UnnamedObjectDiagnostics::Type::ConditionVariableAny);
#if defined(HC_USE_UNNAMED_MUTEX)
    snapshot->usesUnnamedConstructors = true;
#else
    snapshot->usesUnnamedConstructors = false;
#endif
    return S_OK;
}

STDAPI XTaskQueueRunSyncObjectDiagnosticProbe(
    _Out_ XTaskQueueSyncObjectSnapshot* before,
    _Out_ XTaskQueueSyncObjectSnapshot* during,
    _Out_ XTaskQueueSyncObjectSnapshot* after
    ) noexcept
{
    RETURN_HR_IF(E_POINTER, before == nullptr || during == nullptr || after == nullptr);
    RETURN_IF_FAILED(XTaskQueueGetSyncObjectDiagnosticSnapshot(before));

    {
        DefaultUnnamedMutex mutex;
        DefaultUnnamedRecursiveMutex recursiveMutex;
        DefaultUnnamedConditionVariable conditionVariable;
        DefaultUnnamedConditionVariableAny conditionVariableAny;

        std::lock_guard<DefaultUnnamedMutex> mutexLock{ mutex };
        std::lock_guard<DefaultUnnamedRecursiveMutex> outerRecursiveLock{ recursiveMutex };
        std::lock_guard<DefaultUnnamedRecursiveMutex> innerRecursiveLock{ recursiveMutex };
        RETURN_IF_FAILED(XTaskQueueGetSyncObjectDiagnosticSnapshot(during));
    }

    return XTaskQueueGetSyncObjectDiagnosticSnapshot(after);
}
#endif

//
// ApiRefs tracks global refcounts for all APIs. It is used to identify memory leaks
// in tests and can be called to wait for all refs to be released.
//
namespace ApiRefs
{
    std::atomic<uint32_t> g_globalApiRefs{ 0 };
    DefaultUnnamedMutex g_waitMutex;
    DefaultUnnamedConditionVariable g_waitCv;
    
    void GlobalAddRef()
    {
        g_globalApiRefs++;
    }
    
    void GlobalRelease()
    {
        auto priorRefs = g_globalApiRefs.fetch_sub(1);
        if (priorRefs == 1)
        {
            std::unique_lock<std::mutex> lock(g_waitMutex);
            g_waitCv.notify_all();
        }
    }

    bool WaitZeroRefs(_In_ uint32_t timeoutMilliseconds)
    {
        // We are called after queues are deleted so it could be
        // possible that the cv doesn't get signaled, so check for
        // zero first before waiting.
        std::unique_lock<std::mutex> lock(g_waitMutex);
        if (g_globalApiRefs.load() != 0)
        {
            g_waitCv.wait_for(lock, std::chrono::milliseconds(timeoutMilliseconds));
            return g_globalApiRefs == 0;
        }

        return true;
    }

#ifdef HC_UNITTEST_API
    std::atomic<int32_t> g_liveQueues{ 0 };
    std::atomic<int32_t> g_livePorts{ 0 };

    void TrackLiveObject(ApiId id, int32_t delta) noexcept
    {
        switch (id)
        {
        case ApiId::TaskQueue:
            g_liveQueues += delta;
            break;

        case ApiId::TaskQueuePort:
            g_livePorts += delta;
            break;

        default:
            break;
        }
    }
#endif
}

#ifdef HC_UNITTEST_API
STDAPI_(void) XTaskQueueGetLiveObjectCounts(
    _Out_ XTaskQueueLiveObjectCounts* counts
    ) noexcept
{
    if (counts != nullptr)
    {
        counts->globalRefs = ApiRefs::g_globalApiRefs.load();
        counts->queues = static_cast<uint32_t>(ApiRefs::g_liveQueues.load());
        counts->ports = static_cast<uint32_t>(ApiRefs::g_livePorts.load());
    }
}
#endif

namespace ProcessGlobals
{
    const XTaskQueueHandle g_invalidQueueHandle = (XTaskQueueHandle)(-1);
    std::atomic<XTaskQueueHandle> g_defaultProcessQueue = { g_invalidQueueHandle };
    std::atomic<XTaskQueueHandle> g_processQueue = { g_invalidQueueHandle };

#ifdef SUSPEND_API
    SuspendState g_suspendState;
#endif
}

//
// SubmitCallback
//

SubmitCallback::SubmitCallback(
    _In_ XTaskQueueHandle queue)
    : m_queue(queue)
{
    memset(m_buffer1, 0, sizeof(m_buffer1));
    memset(m_buffer2, 0, sizeof(m_buffer2));
}

HRESULT SubmitCallback::Register(_In_opt_ void* context, _In_ XTaskQueueMonitorCallback* callback, _Out_ XTaskQueueRegistrationToken* token)
{
    RETURN_HR_IF(E_POINTER, callback == nullptr || token == nullptr);

    token->token = 0;

    std::lock_guard<std::mutex> lock(m_lock);
    uint32_t bufferReadIdx = (m_indexAndRef & 0x80000000 ? 1 : 0);
    uint32_t bufferWriteIdx = 1 - bufferReadIdx;

    for(uint32_t idx = 0; idx < ARRAYSIZE(m_buffer1); idx++)
    {
        if (token->token == 0 && m_buffers[bufferReadIdx][idx].Callback == nullptr)
        {
            token->token = ++m_nextToken;
            m_buffers[bufferWriteIdx][idx].Token = token->token;
            m_buffers[bufferWriteIdx][idx].Context = context;
            m_buffers[bufferWriteIdx][idx].Callback = callback;
        }
        else
        {
            m_buffers[bufferWriteIdx][idx] = m_buffers[bufferReadIdx][idx];
        }
    }

    RETURN_HR_IF(E_OUTOFMEMORY, token->token == 0);

    // Now spin wait to swap the active buffer.
    uint32_t expected = bufferReadIdx << 31;
    uint32_t desired = bufferWriteIdx << 31;

    for (;;)
    {
        uint32_t expectedSwap = expected;
        if (m_indexAndRef.compare_exchange_weak(expectedSwap, desired))
        {
            break;
        }
    }
    
    return S_OK;
}

void SubmitCallback::Unregister(_In_ XTaskQueueRegistrationToken token)
{
    std::lock_guard<std::mutex> lock(m_lock);
    uint32_t bufferReadIdx = (m_indexAndRef & 0x80000000 ? 1 : 0);
    uint32_t bufferWriteIdx = 1 - bufferReadIdx;

    for(uint32_t idx = 0; idx < ARRAYSIZE(m_buffer1); idx++)
    {
        if (m_buffers[bufferReadIdx][idx].Token == token.token)
        {
            m_buffers[bufferWriteIdx][idx].Callback = nullptr;
        }
        else
        {
            m_buffers[bufferWriteIdx][idx] = m_buffers[bufferReadIdx][idx];
        }
    }

    // Now spin wait to swap the active buffer.
    uint32_t expected = bufferReadIdx << 31;
    uint32_t desired = bufferWriteIdx << 31;

    for (;;)
    {
        uint32_t expectedSwap = expected;
        if (m_indexAndRef.compare_exchange_weak(expectedSwap, desired))
        {
            break;
        }
    }
}

void SubmitCallback::Invoke(_In_ XTaskQueuePort port)
{
    uint32_t indexAndRef = ++m_indexAndRef;
    uint32_t bufferIdx = (indexAndRef & 0x80000000 ? 1 : 0);

    for(uint32_t idx = 0; idx < ARRAYSIZE(m_buffer1); idx++)
    {
        if (m_buffers[bufferIdx][idx].Callback != nullptr)
        {
            m_buffers[bufferIdx][idx].Callback(
                m_buffers[bufferIdx][idx].Context,
                m_queue,
                port);
        }
    }

    m_indexAndRef--;
}

//
// QueueWaitRegistry
//

bool QueueWaitRegistry::Reserve()
{
    std::lock_guard<std::mutex> lock(m_lock);
    if (m_reserved == m_callbacks.capacity()) return false;
    ++m_reserved;
    return true;
}

void QueueWaitRegistry::CancelReservation()
{
    std::lock_guard<std::mutex> lock(m_lock);
    --m_reserved;
}

void QueueWaitRegistry::Publish(XTaskQueuePort port, XTaskQueueRegistrationToken portToken,
    XTaskQueueRegistrationToken* token, bool retiredByPort)
{
    std::lock_guard<std::mutex> lock(m_lock);
    --m_reserved;
    WaitRegistration reg{++m_nextToken, portToken.token, port, retiredByPort};
    m_callbacks.append(reg);
    token->token = reg.Token;
}

QueueWaitRegistry::WaitRegistration QueueWaitRegistry::Lookup(
    XTaskQueueRegistrationToken token)
{
    std::lock_guard<std::mutex> lock(m_lock);
    for (uint32_t i = 0; i < m_callbacks.count(); ++i)
    {
        if (m_callbacks[i].Token == token.token)
            return m_callbacks[i];
    }
    return {};
}

void QueueWaitRegistry::Remove(XTaskQueueRegistrationToken token)
{
    std::lock_guard<std::mutex> lock(m_lock);
    for (uint32_t i = 0; i < m_callbacks.count(); ++i)
    {
        if (m_callbacks[i].Token == token.token)
        {
            m_callbacks.removeAt(i);
            return;
        }
    }
}

//
// TaskQueuePortImpl
//

TaskQueuePortImpl::TaskQueuePortImpl()
    : Api()
{
    m_header.m_signature = TASK_QUEUE_PORT_SIGNATURE;
    m_header.m_runtimeIteration = GetCurrentRuntimeIteration();
    m_header.m_port = this;
    m_header.m_queue = nullptr;
}

TaskQueuePortImpl::~TaskQueuePortImpl()
{
    m_timer.Terminate();

    EraseQueue(m_queueList.get());
    EraseQueue(m_pendingList.get());

#ifdef _WIN32
    // The last port release precludes new pins. Finish any native
    // callbacks that entered before TryAddRef began refusing them.
    while (!m_waits.empty())
    {
        UnregisterWaitHandle({m_waits.begin()->first});
    }
#endif

    m_threadPool.Terminate();

#ifdef _WIN32
    if (m_queueEvent) CloseHandle(m_queueEvent);
#endif

    // Destroy these lists in order, since the pending lsit shares
    // the node heap with the queue list.
    
    m_pendingList.reset();
    m_queueList.reset();
}

HRESULT TaskQueuePortImpl::Initialize(
    _In_ XTaskQueueDispatchMode mode)
{
    m_dispatchMode = mode;

    m_queueList.reset(new (std::nothrow) LocklessQueue<QueueEntry>);
    RETURN_IF_NULL_ALLOC(m_queueList);

    // Lockless lists can share their underlying heaps.  This both reduces
    // memory costs and allows node addresses from one list to be transferred
    // to another.  We use this frequently when transferring data across
    // lists to eliminate the possibility of allocation failures.

    m_pendingList.reset(new (std::nothrow) LocklessQueue<QueueEntry>(*m_queueList.get()));
    RETURN_IF_NULL_ALLOC(m_pendingList);

    // There should be very few simultaneous termination requests, so we can set the
    // default heap size to the minimum.

    m_terminationList.reset(new (std::nothrow) LocklessQueue<TerminationEntry*>(0));
    RETURN_IF_NULL_ALLOC(m_terminationList);

    m_pendingTerminationList.reset(new (std::nothrow) LocklessQueue<TerminationEntry*>(*m_terminationList.get()));
    RETURN_IF_NULL_ALLOC(m_pendingTerminationList);

    RETURN_IF_FAILED(m_timer.Initialize(this, [](void* context)
    {
        TaskQueuePortImpl* pthis = static_cast<TaskQueuePortImpl*>(context);

        // Keep the port alive while we process pending callbacks. The sweep
        // takes and drops temporary references on queues attached to this
        // port, and if a queue is closed concurrently one of those releases
        // can be the last, destroying the queue and its port on this thread.
        // Holding our own reference defers any port destruction to the
        // Release below, after which we touch nothing. The timer can't be
        // torn down underneath us before this AddRef because port
        // destruction waits for in-flight timer callbacks.
        pthis->AddRef();
        pthis->SubmitPendingCallbacks();
        pthis->Release(); // May destroy the port and its timer.
    }));

#ifdef _WIN32
    HANDLE evt = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    RETURN_LAST_ERROR_IF_NULL(evt);
    m_queueEvent = evt;
#endif

    switch (mode)
    {
    case XTaskQueueDispatchMode::Manual:
        // nothing
        break;

    case XTaskQueueDispatchMode::ThreadPool:
    case XTaskQueueDispatchMode::SerializedThreadPool:
        RETURN_IF_FAILED(m_threadPool.Initialize(this, [](void* context, OS::ThreadPoolActionStatus& status)
        {
            TaskQueuePortImpl* pthis = static_cast<TaskQueuePortImpl*>(context);
            pthis->ProcessThreadPoolCallback(status);
        }));
        break;
          
    case XTaskQueueDispatchMode::Immediate:
        // nothing
        break;
    }

    return S_OK;
}

HRESULT __stdcall TaskQueuePortImpl::QueueItem(
    _In_ ITaskQueuePortContext* portContext,
    _In_ uint32_t waitMs,
    _In_opt_ void* callbackContext,
    _In_ XTaskQueueCallback* callback)
{
    RETURN_IF_FAILED(VerifyNotTerminated(portContext));

    QueueEntry entry;
    referenced_ptr<ITaskQueuePortContext> portContextHolder(portContext);

    entry.portContext = portContext;
    entry.callback = callback;
    entry.callbackContext = callbackContext;
    entry.waitRegistration = nullptr;
    entry.id = m_nextId++;

    if (waitMs == 0)
    {
        entry.enqueueTime = 0;
        RETURN_HR_IF(E_OUTOFMEMORY, !AppendEntry(entry));
    }
    else
    {
        // Delayed callbacks are ordered by a monotonic due time so stale timer
        // callbacks and wall-clock adjustments cannot make one pending entry
        // masquerade as another.
        entry.enqueueTime = m_timer.GetDueTime(waitMs);
        RETURN_HR_IF(E_OUTOFMEMORY, !m_pendingList->push_back(entry));

        while (true)
        {
            const uint64_t publishedDueTime = m_timerDue.load();
            const uint64_t now = m_timer.GetCurrentTime();

            // A one-shot timer callback may already be processing an overdue
            // deadline without having advanced m_timerDue yet. Rearm that
            // published deadline so this newly queued entry cannot be stranded
            // behind the in-progress callback.
            if (publishedDueTime != UINT64_MAX && publishedDueTime <= now)
            {
                if (!RearmTimerIfDueTimeUnchanged(publishedDueTime))
                {
                    continue;
                }
            }
            else
            {
                ArmTimerIfEarlier(entry.enqueueTime);
            }

            break;
        }
    }

    // QueueEntry now owns the ref.
    portContextHolder.release();

    // If we raced with termination, cancel the item we just added
    if (portContext->GetStatus() != TaskQueuePortStatus::Active)
    {
        CancelPendingEntries(portContext, true);
    }

    return S_OK;
}

HRESULT __stdcall TaskQueuePortImpl::RegisterWaitHandle(
    _In_ ITaskQueuePortContext* portContext,
    _In_ HANDLE waitHandle,
    _In_opt_ void* callbackContext,
    _In_ XTaskQueueCallback* callback,
    _Out_ XTaskQueueRegistrationToken* token)
{
    return RegisterWaitWithToken(portContext, waitHandle, callbackContext, callback, token, nullptr);
}

HRESULT TaskQueuePortImpl::RegisterWaitWithToken(ITaskQueuePortContext* portContext,
    HANDLE waitHandle, void* callbackContext, XTaskQueueCallback* callback,
    XTaskQueueRegistrationToken* token, QueueWaitRegistry* registry)
{
    RETURN_HR_IF(E_INVALIDARG, callback == nullptr || waitHandle == nullptr || token == nullptr);
    RETURN_HR_IF(E_POINTER, token == nullptr);
    RETURN_IF_FAILED(VerifyNotTerminated(portContext));

#ifdef _WIN32
    referenced_ptr<WaitRegistration> reg(new (std::nothrow) WaitRegistration);
    RETURN_IF_NULL_ALLOC(reg.get());

    void* capability = nullptr;
    RETURN_IF_FAILED(portContext->QueryApi(TaskQueueInternal::kPinnableContextApiId, &capability));
    referenced_ptr<TaskQueuePortContextImpl> pin;
    *pin.address_of() = static_cast<TaskQueuePortContextImpl*>(capability);
    RETURN_HR_IF(E_NOINTERFACE, !pin);

    reg->port = this;
    reg->pinnable = pin.get();
    reg->queueEntry = {portContext, callbackContext, callback, reg.get(), 0, m_nextId++};
    try
    {
        reg->handleOwner = std::make_shared<WaitHandleOwner>();
    }
    CATCH_RETURN();
    RETURN_IF_WIN32_BOOL_FALSE(DuplicateHandle(GetCurrentProcess(), waitHandle, GetCurrentProcess(),
        &reg->handleOwner->handle, 0, FALSE, DUPLICATE_SAME_ACCESS));
    reg->threadpoolWait = CreateThreadpoolWait(WaitCallback, reg.get(), nullptr);
    RETURN_LAST_ERROR_IF_NULL(reg->threadpoolWait);

    try
    {
        std::lock_guard<std::mutex> lock(m_lock);
        RETURN_IF_FAILED(VerifyNotTerminated(portContext));
        RETURN_HR_IF(E_OUTOFMEMORY, m_activeWaits == PORT_WAIT_MAX || m_nextWaitToken == UINT64_MAX);
        reg->token = ++m_nextWaitToken;
        m_waits.emplace(reg->token, reg);
        // Termination must be able to deliver cancellation even under OOM.
        if (!m_queueList->reserve_node(reg->cancelNode))
        {
            m_waits.erase(reg->token);
            return E_OUTOFMEMORY;
        }
        ++m_activeWaits;
        // Lookup never holds the registry lock while entering the port.
        if (registry)
        {
            registry->Publish(portContext->GetType(), {reg->token}, token, true);
            reg->publicRegistry = referenced_ptr<QueueWaitRegistry>(registry);
            reg->publicToken = *token;
        }
        else token->token = reg->token;
        SetThreadpoolWait(reg->threadpoolWait, reg->handleOwner->handle, nullptr);
        InvalidateWaitEpochLocked();
    }
    CATCH_RETURN();

    return S_OK;
#else
    return E_NOTIMPL;
#endif
}

void __stdcall TaskQueuePortImpl::UnregisterWaitHandle(
    _In_ XTaskQueueRegistrationToken token)
{
#ifdef _WIN32
    referenced_ptr<WaitRegistration> reg;
    referenced_ptr<ITaskQueuePortContext> suppressedContext;
    const bool reentrant = WaitCallbackFrame::ConfirmEntry();
    {
        std::lock_guard<std::mutex> lock(m_lock);
        auto it = m_waits.find(token.token);
        if (it == m_waits.end()) return;
        reg = it->second;
        RetireWaitLocked(reg.get(), WaitRegistration::State::Removed);
        suppressedContext = std::move(reg->queuedContext);
        if (suppressedContext) --m_readyEntries;
    }

    DrainRetiredWait(reg);
    {
        std::unique_lock<std::mutex> lock(m_lock);
        m_waitFinished.wait(lock, [&] {
            return !reg->running || (reentrant && reg->enteredUserCode);
        });
        FinishRetirementLocked(reg.get());
    }
#else
    UNREFERENCED_PARAMETER(token);
#endif
}

HRESULT __stdcall TaskQueuePortImpl::PrepareTerminate(
    _In_ ITaskQueuePortContext* portContext,
    _In_ void* callbackContext,
    _In_ XTaskQueueTerminatedCallback* callback,
    _Out_ void** token)
{
    RETURN_HR_IF(E_POINTER, token == nullptr);
    RETURN_HR_IF(E_INVALIDARG, callback == nullptr);

    std::unique_ptr<TerminationEntry> term(new (std::nothrow) TerminationEntry);
    RETURN_IF_NULL_ALLOC(term);

    {
        std::lock_guard<std::mutex> lock(m_terminationLock);
        RETURN_HR_IF(E_OUTOFMEMORY, !m_terminationList->reserve_node(term->node));
    }

    term->callbackContext = callbackContext;
    term->callback = callback;
    term->portContext = portContext;

    // Mark the port as canceled, but don't overwrite
    // terminating or terminated status.
    portContext->TrySetStatus(TaskQueuePortStatus::Active, TaskQueuePortStatus::Canceled);
    *token = term.release();

    return S_OK;
}

void __stdcall TaskQueuePortImpl::CancelTermination(
    _In_ void* token)
{
    TerminationEntry* term = static_cast<TerminationEntry*>(token);
    term->portContext->TrySetStatus(TaskQueuePortStatus::Canceled, TaskQueuePortStatus::Active);

    if (term->node != 0)
    {
        std::lock_guard<std::mutex> lock(m_terminationLock);
        m_terminationList->free_node(term->node);
    }

    delete term;
}

void __stdcall TaskQueuePortImpl::Terminate(
    _In_ void* token)
{
    TerminationEntry* term = static_cast<TerminationEntry*>(token);
    referenced_ptr<ITaskQueuePortContext> cxt(term->portContext);

    // Prevent anything else from coming into the queue
    cxt->SetStatus(TaskQueuePortStatus::Terminating);

    CancelPendingEntries(cxt.get(), true);

    // Are there existing suspends?  AddSuspend returns
    // true if this is the first suspend added.
    if (cxt->AddSuspend())
    {
        ScheduleTermination(term);
    }
    else
    {
        std::lock_guard<std::mutex> lock(m_terminationLock);
        m_pendingTerminationList->push_back(term, term->node);
        term->node = 0;
    }

    // Balance our add.  Note we must use ResumeTermination
    // here so we schedule the termination if this is the
    // last remove.
    ResumeTermination(cxt.get());
}

HRESULT __stdcall TaskQueuePortImpl::Attach(
    _In_ ITaskQueuePortContext* portContext) try
{
    // Discover the private capability while the caller still owns
    // the context. QueryApi returns an owned reference on success.
    void* capability = nullptr;
    HRESULT queryResult = portContext->QueryApi(
        TaskQueueInternal::kPinnableContextApiId,
        &capability);

    if (SUCCEEDED(queryResult) && capability != nullptr)
    {
        auto* pinnable =
            static_cast<TaskQueuePortContextImpl*>(capability);

        struct CapabilityGuard
        {
            TaskQueuePortContextImpl* context;
            ~CapabilityGuard() { context->Release(); }
        } capabilityGuard{ pinnable };

        std::lock_guard<std::mutex> lock(m_attachedContextsLock);

        if (m_nextAttachedContextId == UINT64_MAX)
        {
            return E_OUTOFMEMORY;
        }

        m_attachedContexts.emplace(
            m_nextAttachedContextId,
            AttachedContextRegistration{ portContext, pinnable });

        ++m_nextAttachedContextId;
        return S_OK;
    }

    // Without safe pinning, invoking callbacks outside the lock is unsafe.
    return E_NOINTERFACE;
} CATCH_RETURN();

void __stdcall TaskQueuePortImpl::Detach(
    _In_ ITaskQueuePortContext* portContext)
{
    {
        std::lock_guard<std::mutex> lock(m_attachedContextsLock);

        for (auto it = m_attachedContexts.begin();
             it != m_attachedContexts.end(); ++it)
        {
            if (it->second.context == portContext)
            {
                m_attachedContexts.erase(it);
                break;
            }
        }
    }

    // Cancellation may release references and run reentrant teardown.
    CancelPendingEntries(portContext, false);
}

bool TaskQueuePortImpl::TryAcquireNextAttachedContext(
    uint64_t& cursor,
    uint64_t upperBound,
    ITaskQueuePortContext** result)
{
    *result = nullptr;

    std::lock_guard<std::mutex> lock(m_attachedContextsLock);

    auto it = m_attachedContexts.upper_bound(cursor);

    while (it != m_attachedContexts.end() &&
           it->first <= upperBound)
    {
        cursor = it->first;

        // A successful pin keeps the containing queue alive until
        // the caller releases the context outside the registry lock.
        if (it->second.pinnable->TryPin())
        {
            *result = it->second.context;
            return true;
        }

        ++it;
    }

    return false;
}

bool __stdcall TaskQueuePortImpl::Dispatch(
    _In_ ITaskQueuePortContext* portContext,
    _In_ uint32_t timeoutInMs)
{
    bool found = false;

    while (!found)
    {
        found = DrainOneItem();

        if (!found && !Wait(portContext, timeoutInMs))
        {
            break;
        }
    }

    return found;
}

bool TaskQueuePortImpl::DrainOneItem()
{
    struct DummyActionStatus : OS::ThreadPoolActionStatus
    {
        void Complete() override {}
        void MayRunLong() override {}
    } dummy;

    return DrainOneItem(dummy);
}

bool TaskQueuePortImpl::DrainOneItem(
    _In_ OS::ThreadPoolActionStatus& status)
{
    if (m_suspended && m_dispatchMode != XTaskQueueDispatchMode::Immediate)
    {
        return false;
    }

    m_processingCallback++;
    QueueEntry entry;
    bool popped = false;

    while (m_queueList->pop_front(entry))
    {
        referenced_ptr<ITaskQueuePortContext> context;

#ifdef _WIN32
        if (entry.waitRegistration)
        {
            if (!DispatchWaitEntry(entry, status, context)) continue;
        }
        else
#endif
        {
#ifdef _WIN32
            --m_readyEntries;
#endif
            *context.address_of() = entry.portContext; // Adopt the queue entry's reference.
            if (entry.portContext->GetType() == XTaskQueuePort::Work)
            {
                status.MayRunLong();
            }
            entry.callback(entry.callbackContext, IsCallCanceled(entry));
        }

        m_processingCallback--;
        m_processingCallbackCv.notify_all();
        popped = true;
        break; // context releases after the processing count falls.
    }
    if (!popped)
    {
        m_processingCallback--;
        m_processingCallbackCv.notify_all();
    }

    if (ReadyQueueEmpty() && m_processingCallback.load() == 0)
    {
        SignalTerminations();
        SignalQueue();
    }

    return popped;
}

// Waits up to timeout for something to arrive in the queue.
// Returns true if the wait succeeded.  Returns false if it timed
// out or the port is terminated.
bool TaskQueuePortImpl::Wait(
    _In_ ITaskQueuePortContext* portContext,
    _In_ uint32_t timeout)
{
#ifdef _WIN32
    const ULONGLONG start = GetTickCount64();
    bool attempted = false;
    for (;;)
    {
        WaitSnapshot snapshot;
        CaptureWaitSnapshot(snapshot);
        // A mutation before capture is visible here. A later mutation signals
        // the epoch in this snapshot, including before the kernel wait starts.
        if (portContext->GetStatus() == TaskQueuePortStatus::Terminated) return false;
        if (!m_suspended && ((!ReadyQueueEmpty() && !m_queueList->empty()) || !TerminationListEmpty())) return true;

        DWORD remaining = INFINITE;
        if (timeout != INFINITE)
        {
            const auto elapsed = GetTickCount64() - start;
            if (attempted && timeout != 0 && elapsed >= timeout) return false;
            remaining = elapsed < timeout ? timeout - static_cast<DWORD>(elapsed) : 0;
        }
        // Retirement and termination cannot fail due to event allocation.
        const DWORD interval = snapshot.epoch ? remaining : (std::min)(remaining, DWORD{16});
        attempted = true;
        DWORD result = WaitForMultipleObjects(snapshot.events.count(), snapshot.events.data(), FALSE, interval);
        if (result == WAIT_TIMEOUT)
        {
            if (interval == remaining) return false;
            continue;
        }
        if (result >= WAIT_OBJECT_0 + snapshot.events.count())
        {
            if (result == WAIT_FAILED) { FAIL_FAST_IF_FAILED(HRESULT_FROM_WIN32(GetLastError())); }
            return false;
        }
        if (result > WAIT_OBJECT_0 && result <= WAIT_OBJECT_0 + snapshot.registrations.count())
            ProcessWaitCallback(snapshot.registrations[result - WAIT_OBJECT_0 - 1].get());
    }

#else
    while (m_suspended || (ReadyQueueEmpty() && TerminationListEmpty()))
    {
        if (portContext->GetStatus() == TaskQueuePortStatus::Terminated)
        {
            return false;
        }

        std::unique_lock<std::mutex> lock(m_lock);

        if (!m_signaled &&
            m_event.wait_for(lock, std::chrono::milliseconds(timeout)) == std::cv_status::timeout)
        {
            return false;
        }

        m_signaled = false;
    }
#endif

    return true;
}

bool TaskQueuePortImpl::ReadyQueueEmpty()
{
#ifdef _WIN32
    return m_readyEntries.load() == 0;
#else
    return m_queueList->empty();
#endif
}

bool __stdcall TaskQueuePortImpl::IsEmpty()
{
    bool empty =
        (ReadyQueueEmpty()) &&
        (m_pendingList->empty()) &&
        (m_processingCallback == 0);

    return empty;
}

void __stdcall TaskQueuePortImpl::WaitForUnwind()
{
    DefaultUnnamedMutex mutex;
    std::unique_lock<std::mutex> lock(mutex);

    while(m_processingCallback.load() != 0)
    {
        // wait for 10 ms.  We do not modify m_processingCallback under
        // the protection of a mutex because we don't want the hit of
        // taking a lock.  Therefore, we can't wait forever for the
        // cv here.  We could miss it due to a race. This API is only
        // called during task queue termination and therefore some polling
        // is OK.

        std::chrono::milliseconds ms{10};
        auto when = std::chrono::steady_clock::time_point(ms);
        m_processingCallbackCv.wait_until(lock, when);
    }
}

HRESULT __stdcall TaskQueuePortImpl::SuspendTermination(
    _In_ ITaskQueuePortContext* portContext
    )
{
    // This is necessary to ensure there is no race.
    portContext->AddSuspend();
    HRESULT hr = VerifyNotTerminated(portContext);
    
    if (FAILED(hr))
    {
        ResumeTermination(portContext);
        RETURN_HR(hr);
    }

    return S_OK;
}

void __stdcall TaskQueuePortImpl::ResumeTermination(
    _In_ ITaskQueuePortContext* portContext)
{
    if (portContext->RemoveSuspend())
    {
        // Removed the last external callback.  Look for
        // parked terminations and reschedule them.
        // Use a temporary list sharing the same heap to avoid allocation
        LocklessQueue<TerminationEntry*> entries_to_schedule(*m_pendingTerminationList.get());
        
        {
            std::lock_guard<std::mutex> lock(m_terminationLock);
            m_pendingTerminationList->remove_if([&](auto& entry, auto address)
            {
                if (entry->portContext == portContext)
                {
                    entries_to_schedule.push_back(entry, address);
                    return true;
                }

                return false;
            });
        }

        // Schedule entries outside the lock
        TerminationEntry* entry;
        uint64_t address;
        while (entries_to_schedule.pop_front(entry, address))
        {
            entry->node = address;
            ScheduleTermination(entry);
        }
    }
}

void __stdcall TaskQueuePortImpl::SuspendPort()
{
#ifdef _WIN32
    std::lock_guard<std::mutex> lock(m_lock);
#endif
    m_suspended = true;
#ifdef _WIN32
    InvalidateWaitEpochLocked();
#endif
}

void __stdcall TaskQueuePortImpl::ResumePort()
{
    // Before we resume the port iterate over both the
    // pending list and the termination list and count
    // how many notifies we need. Because of the lock
    // free nature of the queue establishing a count
    // takes some effort.

    uint64_t address;
    size_t notifyCount = 0;
#ifndef _WIN32
    QueueEntry queueEntry;
    LocklessQueue<QueueEntry> retainEntries(*(m_pendingList.get()));

    while (m_queueList->pop_front(queueEntry, address))
    {
        notifyCount++;
        retainEntries.push_back(std::move(queueEntry), address);
    }

    while (retainEntries.pop_front(queueEntry, address))
    {
        m_queueList->push_back(std::move(queueEntry), address);
    }
#endif

    {
        std::lock_guard<std::mutex> lock(m_terminationLock);
        TerminationEntry* terminationEntry;
        LocklessQueue<TerminationEntry*> retainTerminations(*(m_terminationList.get()));

        while (m_terminationList->pop_front(terminationEntry, address))
        {
            notifyCount++;
            retainTerminations.push_back(std::move(terminationEntry), address);
        }

        while (retainTerminations.pop_front(terminationEntry, address))
        {
            m_terminationList->push_back(std::move(terminationEntry), address);
        }
    }

#ifdef _WIN32
    {
        std::lock_guard<std::mutex> lock(m_lock);
        m_suspended = false;
        InvalidateWaitEpochLocked();
        notifyCount += m_readyEntries.load();
    }
#else
    m_suspended = false;
#endif

    while (notifyCount)
    {
        SignalQueue();
        NotifyItemQueued();
        notifyCount--;
    }
}

HRESULT TaskQueuePortImpl::VerifyNotTerminated(
    _In_ ITaskQueuePortContext* portContext)
{
    // N.B.  This looks wrong but it's not.  We only error adding new items
    // if we are terminating or terminated.  If we're just canceled we allow
    // new items in but we invoke them with the canceled flag set to true.
    RETURN_HR_IF(E_ABORT, portContext->GetStatus() > TaskQueuePortStatus::Canceled);
    return S_OK;
}

bool TaskQueuePortImpl::IsCallCanceled(_In_ const QueueEntry& entry)
{
    return entry.portContext->GetStatus() != TaskQueuePortStatus::Active;
}

// Appends the given entry to the active queue.  The entry should already
// be add-refd. This will return false on failure.
bool TaskQueuePortImpl::AppendEntry(
    _In_ const QueueEntry& entry,
    _In_opt_ uint64_t node)
{
#ifdef _WIN32
    if (node == 0 && !m_queueList->reserve_node(node)) return false;
    ++m_readyEntries;
    m_queueList->push_back(entry, node);
#else
    if (node != 0)
    {
        m_queueList->push_back(entry, node);
    }
    else if (!m_queueList->push_back(entry))
    {
        return false;
    }

#endif

    SignalQueue();
    NotifyItemQueued();

    return true;
}

void TaskQueuePortImpl::CancelPendingEntries(
    _In_ ITaskQueuePortContext* portContext,
    _In_ bool appendToQueue)
{
    // Only move entries owned by the terminating queue. Sibling delegates
    // share this port's delayed-callback timer state, so leave m_timer and
    // m_timerDue alone; if we removed the armed earliest entry, the existing
    // timer simply takes one blank fire and re-arms for the next real item.
    //
    // Lifetime: every caller holds its own reference on portContext (and so on
    // its queue, which keeps this port alive), so the entry releases below are
    // never the final release and this port and portContext stay valid for
    // the rest of the function. Still free each node before releasing its
    // reference so nothing touches the port after a release.
    LocklessQueue<QueueEntry> entriesToAppend(*m_queueList.get());

    m_pendingList->remove_if([&](auto& entry, auto address)
    {
        if (entry.portContext == portContext)
        {
            if (appendToQueue)
            {
                entriesToAppend.push_back(std::move(entry), address);
            }
            else
            {
                m_pendingList->free_node(address);
                entry.portContext->Release();
            }

            return true;
        }

        return false;
    });

    while (appendToQueue)
    {
        QueueEntry entry = {};
        uint64_t address = 0;
        if (!entriesToAppend.pop_front(entry, address))
        {
            break;
        }

        if (!AppendEntry(entry, address))
        {
            m_queueList->free_node(address);
            entry.portContext->Release();
        }
    }

    // Test hook: let unit tests enqueue a sibling delayed callback while this
    // termination path still owns the interleaving window that used to race
    // with SubmitPendingCallbacks().
    if (auto hooks = portContext->GetQueue()->GetTestHooks(); hooks != nullptr)
    {
        hooks->PendingEntriesRemovedDuringTermination(portContext->GetType());
    }

    // We may have just removed the pending entry that the shared delayed-callback
    // timer was armed for (a composite that shares this port can terminate while
    // it still owns the armed-earliest entry). Relying on a "blank fire" of the
    // now-orphaned deadline to re-arm is fragile under the STL timer backend, so
    // re-point the timer at the earliest surviving pending entry directly.
    RearmTimerForEarliestPending();
    
#ifdef _WIN32
    
    StaticArray<referenced_ptr<WaitRegistration>, PORT_WAIT_MAX> retired;
    referenced_ptr<ITaskQueuePortContext> discarded[PORT_WAIT_MAX];
    uint32_t notifications = 0;
    {
        std::lock_guard<std::mutex> lock(m_lock);
        for (auto it = m_waits.begin(); it != m_waits.end();)
        {
            auto& reg = it->second;
            if (reg->queueEntry.portContext != portContext ||
                reg->state != WaitRegistration::State::Active)
            {
                ++it;
                continue;
            }

            retired.append(reg);
            if (appendToQueue && !reg->appended)
            {
                uint64_t node = reg->cancelNode;
                reg->cancelNode = 0;
                if (PublishWaitEntryLocked(reg.get(), node)) ++notifications;
            }
            RetireWaitLocked(reg.get(), appendToQueue ?
                WaitRegistration::State::Canceled : WaitRegistration::State::Removed);
            if (!appendToQueue && reg->queuedContext)
            {
                discarded[retired.count() - 1] = std::move(reg->queuedContext);
                --m_readyEntries;
            }
            ++it; // Keep the token discoverable through native drainage.
        }
    }

    for (uint32_t i = 0; i < retired.count(); ++i)
        DrainRetiredWait(retired[i]);
    if (retired.count()) SignalQueue();
    for (uint32_t i = 0; i < notifications; ++i) NotifyItemQueued();
#endif
}

// After cancellation removed pending entries, re-point the shared timer at the
// earliest surviving pending entry instead of leaving it armed for a deadline
// whose entry was just removed. Without this, a composite that shares this
// port's timer can terminate while owning the armed-earliest entry, leaving the
// timer pointed at an orphaned deadline; if the orphaned fire is lost (STL timer
// backend), the remaining delayed callbacks never run and the queue starves.
//
// Only called from CancelPendingEntries, whose callers keep this port alive
// (see there), so releasing the temporary nextItem reference below can never
// destroy this port even if it is the last reference to a sibling queue.
void TaskQueuePortImpl::RearmTimerForEarliestPending()
{
    while (true)
    {
        // Snapshot before scanning so a concurrent QueueItem publish makes the
        // compare-exchange below fail instead of being overwritten.
        uint64_t currentDue = m_timerDue.load();
        QueueEntry nextItem = {};
        bool hasNextItem = false;

        // Read-only scan: keep every entry (return false), tracking the earliest.
        m_pendingList->remove_if([&](auto& entry, auto /*address*/)
        {
            if (!hasNextItem || nextItem.enqueueTime > entry.enqueueTime)
            {
                if (hasNextItem)
                {
                    nextItem.portContext->Release();
                }

                nextItem = entry;
                nextItem.portContext->AddRef();
                hasNextItem = true;
            }

            return false;
        });

        if (hasNextItem)
        {
            bool timerStable = true;
            bool movedTimerLater = false;
            if (nextItem.enqueueTime < currentDue)
            {
                // Surviving entry is earlier than the armed deadline: min-wins arm.
                timerStable = ArmTimerIfEarlier(nextItem.enqueueTime);
            }
            else if (nextItem.enqueueTime > currentDue)
            {
                // The armed deadline belonged to a removed entry; advance the timer
                // to the earliest surviving deadline using the same verified helper
                // the fire path uses (handles a concurrent earlier publish safely).
                timerStable = ArmTimerForNextPendingDueTime(currentDue, nextItem.enqueueTime);
                movedTimerLater = timerStable;
            }

            nextItem.portContext->Release();

            if (!timerStable)
            {
                continue;
            }

            // A submission can land after the scan while the old, earlier due
            // time is still published. Once we move that due time later, scan
            // again to catch the intermediate entry. Repeat after every actual
            // deadline advance until a scan reaches a fixed point.
            if (movedTimerLater)
            {
                continue;
            }

            return;
        }

        if (currentDue == UINT64_MAX)
        {
            return;
        }

        // No pending entries remain on any attached context; let the timer go
        // idle. Mirrors the no-next-item branch of PromoteReadyPendingCallbacks
        // (CAS only, never Cancel(), to avoid racing concurrent Start calls).
        if (!m_timerDue.compare_exchange_strong(currentDue, UINT64_MAX))
        {
            continue;
        }

        // A callback can be queued after the scan while currentDue still names
        // the removed entry. Rescan once after publishing the idle sentinel so
        // that callback is either found here or arms itself against UINT64_MAX.
        continue;
    }
}

void TaskQueuePortImpl::EraseQueue(
    _In_opt_ LocklessQueue<QueueEntry>* queue)
{
    if (queue != nullptr)
    {
        QueueEntry entry;
        
        while(queue->pop_front(entry))
        {
#ifdef _WIN32
            if (entry.waitRegistration)
            {
                entry.waitRegistration->Release();
                continue;
            }
#endif
            entry.portContext->Release();
        }
    }
}

// Arms the OS timer for dueTime using min-wins CAS with post-Start
// verification. If another thread publishes an earlier deadline between
// our CAS and Start, we detect the overwrite and re-arm. This closes the
// TOCTOU window that could strand a pending entry.
//
// Uses <= so callers needing to re-arm for an already-published deadline
// (e.g. SubmitPendingCallbacks on an early timer fire) go through the
// same verified path.
//
// Returns true when the timer is stable (armed at or before dueTime, or
// dueTime is UINT64_MAX). Returns false if m_timerDue moved later (entry
// was promoted), signaling the caller to re-evaluate.
bool TaskQueuePortImpl::ArmTimerIfEarlier(uint64_t dueTime)
{
    while (true)
    {
        uint64_t currentDue = m_timerDue.load();

        if (dueTime <= currentDue)
        {
            if (dueTime == UINT64_MAX)
            {
                return true; // Nothing to arm.
            }

            if (m_timerDue.compare_exchange_weak(currentDue, dueTime))
            {
                m_timer.Start(dueTime);

                // Post-Start verification: did m_timerDue change between
                // our CAS and Start? If not, the timer is correctly armed.
                uint64_t afterDue = m_timerDue.load();
                if (afterDue == dueTime)
                {
                    return true; // Unchanged — timer correctly armed.
                }

                if (afterDue < dueTime)
                {
                    // An earlier deadline was published. Our Start may
                    // have overwritten a concurrent arm. Fix it.
                    dueTime = afterDue;
                    continue;
                }

                // m_timerDue moved later (e.g. UINT64_MAX from promotion).
                // Our entry was already handled. Caller should re-evaluate.
                return false;
            }
            // CAS failed (concurrent modification). Retry with fresh read.
            continue;
        }

        // An earlier deadline is already published; the timer is already
        // armed for it or another thread is in the process of arming it
        // (with their own post-Start verification).
        return true;
    }
}

// Replaces the due time that just fired with the next surviving future
// deadline. Unlike ArmTimerIfEarlier, this helper is allowed to move the
// published due time later, but only while the caller's observed due time is
// still current. If another thread already published an earlier/equal
// deadline, leave it alone. Returns false when the published due time moved
// later after Start(), signaling the caller to rescan the pending list.
bool TaskQueuePortImpl::ArmTimerForNextPendingDueTime(
    uint64_t previousDueTime,
    uint64_t nextDueTime)
{
    while (true)
    {
        if (m_timerDue.compare_exchange_strong(previousDueTime, nextDueTime))
        {
            m_timer.Start(nextDueTime);

            uint64_t afterDue = m_timerDue.load();
            if (afterDue == nextDueTime)
            {
                return true;
            }

            if (afterDue < nextDueTime)
            {
                // Our Start may have overwritten the earlier publisher's Start.
                // Re-arm the currently published deadline so the final OS timer
                // programming always agrees with m_timerDue.
                return ArmTimerIfEarlier(afterDue);
            }

            return false;
        }

        // CAS failed: compare_exchange loaded the current m_timerDue into
        // previousDueTime. If that value is already <= nextDueTime, the
        // timer is armed for an earlier-or-equal deadline and we're done.
        if (previousDueTime <= nextDueTime)
        {
            return true;
        }
    }
}

// Re-arms the exact due time observed by an early/stale timer callback.
// If another thread has already consumed that due time and moved m_timerDue
// later (including to UINT64_MAX), the observed due is stale and the caller
// must re-evaluate instead of resurrecting it.
bool TaskQueuePortImpl::RearmTimerIfDueTimeUnchanged(uint64_t dueTime)
{
    // UINT64_MAX is the "nothing armed" sentinel and must never be programmed
    // as a real deadline (see SubmitPendingCallbacks). Mirrors the guard in
    // ArmTimerIfEarlier.
    if (dueTime == UINT64_MAX)
    {
        return true;
    }

    while (true)
    {
        uint64_t currentDue = m_timerDue.load();

        if (currentDue < dueTime)
        {
            return true;
        }

        if (currentDue > dueTime)
        {
            return false;
        }

        if (m_timerDue.compare_exchange_weak(currentDue, dueTime))
        {
            m_timer.Start(dueTime);

            uint64_t afterDue = m_timerDue.load();
            if (afterDue == dueTime)
            {
                return true;
            }

            if (afterDue < dueTime)
            {
                dueTime = afterDue;
                continue;
            }

            return false;
        }
    }
}

// Promote every pending callback whose deadline has arrived, then arm the
// timer for the earliest remaining future deadline. Sweeping all
// enqueueTime <= now avoids treating timestamps as unique identities, so
// equal-deadline siblings and stale timer callbacks follow the same rule.
void TaskQueuePortImpl::PromoteReadyPendingCallbacks(
    _In_ uint64_t dueTime,
    _In_ uint64_t now)
{
    // Releasing a port context can drop the last reference to its queue, and the queue owns this
    // port. On the STL WaitTimer that release happens right here on the timer thread, so the port
    // (including m_queueList, which readyEntries below shares) would be destroyed while this
    // function is still using it (AddressSanitizer heap-use-after-free). Hand every reference this
    // function would release to this holder instead. Any held context keeps its queue, and so this
    // port, alive; the holder is declared before the loop, so the last release happens after every
    // other local (readyEntries) is gone and nothing touches the port afterwards.
    struct DeferredPortContextRelease
    {
        ITaskQueuePortContext* held = nullptr;

        void Hold(_In_ ITaskQueuePortContext* portContext) noexcept
        {
            ITaskQueuePortContext* previous = held;
            held = portContext;
            if (previous != nullptr)
            {
                previous->Release(); // safe: 'held' keeps the port alive
            }
        }

        ~DeferredPortContextRelease()
        {
            if (held != nullptr)
            {
                held->Release();
            }
        }
    } deferredRelease;

    for (;;)
    {
        // Collect due entries locally first and only touch the active queue
        // after remove_if completes. Keeping the sweep phase and the publish
        // phase separate preserves the "promote all ready entries" behavior
        // without asking remove_if to coexist with queue wakeups and
        // cross-queue node reuse at the same time.
        LocklessQueue<QueueEntry> readyEntries(*m_queueList.get());

        QueueEntry nextItem = {};
        bool hasNextItem = false;

        m_pendingList->remove_if([&](auto& entry, auto address)
        {
            // Any entry whose deadline has passed is ready right now,
            // regardless of whether its timestamp aliases another entry or
            // whether this timer fire is the original notification or a
            // stale callback that arrived late.
            if (entry.enqueueTime <= now)
            {
                readyEntries.push_back(std::move(entry), address);

                return true;
            }

            if (!hasNextItem || nextItem.enqueueTime > entry.enqueueTime)
            {
                // remove_if works by removing items from the list and
                // re-adding them if this callback returns false. If we
                // are going to keep an item beyond this callback we need
                // to make sure fields we're using stay valid. Only the
                // port context is a risk.

                if (hasNextItem)
                {
                    nextItem.portContext->Release();
                }

                nextItem = entry;
                nextItem.portContext->AddRef();
                hasNextItem = true;
            }

            return false;
        });

        // Publish the ready entries after the pending-list walk finishes.
        QueueEntry readyEntry = {};
        uint64_t readyEntryNode = 0;
        while (readyEntries.pop_front(readyEntry, readyEntryNode))
        {
            if (!AppendEntry(readyEntry, readyEntryNode))
            {
                m_queueList->free_node(readyEntryNode);
                deferredRelease.Hold(readyEntry.portContext);
            }
        }

        if (hasNextItem)
        {
            bool timerAdvanced = false;
            if (nextItem.portContext->GetStatus() == TaskQueuePortStatus::Active)
            {
                if (nextItem.enqueueTime != dueTime)
                {
                    VisitAttachedContexts([&](ITaskQueuePortContext* portContext)
                    {
                        auto hooks = portContext->GetQueue()->GetTestHooks();
                        if (hooks != nullptr)
                        {
                            hooks->NextPendingCallbackFound(
                                portContext->GetType(),
                                dueTime,
                                nextItem.enqueueTime);
                        }
                    });
                }

                // A rescue scan that finds the deadline it just published has
                // reached a fixed point. Otherwise replace the fired due time
                // with the earliest future deadline that survived the sweep.
                if (nextItem.enqueueTime != dueTime &&
                    !ArmTimerForNextPendingDueTime(dueTime, nextItem.enqueueTime))
                {
                    deferredRelease.Hold(nextItem.portContext);
                    now = m_timer.GetCurrentTime();
                    dueTime = m_timerDue.load();
                    continue;
                }
                timerAdvanced = nextItem.enqueueTime != dueTime;
            }
            else
            {
                // The port is no longer active. Pending entries are canceled
                // when the port is terminated, but if we were iterating above
                // it's possible that we removed an item while the termination
                // was being processed and it got missed.
                CancelPendingEntries(nextItem.portContext, true);
            }

            deferredRelease.Hold(nextItem.portContext);

            // A callback can be inserted after the scan while the fired due
            // time is still published. It will not arm itself if its deadline
            // falls between dueTime and nextItem.enqueueTime, so rescan after
            // every successful advance until a scan reaches a fixed point.
            // Later submissions see the advanced deadline and arm themselves.
            if (timerAdvanced)
            {
                now = m_timer.GetCurrentTime();
                dueTime = m_timerDue.load();
                continue;
            }

            return;
        }

        // No future entries remain in the pending list.
        uint64_t noDueTime = UINT64_MAX;

        VisitAttachedContexts([&](ITaskQueuePortContext* portContext)
        {
            auto hooks = portContext->GetQueue()->GetTestHooks();
            if (hooks != nullptr)
            {
                hooks->NoNextPendingCallbackFound(
                    portContext->GetType(),
                    dueTime);
            }
        });

        if (m_timerDue.compare_exchange_strong(dueTime, noDueTime))
        {
            // Bug fix: ScheduleNextPendingCallback timer race results
            // in lost delayed task wakes. Don't cancel the timer here
            // as another scheduled callback could have been added.
            // The CAS above is sufficient: the timer has already fired
            // (call site 1: SubmitPendingCallbacks) or was already
            // canceled (call site 2: CancelPendingEntries).  A Cancel()
            // here raced with concurrent QueueItem/Start calls on other
            // threads, permanently stranding entries in m_pendingList.
            // See VerifyDelayedCallbackTimerRaceOnManualQueue for full
            // analysis. The test hook here allows unit tests to verify
            // there is no race.
            VisitAttachedContexts([&](ITaskQueuePortContext* portContext)
            {
                auto hooks = portContext->GetQueue()->GetTestHooks();
                if (hooks != nullptr)                {
                    hooks->NextPendingCallbackScheduled(
                        portContext->GetType(),
                        dueTime,
                        noDueTime);
                }
            });

            // A concurrent QueueItem can append a future entry after our
            // sweep has already concluded there is no next item, but before
            // we publish UINT64_MAX here. Instead of recursing (which has
            // no tail-call guarantee and risks stack growth under sustained
            // contention), loop back for a rescue sweep. If nothing landed,
            // the second pass is a cheap no-op.
            if (dueTime != noDueTime)
            {
                now = m_timer.GetCurrentTime();
                dueTime = noDueTime;
                continue;
            }
        }

        return;
    }
}

void TaskQueuePortImpl::SubmitPendingCallbacks()
{
    while (true)
    {
        uint64_t dueTime = m_timerDue.load();

        // UINT64_MAX is the "nothing armed" sentinel. It must never be programmed
        // as a real deadline: WaitTimer::Start(UINT64_MAX) converts to a duration
        // of -1 (before the steady_clock epoch), a perpetually-past deadline that
        // fires immediately and re-arms the sentinel in an infinite loop, starving
        // every real pending callback. If the timer is disarmed there is nothing
        // to sweep, so return.
        if (dueTime == UINT64_MAX)
        {
            return;
        }

        // Timer callbacks are advisory: a threadpool fire can arrive after
        // retargeting, or slightly before the steady-clock deadline due to
        // clock-source differences on Win32. If the deadline hasn't arrived,
        // re-arm the same published due time rather than silently dropping the
        // callback (which would strand the pending entry).
        const uint64_t now = m_timer.GetCurrentTime();
        if (now < dueTime)
        {
            if (RearmTimerIfDueTimeUnchanged(dueTime))
            {
                return;
            }
            continue;
        }

        PromoteReadyPendingCallbacks(dueTime, now);
        return;
    }
}

// Called from thread pool callback
void TaskQueuePortImpl::ProcessThreadPoolCallback(_In_ OS::ThreadPoolActionStatus& status)
{
    if (m_dispatchMode == XTaskQueueDispatchMode::SerializedThreadPool)
    {
        uint32_t wasProcessing = m_processingSerializedTbCallback++;

        if (wasProcessing == 0)
        {
            // Skipped workers leave their notifications counted for the owner.
            // Short-circuiting decrements the count only after an empty drain.
            // Decrementing to zero atomically releases ownership: a notification
            // before release forces another drain; one after release acquires
            // ownership.
            while (DrainOneItem(status) || --m_processingSerializedTbCallback != 0);
        }
    }
    else
    {
        DrainOneItem(status);
    }

    // Important that this comes before Release; otherwise
    // cleanup may deadlock.
    status.Complete();

    // When we submitted a request to the thread pool we
    // added a reference to ourself.  Balance it here. This
    // may be the final release.
    Release();
}

void TaskQueuePortImpl::SignalQueue()
{
    if (!m_suspended)
    {
#ifdef _WIN32
        SetEvent(m_queueEvent);
#else
        std::unique_lock<std::mutex> lock(m_lock);
        m_signaled = true;
        m_event.notify_all();
#endif
    }
}

void TaskQueuePortImpl::NotifyItemQueued()
{
    // If this is an immediate queue, ignore suspends as
    // we need to dispatch synchronously as part of the call.

    if (!m_suspended || m_dispatchMode == XTaskQueueDispatchMode::Immediate)
    {
        switch (m_dispatchMode)
        {
        case XTaskQueueDispatchMode::Manual:
            // nothing
            break;

        case XTaskQueueDispatchMode::SerializedThreadPool:
        case XTaskQueueDispatchMode::ThreadPool:
            // Addref before submitting to the thread pool in case we
            // are released while there there are outstanding threadpool
            // items. The threadpool does not cancel outstanding callbacks
            // on termination so we need to drain before releasing.
            AddRef();
            m_threadPool.Submit();
            break;

        case XTaskQueueDispatchMode::Immediate:
            // We will handle this after we invoke
            // callback submitted.
            break;
        }

        VisitAttachedContexts([](ITaskQueuePortContext* portContext)
        {
            portContext->ItemQueued();
        });

        // If the queue is immediate, drain the newly queued item
        // now.

        if (m_dispatchMode == XTaskQueueDispatchMode::Immediate)
        {
            DrainOneItem();
        }
    }
}

void TaskQueuePortImpl::SignalTerminations()
{
    // Collect entries to process outside the iteration to avoid concurrent modification races
    // when callbacks invoke nested Terminate() calls.
    // Use a temporary list sharing the same heap to avoid allocation.
    LocklessQueue<TerminationEntry*> entries_to_process(*m_terminationList.get());

    {
        std::lock_guard<std::mutex> lock(m_terminationLock);
        m_terminationList->remove_if([&entries_to_process](auto& entry, auto address)
        {
            if (entry->portContext->GetStatus() >= TaskQueuePortStatus::Terminating)
            {
                entry->portContext->SetStatus(TaskQueuePortStatus::Terminated);
                entries_to_process.push_back(entry, address);
                return true;
            }

            return false;
        });
    }

#ifdef _WIN32
    if (!entries_to_process.empty())
    {
        std::lock_guard<std::mutex> lock(m_lock);
        InvalidateWaitEpochLocked();
    }
#endif

    // Now process callbacks outside the remove_if iteration
    // This prevents races when callbacks invoke nested operations like Terminate()
    TerminationEntry* entry;
    uint64_t address;
    while (entries_to_process.pop_front(entry, address))
    {
        // AddRef portContext to prevent UAF if callback releases the queue
        auto context = entry->portContext;
        context->AddRef();
        
        entry->callback(entry->callbackContext);
                
        {
            std::lock_guard<std::mutex> lock(m_terminationLock);
            m_terminationList->free_node(address);
        }

        delete entry;
        context->Release();
    }
}

void TaskQueuePortImpl::ScheduleTermination(
    _In_ TerminationEntry* term)
{
    // Insert the termination callback into the queue.  Even if the
    // main queue is empty, we still signal it and run through
    // a cycle.  This ensures we flush the queue out with no 
    // races and that the termination callback happens on the right
    // thread.

    // This never fails because we preallocate the
    // list node.
    {
        std::lock_guard<std::mutex> lock(m_terminationLock);
        m_terminationList->push_back(term, term->node);
        term->node = 0; // now owned by the list
    }

    // The port should have already been marked as terminated, so now
    // we can signal it to wake up. This should drain pending calls and
    // invoke the termination callback we just pushed.

    SignalQueue();
    NotifyItemQueued();
}

bool TaskQueuePortImpl::TerminationListEmpty()
{
    std::lock_guard<std::mutex> lock(m_terminationLock);
    return m_terminationList->empty();
}

#ifdef _WIN32
void TaskQueuePortImpl::InvalidateWaitEpochLocked()
{
    // Never reset or reuse a signaled event. A late waiter on the previous
    // snapshot still observes it even if a new epoch already exists.
    if (m_waitEpoch)
    {
        SetEvent(m_waitEpoch->handle);
        m_waitEpoch.reset();
    }
}

void TaskQueuePortImpl::CaptureWaitSnapshot(WaitSnapshot& snapshot)
{
    snapshot.events.append(m_queueEvent);
    std::lock_guard<std::mutex> lock(m_lock);
    static_assert(PORT_EVENT_MAX <= MAXIMUM_WAIT_OBJECTS);
    if (!m_waitEpoch)
    {
        try
        {
            auto epoch = std::make_shared<WaitHandleOwner>();
            epoch->handle = CreateEventW(nullptr, TRUE, FALSE, nullptr);
            if (epoch->handle) m_waitEpoch = std::move(epoch);
        }
        catch (const std::bad_alloc&)
        {
            // Wait() uses bounded polling until an epoch can be allocated.
        }
    }
    snapshot.epoch = m_waitEpoch;
    for (const auto& item : m_waits)
    {
        if (!m_suspended && item.second->state == WaitRegistration::State::Active)
        {
            snapshot.registrations.append(item.second);
            snapshot.owners.append(item.second->handleOwner);
            snapshot.events.append(item.second->handleOwner->handle);
        }
    }
    if (snapshot.epoch) snapshot.events.append(snapshot.epoch->handle);
}

void CALLBACK TaskQueuePortImpl::WaitCallback(
    _In_ PTP_CALLBACK_INSTANCE instance,
    _Inout_opt_ void* context,
    _Inout_ PTP_WAIT wait,
    _In_ TP_WAIT_RESULT waitResult)
{
    UNREFERENCED_PARAMETER(wait);
    if (waitResult != WAIT_OBJECT_0 || !context) return;

    // Retirement holds the registration through WaitForThreadpoolWaitCallbacks.
    // Acquire independent owners before allowing a reentrant close to finish.
    referenced_ptr<TaskQueuePortImpl> port;
    referenced_ptr<WaitRegistration> reg(static_cast<WaitRegistration*>(context));
    if (!reg->port->TryAddRef()) return;
    *port.address_of() = reg->port;
    DisassociateCurrentThreadFromCallback(instance);
    port->ProcessWaitCallback(reg.get());
}

TaskQueuePortImpl::WaitRegistration::~WaitRegistration()
{
    ASSERT(cancelNode == 0);
    if (threadpoolWait) CloseThreadpoolWait(threadpoolWait); // Unarmed registration failure.
}

DefaultUnnamedMutex TaskQueuePortImpl::WaitCallbackFrame::lock;
TaskQueuePortImpl::WaitCallbackFrame* TaskQueuePortImpl::WaitCallbackFrame::head = nullptr;

TaskQueuePortImpl::WaitCallbackFrame::WaitCallbackFrame(WaitRegistration* registration) : reg(registration)
{
    std::lock_guard<std::mutex> guard(lock);
    link = &head;
    next = head;
    if (next) next->link = &next;
    head = this;
}

TaskQueuePortImpl::WaitCallbackFrame::~WaitCallbackFrame()
{
    std::lock_guard<std::mutex> guard(lock);
    *link = next;
    if (next) next->link = link;
}

bool TaskQueuePortImpl::WaitCallbackFrame::ConfirmEntry()
{
    // Only the indirect user call is inside a frame. Synchronous reentry from
    // that call witnesses physical entry, including its enclosing callbacks.
    ULONG_PTR low, high;
    GetCurrentThreadStackLimits(&low, &high);
    // A fiber can suspend its callback and resume on a different thread.
    // Identify the currently executing stack rather than reusing a thread ID.
    std::lock_guard<std::mutex> guard(lock);
    bool reentrant = false;
    for (auto* frame = head; frame; frame = frame->next)
    {
        const auto address = reinterpret_cast<ULONG_PTR>(frame);
        if (address < low || address >= high) continue;
        reentrant = true;
        auto* reg = frame->reg;
        std::lock_guard<std::mutex> lock(reg->port->m_lock);
        reg->enteredUserCode = true;
        reg->port->m_waitFinished.notify_all();
    }
    return reentrant;
}

void TaskQueuePortImpl::DrainRetiredWait(const referenced_ptr<WaitRegistration>& reg)
{
    // Every participant owns a reference. The first caller drains outside the
    // port lock; associated callbacks have no dependency on user code here.
    std::call_once(reg->drainOnce, [&] {
        WaitForThreadpoolWaitCallbacks(reg->threadpoolWait, TRUE);
        CloseThreadpoolWait(reg->threadpoolWait);
        std::lock_guard<std::mutex> lock(m_lock);
        reg->threadpoolWait = nullptr;
        reg->handleOwner.reset();
        reg->drained = true;
        FinishRetirementLocked(reg.get());
        m_waitFinished.notify_all();
    });
}

void TaskQueuePortImpl::FinishRetirementLocked(WaitRegistration* reg)
{
    if (!reg->drained || reg->queuedContext || reg->running) return;
    // The registration owns the token table independently of the queue.
    // Public lookup drops the table lock before entering the port.
    if (reg->publicRegistry)
    {
        reg->publicRegistry->Remove(reg->publicToken);
        reg->publicRegistry.reset();
    }
    m_waits.erase(reg->token);
}

void TaskQueuePortImpl::RetireWaitLocked(WaitRegistration* reg, WaitRegistration::State state)
{
    const bool wasActive = reg->state == WaitRegistration::State::Active;
    reg->state = state;
    if (wasActive)
    {
        SetThreadpoolWait(reg->threadpoolWait, nullptr, nullptr);
        --m_activeWaits;
        if (reg->cancelNode)
        {
            m_queueList->free_node(reg->cancelNode);
            reg->cancelNode = 0;
        }
        InvalidateWaitEpochLocked();
    }
}

bool TaskQueuePortImpl::PublishWaitEntryLocked(WaitRegistration* reg, uint64_t node)
{
    // The node is already reserved. Keep its context pin revocable until
    // dispatch claims it, so unregister can release an undispatched queue.
    ASSERT(node != 0);
    if (!reg->pinnable->TryPin())
    {
        m_queueList->free_node(node);
        return false;
    }
    *reg->queuedContext.address_of() = reg->queueEntry.portContext; // Adopt TryPin.
    reg->AddRef();
    reg->appended = true;
    ++m_readyEntries;
    m_queueList->push_back(reg->queueEntry, node);
    return true;
}

void TaskQueuePortImpl::ProcessWaitCallback(WaitRegistration* reg)
{
    {
        std::lock_guard<std::mutex> lock(m_lock);
        if (reg->state != WaitRegistration::State::Active || reg->appended) return;
        uint64_t node = 0;
        if (!m_queueList->reserve_node(node))
        {
            SetThreadpoolWait(reg->threadpoolWait, reg->handleOwner->handle, nullptr);
            return;
        }
        if (!PublishWaitEntryLocked(reg, node)) return;
    }
    SignalQueue();
    NotifyItemQueued();
}

bool TaskQueuePortImpl::DispatchWaitEntry(const QueueEntry& entry, OS::ThreadPoolActionStatus& status,
    referenced_ptr<ITaskQueuePortContext>& context)
{
    referenced_ptr<WaitRegistration> reg;
    *reg.address_of() = entry.waitRegistration; // Adopt the popped entry's ref.
    {
        std::lock_guard<std::mutex> lock(m_lock);
        if (reg->state == WaitRegistration::State::Removed) return false;
        context = std::move(reg->queuedContext);
        --m_readyEntries;
        reg->running = true;
        reg->enteredUserCode = false;
    }

    if (context->GetType() == XTaskQueuePort::Work) status.MayRunLong();
    const bool canceled = IsCallCanceled(entry);
    {
        WaitCallbackFrame frame(reg.get());
        entry.callback(entry.callbackContext, canceled);
    }

    {
        std::lock_guard<std::mutex> lock(m_lock);
        reg->running = false;
        reg->appended = false;
        if (reg->state == WaitRegistration::State::Active)
            SetThreadpoolWait(reg->threadpoolWait, reg->handleOwner->handle, nullptr);
        else
            FinishRetirementLocked(reg.get());
    }
    m_waitFinished.notify_all();
    return true;
}

#endif

//
// TaskQueuePortContextImpl
//

TaskQueuePortContextImpl::TaskQueuePortContextImpl(
    _In_ TaskQueueImpl* queue,
    _In_ XTaskQueuePort type,
    _In_ SubmitCallback* submitCallback) :
    m_queue(queue),
    m_type(type),
    m_submitCallback(submitCallback)
{
}
    
bool TaskQueuePortContextImpl::TryPin() noexcept
{
    return m_queue->TryAddRef();
}
uint32_t __stdcall TaskQueuePortContextImpl::AddRef()
{
    return m_queue->AddRef();
}

uint32_t __stdcall TaskQueuePortContextImpl::Release()
{
    return m_queue->Release();
}

HRESULT __stdcall TaskQueuePortContextImpl::QueryApi(_In_ ApiId id, _Out_ void** ptr)
{
    if (ptr == nullptr)
    {
        return E_POINTER;
    }

    *ptr = nullptr;

    if (id == TaskQueueInternal::kPinnableContextApiId)
    {
        // This concrete implementation can safely identify itself.
        // The caller owns one reference, just like other QueryApi results.
        *ptr = static_cast<TaskQueuePortContextImpl*>(this);
        AddRef();
        return S_OK;
    }

    if (id == ApiId::Identity || id == ApiId::TaskQueuePortContext)
    {
        *ptr = static_cast<ITaskQueuePortContext*>(this);
        AddRef();
        return S_OK;
    }
    
    return E_NOINTERFACE;
}
    
XTaskQueuePort __stdcall TaskQueuePortContextImpl::GetType()
{
    return m_type;
}

TaskQueuePortStatus __stdcall TaskQueuePortContextImpl::GetStatus()
{
    return m_status;
}

ITaskQueue* __stdcall TaskQueuePortContextImpl::GetQueue()
{
    return m_queue;
}

ITaskQueuePort* __stdcall TaskQueuePortContextImpl::GetPort()
{
    return Port.get();
}

bool __stdcall TaskQueuePortContextImpl::TrySetStatus(
    _In_ TaskQueuePortStatus expectedStatus,
    _In_ TaskQueuePortStatus status)
{
    return m_status.compare_exchange_strong(expectedStatus, status);
}
    
void __stdcall TaskQueuePortContextImpl::SetStatus(
    _In_ TaskQueuePortStatus status)
{
    m_status = status;
}

void __stdcall TaskQueuePortContextImpl::ItemQueued()
{
    m_submitCallback->Invoke(m_type);
}

bool __stdcall TaskQueuePortContextImpl::AddSuspend()
{
    return (m_suspendCount.fetch_add(1) == 0);
}

bool __stdcall TaskQueuePortContextImpl::RemoveSuspend()
{
    for(;;)
    {
        uint32_t current = m_suspendCount.load();

        // These should always be balanced and there is no
        // valid case where this should happen, even
        // for multiple threads racing.

        if (current == 0)
        {
            ASSERT(false);
            return true;
        }

        if (m_suspendCount.compare_exchange_weak(current, current -1))
        {
            return current == 1;
        }
    }
}

//
// TaskQueueImpl
//

TaskQueueImpl::TaskQueueImpl() :
    Api(),
    m_callbackSubmitted(&m_header),
    m_work(this, XTaskQueuePort::Work, &m_callbackSubmitted),
    m_completion(this, XTaskQueuePort::Completion, &m_callbackSubmitted)
{
    m_header.m_signature = TASK_QUEUE_SIGNATURE;
    m_header.m_runtimeIteration = GetCurrentRuntimeIteration();
    m_header.m_queue = this;
    m_termination.terminated = false;
}

TaskQueueImpl::~TaskQueueImpl()
{
    // Zero the header so we get an early warning of using
    // a released object.
    m_header = {};
}

HRESULT TaskQueueImpl::Initialize(
    _In_ XTaskQueueDispatchMode workMode,
    _In_ XTaskQueueDispatchMode completionMode)
{
    RETURN_IF_NULL_ALLOC(m_waitRegistry.get());
    referenced_ptr<TaskQueuePortImpl> work(new (std::nothrow) TaskQueuePortImpl);
    RETURN_IF_NULL_ALLOC(work);
    RETURN_IF_FAILED(work->Initialize(workMode));

    referenced_ptr<TaskQueuePortImpl> completion(new (std::nothrow) TaskQueuePortImpl);
    RETURN_IF_NULL_ALLOC(completion);
    RETURN_IF_FAILED(completion->Initialize(completionMode));
    
    work->GetHandle()->m_queue = this;
    completion->GetHandle()->m_queue = this;
    
    RETURN_IF_FAILED(work->QueryApi(ApiId::TaskQueuePort, (void**)&m_work.Port));
    RETURN_IF_FAILED(completion->QueryApi(ApiId::TaskQueuePort, (void**)&m_completion.Port));

    RETURN_IF_FAILED(m_work.Port->Attach(&m_work));
    RETURN_IF_FAILED(m_completion.Port->Attach(&m_completion));

#ifdef SUSPEND_API
    RETURN_IF_FAILED(m_suspendHandler.Initialize(ProcessGlobals::g_suspendState, this));
#endif

    return S_OK;
}

HRESULT TaskQueueImpl::Initialize(
    _In_ XTaskQueuePortHandle workPort,
    _In_ XTaskQueuePortHandle completionPort)
{
    RETURN_IF_NULL_ALLOC(m_waitRegistry.get());
    RETURN_HR_IF(E_GAMERUNTIME_INVALID_HANDLE, workPort== nullptr || workPort->m_signature != TASK_QUEUE_PORT_SIGNATURE || workPort->m_runtimeIteration != GetCurrentRuntimeIteration());
    RETURN_HR_IF(E_GAMERUNTIME_INVALID_HANDLE, completionPort == nullptr || completionPort->m_signature != TASK_QUEUE_PORT_SIGNATURE || completionPort->m_runtimeIteration != GetCurrentRuntimeIteration());

    m_work.Port = referenced_ptr<ITaskQueuePort>(workPort->m_port);
    m_completion.Port = referenced_ptr<ITaskQueuePort>(completionPort->m_port);
    m_work.Source = referenced_ptr<ITaskQueue>(workPort->m_queue);
    m_completion.Source = referenced_ptr<ITaskQueue>(completionPort->m_queue);
    
    RETURN_IF_FAILED(m_work.Port->Attach(&m_work));
    RETURN_IF_FAILED(m_completion.Port->Attach(&m_completion));

#ifdef SUSPEND_API
    RETURN_IF_FAILED(m_suspendHandler.Initialize(ProcessGlobals::g_suspendState, this));
#endif

    return S_OK;
}

HRESULT __stdcall TaskQueueImpl::GetPortContext(
    _In_ XTaskQueuePort port,
    _Out_ ITaskQueuePortContext** portContext)
{
    RETURN_HR_IF(E_POINTER, portContext == nullptr);
    
    switch(port)
    {
    case XTaskQueuePort::Work:
        *portContext = &m_work;
        m_work.AddRef();
        break;
        
    case XTaskQueuePort::Completion:
        *portContext = &m_completion;
        m_completion.AddRef();
        break;
        
    default:
        RETURN_HR(E_INVALIDARG);
    }
    
    return S_OK;
}

HRESULT __stdcall TaskQueueImpl::RegisterWaitHandle(
    _In_ XTaskQueuePort port,
    _In_ HANDLE waitHandle,
    _In_opt_ void* callbackContext,
    _In_ XTaskQueueCallback* callback,
    _Out_ XTaskQueueRegistrationToken* token)
{
    RETURN_HR_IF(E_POINTER, callback == nullptr || token == nullptr);

    XTaskQueueRegistrationToken portToken;
    referenced_ptr<ITaskQueuePortContext> portContext;

    RETURN_IF_FAILED(GetPortContext(port, portContext.address_of()));
    RETURN_HR_IF(E_OUTOFMEMORY, !m_waitRegistry->Reserve());
    HRESULT hr;
    void* capability = nullptr;
    if (SUCCEEDED(portContext->GetPort()->QueryApi(TaskQueueInternal::kWaitPublicationApiId, &capability)))
    {
        referenced_ptr<TaskQueuePortImpl> localPort;
        *localPort.address_of() = static_cast<TaskQueuePortImpl*>(capability);
        hr = localPort->RegisterWaitWithToken(portContext.get(), waitHandle, callbackContext,
            callback, token, m_waitRegistry.get());
    }
    else
    {
        // Older port implementations use the original immutable interface.
        hr = portContext->GetPort()->RegisterWaitHandle(
            portContext.get(), waitHandle, callbackContext, callback, &portToken);
        if (SUCCEEDED(hr)) m_waitRegistry->Publish(port, portToken, token, false);
    }
    if (FAILED(hr))
    {
        m_waitRegistry->CancelReservation();
        return hr;
    }
    return S_OK;
}

void __stdcall TaskQueueImpl::UnregisterWaitHandle(
    _In_ XTaskQueueRegistrationToken token)
{
    const auto registration = m_waitRegistry->Lookup(token);
    if (registration.PortToken != 0)
    {
        referenced_ptr<ITaskQueuePortContext> portContext;
        if (SUCCEEDED(GetPortContext(registration.Port, portContext.address_of())))
        {
            portContext->GetPort()->UnregisterWaitHandle({registration.PortToken});
            if (!registration.retiredByPort) m_waitRegistry->Remove(token);
        }
    }
}

HRESULT __stdcall TaskQueueImpl::RegisterSubmitCallback(
    _In_opt_ void* context,
    _In_ XTaskQueueMonitorCallback* callback,
    _Out_ XTaskQueueRegistrationToken* token)
{
    return m_callbackSubmitted.Register(context, callback, token);
}

void __stdcall TaskQueueImpl::UnregisterSubmitCallback(
    _In_ XTaskQueueRegistrationToken token)
{
    m_callbackSubmitted.Unregister(token);
}

HRESULT __stdcall TaskQueueImpl::Terminate(
    _In_ bool wait, 
    _In_opt_ void* callbackContext, 
    _In_opt_ XTaskQueueTerminatedCallback* callback)
{
    std::unique_ptr<TerminationEntry> entry(new (std::nothrow) TerminationEntry);
    RETURN_IF_NULL_ALLOC(entry);

    entry->owner = this;
    entry->level = TerminationLevel::Work;
    entry->context = callbackContext;
    entry->callback = callback;

    void* workToken;
    
    RETURN_IF_FAILED(m_work.Port->PrepareTerminate(&m_work, entry.get(), OnTerminationCallback, &workToken));

    HRESULT hr = m_completion.Port->PrepareTerminate(&m_completion, entry.get(), OnTerminationCallback, &entry->completionPortToken);
    if (FAILED(hr))
    {
        m_work.Port->CancelTermination(workToken);
        RETURN_HR(hr);
    }

    // At this point both ports have been marked for termination and have pre-alocated any state they
    // need, so we can proceed with the actual termination.
    
    entry.release();

    // Addref ourself to ensure we don't lose the queue to a close
    // while termination is in flight.  This is released on OnTerminationCallback.
    // If we are waiting we will take another reference and release it
    // after the wait completes.

    AddRef();

    if (wait)
    {
        AddRef();
    }

    m_work.Port->Terminate(workToken);
    
    if (wait)
    {
        std::unique_lock<std::mutex> lock(m_termination.lock);
        while(!m_termination.terminated)
        {
            m_termination.cv.wait(lock);
        }

        // Termination notify happens through a callback and when the
        // terminated flag gets set that callback and other frames
        // are still on the stack. Wait for them to unwind here so
        // if the caller wants to exit the process there isn't code
        // on the stack in another thread.

        m_work.Port->WaitForUnwind();
        m_completion.Port->WaitForUnwind();

        Release();
    }

    return S_OK;
}

void TaskQueueImpl::RundownObject()
{
    m_work.SetStatus(TaskQueuePortStatus::Terminated);
    m_completion.SetStatus(TaskQueuePortStatus::Terminated);

#ifdef SUSPEND_API
    m_suspendHandler.Shutdown();
#endif

    // We can be asked to rundown a partially created
    // task queue, so check for nulls here.

    if (m_work.Port != nullptr)
    {
        m_work.Port->Detach(&m_work);
    }

    if (m_completion.Port != nullptr)
    {
        m_completion.Port->Detach(&m_completion);
    }
}

#ifdef SUSPEND_API
void TaskQueueImpl::OnSuspendResume(_In_ bool isSuspended)
{
    if (isSuspended)
    {
        m_completion.GetPort()->SuspendPort();
        m_work.GetPort()->SuspendPort();
    }
    else
    {
        m_work.GetPort()->ResumePort();
        m_completion.GetPort()->ResumePort();
    }
}
#endif

void TaskQueueImpl::OnTerminationCallback(_In_ void* context)
{
    TerminationEntry* entry = static_cast<TerminationEntry*>(context);
    switch(entry->level)
    {
        case TerminationLevel::Work:
            entry->level = TerminationLevel::Completion;
            entry->owner->m_completion.Port->Terminate(entry->completionPortToken);
            break;

        case TerminationLevel::Completion:
            if (entry->callback != nullptr)
            {
                entry->callback(entry->context);
            }

            {
                std::unique_lock<std::mutex> lock(entry->owner->m_termination.lock);
                entry->owner->m_termination.terminated = true;
                entry->owner->m_termination.cv.notify_all();
            }

            entry->owner->Release();
            delete entry;
            break;

        default:
            ASSERT(false);
    }
}

static HRESULT CreateTaskQueueHandle(
    _In_ ITaskQueue* impl,
    _Out_ XTaskQueueHandle* queue)
{
    *queue = nullptr;

    std::unique_ptr<XTaskQueueObject> q(new (std::nothrow) XTaskQueueObject);
    RETURN_IF_NULL_ALLOC(q);

    q->m_signature = TASK_QUEUE_SIGNATURE;
    q->m_runtimeIteration = GetCurrentRuntimeIteration();
    q->m_queue = impl;
    impl->AddRef();

    *queue = q.release();

    SystemHandleMarkCreated(*queue);
    return S_OK;
}

///////////////////
// XTaskQueue.h APIs
///////////////////

//
// Creates a Task Queue, which can be used to queue
// different calls together.
//
STDAPI XTaskQueueCreate(
    _In_ XTaskQueueDispatchMode workDispatchMode,
    _In_ XTaskQueueDispatchMode completionDispatchMode,
    _Out_ XTaskQueueHandle* queue
    ) noexcept
{
    *queue = nullptr;

    referenced_ptr<TaskQueueImpl> aq(new (std::nothrow) TaskQueueImpl);
    RETURN_IF_NULL_ALLOC(aq);

    RETURN_IF_FAILED(aq->Initialize(
        workDispatchMode, 
        completionDispatchMode));

    RETURN_IF_FAILED(CreateTaskQueueHandle(aq.get(), queue));

    return S_OK;
}

/// <summary>
/// Returns the task queue port handle for the given
/// port. Task queue port handles are owned by the
/// task queue and do not have to be closed. They are
/// used when creating composite task queues.
/// </summary>
/// <param name='queue'>The task queue to get the port from.</param>
/// <param name='port'>The port to get.</param>
STDAPI XTaskQueueGetPort(
    _In_ XTaskQueueHandle queue,
    _In_ XTaskQueuePort port,
    _Out_ XTaskQueuePortHandle* portHandle
    ) noexcept
{
    referenced_ptr<ITaskQueue> aq(GetQueue(queue));
    RETURN_HR_IF(E_GAMERUNTIME_INVALID_HANDLE, aq == nullptr);

    referenced_ptr<ITaskQueuePortContext> portContext;
    RETURN_IF_FAILED(aq->GetPortContext(port, portContext.address_of()));
    
    *portHandle = portContext->GetPort()->GetHandle();
    
    return S_OK;
}

/// <summary>
/// Creates a task queue composed of ports of other
/// task queue ports. A composite task queue will duplicate
/// the handles of queues that own the provided ports.
/// </summary>
/// <param name='workPort'>The port to use for queuing work callbacks.</param>
/// <param name='completionPort'>The port to use for queuing completion callbacks.</param>
STDAPI XTaskQueueCreateComposite(
     _In_ XTaskQueuePortHandle workPort,
     _In_ XTaskQueuePortHandle completionPort,
     _Out_ XTaskQueueHandle* queue
     ) noexcept
{
    *queue = nullptr;

    referenced_ptr<TaskQueueImpl> aq(new (std::nothrow) TaskQueueImpl);
    RETURN_IF_NULL_ALLOC(aq);
    RETURN_IF_FAILED(aq->Initialize(workPort, completionPort));

    RETURN_IF_FAILED(CreateTaskQueueHandle(aq.get(), queue));

    return S_OK;
}

//
// Processes items in the task queue of the given type. If an item
// is processed this will return TRUE. If there are no items to process
// this returns FALSE.  You can pass a timeout, which will cause
// XTaskQueueDispatch to wait for something to arrive in the queue.
//
STDAPI_(bool) XTaskQueueDispatch(
    _In_ XTaskQueueHandle queue,
    _In_ XTaskQueuePort port,
    _In_ uint32_t timeoutInMs
    ) noexcept
{
    referenced_ptr<ITaskQueue> aq(GetQueue(queue));
    if (aq == nullptr)
    {
        return false;
    }

    referenced_ptr<ITaskQueuePortContext> portContext;
    if (FAILED(aq->GetPortContext(port, portContext.address_of())))
    {
        return false;
    }

    return portContext->GetPort()->Dispatch(portContext.get(), timeoutInMs);
}

//
// Returns TRUE if there is no outstanding work in this
// queue.  Note this API is only used for testing and
// is not exposed outside the library.
//
STDAPI_(bool) XTaskQueueIsEmpty(
    _In_ XTaskQueueHandle queue,
    _In_ XTaskQueuePort port
    ) noexcept
{
    referenced_ptr<ITaskQueue> aq(GetQueue(queue));
    if (aq == nullptr)
    {
        return false;
    }

    referenced_ptr<ITaskQueuePortContext> portContext;
    if (FAILED(aq->GetPortContext(port, portContext.address_of())))
    {
        return false;
    }

    return portContext->GetPort()->IsEmpty();
}

//
// Closes the task queue.  A queue can only be closed if it
// is not in use by a task or is empty.  If not true, the queue
// will be marked for closure and closed when it can. 
//
STDAPI_(void) XTaskQueueCloseHandle(
    _In_ XTaskQueueHandle queue
    ) noexcept
{
    ITaskQueue* aq = GetQueue(queue);

    if (aq != nullptr)
    {
        if (USE_UNIQUE_HANDLES() && queue != aq->GetHandle())
        {
            SystemHandleMarkDestroyed(queue);

            queue->m_signature = 0;
            queue->m_queue = nullptr;
            delete queue;
        }

        aq->Release();
    }
}

//
// Terminates a task queue by canceling all pending items and
// preventing new items from being queued.  Once a queue is terminated
// its handle can be closed.
//
STDAPI XTaskQueueTerminate(
    _In_ XTaskQueueHandle queue, 
    _In_ bool wait, 
    _In_opt_ void* callbackContext, 
    _In_opt_ XTaskQueueTerminatedCallback* callback
    ) noexcept
{
    referenced_ptr<ITaskQueue> aq(GetQueue(queue));
    return aq->Terminate(wait, callbackContext, callback);
}

//
// Submits either a work or completion callback immediately.
//
STDAPI XTaskQueueSubmitCallback(
    _In_ XTaskQueueHandle queue,
    _In_ XTaskQueuePort port,
    _In_opt_ void* callbackContext,
    _In_ XTaskQueueCallback* callback
    ) noexcept
{
    return XTaskQueueSubmitDelayedCallback(queue, port, 0, callbackContext, callback);
}

//
// Submits either a work or completion callback.
//
STDAPI XTaskQueueSubmitDelayedCallback(
    _In_ XTaskQueueHandle queue,
    _In_ XTaskQueuePort port,
    _In_ uint32_t delayMs,
    _In_opt_ void* callbackContext,
    _In_ XTaskQueueCallback* callback
    ) noexcept
{
    referenced_ptr<ITaskQueue> aq(GetQueue(queue));
    RETURN_HR_IF(E_GAMERUNTIME_INVALID_HANDLE, aq == nullptr);

    referenced_ptr<ITaskQueuePortContext> portContext;
    RETURN_IF_FAILED(aq->GetPortContext(port, portContext.address_of()));

    RETURN_IF_FAILED(portContext->GetPort()->QueueItem(portContext.get(), delayMs, callbackContext, callback));
    return S_OK;
}

//
// Registers a wait handle with the task queue.  When the wait handle
// is satisfied the task queue will invoke the given callback. This
// provides an efficient way to add items to a task queue in 
// response to handles becoming signaled.
//
STDAPI XTaskQueueRegisterWaiter(
    _In_ XTaskQueueHandle queue,
    _In_ XTaskQueuePort port,
    _In_ HANDLE waitHandle,
    _In_opt_ void* callbackContext,
    _In_ XTaskQueueCallback* callback,
    _Out_ XTaskQueueRegistrationToken* token
    ) noexcept
{
    referenced_ptr<ITaskQueue> aq(GetQueue(queue));
    RETURN_HR_IF(E_GAMERUNTIME_INVALID_HANDLE, aq == nullptr);
    RETURN_IF_FAILED(aq->RegisterWaitHandle(port, waitHandle, callbackContext, callback, token));
    return S_OK;
}

//
// Unregisters a previously registered task queue waiter.
//
STDAPI_(void) XTaskQueueUnregisterWaiter(
    _In_ XTaskQueueHandle queue,
    _In_ XTaskQueueRegistrationToken token
    ) noexcept
{
    referenced_ptr<ITaskQueue> aq(GetQueue(queue));
    if (aq != nullptr)
    {
        aq->UnregisterWaitHandle(token);
    }
}

//
// Increments the refcount on the queue and allows supplying
// options as to how the duplicate is performed.
//
STDAPI XTaskQueueDuplicateHandleWithOptions(
    _In_ XTaskQueueHandle queueHandle,
    _In_ XTaskQueueDuplicateOptions options,
    _Out_ XTaskQueueHandle* duplicatedHandle
    ) noexcept
{
    RETURN_HR_IF(E_POINTER, duplicatedHandle == nullptr);

    *duplicatedHandle = nullptr;

    auto queue = GetQueue(queueHandle);
    RETURN_HR_IF(E_GAMERUNTIME_INVALID_HANDLE, queue == nullptr);

    if (USE_UNIQUE_HANDLES() && (options != XTaskQueueDuplicateOptions::Reference))
    {
        RETURN_IF_FAILED(CreateTaskQueueHandle(queue, duplicatedHandle));
    }
    else
    {
        queue->AddRef();
        *duplicatedHandle = queue->GetHandle();
    }

    return S_OK;
}

//
// Increments the refcount on the queue
//
STDAPI XTaskQueueDuplicateHandle(
    _In_ XTaskQueueHandle queueHandle,
    _Out_ XTaskQueueHandle* duplicatedHandle
    ) noexcept
{
    return XTaskQueueDuplicateHandleWithOptions(
        queueHandle,
        XTaskQueueDuplicateOptions::None,
        duplicatedHandle);
}

//
// Registers a callback that will be called when a new callback
// is submitted. The callback will be directly invoked when
// the call is submitted.
//
STDAPI XTaskQueueRegisterMonitor(
    _In_ XTaskQueueHandle queue,
    _In_opt_ void* callbackContext,
    _In_ XTaskQueueMonitorCallback* callback,
    _Out_ XTaskQueueRegistrationToken* token
    ) noexcept
{
    referenced_ptr<ITaskQueue> aq(GetQueue(queue));
    RETURN_HR_IF(E_GAMERUNTIME_INVALID_HANDLE, aq == nullptr);
    RETURN_IF_FAILED(aq->RegisterSubmitCallback(callbackContext, callback, token));
    return S_OK;
}

//
// Unregisters a previously added callback.
//
STDAPI_(void) XTaskQueueUnregisterMonitor(
    _In_ XTaskQueueHandle queue,
    _In_ XTaskQueueRegistrationToken token
    ) noexcept
{
    referenced_ptr<ITaskQueue> aq(GetQueue(queue));
    if (aq != nullptr)
    {
        aq->UnregisterSubmitCallback(token);
    }
}

//
// Returns a handle to the process task queue, or nullptr if there is no
// process task queue.  By default, there is a default process task queue
// that uses the thread pool for both work and completion ports. This is an
// internal variant that takes duplicate options.
//
STDAPI_(bool) XTaskQueueGetCurrentProcessTaskQueueWithOptions(
    _In_ XTaskQueueDuplicateOptions options,
    _Out_ XTaskQueueHandle* queue
    ) noexcept
{
    *queue = nullptr;

    XTaskQueueHandle processQueue = ProcessGlobals::g_processQueue;
    if (processQueue == ProcessGlobals::g_invalidQueueHandle)
    {
        XTaskQueueHandle defaultProcessQueue = ProcessGlobals::g_defaultProcessQueue;
        if (defaultProcessQueue == ProcessGlobals::g_invalidQueueHandle)
        {
            // The default process queue hasn't been created yet. Create it locally
            // then swap it into the atomic.

            referenced_ptr<TaskQueueImpl> aq(new (std::nothrow) TaskQueueImpl);
            if (aq != nullptr && SUCCEEDED(aq->Initialize(
                XTaskQueueDispatchMode::ThreadPool,
                XTaskQueueDispatchMode::ThreadPool)))
            {
                XTaskQueueHandle expected = ProcessGlobals::g_invalidQueueHandle;
                if (ProcessGlobals::g_defaultProcessQueue.compare_exchange_strong(
                    expected,
                    aq->GetHandle()))
                {
                    // We successfully set the default handle.
                    aq.release();
                }
            }

            defaultProcessQueue = ProcessGlobals::g_defaultProcessQueue;
        }

        processQueue = defaultProcessQueue;
    }

    if (processQueue == ProcessGlobals::g_invalidQueueHandle)
    {
        processQueue = nullptr;
    }

    if (processQueue != nullptr)
    {
        (void)XTaskQueueDuplicateHandleWithOptions(processQueue, options, queue);
    }

    return (*queue) != nullptr;
}

//
// Returns a handle to the process task queue, or nullptr if there is no
// process task queue.  By default, there is a default process task queue
// that uses the thread pool for both work and completion ports.
//
STDAPI_(bool) XTaskQueueGetCurrentProcessTaskQueue(
    _Out_ XTaskQueueHandle* queue
    ) noexcept
{
    return XTaskQueueGetCurrentProcessTaskQueueWithOptions(XTaskQueueDuplicateOptions::None, queue);
}

//
// Sets the given task queue as the process wide task queue.  The
// queue can be set to nullptr, in which case XTaskQueueGetCurrentProcessTaskQueue will
// also return nullptr. The provided queue will have its handle duplicated
// and any existing process task queue will have its handle closed.
//
STDAPI_(void) XTaskQueueSetCurrentProcessTaskQueue(
    _In_opt_ XTaskQueueHandle queue
    ) noexcept
{
    XTaskQueueHandle newQueue = nullptr;

    if (queue != nullptr)
    {
        FAIL_FAST_IF_FAILED(XTaskQueueDuplicateHandle(queue, &newQueue));
    }

    XTaskQueueHandle previous = ProcessGlobals::g_processQueue.exchange(newQueue);
    if (previous != nullptr && previous != ProcessGlobals::g_invalidQueueHandle)
    {
        XTaskQueueCloseHandle(previous);
    }
}

//
// Submits a callback that will be invoked asynchronously via some external
// system.  This is used to prevent queue termination while other work
// is happening.
//
STDAPI XTaskQueueSuspendTermination(
    _In_ XTaskQueueHandle queue
    ) noexcept
{
    referenced_ptr<ITaskQueue> aq(GetQueue(queue));
    RETURN_HR_IF(E_GAMERUNTIME_INVALID_HANDLE, aq == nullptr);

    referenced_ptr<ITaskQueuePortContext> portContext;
    RETURN_IF_FAILED(aq->GetPortContext(XTaskQueuePort::Work, portContext.address_of()));

    RETURN_IF_FAILED(portContext->GetPort()->SuspendTermination(portContext.get()));
    return S_OK;
}

//
// Tells the queue that a previously submitted external callback
// has been dispatched.
//
STDAPI_(void) XTaskQueueResumeTermination(
    _In_ XTaskQueueHandle queue
    ) noexcept
{
    referenced_ptr<ITaskQueue> aq(GetQueue(queue));
    if (aq == nullptr)
    {
        return;
    }
    
    referenced_ptr<ITaskQueuePortContext> portContext;
    if (FAILED(aq->GetPortContext(XTaskQueuePort::Work, portContext.address_of())))
    {
        return;
    }
        
    portContext->GetPort()->ResumeTermination(portContext.get());
}

//
// Suspends the activity of all task queues in the process. When
// a task queue is suspended:
//
// 1. It will not signal when new items are added.
// 2. It will not return items from the dispatcher (it acts like it
//    is empty).
//
#ifdef SUSPEND_API
STDAPI_(void) XTaskQueueGlobalSuspend(
    ) noexcept
{
    ProcessGlobals::g_suspendState.Suspend();
    ProcessGlobals::g_suspendState.WaitForQueuesToSuspend();
}
#endif

//
// Resumes the activity of all task queues in the process. When
// a task queue is resumed:
//
// 1. Queues that are not empty will signal they have items.
// 2. The dispatcher will start returing items again.
//
#ifdef SUSPEND_API
STDAPI_(void) XTaskQueueGlobalResume(
    ) noexcept
{
    ProcessGlobals::g_suspendState.Resume();
}
#endif

//
// Uninitializes global task queue state. This closes handles to per-process
// task queues and resets state back to defaults. If there is a per-process
// task queue, it will be closed but not terminated.  Queue termination is
// up to the caller. This API returns true if all task queues have been
// cleaned up and deleted, or false if there are still outstanding references.
// A timeout may be provided to wait for outstanding queues to be cleaned up.
// The default is not to wait. The maximum timeout capped to 5000ms.
//
STDAPI_(bool) XTaskQueueUninitialize(
    _In_ uint32_t timeoutMilliseconds
    ) noexcept
{
    std::atomic<XTaskQueueHandle>* globals[] =
    {
        &ProcessGlobals::g_defaultProcessQueue,
        &ProcessGlobals::g_processQueue
    };

    for (auto queue : globals)
    {
        auto handle = queue->exchange(ProcessGlobals::g_invalidQueueHandle);
        if (handle != nullptr && handle != ProcessGlobals::g_invalidQueueHandle)
        {
            XTaskQueueCloseHandle(handle);
        }
    }

    constexpr uint32_t maxTimeout = 5000;
    if (timeoutMilliseconds > maxTimeout)
    {
        timeoutMilliseconds = maxTimeout;
    }

    return ApiRefs::WaitZeroRefs(timeoutMilliseconds);
}

/// <summary>
/// Sets or clears test hooks on a task queue.
/// </summary>
STDAPI XTaskQueueSetTestHooks(
    _In_ XTaskQueueHandle queue,
    _In_ XTaskQueueTestHooks* hooks
    ) noexcept
{
    referenced_ptr<ITaskQueue> aq(GetQueue(queue));
    RETURN_HR_IF(E_GAMERUNTIME_INVALID_HANDLE, aq == nullptr);
    aq->SetTestHooks(hooks);
    return S_OK;
}

/// <summary>
/// Submits any pending delayed callbacks that are due to run. This is
/// intended for use in unit tests.
/// </summary>
STDAPI XTaskQueueSubmitPendingCallbacks(
    _In_ XTaskQueueHandle queue,
    _In_ XTaskQueuePort port
    ) noexcept
{
    referenced_ptr<ITaskQueue> aq(GetQueue(queue));
    RETURN_HR_IF(E_GAMERUNTIME_INVALID_HANDLE, aq == nullptr);

    referenced_ptr<ITaskQueuePortContext> portContext;
    RETURN_IF_FAILED(aq->GetPortContext(port, portContext.address_of()));

    portContext->GetPort()->SubmitPendingCallbacks();
    return S_OK;
}
