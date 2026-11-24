// Stand-in for libHttpClient's pch.h, so third_party/xtaskqueue (Source/Task: XTaskQueue and XAsync) builds alone,
// without the HTTP stack. Only the macros and error codes those files use are defined here.
#pragma once
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <objbase.h>

#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <new>
#include <thread>
#include <vector>

#ifndef E_NO_TASK_QUEUE
#define E_NO_TASK_QUEUE _HRESULT_TYPEDEF_(0x800701ABL)
#endif
#ifndef E_ILLEGAL_METHOD_CALL
#define E_ILLEGAL_METHOD_CALL _HRESULT_TYPEDEF_(0x8000000EL)
#endif
#ifndef E_NOT_SUPPORTED
#define E_NOT_SUPPORTED _HRESULT_TYPEDEF_(0x80070032L)
#endif
#define E_GAMERUNTIME_INVALID_HANDLE E_HANDLE

#include <XAsync.h>
#include <XAsyncProvider.h>
#include <XTaskQueue.h>

#define ASSERT(condition) assert(condition)
using DefaultUnnamedMutex = std::mutex;
using DefaultUnnamedConditionVariable = std::condition_variable;

// No tracing, no GDK system-handle tracking, no runtime iterations.
#define HC_TRACE_ERROR(area, msg, ...) ((void)0)
#define ASYNC_LIB_TRACE(result, message) ((void)0)
#define SYSTEM_HANDLE_DEFINE_HELPERS(a, b)
#define SystemHandleAssert(h)
#define SystemHandleMarkCreated(h)
#define SystemHandleMarkDestroyed(h)
#define USE_UNIQUE_HANDLES() (true)
#define GetCurrentRuntimeIteration() 0
#define CATCH_RETURN() catch (std::bad_alloc const&) { return E_OUTOFMEMORY; } catch (...) { return E_FAIL; }
#define CATCH_RETURN_WITH(errCode) catch (...) { return errCode; }

#include "ResultMacros.h"
