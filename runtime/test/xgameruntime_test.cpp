// Checks the registered xgameruntime shim the way a GDK title uses it at boot.
//
// Every function is resolved through rt::shims::find("xgameruntime.dll", name), the same table the loader binds
// imports to. The sequence follows a typical title: initialize the runtime, query the device, create a queue whose
// work runs on the thread pool and whose completions the game loop dispatches by hand, run async work on it and on
// the process queue, drive a custom XAsyncBegin provider to a result, then uninitialize. No game data; nothing
// online.
#include <windows.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>

#include <XAsync.h>
#include <XAsyncProvider.h>
#include <XTaskQueue.h>

#include "file_system.h"
#include "shims.h"

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

template <class Fn>
Fn get(const char* name) {
    return reinterpret_cast<Fn>(rt::shims::find("xgameruntime.dll", name));
}

using InitFn = HRESULT(__stdcall*)();
using UninitFn = void(__stdcall*)();
using DeviceTypeFn = uint32_t(__stdcall*)();
using ConsoleIdFn = HRESULT(__stdcall*)(size_t, char*, size_t*);
using QueueCreateFn = HRESULT(__stdcall*)(XTaskQueueDispatchMode, XTaskQueueDispatchMode, XTaskQueueHandle*);
using QueueCloseFn = void(__stdcall*)(XTaskQueueHandle);
using QueueDispatchFn = bool(__stdcall*)(XTaskQueueHandle, XTaskQueuePort, uint32_t);
using QueueProcessFn = bool(__stdcall*)(XTaskQueueHandle*);
using AsyncRunFn = HRESULT(__stdcall*)(XAsyncBlock*, XAsyncWork*);
using AsyncStatusFn = HRESULT(__stdcall*)(XAsyncBlock*, bool);
using AsyncBeginFn = HRESULT(__stdcall*)(XAsyncBlock*, void*, const void*, const char*, XAsyncProvider*);
using AsyncScheduleFn = HRESULT(__stdcall*)(XAsyncBlock*, uint32_t);
using AsyncCompleteFn = void(__stdcall*)(XAsyncBlock*, HRESULT, size_t);
using AsyncResultSizeFn = HRESULT(__stdcall*)(XAsyncBlock*, size_t*);
using AsyncResultFn = HRESULT(__stdcall*)(XAsyncBlock*, const void*, size_t, void*, size_t*);

struct alignas(8) XVersion {
    uint16_t major;
    uint16_t minor;
    uint16_t build;
    uint16_t revision;
};
static_assert(sizeof(XVersion) == 8, "XVersion must be 8 bytes per GDK spec");

struct XSystemAnalyticsInfo {
    XVersion osVersion;
    XVersion hostingOsVersion;
    char family[64];
    char form[64];
};
static_assert(sizeof(XSystemAnalyticsInfo) == 144, "XSystemAnalyticsInfo must be 144 bytes per GDK spec");

struct XPackageMountHandle {
    uint64_t handle;
};

using AnalyticsInfoFn = XSystemAnalyticsInfo(__stdcall*)();
using PackageIdentifierFn = HRESULT(__stdcall*)(size_t, char*);
using PackageMountFn = HRESULT(__stdcall*)(const char*, XPackageMountHandle*);
using PackageCloseMountHandleFn = void(__stdcall*)(XPackageMountHandle);
using PackageMountPathSizeFn = HRESULT(__stdcall*)(XPackageMountHandle, size_t*);
using PackageMountPathFn = HRESULT(__stdcall*)(XPackageMountHandle, size_t, char*);
using PackagedProcessFn = bool(__stdcall*)();

const char* const kNames[] = {
    "XGameRuntimeInitialize", "XGameRuntimeUninitialize", "XSystemGetDeviceType", "XSystemGetConsoleId",
    "XSystemGetAnalyticsInfo", "XPackageGetCurrentProcessPackageIdentifier", "XPackageMount",
    "XPackageCloseMountHandle", "XPackageGetMountPathSize", "XPackageGetMountPath",
    "XPackageIsPackagedProcess",
    "XTaskQueueCreate", "XTaskQueueCreateComposite", "XTaskQueueGetPort", "XTaskQueueDuplicateHandle",
    "XTaskQueueCloseHandle", "XTaskQueueTerminate", "XTaskQueueDispatch", "XTaskQueueSubmitCallback",
    "XTaskQueueSubmitDelayedCallback", "XTaskQueueRegisterWaiter", "XTaskQueueUnregisterWaiter",
    "XTaskQueueRegisterMonitor", "XTaskQueueUnregisterMonitor", "XTaskQueueGetCurrentProcessTaskQueue",
    "XTaskQueueSetCurrentProcessTaskQueue", "XAsyncGetStatus", "XAsyncGetResultSize", "XAsyncCancel", "XAsyncRun",
    "XAsyncBegin", "XAsyncSchedule", "XAsyncComplete", "XAsyncGetResult",
};

std::atomic<int> g_work_runs{0};
std::atomic<DWORD> g_work_thread{0};
std::atomic<int> g_completions{0};
std::atomic<DWORD> g_completion_thread{0};

HRESULT CALLBACK work(XAsyncBlock* b) {
    g_work_thread = GetCurrentThreadId();
    ++g_work_runs;
    *static_cast<int*>(b->context) = 42;
    return S_OK;
}

void CALLBACK completed(XAsyncBlock*) {
    g_completion_thread = GetCurrentThreadId();
    ++g_completions;
}

// A provider that produces a 4-byte result on the queue's work port: the shape GDK async APIs use internally.
struct ProviderState {
    AsyncCompleteFn complete;
    std::atomic<int> ops{0};
};

HRESULT CALLBACK provider(XAsyncOp op, const XAsyncProviderData* data) {
    auto* st = static_cast<ProviderState*>(data->context);
    st->ops |= 1 << static_cast<int>(op);
    switch (op) {
        case XAsyncOp::DoWork:
            st->complete(data->async, S_OK, sizeof(uint32_t));
            return S_OK;
        case XAsyncOp::GetResult:
            *static_cast<uint32_t*>(data->buffer) = 0x1EA0;
            return S_OK;
        default:
            return S_OK;
    }
}

HMODULE g_teardown_rt_mod = nullptr;
FARPROC(WINAPI* g_teardown_get_proc)(HMODULE, LPCSTR) = nullptr;
BOOL(WINAPI* g_teardown_free_lib)(HMODULE) = nullptr;

// Registered with std::atexit before check_run_time_load() constructs any function-local statics in library_loader.
// LIFO atexit order means this runs after those statics would have been destroyed, simulating mapped-DLL
// DLL_PROCESS_DETACH or earlier atexit handlers calling GetProcAddress/FreeLibrary on a pseudo-handle during exit.
void check_late_teardown() {
    if (!g_teardown_rt_mod || !g_teardown_get_proc || !g_teardown_free_lib) return;
    if (g_teardown_get_proc(g_teardown_rt_mod, "UninitializeApiImpl") == nullptr) {
        std::fprintf(stderr, "FAIL late teardown GetProcAddress on pseudo-handle\n");
        std::_Exit(1);
    }
    SetLastError(0);
    if (g_teardown_get_proc(g_teardown_rt_mod, "LateTeardownMissingSymbol") != nullptr ||
        GetLastError() != ERROR_PROC_NOT_FOUND) {
        std::fprintf(stderr, "FAIL late teardown unserved GetProcAddress on pseudo-handle\n");
        std::_Exit(1);
    }
    if (g_teardown_free_lib(g_teardown_rt_mod) != TRUE) {
        std::fprintf(stderr, "FAIL late teardown FreeLibrary on pseudo-handle\n");
        std::_Exit(1);
    }
}

// ---- The classes measured with tools/gdk_runtime_calls.py ----
//
// Each is asked for exactly as its thunks in the package's libHttpClient.GDK.dll ask: QueryApiImpl with the class id
// and the measured interface id, then one slot with `this` first and the documented arguments, then Release.

using QueryFn = HRESULT(WINAPI*)(const GUID*, REFIID, void**);

template <class F> F method(void* obj, int index) { return reinterpret_cast<F>((*static_cast<void* const* const*>(obj))[index]); }
void release_object(void* obj) { method<ULONG(STDMETHODCALLTYPE*)(void*)>(obj, 2)(obj); }
void* ask(QueryFn query, const GUID& cls, const GUID& iid) {
    void* p = nullptr;
    return query(&cls, iid, &p) == S_OK ? p : nullptr;
}

const GUID kSystem5Id = {0x1861cf2e, 0xe18b, 0x4834, {0xa9, 0xf5, 0xb4, 0xa4, 0xe6, 0xef, 0xb4, 0xcf}};
const GUID kIUnknownId = {0x00000000, 0x0000, 0x0000, {0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x46}};

// XGameRuntimeFeatureImpl and XSystemAnalyticsImpl.
void check_measured_classes(QueryFn query, XSystemAnalyticsInfo (*named_analytics)()) {
    const GUID feature_cls = {0x8836fe87, 0xedb9, 0x4fe3, {0x8d, 0xad, 0x05, 0xf0, 0xd2, 0xcd, 0x5b, 0x40}};
    void* f = ask(query, feature_cls, feature_cls);
    check(f != nullptr, "QueryApiImpl hands out XGameRuntimeFeatureImpl for its class id");
    if (f) {
        using Available = bool(STDMETHODCALLTYPE*)(void*, uint32_t);  // bool, per Microsoft's reference page
        auto available = method<Available>(f, 3);
        // XGameRuntimeFeature values: XAccessibility 0, XLauncher 9, XSpeechSynthesizer 13, XAsync 2, XAsyncProvider 3, XDisplay 4, XGame 5, XGameSave 7, XGameUI 8,
        // XNetworking 10, XPackage 11, XPersistentLocalStorage 12, XStore 14, XSystem 15, XTaskQueue 16, XThread 17,
        // XUser 18, XError 19, XGameEvent 20, XGameStreaming 21.
        bool served = true;
        for (uint32_t v : {0u, 2u, 3u, 5u, 7u, 9u, 13u, 8u, 10u, 11u, 12u, 14u, 15u, 16u, 17u, 18u, 19u, 20u, 21u})
            served = served && available(f, v);
        check(served, "slot 3 (XGameRuntimeIsFeatureAvailable): true for the features whose classes are served, "
                      "XPersistentLocalStorage among them");
        check(!available(f, 4) && !available(f, 22) && !available(f, 0xFFFFFFFFu),
              "slot 3: false for XDisplay (served, but documented for the consoles only) and values past the enum");
        // Every feature but XDisplay follows whether QueryApiImpl serves its class, whichever file serves it.
        const struct {
            uint32_t feature;
            GUID cls;
        } follow[] = {
            {0, {0x3e241977, 0x6237, 0x41e9, {0x85, 0x59, 0x10, 0x2c, 0x2d, 0x99, 0x83, 0xf1}}},   // XAccessibility
            {1, {0xa4f1aee2, 0x4bf1, 0x4485, {0xb0, 0x08, 0xa7, 0xc2, 0x6d, 0x52, 0xac, 0x27}}},   // XAppCapture
            {2, {0x073b7dcb, 0x1fcf, 0x4030, {0x94, 0xbe, 0xe3, 0xc9, 0xeb, 0x62, 0x34, 0x28}}},   // XAsync
            {5, {0x973a344e, 0x24bf, 0x4d0f, {0x84, 0x57, 0x56, 0xc5, 0x34, 0x89, 0x2b, 0x29}}},   // XGame
            {6, {0x0651aae2, 0x4012, 0x4077, {0xbf, 0x84, 0x8b, 0x90, 0x97, 0x09, 0x0e, 0x2c}}},   // XGameInvite
            {7, {0x704c3f58, 0xe629, 0x4cc2, {0xb1, 0x97, 0x30, 0x51, 0x1b, 0x99, 0x6f, 0xe2}}},   // XGameSave
            {8, {0xdfcd4649, 0x4ff8, 0x4043, {0xba, 0x07, 0x35, 0xd6, 0x07, 0xdf, 0x98, 0xb0}}},   // XGameUI
            {9, {0x1b339674, 0x328d, 0x4283, {0xa2, 0x00, 0x31, 0x71, 0xf1, 0x8d, 0x36, 0x39}}},   // XLauncher
            {10, {0x37e56907, 0x2f10, 0x41e8, {0xb7, 0x2f, 0x36, 0xed, 0xb1, 0x85, 0x33, 0x1a}}},  // XNetworking
            {11, {0xaf406016, 0xe850, 0x4aa8, {0xa8, 0x8d, 0x2f, 0x3d, 0xcb, 0x9d, 0xac, 0x7e}}},  // XPackage
            {12, {0xf4faf4d4, 0x2d04, 0x4fce, {0xb3, 0xe0, 0x47, 0x4a, 0x71, 0x3a, 0x3e, 0x84}}},  // XPersistentLocalStorage
            {13, {0x3e241977, 0x6237, 0x41e9, {0x85, 0x59, 0x10, 0x2c, 0x2d, 0x99, 0x83, 0xf1}}},  // XSpeechSynthesizer
            {14, {0x0dd112ac, 0x7c24, 0x448c, {0xb9, 0x2b, 0x39, 0x60, 0xfb, 0x5b, 0xd3, 0x0c}}},  // XStore
            {15, {0xe349bd1a, 0xfc20, 0x4e40, {0xb9, 0x9c, 0x41, 0x78, 0xcc, 0x6b, 0x40, 0x9f}}},  // XSystem
            {18, {0x01acd177, 0x91f9, 0x4763, {0xa3, 0x8e, 0xcc, 0xbb, 0x55, 0xce, 0x32, 0xe0}}},  // XUser
            {19, {0x8ca467f7, 0x22e8, 0x4096, {0x84, 0x56, 0xbb, 0x8a, 0xa1, 0x3f, 0x79, 0xd8}}},  // XError
            {20, {0xbbfbdcc7, 0xbfe7, 0x409b, {0xa5, 0xca, 0xed, 0xf0, 0x54, 0x96, 0x0b, 0x4d}}},  // XGameEvent
            {21, {0x0a2192aa, 0xb2d5, 0x4d58, {0x83, 0xbe, 0x38, 0x3b, 0x6d, 0x80, 0x79, 0x9e}}},  // XGameStreaming
        };
        bool consistent = true;
        for (const auto& entry : follow) {
            void* obj = nullptr;
            const bool class_served = query(&entry.cls, kIUnknownId, &obj) != static_cast<HRESULT>(0x80070032);
            if (obj) release_object(obj);
            consistent = consistent && available(f, entry.feature) == class_served;
        }
        check(consistent, "slot 3: every other feature is available exactly when QueryApiImpl serves its class");
        release_object(f);
    }

    // A member function returning a struct, called the way MSVC calls one (the result's address after `this`, the
    // same address back in rax): what the thunk does.
    struct AnalyticsView {
        virtual HRESULT STDMETHODCALLTYPE QueryInterface(REFIID, void**) = 0;
        virtual ULONG STDMETHODCALLTYPE AddRef() = 0;
        virtual ULONG STDMETHODCALLTYPE Release() = 0;
        virtual XSystemAnalyticsInfo STDMETHODCALLTYPE GetAnalyticsInfo() = 0;
    };
    const GUID analytics_cls = {0xb884675d, 0xb738, 0x4a9c, {0x81, 0x5d, 0x9a, 0x9a, 0x1e, 0x0c, 0x6c, 0x9b}};
    void* a = ask(query, analytics_cls, analytics_cls);
    check(a != nullptr, "QueryApiImpl hands out XSystemAnalyticsImpl for its class id");
    if (a) {
        const XSystemAnalyticsInfo via_slot = static_cast<AnalyticsView*>(a)->GetAnalyticsInfo();
        const XSystemAnalyticsInfo named = named_analytics();
        check(std::memcmp(&via_slot, &named, sizeof named) == 0 && std::strcmp(via_slot.family, "Xbox") == 0,
              "slot 3 (XSystemGetAnalyticsInfo) returns the named export's struct by value");
        void* other = reinterpret_cast<void*>(1);
        check(query(&analytics_cls, kSystem5Id, &other) == E_NOINTERFACE && other == nullptr,
              "XSystemAnalyticsImpl answers its class id, not another class's interface");
        release_object(a);
    }

    // Classes nobody serves are still refused, and logged, as before.
    const GUID refused[] = {
        {0xa4f1aee2, 0x4bf1, 0x4485, {0xb0, 0x08, 0xa7, 0xc2, 0x6d, 0x52, 0xac, 0x27}},  // XAppCaptureImpl
        {0x12345678, 0x9abc, 0xdef0, {0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0}},  // no such class
    };
    bool all_refused = true;
    for (const GUID& cls : refused) {
        void* out = reinterpret_cast<void*>(1);
        all_refused = all_refused && query(&cls, cls, &out) == static_cast<HRESULT>(0x80070032) && out == nullptr;
    }
    check(all_refused, "XAppCaptureImpl and an unknown class are still refused");
}

// xgameruntime.lib's thunks don't import the runtime: they LoadLibraryW(L"xgameruntime.dll") and GetProcAddress five
// names (measured in the package's libHttpClient.GDK.dll and XCurl.dll). Drive that path through the loader shims,
// bound by the api set XCurl.dll imports them from, as a mapped image would be.
void check_run_time_load() {
    using LoadW = HMODULE(WINAPI*)(LPCWSTR);
    using LoadExA = HMODULE(WINAPI*)(LPCSTR, HANDLE, DWORD);
    using GetModW = HMODULE(WINAPI*)(LPCWSTR);
    using GetProc = FARPROC(WINAPI*)(HMODULE, LPCSTR);
    using Free = BOOL(WINAPI*)(HMODULE);
    const char* set = "api-ms-win-core-libraryloader-l1-2-0.dll";
    auto load_w = reinterpret_cast<LoadW>(rt::shims::find(set, "LoadLibraryW"));
    auto load_ex_a = reinterpret_cast<LoadExA>(rt::shims::find("kernel32.dll", "LoadLibraryExA"));
    auto get_mod_w = reinterpret_cast<GetModW>(rt::shims::find(set, "GetModuleHandleW"));
    auto get_proc = reinterpret_cast<GetProc>(rt::shims::find(set, "GetProcAddress"));
    auto free_lib = reinterpret_cast<Free>(rt::shims::find(set, "FreeLibrary"));
    check(load_w && load_ex_a && get_mod_w && get_proc && free_lib, "the loader functions are registered");
    if (!(load_w && load_ex_a && get_mod_w && get_proc && free_lib)) return;

    check(rt::shims::owns_dll("xgameruntime.dll") && !rt::shims::owns_dll("kernel32.dll") &&
              !rt::shims::owns_dll(set) && !rt::shims::owns_dll("nosuch.dll"),
          "owns_dll: xgameruntime.dll yes; kernel32, api sets and unknown modules no");

    HMODULE rt_mod = load_w(L"xgameruntime.dll");
    check(rt_mod != nullptr, "LoadLibraryW(xgameruntime.dll) is served without the host runtime");
    check(load_w(L"XGameRuntime") == rt_mod && load_ex_a("C:\\Some\\Dir\\XGAMERUNTIME.DLL", nullptr, 0) == rt_mod &&
              get_mod_w(L"xgameruntime.dll") == rt_mod,
          "the same module for a bare name, a full path in any case, LoadLibraryExA and GetModuleHandleW");

    const char* entry_points[] = {"InitializeApiImpl", "InitializeApiImplEx", "InitializeApiImplEx2",
                                  "UninitializeApiImpl", "QueryApiImpl"};
    bool all = true;
    for (const char* n : entry_points)
        all = all && get_proc(rt_mod, n) &&
              reinterpret_cast<void*>(get_proc(rt_mod, n)) == rt::shims::find("xgameruntime.dll", n);
    check(all, "GetProcAddress serves the five entry points xgameruntime.lib looks up");
    SetLastError(0);
    check(get_proc(rt_mod, "NoSuchRuntimeFunction") == nullptr && GetLastError() == ERROR_PROC_NOT_FOUND,
          "an unserved name is NULL with ERROR_PROC_NOT_FOUND, as from a real module");
    SetLastError(0);
    check(get_proc(rt_mod, nullptr) == nullptr && GetLastError() == ERROR_PROC_NOT_FOUND,
          "a null proc name is NULL with ERROR_PROC_NOT_FOUND");

    // XCurl.dll's other way in: LoadLibraryExW(L"gameplatformservices.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32),
    // then GetApiVersionImpl(uint64_t*, uint64_t*) and QueryApiImpl. Served by the shims, never the PC's own.
    using LoadExW = HMODULE(WINAPI*)(LPCWSTR, HANDLE, DWORD);
    auto load_ex_w = reinterpret_cast<LoadExW>(rt::shims::find(set, "LoadLibraryExW"));
    HMODULE gps = load_ex_w ? load_ex_w(L"gameplatformservices.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32) : nullptr;
    check(rt::shims::owns_dll("gameplatformservices.dll") && gps != nullptr && gps != rt_mod,
          "LoadLibraryExW(gameplatformservices.dll, SEARCH_SYSTEM32) is served by the shims");
    using ApiVersion = HRESULT(WINAPI*)(uint64_t*, uint64_t*);
    auto api_version = gps ? reinterpret_cast<ApiVersion>(get_proc(gps, "GetApiVersionImpl")) : nullptr;
    uint64_t v1 = 0, v2 = 0, w1 = 0, w2 = 0;
    check(api_version && api_version(&v1, &v2) == S_OK && api_version(&w1, &w2) == S_OK && v1 && v2 && v1 == w1 &&
              v2 == w2,
          "GetApiVersionImpl writes the same two non-zero versions on every call");
    check(gps && reinterpret_cast<void*>(get_proc(gps, "QueryApiImpl")) ==
                     rt::shims::find("xgameruntime.dll", "QueryApiImpl"),
          "its QueryApiImpl is the offline runtime's");

    struct Options {
        uint32_t unknown;
        BOOL is_inline;
        const char* game_config;
    } options{1, FALSE, nullptr};
    using InitEx2 = HRESULT(WINAPI*)(ULONG, ULONG, char, const Options*);
    using Query = HRESULT(WINAPI*)(const GUID*, REFIID, void**);
    using Uninit = HRESULT(WINAPI*)();
    auto init_ex2 = reinterpret_cast<InitEx2>(get_proc(rt_mod, "InitializeApiImplEx2"));
    auto query = reinterpret_cast<Query>(get_proc(rt_mod, "QueryApiImpl"));
    auto uninit_api = reinterpret_cast<Uninit>(get_proc(rt_mod, "UninitializeApiImpl"));
    check(init_ex2(0x2410, 0x0, 0x02, &options) == S_OK, "InitializeApiImplEx2 succeeds offline");
    // An id no runtime class has (made up for this test).
    const GUID some_class = {0x4c454f4e, 0x4944, 0x4100, {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01}};
    void* out = reinterpret_cast<void*>(1);
    check(query(&some_class, some_class, &out) == static_cast<HRESULT>(0x80070032) && out == nullptr,
          "QueryApiImpl refuses a class it doesn't provide yet (HRESULT 0x80070032) and clears the out pointer");

    // XThreadingImpl and XErrorImpl, driven the way xgameruntime.lib's thunks in the package's DLLs do: QueryApiImpl
    // with the class id as both ids, one vtable slot with `this` first, then Release (slot 2).
    const GUID threading = {0x073b7dcb, 0x1fcf, 0x4030, {0x94, 0xbe, 0xe3, 0xc9, 0xeb, 0x62, 0x34, 0x28}};
    const GUID error_cls = {0x8ca467f7, 0x22e8, 0x4096, {0x84, 0x56, 0xbb, 0x8a, 0xa1, 0x3f, 0x79, 0xd8}};
    using Fn = void*;
    auto slot = [](void* obj, int index) { return (*static_cast<Fn* const*>(obj))[index]; };
    auto release = [&](void* obj) { reinterpret_cast<ULONG(STDMETHODCALLTYPE*)(void*)>(slot(obj, 2))(obj); };
    auto get = [&](const GUID& cls) {
        void* p = nullptr;
        return query(&cls, cls, &p) == S_OK ? p : nullptr;
    };
    void* t = get(threading);
    check(t != nullptr, "QueryApiImpl hands out XThreadingImpl for its class id");
    if (t) {
        void* again = get(threading);
        check(again == t, "XThreadingImpl is one process-wide object");
        release(again);
        XTaskQueueHandle queue = nullptr;
        using Create = HRESULT(STDMETHODCALLTYPE*)(void*, XTaskQueueDispatchMode, XTaskQueueDispatchMode,
                                                  XTaskQueueHandle*);
        check(reinterpret_cast<Create>(slot(t, 12))(t, XTaskQueueDispatchMode::Manual, XTaskQueueDispatchMode::Manual,
                                                    &queue) == S_OK && queue,
              "slot 12 is XTaskQueueCreate");
        static std::atomic<int> s_ran{0};
        using Submit = HRESULT(STDMETHODCALLTYPE*)(void*, XTaskQueueHandle, XTaskQueuePort, void*,
                                                  XTaskQueueCallback*);
        XTaskQueueCallback* cb = [](void* ctx, bool canceled) {
            if (!canceled) static_cast<std::atomic<int>*>(ctx)->fetch_add(1);
        };
        check(reinterpret_cast<Submit>(slot(t, 18))(t, queue, XTaskQueuePort::Work, &s_ran, cb) == S_OK,
              "slot 18 is XTaskQueueSubmitCallback");
        using Dispatch = bool(STDMETHODCALLTYPE*)(void*, XTaskQueueHandle, XTaskQueuePort, uint32_t);
        check(reinterpret_cast<Dispatch>(slot(t, 16))(t, queue, XTaskQueuePort::Work, 0) && s_ran.load() == 1,
              "slot 16 is XTaskQueueDispatch: the callback ran on the work port");
        XTaskQueueHandle dup = nullptr;
        using Dup = HRESULT(STDMETHODCALLTYPE*)(void*, XTaskQueueHandle, XTaskQueueHandle*);
        check(reinterpret_cast<Dup>(slot(t, 15))(t, queue, &dup) == S_OK && dup,
              "slot 15 is XTaskQueueDuplicateHandle");
        using Close = void(STDMETHODCALLTYPE*)(void*, XTaskQueueHandle);
        reinterpret_cast<Close>(slot(t, 17))(t, dup);
        reinterpret_cast<Close>(slot(t, 17))(t, queue);
        using SetSensitive = HRESULT(STDMETHODCALLTYPE*)(void*, bool);
        using IsSensitive = bool(STDMETHODCALLTYPE*)(void*);
        check(reinterpret_cast<SetSensitive>(slot(t, 27))(t, true) == S_OK &&
                  reinterpret_cast<IsSensitive>(slot(t, 30))(t) &&
                  reinterpret_cast<SetSensitive>(slot(t, 27))(t, false) == S_OK &&
                  !reinterpret_cast<IsSensitive>(slot(t, 30))(t),
              "slots 27 and 30 set and read this thread's time-sensitive flag");
        // Slot 8 (XAsyncBeginAlloc, private, signature not measured) must not report a begun call whose provider
        // never received XAsyncOp::Begin: either it refuses, or the provider saw Begin exactly once.
        static std::atomic<int> s_begins{0};
        XAsyncProvider* provider = [](XAsyncOp op, const XAsyncProviderData*) -> HRESULT {
            if (op == XAsyncOp::Begin) s_begins.fetch_add(1);
            return S_OK;
        };
        XAsyncBlock alloc_block{};
        using BeginAlloc = HRESULT(STDMETHODCALLTYPE*)(void*, XAsyncBlock*, const void*, const char*, XAsyncProvider*,
                                                        size_t, size_t, void*);
        const HRESULT alloc_hr =
            reinterpret_cast<BeginAlloc>(slot(t, 8))(t, &alloc_block, nullptr, "test", provider, 16, 0, nullptr);
        check(FAILED(alloc_hr) || s_begins.load() == 1,
              "slot 8 (XAsyncBeginAlloc) never reports success for a call whose provider did not begin");
        void* unk = nullptr;
        const GUID iunknown = {0, 0, 0, {0xc0, 0, 0, 0, 0, 0, 0, 0x46}};
        using QI = HRESULT(STDMETHODCALLTYPE*)(void*, REFIID, void**);
        check(reinterpret_cast<QI>(slot(t, 0))(t, iunknown, &unk) == S_OK && unk == t &&
                  reinterpret_cast<QI>(slot(t, 0))(t, error_cls, &unk) == E_NOINTERFACE && !unk,
              "QueryInterface answers IUnknown and the class id, not another class");
        release(t);
        release(t);
    }
    void* e = get(error_cls);
    check(e != nullptr, "QueryApiImpl hands out XErrorImpl for its class id");
    if (e) {
        struct Seen {
            HRESULT hr;
            char msg[32];
        } seen{};
        using Callback = bool(CALLBACK*)(HRESULT, const char*, void*);
        Callback on_error = [](HRESULT hr, const char* msg, void* ctx) {
            auto* s = static_cast<Seen*>(ctx);
            s->hr = hr;
            strncpy_s(s->msg, msg ? msg : "", _TRUNCATE);
            return true;
        };
        using SetCallback = void(STDMETHODCALLTYPE*)(void*, Callback, void*);
        using Report = HRESULT(STDMETHODCALLTYPE*)(void*, HRESULT, const char*);
        reinterpret_cast<SetCallback>(slot(e, 4))(e, on_error, &seen);
        const HRESULT reported = reinterpret_cast<Report>(slot(e, 3))(e, E_FAIL, "offline test");
        check(reported == S_OK && seen.hr == E_FAIL && std::strcmp(seen.msg, "offline test") == 0,
              "slot 3 is XErrorReport and reaches the callback slot 4 (XErrorSetCallback) set");
        reinterpret_cast<SetCallback>(slot(e, 4))(e, nullptr, nullptr);
        using SetOptions = void(STDMETHODCALLTYPE*)(void*, uint32_t, uint32_t);  // void, per Microsoft's page
        reinterpret_cast<SetOptions>(slot(e, 5))(e, 0x01, 0x00);  // OutputDebugStringOnError / None
        release(e);
    }

    // XNetworkingImpl, asked for as XCurl.dll does: the class id with IXNetworkingImpl2's interface id.
    const GUID networking = {0x37e56907, 0x2f10, 0x41e8, {0xb7, 0x2f, 0x36, 0xed, 0xb1, 0x85, 0x33, 0x1a}};
    const GUID networking2 = {0xbf2346b2, 0x39af, 0x4658, {0xb5, 0xea, 0x44, 0x71, 0x3c, 0x7e, 0x83, 0xb3}};
    const HRESULT no_network = HRESULT_FROM_WIN32(ERROR_NETWORK_UNREACHABLE);
    void* n = nullptr;
    check(query(&networking, networking2, &n) == S_OK && n, "QueryApiImpl hands out XNetworkingImpl for IXNetworkingImpl2");
    void* n1 = nullptr;
    check(query(&networking, networking, &n1) == S_OK && n1 == n, "and the same object for IXNetworkingImpl");
    if (n1) release(n1);
    if (n) {
        struct Hint {
            uint32_t level, cost, iana;
            bool initialized, approaching, over, roaming;
        } hint{9, 9, 9, false, true, true, true};
        using GetHint = HRESULT(STDMETHODCALLTYPE*)(void*, Hint*);
        check(reinterpret_cast<GetHint>(slot(n, 15))(n, &hint) == S_OK && hint.level == 1 && hint.cost == 0 &&
                  hint.initialized && !hint.approaching && !hint.over && !hint.roaming,
              "slot 15 (XNetworkingGetConnectivityHint): initialized, no connectivity");
        XAsyncBlock block{};
        using SecurityAsync = HRESULT(STDMETHODCALLTYPE*)(void*, const wchar_t*, XAsyncBlock*);
        using ResultSize = HRESULT(STDMETHODCALLTYPE*)(void*, XAsyncBlock*, size_t*);
        size_t size = 99;
        check(reinterpret_cast<SecurityAsync>(slot(n, 11))(n, L"https://example.invalid/", &block) == S_OK &&
                  XAsyncGetStatus(&block, true) == no_network &&
                  reinterpret_cast<ResultSize>(slot(n, 12))(n, &block, &size) == no_network && size == 0,
              "slots 11/12 (UTF-16 URL security info): the async call completes with no network");
        // Slots 10/13 in Microsoft's order: (async, bufferByteCount, bufferByteCountUsed*, buffer*, information**).
        // A sentinel buffer must stay untouched; the two outputs are written where they are, not into the buffer.
        using SecurityResult = HRESULT(STDMETHODCALLTYPE*)(void*, XAsyncBlock*, size_t, size_t*, uint8_t*, void**);
        for (int s : {10, 13}) {
            uint8_t buffer[32];
            std::memset(buffer, 0xA5, sizeof(buffer));
            size_t used = 77;
            void* info = &used;
            const HRESULT hr = reinterpret_cast<SecurityResult>(slot(n, s))(n, &block, sizeof(buffer), &used, buffer, &info);
            bool untouched = true;
            for (uint8_t byte : buffer) untouched = untouched && byte == 0xA5;
            check(hr == no_network && used == 0 && info == nullptr && untouched,
                  s == 10 ? "slot 10 (ANSI result): outputs zeroed in place, the caller's buffer untouched"
                          : "slot 13 (UTF-16 result): outputs zeroed in place, the caller's buffer untouched");
        }
        using Verify = HRESULT(STDMETHODCALLTYPE*)(void*, void*, const void*);
        check(reinterpret_cast<Verify>(slot(n, 14))(n, nullptr, nullptr) == no_network,
              "slot 14 (XNetworkingVerifyServerCertificate) fails offline");
        struct Seen {
            int calls;
            Hint hint;
            uint16_t port;
        };
        using HintCallback = void(CALLBACK*)(void*, const Hint*);
        using PortCallback = void(CALLBACK*)(void*, uint16_t);
        using RegisterHint = HRESULT(STDMETHODCALLTYPE*)(void*, XTaskQueueHandle, void*, HintCallback,
                                                         XTaskQueueRegistrationToken*);
        using RegisterPort = HRESULT(STDMETHODCALLTYPE*)(void*, XTaskQueueHandle, void*, PortCallback,
                                                         XTaskQueueRegistrationToken*);
        using Unregister = bool(STDMETHODCALLTYPE*)(void*, XTaskQueueRegistrationToken, bool);
        const auto register_hint = reinterpret_cast<RegisterHint>(slot(n, 16));
        const auto register_port = reinterpret_cast<RegisterPort>(slot(n, 6));
        const auto unregister_hint = reinterpret_cast<Unregister>(slot(n, 17));
        const auto unregister_port = reinterpret_cast<Unregister>(slot(n, 7));
        HintCallback on_hint = [](void* ctx, const Hint* h) {
            auto* s = static_cast<Seen*>(ctx);
            ++s->calls;
            if (h) s->hint = *h;
        };
        PortCallback on_port = [](void* ctx, uint16_t p) {
            auto* s = static_cast<Seen*>(ctx);
            ++s->calls;
            s->port = p;
        };
        XTaskQueueRegistrationToken token{};
        check(register_hint(n, nullptr, nullptr, nullptr, &token) == E_INVALIDARG &&
                  register_port(n, nullptr, nullptr, nullptr, &token) == E_INVALIDARG,
              "slots 6/16 refuse a null callback");
        XTaskQueueHandle manual = nullptr;
        XTaskQueueCreate(XTaskQueueDispatchMode::Manual, XTaskQueueDispatchMode::Manual, &manual);
        Seen hint_seen{0, {9, 9, 9, false, true, true, true}, 0};
        Seen port_seen{0, {}, 7};
        XTaskQueueRegistrationToken hint_token{}, port_token{};
        const bool registered = register_hint(n, manual, &hint_seen, on_hint, &hint_token) == S_OK && hint_token.token &&
                                register_port(n, manual, &port_seen, on_port, &port_token) == S_OK && port_token.token;
        const bool none_before_dispatch = hint_seen.calls == 0 && port_seen.calls == 0;
        while (XTaskQueueDispatch(manual, XTaskQueuePort::Completion, 0)) {}
        const Hint& h = hint_seen.hint;
        check(registered && none_before_dispatch && hint_seen.calls == 1 && h.level == 1 && h.cost == 0 &&
                  h.initialized && !h.approaching && !h.over && !h.roaming,
              "slot 16 sends one initial notification on the queue's completion port: initialized, no connectivity");
        check(port_seen.calls == 1 && port_seen.port == 0, "slot 6 sends one initial notification too: port 0");
        check(unregister_hint(n, hint_token, true) && unregister_port(n, port_token, true),
              "slots 17/7 unregister them");
        Seen late{};
        XTaskQueueRegistrationToken late_hint{}, late_port{};
        const bool unregistered = register_hint(n, manual, &late, on_hint, &late_hint) == S_OK &&
                                  register_port(n, manual, &late, on_port, &late_port) == S_OK &&
                                  unregister_hint(n, late_hint, false) && unregister_port(n, late_port, false);
        while (XTaskQueueDispatch(manual, XTaskQueuePort::Completion, 0)) {}
        check(unregistered && late.calls == 0, "a registration ended before its queue is dispatched sends nothing");
        XTaskQueueCloseHandle(manual);
        uint16_t port = 7;
        using QueryPort = HRESULT(STDMETHODCALLTYPE*)(void*, uint16_t*);
        using NotImpl = HRESULT(STDMETHODCALLTYPE*)(void*);
        check(reinterpret_cast<QueryPort>(slot(n, 3))(n, &port) == no_network && port == 0 &&
                  reinterpret_cast<NotImpl>(slot(n, 20))(n) == E_NOTIMPL,
              "slot 3 (preferred UDP port) has no answer offline; slot 20 (IXNetworkingImpl2) is E_NOTIMPL");
        release(n);
    }

    // XPersistentLocalStorageImpl (RT-219): the class id with IXPersistentLocalStorageImpl3's interface id, over a
    // temporary overlay. The storage folder is <overlay>\.gta6rt-persistent.
    const GUID pls_class = {0xf4faf4d4, 0x2d04, 0x4fce, {0xb3, 0xe0, 0x47, 0x4a, 0x71, 0x3a, 0x3e, 0x84}};
    const GUID pls3 = {0x41a4e10c, 0x5a7e, 0x41d9, {0x8c, 0x37, 0x37, 0xbd, 0xe6, 0x2a, 0x07, 0xd6}};
    const std::filesystem::path scratch =
        std::filesystem::temp_directory_path() / (L"gta6rt_pls_test_" + std::to_wstring(GetCurrentProcessId()));
    std::error_code scratch_ec;
    std::filesystem::remove_all(scratch, scratch_ec);
    rt::fs::configure(scratch / L"game", scratch / L"overlay");
    void* pls = nullptr;
    check(query(&pls_class, pls3, &pls) == S_OK && pls, "QueryApiImpl hands out XPersistentLocalStorageImpl for IXPersistentLocalStorageImpl3");
    if (pls) {
        using GetPathSize = HRESULT(STDMETHODCALLTYPE*)(void*, size_t*);
        using GetPath = HRESULT(STDMETHODCALLTYPE*)(void*, size_t, char*, size_t*);
        using GetSpaceInfo = HRESULT(STDMETHODCALLTYPE*)(void*, uint64_t*);
        using PromptAsync = HRESULT(STDMETHODCALLTYPE*)(void*, uint64_t, XAsyncBlock*);
        using PromptResult = HRESULT(STDMETHODCALLTYPE*)(void*, XAsyncBlock*);
        using Mount = HRESULT(STDMETHODCALLTYPE*)(void*, const char*, void**);

        size_t size = 0;
        check(reinterpret_cast<GetPathSize>(slot(pls, 3))(pls, &size) == S_OK && size > 1,
              "slot 3 (GetPathSize) reports the length of the path with its NUL");
        std::string path(size, '\xA5');
        size_t used = 0;
        check(reinterpret_cast<GetPath>(slot(pls, 4))(pls, size, path.data(), &used) == S_OK && used == size &&
                  path[size - 1] == '\0',
              "slot 4 (GetPath) fills the buffer and reports the bytes used");
        const std::filesystem::path folder = std::filesystem::path(path.c_str());
        check(folder == scratch / L"overlay" / L".gta6rt-persistent" && std::filesystem::is_directory(folder),
              "the path is <overlay>\\.gta6rt-persistent and the folder exists");
        char tiny[2] = {'x', 'x'};
        used = 99;
        check(reinterpret_cast<GetPath>(slot(pls, 4))(pls, 2, tiny, &used) == HRESULT_FROM_WIN32(ERROR_INSUFFICIENT_BUFFER) &&
                  used == 0,
              "slot 4 refuses a buffer that is too small");

        {
            std::ofstream cache_file(folder / L"cache.bin", std::ios::binary);
            const std::string kilobyte(1024, 'z');
            cache_file << kilobyte;
        }
        uint64_t info[4] = {9, 9, 9, 9};  // available free, total free, used, total
        const HRESULT info_hr = reinterpret_cast<GetSpaceInfo>(slot(pls, 5))(pls, info);
        check(info_hr == S_OK && info[3] == 2048ull * 1024 * 1024 && info[2] == 1024 && info[1] == info[3] - 1024 &&
                  info[0] > 0 && info[0] <= info[1],
              "slot 5 (GetSpaceInfo): the 2048 MB quota, the bytes used, and what is left");

        XAsyncBlock fits{};
        check(reinterpret_cast<PromptAsync>(slot(pls, 6))(pls, 1024, &fits) == S_OK &&
                  XAsyncGetStatus(&fits, true) == S_OK && reinterpret_cast<PromptResult>(slot(pls, 7))(pls, &fits) == S_OK,
              "slots 6/7: a prompt for space the quota can hold succeeds");
        XAsyncBlock too_big{};
        check(reinterpret_cast<PromptAsync>(slot(pls, 6))(pls, 4096ull * 1024 * 1024, &too_big) == S_OK &&
                  XAsyncGetStatus(&too_big, true) == HRESULT_FROM_WIN32(ERROR_DISK_FULL) &&
                  reinterpret_cast<PromptResult>(slot(pls, 7))(pls, &too_big) == HRESULT_FROM_WIN32(ERROR_DISK_FULL),
              "slots 6/7: a prompt for more than the quota fails with ERROR_DISK_FULL");

        {
            using CreateFileWFn = HANDLE(WINAPI*)(LPCWSTR, DWORD, DWORD, LPSECURITY_ATTRIBUTES, DWORD, DWORD, HANDLE);
            using FindFirstFn = HANDLE(WINAPI*)(LPCWSTR, LPWIN32_FIND_DATAW);
            using FindNextFn = BOOL(WINAPI*)(HANDLE, LPWIN32_FIND_DATAW);
            using FindCloseFn = BOOL(WINAPI*)(HANDLE);
            using DeleteFn = BOOL(WINAPI*)(LPCWSTR);
            const auto create_w = reinterpret_cast<CreateFileWFn>(rt::shims::find("kernel32.dll", "CreateFileW"));
            const auto first_w = reinterpret_cast<FindFirstFn>(rt::shims::find("kernel32.dll", "FindFirstFileW"));
            const auto next_w = reinterpret_cast<FindNextFn>(rt::shims::find("kernel32.dll", "FindNextFileW"));
            const auto close_w = reinterpret_cast<FindCloseFn>(rt::shims::find("kernel32.dll", "FindClose"));
            const auto delete_w = reinterpret_cast<DeleteFn>(rt::shims::find("kernel32.dll", "DeleteFileW"));
            auto lists = [&](const wchar_t* pattern, const wchar_t* name) {
                bool found = false;
                WIN32_FIND_DATAW fd{};
                const HANDLE h = first_w(pattern, &fd);
                if (h == INVALID_HANDLE_VALUE) return false;
                do found = found || _wcsicmp(fd.cFileName, name) == 0;
                while (next_w(h, &fd));
                close_w(h);
                return found;
            };
            std::filesystem::create_directories(scratch / L"game" / L"persistent");
            std::ofstream(scratch / L"game" / L"persistent" / L"cache.bin", std::ios::binary) << "package";
            std::ofstream(folder / L"index.bin", std::ios::binary) << "storage";

            check(!lists(L"GAME:\\*", L".gta6rt-persistent") && !lists(L"GAME:\\persistent\\*", L"index.bin"),
                  "GAME: listings show neither the storage folder nor its files");
            char bytes[16] = {};
            DWORD got = 0;
            const HANDLE h = create_w(L"GAME:\\persistent\\cache.bin", GENERIC_READ, FILE_SHARE_READ, nullptr,
                                      OPEN_EXISTING, 0, nullptr);
            if (h != INVALID_HANDLE_VALUE) {
                ::ReadFile(h, bytes, sizeof bytes - 1, &got, nullptr);
                ::CloseHandle(h);
            }
            check(std::string(bytes) == "package",
                  "GAME:\\persistent\\cache.bin is the package's file, not the storage file of that name");
            std::error_code size_ec;
            check(delete_w(L"GAME:\\persistent\\cache.bin") &&
                      std::filesystem::file_size(folder / L"cache.bin", size_ec) == 1024,
                  "deleting the GAME: name leaves the storage file alone");
        }

        void* handle = reinterpret_cast<void*>(1);
        check(reinterpret_cast<Mount>(slot(pls, 8))(pls, "pkg", &handle) == E_NOTIMPL && handle == nullptr,
              "slot 8 (MountForPackage) is E_NOTIMPL and leaves no handle");
        release(pls);
    }
    rt::fs::configure({}, {});
    check(query(&pls_class, pls_class, &pls) == S_OK, "the class answers with its own id too");
    if (pls) {
        using GetPathSize = HRESULT(STDMETHODCALLTYPE*)(void*, size_t*);
        size_t size = 99;
        check(reinterpret_cast<GetPathSize>(slot(pls, 3))(pls, &size) == HRESULT_FROM_WIN32(ERROR_PATH_NOT_FOUND) && size == 0,
              "with no overlay configured the path size fails instead of answering S_OK with nothing");
        release(pls);
    }
    std::filesystem::remove_all(scratch, scratch_ec);

    // XSystemImpl, asked for as libHttpClient.GDK.dll does: the class id with IXSystemImpl5's interface id.
    const GUID system_cls = {0xe349bd1a, 0xfc20, 0x4e40, {0xb9, 0x9c, 0x41, 0x78, 0xcc, 0x6b, 0x40, 0x9f}};
    const GUID system5 = {0x1861cf2e, 0xe18b, 0x4834, {0xa9, 0xf5, 0xb4, 0xa4, 0xe6, 0xef, 0xb4, 0xcf}};
    void* sy = nullptr;
    check(query(&system_cls, system5, &sy) == S_OK && sy, "QueryApiImpl hands out XSystemImpl for IXSystemImpl5");
    if (sy) {
        using IdQuery = HRESULT(STDMETHODCALLTYPE*)(void*, size_t, char*, size_t*);
        char buf[64] = {};
        size_t used = 0;
        check(reinterpret_cast<IdQuery>(slot(sy, 3))(sy, sizeof(buf), buf, &used) == S_OK && used == 39 &&
                  std::strlen(buf) == 38,
              "slot 3 (XSystemGetConsoleId): the 39-byte placeholder");
        check(reinterpret_cast<IdQuery>(slot(sy, 4))(sy, sizeof(buf), buf, &used) == S_OK &&
                  std::strcmp(buf, "RETAIL") == 0 && used == 7,
              "slot 4 (XSystemGetXboxLiveSandboxId): RETAIL");
        check(reinterpret_cast<IdQuery>(slot(sy, 4))(sy, 15, buf, &used) == HRESULT_FROM_WIN32(ERROR_INSUFFICIENT_BUFFER) &&
                  reinterpret_cast<IdQuery>(slot(sy, 4))(sy, 16, buf, &used) == S_OK,
              "slot 4 needs XSystemXboxLiveSandboxIdMaxBytes (16), not just the id's 7 bytes");
        const GUID chain[] = {
            system_cls,
            {0x6fd71f09, 0x7513, 0x49f0, {0x89, 0xbc, 0xbf, 0xaf, 0x5d, 0xf6, 0xf8, 0x52}},
            {0x67ce4bfc, 0xb1d1, 0x4ac7, {0xbc, 0x3a, 0xcb, 0x92, 0x19, 0xa9, 0x7a, 0x85}},
            {0xdadc2895, 0x34b0, 0x4ef5, {0xa8, 0x3e, 0x45, 0x11, 0x4d, 0x62, 0x9b, 0x80}},
            system5};
        bool all_inherited = true;
        for (const GUID& iid : chain) {
            void* same = nullptr;
            all_inherited = all_inherited && query(&system_cls, iid, &same) == S_OK && same == sy;
            if (same) release(same);
        }
        check(all_inherited, "IXSystemImpl5 answers for every interface it inherits (IXSystemImpl .. IXSystemImpl4)");
        check(reinterpret_cast<IdQuery>(slot(sy, 5))(sy, sizeof(buf), buf, &used) == S_OK && used == 45 &&
                  reinterpret_cast<IdQuery>(slot(sy, 5))(sy, 44, buf, &used) ==
                      HRESULT_FROM_WIN32(ERROR_INSUFFICIENT_BUFFER) &&
                  reinterpret_cast<IdQuery>(slot(sy, 5))(sy, 45, nullptr, &used) == E_INVALIDARG,
              "slot 5 (XSystemGetAppSpecificDeviceId): 45 bytes; a short or null buffer is refused");
        using Track = HRESULT(STDMETHODCALLTYPE*)(void*, void*, void*);
        using IsValid = bool(STDMETHODCALLTYPE*)(void*, void*);  // bool, per Microsoft's reference page
        using Bandwidth = void(STDMETHODCALLTYPE*)(void*, bool);
        int ctx = 0;
        reinterpret_cast<Bandwidth>(slot(sy, 8))(sy, true);
        check(reinterpret_cast<Track>(slot(sy, 6))(sy, &ctx, &ctx) == S_OK &&
                  reinterpret_cast<IsValid>(slot(sy, 7))(sy, &ctx) &&
                  !reinterpret_cast<IsValid>(slot(sy, 7))(sy, nullptr),
              "slots 6-8 (handle tracking, handle check, download bandwidth) answer offline");
        release(sy);
    }

    // XPackageImpl, asked for as libHttpClient.GDK.dll does: the class id with IXPackageImpl3's interface id. The
    // slots with an offline answer agree with the named exports; the others are E_NOTIMPL.
    const GUID package_cls = {0xaf406016, 0xe850, 0x4aa8, {0xa8, 0x8d, 0x2f, 0x3d, 0xcb, 0x9d, 0xac, 0x7e}};
    const GUID package3 = {0xe2a4734b, 0x2f4a, 0x456d, {0xaa, 0x8f, 0xd0, 0x65, 0xe0, 0x4f, 0xb2, 0x09}};
    void* pk = nullptr;
    check(query(&package_cls, package3, &pk) == S_OK && pk, "QueryApiImpl hands out XPackageImpl for IXPackageImpl3");
    if (pk) {
        using Ident = HRESULT(STDMETHODCALLTYPE*)(void*, size_t, char*);
        using IdentExport = HRESULT(__stdcall*)(size_t, char*);
        char a[64] = {}, b[64] = {};
        const auto ident_export = reinterpret_cast<IdentExport>(
            rt::shims::find("xgameruntime.dll", "XPackageGetCurrentProcessPackageIdentifier"));
        check(reinterpret_cast<Ident>(slot(pk, 3))(pk, sizeof(a), a) == S_OK && ident_export &&
                  ident_export(sizeof(b), b) == S_OK && std::strcmp(a, b) == 0,
              "slot 3 is XPackageGetCurrentProcessPackageIdentifier, the same answer as the export");
        using IsPackaged = bool(STDMETHODCALLTYPE*)(void*);
        using IsPackagedExport = bool(__stdcall*)();
        const auto packaged_export =
            reinterpret_cast<IsPackagedExport>(rt::shims::find("xgameruntime.dll", "XPackageIsPackagedProcess"));
        check(packaged_export && reinterpret_cast<IsPackaged>(slot(pk, 4))(pk) == packaged_export(),
              "slot 4 is XPackageIsPackagedProcess");
        // Slot 11, XPackageGetUserLocale: one of the package's declared locales, never the host's; nothing written
        // (but S_OK) when the package declares none.
        using Locale = HRESULT(STDMETHODCALLTYPE*)(void*, size_t, char*);
        char tmp[MAX_PATH] = {};
        GetTempPathA(sizeof(tmp), tmp);
        const std::string game = std::string(tmp) + "leonida_xpackage_locale_test";
        CreateDirectoryA(game.c_str(), nullptr);
        SetEnvironmentVariableA("LEONIDA_GAME_DIR", game.c_str());
        char locale[128];
        std::memset(locale, 'z', sizeof(locale));
        check(reinterpret_cast<Locale>(slot(pk, 11))(pk, sizeof(locale), locale) == S_OK && locale[0] == 'z',
              "slot 11 without declared locales: S_OK and nothing written");
        if (FILE* f = std::fopen((game + "\\MicrosoftGame.Config").c_str(), "wb")) {
            std::fputs("<Game><Resources><Resource Language=\"zz-ZZ\"/><Resource Language='xx-YY'/></Resources></Game>",
                       f);
            std::fclose(f);
        }
        check(reinterpret_cast<Locale>(slot(pk, 11))(pk, sizeof(locale), locale) == S_OK &&
                  (std::strcmp(locale, "zz-ZZ") == 0 || std::strcmp(locale, "xx-YY") == 0) &&
                  reinterpret_cast<Locale>(slot(pk, 11))(pk, 3, locale) == HRESULT_FROM_WIN32(ERROR_INSUFFICIENT_BUFFER),
              "slot 11 gives one of the package's declared locales, not the host's; a short buffer is refused");
        DeleteFileA((game + "\\MicrosoftGame.Config").c_str());
        RemoveDirectoryA(game.c_str());
        SetEnvironmentVariableA("LEONIDA_GAME_DIR", nullptr);
        {
            // A game folder outside every ANSI code page (U+16A0, a runic letter) reaches MicrosoftGame.Config intact.
            wchar_t wtmp[MAX_PATH] = {};
            GetTempPathW(MAX_PATH, wtmp);
            const std::wstring wide = std::wstring(wtmp) + L"leonida_xpackage_\u16a0ocale";
            CreateDirectoryW(wide.c_str(), nullptr);
            SetEnvironmentVariableW(L"LEONIDA_GAME_DIR", wide.c_str());
            const std::wstring wconfig = wide + L"\\MicrosoftGame.Config";
            if (FILE* f = _wfopen(wconfig.c_str(), L"wb")) {
                std::fputs("<Game><Resources><Resource Language=\"qq-QQ\"/></Resources></Game>", f);
                std::fclose(f);
            }
            char wlocale[32] = {};
            check(reinterpret_cast<Locale>(slot(pk, 11))(pk, sizeof(wlocale), wlocale) == S_OK &&
                      std::strcmp(wlocale, "qq-QQ") == 0,
                  "slot 11 reads MicrosoftGame.Config from a game folder named outside the ANSI code page");
            DeleteFileW(wconfig.c_str());
            RemoveDirectoryW(wide.c_str());
            SetEnvironmentVariableW(L"LEONIDA_GAME_DIR", nullptr);
        }
        // Slots 23-26 are the named exports' content-package mounts: XPackageMount, the path size and path of the
        // handle, and closing it. XPackageMountHandle is one u64, passed by value.
        using Mount = HRESULT(STDMETHODCALLTYPE*)(void*, const char*, uint64_t*);
        using PathSize = HRESULT(STDMETHODCALLTYPE*)(void*, uint64_t, size_t*);
        using Path = HRESULT(STDMETHODCALLTYPE*)(void*, uint64_t, size_t, char*);
        using CloseMount = void(STDMETHODCALLTYPE*)(void*, uint64_t);
        SetEnvironmentVariableA("LEONIDA_CONTENT_PACKAGES", "SLOTDLC_12345678901234567890123=C:\\LeonidaSlotDLC");
        uint64_t handle = 0;
        size_t need = 0;
        char path[MAX_PATH] = {};
        check(reinterpret_cast<Mount>(slot(pk, 23))(pk, "SLOTDLC_12345678901234567890123", &handle) == S_OK &&
                  handle != 0,
              "slot 23 (XPackageMount) mounts a content package and hands out a handle");
        check(reinterpret_cast<PathSize>(slot(pk, 24))(pk, handle, &need) == S_OK &&
                  need == std::strlen("C:\\LeonidaSlotDLC") + 1 &&
                  reinterpret_cast<Path>(slot(pk, 25))(pk, handle, need, path) == S_OK &&
                  std::strcmp(path, "C:\\LeonidaSlotDLC") == 0 &&
                  reinterpret_cast<Path>(slot(pk, 25))(pk, handle, need - 1, path) ==
                      HRESULT_FROM_WIN32(ERROR_INSUFFICIENT_BUFFER),
              "slots 24/25: the mount's path, and exactly the size XPackageGetMountPath needs");
        reinterpret_cast<CloseMount>(slot(pk, 26))(pk, handle);
        check(reinterpret_cast<PathSize>(slot(pk, 24))(pk, handle, &need) == HRESULT_FROM_WIN32(ERROR_NOT_FOUND),
              "slot 26 (XPackageCloseMountHandle) closes the handle");
        SetEnvironmentVariableA("LEONIDA_CONTENT_PACKAGES", nullptr);
        using Stub = HRESULT(STDMETHODCALLTYPE*)(void*);
        check(reinterpret_cast<Stub>(slot(pk, 5))(pk) == E_NOTIMPL && reinterpret_cast<Stub>(slot(pk, 40))(pk) == E_NOTIMPL,
              "the slots without an offline answer (5, 40) are E_NOTIMPL");
        struct Progress {
            uint64_t total, installed, launch;
            bool launchable, completed;
        } progress{~0ull, 7, 7, false, false};
        using GetProgress = void(STDMETHODCALLTYPE*)(void*, void*, Progress*);
        reinterpret_cast<GetProgress>(slot(pk, 7))(pk, nullptr, &progress);
        check(progress.launchable && progress.completed && progress.installed == progress.total,
              "slot 7 (void XPackageGetInstallationProgress) writes an installed, launchable, completed package");
        using Update = bool(STDMETHODCALLTYPE*)(void*, void*);
        using UnregProgress = bool(STDMETHODCALLTYPE*)(void*, void*, XTaskQueueRegistrationToken, bool);
        using UnregInstalled = bool(STDMETHODCALLTYPE*)(void*, XTaskQueueRegistrationToken, bool);
        using Uninstall = bool(STDMETHODCALLTYPE*)(void*, const char*);
        check(!reinterpret_cast<Update>(slot(pk, 8))(pk, nullptr) &&
                  reinterpret_cast<UnregProgress>(slot(pk, 10))(pk, nullptr, XTaskQueueRegistrationToken{1}, true) &&
                  reinterpret_cast<UnregInstalled>(slot(pk, 22))(pk, XTaskQueueRegistrationToken{1}, false) &&
                  !reinterpret_cast<Uninstall>(slot(pk, 34))(pk, "pkg"),
              "the bool slots 8, 10, 22 and 34 return their documented offline answers");
        const GUID chain[] = {
            package_cls,
            {0x3720de07, 0xe8e4, 0x44a3, {0xad, 0x32, 0xb3, 0x59, 0xe8, 0xad, 0xbe, 0x55}},
            {0xf92d8712, 0x2b27, 0x4d8a, {0xbf, 0x01, 0x11, 0xa6, 0xf8, 0xe3, 0xeb, 0x42}},
            package3};
        bool all = true;
        for (const GUID& iid : chain) {
            void* same = nullptr;
            all = all && query(&package_cls, iid, &same) == S_OK && same == pk;
            if (same) release(same);
        }
        check(all, "IXPackageImpl3 answers for IXPackageImpl and IXPackageImpl2 too");
        release(pk);
    }
    {
        // XGameImpl (IXGameImpl3 : IXGameImpl2 : IXGameImpl, xgameruntime include/xgame.idl); IXGameImpl's id is the
        // class id. Slots: 3 XGameGetXboxTitleId, 4 XLaunchNewGame (void), 5 XLaunchRestartOnCrash.
        const GUID game_cls = {0x973a344e, 0x24bf, 0x4d0f, {0x84, 0x57, 0x56, 0xc5, 0x34, 0x89, 0x2b, 0x29}};
        const GUID game2 = {0x50849859, 0x0ad8, 0x4f81, {0x80, 0xe4, 0x5b, 0xc7, 0x86, 0x26, 0xf8, 0x52}};
        const GUID game3 = {0x2549f142, 0x6419, 0x4a06, {0x97, 0xb5, 0x93, 0x1a, 0xab, 0x7c, 0x2f, 0x34}};
        void* g = get(game_cls);
        check(g != nullptr, "QueryApiImpl hands out XGameImpl for its class id");
        bool every = g != nullptr;
        for (const GUID& iid : {game2, game3}) {
            void* same = nullptr;
            every = every && query(&game_cls, iid, &same) == S_OK && same == g;
            if (same) release(same);
        }
        check(every, "XGameImpl answers for IXGameImpl2's and IXGameImpl3's ids, the same object");
        if (g) {
            using TitleId = HRESULT(STDMETHODCALLTYPE*)(void*, uint32_t*);
            using LaunchNew = void(STDMETHODCALLTYPE*)(void*, const char*, const char*, void*);
            using Restart = HRESULT(STDMETHODCALLTYPE*)(void*, const char*, uint32_t);
            char tmp[MAX_PATH] = {};
            GetTempPathA(sizeof(tmp), tmp);
            const std::string game = std::string(tmp) + "leonida_xgame_title_test";
            CreateDirectoryA(game.c_str(), nullptr);
            SetEnvironmentVariableA("LEONIDA_GAME_DIR", game.c_str());
            const std::string config = game + "\\MicrosoftGame.Config";
            auto write_config = [&](const char* text) {
                if (FILE* f = std::fopen(config.c_str(), "wb")) {
                    std::fputs(text, f);
                    std::fclose(f);
                }
            };
            uint32_t id = 0x5a5a5a5a;
            DeleteFileA(config.c_str());
            check(reinterpret_cast<TitleId>(slot(g, 3))(g, &id) == E_FAIL && id == 0x5a5a5a5a,
                  "slot 3 (XGameGetXboxTitleId) without a config: E_FAIL, nothing written");
            write_config("<Game configVersion=\"1\">\n  <TitleId> 1a2B3c4D </TitleId>\n</Game>\n");
            check(reinterpret_cast<TitleId>(slot(g, 3))(g, &id) == S_OK && id == 0x1A2B3C4Du,
                  "slot 3 gives the config's <TitleId> (hex, either case, whitespace allowed)");
            check(reinterpret_cast<TitleId>(slot(g, 3))(g, nullptr) == E_INVALIDARG,
                  "slot 3 with a null pointer is E_INVALIDARG");
            write_config("<Game><TitleId>not-hex</TitleId></Game>");
            id = 0x5a5a5a5a;
            check(reinterpret_cast<TitleId>(slot(g, 3))(g, &id) == E_FAIL && id == 0x5a5a5a5a,
                  "slot 3 refuses a <TitleId> that is not hex");
            write_config("<Game><TitleId>123456789</TitleId></Game>");
            check(reinterpret_cast<TitleId>(slot(g, 3))(g, &id) == E_FAIL, "slot 3 refuses more than 8 hex digits");
            DeleteFileA(config.c_str());
            RemoveDirectoryA(game.c_str());
            SetEnvironmentVariableA("LEONIDA_GAME_DIR", nullptr);
            reinterpret_cast<LaunchNew>(slot(g, 4))(g, "other.exe", "", nullptr);
            check(true, "slot 4 (void XLaunchNewGame) returns offline");
            check(reinterpret_cast<Restart>(slot(g, 5))(g, "--restarted", 0) == S_OK,
                  "slot 5 (XLaunchRestartOnCrash) succeeds offline");
            release(g);
        }
    }
    check_measured_classes(query,
                           reinterpret_cast<AnalyticsInfoFn>(rt::shims::find("xgameruntime.dll", "XSystemGetAnalyticsInfo")));
    check(uninit_api() == S_OK, "UninitializeApiImpl succeeds");

    HMODULE k32 = load_w(L"kernel32.dll");
    check(k32 == GetModuleHandleW(L"kernel32.dll"), "a host module still loads from the host");
    check(reinterpret_cast<void*>(get_proc(k32, "GetTickCount")) ==
              reinterpret_cast<void*>(GetProcAddress(k32, "GetTickCount")),
          "GetProcAddress on a host module is the host's");
    check(free_lib(rt_mod) == TRUE && load_w(L"xgameruntime.dll") == rt_mod,
          "FreeLibrary on the runtime succeeds and keeps it loaded");

    g_teardown_rt_mod = rt_mod;
    g_teardown_get_proc = get_proc;
    g_teardown_free_lib = free_lib;
}

}  // namespace

int main() {
    std::printf("xgameruntime_test: the offline GDK runtime shim, through the shim registry\n");

    int missing = 0;
    for (const char* n : kNames)
        if (!rt::shims::find("xgameruntime.dll", n)) {
            std::printf("     not registered: %s\n", n);
            ++missing;
        }
    check(missing == 0, "every xgameruntime.dll function in the table resolves through rt::shims::find");
    if (missing) {
        std::printf("xgameruntime_test: %d of %d checks failed\n", g_failures, g_checks);
        return 1;
    }

    auto init = get<InitFn>("XGameRuntimeInitialize");
    auto uninit = get<UninitFn>("XGameRuntimeUninitialize");
    check(init() == S_OK, "XGameRuntimeInitialize returns S_OK");
    check(init() == S_OK, "a second XGameRuntimeInitialize also returns S_OK (reference counted)");

    check(get<DeviceTypeFn>("XSystemGetDeviceType")() == 7, "XSystemGetDeviceType reports XboxScarlettAnaconda (7)");

    auto console_id = get<ConsoleIdFn>("XSystemGetConsoleId");
    char id[64] = {};
    size_t used = 0;
    check(console_id(8, id, &used) == HRESULT_FROM_WIN32(ERROR_INSUFFICIENT_BUFFER),
          "XSystemGetConsoleId rejects a buffer smaller than XSystemConsoleIdBytes");
    check(console_id(sizeof(id), id, &used) == S_OK && used == 39 && std::strlen(id) == 38,
          "XSystemGetConsoleId fills 39 bytes (38 characters and the terminator)");

    auto analytics_info = get<AnalyticsInfoFn>("XSystemGetAnalyticsInfo");
    XSystemAnalyticsInfo info = analytics_info();
    check(info.osVersion.major == 10 && info.osVersion.minor == 0 && info.osVersion.build == 22000 &&
              info.osVersion.revision == 0,
          "XSystemGetAnalyticsInfo returns osVersion 10.0.22000.0");
    check(info.hostingOsVersion.major == 10 && info.hostingOsVersion.minor == 0 &&
              info.hostingOsVersion.build == 22000 && info.hostingOsVersion.revision == 0,
          "XSystemGetAnalyticsInfo returns hostingOsVersion 10.0.22000.0");
    check(std::strcmp(info.family, "Xbox") == 0, "XSystemGetAnalyticsInfo returns family 'Xbox'");
    check(std::strcmp(info.form, "Xbox Series X") == 0, "XSystemGetAnalyticsInfo returns form 'Xbox Series X'");

    auto package_id = get<PackageIdentifierFn>("XPackageGetCurrentProcessPackageIdentifier");
    check(package_id(0, nullptr) == E_INVALIDARG, "XPackageGetCurrentProcessPackageIdentifier rejects null buffer and size 0");
    char pkg_id[64] = {};
    check(package_id(0, pkg_id) == E_INVALIDARG, "XPackageGetCurrentProcessPackageIdentifier rejects size 0 with buffer");
    check(package_id(10, nullptr) == E_INVALIDARG, "XPackageGetCurrentProcessPackageIdentifier rejects null pointer with size");
    check(package_id(16, pkg_id) == HRESULT_FROM_WIN32(ERROR_INSUFFICIENT_BUFFER),
          "XPackageGetCurrentProcessPackageIdentifier rejects insufficient buffer");
    check(package_id(sizeof(pkg_id), pkg_id) == S_OK && std::strlen(pkg_id) == 32,
          "XPackageGetCurrentProcessPackageIdentifier fills 33-byte package identifier");

    SetEnvironmentVariableA("LEONIDA_PACKAGE_ID", "CustomTestPackageId_123456789012");
    check(package_id(sizeof(pkg_id), pkg_id) == S_OK &&
              std::strcmp(pkg_id, "CustomTestPackageId_123456789012") == 0,
          "XPackageGetCurrentProcessPackageIdentifier respects LEONIDA_PACKAGE_ID override");
    SetEnvironmentVariableA("LEONIDA_PACKAGE_ID", nullptr);
    // pkg_id still holds the override; the mount checks below need the game's own identifier again.
    check(package_id(sizeof(pkg_id), pkg_id) == S_OK && std::strlen(pkg_id) == 32,
          "XPackageGetCurrentProcessPackageIdentifier returns the default again once the override is cleared");

    auto mount_pkg = get<PackageMountFn>("XPackageMount");
    auto close_mount = get<PackageCloseMountHandleFn>("XPackageCloseMountHandle");
    auto mount_path_size = get<PackageMountPathSizeFn>("XPackageGetMountPathSize");
    auto mount_path = get<PackageMountPathFn>("XPackageGetMountPath");

    const XPackageMountHandle kInvalidMount{0};
    const XPackageMountHandle kStaleMount{999};

    XPackageMountHandle test_out{0};
    check(mount_pkg(nullptr, nullptr) == E_INVALIDARG, "XPackageMount rejects null identifier and null out pointer");
    check(mount_pkg(pkg_id, nullptr) == E_INVALIDARG, "XPackageMount rejects null out pointer with identifier");
    check(mount_pkg(nullptr, &test_out) == E_INVALIDARG, "XPackageMount rejects null identifier with out pointer");
    check(mount_pkg(pkg_id, &test_out) == E_ACCESSDENIED,
          "XPackageMount rejects game package identifier with E_ACCESSDENIED");
    check(mount_pkg("UninstalledPackage_0000000000000", &test_out) == HRESULT_FROM_WIN32(ERROR_NOT_FOUND),
          "XPackageMount rejects uninstalled package identifier with ERROR_NOT_FOUND");

    size_t req_size = 0;
    check(mount_path_size(kInvalidMount, &req_size) == E_INVALIDARG,
          "XPackageGetMountPathSize rejects invalid mount handle 0");
    check(mount_path_size(kStaleMount, &req_size) == HRESULT_FROM_WIN32(ERROR_NOT_FOUND),
          "XPackageGetMountPathSize rejects unknown mount handle");

    char mnt_path[MAX_PATH] = {};
    check(mount_path(kInvalidMount, 0, nullptr) == E_INVALIDARG,
          "XPackageGetMountPath rejects invalid mount handle 0");
    check(mount_path(kStaleMount, sizeof(mnt_path), mnt_path) == HRESULT_FROM_WIN32(ERROR_NOT_FOUND),
          "XPackageGetMountPath rejects unknown mount handle");

    SetEnvironmentVariableA("LEONIDA_CONTENT_PACKAGES", "DLC1_123456789012345678901234567=C:\\LeonidaDLC1");
    XPackageMountHandle dlc_mount{0};
    check(mount_pkg("DLC1_123456789012345678901234567", &dlc_mount) == S_OK && dlc_mount.handle != 0,
          "XPackageMount mounts content package from LEONIDA_CONTENT_PACKAGES");
    check(mount_path_size(dlc_mount, nullptr) == E_INVALIDARG,
          "XPackageGetMountPathSize rejects null size pointer");
    check(mount_path_size(dlc_mount, &req_size) == S_OK && req_size == std::strlen("C:\\LeonidaDLC1") + 1,
          "XPackageGetMountPathSize returns required buffer size including null terminator for mounted handle");
    check(mount_path(dlc_mount, 0, mnt_path) == E_INVALIDARG,
          "XPackageGetMountPath rejects size 0 with buffer");
    check(mount_path(dlc_mount, 10, nullptr) == E_INVALIDARG,
          "XPackageGetMountPath rejects null pointer with size");
    check(mount_path(dlc_mount, 5, mnt_path) == HRESULT_FROM_WIN32(ERROR_INSUFFICIENT_BUFFER),
          "XPackageGetMountPath rejects insufficient buffer");
    check(mount_path(dlc_mount, sizeof(mnt_path), mnt_path) == S_OK &&
              std::strcmp(mnt_path, "C:\\LeonidaDLC1") == 0,
          "XPackageGetMountPath returns configured content directory for mounted handle");

    close_mount(dlc_mount);
    check(mount_path_size(dlc_mount, &req_size) == HRESULT_FROM_WIN32(ERROR_NOT_FOUND),
          "XPackageGetMountPathSize rejects closed/stale mount handle");
    check(mount_path(dlc_mount, sizeof(mnt_path), mnt_path) == HRESULT_FROM_WIN32(ERROR_NOT_FOUND),
          "XPackageGetMountPath rejects closed/stale mount handle");
    SetEnvironmentVariableA("LEONIDA_CONTENT_PACKAGES", nullptr);

    auto is_packaged = get<PackagedProcessFn>("XPackageIsPackagedProcess");
    check(is_packaged() == true, "XPackageIsPackagedProcess returns true by default");
    SetEnvironmentVariableA("LEONIDA_PACKAGED", "0");
    check(is_packaged() == false, "XPackageIsPackagedProcess respects LEONIDA_PACKAGED=0 override");
    SetEnvironmentVariableA("LEONIDA_PACKAGED", nullptr);

    // Work on the thread pool, completions dispatched by the "game loop" on this thread.
    auto create = get<QueueCreateFn>("XTaskQueueCreate");
    auto close = get<QueueCloseFn>("XTaskQueueCloseHandle");
    auto dispatch = get<QueueDispatchFn>("XTaskQueueDispatch");
    auto run = get<AsyncRunFn>("XAsyncRun");
    auto status = get<AsyncStatusFn>("XAsyncGetStatus");
    XTaskQueueHandle q = nullptr;
    check(create(XTaskQueueDispatchMode::ThreadPool, XTaskQueueDispatchMode::Manual, &q) == S_OK && q,
          "XTaskQueueCreate(ThreadPool work, Manual completion)");
    int result = 0;
    XAsyncBlock block{};
    block.queue = q;
    block.context = &result;
    block.callback = completed;
    check(run(&block, work) == S_OK, "XAsyncRun accepts work on that queue");
    for (int i = 0; i < 500 && g_completions == 0; ++i) {
        while (dispatch(q, XTaskQueuePort::Completion, 0)) {}
        if (!g_completions) Sleep(2);
    }
    check(g_work_runs == 1 && result == 42, "the work callback ran once and wrote its result");
    check(g_work_thread != GetCurrentThreadId(), "the work callback ran on a thread-pool thread");
    check(g_completions == 1 && g_completion_thread == GetCurrentThreadId(),
          "the completion callback ran once, on the thread that dispatched the completion port");
    check(status(&block, false) == S_OK, "XAsyncGetStatus reports S_OK after completion");
    close(q);

    // The process task queue, used when XAsyncBlock::queue is null.
    XTaskQueueHandle pq = nullptr;
    check(get<QueueProcessFn>("XTaskQueueGetCurrentProcessTaskQueue")(&pq) && pq,
          "XTaskQueueGetCurrentProcessTaskQueue returns a queue");
    if (pq) close(pq);
    int result2 = 0;
    XAsyncBlock block2{};
    block2.context = &result2;
    check(run(&block2, work) == S_OK && status(&block2, true) == S_OK && result2 == 42,
          "XAsyncRun with a null queue completes on the process queue");

    // A custom provider: XAsyncBegin, XAsyncSchedule, XAsyncComplete, XAsyncGetResultSize, XAsyncGetResult.
    ProviderState st{get<AsyncCompleteFn>("XAsyncComplete")};
    XAsyncBlock block3{};
    static const int kIdentity = 0;
    HRESULT hr = get<AsyncBeginFn>("XAsyncBegin")(&block3, &st, &kIdentity, "xgameruntime_test", provider);
    check(hr == S_OK, "XAsyncBegin starts a provider");
    check(get<AsyncScheduleFn>("XAsyncSchedule")(&block3, 0) == S_OK, "XAsyncSchedule queues its work");
    check(status(&block3, true) == S_OK, "the provider completes with S_OK");
    size_t size = 0;
    check(get<AsyncResultSizeFn>("XAsyncGetResultSize")(&block3, &size) == S_OK && size == sizeof(uint32_t),
          "XAsyncGetResultSize reports the 4-byte result");
    uint32_t value = 0;
    size_t got = 0;
    check(get<AsyncResultFn>("XAsyncGetResult")(&block3, &kIdentity, sizeof(value), &value, &got) == S_OK &&
              value == 0x1EA0 && got == sizeof(value),
          "XAsyncGetResult returns the provider's result");
    // Cleanup comes when the async state's last reference goes, which can be on the work thread just after the
    // result is read, so allow it a moment.
    const int cleanup_bit = 1 << static_cast<int>(XAsyncOp::Cleanup);
    for (int i = 0; i < 500 && !(st.ops & cleanup_bit); ++i) Sleep(2);
    check((st.ops & (1 << static_cast<int>(XAsyncOp::Begin))) && (st.ops & (1 << static_cast<int>(XAsyncOp::DoWork))) &&
              (st.ops & (1 << static_cast<int>(XAsyncOp::GetResult))) && (st.ops & (1 << static_cast<int>(XAsyncOp::Cleanup))),
          "the provider saw Begin, DoWork, GetResult and Cleanup");

    uninit();
    uninit();
    check(init() == S_OK, "XGameRuntimeInitialize works again after balanced uninitialize calls");
    uninit();

    std::atexit(check_late_teardown);
    check_run_time_load();

    if (g_failures) {
        std::printf("xgameruntime_test: %d of %d checks failed\n", g_failures, g_checks);
        return 1;
    }
    std::printf("xgameruntime_test: OK (%d checks)\n", g_checks);
    return 0;
}
