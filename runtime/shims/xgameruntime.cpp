// xgameruntime: the GDK runtime surface a title calls before anything else, offline.
//
// Every GDK title starts with XGameRuntimeInitialize and builds its async work on XTaskQueue/XAsync. This table
// serves those from third_party/xtaskqueue (Microsoft's MIT-licensed implementation from libHttpClient), plus the
// XGameRuntime and XSystem calls a title makes at boot. Signatures and enum values follow Microsoft's public GDK
// documentation (learn.microsoft.com/gaming/gdk); no GDK header or library code is used.
//
// Nothing here goes online.
//
// Measured on the retail executable (docs/runtime_imports.md, "The retail executable's runtime classes"): it does not
// import xgameruntime.dll. Its 161 runtime thunks are linked into the image and reach the classes through
// QueryApiImpl, as libHttpClient.GDK.dll's do, so "xgameruntime.dll" below is the only module name needed. The name
// occurs in the image as a UTF-16 string, which would be a run-time load.
#include <windows.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <string>
#include <system_error>
#include <unordered_map>
#include <vector>

#include <XAsync.h>
#include <XAsyncProvider.h>
#include <XTaskQueue.h>

#include "file_system.h"
#include "shims.h"

namespace rt::shims {
bool xgameruntime_user_query(const GUID& class_id, REFIID interface_id, void** out, HRESULT* hr);
bool xgameruntime_gamesave_query(const GUID& class_id, REFIID interface_id, void** out, HRESULT* hr);
bool xgameruntime_store_query(const GUID& class_id, REFIID interface_id, void** out, HRESULT* hr);
bool xgameruntime_offline_query(const GUID& class_id, REFIID interface_id, void** out, HRESULT* hr);
}

namespace {

// ---- XGameRuntime ----

std::atomic<int> g_runtime_refs{0};

// ---- The run-time entry points xgameruntime.lib looks up ----
//
// A title linked against xgameruntime.lib doesn't import XGameRuntimeInitialize: the static library's thunks load
// "xgameruntime.dll" and GetProcAddress these five names (measured in the package's libHttpClient.GDK.dll and
// XCurl.dll, docs/runtime_imports.md). Signatures, the options struct and the error for an unknown class follow
// xodus-gaming/xgameruntime-docs (GPL-3.0; facts only, nothing copied): InitializeApiImplEx2 is the one
// XGameRuntimeInitialize[WithOptions] calls, and QueryApiImpl hands out the runtime's COM classes. XThreadingImpl,
// XErrorImpl, XNetworkingImpl, XSystemImpl, XPackageImpl, XGameRuntimeFeatureImpl and XSystemAnalyticsImpl are
// served (their layouts are measured, below), XPersistentLocalStorageImpl and XGameImpl from the public docs' layout
// (no caller in the shipped DLLs to measure), and the classes of the other xgameruntime_*.cpp files through their
// query functions; for any other class QueryApiImpl logs the class and interface id and refuses it, so the first boot
// with the executable lists the interfaces to implement next.

struct ApiImplOptions {    // InitializeApiImplEx2's fourth argument
    uint32_t unknown;      // documented as always 1
    BOOL is_inline;        // 1 when XGameRuntimeOptions::gameConfigSource is Inline
    const char* game_config;
};

constexpr HRESULT kClassNotProvided = static_cast<HRESULT>(0x80070032);  // HRESULT_FROM_WIN32(ERROR_NOT_SUPPORTED)

std::atomic<int> g_api_refs{0};

HRESULT api_initialize(const char* which, ULONG gdk_ver, ULONG gs_ver, int mode, const ApiImplOptions* options) {
    if (g_api_refs.fetch_add(1) == 0)
        std::fprintf(stderr, "[rt] xgameruntime: %s(gdk=%lu, gs=%lu, mode=%#x, options=%s) (offline)\n", which,
                     gdk_ver, gs_ver, mode & 0xFF,
                     options ? (options->is_inline ? "inline game config" : "default game config") : "none");
    return S_OK;
}

HRESULT WINAPI InitializeApiImpl(ULONG gdk_ver, ULONG gs_ver) {
    return api_initialize("InitializeApiImpl", gdk_ver, gs_ver, -1, nullptr);
}

HRESULT WINAPI InitializeApiImplEx(ULONG gdk_ver, ULONG gs_ver, char mode) {
    return api_initialize("InitializeApiImplEx", gdk_ver, gs_ver, static_cast<unsigned char>(mode), nullptr);
}

HRESULT WINAPI InitializeApiImplEx2(ULONG gdk_ver, ULONG gs_ver, char mode, const ApiImplOptions* options) {
    return api_initialize("InitializeApiImplEx2", gdk_ver, gs_ver, static_cast<unsigned char>(mode), options);
}

HRESULT WINAPI UninitializeApiImpl() {
    int prev = g_api_refs.load();
    while (prev > 0 && !g_api_refs.compare_exchange_weak(prev, prev - 1)) {}
    return S_OK;
}

void print_guid(char (&out)[40], const GUID* g) {
    if (!g) {
        std::snprintf(out, sizeof out, "(null)");
        return;
    }
    std::snprintf(out, sizeof out, "%08lx-%04x-%04x-%02x%02x-%02x%02x%02x%02x%02x%02x", g->Data1, g->Data2,
                  g->Data3, g->Data4[0], g->Data4[1], g->Data4[2], g->Data4[3], g->Data4[4], g->Data4[5],
                  g->Data4[6], g->Data4[7]);
}

// ---- The runtime classes QueryApiImpl hands out ----
//
// xgameruntime.lib's thunk for each function asks QueryApiImpl for the class (passing the class id as the interface
// id too), calls one slot of the object's vtable with the function's own arguments after `this`, and Releases it.
// The slot order is xgameruntime-docs' "Layout" list after IUnknown's three slots, and it is measured: the thunks
// statically linked into the package's libHttpClient.GDK.dll and XCurl.dll (disassembled, their code not run)
// call IXThreadingImpl slots 3-7 and 9-30, each with the argument shape of the documented function in that
// position (XAsyncBegin's five arguments at slot 7, XTaskQueueCreate's two modes and a handle pointer at 12,
// XTaskQueueTerminate's bool at 22, XThreadSetTimeSensitive's bool at 27, ...), and IXErrorImpl slot 3 with an
// HRESULT and a pointer. docs/runtime_imports.md has the table. Each function is served by the implementation the
// named export already uses (third_party/xtaskqueue), so the two ways in behave the same.

constexpr GUID kThreadingClass = {0x073b7dcb, 0x1fcf, 0x4030, {0x94, 0xbe, 0xe3, 0xc9, 0xeb, 0x62, 0x34, 0x28}};
constexpr GUID kErrorClass = {0x8ca467f7, 0x22e8, 0x4096, {0x84, 0x56, 0xbb, 0x8a, 0xa1, 0x3f, 0x79, 0xd8}};

// A function's arguments after `this`, forwarded unchanged (x64 has one calling convention).
// Specialised on the function's exact type, so a noexcept function matches only one form.
template <auto F, class T = decltype(F)> struct Method;
template <auto F, class R, class... A> struct Method<F, R (*)(A...)> {
    static R STDMETHODCALLTYPE call(void*, A... a) { return F(a...); }
};
template <auto F, class R, class... A> struct Method<F, R (*)(A...) noexcept> {
    static R STDMETHODCALLTYPE call(void*, A... a) noexcept { return F(a...); }
};

constexpr GUID kIUnknown = {0x00000000, 0x0000, 0x0000, {0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x46}};

// The runtime's objects are process-wide singletons: AddRef/Release count but never free.
struct RuntimeObject {
    const void* const* vtable;
    const GUID* class_id;               // the coclass id
    const GUID* const* interface_ids;   // every interface the vtable serves, base first, null-terminated: a derived
                                        // interface answers for each one it inherits (IXSystemImpl5 : ... : IXSystemImpl)
    std::atomic<long> refs{1};
};

HRESULT STDMETHODCALLTYPE object_query_interface(void* self, REFIID iid, void** out) {
    if (!out) return E_POINTER;
    auto* o = static_cast<RuntimeObject*>(self);
    bool known = iid == kIUnknown || iid == *o->class_id;
    for (const GUID* const* i = o->interface_ids; !known && i && *i; ++i) known = iid == **i;
    if (known) {
        o->refs.fetch_add(1);
        *out = self;
        return S_OK;
    }
    *out = nullptr;
    return E_NOINTERFACE;
}
ULONG STDMETHODCALLTYPE object_add_ref(void* self) { return static_cast<ULONG>(static_cast<RuntimeObject*>(self)->refs.fetch_add(1) + 1); }
ULONG STDMETHODCALLTYPE object_release(void* self) {
    long prev = static_cast<RuntimeObject*>(self)->refs.load();
    while (prev > 1 && !static_cast<RuntimeObject*>(self)->refs.compare_exchange_weak(prev, prev - 1)) {}
    return static_cast<ULONG>(prev > 1 ? prev - 1 : 1);  // the singleton's own reference stays
}

// A private slot whose signature isn't public: logged once, answers 0 (S_OK / false / nothing to do).
template <int Slot> uint64_t STDMETHODCALLTYPE private_slot(void*) {
    static std::atomic<bool> s_logged{false};
    if (!s_logged.exchange(true))
        std::fprintf(stderr, "[rt] xgameruntime: XThreadingImpl private slot %d called; answering 0\n", Slot);
    return 0;
}

// Slot 8, XAsyncBeginAlloc: private (no public page) and never called by the package's DLLs, so its signature here
// is not measured. Refuse it: answering S_OK would tell the caller its async call has begun while no provider ever
// receives XAsyncOp::Begin, so its completion never arrives.
HRESULT STDMETHODCALLTYPE begin_alloc_not_provided(void*) {
    static std::atomic<bool> s_logged{false};
    if (!s_logged.exchange(true))
        std::fprintf(stderr, "[rt] xgameruntime: XThreadingImpl slot 8 (XAsyncBeginAlloc) called; not provided, "
                             "returning E_NOTIMPL\n");
    return E_NOTIMPL;
}

// XThreadSetTimeSensitive / XThreadIsTimeSensitive / XThreadAssertNotTimeSensitive. The flag is per thread, as in
// the GDK; nothing is enforced offline (a time-sensitive thread may still block).
thread_local bool t_time_sensitive = false;
HRESULT STDMETHODCALLTYPE thread_set_time_sensitive(void*, bool value) {
    t_time_sensitive = value;
    return S_OK;
}
void STDMETHODCALLTYPE thread_assert_not_time_sensitive(void*) {}
bool STDMETHODCALLTYPE thread_is_time_sensitive(void*) { return t_time_sensitive; }

#define XM(fn) reinterpret_cast<const void*>(&Method<&fn>::call)
const void* const kThreadingVtable[] = {
    reinterpret_cast<const void*>(&object_query_interface), reinterpret_cast<const void*>(&object_add_ref),
    reinterpret_cast<const void*>(&object_release),
    XM(XAsyncGetStatus),                    // 3
    XM(XAsyncGetResultSize),                // 4
    XM(XAsyncCancel),                       // 5
    XM(XAsyncRun),                          // 6
    XM(XAsyncBegin),                        // 7
    reinterpret_cast<const void*>(&begin_alloc_not_provided),  // 8  XAsyncBeginAlloc (private, not called by the package)
    XM(XAsyncSchedule),                     // 9
    XM(XAsyncComplete),                     // 10
    XM(XAsyncGetResult),                    // 11
    XM(XTaskQueueCreate),                   // 12
    XM(XTaskQueueCreateComposite),          // 13
    XM(XTaskQueueGetPort),                  // 14
    XM(XTaskQueueDuplicateHandle),          // 15
    XM(XTaskQueueDispatch),                 // 16
    XM(XTaskQueueCloseHandle),              // 17
    XM(XTaskQueueSubmitCallback),           // 18
    XM(XTaskQueueSubmitDelayedCallback),    // 19
    XM(XTaskQueueRegisterWaiter),           // 20
    XM(XTaskQueueUnregisterWaiter),         // 21
    XM(XTaskQueueTerminate),                // 22
    XM(XTaskQueueRegisterMonitor),          // 23
    XM(XTaskQueueUnregisterMonitor),        // 24
    XM(XTaskQueueGetCurrentProcessTaskQueue),  // 25
    XM(XTaskQueueSetCurrentProcessTaskQueue),  // 26
    reinterpret_cast<const void*>(&thread_set_time_sensitive),         // 27
    reinterpret_cast<const void*>(&private_slot<28>),                  // 28 XTaskQueueIsEmpty (private; no arguments)
    reinterpret_cast<const void*>(&thread_assert_not_time_sensitive),  // 29
    reinterpret_cast<const void*>(&thread_is_time_sensitive),          // 30
};
#undef XM
static_assert(std::size(kThreadingVtable) == 31, "IUnknown + IXThreadingImpl's 28 methods");

// IXErrorImpl: XErrorReport(HRESULT, const char*) at slot 3 (the docs' "_unknown_"; XCurl calls it with an HRESULT
// and a string), XErrorSetCallback(XErrorCallback*, void*) and XErrorSetOptions(options, options). Offline, a report
// is logged and handed to the callback the title set, if any.
using XErrorCallback = bool(CALLBACK*)(HRESULT hr, const char* message, void* context);
std::atomic<XErrorCallback> g_error_callback{nullptr};
std::atomic<void*> g_error_context{nullptr};
std::mutex g_error_mutex{};   // guards g_error_callback + g_error_context as one consistent pair

// HRESULT XErrorReport(HRESULT, const char*) (xgameruntime-docs' XErrorReport.md): S_OK once reported.
HRESULT STDMETHODCALLTYPE error_report(void*, HRESULT hr, const char* message) {
    static std::atomic<int> s_logged{0};
    if (s_logged.fetch_add(1) < 64)
        std::fprintf(stderr, "[rt] xgameruntime: XErrorReport(0x%08lx, %s)\n", static_cast<unsigned long>(hr),
                     message ? message : "(null)");
    std::lock_guard<std::mutex> lock(g_error_mutex);
    if (XErrorCallback cb = g_error_callback.load()) cb(hr, message, g_error_context.load());
    return S_OK;
}
void STDMETHODCALLTYPE error_set_callback(void*, XErrorCallback callback, void* context) {
    std::lock_guard<std::mutex> lock(g_error_mutex);
    g_error_context = context;
    g_error_callback = callback;
}
// void XErrorSetOptions(XErrorOptions, XErrorOptions), as Microsoft documents it: nothing to configure offline.
void STDMETHODCALLTYPE error_set_options(void*, uint32_t, uint32_t) {}

const void* const kErrorVtable[] = {
    reinterpret_cast<const void*>(&object_query_interface), reinterpret_cast<const void*>(&object_add_ref),
    reinterpret_cast<const void*>(&object_release),
    reinterpret_cast<const void*>(&error_report),        // 3
    reinterpret_cast<const void*>(&error_set_callback),  // 4
    reinterpret_cast<const void*>(&error_set_options),   // 5
};

// ---- XNetworkingImpl (IXNetworkingImpl, IXNetworkingImpl2), offline ----
//
// XCurl.dll asks for the class with IXNetworkingImpl2's interface id and calls slots 11-17 (docs/runtime_imports.md,
// "Class layouts, measured"): the UTF-16 security-information query for a URL, the server certificate check, the
// connectivity hint and its change registration. Signatures follow Microsoft's public XNetworking documentation.
// Offline there is no network: the hint says so (initialized, no connectivity), every query that would need the
// network fails with kNoNetwork (asynchronously, through XAsync, for the *Async calls), a change registration sends
// the initial notification Microsoft documents (on the queue's completion port) and none after, since nothing
// changes, and the IXNetworkingImpl2 additions are E_NOTIMPL.

constexpr GUID kNetworkingClass = {0x37e56907, 0x2f10, 0x41e8, {0xb7, 0x2f, 0x36, 0xed, 0xb1, 0x85, 0x33, 0x1a}};
constexpr GUID kNetworking2 = {0xbf2346b2, 0x39af, 0x4658, {0xb5, 0xea, 0x44, 0x71, 0x3c, 0x7e, 0x83, 0xb3}};
constexpr HRESULT kNoNetwork = HRESULT_FROM_WIN32(ERROR_NETWORK_UNREACHABLE);

// XNetworkingConnectivityHint: level, cost, IANA interface type (u32 each), then four bools.
struct ConnectivityHint {
    uint32_t connectivity_level;   // XNetworkingConnectivityLevelHint: Unknown 0, None 1, LocalAccess 2, Internet 3
    uint32_t connectivity_cost;    // XNetworkingConnectivityCostHint: Unknown 0, Unrestricted 1, Fixed 2, Variable 3
    uint32_t iana_interface_type;
    bool network_initialized;
    bool approaching_data_limit;
    bool over_data_limit;
    bool roaming;
};
static_assert(sizeof(ConnectivityHint) == 16, "XNetworkingConnectivityHint is 16 bytes");

HRESULT STDMETHODCALLTYPE net_query_port(void*, uint16_t* port) {
    if (!port) return E_POINTER;
    *port = 0;
    return kNoNetwork;
}
// Completes `async` with kNoNetwork on its queue, as the GDK does for a query that fails.
HRESULT start_failed_async(XAsyncBlock* async) {
    if (!async) return E_POINTER;
    return XAsyncRun(async, [](XAsyncBlock*) -> HRESULT { return kNoNetwork; });
}
HRESULT STDMETHODCALLTYPE net_query_port_async(void*, XAsyncBlock* async) { return start_failed_async(async); }
HRESULT STDMETHODCALLTYPE net_query_port_result(void*, XAsyncBlock* async, uint16_t* port) {
    if (!async || !port) return E_POINTER;
    *port = 0;
    return XAsyncGetStatus(async, false);
}
HRESULT STDMETHODCALLTYPE net_security_async(void*, const void* url, XAsyncBlock* async) {
    if (!url) return E_INVALIDARG;
    return start_failed_async(async);
}
HRESULT STDMETHODCALLTYPE net_security_result_size(void*, XAsyncBlock* async, size_t* size) {
    if (!async || !size) return E_POINTER;
    *size = 0;
    return XAsyncGetStatus(async, false);
}
// XNetworkingQuerySecurityInformationForUrl[Utf16]AsyncResult(XAsyncBlock*, size_t bufferByteCount,
// size_t* bufferByteCountUsed, uint8_t* buffer, XNetworkingSecurityInformation** securityInformation), in the order
// of Microsoft's reference pages. The buffer itself is never written: there is no result to copy offline.
HRESULT STDMETHODCALLTYPE net_security_result(void*, XAsyncBlock* async, size_t, size_t* used, uint8_t*, void** info) {
    if (!async) return E_POINTER;
    if (used) *used = 0;
    if (info) *info = nullptr;
    return XAsyncGetStatus(async, false);
}
HRESULT STDMETHODCALLTYPE net_verify_certificate(void*, void*, const void*) { return kNoNetwork; }
HRESULT STDMETHODCALLTYPE net_connectivity_hint(void*, ConnectivityHint* hint) {
    if (!hint) return E_POINTER;
    *hint = {};
    hint->connectivity_level = 1;   // None: the stack is up, nothing is reachable
    hint->network_initialized = true;
    return S_OK;
}

using PortChangedCallback = void(CALLBACK*)(void* context, uint16_t port);
using HintChangedCallback = void(CALLBACK*)(void* context, const ConnectivityHint* hint);

struct NetRegistration {
    bool running = false;
    bool unregistered = false;
};
std::mutex g_net_mu;
std::condition_variable g_net_cv;
std::unordered_map<uint64_t, NetRegistration> g_net_registrations;
std::atomic<uint64_t> g_net_tokens{0};

void notify(PortChangedCallback callback, void* context) {
    uint16_t port = 0;
    net_query_port(nullptr, &port);
    callback(context, port);
}
void notify(HintChangedCallback callback, void* context) {
    ConnectivityHint hint{};
    net_connectivity_hint(nullptr, &hint);
    callback(context, &hint);
}

template <class Callback> struct InitialNotification {
    uint64_t token;
    void* context;
    Callback callback;

    static void CALLBACK run(void* self, bool canceled) {
        std::unique_ptr<InitialNotification> n(static_cast<InitialNotification*>(self));
        {
            std::lock_guard lock(g_net_mu);
            const auto it = g_net_registrations.find(n->token);
            if (canceled || it == g_net_registrations.end()) return;
            it->second.running = true;
        }
        notify(n->callback, n->context);
        {
            std::lock_guard lock(g_net_mu);
            const auto it = g_net_registrations.find(n->token);
            if (it->second.unregistered) g_net_registrations.erase(it);
            else it->second.running = false;
        }
        g_net_cv.notify_all();
    }
};

template <class Callback>
HRESULT STDMETHODCALLTYPE net_register(void*, XTaskQueueHandle queue, void* context, Callback callback,
                                       XTaskQueueRegistrationToken* token) {
    if (!callback) return E_INVALIDARG;
    if (!token) return E_POINTER;
    XTaskQueueHandle process_queue = nullptr;
    if (!queue) {
        if (!XTaskQueueGetCurrentProcessTaskQueue(&process_queue)) return HRESULT_FROM_WIN32(ERROR_NO_TASK_QUEUE);
        queue = process_queue;
    }
    const uint64_t id = g_net_tokens.fetch_add(1) + 1;
    {
        std::lock_guard lock(g_net_mu);
        g_net_registrations.emplace(id, NetRegistration{});
    }
    token->token = id;
    auto* initial = new InitialNotification<Callback>{id, context, callback};
    const HRESULT hr = XTaskQueueSubmitCallback(queue, XTaskQueuePort::Completion, initial,
                                                &InitialNotification<Callback>::run);
    if (process_queue) XTaskQueueCloseHandle(process_queue);
    if (FAILED(hr)) {
        delete initial;
        token->token = 0;
        std::lock_guard lock(g_net_mu);
        g_net_registrations.erase(id);
    }
    return hr;
}
bool STDMETHODCALLTYPE net_unregister(void*, XTaskQueueRegistrationToken token, bool wait) {
    std::unique_lock lock(g_net_mu);
    const auto it = g_net_registrations.find(token.token);
    if (it == g_net_registrations.end()) return true;
    if (!it->second.running) {
        g_net_registrations.erase(it);
        return true;
    }
    it->second.unregistered = true;
    if (wait) g_net_cv.wait(lock, [&] { return !g_net_registrations.contains(token.token); });
    return wait;
}
HRESULT STDMETHODCALLTYPE net_not_implemented(void*) { return E_NOTIMPL; }

#define NM(fn) reinterpret_cast<const void*>(&fn)
const void* const kNetworkingVtable[] = {
    NM(object_query_interface), NM(object_add_ref), NM(object_release),
    NM(net_query_port),            // 3  XNetworkingQueryPreferredLocalUdpMultiplayerPort
    NM(net_query_port_async),      // 4  ...Async
    NM(net_query_port_result),     // 5  ...AsyncResult
    NM(net_register<PortChangedCallback>),  // 6  XNetworkingRegisterPreferredLocalUdpMultiplayerPortChanged
    NM(net_unregister),            // 7  XNetworkingUnregisterPreferredLocalUdpMultiplayerPortChanged
    NM(net_security_async),        // 8  XNetworkingQuerySecurityInformationForUrlAsync
    NM(net_security_result_size),  // 9  ...AsyncResultSize
    NM(net_security_result),       // 10 ...AsyncResult
    NM(net_security_async),        // 11 XNetworkingQuerySecurityInformationForUrlUtf16Async (XCurl)
    NM(net_security_result_size),  // 12 ...Utf16AsyncResultSize (XCurl)
    NM(net_security_result),       // 13 ...Utf16AsyncResult (XCurl)
    NM(net_verify_certificate),    // 14 XNetworkingVerifyServerCertificate (XCurl)
    NM(net_connectivity_hint),     // 15 XNetworkingGetConnectivityHint (XCurl)
    NM(net_register<HintChangedCallback>),  // 16 XNetworkingRegisterConnectivityHintChanged (XCurl)
    NM(net_unregister),            // 17 XNetworkingUnregisterConnectivityHintChanged (XCurl)
    NM(net_not_implemented),       // 18 XNetworkingQueryConfigurationSetting (IXNetworkingImpl2)
    NM(net_not_implemented),       // 19 XNetworkingSetConfigurationSetting
    NM(net_not_implemented),       // 20 XNetworkingQueryStatistics
};
#undef NM
static_assert(std::size(kNetworkingVtable) == 21, "IUnknown + IXNetworkingImpl's 15 + IXNetworkingImpl2's 3");

// ---- XPersistentLocalStorageImpl (IXPersistentLocalStorageImpl .. 3), a folder beside the overlay ----
//
// The package declares persistent local storage (MicrosoftGame.Config, PersistentLocalStorage, 2048 MB), so the title
// can ask where it may keep its cache. Slots 3-7 follow Microsoft's XPersistentLocalStorage reference and the ids and
// layout in the public xgameruntime-docs and Wine IDL (facts only): GetPathSize, GetPath, GetSpaceInfo,
// PromptUserForSpaceAsync and PromptUserForSpaceResult; IXPersistentLocalStorageImpl3 adds MountForPackage at slot 8.
// The storage is <overlay>\.gta6rt-persistent, so the package stays read-only, --overlay moves it with everything
// else a title writes, and GAME: does not see it. Offline there is nobody to ask for more room: a prompt for space the
// quota can hold succeeds, one it cannot fails with ERROR_DISK_FULL.

constexpr GUID kPersistentStorageClass = {0xf4faf4d4, 0x2d04, 0x4fce, {0xb3, 0xe0, 0x47, 0x4a, 0x71, 0x3a, 0x3e, 0x84}};
constexpr GUID kPersistentStorage2 = {0xd29411df, 0x0794, 0x4553, {0x8b, 0x27, 0x95, 0xfc, 0x02, 0xd0, 0xf7, 0x5d}};
constexpr GUID kPersistentStorage3 = {0x41a4e10c, 0x5a7e, 0x41d9, {0x8c, 0x37, 0x37, 0xbd, 0xe6, 0x2a, 0x07, 0xd6}};
constexpr uint64_t kPersistentQuotaBytes = 2048ull * 1024 * 1024;  // MicrosoftGame.Config of package 1.0.2.1

// XPersistentLocalStorageSpaceInfo: four 64-bit counts.
struct PersistentSpaceInfo {
    uint64_t available_free_bytes;  // what the title can still write: the quota left, capped by the host volume
    uint64_t total_free_bytes;      // the quota left
    uint64_t used_bytes;
    uint64_t total_bytes;           // the quota
};
static_assert(sizeof(PersistentSpaceInfo) == 32, "XPersistentLocalStorageSpaceInfo is four UINT64");

// The storage folder, created on first use. Empty when there is no overlay (no place to keep files) or the folder
// cannot be created.
std::filesystem::path persistent_storage_dir() {
    const std::filesystem::path overlay = rt::fs::overlay_dir();
    if (overlay.empty()) return {};
    std::filesystem::path dir = overlay / rt::fs::kPersistentStorageDir;
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    return ec ? std::filesystem::path{} : dir;
}

std::string utf8_of(const std::filesystem::path& path) {
    const std::wstring& wide = path.native();
    const int n = WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), static_cast<int>(wide.size()), nullptr, 0, nullptr, nullptr);
    if (n <= 0) return {};
    std::string utf8(static_cast<size_t>(n), '\0');
    WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), static_cast<int>(wide.size()), utf8.data(), n, nullptr, nullptr);
    return utf8;
}

uint64_t folder_bytes(const std::filesystem::path& dir) {
    uint64_t total = 0;
    std::error_code ec;
    for (std::filesystem::recursive_directory_iterator it(dir, std::filesystem::directory_options::skip_permission_denied, ec), end;
         !ec && it != end; it.increment(ec)) {
        std::error_code file_ec;
        if (!it->is_regular_file(file_ec) || file_ec) continue;
        const uintmax_t size = it->file_size(file_ec);
        if (!file_ec) total += size;
    }
    return total;
}

HRESULT STDMETHODCALLTYPE pls_get_path_size(void*, size_t* path_size) {
    if (!path_size) return E_POINTER;
    *path_size = 0;
    const std::string path = utf8_of(persistent_storage_dir());
    if (path.empty()) return HRESULT_FROM_WIN32(ERROR_PATH_NOT_FOUND);
    *path_size = path.size() + 1;  // with the terminating NUL
    return S_OK;
}

HRESULT STDMETHODCALLTYPE pls_get_path(void*, size_t path_size, char* path, size_t* path_used) {
    if (path_used) *path_used = 0;
    if (!path) return E_POINTER;
    const std::string folder = utf8_of(persistent_storage_dir());
    if (folder.empty()) return HRESULT_FROM_WIN32(ERROR_PATH_NOT_FOUND);
    if (path_size < folder.size() + 1) return HRESULT_FROM_WIN32(ERROR_INSUFFICIENT_BUFFER);
    std::memcpy(path, folder.c_str(), folder.size() + 1);
    if (path_used) *path_used = folder.size() + 1;
    return S_OK;
}

HRESULT STDMETHODCALLTYPE pls_get_space_info(void*, PersistentSpaceInfo* info) {
    if (!info) return E_POINTER;
    *info = {};
    const std::filesystem::path dir = persistent_storage_dir();
    if (dir.empty()) return HRESULT_FROM_WIN32(ERROR_PATH_NOT_FOUND);
    const uint64_t used = folder_bytes(dir);
    const uint64_t left = used < kPersistentQuotaBytes ? kPersistentQuotaBytes - used : 0;
    std::error_code ec;
    const std::filesystem::space_info host = std::filesystem::space(dir, ec);
    info->total_bytes = kPersistentQuotaBytes;
    info->used_bytes = used;
    info->total_free_bytes = left;
    info->available_free_bytes = ec || left < host.available ? left : host.available;
    return S_OK;
}

HRESULT STDMETHODCALLTYPE pls_prompt_async(void*, uint64_t requested_bytes, XAsyncBlock* async) {
    if (!async) return E_POINTER;
    const std::filesystem::path dir = persistent_storage_dir();
    const bool fits = !dir.empty() && requested_bytes <= kPersistentQuotaBytes - (std::min)(folder_bytes(dir), kPersistentQuotaBytes);
    if (fits) return XAsyncRun(async, [](XAsyncBlock*) -> HRESULT { return S_OK; });
    return XAsyncRun(async, [](XAsyncBlock*) -> HRESULT { return HRESULT_FROM_WIN32(ERROR_DISK_FULL); });
}

HRESULT STDMETHODCALLTYPE pls_prompt_result(void*, XAsyncBlock* async) {
    if (!async) return E_POINTER;
    return XAsyncGetStatus(async, false);
}

// IXPersistentLocalStorageImpl3's package mount has nothing to mount offline; the handle is left empty.
HRESULT STDMETHODCALLTYPE pls_mount_for_package(void*, const char*, void** mount_handle) {
    if (!mount_handle) return E_POINTER;
    *mount_handle = nullptr;
    return E_NOTIMPL;
}

#define PLS(fn) reinterpret_cast<const void*>(&fn)
const void* const kPersistentStorageVtable[] = {
    PLS(object_query_interface), PLS(object_add_ref), PLS(object_release),
    PLS(pls_get_path_size),      // 3 XPersistentLocalStorageGetPathSize
    PLS(pls_get_path),           // 4 XPersistentLocalStorageGetPath
    PLS(pls_get_space_info),     // 5 XPersistentLocalStorageGetSpaceInfo
    PLS(pls_prompt_async),       // 6 XPersistentLocalStoragePromptUserForSpaceAsync
    PLS(pls_prompt_result),      // 7 XPersistentLocalStoragePromptUserForSpaceResult
    PLS(pls_mount_for_package),  // 8 XPersistentLocalStorageMountForPackage (IXPersistentLocalStorageImpl3)
};
#undef PLS
static_assert(std::size(kPersistentStorageVtable) == 9, "IUnknown + IXPersistentLocalStorageImpl's 5 + Impl3's 1");

// The ids xgameruntime-docs and the xgameruntime IDL give: XThreadingImpl's and XErrorImpl's class id is also their
// one interface's; IXNetworkingImpl's id is the class id, and IXNetworkingImpl2 derives from it.
const GUID* const kNetworkingIids[] = {&kNetworking2, nullptr};
RuntimeObject g_threading{kThreadingVtable, &kThreadingClass, nullptr};
RuntimeObject g_error{kErrorVtable, &kErrorClass, nullptr};
RuntimeObject g_networking{kNetworkingVtable, &kNetworkingClass, kNetworkingIids};
const GUID* const kPersistentStorageIids[] = {&kPersistentStorage2, &kPersistentStorage3, nullptr};
RuntimeObject g_persistent_storage{kPersistentStorageVtable, &kPersistentStorageClass, kPersistentStorageIids};

RuntimeObject& system_object();   // XSystemImpl, defined after the XSystem functions it serves
RuntimeObject& package_object();  // XPackageImpl, after the XPackage functions
RuntimeObject& game_object();     // XGameImpl, after XPackageImpl
RuntimeObject* const* measured_objects();  // XGameRuntimeFeatureImpl and XSystemAnalyticsImpl, after XGameImpl

// The object this file hands out for a class id, or null when the class isn't one of this file's.
RuntimeObject* find_object(const GUID& class_id) {
    for (RuntimeObject* o : {&g_threading, &g_error, &g_networking, &g_persistent_storage, &system_object(),
                             &package_object(), &game_object()})
        if (class_id == *o->class_id) return o;
    for (RuntimeObject* const* o = measured_objects(); *o; ++o)
        if (class_id == *(*o)->class_id) return *o;
    return nullptr;
}

// QueryApiImpl's answer for a served class: true with *hr set (and *out on success) when this file or one of the
// other xgameruntime_*.cpp files serves the class, false when none does.
bool query_served(const GUID& class_id, REFIID interface_id, void** out, HRESULT* hr) {
    if (RuntimeObject* o = find_object(class_id)) {
        *hr = object_query_interface(o, interface_id, out);
        return true;
    }
    return rt::shims::xgameruntime_user_query(class_id, interface_id, out, hr) ||       // xgameruntime_user.cpp
           rt::shims::xgameruntime_gamesave_query(class_id, interface_id, out, hr) ||   // xgameruntime_gamesave.cpp
           rt::shims::xgameruntime_store_query(class_id, interface_id, out, hr) ||      // xgameruntime_store.cpp
           rt::shims::xgameruntime_offline_query(class_id, interface_id, out, hr);      // xgameruntime_offline.cpp
}

// Whether QueryApiImpl serves the class, whichever file serves it (the reference taken to find out is released).
bool class_served(const GUID& class_id) {
    void* obj = nullptr;
    HRESULT hr = S_OK;
    if (!query_served(class_id, kIUnknown, &obj, &hr)) return false;
    if (obj) (*static_cast<ULONG(STDMETHODCALLTYPE* const* const*)(void*)>(obj))[2](obj);   // Release
    return true;
}

HRESULT WINAPI QueryApiImpl(const GUID* class_id, REFIID interface_id, void** out) {
    if (out) *out = nullptr;
    if (class_id && out) {
        HRESULT hr = S_OK;
        if (query_served(*class_id, interface_id, out, &hr)) return hr;
    }
    char cls[40], iid[40];
    print_guid(cls, class_id);
    print_guid(iid, &interface_id);
    static std::atomic<int> s_logged{0};
    if (s_logged.fetch_add(1) < 256)   // one line per request, bounded
        std::fprintf(stderr, "[rt] xgameruntime: QueryApiImpl(class %s, interface %s): not provided yet\n", cls, iid);
    return kClassNotProvided;
}

// gameplatformservices.dll: XCurl.dll's other way into the runtime (docs/runtime_imports.md). It reads
// HKLM\Microsoft\GamingServices ServiceRetryTimeout, LoadLibraryExW's this module from System32, calls
// GetApiVersionImpl through the CFG dispatch with two pointers to 64-bit locals (the result unread), compares them
// with the pair it saw last and keeps them, and then GetProcAddress'es QueryApiImpl. Served by the shims, so that call
// never reaches a PC's Gaming Services runtime (which would bypass these offline classes, and could go online) and
// never fails because the PC has none. The two values only have to be stable: the offline runtime has one version.
constexpr uint64_t kOfflineApiVersion = 1;
HRESULT WINAPI GetApiVersionImpl(uint64_t* first, uint64_t* second) {
    if (first) *first = kOfflineApiVersion;
    if (second) *second = kOfflineApiVersion;
    return S_OK;
}

// HRESULT XGameRuntimeInitialize(). Reference counted, as in the GDK: each call needs an XGameRuntimeUninitialize.
HRESULT __stdcall XGameRuntimeInitialize() {
    if (g_runtime_refs.fetch_add(1) == 0) std::fprintf(stderr, "[rt] xgameruntime: initialized (offline)\n");
    return S_OK;
}

// void XGameRuntimeUninitialize()
void __stdcall XGameRuntimeUninitialize() {
    int prev = g_runtime_refs.load();
    while (prev > 0 && !g_runtime_refs.compare_exchange_weak(prev, prev - 1)) {}
}

// ---- XSystem ----

// XSystemDeviceType, in the order of the public documentation: Unknown, Pc, XboxOne, XboxOneS, XboxOneX,
// XboxOneXDevkit, XboxScarlettLockhart, XboxScarlettAnaconda, XboxScarlettDevkit.
enum class XSystemDeviceType : uint32_t {
    Unknown = 0, Pc = 1, XboxOne = 2, XboxOneS = 3, XboxOneX = 4, XboxOneXDevkit = 5,
    XboxScarlettLockhart = 6, XboxScarlettAnaconda = 7, XboxScarlettDevkit = 8,
};

// XSystemDeviceType XSystemGetDeviceType(). The package is the Xbox Series build, so report Series X (Anaconda).
// LEONIDA_DEVICE_TYPE=<0..8> overrides it, for trying other code paths on day one.
uint32_t __stdcall XSystemGetDeviceType() {
    char s[16];
    const DWORD n = GetEnvironmentVariableA("LEONIDA_DEVICE_TYPE", s, sizeof(s));
    if (n > 0 && n < sizeof(s)) {
        int v = std::atoi(s);
        if (v >= 0 && v <= 8) return static_cast<uint32_t>(v);
    }
    return static_cast<uint32_t>(XSystemDeviceType::XboxScarlettAnaconda);
}

// HRESULT XSystemGetConsoleId(size_t consoleIdSize, char* consoleId, size_t* consoleIdUsed).
// XSystemConsoleIdBytes is 39 (including the terminator). An offline PC has no console id, so this returns a
// fixed placeholder of that length.
constexpr size_t kConsoleIdBytes = 39;
constexpr char kConsoleId[kConsoleIdBytes] = "00000000.00000000.00000000.00000000.00";
static_assert(sizeof(kConsoleId) == kConsoleIdBytes, "placeholder console id must fill XSystemConsoleIdBytes");

HRESULT __stdcall XSystemGetConsoleId(size_t size, char* out, size_t* used) {
    if (!out) return E_INVALIDARG;
    if (size < kConsoleIdBytes) return HRESULT_FROM_WIN32(ERROR_INSUFFICIENT_BUFFER);
    std::memcpy(out, kConsoleId, kConsoleIdBytes);
    if (used) *used = kConsoleIdBytes;
    return S_OK;
}

// uint64_t XSystemGetAppMemoryUsageLimit(). Reports total physical RAM, or 12 GB fallback for Series X.
uint64_t __stdcall XSystemGetAppMemoryUsageLimit() {
    MEMORYSTATUSEX ms{sizeof(ms)};
    if (GlobalMemoryStatusEx(&ms)) return ms.ullTotalPhys;
    return 12ULL * 1024 * 1024 * 1024;
}

// uint64_t XSystemGetAppMemoryUsage()
uint64_t __stdcall XSystemGetAppMemoryUsage() {
    return 0;
}

// ---- XSystemImpl (asked for as IXSystemImpl5), offline ----
//
// libHttpClient.GDK.dll asks QueryApiImpl for XSystemImpl with IXSystemImpl5's id and calls slots 3-8, each with the
// argument shape of the function at that position of xgameruntime-docs' IXSystemImpl / IXSystemImpl3 /
// IXSystemImpl4 lists (IXSystemImpl2 and 5 add nothing known): three (size, buffer, used) id queries, then
// XSystemHandleTrack (two pointers), XSystemIsHandleValid (a handle) and XSystemAllowFullDownloadBandwidth (a bool).
// docs/runtime_imports.md has the measurement.

constexpr GUID kSystemClass = {0xe349bd1a, 0xfc20, 0x4e40, {0xb9, 0x9c, 0x41, 0x78, 0xcc, 0x6b, 0x40, 0x9f}};
// IXSystemImpl's id is the class id; IXSystemImpl5 : IXSystemImpl4 : IXSystemImpl3 : IXSystemImpl2 : IXSystemImpl
// (xgameruntime's include/xsystem.idl). One vtable serves all five.
constexpr GUID kSystem2 = {0x6fd71f09, 0x7513, 0x49f0, {0x89, 0xbc, 0xbf, 0xaf, 0x5d, 0xf6, 0xf8, 0x52}};
constexpr GUID kSystem3 = {0x67ce4bfc, 0xb1d1, 0x4ac7, {0xbc, 0x3a, 0xcb, 0x92, 0x19, 0xa9, 0x7a, 0x85}};
constexpr GUID kSystem4 = {0xdadc2895, 0x34b0, 0x4ef5, {0xa8, 0x3e, 0x45, 0x11, 0x4d, 0x62, 0x9b, 0x80}};
constexpr GUID kSystem5 = {0x1861cf2e, 0xe18b, 0x4834, {0xa9, 0xf5, 0xb4, 0xa4, 0xe6, 0xef, 0xb4, 0xcf}};
const GUID* const kSystemIids[] = {&kSystem2, &kSystem3, &kSystem4, &kSystem5, nullptr};

// A fixed string into a (size, buffer, used) out parameter, as the XSystem id queries take one. `minimum` is the
// buffer size the API requires (its *Bytes constant), which can be larger than the string.
HRESULT copy_id(const char* id, size_t bytes, size_t minimum, size_t size, char* out, size_t* used) {
    if (!out) return E_INVALIDARG;
    if (size < minimum || size < bytes) return HRESULT_FROM_WIN32(ERROR_INSUFFICIENT_BUFFER);
    std::memcpy(out, id, bytes);
    if (used) *used = bytes;
    return S_OK;
}

// XSystemXboxLiveSandboxIdMaxBytes is 16, and Microsoft's contract asks for a buffer at least that large (smaller
// ones get ERROR_INSUFFICIENT_BUFFER, whatever the id's length); an offline title runs in the retail sandbox.
constexpr size_t kSandboxIdMaxBytes = 16;
constexpr char kSandboxId[] = "RETAIL";
static_assert(sizeof(kSandboxId) <= kSandboxIdMaxBytes, "fits XSystemXboxLiveSandboxIdMaxBytes");
// XSystemAppSpecificDeviceIdBytes is 45 (including the terminator): a fixed placeholder of that length, like the
// console id.
constexpr char kAppDeviceId[] = "00000000000000000000000000000000000000000000";
static_assert(sizeof(kAppDeviceId) == 45, "fills XSystemAppSpecificDeviceIdBytes");

HRESULT STDMETHODCALLTYPE sys_console_id(void*, size_t size, char* out, size_t* used) {
    return XSystemGetConsoleId(size, out, used);
}
HRESULT STDMETHODCALLTYPE sys_sandbox_id(void*, size_t size, char* out, size_t* used) {
    return copy_id(kSandboxId, sizeof(kSandboxId), kSandboxIdMaxBytes, size, out, used);
}
HRESULT STDMETHODCALLTYPE sys_app_device_id(void*, size_t size, char* out, size_t* used) {
    return copy_id(kAppDeviceId, sizeof(kAppDeviceId), sizeof(kAppDeviceId), size, out, used);
}
// XSystemHandleTrack(callback, context): the GDK reports handle creation and destruction to it. Offline nothing is
// tracked, so the callback is accepted and never called.
HRESULT STDMETHODCALLTYPE sys_handle_track(void*, void*, void*) { return S_OK; }
// bool XSystemIsHandleValid(XSystemHandle) and void XSystemAllowFullDownloadBandwidth(bool), as Microsoft documents
// them: a bool, not an HRESULT (S_OK would read as false).
bool STDMETHODCALLTYPE sys_is_handle_valid(void*, void* handle) { return handle != nullptr; }
void STDMETHODCALLTYPE sys_allow_full_download_bandwidth(void*, bool) {}

const void* const kSystemVtable[] = {
    reinterpret_cast<const void*>(&object_query_interface), reinterpret_cast<const void*>(&object_add_ref),
    reinterpret_cast<const void*>(&object_release),
    reinterpret_cast<const void*>(&sys_console_id),                     // 3 XSystemGetConsoleId
    reinterpret_cast<const void*>(&sys_sandbox_id),                     // 4 XSystemGetXboxLiveSandboxId
    reinterpret_cast<const void*>(&sys_app_device_id),                  // 5 XSystemGetAppSpecificDeviceId
    reinterpret_cast<const void*>(&sys_handle_track),                   // 6 XSystemHandleTrack (IXSystemImpl3)
    reinterpret_cast<const void*>(&sys_is_handle_valid),                // 7 XSystemIsHandleValid
    reinterpret_cast<const void*>(&sys_allow_full_download_bandwidth),  // 8 XSystemAllowFullDownloadBandwidth (4)
};
static_assert(std::size(kSystemVtable) == 9, "IUnknown + the six methods IXSystemImpl5 is known to have");

RuntimeObject& system_object() {
    static RuntimeObject object{kSystemVtable, &kSystemClass, kSystemIids};
    return object;
}

// XVersion: 4-part version structure from GDK XGameRuntimeTypes.h (8 bytes)
struct alignas(8) XVersion {
    uint16_t major;
    uint16_t minor;
    uint16_t build;
    uint16_t revision;
};
static_assert(sizeof(XVersion) == 8, "XVersion must be 8 bytes");

// XSystemAnalyticsInfo: hardware and OS version telemetry (144 bytes)
struct XSystemAnalyticsInfo {
    XVersion osVersion;
    XVersion hostingOsVersion;
    char family[64];
    char form[64];
};
static_assert(sizeof(XSystemAnalyticsInfo) == 144, "XSystemAnalyticsInfo must be 144 bytes");

static void copy_string_padded(char* dst, size_t dst_cap, const char* src) {
    if (!dst || dst_cap == 0) return;
    std::memset(dst, 0, dst_cap);
    if (!src) return;
    const size_t len = std::strlen(src);
    const size_t copy_len = len < dst_cap - 1 ? len : dst_cap - 1;
    std::memcpy(dst, src, copy_len);
}

// XSystemAnalyticsInfo XSystemGetAnalyticsInfo()
// Returned by value per Microsoft GDK specification.
XSystemAnalyticsInfo __stdcall XSystemGetAnalyticsInfo() {
    XSystemAnalyticsInfo info{};
    info.osVersion = {10, 0, 22000, 0};
    info.hostingOsVersion = {10, 0, 22000, 0};
    copy_string_padded(info.family, sizeof(info.family), "Xbox");

    const uint32_t dt = XSystemGetDeviceType();
    if (dt == static_cast<uint32_t>(XSystemDeviceType::XboxScarlettLockhart)) {
        copy_string_padded(info.form, sizeof(info.form), "Xbox Series S");
    } else {
        copy_string_padded(info.form, sizeof(info.form), "Xbox Series X");
    }

    char form_override[64];
    const DWORD n = GetEnvironmentVariableA("LEONIDA_ANALYTICS_FORM", form_override, sizeof(form_override));
    if (n > 0 && n < sizeof(form_override)) {
        copy_string_padded(info.form, sizeof(info.form), form_override);
    }
    return info;
}

// ---- XPackage ----

constexpr size_t XPACKAGE_IDENTIFIER_MAX_LENGTH = 33;
constexpr char kDefaultPackageIdentifier[XPACKAGE_IDENTIFIER_MAX_LENGTH] = "LeonidaPackageIdentifier00000000";
static_assert(sizeof(kDefaultPackageIdentifier) == XPACKAGE_IDENTIFIER_MAX_LENGTH,
              "default package identifier must match XPACKAGE_IDENTIFIER_MAX_LENGTH");

// HRESULT XPackageGetCurrentProcessPackageIdentifier(size_t bufferSize, char* buffer).
// Retrieves the package identifier string for the current process.
HRESULT __stdcall XPackageGetCurrentProcessPackageIdentifier(size_t bufferSize, char* buffer) {
    if (!buffer || bufferSize == 0) return E_INVALIDARG;

    char id[XPACKAGE_IDENTIFIER_MAX_LENGTH] = {};
    const DWORD n = GetEnvironmentVariableA("LEONIDA_PACKAGE_ID", id, sizeof(id));
    const char* selected = (n > 0 && n < sizeof(id)) ? id : kDefaultPackageIdentifier;
    const size_t len = std::strlen(selected);

    if (bufferSize < len + 1) return HRESULT_FROM_WIN32(ERROR_INSUFFICIENT_BUFFER);
    std::memcpy(buffer, selected, len + 1);
    return S_OK;
}

// XPackageMountHandle: opaque 64-bit mount handle per GDK XPackage.h
struct XPackageMountHandle {
    uint64_t handle;
};

static std::mutex s_mount_mutex;
static std::atomic<uint64_t> s_next_mount_id{1};
static std::unordered_map<uint64_t, std::string> s_active_mounts;

// Resolves the installation path for a specific content package identifier.
// GDK specifies that XPackageMount is only for content packages; mounting the main game returns E_ACCESSDENIED.
// Returns HRESULT_FROM_WIN32(ERROR_NOT_FOUND) if the package identifier is not installed.
static HRESULT resolve_package_path(const char* packageIdentifier, char* out, size_t out_cap) {
    if (!packageIdentifier || !out || out_cap == 0) return E_INVALIDARG;

    // Check against configured process/game package identifier; mounting the game itself is denied
    char game_id[XPACKAGE_IDENTIFIER_MAX_LENGTH] = {};
    const DWORD n = GetEnvironmentVariableA("LEONIDA_PACKAGE_ID", game_id, sizeof(game_id));
    const char* process_id = (n > 0 && n < sizeof(game_id)) ? game_id : kDefaultPackageIdentifier;

    if (std::strcmp(packageIdentifier, process_id) == 0) {
        return E_ACCESSDENIED;
    }

    // Check optional installed content packages mapping: "id1=path1;id2=path2"
    char mapping[1024] = {};
    const DWORD map_len = GetEnvironmentVariableA("LEONIDA_CONTENT_PACKAGES", mapping, sizeof(mapping));
    if (map_len > 0 && map_len < sizeof(mapping)) {
        char* token = mapping;
        while (token && *token) {
            char* sep = std::strchr(token, ';');
            if (sep) *sep = '\0';
            char* eq = std::strchr(token, '=');
            if (eq) {
                *eq = '\0';
                if (std::strcmp(token, packageIdentifier) == 0) {
                    const char* val = eq + 1;
                    const size_t vlen = std::strlen(val);
                    if (vlen + 1 > out_cap) return HRESULT_FROM_WIN32(ERROR_INSUFFICIENT_BUFFER);
                    std::memcpy(out, val, vlen + 1);
                    return S_OK;
                }
            }
            token = sep ? sep + 1 : nullptr;
        }
    }

    return HRESULT_FROM_WIN32(ERROR_NOT_FOUND);
}

// HRESULT XPackageMount(const char* packageIdentifier, XPackageMountHandle* mount)
// Mounts an installed content package given its identifier and yields a valid mount handle.
// Returns ERROR_NOT_FOUND if the package identifier is not recognized/installed.
HRESULT __stdcall XPackageMount(const char* packageIdentifier, XPackageMountHandle* mount) {
    if (!packageIdentifier || !mount) return E_INVALIDARG;
    mount->handle = 0;

    char buf[MAX_PATH] = {};
    const HRESULT hr = resolve_package_path(packageIdentifier, buf, sizeof(buf));
    if (FAILED(hr)) return hr;

    const uint64_t id = s_next_mount_id.fetch_add(1);
    {
        std::lock_guard lock(s_mount_mutex);
        s_active_mounts[id] = buf;
    }
    mount->handle = id;
    return S_OK;
}

// void XPackageCloseMountHandle(XPackageMountHandle mount)
// Closes a previously mounted package handle.
void __stdcall XPackageCloseMountHandle(XPackageMountHandle mount) {
    if (mount.handle == 0) return;
    std::lock_guard lock(s_mount_mutex);
    s_active_mounts.erase(mount.handle);
}

// HRESULT XPackageGetMountPathSize(XPackageMountHandle mount, size_t* pathSize).
// Retrieves the size of the buffer needed to store the mounted package path, including the null terminator.
HRESULT __stdcall XPackageGetMountPathSize(XPackageMountHandle mount, size_t* pathSize) {
    if (mount.handle == 0 || !pathSize) return E_INVALIDARG;

    std::string path;
    {
        std::lock_guard lock(s_mount_mutex);
        auto it = s_active_mounts.find(mount.handle);
        if (it == s_active_mounts.end()) return HRESULT_FROM_WIN32(ERROR_NOT_FOUND);
        path = it->second;
    }

    *pathSize = path.size() + 1;
    return S_OK;
}

// HRESULT XPackageGetMountPath(XPackageMountHandle mount, size_t pathSize, char* path).
// Retrieves the installed filesystem path for the mounted package.
HRESULT __stdcall XPackageGetMountPath(XPackageMountHandle mount, size_t pathSize, char* path) {
    if (mount.handle == 0 || !path || pathSize == 0) return E_INVALIDARG;

    std::string mount_path;
    {
        std::lock_guard lock(s_mount_mutex);
        auto it = s_active_mounts.find(mount.handle);
        if (it == s_active_mounts.end()) return HRESULT_FROM_WIN32(ERROR_NOT_FOUND);
        mount_path = it->second;
    }

    if (pathSize < mount_path.size() + 1) return HRESULT_FROM_WIN32(ERROR_INSUFFICIENT_BUFFER);
    std::memcpy(path, mount_path.c_str(), mount_path.size() + 1);
    return S_OK;
}

// bool XPackageIsPackagedProcess().
// Returns true on Xbox Series console. LEONIDA_PACKAGED=0 overrides it.
bool __stdcall XPackageIsPackagedProcess() {
    char s[16];
    const DWORD n = GetEnvironmentVariableA("LEONIDA_PACKAGED", s, sizeof(s));
    if (n > 0 && n < sizeof(s)) {
        if (std::atoi(s) == 0) return false;
    }
    return true;
}

// ---- XPackageImpl (asked for as IXPackageImpl3), offline ----
//
// libHttpClient.GDK.dll asks QueryApiImpl for XPackageImpl with IXPackageImpl3's id through 30 thunks (slots 3-19,
// 22, 24-27, 30, 32-34, 37-40), each in the argument shape of the method at that position of xgameruntime-docs'
// IXPackageImpl (slots 3-34), IXPackageImpl2 (35-38) and IXPackageImpl3 (39-40) lists (docs/runtime_imports.md).
// The slots with an offline answer forward to the XPackage functions above (the package identifier, whether the
// process is packaged, the mount path and its size, the user's locale); every other slot logs its number once and
// returns E_NOTIMPL, so the first boot shows which ones the title needs.

constexpr GUID kPackageClass = {0xaf406016, 0xe850, 0x4aa8, {0xa8, 0x8d, 0x2f, 0x3d, 0xcb, 0x9d, 0xac, 0x7e}};
// IXPackageImpl3 : IXPackageImpl2 : IXPackageImpl (xgameruntime include/xpackage.idl); the class id is its own.
constexpr GUID kPackage1 = {0x3720de07, 0xe8e4, 0x44a3, {0xad, 0x32, 0xb3, 0x59, 0xe8, 0xad, 0xbe, 0x55}};
constexpr GUID kPackage2 = {0xf92d8712, 0x2b27, 0x4d8a, {0xbf, 0x01, 0x11, 0xa6, 0xf8, 0xe3, 0xeb, 0x42}};
constexpr GUID kPackage3 = {0xe2a4734b, 0x2f4a, 0x456d, {0xaa, 0x8f, 0xd0, 0x65, 0xe0, 0x4f, 0xb2, 0x09}};
const GUID* const kPackageIids[] = {&kPackage1, &kPackage2, &kPackage3, nullptr};

template <int Slot> HRESULT STDMETHODCALLTYPE package_not_implemented(void*) {
    static std::atomic<bool> s_logged{false};
    if (!s_logged.exchange(true))
        std::fprintf(stderr, "[rt] xgameruntime: XPackageImpl slot %d not implemented offline (E_NOTIMPL)\n", Slot);
    return E_NOTIMPL;
}
HRESULT STDMETHODCALLTYPE pkg_identifier(void*, size_t size, char* buffer) {
    return XPackageGetCurrentProcessPackageIdentifier(size, buffer);
}
bool STDMETHODCALLTYPE pkg_is_packaged(void*) { return XPackageIsPackagedProcess(); }
// The locales the package declares: <Resources><Resource Language="xx-YY"/> in MicrosoftGame.Config, read from
// the game folder (LEONIDA_GAME_DIR, else --game-dir). Empty when there is no config or it declares none.
// The package's MicrosoftGame.Config as text, from the game folder (LEONIDA_GAME_DIR, else --game-dir); empty when
// there is no folder or no config. Wide on purpose: a game folder named outside the ANSI code page must reach the file
// intact.
std::string package_config() {
    wchar_t env[MAX_PATH] = {};
    const DWORD n = GetEnvironmentVariableW(L"LEONIDA_GAME_DIR", env, MAX_PATH);
    const std::filesystem::path dir = (n > 0 && n < MAX_PATH) ? std::filesystem::path(env) : rt::fs::game_dir();
    if (dir.empty()) return {};
    std::ifstream f(dir / L"MicrosoftGame.Config", std::ios::binary);
    if (!f) return {};
    return std::string{std::istreambuf_iterator<char>(f), {}};
}

std::vector<std::string> package_locales() {
    std::vector<std::string> out;
    const std::string xml = package_config();
    if (xml.empty()) return out;
    for (size_t at = 0; (at = xml.find("<Resource", at)) != std::string::npos; ++at) {
        if (xml.compare(at, 10, "<Resources") == 0) continue;
        const size_t end = xml.find('>', at);
        const size_t key = xml.find("Language=", at);
        if (end == std::string::npos || key == std::string::npos || key + 10 > end) continue;
        const char quote = xml[key + 9];
        const size_t close = xml.find(quote, key + 10);
        if ((quote == '"' || quote == '\'') && close != std::string::npos && close < end)
            out.push_back(xml.substr(key + 10, close - key - 10));
    }
    return out;
}

// XPackageGetUserLocale(size_t localeSize, char* locale): of the locales the package declares, the one closest to
// the user's (the same name, else the same language, else the first declared). Microsoft's contract: "If no locale
// information is declared for the package, this function succeeds but does not return a user locale". The host's
// locale is never handed out as the package's.
HRESULT STDMETHODCALLTYPE pkg_user_locale(void*, size_t size, char* locale) {
    if (!locale || !size) return E_INVALIDARG;
    const std::vector<std::string> declared = package_locales();
    if (declared.empty()) return S_OK;
    wchar_t wide[LOCALE_NAME_MAX_LENGTH] = {};
    char user[LOCALE_NAME_MAX_LENGTH * 4] = "";
    if (GetUserDefaultLocaleName(wide, LOCALE_NAME_MAX_LENGTH) > 0)
        WideCharToMultiByte(CP_UTF8, 0, wide, -1, user, static_cast<int>(sizeof(user)), nullptr, nullptr);
    const auto language = [](const std::string& name) { return name.substr(0, name.find('-')); };
    const auto same = [](const std::string& a, const std::string& b) { return _stricmp(a.c_str(), b.c_str()) == 0; };
    const std::string* pick = &declared.front();
    bool exact = false;
    for (const std::string& d : declared)
        if (same(d, user)) { pick = &d; exact = true; break; }
    if (!exact)
        for (const std::string& d : declared)
            if (same(language(d), language(user))) { pick = &d; break; }
    if (size < pick->size() + 1) return HRESULT_FROM_WIN32(ERROR_INSUFFICIENT_BUFFER);
    std::memcpy(locale, pick->c_str(), pick->size() + 1);
    return S_OK;
}
// Slots 23-26 forward to the named exports' content-package mounts (XPackageMount and its handle registry), so the two
// ways in behave the same. XPackageMountHandle is one u64, passed in a register like the named exports take it.
HRESULT STDMETHODCALLTYPE pkg_mount(void*, const char* package_identifier, XPackageMountHandle* mount) {
    return XPackageMount(package_identifier, mount);
}
HRESULT STDMETHODCALLTYPE pkg_mount_path_size(void*, XPackageMountHandle mount, size_t* size) {
    return XPackageGetMountPathSize(mount, size);
}
HRESULT STDMETHODCALLTYPE pkg_mount_path(void*, XPackageMountHandle mount, size_t size, char* path) {
    return XPackageGetMountPath(mount, size, path);
}
void STDMETHODCALLTYPE pkg_close_mount(void*, XPackageMountHandle mount) { XPackageCloseMountHandle(mount); }

// XPackageInstallationProgress: totalBytes, installedBytes, launchBytes (u64), launchable, completed (bool).
struct InstallationProgress {
    uint64_t total_bytes, installed_bytes, launch_bytes;
    bool launchable, completed;
};
static_assert(sizeof(InstallationProgress) == 32, "XPackageInstallationProgress layout");
// void XPackageGetInstallationProgress(monitor, XPackageInstallationProgress*): an offline package is installed,
// launchable and completed, every byte present.
void STDMETHODCALLTYPE pkg_installation_progress(void*, void*, InstallationProgress* progress) {
    if (progress) *progress = {0, 0, 0, true, true};
}
// The bool methods with no offline work, as Microsoft documents them: XPackageUpdateInstallationMonitor is false
// (nothing changed), the two Unregister* calls are true (nothing pending), XPackageUninstallPackage is false (not
// uninstalled). A generic E_NOTIMPL stub would read as true in al.
bool STDMETHODCALLTYPE pkg_update_monitor(void*, void*) { return false; }
bool STDMETHODCALLTYPE pkg_unregister_progress(void*, void*, XTaskQueueRegistrationToken, bool) { return true; }
bool STDMETHODCALLTYPE pkg_unregister_installed(void*, XTaskQueueRegistrationToken, bool) { return true; }
bool STDMETHODCALLTYPE pkg_uninstall_package(void*, const char*) { return false; }

#define PN(n) reinterpret_cast<const void*>(&package_not_implemented<n>)
const void* const kPackageVtable[] = {
    reinterpret_cast<const void*>(&object_query_interface), reinterpret_cast<const void*>(&object_add_ref),
    reinterpret_cast<const void*>(&object_release),
    reinterpret_cast<const void*>(&pkg_identifier),   // 3  XPackageGetCurrentProcessPackageIdentifier
    reinterpret_cast<const void*>(&pkg_is_packaged),  // 4  XPackageIsPackagedProcess
    PN(5), PN(6),                                     // 5-6 installation monitor: create, close
    reinterpret_cast<const void*>(&pkg_installation_progress),  // 7  XPackageGetInstallationProgress (void, writes)
    reinterpret_cast<const void*>(&pkg_update_monitor),         // 8  XPackageUpdateInstallationMonitor (bool)
    PN(9),                                                      // 9  XPackageRegisterInstallationProgressChanged
    reinterpret_cast<const void*>(&pkg_unregister_progress),    // 10 ...UnregisterInstallationProgressChanged (bool)
    reinterpret_cast<const void*>(&pkg_user_locale),  // 11 XPackageGetUserLocale
    PN(12), PN(13), PN(14), PN(15), PN(16), PN(17),   // 12-19 chunks: find / enumerate availability, change order,
    PN(18), PN(19),                                   //       install (+ async, result), estimate size, uninstall
    PN(20), PN(21),                                   // 20-21 _unknown_ (older enumerate / register installed)
    reinterpret_cast<const void*>(&pkg_unregister_installed),  // 22 XPackageUnregisterPackageInstalled (bool)
    reinterpret_cast<const void*>(&pkg_mount),            // 23 XPackageMount
    reinterpret_cast<const void*>(&pkg_mount_path_size),  // 24 XPackageGetMountPathSize
    reinterpret_cast<const void*>(&pkg_mount_path),       // 25 XPackageGetMountPath
    reinterpret_cast<const void*>(&pkg_close_mount),      // 26 XPackageCloseMountHandle
    PN(27), PN(28), PN(29), PN(30), PN(31),           // 27 _unknown_, 28 EnumeratePackages, 29 RegisterPackageInstalled,
                                                      // 30 GetWriteStats, 31 _unknown_
    PN(32), PN(33),                                   // 32 UninstallUWPInstance, 33 EnumerateFeatures
    reinterpret_cast<const void*>(&pkg_uninstall_package),     // 34 XPackageUninstallPackage (bool)
    PN(35), PN(36), PN(37), PN(38),                   // IXPackageImpl2: EnumeratePackages, RegisterPackageInstalled,
                                                      //   MountWithUiAsync, MountWithUiResult
    PN(39), PN(40),                                   // IXPackageImpl3: EnumeratePackages, RegisterPackageInstalled
};
#undef PN
static_assert(std::size(kPackageVtable) == 41, "IUnknown + IXPackageImpl's 32 + IXPackageImpl2's 4 + IXPackageImpl3's 2");

RuntimeObject& package_object() {
    static RuntimeObject object{kPackageVtable, &kPackageClass, kPackageIids};
    return object;
}

// ---- XGameImpl (asked for as IXGameImpl3), offline ----
//
// IXGameImpl3 : IXGameImpl2 : IXGameImpl (xgameruntime include/xgame.idl); the class id is IXGameImpl's. Three
// methods after IUnknown: 3 XGameGetXboxTitleId (IXGameImpl), 4 XLaunchNewGame (IXGameImpl2), 5 XLaunchRestartOnCrash
// (IXGameImpl3). A retail Gen9 title asks QueryApiImpl for this class (Freminet/ScarlettX's README, observed on
// Concept Destruction); the executable here is not available yet, so its own requests are unmeasured.

constexpr GUID kGameClass = {0x973a344e, 0x24bf, 0x4d0f, {0x84, 0x57, 0x56, 0xc5, 0x34, 0x89, 0x2b, 0x29}};
constexpr GUID kGame2 = {0x50849859, 0x0ad8, 0x4f81, {0x80, 0xe4, 0x5b, 0xc7, 0x86, 0x26, 0xf8, 0x52}};
constexpr GUID kGame3 = {0x2549f142, 0x6419, 0x4a06, {0x97, 0xb5, 0x93, 0x1a, 0xab, 0x7c, 0x2f, 0x34}};
const GUID* const kGameIids[] = {&kGame2, &kGame3, nullptr};

// The title id MicrosoftGame.Config declares (<TitleId>, eight hex digits), or false when there is none or it is not
// hex. The first <TitleId> element counts; surrounding whitespace is allowed.
bool package_title_id(uint32_t& out) {
    const std::string xml = package_config();
    const size_t open = xml.find("<TitleId>");
    if (open == std::string::npos) return false;
    const size_t begin = xml.find_first_not_of(" \t\r\n", open + 9);
    const size_t close = xml.find("</TitleId>", open);
    if (begin == std::string::npos || close == std::string::npos || begin >= close) return false;
    size_t end = close;
    while (end > begin && std::isspace(static_cast<unsigned char>(xml[end - 1]))) --end;
    if (end - begin == 0 || end - begin > 8) return false;
    uint32_t value = 0;
    for (size_t i = begin; i < end; ++i) {
        const char c = xml[i];
        const int digit = c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10
                        : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1;
        if (digit < 0) return false;
        value = value << 4 | static_cast<uint32_t>(digit);
    }
    out = value;
    return true;
}

// HRESULT XGameGetXboxTitleId(uint32_t* titleId): the package's title id. Without a config that declares one there
// is no title id to give (E_FAIL), rather than a made-up one.
HRESULT STDMETHODCALLTYPE game_get_title_id(void*, uint32_t* title_id) {
    if (!title_id) return E_INVALIDARG;
    uint32_t id = 0;
    if (!package_title_id(id)) {
        static std::atomic<bool> s_logged{false};
        if (!s_logged.exchange(true))
            std::fprintf(stderr, "[rt] xgameruntime: XGameGetXboxTitleId: no <TitleId> in MicrosoftGame.Config\n");
        return E_FAIL;
    }
    *title_id = id;
    return S_OK;
}

// void XLaunchNewGame(const char* exePath, const char* args, XUserHandle defaultUser): on the console this ends the
// title and starts another. Offline there is no other title to start; the call is logged and returns.
void STDMETHODCALLTYPE game_launch_new(void*, const char* exe_path, const char* args, void*) {
    std::fprintf(stderr, "[rt] xgameruntime: XLaunchNewGame(\"%s\", \"%s\") ignored offline\n",
                 exe_path ? exe_path : "", args ? args : "");
}

// HRESULT XLaunchRestartOnCrash(const char* args, uint32_t reserved): asks the console to restart the title after a
// crash. Nothing restarts the process here; it succeeds with nothing registered.
HRESULT STDMETHODCALLTYPE game_restart_on_crash(void*, const char*, uint32_t) {
    static std::atomic<bool> s_logged{false};
    if (!s_logged.exchange(true))
        std::fprintf(stderr, "[rt] xgameruntime: XLaunchRestartOnCrash: no restart is registered offline\n");
    return S_OK;
}

const void* const kGameVtable[] = {
    reinterpret_cast<const void*>(&object_query_interface), reinterpret_cast<const void*>(&object_add_ref),
    reinterpret_cast<const void*>(&object_release),
    reinterpret_cast<const void*>(&game_get_title_id),      // 3 XGameGetXboxTitleId (IXGameImpl)
    reinterpret_cast<const void*>(&game_launch_new),        // 4 XLaunchNewGame (IXGameImpl2, void)
    reinterpret_cast<const void*>(&game_restart_on_crash),  // 5 XLaunchRestartOnCrash (IXGameImpl3)
};
static_assert(std::size(kGameVtable) == 6, "IUnknown + IXGameImpl3's 3 methods");

RuntimeObject& game_object() {
    static RuntimeObject object{kGameVtable, &kGameClass, kGameIids};
    return object;
}

// ---- XGameRuntimeFeatureImpl, XSystemAnalyticsImpl ----
//
// Measured with tools/gdk_runtime_calls.py, which finds xgameruntime.lib's thunks in an image's code without running
// it: for each class, the interface id the thunks ask QueryApiImpl for, the slots they call and the number of
// arguments each passes after `this`. On the package's libHttpClient.GDK.dll every slot below is the function at that
// position of the interface's "Layout" list in xodus-gaming/xgameruntime-docs (GPL-3.0; facts only), and the measured
// argument count is the documented one. Signatures and semantics are Microsoft's public GDK reference pages,
// learn.microsoft.com/gaming/gdk/docs/reference/system/<family>/functions/<name>, cited per function as
// <family>/<name>.
// The thunks are in libHttpClient.GDK.dll but nothing there calls them; a title linked with the same xgameruntime.lib
// reaches these classes through the same thunks. docs/runtime_imports.md has the table.

#define VT(fn) reinterpret_cast<const void*>(&fn)

// XSystemAnalyticsImpl (IXSystemAnalyticsImpl, whose id is the class id): slot 3 is
//   XSystemAnalyticsInfo XSystemGetAnalyticsInfo()   (xsystem/xsystemgetanalyticsinfo)
// The one argument measured after `this` is where the result goes: a member function returning a struct takes its
// address after `this` and returns it (MSVC x64), and the thunk copies the struct out of the returned pointer. The
// answer is the named export's.
constexpr GUID kSystemAnalyticsClass = {0xb884675d, 0xb738, 0x4a9c, {0x81, 0x5d, 0x9a, 0x9a, 0x1e, 0x0c, 0x6c, 0x9b}};

XSystemAnalyticsInfo* STDMETHODCALLTYPE system_analytics_info(void*, XSystemAnalyticsInfo* result) {
    if (result) *result = XSystemGetAnalyticsInfo();
    return result;
}

const void* const kSystemAnalyticsVtable[] = {
    VT(object_query_interface), VT(object_add_ref), VT(object_release),
    VT(system_analytics_info),  // 3
};

// XGameRuntimeFeatureImpl (IXGameRuntimeFeatureImpl, whose id is the class id): slot 3 is
//   bool XGameRuntimeIsFeatureAvailable(XGameRuntimeFeature feature)
// (xgameruntimefeature/xgameruntimeisfeatureavailable), measured with one 32-bit argument and a byte result.
// XGameRuntimeFeature : uint32_t has the values of Microsoft's enum page, XAccessibility = 0 to XGameStreaming = 21.
// Offline a feature is available when both hold:
//   - QueryApiImpl serves the class its functions go through (xgameruntime-docs' layouts), from this file or another
//     xgameruntime_*.cpp, so a call can reach it; the answer follows each class as it is served;
//   - Microsoft documents it for Windows: an offline PC lacks what is documented only for the consoles (XDisplay:
//     xdisplay/xdisplaytryenablehdrmode lists only the Xbox consoles as supported platforms).
// Values past the enum are false.
constexpr GUID kGameRuntimeFeatureClass = {0x8836fe87, 0xedb9, 0x4fe3, {0x8d, 0xad, 0x05, 0xf0, 0xd2, 0xcd, 0x5b, 0x40}};

// The classes no object in this file stands for; the other files serve some of them. Named ...ClassId, not ...Class,
// because tools/gdk_runtime_calls.py counts every `constexpr GUID k...Class` in these files as a served class.
constexpr GUID kAccessibilityClassId = {0x3e241977, 0x6237, 0x41e9, {0x85, 0x59, 0x10, 0x2c, 0x2d, 0x99, 0x83, 0xf1}};
constexpr GUID kAppCaptureClassId = {0xa4f1aee2, 0x4bf1, 0x4485, {0xb0, 0x08, 0xa7, 0xc2, 0x6d, 0x52, 0xac, 0x27}};
constexpr GUID kDisplayClassId = {0x03f0fe74, 0xfdd9, 0x4e5c, {0xb6, 0x30, 0xf9, 0x33, 0x9c, 0x47, 0xac, 0xc5}};
constexpr GUID kGameEventClassId = {0xbbfbdcc7, 0xbfe7, 0x409b, {0xa5, 0xca, 0xed, 0xf0, 0x54, 0x96, 0x0b, 0x4d}};
constexpr GUID kGameInviteClassId = {0x0651aae2, 0x4012, 0x4077, {0xbf, 0x84, 0x8b, 0x90, 0x97, 0x09, 0x0e, 0x2c}};
constexpr GUID kGameSaveClassId = {0x704c3f58, 0xe629, 0x4cc2, {0xb1, 0x97, 0x30, 0x51, 0x1b, 0x99, 0x6f, 0xe2}};
constexpr GUID kGameStreamingClassId = {0x0a2192aa, 0xb2d5, 0x4d58, {0x83, 0xbe, 0x38, 0x3b, 0x6d, 0x80, 0x79, 0x9e}};
constexpr GUID kGameUiClassId = {0xdfcd4649, 0x4ff8, 0x4043, {0xba, 0x07, 0x35, 0xd6, 0x07, 0xdf, 0x98, 0xb0}};
constexpr GUID kLauncherClassId = {0x1b339674, 0x328d, 0x4283, {0xa2, 0x00, 0x31, 0x71, 0xf1, 0x8d, 0x36, 0x39}};
constexpr GUID kStoreClassId = {0x0dd112ac, 0x7c24, 0x448c, {0xb9, 0x2b, 0x39, 0x60, 0xfb, 0x5b, 0xd3, 0x0c}};
constexpr GUID kUserClassId = {0x01acd177, 0x91f9, 0x4763, {0xa3, 0x8e, 0xcc, 0xbb, 0x55, 0xce, 0x32, 0xe0}};

struct RuntimeFeature {
    const GUID* class_id;  // the class the feature's functions go through
    bool console_only;     // documented for the Xbox consoles only
};
const RuntimeFeature kRuntimeFeatures[] = {
    {&kAccessibilityClassId, false},    // 0  XAccessibility
    {&kAppCaptureClassId, false},       // 1  XAppCapture
    {&kThreadingClass, false},          // 2  XAsync
    {&kThreadingClass, false},          // 3  XAsyncProvider
    {&kDisplayClassId, true},           // 4  XDisplay
    {&kGameClass, false},               // 5  XGame
    {&kGameInviteClassId, false},       // 6  XGameInvite
    {&kGameSaveClassId, false},         // 7  XGameSave
    {&kGameUiClassId, false},           // 8  XGameUI
    {&kLauncherClassId, false},         // 9  XLauncher
    {&kNetworkingClass, false},         // 10 XNetworking
    {&kPackageClass, false},            // 11 XPackage
    {&kPersistentStorageClass, false},  // 12 XPersistentLocalStorage
    {&kAccessibilityClassId, false},    // 13 XSpeechSynthesizer (IXAccessibilityImpl holds its functions)
    {&kStoreClassId, false},            // 14 XStore
    {&kSystemClass, false},             // 15 XSystem
    {&kThreadingClass, false},          // 16 XTaskQueue
    {&kThreadingClass, false},          // 17 XThread
    {&kUserClassId, false},             // 18 XUser
    {&kErrorClass, false},              // 19 XError
    {&kGameEventClassId, false},        // 20 XGameEvent
    {&kGameStreamingClassId, false},    // 21 XGameStreaming
};
static_assert(std::size(kRuntimeFeatures) == 22, "XGameRuntimeFeature has 22 values");

bool STDMETHODCALLTYPE runtime_is_feature_available(void*, uint32_t feature) {
    if (feature >= std::size(kRuntimeFeatures)) return false;
    const RuntimeFeature& f = kRuntimeFeatures[feature];
    return !f.console_only && class_served(*f.class_id);
}

const void* const kGameRuntimeFeatureVtable[] = {
    VT(object_query_interface), VT(object_add_ref), VT(object_release),
    VT(runtime_is_feature_available),  // 3
};
#undef VT

RuntimeObject g_game_runtime_feature{kGameRuntimeFeatureVtable, &kGameRuntimeFeatureClass, nullptr};
RuntimeObject g_system_analytics{kSystemAnalyticsVtable, &kSystemAnalyticsClass, nullptr};

RuntimeObject* const* measured_objects() {
    static RuntimeObject* const objects[] = {&g_game_runtime_feature, &g_system_analytics, nullptr};
    return objects;
}

#define XGR(name) {"xgameruntime.dll", #name, reinterpret_cast<void*>(&name)}

}  // namespace

SHIM_TABLE(xgameruntime,
    // what xgameruntime.lib's thunks GetProcAddress (library_loader.cpp serves the LoadLibrary before it)
    XGR(InitializeApiImpl),
    XGR(InitializeApiImplEx),
    XGR(InitializeApiImplEx2),
    XGR(UninitializeApiImpl),
    XGR(QueryApiImpl),
    // the same runtime through gameplatformservices.dll, which XCurl.dll loads from System32 (GetApiVersionImpl above)
    {"gameplatformservices.dll", "QueryApiImpl", reinterpret_cast<void*>(&QueryApiImpl)},
    {"gameplatformservices.dll", "GetApiVersionImpl", reinterpret_cast<void*>(&GetApiVersionImpl)},
    XGR(XGameRuntimeInitialize),
    XGR(XGameRuntimeUninitialize),
    XGR(XSystemGetDeviceType),
    XGR(XSystemGetConsoleId),
    XGR(XSystemGetAppMemoryUsageLimit),
    XGR(XSystemGetAppMemoryUsage),
    XGR(XSystemGetAnalyticsInfo),
    XGR(XPackageGetCurrentProcessPackageIdentifier),
    XGR(XPackageMount),
    XGR(XPackageCloseMountHandle),
    XGR(XPackageGetMountPathSize),
    XGR(XPackageGetMountPath),
    XGR(XPackageIsPackagedProcess),
    // XTaskQueue (third_party/xtaskqueue)
    XGR(XTaskQueueCreate),
    XGR(XTaskQueueCreateComposite),
    XGR(XTaskQueueGetPort),
    XGR(XTaskQueueDuplicateHandle),
    XGR(XTaskQueueCloseHandle),
    XGR(XTaskQueueTerminate),
    XGR(XTaskQueueDispatch),
    XGR(XTaskQueueSubmitCallback),
    XGR(XTaskQueueSubmitDelayedCallback),
    XGR(XTaskQueueRegisterWaiter),
    XGR(XTaskQueueUnregisterWaiter),
    XGR(XTaskQueueRegisterMonitor),
    XGR(XTaskQueueUnregisterMonitor),
    XGR(XTaskQueueGetCurrentProcessTaskQueue),
    XGR(XTaskQueueSetCurrentProcessTaskQueue),
    // XAsync (third_party/xtaskqueue)
    XGR(XAsyncGetStatus),
    XGR(XAsyncGetResultSize),
    XGR(XAsyncCancel),
    XGR(XAsyncRun),
    XGR(XAsyncBegin),
    XGR(XAsyncSchedule),
    XGR(XAsyncComplete),
    XGR(XAsyncGetResult),
);
