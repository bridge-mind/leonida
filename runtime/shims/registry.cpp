#include <windows.h>
#include <intrin.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "shims.h"
#include "startup_trace.h"

namespace rt::shims {
namespace {

std::map<std::string, void*>& table() {
    static std::map<std::string, void*> t;  // "dll!name"
    return t;
}

std::string lower_ascii(std::string s) {
    for (auto& c : s)
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c + 32);
    return s;
}

bool starts_with(const std::string& s, const char* prefix) { return s.rfind(prefix, 0) == 0; }

std::mutex g_called_mutex;
std::set<std::string> g_called;
std::map<const char*, ReturnClass> g_seen;  // by the thunk's `what` pointer; both under g_called_mutex
std::atomic<TestHook> g_test_hook{nullptr};

// ---- Return-value policy ----------------------------------------------------------------------------------------

constexpr int kStopExitCode = 3;  // what abort() exits with; documented, and checked by the tests

std::atomic<int> g_policy{-1};  // -1: not decided yet, LEONIDA_UNIMPLEMENTED is read on first use

struct Lookup {
    void* ctx;
    ModuleLookup fn;
};
std::mutex g_lookup_mutex;
std::vector<Lookup> g_lookups;

// The imports known to return HRESULT. Everything else is "other" and keeps returning 0 under `fail`.
// Each family was checked against the public Microsoft documentation (learn.microsoft.com), one signature per
// function; no SDK headers were read. An HRESULT function that is not listed just returns 0, as before, so a missing
// entry is the safe direction; a wrong entry would hand E_NOTIMPL (nonzero, "true") to a bool or pointer caller.
//
//  d3d12.dll, d3d12_xs.dll   D3D12Create*: D3D12CreateDevice, D3D12CreateRootSignatureDeserializer,
//                            D3D12CreateVersionedRootSignatureDeserializer return HRESULT
//                            (learn.microsoft.com/windows/win32/api/d3d12/nf-d3d12-d3d12createdevice and siblings).
//                            D3D12Serialize*: D3D12SerializeRootSignature, D3D12SerializeVersionedRootSignature: HRESULT.
//                            D3D12GetDebugInterface, D3D12EnableExperimentalFeatures: HRESULT.
//                            D3D12XboxCreateDevice is matched by exact name: the console-only function has no public
//                            reference page that could be reached; it is the factory this repo measured (it returns
//                            an HRESULT with the device in an out parameter, docs/d3d12_vulkan.md), not a documented one.
//                            D3D12XboxCreateSyntheticDevice is the other console-only factory the sibling export
//                            inventories list (docs/d3d12_vulkan.md §7); same Create*Device HRESULT shape.
//                            D3D12XboxSetProcessDebugFlags is left out: nothing measured its return.
//  dxgi.dll                  CreateDXGIFactory, CreateDXGIFactory1, CreateDXGIFactory2, DXGIGetDebugInterface1 and
//                            DXGIDeclareAdapterRemovalSupport return HRESULT (win32/api/dxgi/nf-dxgi-createdxgifactory,
//                            dxgi1_3 and dxgi1_6 pages). Not the whole DXGI* prefix: only these names were verified.
//  xgameruntime.dll          GDK reference, gaming/gdk/docs/reference/system/<family>/functions/<name>. These families
//                            return HRESULT except the functions in kNotHResult, which the same pages show as
//                            void/bool/int: XAsync*, XTaskQueue*, XUser*, XStore*, XGameSave*, XPackage*, XGameUi*.
//                            XGameRuntimeInitialize and XGameEventWrite: HRESULT.
//                            Mixed families (XSystem*, XGameStreaming*, XThread*, XDisplay*, XGameRuntimeFeature*) are
//                            left out on purpose.
//  gameinput.dll             GameInputCreate: HRESULT (gaming/gdk/docs/reference/input/gameinput/functions/
//                            gameinputcreate). The shim implements it (runtime/shims/gameinput.cpp); the class is
//                            listed anyway for a build or import it misses.
//  xaudio2_9.dll             XAudio2Create: HRESULT (learn.microsoft.com/windows/win32/api/xaudio2/
//  xaudio2_9redist.dll       nf-xaudio2-xaudio2create). xaudio2_9.dll is the console name (docs/d3d12_vulkan.md §7).
//  dstorage.dll              DStorageGetFactory: HRESULT (learn.microsoft.com/windows/win32/api/dstorage/
//  dstorage_xs.dll           nf-dstorage-dstoragegetfactory). dstorage_xs.dll is the console name.
const char* const kNotHResult[] = {
    "XAsyncCancel", "XAsyncComplete",
    "XTaskQueueCloseHandle", "XTaskQueueDispatch", "XTaskQueueGetCurrentProcessTaskQueue",
    "XTaskQueueSetCurrentProcessTaskQueue", "XTaskQueueUnregisterMonitor", "XTaskQueueUnregisterWaiter",
    "XUserCloseHandle", "XUserCloseSignOutDeferralHandle", "XUserCompare", "XUserIsStoreUser",
    "XUserUnregisterForChangeEvent", "XUserUnregisterForDefaultAudioEndpointUtf16Changed",
    "XUserUnregisterForDeviceAssociationChanged",
    "XStoreCloseContextHandle", "XStoreCloseLicenseHandle", "XStoreCloseProductsQueryHandle", "XStoreIsLicenseValid",
    "XStoreProductsQueryHasMorePages", "XStoreUnregisterGameLicenseChanged", "XStoreUnregisterPackageLicenseLost",
    "XGameSaveCloseContainer", "XGameSaveCloseProvider", "XGameSaveCloseUpdate",
    "XPackageCloseInstallationMonitorHandle", "XPackageCloseMountHandle", "XPackageGetInstallationProgress",
    "XPackageIsPackagedProcess", "XPackageUninstallPackage", "XPackageUnregisterInstallationProgressChanged",
    "XPackageUnregisterPackageInstalled", "XPackageUpdateInstallationMonitor",
};
const char* const kXFamilies[] = {"XAsync", "XTaskQueue", "XUser", "XStore", "XGameSave", "XPackage", "XGameUi"};

ReturnClass classify(const std::string& dll, const std::string& name) {
    if (dll == "d3d12.dll" || dll == "d3d12_xs.dll") {
        if (starts_with(name, "D3D12Create") || starts_with(name, "D3D12Serialize") ||
            name == "D3D12GetDebugInterface" || name == "D3D12EnableExperimentalFeatures" ||
            name == "D3D12XboxCreateDevice" || name == "D3D12XboxCreateSyntheticDevice")
            return ReturnClass::HResult;
    } else if (dll == "dxgi.dll") {
        if (starts_with(name, "CreateDXGIFactory") || name == "DXGIGetDebugInterface1" ||
            name == "DXGIDeclareAdapterRemovalSupport")
            return ReturnClass::HResult;
    } else if (dll == "xaudio2_9.dll" || dll == "xaudio2_9redist.dll") {
        if (name == "XAudio2Create") return ReturnClass::HResult;
    } else if (dll == "gameinput.dll") {
        if (name == "GameInputCreate") return ReturnClass::HResult;
    } else if (dll == "dstorage.dll" || dll == "dstorage_xs.dll") {
        if (name == "DStorageGetFactory") return ReturnClass::HResult;
    } else if (dll == "xgameruntime.dll") {
        if (name == "XGameRuntimeInitialize" || name == "XGameEventWrite") return ReturnClass::HResult;
        for (const char* n : kNotHResult)
            if (name == n) return ReturnClass::Other;
        for (const char* f : kXFamilies)
            if (starts_with(name, f)) return ReturnClass::HResult;
    }
    return ReturnClass::Other;
}

struct CallInfo {  // what the common stub saved, in this order
    uint64_t rcx, rdx, r8, r9, ret;
};

// "0x00007ff6... = module+0x1234", resolved through the loaders' module lists, else through the host's modules.
std::string describe_address(uint64_t addr) {
    char head[32], tail[64];
    std::snprintf(head, sizeof head, "0x%016llx", static_cast<unsigned long long>(addr));
    std::string module;
    uint32_t rva = 0;
    {
        std::lock_guard lock(g_lookup_mutex);
        for (auto& l : g_lookups)
            if (l.fn(l.ctx, addr, module, rva)) {
                std::snprintf(tail, sizeof tail, "+0x%x", rva);
                return std::string(head) + " = " + module + tail;
            }
    }
    HMODULE host = nullptr;
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCSTR>(addr), &host) && host) {
        char path[MAX_PATH] = {};
        GetModuleFileNameA(host, path, MAX_PATH);
        const char* slash = std::strrchr(path, '\\');
        std::snprintf(tail, sizeof tail, "+0x%llx",
                      static_cast<unsigned long long>(addr - reinterpret_cast<uint64_t>(host)));
        return std::string(head) + " = host module " + (slash ? slash + 1 : path) + tail;
    }
    return std::string(head) + " = not inside any module";
}

// Called by the common stub with the import's "dll!name" string and the registers it saved. The return value goes
// back to the game's caller in rax; the stub zeroes xmm0.
extern "C" uint64_t rt_unimplemented(const char* what, const CallInfo* info) {
    if (auto hook = g_test_hook.load()) hook(_AddressOfReturnAddress());
    const UnimplementedPolicy policy = unimplemented_policy();

    // `what` is a string that lives for the process and is unique per import, so the pointer keys the per-import
    // state. Strings are built only the first time an import is seen; a repeat call costs one map lookup.
    ReturnClass cls;
    bool first = false;
    {
        std::lock_guard lock(g_called_mutex);
        auto it = g_seen.find(what);
        if (it == g_seen.end()) {
            const std::string full = what;
            const auto bang = full.find('!');
            cls = classify(lower_ascii(bang == std::string::npos ? "" : full.substr(0, bang)),
                           bang == std::string::npos ? full : full.substr(bang + 1));
            g_seen.emplace(what, cls);
            g_called.insert(full);
            first = true;
        } else {
            cls = it->second;
        }
    }
    const uint64_t fail_value = cls == ReturnClass::HResult ? kENotImpl : 0;
    if (first && startup_trace::enabled())
        startup_trace::event("unimplemented_call", "\"import\":" + startup_trace::quote(what) +
                                                       ",\"caller\":" + startup_trace::quote(module_offset(info->ret)));

    if (policy == UnimplementedPolicy::Stop) {
        std::fprintf(stderr,
                     "[rt] STOP: unimplemented import %s called\n"
                     "[rt]   caller   %s\n"
                     "[rt]   rcx=0x%llx rdx=0x%llx r8=0x%llx r9=0x%llx\n"
                     "[rt]   return class: %s (LEONIDA_UNIMPLEMENTED=fail returns %s)\n",
                     what, describe_address(info->ret).c_str(), static_cast<unsigned long long>(info->rcx),
                     static_cast<unsigned long long>(info->rdx), static_cast<unsigned long long>(info->r8),
                     static_cast<unsigned long long>(info->r9), cls == ReturnClass::HResult ? "hresult" : "other",
                     cls == ReturnClass::HResult ? "E_NOTIMPL" : "0");
        if (IsDebuggerPresent()) {
            std::fprintf(stderr, "[rt]   debugger attached: breaking; continuing returns the `fail` value\n");
            std::fflush(stderr);
            __debugbreak();
            return fail_value;
        }
        std::fprintf(stderr, "[rt]   no debugger attached: terminating the process (exit code %d)\n", kStopExitCode);
        std::fflush(stderr);
        TerminateProcess(GetCurrentProcess(), kStopExitCode);
        return 0;  // not reached
    }
    if (first)
        std::fprintf(stderr, "[rt] unimplemented: %s%s\n", what,
                     policy == UnimplementedPolicy::Fail
                         ? (cls == ReturnClass::HResult ? " (returns E_NOTIMPL)" : " (returns 0)")
                         : "");
    return policy == UnimplementedPolicy::Fail ? fail_value : 0;
}

// Thunk layout (x86-64), one per import:
//   49 BA <imm64>   mov r10, what            ("dll!name"; r10 is volatile and carries no argument)
//   48 B8 <imm64>   mov rax, common_stub
//   FF E0           jmp rax
// The game's `call` pushed the return address, so at thunk entry [rsp] is the caller. rcx/rdx/r8/r9 are untouched.
//
// The common stub (one copy, made on first use) saves those registers and the return address, calls
// rt_unimplemented(what, &saved), and returns its rax straight to the game's caller with xmm0 zeroed (so a caller
// that expects a float or double result sees 0.0):
//   sub rsp, 0x48                       ; 8 (return address) + 0x48 leaves rsp 16-byte aligned at the call
//   mov [rsp+0x20..0x38], rcx, rdx, r8, r9   ; the CallInfo, above the 32-byte shadow space
//   mov rax, [rsp+0x48]; mov [rsp+0x40], rax  ; the return address
//   mov rcx, r10; lea rdx, [rsp+0x20]; mov rax, rt_unimplemented; call rax
//   add rsp, 0x48; xorps xmm0, xmm0; ret
// An UNWIND_INFO (UWOP_ALLOC_SMALL 0x48) is registered for the stub, so a debugger stopped inside rt_unimplemented
// can walk up into the game.
constexpr size_t kThunkSize = 32;

uint8_t* g_pool = nullptr;
size_t g_pool_used = 0, g_pool_size = 0;
std::mutex g_pool_mutex;
void* g_stub = nullptr;  // under g_pool_mutex

void* make_common_stub() {
    constexpr size_t kUnwindOffset = 0x80, kFunctionOffset = 0x90, kRegionSize = 0x1000;
    auto* region =
        static_cast<uint8_t*>(VirtualAlloc(nullptr, kRegionSize, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE));
    if (!region) return nullptr;
    static const uint8_t head[] = {
        0x48, 0x83, 0xEC, 0x48,        // sub rsp, 0x48
        0x48, 0x89, 0x4C, 0x24, 0x20,  // mov [rsp+0x20], rcx
        0x48, 0x89, 0x54, 0x24, 0x28,  // mov [rsp+0x28], rdx
        0x4C, 0x89, 0x44, 0x24, 0x30,  // mov [rsp+0x30], r8
        0x4C, 0x89, 0x4C, 0x24, 0x38,  // mov [rsp+0x38], r9
        0x48, 0x8B, 0x44, 0x24, 0x48,  // mov rax, [rsp+0x48]
        0x48, 0x89, 0x44, 0x24, 0x40,  // mov [rsp+0x40], rax
        0x4C, 0x89, 0xD1,              // mov rcx, r10
        0x48, 0x8D, 0x54, 0x24, 0x20,  // lea rdx, [rsp+0x20]
        0x48, 0xB8,                    // mov rax, imm64
    };
    static const uint8_t tail[] = {
        0xFF, 0xD0,              // call rax
        0x48, 0x83, 0xC4, 0x48,  // add rsp, 0x48
        0x0F, 0x57, 0xC0,        // xorps xmm0, xmm0
        0xC3,                    // ret
    };
    static_assert(sizeof(head) + 8 + sizeof(tail) < kUnwindOffset);
    const uint64_t fn = reinterpret_cast<uint64_t>(&rt_unimplemented);
    size_t n = 0;
    std::memcpy(region + n, head, sizeof head);
    n += sizeof head;
    std::memcpy(region + n, &fn, 8);
    n += 8;
    std::memcpy(region + n, tail, sizeof tail);
    n += sizeof tail;

    // UNWIND_INFO: version 1, no flags, prolog of 4 bytes, one code (offset 4, UWOP_ALLOC_SMALL, (0x48 - 8) / 8 = 8).
    const uint8_t unwind[] = {0x01, 0x04, 0x01, 0x00, 0x04, 0x82, 0x00, 0x00};
    std::memcpy(region + kUnwindOffset, unwind, sizeof unwind);
    auto* rf = reinterpret_cast<RUNTIME_FUNCTION*>(region + kFunctionOffset);
    rf->BeginAddress = 0;
    rf->EndAddress = static_cast<DWORD>(n);
    rf->UnwindData = kUnwindOffset;
    // Without the table the stub still works; only a stack walk through it (a debugger, an exception) stops there.
    if (!RtlAddFunctionTable(rf, 1, reinterpret_cast<DWORD64>(region)))
        std::fprintf(stderr, "[rt] RtlAddFunctionTable failed for the unimplemented-import stub (error %lu): "
                             "stack walks cannot pass through it\n", GetLastError());
    FlushInstructionCache(GetCurrentProcess(), region, kRegionSize);
    return region;
}

}  // namespace

UnimplementedPolicy parse_unimplemented_policy(const char* text, bool* valid) {
    if (valid) *valid = true;
    if (!text || !*text) return UnimplementedPolicy::Log;
    const std::string v = lower_ascii(text);
    if (v == "log") return UnimplementedPolicy::Log;
    if (v == "fail") return UnimplementedPolicy::Fail;
    if (v == "stop") return UnimplementedPolicy::Stop;
    if (valid) *valid = false;
    return UnimplementedPolicy::Log;
}

UnimplementedPolicy unimplemented_policy() {
    int p = g_policy.load();
    if (p < 0) {
        char buf[32] = {};
        const DWORD n = GetEnvironmentVariableA("LEONIDA_UNIMPLEMENTED", buf, sizeof buf);
        bool valid = true;
        const UnimplementedPolicy parsed = parse_unimplemented_policy(n < sizeof buf ? buf : "?", &valid);
        if (!valid) std::fprintf(stderr, "[rt] LEONIDA_UNIMPLEMENTED: expected log, fail or stop; using log\n");
        int expected = -1;
        g_policy.compare_exchange_strong(expected, static_cast<int>(parsed));
        p = g_policy.load();
    }
    return static_cast<UnimplementedPolicy>(p);
}

void set_unimplemented_policy(UnimplementedPolicy p) { g_policy.store(static_cast<int>(p)); }

ReturnClass classify_import(const std::string& dll, const std::string& name) { return classify(lower_ascii(dll), name); }

void set_unimplemented_test_hook(TestHook hook) { g_test_hook.store(hook); }

void add_module_lookup(void* ctx, ModuleLookup fn) {
    std::lock_guard lock(g_lookup_mutex);
    g_lookups.push_back({ctx, fn});
}

bool mapped_image_base(uint64_t address, uint64_t& base) {
    std::string module;
    uint32_t rva = 0;
    std::lock_guard lock(g_lookup_mutex);
    for (auto& l : g_lookups)
        if (l.fn(l.ctx, address, module, rva)) {
            base = address - rva;
            return true;
        }
    return false;
}

std::string module_offset(uint64_t address) {
    char b[48];
    std::string module;
    uint32_t rva = 0;
    {
        std::lock_guard lock(g_lookup_mutex);
        for (auto& l : g_lookups)
            if (l.fn(l.ctx, address, module, rva)) {
                std::snprintf(b, sizeof b, "+0x%x", rva);
                return module + b;
            }
    }
    HMODULE host = nullptr;
    if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCWSTR>(address), &host) && host) {
        // Wide, then UTF-8: the ANSI form would put code-page bytes into the trace's JSON lines.
        wchar_t path[MAX_PATH] = {};
        const DWORD len = GetModuleFileNameW(host, path, MAX_PATH);
        const wchar_t* slash = std::wcsrchr(path, L'\\');
        const wchar_t* file = slash ? slash + 1 : path;
        const int wlen = len ? static_cast<int>(path + len - file) : 0;
        const int bytes = wlen > 0 ? WideCharToMultiByte(CP_UTF8, 0, file, wlen, nullptr, 0, nullptr, nullptr) : 0;
        std::string name(bytes > 0 ? bytes : 0, '\0');
        if (bytes > 0) WideCharToMultiByte(CP_UTF8, 0, file, wlen, name.data(), bytes, nullptr, nullptr);
        std::snprintf(b, sizeof b, "+0x%llx", static_cast<unsigned long long>(address - reinterpret_cast<uint64_t>(host)));
        return "host:" + name + b;
    }
    std::snprintf(b, sizeof b, "0x%llx", static_cast<unsigned long long>(address));
    return b;
}

void remove_module_lookup(void* ctx) {
    std::lock_guard lock(g_lookup_mutex);
    for (auto it = g_lookups.begin(); it != g_lookups.end(); ++it)
        if (it->ctx == ctx) {
            g_lookups.erase(it);
            break;
        }
}

namespace {

struct BundledLoaderEntry {
    void* ctx;
    const BundledModuleOps* ops;
};

// Immortal: late mapped-DLL detach or atexit handlers may call GetProcAddress/FreeLibrary during process teardown.
std::mutex& bundled_loader_mutex() {
    static auto* m = new std::mutex;
    return *m;
}

std::vector<BundledLoaderEntry>& bundled_loaders() {
    static auto* v = new std::vector<BundledLoaderEntry>;
    return *v;
}

std::vector<BundledLoaderEntry> snapshot_bundled_loaders() {
    std::lock_guard lock(bundled_loader_mutex());
    return bundled_loaders();
}

}  // namespace

void add_bundled_loader(void* ctx, const BundledModuleOps* ops) {
    if (!ctx || !ops) return;
    std::lock_guard lock(bundled_loader_mutex());
    bundled_loaders().push_back({ctx, ops});
}

void remove_bundled_loader(void* ctx) {
    std::lock_guard lock(bundled_loader_mutex());
    auto& list = bundled_loaders();
    for (auto it = list.begin(); it != list.end(); ++it)
        if (it->ctx == ctx) {
            list.erase(it);
            break;
        }
}

void* find_loaded_bundled_module(const std::string& dll_lower) {
    for (const auto& entry : snapshot_bundled_loaders())
        if (void* h = entry.ops->find_loaded(entry.ctx, dll_lower)) return h;
    return nullptr;
}

void* load_bundled_module(const std::string& dll_lower, uint32_t flags, bool& handled) {
    handled = false;
    for (const auto& entry : snapshot_bundled_loaders()) {
        bool loader_handled = false;
        void* h = entry.ops->load_bundled(entry.ctx, dll_lower, flags, loader_handled);
        if (loader_handled) {
            handled = true;
            return h;
        }
    }
    return nullptr;
}

void* find_loaded_bundled_module_w(const std::wstring& basename) {
    for (const auto& entry : snapshot_bundled_loaders())
        if (entry.ops->find_loaded_w)
            if (void* h = entry.ops->find_loaded_w(entry.ctx, basename)) return h;
    return nullptr;
}

void* load_bundled_module_w(const std::wstring& basename, uint32_t flags, bool& handled) {
    handled = false;
    for (const auto& entry : snapshot_bundled_loaders()) {
        if (!entry.ops->load_bundled_w) continue;
        bool loader_handled = false;
        void* h = entry.ops->load_bundled_w(entry.ctx, basename, flags, loader_handled);
        if (loader_handled) {
            handled = true;
            return h;
        }
    }
    return nullptr;
}

void* get_bundled_proc_address(void* module, const std::string& sym, bool& owns_handle) {
    owns_handle = false;
    for (const auto& entry : snapshot_bundled_loaders()) {
        bool loader_owns = false;
        void* fn = entry.ops->get_proc(entry.ctx, module, sym, loader_owns);
        if (loader_owns) {
            owns_handle = true;
            return fn;
        }
    }
    return nullptr;
}

bool owns_bundled_module_handle(void* module) {
    for (const auto& entry : snapshot_bundled_loaders())
        if (entry.ops->owns_handle(entry.ctx, module)) return true;
    return false;
}

Table::Table(const Entry* entries, size_t count) {
    for (size_t i = 0; i < count; ++i) table()[lower_ascii(entries[i].dll) + "!" + entries[i].name] = entries[i].fn;
}

void* find(const std::string& dll_lower, const std::string& name) {
    auto it = table().find(dll_lower + "!" + name);
    return it == table().end() ? nullptr : it->second;
}

bool owns_dll(const std::string& dll_lower) {
    static const char* const host[] = {"kernel32.dll", "kernelbase.dll", "msvcrt.dll", "ucrtbase.dll", "ucrtbased.dll",
                                       "vulkan-1.dll", "ntdll.dll", "user32.dll"};
    if (starts_with(dll_lower, "api-ms-") || starts_with(dll_lower, "ext-ms-")) return false;
    for (const char* h : host)
        if (dll_lower == h) return false;
    const std::string prefix = dll_lower + "!";
    auto it = table().lower_bound(prefix);
    return it != table().end() && it->first.compare(0, prefix.size(), prefix) == 0;
}

void* make_unimplemented_thunk(const std::string& dll, const std::string& name) {
    std::lock_guard lock(g_pool_mutex);
    static std::map<std::string, void*> s_thunks;
    const std::string key = lower_ascii(dll) + "!" + name;
    auto it = s_thunks.find(key);
    if (it != s_thunks.end()) return it->second;

    if (!g_stub) g_stub = make_common_stub();
    if (!g_stub) return nullptr;
    if (!g_pool || g_pool_used + kThunkSize > g_pool_size) {
        g_pool_size = 1 << 20;
        g_pool = static_cast<uint8_t*>(VirtualAlloc(nullptr, g_pool_size, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE));
        g_pool_used = 0;
        if (!g_pool) return nullptr;
    }
    const char* what = _strdup(key.c_str());  // lives for the process
    uint8_t* t = g_pool + g_pool_used;
    g_pool_used += kThunkSize;
    const uint64_t a = reinterpret_cast<uint64_t>(what), f = reinterpret_cast<uint64_t>(g_stub);
    uint8_t code[kThunkSize] = {0x49, 0xBA};
    std::memcpy(code + 2, &a, 8);
    code[10] = 0x48;
    code[11] = 0xB8;
    std::memcpy(code + 12, &f, 8);
    code[20] = 0xFF;
    code[21] = 0xE0;
    std::memset(code + 22, 0xCC, kThunkSize - 22);
    std::memcpy(t, code, kThunkSize);
    FlushInstructionCache(GetCurrentProcess(), t, kThunkSize);
    s_thunks.emplace(key, t);
    return t;
}

void report_called_unimplemented() {
    std::lock_guard lock(g_called_mutex);
    for (auto& s : g_called) std::fprintf(stderr, "  called: %s\n", s.c_str());
}

}  // namespace rt::shims
