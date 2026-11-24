// A mapped program that reaches the GDK runtime the way xgameruntime.lib's thunks do: it has no import of
// xgameruntime.dll, loads it with LoadLibraryW and looks the five run-time entry points up with GetProcAddress
// (measured in the package's libHttpClient.GDK.dll and XCurl.dll, docs/runtime_imports.md "Run-time loaded
// modules"). Run under `gta6rt --run`, its own kernel32 imports are bound by the loader, so LoadLibraryW and
// GetProcAddress must reach runtime/shims/library_loader.cpp and the offline xgameruntime shim. Run on its own,
// LoadLibraryW goes to Windows and either fails or loads the PC's Gaming Services runtime, and this never prints
// PASS. Ordinary test code; no game data.
#include <windows.h>

#include <cstdio>
#include <cstring>
#include <cwchar>
#include <initializer_list>

namespace {

struct Options {        // InitializeApiImplEx2's fourth argument (xodus-gaming/xgameruntime-docs)
    UINT32 unknown;     // documented as always 1
    BOOL is_inline;
    const char* game_config;
};

using InitEx2 = HRESULT(WINAPI*)(ULONG, ULONG, char, const Options*);
using Query = HRESULT(WINAPI*)(const GUID*, REFIID, void**);
using Uninit = HRESULT(WINAPI*)();

int fail(const char* stage, unsigned long value) {
    std::printf("tiny_gdk_load: FAIL %s (0x%08lx)\n", stage, value);
    return 1;
}

}  // namespace

int main() {
    for (const wchar_t* dll : {L"kernel32.dll", L"kernelbase.dll"}) {
        HMODULE host = GetModuleHandleW(dll);
        if (!host) return fail("host module handle", GetLastError());
        auto command_a = reinterpret_cast<decltype(&GetCommandLineA)>(GetProcAddress(host, "GetCommandLineA"));
        auto command_w = reinterpret_cast<decltype(&GetCommandLineW)>(GetProcAddress(host, "GetCommandLineW"));
        if (!command_a || !command_w || std::strcmp(command_a(), GetCommandLineA()) != 0 ||
            std::wcscmp(command_w(), GetCommandLineW()) != 0 ||
            !std::strstr(command_a(), "host-shim-probe") || !std::wcsstr(command_w(), L"host-shim-probe"))
            return fail("dynamic command line matches direct imports", 0);
        auto process_id = reinterpret_cast<decltype(&GetCurrentProcessId)>(GetProcAddress(host, "GetCurrentProcessId"));
        if (!process_id || process_id() != GetCurrentProcessId())
            return fail("unshimmed host export", GetLastError());
        SetLastError(0);
        if (GetProcAddress(host, "LeonidaNoSuchHostExport") || GetLastError() != ERROR_PROC_NOT_FOUND)
            return fail("missing host export", GetLastError());
        auto create_file = reinterpret_cast<decltype(&CreateFileW)>(GetProcAddress(host, "CreateFileW"));
        if (!create_file) return fail("dynamic CreateFileW", GetLastError());
        HANDLE file = create_file(L"GAME:\\host_shim_probe.txt", GENERIC_READ, FILE_SHARE_READ, nullptr,
                                  OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file == INVALID_HANDLE_VALUE) return fail("dynamic GAME: open", GetLastError());
        char data[4]{};
        DWORD count = 0;
        const BOOL read = ReadFile(file, data, sizeof(data), &count, nullptr);
        CloseHandle(file);
        if (!read || count != sizeof(data) || std::memcmp(data, "shim", sizeof(data)) != 0)
            return fail("dynamic GAME: contents", 0);
    }
    HMODULE runtime = LoadLibraryW(L"xgameruntime.dll");
    if (!runtime) return fail("LoadLibraryW(xgameruntime.dll)", GetLastError());
    // The loader shim hands out a pseudo-handle for a module it owns: not an image the host loaded.
    HMODULE host_view = nullptr;
    if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, L"xgameruntime.dll", &host_view) && host_view == runtime)
        return fail("xgameruntime.dll came from the host loader, not the shims", 0);

    const char* names[] = {"InitializeApiImpl", "InitializeApiImplEx", "InitializeApiImplEx2", "UninitializeApiImpl",
                           "QueryApiImpl"};
    for (const char* name : names)
        if (!GetProcAddress(runtime, name)) return fail(name, GetLastError());
    SetLastError(0);
    if (GetProcAddress(runtime, "NoSuchRuntimeFunction") || GetLastError() != ERROR_PROC_NOT_FOUND)
        return fail("an unserved name is NULL with ERROR_PROC_NOT_FOUND", GetLastError());

    auto init = reinterpret_cast<InitEx2>(GetProcAddress(runtime, "InitializeApiImplEx2"));
    auto query = reinterpret_cast<Query>(GetProcAddress(runtime, "QueryApiImpl"));
    auto uninit = reinterpret_cast<Uninit>(GetProcAddress(runtime, "UninitializeApiImpl"));
    const Options options{1, FALSE, nullptr};
    HRESULT hr = init(0, 0, 0x02, &options);
    if (hr != S_OK) return fail("InitializeApiImplEx2", static_cast<unsigned long>(hr));

    // XThreadingImpl, the first class XCurl.dll references (docs/runtime_imports.md), used as its thunks do: the
    // class id as both ids, a slot with `this` first, then Release (slot 2). Slot 12 is XTaskQueueCreate, 17
    // XTaskQueueCloseHandle.
    const GUID threading = {0x073b7dcb, 0x1fcf, 0x4030, {0x94, 0xbe, 0xe3, 0xc9, 0xeb, 0x62, 0x34, 0x28}};
    void* obj = nullptr;
    hr = query(&threading, threading, &obj);
    if (hr != S_OK || !obj) return fail("QueryApiImpl hands out XThreadingImpl", static_cast<unsigned long>(hr));
    void* const* vtable = *static_cast<void* const* const*>(obj);
    using Create = HRESULT(STDMETHODCALLTYPE*)(void*, UINT32, UINT32, void**);
    using Close = void(STDMETHODCALLTYPE*)(void*, void*);
    using Release = ULONG(STDMETHODCALLTYPE*)(void*);
    void* queue = nullptr;
    hr = reinterpret_cast<Create>(vtable[12])(obj, 0, 0, &queue);  // XTaskQueueDispatchMode::Manual twice
    if (hr != S_OK || !queue) return fail("XThreadingImpl slot 12 (XTaskQueueCreate)", static_cast<unsigned long>(hr));
    reinterpret_cast<Close>(vtable[17])(obj, queue);
    reinterpret_cast<Release>(vtable[2])(obj);

    // A class not served (an id no runtime class has, made up for this test) is refused, with the out pointer
    // cleared.
    const GUID unserved = {0x4c454f4e, 0x4944, 0x4100, {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01}};
    void* out = reinterpret_cast<void*>(1);
    hr = query(&unserved, unserved, &out);
    if (hr != static_cast<HRESULT>(0x80070032) || out != nullptr)
        return fail("QueryApiImpl refuses an unprovided class and clears its out pointer", static_cast<unsigned long>(hr));
    hr = uninit();
    if (hr != S_OK) return fail("UninitializeApiImpl", static_cast<unsigned long>(hr));

    // Host modules still come from the host.
    HMODULE k32 = LoadLibraryW(L"kernel32.dll");
    if (!k32 || GetProcAddress(k32, "GetTickCount") != GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "GetTickCount"))
        return fail("kernel32.dll from the host", GetLastError());
    if (!FreeLibrary(runtime)) return fail("FreeLibrary(xgameruntime.dll)", GetLastError());

    std::puts("tiny_gdk_load: PASS");
    return 0;
}
