// library_loader: LoadLibrary / GetModuleHandle / GetProcAddress for modules the shims own.
//
// GDK code does not always import the console runtime. The package's libHttpClient.GDK.dll and XCurl.dll have no
// import entry for xgameruntime.dll at all: their statically linked xgameruntime.lib thunks hold the wide string
// "xgameruntime.dll" and the names InitializeApiImpl, InitializeApiImplEx, InitializeApiImplEx2, QueryApiImpl and
// UninitializeApiImpl, and look them up at run time (docs/runtime_imports.md, "Run-time loaded modules"). Bound to the
// host, that LoadLibraryW would load the PC's Gaming Services runtime or fail, and never reach the offline shims.
//
// So for a module the shim registry owns (rt::shims::owns_dll: xgameruntime.dll, d3d12_xs.dll, xmem.dll, ...), these
// return a pseudo-handle, and GetProcAddress on it resolves names (and "#ordinal") through rt::shims::find. A name the
// registry doesn't serve returns NULL with ERROR_PROC_NOT_FOUND, as a real module would, and is logged once. Every
// other module uses the same named shims as static imports, falling back to the host. Only images mapped by the
// loader see these: they are bound through the registry, the runtime's own code calls the host directly.
#include <windows.h>

#include <cstdio>
#include <deque>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "shims.h"

namespace rt::shims::library_loader {

// A module the shims own, identified by a pointer to one of these. Host modules are image bases (64 KiB aligned);
// these live in the runtime's heap, and GetProcAddress tells them apart by lookup, not by address range.
struct Pseudo {
    char tag[8] = {'L', 'E', 'O', 'S', 'H', 'I', 'M', 0};
    std::string dll;  // lower case, with .dll
};

namespace {

// Intentionally immortal: mapped-DLL DLL_PROCESS_DETACH and earlier atexit handlers can call GetProcAddress or
// FreeLibrary on pseudo-handles during process teardown, after normal statics would have been destroyed.
std::mutex& loader_mutex() {
    static auto* m = new std::mutex;
    return *m;
}

std::deque<Pseudo>& pseudos() {   // a deque never moves its elements, so handed-out handles stay valid
    static auto* d = new std::deque<Pseudo>;
    return *d;
}

std::string lower_ascii(std::string s) {
    for (auto& c : s)
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c + 32);
    return s;
}

std::string narrow(const wchar_t* w) {
    std::string s;
    for (; w && *w; ++w) s += (*w < 0x80) ? static_cast<char>(*w) : '?';
    return s;
}

bool is_ascii(const wchar_t* w) {
    for (; w && *w; ++w)
        if (*w >= 0x80) return false;
    return true;
}

// module_name for a name narrow() cannot spell: the last path component as given (the loader matches it
// case-insensitively, as Windows does), ".dll" added when there is no extension.
std::wstring module_name_w(std::wstring path) {
    const size_t slash = path.find_last_of(L"\\/");
    if (slash != std::wstring::npos) path.erase(0, slash + 1);
    if (!path.empty() && path.back() == L'.') path.pop_back();
    else if (path.find(L'.') == std::wstring::npos) path += L".dll";
    return path;
}

// Whether the module a wide name refers to has an ASCII name. Only the last component decides: the directory may be
// outside ASCII (narrow() turns it into '?', which the basename-only lookups never see) while the module is a shim's.
bool ascii_basename(const wchar_t* name) {
    return is_ascii(module_name_w(name).c_str());
}

// LoadLibrary's file-name rules for the module a path names: the last path component, lower case, and ".dll" added
// when there is no extension (a trailing '.' means "no extension, add nothing").
std::string module_name(std::string path) {
    const size_t slash = path.find_last_of("\\/");
    if (slash != std::string::npos) path.erase(0, slash + 1);
    path = lower_ascii(path);
    if (!path.empty() && path.back() == '.') path.pop_back();
    else if (path.find('.') == std::string::npos) path += ".dll";
    return path;
}

std::string host_module_name(HMODULE module) {
    std::lock_guard lock(loader_mutex());
    static auto* names = new std::map<HMODULE, std::string>;
    if (auto it = names->find(module); it != names->end()) return it->second;

    std::vector<wchar_t> path(MAX_PATH);
    for (;;) {
        const DWORD length = GetModuleFileNameW(module, path.data(), static_cast<DWORD>(path.size()));
        if (!length) return {};
        if (length < path.size()) {
            const std::string name = module_name(narrow(path.data()));
            names->emplace(module, name);
            return name;
        }
        if (path.size() == 32768) return {};
        path.resize(path.size() > 16384 ? 32768 : path.size() * 2);
    }
}

}  // namespace

// The pseudo-handle for an owned module, or nullptr when the shims don't own `path`'s module.
HMODULE owned_handle(const std::string& path) {
    const std::string dll = module_name(path);
    if (dll.empty() || !owns_dll(dll)) return nullptr;
    std::lock_guard lock(loader_mutex());
    for (auto& p : pseudos())
        if (p.dll == dll) return reinterpret_cast<HMODULE>(&p);
    pseudos().push_back(Pseudo{});
    pseudos().back().dll = dll;
    std::fprintf(stderr, "[rt] loader: %s is served by the shims (run-time load)\n", dll.c_str());
    return reinterpret_cast<HMODULE>(&pseudos().back());
}

// The owned module a handle names, or nullptr for a host module.
const Pseudo* pseudo(HMODULE h) {
    std::lock_guard lock(loader_mutex());
    for (auto& p : pseudos())
        if (reinterpret_cast<HMODULE>(&p) == h) return &p;
    return nullptr;
}

namespace {

constexpr DWORD kDataFileLoadFlags =
    LOAD_LIBRARY_AS_DATAFILE | LOAD_LIBRARY_AS_DATAFILE_EXCLUSIVE | LOAD_LIBRARY_AS_IMAGE_RESOURCE;

HMODULE load_shim_or_bundled(const std::string& path, DWORD flags, bool& handled) {
    handled = false;
    if (HMODULE h = owned_handle(path)) {
        handled = true;
        return h;
    }
    const std::string dll = module_name(path);
    if (!dll.empty()) {
        if (void* h = load_bundled_module(dll, static_cast<uint32_t>(flags), handled); handled)
            return reinterpret_cast<HMODULE>(h);
    }
    return nullptr;
}

HMODULE WINAPI load_library_a(LPCSTR name) {
    if (name) {
        bool handled = false;
        HMODULE h = load_shim_or_bundled(name, 0, handled);
        if (handled) return h;
    }
    return LoadLibraryA(name);
}

// A name with a character outside ASCII names no shim (their names are ASCII) and narrow() would turn it into '?',
// so it goes to the bundled loader by its wide basename: an image already mapped under that name is returned, as
// Windows returns the loaded module for LoadLibraryW of its own file.
HMODULE load_bundled_w(LPCWSTR name, DWORD flags, bool& handled) {
    handled = false;
    const std::wstring dll = module_name_w(name);
    if (dll.empty()) return nullptr;
    return reinterpret_cast<HMODULE>(load_bundled_module_w(dll, static_cast<uint32_t>(flags), handled));
}

HMODULE WINAPI load_library_w(LPCWSTR name) {
    if (name) {
        bool handled = false;
        HMODULE h = ascii_basename(name) ? load_shim_or_bundled(narrow(name), 0, handled) : load_bundled_w(name, 0, handled);
        if (handled) return h;
    }
    return LoadLibraryW(name);
}

HMODULE WINAPI load_library_ex_a(LPCSTR name, HANDLE file, DWORD flags) {
    if (name && (flags & kDataFileLoadFlags) == 0) {
        bool handled = false;
        HMODULE h = load_shim_or_bundled(name, flags, handled);
        if (handled) return h;
    }
    return LoadLibraryExA(name, file, flags);
}

HMODULE WINAPI load_library_ex_w(LPCWSTR name, HANDLE file, DWORD flags) {
    if (name && (flags & kDataFileLoadFlags) == 0) {
        bool handled = false;
        HMODULE h = ascii_basename(name) ? load_shim_or_bundled(narrow(name), flags, handled)
                                   : load_bundled_w(name, flags, handled);
        if (handled) return h;
    }
    return LoadLibraryExW(name, file, flags);
}

// GetModuleHandle answers for an owned module as if it were loaded (the GDK may probe before it loads), and for any
// bundled package module already mapped by the native loader.
HMODULE WINAPI get_module_handle_a(LPCSTR name) {
    if (name) {
        if (HMODULE h = owned_handle(name)) return h;
        const std::string dll = module_name(name);
        if (!dll.empty())
            if (void* h = find_loaded_bundled_module(dll)) return reinterpret_cast<HMODULE>(h);
    }
    return GetModuleHandleA(name);
}

HMODULE WINAPI get_module_handle_w(LPCWSTR name) {
    if (name) {
        if (ascii_basename(name)) {
            const std::string path = narrow(name);
            if (HMODULE h = owned_handle(path)) return h;
            const std::string dll = module_name(path);
            if (!dll.empty())
                if (void* h = find_loaded_bundled_module(dll)) return reinterpret_cast<HMODULE>(h);
        } else {
            // Outside ASCII: by the wide basename (narrow() would turn it into '?'), as for load_library_w.
            const std::wstring dll = module_name_w(name);
            if (!dll.empty())
                if (void* h = find_loaded_bundled_module_w(dll)) return reinterpret_cast<HMODULE>(h);
        }
    }
    return GetModuleHandleW(name);
}

FARPROC WINAPI get_proc_address(HMODULE module, LPCSTR name) {
    if (const Pseudo* p = pseudo(module)) {
        if (!name) {
            SetLastError(ERROR_PROC_NOT_FOUND);
            return nullptr;
        }
        // A value below 0x10000 is an ordinal (IS_INTRESOURCE), as in the real GetProcAddress.
        const std::string what =
            IS_INTRESOURCE(name) ? "#" + std::to_string(reinterpret_cast<uintptr_t>(name)) : name;
        if (void* fn = find(p->dll, what)) return reinterpret_cast<FARPROC>(fn);
        {
            static auto* m = new std::mutex;
            static auto* seen = new std::set<std::string>;
            std::lock_guard lock(*m);
            if (seen->insert(p->dll + "!" + what).second)
                std::fprintf(stderr, "[rt] loader: GetProcAddress(%s, %s): not served\n", p->dll.c_str(), what.c_str());
        }
        SetLastError(ERROR_PROC_NOT_FOUND);
        return nullptr;
    }

    if (!name) {
        if (owns_bundled_module_handle(module)) {
            SetLastError(ERROR_PROC_NOT_FOUND);
            return nullptr;
        }
        return GetProcAddress(module, name);
    }

    const std::string what = IS_INTRESOURCE(name) ? "#" + std::to_string(reinterpret_cast<uintptr_t>(name)) : name;
    bool owns_handle = false;
    void* fn = get_bundled_proc_address(module, what, owns_handle);
    if (owns_handle) {
        if (fn) return reinterpret_cast<FARPROC>(fn);
        SetLastError(ERROR_PROC_NOT_FOUND);
        return nullptr;
    }
    if (module && !IS_INTRESOURCE(name)) {
        const DWORD error = GetLastError();
        const std::string dll = host_module_name(module);
        void* shim = dll.empty() ? nullptr : find(dll, what);
        SetLastError(error);
        if (shim) return reinterpret_cast<FARPROC>(shim);
    }
    return GetProcAddress(module, name);
}

BOOL WINAPI free_library(HMODULE module) {
    if (pseudo(module)) return TRUE;   // owned modules are never unloaded
    if (owns_bundled_module_handle(module)) return TRUE;  // mapped modules stay live until ~Loader detaches them
    return FreeLibrary(module);
}

// DisableThreadLibraryCalls(hinstance) from a mapped image's DllMain. The real function only accepts a module the
// Windows loader mapped, so for ours it fails, and a DllMain that returns its result (the MSVC runtime's satellite
// DLLs do) fails Loader::initialise() with "DllMain returned FALSE". Owned and mapped modules answer TRUE. The loader
// sends no thread events to a DllMain, so there is nothing further to switch off; every other handle goes to the
// real function.
BOOL WINAPI disable_thread_calls(HMODULE module) {
    if (pseudo(module)) return TRUE;
    if (owns_bundled_module_handle(module)) return TRUE;
    return DisableThreadLibraryCalls(module);
}

#define LOADER(dll)                                                                         \
    {dll, "LoadLibraryA", reinterpret_cast<void*>(&rt::shims::library_loader::load_library_a)},                        \
    {dll, "LoadLibraryW", reinterpret_cast<void*>(&rt::shims::library_loader::load_library_w)},                        \
    {dll, "LoadLibraryExA", reinterpret_cast<void*>(&rt::shims::library_loader::load_library_ex_a)},                   \
    {dll, "LoadLibraryExW", reinterpret_cast<void*>(&rt::shims::library_loader::load_library_ex_w)},                   \
    {dll, "GetModuleHandleA", reinterpret_cast<void*>(&rt::shims::library_loader::get_module_handle_a)},               \
    {dll, "GetModuleHandleW", reinterpret_cast<void*>(&rt::shims::library_loader::get_module_handle_w)},               \
    {dll, "GetProcAddress", reinterpret_cast<void*>(&rt::shims::library_loader::get_proc_address)},                    \
    {dll, "FreeLibrary", reinterpret_cast<void*>(&rt::shims::library_loader::free_library)},                           \
    {dll, "DisableThreadLibraryCalls", reinterpret_cast<void*>(&rt::shims::library_loader::disable_thread_calls)}

}  // namespace
}  // namespace rt::shims::library_loader

// The package's DLLs import the loader functions through these api sets (XCurl.dll: libraryloader-l1-2-0 and
// -l1-2-1); kernel32 and kernelbase cover code linked against the classic libraries.
SHIM_TABLE(library_loader,
    LOADER("kernel32.dll"),
    LOADER("kernelbase.dll"),
    LOADER("api-ms-win-core-libraryloader-l1-1-0.dll"),
    LOADER("api-ms-win-core-libraryloader-l1-1-1.dll"),
    LOADER("api-ms-win-core-libraryloader-l1-2-0.dll"),
    LOADER("api-ms-win-core-libraryloader-l1-2-1.dll"),
    LOADER("api-ms-win-core-libraryloader-l1-2-2.dll"),
);
