// A mapped program that loads a bundled Xbox DLL (tiny_bundled_dll.dll, which imports xmem.dll) at run time via
// LoadLibraryW, GetModuleHandle, GetProcAddress and FreeLibrary — the pattern used by titles that dynamically load
// Bink2Scarlett.dll, XboxBvhBuilder.dll or the FidelityFX DLLs. Under `gta6rt --run`, library_loader.cpp routes the
// load to rt::Loader so the DLL's xmem.dll imports bind to the shim registry instead of failing under the host loader.
// Ordinary test code; no game data.
#include <windows.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace {

using AttachCountFn = int(__stdcall*)();
using AllocSumFn = uint32_t(__stdcall*)();
using XMemAllocFn = void*(__stdcall*)(size_t, uint64_t);
using XMemFreeFn = void(__stdcall*)(void*, uint64_t);

int fail(const char* stage, unsigned long value) {
    std::printf("tiny_bundled_load: FAIL %s (0x%08lx)\n", stage, value);
    return 1;
}

}  // namespace

int main() {
    if (GetModuleHandleW(L"tiny_bundled_dll.dll") != nullptr)
        return fail("GetModuleHandleW before LoadLibrary should be NULL", 0);

    // DONT_RESOLVE_DLL_REFERENCES must fail explicitly rather than silently resolving imports and running DllMain.
    SetLastError(0);
    if (LoadLibraryExW(L"tiny_bundled_dll.dll", nullptr, DONT_RESOLVE_DLL_REFERENCES) != nullptr ||
        GetLastError() != ERROR_NOT_SUPPORTED) {
        return fail("LoadLibraryExW(DONT_RESOLVE_DLL_REFERENCES) fails with ERROR_NOT_SUPPORTED", GetLastError());
    }
    if (GetModuleHandleW(L"tiny_bundled_dll.dll") != nullptr)
        return fail("DONT_RESOLVE_DLL_REFERENCES must not map or register the bundled DLL", 0);

    HMODULE mod = LoadLibraryW(L"tiny_bundled_dll.dll");
    if (!mod) return fail("LoadLibraryW(tiny_bundled_dll.dll)", GetLastError());

    // Must be mapped by rt::Loader, not by the host Windows loader.
    HMODULE host_view = nullptr;
    if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, L"tiny_bundled_dll.dll", &host_view) &&
        host_view == mod) {
        return fail("tiny_bundled_dll.dll came from the host loader, not rt::Loader", 0);
    }

    if (GetModuleHandleW(L"tiny_bundled_dll.dll") != mod)
        return fail("GetModuleHandleW after LoadLibrary returns the same handle", 0);
    if (GetModuleHandleA("TINY_BUNDLED_DLL") != mod)
        return fail("GetModuleHandleA case-insensitive without .dll returns the same handle", 0);
    if (LoadLibraryExA("C:\\Fake\\Path\\tiny_bundled_dll.dll", nullptr, 0) != mod)
        return fail("LoadLibraryExA with full path returns the same already-mapped handle", GetLastError());

    auto attach_count = reinterpret_cast<AttachCountFn>(GetProcAddress(mod, "bundled_attach_count"));
    if (!attach_count) return fail("GetProcAddress(bundled_attach_count)", GetLastError());
    const int count = attach_count();
    if (count != 1) return fail("DllMain(DLL_PROCESS_ATTACH) ran once", static_cast<unsigned long>(count));

    auto alloc_sum = reinterpret_cast<AllocSumFn>(GetProcAddress(mod, "bundled_alloc_sum"));
    if (!alloc_sum) return fail("GetProcAddress(bundled_alloc_sum)", GetLastError());
    if (reinterpret_cast<void*>(GetProcAddress(mod, MAKEINTRESOURCEA(2))) != reinterpret_cast<void*>(alloc_sum))
        return fail("GetProcAddress by ordinal #2 matches bundled_alloc_sum", GetLastError());
    const uint32_t sum = alloc_sum();
    if (sum != 136u) return fail("bundled_alloc_sum via xmem shim returned 136", sum);

    auto fwd_alloc = reinterpret_cast<XMemAllocFn>(GetProcAddress(mod, "bundled_fwd_alloc"));
    auto fwd_free = reinterpret_cast<XMemFreeFn>(GetProcAddress(mod, "bundled_fwd_free"));
    if (!fwd_alloc || !fwd_free) return fail("GetProcAddress follows forwarders to xmem shim", GetLastError());
    void* block = fwd_alloc(128, 0);
    if (!block || (reinterpret_cast<uintptr_t>(block) & 63u) != 0)
        return fail("forwarded xmem.XMemAlloc returned aligned block", 0);
    fwd_free(block, 0);

    SetLastError(0);
    if (GetProcAddress(mod, "NoSuchExport") != nullptr || GetLastError() != ERROR_PROC_NOT_FOUND)
        return fail("missing export returns NULL with ERROR_PROC_NOT_FOUND", GetLastError());
    SetLastError(0);
    if (GetProcAddress(mod, nullptr) != nullptr || GetLastError() != ERROR_PROC_NOT_FOUND)
        return fail("null export name returns NULL with ERROR_PROC_NOT_FOUND", GetLastError());

    if (!FreeLibrary(mod)) return fail("FreeLibrary(tiny_bundled_dll.dll)", GetLastError());

    // Host modules still resolve through the host loader.
    HMODULE k32 = LoadLibraryW(L"kernel32.dll");
    if (!k32 || GetProcAddress(k32, "GetTickCount") != GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "GetTickCount"))
        return fail("kernel32.dll still comes from the host", GetLastError());

    std::puts("tiny_bundled_load: PASS");
    return 0;
}
