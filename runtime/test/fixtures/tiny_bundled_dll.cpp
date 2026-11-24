// A stand-in bundled Xbox DLL (like Bink2Scarlett.dll or amd_fidelityfx_upscaler_xbox.dll) that imports console
// memory functions from xmem.dll and is loaded at run time through LoadLibraryW by tiny_bundled_load.exe.
// Test code, not game content.
#include <windows.h>

#include <cstddef>
#include <cstdint>

extern "C" {
__declspec(dllimport) void* __stdcall XMemAlloc(size_t size, uint64_t attributes);
__declspec(dllimport) void __stdcall XMemFree(void* ptr, uint64_t attributes);
}

namespace {
int g_attach_count = 0;
}

BOOL WINAPI DllMain(HINSTANCE, DWORD reason, LPVOID) {
    if (reason == DLL_PROCESS_ATTACH) ++g_attach_count;
    return TRUE;
}

extern "C" __declspec(dllexport) int __stdcall bundled_attach_count() { return g_attach_count; }

extern "C" __declspec(dllexport) uint32_t __stdcall bundled_alloc_sum() {
    auto* buf = static_cast<uint32_t*>(XMemAlloc(16 * sizeof(uint32_t), 0x5e50000));
    if (!buf) return 0;
    if ((reinterpret_cast<uintptr_t>(buf) & 63u) != 0) {
        XMemFree(buf, 0x5e50000);
        return 0;
    }
    uint32_t sum = 0;
    for (uint32_t i = 0; i < 16; ++i) {
        buf[i] = i + 1;
        sum += buf[i];
    }
    XMemFree(buf, 0x5e50000);
    return sum;
}
