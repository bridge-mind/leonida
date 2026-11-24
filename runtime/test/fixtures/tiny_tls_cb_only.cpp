// Callback-only static TLS fixture DLL: defines _tls_used with StartAddressOfRawData == EndAddressOfRawData == 0
// and SizeOfZeroFill == 0 while providing a TLS callback (no CRT .tls section). Ordinary test code, not game data.
#include <windows.h>

#ifdef __MSVC_RUNTIME_CHECKS
// Debug builds compile with /RTC1 (the Visual Studio generator's default), whose checks call into the CRT; /NODEFAULTLIB
// leaves these unresolved (LNK1120). The checks have nothing to check here, so they do nothing.
extern "C" {
void __cdecl _RTC_InitBase() {}
void __cdecl _RTC_Shutdown() {}
}
#endif

namespace {

ULONG g_tls_index = 0;
int g_attach_count = 0;
int g_detach_count = 0;
volatile DWORD g_watched_thread = 0;
int g_thread_attach_count = 0;
int g_thread_detach_count = 0;

void NTAPI tls_callback(PVOID, DWORD reason, PVOID) {
    if (reason == DLL_PROCESS_ATTACH) ++g_attach_count;
    if (reason == DLL_PROCESS_DETACH) ++g_detach_count;
    if (GetCurrentThreadId() == g_watched_thread) {
        if (reason == DLL_THREAD_ATTACH) ++g_thread_attach_count;
        if (reason == DLL_THREAD_DETACH) ++g_thread_detach_count;
    }
}

PIMAGE_TLS_CALLBACK g_callbacks[] = {tls_callback, nullptr};

}  // namespace

extern "C" const IMAGE_TLS_DIRECTORY64 _tls_used = {
    0,                                         // StartAddressOfRawData
    0,                                         // EndAddressOfRawData
    reinterpret_cast<ULONGLONG>(&g_tls_index), // AddressOfIndex
    reinterpret_cast<ULONGLONG>(g_callbacks),  // AddressOfCallBacks
    0,                                         // SizeOfZeroFill
    0,                                         // Characteristics
};

extern "C" __declspec(dllexport) void __cdecl tiny_tls_cb_watch_thread(DWORD id) { g_watched_thread = id; }
extern "C" __declspec(dllexport) int __cdecl tiny_tls_cb_attach_count() { return g_attach_count; }
extern "C" __declspec(dllexport) int __cdecl tiny_tls_cb_detach_count() { return g_detach_count; }
extern "C" __declspec(dllexport) int __cdecl tiny_tls_cb_thread_attach_count() { return g_thread_attach_count; }
extern "C" __declspec(dllexport) int __cdecl tiny_tls_cb_thread_detach_count() { return g_thread_detach_count; }

extern "C" BOOL WINAPI DllMainCRTStartup(HINSTANCE, DWORD, LPVOID) { return TRUE; }
