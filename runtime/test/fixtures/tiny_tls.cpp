// Tiny fixture DLL for the loader's static-TLS test. It is ordinary test code, not game content.
//
// __declspec(thread) makes the compiler read TEB->ThreadLocalStoragePointer[_tls_index], so the exported getter only
// works when the loader has written a real module TLS index into AddressOfIndex and installed a copy of the template
// in that slot. The callback is registered in .CRT$XLB (the linker's TLS callback array); the loader runs it through
// run_tls_callbacks, and with the thread events once initialise() has run. Thread events are counted only for the
// thread named by tiny_tls_watch_thread, so threads the process starts for itself don't disturb the counts.
#include <windows.h>

namespace {

__declspec(thread) int g_tls_value = 7;

int g_callback_count = 0;
int g_attach_count = 0;
int g_detach_count = 0;
DWORD g_last_reason = 0;
PVOID g_callback_base = nullptr;
PVOID g_callback_reserved = reinterpret_cast<PVOID>(1);
int g_dllmain_count = 0;
volatile DWORD g_watched_thread = 0;
int g_thread_attach_count = 0;
int g_thread_detach_count = 0;
int g_value_at_thread_attach = -1;

void NTAPI tls_callback(PVOID base, DWORD reason, PVOID reserved) {
    ++g_callback_count;
    g_last_reason = reason;
    g_callback_base = base;
    g_callback_reserved = reserved;
    if (reason == DLL_PROCESS_ATTACH) ++g_attach_count;
    if (reason == DLL_PROCESS_DETACH) ++g_detach_count;
    if (GetCurrentThreadId() == g_watched_thread) {
        if (reason == DLL_THREAD_ATTACH) {
            ++g_thread_attach_count;
            g_value_at_thread_attach = g_tls_value;  // the thread's own copy must already be in place
        }
        if (reason == DLL_THREAD_DETACH) ++g_thread_detach_count;
    }
}

}  // namespace

#pragma section(".CRT$XLB", read)
__declspec(allocate(".CRT$XLB")) PIMAGE_TLS_CALLBACK g_tls_callback = tls_callback;

extern "C" __declspec(dllexport) int __cdecl tiny_tls_get() { return g_tls_value; }
extern "C" __declspec(dllexport) void __cdecl tiny_tls_set(int value) { g_tls_value = value; }
extern "C" __declspec(dllexport) int __cdecl tiny_tls_plain() { return 123; }

extern "C" __declspec(dllexport) int __cdecl tiny_tls_callback_count() { return g_callback_count; }
extern "C" __declspec(dllexport) int __cdecl tiny_tls_attach_count() { return g_attach_count; }
extern "C" __declspec(dllexport) int __cdecl tiny_tls_detach_count() { return g_detach_count; }
extern "C" __declspec(dllexport) DWORD __cdecl tiny_tls_last_reason() { return g_last_reason; }
extern "C" __declspec(dllexport) PVOID __cdecl tiny_tls_callback_base() { return g_callback_base; }
extern "C" __declspec(dllexport) PVOID __cdecl tiny_tls_callback_reserved() { return g_callback_reserved; }
extern "C" __declspec(dllexport) int __cdecl tiny_tls_dllmain_count() { return g_dllmain_count; }
extern "C" __declspec(dllexport) void __cdecl tiny_tls_watch_thread(DWORD id) { g_watched_thread = id; }
extern "C" __declspec(dllexport) int __cdecl tiny_tls_thread_attach_count() { return g_thread_attach_count; }
extern "C" __declspec(dllexport) int __cdecl tiny_tls_thread_detach_count() { return g_thread_detach_count; }
extern "C" __declspec(dllexport) int __cdecl tiny_tls_value_at_thread_attach() { return g_value_at_thread_attach; }

BOOL WINAPI DllMain(HINSTANCE, DWORD, LPVOID) {
    ++g_dllmain_count;
    return TRUE;
}
