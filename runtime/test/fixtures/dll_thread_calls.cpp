// A DLL whose DllMain returns the result of DisableThreadLibraryCalls(hinstance), the way the MSVC runtime's satellite
// DLLs do (msvcp140_1.dll, msvcp140_2.dll, msvcp140_atomic_wait.dll, msvcp140_codecvt_ids.dll). Built without the
// CRT, so nothing but this one call runs at DLL_PROCESS_ATTACH. dll_thread_calls_test maps it with the loader. No game
// data.
#include <windows.h>

#ifdef __MSVC_RUNTIME_CHECKS
// Debug builds compile with /RTC1 (the Visual Studio generator's default), whose checks call into the CRT; /NODEFAULTLIB
// leaves these unresolved (LNK1120). The checks have nothing to check here, so they do nothing.
extern "C" {
void __cdecl _RTC_InitBase() {}
void __cdecl _RTC_Shutdown() {}
}
#endif

extern "C" BOOL WINAPI dll_thread_calls_main(HINSTANCE instance, DWORD reason, LPVOID) {
    if (reason == DLL_PROCESS_ATTACH) return DisableThreadLibraryCalls(instance);
    return TRUE;
}

// The same call on a handle that names no module: the answer must stay the Windows one (FALSE).
extern "C" __declspec(dllexport) BOOL dll_thread_calls_bogus_handle() {
    return DisableThreadLibraryCalls(reinterpret_cast<HMODULE>(static_cast<uintptr_t>(0x10000)));
}
