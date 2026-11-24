// Link-time stand-in for the GDK's xgameruntime.dll, so tiny_game.exe gets real imports of these names. Under
// `gta6rt --run` the shim registry answers them first and this DLL never loads; run directly, these failing answers
// are what tiny_game would see. Test code, not game content.
#include <windows.h>

#include <cstdint>
#include <XAsync.h>

extern "C" __declspec(dllexport) HRESULT __stdcall XGameRuntimeInitialize() { return E_FAIL; }
extern "C" __declspec(dllexport) void __stdcall XGameRuntimeUninitialize() {}
extern "C" __declspec(dllexport) uint32_t __stdcall XSystemGetDeviceType() { return 0; }

STDAPI XTaskQueueCreate(
    XTaskQueueDispatchMode, XTaskQueueDispatchMode, XTaskQueueHandle* queue) noexcept {
    *queue = nullptr;
    return E_NOTIMPL;
}
STDAPI_(bool) XTaskQueueDispatch(
    XTaskQueueHandle, XTaskQueuePort, uint32_t) noexcept { return false; }
STDAPI_(void) XTaskQueueCloseHandle(XTaskQueueHandle) noexcept {}
STDAPI XAsyncRun(XAsyncBlock*, XAsyncWork*) noexcept { return E_NOTIMPL; }
STDAPI XAsyncGetStatus(XAsyncBlock*, bool) noexcept { return E_NOTIMPL; }
