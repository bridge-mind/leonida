// Link-time stand-in for the console's d3d12_xs.dll, so tiny_frame.exe gets a real import of
// D3D12XboxCreateDevice. Under `gta6rt --run` the shim registry answers the import and this DLL never loads; it is
// built into a folder the loader does not search. Test code, not game content.
#include <windows.h>
#include <unknwn.h>

extern "C" HRESULT __stdcall D3D12XboxCreateDevice(IUnknown*, const void*, const IID*, void** out) {
    if (out) *out = nullptr;
    return E_FAIL;
}
