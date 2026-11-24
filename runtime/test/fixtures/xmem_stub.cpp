// Link-time stand-in for the console's xmem.dll, so tiny_bundled_dll.dll gets real imports of XMemAlloc and
// XMemFree (the same imports Bink2Scarlett.dll and amd_fidelityfx_upscaler_xbox.dll carry). Under `gta6rt --run` the
// shim registry answers them instead; if the host loader ever loaded this stub directly, XMemAlloc returns nullptr
// so the test fails loudly. Test code, not game content.
#include <cstddef>
#include <cstdint>

extern "C" __declspec(dllexport) void* __stdcall XMemAlloc(size_t, uint64_t) { return nullptr; }
extern "C" __declspec(dllexport) void __stdcall XMemFree(void*, uint64_t) {}
