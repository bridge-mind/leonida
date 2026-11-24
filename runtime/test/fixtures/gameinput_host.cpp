// A DLL that does what a title's input code does first: GameInputCreate, a timestamp, a gamepad reading. It uses the
// Windows SDK's public GameInput.h and links against the SDK's gameinput.lib, so GameInputCreate is imported from
// GameInput.dll (by ordinal) like any title's import. Built without the CRT: DllMain does nothing and the exports call
// only GameInput. gameinput_host_test maps it with the loader. No game data.
#include <windows.h>

#include <GameInput.h>

extern "C" BOOL WINAPI gameinput_host_main(HINSTANCE, DWORD, LPVOID) { return TRUE; }

#ifdef __MSVC_RUNTIME_CHECKS
// Debug builds compile with /RTC1 (the Visual Studio generator's default), whose checks call into the CRT; /NODEFAULTLIB
// leaves these three unresolved (LNK1120). The checks have nothing to check here, so they do nothing.
extern "C" {
void __cdecl _RTC_InitBase() {}
void __cdecl _RTC_Shutdown() {}
void __fastcall _RTC_CheckStackVars(void*, void*) {}
}
#endif

// Returns 1 when the library was created. reading_hr is S_OK with a gamepad attached and
// GAMEINPUT_E_READING_NOT_FOUND without one (E_FAIL when the library was not created); state_ok is 1 when
// GetGamepadState succeeded on a reading.
extern "C" __declspec(dllexport) int gameinput_host_probe(long* create_hr, unsigned long long* timestamp,
                                                          long* reading_hr, int* state_ok) {
    IGameInput* input = nullptr;
    IGameInputReading* reading = nullptr;
    const HRESULT hr = GameInputCreate(&input);
    *create_hr = hr;
    *timestamp = 0;
    *reading_hr = E_FAIL;  // not S_OK: a failed GameInputCreate must not look like an attached gamepad
    *state_ok = 0;
    if (FAILED(hr) || !input) return 0;
    *timestamp = input->GetCurrentTimestamp();
    *reading_hr = input->GetCurrentReading(GameInputKindGamepad, nullptr, &reading);
    if (SUCCEEDED(*reading_hr) && reading) {
        GameInputGamepadState state;
        *state_ok = reading->GetGamepadState(&state) ? 1 : 0;
        reading->Release();
    }
    input->Release();
    return 1;
}
