// Whether the test runs where windows and outputs exist. From an SSH session or a service the process has a window
// station that is not visible: the host DXGI then answers DXGI_ERROR_NOT_CURRENTLY_AVAILABLE for EnumOutputs and
// CreateSwapChainForHwnd, and a window cannot be shown, so tests that need a desktop skip there.
#pragma once

#include <windows.h>

inline bool has_interactive_window_station() {
    HWINSTA station = GetProcessWindowStation();
    if (!station) return true;  // Unknown environment: let window creation report the real failure.
    USEROBJECTFLAGS flags{};
    DWORD bytes = 0;
    if (!GetUserObjectInformationW(station, UOI_FLAGS, &flags, sizeof(flags), &bytes)) return true;
    return (flags.dwFlags & WSF_VISIBLE) != 0;
}
