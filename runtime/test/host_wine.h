// Whether the host D3D12 these tests compare against is Wine's (vkd3d) rather than Microsoft's runtime. The host
// comparisons take Microsoft's d3d12.dll and WARP as the reference: under Wine, vkd3d answers differently in ways
// that aren't the layer's (it accepts copies Microsoft refuses, reports other heap flags and alignments, keeps no
// separate WARP), so such a test skips its host comparisons there and says why. Windows runs them all. The same
// check as copycmp::host_is_wine (d3d12_copy_compare.h) and loader_test, for tests that create their host
// devices themselves.
#pragma once

#include <windows.h>

#include <cstdio>

namespace hostwine {

inline bool host_is_wine() {
    static const bool wine = GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "wine_get_version") != nullptr;
    static bool said = false;
    if (wine && !said) std::printf("host D3D12 is Wine's, not Microsoft's: comparisons with the host are skipped\n");
    said = true;
    return wine;
}

}  // namespace hostwine
