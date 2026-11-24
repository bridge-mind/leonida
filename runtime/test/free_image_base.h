#pragma once

// A preferred base for a synthetic test image without relocations, so the loader can map it where it asks.
//
// The tests fix their images at bases from 0x140000000 (the usual EXE base) up to 0x1a0000000 (DLLs). On Windows
// that whole span is free, and free_image_base() returns the constants unchanged. Under arm64 Wine (CrossOver on
// Apple Silicon) the process's own allocations sit at 0x100000000 and up and macOS maps its shared region at
// 0x180000000 (RT-22), so such an image is rejected with "preferred base is taken" before a test checks what it is
// about, and the span can be free when a test starts and taken by Wine's own allocations a few tests later. Under
// Wine the span is therefore moved as a whole to a free window the system picks top-down, away from where the heap
// grows: each constant keeps its offset, so images loaded together still get different bases.
#include <windows.h>

#include <cstdint>

namespace leonida_test {

inline std::uint64_t free_image_base(std::uint64_t preferred) {
    constexpr std::uint64_t kFirst = 0x140000000, kEnd = 0x200000000;
    if (preferred < kFirst || preferred >= kEnd) return preferred;
    static const std::uint64_t window = [] {
        constexpr SIZE_T size = kEnd - kFirst;
        const HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
        if (!ntdll || !GetProcAddress(ntdll, "wine_get_version")) return kFirst;  // Windows: the constants
        if (void* p = VirtualAlloc(nullptr, size, MEM_RESERVE | MEM_TOP_DOWN, PAGE_NOACCESS)) {
            VirtualFree(p, 0, MEM_RELEASE);
            return reinterpret_cast<std::uint64_t>(p);  // 64 KiB aligned, as an image base must be
        }
        return kFirst;
    }();
    return window + (preferred - kFirst);
}

}  // namespace leonida_test
