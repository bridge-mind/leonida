// xg_xs.dll: the console's texture layout library. The game executable (1.0.2.1) imports two functions from it.
//
// Measured from the call sites in the executable, read statically (shapes and checks only):
//   HRESULT XGSuggestSwizzleMode(const desc*, 0, uint32_t* mode)
//     2 call sites, both in the game's helper that fills a texture description before XGCreateTextureComputer. The
//     out pointer is a 32-bit field of that description, which the helper first sets to 0 and, when the mode is
//     known, writes itself as 0x100 | mode. The HRESULT is only passed back up; no caller branches on it.
//   HRESULT XGCreateTextureComputer(const desc*, computer** out)
//     4 call sites. Two test the HRESULT and take their failure path when it is negative; two ignore it and call
//     through *out at once. On the computer the callers use vtable slot 1 after they are done (3 sites, read as
//     Release; no site calls slot 0, read as AddRef, nor QueryInterface), slot 2 with one out pointer to a large
//     stack structure (2 sites), slot 8 with six arguments in a loop over subresources (1 site) and slot 10 with
//     seven arguments (1 site). No site reads the return value of slots 2, 8 or 10.
//
// The runtime has no model of the console's resource layout (runtime/src/gfx10_tiling.h covers two swizzle modes of
// one 2D level, without mips or the layout structure slot 2 fills), so nothing here computes a layout:
// - XGSuggestSwizzleMode returns E_NOTIMPL and writes nothing, so the field keeps the 0 the caller put there.
// - XGCreateTextureComputer returns E_NOTIMPL, but still stores a computer in *out. The two checking callers then
//   fail cleanly; the two that do not check would otherwise call through a null pointer. COM says *out is null on
//   failure; this breaks that on purpose, because the measured callers need it.
// - Slot 0 and slot 1 count nothing (the computer is one static object). Every other slot, up to kSlots, prints its
//   index and caller once and returns E_NOTIMPL without writing its arguments, so a caller's own zero-filled outputs
//   stay zero and a stack structure the caller did not fill stays as it was.
// Hypothesis, not measured: slots 0 and 1 being AddRef and Release; the meaning of slots 2, 8 and 10; that callers
// reach no slot above kSlots - 1 (the computer is also stored in an object at one site and may be called later).
#include <windows.h>
#include <intrin.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <utility>

#include "shims.h"

namespace {

constexpr HRESULT kNotImpl = static_cast<HRESULT>(rt::shims::kENotImpl);
constexpr size_t kSlots = 32;

// Prints once per entry point; `slot` is the vtable index for a computer method, or -1 for an export.
void warn_once(std::atomic<bool>& warned, const char* what, int slot, const void* caller) {
    if (warned.exchange(true, std::memory_order_relaxed)) return;
    char name[64];
    if (slot < 0) std::snprintf(name, sizeof(name), "%s", what);
    else std::snprintf(name, sizeof(name), "%s%d", what, slot);
    std::fprintf(stderr, "[rt] xg_xs: %s: no console texture layout on the PC; E_NOTIMPL (caller %s, logged once)\n",
                 name, rt::shims::module_offset(reinterpret_cast<uint64_t>(caller)).c_str());
}

ULONG __stdcall computer_add_ref(void*) { return 1; }
ULONG __stdcall computer_release(void*) { return 1; }

template <size_t Slot>
HRESULT __stdcall computer_slot(void*) {
    static std::atomic<bool> warned{false};
    warn_once(warned, "texture computer slot ", static_cast<int>(Slot), _ReturnAddress());
    return kNotImpl;
}

template <size_t... Slot>
constexpr auto make_vtable(std::index_sequence<Slot...>) {
    return std::array<void*, sizeof...(Slot)>{
        (Slot == 0   ? reinterpret_cast<void*>(&computer_add_ref)
         : Slot == 1 ? reinterpret_cast<void*>(&computer_release)
                     : reinterpret_cast<void*>(&computer_slot<Slot>))...};
}

const auto kVtable = make_vtable(std::make_index_sequence<kSlots>{});
struct Computer {
    const void* const* vtable;
};
Computer g_computer{kVtable.data()};

HRESULT __stdcall XGCreateTextureComputer(const void* /*desc*/, void** out) {
    static std::atomic<bool> warned{false};
    warn_once(warned, "XGCreateTextureComputer", -1, _ReturnAddress());
    if (!out) return E_POINTER;
    *out = &g_computer;  // see the top of the file: two measured callers use *out without checking the HRESULT
    return kNotImpl;
}

HRESULT __stdcall XGSuggestSwizzleMode(const void* /*desc*/, uint32_t /*flags*/, uint32_t* /*mode*/) {
    static std::atomic<bool> warned{false};
    warn_once(warned, "XGSuggestSwizzleMode", -1, _ReturnAddress());
    return kNotImpl;
}

}  // namespace

SHIM_TABLE(xg_xs,
    {"xg_xs.dll", "XGCreateTextureComputer", reinterpret_cast<void*>(&XGCreateTextureComputer)},
    {"xg_xs.dll", "XGSuggestSwizzleMode", reinterpret_cast<void*>(&XGSuggestSwizzleMode)},
);
