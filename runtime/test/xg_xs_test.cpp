// Checks the registered xg_xs.dll shim against the call shapes measured in the game executable (runtime/shims/xg_xs.cpp).
//
// Both exports are resolved through rt::shims::find, so this is the table the runtime installs. The checks: both
// return E_NOTIMPL, XGSuggestSwizzleMode leaves the caller's field alone, XGCreateTextureComputer still hands out a
// computer (two measured callers use it without checking the HRESULT), and the computer's slots called at the
// measured sites (1, 2, 8, 10) return without writing what their arguments point to.
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

#include "shims.h"

namespace {

using CreateFn = HRESULT(__stdcall*)(const void*, void**);
using SuggestFn = HRESULT(__stdcall*)(const void*, uint32_t, uint32_t*);
using ReleaseFn = ULONG(__stdcall*)(void*);
using Slot2Fn = HRESULT(__stdcall*)(void*, void*);
using Slot8Fn = HRESULT(__stdcall*)(void*, void*, uint32_t, uint32_t, const void*, uint32_t, uint32_t);
using Slot10Fn = HRESULT(__stdcall*)(void*, uint32_t*, void*, void*, uint32_t*, uint32_t, void*);
constexpr HRESULT kNotImpl = static_cast<HRESULT>(rt::shims::kENotImpl);

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

bool all_bytes(const uint8_t* p, size_t n, uint8_t value) {
    for (size_t i = 0; i < n; ++i)
        if (p[i] != value) return false;
    return true;
}

}  // namespace

int main() {
    auto create = reinterpret_cast<CreateFn>(rt::shims::find("xg_xs.dll", "XGCreateTextureComputer"));
    auto suggest = reinterpret_cast<SuggestFn>(rt::shims::find("xg_xs.dll", "XGSuggestSwizzleMode"));
    check(create != nullptr, "XGCreateTextureComputer is registered");
    check(suggest != nullptr, "XGSuggestSwizzleMode is registered");
    check(rt::shims::owns_dll("xg_xs.dll"), "the registry owns xg_xs.dll");
    if (!create || !suggest) return 1;

    uint8_t desc[0x38] = {};
    uint32_t mode = 0x1234;
    check(suggest(desc, 0, &mode) == kNotImpl, "XGSuggestSwizzleMode returns E_NOTIMPL");
    check(mode == 0x1234, "XGSuggestSwizzleMode leaves the mode field as the caller set it");

    check(create(desc, nullptr) == E_POINTER, "XGCreateTextureComputer(desc, nullptr) is E_POINTER");
    void* computer = nullptr;
    check(create(desc, &computer) == kNotImpl, "XGCreateTextureComputer returns E_NOTIMPL");
    check(computer != nullptr, "XGCreateTextureComputer still stores a computer for the callers that do not check");
    if (!computer) return 1;

    void* const* vtable = *static_cast<void* const* const*>(computer);
    uint8_t layout[0x800];
    std::memset(layout, 0xcd, sizeof(layout));
    check(reinterpret_cast<Slot2Fn>(vtable[2])(computer, layout) == kNotImpl, "slot 2 returns E_NOTIMPL");
    check(all_bytes(layout, sizeof(layout), 0xcd), "slot 2 writes nothing to its output");

    uint8_t dst[64];
    std::memset(dst, 0xcd, sizeof(dst));
    const uint8_t src[64] = {};
    check(reinterpret_cast<Slot8Fn>(vtable[8])(computer, dst, 0, 0, src, 16, 64) == kNotImpl,
          "slot 8 returns E_NOTIMPL");
    check(all_bytes(dst, sizeof(dst), 0xcd), "slot 8 writes nothing to its destination");

    uint32_t count = 0, inout = 0x10;
    uint8_t a[12] = {}, b[12] = {}, list[0xc0] = {};
    check(reinterpret_cast<Slot10Fn>(vtable[10])(computer, &count, a, b, &inout, 0, list) == kNotImpl,
          "slot 10 returns E_NOTIMPL");
    check(count == 0 && inout == 0x10 && all_bytes(a, 12, 0) && all_bytes(b, 12, 0) &&
              all_bytes(list, sizeof(list), 0),
          "slot 10 leaves the caller's zero-filled outputs as they were");
    check(reinterpret_cast<Slot2Fn>(vtable[31])(computer, layout) == kNotImpl, "the last slot (31) is traced too");

    auto release = reinterpret_cast<ReleaseFn>(vtable[1]);
    release(computer);
    release(computer);
    check(reinterpret_cast<Slot2Fn>(vtable[2])(computer, layout) == kNotImpl,
          "the computer stays usable after more Releases than it was handed out");

    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
