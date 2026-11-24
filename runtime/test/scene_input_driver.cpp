// leonida_scene_input.dll: the test-only shim module that the scene guest (guest_scene_state.h) loads to feed its
// input trace. CMake links it to gta6rt only for the opt-in LEONIDA_GUEST_SCENE build.
//
// SetTraceKeys(frame) does three things on the caller's thread: it points the keyboard pad at trace_keys(frame)
// (guestscene::kNoFrame means every key is up), it pins the cursor so the right stick stays at rest, and it runs one
// rt::gameinput::poll(). The guest's next reading is therefore that frame's. Nothing reaches the OS input queue: the
// seams replace GetAsyncKeyState and GetCursorPos inside this process only.
// SetTraceKeys returns 1. It returns 0 when:
//   - the backend is not in the mode the trace needs (LEONIDA_KEYBOARD_PAD=always, LEONIDA_GAMEINPUT_THREAD=0),
//   - the frame is out of range, or
//   - a second thread calls it (SDL's joystick calls must stay on one thread).
//
#include <windows.h>

#include <cstdint>

#include "guest_scene_trace.h"
#include "../shims/shims.h"
#include "../shims/gameinput.h"

namespace {

uint32_t g_frame = guestscene::kNoFrame;  // the frame whose keys the pad reads
DWORD g_thread = 0;                       // poll()'s thread, fixed by the first call

bool scene_key_down(int virtual_key) { return guestscene::trace_key_down(g_frame, virtual_key); }
bool always_focused() { return true; }
bool fixed_cursor(int* x, int* y) {  // a cursor that never moves keeps the right stick at rest
    if (!x || !y) return false;
    *x = 0;
    *y = 0;
    return true;
}

}  // namespace

extern "C" int WINAPI SetTraceKeys(uint32_t frame) {
    const rt::gameinput::Config& c = rt::gameinput::config();
    if (c.background_thread || c.keyboard_pad != rt::gameinput::KeyboardPadMode::Always) return 0;
    if (frame != guestscene::kNoFrame && frame >= guestscene::kFrames) return 0;
    const DWORD self = GetCurrentThreadId();
    if (g_thread == 0) g_thread = self;
    if (g_thread != self) return 0;
    g_frame = frame;
    rt::gameinput::set_keyboard_source(&scene_key_down, &always_focused);
    rt::gameinput::set_cursor_source(&fixed_cursor);
    rt::gameinput::poll();
    return 1;
}

SHIM_TABLE(scene_input, {"leonida_scene_input.dll", "SetTraceKeys", reinterpret_cast<void*>(&SetTraceKeys)});
