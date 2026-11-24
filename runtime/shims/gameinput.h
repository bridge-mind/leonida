// GameInput shim on SDL3: runtime-side controls and test seams. The GameInput API itself is reached the way a title
// reaches it, through GameInputCreate in the shim registry (gameinput.dll / xgameruntime.dll). See docs/gameinput.md.
#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "../src/active_input.h"  // leonida::InputFamily (and std::optional) for Config::prompts and active_input()

namespace rt::gameinput {

// Takes a reference on a reference-counted object only while it is alive, that is while its count isn't 0. A thread
// that finds the object in a shared list (the poll thread and a device's force-feedback effects) then can't revive one
// whose last Release has already run and whose destructor is waiting for the lock to unlink it. False: leave it be.
template <class Count>
bool add_ref_if_live(std::atomic<Count>& refs) {
    Count n = refs.load();
    while (n != 0 && !refs.compare_exchange_weak(n, n + 1)) {}
    return n != 0;
}

// How the impulse-trigger motors (GameInputRumbleParams::leftTrigger/rightTrigger) reach a DualSense.
enum class DsTriggerMode { Off, Rumble, Resistance };

// The keyboard pad: one extra virtual gamepad driven by the keyboard. Focused reads keys only while a window of this
// process is in the foreground; Always reads them whatever has focus (console tools such as gameinput_probe).
enum class KeyboardPadMode { Off, Focused, Always };

struct Config {
    // LEONIDA_PROMPTS=auto|xbox|playstation|ps5|ps4|ps3|nintendo|keyboard: a family that active_input() reports
    // whatever is in use; unset or auto follows the devices.
    std::optional<leonida::InputFamily> prompts;
    uint32_t prompts_switch_ms = 300;  // LEONIDA_PROMPTS_SWITCH_MS, 0..2000: shortest time between two changes
    DsTriggerMode ds_triggers = DsTriggerMode::Rumble;  // LEONIDA_DS_TRIGGERS=off|rumble|resistance
    float ds_trigger_gain = 1.0f;                       // LEONIDA_DS_TRIGGER_GAIN, 0..4, scales the effect strength
    float mouse_sensitivity = 1.0f;                     // LEONIDA_MOUSE_SENSITIVITY, 0.1..10, scales mouse movement
    uint8_t ds_trigger_frequency = 30;                  // LEONIDA_DS_TRIGGER_FREQ, Hz for the rumble mode, 1..255
    bool lightbar_set = false;                          // LEONIDA_DS_LIGHTBAR=RRGGBB|off; unset keeps SDL's
    uint8_t lightbar[3] = {0, 0, 0};                    //   player colour
    bool rumble = true;                                 // LEONIDA_RUMBLE=0 turns all rumble off
    bool background_thread = true;                      // LEONIDA_GAMEINPUT_THREAD=0: poll only from poll()
    KeyboardPadMode keyboard_pad = KeyboardPadMode::Off;  // LEONIDA_KEYBOARD_PAD=1|focus|always|off
};

// The configuration in use. Read from the environment on the first GameInputCreate.
const Config& config();
// Re-reads the environment (tests). Takes effect for the next rumble update.
void reload_config();

// One step of the backend, on the calling thread: initialise SDL if needed, open new gamepads, close removed ones,
// take a reading from each, apply pending rumble/trigger output, and deliver queued callbacks unless a dispatcher
// owns them. With the background thread on (the default) this runs every 2 ms by itself; with it off
// (LEONIDA_GAMEINPUT_THREAD=0) a test calls it, always from the same thread, because SDL's joystick calls must be.
void poll();

// Number of gamepads currently open by the backend (tests, gameinput_probe).
int connected_count();

// The kind of device the player used last (src/active_input.h has the rules), or the one LEONIDA_PROMPTS forces.
// The title is told every pad is an Xbox One pad; this is for the runtime's own use, such as button prompt glyphs.
// Safe from any thread. `generation` goes up by one with each change of `family`, so a reader can check it every
// frame and act only when it moved. Xbox, generation 0, until the first deliberate input.
struct ActiveInput {
    leonida::InputFamily family = leonida::InputFamily::Xbox;
    uint64_t generation = 0;
    bool forced = false;  // LEONIDA_PROMPTS set the family
};
ActiveInput active_input();

// The SDL haptic calls the shim makes, one entry per call, so a test can stand in for them: SDL's virtual joysticks
// report no haptics, and a pad that does is the only other way to reach this path. `joystick` is the SDL_Joystick the
// gamepad was opened from, `haptic` what open_joystick returned, `effect` is an SDL_HapticEffect. Returning null from
// open_joystick says the pad has no force feedback, which is what a pad without it does.
struct HapticApi {
    void* (*open_joystick)(void* joystick);   // SDL_OpenHapticFromJoystick
    void (*close)(void* haptic);              // SDL_CloseHaptic
    uint32_t (*features)(void* haptic);       // SDL_GetHapticFeatures: the SDL_HAPTIC_* bits the device supports
    int (*max_effects)(void* haptic);         // SDL_GetMaxHapticEffects
    int (*create_effect)(void* haptic, const void* effect);      // SDL_CreateHapticEffect, -1 on failure
    bool (*update_effect)(void* haptic, int effect, const void* data);  // SDL_UpdateHapticEffect
    bool (*run_effect)(void* haptic, int effect, uint32_t iterations);  // SDL_RunHapticEffect
    bool (*stop_effect)(void* haptic, int effect);            // SDL_StopHapticEffect
    void (*destroy_effect)(void* haptic, int effect);         // SDL_DestroyHapticEffect
    bool (*set_gain)(void* haptic, int gain);                 // SDL_SetHapticGain, 0..100
};
// Where the haptics come from (tests). nullptr restores SDL. Applies to pads opened after the call: a pad keeps the
// haptic device it was opened with.
void set_haptic_source(const HapticApi* api);

// What the shim last sent to SDL for the device that is index `i` in connection order. Test seam.
struct OutputState {
    uint16_t low = 0, high = 0;           // SDL_RumbleGamepad
    uint16_t left = 0, right = 0;         // SDL_RumbleGamepadTriggers
    uint8_t ds_right[11] = {}, ds_left[11] = {};  // last DualSense trigger effects sent
    bool is_dualsense = false;
    bool trigger_rumble = false;          // the pad takes SDL_RumbleGamepadTriggers
    bool force_feedback = false;          // SDL opened the pad as a haptic device
    uint32_t haptic_motors = 0;           // haptic motors the pad reports (0 or 2)
    // The last force-feedback effect the shim asked SDL for: SDL_CreateHapticEffect's id (-1 when SDL refused),
    // whether SDL_RunHapticEffect then returned true, and SDL's error text when a call failed. `force_feedback` only
    // says a haptic device was opened; an effect SDL creates but will not run, or refuses outright, shows up here.
    int effect_id = -1;
    bool effect_ran = false;
    char effect_error[128] = {};
};
bool output_state(int i, OutputState* out);

// Where the keyboard pad reads keys from. Test seam: key_down(virtual_key) replaces GetAsyncKeyState, has_focus
// replaces the foreground-window check of KeyboardPadMode::Focused. nullptr restores the default for either.
using KeyDownFn = bool (*)(int virtual_key);
using HasFocusFn = bool (*)();
void set_keyboard_source(KeyDownFn key_down, HasFocusFn has_focus);

// Where the keyboard pad reads the mouse cursor from (screen pixels; false when it cannot be read). Test seam:
// replaces GetCursorPos. nullptr restores the default.
using CursorPosFn = bool (*)(int* x, int* y);
void set_cursor_source(CursorPosFn cursor_pos);

// Raw HID (RT-175, docs/gameinput.md "Raw HID"). One report a HID report descriptor declares: its kind
// (GameInputRawDeviceReportKind: 0 input, 1 output, 2 feature), its report id (0 when the descriptor declares none)
// and its payload size in bytes, without the id byte.
struct RawReportSize {
    int kind;
    uint32_t id;
    uint32_t bytes;
};
// Reads the reports out of a HID report descriptor, sorted by kind then id. Main items Input/Output/Feature add
// Report Size x Report Count bits to the current report id; Push and Pop save and restore the globals; long items are
// skipped. False on a truncated descriptor, a Report ID of 0 or above 255, a Pop without a Push, or a report over
// 64 KiB.
bool parse_report_descriptor(const uint8_t* data, size_t size, std::vector<RawReportSize>* out);

// Where the shim does a pad's HID I/O. The default is SDL's hidapi (SDL_hid_*). Report buffers follow hidapi: they
// start with the report id byte (0 for a device without ids) for write, get_feature and send_feature, and read
// returns it first only when the device uses ids. read never blocks: it returns 0 when nothing is pending. Every
// call returns a byte count, or a negative value on error.
struct HidBackend {
    void* (*open)(const char* path, uint16_t vendor, uint16_t product);  // nullptr: no HID device for this pad
    void (*close)(void* hid);
    int (*descriptor)(void* hid, uint8_t* buf, size_t size);
    int (*read)(void* hid, uint8_t* buf, size_t size);
    int (*write)(void* hid, const uint8_t* buf, size_t size);
    int (*get_feature)(void* hid, uint8_t* buf, size_t size);
    int (*send_feature)(void* hid, const uint8_t* buf, size_t size);
};
// Test seam: the backend for pads opened from now on (pads already open keep theirs until they reconnect). nullptr
// restores SDL's.
void set_hid_backend(const HidBackend* backend);

// One row of the keyboard pad's key map: a Windows virtual-key code and the SDL gamepad button it presses.
struct KeyBinding {
    int virtual_key;
    int sdl_button;  // SDL_GamepadButton
    const char* key_name;
};
// The fixed key map (docs/gameinput.md lists it). `count` receives the number of rows.
const KeyBinding* keyboard_bindings(int* count);

}  // namespace rt::gameinput
