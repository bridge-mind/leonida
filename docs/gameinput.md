# Controller input: the GameInput shim on SDL3

The Xbox build reads controllers through Microsoft's GameInput API. `runtime/shims/gameinput.cpp` answers
`GameInputCreate` with its own implementation of that API, backed by [SDL3](https://libsdl.org) gamepads, so
every controller SDL supports works the way an Xbox pad would: DualSense, DualShock 4, Switch Pro, Xbox pads, and the
rest of SDL's controller database. On a DualSense the impulse-trigger rumble becomes adaptive-trigger effects, and
the lightbar colour can be set.

Without the shim the loader would bind `GameInputCreate` to the host's `GameInput.dll` if there is one
([#279](https://github.com/ItsNotRexed/Leonida/pull/279)). That works for Xbox pads only, and gives no DualSense
features. That path is tested without the game: `gameinput_host_test` (ctest `gameinput_host`) maps a synthetic DLL
built against the SDK's `gameinput.lib` and checks that its `GameInput.dll` import binds to the host's copy, that
`GameInputCreate` and `GetCurrentTimestamp` work from mapped code, and that `GetCurrentReading` answers; with a pad
attached, `GetGamepadState` must succeed on the reading. It passes on Windows 11 build 26300 (GameInput 0.2309) without
a pad attached. The test skips only where Windows cannot load `GameInput.dll`; with the DLL present, an import that binds to nothing fails it. The game executable is not available yet,
so whether it imports GameInput this way, and under which
module, is unverified (see [VERIFY](#verify)).

## Building

The shim is optional, and the build downloads nothing unless asked. SDL3 is zlib-licensed. CMake builds the shim when
SDL3 is available in one of these ways:

- **Installed SDL3**, 3.2 or newer, found by `find_package(SDL3)`: set `SDL3_DIR` or `CMAKE_PREFIX_PATH`, for example
  `-DSDL3_DIR=C:/SDL3/cmake`. No flag is needed.
- **`-DLEONIDA_SDL3_FETCH=ON`**: downloads the pinned release `SDL3-3.4.20.tar.gz` from SDL's GitHub releases once at
  configure time, checks its SHA-256, and builds it static with only the gamepad parts (no audio, video, GPU or
  render). That takes about a minute on the first build. `scripts/build_runtime.ps1 -FetchSdl3` passes it.
- **A local copy of that tarball**: extract it and pass `-DFETCHCONTENT_SOURCE_DIR_SDL3=<path to SDL3-3.4.20>`; it is
  built the same way, without a download.

Without any of these, or with `-DLEONIDA_SDL3=OFF`, configure prints a message and `gta6rt` is built without the shim:
it has no `GameInputCreate`, as before, so the loader binds it to the host's `GameInput.dll` or the unimplemented
thunk. `gameinput_test` and `gameinput_probe` are not built, and CTest lists `gameinput` as not run (disabled) rather
than failing.

`scripts/build_runtime.ps1` (what `make_game.py` runs) builds the shim only on its CMake path, when Visual Studio's
C++ CMake tools are installed. Its direct-`cl.exe` fallback compiles a fixed file list without the shim and warns
about it.

## Mapping

GameInput's buttons are positional, as on an Xbox pad, and SDL's are too, so the face buttons go by position, not by
label:

| GameInput | SDL3 | DualSense | Switch Pro |
|---|---|---|---|
| `A` | south | Cross | B |
| `B` | east | Circle | A |
| `X` | west | Square | Y |
| `Y` | north | Triangle | X |
| `View` | back | Create | Minus |
| `Menu` | start | Options | Plus |
| shoulders, stick clicks, d-pad | same | L1/R1, L3/R3, d-pad | L/R, stick clicks, d-pad |
| guide button callback | guide | PS | Home |

The DualSense touchpad click, mute button and paddles are not in GameInput's gamepad state; they are in the
controller-button array (`GetControllerButtonState`, indexed in SDL's `SDL_GamepadButton` order).

- Sticks: -1..1 with +Y up (SDL reports +Y down; the shim flips it). Triggers: 0..1. No deadzone is applied, as on
  console: the title applies its own.
- `GetUiNavigationState` is derived from the pad: A accept, B cancel, X/Y context 1/2, stick clicks context 3/4, d-pad
  or left stick past half for up/down/left/right, shoulders for page left/right, triggers past half for page up/down,
  right stick past half for scrolling, Menu and View as themselves.
- Controller axes (`GetControllerAxisState`): LX, LY, RX, RY, LT, RT in 0..1, sticks at rest at 0.5.
- Every pad is reported as family `GameInputFamilyXboxOne` with its real USB vendor and product id.
- Player LEDs (and SDL's default lightbar colour) follow connection order: the first pad is player 1, and a pad
  plugged in later takes the lowest free slot.

## Keyboard pad

The title reads gamepads only, so keyboard input has to arrive as a gamepad. With `LEONIDA_KEYBOARD_PAD` set, the
shim attaches one SDL virtual gamepad named `Keyboard pad (Leonida)` and presses its buttons from the keyboard on
every poll. It then goes through the same path as a real pad: a device object, the next free player slot, readings,
reading and device callbacks, and UI navigation derived from its buttons. It reports no rumble motors. It is off by
default.

| key | button |
|---|---|
| arrow keys | d-pad |
| Enter | `A` |
| Backspace | `B` |
| X | `X` |
| Y | `Y` |
| Q / E | left / right shoulder |
| Esc | `Menu` |
| Tab | `View` |
| W / A / S / D | left stick up / left / down / right |
| mouse movement | right stick |
| right mouse button | left trigger (aim) |
| left mouse button | right trigger (fire) |

Letters are Windows virtual-key codes, so `Y` is the key labelled Y on the active layout (it sits where Z does on a
QWERTY keyboard with a German layout). Mouse buttons are the physical ones: `GetAsyncKeyState` ignores the
swapped-buttons setting.

WASD pushes the left stick to full deflection. Opposite keys cancel (W+S is centered on Y), and a diagonal sits on
the round gate as a real stick reports it: about 0.707 on each axis, not 1.

The right stick follows the mouse's movement, not the cursor's position. Each poll (2 ms with the background thread)
the stick keeps 7/8 of its value and adds 1024 axis units (of 32767, times `LEONIDA_MOUSE_SENSITIVITY`) per count the mouse moved since the last poll,
clamped to full. Steady movement of 4 counts a poll (2000 a second) is full deflection; when the mouse stops, the
stick falls back from full to exactly 0 in 67 polls (134 ms). Moving the mouse down pushes the stick down.

The movement comes from raw input (`WM_INPUT`): what the mouse itself reports, so it does not stop at the edge of the
screen and Windows' pointer acceleration does not change it. When the keyboard pad attaches, the shim starts one
thread with a message-only window (never shown) registered for raw mouse input (`RIDEV_INPUTSINK`), which adds each
relative movement to a running total; the poll takes the difference of that total between polls. Absolute reports
(remote desktop, some virtual machines, pen tablets) are ignored. If the window or the registration fails, the shim
logs it and falls back to `GetCursorPos` deltas, which stop at the screen edge. A count is one step of the mouse's
sensor, about a pixel at Windows' default pointer speed without acceleration, so the feel depends on the mouse's DPI.

With raw input on and a window of this process in the foreground, the cursor is kept inside that window's client
area (`ClipCursor`, re-applied every 100 ms) so that a click cannot land on another window; it is released when focus
moves away or the keyboard pad is turned off. The cursor stays visible: hiding it has to happen on the thread that
owns the game's window, which is the title's, not the shim's.

Keys are read with `GetAsyncKeyState`, which needs no window or message loop (SDL's keyboard is part of its video
subsystem, which this build leaves out) and works under Wine. With `LEONIDA_KEYBOARD_PAD=1` keys count only while a
window of this process is in the foreground, so typing in another application presses nothing, and every button,
stick and trigger is released when focus moves away (the cursor is read afresh when focus returns, so the right
stick does not jump). `LEONIDA_KEYBOARD_PAD=always` reads them whatever has focus; a console tool such as
  `gameinput_probe` needs it, because its console window belongs to the terminal, not to the tool.
- Force feedback and haptics run through a fake haptic backend (`rt::gameinput::set_haptic_source`), because SDL's
  virtual joysticks report no haptics: the motor count, axes and effect flags a wheel reports, and a pad without SDL
  haptics reporting no motor at all; `CreateForceFeedbackEffect` and its `E_INVALIDARG` cases; a constant effect
  becoming `SDL_HAPTIC_CONSTANT` at the right level, not re-sent while unchanged, updated once when the parameters
  change; the effect's gain and the motor's device gain; a 50 Hz sine's period, phase, length and play count; a
  spring's coefficients, saturations, dead zone and centre; a finished effect reporting stopped; `Paused` stopping it;
  `LEONIDA_RUMBLE=0` silencing it and the mute lifting; the SDL effect destroyed when the title releases it; and, on a
  pad with trigger motors, the two haptic motors, `SetHapticMotorState`'s intensity, deadline, `playCount` 0 and null,
  and the trigger rumble sharing the one message.

To try it by hand: `set LEONIDA_KEYBOARD_PAD=always`, then `gameinput_probe --seconds 20` and press keys.

## Rumble and triggers

`IGameInputDevice::SetRumbleState` takes four motor levels, 0..1. The two body motors go to `SDL_RumbleGamepad` on
every pad that has rumble. The two trigger motors depend on the pad:

- Xbox One and Series pads (and anything else SDL reports trigger rumble for): `SDL_RumbleGamepadTriggers`, the same
  impulse triggers as on console.
- DualSense: the trigger level becomes an adaptive-trigger effect sent with `SDL_SendGamepadEffect`, chosen by
  `LEONIDA_DS_TRIGGERS`.
- Others (DualShock 4, Switch Pro): the trigger levels are dropped and the device reports no trigger motors.

**This is an approximation.** The Xbox title only sends an impulse-trigger vibration level per frame. The PS5 version
of a game has effects authored per weapon and vehicle (weapon resistance, a click at a trigger position, and so on);
none of that exists in the Xbox build, so the shim can only turn the level it gets into a generic effect. Expect it to
feel like the Xbox signal played through the DualSense's trigger motors, not like the PS5 release.

The DualSense effect is SDL's 47-byte DualSense effects block (`DS5EffectsState_t` in SDL's `SDL_hidapi_ps5.c`, the
format `test/testcontroller.c` uses): byte 0 holds the valid flags `0x04 | 0x08` (right and left trigger effect),
bytes 10-20 the right trigger effect and bytes 21-31 the left one. The effect modes are the three SDL's test program
uses:

| `LEONIDA_DS_TRIGGERS` | level 0 | level > 0 |
|---|---|---|
| `rumble` (default) | `05` (effect off) | `06 <freq> <level*255> 00`: vibration at `LEONIDA_DS_TRIGGER_FREQ` Hz from the start of the pull |
| `resistance` | `05` | `01 00 <level*255>`: constant resistance over the whole pull |
| `off` | nothing is sent; the triggers stay as they are | |

An effect is sent only when its bytes change, so a steady level costs one report.

## Force feedback and haptics

`IGameInputDevice` reports the motors a pad really has, and drives them. There are two kinds, and they come from two
places:

| GameInput | comes from | the shim reports |
|---|---|---|
| `forceFeedbackMotorCount`, `forceFeedbackMotorInfo` | SDL opened the pad's joystick as a haptic device (`SDL_OpenHapticFromJoystick`), which it does for a wheel or a stick, never for a gamepad | one motor at index 0: `supportedAxes` LinearX, `maxSimultaneousEffects` SDL's limit, and one flag per effect kind from SDL's feature bits |
| `hapticFeedbackMotorCount`, `hapticFeedbackMotorInfo` | the trigger motors the pad already has for rumble: Xbox impulse triggers, or a DualSense with `LEONIDA_DS_TRIGGERS` on | two motors, left and right trigger, one waveform each |

A pad with neither keeps the answer the shim gave before this existed: no force-feedback or haptic motors,
`CreateForceFeedbackEffect` `E_INVALIDARG`, `IsForceFeedbackMotorPoweredOn` false, `SetHapticMotorState` `E_INVALIDARG`.
The keyboard pad is one of those, and so is any pad SDL opens as a gamepad only.

### Force feedback: `CreateForceFeedbackEffect`

`IGameInputForceFeedbackEffect` is the title's object; SDL is touched only on the poll thread. `SetParams`, `SetGain`
and `SetState` record the change and bump a version, and the next poll creates, updates, runs or stops the SDL effect,
so a title call still costs a lock and a copy. GameInput's eleven effect kinds are SDL's eleven force-feedback effects:

| GameInput kind | SDL type | fields |
|---|---|---|
| `Constant` | `SDL_HAPTIC_CONSTANT` | `level` = magnitude × gain |
| `Ramp` | `SDL_HAPTIC_RAMP` | `start` and `end` from the start and end magnitude |
| `SineWave`, `SquareWave`, `TriangleWave`, `SawtoothUpWave`, `SawtoothDownWave` | `SDL_HAPTIC_SINE`, `SQUARE`, `TRIANGLE`, `SAWTOOTHUP`, `SAWTOOTHDOWN` | `period` = 1000/frequency in ms, `magnitude`, `offset` = bias, `phase` = the effect's degrees × 100 |
| `Spring`, `Friction`, `Damper`, `Inertia` | `SDL_HAPTIC_SPRING`, `FRICTION`, `DAMPER`, `INERTIA` | positive and negative coefficient, saturation and dead zone on all three axes, `center` from the magnitude |

- Only the X magnitude is read, because the motor reports LinearX: the axis a wheel's or a stick's motor acts on. An
effect for another axis is played on this one, not dropped.
- `SetForceFeedbackMotorGain` is SDL's device gain (`SDL_SetHapticGain`, 0..100) and scales every effect on the motor;
`IGameInputForceFeedbackEffect::SetGain` scales that one effect's levels. Both are clamped to 0..1.
- The envelope: attack and release become SDL's `attack_length` and `fade_length` with the gains as their levels,
`sustainGain` scales the magnitude, attack + sustain + release is SDL's `length`, and `playCount` is the iteration count,
where 0 is "until stopped". No duration at all is a 60 s SDL length re-run every 20 s — the same refresh the rumble
motors get — so an effect with no end keeps playing without being restarted every poll. An effect that plays its
length reports `GameInputFeedbackStopped` afterwards.
- An all-zero envelope (what a title that only sets a magnitude leaves behind) means "no envelope", not silence, so
the effect plays at its own magnitude rather than at gain 0.
- A new effect starts `GameInputFeedbackRunning`: `CreateForceFeedbackEffect` already carried its parameters, and the
documented use — create once, `SetParams` per frame — never reaches `SetState`. `GameInputFeedbackPaused` stops the
SDL effect, because SDL pauses the device where GameInput pauses one effect.
- `LEONIDA_RUMBLE=0` stops every effect, like every other output path.
- The kind is checked against the motor's flags when the effect is created, so a kind the pad does not list is
`E_INVALIDARG`. SDL is not asked there — that is a title call — so if SDL still refuses an effect the poll thread sends
nothing for it rather than guessing.

### Haptics: `SetHapticMotorState`

The trigger motors play one waveform each: an intensity, which is the one thing SDL lets the shim send on a trigger,
through the same output as the trigger rumble — `SDL_RumbleGamepadTriggers`, or the DualSense effect block. `duration`
and `repeatDelay` are microseconds, `playCount` 0 plays until the state is set again, and a null `params` stops the
motor. The trigger rumble level and the haptic intensity share that one message, the stronger of the two winning, so a
title mixing them does not get one cancel the other.

**This is an approximation too, and a coarser one than the rumble.** A real haptic motor has a table of waveforms, each
with its own HID usage; SDL exposes no such table — a DualSense takes one effects block and everything else takes a
trigger impulse — so the shim reports a single waveform per motor (an intensity, usage 0x0F/0x58) instead of the pad's
real ones, and plays any request as an intensity. A title that picks waveform 3 by name cannot be served beyond
intensity 0; waveform indices other than 0 are `E_INVALIDARG`.

## Motion

Pads whose SDL gamepad reports both a gyroscope and an accelerometer (a DualSense, a Switch Pro, Steam Input pads) get
`GameInputKindMotion` in `supportedInput` and a `motionInfo` (`GameInputMotionInfo`, Microsoft's v0 layout); `IGameInputReading::GetMotionState`
returns a `GameInputMotionState` on their readings, and `GetCurrentReading`/`GetNextReading` answer a
`GameInputKindMotion` query for such a pad and return `GAMEINPUT_E_READING_NOT_FOUND` for a pad without sensors. A
reading callback registered for `GameInputKindMotion` hears only such pads: like a device callback, it is limited to
devices that support at least one of the kinds it names (Microsoft's `RegisterReadingCallback` reference). The
sensors are enabled when the pad is opened, at SDL's requested rate, and every poll reads the latest sample.

- The state carries **signed fractions of the maxima `motionInfo` reports** (1.0 = full magnitude), which is the
  convention Microsoft documents for `GameInputMotionState`. SDL delivers accel in m/s² (gravity included) and gyro
  in rad/s, so the shim divides by the maxima. The maxima are this shim's choice — SDL does not expose a sensor
  range — and are 2 g (19.6133 m/s²) and 2000 deg/s (34.9066 rad/s), a DualSense's accelerometer and gyro ranges
  (VERIFY against the game and a real pad).
- A pad at rest sits at about 0.5 on the gravity axis and 0.0 on all gyro axes. SDL answers sensor reads with zeros
  before the device's first sample arrives; since a real accelerometer never reads exactly zero (gravity), all-zero
  accel means "no sample yet" and the reading carries no motion at all — `GetMotionState` returns false.
- SDL's gamepad API has no compass and the shim synthesises no orientation quaternion, so the magnetic-field and
  orientation members are zero with `GameInputMotionAccuracyUnknown`.
- A pad with sensors streams a reading every ~8 ms (~125 Hz) even when nothing else changes, so motion is not stuck
  at the last button event; pads without sensors keep the old change-driven behaviour. The 8 ms window and the 512-
  reading history cap bound the cost.
- `gameinput_probe` prints the accel and gyro fractions when the pad has sensors (a resting pad shows ~0.5 g).

## Active input

The title is told that every pad is an Xbox One pad, so it cannot know what the player is holding. The shim keeps
that record itself, for the runtime's own use, such as button prompt glyphs that follow the device in use:
`rt::gameinput::active_input()` returns the family of the device used last, a generation that goes up by one with each
change, and whether `LEONIDA_PROMPTS` forced it. It needs no lock and can be read from any thread, every frame.

| SDL gamepad type | family |
|---|---|
| `PS5` (DualSense, DualSense Edge) | PlayStation 5 |
| `PS4` (DualShock 4) | PlayStation 4 |
| `PS3` (DualShock 3) | PlayStation 3 |
| `NINTENDO_SWITCH_PRO`, the Joy-Con types | Nintendo |
| the [keyboard pad](#keyboard-pad) | keyboard and mouse |
| everything else (Xbox One/Series, Xbox 360, generic, unknown) | Xbox |

SDL takes the type from the pad's vendor and product id. Only deliberate input changes the answer
(`runtime/src/active_input.h`, tested by `active_input_test` without SDL):

- an edge: a button going down (any of SDL's, the DualSense touchpad included), a trigger rising past 0.3, or a stick
  leaving a radius of 0.35. A held button, trigger or stick, stick drift inside the radius and analog noise never
  count. The mouse moves the keyboard pad's right stick, so a real mouse movement counts through the same rule and a
  jitter of a count or two does not;
- a pad's first reading after it connects is its baseline, so connecting a pad with a button held, or with a worn
  stick resting past the radius, changes nothing;
- after a change, the next one waits `LEONIDA_PROMPTS_SWITCH_MS` (300 ms by default). Input from another family inside
  that time is ignored, not queued, so two devices used at once cannot make the answer flicker;
- input from a second device of the family already in use changes nothing.

Before the first deliberate input the answer is Xbox. Each change logs one line, for example
`input in use: PlayStation 5 (DualSense Wireless Controller, button press)`. `LEONIDA_PROMPTS` forces a family (a
line is logged when it takes effect); tracking goes on underneath, so clearing it returns at once to the device in use.

## Environment variables

| variable | default | effect |
|---|---|---|
| `LEONIDA_DS_TRIGGERS` | `rumble` | DualSense trigger effects: `off`, `rumble` or `resistance` (above) |
| `LEONIDA_DS_TRIGGER_GAIN` | `1` | scales the trigger level before it becomes an effect, 0..4 |
| `LEONIDA_DS_TRIGGER_FREQ` | `30` | vibration frequency of the `rumble` mode, 1..255 |
| `LEONIDA_DS_LIGHTBAR` | unset | `RRGGBB` (hex) sets the DualSense/DualShock 4 lightbar when the pad connects; `off` turns it dark; unset keeps SDL's player colour |
| `LEONIDA_RUMBLE` | `1` | `0` turns all rumble, trigger and force-feedback output off |
| `LEONIDA_PROMPTS` | `auto` | forces the [active input](#active-input) family: `xbox`, `playstation` (or `ps5`), `ps4`, `ps3`, `nintendo` or `keyboard`; `auto` follows the devices |
| `LEONIDA_PROMPTS_SWITCH_MS` | `300` | shortest time between two changes of the active input family, 0..2000 ms |
| `LEONIDA_GAMEINPUT_THREAD` | `1` | `0` stops the background poll thread; only `rt::gameinput::poll()` then reads the pads (tests) |
| `LEONIDA_KEYBOARD_PAD` | `off` | `1` adds the [keyboard pad](#keyboard-pad), read while this process has focus; `always` reads keys whatever has focus |
| `LEONIDA_MOUSE_SENSITIVITY` | `1` | scales how far the mouse moves the keyboard pad's right stick, 0.1..10 |

SDL's own hints (`SDL_JOYSTICK_*`, see SDL's `SDL_hints.h`) also apply, for example `SDL_JOYSTICK_HIDAPI_PS5=0` to
leave a DualSense to the generic driver, or `SDL_GAMECONTROLLERCONFIG` to add a mapping for an unknown pad.

## How it runs

One thread owns SDL: a background thread started by the first `GameInputCreate` that polls every 2 ms. It opens new
pads, closes removed ones, takes a reading of each, and applies rumble, haptic and force-feedback requests. Title calls
never touch SDL: they read the latest reading or leave a request, so a call costs a lock and a copy. Callbacks (device, reading, guide
button) run on that thread, as GameInput runs them on its own threads. Once the title creates a dispatcher
(`IGameInput::CreateDispatcher`), callbacks are queued instead, the dispatcher's wait handle is signalled, and
`IGameInputDispatcher::Dispatch` runs them on the title's thread, oldest first, within its quota: at least one queued
event per call (a quota of 0 runs exactly one), no further event once the quota has passed, and only the events
queued when the call started (later ones wait for the next call, so even an unlimited quota returns). The wait
handle is signalled again while events remain.

The API surface:

| interface | implemented | not implemented |
|---|---|---|
| `IGameInput` | `GetCurrentTimestamp` (µs), `GetCurrentReading`, `GetNextReading`, `GetPreviousReading`, `RegisterReadingCallback`, `RegisterDeviceCallback` (blocking and async enumeration), `RegisterGuideButtonCallback`, `RegisterKeyboardLayoutCallback` (never fires), `StopCallback`, `UnregisterCallback`, `CreateDispatcher`, `FindDeviceFromId`, `EnableOemDeviceSupport` (no-op), `SetFocusPolicy` (no-op) | `GetTemporalReading` (`E_NOTIMPL`, as in GameInput v0 itself; see below), `CreateAggregateDevice`, `FindDeviceFromObject/PlatformHandle/PlatformString` |
| `IGameInputReading` | kind, sequence number, timestamp, device, gamepad state, UI navigation state, controller axes and buttons, motion state, raw report (pads opened through HID, see [Raw HID](#raw-hid)) | keyboard, mouse, touch, arcade/flight stick, racing wheel (all return nothing) |
| `IGameInputDevice` | device info, status, battery, `SetRumbleState`, force-feedback motors with `CreateForceFeedbackEffect`/`IGameInputForceFeedbackEffect`, `IsForceFeedbackMotorPoweredOn`, `SetForceFeedbackMotorGain`, haptic trigger motors with `SetHapticMotorState`, `CreateRawDeviceReport`, `GetRawDeviceFeature`, `SetRawDeviceFeature`, `SendRawDeviceOutput` (pads opened through HID) | `SendRawDeviceOutputWithResponse`, `ExecuteRawDeviceIoControl`, exclusive raw access, power off, input synchronization (no-ops) |
| `IGameInputRawDeviceReport` | device, report info (kind, id, size), raw data size, get and set raw data | item values (`Get/Set/ResetItemValue`, `ResetAllItems` answer false: no item tables) |
| `IGameInputReading` | kind, sequence number, timestamp, device, gamepad state, UI navigation state, controller axes and buttons, motion state | raw report, keyboard, mouse, touch, arcade/flight stick, racing wheel (all return nothing) |
| `IGameInputDevice` | device info, status, battery, `SetRumbleState`, force-feedback motors with `CreateForceFeedbackEffect`/`IGameInputForceFeedbackEffect`, `IsForceFeedbackMotorPoweredOn`, `SetForceFeedbackMotorGain`, haptic trigger motors with `SetHapticMotorState` | raw device I/O, power off, input synchronization (no-ops) |
| `IGameInputDispatcher` | `Dispatch` (with its quota), `OpenWaitHandle` | |

`CreateAggregateDevice` takes exactly one of the keyboard, mouse, gamepad, arcade stick, flight stick or racing wheel
kinds (none, several, or another kind is `E_INVALIDARG`) and returns one device per kind, the same object on every
call: family `GameInputFamilyAggregate`, always connected, with that kind as its `supportedInput`, and a fixed
`deviceId` per kind that `FindDeviceFromId` also answers. Readings asked
through it (`GetCurrentReading`, `GetNextReading`, `GetPreviousReading`) come from every connected device of that
kind (`GetCurrentReading` answers the most recent of them); asking it for another kind is
`GAMEINPUT_E_READING_NOT_FOUND`. A reading or device callback registered with an aggregate device as its filter is
refused with `E_NOTIMPL` and an invalid token.

A device object stays alive for the whole process. When a pad is unplugged its status drops to
`GameInputDeviceNoStatus` and its device callback runs; when the same pad is plugged in again (matched by SDL GUID and
serial or device path) the title gets the same object back, connected again.

## Raw HID

A title can read a pad's raw HID input reports and send it output and feature reports (RT-175). The shim opens each
pad's HID device through SDL's hidapi (`SDL_hid_*`), by the device path SDL reports for the pad. When that path is not
a HID path, it falls back to the pad's vendor and product id, but only when exactly one game-pad collection (usage page
1, usage 5) has them, so twins never mix. It reads the report descriptor to find every input, output and feature
report: its id (0 when the descriptor declares none) and its payload size in bytes, without the id byte.

A pad opened this way gets the following:

- `GameInputKindRawDeviceReport` in `supportedInput` and `GameInputDeviceRawIoEnabled` in its status.
- `inputReportInfo`, `outputReportInfo` and `featureReportInfo` tables with their counts. Every report has `itemCount`
  0: the descriptor's items (usages, logical ranges, units) are not decoded.
- Readings that carry the newest raw input report. Each poll drains the reports that arrived, the last declared one
  wins, and a report with new bytes makes a reading of its own. Before the first report, a reading carries none.
- These device calls, each on any `IGameInputRawDeviceReport` of the right kind whose id the pad declares:
  - `CreateRawDeviceReport(id, kind)` makes a zeroed report.
  - `GetRawDeviceFeature(id)` reads a feature report.
  - `SetRawDeviceFeature` sends one.
  - `SendRawDeviceOutput` writes an output report.

  An undeclared id or kind is `E_INVALIDARG`, and a failed transfer `E_FAIL`.

`SetRawData` takes a buffer of exactly the report's size and refuses any other size, so a report is never half old and
half new. The item calls answer false. `SendRawDeviceOutputWithResponse`, `ExecuteRawDeviceIoControl` and exclusive
access still return `E_NOTIMPL` / false.

A pad without such a HID device keeps the old answers: no raw kind, no tables, `E_NOTIMPL` from the raw calls.

- **Xbox pads on Windows** are read by SDL through XInput (path `XInput#n`). SDL's hidapi skips their XInput HID
  interface (`&IG_` in the path), noting that it carries no full HID reports. So they have no raw I/O here, and the log
  says so.
- **SDL virtual joysticks and the keyboard pad** have no path, so no HID device either.
- **Pads SDL reads through hidapi** can have raw I/O: DualSense, DualShock 4, Switch Pro, and most others in DirectInput
  mode.

`rt::gameinput::set_hid_backend` replaces the hidapi backend (tests). `rt::gameinput::parse_report_descriptor` is the
descriptor walk: main items Input, Output and Feature add Report Size × Report Count bits to the current Report ID, Push
and Pop save and restore the globals, and long items are skipped. It refuses a truncated item, Report ID 0 or above
255, a Pop without a Push, and a report over 64 KiB.

## Reading history

Each pad keeps the readings the poll took for it, one per change it saw, so `GetNextReading` and `GetPreviousReading`
can step through input the title has not read yet, such as a button pressed and released between two of its frames.
A reading leaves the history once the reading after it is more than 500 ms old, so the current state stays reachable
however long the pad then idles. A pad keeps at most 512 readings, and none once it is unplugged. A reading the title
holds stays valid after it leaves the history.

- With a device the search covers that pad; with none, every connected pad. Readings of different pads are ordered by
  when the shim took them, not by timestamp, since pads polled together share one.
- `GetNextReading` returns the first reading after the reference, `GetPreviousReading` the last one before it.
- The horizon is the newest reading dropped from the histories searched (with no device, the reference's pad counts
  too, even unplugged). A step that would cross it fails rather than skip input.

| situation | result |
|---|---|
| output null; reference null | `E_POINTER`; `E_INVALIDARG` |
| no supported kind in the mask | `GAMEINPUT_E_READING_NOT_FOUND` |
| the device given is unplugged | `GAMEINPUT_E_DEVICE_DISCONNECTED` |
| next: the horizon is after the reference; previous: it is not before it | `GAMEINPUT_E_REFERENCE_READING_TOO_OLD` |
| nothing newer, or nothing older after the horizon | `GAMEINPUT_E_READING_NOT_FOUND` |

After a pad is plugged back in, its last reading before the unplug leads to the reconnect snapshot, older ones are
`REFERENCE_READING_TOO_OLD`, and sequence numbers carry on. Timestamps are poll times, so a change shorter than one poll
(2 ms with the thread) is not seen.

### Device identity

`AppLocalDeviceId` is the SDL GUID in bytes 0-15 and an FNV-1a hash of the serial or path in bytes 16-31. Pads that
SDL cannot tell apart, such as two of the same model with no serial and no path, would share that id, and
`FindDeviceFromId` would return the first for both. They are twins: the first keeps the id, so a single pad's id is
unchanged, and each further one gets an id hashed from its twin number (`t:<n>:` before the serial or path). A new id
is checked against every device seen in the process and moved to the next number if taken, so no two devices share
one. A pad that comes back takes the lowest-numbered disconnected twin's object and id; a connected device is never
reused.

That is assignment by order, not identification. With one twin unplugged, the right object comes back if the same pad
returns, but another identical pad plugged in instead gets it too. With two or more unplugged, nothing tells them
apart: if they return in a different order, the objects and ids are swapped between the physical pads, and the shim
cannot detect it. The same sequence of plugs gives the same ids in every run, since the ids depend only on the GUID,
the serial or path and the order; for pads already connected at start, that order is SDL's (see [VERIFY](#verify)).

## ABI

`runtime/shims/gameinput_abi.h` declares the v0 GameInput API as Leonida's own code: the interfaces in vtable order,
the enums the shim uses and the structs it has to lay out. It was written against Microsoft's public v0 header (the
GDK version, also published as the `Microsoft.GameInput` NuGet package 0.2303.22621.3038). The header is not in the
repository. Its struct offsets were checked by compiling both side by side with `offsetof` asserts for every field the
shim fills (`GameInputDeviceInfo` 320 bytes with `deviceFamily` at 96 and the info pointers from 160,
`GameInputGamepadState` 28 bytes, `GameInputControllerAxisInfo` 48, and the rest in `gameinput_abi.h`).

The force-feedback and haptic declarations (`GameInputForceFeedbackMotorInfo`, `GameInputHapticWaveformInfo`,
`GameInputHapticFeedbackMotorInfo`, the envelope and parameter structs, `IGameInputForceFeedbackEffect`) were
transcribed from that header's public source and pinned with the same kind of `static_assert` on their sizes and
fields — they were not compiled against it, since it is not in the repository. That is the one place the shim's ABI
rests on a transcription rather than on a side-by-side compile.

## Tests

- `active_input_test` (CTest, no SDL): the [active input](#active-input) tracker's rules on synthetic samples with
  its own clock. `gameinput_test` runs them on a DualSense, a DualShock 4 (Sony `054c:09cc`), an Xbox pad and the
  keyboard pad: no change when pads connect (one with a button held) or a held button is let go, each family by a
  button, the touchpad, a trigger or a stick, stick drift and a one-count mouse jitter ignored, a real mouse movement
  counted, the 300 ms window on the real clock, and `LEONIDA_PROMPTS` forcing, changing, ignoring an unknown value
  and clearing.
- `gameinput_test` (CTest, no controller): turns SDL's real backends off and drives the shim through
  `rt::shims::find` with SDL virtual joysticks, an Xbox-style pad with trigger motors, a DualSense (Sony
  `054c:0ce6`) and a pad without trigger motors. It checks enumeration and device callbacks, the button, stick,
  trigger and UI-navigation mapping, reading callbacks and sequence numbers, the reading history (a tap read after
  the fact, both directions, two pads interleaved and filtered by device, ageing out, held readings, unplug and
  replug), the guide button, motor and trigger
  rumble as SDL receives them, the DualSense effect bytes in each `LEONIDA_DS_TRIGGERS` mode, the lightbar, a
  dispatcher and its quota, async and blocking enumeration, unregistering, unplugging and replugging, and three identical pads
  (an id each, `FindDeviceFromId` and device-filtered readings per pad, replugging one, two and all three, and the
  first twin's id equal to the single-pad id). Motion runs on a virtual pad with sensors (`nsensors` in the
  virtual-joystick description, fed with `SDL_SendJoystickVirtualSensorData`): the kind and `motionInfo` on a sensor
  pad and not on a plain one, no motion before the first sample, normalisation against the `motionInfo` maxima, the
  accuracy values, the ±1 clamp, the ~8 ms cadence, that a motion query on a pad without sensors is
  `GAMEINPUT_E_READING_NOT_FOUND`, and that a Motion reading callback hears only the sensor pad. The keyboard pad is
  driven through `rt::gameinput::set_keyboard_source` and `set_cursor_source` with fake keys, focus and cursor: every
  binding, two keys at once, each WASD direction, opposite keys cancelling, diagonals, the mouse moving the right
  stick, clamping and decaying to 0, the mouse-button triggers, no input out of focus, release of buttons, sticks and
  triggers on focus loss, the `always` mode, and turning it off and on again.
  Raw HID (RT-175) is tested in two ways:
  - `parse_report_descriptor` runs on synthetic descriptors: report ids, none, Push/Pop, a long item, and the refused
    forms.
  - A fake backend (`set_hid_backend`) stands behind a virtual pad with pid.codes test ids. The test checks the report
    tables, the kind and status bits, a reading's raw report (newest declared id, without the id byte, an undeclared id
    ignored), creating, sending and reading back output and feature reports with their id byte, `SetRawData`'s exact
    size, the item calls, `E_INVALIDARG` / `E_FAIL` / `E_NOTIMPL`, and the HID device closing on unplug.
- `gameinput_probe [--seconds N] [--rumble] [--force-feedback]` (by hand, with a pad connected): lists the pads with
  each one's raw HID report tables (or why it has none), plays each motor and trigger for a second with `--rumble`,
  holds a constant force at magnitude 1.0 on motor 0 for a second with `--force-feedback` (skipped, with the reason
  printed, on a pad the shim reports no force-feedback motor for), and prints whether SDL created that effect and is
  running it, separately from GameInput accepting it, then prints the live reading and the newest raw input report.

On hardware so far, with `gameinput_probe` and SDL 3.4.20 on Windows 11: a DualSense (`054c:0ce6`) over USB. It
connects as player 1 with trigger effects and gyro (`motors=0xf forcefeedback=0 haptic=2`). Pressed one at a time,
each of the 14 gamepad buttons set the one bit of the name [Mapping](#mapping) gives it: Cross, Circle, Square and Triangle
as `A`, `B`, `X` and `Y`, the d-pad, L1/R1, L3/R3, Create as `View` and Options as `Menu`. Both triggers reached
1.00 and both sticks -1.00 and +1.00 on each axis. At rest the accelerometer read about 0.5 on one axis and the gyro
0.0; shaking the pad by hand reached 1.00 on the accelerometer. With `--rumble` (the default `rumble` trigger mode)
and `LEONIDA_DS_LIGHTBAR=00ff00`, the person holding the pad reported the motors, the trigger effects and the lightbar
as working; nothing measured them, and the `resistance` mode was not tried.
    size, the item calls, `E_INVALIDARG` / `E_FAIL` / `E_NOTIMPL`, the HID device closing on unplug, and a held
    reading's `GetRawReport` waiting while a reconnect rebuilds the report tables.
- `gameinput_probe [--seconds N] [--rumble]` (by hand, with a pad connected): lists the pads with each one's raw HID
  report tables (or why it has none), plays each motor and trigger for a second with `--rumble`, then prints the live
  reading and the newest raw input report.

## VERIFY

These need the game executable or a test on hardware:

- **Which API version and module.** The shim is GameInput v0 and registers `GameInputCreate` under `gameinput.dll`
  and `xgameruntime.dll`. GameInput v1 and later (NuGet 1.x and up) changed the interfaces; if the executable was
  built against one of those, the vtables need a second set.
- **IIDs.** `QueryInterface` answers the v0 IIDs from the public header.
- **Reading history.** Microsoft's [next reading](https://learn.microsoft.com/en-us/gaming/gdk/docs/reference/input/gameinput/interfaces/igameinput/methods/igameinput_getnextreading)
  and [previous reading](https://learn.microsoft.com/en-us/gaming/gdk/docs/reference/input/gameinput/interfaces/igameinput/methods/igameinput_getpreviousreading) pages describe half a second of readings per device and failure when old readings leave the stream. The table above records the shim's HRESULT choices. Changes faster than the poll are not seen.
  `GetTemporalReading` returns `E_NOTIMPL` with a null reading on purpose: Microsoft's
  [reference page](https://learn.microsoft.com/en-us/gaming/gdk/docs/reference/input/gameinput/deprecated/interfaces/igameinput/methods/igameinput_gettemporalreading)
  says it "was never implemented in v0" and returns `E_NOTIMPL`, and v1 removed it. The v0 header (NuGet
  `Microsoft.GameInput` 0.2303.22621.3038) declares it as `(timestamp, device, reading)`, the layout the shim has;
  the page's five-parameter form does not match that header.
- **UI navigation mapping** and the **controller-axis convention** (0..1 with sticks at 0.5) are this shim's choice,
  not measured from a console.
- **Battery capacity** is reported as a fraction (remaining 0..1 of full 1), not in device units.
- **`UnregisterCallback` timeout** is not honoured; it waits for a callback that is running.
- **Motion normalisation maxima** (2 g accel, 2000 deg/s gyro) are this shim's choice; SDL exposes no sensor range,
  and whether the game reads motion, and against what scale, is unknown until it runs. The accelerometer range a
  DualSense exposes can also be wider than 2 g, so hard shakes may clamp at 1.0: a hand shake of a real DualSense
  did reach 1.00 on one axis.
- **Motion members the shim does not provide** (magnetic field, orientation) are zero with
  `GameInputMotionAccuracyUnknown`; whether a title wants a synthesised quaternion there is unknown.
- **Device family**: every pad is reported as `GameInputFamilyXboxOne`; a title that filters on family sees only
  that.
- **Active input behind Steam Input**: when Steam turns a PlayStation pad into a virtual Xbox pad, SDL may report the
  virtual one's type, and the active input then says Xbox; `LEONIDA_PROMPTS` is the way round it until that is
  checked on a real setup.
- **Identical pads at start**: twins are numbered in the order the shim opens them. For pads already connected when
  it starts, that is SDL's enumeration order, not measured on Windows for identical pads; if it varies between runs,
  so does which pad gets which id.
- **Raw mouse registration**: Windows keeps one raw-input target window per device type and process, so if the title
  registers raw mouse input itself (a console title is not expected to), it replaces the shim's registration and the
  right stick falls back to nothing. The shim does not detect that yet.
- **Keyboard pad as a second pad**: it is one more connected gamepad, so with a controller plugged in the title sees
  two. A title that follows only the first pad it saw, or asks the player to press a button to pick the active pad,
  may ignore one of them; whether the game does either is unknown.
- **Mouse on the right stick**: the scale (1024 units per pixel per poll) and decay (7/8 per poll) are a starting
  point, not tuned in game. The title adds its own stick acceleration and deadzone on top, so aiming may feel slow
  near center or jumpy.
- **Force feedback on hardware**: which pads SDL opens as haptic devices on Windows (a DirectInput wheel or stick,
  not an Xbox pad) and how many force-feedback motors a title would really see there. The shim reports one motor with
  LinearX; whether a wheel wants a second axis, or LinearY for a stick, is unknown until one is attached.
  `gameinput_probe` prints `forcefeedback=` and `haptic=` per pad.
- **How a force-feedback effect feels**: the magnitudes go out 1:1 (SDL's -32767..32767 from GameInput's -1..1) with
  no scaling beyond the two gains. Whether a game's curb weight, road texture and crash feel right, and whether
  `SDL_SetHapticGain` is the right place for a motor's master gain on the driver SDL picks, needs a wheel in game.
- **Envelope and phase units**: GameInput's `phase` is read as degrees (SDL takes hundredths of a degree) and its
  durations as microseconds summed into SDL's `length`. Neither came from a measurement, only from the header's
  remarks.
- **A new effect starting stopped or running**: the shim starts it running, because the documented use creates an
  effect once and only ever calls `SetParams`. A title that creates one and waits for `SetState` would play a frame
  early; whether this one does is unknown until it runs.
- **Haptic waveforms**: the single intensity waveform (usage 0x0F/0x58) is this shim's choice, not the pad's table, so
  a title that asks for a particular waveform gets intensity-only behaviour. Whether the game uses haptic motors at
  all — as opposed to `SetRumbleState` — is unknown.
- **Raw HID on a real pad**: the synthetic tests, the XInput refusal and one DualSense over USB have run.
  - What ran: an Xbox Series X controller over Bluetooth on Windows 11 connects and plays normally, with no raw I/O,
    and the log names the reason.
  - What ran: a DualSense (`054c:0ce6`) over USB on Windows 11 through `gameinput_probe`. Its descriptor gives 1
    input report (id 1, 63 bytes), 1 output report (id 2, 47 bytes) and 20 feature reports, and the raw input report
    changed on every reading (2,442 distinct reports in a 40 s capture). No output or feature report was sent
    through the raw calls.
  - Still to confirm: a DualSense over Bluetooth, a DualShock 4 or a Switch Pro through `gameinput_probe`, which
    should print the descriptor's reports and changing raw input bytes.
  - Size convention: report sizes exclude the id byte, and `GetRawData` returns the payload without it. What
    Microsoft's runtime does for a device with report ids is not documented on the reference pages.
  - Readings: a pad that streams reports makes a reading for each changed one, like the runtime's
    `GameInputKindRawDeviceReport` stream, but whether the title asks for raw reports at all is unknown.
