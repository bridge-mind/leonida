// gameinput: Microsoft's GameInput API (v0) for gamepads, backed by SDL3, so any controller SDL knows works:
// DualSense, DualShock 4, Switch Pro, Xbox and the rest of SDL's gamepad database.
//
// The Xbox build reads controllers through GameInput. This shim answers GameInputCreate with a COM object whose
// vtables follow the public v0 API (gameinput_abi.h) and serves gamepads from SDL3:
//
//   - device enumeration, connect/disconnect and device callbacks (blocking and async enumeration)
//   - GetCurrentReading, and GetNextReading / GetPreviousReading over each pad's last half second of readings
//   - GetGamepadState, GetUiNavigationState, controller axes and buttons
//   - SetRumbleState: the two body motors through SDL_RumbleGamepad, the impulse-trigger motors through
//     SDL_RumbleGamepadTriggers on pads that have them; on a DualSense the trigger signal becomes an adaptive-trigger
//     effect (SDL_SendGamepadEffect), as set by LEONIDA_DS_TRIGGERS. That is an approximation of the Xbox impulse
//     trigger signal, not the PS5 version's authored effects.
//   - DualSense/DualShock 4 lightbar colour (LEONIDA_DS_LIGHTBAR) and the player LEDs by connection order
//   - force-feedback motors, reported from the SDL haptic device SDL opens for the pad's joystick, and played through
//     SDL_CreateHapticEffect/RunHapticEffect (IGameInputForceFeedbackEffect); haptic trigger motors, reported from the
//     trigger motors the pad already has, played as an intensity through the same trigger output as SetRumbleState
//   - the keyboard pad (LEONIDA_KEYBOARD_PAD): one SDL virtual gamepad driven by the keyboard and mouse
//   - which kind of device the player used last (rt::gameinput::active_input, LEONIDA_PROMPTS), for the runtime's
//     own use: the title is told every pad is an Xbox One pad
//
// One thread owns SDL: the background poll thread by default, or the caller of rt::gameinput::poll() when
// LEONIDA_GAMEINPUT_THREAD=0. Title calls never touch SDL: they read the latest snapshot and leave rumble requests
// that the next poll applies (2 ms at most with the thread). Callbacks run on the poll thread, as GameInput runs them
// on its own threads, unless the title created a dispatcher; then IGameInputDispatcher::Dispatch delivers them.
//
// Buttons are positional, as on an Xbox pad: A is the bottom face button (Cross on a DualSense, B on a Switch Pro),
// B the right one, X the left, Y the top. View is Back/Create/Share/Minus, Menu is Start/Options/Plus.
//
// Nothing here goes online. docs/gameinput.md has the mapping, the environment variables and the VERIFY list.
//
// ASSUMPTION: the console executable is not available, so the DLL it imports GameInputCreate from is unknown. The PC
// API lives in gameinput.dll; on console the GDK links it through xgameruntime. Both names are registered.
#include <windows.h>

#include <SDL3/SDL.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "gameinput.h"
#include "gameinput_abi.h"
#include "shims.h"

using namespace gi;

namespace {

constexpr uint32_t kBaseKinds =
    GameInputKindControllerAxis | GameInputKindControllerButton | GameInputKindGamepad | GameInputKindUiNavigation;
// Everything a reading may carry: motion on pads with sensors, raw reports on pads opened through HID (RT-175).
constexpr uint32_t kSupportedKinds = kBaseKinds | GameInputKindMotion | GameInputKindRawDeviceReport;
constexpr uint32_t kAxisCount = 6;  // LX, LY, RX, RY, LT, RT
constexpr uint32_t kButtonCount = SDL_GAMEPAD_BUTTON_COUNT;
constexpr uint32_t kConnectedStatus = GameInputDeviceConnected | GameInputDeviceInputEnabled | GameInputDeviceOutputEnabled;
constexpr Uint32 kRumbleDurationMs = 0xFFFF;  // SDL's maximum; refreshed well before it runs out
constexpr uint64_t kRumbleRefreshUs = 20'000'000;
constexpr uint64_t kBatteryRefreshUs = 1'000'000;
constexpr int kDsEffectSize = 47;  // the DualSense effects block SDL_SendGamepadEffect forwards (SDL's DS5EffectsState_t)
constexpr uint32_t kForceFeedbackMotors = 1;  // one per SDL haptic device: the wheel or stick it belongs to
constexpr uint32_t kHapticMotors = 2;         // the left and right trigger, on a pad that backs them
constexpr uint32_t kHapticHoldMs = 60'000;    // length of an effect GameInput gives no duration; re-run before it ends
constexpr uint64_t kHistoryUs = 500'000;  // a reading stays in the history until the one after it is this old
constexpr size_t kHistoryCap = 512;       // readings per pad, whatever the poll rate
// Motion. SDL gives accel in m/s^2 (gravity included) and gyro in rad/s; the state the ABI defines carries signed
// fractions of the maxima motionInfo reports, so a reading stays within -1..1 however hard the pad is shaken. The
// maxima are this shim's choice: SDL does not expose a sensor range, and 2 g / 2000 deg/s are the ranges a
// DualSense's accelerometer and gyro can express (VERIFY: against the game and a real pad, docs/gameinput.md).
constexpr float kMaxAcceleration = 19.6133f;     // 2 g in m/s^2
constexpr float kMaxAngularVelocity = 34.9066f;  // 2000 deg/s in rad/s
constexpr float kMotionSensorRate = 250.0f;      // requested sample rate; the device clamps to what it has
constexpr uint64_t kMotionReadingUs = 8'000;     // a motion pad streams readings at ~125 Hz so the history stays sane

uint64_t now_us() {
    static const LARGE_INTEGER freq = [] { LARGE_INTEGER f; QueryPerformanceFrequency(&f); return f; }();
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return static_cast<uint64_t>(c.QuadPart / freq.QuadPart * 1'000'000 + c.QuadPart % freq.QuadPart * 1'000'000 / freq.QuadPart);
}

// Release for objects the shim keeps for the process: the count never drops below one, so they are never freed.
ULONG release_floor(std::atomic<ULONG>& refs) {
    ULONG r = refs.load();
    while (r > 1 && !refs.compare_exchange_weak(r, r - 1)) {}
    return r > 1 ? r - 1 : 1;
}

bool same_guid(const GUID& a, const GUID& b) { return std::memcmp(&a, &b, sizeof(GUID)) == 0; }

// ---- SDL haptics ------------------------------------------------------------------------------------------------
//
// Force feedback and the trigger haptic motors go through SDL's haptic API, opened from the pad's joystick: SDL's
// gamepad API has no force feedback at all, and a wheel or stick it models as a joystick exposes one only as a haptic
// device. Every call goes through one table so gameinput_test can stand in for it, because SDL's virtual joysticks
// report no haptics and a wheel is the only other way to reach this path.

void* sdl_open_haptic(void* js) { return SDL_OpenHapticFromJoystick(static_cast<SDL_Joystick*>(js)); }
void sdl_close_haptic(void* h) { SDL_CloseHaptic(static_cast<SDL_Haptic*>(h)); }
uint32_t sdl_haptic_features(void* h) { return SDL_GetHapticFeatures(static_cast<SDL_Haptic*>(h)); }
int sdl_haptic_max_effects(void* h) { return SDL_GetMaxHapticEffects(static_cast<SDL_Haptic*>(h)); }
int sdl_haptic_create(void* h, const void* e) {
    return SDL_CreateHapticEffect(static_cast<SDL_Haptic*>(h), static_cast<const SDL_HapticEffect*>(e));
}
bool sdl_haptic_update(void* h, int id, const void* e) {
    return SDL_UpdateHapticEffect(static_cast<SDL_Haptic*>(h), id, static_cast<const SDL_HapticEffect*>(e));
}
bool sdl_haptic_run(void* h, int id, uint32_t iterations) {
    return SDL_RunHapticEffect(static_cast<SDL_Haptic*>(h), id, iterations);
}
bool sdl_haptic_stop(void* h, int id) { return SDL_StopHapticEffect(static_cast<SDL_Haptic*>(h), id); }
void sdl_haptic_destroy(void* h, int id) { SDL_DestroyHapticEffect(static_cast<SDL_Haptic*>(h), id); }
bool sdl_haptic_gain(void* h, int gain) { return SDL_SetHapticGain(static_cast<SDL_Haptic*>(h), gain); }

const rt::gameinput::HapticApi kSdlHaptic{sdl_open_haptic,     sdl_close_haptic, sdl_haptic_features,
                                          sdl_haptic_max_effects, sdl_haptic_create, sdl_haptic_update,
                                          sdl_haptic_run,      sdl_haptic_stop,  sdl_haptic_destroy,
                                          sdl_haptic_gain};
std::atomic<const rt::gameinput::HapticApi*> g_haptic_api{&kSdlHaptic};
const rt::gameinput::HapticApi& haptics() { return *g_haptic_api.load(); }

// ---- configuration ----------------------------------------------------------------------------------------------

std::string env(const char* name) {
    char buf[64];
    const DWORD n = GetEnvironmentVariableA(name, buf, sizeof buf);
    if (n == 0 || n >= sizeof buf) return {};
    std::string s(buf, n);
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

rt::gameinput::Config read_config() {
    rt::gameinput::Config c;
    const std::string t = env("LEONIDA_DS_TRIGGERS");
    if (t == "off" || t == "0") c.ds_triggers = rt::gameinput::DsTriggerMode::Off;
    else if (t == "resistance") c.ds_triggers = rt::gameinput::DsTriggerMode::Resistance;
    else if (!t.empty() && t != "rumble") std::fprintf(stderr, "[rt] gameinput: LEONIDA_DS_TRIGGERS=%s unknown, using rumble\n", t.c_str());
    if (const std::string g = env("LEONIDA_DS_TRIGGER_GAIN"); !g.empty())
        c.ds_trigger_gain = std::clamp(static_cast<float>(std::atof(g.c_str())), 0.0f, 4.0f);
    if (const std::string g = env("LEONIDA_MOUSE_SENSITIVITY"); !g.empty())
        c.mouse_sensitivity = std::clamp(static_cast<float>(std::atof(g.c_str())), 0.1f, 10.0f);
    if (const std::string f = env("LEONIDA_DS_TRIGGER_FREQ"); !f.empty())
        c.ds_trigger_frequency = static_cast<uint8_t>(std::clamp(std::atoi(f.c_str()), 1, 255));
    if (const std::string l = env("LEONIDA_DS_LIGHTBAR"); !l.empty()) {
        if (l == "off") {
            c.lightbar_set = true;
        } else if (l.size() == 6 && l.find_first_not_of("0123456789abcdef") == std::string::npos) {
            const unsigned long rgb = std::strtoul(l.c_str(), nullptr, 16);
            c.lightbar_set = true;
            c.lightbar[0] = static_cast<uint8_t>(rgb >> 16);
            c.lightbar[1] = static_cast<uint8_t>(rgb >> 8);
            c.lightbar[2] = static_cast<uint8_t>(rgb);
        } else {
            std::fprintf(stderr, "[rt] gameinput: LEONIDA_DS_LIGHTBAR=%s is not RRGGBB or off, ignored\n", l.c_str());
        }
    }
    if (env("LEONIDA_RUMBLE") == "0") c.rumble = false;
    if (env("LEONIDA_GAMEINPUT_THREAD") == "0") c.background_thread = false;
    const std::string k = env("LEONIDA_KEYBOARD_PAD");
    if (k == "1" || k == "on" || k == "focus") c.keyboard_pad = rt::gameinput::KeyboardPadMode::Focused;
    else if (k == "always") c.keyboard_pad = rt::gameinput::KeyboardPadMode::Always;
    else if (!k.empty() && k != "0" && k != "off")
        std::fprintf(stderr, "[rt] gameinput: LEONIDA_KEYBOARD_PAD=%s unknown (1, always or off), keyboard pad off\n",
                     k.c_str());
    using leonida::InputFamily;
    const std::string p = env("LEONIDA_PROMPTS");
    if (p == "xbox") c.prompts = InputFamily::Xbox;
    else if (p == "playstation" || p == "ps5") c.prompts = InputFamily::PlayStation5;
    else if (p == "ps4") c.prompts = InputFamily::PlayStation4;
    else if (p == "ps3") c.prompts = InputFamily::PlayStation3;
    else if (p == "nintendo" || p == "switch") c.prompts = InputFamily::Nintendo;
    else if (p == "keyboard") c.prompts = InputFamily::Keyboard;
    else if (!p.empty() && p != "auto")
        std::fprintf(stderr, "[rt] gameinput: LEONIDA_PROMPTS=%s unknown (auto, xbox, playstation, ps5, ps4, ps3, "
                             "nintendo or keyboard), following the devices\n", p.c_str());
    if (const std::string m = env("LEONIDA_PROMPTS_SWITCH_MS"); !m.empty())
        c.prompts_switch_ms = static_cast<uint32_t>(std::clamp(std::atoi(m.c_str()), 0, 2000));
    return c;
}

// ---- objects ----------------------------------------------------------------------------------------------------

class Device;
class ForceFeedbackEffect;

struct Snapshot {
    GameInputGamepadState gamepad{};
    uint32_t ui = 0;
    float axes[kAxisCount] = {};
    bool buttons[kButtonCount] = {};
    bool guide = false;
    GameInputMotionState motion{};  // valid only when has_motion; does not take part in operator==
    bool has_motion = false;
    // The newest raw input report of a pad opened through HID (RT-175): its report id and payload, without the id
    // byte. A new report with other bytes is a change, so it makes a reading of its own.
    bool has_raw = false;
    uint32_t raw_id = 0;
    std::vector<uint8_t> raw;
    bool operator==(const Snapshot& o) const {  // field by field: the padding is not initialised
        const GameInputGamepadState &a = gamepad, &b = o.gamepad;
        return a.buttons == b.buttons && a.leftTrigger == b.leftTrigger && a.rightTrigger == b.rightTrigger &&
               a.leftThumbstickX == b.leftThumbstickX && a.leftThumbstickY == b.leftThumbstickY &&
               a.rightThumbstickX == b.rightThumbstickX && a.rightThumbstickY == b.rightThumbstickY && ui == o.ui &&
               std::memcmp(buttons, o.buttons, sizeof buttons) == 0 && guide == o.guide && has_raw == o.has_raw &&
               raw_id == o.raw_id && raw == o.raw;
    }
};

class Reading final : public IGameInputReading {
public:
    Reading(Device* device, const Snapshot& s, uint64_t seq, uint64_t ts, uint64_t order);
    ~Reading();

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** out) override {
        if (!out) return E_POINTER;
        if (same_guid(riid, IID_IUnknown) || same_guid(riid, IID_IGameInputReading)) {
            AddRef();
            *out = static_cast<IGameInputReading*>(this);
            return S_OK;
        }
        *out = nullptr;
        return E_NOINTERFACE;
    }
    ULONG STDMETHODCALLTYPE AddRef() override { return ++refs_; }
    ULONG STDMETHODCALLTYPE Release() override {
        const ULONG r = --refs_;
        if (r == 0) delete this;
        return r;
    }

    uint32_t STDMETHODCALLTYPE GetInputKind() override {
        // A reading of a pad without sensors carries no motion, and one without a raw report no raw report.
        return kBaseKinds | (s_.has_motion ? GameInputKindMotion : 0u) | (s_.has_raw ? GameInputKindRawDeviceReport : 0u);
    }
    uint64_t STDMETHODCALLTYPE GetSequenceNumber(uint32_t) override { return seq_; }
    uint64_t STDMETHODCALLTYPE GetTimestamp() override { return ts_; }
    void STDMETHODCALLTYPE GetDevice(IGameInputDevice** device) override;
    bool STDMETHODCALLTYPE GetRawReport(void** report) override;
    uint32_t STDMETHODCALLTYPE GetControllerAxisCount() override { return kAxisCount; }
    uint32_t STDMETHODCALLTYPE GetControllerAxisState(uint32_t count, float* out) override {
        const uint32_t n = out ? std::min(count, kAxisCount) : 0;
        std::memcpy(out, s_.axes, n * sizeof(float));
        return n;
    }
    uint32_t STDMETHODCALLTYPE GetControllerButtonCount() override { return kButtonCount; }
    uint32_t STDMETHODCALLTYPE GetControllerButtonState(uint32_t count, bool* out) override {
        const uint32_t n = out ? std::min(count, kButtonCount) : 0;
        std::memcpy(out, s_.buttons, n * sizeof(bool));
        return n;
    }
    uint32_t STDMETHODCALLTYPE GetControllerSwitchCount() override { return 0; }
    uint32_t STDMETHODCALLTYPE GetControllerSwitchState(uint32_t, GameInputSwitchPosition*) override { return 0; }
    uint32_t STDMETHODCALLTYPE GetKeyCount() override { return 0; }
    uint32_t STDMETHODCALLTYPE GetKeyState(uint32_t, void*) override { return 0; }
    bool STDMETHODCALLTYPE GetMouseState(void*) override { return false; }
    uint32_t STDMETHODCALLTYPE GetTouchCount() override { return 0; }
    uint32_t STDMETHODCALLTYPE GetTouchState(uint32_t, void*) override { return 0; }
    bool STDMETHODCALLTYPE GetMotionState(void* state) override {
        if (!state || !s_.has_motion) return false;
        *static_cast<GameInputMotionState*>(state) = s_.motion;
        return true;
    }
    bool STDMETHODCALLTYPE GetArcadeStickState(void*) override { return false; }
    bool STDMETHODCALLTYPE GetFlightStickState(void*) override { return false; }
    bool STDMETHODCALLTYPE GetGamepadState(GameInputGamepadState* state) override {
        if (!state) return false;
        *state = s_.gamepad;
        return true;
    }
    bool STDMETHODCALLTYPE GetRacingWheelState(void*) override { return false; }
    bool STDMETHODCALLTYPE GetUiNavigationState(GameInputUiNavigationState* state) override {
        if (!state) return false;
        state->buttons = s_.ui;
        return true;
    }

    Device* device() const { return device_; }
    uint64_t order() const { return order_; }

private:
    std::atomic<ULONG> refs_{1};
    Device* device_;
    Snapshot s_;
    uint64_t seq_, ts_;
    uint64_t order_;  // place among every pad's readings, the order GetNextReading/GetPreviousReading follow
};

struct Rumble {
    float low = 0, high = 0, left = 0, right = 0;
};

// GameInput -1..1 magnitudes and 0..1 saturations against SDL's levels and saturations.
Sint16 hlevel(float v) { return static_cast<Sint16>(std::lround(std::clamp(v, -1.0f, 1.0f) * 32767.0f)); }
Uint16 hsat(float v) { return static_cast<Uint16>(std::lround(std::clamp(v, 0.0f, 1.0f) * 65535.0f)); }
uint16_t ms_from_us(uint64_t us) { return static_cast<uint16_t>(std::min<uint64_t>((us + 999) / 1000, 65535)); }

// The SDL_HAPTIC_* bit an effect kind needs, for the motor flags and the create-time check.
uint32_t kind_feature(GameInputForceFeedbackEffectKind kind) {
    switch (kind) {
        case GameInputForceFeedbackConstant: return SDL_HAPTIC_CONSTANT;
        case GameInputForceFeedbackRamp: return SDL_HAPTIC_RAMP;
        case GameInputForceFeedbackSineWave: return SDL_HAPTIC_SINE;
        case GameInputForceFeedbackSquareWave: return SDL_HAPTIC_SQUARE;
        case GameInputForceFeedbackTriangleWave: return SDL_HAPTIC_TRIANGLE;
        case GameInputForceFeedbackSawtoothUpWave: return SDL_HAPTIC_SAWTOOTHUP;
        case GameInputForceFeedbackSawtoothDownWave: return SDL_HAPTIC_SAWTOOTHDOWN;
        case GameInputForceFeedbackSpring: return SDL_HAPTIC_SPRING;
        case GameInputForceFeedbackFriction: return SDL_HAPTIC_FRICTION;
        case GameInputForceFeedbackDamper: return SDL_HAPTIC_DAMPER;
        case GameInputForceFeedbackInertia: return SDL_HAPTIC_INERTIA;
    }
    return 0;
}

bool valid_kind(GameInputForceFeedbackEffectKind kind) {
    return kind >= GameInputForceFeedbackConstant && kind <= GameInputForceFeedbackInertia;
}

// An all-zero envelope is how a title that only sets a magnitude leaves it: that means no envelope, not silence.
bool envelope_used(const GameInputForceFeedbackEnvelope& e) {
    return e.attackDuration || e.sustainDuration || e.releaseDuration || e.attackGain != 0.0f ||
           e.sustainGain != 0.0f || e.releaseGain != 0.0f || e.playCount || e.repeatDelay;
}

uint32_t play_count(const GameInputForceFeedbackParams& p) {
    return p.kind <= GameInputForceFeedbackSawtoothDownWave ? p.data.constant.envelope.playCount : 0;
}

// One GameInput effect as SDL's template. GameInput's eleven kinds are SDL's eleven force-feedback effects, so the
// kind picks the SDL type and the fields are magnitudes; conditions have no envelope in GameInput either. Returns
// false for a kind the shim does not map. `endless` says the effect has no duration of its own (GameInput left the
// envelope empty), which is what the poll thread re-runs before the SDL length runs out.
bool build_effect(const GameInputForceFeedbackParams& p, float gain, bool* endless, uint32_t* length_ms,
                  SDL_HapticEffect* out) {
    std::memset(out, 0, sizeof *out);
    if (!valid_kind(p.kind)) return false;
    const bool envelope_kind = p.kind <= GameInputForceFeedbackSawtoothDownWave;
    const GameInputForceFeedbackEnvelope env = envelope_kind ? p.data.constant.envelope : GameInputForceFeedbackEnvelope{};
    const bool used = envelope_kind && envelope_used(env);
    const float level = gain * (used ? env.sustainGain : 1.0f);  // GameInput's sustainGain is the effect's level
    const uint64_t total_us = used ? env.attackDuration + env.sustainDuration + env.releaseDuration : 0;
    *endless = total_us == 0;
    *length_ms = total_us == 0 ? kHapticHoldMs
                              : static_cast<uint32_t>(std::clamp<uint64_t>((total_us + 999) / 1000, 1, 86'400'000));
    // The envelope's attack and release, when the title gave one. SDL only uses the envelope when attack_length or
    // fade_level is set, so a zero-gain envelope is left alone rather than muting the effect.
    auto envelope = [&](SDL_HapticConstant& e) {
        if (!used || (!env.attackDuration && !env.releaseDuration)) return;
        e.attack_length = ms_from_us(env.attackDuration);
        e.attack_level = hsat(env.attackGain);
        e.fade_length = ms_from_us(env.releaseDuration);
        e.fade_level = hsat(env.releaseGain);
    };
    // Force along +X: the axis a wheel's or a stick's motor acts on, which is the one the motor reports (LinearX).
    auto direction = [](SDL_HapticDirection& d) {
        d.type = SDL_HAPTIC_CARTESIAN;
        d.dir[0] = 1;
    };
    switch (p.kind) {
        case GameInputForceFeedbackConstant: {
            SDL_HapticConstant& e = out->constant;
            e.type = SDL_HAPTIC_CONSTANT;
            direction(e.direction);
            e.length = *length_ms;
            e.level = hlevel(p.data.constant.magnitude.linearX * level);
            envelope(e);
            break;
        }
        case GameInputForceFeedbackRamp: {
            SDL_HapticRamp& e = out->ramp;
            e.type = SDL_HAPTIC_RAMP;
            direction(e.direction);
            e.length = *length_ms;
            e.start = hlevel(p.data.ramp.startMagnitude.linearX * level);
            e.end = hlevel(p.data.ramp.endMagnitude.linearX * level);
            if (used) {
                e.attack_length = ms_from_us(env.attackDuration);
                e.attack_level = hsat(env.attackGain);
                e.fade_length = ms_from_us(env.releaseDuration);
                e.fade_level = hsat(env.releaseGain);
            }
            break;
        }
        case GameInputForceFeedbackSineWave:
        case GameInputForceFeedbackSquareWave:
        case GameInputForceFeedbackTriangleWave:
        case GameInputForceFeedbackSawtoothUpWave:
        case GameInputForceFeedbackSawtoothDownWave: {
            SDL_HapticPeriodic& e = out->periodic;
            e.type = p.kind == GameInputForceFeedbackSineWave       ? SDL_HAPTIC_SINE
                     : p.kind == GameInputForceFeedbackSquareWave   ? SDL_HAPTIC_SQUARE
                     : p.kind == GameInputForceFeedbackTriangleWave ? SDL_HAPTIC_TRIANGLE
                     : p.kind == GameInputForceFeedbackSawtoothUpWave ? SDL_HAPTIC_SAWTOOTHUP
                                                                      : SDL_HAPTIC_SAWTOOTHDOWN;
            direction(e.direction);
            e.length = *length_ms;
            const GameInputForceFeedbackPeriodicParams& pp = p.data.sineWave;
            // GameInput gives the wave's frequency in Hz and its phase in degrees; SDL wants a period in ms and a
            // phase in hundredths of a degree.
            e.period = pp.frequency > 0.0f ? static_cast<Uint16>(std::clamp(std::lround(1000.0f / pp.frequency), 1L, 65535L)) : 100;
            e.magnitude = hlevel(pp.magnitude.linearX * level);
            e.offset = hlevel(pp.bias * level);
            e.phase = static_cast<Uint16>(std::clamp(std::lround(pp.phase * 100.0f), 0L, 35999L));
            if (used) {
                e.attack_length = ms_from_us(env.attackDuration);
                e.attack_level = hsat(env.attackGain);
                e.fade_length = ms_from_us(env.releaseDuration);
                e.fade_level = hsat(env.releaseGain);
            }
            break;
        }
        default: {  // spring, friction, damper, inertia
            SDL_HapticCondition& e = out->condition;
            e.type = p.kind == GameInputForceFeedbackSpring    ? SDL_HAPTIC_SPRING
                     : p.kind == GameInputForceFeedbackFriction ? SDL_HAPTIC_FRICTION
                     : p.kind == GameInputForceFeedbackDamper  ? SDL_HAPTIC_DAMPER
                                                                : SDL_HAPTIC_INERTIA;
            e.length = *length_ms;
            const GameInputForceFeedbackConditionParams& cp = p.data.spring;
            const float center[3] = {cp.magnitude.linearX, cp.magnitude.linearY, cp.magnitude.linearZ};
            for (int a = 0; a < 3; ++a) {
                e.right_sat[a] = hsat(cp.maxPositiveMagnitude);
                e.left_sat[a] = hsat(cp.maxNegativeMagnitude);
                e.right_coeff[a] = hlevel(cp.positiveCoefficient * level);
                e.left_coeff[a] = hlevel(-cp.negativeCoefficient * level);
                e.deadband[a] = hsat(cp.deadZone);
                e.center[a] = hlevel(center[a]);
            }
            break;
        }
    }
    return true;
}

class Device final : public IGameInputDevice {
public:
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** out) override {
        if (!out) return E_POINTER;
        if (same_guid(riid, IID_IUnknown) || same_guid(riid, IID_IGameInputDevice)) {
            AddRef();
            *out = static_cast<IGameInputDevice*>(this);
            return S_OK;
        }
        *out = nullptr;
        return E_NOINTERFACE;
    }
    // Devices live for the process: the registry keeps every device it has seen, so that a title holding one across a
    // disconnect gets the same object back when the pad returns.
    ULONG STDMETHODCALLTYPE AddRef() override { return ++refs_; }
    ULONG STDMETHODCALLTYPE Release() override { return release_floor(refs_); }

    const GameInputDeviceInfo* STDMETHODCALLTYPE GetDeviceInfo() override { return &info; }
    uint32_t STDMETHODCALLTYPE GetDeviceStatus() override { return status.load(); }
    void STDMETHODCALLTYPE GetBatteryState(GameInputBatteryState* state) override;
    HRESULT STDMETHODCALLTYPE CreateForceFeedbackEffect(uint32_t motorIndex, const void* params, void** effect) override;
    bool STDMETHODCALLTYPE IsForceFeedbackMotorPoweredOn(uint32_t motorIndex) override;
    void STDMETHODCALLTYPE SetForceFeedbackMotorGain(uint32_t motorIndex, float masterGain) override;
    HRESULT STDMETHODCALLTYPE SetHapticMotorState(uint32_t motorIndex, const void* params) override;
    void STDMETHODCALLTYPE SetRumbleState(const GameInputRumbleParams* params) override;
    void STDMETHODCALLTYPE SetInputSynchronizationState(bool) override {}
    void STDMETHODCALLTYPE SendInputSynchronizationHint() override {}
    void STDMETHODCALLTYPE PowerOff() override {}
    // Raw HID (RT-175): reports of the pad's HID device through rt::gameinput::HidBackend (SDL_hid_* by default).
    HRESULT STDMETHODCALLTYPE CreateRawDeviceReport(uint32_t reportId, int32_t reportKind, void** report) override;
    HRESULT STDMETHODCALLTYPE GetRawDeviceFeature(uint32_t reportId, void** report) override;
    HRESULT STDMETHODCALLTYPE SetRawDeviceFeature(void* report) override;
    HRESULT STDMETHODCALLTYPE SendRawDeviceOutput(void* report) override;
    HRESULT STDMETHODCALLTYPE SendRawDeviceOutputWithResponse(void*, void** response) override {
        if (response) *response = nullptr;
        return E_NOTIMPL;
    }
    HRESULT STDMETHODCALLTYPE ExecuteRawDeviceIoControl(uint32_t, size_t, const void*, size_t, void*, size_t* size) override {
        if (size) *size = 0;
        return E_NOTIMPL;
    }
    bool STDMETHODCALLTYPE AcquireExclusiveRawDeviceAccess(uint64_t) override { return false; }
    void STDMETHODCALLTYPE ReleaseExclusiveRawDeviceAccess() override {}

    // Shared state, under g().mu unless noted.
    GameInputDeviceInfo info{};
    std::string name;
    GameInputString display_name{};
    GameInputControllerAxisInfo axis_info[kAxisCount]{};
    GameInputControllerButtonInfo button_info[kButtonCount]{};
    GameInputGamepadInfo gamepad_info{};
    GameInputUiNavigationInfo ui_info{};
    GameInputMotionInfo motion_info{};  // filled only when the pad has sensors; info.motionInfo points at it
    GameInputForceFeedbackMotorInfo ff_motor_info{};  // filled when SDL opened the pad as a haptic device
    GameInputHapticFeedbackMotorInfo haptic_motor_info[kHapticMotors]{};   // the trigger motors, when the pad has them
    GameInputHapticWaveformInfo haptic_waveform_info[kHapticMotors]{};     // one waveform each, an intensity
    std::atomic<uint32_t> status{GameInputDeviceNoStatus};
    std::deque<Reading*> history;  // one reference each, oldest first; the newest is the current reading
    uint64_t evicted_order = 0;    // order of the newest reading dropped from history
    Reading* latest() const { return history.empty() ? nullptr : history.back(); }
    uint64_t seq = 0;
    Rumble wanted;
    float motor_gain = 1.0f;                     // SetForceFeedbackMotorGain, 0..1
    std::vector<ForceFeedbackEffect*> effects;    // every live effect, so the poll thread can drive them
    std::vector<int> retired;                     // SDL effect ids whose object is gone: destroyed on the poll thread
    float haptic_intensity[kHapticMotors] = {};   // SetHapticMotorState, 0..1
    uint64_t haptic_until[kHapticMotors] = {};    // when the request ends, 0 for "until changed"
    GameInputBatteryState battery{0, 0, 0, 0, GameInputBatteryUnknown};
    AppLocalDeviceId base_id{};  // id_bytes of the identity alone: twins share it, info.deviceId tells them apart

    // Poll-thread state.
    SDL_Gamepad* pad = nullptr;
    SDL_JoystickID id = 0;
    int player = -1;
    bool is_dualsense = false, has_rumble = false, has_trigger_rumble = false, has_motion = false;
    bool has_trigger_output = false;  // the pad backs trigger motors: rumble, DualSense effects or haptic motors
    void* haptic = nullptr;           // SDL_Haptic behind the joystick, when SDL has one for this pad
    uint32_t ff_features = 0;         // the SDL_HAPTIC_* bits it reports: which effect kinds it can play
    int sent_gain = -1;               // device gain 0..100 last sent, so a steady gain costs one call
    uint32_t haptic_motors = 0;       // trigger haptic motors the pad reports (0 or 2)
    rt::gameinput::OutputState sent{};
    leonida::InputFamily family = leonida::InputFamily::Xbox;  // what kind of device it really is (active_input)
    Snapshot last_snap{};  // what the latest reading holds
    uint64_t last_reading_us = 0;  // for the motion cadence: readings stream at ~125 Hz on pads with sensors
    uint64_t motors_sent_at = 0, triggers_sent_at = 0, battery_at = 0;
    bool ds_sent_once = false;

    // Raw HID (RT-175). hid is the backend's handle, null when the pad has no HID device the backend could open;
    // hid_mu serialises every call on it, from the poll thread (reads) and the title's threads (output, features).
    // The report tables come from the HID report descriptor; info.input/output/featureReportInfo point into them.
    std::mutex hid_mu;
    void* hid = nullptr;
    const rt::gameinput::HidBackend* hid_backend = nullptr;  // the backend hid was opened with
    bool hid_ids = false;  // the descriptor declares report ids, so an input report starts with its id byte
    std::vector<GameInputRawDeviceReportInfo> raw_in, raw_out, raw_feature;
    const GameInputRawDeviceReportInfo* find_raw(int32_t kind, uint32_t report_id) const;
    HRESULT send_raw(void* report, GameInputRawDeviceReportKind kind);

private:
    std::atomic<ULONG> refs_{1};
};

Reading::Reading(Device* device, const Snapshot& s, uint64_t seq, uint64_t ts, uint64_t order)
    : device_(device), s_(s), seq_(seq), ts_(ts), order_(order) {
    device_->AddRef();
}
Reading::~Reading() { device_->Release(); }
void Reading::GetDevice(IGameInputDevice** device) {
    if (!device) return;
    device_->AddRef();
    *device = device_;
}

// ---- raw HID (RT-175) -------------------------------------------------------------------------------------------

// One raw report: its own copy of the report info (so it outlives a reconnect that rebuilds the device's tables) and
// payload. There are no item tables (itemCount 0), so the item calls answer false; see docs/gameinput.md "Raw HID".
class RawReport final : public IGameInputRawDeviceReport {
public:
    RawReport(Device* device, const GameInputRawDeviceReportInfo& info, const uint8_t* data, size_t size)
        : device_(device), info_(info), data_(info.size, 0) {
        info_.itemCount = 0;
        info_.items = nullptr;
        if (data) std::memcpy(data_.data(), data, std::min(size, data_.size()));
        device_->AddRef();
    }
    ~RawReport() { device_->Release(); }

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** out) override {
        if (!out) return E_POINTER;
        if (same_guid(riid, IID_IUnknown) || same_guid(riid, IID_IGameInputRawDeviceReport)) {
            AddRef();
            *out = static_cast<IGameInputRawDeviceReport*>(this);
            return S_OK;
        }
        *out = nullptr;
        return E_NOINTERFACE;
    }
    ULONG STDMETHODCALLTYPE AddRef() override { return ++refs_; }
    ULONG STDMETHODCALLTYPE Release() override {
        const ULONG r = --refs_;
        if (r == 0) delete this;
        return r;
    }

    void STDMETHODCALLTYPE GetDevice(IGameInputDevice** device) override;
    const GameInputRawDeviceReportInfo* STDMETHODCALLTYPE GetReportInfo() override { return &info_; }
    size_t STDMETHODCALLTYPE GetRawDataSize() override { return data_.size(); }
    size_t STDMETHODCALLTYPE GetRawData(size_t bufferSize, void* buffer) override {
        const size_t n = buffer ? std::min(bufferSize, data_.size()) : 0;
        if (n) std::memcpy(buffer, data_.data(), n);
        return n;
    }
    // The whole payload at once: a buffer of another size is refused, so a report is never half old, half new.
    bool STDMETHODCALLTYPE SetRawData(size_t bufferSize, const void* buffer) override {
        if (!buffer || bufferSize != data_.size()) return false;
        std::memcpy(data_.data(), buffer, bufferSize);
        return true;
    }
    bool STDMETHODCALLTYPE GetItemValue(uint32_t, int64_t* value) override {
        if (value) *value = 0;
        return false;
    }
    bool STDMETHODCALLTYPE SetItemValue(uint32_t, int64_t) override { return false; }
    bool STDMETHODCALLTYPE ResetItemValue(uint32_t) override { return false; }
    bool STDMETHODCALLTYPE ResetAllItems() override { return false; }

private:
    std::atomic<ULONG> refs_{1};
    Device* device_;
    GameInputRawDeviceReportInfo info_;
    std::vector<uint8_t> data_;
};

void RawReport::GetDevice(IGameInputDevice** device) {
    if (!device) return;
    device_->AddRef();
    *device = device_;
}

// The default backend: SDL's hidapi. A pad is opened by the path SDL reports for it; with no such HID path, by its
// vendor and product id when exactly one game-pad collection (usage page 1, usage 5) matches, so twins never mix.
// A pad without a path (SDL virtual joysticks, the keyboard pad) gets no HID device.
void* sdl_hid_open(const char* path, uint16_t vendor, uint16_t product) {
    if (!path || !*path) return nullptr;
    static const bool ok = SDL_hid_init() == 0;
    if (!ok) return nullptr;
    if (SDL_hid_device* h = SDL_hid_open_path(path)) return h;
    SDL_hid_device_info* list = SDL_hid_enumerate(vendor, product);
    const char* only = nullptr;
    int matches = 0;
    for (SDL_hid_device_info* i = list; i; i = i->next)
        if (i->usage_page == 0x01 && i->usage == 0x05 && ++matches == 1) only = i->path;
    SDL_hid_device* h = (matches == 1 && only) ? SDL_hid_open_path(only) : nullptr;
    SDL_hid_free_enumeration(list);
    return h;
}
void sdl_hid_close(void* h) { SDL_hid_close(static_cast<SDL_hid_device*>(h)); }
int sdl_hid_descriptor(void* h, uint8_t* buf, size_t size) {
    return SDL_hid_get_report_descriptor(static_cast<SDL_hid_device*>(h), buf, size);
}
int sdl_hid_read(void* h, uint8_t* buf, size_t size) {
    return SDL_hid_read_timeout(static_cast<SDL_hid_device*>(h), buf, size, 0);
}
int sdl_hid_write(void* h, const uint8_t* buf, size_t size) {
    return SDL_hid_write(static_cast<SDL_hid_device*>(h), buf, size);
}
int sdl_hid_get_feature(void* h, uint8_t* buf, size_t size) {
    return SDL_hid_get_feature_report(static_cast<SDL_hid_device*>(h), buf, size);
}
int sdl_hid_send_feature(void* h, const uint8_t* buf, size_t size) {
    return SDL_hid_send_feature_report(static_cast<SDL_hid_device*>(h), buf, size);
}
const rt::gameinput::HidBackend kSdlHid = {sdl_hid_open,  sdl_hid_close,       sdl_hid_descriptor, sdl_hid_read,
                                           sdl_hid_write, sdl_hid_get_feature, sdl_hid_send_feature};
std::atomic<const rt::gameinput::HidBackend*> g_hid{&kSdlHid};

bool Reading::GetRawReport(void** report) {
    if (!report) return false;
    *report = nullptr;
    if (!s_.has_raw) return false;
    std::lock_guard lock(device_->hid_mu);
    const GameInputRawDeviceReportInfo* info = device_->find_raw(GameInputRawInputReport, s_.raw_id);
    if (!info) return false;
    *report = static_cast<IGameInputRawDeviceReport*>(new RawReport(device_, *info, s_.raw.data(), s_.raw.size()));
    return true;
}

const GameInputRawDeviceReportInfo* Device::find_raw(int32_t kind, uint32_t report_id) const {
    const std::vector<GameInputRawDeviceReportInfo>* list = kind == GameInputRawInputReport    ? &raw_in
                                                           : kind == GameInputRawOutputReport  ? &raw_out
                                                           : kind == GameInputRawFeatureReport ? &raw_feature
                                                                                               : nullptr;
    if (!list) return nullptr;
    for (const GameInputRawDeviceReportInfo& r : *list)
        if (r.id == report_id) return &r;
    return nullptr;
}

HRESULT Device::CreateRawDeviceReport(uint32_t reportId, int32_t reportKind, void** report) {
    if (!report) return E_POINTER;
    *report = nullptr;
    std::lock_guard lock(hid_mu);
    if (!hid) return E_NOTIMPL;  // no raw I/O on this pad (GameInputDeviceRawIoEnabled is not set)
    const GameInputRawDeviceReportInfo* ri = find_raw(reportKind, reportId);
    if (!ri) return E_INVALIDARG;
    *report = static_cast<IGameInputRawDeviceReport*>(new RawReport(this, *ri, nullptr, 0));
    return S_OK;
}

HRESULT Device::GetRawDeviceFeature(uint32_t reportId, void** report) {
    if (!report) return E_POINTER;
    *report = nullptr;
    std::lock_guard lock(hid_mu);
    if (!hid) return E_NOTIMPL;
    const GameInputRawDeviceReportInfo* ri = find_raw(GameInputRawFeatureReport, reportId);
    if (!ri) return E_INVALIDARG;
    std::vector<uint8_t> buf(ri->size + 1, 0);
    buf[0] = static_cast<uint8_t>(reportId);  // hidapi: the id goes first, 0 for a device without ids
    const int n = hid_backend->get_feature(hid, buf.data(), buf.size());
    if (n < 1) return E_FAIL;
    *report = static_cast<IGameInputRawDeviceReport*>(new RawReport(this, *ri, buf.data() + 1, static_cast<size_t>(n) - 1));
    return S_OK;
}

// SetRawDeviceFeature and SendRawDeviceOutput: any IGameInputRawDeviceReport of the right kind whose id this pad
// declares; the payload is sent with the id byte in front, as hidapi takes it.
HRESULT Device::send_raw(void* report, GameInputRawDeviceReportKind kind) {
    if (!report) return E_INVALIDARG;
    auto* r = static_cast<IGameInputRawDeviceReport*>(report);
    const GameInputRawDeviceReportInfo* given = r->GetReportInfo();
    std::lock_guard lock(hid_mu);
    if (!hid) return E_NOTIMPL;
    if (!given || given->kind != kind) return E_INVALIDARG;
    const GameInputRawDeviceReportInfo* ri = find_raw(kind, given->id);
    if (!ri) return E_INVALIDARG;
    std::vector<uint8_t> buf(ri->size + 1, 0);
    buf[0] = static_cast<uint8_t>(ri->id);
    r->GetRawData(ri->size, buf.data() + 1);
    const rt::gameinput::HidBackend* b = hid_backend;
    const int n = kind == GameInputRawFeatureReport ? b->send_feature(hid, buf.data(), buf.size())
                                                     : b->write(hid, buf.data(), buf.size());
    return n < 0 ? E_FAIL : S_OK;
}
HRESULT Device::SetRawDeviceFeature(void* report) { return send_raw(report, GameInputRawFeatureReport); }
HRESULT Device::SendRawDeviceOutput(void* report) { return send_raw(report, GameInputRawOutputReport); }

// ---- callbacks --------------------------------------------------------------------------------------------------

enum class CbType { Device, Reading, Guide, KeyboardLayout };

struct Callback {
    GameInputCallbackToken token;
    CbType type;
    Device* device;  // filter, or null for all
    uint32_t kind;
    uint32_t status_filter;
    void* context;
    void* fn;
    bool active;
};

struct Event {
    CbType type;
    Device* device;          // referenced
    Reading* reading;        // referenced, Reading events only
    uint64_t timestamp;
    uint32_t current, previous;
    bool pressed;
    GameInputCallbackToken only;  // enumeration: deliver to this callback only
};

// One effect a title created, on the device it belongs to. The object and its parameters belong to the title; SDL's
// effect is created, run and destroyed on the poll thread, so a title call only writes its state and bumps `version_`,
// which the poll thread compares before it touches SDL. A new effect is Running: CreateForceFeedbackEffect already
// carried the parameters, and the documented use (create once, SetParams per frame) never reaches SetState.
class ForceFeedbackEffect final : public IGameInputForceFeedbackEffect {
public:
    ForceFeedbackEffect(Device* device, uint32_t motor, const GameInputForceFeedbackParams& params);
    ~ForceFeedbackEffect();

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** out) override;
    ULONG STDMETHODCALLTYPE AddRef() override;
    ULONG STDMETHODCALLTYPE Release() override;
    void STDMETHODCALLTYPE GetDevice(IGameInputDevice** device) override;
    uint32_t STDMETHODCALLTYPE GetMotorIndex() override;
    float STDMETHODCALLTYPE GetGain() override;
    void STDMETHODCALLTYPE SetGain(float gain) override;
    void STDMETHODCALLTYPE GetParams(GameInputForceFeedbackParams* params) override;
    bool STDMETHODCALLTYPE SetParams(const GameInputForceFeedbackParams* params) override;
    GameInputFeedbackEffectState STDMETHODCALLTYPE GetState() override;
    void STDMETHODCALLTYPE SetState(GameInputFeedbackEffectState state) override;

    // Poll thread: make SDL match the effect's parameters, gain and state. `muted` says every output is off
    // (LEONIDA_RUMBLE=0); the motor's master gain is SDL's device gain, set once by apply_effects.
    void sync(Device* device, uint64_t now, bool muted);
    // Poll thread, on unplug: the SDL device went with the pad, so forget its effect id without touching SDL.
    void orphan();
    // Poll thread, under g().mu: a reference unless the title's last Release already ran (the destructor then waits
    // for g().mu to unlink the effect, so it is still in the device's list but must not be used or revived).
    bool add_ref_if_live() { return rt::gameinput::add_ref_if_live(refs_); }

private:
    std::atomic<ULONG> refs_{1};
    Device* device_;
    uint32_t motor_;
    // Written by title calls under g().mu and read by the poll thread.
    GameInputForceFeedbackParams params_;
    float gain_ = 1.0f;
    GameInputFeedbackEffectState state_ = GameInputFeedbackRunning;
    uint64_t version_ = 1;
    // Poll thread only.
    uint64_t applied_ = 0;   // the version SDL has
    int sdl_effect_ = -1;
    bool sdl_running_ = false;
    bool endless_ = true;    // GameInput gave no duration: re-run before the SDL length runs out
    uint64_t run_at_ = 0, run_for_ = 0;
};

class Dispatcher;

struct State {
    std::recursive_mutex mu;       // everything below
    std::mutex poll_mu;            // one poller at a time
    std::recursive_mutex deliver_mu;  // held while callbacks run; UnregisterCallback waits on it
    std::vector<Device*> devices;  // every device seen, for the process lifetime
    std::vector<Device*> connected;  // open now, in connection order
    std::vector<Callback> callbacks;
    std::vector<Event> queue;
    GameInputCallbackToken next_token = 1;
    uint64_t last_order = 0;  // Reading::order of the newest reading
    int dispatchers = 0;
    HANDLE wait_event = nullptr;  // auto-reset, signalled when a dispatcher has work
    rt::gameinput::Config config;
    bool config_read = false;
    bool sdl_ready = false, sdl_failed = false;
    bool thread_started = false;
    std::atomic<DWORD> delivering_thread{0};  // written by the delivering thread, read unlocked by UnregisterCallback
};

// Never destroyed: the poll thread may still run while the process exits.
State& g() {
    static State* s = new State;
    return *s;
}

void queue_event(const Event& e) {
    if (e.device) e.device->AddRef();
    if (e.reading) e.reading->AddRef();
    g().queue.push_back(e);
}

bool matches(const Callback& c, const Event& e) {
    if (!c.active || c.type != e.type) return false;
    if (e.only && e.only != c.token) return false;
    if (c.device && c.device != e.device) return false;
    switch (e.type) {
        case CbType::Device:
            if (!(c.kind & e.device->info.supportedInput) && c.kind != GameInputKindUnknown) return false;
            return e.only || ((e.current ^ e.previous) & c.status_filter);
        case CbType::Reading:
            return c.kind == GameInputKindUnknown || (c.kind & e.device->info.supportedInput);
        default:
            return true;
    }
}

void call(const Callback& c, const Event& e) {
    switch (c.type) {
        case CbType::Device:
            reinterpret_cast<GameInputDeviceCallback>(c.fn)(c.token, c.context, e.device, e.timestamp, e.current, e.previous);
            break;
        case CbType::Reading:
            reinterpret_cast<GameInputReadingCallback>(c.fn)(c.token, c.context, e.reading, false);
            break;
        case CbType::Guide:
            reinterpret_cast<GameInputGuideButtonCallback>(c.fn)(c.token, c.context, e.device, e.timestamp, e.pressed);
            break;
        case CbType::KeyboardLayout:
            break;
    }
}

// Runs one queued event's callbacks on the calling thread, without holding the state lock.
void deliver_event(const Event& e) {
    State& s = g();
    std::vector<Callback> targets;
    {
        std::lock_guard lock(s.mu);
        for (const Callback& c : s.callbacks)
            if (matches(c, e)) targets.push_back(c);
    }
    for (const Callback& c : targets) {
        bool still = false;
        {
            std::lock_guard lock(s.mu);
            for (const Callback& now : s.callbacks)
                if (now.token == c.token) still = now.active;
        }
        if (still) call(c, e);
    }
    if (e.device) e.device->Release();
    if (e.reading) e.reading->Release();
}

// Runs queued callbacks on the calling thread, without holding the state lock. Returns whether more events arrived
// meanwhile.
bool deliver() {
    State& s = g();
    std::lock_guard dl(s.deliver_mu);
    std::vector<Event> events;
    {
        std::lock_guard lock(s.mu);
        events.swap(s.queue);
    }
    s.delivering_thread = GetCurrentThreadId();
    for (const Event& e : events) deliver_event(e);
    s.delivering_thread = 0;
    std::lock_guard lock(s.mu);
    return !s.queue.empty();
}

// IGameInputDispatcher::Dispatch: runs the events queued when the call starts, oldest first: at least one even with a
// quota of 0, and no further event once quota_us microseconds have passed (an event already started runs to the end,
// so the quota is a target, as on Windows). Events queued during the call wait for the next one, so a busy poll thread
// cannot keep the title inside Dispatch even with an unlimited quota. Events left over go back to the front of the
// queue, ahead of newer ones, and the wait handle is signalled again while any remain. Returns whether events remain.
bool deliver_within(uint64_t quota_us) {
    State& s = g();
    std::lock_guard dl(s.deliver_mu);
    const auto start = std::chrono::steady_clock::now();
    std::vector<Event> batch;
    {
        std::lock_guard lock(s.mu);
        batch.swap(s.queue);
    }
    s.delivering_thread = GetCurrentThreadId();
    size_t next = 0;
    while (next < batch.size()) {
        if (next > 0) {
            const auto spent = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start);
            if (static_cast<uint64_t>(spent.count()) >= quota_us) break;
        }
        deliver_event(batch[next++]);
    }
    s.delivering_thread = 0;
    std::lock_guard lock(s.mu);
    if (next < batch.size()) s.queue.insert(s.queue.begin(), batch.begin() + static_cast<std::ptrdiff_t>(next), batch.end());
    const bool remaining = !s.queue.empty();
    if (remaining && s.wait_event) SetEvent(s.wait_event);
    return remaining;
}

// ---- SDL backend (poll thread only) -----------------------------------------------------------------------------

uint16_t to_u16(float v) { return static_cast<uint16_t>(std::lround(std::clamp(v, 0.0f, 1.0f) * 65535.0f)); }

float axis(SDL_Gamepad* p, SDL_GamepadAxis a) { return std::max(-1.0f, SDL_GetGamepadAxis(p, a) / 32767.0f); }

float norm(float v, float max) { return std::clamp(v / max, -1.0f, 1.0f); }

bool is_keyboard_pad(SDL_JoystickID jid);  // with the keyboard pad below

// What kind of device a pad really is, for active_input(): SDL's gamepad type, which it takes from the vendor and
// product id. Anything SDL does not place (a generic or unknown pad, a GameCube adapter) counts as Xbox, the layout
// GameInput itself describes.
leonida::InputFamily family_of(SDL_JoystickID jid, SDL_Gamepad* p) {
    using leonida::InputFamily;
    if (is_keyboard_pad(jid)) return InputFamily::Keyboard;
    switch (SDL_GetGamepadType(p)) {
    case SDL_GAMEPAD_TYPE_PS5: return InputFamily::PlayStation5;
    case SDL_GAMEPAD_TYPE_PS4: return InputFamily::PlayStation4;
    case SDL_GAMEPAD_TYPE_PS3: return InputFamily::PlayStation3;
    case SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_PRO:
    case SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_LEFT:
    case SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_RIGHT:
    case SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_PAIR: return InputFamily::Nintendo;
    default: return InputFamily::Xbox;
    }
}

// ---- active input (src/active_input.h) ----
//
// The tracker is poll-thread state, under poll_mu like the rest of the poll. What it answers is published in one word,
// so active_input() needs no lock and never sees a family from one change with the generation of another: the family
// in bits 0-7, forced in bit 8, the generation from bit 16.
leonida::ActiveInputTracker g_tracker;
std::atomic<uint64_t> g_active{0};

void publish_active() {
    g_active.store(static_cast<uint64_t>(g_tracker.active()) | (g_tracker.forced() ? 1ull << 8 : 0) |
                       (g_tracker.generation() << 16),
                   std::memory_order_release);
}

leonida::InputSample sample_of(const Snapshot& s) {
    static_assert(kButtonCount <= 32, "every SDL gamepad button needs a bit");
    leonida::InputSample out;
    for (uint32_t i = 0; i < kButtonCount; ++i)  // all of SDL's buttons: guide, touchpad and paddles included
        if (s.buttons[i]) out.buttons |= 1u << i;
    out.left_trigger = s.gamepad.leftTrigger;
    out.right_trigger = s.gamepad.rightTrigger;
    out.left_x = s.gamepad.leftThumbstickX;
    out.left_y = s.gamepad.leftThumbstickY;
    out.right_x = s.gamepad.rightThumbstickX;
    out.right_y = s.gamepad.rightThumbstickY;
    return out;
}

// LEONIDA_PROMPTS and LEONIDA_PROMPTS_SWITCH_MS, applied at each poll: reload_config can change them at any time.
void apply_prompt_config() {
    std::optional<leonida::InputFamily> want;
    uint32_t switch_ms;
    {
        std::lock_guard lock(g().mu);
        want = g().config.prompts;
        switch_ms = g().config.prompts_switch_ms;
    }
    leonida::ActiveInputTracker::Thresholds t = g_tracker.thresholds();
    t.min_switch_us = uint64_t{switch_ms} * 1000;
    g_tracker.set_thresholds(t);
    const std::optional<leonida::InputFamily> now =
        g_tracker.forced() ? std::optional<leonida::InputFamily>(g_tracker.active()) : std::nullopt;
    if (want == now) return;
    g_tracker.force(want);
    publish_active();
    if (want)
        std::fprintf(stderr, "[rt] gameinput: LEONIDA_PROMPTS: input family set to %s\n",
                     leonida::input_family_name(*want));
    else
        std::fprintf(stderr, "[rt] gameinput: LEONIDA_PROMPTS cleared: input family follows the devices (%s now)\n",
                     leonida::input_family_name(g_tracker.active()));
}

Snapshot read_pad(Device* d) {
    SDL_Gamepad* p = d->pad;
    Snapshot s;
    auto down = [p](SDL_GamepadButton b) { return SDL_GetGamepadButton(p, b); };
    struct Map { SDL_GamepadButton sdl; uint32_t gi; };
    static const Map kMap[] = {
        {SDL_GAMEPAD_BUTTON_START, GameInputGamepadMenu},
        {SDL_GAMEPAD_BUTTON_BACK, GameInputGamepadView},
        {SDL_GAMEPAD_BUTTON_SOUTH, GameInputGamepadA},
        {SDL_GAMEPAD_BUTTON_EAST, GameInputGamepadB},
        {SDL_GAMEPAD_BUTTON_WEST, GameInputGamepadX},
        {SDL_GAMEPAD_BUTTON_NORTH, GameInputGamepadY},
        {SDL_GAMEPAD_BUTTON_DPAD_UP, GameInputGamepadDPadUp},
        {SDL_GAMEPAD_BUTTON_DPAD_DOWN, GameInputGamepadDPadDown},
        {SDL_GAMEPAD_BUTTON_DPAD_LEFT, GameInputGamepadDPadLeft},
        {SDL_GAMEPAD_BUTTON_DPAD_RIGHT, GameInputGamepadDPadRight},
        {SDL_GAMEPAD_BUTTON_LEFT_SHOULDER, GameInputGamepadLeftShoulder},
        {SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER, GameInputGamepadRightShoulder},
        {SDL_GAMEPAD_BUTTON_LEFT_STICK, GameInputGamepadLeftThumbstick},
        {SDL_GAMEPAD_BUTTON_RIGHT_STICK, GameInputGamepadRightThumbstick},
    };
    for (const Map& m : kMap)
        if (down(m.sdl)) s.gamepad.buttons |= m.gi;
    // GameInput: sticks in -1..1 with +Y up, triggers in 0..1. SDL: +Y down, -32768..32767, triggers 0..32767.
    s.gamepad.leftThumbstickX = axis(p, SDL_GAMEPAD_AXIS_LEFTX);
    s.gamepad.leftThumbstickY = -axis(p, SDL_GAMEPAD_AXIS_LEFTY);
    s.gamepad.rightThumbstickX = axis(p, SDL_GAMEPAD_AXIS_RIGHTX);
    s.gamepad.rightThumbstickY = -axis(p, SDL_GAMEPAD_AXIS_RIGHTY);
    s.gamepad.leftTrigger = std::max(0.0f, axis(p, SDL_GAMEPAD_AXIS_LEFT_TRIGGER));
    s.gamepad.rightTrigger = std::max(0.0f, axis(p, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER));

    // Controller axes, 0..1 with 0.5 at rest for the sticks (VERIFY: the raw-axis convention of a GameInput gamepad).
    s.axes[0] = (s.gamepad.leftThumbstickX + 1) * 0.5f;
    s.axes[1] = (s.gamepad.leftThumbstickY + 1) * 0.5f;
    s.axes[2] = (s.gamepad.rightThumbstickX + 1) * 0.5f;
    s.axes[3] = (s.gamepad.rightThumbstickY + 1) * 0.5f;
    s.axes[4] = s.gamepad.leftTrigger;
    s.axes[5] = s.gamepad.rightTrigger;
    for (uint32_t i = 0; i < kButtonCount; ++i) s.buttons[i] = down(static_cast<SDL_GamepadButton>(i));
    s.guide = down(SDL_GAMEPAD_BUTTON_GUIDE);

    // Motion, normalised against the maxima motionInfo reports (the ABI's fraction convention). Sensors were enabled
    // when the pad was opened; a pad whose sensor read fails just reports no motion this poll. SDL answers reads
    // with zeros before the device's first sample arrives, and gravity makes a real accelerometer never read
    // exactly zero, so all-zero accel means "no sample yet" and the reading carries no motion at all.
    if (d->has_motion) {
        float accel[3], gyro[3];
        if (SDL_GetGamepadSensorData(p, SDL_SENSOR_ACCEL, accel, 3) &&
            SDL_GetGamepadSensorData(p, SDL_SENSOR_GYRO, gyro, 3) &&
            (accel[0] != 0.0f || accel[1] != 0.0f || accel[2] != 0.0f)) {
            s.has_motion = true;
            s.motion.accelerationX = norm(accel[0], kMaxAcceleration);
            s.motion.accelerationY = norm(accel[1], kMaxAcceleration);
            s.motion.accelerationZ = norm(accel[2], kMaxAcceleration);
            s.motion.angularVelocityX = norm(gyro[0], kMaxAngularVelocity);
            s.motion.angularVelocityY = norm(gyro[1], kMaxAngularVelocity);
            s.motion.angularVelocityZ = norm(gyro[2], kMaxAngularVelocity);
            // SDL's gamepad API has no compass and the shim synthesises no orientation, so those stay zero and
            // unknown rather than a made-up value.
            s.motion.accelerometerAccuracy = GameInputMotionAccuracyHigh;
            s.motion.gyroscopeAccuracy = GameInputMotionAccuracyHigh;
            s.motion.magnetometerAccuracy = GameInputMotionAccuracyUnknown;
            s.motion.orientationAccuracy = GameInputMotionAccuracyUnknown;
        }
    }

    // UI navigation derived from the gamepad (VERIFY against the console's own mapping).
    const uint32_t b = s.gamepad.buttons;
    const float t = 0.5f;
    uint32_t& u = s.ui;
    if (b & GameInputGamepadMenu) u |= GameInputUiNavigationMenu;
    if (b & GameInputGamepadView) u |= GameInputUiNavigationView;
    if (b & GameInputGamepadA) u |= GameInputUiNavigationAccept;
    if (b & GameInputGamepadB) u |= GameInputUiNavigationCancel;
    if (b & GameInputGamepadX) u |= GameInputUiNavigationContext1;
    if (b & GameInputGamepadY) u |= GameInputUiNavigationContext2;
    if (b & GameInputGamepadLeftThumbstick) u |= GameInputUiNavigationContext3;
    if (b & GameInputGamepadRightThumbstick) u |= GameInputUiNavigationContext4;
    if ((b & GameInputGamepadDPadUp) || s.gamepad.leftThumbstickY > t) u |= GameInputUiNavigationUp;
    if ((b & GameInputGamepadDPadDown) || s.gamepad.leftThumbstickY < -t) u |= GameInputUiNavigationDown;
    if ((b & GameInputGamepadDPadLeft) || s.gamepad.leftThumbstickX < -t) u |= GameInputUiNavigationLeft;
    if ((b & GameInputGamepadDPadRight) || s.gamepad.leftThumbstickX > t) u |= GameInputUiNavigationRight;
    if (b & GameInputGamepadLeftShoulder) u |= GameInputUiNavigationPageLeft;
    if (b & GameInputGamepadRightShoulder) u |= GameInputUiNavigationPageRight;
    if (s.gamepad.leftTrigger > t) u |= GameInputUiNavigationPageUp;
    if (s.gamepad.rightTrigger > t) u |= GameInputUiNavigationPageDown;
    if (s.gamepad.rightThumbstickY > t) u |= GameInputUiNavigationScrollUp;
    if (s.gamepad.rightThumbstickY < -t) u |= GameInputUiNavigationScrollDown;
    if (s.gamepad.rightThumbstickX < -t) u |= GameInputUiNavigationScrollLeft;
    if (s.gamepad.rightThumbstickX > t) u |= GameInputUiNavigationScrollRight;
    return s;
}

void id_bytes(const std::string& identity, const SDL_GUID& guid, AppLocalDeviceId* out) {
    std::memcpy(out->value, guid.data, 16);
    for (int half = 0; half < 2; ++half) {  // FNV-1a, two seeds
        uint64_t h = half ? 0x84222325cbf29ce4ull : 0xcbf29ce484222325ull;
        for (unsigned char c : identity) h = (h ^ c) * 0x100000001b3ull;
        std::memcpy(out->value + 16 + half * 8, &h, 8);
    }
}

// Twin n of an identity. Twin 0 is the plain id, so a pad without twins keeps the id it always had. Real identities
// start with "s:", "p:" or are empty, never "t:", so a twin id cannot be another pad's own id.
void id_bytes(const std::string& identity, const SDL_GUID& guid, uint32_t twin, AppLocalDeviceId* out) {
    id_bytes(twin ? "t:" + std::to_string(twin) + ":" + identity : identity, guid, out);
}

void fill_info(Device* d, SDL_Gamepad* p, const AppLocalDeviceId& id) {
    GameInputDeviceInfo& i = d->info;
    i = {};
    i.infoSize = sizeof(GameInputDeviceInfo);
    i.vendorId = SDL_GetGamepadVendor(p);
    i.productId = SDL_GetGamepadProduct(p);
    i.revisionNumber = SDL_GetGamepadProductVersion(p);
    i.usage = {0x01, 0x05};  // HID generic desktop / game pad
    const uint16_t fw = SDL_GetGamepadFirmwareVersion(p);
    i.firmwareVersion = {static_cast<uint16_t>(fw >> 8), static_cast<uint16_t>(fw & 0xFF), 0, 0};
    i.deviceId = id;
    i.deviceRootId = id;
    // Every pad is presented as an Xbox gamepad, which is all a console title expects (VERIFY: whether a title
    // filters on the family).
    i.deviceFamily = GameInputFamilyXboxOne;
    if (SDL_GetGamepadConnectionState(p) == SDL_JOYSTICK_CONNECTION_WIRELESS) i.capabilities |= GameInputDeviceCapabilityWireless;
    // The per-device kind set: the reading-kind searches take the global kSupportedKinds as the outer bound and this
    // as the per-device one, so only pads with sensors answer a motion query.
    i.supportedInput = kBaseKinds;
    if (d->has_rumble) i.supportedRumbleMotors |= GameInputRumbleLowFrequency | GameInputRumbleHighFrequency;
    if (d->has_trigger_output) i.supportedRumbleMotors |= GameInputRumbleLeftTrigger | GameInputRumbleRightTrigger;
    if (d->has_motion) {
        i.supportedInput |= GameInputKindMotion;
        // The maxima GetMotionState's fractions divide by. The magnetometer is not available: SDL's gamepad API has
        // no compass, and maxMagneticFieldStrength 0 says so (Microsoft's GameInputMotionInfo remarks).
        d->motion_info = {kMaxAcceleration, kMaxAngularVelocity, 0.0f};
        i.motionInfo = &d->motion_info;
    }
    // The motors this pad can back. Force feedback is whatever SDL opened a haptic device for with effect kinds of its
    // own; the haptic motors are the trigger motors the pad already has for rumble, which the shim can drive as an
    // intensity. A pad with neither keeps the old answers: no motors at all.
    if (d->haptic) {
        i.forceFeedbackMotorCount = kForceFeedbackMotors;
        i.forceFeedbackMotorInfo = &d->ff_motor_info;
    }
    if (d->haptic_motors) {
        i.hapticFeedbackMotorCount = d->haptic_motors;
        i.hapticFeedbackMotorInfo = d->haptic_motor_info;
    }
    i.controllerAxisCount = kAxisCount;
    i.controllerButtonCount = kButtonCount;
    for (uint32_t a = 0; a < kAxisCount; ++a) {
        d->axis_info[a] = {};
        d->axis_info[a].mappedInputKinds = GameInputKindGamepad;
        d->axis_info[a].isContinuous = true;
        d->axis_info[a].hasRestValue = true;
        d->axis_info[a].restValue = a < 4 ? 0.5f : 0.0f;
        d->axis_info[a].resolution = 65536;
        d->axis_info[a].legacyHidIndex = static_cast<uint16_t>(a);
        d->axis_info[a].legacyDInputIndex = static_cast<uint16_t>(a);
    }
    for (uint32_t b = 0; b < kButtonCount; ++b) {
        d->button_info[b] = {};
        d->button_info[b].mappedInputKinds = GameInputKindGamepad;
        d->button_info[b].legacyHidIndex = static_cast<uint16_t>(b);
        d->button_info[b].legacyDInputIndex = static_cast<uint16_t>(b);
    }
    i.controllerAxisInfo = d->axis_info;
    i.controllerButtonInfo = d->button_info;
    i.gamepadInfo = &d->gamepad_info;
    i.uiNavigationInfo = &d->ui_info;
    const char* n = SDL_GetGamepadName(p);
    d->name = n ? n : "Gamepad";
    uint32_t points = 0;
    for (unsigned char c : d->name) points += (c & 0xC0) != 0x80;
    d->display_name = {static_cast<uint32_t>(d->name.size()), points, d->name.c_str()};
    i.displayName = &d->display_name;
}

// Makes snap the device's current reading. A reading leaves the history once the one after it is more than kHistoryUs
// old, so the last half second stays reachable however long the pad then idles; kHistoryCap bounds a flood. Readings
// the title holds outlive their place here (~Reading only releases the device, which takes no lock). Under g().mu.
Reading* push_reading(Device* d, const Snapshot& snap, uint64_t ts) {
    std::deque<Reading*>& h = d->history;
    d->last_reading_us = ts;
    h.push_back(new Reading(d, snap, ++d->seq, ts, ++g().last_order));
    while (h.size() > kHistoryCap || (h.size() > 1 && h[1]->GetTimestamp() + kHistoryUs < ts)) {
        d->evicted_order = h.front()->order();
        h.front()->Release();
        h.pop_front();
    }
    return h.back();
}

// Raw HID (RT-175): opens the pad's HID device with the current backend and builds the report tables from its report
// descriptor. A pad the backend cannot open, or whose descriptor cannot be read or declares no report, gets no raw
// I/O: no GameInputKindRawDeviceReport, no report tables, and E_NOTIMPL from the raw device calls. Under g().mu.
void open_raw(Device* d, const char* path) {
    std::lock_guard lock(d->hid_mu);
    if (d->hid) d->hid_backend->close(d->hid);
    d->hid = nullptr;
    d->hid_ids = false;
    d->raw_in.clear();
    d->raw_out.clear();
    d->raw_feature.clear();
    const rt::gameinput::HidBackend* b = g_hid.load();
    void* h = b->open(path, d->info.vendorId, d->info.productId);
    if (!h) {
        // SDL reads Xbox pads on Windows through XInput ("XInput#n"); SDL's hidapi skips their XInput HID interface
        // (&IG_ in the path), which carries no full HID reports, so they have no raw I/O here.
        const bool xinput = path && std::strncmp(path, "XInput#", 7) == 0;
        std::fprintf(stderr, "[rt] gameinput: %s: no raw HID (%s)\n", d->name.c_str(),
                     xinput             ? "an XInput pad: SDL's hidapi does not open XInput HID interfaces"
                     : path && *path    ? "its HID device could not be opened"
                                        : "SDL reports no device path");
        return;
    }
    std::vector<uint8_t> desc(4096);
    const int n = b->descriptor(h, desc.data(), desc.size());
    std::vector<rt::gameinput::RawReportSize> reports;
    const bool parsed = n > 0 && rt::gameinput::parse_report_descriptor(desc.data(), static_cast<size_t>(n), &reports);
    if (!parsed || reports.empty()) {
        std::fprintf(stderr, "[rt] gameinput: %s: no raw HID (%s)\n", d->name.c_str(),
                     n <= 0 ? "no report descriptor" : !parsed ? "malformed report descriptor" : "no reports declared");
        b->close(h);
        return;
    }
    for (const rt::gameinput::RawReportSize& r : reports) {
        const GameInputRawDeviceReportInfo info{static_cast<GameInputRawDeviceReportKind>(r.kind), r.id, r.bytes, 0, nullptr};
        (r.kind == GameInputRawInputReport ? d->raw_in : r.kind == GameInputRawOutputReport ? d->raw_out : d->raw_feature)
            .push_back(info);
        d->hid_ids |= r.id != 0;
    }
    d->hid = h;
    d->hid_backend = b;
    GameInputDeviceInfo& i = d->info;
    i.supportedInput |= GameInputKindRawDeviceReport;
    i.inputReportCount = static_cast<uint32_t>(d->raw_in.size());
    i.inputReportInfo = d->raw_in.empty() ? nullptr : d->raw_in.data();
    i.outputReportCount = static_cast<uint32_t>(d->raw_out.size());
    i.outputReportInfo = d->raw_out.empty() ? nullptr : d->raw_out.data();
    i.featureReportCount = static_cast<uint32_t>(d->raw_feature.size());
    i.featureReportInfo = d->raw_feature.empty() ? nullptr : d->raw_feature.data();
}

void close_raw(Device* d) {
    std::lock_guard lock(d->hid_mu);
    if (d->hid) d->hid_backend->close(d->hid);
    d->hid = nullptr;
}

// Drains the input reports pending since the last poll; the newest one the pad declares goes into snap. With none
// pending, snap keeps the previous report, so a quiet pad makes no new reading. Poll thread.
void read_raw(Device* d, Snapshot* snap) {
    snap->has_raw = d->last_snap.has_raw;
    snap->raw_id = d->last_snap.raw_id;
    snap->raw = d->last_snap.raw;
    std::lock_guard lock(d->hid_mu);
    if (!d->hid) return;
    std::vector<uint8_t> buf(4096);
    for (int k = 0; k < 64; ++k) {  // bounded: a pad that floods cannot hold up the poll
        const int n = d->hid_backend->read(d->hid, buf.data(), buf.size());
        if (n <= 0) break;
        uint32_t id = 0;
        const uint8_t* p = buf.data();
        size_t len = static_cast<size_t>(n);
        if (d->hid_ids) {
            id = buf[0];
            ++p;
            --len;
        }
        const GameInputRawDeviceReportInfo* info = d->find_raw(GameInputRawInputReport, id);
        if (!info) continue;
        snap->has_raw = true;
        snap->raw_id = id;
        snap->raw.assign(p, p + std::min(len, static_cast<size_t>(info->size)));
        snap->raw.resize(info->size, 0);
    }
}

void open_pad(SDL_JoystickID jid, uint64_t ts) {
    State& s = g();
    SDL_Gamepad* p = SDL_OpenGamepad(jid);
    if (!p) {
        std::fprintf(stderr, "[rt] gameinput: cannot open gamepad %u: %s\n", static_cast<unsigned>(jid), SDL_GetError());
        return;
    }
    const char* serial = SDL_GetGamepadSerial(p);
    const char* path = SDL_GetGamepadPath(p);
    const std::string identity = (serial && *serial) ? std::string("s:") + serial : (path && *path) ? std::string("p:") + path : "";
    const SDL_GUID guid = SDL_GetGamepadGUIDForID(jid);
    AppLocalDeviceId base{};
    id_bytes(identity, guid, &base);

    const SDL_PropertiesID props = SDL_GetGamepadProperties(p);
    std::lock_guard lock(s.mu);
    // The same pad coming back gets its old object. Pads with the same GUID and serial or path (identical pads with
    // neither, mostly) are twins sharing one base id, told apart by order alone: the lowest disconnected twin is
    // reused, and a new one gets the next free twin id. Which physical pad was which cannot be recovered, so twins
    // that return in a different order swap objects and ids (docs/gameinput.md).
    Device* d = nullptr;
    uint32_t twins = 0;
    for (Device* old : s.devices) {  // creation order, so twins in twin order
        if (std::memcmp(&old->base_id, &base, sizeof base) != 0) continue;
        ++twins;
        if (!d && old->status == GameInputDeviceNoStatus) d = old;
    }
    AppLocalDeviceId id = d ? d->info.deviceId : base;  // a copy: fill_info clears info before it writes the id
    if (!d) {
        const auto taken = [&s](const AppLocalDeviceId& v) {
            for (Device* o : s.devices)
                if (std::memcmp(&o->info.deviceId, &v, sizeof v) == 0) return true;
            return false;
        };
        id_bytes(identity, guid, twins, &id);
        for (uint32_t n = twins + 1; taken(id); ++n) id_bytes(identity, guid, n, &id);
        d = new Device;
        d->base_id = base;
        s.devices.push_back(d);
    }
    d->pad = p;
    d->id = jid;
    d->family = family_of(jid, p);
    d->is_dualsense = SDL_GetGamepadType(p) == SDL_GAMEPAD_TYPE_PS5;
    d->has_rumble = SDL_GetBooleanProperty(props, SDL_PROP_GAMEPAD_CAP_RUMBLE_BOOLEAN, false);
    d->has_trigger_rumble = SDL_GetBooleanProperty(props, SDL_PROP_GAMEPAD_CAP_TRIGGER_RUMBLE_BOOLEAN, false);
    // Motion needs both sensors; SDL's orientation quaternion is left alone (no title asks for it, and a synthesised
    // one would lie). Enabling at open: SDL delivers samples only for enabled sensors.
    d->has_motion = SDL_GamepadHasSensor(p, SDL_SENSOR_GYRO) && SDL_GamepadHasSensor(p, SDL_SENSOR_ACCEL);
    if (d->has_motion) {
        SDL_SetGamepadSensorEnabled(p, SDL_SENSOR_GYRO, kMotionSensorRate);
        SDL_SetGamepadSensorEnabled(p, SDL_SENSOR_ACCEL, kMotionSensorRate);
    }
    d->has_trigger_output = d->has_trigger_rumble ||
                            (d->is_dualsense && s.config.ds_triggers != rt::gameinput::DsTriggerMode::Off);
    // Force feedback: SDL opens a wheel or a stick as a joystick with a haptic device behind it and never through the
    // gamepad API, so it is opened from the joystick and closed with the pad. Only what the device really reports is
    // passed on: no haptic device, no motor, and an effect kind SDL does not list is not offered to a title.
    d->haptic = haptics().open_joystick(SDL_GetGamepadJoystick(p));
    d->ff_features = 0;
    d->ff_motor_info = {};
    d->sent_gain = -1;
    if (d->haptic) {
        const uint32_t features = haptics().features(d->haptic);
        if (features) {
            GameInputForceFeedbackMotorInfo& m = d->ff_motor_info;
            d->ff_features = features;
            m.supportedAxes = GameInputFeedbackAxisLinearX;  // the axis a wheel's or a stick's motor acts on
            m.location = GameInputLocationChassis;
            m.locationId = 0;
            m.maxSimultaneousEffects = static_cast<uint32_t>(std::max(1, haptics().max_effects(d->haptic)));
            m.isConstantEffectSupported = (features & SDL_HAPTIC_CONSTANT) != 0;
            m.isRampEffectSupported = (features & SDL_HAPTIC_RAMP) != 0;
            m.isSineWaveEffectSupported = (features & SDL_HAPTIC_SINE) != 0;
            m.isSquareWaveEffectSupported = (features & SDL_HAPTIC_SQUARE) != 0;
            m.isTriangleWaveEffectSupported = (features & SDL_HAPTIC_TRIANGLE) != 0;
            m.isSawtoothUpWaveEffectSupported = (features & SDL_HAPTIC_SAWTOOTHUP) != 0;
            m.isSawtoothDownWaveEffectSupported = (features & SDL_HAPTIC_SAWTOOTHDOWN) != 0;
            m.isSpringEffectSupported = (features & SDL_HAPTIC_SPRING) != 0;
            m.isFrictionEffectSupported = (features & SDL_HAPTIC_FRICTION) != 0;
            m.isDamperEffectSupported = (features & SDL_HAPTIC_DAMPER) != 0;
            m.isInertiaEffectSupported = (features & SDL_HAPTIC_INERTIA) != 0;
        } else {
            haptics().close(d->haptic);
            d->haptic = nullptr;
        }
    }
    // Haptic motors: the trigger motors, one waveform each. The waveform identifier is this shim's choice, an
    // intensity (HID physical-interface usage 0x0F/0x58), because an intensity is the one thing SDL lets it send.
    d->haptic_motors = 0;
    if (d->has_trigger_output) {
        for (uint32_t m = 0; m < kHapticMotors; ++m) {
            d->haptic_waveform_info[m] = {{0x0F, 0x58}, true, true, true, false, 0};
            d->haptic_motor_info[m] = {};
            d->haptic_motor_info[m].mappedRumbleMotors = m ? GameInputRumbleRightTrigger : GameInputRumbleLeftTrigger;
            d->haptic_motor_info[m].location = GameInputLocationChassis;
            d->haptic_motor_info[m].locationId = m;
            d->haptic_motor_info[m].waveformCount = 1;
            d->haptic_motor_info[m].waveformInfo = &d->haptic_waveform_info[m];
        }
        d->haptic_motors = kHapticMotors;
    }
    d->sent = {};
    d->sent.is_dualsense = d->is_dualsense;
    d->sent.trigger_rumble = d->has_trigger_rumble;
    d->sent.force_feedback = d->haptic != nullptr;
    d->sent.haptic_motors = d->haptic_motors;
    d->ds_sent_once = false;
    d->motors_sent_at = d->triggers_sent_at = d->battery_at = 0;
    d->wanted = {};
    fill_info(d, p, id);
    open_raw(d, path);

    // Player LEDs and SDL's default lightbar colour follow the slot: the lowest one not in use.
    int player = 0;
    for (bool used = true; used; ++player) {
        used = false;
        for (Device* c : s.connected) used |= c->player == player;
        if (!used) break;
    }
    d->player = player;
    SDL_SetGamepadPlayerIndex(p, player);
    if (s.config.lightbar_set && SDL_GetBooleanProperty(props, SDL_PROP_GAMEPAD_CAP_RGB_LED_BOOLEAN, false))
        SDL_SetGamepadLED(p, s.config.lightbar[0], s.config.lightbar[1], s.config.lightbar[2]);

    uint32_t status = kConnectedStatus;
    if (d->info.capabilities & GameInputDeviceCapabilityWireless) status |= GameInputDeviceWireless;
    if (d->hid) status |= GameInputDeviceRawIoEnabled;
    d->status = status;
    s.connected.push_back(d);

    const Snapshot snap = read_pad(d);
    d->last_snap = snap;
    push_reading(d, snap, ts);
    queue_event({CbType::Device, d, nullptr, ts, status, GameInputDeviceNoStatus, false, 0});
    std::fprintf(stderr, "[rt] gameinput: connected %s (%04x:%04x, player %d%s%s%s%s)\n", d->name.c_str(),
                 d->info.vendorId, d->info.productId, player + 1, d->is_dualsense ? ", DualSense trigger effects" : "",
                 d->has_trigger_rumble ? ", trigger rumble" : "", d->has_motion ? ", gyro" : "",
                 d->haptic ? ", force feedback" : "");
}

void close_pad(Device* d, uint64_t ts) {
    State& s = g();
    g_tracker.forget(d->id);  // poll-thread state: a pad that comes back starts from a fresh baseline
    std::lock_guard lock(s.mu);
    const uint32_t old = d->status.exchange(GameInputDeviceNoStatus);
    SDL_CloseGamepad(d->pad);
    close_raw(d);
    d->pad = nullptr;
    d->player = -1;
    // The haptic device belongs to the joystick, which goes with the gamepad. Effects a title still holds stay alive
    // and pick up again when the pad comes back.
    if (d->haptic) {
        haptics().close(d->haptic);
        d->haptic = nullptr;
    }
    d->retired.clear();
    d->ff_features = 0;
    d->ff_motor_info = {};
    d->sent_gain = -1;
    d->haptic_motors = 0;
    for (ForceFeedbackEffect* e : d->effects) e->orphan();
    for (uint32_t m = 0; m < kHapticMotors; ++m) d->haptic_intensity[m] = 0, d->haptic_until[m] = 0;
    if (!d->history.empty()) d->evicted_order = d->history.back()->order();  // the history ends with the session
    for (Reading* r : d->history) r->Release();
    d->history.clear();
    s.connected.erase(std::remove(s.connected.begin(), s.connected.end(), d), s.connected.end());
    queue_event({CbType::Device, d, nullptr, ts, GameInputDeviceNoStatus, old, false, 0});
    std::fprintf(stderr, "[rt] gameinput: disconnected %s\n", d->name.c_str());
}

void ds_effect(float strength, uint8_t out[11]) {
    const rt::gameinput::Config& c = g().config;
    std::memset(out, 0, 11);
    const uint8_t v = static_cast<uint8_t>(std::lround(std::clamp(strength * c.ds_trigger_gain, 0.0f, 1.0f) * 255.0f));
    if (v == 0) {
        out[0] = 0x05;  // effect off
    } else if (c.ds_triggers == rt::gameinput::DsTriggerMode::Resistance) {
        out[0] = 0x01;  // continuous resistance: start position, force
        out[1] = 0;
        out[2] = v;
    } else {
        out[0] = 0x06;  // vibration: frequency, amplitude, start position
        out[1] = c.ds_trigger_frequency;
        out[2] = v;
        out[3] = 0;
    }
}

// The force-feedback effects and the device gain, on the poll thread: SDL's effects are created, updated, run and
// destroyed here and nowhere else, so a title call costs a lock and a copy.
void apply_effects(Device* d, uint64_t now) {
    State& s = g();
    std::vector<ForceFeedbackEffect*> live;
    std::vector<int> retire;
    float motor_gain;
    bool muted;
    {
        std::lock_guard lock(s.mu);
        live.reserve(d->effects.size());
        // Referenced: the title may release one while this runs. An effect whose count already reached 0 is being
        // destroyed (its destructor waits for this lock to unlink it); a plain AddRef revived it, and this loop's own
        // Release below freed it a second time.
        for (ForceFeedbackEffect* e : d->effects)
            if (e->add_ref_if_live()) live.push_back(e);
        retire.swap(d->retired);
        motor_gain = d->motor_gain;
        muted = !s.config.rumble;
    }
    if (d->haptic) {
        for (int id : retire) haptics().destroy_effect(d->haptic, id);
        const int gain = static_cast<int>(std::lround(std::clamp(motor_gain, 0.0f, 1.0f) * 100.0f));
        if (gain != d->sent_gain) {
            haptics().set_gain(d->haptic, gain);
            d->sent_gain = gain;
        }
    }
    for (ForceFeedbackEffect* e : live) {
        e->sync(d, now, muted);
        e->Release();
    }
}

void apply_output(Device* d, uint64_t now) {
    State& s = g();
    Rumble want;
    float haptic[kHapticMotors] = {};
    {
        std::lock_guard lock(s.mu);
        want = d->wanted;
        for (uint32_t m = 0; m < d->haptic_motors; ++m)
            if (!d->haptic_until[m] || now < d->haptic_until[m]) haptic[m] = d->haptic_intensity[m];
    }
    const bool muted = !s.config.rumble;
    if (muted) want = {};
    // The haptic motors sit on the trigger motors, so their intensity and SetRumbleState's trigger level go out on one
    // output: the stronger of the two wins, and a request that has run out is dropped here.
    want.left = std::max(want.left, muted ? 0.0f : haptic[0]);
    want.right = std::max(want.right, muted ? 0.0f : haptic[1]);
    rt::gameinput::OutputState& o = d->sent;

    if (d->has_rumble) {
        const uint16_t lo = to_u16(want.low), hi = to_u16(want.high);
        const bool stale = (lo || hi) && now - d->motors_sent_at > kRumbleRefreshUs;
        if (lo != o.low || hi != o.high || stale) {
            SDL_RumbleGamepad(d->pad, lo, hi, kRumbleDurationMs);
            o.low = lo;
            o.high = hi;
            d->motors_sent_at = now;
        }
    }
    if (d->is_dualsense) {
        // LEONIDA_DS_TRIGGERS=off leaves the triggers alone.
        if (s.config.ds_triggers != rt::gameinput::DsTriggerMode::Off) {
            uint8_t right[11], left[11];
            ds_effect(want.right, right);
            ds_effect(want.left, left);
            if (!d->ds_sent_once || std::memcmp(right, o.ds_right, 11) || std::memcmp(left, o.ds_left, 11)) {
                uint8_t block[kDsEffectSize] = {};
                block[0] = 0x04 | 0x08;  // valid flags: right and left trigger effect
                std::memcpy(block + 10, right, 11);
                std::memcpy(block + 21, left, 11);
                if (SDL_SendGamepadEffect(d->pad, block, sizeof block)) {
                    std::memcpy(o.ds_right, right, 11);
                    std::memcpy(o.ds_left, left, 11);
                    d->ds_sent_once = true;
                }
            }
        }
    } else if (d->has_trigger_rumble) {
        const uint16_t l = to_u16(want.left), r = to_u16(want.right);
        const bool stale = (l || r) && now - d->triggers_sent_at > kRumbleRefreshUs;
        if (l != o.left || r != o.right || stale) {
            SDL_RumbleGamepadTriggers(d->pad, l, r, kRumbleDurationMs);
            o.left = l;
            o.right = r;
            d->triggers_sent_at = now;
        }
    }
    apply_effects(d, now);
}

void refresh_battery(Device* d, uint64_t now) {
    if (now - d->battery_at < kBatteryRefreshUs && d->battery_at) return;
    d->battery_at = now;
    int percent = -1;
    const SDL_PowerState ps = SDL_GetGamepadPowerInfo(d->pad, &percent);
    GameInputBatteryState b{0, 0, 0, 0, GameInputBatteryUnknown};
    switch (ps) {
        case SDL_POWERSTATE_ON_BATTERY: b.status = GameInputBatteryDischarging; break;
        case SDL_POWERSTATE_CHARGING: b.status = GameInputBatteryCharging; break;
        case SDL_POWERSTATE_CHARGED: b.status = GameInputBatteryIdle; break;
        case SDL_POWERSTATE_NO_BATTERY: b.status = GameInputBatteryNotPresent; break;
        default: break;
    }
    if (percent >= 0) {  // relative capacity (VERIFY: GameInput reports the capacity in device units)
        b.remainingCapacity = percent / 100.0f;
        b.fullChargeCapacity = 1.0f;
    }
    std::lock_guard lock(g().mu);
    d->battery = b;
}

// ---- keyboard pad (poll thread only) ----------------------------------------------------------------------------
//
// The title reads gamepads, not the keyboard, so keys reach it as one more pad: an SDL virtual gamepad that this
// code presses buttons on each poll. It then goes through open_pad and read_pad like any controller, so it gets a
// device object, a player slot, readings, callbacks and UI navigation the same way. It has no motors, so rumble sent
// to it is dropped. Keys are read with GetAsyncKeyState, which needs no window or message loop and works under Wine.
// WASD is the left stick, the mouse the right stick, and the right/left mouse buttons the left/right triggers.

constexpr rt::gameinput::KeyBinding kKeyBindings[] = {
    {VK_UP, SDL_GAMEPAD_BUTTON_DPAD_UP, "Up"},
    {VK_DOWN, SDL_GAMEPAD_BUTTON_DPAD_DOWN, "Down"},
    {VK_LEFT, SDL_GAMEPAD_BUTTON_DPAD_LEFT, "Left"},
    {VK_RIGHT, SDL_GAMEPAD_BUTTON_DPAD_RIGHT, "Right"},
    {VK_RETURN, SDL_GAMEPAD_BUTTON_SOUTH, "Enter"},
    {VK_BACK, SDL_GAMEPAD_BUTTON_EAST, "Backspace"},
    {'X', SDL_GAMEPAD_BUTTON_WEST, "X"},
    {'Y', SDL_GAMEPAD_BUTTON_NORTH, "Y"},
    {'Q', SDL_GAMEPAD_BUTTON_LEFT_SHOULDER, "Q"},
    {'E', SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER, "E"},
    {VK_ESCAPE, SDL_GAMEPAD_BUTTON_START, "Esc"},
    {VK_TAB, SDL_GAMEPAD_BUTTON_BACK, "Tab"},
};

// WASD on the left stick: a full press is the edge of the stick; a diagonal sits on the round gate, as a real stick
// reports it, so each axis is 1/sqrt(2) of full.
constexpr int kStickFull = 32767;
constexpr int kStickDiagonal = 23170;  // 32767 / sqrt(2)

// The mouse on the right stick, in SDL axis units per poll (2 ms with the background thread): each poll the stick
// keeps kMouseDecayNum/kMouseDecayDen of its value and adds kMouseAxisPerPixel for every pixel the cursor moved, then
// is clamped to full. Steady movement settles at 8 x 1024 units per pixel per poll, so 4 pixels per poll (2000
// pixels a second) is full deflection; a stopped mouse falls back from full to exactly 0 in 67 polls (134 ms).
// Integer math rounds toward zero, so the stick always reaches center.
constexpr int kMouseAxisPerPixel = 1024;
constexpr int kMouseDecayNum = 7, kMouseDecayDen = 8;

bool async_key_down(int vk) { return (GetAsyncKeyState(vk) & 0x8000) != 0; }

// ---- raw mouse input (its own thread) ----
//
// The cursor stops at the edge of the screen, so cursor deltas stop too. Raw input reports what the mouse itself
// moved, with no edge and no pointer acceleration. A message-only window (never shown) on its own thread receives
// WM_INPUT and adds each movement to a running total; system_cursor_pos then returns that total as an edgeless
// "cursor", and update_keyboard_pad takes the difference between polls exactly as it did with the real cursor.
std::atomic<long long> g_raw_x{0}, g_raw_y{0};  // total raw movement since the thread started, mouse counts
std::atomic<bool> g_raw_ready{false};           // registration worked; system_cursor_pos uses the totals
std::atomic<bool> g_raw_started{false};         // the thread is started once per process

LRESULT CALLBACK raw_mouse_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    if (msg == WM_INPUT) {
        RAWINPUT raw{};
        UINT size = sizeof(raw);
        if (GetRawInputData(reinterpret_cast<HRAWINPUT>(lp), RID_INPUT, &raw, &size, sizeof(RAWINPUTHEADER)) !=
                static_cast<UINT>(-1) &&
            raw.header.dwType == RIM_TYPEMOUSE && !(raw.data.mouse.usFlags & MOUSE_MOVE_ABSOLUTE)) {
            // Relative movement only: absolute reports (remote desktop, some VMs, tablets) are positions, not
            // movement, and would make the stick jump.
            g_raw_x += raw.data.mouse.lLastX;
            g_raw_y += raw.data.mouse.lLastY;  // +Y is down, as for the cursor
        }
    }
    return DefWindowProcW(hwnd, msg, wp, lp);  // also for WM_INPUT, which lets Windows clean up the input
}

void raw_mouse_thread() {
    const HINSTANCE inst = GetModuleHandleW(nullptr);
    WNDCLASSEXW wc{};
    wc.cbSize = sizeof(wc);
    wc.lpfnWndProc = raw_mouse_proc;
    wc.hInstance = inst;
    wc.lpszClassName = L"LeonidaRawMouse";
    RegisterClassExW(&wc);
    // HWND_MESSAGE: a window that is never shown and only receives messages. It must be created on the thread that
    // runs the message loop below, because a window's messages go to the thread that created it.
    HWND hwnd = CreateWindowExW(0, L"LeonidaRawMouse", L"", 0, 0, 0, 0, 0,
                                HWND_MESSAGE, nullptr, inst, nullptr);
    if (!hwnd) {
        std::fprintf(stderr, "[rt] gameinput: no raw mouse input (window: error %lu); using the cursor position\n",
                     GetLastError());
        return;
    }
    RAWINPUTDEVICE dev{};
    dev.usUsagePage = 0x01;  // HID generic desktop
    dev.usUsage = 0x02;      // mouse
    dev.dwFlags = RIDEV_INPUTSINK;  // also while this (invisible) window has no focus; focus is checked per poll
    dev.hwndTarget = hwnd;
    if (!RegisterRawInputDevices(&dev, 1, sizeof(dev))) {
        std::fprintf(stderr, "[rt] gameinput: no raw mouse input (register: error %lu); using the cursor position\n",
                     GetLastError());
        DestroyWindow(hwnd);
        return;
    }
    g_raw_ready = true;
    std::fprintf(stderr, "[rt] gameinput: raw mouse input on (the right stick does not stop at the screen edge)\n");
    MSG m;
    while (GetMessageW(&m, nullptr, 0, 0) > 0) DispatchMessageW(&m);
}

void start_raw_mouse() {
    if (!g_raw_started.exchange(true)) std::thread(raw_mouse_thread).detach();
}

bool system_cursor_pos(int* x, int* y) {
    if (g_raw_ready) {  // the edgeless total; it wraps only after ±2^31 counts, and the poll uses differences
        *x = static_cast<int>(g_raw_x.load());
        *y = static_cast<int>(g_raw_y.load());
        return true;
    }
    POINT p;
    if (!GetCursorPos(&p)) return false;
    *x = p.x;
    *y = p.y;
    return true;
}

bool foreground_is_ours() {
    const HWND w = GetForegroundWindow();
    DWORD pid = 0;
    return w && GetWindowThreadProcessId(w, &pid) && pid == GetCurrentProcessId();
}

std::atomic<rt::gameinput::KeyDownFn> g_key_down{async_key_down};
std::atomic<rt::gameinput::HasFocusFn> g_has_focus{foreground_is_ours};
std::atomic<rt::gameinput::CursorPosFn> g_cursor_pos{system_cursor_pos};

struct KeyboardPad {
    SDL_JoystickID id = 0;
    SDL_Joystick* js = nullptr;
    bool failed = false;  // attaching failed once; not retried every 2 ms
    int right_x = 0, right_y = 0;    // the mouse's right stick, SDL axis units (+Y down)
    int cursor_x = 0, cursor_y = 0;  // the cursor at the last poll
    bool have_cursor = false;        // cursor_x/y are valid; false after focus loss, so the first poll moves nothing
};
KeyboardPad g_keyboard;

bool is_keyboard_pad(SDL_JoystickID jid) { return jid != 0 && jid == g_keyboard.id; }

void detach_keyboard_pad() {
    if (g_keyboard.js) SDL_CloseJoystick(g_keyboard.js);
    if (g_keyboard.id) SDL_DetachVirtualJoystick(g_keyboard.id);
    if (g_keyboard.id) std::fprintf(stderr, "[rt] gameinput: keyboard pad removed\n");
    g_keyboard = {};
}

// Keeps the cursor inside the game window while it has focus, so a click cannot land on another window, and releases
// it on focus loss. Only with raw input (a clipped plain cursor would stop the stick at the window edge) and only for
// a real foreground window of this process: not KeyboardPadMode::Always over a terminal, nor a test's fake focus.
// Re-applied every 100 ms, because Windows drops the clip on some focus and resize changes.
bool g_clipped = false;
uint64_t g_clip_at = 0;

void clip_cursor(bool want) {
    if (!want) {
        if (g_clipped) ClipCursor(nullptr);
        g_clipped = false;
        return;
    }
    const uint64_t now = now_us();
    if (g_clipped && now - g_clip_at < 100'000) return;
    const HWND w = GetForegroundWindow();
    RECT r;
    if (!w || !GetClientRect(w, &r)) return;
    POINT top_left{r.left, r.top}, bottom_right{r.right, r.bottom};
    if (!ClientToScreen(w, &top_left) || !ClientToScreen(w, &bottom_right)) return;
    const RECT screen{top_left.x, top_left.y, bottom_right.x, bottom_right.y};
    if (screen.right > screen.left && screen.bottom > screen.top && ClipCursor(&screen)) {
        g_clipped = true;
        g_clip_at = now;
    }
}

bool attach_keyboard_pad() {
    SDL_VirtualJoystickDesc d;
    SDL_INIT_INTERFACE(&d);
    d.type = SDL_JOYSTICK_TYPE_GAMEPAD;
    d.naxes = SDL_GAMEPAD_AXIS_COUNT;
    d.nbuttons = SDL_GAMEPAD_BUTTON_COUNT;
    d.button_mask = (1u << SDL_GAMEPAD_BUTTON_COUNT) - 1;
    d.axis_mask = (1u << SDL_GAMEPAD_AXIS_COUNT) - 1;
    d.name = "Keyboard pad (Leonida)";
    g_keyboard.id = SDL_AttachVirtualJoystick(&d);
    g_keyboard.js = g_keyboard.id ? SDL_OpenJoystick(g_keyboard.id) : nullptr;
    if (!g_keyboard.js) {
        std::fprintf(stderr, "[rt] gameinput: cannot attach the keyboard pad: %s\n", SDL_GetError());
        detach_keyboard_pad();
        g_keyboard.failed = true;
        return false;
    }
    std::fprintf(stderr, "[rt] gameinput: keyboard pad attached (arrows d-pad, Enter A, Backspace B, X X, Y Y, "
                         "Q/E shoulders, Esc Menu, Tab View, WASD left stick, mouse right stick, right/left "
                         "button LT/RT)\n");
    start_raw_mouse();
    return true;
}

// Before SDL_UpdateGamepads, so the buttons set here are in this poll's reading.
void update_keyboard_pad() {
    float sens;
    rt::gameinput::KeyboardPadMode mode;
    {
        std::lock_guard lock(g().mu);
        mode = g().config.keyboard_pad;
        sens = g().config.mouse_sensitivity;
    }
    if (mode == rt::gameinput::KeyboardPadMode::Off) {
        clip_cursor(false);
        if (g_keyboard.id) detach_keyboard_pad();
        g_keyboard.failed = false;
        return;
    }
    if (!g_keyboard.js && (g_keyboard.failed || !attach_keyboard_pad())) return;

    bool down[SDL_GAMEPAD_BUTTON_COUNT] = {};
    int lx = 0, ly = 0;
    bool aim = false, fire = false;
    // Out of focus every button, stick and trigger is released, so a key held while switching windows does not stay
    // pressed, and the cursor is read afresh on return, so moving it in another window does not swing the stick.
    const bool active = mode == rt::gameinput::KeyboardPadMode::Always || g_has_focus.load()();
    if (active) {
        const rt::gameinput::KeyDownFn key_down = g_key_down.load();
        for (const auto& b : kKeyBindings) down[b.sdl_button] |= key_down(b.virtual_key);
        lx = key_down('D') - key_down('A');  // opposite keys cancel
        ly = key_down('S') - key_down('W');  // SDL's +Y is down
        aim = key_down(VK_RBUTTON);
        fire = key_down(VK_LBUTTON);

        int x = 0, y = 0;
        const bool have = g_cursor_pos.load()(&x, &y);
        const bool moved = have && g_keyboard.have_cursor;
        auto step = [sens](int v, int pixels) {
            const long long n = static_cast<long long>(v) * kMouseDecayNum / kMouseDecayDen +
                                std::lround(static_cast<long long>(pixels) * kMouseAxisPerPixel * sens);
            return static_cast<int>(std::clamp<long long>(n, -kStickFull, kStickFull));
        };
        g_keyboard.right_x = step(g_keyboard.right_x, moved ? x - g_keyboard.cursor_x : 0);
        g_keyboard.right_y = step(g_keyboard.right_y, moved ? y - g_keyboard.cursor_y : 0);
        if (have) g_keyboard.cursor_x = x, g_keyboard.cursor_y = y;
        g_keyboard.have_cursor = have;
    } else {
        g_keyboard.right_x = g_keyboard.right_y = 0;
        g_keyboard.have_cursor = false;
    }
    clip_cursor(active && g_raw_ready && g_cursor_pos.load() == system_cursor_pos && foreground_is_ours());
    for (const auto& b : kKeyBindings) SDL_SetJoystickVirtualButton(g_keyboard.js, b.sdl_button, down[b.sdl_button]);
    const int reach = lx && ly ? kStickDiagonal : kStickFull;
    SDL_SetJoystickVirtualAxis(g_keyboard.js, SDL_GAMEPAD_AXIS_LEFTX, static_cast<Sint16>(lx * reach));
    SDL_SetJoystickVirtualAxis(g_keyboard.js, SDL_GAMEPAD_AXIS_LEFTY, static_cast<Sint16>(ly * reach));
    SDL_SetJoystickVirtualAxis(g_keyboard.js, SDL_GAMEPAD_AXIS_RIGHTX, static_cast<Sint16>(g_keyboard.right_x));
    SDL_SetJoystickVirtualAxis(g_keyboard.js, SDL_GAMEPAD_AXIS_RIGHTY, static_cast<Sint16>(g_keyboard.right_y));
    // SDL maps a full-range virtual axis to the trigger, so -32768 is released, 0 half pulled, 32767 full.
    SDL_SetJoystickVirtualAxis(g_keyboard.js, SDL_GAMEPAD_AXIS_LEFT_TRIGGER, aim ? 32767 : -32768);
    SDL_SetJoystickVirtualAxis(g_keyboard.js, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER, fire ? 32767 : -32768);
}

bool init_sdl() {
    State& s = g();
    if (s.sdl_ready) return true;
    if (s.sdl_failed) return false;
    // No window: keep reading pads when another application has focus.
    SDL_SetHint(SDL_HINT_JOYSTICK_ALLOW_BACKGROUND_EVENTS, "1");
    // Never read pads through the host's GameInput underneath this GameInput shim (SDL's default is already off
    // outside the GDK; this keeps it off if that default changes).
#ifdef SDL_HINT_JOYSTICK_GAMEINPUT
    SDL_SetHint(SDL_HINT_JOYSTICK_GAMEINPUT, "0");
#endif
    SDL_SetGamepadEventsEnabled(false);  // the shim reads state directly; nothing drains an event queue
    if (!SDL_InitSubSystem(SDL_INIT_GAMEPAD)) {
        std::fprintf(stderr, "[rt] gameinput: SDL gamepad init failed: %s; no controllers\n", SDL_GetError());
        s.sdl_failed = true;
        return false;
    }
    SDL_SetGamepadEventsEnabled(false);
    s.sdl_ready = true;
    const int v = SDL_GetVersion();
    std::fprintf(stderr, "[rt] gameinput: SDL %d.%d.%d gamepads\n", SDL_VERSIONNUM_MAJOR(v), SDL_VERSIONNUM_MINOR(v),
                 SDL_VERSIONNUM_MICRO(v));
    return true;
}

void poll_once() {
    State& s = g();
    std::lock_guard pl(s.poll_mu);
    if (!init_sdl()) return;
    apply_prompt_config();
    update_keyboard_pad();
    SDL_UpdateGamepads();
    const uint64_t now = now_us();

    int count = 0;
    SDL_JoystickID* ids = SDL_GetGamepads(&count);
    std::vector<SDL_JoystickID> present(ids, ids + (ids ? count : 0));
    SDL_free(ids);

    std::vector<Device*> connected;
    {
        std::lock_guard lock(s.mu);
        connected = s.connected;
    }
    for (Device* d : connected)
        if (std::find(present.begin(), present.end(), d->id) == present.end() || !SDL_GamepadConnected(d->pad))
            close_pad(d, now);
    for (SDL_JoystickID jid : present) {
        bool known = false;
        {
            std::lock_guard lock(s.mu);
            for (Device* d : s.connected) known |= d->id == jid;
        }
        if (!known) open_pad(jid, now);
    }

    {
        std::lock_guard lock(s.mu);
        connected = s.connected;
    }
    for (Device* d : connected) {
        Snapshot snap = read_pad(d);
        read_raw(d, &snap);
        const bool guide_changed = d->last_snap.guide != snap.guide;
        const bool changed = !(d->last_snap == snap);
        d->last_snap = snap;
        // A newly opened pad's first sample here is its baseline (active_input.h), so connecting a pad with a button
        // held changes nothing.
        if (const leonida::InputEdge edge = g_tracker.observe(d->id, d->family, sample_of(snap), now);
            edge != leonida::InputEdge::None) {
            publish_active();
            if (!g_tracker.forced())
                std::fprintf(stderr, "[rt] gameinput: input in use: %s (%s, %s)\n",
                             leonida::input_family_name(d->family), d->name.c_str(), leonida::input_edge_name(edge));
        }
        // A pad with sensors streams readings at ~125 Hz even when nothing else changes, so motion the title reads
        // between its frames is never stuck at the last button event; pads without sensors only read on change, as
        // before. push_reading stamps last_reading_us, so the cadence follows the actual history.
        const bool motion_tick = snap.has_motion && now - d->last_reading_us >= kMotionReadingUs;
        if (changed || motion_tick) {
            std::lock_guard lock(s.mu);
            queue_event({CbType::Reading, d, push_reading(d, snap, now), now, 0, 0, false, 0});
        }
        if (guide_changed) {
            std::lock_guard lock(s.mu);
            queue_event({CbType::Guide, d, nullptr, now, 0, 0, snap.guide, 0});
        }
        refresh_battery(d, now);
        apply_output(d, now);
    }
}

void poll_and_deliver() {
    poll_once();
    State& s = g();
    bool dispatch;
    bool pending;
    {
        std::lock_guard lock(s.mu);
        dispatch = s.dispatchers > 0;
        pending = !s.queue.empty();
    }
    if (!pending) return;
    if (dispatch) SetEvent(s.wait_event);
    else deliver();
}

void poll_thread() {
    HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
    if (!timer) timer = CreateWaitableTimerW(nullptr, FALSE, nullptr);
    LARGE_INTEGER due;
    due.QuadPart = -20'000;  // 2 ms, relative
    for (;;) {
        poll_and_deliver();
        if (timer && SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE)) WaitForSingleObject(timer, INFINITE);
        else Sleep(1);
    }
}

void ensure_started() {
    State& s = g();
    std::lock_guard lock(s.mu);
    if (!s.config_read) {
        s.config = read_config();
        s.config_read = true;
        s.wait_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    }
    if (s.config.background_thread && !s.thread_started) {
        s.thread_started = true;
        std::thread(poll_thread).detach();
    }
}

// ---- IGameInputDevice methods that need State ----

void Device::GetBatteryState(GameInputBatteryState* state) {
    if (!state) return;
    std::lock_guard lock(g().mu);
    *state = battery;
}

void Device::SetRumbleState(const GameInputRumbleParams* params) {
    std::lock_guard lock(g().mu);
    wanted = params ? Rumble{params->lowFrequency, params->highFrequency, params->leftTrigger, params->rightTrigger} : Rumble{};
}

HRESULT Device::CreateForceFeedbackEffect(uint32_t motorIndex, const void* params, void** effect) {
    if (!effect) return E_POINTER;
    *effect = nullptr;
    if (!params || motorIndex >= info.forceFeedbackMotorCount) return E_INVALIDARG;
    const auto* p = static_cast<const GameInputForceFeedbackParams*>(params);
    // Only what the motor reported. SDL is not asked here: this is a title call, so the cached feature mask decides,
    // and the poll thread is what finds out if SDL refuses the effect anyway.
    if (!valid_kind(p->kind) || !(ff_features & kind_feature(p->kind))) return E_INVALIDARG;
    *effect = new ForceFeedbackEffect(this, motorIndex, *p);
    return S_OK;
}

bool Device::IsForceFeedbackMotorPoweredOn(uint32_t motorIndex) {
    return motorIndex < info.forceFeedbackMotorCount && haptic != nullptr && status.load() != GameInputDeviceNoStatus;
}

void Device::SetForceFeedbackMotorGain(uint32_t motorIndex, float masterGain) {
    std::lock_guard lock(g().mu);
    if (motorIndex < info.forceFeedbackMotorCount) motor_gain = std::clamp(masterGain, 0.0f, 1.0f);
}

HRESULT Device::SetHapticMotorState(uint32_t motorIndex, const void* params) {
    std::lock_guard lock(g().mu);
    if (motorIndex >= haptic_motors) return E_INVALIDARG;
    if (!params) {  // stop the motor
        haptic_intensity[motorIndex] = 0.0f;
        haptic_until[motorIndex] = 0;
        return S_OK;
    }
    const auto* p = static_cast<const GameInputHapticFeedbackParams*>(params);
    if (p->waveformIndex >= haptic_motor_info[motorIndex].waveformCount) return E_INVALIDARG;
    haptic_intensity[motorIndex] = std::clamp(p->intensity, 0.0f, 1.0f);
    // duration and repeatDelay are microseconds; playCount 0 plays until the state is set again. The poll thread sends
    // the level and drops it when the deadline passes.
    uint64_t total = p->playCount == 0 ? 0 : p->duration;
    if (p->playCount > 1) total = p->duration * p->playCount + p->repeatDelay * (p->playCount - 1);
    haptic_until[motorIndex] = total ? now_us() + total : 0;
    return S_OK;
}

// ---- IGameInputForceFeedbackEffect ----------------------------------------------------------------

ForceFeedbackEffect::ForceFeedbackEffect(Device* device, uint32_t motor, const GameInputForceFeedbackParams& params)
    : device_(device), motor_(motor), params_(params) {
    device_->AddRef();
    std::lock_guard lock(g().mu);
    device_->effects.push_back(this);
}

ForceFeedbackEffect::~ForceFeedbackEffect() {
    std::lock_guard lock(g().mu);
    std::vector<ForceFeedbackEffect*>& list = device_->effects;
    list.erase(std::remove(list.begin(), list.end(), this), list.end());
    if (sdl_effect_ >= 0) {  // SDL's effect is destroyed by the poll thread, which is the only thread that may
        device_->retired.push_back(sdl_effect_);
        sdl_effect_ = -1;
    }
    device_->Release();
}

HRESULT STDMETHODCALLTYPE ForceFeedbackEffect::QueryInterface(REFIID riid, void** out) {
    if (!out) return E_POINTER;
    if (same_guid(riid, IID_IUnknown) || same_guid(riid, IID_IGameInputForceFeedbackEffect)) {
        AddRef();
        *out = static_cast<IGameInputForceFeedbackEffect*>(this);
        return S_OK;
    }
    *out = nullptr;
    return E_NOINTERFACE;
}

ULONG STDMETHODCALLTYPE ForceFeedbackEffect::AddRef() { return ++refs_; }

ULONG STDMETHODCALLTYPE ForceFeedbackEffect::Release() {
    const ULONG r = --refs_;
    if (r == 0) delete this;
    return r;
}

void STDMETHODCALLTYPE ForceFeedbackEffect::GetDevice(IGameInputDevice** device) {
    if (!device) return;
    device_->AddRef();
    *device = device_;
}

uint32_t STDMETHODCALLTYPE ForceFeedbackEffect::GetMotorIndex() { return motor_; }

float STDMETHODCALLTYPE ForceFeedbackEffect::GetGain() {
    std::lock_guard lock(g().mu);
    return gain_;
}

void STDMETHODCALLTYPE ForceFeedbackEffect::SetGain(float gain) {
    std::lock_guard lock(g().mu);
    gain_ = std::clamp(gain, 0.0f, 1.0f);
    ++version_;
}

void STDMETHODCALLTYPE ForceFeedbackEffect::GetParams(GameInputForceFeedbackParams* params) {
    if (!params) return;
    std::lock_guard lock(g().mu);
    *params = params_;
}

bool STDMETHODCALLTYPE ForceFeedbackEffect::SetParams(const GameInputForceFeedbackParams* params) {
    if (!params || !valid_kind(params->kind)) return false;
    std::lock_guard lock(g().mu);
    params_ = *params;
    ++version_;
    return true;
}

GameInputFeedbackEffectState STDMETHODCALLTYPE ForceFeedbackEffect::GetState() {
    std::lock_guard lock(g().mu);
    return state_;
}

void STDMETHODCALLTYPE ForceFeedbackEffect::SetState(GameInputFeedbackEffectState state) {
    std::lock_guard lock(g().mu);
    state_ = state;
    ++version_;
}

void ForceFeedbackEffect::orphan() {
    sdl_effect_ = -1;
    sdl_running_ = false;
    applied_ = 0;
}

// What SDL said about the effect, for output_state (gameinput_probe and the tests): the id it returned, whether it is
// running, and its error text when it is not. SDL_GetError is read here, on the poll thread that made the call.
void note_effect(Device* d, int id, bool ran) {
    const char* error = ran ? "" : SDL_GetError();
    std::lock_guard lock(g().mu);
    d->sent.effect_id = id;
    d->sent.effect_ran = ran;
    std::snprintf(d->sent.effect_error, sizeof d->sent.effect_error, "%s", error ? error : "");
}

void ForceFeedbackEffect::sync(Device* d, uint64_t now, bool muted) {
    GameInputForceFeedbackParams params;
    float gain;
    GameInputFeedbackEffectState state;
    uint64_t version;
    {
        std::lock_guard lock(g().mu);
        params = params_;
        gain = gain_;
        state = state_;
        version = version_;
    }
    if (!d->haptic) return;
    // A stopped or paused effect is stopped in SDL: GameInput pauses one effect where SDL pauses the device, so a
    // pause here is a stop that keeps the parameters. LEONIDA_RUMBLE=0 silences this path like every other.
    if (state != GameInputFeedbackRunning || muted) {
        if (sdl_running_) {
            haptics().stop_effect(d->haptic, sdl_effect_);
            sdl_running_ = false;
        }
        return;
    }
    // An effect that played the length GameInput gave it is over, as the API says of a finished effect. The run is
    // forgotten so that a second SetState(Running) starts it again instead of expiring at once.
    if (!endless_ && run_for_ && now >= run_at_ + run_for_) {
        if (sdl_running_) {
            haptics().stop_effect(d->haptic, sdl_effect_);
            sdl_running_ = false;
        }
        endless_ = true;
        run_for_ = 0;
        std::lock_guard lock(g().mu);
        state_ = GameInputFeedbackStopped;
        return;
    }
    const bool stale = sdl_running_ && endless_ && now - run_at_ > kRumbleRefreshUs;
    if (sdl_running_ && version == applied_ && !stale) return;

    SDL_HapticEffect fx;
    bool endless = true;
    uint32_t length_ms = kHapticHoldMs;
    if (!build_effect(params, gain, &endless, &length_ms, &fx)) {
        if (sdl_running_) haptics().stop_effect(d->haptic, sdl_effect_);
        sdl_running_ = false;
        return;
    }
    if (sdl_effect_ < 0) {
        sdl_effect_ = haptics().create_effect(d->haptic, &fx);  // SDL refused: nothing to drive this effect
        if (sdl_effect_ < 0) {
            note_effect(d, -1, false);
            return;
        }
    } else if (!haptics().update_effect(d->haptic, sdl_effect_, &fx)) {
        note_effect(d, sdl_effect_, false);
        return;
    }
    // GameInput's playCount 0 means "until stopped", which is what SDL's 0 iterations means.
    const uint32_t iterations = endless ? 0 : play_count(params);
    if (!haptics().run_effect(d->haptic, sdl_effect_, iterations)) {
        note_effect(d, sdl_effect_, false);
        return;
    }
    note_effect(d, sdl_effect_, true);
    sdl_running_ = true;
    applied_ = version;
    endless_ = endless;
    run_at_ = now;
    run_for_ = endless || iterations == 0 ? 0 : static_cast<uint64_t>(length_ms) * 1000ull * iterations;
}

// ---- IGameInputDispatcher ----

class Dispatcher final : public IGameInputDispatcher {
public:
    Dispatcher() {
        std::lock_guard lock(g().mu);
        ++g().dispatchers;
    }
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** out) override {
        if (!out) return E_POINTER;
        if (same_guid(riid, IID_IUnknown) || same_guid(riid, IID_IGameInputDispatcher)) {
            AddRef();
            *out = static_cast<IGameInputDispatcher*>(this);
            return S_OK;
        }
        *out = nullptr;
        return E_NOINTERFACE;
    }
    ULONG STDMETHODCALLTYPE AddRef() override { return ++refs_; }
    ULONG STDMETHODCALLTYPE Release() override {
        const ULONG r = --refs_;
        if (r == 0) {
            {
                std::lock_guard lock(g().mu);
                --g().dispatchers;
            }
            delete this;
        }
        return r;
    }
    // Runs the queued callbacks on the caller's thread within the quota (see deliver_within).
    bool STDMETHODCALLTYPE Dispatch(uint64_t quotaInMicroseconds) override { return deliver_within(quotaInMicroseconds); }
    HRESULT STDMETHODCALLTYPE OpenWaitHandle(HANDLE* handle) override {
        if (!handle) return E_POINTER;
        *handle = nullptr;
        if (!DuplicateHandle(GetCurrentProcess(), g().wait_event, GetCurrentProcess(), handle, 0, FALSE, DUPLICATE_SAME_ACCESS))
            return HRESULT_FROM_WIN32(GetLastError());
        return S_OK;
    }

private:
    std::atomic<ULONG> refs_{1};
};

// ---- IGameInput ----

Device* as_device(IGameInputDevice* d) { return static_cast<Device*>(d); }

// GetNextReading (forward) and GetPreviousReading: the reading right after or right before the reference in the order
// readings were taken, from the filter device's history or, with no filter, every connected pad's. Readings compare by
// Reading::order, not timestamp, since pads polled together share one. The horizon is the newest reading dropped from
// the histories searched (with no filter the reference's pad counts too, even unplugged): a step that would cross it
// is REFERENCE_READING_TOO_OLD rather than a silent skip.
HRESULT find_reading(IGameInputReading* ref, uint32_t kind, IGameInputDevice* device, IGameInputReading** out,
                     bool forward) {
    if (!out) return E_POINTER;
    *out = nullptr;
    if (!ref) return E_INVALIDARG;
    if (kind != GameInputKindUnknown && !(kind & kSupportedKinds)) return GAMEINPUT_E_READING_NOT_FOUND;
    const Reading* r = static_cast<Reading*>(ref);
    State& s = g();
    std::lock_guard lock(s.mu);
    std::vector<Device*> scope = s.connected;
    uint64_t horizon = 0;
    if (device) {
        Device* dev = as_device(device);
        if (dev->info.deviceFamily == GameInputFamilyAggregate) {
            if (kind != GameInputKindUnknown && !(kind & dev->info.supportedInput))
                return GAMEINPUT_E_READING_NOT_FOUND;
            const uint32_t effective_kind = (kind != GameInputKindUnknown) ? (kind & dev->info.supportedInput) : dev->info.supportedInput;
            scope.clear();
            for (Device* cd : s.connected) {
                if (effective_kind & cd->info.supportedInput) {
                    scope.push_back(cd);
                }
            }
        } else {
            if (dev->status == GameInputDeviceNoStatus) return GAMEINPUT_E_DEVICE_DISCONNECTED;
            if (kind != GameInputKindUnknown && !(kind & dev->info.supportedInput))
                return GAMEINPUT_E_READING_NOT_FOUND;  // the device does not do this kind (motion on a plain pad)
            scope = {dev};
        }
    } else {
        horizon = r->device()->evicted_order;
    }
    for (Device* d : scope) horizon = std::max(horizon, d->evicted_order);
    if (forward ? horizon > r->order() : horizon >= r->order()) return GAMEINPUT_E_REFERENCE_READING_TOO_OLD;
    Reading* best = nullptr;
    for (Device* d : scope)
        for (Reading* c : d->history) {
            const bool beyond = forward ? c->order() > r->order() : c->order() < r->order() && c->order() > horizon;
            if (beyond && (!best || (c->order() < best->order()) == forward)) best = c;  // the nearest one
        }
    if (!best) return GAMEINPUT_E_READING_NOT_FOUND;
    best->AddRef();
    *out = best;
    return S_OK;
}

class GameInput final : public IGameInput {
public:
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** out) override {
        if (!out) return E_POINTER;
        if (same_guid(riid, IID_IUnknown) || same_guid(riid, IID_IGameInput)) {
            AddRef();
            *out = static_cast<IGameInput*>(this);
            return S_OK;
        }
        *out = nullptr;
        return E_NOINTERFACE;
    }
    ULONG STDMETHODCALLTYPE AddRef() override { return ++refs_; }
    ULONG STDMETHODCALLTYPE Release() override { return release_floor(refs_); }  // a process-wide singleton

    uint64_t STDMETHODCALLTYPE GetCurrentTimestamp() override { return now_us(); }

    HRESULT STDMETHODCALLTYPE GetCurrentReading(uint32_t kind, IGameInputDevice* device, IGameInputReading** out) override {
        if (!out) return E_POINTER;
        *out = nullptr;
        if (kind != GameInputKindUnknown && !(kind & kSupportedKinds)) return GAMEINPUT_E_READING_NOT_FOUND;
        std::lock_guard lock(g().mu);
        Reading* best = nullptr;
        if (device) {
            Device* d = as_device(device);
            if (d->info.deviceFamily == GameInputFamilyAggregate) {
                if (kind != GameInputKindUnknown && !(kind & d->info.supportedInput))
                    return GAMEINPUT_E_READING_NOT_FOUND;
                const uint32_t effective_kind = (kind != GameInputKindUnknown) ? (kind & d->info.supportedInput) : d->info.supportedInput;
                for (Device* cd : g().connected) {
                    if (!(effective_kind & cd->info.supportedInput)) continue;
                    if (Reading* r = cd->latest(); r && (!best || r->GetTimestamp() >= best->GetTimestamp())) best = r;
                }
            } else {
                if (d->status == GameInputDeviceNoStatus) return GAMEINPUT_E_DEVICE_DISCONNECTED;
                if (kind != GameInputKindUnknown && !(kind & d->info.supportedInput))
                    return GAMEINPUT_E_READING_NOT_FOUND;  // the device does not do this kind (motion on a plain pad)
                best = d->latest();
            }
        } else {
            for (Device* d : g().connected)  // the most recent reading from any gamepad
                if (Reading* r = d->latest(); r && (!best || r->GetTimestamp() >= best->GetTimestamp())) best = r;
        }
        if (!best) return GAMEINPUT_E_READING_NOT_FOUND;
        best->AddRef();
        *out = best;
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE GetNextReading(IGameInputReading* ref, uint32_t kind, IGameInputDevice* device,
                                             IGameInputReading** out) override {
        return find_reading(ref, kind, device, out, true);
    }
    HRESULT STDMETHODCALLTYPE GetPreviousReading(IGameInputReading* ref, uint32_t kind, IGameInputDevice* device,
                                                 IGameInputReading** out) override {
        return find_reading(ref, kind, device, out, false);
    }
    // E_NOTIMPL is GameInput v0's own answer: Microsoft's reference page says the method "was never implemented in
    // v0" and returns E_NOTIMPL, and v1 removed it. A history lookup here would give a title what no GameInput does.
    HRESULT STDMETHODCALLTYPE GetTemporalReading(uint64_t, IGameInputDevice*, IGameInputReading** out) override {
        if (!out) return E_POINTER;
        *out = nullptr;
        return E_NOTIMPL;
    }

    HRESULT STDMETHODCALLTYPE RegisterReadingCallback(IGameInputDevice* device, uint32_t kind, float, void* context,
                                                      GameInputReadingCallback fn, GameInputCallbackToken* token) override {
        if (token) *token = kInvalidCallbackToken;
        if (!fn) return E_INVALIDARG;
        if (device && as_device(device)->info.deviceFamily == GameInputFamilyAggregate) return E_NOTIMPL;
        return add({0, CbType::Reading, as_device(device), kind, 0, context, reinterpret_cast<void*>(fn), true}, token);
    }

    HRESULT STDMETHODCALLTYPE RegisterDeviceCallback(IGameInputDevice* device, uint32_t kind, uint32_t status_filter,
                                                     GameInputEnumerationKind enumeration, void* context,
                                                     GameInputDeviceCallback fn, GameInputCallbackToken* token) override {
        if (token) *token = kInvalidCallbackToken;
        if (!fn) return E_INVALIDARG;
        if (device && as_device(device)->info.deviceFamily == GameInputFamilyAggregate) return E_NOTIMPL;
        State& s = g();
        GameInputCallbackToken t;
        std::vector<Event> now;
        {
            std::lock_guard lock(s.mu);
            t = s.next_token++;
            s.callbacks.push_back({t, CbType::Device, as_device(device), kind, status_filter, context,
                                   reinterpret_cast<void*>(fn), true});
            if (enumeration != GameInputNoEnumeration) {
                const uint64_t ts = now_us();
                for (Device* d : s.connected) {
                    if (device && as_device(device) != d) continue;
                    if (kind != GameInputKindUnknown && !(kind & d->info.supportedInput)) continue;
                    const uint32_t st = d->status;
                    if (!(st & status_filter)) continue;
                    const Event e{CbType::Device, d, nullptr, ts, st, GameInputDeviceNoStatus, false, t};
                    if (enumeration == GameInputBlockingEnumeration) {
                        d->AddRef();
                        now.push_back(e);
                    } else {
                        queue_event(e);
                    }
                }
            }
        }
        if (token) *token = t;
        // Blocking enumeration: every connected device is reported before this returns, on the caller's thread.
        for (const Event& e : now) {
            fn(t, context, e.device, e.timestamp, e.current, e.previous);
            e.device->Release();
        }
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE RegisterGuideButtonCallback(IGameInputDevice* device, void* context,
                                                          GameInputGuideButtonCallback fn, GameInputCallbackToken* token) override {
        return add({0, CbType::Guide, as_device(device), 0, 0, context, reinterpret_cast<void*>(fn), true}, token);
    }
    // Gamepads have no keyboard layout; the callback is accepted and never runs.
    HRESULT STDMETHODCALLTYPE RegisterKeyboardLayoutCallback(IGameInputDevice* device, void* context,
                                                             GameInputKeyboardLayoutCallback fn, GameInputCallbackToken* token) override {
        return add({0, CbType::KeyboardLayout, as_device(device), 0, 0, context, reinterpret_cast<void*>(fn), true}, token);
    }

    void STDMETHODCALLTYPE StopCallback(GameInputCallbackToken token) override {
        std::lock_guard lock(g().mu);
        for (Callback& c : g().callbacks)
            if (c.token == token) c.active = false;
    }

    // Removes the callback and, unless called from inside a callback, waits for one that is running to return.
    // VERIFY: the timeout is not honoured; the wait is for at most one callback.
    bool STDMETHODCALLTYPE UnregisterCallback(GameInputCallbackToken token, uint64_t) override {
        State& s = g();
        bool found = false;
        {
            std::lock_guard lock(s.mu);
            auto it = std::find_if(s.callbacks.begin(), s.callbacks.end(), [token](const Callback& c) { return c.token == token; });
            if (it != s.callbacks.end()) {
                s.callbacks.erase(it);
                found = true;
            }
        }
        if (found && s.delivering_thread != GetCurrentThreadId()) std::lock_guard wait(s.deliver_mu);
        return found;
    }

    HRESULT STDMETHODCALLTYPE CreateDispatcher(IGameInputDispatcher** out) override {
        if (!out) return E_POINTER;
        *out = new Dispatcher;
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE CreateAggregateDevice(uint32_t kind, IGameInputDevice** out) override {
        if (!out) return E_POINTER;
        *out = nullptr;
        if (kind == GameInputKindUnknown) return E_INVALIDARG;
        // Combining multiple kinds into a single call is unsupported per Microsoft GDK spec
        if ((kind & (kind - 1)) != 0) return E_INVALIDARG;
        constexpr uint32_t kAllowedKinds =
            GameInputKindKeyboard | GameInputKindMouse | GameInputKindGamepad |
            GameInputKindArcadeStick | GameInputKindFlightStick | GameInputKindRacingWheel;
        if (!(kind & kAllowedKinds)) return E_INVALIDARG;

        std::lock_guard lock(g().mu);
        for (Device* d : g().devices) {
            if (d->info.deviceFamily == GameInputFamilyAggregate && d->info.supportedInput == kind) {
                d->AddRef();
                *out = d;
                return S_OK;
            }
        }

        auto* agg = new Device();
        agg->info.deviceFamily = GameInputFamilyAggregate;
        agg->info.supportedInput = kind;
        agg->info.capabilities = GameInputDeviceCapabilityNone;
        agg->status = GameInputDeviceConnected;
        agg->name = "Aggregate GameInput Device";
        agg->display_name.data = agg->name.c_str();
        agg->display_name.sizeInBytes = static_cast<uint32_t>(agg->name.size());
        agg->display_name.codePointCount = static_cast<uint32_t>(agg->name.size());
        agg->info.displayName = &agg->display_name;

        std::memcpy(agg->info.deviceId.value, "LEONIDA_AGGREGATE_DEVICE_ID\0\0\0\0", 32);
        agg->info.deviceId.value[28] = static_cast<uint8_t>(kind & 0xff);
        agg->info.deviceId.value[29] = static_cast<uint8_t>((kind >> 8) & 0xff);

        g().devices.push_back(agg);
        agg->AddRef();
        *out = agg;
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE FindDeviceFromId(const AppLocalDeviceId* id, IGameInputDevice** out) override {
        if (!out) return E_POINTER;
        *out = nullptr;
        if (!id) return E_INVALIDARG;
        std::lock_guard lock(g().mu);
        for (Device* d : g().devices)
            if (std::memcmp(&d->info.deviceId, id, sizeof *id) == 0) {
                d->AddRef();
                *out = d;
                return S_OK;
            }
        return GAMEINPUT_E_DEVICE_NOT_FOUND;
    }
    HRESULT STDMETHODCALLTYPE FindDeviceFromObject(IUnknown*, IGameInputDevice** out) override { return not_found(out); }
    HRESULT STDMETHODCALLTYPE FindDeviceFromPlatformHandle(HANDLE, IGameInputDevice** out) override { return not_found(out); }
    HRESULT STDMETHODCALLTYPE FindDeviceFromPlatformString(LPCWSTR, IGameInputDevice** out) override { return not_found(out); }
    HRESULT STDMETHODCALLTYPE EnableOemDeviceSupport(uint16_t, uint16_t, uint8_t, uint8_t) override { return S_OK; }
    // Input keeps arriving in the background: the runtime has no focus model yet.
    void STDMETHODCALLTYPE SetFocusPolicy(uint32_t) override {}

private:
    static HRESULT not_found(IGameInputDevice** out) {
        if (out) *out = nullptr;
        return GAMEINPUT_E_DEVICE_NOT_FOUND;
    }
    static HRESULT add(Callback c, GameInputCallbackToken* token) {
        if (token) *token = kInvalidCallbackToken;
        if (!c.fn) return E_INVALIDARG;
        std::lock_guard lock(g().mu);
        c.token = g().next_token++;
        g().callbacks.push_back(c);
        if (token) *token = c.token;
        return S_OK;
    }
    std::atomic<ULONG> refs_{1};
};

GameInput& instance() {
    static GameInput* gi = new GameInput;
    return *gi;
}

// HRESULT GameInputCreate(IGameInput** gameInput). One instance per process; each call adds a reference.
HRESULT WINAPI GameInputCreate(IGameInput** out) {
    if (!out) return E_POINTER;
    ensure_started();
    GameInput& gi = instance();
    gi.AddRef();
    *out = &gi;
    return S_OK;
}

}  // namespace

namespace rt::gameinput {

const Config& config() {
    ensure_started();
    return g().config;
}

void reload_config() {
    std::lock_guard lock(g().mu);
    g().config = read_config();
    g().config_read = true;
}

void poll() {
    ensure_started();
    poll_and_deliver();
}

int connected_count() {
    std::lock_guard lock(g().mu);
    return static_cast<int>(g().connected.size());
}

ActiveInput active_input() {
    const uint64_t v = g_active.load(std::memory_order_acquire);
    return {static_cast<leonida::InputFamily>(v & 0xFF), v >> 16, ((v >> 8) & 1) != 0};
}

bool output_state(int i, OutputState* out) {
    std::lock_guard lock(g().mu);
    if (i < 0 || i >= static_cast<int>(g().connected.size()) || !out) return false;
    *out = g().connected[static_cast<size_t>(i)]->sent;
    return true;
}

void set_keyboard_source(KeyDownFn key_down, HasFocusFn has_focus) {
    g_key_down = key_down ? key_down : async_key_down;
    g_has_focus = has_focus ? has_focus : foreground_is_ours;
}

void set_cursor_source(CursorPosFn cursor_pos) { g_cursor_pos = cursor_pos ? cursor_pos : system_cursor_pos; }

void set_haptic_source(const HapticApi* api) { g_haptic_api = api ? api : &kSdlHaptic; }

const KeyBinding* keyboard_bindings(int* count) {
    if (count) *count = static_cast<int>(std::size(kKeyBindings));
    return kKeyBindings;
}

// HID 1.11 section 6.2.2: a short item is one prefix byte (bSize in bits 0-1, 3 meaning 4 bytes; bType in bits 2-3;
// bTag in bits 4-7) and up to four data bytes, little-endian. 0xFE starts a long item (data size, tag, data).
bool parse_report_descriptor(const uint8_t* data, size_t size, std::vector<RawReportSize>* out) {
    if (!out || (!data && size)) return false;
    out->clear();
    struct Globals {
        uint32_t report_size = 0, report_count = 0, report_id = 0;
    };
    Globals g;
    std::vector<Globals> stack;
    struct Key {
        int kind;
        uint32_t id;
        uint64_t bits;
    };
    std::vector<Key> reports;  // first-seen order; sorted at the end
    size_t i = 0;
    while (i < size) {
        const uint8_t prefix = data[i++];
        if (prefix == 0xFE) {  // long item: bDataSize, bLongItemTag, data
            if (i + 2 > size) return false;
            i += 2 + data[i];
            if (i > size) return false;
            continue;
        }
        const size_t n = (prefix & 3) == 3 ? 4 : (prefix & 3);
        if (i + n > size) return false;
        uint32_t v = 0;
        for (size_t k = 0; k < n; ++k) v |= static_cast<uint32_t>(data[i + k]) << (8 * k);
        i += n;
        const int type = (prefix >> 2) & 3, tag = prefix >> 4;
        if (type == 1) {  // global
            if (tag == 7) g.report_size = v;
            else if (tag == 8) {
                if (v == 0 || v > 255) return false;
                g.report_id = v;
            } else if (tag == 9) g.report_count = v;
            else if (tag == 10) stack.push_back(g);
            else if (tag == 11) {
                if (stack.empty()) return false;
                g = stack.back();
                stack.pop_back();
            }
        } else if (type == 0) {  // main: Input 8, Output 9, Feature 11 (Collection 10 and End Collection 12 add nothing)
            const int kind = tag == 8 ? 0 : tag == 9 ? 1 : tag == 11 ? 2 : -1;
            if (kind < 0) continue;
            const uint64_t add = static_cast<uint64_t>(g.report_size) * g.report_count;
            Key* k = nullptr;
            for (Key& r : reports)
                if (r.kind == kind && r.id == g.report_id) k = &r;
            if (!k) {
                reports.push_back({kind, g.report_id, 0});
                k = &reports.back();
            }
            k->bits += add;
            if (k->bits > 65536ull * 8) return false;
        }
    }
    for (const Key& k : reports) out->push_back({k.kind, k.id, static_cast<uint32_t>((k.bits + 7) / 8)});
    std::sort(out->begin(), out->end(),
              [](const RawReportSize& a, const RawReportSize& b) { return a.kind != b.kind ? a.kind < b.kind : a.id < b.id; });
    return true;
}

void set_hid_backend(const HidBackend* backend) { g_hid = backend ? backend : &kSdlHid; }

}  // namespace rt::gameinput

SHIM_TABLE(gameinput,
    {"gameinput.dll", "GameInputCreate", reinterpret_cast<void*>(&GameInputCreate)},
    {"xgameruntime.dll", "GameInputCreate", reinterpret_cast<void*>(&GameInputCreate)},
);
