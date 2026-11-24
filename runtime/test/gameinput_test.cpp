// Checks the GameInput shim the way a title uses it, on SDL virtual joysticks, so no controller is needed.
//
// GameInputCreate is resolved through the shim registry like any import. Real controller backends are turned off
// with SDL hints, then virtual pads are attached and driven: an Xbox-style pad with trigger rumble, a DualSense
// (Sony vendor/product id, so SDL treats it as one and the shim takes the trigger-effect path) and a plain pad
// without trigger motors. The test covers enumeration and device callbacks, readings and their button/axis mapping,
// UI navigation, the guide button, rumble and trigger rumble as SDL receives them, the DualSense effect bytes for
// each LEONIDA_DS_TRIGGERS mode, the lightbar, a dispatcher, unplug/replug, reading history,
// identification of two or three otherwise indistinguishable pads, and which kind of device was used last
// (active_input, LEONIDA_PROMPTS). No game data.
#include <windows.h>

#include <SDL3/SDL.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <string>
#include <thread>
#include <vector>

#include "gameinput.h"
#include "gameinput_abi.h"
#include "shims.h"

using namespace gi;

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

bool approx(float a, float b) { return std::fabs(a - b) < 0.002f; }

// What SDL handed to one virtual pad.
struct Pad {
    SDL_JoystickID id = 0;
    SDL_Joystick* js = nullptr;
    int rumble_calls = 0, trigger_calls = 0, effect_calls = 0, led_calls = 0;
    uint16_t low = 0, high = 0, left = 0, right = 0;
    uint8_t effect[64] = {};
    int effect_size = 0;
    uint8_t led[3] = {};
    int player = -100;
};

bool SDLCALL on_rumble(void* u, Uint16 lo, Uint16 hi) {
    auto* p = static_cast<Pad*>(u);
    ++p->rumble_calls;
    p->low = lo;
    p->high = hi;
    return true;
}
bool SDLCALL on_triggers(void* u, Uint16 l, Uint16 r) {
    auto* p = static_cast<Pad*>(u);
    ++p->trigger_calls;
    p->left = l;
    p->right = r;
    return true;
}
bool SDLCALL on_led(void* u, Uint8 r, Uint8 g, Uint8 b) {
    auto* p = static_cast<Pad*>(u);
    ++p->led_calls;
    p->led[0] = r;
    p->led[1] = g;
    p->led[2] = b;
    return true;
}
bool SDLCALL on_effect(void* u, const void* data, int size) {
    auto* p = static_cast<Pad*>(u);
    ++p->effect_calls;
    p->effect_size = size;
    std::memcpy(p->effect, data, std::min<size_t>(size, sizeof p->effect));
    return true;
}
void SDLCALL on_player(void* u, int index) { static_cast<Pad*>(u)->player = index; }

void attach(Pad& pad, const char* name, uint16_t vendor, uint16_t product, bool trigger_motors, bool sensors = false) {
    SDL_VirtualJoystickDesc d;
    SDL_INIT_INTERFACE(&d);
    d.type = SDL_JOYSTICK_TYPE_GAMEPAD;
    d.vendor_id = vendor;
    d.product_id = product;
    d.naxes = SDL_GAMEPAD_AXIS_COUNT;
    d.nbuttons = SDL_GAMEPAD_BUTTON_COUNT;
    d.button_mask = (1u << SDL_GAMEPAD_BUTTON_COUNT) - 1;
    d.axis_mask = (1u << SDL_GAMEPAD_AXIS_COUNT) - 1;
    d.name = name;
    d.userdata = &pad;
    d.Rumble = on_rumble;
    d.RumbleTriggers = trigger_motors ? on_triggers : nullptr;
    d.SetLED = on_led;
    d.SendEffect = on_effect;
    d.SetPlayerIndex = on_player;
    static const SDL_VirtualJoystickSensorDesc kSensors[] = {{SDL_SENSOR_GYRO, 250.0f}, {SDL_SENSOR_ACCEL, 250.0f}};
    if (sensors) {
        d.nsensors = 2;
        d.sensors = kSensors;
    }
    pad.id = SDL_AttachVirtualJoystick(&d);
    pad.js = pad.id ? SDL_OpenJoystick(pad.id) : nullptr;
    if (pad.js) {  // triggers at rest: the full-range virtual axis maps -32768 to 0
        SDL_SetJoystickVirtualAxis(pad.js, SDL_GAMEPAD_AXIS_LEFT_TRIGGER, -32768);
        SDL_SetJoystickVirtualAxis(pad.js, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER, -32768);
    }
}

void detach(Pad& pad) {
    if (pad.js) SDL_CloseJoystick(pad.js);
    SDL_DetachVirtualJoystick(pad.id);
    pad.js = nullptr;
}

// Callback recorders.
struct DeviceEvent {
    IGameInputDevice* device;
    uint32_t current, previous;
    GameInputCallbackToken token;
};
std::vector<DeviceEvent> g_device_events;
int g_reading_events = 0;
std::vector<bool> g_guide_events;

void CALLBACK on_device(GameInputCallbackToken token, void*, IGameInputDevice* device, uint64_t, uint32_t current,
                        uint32_t previous) {
    g_device_events.push_back({device, current, previous, token});
}
DWORD g_reading_sleep_ms = 0;  // makes on_reading slow, for the Dispatch quota checks
SDL_Joystick* g_requeue_pad = nullptr;  // set: on_reading queues one more reading from inside the callback
bool g_requeue_down = false;

void CALLBACK on_reading(GameInputCallbackToken, void*, IGameInputReading* reading, bool) {
    if (reading) ++g_reading_events;
    if (g_reading_sleep_ms) Sleep(g_reading_sleep_ms);
    if (g_requeue_pad) {
        g_requeue_down = !g_requeue_down;
        SDL_SetJoystickVirtualButton(g_requeue_pad, SDL_GAMEPAD_BUTTON_WEST, g_requeue_down);
        rt::gameinput::poll();
    }
}
void CALLBACK on_guide(GameInputCallbackToken, void*, IGameInputDevice*, uint64_t, bool pressed) {
    g_guide_events.push_back(pressed);
}

void set_env(const char* name, const char* value) { SetEnvironmentVariableA(name, value); }

uint32_t buttons_of(IGameInputReading* r) {
    GameInputGamepadState s{};
    return r && r->GetGamepadState(&s) ? s.buttons : ~0u;
}

IGameInputDevice* device_of(IGameInputReading* r) {  // devices live for the process: no reference is kept
    IGameInputDevice* d = nullptr;
    if (r) r->GetDevice(&d);
    if (d) d->Release();
    return d;
}

IGameInputReading* current(IGameInput* input, IGameInputDevice* device) {
    IGameInputReading* r = nullptr;
    input->GetCurrentReading(GameInputKindGamepad, device, &r);
    return r;
}

// Steps GetNextReading (forward) or GetPreviousReading from ref until a step fails or limit readings are collected.
// The readings are referenced; *last is the final result.
std::vector<IGameInputReading*> walk(IGameInput* input, IGameInputReading* ref, IGameInputDevice* device, bool forward,
                                     size_t limit, HRESULT* last = nullptr) {
    std::vector<IGameInputReading*> out;
    HRESULT hr = E_INVALIDARG;
    for (IGameInputReading* at = ref; at && out.size() < limit; at = out.back()) {
        IGameInputReading* r = nullptr;
        hr = forward ? input->GetNextReading(at, GameInputKindGamepad, device, &r)
                     : input->GetPreviousReading(at, GameInputKindGamepad, device, &r);
        if (hr != S_OK) break;
        out.push_back(r);
    }
    if (last) *last = hr;
    return out;
}

void release_all(std::vector<IGameInputReading*>& v) {
    for (IGameInputReading* r : v) r->Release();
    v.clear();
}

// ---- keyboard pad: a fake key source instead of GetAsyncKeyState ----
bool g_keys[256] = {};
bool g_focus = false;
bool fake_key_down(int vk) { return vk >= 0 && vk < 256 && g_keys[vk]; }
bool fake_focus() { return g_focus; }
int g_cursor_x = 0, g_cursor_y = 0;
bool fake_cursor(int* x, int* y) {
    *x = g_cursor_x;
    *y = g_cursor_y;
    return true;
}

GameInputGamepadState pad_state(IGameInput* input, IGameInputDevice* dev, uint32_t* ui = nullptr) {
    GameInputGamepadState st{};
    IGameInputReading* r = nullptr;
    if (input->GetCurrentReading(GameInputKindGamepad, dev, &r) == S_OK && r) {
        r->GetGamepadState(&st);
        if (ui) {
            GameInputUiNavigationState nav{};
            r->GetUiNavigationState(&nav);
            *ui = nav.buttons;
        }
        r->Release();
    }
    return st;
}

bool sticks_at_rest(const GameInputGamepadState& st) {
    return st.leftThumbstickX == 0 && st.leftThumbstickY == 0 && st.rightThumbstickX == 0 &&
           st.rightThumbstickY == 0 && st.leftTrigger == 0 && st.rightTrigger == 0;
}

// ---- motion: SDL sensors through the shim ----
void send_motion(Pad& pad, float ax, float ay, float az, float gx, float gy, float gz) {
    const Uint64 ts = SDL_GetTicksNS();
    const float accel[3] = {ax, ay, az};
    const float gyro[3] = {gx, gy, gz};
    SDL_SendJoystickVirtualSensorData(pad.js, SDL_SENSOR_ACCEL, ts, accel, 3);
    SDL_SendJoystickVirtualSensorData(pad.js, SDL_SENSOR_GYRO, ts, gyro, 3);
}

GameInputMotionState motion_state(IGameInput* input, IGameInputDevice* dev, bool* ok = nullptr) {
    GameInputMotionState ms{};
    IGameInputReading* r = nullptr;
    const bool got = input->GetCurrentReading(GameInputKindMotion, dev, &r) == S_OK && r && r->GetMotionState(&ms);
    if (r) r->Release();
    if (ok) *ok = got;
    return ms;
}

struct MotionReadings {
    IGameInputDevice* gyro;
    IGameInputDevice* plain;
    int gyro_calls, plain_calls;
};
void CALLBACK on_motion_reading(GameInputCallbackToken, void* context, IGameInputReading* reading, bool) {
    auto* seen = static_cast<MotionReadings*>(context);
    IGameInputDevice* dev = nullptr;
    if (reading) reading->GetDevice(&dev);
    if (dev == seen->gyro) ++seen->gyro_calls;
    if (dev == seen->plain) ++seen->plain_calls;
    if (dev) dev->Release();
}

// ---- raw HID (RT-175) -------------------------------------------------------------------------------------------

using rt::gameinput::RawReportSize;

bool same_reports(const std::vector<RawReportSize>& got, std::initializer_list<RawReportSize> want) {
    if (got.size() != want.size()) return false;
    size_t i = 0;
    for (const RawReportSize& w : want) {
        if (got[i].kind != w.kind || got[i].id != w.id || got[i].bytes != w.bytes) return false;
        ++i;
    }
    return true;
}

// A game pad with numbered reports: input 1 (3 bytes), output 2 (2 bytes), feature 3 (4 bytes).
const uint8_t kRawPadDescriptor[] = {
    0x05, 0x01, 0x09, 0x05, 0xA1, 0x01,  // usage page generic desktop, usage game pad, collection application
    0x85, 0x01, 0x75, 0x08, 0x95, 0x03, 0x81, 0x02,  // report id 1, size 8, count 3, input
    0x85, 0x02, 0x95, 0x02, 0x91, 0x02,              // report id 2, count 2, output
    0x85, 0x03, 0x95, 0x04, 0xB1, 0x02,              // report id 3, count 4, feature
    0xC0,                                            // end collection
};

void test_parse_report_descriptor() {
    std::vector<RawReportSize> r;
    check(rt::gameinput::parse_report_descriptor(kRawPadDescriptor, sizeof kRawPadDescriptor, &r) &&
              same_reports(r, {{0, 1, 3}, {1, 2, 2}, {2, 3, 4}}),
          "raw hid: a descriptor with report ids gives input 1 (3), output 2 (2), feature 3 (4)");
    // No report id: one 16-bit input report under id 0, from two main items that add up (12 + 4 bits).
    const uint8_t no_ids[] = {0x05, 0x01, 0x09, 0x05, 0xA1, 0x01, 0x75, 0x0C, 0x95, 0x01, 0x81, 0x02,
                              0x75, 0x04, 0x81, 0x03, 0xC0};
    check(rt::gameinput::parse_report_descriptor(no_ids, sizeof no_ids, &r) && same_reports(r, {{0, 0, 2}}),
          "raw hid: without report ids one input report of the summed bits (12 + 4 bits = 2 bytes)");
    // Push saves the globals (size 8, count 2) and Pop restores them after a 32-bit item; a long item is skipped.
    const uint8_t push_pop[] = {0x75, 0x08, 0x95, 0x02, 0xA4, 0x75, 0x20, 0x95, 0x01, 0x81, 0x02, 0xB4,
                                0xFE, 0x02, 0x10, 0xAA, 0xBB, 0x81, 0x02};
    check(rt::gameinput::parse_report_descriptor(push_pop, sizeof push_pop, &r) && same_reports(r, {{0, 0, 6}}),
          "raw hid: Push/Pop restore Report Size and Count, a long item is skipped (4 + 2 bytes)");
    const uint8_t truncated[] = {0x75, 0x08, 0x96, 0x01};      // a two-byte Report Count cut after one byte
    const uint8_t id_zero[] = {0x85, 0x00, 0x81, 0x02};       // Report ID 0 is reserved
    const uint8_t pop_empty[] = {0xB4, 0x81, 0x02};           // Pop without Push
    const uint8_t huge[] = {0x77, 0xFF, 0xFF, 0x00, 0x00, 0x97, 0xFF, 0xFF, 0x00, 0x00, 0x81, 0x02};  // > 64 KiB
    check(!rt::gameinput::parse_report_descriptor(truncated, sizeof truncated, &r) &&
              !rt::gameinput::parse_report_descriptor(id_zero, sizeof id_zero, &r) &&
              !rt::gameinput::parse_report_descriptor(pop_empty, sizeof pop_empty, &r) &&
              !rt::gameinput::parse_report_descriptor(huge, sizeof huge, &r),
          "raw hid: a truncated item, Report ID 0, Pop without Push and a report over 64 KiB are refused");
    check(rt::gameinput::parse_report_descriptor(nullptr, 0, &r) && r.empty(), "raw hid: an empty descriptor declares nothing");
}

// The fake HID device behind the raw pad: input reports to hand out, what the shim wrote, the feature reply.
struct FakeRawHid {
    uint16_t vendor = 0x1209, product = 0x0001;  // pid.codes test ids, not a real pad
    int opens = 0, closes = 0;
    std::deque<std::vector<uint8_t>> inputs;
    std::vector<std::vector<uint8_t>> writes, features_sent;
    std::vector<uint8_t> feature_reply;
    bool fail = false;
    HANDLE in_open = nullptr, release_open = nullptr;
};
FakeRawHid g_raw;
void* raw_open(const char*, uint16_t vendor, uint16_t product) {
    if (vendor != g_raw.vendor || product != g_raw.product) return nullptr;
    ++g_raw.opens;
    if (g_raw.release_open) {
        SetEvent(g_raw.in_open);
        WaitForSingleObject(g_raw.release_open, 5000);
    }
    return &g_raw;
}
void raw_close(void*) { ++g_raw.closes; }
int raw_descriptor(void*, uint8_t* buf, size_t size) {
    if (size < sizeof kRawPadDescriptor) return -1;
    std::memcpy(buf, kRawPadDescriptor, sizeof kRawPadDescriptor);
    return static_cast<int>(sizeof kRawPadDescriptor);
}
int raw_read(void*, uint8_t* buf, size_t size) {
    if (g_raw.inputs.empty()) return 0;
    const std::vector<uint8_t> r = g_raw.inputs.front();
    g_raw.inputs.pop_front();
    const size_t n = std::min(size, r.size());
    std::memcpy(buf, r.data(), n);
    return static_cast<int>(n);
}
int raw_write(void*, const uint8_t* buf, size_t size) {
    if (g_raw.fail) return -1;
    g_raw.writes.emplace_back(buf, buf + size);
    return static_cast<int>(size);
}
int raw_get_feature(void*, uint8_t* buf, size_t size) {
    if (g_raw.fail || g_raw.feature_reply.empty() || g_raw.feature_reply[0] != buf[0]) return -1;
    const size_t n = std::min(size, g_raw.feature_reply.size());
    std::memcpy(buf, g_raw.feature_reply.data(), n);
    return static_cast<int>(n);
}
int raw_send_feature(void*, const uint8_t* buf, size_t size) {
    if (g_raw.fail) return -1;
    g_raw.features_sent.emplace_back(buf, buf + size);
    return static_cast<int>(size);
}
const rt::gameinput::HidBackend kFakeRawHid = {raw_open, raw_close, raw_descriptor, raw_read,
                                            raw_write, raw_get_feature, raw_send_feature};

std::vector<uint8_t> raw_bytes(IGameInputRawDeviceReport* r) {
    std::vector<uint8_t> v(r ? r->GetRawDataSize() : 0);
    if (r) r->GetRawData(v.size(), v.data());
    return v;
}

// RT-262: the poll thread takes a reference on a force-feedback effect only while it is alive.
void test_add_ref_if_live() {
    std::atomic<unsigned long> live{1}, dying{0};
    check(rt::gameinput::add_ref_if_live(live) && live.load() == 2, "add_ref_if_live: a live count goes up by one");
    check(!rt::gameinput::add_ref_if_live(dying) && dying.load() == 0,
          "add_ref_if_live: a count of 0 stays 0, so an effect being destroyed isn't revived");
}

void test_raw_hid(IGameInput* input) {
    test_parse_report_descriptor();
    rt::gameinput::set_hid_backend(&kFakeRawHid);
    const size_t events_before = g_device_events.size();
    Pad raw_pad, plain;
    attach(raw_pad, "Leonida virtual raw pad", g_raw.vendor, g_raw.product, false);
    attach(plain, "Leonida virtual pad without HID", 0x045E, 0x02FF, false);
    rt::gameinput::poll();
    std::vector<IGameInputDevice*> found;  // attach order: the raw pad first (see test_motion)
    for (size_t i = events_before; i < g_device_events.size() && found.size() < 2; ++i) {
        const DeviceEvent& e = g_device_events[i];
        if ((e.current & GameInputDeviceConnected) && std::find(found.begin(), found.end(), e.device) == found.end())
            found.push_back(e.device);
    }
    IGameInputDevice* rdev = found.size() > 0 ? found[0] : nullptr;
    IGameInputDevice* pdev = found.size() > 1 ? found[1] : nullptr;
    check(rdev && pdev && g_raw.opens == 1, "raw hid: both pads connected, the backend opened the raw pad only");
    if (!rdev || !pdev) return;
    rdev->AddRef();
    pdev->AddRef();

    const GameInputDeviceInfo* ri = rdev->GetDeviceInfo();
    check((ri->supportedInput & GameInputKindRawDeviceReport) && (rdev->GetDeviceStatus() & GameInputDeviceRawIoEnabled),
          "raw hid: the raw pad reports GameInputKindRawDeviceReport and GameInputDeviceRawIoEnabled");
    const auto* in = static_cast<const GameInputRawDeviceReportInfo*>(ri->inputReportInfo);
    const auto* out = static_cast<const GameInputRawDeviceReportInfo*>(ri->outputReportInfo);
    const auto* feat = static_cast<const GameInputRawDeviceReportInfo*>(ri->featureReportInfo);
    check(ri->inputReportCount == 1 && in && in->kind == GameInputRawInputReport && in->id == 1 && in->size == 3 &&
              in->itemCount == 0 && ri->outputReportCount == 1 && out && out->id == 2 && out->size == 2 &&
              ri->featureReportCount == 1 && feat && feat->id == 3 && feat->size == 4,
          "raw hid: the report tables follow the descriptor (input 1/3, output 2/2, feature 3/4, no items)");
    const GameInputDeviceInfo* pi = pdev->GetDeviceInfo();
    check(!(pi->supportedInput & GameInputKindRawDeviceReport) && pi->inputReportCount == 0 &&
              !(pdev->GetDeviceStatus() & GameInputDeviceRawIoEnabled),
          "raw hid: a pad without a HID device has no raw reports");

    IGameInputReading* reading = nullptr;
    void* report = nullptr;
    check(input->GetCurrentReading(GameInputKindGamepad, rdev, &reading) == S_OK && reading &&
              !(reading->GetInputKind() & GameInputKindRawDeviceReport) && !reading->GetRawReport(&report) && !report,
          "raw hid: before the first input report a reading carries no raw report");
    if (reading) reading->Release();

    g_raw.inputs.push_back({0x01, 0x11, 0x22, 0x33});
    g_raw.inputs.push_back({0x09, 0x99});              // an id the descriptor does not declare: ignored
    rt::gameinput::poll();
    reading = nullptr;
    check(input->GetCurrentReading(GameInputKindRawDeviceReport, rdev, &reading) == S_OK && reading &&
              (reading->GetInputKind() & GameInputKindRawDeviceReport),
          "raw hid: an input report makes a reading of kind GameInputKindRawDeviceReport");
    IGameInputRawDeviceReport* rr = nullptr;
    if (reading && reading->GetRawReport(&report)) rr = static_cast<IGameInputRawDeviceReport*>(report);
    const GameInputRawDeviceReportInfo* rinfo = rr ? rr->GetReportInfo() : nullptr;
    IGameInputDevice* owner = nullptr;
    if (rr) rr->GetDevice(&owner);
    check(rr && rinfo && rinfo->kind == GameInputRawInputReport && rinfo->id == 1 &&
              raw_bytes(rr) == std::vector<uint8_t>({0x11, 0x22, 0x33}) && owner == rdev,
          "raw hid: GetRawReport gives the newest declared input report, without its id byte, of this device");
    if (owner) owner->Release();
    int64_t item = 7;
    check(rr && !rr->GetItemValue(0, &item) && item == 0 && !rr->SetItemValue(0, 1) && !rr->ResetAllItems(),
          "raw hid: the item calls answer false (no item tables)");
    if (rr) rr->Release();
    if (reading) reading->Release();

    void* created = nullptr;
    check(rdev->CreateRawDeviceReport(2, GameInputRawOutputReport, &created) == S_OK && created,
          "raw hid: CreateRawDeviceReport(2, output)");
    auto* outr = static_cast<IGameInputRawDeviceReport*>(created);
    const uint8_t payload[] = {0xAA, 0xBB};
    check(outr && outr->GetRawDataSize() == 2 && raw_bytes(outr) == std::vector<uint8_t>({0, 0}) &&
              !outr->SetRawData(1, payload) && outr->SetRawData(2, payload),
          "raw hid: a new report is zeroed, and SetRawData takes exactly its size");
    check(rdev->SendRawDeviceOutput(outr) == S_OK && !g_raw.writes.empty() &&
              g_raw.writes.back() == std::vector<uint8_t>({0x02, 0xAA, 0xBB}),
          "raw hid: SendRawDeviceOutput writes the id byte then the payload");
    check(rdev->SetRawDeviceFeature(outr) == E_INVALIDARG && rdev->SendRawDeviceOutput(nullptr) == E_INVALIDARG,
          "raw hid: an output report is not a feature report, and a null report is refused");
    g_raw.fail = true;
    check(rdev->SendRawDeviceOutput(outr) == E_FAIL, "raw hid: a failed write is E_FAIL");
    g_raw.fail = false;
    if (outr) outr->Release();

    g_raw.feature_reply = {0x03, 0x01, 0x02, 0x03, 0x04};
    void* fr = nullptr;
    check(rdev->GetRawDeviceFeature(3, &fr) == S_OK && fr &&
              raw_bytes(static_cast<IGameInputRawDeviceReport*>(fr)) == std::vector<uint8_t>({1, 2, 3, 4}),
          "raw hid: GetRawDeviceFeature(3) reads the feature report");
    check(fr && rdev->SetRawDeviceFeature(fr) == S_OK && !g_raw.features_sent.empty() &&
              g_raw.features_sent.back() == std::vector<uint8_t>({0x03, 1, 2, 3, 4}),
          "raw hid: SetRawDeviceFeature sends it back with its id byte");
    if (fr) static_cast<IGameInputRawDeviceReport*>(fr)->Release();
    void* none = reinterpret_cast<void*>(1);
    check(rdev->CreateRawDeviceReport(9, GameInputRawOutputReport, &none) == E_INVALIDARG && !none &&
              rdev->CreateRawDeviceReport(1, 7, &none) == E_INVALIDARG && rdev->GetRawDeviceFeature(2, &none) == E_INVALIDARG,
          "raw hid: an undeclared id or kind is E_INVALIDARG");
    check(pdev->CreateRawDeviceReport(1, GameInputRawInputReport, &none) == E_NOTIMPL && !none &&
              pdev->GetRawDeviceFeature(3, &none) == E_NOTIMPL,
          "raw hid: a pad without raw I/O answers E_NOTIMPL, as before");

    detach(raw_pad);
    detach(plain);
    rt::gameinput::poll();
    check(g_raw.closes == 1 && rt::gameinput::connected_count() == 0, "raw hid: unplugging closes the HID device");
    check(rdev->CreateRawDeviceReport(2, GameInputRawOutputReport, &none) == E_NOTIMPL,
          "raw hid: no raw I/O once the pad is gone");
    rdev->Release();
    pdev->Release();
    rt::gameinput::set_hid_backend(nullptr);
}

void test_raw_report_during_reconnect(IGameInput* input) {
    rt::gameinput::set_hid_backend(&kFakeRawHid);
    Pad raw_pad;
    attach(raw_pad, "Leonida virtual raw pad", g_raw.vendor, g_raw.product, false);
    rt::gameinput::poll();
    g_raw.inputs.push_back({0x01, 0x44, 0x55, 0x66});
    rt::gameinput::poll();
    IGameInputReading* held = nullptr;
    check(input->GetCurrentReading(GameInputKindRawDeviceReport, nullptr, &held) == S_OK && held,
          "raw hid reconnect: a reading with a raw report to hold");
    detach(raw_pad);
    rt::gameinput::poll();
    if (!held) {
        rt::gameinput::set_hid_backend(nullptr);
        return;
    }

    g_raw.in_open = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    g_raw.release_open = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    HANDLE read_done = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    attach(raw_pad, "Leonida virtual raw pad", g_raw.vendor, g_raw.product, false);
    std::thread poller([] { rt::gameinput::poll(); });
    const bool in_open = WaitForSingleObject(g_raw.in_open, 5000) == WAIT_OBJECT_0;
    bool got = false;
    void* report = nullptr;
    std::thread reader([&] {
        got = held->GetRawReport(&report);
        SetEvent(read_done);
    });
    const bool waited = WaitForSingleObject(read_done, 200) == WAIT_TIMEOUT;
    SetEvent(g_raw.release_open);
    poller.join();
    const bool finished = WaitForSingleObject(read_done, 5000) == WAIT_OBJECT_0;
    reader.join();
    check(in_open && waited, "raw hid reconnect: GetRawReport waits while open_raw rebuilds the report tables");
    auto* rr = static_cast<IGameInputRawDeviceReport*>(report);
    check(finished && got && rr && raw_bytes(rr) == std::vector<uint8_t>({0x44, 0x55, 0x66}),
          "raw hid reconnect: then it gives the held reading's report");
    if (rr) rr->Release();
    IGameInputDevice* dev = nullptr;
    held->GetDevice(&dev);
    void* created = nullptr;
    check(dev && (dev->GetDeviceStatus() & GameInputDeviceRawIoEnabled) &&
              dev->CreateRawDeviceReport(2, GameInputRawOutputReport, &created) == S_OK && created,
          "raw hid reconnect: and the reconnected pad has raw I/O again");
    if (created) static_cast<IGameInputRawDeviceReport*>(created)->Release();
    if (dev) dev->Release();
    held->Release();

    CloseHandle(read_done);
    CloseHandle(g_raw.in_open);
    CloseHandle(g_raw.release_open);
    g_raw.in_open = g_raw.release_open = nullptr;
    detach(raw_pad);
    rt::gameinput::poll();
    rt::gameinput::set_hid_backend(nullptr);
}

void test_motion(IGameInput* input) {
    const size_t events_before = g_device_events.size();  // other tests' events are still queued in the vector
    Pad gyro_pad, plain;
    attach(gyro_pad, "Leonida virtual gyro pad", 0x054C, 0x0CE6, false, true);  // Sony ids, sensors on
    attach(plain, "Leonida virtual plain pad", 0x045E, 0x02FF, false);
    rt::gameinput::poll();
    check(rt::gameinput::connected_count() == 2, "motion: both pads connected");
    // The connect events for these two pads are the first two CONNECTED events after `events_before`, deduplicated
    // by device: earlier test sections leave their device callbacks registered, so every connect is delivered more
    // than once, and a disconnect from a previous test may be queued in between.
    std::vector<IGameInputDevice*> found;
    for (size_t i = events_before; i < g_device_events.size() && found.size() < 2; ++i) {
        const DeviceEvent& e = g_device_events[i];
        if (!(e.current & GameInputDeviceConnected)) continue;
        if (std::find(found.begin(), found.end(), e.device) != found.end()) continue;
        found.push_back(e.device);
    }
    IGameInputDevice* mdev = found.size() > 0 ? found[0] : nullptr;  // attach order: gyro pad first
    IGameInputDevice* pdev = found.size() > 1 ? found[1] : nullptr;
    if (mdev) mdev->AddRef();
    if (pdev) pdev->AddRef();
    check(mdev && pdev, "motion: both device objects arrived");
    if (!mdev || !pdev) return;

    const GameInputDeviceInfo* mi = mdev->GetDeviceInfo();
    const GameInputDeviceInfo* pi = pdev->GetDeviceInfo();
    check((mi->supportedInput & GameInputKindMotion) != 0, "motion: the gyro pad reports GameInputKindMotion");
    check((pi->supportedInput & GameInputKindMotion) == 0, "motion: the plain pad does not");
    const GameInputMotionInfo* mmotion = static_cast<const GameInputMotionInfo*>(mi->motionInfo);
    check(mmotion != nullptr && mmotion->maxAcceleration == 19.6133f && mmotion->maxAngularVelocity == 34.9066f &&
              mmotion->maxMagneticFieldStrength == 0.0f,
          "motion: motionInfo carries the shim's maxima (magnetometer unavailable)");
    check(pi->motionInfo == nullptr, "motion: the plain pad's motionInfo is null");

    // Before the first sensor sample arrives, SDL answers reads with zeros and the shim reports no motion at all.
    IGameInputReading* r = nullptr;
    GameInputMotionState unused{};
    check(input->GetCurrentReading(GameInputKindGamepad, mdev, &r) == S_OK && r && !r->GetMotionState(&unused),
          "motion: before the first sample, GetMotionState is false even on a sensor pad");
    if (r) r->Release();

    // 9.80665 m/s^2 is half of the 2 g maximum; 17.4533 rad/s is half of 2000 deg/s. A reading with the sample needs
    // either a gamepad change or the ~8 ms motion cadence, so the test waits out the window before polling.
    send_motion(gyro_pad, 0.0f, 0.0f, 9.80665f, 17.4533f, 0.0f, 0.0f);
    Sleep(12);
    rt::gameinput::poll();
    bool got = false;
    const GameInputMotionState ms = motion_state(input, mdev, &got);
    check(got, "motion: GetMotionState true on a motion reading");
    check(approx(ms.accelerationZ, 0.5f) && approx(ms.angularVelocityX, 0.5f),
          "motion: accel and gyro normalised against the motionInfo maxima");
    check(ms.accelerometerAccuracy == GameInputMotionAccuracyHigh && ms.gyroscopeAccuracy == GameInputMotionAccuracyHigh,
          "motion: live sensors report High accuracy");
    check(ms.magnetometerAccuracy == GameInputMotionAccuracyUnknown &&
              ms.orientationAccuracy == GameInputMotionAccuracyUnknown,
          "motion: sensors the shim does not provide report Unknown");

    r = nullptr;
    check(input->GetCurrentReading(GameInputKindMotion, pdev, &r) == GAMEINPUT_E_READING_NOT_FOUND && !r,
          "motion: a motion query on the plain pad is READING_NOT_FOUND");
    check(input->GetCurrentReading(GameInputKindGamepad, mdev, &r) == S_OK && r && !r->GetMotionState(nullptr) &&
              (r->GetInputKind() & GameInputKindMotion),
          "motion: GetMotionState(nullptr) is false; the gyro pad's reading kind includes Motion");
    if (r) r->Release();
    unused = {};
    check(input->GetCurrentReading(GameInputKindGamepad, pdev, &r) == S_OK && r &&
              !(r->GetInputKind() & GameInputKindMotion) && !r->GetMotionState(&unused),
          "motion: the plain pad's reading has no motion kind and GetMotionState is false");
    if (r) r->Release();

    send_motion(gyro_pad, 100.0f, -100.0f, 0.0f, 0.0f, 0.0f, -100.0f);
    Sleep(12);
    rt::gameinput::poll();
    const GameInputMotionState sat = motion_state(input, mdev);
    check(sat.accelerationX == 1.0f && sat.accelerationY == -1.0f && sat.angularVelocityZ == -1.0f,
          "motion: out-of-range sensor values clamp to the +-1 fraction range");

    // The cadence: with the buttons idle, a changed gyro yields a new reading only once per ~8 ms window, not one
    // per poll, so the history still spans the half second it is documented for.
    send_motion(gyro_pad, 0.0f, 0.0f, 9.80665f, 0.0f, 0.0f, 0.0f);
    rt::gameinput::poll();
    IGameInputReading* first = nullptr;
    check(input->GetCurrentReading(GameInputKindMotion, mdev, &first) == S_OK && first,
          "motion: a reading exists after a sensor update");
    send_motion(gyro_pad, 0.0f, 0.0f, 9.80665f, 34.9066f, 0.0f, 0.0f);  // gravity stays: zero accel means "no sample"
    rt::gameinput::poll();
    IGameInputReading* none = nullptr;
    check(input->GetNextReading(first, GameInputKindMotion, mdev, &none) == GAMEINPUT_E_READING_NOT_FOUND && !none,
          "motion: no new reading within the 8 ms cadence window");
    Sleep(12);
    rt::gameinput::poll();
    IGameInputReading* next = nullptr;
    check(input->GetNextReading(first, GameInputKindMotion, mdev, &next) == S_OK && next,
          "motion: after the window the latest sensor state is in a new reading");
    if (next) {
        GameInputMotionState later{};
        check(next->GetMotionState(&later) && approx(later.angularVelocityX, 1.0f),
              "motion: the new reading carries the newest gyro sample");
        next->Release();
    }
    if (first) first->Release();

    MotionReadings seen{mdev, pdev, 0, 0};
    GameInputCallbackToken motion_token = 0;
    check(input->RegisterReadingCallback(nullptr, GameInputKindMotion, 0.0f, &seen, on_motion_reading, &motion_token) == S_OK,
          "motion: RegisterReadingCallback(GameInputKindMotion)");
    for (int i = 0; i < 4; ++i) {
        SDL_SetJoystickVirtualButton(gyro_pad.js, SDL_GAMEPAD_BUTTON_SOUTH, i % 2 == 0);
        SDL_SetJoystickVirtualButton(plain.js, SDL_GAMEPAD_BUTTON_SOUTH, i % 2 == 0);
        rt::gameinput::poll();
    }
    input->UnregisterCallback(motion_token, 0);
    check(seen.gyro_calls > 0 && seen.plain_calls == 0,
          "motion: a Motion reading callback fires for the gyro pad's readings, not for the plain pad's");

    detach(gyro_pad);
    detach(plain);
    rt::gameinput::poll();
    if (mdev) mdev->Release();
    if (pdev) pdev->Release();
    check(rt::gameinput::connected_count() == 0, "motion: both pads closed again");
}

// WASD, the mouse and its buttons. Runs in the focused mode with focus and nothing held.
void test_keyboard_pad_sticks(IGameInput* input, IGameInputDevice* kb) {
    rt::gameinput::poll();
    check(sticks_at_rest(pad_state(input, kb)), "keyboard pad: sticks and triggers at rest");

    struct Dir { std::vector<int> keys; float x, y; const char* what; };
    const float d = 23170 / 32767.0f;
    const Dir dirs[] = {
        {{'W'}, 0, 1, "keyboard pad: W is left stick up"},
        {{'S'}, 0, -1, "keyboard pad: S is left stick down"},
        {{'A'}, -1, 0, "keyboard pad: A is left stick left"},
        {{'D'}, 1, 0, "keyboard pad: D is left stick right"},
        {{'W', 'S'}, 0, 0, "keyboard pad: W and S cancel"},
        {{'A', 'D'}, 0, 0, "keyboard pad: A and D cancel"},
        {{'W', 'S', 'D'}, 1, 0, "keyboard pad: W+S cancel, D still full right"},
        {{'W', 'D'}, d, d, "keyboard pad: W+D is up-right on the round gate"},
        {{'S', 'A'}, -d, -d, "keyboard pad: S+A is down-left on the round gate"},
    };
    for (const Dir& e : dirs) {
        for (int k : e.keys) g_keys[k] = true;
        rt::gameinput::poll();
        const GameInputGamepadState st = pad_state(input, kb);
        check(approx(st.leftThumbstickX, e.x) && approx(st.leftThumbstickY, e.y) && st.buttons == 0 &&
                  st.rightThumbstickX == 0 && st.rightThumbstickY == 0,
              e.what);
        for (int k : e.keys) g_keys[k] = false;
    }
    rt::gameinput::poll();
    check(sticks_at_rest(pad_state(input, kb)), "keyboard pad: left stick centers on release");

    // 10 px right, 10 px down in one poll: 10 x 1024 axis units each, +Y down on screen is stick down.
    g_cursor_x += 10;
    g_cursor_y += 10;
    rt::gameinput::poll();
    GameInputGamepadState st = pad_state(input, kb);
    check(approx(st.rightThumbstickX, 10240 / 32767.0f) && approx(st.rightThumbstickY, -10240 / 32767.0f) &&
              st.leftThumbstickX == 0,
          "keyboard pad: mouse delta moves the right stick");
    g_cursor_x -= 500;
    rt::gameinput::poll();
    check(approx(pad_state(input, kb).rightThumbstickX, -1), "keyboard pad: a large mouse delta clamps to full");
    // The mouse stops: the stick falls back, never past center, to exactly 0.
    int polls = 0;
    float last = -1;
    bool monotonic = true;
    for (; polls < 200; ++polls) {
        rt::gameinput::poll();
        st = pad_state(input, kb);
        monotonic &= st.rightThumbstickX >= last && st.rightThumbstickX <= 0;
        last = st.rightThumbstickX;
        if (st.rightThumbstickX == 0 && st.rightThumbstickY == 0) break;
    }
    check(monotonic && polls < 100 && sticks_at_rest(st), "keyboard pad: right stick decays to 0 when the mouse stops");

    // LEONIDA_MOUSE_SENSITIVITY scales the axis units per count: 2 doubles, 0.5 halves; out of range clamps.
    auto settle = [&] {
        for (int i = 0; i < 200 && !sticks_at_rest(pad_state(input, kb)); ++i) rt::gameinput::poll();
    };
    set_env("LEONIDA_MOUSE_SENSITIVITY", "2");
    rt::gameinput::reload_config();
    g_cursor_x += 10;
    rt::gameinput::poll();
    check(approx(pad_state(input, kb).rightThumbstickX, 20480 / 32767.0f), "keyboard pad: mouse sensitivity 2 doubles the stick");
    settle();
    set_env("LEONIDA_MOUSE_SENSITIVITY", "0.5");
    rt::gameinput::reload_config();
    g_cursor_x -= 10;
    rt::gameinput::poll();
    check(approx(pad_state(input, kb).rightThumbstickX, -5120 / 32767.0f), "keyboard pad: mouse sensitivity 0.5 halves the stick");
    settle();
    set_env("LEONIDA_MOUSE_SENSITIVITY", "100");
    rt::gameinput::reload_config();
    check(rt::gameinput::config().mouse_sensitivity == 10.0f, "keyboard pad: mouse sensitivity clamps to 10");
    set_env("LEONIDA_MOUSE_SENSITIVITY", "0");
    rt::gameinput::reload_config();
    check(approx(rt::gameinput::config().mouse_sensitivity, 0.1f), "keyboard pad: mouse sensitivity clamps to 0.1");
    set_env("LEONIDA_MOUSE_SENSITIVITY", nullptr);
    rt::gameinput::reload_config();
    check(rt::gameinput::config().mouse_sensitivity == 1.0f && rt::gameinput::config().ds_trigger_gain == 1.0f,
          "keyboard pad: mouse sensitivity defaults to 1 and leaves the trigger gain alone");

    g_keys[VK_RBUTTON] = true;
    rt::gameinput::poll();
    st = pad_state(input, kb);
    check(approx(st.leftTrigger, 1) && st.rightTrigger == 0 && st.buttons == 0,
          "keyboard pad: right mouse button is the left trigger");
    g_keys[VK_LBUTTON] = true;
    rt::gameinput::poll();
    st = pad_state(input, kb);
    check(approx(st.leftTrigger, 1) && approx(st.rightTrigger, 1), "keyboard pad: left mouse button is the right trigger");
    g_keys[VK_RBUTTON] = false;
    rt::gameinput::poll();
    st = pad_state(input, kb);
    check(st.leftTrigger == 0 && approx(st.rightTrigger, 1), "keyboard pad: releasing a mouse button releases its trigger");

    // Losing focus releases sticks and triggers; the cursor moved meanwhile does not swing the stick on return.
    g_keys['W'] = g_keys['D'] = g_keys[VK_LBUTTON] = true;
    g_cursor_x += 20;
    rt::gameinput::poll();
    check(!sticks_at_rest(pad_state(input, kb)), "keyboard pad: sticks and trigger held before focus loss");
    g_focus = false;
    g_cursor_x += 300;
    rt::gameinput::poll();
    check(sticks_at_rest(pad_state(input, kb)), "keyboard pad: losing focus releases sticks and triggers");
    g_keys['W'] = g_keys['D'] = g_keys[VK_LBUTTON] = false;
    g_cursor_x += 300;
    g_focus = true;
    rt::gameinput::poll();
    check(sticks_at_rest(pad_state(input, kb)), "keyboard pad: no mouse jump when focus returns");
}

// Runs with no other pad connected and the device callback from main still registered.
void test_keyboard_pad(IGameInput* input) {
    rt::gameinput::set_keyboard_source(fake_key_down, fake_focus);
    rt::gameinput::set_cursor_source(fake_cursor);

    set_env("LEONIDA_KEYBOARD_PAD", nullptr);
    rt::gameinput::reload_config();
    rt::gameinput::poll();
    check(rt::gameinput::connected_count() == 0, "keyboard pad: off by default");

    set_env("LEONIDA_KEYBOARD_PAD", "1");
    rt::gameinput::reload_config();
    check(rt::gameinput::config().keyboard_pad == rt::gameinput::KeyboardPadMode::Focused,
          "keyboard pad: LEONIDA_KEYBOARD_PAD=1 is the focused mode");
    const size_t events_before = g_device_events.size();
    rt::gameinput::poll();
    check(rt::gameinput::connected_count() == 1, "keyboard pad: attached as one gamepad");
    // Main leaves three device callbacks registered (blocking, async and blocking enumeration), so one connect is
    // reported to each of them: all with the same device.
    IGameInputDevice* kb = g_device_events.size() > events_before ? g_device_events.back().device : nullptr;
    bool connect_events = g_device_events.size() == events_before + 3;
    for (size_t i = events_before; i < g_device_events.size(); ++i)
        connect_events &= g_device_events[i].device == kb && (g_device_events[i].current & GameInputDeviceConnected) &&
                          g_device_events[i].previous == GameInputDeviceNoStatus;
    check(connect_events, "keyboard pad: device callbacks on connect");
    if (!kb) return;
    const GameInputDeviceInfo* info = kb->GetDeviceInfo();
    check(info->displayName && std::strcmp(info->displayName->data, "Keyboard pad (Leonida)") == 0,
          "keyboard pad: display name");
    check((info->supportedInput & GameInputKindGamepad) && info->supportedRumbleMotors == 0,
          "keyboard pad: a gamepad without motors");
    check(pad_state(input, kb).buttons == 0, "keyboard pad: at rest");

    // Out of focus, keys are ignored.
    g_focus = false;
    g_keys[VK_RETURN] = true;
    rt::gameinput::poll();
    check(pad_state(input, kb).buttons == 0, "keyboard pad: no buttons while another window has focus");

    // In focus, every binding presses its button.
    g_focus = true;
    g_keys[VK_RETURN] = false;
    struct Expect { int vk; uint32_t button; const char* what; };
    const Expect expect[] = {
        {VK_UP, GameInputGamepadDPadUp, "keyboard pad: Up is d-pad up"},
        {VK_DOWN, GameInputGamepadDPadDown, "keyboard pad: Down is d-pad down"},
        {VK_LEFT, GameInputGamepadDPadLeft, "keyboard pad: Left is d-pad left"},
        {VK_RIGHT, GameInputGamepadDPadRight, "keyboard pad: Right is d-pad right"},
        {VK_RETURN, GameInputGamepadA, "keyboard pad: Enter is A"},
        {VK_BACK, GameInputGamepadB, "keyboard pad: Backspace is B"},
        {'X', GameInputGamepadX, "keyboard pad: X is X"},
        {'Y', GameInputGamepadY, "keyboard pad: Y is Y"},
        {'Q', GameInputGamepadLeftShoulder, "keyboard pad: Q is the left shoulder"},
        {'E', GameInputGamepadRightShoulder, "keyboard pad: E is the right shoulder"},
        {VK_ESCAPE, GameInputGamepadMenu, "keyboard pad: Esc is Menu"},
        {VK_TAB, GameInputGamepadView, "keyboard pad: Tab is View"},
    };
    for (const Expect& e : expect) {
        g_keys[e.vk] = true;
        rt::gameinput::poll();
        check(pad_state(input, kb).buttons == e.button, e.what);
        g_keys[e.vk] = false;
    }
    int bindings = 0;
    rt::gameinput::keyboard_bindings(&bindings);
    check(bindings == static_cast<int>(std::size(expect)), "keyboard pad: every binding is tested");

    // Two keys at once, and UI navigation from the d-pad and A.
    g_keys[VK_DOWN] = g_keys[VK_RETURN] = true;
    rt::gameinput::poll();
    uint32_t ui = 0;
    const GameInputGamepadState both = pad_state(input, kb, &ui);
    check(both.buttons == (GameInputGamepadDPadDown | GameInputGamepadA), "keyboard pad: Down and Enter together");
    check((ui & GameInputUiNavigationDown) && (ui & GameInputUiNavigationAccept), "keyboard pad: UI navigation down and accept");
    check(approx(both.leftTrigger, 0) && approx(both.rightTrigger, 0) && approx(both.leftThumbstickX, 0) &&
              approx(both.rightThumbstickY, 0),
          "keyboard pad: sticks and triggers stay at rest");

    // Losing focus releases held keys.
    g_focus = false;
    rt::gameinput::poll();
    check(pad_state(input, kb).buttons == 0, "keyboard pad: losing focus releases every button");

    // The always mode ignores focus.
    set_env("LEONIDA_KEYBOARD_PAD", "always");
    rt::gameinput::reload_config();
    rt::gameinput::poll();
    check(pad_state(input, kb).buttons == (GameInputGamepadDPadDown | GameInputGamepadA),
          "keyboard pad: LEONIDA_KEYBOARD_PAD=always reads keys without focus");
    g_keys[VK_DOWN] = g_keys[VK_RETURN] = false;

    set_env("LEONIDA_KEYBOARD_PAD", "1");
    rt::gameinput::reload_config();
    g_focus = true;
    test_keyboard_pad_sticks(input, kb);

    // Turning it off removes the pad; turning it on again brings back the same device object.
    set_env("LEONIDA_KEYBOARD_PAD", "off");
    rt::gameinput::reload_config();
    rt::gameinput::poll();
    check(rt::gameinput::connected_count() == 0 && kb->GetDeviceStatus() == GameInputDeviceNoStatus,
          "keyboard pad: off removes it");
    set_env("LEONIDA_KEYBOARD_PAD", "1");
    rt::gameinput::reload_config();
    rt::gameinput::poll();
    check(rt::gameinput::connected_count() == 1 && g_device_events.back().device == kb &&
              (kb->GetDeviceStatus() & GameInputDeviceConnected),
          "keyboard pad: on again is the same device object");

    set_env("LEONIDA_KEYBOARD_PAD", nullptr);
    rt::gameinput::reload_config();
    rt::gameinput::poll();
    check(rt::gameinput::connected_count() == 0, "keyboard pad: removed at the end");
    rt::gameinput::set_keyboard_source(nullptr, nullptr);
    rt::gameinput::set_cursor_source(nullptr);
}

// active_input(): which kind of device the player used last. Virtual pads with a DualSense, a DualShock 4 and an Xbox
// pad's vendor and product id (SDL takes the gamepad type from them) plus the keyboard pad; first with no wait
// between changes (LEONIDA_PROMPTS_SWITCH_MS=0), then with the default 300 ms window on the real clock, then a family
// forced by LEONIDA_PROMPTS.
void test_active_input() {
    using leonida::InputFamily;
    const auto family = [] { return rt::gameinput::active_input().family; };
    const auto generation = [] { return rt::gameinput::active_input().generation; };

    set_env("LEONIDA_PROMPTS", nullptr);
    set_env("LEONIDA_PROMPTS_SWITCH_MS", "0");
    rt::gameinput::reload_config();
    rt::gameinput::poll();
    check(!rt::gameinput::active_input().forced, "active input: not forced without LEONIDA_PROMPTS");

    Pad ds, ds4, xbox;
    attach(ds, "Leonida virtual DualSense", 0x054C, 0x0CE6, false);
    attach(ds4, "Leonida virtual DualShock 4", 0x054C, 0x09CC, false);
    attach(xbox, "Leonida virtual Xbox pad", 0x045E, 0x0B12, true);
    SDL_SetJoystickVirtualButton(ds.js, SDL_GAMEPAD_BUTTON_SOUTH, true);  // held while it connects
    const uint64_t at_connect = generation();
    rt::gameinput::poll();
    check(rt::gameinput::connected_count() == 3, "active input: three pads open");
    check(generation() == at_connect, "active input: connecting pads (one with a button held) changes nothing");

    auto press = [](Pad& p, SDL_GamepadButton b) {
        SDL_SetJoystickVirtualButton(p.js, b, true);
        rt::gameinput::poll();
        SDL_SetJoystickVirtualButton(p.js, b, false);
        rt::gameinput::poll();
    };
    SDL_SetJoystickVirtualButton(ds.js, SDL_GAMEPAD_BUTTON_SOUTH, false);  // the held button, let go
    rt::gameinput::poll();
    check(generation() == at_connect, "active input: letting go of a button is not input");

    // The tests before this one leave whichever pad they used last; an Xbox press makes the start known.
    press(xbox, SDL_GAMEPAD_BUTTON_SOUTH);
    check(family() == InputFamily::Xbox, "active input: the Xbox pad in use");
    const uint64_t g0 = generation();

    press(ds, SDL_GAMEPAD_BUTTON_SOUTH);
    check(family() == InputFamily::PlayStation5 && generation() == g0 + 1, "active input: DualSense press");
    press(ds4, SDL_GAMEPAD_BUTTON_EAST);
    check(family() == InputFamily::PlayStation4 && generation() == g0 + 2, "active input: DualShock 4 press");

    // Stick drift on the Xbox pad (0.18) is not input; the stick pushed over is.
    SDL_SetJoystickVirtualAxis(xbox.js, SDL_GAMEPAD_AXIS_LEFTX, 6000);
    rt::gameinput::poll();
    check(family() == InputFamily::PlayStation4 && generation() == g0 + 2, "active input: stick drift changes nothing");
    SDL_SetJoystickVirtualAxis(xbox.js, SDL_GAMEPAD_AXIS_LEFTX, 32767);
    rt::gameinput::poll();
    check(family() == InputFamily::Xbox && generation() == g0 + 3, "active input: Xbox stick pushed over");
    SDL_SetJoystickVirtualAxis(xbox.js, SDL_GAMEPAD_AXIS_LEFTX, 0);
    rt::gameinput::poll();

    // Buttons GameInput's gamepad state leaves out count too: the DualSense touchpad.
    press(ds, SDL_GAMEPAD_BUTTON_TOUCHPAD);
    check(family() == InputFamily::PlayStation5 && generation() == g0 + 4, "active input: DualSense touchpad click");
    // A trigger pulled on the DualShock 4.
    SDL_SetJoystickVirtualAxis(ds4.js, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER, 32767);
    rt::gameinput::poll();
    check(family() == InputFamily::PlayStation4 && generation() == g0 + 5, "active input: DualShock 4 trigger pull");
    SDL_SetJoystickVirtualAxis(ds4.js, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER, -32768);
    rt::gameinput::poll();

    // The keyboard pad: a key, and the mouse through the keyboard pad's right stick.
    rt::gameinput::set_keyboard_source(fake_key_down, fake_focus);
    rt::gameinput::set_cursor_source(fake_cursor);
    std::memset(g_keys, 0, sizeof g_keys);
    g_cursor_x = g_cursor_y = 0;
    set_env("LEONIDA_KEYBOARD_PAD", "always");
    rt::gameinput::reload_config();
    rt::gameinput::poll();  // attaches the keyboard pad
    rt::gameinput::poll();  // opens it: its first sample is the baseline
    check(rt::gameinput::connected_count() == 4 && generation() == g0 + 5,
          "active input: the keyboard pad connecting changes nothing");
    g_keys[VK_RETURN] = true;
    rt::gameinput::poll();
    g_keys[VK_RETURN] = false;
    rt::gameinput::poll();
    check(family() == InputFamily::Keyboard && generation() == g0 + 6, "active input: Enter on the keyboard pad");
    press(xbox, SDL_GAMEPAD_BUTTON_SOUTH);
    check(family() == InputFamily::Xbox && generation() == g0 + 7, "active input: back to the Xbox pad");
    g_cursor_x += 1;  // a 1-count jitter stays well inside the stick radius
    rt::gameinput::poll();
    check(family() == InputFamily::Xbox, "active input: mouse jitter changes nothing");
    g_cursor_x += 20;  // a real movement pushes the right stick past the radius
    rt::gameinput::poll();
    check(family() == InputFamily::Keyboard && generation() == g0 + 8, "active input: mouse movement");
    for (int i = 0; i < 80; ++i) rt::gameinput::poll();  // the stick falls back to rest

    // The default window on the real clock: a second change within 300 ms of the last is ignored.
    set_env("LEONIDA_PROMPTS_SWITCH_MS", nullptr);
    rt::gameinput::reload_config();
    check(rt::gameinput::config().prompts_switch_ms == 300, "active input: 300 ms by default");
    Sleep(350);
    press(ds, SDL_GAMEPAD_BUTTON_NORTH);
    check(family() == InputFamily::PlayStation5 && generation() == g0 + 9, "active input: DualSense after the window");
    press(xbox, SDL_GAMEPAD_BUTTON_WEST);
    check(family() == InputFamily::PlayStation5 && generation() == g0 + 9,
          "active input: an Xbox press right after a change is ignored");
    Sleep(350);
    press(xbox, SDL_GAMEPAD_BUTTON_WEST);
    check(family() == InputFamily::Xbox && generation() == g0 + 10,
          "active input: the Xbox press once the window passed");

    // LEONIDA_PROMPTS forces a family; tracking goes on underneath and takes over again when it is cleared.
    set_env("LEONIDA_PROMPTS_SWITCH_MS", "0");
    set_env("LEONIDA_PROMPTS", "keyboard");
    rt::gameinput::reload_config();
    rt::gameinput::poll();
    check(family() == InputFamily::Keyboard && rt::gameinput::active_input().forced && generation() == g0 + 11,
          "active input: LEONIDA_PROMPTS=keyboard");
    press(ds, SDL_GAMEPAD_BUTTON_SOUTH);
    check(family() == InputFamily::Keyboard && generation() == g0 + 11,
          "active input: forced, a DualSense press shows nothing");
    set_env("LEONIDA_PROMPTS", "PlayStation");  // case does not matter
    rt::gameinput::reload_config();
    rt::gameinput::poll();
    check(family() == InputFamily::PlayStation5 && rt::gameinput::active_input().forced && generation() == g0 + 12,
          "active input: LEONIDA_PROMPTS=PlayStation (any case) is PlayStation 5");
    set_env("LEONIDA_PROMPTS", "ps4");
    rt::gameinput::reload_config();
    rt::gameinput::poll();
    check(family() == InputFamily::PlayStation4 && generation() == g0 + 13, "active input: LEONIDA_PROMPTS=ps4");
    set_env("LEONIDA_PROMPTS", "nonsense");  // warns and follows the devices
    rt::gameinput::reload_config();
    rt::gameinput::poll();
    check(!rt::gameinput::active_input().forced && family() == InputFamily::PlayStation5 && generation() == g0 + 14,
          "active input: an unknown LEONIDA_PROMPTS follows the devices (the DualSense pressed while forced)");
    set_env("LEONIDA_PROMPTS", "playstation");
    rt::gameinput::reload_config();
    rt::gameinput::poll();
    check(family() == InputFamily::PlayStation5 && rt::gameinput::active_input().forced && generation() == g0 + 14,
          "active input: forcing the family already in use changes no generation");

    // Clean up: everything off and unplugged.
    set_env("LEONIDA_PROMPTS", nullptr);
    set_env("LEONIDA_PROMPTS_SWITCH_MS", nullptr);
    set_env("LEONIDA_KEYBOARD_PAD", nullptr);
    rt::gameinput::reload_config();
    detach(ds);
    detach(ds4);
    detach(xbox);
    rt::gameinput::poll();
    check(rt::gameinput::connected_count() == 0, "active input: all pads closed at the end");
    rt::gameinput::set_keyboard_source(nullptr, nullptr);
    rt::gameinput::set_cursor_source(nullptr);
}

bool same_id(const AppLocalDeviceId& a, const AppLocalDeviceId& b) { return std::memcmp(&a, &b, sizeof a) == 0; }

// The id the shim derives from SDL: the GUID, then two FNV-1a hashes of "s:" serial, "p:" path or nothing, with
// "t:<n>:" before it for twin n. Twin 0 is the id every pad had before twins were told apart.
AppLocalDeviceId expected_id(const Pad& pad, int twin) {
    const char* serial = SDL_GetJoystickSerial(pad.js);
    const char* path = SDL_GetJoystickPath(pad.js);
    std::string identity = (serial && *serial) ? std::string("s:") + serial : (path && *path) ? std::string("p:") + path : "";
    if (twin) identity = "t:" + std::to_string(twin) + ":" + identity;
    AppLocalDeviceId id{};
    const SDL_GUID guid = SDL_GetJoystickGUIDForID(pad.id);
    std::memcpy(id.value, guid.data, 16);
    for (int half = 0; half < 2; ++half) {
        uint64_t h = half ? 0x84222325cbf29ce4ull : 0xcbf29ce484222325ull;
        for (unsigned char c : identity) h = (h ^ c) * 0x100000001b3ull;
        std::memcpy(id.value + 16 + half * 8, &h, 8);
    }
    return id;
}

// Attaches one more identical twin and returns the device the shim reported for it.
IGameInputDevice* plug(Pad& pad) {
    const size_t before = g_device_events.size();
    attach(pad, "Leonida virtual twin pad", 0x0F0D, 0x00C1, false);
    rt::gameinput::poll();
    return g_device_events.size() > before ? g_device_events.back().device : nullptr;
}

bool found_is(IGameInput* input, IGameInputDevice* dev) {
    if (!dev) return false;
    IGameInputDevice* found = nullptr;
    const bool ok = input->FindDeviceFromId(&dev->GetDeviceInfo()->deviceId, &found) == S_OK && found == dev;
    if (found) found->Release();
    return ok;
}

// Pads SDL cannot tell apart: the same name, vendor and product, no serial, no path. Runs with no other pad connected.
void test_aggregate_device(IGameInput* input) {
    check(input->CreateAggregateDevice(GameInputKindGamepad, nullptr) == E_POINTER,
          "CreateAggregateDevice rejects null output pointer");
    IGameInputDevice* agg = nullptr;
    check(input->CreateAggregateDevice(GameInputKindUnknown, &agg) == E_INVALIDARG && !agg,
          "CreateAggregateDevice rejects GameInputKindUnknown");
    check(input->CreateAggregateDevice(GameInputKindGamepad | GameInputKindKeyboard, &agg) == E_INVALIDARG && !agg,
          "CreateAggregateDevice rejects multiple combined kind flags");
    check(input->CreateAggregateDevice(0x80000000, &agg) == E_INVALIDARG && !agg,
          "CreateAggregateDevice rejects unsupported kind flag");

    check(input->CreateAggregateDevice(GameInputKindGamepad, &agg) == S_OK && agg != nullptr,
          "CreateAggregateDevice creates an aggregate device for GameInputKindGamepad");
    if (agg) {
        const GameInputDeviceInfo* info = agg->GetDeviceInfo();
        check(info && info->deviceFamily == GameInputFamilyAggregate,
              "aggregate device reports GameInputFamilyAggregate");
        check(info && info->supportedInput == GameInputKindGamepad,
              "aggregate device reports supportedInput == GameInputKindGamepad");
        check(agg->GetDeviceStatus() == GameInputDeviceConnected,
              "aggregate device status is GameInputDeviceConnected");

        IGameInputDevice* agg2 = nullptr;
        check(input->CreateAggregateDevice(GameInputKindGamepad, &agg2) == S_OK && agg2 == agg,
              "CreateAggregateDevice returns existing aggregate instance for the same kind");
        if (agg2) agg2->Release();

        // Reading test with no connected pad
        IGameInputReading* r = nullptr;
        check(input->GetCurrentReading(GameInputKindGamepad, agg, &r) == GAMEINPUT_E_READING_NOT_FOUND && !r,
              "aggregate device returns READING_NOT_FOUND when no pads are connected");
        check(input->GetCurrentReading(GameInputKindKeyboard, agg, &r) == GAMEINPUT_E_READING_NOT_FOUND && !r,
              "aggregate device returns READING_NOT_FOUND for kind mismatch with no pads");

        // Callback registration guards for aggregate device
        GameInputCallbackToken reading_tok = 0xbeef;
        check(input->RegisterReadingCallback(agg, GameInputKindGamepad, 0.0f, nullptr, on_reading, &reading_tok) == E_NOTIMPL &&
                  reading_tok == kInvalidCallbackToken,
              "RegisterReadingCallback rejects aggregate device with E_NOTIMPL and invalid token");

        GameInputCallbackToken device_tok = 0xbeef;
        check(input->RegisterDeviceCallback(agg, GameInputKindGamepad, GameInputDeviceConnected,
                                            GameInputBlockingEnumeration, nullptr, on_device, &device_tok) == E_NOTIMPL &&
                  device_tok == kInvalidCallbackToken,
              "RegisterDeviceCallback rejects aggregate device with E_NOTIMPL and invalid token");

        // Attach a virtual pad and verify aggregate device yields readings from it
        Pad xbox;
        attach(xbox, "Leonida virtual Xbox pad", 0x045E, 0x0B12, true);
        rt::gameinput::poll();

        SDL_SetJoystickVirtualButton(xbox.js, SDL_GAMEPAD_BUTTON_SOUTH, true);
        rt::gameinput::poll();

        check(input->GetCurrentReading(GameInputKindGamepad, agg, &r) == S_OK && r != nullptr,
              "aggregate device yields reading from connected pad");
        if (r) {
            check(buttons_of(r) == GameInputGamepadA, "aggregate device reading reflects pad input state");
            r->Release();
        }

        IGameInputReading* r_mismatch = nullptr;
        check(input->GetCurrentReading(GameInputKindKeyboard, agg, &r_mismatch) == GAMEINPUT_E_READING_NOT_FOUND && !r_mismatch,
              "aggregate device rejects kind mismatch even when pad is connected");

        SDL_SetJoystickVirtualButton(xbox.js, SDL_GAMEPAD_BUTTON_SOUTH, false);
        rt::gameinput::poll();
        IGameInputReading* r2 = nullptr;
        check(input->GetCurrentReading(GameInputKindGamepad, agg, &r2) == S_OK && r2 != nullptr,
              "aggregate device yields updated reading after button release");
        if (r2) {
            check(buttons_of(r2) == 0, "aggregate reading shows button released");
            IGameInputReading* prev = nullptr;
            check(input->GetPreviousReading(r2, GameInputKindGamepad, agg, &prev) == S_OK && prev != nullptr,
                  "GetPreviousReading with aggregate device filter finds prior reading");
            if (prev) {
                check(buttons_of(prev) == GameInputGamepadA, "prior reading has pressed button");
                prev->Release();
            }
            r2->Release();
        }

        detach(xbox);
        rt::gameinput::poll();
        agg->Release();
    }
}

void test_twins(IGameInput* input) {
    Pad pads[3];
    IGameInputDevice* dev[3] = {};
    AppLocalDeviceId id[3] = {};
    for (int i = 0; i < 2; ++i) dev[i] = plug(pads[i]);
    check(dev[0] && dev[1] && dev[0] != dev[1], "twins: two identical pads are two device objects");
    if (!dev[0] || !dev[1]) {
        for (Pad& p : pads) detach(p);
        rt::gameinput::poll();
        return;
    }
    const AppLocalDeviceId legacy = expected_id(pads[0], 0);
    check(same_id(legacy, expected_id(pads[1], 0)), "twins: SDL gives them nothing to tell them apart");
    for (int i = 0; i < 2; ++i) id[i] = dev[i]->GetDeviceInfo()->deviceId;
    check(same_id(id[0], legacy), "twins: the first keeps the id a single pad has always had");
    check(!same_id(id[0], id[1]) && same_id(id[1], expected_id(pads[1], 1)),
          "twins: the second gets twin id 1, same GUID bytes, different from the first");
    check(found_is(input, dev[0]) && found_is(input, dev[1]), "twins: FindDeviceFromId returns each twin for its own id");

    SDL_SetJoystickVirtualButton(pads[0].js, SDL_GAMEPAD_BUTTON_SOUTH, true);
    SDL_SetJoystickVirtualButton(pads[1].js, SDL_GAMEPAD_BUTTON_EAST, true);
    rt::gameinput::poll();
    check(pad_state(input, dev[0]).buttons == GameInputGamepadA && pad_state(input, dev[1]).buttons == GameInputGamepadB,
          "twins: a reading filtered to each twin has only its own button");
    IGameInputDevice* found = nullptr;
    input->FindDeviceFromId(&id[1], &found);
    check(found && pad_state(input, found).buttons == GameInputGamepadB,
          "twins: the device found from the second id reads the second pad");
    if (found) found->Release();

    dev[2] = plug(pads[2]);
    if (dev[2]) id[2] = dev[2]->GetDeviceInfo()->deviceId;
    check(dev[2] && dev[2] != dev[0] && dev[2] != dev[1] && same_id(id[2], expected_id(pads[2], 2)) &&
              !same_id(id[2], id[0]) && !same_id(id[2], id[1]) && found_is(input, dev[2]),
          "twins: a third gets a third object and twin id 2");

    // One twin unplugged: it gets its object and id back.
    detach(pads[1]);
    rt::gameinput::poll();
    check(dev[1]->GetDeviceStatus() == GameInputDeviceNoStatus && found_is(input, dev[1]),
          "twins: an unplugged twin is still found from its id");
    check(plug(pads[1]) == dev[1] && same_id(dev[1]->GetDeviceInfo()->deviceId, id[1]),
          "twins: the middle twin plugged back in gets its object and id back");
    detach(pads[0]);
    rt::gameinput::poll();
    check(plug(pads[0]) == dev[0] && same_id(dev[0]->GetDeviceInfo()->deviceId, legacy),
          "twins: so does the first, with the legacy id");

    // Two unplugged: nothing tells them apart, so the first back takes the lower twin whichever pad it is.
    detach(pads[1]);
    detach(pads[2]);
    rt::gameinput::poll();
    IGameInputDevice* first_back = plug(pads[2]);
    IGameInputDevice* second_back = plug(pads[1]);
    check(first_back == dev[1] && second_back == dev[2], "twins: with two unplugged, objects follow arrival order");

    // All unplugged and plugged in again: the same objects and ids in creation order, no new object.
    for (Pad& p : pads) detach(p);
    rt::gameinput::poll();
    bool same = true;
    for (int i = 0; i < 3; ++i) {
        IGameInputDevice* d = plug(pads[i]);
        same &= d && d == dev[i] && same_id(d->GetDeviceInfo()->deviceId, id[i]);
    }
    check(same && rt::gameinput::connected_count() == 3, "twins: all three replugged get the same objects and ids in order");

    for (Pad& p : pads) detach(p);
    rt::gameinput::poll();
    check(rt::gameinput::connected_count() == 0, "twins: removed at the end");
}

// ---- force feedback and haptic motors: a fake SDL haptic backend ----
// SDL's virtual joysticks report no haptics, so the shim's haptic calls go through this table instead. It records what
// the shim asked SDL to do and reports the capabilities a force-feedback wheel has.
struct FakeHaptic {
    uint32_t features = SDL_HAPTIC_CONSTANT | SDL_HAPTIC_SINE | SDL_HAPTIC_SQUARE | SDL_HAPTIC_TRIANGLE |
                        SDL_HAPTIC_SAWTOOTHUP | SDL_HAPTIC_SAWTOOTHDOWN | SDL_HAPTIC_RAMP | SDL_HAPTIC_SPRING |
                        SDL_HAPTIC_FRICTION | SDL_HAPTIC_DAMPER | SDL_HAPTIC_INERTIA;
    int max_effects = 8;
    int create_result = 1;  // what create returns; negative is SDL refusing the effect
    int opens = 0, closes = 0, creates = 0, updates = 0, runs = 0, stops = 0, destroys = 0;
    int gain_calls = 0, last_gain = -1;
    uint32_t last_iterations = 0;
    SDL_HapticEffect last{};
};
FakeHaptic g_fake;
bool g_fake_open = false;  // pads attached while this is set are opened as haptic devices

void* fake_open(void*) {
    if (!g_fake_open) return nullptr;
    ++g_fake.opens;
    return &g_fake;
}
void fake_close(void*) { ++g_fake.closes; }
uint32_t fake_features(void* h) { return static_cast<FakeHaptic*>(h)->features; }
int fake_max_effects(void* h) { return static_cast<FakeHaptic*>(h)->max_effects; }
int fake_create(void* h, const void* e) {
    FakeHaptic* f = static_cast<FakeHaptic*>(h);
    ++f->creates;
    if (f->create_result < 0) {
        SDL_SetError("fake: refused");
        return -1;
    }
    f->last = *static_cast<const SDL_HapticEffect*>(e);
    return f->create_result;
}
bool fake_update(void* h, int, const void* e) {
    FakeHaptic* f = static_cast<FakeHaptic*>(h);
    ++f->updates;
    f->last = *static_cast<const SDL_HapticEffect*>(e);
    return true;
}
bool fake_run(void* h, int, uint32_t iterations) {
    FakeHaptic* f = static_cast<FakeHaptic*>(h);
    ++f->runs;
    f->last_iterations = iterations;
    return true;
}
bool fake_stop(void* h, int) {
    ++static_cast<FakeHaptic*>(h)->stops;
    return true;
}
void fake_destroy(void* h, int) { ++static_cast<FakeHaptic*>(h)->destroys; }
bool fake_gain(void* h, int gain) {
    FakeHaptic* f = static_cast<FakeHaptic*>(h);
    ++f->gain_calls;
    f->last_gain = gain;
    return true;
}
const rt::gameinput::HapticApi kFakeHaptic{fake_open,    fake_close, fake_features, fake_max_effects, fake_create,
                                           fake_update,  fake_run,   fake_stop,     fake_destroy,    fake_gain};

// The motors of one device, and the effect object its force-feedback motor hands out. Runs with no other pad.
void test_force_feedback(IGameInput* input) {
    rt::gameinput::set_haptic_source(&kFakeHaptic);
    g_fake = {};
    g_fake_open = true;
    Pad wheel;
    attach(wheel, "Leonida virtual wheel", 0x046D, 0xC24F, false);
    const size_t before = g_device_events.size();
    rt::gameinput::poll();
    IGameInputDevice* dev = g_device_events.size() > before ? g_device_events.back().device : nullptr;
    check(dev && rt::gameinput::connected_count() == 1 && g_fake.opens == 1,
          "force feedback: the pad's joystick is opened as an SDL haptic device");
    if (!dev) return;
    dev->AddRef();
    rt::gameinput::OutputState out{};
    check(rt::gameinput::output_state(0, &out) && out.force_feedback, "force feedback: the output state says so");
    IGameInputReading* r = current(input, dev);
    check(r != nullptr, "force feedback: the pad still reads normally with a haptic device open");
    if (r) r->Release();

    const GameInputDeviceInfo* info = dev->GetDeviceInfo();
    check(info->forceFeedbackMotorCount == 1 && info->forceFeedbackMotorInfo != nullptr,
          "force feedback: the device reports one force-feedback motor");
    const GameInputForceFeedbackMotorInfo* motor =
        static_cast<const GameInputForceFeedbackMotorInfo*>(info->forceFeedbackMotorInfo);
    check(motor->supportedAxes == GameInputFeedbackAxisLinearX && motor->location == GameInputLocationChassis &&
              motor->maxSimultaneousEffects == 8,
          "force feedback: motorInfo carries the axis, the location and the effect limit");
    check(motor->isConstantEffectSupported && motor->isRampEffectSupported && motor->isSpringEffectSupported &&
              motor->isSawtoothDownWaveEffectSupported,
          "force feedback: the effect kinds come from SDL's feature bits");
    check(info->hapticFeedbackMotorCount == 0, "force feedback: a wheel without trigger motors reports no haptics");
    check(dev->IsForceFeedbackMotorPoweredOn(0) && !dev->IsForceFeedbackMotorPoweredOn(1),
          "force feedback: motor 0 is powered on, and motor 1 does not exist");

    IGameInputForceFeedbackEffect* effect = nullptr;
    GameInputForceFeedbackParams params{};
    params.kind = GameInputForceFeedbackConstant;
    params.data.constant.magnitude.linearX = 0.5f;
    check(dev->CreateForceFeedbackEffect(1, &params, reinterpret_cast<void**>(&effect)) == E_INVALIDARG && !effect,
          "force feedback: motor 1 does not exist");
    check(dev->CreateForceFeedbackEffect(0, &params, nullptr) == E_POINTER, "force feedback: a null effect is E_POINTER");
    check(dev->CreateForceFeedbackEffect(0, nullptr, reinterpret_cast<void**>(&effect)) == E_INVALIDARG,
          "force feedback: null parameters are E_INVALIDARG");
    GameInputForceFeedbackParams unknown{};
    unknown.kind = static_cast<GameInputForceFeedbackEffectKind>(11);
    check(dev->CreateForceFeedbackEffect(0, &unknown, reinterpret_cast<void**>(&effect)) == E_INVALIDARG,
          "force feedback: an effect kind outside the enum is E_INVALIDARG");
    check(dev->CreateForceFeedbackEffect(0, &params, reinterpret_cast<void**>(&effect)) == S_OK && effect,
          "force feedback: CreateForceFeedbackEffect");
    if (!effect) {  // leave nothing behind for the next test
        detach(wheel);
        dev->Release();
        rt::gameinput::poll();
        rt::gameinput::set_haptic_source(nullptr);
        return;
    }
    check(effect->GetMotorIndex() == 0 && approx(effect->GetGain(), 1.0f) && effect->GetState() == GameInputFeedbackRunning,
          "force feedback: a new effect is on motor 0, at full gain, running");
    IGameInputDevice* owner = nullptr;
    effect->GetDevice(&owner);
    check(owner == dev, "force feedback: the effect names its device");
    if (owner) owner->Release();

    // Nothing is asked of SDL before the poll thread runs: the title only wrote the parameters.
    const int runs0 = g_fake.runs;
    rt::gameinput::poll();
    check(g_fake.creates == 1 && g_fake.runs == runs0 + 1, "force feedback: the poll thread creates and runs the effect");
    rt::gameinput::OutputState sent{};
    check(rt::gameinput::output_state(0, &sent) && sent.effect_id == 1 && sent.effect_ran && !sent.effect_error[0],
          "force feedback: the output state reports the effect SDL created and is running");
    check(g_fake.last.type == SDL_HAPTIC_CONSTANT && g_fake.last.constant.level == 16384,
          "force feedback: a constant 0.5 becomes SDL_HAPTIC_CONSTANT level 16384");
    check(g_fake.last.constant.length == 60000 && g_fake.last_iterations == 0,
          "force feedback: no duration in GameInput is SDL's 60 s hold, run until stopped");
    rt::gameinput::poll();
    check(g_fake.runs == runs0 + 1, "force feedback: an unchanged effect is not re-sent every poll");

    // SDL refusing an effect is visible, not swallowed: the output state carries the failure and SDL's error.
    g_fake.create_result = -1;
    IGameInputForceFeedbackEffect* refused = nullptr;
    check(dev->CreateForceFeedbackEffect(0, &params, reinterpret_cast<void**>(&refused)) == S_OK && refused,
          "force feedback: GameInput accepts an effect SDL will refuse");
    if (refused) {
        rt::gameinput::poll();
        rt::gameinput::OutputState refused_out{};
        check(rt::gameinput::output_state(0, &refused_out) && refused_out.effect_id < 0 && !refused_out.effect_ran &&
                  std::strstr(refused_out.effect_error, "fake: refused"),
              "force feedback: SDL refusing the effect is recorded with its error");
        refused->Release();
        rt::gameinput::poll();  // the released effect leaves the device
    }
    g_fake.create_result = 1;

    params.data.constant.magnitude.linearX = -1.0f;
    check(effect->SetParams(&params), "force feedback: SetParams");
    rt::gameinput::poll();
    check(g_fake.updates == 1 && g_fake.runs == runs0 + 2 && g_fake.last.constant.level == -32767,
          "force feedback: changed parameters update the SDL effect once");
    GameInputForceFeedbackParams read{};
    effect->GetParams(&read);
    check(read.kind == GameInputForceFeedbackConstant && read.data.constant.magnitude.linearX == -1.0f,
          "force feedback: GetParams returns what was set");

    effect->SetGain(0.5f);
    rt::gameinput::poll();
    check(approx(effect->GetGain(), 0.5f) && g_fake.last.constant.level == -16384,
          "force feedback: the effect's gain scales the SDL level");
    // The device gain is SDL's, sent once when the pad is opened (100) and once per change.
    check(g_fake.last_gain == 100 && g_fake.gain_calls == 1, "force feedback: the device gain starts at SDL's full 100");
    dev->SetForceFeedbackMotorGain(0, 0.5f);
    rt::gameinput::poll();
    check(g_fake.last_gain == 50 && g_fake.gain_calls == 2,
          "force feedback: SetForceFeedbackMotorGain sets SDL's device gain once");
    dev->SetForceFeedbackMotorGain(0, 1.0f);
    effect->SetGain(1.0f);
    rt::gameinput::poll();
    check(g_fake.last_gain == 100, "force feedback: the master gain follows the request");

    // A wave with GameInput's envelope: 50 Hz for 90 degrees over 100 ms, played once.
    GameInputForceFeedbackParams wave{};
    wave.kind = GameInputForceFeedbackSineWave;
    wave.data.sineWave.envelope.sustainDuration = 100'000;  // microseconds
    wave.data.sineWave.envelope.sustainGain = 1.0f;
    wave.data.sineWave.envelope.playCount = 1;
    wave.data.sineWave.magnitude.linearX = 1.0f;
    wave.data.sineWave.frequency = 50.0f;
    wave.data.sineWave.phase = 90.0f;
    check(effect->SetParams(&wave), "force feedback: SetParams for a sine wave");
    rt::gameinput::poll();
    check(g_fake.last.type == SDL_HAPTIC_SINE && g_fake.last.periodic.period == 20 &&
              g_fake.last.periodic.magnitude == 32767 && g_fake.last.periodic.phase == 9000,
          "force feedback: a 50 Hz sine with a 90 degree phase becomes SDL's period and hundredths of a degree");
    check(g_fake.last.periodic.length == 100 && g_fake.last_iterations == 1,
          "force feedback: the envelope's 100 ms is SDL's length and its playCount the iteration count");
    Sleep(130);
    rt::gameinput::poll();
    check(effect->GetState() == GameInputFeedbackStopped && g_fake.stops >= 1,
          "force feedback: an effect that ran its duration reports stopped");

    // A condition effect: coefficients, saturations, dead zone and centre.
    GameInputForceFeedbackParams spring{};
    spring.kind = GameInputForceFeedbackSpring;
    spring.data.spring.magnitude.linearX = 0.5f;
    spring.data.spring.positiveCoefficient = 0.5f;
    spring.data.spring.negativeCoefficient = 0.25f;
    spring.data.spring.maxPositiveMagnitude = 1.0f;
    spring.data.spring.maxNegativeMagnitude = 0.5f;
    spring.data.spring.deadZone = 0.1f;
    check(effect->SetParams(&spring), "force feedback: SetParams for a spring");
    effect->SetState(GameInputFeedbackRunning);
    rt::gameinput::poll();
    check(g_fake.last.type == SDL_HAPTIC_SPRING && g_fake.last.condition.right_coeff[0] == 16384 &&
              g_fake.last.condition.left_coeff[0] == -8192 && g_fake.last.condition.right_sat[0] == 65535 &&
              g_fake.last.condition.left_sat[0] == 32768 && g_fake.last.condition.deadband[0] == 6554 &&
              g_fake.last.condition.center[0] == 16384,
          "force feedback: a spring maps its coefficients, saturations, dead zone and centre");

    effect->SetState(GameInputFeedbackPaused);
    const int stops0 = g_fake.stops;
    rt::gameinput::poll();
    check(g_fake.stops == stops0 + 1 && effect->GetState() == GameInputFeedbackPaused,
          "force feedback: a paused effect is stopped in SDL");

    // LEONIDA_RUMBLE=0 silences this path too: a running effect is stopped, and it runs again when the mute lifts.
    effect->SetState(GameInputFeedbackRunning);
    rt::gameinput::poll();
    const int runs1 = g_fake.runs, stops1 = g_fake.stops;
    check(g_fake.runs == runs1, "force feedback: the effect is running before the mute");
    set_env("LEONIDA_RUMBLE", "0");
    rt::gameinput::reload_config();
    rt::gameinput::poll();
    check(g_fake.runs == runs1 && g_fake.stops == stops1 + 1, "force feedback: LEONIDA_RUMBLE=0 stops the effect");
    set_env("LEONIDA_RUMBLE", nullptr);
    rt::gameinput::reload_config();
    rt::gameinput::poll();
    check(g_fake.runs == runs1 + 1, "force feedback: and it runs again once the mute is lifted");

    check(effect->Release() == 0, "force feedback: the effect is freed once the title releases it");
    rt::gameinput::poll();
    check(g_fake.destroys == 1, "force feedback: its SDL effect is destroyed on the poll thread");

    // RT-262: the title releases effects while the poll thread applies them. The poll thread used to AddRef an effect
    // whose count had already reached 0 (its destructor waiting for the lock), use it after it was freed, and free it
    // a second time with its own Release.
    {
        std::atomic<bool> stop{false};
        std::thread poller([&] {
            while (!stop.load()) rt::gameinput::poll();
        });
        int created = 0;
        for (int i = 0; i < 20000; ++i) {
            IGameInputForceFeedbackEffect* e = nullptr;
            if (dev->CreateForceFeedbackEffect(0, &params, reinterpret_cast<void**>(&e)) == S_OK && e) {
                ++created;
                e->Release();
            }
        }
        stop = true;
        poller.join();
        rt::gameinput::poll();
        check(created == 20000, "force feedback: 20,000 effects created and released while the poll thread runs");
    }

    detach(wheel);
    rt::gameinput::poll();
    check(g_fake.closes == 1 && !dev->IsForceFeedbackMotorPoweredOn(0),
          "force feedback: unplugging closes the haptic device");
    dev->Release();

    // A pad SDL opens no haptic device for keeps the old answer: no motor, and no effect.
    g_fake_open = false;
    Pad plain;
    attach(plain, "Leonida virtual plain pad", 0x045E, 0x02FF, false);
    const size_t plain_before = g_device_events.size();
    rt::gameinput::poll();
    IGameInputDevice* plaindev = g_device_events.size() > plain_before ? g_device_events.back().device : nullptr;
    check(plaindev && plaindev->GetDeviceInfo()->forceFeedbackMotorCount == 0 &&
              plaindev->GetDeviceInfo()->forceFeedbackMotorInfo == nullptr,
          "force feedback: a pad without SDL haptics reports no motor");
    if (plaindev) {
        effect = nullptr;
        check(plaindev->CreateForceFeedbackEffect(0, &params, reinterpret_cast<void**>(&effect)) == E_INVALIDARG && !effect,
              "force feedback: so CreateForceFeedbackEffect is E_INVALIDARG");
        check(!plaindev->IsForceFeedbackMotorPoweredOn(0), "force feedback: and no motor is powered on");
    }
    detach(plain);
    rt::gameinput::poll();
    rt::gameinput::set_haptic_source(nullptr);
}

// The trigger haptic motors of a pad that has them. Runs with no other pad.
void test_haptic_motors(IGameInput* input) {
    set_env("LEONIDA_DS_TRIGGERS", nullptr);
    rt::gameinput::reload_config();
    Pad xbox;
    attach(xbox, "Leonida virtual Xbox pad", 0x045E, 0x0B12, true);
    const size_t before = g_device_events.size();
    rt::gameinput::poll();
    IGameInputDevice* dev = g_device_events.size() > before ? g_device_events.back().device : nullptr;
    check(dev && rt::gameinput::connected_count() == 1, "haptics: the pad with trigger motors is connected");
    if (!dev) return;
    dev->AddRef();
    const GameInputDeviceInfo* info = dev->GetDeviceInfo();
    check(info->hapticFeedbackMotorCount == 2 && info->hapticFeedbackMotorInfo != nullptr,
          "haptics: the pad reports two haptic motors");
    const GameInputHapticFeedbackMotorInfo* motors =
        static_cast<const GameInputHapticFeedbackMotorInfo*>(info->hapticFeedbackMotorInfo);
    check(motors[0].mappedRumbleMotors == GameInputRumbleLeftTrigger &&
              motors[1].mappedRumbleMotors == GameInputRumbleRightTrigger && motors[0].waveformCount == 1 &&
              motors[0].waveformInfo != nullptr && motors[0].waveformInfo->isIntensitySupported,
          "haptics: they are the left and right trigger, one waveform each, an intensity");
    check(sticks_at_rest(pad_state(input, dev)), "haptics: the pad still reads normally");

    GameInputHapticFeedbackParams p{};
    p.intensity = 0.5f;
    p.duration = 100'000;  // microseconds
    p.playCount = 1;
    check(dev->SetHapticMotorState(2, &p) == E_INVALIDARG, "haptics: motor 2 does not exist");
    GameInputHapticFeedbackParams other_wave = p;
    other_wave.waveformIndex = 1;
    check(dev->SetHapticMotorState(0, &other_wave) == E_INVALIDARG, "haptics: waveform 1 does not exist");
    check(dev->SetHapticMotorState(0, &p) == S_OK, "haptics: SetHapticMotorState on the left trigger");
    rt::gameinput::poll();
    check(xbox.left == 32768 && xbox.right == 0, "haptics: the intensity reaches the trigger motor");
    Sleep(130);
    rt::gameinput::poll();
    check(xbox.left == 0, "haptics: the request ends with its duration");

    p.intensity = 1.0f;
    p.playCount = 0;  // until changed
    check(dev->SetHapticMotorState(1, &p) == S_OK, "haptics: playCount 0 plays until changed");
    rt::gameinput::poll();
    check(xbox.right == 65535 && xbox.left == 0, "haptics: the right trigger takes its own motor");
    Sleep(130);
    rt::gameinput::poll();
    check(xbox.right == 65535, "haptics: and keeps playing past the duration");
    check(dev->SetHapticMotorState(1, nullptr) == S_OK, "haptics: a null state stops the motor");
    rt::gameinput::poll();
    check(xbox.right == 0 && xbox.left == 0, "haptics: both triggers are silent again");

    // SetRumbleState still works on the same pad, and the two share the trigger output.
    GameInputRumbleParams rumble{0, 0, 0.25f, 0};
    dev->SetRumbleState(&rumble);
    rt::gameinput::poll();
    check(xbox.left == 16384, "haptics: SetRumbleState still drives the trigger motor");
    p.intensity = 1.0f;
    p.playCount = 1;
    p.duration = 0;  // no deadline at playCount 1 either: the level holds until it is set again
    dev->SetHapticMotorState(0, &p);
    rt::gameinput::poll();
    check(xbox.left == 65535, "haptics: the stronger of the two wins on one trigger");
    dev->SetRumbleState(nullptr);
    dev->SetHapticMotorState(0, nullptr);
    rt::gameinput::poll();
    check(xbox.left == 0, "haptics: releasing both leaves the trigger silent");

    detach(xbox);
    rt::gameinput::poll();
    check(rt::gameinput::connected_count() == 0, "haptics: detached");
    dev->Release();
}

}  // namespace

int main() {
    // Only virtual pads: no real controller on the test machine may show up.
    for (const char* hint : {SDL_HINT_JOYSTICK_HIDAPI, SDL_HINT_JOYSTICK_RAWINPUT, SDL_HINT_JOYSTICK_WGI,
                             SDL_HINT_JOYSTICK_GAMEINPUT, SDL_HINT_XINPUT_ENABLED, SDL_HINT_JOYSTICK_DIRECTINPUT})
        SDL_SetHintWithPriority(hint, "0", SDL_HINT_OVERRIDE);
    set_env("LEONIDA_GAMEINPUT_THREAD", "0");
    set_env("LEONIDA_DS_TRIGGERS", nullptr);
    set_env("LEONIDA_DS_LIGHTBAR", nullptr);
    set_env("LEONIDA_RUMBLE", nullptr);
    test_add_ref_if_live();

    auto create = reinterpret_cast<GameInputCreateFn>(rt::shims::find("gameinput.dll", "GameInputCreate"));
    check(create != nullptr, "gameinput.dll!GameInputCreate is registered");
    check(rt::shims::find("xgameruntime.dll", "GameInputCreate") == reinterpret_cast<void*>(create),
          "xgameruntime.dll!GameInputCreate is the same function");
    if (!create) return 1;

    IGameInput* input = nullptr;
    check(create(&input) == S_OK && input, "GameInputCreate returns an instance");
    check(create(nullptr) == E_POINTER, "GameInputCreate(nullptr) is E_POINTER");
    IUnknown* unk = nullptr;
    check(input->QueryInterface(IID_IGameInput, reinterpret_cast<void**>(&unk)) == S_OK && unk == input,
          "QueryInterface(IID_IGameInput)");
    if (unk) unk->Release();

    rt::gameinput::poll();  // initialises SDL on this thread
    check(rt::gameinput::connected_count() == 0, "no gamepads before any is attached");

    IGameInputReading* reading = nullptr;
    check(input->GetCurrentReading(GameInputKindGamepad, nullptr, &reading) == GAMEINPUT_E_READING_NOT_FOUND && !reading,
          "GetCurrentReading with no device is READING_NOT_FOUND");

    GameInputCallbackToken device_token = 0, reading_token = 0, guide_token = 0;
    check(input->RegisterDeviceCallback(nullptr, GameInputKindGamepad, GameInputDeviceConnected,
                                        GameInputBlockingEnumeration, nullptr, on_device, &device_token) == S_OK &&
              device_token != 0,
          "RegisterDeviceCallback (blocking enumeration, nothing connected)");
    check(g_device_events.empty(), "no enumeration callback without devices");
    check(input->RegisterReadingCallback(nullptr, GameInputKindGamepad, 0.0f, nullptr, on_reading, &reading_token) == S_OK,
          "RegisterReadingCallback");
    check(input->RegisterGuideButtonCallback(nullptr, nullptr, on_guide, &guide_token) == S_OK, "RegisterGuideButtonCallback");

    // ---- an Xbox-style pad with impulse triggers ----
    Pad xbox;
    attach(xbox, "Leonida virtual Xbox pad", 0x045E, 0x0B12, true);
    check(xbox.js != nullptr, "virtual Xbox pad attached");
    rt::gameinput::poll();
    check(rt::gameinput::connected_count() == 1, "the shim opened it");
    check(g_device_events.size() == 1 && g_device_events[0].previous == GameInputDeviceNoStatus &&
              (g_device_events[0].current & GameInputDeviceConnected),
          "device callback: connected");
    IGameInputDevice* dev = g_device_events.empty() ? nullptr : g_device_events[0].device;
    if (!dev) return 1;
    dev->AddRef();

    const GameInputDeviceInfo* info = dev->GetDeviceInfo();
    check(info->infoSize == sizeof(GameInputDeviceInfo), "info.infoSize");
    check(info->vendorId == 0x045E && info->productId == 0x0B12, "info vendor and product id");
    check((info->supportedInput & GameInputKindGamepad) && (info->supportedInput & GameInputKindUiNavigation),
          "info.supportedInput has Gamepad and UiNavigation");
    check(info->supportedRumbleMotors == (GameInputRumbleLowFrequency | GameInputRumbleHighFrequency |
                                          GameInputRumbleLeftTrigger | GameInputRumbleRightTrigger),
          "info.supportedRumbleMotors: both motors and both triggers");
    check(info->controllerAxisCount == 6 && info->controllerAxisInfo && info->gamepadInfo && info->displayName &&
              std::strcmp(info->displayName->data, "Leonida virtual Xbox pad") == 0,
          "info axis table, gamepad info and display name");
    check(dev->GetDeviceStatus() & GameInputDeviceConnected, "GetDeviceStatus: connected");
    check(xbox.player == 0, "player index 0 for the first pad");

    // Initial reading: everything at rest.
    check(input->GetCurrentReading(GameInputKindGamepad, dev, &reading) == S_OK && reading, "initial reading");
    GameInputGamepadState st{};
    if (reading) {
        check(reading->GetGamepadState(&st) && st.buttons == 0 && approx(st.leftTrigger, 0) && approx(st.leftThumbstickX, 0),
              "initial gamepad state is at rest");
        IGameInputDevice* rd = nullptr;
        reading->GetDevice(&rd);
        check(rd == dev, "reading->GetDevice is the device");
        if (rd) rd->Release();
    }
    const uint64_t seq0 = reading ? reading->GetSequenceNumber(GameInputKindGamepad) : 0;

    // Press A, push the left stick up-right, pull the left trigger, press the right shoulder and dpad down.
    const int reading_events_before = g_reading_events;
    SDL_SetJoystickVirtualButton(xbox.js, SDL_GAMEPAD_BUTTON_SOUTH, true);
    SDL_SetJoystickVirtualButton(xbox.js, SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER, true);
    SDL_SetJoystickVirtualButton(xbox.js, SDL_GAMEPAD_BUTTON_DPAD_DOWN, true);
    SDL_SetJoystickVirtualButton(xbox.js, SDL_GAMEPAD_BUTTON_START, true);
    SDL_SetJoystickVirtualAxis(xbox.js, SDL_GAMEPAD_AXIS_LEFTX, 32767);
    SDL_SetJoystickVirtualAxis(xbox.js, SDL_GAMEPAD_AXIS_LEFTY, -32768);  // SDL: up is negative
    SDL_SetJoystickVirtualAxis(xbox.js, SDL_GAMEPAD_AXIS_RIGHTX, -32768);
    SDL_SetJoystickVirtualAxis(xbox.js, SDL_GAMEPAD_AXIS_LEFT_TRIGGER, 32767);
    rt::gameinput::poll();
    check(g_reading_events > reading_events_before, "reading callback after a change");

    IGameInputReading* next = nullptr;
    check(input->GetNextReading(reading, GameInputKindGamepad, nullptr, &next) == S_OK && next,
          "GetNextReading after a change");
    if (next) next->Release();
    if (reading) reading->Release();
    reading = nullptr;
    check(input->GetCurrentReading(GameInputKindGamepad, nullptr, &reading) == S_OK && reading,
          "GetCurrentReading(any device)");
    if (reading) {
        st = {};
        reading->GetGamepadState(&st);
        std::printf("     buttons=0x%x lt=%.3f rt=%.3f lx=%.3f ly=%.3f rx=%.3f ry=%.3f\n", st.buttons, st.leftTrigger,
                    st.rightTrigger, st.leftThumbstickX, st.leftThumbstickY, st.rightThumbstickX, st.rightThumbstickY);
        check(st.buttons == (GameInputGamepadA | GameInputGamepadRightShoulder | GameInputGamepadDPadDown | GameInputGamepadMenu),
              "buttons: A, right shoulder, dpad down, Menu (Start)");
        check(approx(st.leftThumbstickX, 1) && approx(st.leftThumbstickY, 1), "left stick up-right is (+1, +1)");
        check(approx(st.rightThumbstickX, -1) && approx(st.rightThumbstickY, 0), "right stick left is (-1, 0)");
        check(approx(st.leftTrigger, 1) && approx(st.rightTrigger, 0), "left trigger 1, right trigger 0");
        check(reading->GetSequenceNumber(GameInputKindGamepad) > seq0, "sequence number advanced");
        GameInputUiNavigationState ui{};
        check(reading->GetUiNavigationState(&ui), "GetUiNavigationState");
        const uint32_t want_ui = GameInputUiNavigationAccept | GameInputUiNavigationMenu | GameInputUiNavigationDown |
                                 GameInputUiNavigationUp | GameInputUiNavigationRight | GameInputUiNavigationPageRight |
                                 GameInputUiNavigationPageUp | GameInputUiNavigationScrollLeft;
        check(ui.buttons == want_ui, "UI navigation derived from the pad");
        float axes[8] = {};
        check(reading->GetControllerAxisState(8, axes) == 6 && approx(axes[0], 1) && approx(axes[3], 0.5f) && approx(axes[4], 1),
              "controller axes (0..1, sticks centred at 0.5)");
        bool buttons[64] = {};
        check(reading->GetControllerButtonState(64, buttons) == SDL_GAMEPAD_BUTTON_COUNT && buttons[SDL_GAMEPAD_BUTTON_SOUTH] &&
                  !buttons[SDL_GAMEPAD_BUTTON_EAST],
              "controller buttons");
        check(!reading->GetMotionState(nullptr) && !reading->GetMouseState(nullptr), "no motion or mouse state");
        IGameInputReading* none = nullptr;
        check(input->GetNextReading(reading, GameInputKindGamepad, nullptr, &none) == GAMEINPUT_E_READING_NOT_FOUND && !none,
              "GetNextReading at the newest reading is READING_NOT_FOUND");
    }
    // Nothing changed: no new reading, no callback.
    const int events_now = g_reading_events;
    rt::gameinput::poll();
    check(g_reading_events == events_now, "no reading callback without a change");

    // Guide button.
    SDL_SetJoystickVirtualButton(xbox.js, SDL_GAMEPAD_BUTTON_GUIDE, true);
    rt::gameinput::poll();
    SDL_SetJoystickVirtualButton(xbox.js, SDL_GAMEPAD_BUTTON_GUIDE, false);
    rt::gameinput::poll();
    check(g_guide_events.size() == 2 && g_guide_events[0] && !g_guide_events[1], "guide button callback: down, up");

    // Rumble: both motors and both trigger motors reach SDL.
    GameInputRumbleParams rumble{0.5f, 1.0f, 0.25f, 0.75f};
    dev->SetRumbleState(&rumble);
    rt::gameinput::poll();
    std::printf("     rumble low=%u high=%u left=%u right=%u\n", xbox.low, xbox.high, xbox.left, xbox.right);
    check(xbox.low == 32768 && xbox.high == 65535, "SetRumbleState motors -> SDL_RumbleGamepad");
    check(xbox.left == 16384 && xbox.right == 49151, "SetRumbleState triggers -> SDL_RumbleGamepadTriggers");
    check(xbox.effect_calls == 0, "no DualSense effect for an Xbox pad");
    const int rumble_calls = xbox.rumble_calls;
    rt::gameinput::poll();
    check(xbox.rumble_calls == rumble_calls, "unchanged rumble is not re-sent every poll");
    dev->SetRumbleState(nullptr);
    rt::gameinput::poll();
    check(xbox.low == 0 && xbox.high == 0 && xbox.left == 0 && xbox.right == 0, "SetRumbleState(nullptr) stops rumble");

    set_env("LEONIDA_RUMBLE", "0");
    rt::gameinput::reload_config();
    dev->SetRumbleState(&rumble);
    rt::gameinput::poll();
    check(xbox.low == 0 && xbox.left == 0, "LEONIDA_RUMBLE=0 keeps the motors off");
    set_env("LEONIDA_RUMBLE", nullptr);
    rt::gameinput::reload_config();
    dev->SetRumbleState(nullptr);
    rt::gameinput::poll();

    // ---- a DualSense: trigger rumble becomes adaptive-trigger effects ----
    set_env("LEONIDA_DS_LIGHTBAR", "ff8000");
    rt::gameinput::reload_config();
    Pad ds;
    attach(ds, "Leonida virtual DualSense", 0x054C, 0x0CE6, true);
    rt::gameinput::poll();
    rt::gameinput::OutputState out{};
    check(rt::gameinput::connected_count() == 2 && rt::gameinput::output_state(1, &out) && out.is_dualsense,
          "a Sony 054c:0ce6 pad is handled as a DualSense");
    check(ds.player == 1, "player index 1 for the second pad");
    check(ds.led_calls >= 1 && ds.led[0] == 0xFF && ds.led[1] == 0x80 && ds.led[2] == 0x00, "LEONIDA_DS_LIGHTBAR=ff8000");
    IGameInputDevice* dsdev = g_device_events.empty() ? nullptr : g_device_events.back().device;
    check(dsdev && dsdev != dev && (dsdev->GetDeviceInfo()->supportedRumbleMotors & GameInputRumbleLeftTrigger),
          "DualSense device reported with trigger motors");
    if (!dsdev) return 1;

    GameInputRumbleParams ds_rumble{0.0f, 0.0f, 1.0f, 0.5f};
    dsdev->SetRumbleState(&ds_rumble);
    rt::gameinput::poll();
    std::printf("     effect size=%d bits=%02x R=%02x %02x %02x %02x L=%02x %02x %02x %02x\n", ds.effect_size,
                ds.effect[0], ds.effect[10], ds.effect[11], ds.effect[12], ds.effect[13], ds.effect[21], ds.effect[22],
                ds.effect[23], ds.effect[24]);
    check(ds.effect_size == 47 && ds.effect[0] == 0x0C, "effect block: 47 bytes, right+left trigger flags");
    check(ds.effect[10] == 0x06 && ds.effect[11] == 30 && ds.effect[12] == 128, "rumble mode: right trigger vibrates at half");
    check(ds.effect[21] == 0x06 && ds.effect[23] == 255, "rumble mode: left trigger vibrates at full");
    check(ds.trigger_calls == 0, "a DualSense gets effects, not SDL_RumbleGamepadTriggers");

    set_env("LEONIDA_DS_TRIGGERS", "resistance");
    rt::gameinput::reload_config();
    ds_rumble = {0, 0, 0.0f, 1.0f};
    dsdev->SetRumbleState(&ds_rumble);
    rt::gameinput::poll();
    check(ds.effect[10] == 0x01 && ds.effect[12] == 255 && ds.effect[21] == 0x05,
          "resistance mode: right trigger resists, left trigger off");

    set_env("LEONIDA_DS_TRIGGERS", "off");
    rt::gameinput::reload_config();
    const int effect_calls = ds.effect_calls;
    ds_rumble = {0, 0, 1.0f, 1.0f};
    dsdev->SetRumbleState(&ds_rumble);
    rt::gameinput::poll();
    check(ds.effect_calls == effect_calls, "LEONIDA_DS_TRIGGERS=off sends no trigger effect");
    set_env("LEONIDA_DS_TRIGGERS", nullptr);
    set_env("LEONIDA_DS_LIGHTBAR", nullptr);
    rt::gameinput::reload_config();

    // The most recent reading of any device: move the DualSense.
    SDL_SetJoystickVirtualButton(ds.js, SDL_GAMEPAD_BUTTON_EAST, true);
    rt::gameinput::poll();
    if (reading) reading->Release();
    reading = nullptr;
    if (input->GetCurrentReading(GameInputKindGamepad, nullptr, &reading) == S_OK && reading) {
        IGameInputDevice* rd = nullptr;
        reading->GetDevice(&rd);
        st = {};
        reading->GetGamepadState(&st);
        check(rd == dsdev && st.buttons == GameInputGamepadB, "GetCurrentReading(any) follows the latest change (B on the DualSense)");
        if (rd) rd->Release();
    } else {
        check(false, "GetCurrentReading(any) with two pads");
    }

    // ---- a pad without trigger motors ----
    Pad plain;
    attach(plain, "Leonida virtual plain pad", 0x057E, 0x2009, false);  // Switch Pro ids: rumble, no trigger motors
    rt::gameinput::poll();
    IGameInputDevice* plaindev = g_device_events.empty() ? nullptr : g_device_events.back().device;
    check(plaindev && (plaindev->GetDeviceInfo()->supportedRumbleMotors & GameInputRumbleLeftTrigger) == 0,
          "a pad without trigger motors does not report them");
    if (plaindev) {
        GameInputRumbleParams p{0.25f, 0.25f, 1.0f, 1.0f};
        plaindev->SetRumbleState(&p);
        rt::gameinput::poll();
        check(plain.low == 16384 && plain.trigger_calls == 0 && plain.effect_calls == 0,
              "its body motors rumble and the trigger request is dropped");
    }
    detach(plain);
    rt::gameinput::poll();

    // ---- dispatcher: callbacks wait for Dispatch ----
    IGameInputDispatcher* dispatcher = nullptr;
    check(input->CreateDispatcher(&dispatcher) == S_OK && dispatcher, "CreateDispatcher");
    HANDLE wait = nullptr;
    check(dispatcher && dispatcher->OpenWaitHandle(&wait) == S_OK && wait, "OpenWaitHandle");
    const int before_dispatch = g_reading_events;
    SDL_SetJoystickVirtualButton(xbox.js, SDL_GAMEPAD_BUTTON_NORTH, true);
    rt::gameinput::poll();
    check(g_reading_events == before_dispatch, "with a dispatcher, poll does not run callbacks");
    check(wait && WaitForSingleObject(wait, 0) == WAIT_OBJECT_0, "the wait handle is signalled");
    if (dispatcher) dispatcher->Dispatch(1000);
    check(g_reading_events > before_dispatch, "Dispatch runs them");

    // The quota: Dispatch runs at least one queued event, takes no more once the quota has passed, and returns whether
    // events remain (re-signalling the wait handle). Four button changes queue four reading events.
    auto queue_changes = [&](int n) {
        for (int i = 0; i < n; ++i) {
            SDL_SetJoystickVirtualButton(xbox.js, SDL_GAMEPAD_BUTTON_NORTH, i % 2 == 1);  // NORTH is down now: up, down, up, down
            rt::gameinput::poll();
        }
    };
    if (dispatcher) {
        while (dispatcher->Dispatch(UINT64_MAX)) {}
        queue_changes(4);
        int before_quota = g_reading_events;
        if (wait) WaitForSingleObject(wait, 0);  // consume the signal from queueing
        check(dispatcher->Dispatch(0) && g_reading_events == before_quota + 1, "Dispatch(0) runs exactly one event and reports more");
        check(wait && WaitForSingleObject(wait, 0) == WAIT_OBJECT_0, "the wait handle is signalled again while events remain");
        // The callback sleeps 5 ms, so at least 5 ms have passed after the first event however loaded the machine
        // is; a slower run only makes the 1 ms quota more clearly spent.
        g_reading_sleep_ms = 5;
        before_quota = g_reading_events;
        check(dispatcher->Dispatch(1000) && g_reading_events == before_quota + 1,
              "a 1 ms quota stops after a 5 ms callback and reports more");
        g_reading_sleep_ms = 0;
        before_quota = g_reading_events;
        check(!dispatcher->Dispatch(1'000'000) && g_reading_events >= before_quota + 2,
              "a large quota drains the queue and returns false");
        check(!dispatcher->Dispatch(0), "Dispatch on an empty queue returns false");

        // Events queued during a call wait for the next one: each callback here queues another reading, so a
        // Dispatch that kept draining would never return. With two events queued, each call runs exactly the batch
        // it started with and reports the readings queued meanwhile.
        queue_changes(2);
        g_requeue_pad = xbox.js;
        before_quota = g_reading_events;
        check(dispatcher->Dispatch(UINT64_MAX) && g_reading_events == before_quota + 2,
              "Dispatch runs only the events queued when it started");
        before_quota = g_reading_events;
        check(dispatcher->Dispatch(UINT64_MAX) && g_reading_events == before_quota + 2,
              "the events queued during the call run on the next one");
        g_requeue_pad = nullptr;
        while (dispatcher->Dispatch(UINT64_MAX)) {}
        if (g_requeue_down) {
            g_requeue_down = false;
            SDL_SetJoystickVirtualButton(xbox.js, SDL_GAMEPAD_BUTTON_WEST, false);
            rt::gameinput::poll();
            while (dispatcher->Dispatch(UINT64_MAX)) {}
        }
    }
    if (wait) CloseHandle(wait);
    if (dispatcher) dispatcher->Release();

    // ---- unplug and replug ----
    const size_t events_before_unplug = g_device_events.size();
    detach(xbox);
    rt::gameinput::poll();
    check(g_device_events.size() == events_before_unplug + 1 && g_device_events.back().device == dev &&
              g_device_events.back().current == GameInputDeviceNoStatus &&
              (g_device_events.back().previous & GameInputDeviceConnected),
          "device callback: disconnected");
    check(dev->GetDeviceStatus() == GameInputDeviceNoStatus, "GetDeviceStatus after unplug");
    IGameInputReading* gone = nullptr;
    check(input->GetCurrentReading(GameInputKindGamepad, dev, &gone) == GAMEINPUT_E_DEVICE_DISCONNECTED && !gone,
          "GetCurrentReading on an unplugged device is DEVICE_DISCONNECTED");
    AppLocalDeviceId id = info->deviceId;
    IGameInputDevice* found = nullptr;
    check(input->FindDeviceFromId(&id, &found) == S_OK && found == dev, "FindDeviceFromId");
    if (found) found->Release();

    Pad xbox2;
    attach(xbox2, "Leonida virtual Xbox pad", 0x045E, 0x0B12, true);
    rt::gameinput::poll();
    check(g_device_events.back().device == dev && (g_device_events.back().current & GameInputDeviceConnected),
          "the same pad plugged back in is the same device object");
    check(xbox2.player == 0, "it takes the free player slot 0");

    // Async enumeration delivers on the next poll; unregistering stops callbacks.
    std::vector<DeviceEvent> before = g_device_events;
    GameInputCallbackToken async_token = 0;
    input->RegisterDeviceCallback(nullptr, GameInputKindGamepad, GameInputDeviceConnected, GameInputAsyncEnumeration,
                                  nullptr, on_device, &async_token);
    check(g_device_events.size() == before.size(), "async enumeration does not call back before returning");
    rt::gameinput::poll();
    size_t async_events = 0;
    for (size_t i = before.size(); i < g_device_events.size(); ++i) async_events += g_device_events[i].token == async_token;
    check(async_events == 2, "async enumeration reports both connected pads to the new callback only");

    check(input->UnregisterCallback(reading_token, 0), "UnregisterCallback");
    check(!input->UnregisterCallback(reading_token, 0), "a second UnregisterCallback returns false");
    const int after_unregister = g_reading_events;
    SDL_SetJoystickVirtualButton(xbox2.js, SDL_GAMEPAD_BUTTON_WEST, true);
    rt::gameinput::poll();
    check(g_reading_events == after_unregister, "no reading callback after unregistering");

    // Blocking enumeration with pads present runs before returning.
    before = g_device_events;
    GameInputCallbackToken blocking_token = 0;
    input->RegisterDeviceCallback(nullptr, GameInputKindGamepad, GameInputDeviceConnected, GameInputBlockingEnumeration,
                                  nullptr, on_device, &blocking_token);
    check(g_device_events.size() == before.size() + 2, "blocking enumeration reports both pads before returning");

    // ---- reading history: GetNextReading and GetPreviousReading ----
    // A tap of A on the Xbox pad, pressed and released before the title reads again.
    SDL_SetJoystickVirtualButton(xbox2.js, SDL_GAMEPAD_BUTTON_WEST, false);
    SDL_SetJoystickVirtualButton(ds.js, SDL_GAMEPAD_BUTTON_EAST, false);
    rt::gameinput::poll();
    IGameInputReading* idle = current(input, dev);
    check(buttons_of(idle) == 0, "history: the reference reading is at rest");
    SDL_SetJoystickVirtualButton(xbox2.js, SDL_GAMEPAD_BUTTON_SOUTH, true);
    rt::gameinput::poll();
    SDL_SetJoystickVirtualButton(xbox2.js, SDL_GAMEPAD_BUTTON_SOUTH, false);
    rt::gameinput::poll();
    HRESULT hr = S_OK;
    std::vector<IGameInputReading*> seen = walk(input, idle, nullptr, true, 8, &hr);
    check(seen.size() == 2 && buttons_of(seen[0]) == GameInputGamepadA && buttons_of(seen[1]) == 0 &&
              hr == GAMEINPUT_E_READING_NOT_FOUND,
          "history: GetNextReading gives a tap's press, its release, then READING_NOT_FOUND");
    std::vector<IGameInputReading*> again = walk(input, seen.empty() ? nullptr : seen.back(), nullptr, false, 2);
    check(again.size() == 2 && again[0] == seen[0] && again[1] == idle,
          "history: GetPreviousReading walks back over the press to the reference");
    release_all(again);
    release_all(seen);
    IGameInputReading* hit = nullptr;
    check(input->GetNextReading(idle, GameInputKindGamepad | GameInputKindKeyboard, dev, &hit) == S_OK &&
              buttons_of(hit) == GameInputGamepadA,
          "history: a kind mask that includes the gamepad matches");
    if (hit) hit->Release();
    check(input->GetNextReading(idle, GameInputKindKeyboard, dev, &hit) == GAMEINPUT_E_READING_NOT_FOUND && !hit,
          "history: keyboard alone finds no reading");
    check(input->GetPreviousReading(idle, GameInputKindGamepad, dev, nullptr) == E_POINTER &&
              input->GetPreviousReading(nullptr, GameInputKindGamepad, dev, &hit) == E_INVALIDARG && !hit,
          "history: a null output is E_POINTER, a null reference E_INVALIDARG");
    // GetTemporalReading: E_NOTIMPL with a null reading, as GameInput v0 itself, for a timestamp in the history, one
    // before it and the current time, with and without a device filter.
    const uint64_t idle_us = idle->GetTimestamp();
    bool temporal_notimpl = true;
    for (const uint64_t at : {idle_us, idle_us - 1'000'000, input->GetCurrentTimestamp()})
        for (IGameInputDevice* filter : {dev, static_cast<IGameInputDevice*>(nullptr)}) {
            IGameInputReading* temporal = idle;  // a stale value the call must clear
            temporal_notimpl &= input->GetTemporalReading(at, filter, &temporal) == E_NOTIMPL && !temporal;
        }
    check(temporal_notimpl && input->GetTemporalReading(idle_us, dev, nullptr) == E_POINTER,
          "history: GetTemporalReading is E_NOTIMPL with a null reading, as in GameInput v0");

    // Two pads in one stream, one poll each: A on the Xbox pad, B on the DualSense, Y on the Xbox pad.
    IGameInputReading* start = current(input, dev);
    SDL_SetJoystickVirtualButton(xbox2.js, SDL_GAMEPAD_BUTTON_SOUTH, true);
    rt::gameinput::poll();
    SDL_SetJoystickVirtualButton(ds.js, SDL_GAMEPAD_BUTTON_EAST, true);
    rt::gameinput::poll();
    SDL_SetJoystickVirtualButton(xbox2.js, SDL_GAMEPAD_BUTTON_NORTH, true);
    rt::gameinput::poll();
    seen = walk(input, start, nullptr, true, 8, &hr);
    const bool interleaved = seen.size() == 3 && device_of(seen[0]) == dev && device_of(seen[1]) == dsdev &&
                             device_of(seen[2]) == dev && buttons_of(seen[1]) == GameInputGamepadB;
    check(interleaved && hr == GAMEINPUT_E_READING_NOT_FOUND,
          "history: with no device filter, both pads' readings come in the order they were taken");
    if (interleaved) {
        again = walk(input, seen[2], nullptr, false, 3);
        check(again.size() == 3 && again[0] == seen[1] && again[1] == seen[0] && again[2] == start,
              "history: GetPreviousReading gives the exact reverse");
        release_all(again);
        again = walk(input, start, dev, true, 8);
        check(again.size() == 2 && again[0] == seen[0] && again[1] == seen[2], "history: a device filter skips the other pad");
        release_all(again);
        check(input->GetNextReading(seen[0], GameInputKindGamepad, dsdev, &hit) == S_OK && hit == seen[1],
              "history: filtering on the other pad gives its next reading after the reference");
        if (hit) hit->Release();
        check(input->GetNextReading(seen[1], GameInputKindGamepad, dsdev, &hit) == GAMEINPUT_E_READING_NOT_FOUND && !hit,
              "history: and never one of the reference's pad");
    }
    release_all(seen);
    if (start) start->Release();

    // Both pads change in one poll and their readings share a timestamp: each still comes once, in both directions.
    IGameInputReading* pair_ref = current(input, dev);
    SDL_SetJoystickVirtualButton(xbox2.js, SDL_GAMEPAD_BUTTON_NORTH, false);
    SDL_SetJoystickVirtualButton(ds.js, SDL_GAMEPAD_BUTTON_EAST, false);
    rt::gameinput::poll();
    seen = walk(input, pair_ref, nullptr, true, 8, &hr);
    const bool paired = seen.size() == 2 && seen[0]->GetTimestamp() == seen[1]->GetTimestamp() &&
                        device_of(seen[0]) != device_of(seen[1]);
    check(paired && hr == GAMEINPUT_E_READING_NOT_FOUND, "history: two readings with one timestamp both come, once each");
    if (paired) {
        again = walk(input, seen[1], nullptr, false, 2);
        check(again.size() == 2 && again[0] == seen[0] && again[1] == pair_ref, "history: and in reverse");
        release_all(again);
    }
    release_all(seen);
    if (pair_ref) pair_ref->Release();

    // Ageing: a reading leaves the history once the reading after it is more than half a second old.
    IGameInputReading* aged = current(input, dev);
    SDL_SetJoystickVirtualButton(xbox2.js, SDL_GAMEPAD_BUTTON_LEFT_SHOULDER, true);
    rt::gameinput::poll();
    Sleep(600);
    SDL_SetJoystickVirtualButton(xbox2.js, SDL_GAMEPAD_BUTTON_LEFT_SHOULDER, false);
    rt::gameinput::poll();
    const uint32_t a_ls = GameInputGamepadA | GameInputGamepadLeftShoulder;
    check(input->GetNextReading(aged, GameInputKindGamepad, dev, &hit) == S_OK && buttons_of(hit) == a_ls,
          "history: a reference that has aged out still leads to its successor while that is kept");
    if (hit) hit->Release();
    check(input->GetPreviousReading(aged, GameInputKindGamepad, dev, &hit) == GAMEINPUT_E_REFERENCE_READING_TOO_OLD && !hit,
          "history: GetPreviousReading on an aged-out reference is REFERENCE_READING_TOO_OLD");
    Sleep(600);
    SDL_SetJoystickVirtualButton(xbox2.js, SDL_GAMEPAD_BUTTON_LEFT_SHOULDER, true);
    rt::gameinput::poll();
    check(input->GetNextReading(aged, GameInputKindGamepad, dev, &hit) == GAMEINPUT_E_REFERENCE_READING_TOO_OLD && !hit &&
              input->GetNextReading(aged, GameInputKindGamepad, nullptr, &hit) == GAMEINPUT_E_REFERENCE_READING_TOO_OLD && !hit,
          "history: once a reading after the reference is dropped, GetNextReading is REFERENCE_READING_TOO_OLD");
    check(buttons_of(aged) == GameInputGamepadA && device_of(aged) == dev,
          "history: a reading the title holds stays readable after it leaves the history");
    IGameInputReading* fresh = current(input, dev);
    check(fresh && input->GetNextReading(fresh, GameInputKindGamepad, dev, &hit) == GAMEINPUT_E_READING_NOT_FOUND && !hit,
          "history: after GetCurrentReading the title is caught up");

    // Unplugging drops the pad's history; plugged back in, its last reading leads to the reconnect snapshot.
    const uint64_t last_seq = fresh ? fresh->GetSequenceNumber(GameInputKindGamepad) : 0;
    detach(xbox2);
    rt::gameinput::poll();
    check(input->GetNextReading(fresh, GameInputKindGamepad, dev, &hit) == GAMEINPUT_E_DEVICE_DISCONNECTED && !hit &&
              input->GetPreviousReading(fresh, GameInputKindGamepad, dev, &hit) == GAMEINPUT_E_DEVICE_DISCONNECTED && !hit,
          "history: filtering on an unplugged pad is DEVICE_DISCONNECTED");
    check(input->GetNextReading(fresh, GameInputKindGamepad, nullptr, &hit) == GAMEINPUT_E_READING_NOT_FOUND && !hit,
          "history: with no filter, nothing follows an unplugged pad's last reading");
    check(buttons_of(fresh) == a_ls && device_of(fresh) == dev, "history: a held reading stays readable after unplugging");
    attach(xbox2, "Leonida virtual Xbox pad", 0x045E, 0x0B12, true);
    rt::gameinput::poll();
    IGameInputReading* reconnect = nullptr;
    check(input->GetNextReading(fresh, GameInputKindGamepad, dev, &reconnect) == S_OK && device_of(reconnect) == dev &&
              buttons_of(reconnect) == 0 && reconnect->GetSequenceNumber(GameInputKindGamepad) > last_seq,
          "history: after replugging, the last reading before the unplug leads to the reconnect snapshot");
    check(reconnect && input->GetPreviousReading(reconnect, GameInputKindGamepad, dev, &hit) == GAMEINPUT_E_READING_NOT_FOUND && !hit,
          "history: nothing comes before the reconnect snapshot");
    check(input->GetNextReading(aged, GameInputKindGamepad, dev, &hit) == GAMEINPUT_E_REFERENCE_READING_TOO_OLD && !hit,
          "history: an older reading from before the unplug is REFERENCE_READING_TOO_OLD");
    if (reconnect) reconnect->Release();
    check(idle && aged && fresh && idle->Release() == 0 && aged->Release() == 0 && fresh->Release() == 0,
          "history: dropped readings are freed once the title releases them");

    if (reading) reading->Release();
    dev->Release();
    detach(xbox2);
    detach(ds);
    rt::gameinput::poll();
    check(rt::gameinput::connected_count() == 0, "all pads closed after detaching");

    test_active_input();
    test_twins(input);
    test_keyboard_pad(input);
    test_motion(input);
    test_aggregate_device(input);
    test_force_feedback(input);
    test_haptic_motors(input);
    test_raw_hid(input);
    test_raw_report_during_reconnect(input);
    input->Release();

    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
