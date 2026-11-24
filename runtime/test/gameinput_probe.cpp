// gameinput_probe: the GameInput shim on a real controller. Not a CTest; run it by hand with a pad connected.
//
//   gameinput_probe [--seconds N] [--rumble] [--force-feedback]
//
// Lists the gamepads the shim sees, with the raw HID report tables of each pad opened through HID, then prints the
// live gamepad reading (and its newest raw input report, when there is one) for N seconds (default 10). With
// --rumble it first plays, on every pad: the low-frequency motor, the high-frequency motor, the left and the right
// impulse trigger, one second each. On a DualSense the trigger steps run in the LEONIDA_DS_TRIGGERS mode (rumble by
// default; set it to resistance to feel that one). With --force-feedback it then creates a constant force at
// magnitude 1.0 on motor 0 and holds it for a second, which a wheel feels as a steady pull; a pad the shim reports no
// force-feedback motor for is skipped with the reason shown. Everything goes through GameInputCreate from the shim
// registry, as a title would call it. No game data.
#include <windows.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gameinput.h"
#include "gameinput_abi.h"
#include "shims.h"

using namespace gi;

namespace {

std::vector<IGameInputDevice*> g_devices;

void CALLBACK on_device(GameInputCallbackToken, void*, IGameInputDevice* device, uint64_t, uint32_t current, uint32_t previous) {
    const GameInputDeviceInfo* i = device->GetDeviceInfo();
    std::printf("\n%s %s (%04x:%04x) motors=0x%x forcefeedback=%u haptic=%u\n",
                (current & GameInputDeviceConnected) ? "connected   " : "disconnected",
                i->displayName ? i->displayName->data : "?", i->vendorId, i->productId, i->supportedRumbleMotors,
                i->forceFeedbackMotorCount, i->hapticFeedbackMotorCount);
    // Raw HID (RT-175): the report tables read from the pad's HID report descriptor, or why there are none.
    if (current & GameInputDeviceConnected) {
        if (!(current & GameInputDeviceRawIoEnabled)) {
            std::printf("raw hid: no raw I/O (no HID device the backend could open, or no readable descriptor)\n");
        } else {
            const auto list = [](const char* kind, uint32_t n, const void* p) {
                const auto* r = static_cast<const GameInputRawDeviceReportInfo*>(p);
                std::printf("raw hid: %u %s report(s):", n, kind);
                for (uint32_t k = 0; k < n; ++k) std::printf(" id %u = %u bytes;", r[k].id, r[k].size);
                std::printf("\n");
            };
            list("input", i->inputReportCount, i->inputReportInfo);
            list("output", i->outputReportCount, i->outputReportInfo);
            list("feature", i->featureReportCount, i->featureReportInfo);
        }
    }
    if ((current & GameInputDeviceConnected) && !(previous & GameInputDeviceConnected)) {
        device->AddRef();
        g_devices.push_back(device);
    }
}

void play(const char* what, GameInputRumbleParams p) {
    std::printf("rumble: %s\n", what);
    for (IGameInputDevice* d : g_devices) d->SetRumbleState(&p);
    Sleep(1000);
    for (IGameInputDevice* d : g_devices) d->SetRumbleState(nullptr);
    Sleep(300);
}

// A constant force at full magnitude on the pad's first motor, held for a second: the effect a wheel feels as a
// steady pull. A pad SDL did not open as a haptic device reports no motor, and the step says so and moves on.
void constant_force() {
    // g_devices is in connection order, which is what output_state indexes; the SDL outcome is read back per pad.
    for (size_t n = 0; n < g_devices.size(); ++n) {
        IGameInputDevice* d = g_devices[n];
        const GameInputDeviceInfo* i = d->GetDeviceInfo();
        if (i->forceFeedbackMotorCount == 0) {
            std::printf("force feedback: %s reports no force-feedback motor; skipped\n",
                        i->displayName ? i->displayName->data : "?");
            continue;
        }
        const auto* motor = static_cast<const GameInputForceFeedbackMotorInfo*>(i->forceFeedbackMotorInfo);
        std::printf("force feedback: %u motor(s), motor 0 axes=0x%x constant=%d\n", i->forceFeedbackMotorCount,
                    motor ? static_cast<unsigned>(motor->supportedAxes) : 0u,
                    motor != nullptr && motor->isConstantEffectSupported);
        GameInputForceFeedbackParams p{};
        p.kind = GameInputForceFeedbackConstant;
        p.data.constant.magnitude.linearX = 1.0f;
        IGameInputForceFeedbackEffect* e = nullptr;
        const HRESULT hr = d->CreateForceFeedbackEffect(0, &p, reinterpret_cast<void**>(&e));
        if (FAILED(hr) || !e) {
            std::printf("force feedback: CreateForceFeedbackEffect failed (0x%08lx)\n", static_cast<unsigned long>(hr));
            continue;
        }
        std::printf("force feedback: constant linearX 1.0 on motor 0, 1 s\n");
        e->SetState(GameInputFeedbackRunning);
        Sleep(1000);  // the poll thread creates and runs the effect
        char seen[256] = "no output state";
        rt::gameinput::OutputState out{};
        if (rt::gameinput::output_state(static_cast<int>(n), &out)) {
            if (out.effect_id < 0) {
                std::snprintf(seen, sizeof seen, "SDL did not create the effect (%s)",
                              out.effect_error[0] ? out.effect_error : "no error text");
            } else if (!out.effect_ran) {
                std::snprintf(seen, sizeof seen, "SDL created effect %d but did not run it (%s)", out.effect_id,
                              out.effect_error[0] ? out.effect_error : "no error text");
            } else {
                std::snprintf(seen, sizeof seen, "SDL created effect %d and is running it", out.effect_id);
            }
        }
        std::printf("force feedback: %s\n", seen);
        e->SetState(GameInputFeedbackStopped);
        e->Release();  // the poll thread destroys the SDL effect
        Sleep(300);
    }
}

}  // namespace

int main(int argc, char** argv) {
    int seconds = 10;
    bool rumble = false, force_feedback = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--seconds") && i + 1 < argc) seconds = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--rumble")) rumble = true;
        else if (!std::strcmp(argv[i], "--force-feedback")) force_feedback = true;
        else {
            std::printf("usage: gameinput_probe [--seconds N] [--rumble] [--force-feedback]\n");
            return 2;
        }
    }
    auto create = reinterpret_cast<GameInputCreateFn>(rt::shims::find("gameinput.dll", "GameInputCreate"));
    IGameInput* input = nullptr;
    if (!create || create(&input) != S_OK) {
        std::printf("GameInputCreate failed\n");
        return 1;
    }
    Sleep(500);  // let the poll thread open what is connected
    GameInputCallbackToken token = 0;
    input->RegisterDeviceCallback(nullptr, GameInputKindGamepad, GameInputDeviceAnyStatus, GameInputBlockingEnumeration,
                                  nullptr, on_device, &token);
    if (g_devices.empty()) std::printf("no gamepad connected; plug one in (watching for %d s)\n", seconds);

    if (rumble && !g_devices.empty()) {
        const auto& c = rt::gameinput::config();
        std::printf("DualSense trigger mode: %s\n", c.ds_triggers == rt::gameinput::DsTriggerMode::Off ? "off"
                                                   : c.ds_triggers == rt::gameinput::DsTriggerMode::Resistance ? "resistance"
                                                                                                               : "rumble");
        play("low-frequency motor", {1, 0, 0, 0});
        play("high-frequency motor", {0, 1, 0, 0});
        play("left trigger", {0, 0, 1, 0});
        play("right trigger", {0, 0, 0, 1});
        play("left trigger at 30%", {0, 0, 0.3f, 0});
    }

    if (force_feedback && !g_devices.empty()) constant_force();

    const ULONGLONG end = GetTickCount64() + static_cast<ULONGLONG>(seconds) * 1000;
    while (GetTickCount64() < end) {
        IGameInputReading* r = nullptr;
        if (input->GetCurrentReading(GameInputKindGamepad, nullptr, &r) == S_OK) {
            GameInputGamepadState s{};
            GameInputUiNavigationState ui{};
            GameInputMotionState m{};
            const bool motion = r->GetMotionState(&m);
            r->GetGamepadState(&s);
            r->GetUiNavigationState(&ui);
            std::printf("\rbuttons=%04x ui=%06x LT=%.2f RT=%.2f L=(%+.2f,%+.2f) R=(%+.2f,%+.2f) seq=%llu   ", s.buttons,
                        ui.buttons, s.leftTrigger, s.rightTrigger, s.leftThumbstickX, s.leftThumbstickY,
                        s.rightThumbstickX, s.rightThumbstickY,
                        static_cast<unsigned long long>(r->GetSequenceNumber(GameInputKindGamepad)));
            if (motion)  // fractions of motionInfo's maxima: accel includes gravity, so a resting pad shows ~0.5 g
                std::printf("\n[a]=(%+.2f,%+.2f,%+.2f) [w]=(%+.2f,%+.2f,%+.2f)   ", m.accelerationX, m.accelerationY,
                            m.accelerationZ, m.angularVelocityX, m.angularVelocityY, m.angularVelocityZ);
            void* raw = nullptr;
            if (r->GetRawReport(&raw) && raw) {  // the newest raw input report: id and its first 16 bytes
                auto* rr = static_cast<IGameInputRawDeviceReport*>(raw);
                unsigned char b[16] = {};
                const size_t n = rr->GetRawData(sizeof b, b);
                std::printf("\nraw id %u (%zu bytes):", rr->GetReportInfo()->id, rr->GetRawDataSize());
                for (size_t k = 0; k < n; ++k) std::printf(" %02x", b[k]);
                std::printf("   ");
                rr->Release();
            }
            std::fflush(stdout);
            r->Release();
        }
        Sleep(16);
    }
    std::printf("\n");
    input->UnregisterCallback(token, 0);
    for (IGameInputDevice* d : g_devices) d->Release();
    input->Release();
    return 0;
}
