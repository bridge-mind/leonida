#pragma once

// Which kind of input device the player used last: an Xbox-style pad, a PlayStation pad, a Nintendo pad or the
// keyboard pad (docs/gameinput.md). The GameInput shim reports every pad to the title as an Xbox One pad, so the title
// itself cannot tell; this is the port's own record, for whatever wants to follow the device in use (button prompt
// glyphs first). No SDL, no Windows headers: the shim feeds it one sample per device per poll, and tests feed it
// synthetic samples with their own times.
//
// The rules, chosen so that only deliberate input changes the answer:
//   * Only an edge counts: a button going down, a trigger rising past `trigger`, a stick leaving the `stick` radius.
//     A button held down, a trigger held, a stick held over, stick drift that stays inside the radius and analog
//     noise never count, so a pad lying on the desk cannot take over. The keyboard pad's right stick follows the
//     mouse, so a real mouse movement counts as keyboard input through the same stick rule.
//   * A device's first sample is its baseline and counts for nothing: a pad that connects with a button held or a
//     worn stick resting past the radius does not switch anything until it produces a new edge.
//   * After a change, another change waits until `min_switch_us` has passed. An edge from another family inside that
//     window is ignored (not queued), so two devices used at once cannot make the answer flicker.
//   * Edges from a device of the family already active change nothing.
//   * Before any edge the answer is `initial` (Xbox unless the caller says otherwise).
//   * A forced family (LEONIDA_PROMPTS) wins over all of this; the tracking still runs underneath, so clearing the
//     force goes straight back to the device in use.
// `generation` counts changes of the answer (tracked or forced), so a reader can tell that something changed without
// comparing families.

#include <cmath>
#include <cstdint>
#include <optional>
#include <unordered_map>

namespace leonida {

enum class InputFamily : std::uint8_t {
    Xbox,          // Xbox One/Series, Xbox 360, and any pad SDL does not know better (generic, unknown)
    PlayStation5,  // DualSense, DualSense Edge
    PlayStation4,  // DualShock 4
    PlayStation3,  // DualShock 3
    Nintendo,      // Switch Pro controller and Joy-Cons
    Keyboard,      // the keyboard pad: keyboard and mouse
};

inline const char* input_family_name(InputFamily f) {
    switch (f) {
    case InputFamily::Xbox: return "Xbox";
    case InputFamily::PlayStation5: return "PlayStation 5";
    case InputFamily::PlayStation4: return "PlayStation 4";
    case InputFamily::PlayStation3: return "PlayStation 3";
    case InputFamily::Nintendo: return "Nintendo";
    case InputFamily::Keyboard: return "keyboard and mouse";
    }
    return "unknown";
}

// One device's state at one poll, in GameInput's units: sticks -1..1, triggers 0..1. `buttons` is any bit set per
// pressed button (the shim passes the GameInput gamepad bits plus one bit for the guide button); only its changes
// matter.
struct InputSample {
    std::uint32_t buttons = 0;
    float left_trigger = 0, right_trigger = 0;
    float left_x = 0, left_y = 0, right_x = 0, right_y = 0;
};

// What made the answer change, for the log line.
enum class InputEdge : std::uint8_t { None, Button, Trigger, Stick };

inline const char* input_edge_name(InputEdge e) {
    switch (e) {
    case InputEdge::None: return "none";
    case InputEdge::Button: return "button press";
    case InputEdge::Trigger: return "trigger pull";
    case InputEdge::Stick: return "stick movement";
    }
    return "unknown";
}

class ActiveInputTracker {
public:
    struct Thresholds {
        float stick = 0.35f;                  // radius a stick has to leave
        float trigger = 0.3f;                 // a trigger has to rise past this
        std::uint64_t min_switch_us = 300'000;  // shortest time between two changes
    };

    explicit ActiveInputTracker(InputFamily initial = InputFamily::Xbox) : tracked_(initial) {}
    ActiveInputTracker(InputFamily initial, Thresholds t) : thresholds_(t), tracked_(initial) {}

    // One sample of one device (`device` is any key that stays the same while the device is connected). Returns the
    // edge that changed the tracked family, or None when it did not change. With a family forced, the tracked family
    // still follows the devices, but active() does not move.
    InputEdge observe(std::uint64_t device, InputFamily family, const InputSample& now, std::uint64_t time_us) {
        auto it = devices_.find(device);
        if (it == devices_.end()) {
            devices_.emplace(device, now);  // the baseline: counts for nothing
            return InputEdge::None;
        }
        const InputEdge edge = edge_between(it->second, now);
        it->second = now;
        if (edge == InputEdge::None || family == tracked_) return InputEdge::None;
        if (changed_once_ && time_us - changed_at_ < thresholds_.min_switch_us) return InputEdge::None;
        const InputFamily before = active();
        tracked_ = family;
        changed_at_ = time_us;
        changed_once_ = true;
        if (active() != before) ++generation_;
        return edge;
    }

    // The device disconnected: its next sample, if it comes back, is a baseline again.
    void forget(std::uint64_t device) { devices_.erase(device); }

    // Forces the answer (a family) or returns to tracking (nullopt). Changing what active() returns bumps generation.
    void force(std::optional<InputFamily> family) {
        const InputFamily before = active();
        forced_ = family;
        if (active() != before) ++generation_;
    }

    InputFamily active() const { return forced_ ? *forced_ : tracked_; }
    InputFamily tracked() const { return tracked_; }
    bool forced() const { return forced_.has_value(); }
    std::uint64_t generation() const { return generation_; }
    const Thresholds& thresholds() const { return thresholds_; }
    void set_thresholds(const Thresholds& t) { thresholds_ = t; }

private:
    InputEdge edge_between(const InputSample& was, const InputSample& now) const {
        if (now.buttons & ~was.buttons) return InputEdge::Button;
        const float t = thresholds_.trigger;
        if ((was.left_trigger <= t && now.left_trigger > t) || (was.right_trigger <= t && now.right_trigger > t))
            return InputEdge::Trigger;
        const auto out = [this](float x, float y) { return std::hypot(x, y) > thresholds_.stick; };
        if ((!out(was.left_x, was.left_y) && out(now.left_x, now.left_y)) ||
            (!out(was.right_x, was.right_y) && out(now.right_x, now.right_y)))
            return InputEdge::Stick;
        return InputEdge::None;
    }

    Thresholds thresholds_{};
    InputFamily tracked_;
    std::optional<InputFamily> forced_;
    std::uint64_t generation_ = 0;
    std::uint64_t changed_at_ = 0;
    bool changed_once_ = false;
    std::unordered_map<std::uint64_t, InputSample> devices_;
};

}  // namespace leonida
