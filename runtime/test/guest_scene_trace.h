// Synthetic keyboard-pad trace for the mapped scene and the fixture input source.
#pragma once
#include <cstdint>

namespace guestscene {

constexpr uint32_t kFrames = 600;
constexpr uint32_t kSplit = 300;  // checkpoint.bin holds the state after frames 0..299
constexpr uint32_t kPageFrames = 60;
constexpr uint32_t kPages = kFrames / kPageFrames;
constexpr uint32_t kPageBytes = 16;
constexpr uint32_t kExtent = 64;  // readback width and height
constexpr uint32_t kGrid = 64;    // the position wraps on a 64x64 grid
constexpr uint32_t kStartX = 8, kStartY = 8;
constexpr uint32_t kNoFrame = 0xFFFFFFFFu;
constexpr uint32_t kAFrames[2] = {137, 422};  // the only frames with A down
static_assert(kPages == 10, "scene.bin has exactly 10 pages");

// Trace keys, and the virtual keys the keyboard pad maps them from (arrows to the d-pad, Enter to A).
constexpr uint32_t kKeyLeft = 1, kKeyUp = 2, kKeyRight = 4, kKeyDown = 8, kKeyA = 16;
constexpr int kVkReturn = 0x0D, kVkLeft = 0x25, kVkUp = 0x26, kVkRight = 0x27, kVkDown = 0x28;
// The GameInputGamepadButtons bits the session reads; the rest of the reading is ignored.
constexpr uint32_t kPadA = 0x4, kPadUp = 0x40, kPadDown = 0x80, kPadLeft = 0x100, kPadRight = 0x200;
constexpr uint32_t kPadMask = kPadA | kPadUp | kPadDown | kPadLeft | kPadRight;

constexpr uint8_t scene_byte_at(uint32_t page, uint32_t i) { return uint8_t((page * 29u + i * 7u + 11u) & 255u); }

// The keys held on `frame`: one direction set per 50-frame segment, plus A on kAFrames. 0 outside 0..599.
constexpr uint32_t trace_keys(uint32_t frame) {
    constexpr uint32_t segments[kFrames / 50] = {
        kKeyRight, kKeyRight | kKeyDown, kKeyDown, 0,        kKeyLeft, kKeyLeft | kKeyUp,
        kKeyUp,    kKeyRight,            0,        kKeyDown, kKeyLeft, kKeyRight | kKeyUp,
    };
    if (frame >= kFrames) return 0;
    return segments[frame / 50] | ((frame == kAFrames[0] || frame == kAFrames[1]) ? kKeyA : 0u);
}

constexpr bool trace_key_down(uint32_t frame, int virtual_key) {
    const uint32_t k = trace_keys(frame);
    switch (virtual_key) {
    case kVkLeft: return (k & kKeyLeft) != 0;
    case kVkUp: return (k & kKeyUp) != 0;
    case kVkRight: return (k & kKeyRight) != 0;
    case kVkDown: return (k & kKeyDown) != 0;
    case kVkReturn: return (k & kKeyA) != 0;
    default: return false;
    }
}

constexpr uint32_t expected_buttons(uint32_t frame) {
    const uint32_t k = trace_keys(frame);
    return ((k & kKeyLeft) ? kPadLeft : 0u) | ((k & kKeyUp) ? kPadUp : 0u) | ((k & kKeyRight) ? kPadRight : 0u) |
           ((k & kKeyDown) ? kPadDown : 0u) | ((k & kKeyA) ? kPadA : 0u);
}

constexpr uint32_t wrap_step(uint32_t v, uint32_t plus, uint32_t minus) { return (v + kGrid + plus - minus) % kGrid; }
constexpr uint32_t scene_index(uint32_t x, uint32_t y) { return (x + 3 * y) % kPageBytes; }
constexpr uint32_t bias_for(uint32_t scene_byte, uint32_t buttons) {
    return ((scene_byte >> 5) + ((buttons & ~kPadA) ? 1u : 0u) + ((buttons & kPadA) ? 2u : 0u)) & 3u;
}

// Where the trace leaves the pad after frames 0..frames-1.
struct Predicted {
    uint32_t x, y, buttons, a_edges;
};
constexpr Predicted predict(uint32_t frames) {
    Predicted p{kStartX, kStartY, 0, 0};
    for (uint32_t f = 0; f < frames && f < kFrames; ++f) {
        const uint32_t b = expected_buttons(f);
        p.x = wrap_step(p.x, (b & kPadRight) ? 1u : 0u, (b & kPadLeft) ? 1u : 0u);
        p.y = wrap_step(p.y, (b & kPadDown) ? 1u : 0u, (b & kPadUp) ? 1u : 0u);
        if ((b & kPadA) && !(p.buttons & kPadA)) ++p.a_edges;
        p.buttons = b;
    }
    return p;
}
static_assert(predict(kSplit).a_edges == 1 && predict(kFrames).a_edges == 2, "one A rising edge per half");

}  // namespace guestscene
