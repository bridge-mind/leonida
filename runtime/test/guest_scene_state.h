// The scene guest's session: a deterministic 600-frame run that the D3D12 guest fixture drives. Test only.
//
// For each frame the fixture calls step(), draws with bias() and x_offset(), reads its 64x64 RGBA32F target back and
// passes the floats to finish_frame(). No clock is read: the frame index is the only time. A run in one process and
// the same frames split over two processes therefore produce the same bytes.
//
// Environment:
//   Paths use the console GAME: namespace. The launcher assigns each run a fresh write overlay.
//   LEONIDA_SCENE_MODE  full    frames 0..599; writes checkpoint.bin after frame 299 and continues
//                       save    frames 0..299; writes checkpoint.bin and stops
//                       resume  frames 300..599 from checkpoint.bin (copied in by the orchestrator), fresh process
//   LEONIDA_KEYBOARD_PAD=always and LEONIDA_GAMEINPUT_THREAD=0 are required (checked, fail closed)
//   LEONIDA_SCENE_AUDIO=winmm enables synchronous host playback; the default writes PCM offline.
//
// Files. Binary layouts are arrays of little-endian uint32_t words. "checksum" is FNV-1a 32 over the words before it.
//   scene.bin       10 pages x 16 bytes; byte i of page p = (p*29 + i*7 + 11) & 255. Supplied by the runner and
//                   checked whole. Every 60 frames, page f/60 is read again with CreateFileW/SetFilePointerEx/ReadFile.
//   frames.bin      8 words per frame, in frame order: frame, digest low, digest high, pad buttons, x, y, scene byte,
//                   bias index. The digest is FNV-1a 64 over the readback's float bit patterns (LE). It is streamed
//                   into frames.bin.tmp and renamed by finish(); save + resume concatenated must equal full.
//   checkpoint.bin  the state words with magic 'LSCK'; state.bin holds the same words at the last frame, magic 'LSST':
//                   magic, version, word count, next frame, x, y, buttons, A rising edges, audio events, audio frame,
//                   page index, page bytes (4 words), scene byte, bias index, frame chain low, high, checksum.
//                   The chain is FNV-1a 64 over every frames.bin record so far. A loaded state must also match
//                   predict(): the trace decides everything except the chain.
//   event.wav       the pulse at the first A rising edge: 44.1 kHz mono 16-bit PCM, RIFF.
//   audio.bin       magic 'LSAU', version, word count, frame, sample rate, samples, FNV-1a 32 of the PCM bytes,
//                   playback (0 offline, 1 WinMM returned success, 3 unavailable or failed), error flag, checksum.
//   shot_NNN.ppm    binary PPM (P6) of the readback at frames 0, 137, 299, 300, 422 and 599 (the 600th frame).
// Each complete output is flushed to a temporary file before MoveFileExW publishes it in the overlay.
//
// Input comes the way a title reads it. GameInputCreate comes from gameinput.dll. A device callback picks out the
// runtime's keyboard pad by its display name, so a physical pad is never used. GetCurrentReading and GetGamepadState
// then run once per frame. Before each reading the guest calls SetTraceKeys(frame) from leonida_scene_input.dll
// (scene_input_driver.cpp), which presents trace_keys(frame) to the keyboard pad and polls the backend on this thread.
// The masked reading must equal expected_buttons(frame); position and events advance from the reading itself.
#pragma once
#include "guest_scene_trace.h"
#include <windows.h>
#include <mmsystem.h>
#include <cstdio>
#include "../shims/gameinput_abi.h"
#include <unknwn.h>

#include <cstring>
#include <string>
#include <vector>

namespace guestscene {
namespace detail {

using GiDevice = gi::IGameInputDevice;
using GiReading = gi::IGameInputReading;
using GiInput = gi::IGameInput;
using GiGamepadState = gi::GameInputGamepadState;
using GameInputCreateFn = gi::GameInputCreateFn;
constexpr auto kKindGamepad = gi::GameInputKindGamepad;
constexpr auto kStatusConnected = gi::GameInputDeviceConnected;
constexpr auto kBlockingEnumeration = gi::GameInputBlockingEnumeration;
using SetTraceKeysFn = int(WINAPI*)(uint32_t frame);

// Owners: each releases what it holds exactly once, and none can be copied.
class File {
public:
    File() = default;
    File(const File&) = delete;
    File& operator=(const File&) = delete;
    ~File() { reset(); }
    bool open(const std::wstring& path, DWORD access, DWORD disposition) {
        reset();
        h_ = CreateFileW(path.c_str(), access, access == GENERIC_READ ? FILE_SHARE_READ : 0, nullptr, disposition,
                         FILE_ATTRIBUTE_NORMAL, nullptr);
        return ok();
    }
    bool ok() const { return h_ != INVALID_HANDLE_VALUE; }
    void reset() {
        if (ok()) CloseHandle(h_);
        h_ = INVALID_HANDLE_VALUE;
    }
    bool size_is(uint64_t n) const {
        LARGE_INTEGER size{};
        return GetFileSizeEx(h_, &size) && uint64_t(size.QuadPart) == n;
    }
    bool read_at(uint64_t offset, void* out, uint32_t n) const {
        LARGE_INTEGER at{};
        at.QuadPart = LONGLONG(offset);
        DWORD got = 0;
        return SetFilePointerEx(h_, at, nullptr, FILE_BEGIN) && ReadFile(h_, out, n, &got, nullptr) && got == n;
    }
    bool write(const void* data, size_t n) const {
        const auto* p = static_cast<const uint8_t*>(data);
        while (n > 0) {
            const DWORD chunk = n > 0x40000000u ? 0x40000000u : DWORD(n);
            DWORD done = 0;
            if (!WriteFile(h_, p, chunk, &done, nullptr) || done != chunk) return false;
            p += chunk;
            n -= chunk;
        }
        return true;
    }
    bool close_flushed() {
        if (!ok()) return false;
        const bool flushed = FlushFileBuffers(h_) != FALSE;
        const bool closed = CloseHandle(h_) != FALSE;
        h_ = INVALID_HANDLE_VALUE;
        return flushed && closed;
    }

private:
    HANDLE h_ = INVALID_HANDLE_VALUE;
};

class Module {
public:
    Module() = default;
    Module(const Module&) = delete;
    Module& operator=(const Module&) = delete;
    ~Module() { reset(); }
    bool load(const char* name) {
        reset();
        m_ = LoadLibraryA(name);
        return m_ != nullptr;
    }
    template <class Fn>
    Fn proc(const char* name) const {
        return m_ ? reinterpret_cast<Fn>(reinterpret_cast<void*>(GetProcAddress(m_, name))) : nullptr;
    }
    void reset() {
        if (m_) FreeLibrary(m_);
        m_ = nullptr;
    }

private:
    HMODULE m_ = nullptr;
};

template <class T>
class Ref {
public:
    Ref() = default;
    Ref(const Ref&) = delete;
    Ref& operator=(const Ref&) = delete;
    ~Ref() { reset(); }
    T* get() const { return p_; }
    T* operator->() const { return p_; }
    T** put() {
        reset();
        return &p_;
    }
    void adopt(T* p) {  // takes over one reference the caller already holds
        reset();
        p_ = p;
    }
    void reset() {
        if (T* p = p_) {
            p_ = nullptr;
            p->Release();
        }
    }

private:
    T* p_ = nullptr;
};

constexpr uint64_t kFnv64Basis = 0xCBF29CE484222325ull;
constexpr uint32_t fourcc(char a, char b, char c, char d) {
    return uint32_t(uint8_t(a)) | uint32_t(uint8_t(b)) << 8 | uint32_t(uint8_t(c)) << 16 | uint32_t(uint8_t(d)) << 24;
}
inline void put16(uint8_t* p, uint16_t v) {
    p[0] = uint8_t(v);
    p[1] = uint8_t(v >> 8);
}
inline void put32(uint8_t* p, uint32_t v) {
    for (int i = 0; i < 4; ++i) p[i] = uint8_t(v >> (8 * i));
}
inline uint32_t get32(const uint8_t* p) {
    return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
}
inline uint32_t fnv32(const uint8_t* p, size_t n) {
    uint32_t h = 0x811C9DC5u;
    for (size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 0x01000193u;
    return h;
}
inline uint64_t fnv64(const uint8_t* p, size_t n, uint64_t h) {
    for (size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 0x100000001B3ull;
    return h;
}

inline std::string env_a(const char* name) {
    char buf[64];
    const DWORD n = GetEnvironmentVariableA(name, buf, sizeof buf);
    return n > 0 && n < sizeof buf ? std::string(buf, n) : std::string();
}
// The whole file, which must be exactly n bytes long.
inline bool read_whole(const std::wstring& path, void* out, uint32_t n) {
    File f;
    return f.open(path, GENERIC_READ, OPEN_EXISTING) && f.size_is(n) && f.read_at(0, out, n);
}
// Flush the temporary file before replacing the destination.
inline bool write_atomic(const std::wstring& path, const void* data, size_t n) {
    const std::wstring tmp = path + L".tmp";
    File f;
    const bool written = f.open(tmp, GENERIC_WRITE, CREATE_ALWAYS) && f.write(data, n) && f.close_flushed();
    f.reset();
    if (written && MoveFileExW(tmp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        return true;
    DeleteFileW(tmp.c_str());
    return false;
}

}  // namespace detail

class Session {
public:
    Session() = default;
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;
    ~Session() {
        release_input();
        if (frames_.ok()) {  // never finished: leave no frames.bin behind
            frames_.reset();
            DeleteFileW(path(L"frames.bin.tmp").c_str());
        }
    }

    bool begin() {
        if (phase_ != Phase::Idle) return fail("begin: called twice");
        if (!read_env()) return false;
        if (!prepare_scene()) return false;
        if (mode_ == Mode::Resume && !load_checkpoint()) return false;
        stop_ = mode_ == Mode::Save ? kSplit : kFrames;
        if (!frames_.open(path(L"frames.bin.tmp"), GENERIC_WRITE, CREATE_ALWAYS))
            return fail("frames.bin.tmp: cannot create");
        if (!open_input()) return false;
        phase_ = Phase::Ready;
        return true;
    }

    // Streams the page when one is due, applies this frame's trace, reads the pad and advances the state.
    bool step() {
        if (phase_ != Phase::Ready) return fail("step: not ready for a frame");
        const uint32_t f = st_.next_frame;
        if (f % kPageFrames == 0 && !stream_page(f / kPageFrames)) return false;
        if (!set_trace_(f)) return fail("SetTraceKeys refused the frame");
        if (pad_lost_) return fail("the keyboard pad disconnected");
        detail::Ref<detail::GiReading> reading;
        detail::GiGamepadState pad{};
        if (FAILED(input_->GetCurrentReading(detail::kKindGamepad, pad_.get(), reading.put())) || !reading.get() ||
            !reading->GetGamepadState(&pad))
            return fail("no gamepad reading from the keyboard pad");
        const uint32_t b = pad.buttons & kPadMask;
        if (b != expected_buttons(f)) return fail("the reading differs from the trace");
        st_.x = wrap_step(st_.x, (b & kPadRight) ? 1u : 0u, (b & kPadLeft) ? 1u : 0u);
        st_.y = wrap_step(st_.y, (b & kPadDown) ? 1u : 0u, (b & kPadUp) ? 1u : 0u);
        const bool a_edge = (b & kPadA) && !(st_.buttons & kPadA);
        st_.buttons = b;
        if (a_edge) {
            ++st_.a_edges;
            if (st_.audio_events == 0) {
                if (!emit_audio(f)) return false;
                st_.audio_events = 1;
                st_.audio_frame = f;
            }
        }
        st_.scene_byte = st_.page[scene_index(st_.x, st_.y)];
        st_.bias_index = bias_for(st_.scene_byte, b);
        phase_ = Phase::Stepped;
        return true;
    }

    // rgba: the frame's readback, extent x extent pixels of 4 floats, top row first.
    bool finish_frame(const float* rgba, unsigned extent) {
        if (phase_ != Phase::Stepped) return fail("finish_frame: no step for this frame");
        if (!rgba || extent != kExtent) return fail("finish_frame: the readback must be 64x64 RGBA floats");
        const uint32_t f = st_.next_frame;
        uint64_t digest = detail::kFnv64Basis;
        for (size_t i = 0; i < size_t(kExtent) * kExtent * 4; ++i) {
            uint32_t bits = 0;
            std::memcpy(&bits, rgba + i, 4);
            uint8_t le[4];
            detail::put32(le, bits);
            digest = detail::fnv64(le, 4, digest);
        }
        const uint32_t words[kRecordWords] = {f,     uint32_t(digest), uint32_t(digest >> 32), st_.buttons,
                                              st_.x, st_.y,            st_.scene_byte,         st_.bias_index};
        uint8_t record[kRecordWords * 4];
        for (uint32_t i = 0; i < kRecordWords; ++i) detail::put32(record + 4 * i, words[i]);
        if (!frames_.write(record, sizeof record)) return fail("frames.bin.tmp: write failed");
        st_.chain = detail::fnv64(record, sizeof record, st_.chain);
        for (uint32_t s : kShots)
            if (s == f && !write_shot(f, rgba)) return false;
        st_.next_frame = f + 1;
        if (st_.next_frame == kSplit && mode_ != Mode::Resume && !save_state(L"checkpoint.bin", kCheckpointMagic))
            return false;
        phase_ = st_.next_frame >= stop_ ? Phase::Done : Phase::Ready;
        return true;
    }

    // Commits frames.bin and writes state.bin. Only valid once done().
    bool finish() {
        if (phase_ != Phase::Done) return fail("finish: frames remain");
        if (!frames_.close_flushed() ||
            !MoveFileExW(path(L"frames.bin.tmp").c_str(), path(L"frames.bin").c_str(),
                         MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
            DeleteFileW(path(L"frames.bin.tmp").c_str());
            return fail("frames.bin: cannot commit");
        }
        if (!save_state(L"state.bin", kStateMagic)) return false;
        release_input();
        phase_ = Phase::Finished;
        return true;
    }

    bool done() const { return phase_ == Phase::Done || phase_ == Phase::Finished; }
    // RGB modulation for this frame's draw: 0, 0.125, 0.25 or 0.375.
    float bias() const { return 0.125f * float(st_.bias_index); }
    // Horizontal offset in NDC: x mod 8 whole pixels at 64x64 (2/64 per pixel), 0 to 0.21875; 0.125 is 4 pixels.
    float x_offset() const { return float(st_.x % 8u) * (2.0f / float(kExtent)); }
    uint32_t frame() const { return st_.next_frame; }
    const char* error() const { return error_ ? error_ : ""; }

private:
    enum class Mode { Full, Save, Resume };
    enum class Phase { Idle, Ready, Stepped, Done, Finished, Failed };
    struct State {
        uint32_t next_frame = 0, x = kStartX, y = kStartY, buttons = 0, a_edges = 0, audio_events = 0;
        uint32_t audio_frame = kNoFrame, page_index = kNoFrame, scene_byte = 0, bias_index = 0;
        uint8_t page[kPageBytes] = {};
        uint64_t chain = detail::kFnv64Basis;
    };
    enum : uint32_t { kAudioOffline = 0, kAudioPlayed = 1, kAudioError = 3 };

    static constexpr uint32_t kVersion = 1, kStateWords = 20, kStateBytes = kStateWords * 4, kRecordWords = 8;
    static constexpr uint32_t kCheckpointMagic = detail::fourcc('L', 'S', 'C', 'K');
    static constexpr uint32_t kStateMagic = detail::fourcc('L', 'S', 'S', 'T');
    static constexpr uint32_t kAudioMagic = detail::fourcc('L', 'S', 'A', 'U');
    static constexpr uint32_t kAudioRate = 44100, kAudioSamples = kAudioRate / 4, kAudioHalfPeriod = 50;
    static constexpr uint32_t kAudioRamp = 441, kAudioPeak = 9000;
    static constexpr uint32_t kShots[6] = {0, 137, 299, 300, 422, kFrames - 1};
    static constexpr int kWarmupPolls = 32;
    static constexpr const char* kPadName = "Keyboard pad (Leonida)";

    bool fail(const char* why) {
        if (!error_) {
            error_ = why;
            std::fprintf(stderr, "guest scene frame %u: %s (Win32 %lu)\n", st_.next_frame, why, GetLastError());
        }
        phase_ = Phase::Failed;
        return false;
    }
    std::wstring path(const std::wstring& name) const { return dir_ + L'\\' + name; }
    static std::wstring shot_name(uint32_t f) {
        wchar_t name[] = L"shot_000.ppm";
        name[5] = wchar_t(L'0' + f / 100 % 10);
        name[6] = wchar_t(L'0' + f / 10 % 10);
        name[7] = wchar_t(L'0' + f % 10);
        return name;
    }

    bool read_env() {
        dir_ = L"GAME:";
        const std::string m = detail::env_a("LEONIDA_SCENE_MODE");
        if (m == "full") mode_ = Mode::Full;
        else if (m == "save") mode_ = Mode::Save;
        else if (m == "resume") mode_ = Mode::Resume;
        else return fail("LEONIDA_SCENE_MODE must be full, save or resume");
        if (detail::env_a("LEONIDA_KEYBOARD_PAD") != "always" || detail::env_a("LEONIDA_GAMEINPUT_THREAD") != "0")
            return fail("needs LEONIDA_KEYBOARD_PAD=always and LEONIDA_GAMEINPUT_THREAD=0");
        play_audio_ = detail::env_a("LEONIDA_SCENE_AUDIO") == "winmm";
        return true;
    }

    bool prepare_scene() {
        uint8_t want[kPages * kPageBytes], got[kPages * kPageBytes];
        for (uint32_t p = 0; p < kPages; ++p)
            for (uint32_t i = 0; i < kPageBytes; ++i) want[p * kPageBytes + i] = scene_byte_at(p, i);
        const std::wstring file = path(L"scene.bin");
        if (!detail::read_whole(file, got, sizeof got) || std::memcmp(got, want, sizeof want) != 0)
            return fail("scene.bin: not the 10 expected pages");
        return true;
    }

    bool stream_page(uint32_t page) {
        detail::File f;
        uint8_t buf[kPageBytes];
        if (!f.open(path(L"scene.bin"), GENERIC_READ, OPEN_EXISTING) || !f.size_is(kPages * kPageBytes) ||
            !f.read_at(uint64_t(page) * kPageBytes, buf, kPageBytes))
            return fail("scene.bin: page read failed");
        for (uint32_t i = 0; i < kPageBytes; ++i)
            if (buf[i] != scene_byte_at(page, i)) return fail("scene.bin: page contents changed");
        std::memcpy(st_.page, buf, kPageBytes);
        st_.page_index = page;
        return true;
    }

    bool open_input() {
        if (!driver_.load("leonida_scene_input.dll") ||
            !(set_trace_ = driver_.proc<detail::SetTraceKeysFn>("SetTraceKeys")))
            return fail("leonida_scene_input.dll: SetTraceKeys not found");
        const auto create =
            gameinput_.load("gameinput.dll") ? gameinput_.proc<detail::GameInputCreateFn>("GameInputCreate") : nullptr;
        if (!create || FAILED(create(input_.put())) || !input_.get()) return fail("GameInputCreate failed");
        if (FAILED(input_->RegisterDeviceCallback(nullptr, detail::kKindGamepad, detail::kStatusConnected,
                                                  detail::kBlockingEnumeration, this, &Session::on_device, &token_)))
            return fail("RegisterDeviceCallback failed");
        // The keyboard pad attaches on the backend's first poll; its callback arrives with that poll or a later one.
        for (int i = 0; i < kWarmupPolls && !pad_.get(); ++i)
            if (!set_trace_(kNoFrame)) return fail("SetTraceKeys refused: the backend is not in trace mode");
        if (!pad_.get() || pad_lost_) return fail("the keyboard pad was not enumerated");
        return true;
    }

    static bool is_scene_pad(detail::GiDevice* device) {
        const auto* info = device->GetDeviceInfo();
        const auto* name = info ? info->displayName : nullptr;
        return name && name->data && std::strcmp(name->data, kPadName) == 0;
    }

    // Delivered on this thread from inside poll() (LEONIDA_GAMEINPUT_THREAD=0) or RegisterDeviceCallback.
    static void CALLBACK on_device(uint64_t, void* context, detail::GiDevice* device, uint64_t, uint32_t current,
                                   uint32_t) {
        auto* s = static_cast<Session*>(context);
        if (!s || !device) return;
        if (device == s->pad_.get()) {
            if (!(current & detail::kStatusConnected)) s->pad_lost_ = true;
            return;
        }
        if (s->pad_.get() || !(current & detail::kStatusConnected) || !is_scene_pad(device)) return;
        device->AddRef();
        s->pad_.adopt(device);
    }

    void release_input() {
        if (input_.get() && token_) input_->UnregisterCallback(token_, 1000000);
        token_ = 0;
        pad_.reset();
        input_.reset();
    }

    static void encode(const State& s, uint32_t magic, uint8_t* out) {
        uint32_t w[kStateWords] = {magic,          kVersion,       kStateWords,          s.next_frame,
                                   s.x,            s.y,            s.buttons,            s.a_edges,
                                   s.audio_events, s.audio_frame,  s.page_index,         0,
                                   0,              0,              0,                    s.scene_byte,
                                   s.bias_index,   uint32_t(s.chain), uint32_t(s.chain >> 32), 0};
        for (uint32_t i = 0; i < 4; ++i) w[11 + i] = detail::get32(s.page + 4 * i);
        for (uint32_t i = 0; i + 1 < kStateWords; ++i) detail::put32(out + 4 * i, w[i]);
        detail::put32(out + 4 * (kStateWords - 1), detail::fnv32(out, 4 * (kStateWords - 1)));
    }
    static bool decode(const uint8_t* in, uint32_t magic, State* s) {
        uint32_t w[kStateWords];
        for (uint32_t i = 0; i < kStateWords; ++i) w[i] = detail::get32(in + 4 * i);
        if (w[0] != magic || w[1] != kVersion || w[2] != kStateWords ||
            w[kStateWords - 1] != detail::fnv32(in, 4 * (kStateWords - 1)))
            return false;
        s->next_frame = w[3];
        s->x = w[4];
        s->y = w[5];
        s->buttons = w[6];
        s->a_edges = w[7];
        s->audio_events = w[8];
        s->audio_frame = w[9];
        s->page_index = w[10];
        for (uint32_t i = 0; i < 4; ++i) detail::put32(s->page + 4 * i, w[11 + i]);
        s->scene_byte = w[15];
        s->bias_index = w[16];
        s->chain = uint64_t(w[17]) | uint64_t(w[18]) << 32;
        return true;
    }
    // Everything but the chain follows from the trace, so a state that disagrees with predict() is corrupt.
    static bool consistent(const State& s) {
        if (s.next_frame > kFrames) return false;
        const Predicted p = predict(s.next_frame);
        if (s.x != p.x || s.y != p.y || s.buttons != p.buttons || s.a_edges != p.a_edges) return false;
        if (s.audio_events != (p.a_edges ? 1u : 0u) || s.audio_frame != (p.a_edges ? kAFrames[0] : kNoFrame))
            return false;
        if (s.next_frame == 0) return s.page_index == kNoFrame;
        if (s.page_index != (s.next_frame - 1) / kPageFrames) return false;
        for (uint32_t i = 0; i < kPageBytes; ++i)
            if (s.page[i] != scene_byte_at(s.page_index, i)) return false;
        return s.scene_byte == s.page[scene_index(s.x, s.y)] && s.bias_index == bias_for(s.scene_byte, s.buttons);
    }

    bool save_state(const wchar_t* name, uint32_t magic) {
        if (!consistent(st_)) return fail("the state does not match the trace");
        uint8_t bytes[kStateBytes];
        encode(st_, magic, bytes);
        if (!detail::write_atomic(path(name), bytes, sizeof bytes)) return fail("state write failed");
        return true;
    }
    bool load_checkpoint() {
        uint8_t bytes[kStateBytes];
        State s;
        if (!detail::read_whole(path(L"checkpoint.bin"), bytes, sizeof bytes))
            return fail("checkpoint.bin: missing, truncated or oversized");
        if (!decode(bytes, kCheckpointMagic, &s)) return fail("checkpoint.bin: bad header or checksum");
        if (s.next_frame != kSplit || !consistent(s)) return fail("checkpoint.bin: state does not match the trace");
        st_ = s;
        return true;
    }

    static uint8_t to_byte(float v) {
        if (!(v > 0.0f)) return 0;  // also NaN
        if (v >= 1.0f) return 255;
        return uint8_t(v * 255.0f + 0.5f);
    }
    bool write_shot(uint32_t f, const float* rgba) {
        static_assert(kExtent == 64, "the PPM header spells 64 64");
        static const char kHeader[] = "P6\n64 64\n255\n";
        std::vector<uint8_t> img(kHeader, kHeader + sizeof kHeader - 1);
        img.reserve(img.size() + kExtent * kExtent * 3);
        for (uint32_t i = 0; i < kExtent * kExtent; ++i)
            for (uint32_t c = 0; c < 3; ++c) img.push_back(to_byte(rgba[4 * i + c]));
        if (!detail::write_atomic(path(shot_name(f)), img.data(), img.size())) return fail("shot_NNN.ppm: write failed");
        return true;
    }

    // event.wav (always, when the edge comes) and audio.bin, which says whether a device played it.
    bool emit_audio(uint32_t f) {
        const uint32_t data_bytes = kAudioSamples * 2;
        std::vector<uint8_t> wav(44 + data_bytes);
        uint8_t* h = wav.data();
        std::memcpy(h, "RIFF", 4);
        detail::put32(h + 4, 36 + data_bytes);
        std::memcpy(h + 8, "WAVEfmt ", 8);
        detail::put32(h + 16, 16);
        detail::put16(h + 20, 1);  // PCM
        detail::put16(h + 22, 1);  // mono
        detail::put32(h + 24, kAudioRate);
        detail::put32(h + 28, kAudioRate * 2);
        detail::put16(h + 32, 2);
        detail::put16(h + 34, 16);
        std::memcpy(h + 36, "data", 4);
        detail::put32(h + 40, data_bytes);
        for (uint32_t i = 0; i < kAudioSamples; ++i) {  // 441 Hz square wave, 10 ms linear ramps at both ends
            uint32_t edge = i < kAudioSamples - 1 - i ? i : kAudioSamples - 1 - i;
            if (edge > kAudioRamp) edge = kAudioRamp;
            const int32_t amp = int32_t(kAudioPeak * edge / kAudioRamp);
            detail::put16(h + 44 + 2 * i, uint16_t(int16_t(((i / kAudioHalfPeriod) & 1u) ? -amp : amp)));
        }
        if (!detail::write_atomic(path(L"event.wav"), wav.data(), wav.size())) return fail("event.wav: write failed");
        uint32_t mm = 0;
        const uint32_t playback = play(wav.data(), &mm);
        const uint32_t w[9] = {kAudioMagic,   kVersion, 10, f, kAudioRate, kAudioSamples,
                               detail::fnv32(h + 44, data_bytes), playback, mm};
        uint8_t out[10 * 4];
        for (uint32_t i = 0; i < 9; ++i) detail::put32(out + 4 * i, w[i]);
        detail::put32(out + 36, detail::fnv32(out, 36));
        if (!detail::write_atomic(path(L"audio.bin"), out, sizeof out)) return fail("audio.bin: write failed");
        return true;
    }

    // Optional host playback of the exact WAV that was written. SND_SYNC keeps the
    // backing bytes alive until WinMM returns; the runner bounds the process duration.
    uint32_t play(const uint8_t* wav, uint32_t* mm) const {
        if (!play_audio_) return kAudioOffline;
        detail::Module winmm;
        if (!winmm.load("winmm.dll")) return kAudioError;
        const auto play_sound = winmm.proc<decltype(&PlaySoundW)>("PlaySoundW");
        if (!play_sound) return kAudioError;
        if (play_sound(reinterpret_cast<LPCWSTR>(wav), nullptr, SND_MEMORY | SND_SYNC | SND_NODEFAULT))
            return kAudioPlayed;
        *mm = MMSYSERR_ERROR;
        return kAudioError;
    }

    // Declaration order is release order reversed: the modules outlive the interfaces they created.
    detail::Module driver_, gameinput_;
    detail::SetTraceKeysFn set_trace_ = nullptr;
    detail::Ref<detail::GiInput> input_;
    detail::Ref<detail::GiDevice> pad_;
    detail::File frames_;
    uint64_t token_ = 0;
    bool pad_lost_ = false, play_audio_ = false;
    std::wstring dir_;
    Mode mode_ = Mode::Full;
    Phase phase_ = Phase::Idle;
    uint32_t stop_ = kFrames;
    State st_;
    const char* error_ = nullptr;
};

}  // namespace guestscene
