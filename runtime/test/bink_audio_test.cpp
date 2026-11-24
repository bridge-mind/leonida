// Optional Windows test: load Bink2Scarlett.dll through rt::Loader and inspect/extract the audio tracks of one
// Bink 2 movie. Skips unless both LEONIDA_GAME_DIR and LEONIDA_MOVIE_AUDIO are set.
//
//   LEONIDA_GAME_DIR=<package root>       (contains Bink2Scarlett.dll)
//   LEONIDA_MOVIE_AUDIO=<file.bk2>        (a carved Bink 2 stream with audio)
//   LEONIDA_AUDIO_FRAMES=<n>              (optional: stop after n video frames; default all)
//
// Output stays under %TEMP%\leonida_bink_audio. No movie, packet or PCM is committed.
#include <windows.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../shims/shims.h"
#include "../src/pe_loader.h"

namespace {

int g_failures = 0;
const char* g_step = "start";

LONG CALLBACK log_exception(EXCEPTION_POINTERS* info) {
    std::printf("VEH exception code=0x%08lx address=%p step=%s\n", info->ExceptionRecord->ExceptionCode,
                info->ExceptionRecord->ExceptionAddress, g_step);
    return EXCEPTION_CONTINUE_SEARCH;
}

void check(bool condition, const std::string& what) {
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what.c_str());
    if (!condition) ++g_failures;
}

[[noreturn]] void fatal(const std::string& what) {
    std::fprintf(stderr, "FAIL %s\n", what.c_str());
    std::exit(1);
}

std::string env(const char* name) {
    const char* v = std::getenv(name);
    return v ? v : "";
}

uint32_t read_u32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

uint16_t read_u16(const uint8_t* p) { return (uint16_t)(p[0] | (p[1] << 8)); }

struct MovieHeader {
    uint32_t stored = 0;
    uint32_t frames = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t fps_num = 0;
    uint32_t fps_den = 0;
    uint32_t audio_tracks = 0;
    std::vector<uint32_t> track_ids;
    std::vector<uint32_t> max_sizes;
    std::vector<uint16_t> rates;
    std::vector<uint16_t> flags;
};

MovieHeader read_header(const std::filesystem::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) fatal("cannot open movie " + path.string());
    uint8_t h[512]{};
    f.read(reinterpret_cast<char*>(h), sizeof(h));
    size_t got = (size_t)f.gcount();
    if (got < 0x2C) fatal("movie is shorter than the 44-byte Bink 2 header");
    if (h[0] != 'K' || h[1] != 'B' || h[2] != '2' || h[3] < 'a' || h[3] > 'z')
        fatal("movie does not start with KB2[a-z]");
    MovieHeader out;
    out.stored = read_u32(h + 0x04) + 8;
    out.frames = read_u32(h + 0x08);
    out.width = read_u32(h + 0x14);
    out.height = read_u32(h + 0x18);
    out.fps_num = read_u32(h + 0x1C);
    out.fps_den = read_u32(h + 0x20);
    out.audio_tracks = read_u32(h + 0x28);
    std::error_code ec;
    auto size = std::filesystem::file_size(path, ec);
    if (ec || out.stored > size) fatal("movie header's stored size is past EOF");
    if (!out.frames || !out.width || !out.height || !out.fps_num || !out.fps_den)
        fatal("movie header has a zero frame count, size or fps");
    if (out.frames > 1'000'000 || out.width > 7680 || out.height > 4800)
        fatal("movie header dimensions or frame count are out of range");
    if (out.audio_tracks > 16) fatal("movie header has an implausible audio track count");
    if (out.audio_tracks) {
        // Revisions i/j/k/n put one opaque word at 0x2C before the audio metadata.
        size_t p = 0x30;
        if (got < p + 12 * out.audio_tracks) fatal("movie header is too short for its audio descriptors");
        for (uint32_t i = 0; i < out.audio_tracks; ++i)
            out.max_sizes.push_back(read_u32(h + p + 4 * i));
        for (uint32_t i = 0; i < out.audio_tracks; ++i) {
            out.rates.push_back(read_u16(h + p + 4 * out.audio_tracks + 4 * i));
            out.flags.push_back(read_u16(h + p + 4 * out.audio_tracks + 4 * i + 2));
        }
        for (uint32_t i = 0; i < out.audio_tracks; ++i)
            out.track_ids.push_back(read_u32(h + p + 8 * out.audio_tracks + 4 * i));
    }
    return out;
}

struct BinkTrack {
    uint32_t frequency;   // +0  measured 48000
    uint32_t channels;    // +4  measured 2
    uint32_t max_size;    // +8  measured 154624
    uint32_t unknown12;   // +12
    void* bink;           // +16
    uintptr_t sndcomp;    // +24
    int32_t track_index;  // +32
};
static_assert(sizeof(BinkTrack) == 40, "observed BINKTRACK x64 prefix");

struct BinkAudioApi {
    using Open = void* (*)(const char*, uint32_t);
    using Close = void (*)(void*);
    using Wait = uint32_t (*)(void*);
    using DoFrame = uint32_t (*)(void*);
    using CopyToBuffer = uint32_t (*)(void*, void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
    using NextFrame = void (*)(void*);
    using Service = void (*)(void*);
    using SetSoundTrack = void (*)(uint32_t, uint32_t*);
    using OpenTrack = BinkTrack* (*)(void*, uint32_t);
    using CloseTrack = void (*)(BinkTrack*);
    using GetTrackType = uint32_t (*)(void*, uint32_t);
    using GetTrackMaxSize = uint32_t (*)(void*, uint32_t);
    using GetTrackID = uint32_t (*)(void*, uint32_t);
    using GetTrackData = uint32_t (*)(BinkTrack*, void*);

    Open open = nullptr;
    Close close = nullptr;
    Wait wait = nullptr;
    DoFrame do_frame = nullptr;
    CopyToBuffer copy = nullptr;
    NextFrame next = nullptr;
    Service service = nullptr;
    SetSoundTrack set_sound_track = nullptr;
    OpenTrack open_track = nullptr;
    CloseTrack close_track = nullptr;
    GetTrackType type = nullptr;
    GetTrackMaxSize max_size = nullptr;
    GetTrackID id = nullptr;
    GetTrackData data = nullptr;
};

void* symbol(rt::Module& m, const char* name) {
    auto it = m.exports.find(name);
    if (it == m.exports.end()) fatal(std::string("Bink2Scarlett.dll does not export ") + name);
    return it->second;
}

void* iat_slot(rt::Module& m, const std::string& want_dll, const std::string& want_name) {
    auto* dos = reinterpret_cast<IMAGE_DOS_HEADER*>(m.base);
    auto* nt = reinterpret_cast<IMAGE_NT_HEADERS64*>(m.base + dos->e_lfanew);
    auto& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    auto* desc = reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR*>(m.base + dir.VirtualAddress);
    for (; desc->Name; ++desc) {
        const char* dll = reinterpret_cast<const char*>(m.base + desc->Name);
        if (rt::lower(dll) != rt::lower(want_dll)) continue;
        auto* lookup = reinterpret_cast<IMAGE_THUNK_DATA64*>(
            m.base + (desc->OriginalFirstThunk ? desc->OriginalFirstThunk : desc->FirstThunk));
        auto* iat = reinterpret_cast<IMAGE_THUNK_DATA64*>(m.base + desc->FirstThunk);
        for (; lookup->u1.AddressOfData; ++lookup, ++iat) {
            if (IMAGE_SNAP_BY_ORDINAL64(lookup->u1.Ordinal)) continue;
            auto* name = reinterpret_cast<IMAGE_IMPORT_BY_NAME*>(m.base + (uint32_t)lookup->u1.AddressOfData);
            if (want_name == name->Name) return &iat->u1.Function;
        }
    }
    return nullptr;
}

void write_u16(std::ofstream& f, uint16_t v) {
    char b[2] = {(char)(v & 0xFF), (char)(v >> 8)};
    f.write(b, 2);
}
void write_u32(std::ofstream& f, uint32_t v) {
    char b[4] = {(char)(v & 0xFF), (char)((v >> 8) & 0xFF), (char)((v >> 16) & 0xFF), (char)(v >> 24)};
    f.write(b, 4);
}

void write_wav(const std::filesystem::path& path, const std::vector<uint8_t>& pcm, uint32_t rate, uint32_t bits,
               uint32_t channels) {
    const uint32_t block = channels * (bits / 8);
    const uint32_t byte_rate = rate * block;
    const uint32_t data_size = (uint32_t)pcm.size();
    std::ofstream f(path, std::ios::binary);
    if (!f) fatal("cannot write " + path.string());
    f.write("RIFF", 4);
    write_u32(f, 36 + data_size);
    f.write("WAVEfmt ", 8);
    write_u32(f, 16);
    write_u16(f, 1);  // PCM; Bink's 16-bit path is integer PCM.
    write_u16(f, (uint16_t)channels);
    write_u32(f, rate);
    write_u32(f, byte_rate);
    write_u16(f, (uint16_t)block);
    write_u16(f, (uint16_t)bits);
    f.write("data", 4);
    write_u32(f, data_size);
    f.write(reinterpret_cast<const char*>(pcm.data()), (std::streamsize)pcm.size());
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    AddVectoredExceptionHandler(1, log_exception);
    const std::string game_dir_arg = env("LEONIDA_GAME_DIR");
    const std::string movie_arg = env("LEONIDA_MOVIE_AUDIO");
    if (game_dir_arg.empty() || movie_arg.empty()) {
        std::printf("SKIP (no package/audio movie)\n");
        return 77;
    }

    const std::filesystem::path game_dir = game_dir_arg;
    const std::filesystem::path movie = movie_arg;
    std::error_code ec;
    if (!std::filesystem::is_directory(game_dir, ec)) fatal("LEONIDA_GAME_DIR is not a directory");
    if (!std::filesystem::is_regular_file(game_dir / "Bink2Scarlett.dll", ec))
        fatal("LEONIDA_GAME_DIR does not contain Bink2Scarlett.dll");
    if (!std::filesystem::is_regular_file(movie, ec)) fatal("LEONIDA_MOVIE_AUDIO is not a file");

    const MovieHeader header = read_header(movie);
    std::printf("movie: %s  %ux%u  %u frames  %u audio tracks\n", movie.string().c_str(), header.width,
                header.height, header.frames, header.audio_tracks);
    for (uint32_t i = 0; i < header.audio_tracks; ++i)
        std::printf("  header track %u: id=%u rate=%u flags=0x%04x\n", i, header.track_ids[i], header.rates[i],
                    header.flags[i]);
    if (!header.audio_tracks) {
        std::printf("SKIP (movie has no audio tracks)\n");
        return 77;  // nothing was checked: a ctest SKIP (SKIP_RETURN_CODE 77), not a pass
    }

    try {
        rt::Loader loader(game_dir);
        rt::Module& bink = loader.load(game_dir / "Bink2Scarlett.dll");
        std::printf("mapped %s at %p\n", bink.name.c_str(), bink.base);

        int shim = 0, host = 0, module = 0, unimplemented = 0;
        bool xmem_alloc = false, xmem_free = false, non_audio_unimplemented = false;
        for (auto& r : bink.imports) {
            switch (r.binding) {
                case rt::Binding::Shim: ++shim; break;
                case rt::Binding::Host: ++host; break;
                case rt::Binding::Module: ++module; break;
                default: ++unimplemented; break;
            }
            if (rt::lower(r.dll) == "xmem.dll") {
                if (r.name == "XMemAlloc") xmem_alloc = true;
                if (r.name == "XMemFree") xmem_free = true;
                check(r.binding == rt::Binding::Shim, "xmem.dll!" + r.name + " binds to the shim");
                void* expected = rt::shims::find("xmem.dll", r.name);
                void* slot = iat_slot(bink, r.dll, r.name);
                check(expected && slot && *static_cast<void**>(slot) == expected,
                      "xmem.dll!" + r.name + " IAT points at rt::shims::find");
            }
            if (r.binding == rt::Binding::Unimplemented) {
                std::printf("  UNIMPLEMENTED %s!%s\n", r.dll.c_str(), r.name.c_str());
                if (rt::lower(r.dll).find("xaudio") == std::string::npos) non_audio_unimplemented = true;
            }
        }
        std::printf("imports: shim=%d host=%d module=%d unimplemented=%d\n", shim, host, module, unimplemented);
        check(xmem_alloc && xmem_free, "both XMemAlloc and XMemFree imports are present");
        check(!non_audio_unimplemented, "no non-audio unimplemented imports");

        BinkAudioApi api;
        api.open = reinterpret_cast<BinkAudioApi::Open>(symbol(bink, "BinkOpen"));
        api.close = reinterpret_cast<BinkAudioApi::Close>(symbol(bink, "BinkClose"));
        api.wait = reinterpret_cast<BinkAudioApi::Wait>(symbol(bink, "BinkWait"));
        api.do_frame = reinterpret_cast<BinkAudioApi::DoFrame>(symbol(bink, "BinkDoFrame"));
        api.copy = reinterpret_cast<BinkAudioApi::CopyToBuffer>(symbol(bink, "BinkCopyToBuffer"));
        api.next = reinterpret_cast<BinkAudioApi::NextFrame>(symbol(bink, "BinkNextFrame"));
        api.service = reinterpret_cast<BinkAudioApi::Service>(symbol(bink, "BinkService"));
        api.set_sound_track = reinterpret_cast<BinkAudioApi::SetSoundTrack>(symbol(bink, "BinkSetSoundTrack"));
        api.open_track = reinterpret_cast<BinkAudioApi::OpenTrack>(symbol(bink, "BinkOpenTrack"));
        api.close_track = reinterpret_cast<BinkAudioApi::CloseTrack>(symbol(bink, "BinkCloseTrack"));
        api.type = reinterpret_cast<BinkAudioApi::GetTrackType>(symbol(bink, "BinkGetTrackType"));
        api.max_size = reinterpret_cast<BinkAudioApi::GetTrackMaxSize>(symbol(bink, "BinkGetTrackMaxSize"));
        api.id = reinterpret_cast<BinkAudioApi::GetTrackID>(symbol(bink, "BinkGetTrackID"));
        api.data = reinterpret_cast<BinkAudioApi::GetTrackData>(symbol(bink, "BinkGetTrackData"));
        std::printf("addresses: DoFrame=%p Copy=%p Wait=%p Next=%p GetTrackData=%p OpenTrack=%p\n",
                    (void*)api.do_frame, (void*)api.copy, (void*)api.wait, (void*)api.next, (void*)api.data,
                    (void*)api.open_track);

        void* handle = nullptr;
        if (env("LEONIDA_RUN_DLLMAIN") == "1") {
            auto dll_main = reinterpret_cast<BOOL(WINAPI*)(HMODULE, DWORD, LPVOID)>(bink.base + bink.entry_rva);
            check(dll_main(reinterpret_cast<HMODULE>(bink.base), DLL_PROCESS_ATTACH, nullptr) != FALSE,
                  "DllMain(DLL_PROCESS_ATTACH) returned TRUE");
        }
        uint32_t selected = 0;
        if (!env("LEONIDA_AUDIO_TRACK").empty()) {
            char* end = nullptr;
            unsigned long n = std::strtoul(env("LEONIDA_AUDIO_TRACK").c_str(), &end, 10);
            if (!end || *end || n >= header.audio_tracks) fatal("LEONIDA_AUDIO_TRACK is not a track index");
            selected = (uint32_t)n;
        }
        if (!header.track_ids.empty()) {
            auto ids = header.track_ids;
            api.set_sound_track(1, &ids[selected]);
            std::printf("BinkSetSoundTrack(1, [%u])\n", ids[selected]);
        }
        handle = api.open(movie.string().c_str(), 0);
        if (!handle) fatal("BinkOpen returned null");
        std::printf("PASS BinkOpen returned %p\n", handle);

        struct Track {
            uint32_t index;
            uint32_t id;
            uint32_t type;
            BinkTrack* handle;
            std::vector<uint8_t> pcm;
            uint32_t calls;
            uint32_t bytes;
        };
        std::vector<Track> tracks;
        for (uint32_t i = 0; i < header.audio_tracks; ++i) {
            Track t{};
            t.index = i;
            t.id = api.id(handle, i);
            t.type = api.type(handle, i);
            const uint32_t max_size = api.max_size(handle, i);
            const uint32_t expected_type = ((uint32_t)header.flags[i] << 16) | header.rates[i];
            check(t.id == header.track_ids[i], "BinkGetTrackID matches the header track ID");
            check(t.type == expected_type, "BinkGetTrackType is (flags << 16) | sample rate");
            check(max_size >= header.max_sizes[i], "BinkGetTrackMaxSize covers the header max size");
            std::printf("track index %u: id=%u type=0x%08x max_size=%u (header %u)", i, t.id, t.type, max_size,
                        header.max_sizes[i]);
            if (i != selected) {
                std::printf(" (not opened; LEONIDA_AUDIO_TRACK=%u)\n", selected);
                continue;
            }
            t.handle = api.open_track(handle, i);
            std::printf(" open=%p", (void*)t.handle);
            if (t.handle) {
                check(t.handle->frequency == header.rates[i], "BINKTRACK.Frequency matches the header rate");
                check(t.handle->channels == ((header.flags[i] & 0x2000) ? 2u : 1u),
                      "BINKTRACK.Channels matches the header stereo flag");
                check(t.handle->track_index == (int32_t)i, "BINKTRACK.trackindex matches the requested index");
                std::printf("  BINKTRACK={freq=%u channels=%u max=%u trackindex=%d}", t.handle->frequency,
                            t.handle->channels, t.handle->max_size, t.handle->track_index);
                const uint8_t* raw = reinterpret_cast<const uint8_t*>(t.handle);
                std::printf("\n  track bytes:");
                for (int k = 0; k < 64; ++k) std::printf(" %02x", raw[k]);
            }
            std::printf("\n");
            check(t.handle != nullptr, "BinkOpenTrack returned a track");
            if (t.handle) tracks.push_back(std::move(t));
        }

        if (tracks.empty()) {
            api.close(handle);
            return 1;
        }

        uint32_t limit = header.frames;
        if (!env("LEONIDA_AUDIO_FRAMES").empty()) {
            char* end = nullptr;
            unsigned long n = std::strtoul(env("LEONIDA_AUDIO_FRAMES").c_str(), &end, 10);
            if (!end || *end || !n) fatal("LEONIDA_AUDIO_FRAMES is not a positive frame count");
            limit = (uint32_t)std::min<unsigned long>(n, header.frames);
        }

        std::vector<uint8_t> packet(16 << 20);
        std::vector<uint8_t> video((size_t)header.width * header.height * 4);
        uint32_t frame_sizes_printed = 0;
        const bool trace = env("LEONIDA_AUDIO_TRACE") == "1";
        const bool skip_sleep = env("LEONIDA_AUDIO_NO_SLEEP") == "1";
        for (uint32_t frame = 0; frame < limit; ++frame) {
            if (trace) std::printf("trace: frame %u before BinkDoFrame\n", frame);
            g_step = "BinkDoFrame";
            const uint32_t do_rc = api.do_frame(handle);
            uint32_t wait = 0;
            g_step = "BinkWait(after BinkDoFrame)";
            for (int spin = 0; spin < 1000 && (wait = api.wait(handle)) != 0; ++spin) Sleep(1);
            g_step = "BinkCopyToBuffer";
            const uint32_t copy_rc = api.copy(handle, video.data(), header.width * 4, header.height, 0, 0, 3);
            uint32_t wait2 = 0;
            g_step = "BinkWait(after BinkCopyToBuffer)";
            for (int spin = 0; spin < 1000 && (wait2 = api.wait(handle)) != 0; ++spin) Sleep(1);
            g_step = "BinkService";
            api.service(handle);
            // The package DLL's audio decode is asynchronous. Without this delay BinkGetTrackData intermittently
            // faults inside Bink2Scarlett.dll; BinkWait and BinkService both return before the audio path is ready.
            // 1 ms is the smallest delay that made every tested stream stable. This is a property of the DLL, not
            // the loader or the xmem shim.
            if (!skip_sleep) Sleep(1);
            if (trace)
                std::printf("trace: frame %u after BinkDoFrame=%u wait=%u Copy=%u wait2=%u\n", frame, do_rc, wait,
                            copy_rc, wait2);
            for (auto& t : tracks) {
                g_step = "BinkGetTrackData";
                const uint32_t n = api.data(t.handle, packet.data());
                if (trace) std::printf("trace: frame %u track %u after BinkGetTrackData=%u\n", frame, t.index, n);
                if (n > packet.size()) fatal("BinkGetTrackData returned a size larger than the packet buffer");
                ++t.calls;
                t.bytes += n;
                if (n) t.pcm.insert(t.pcm.end(), packet.begin(), packet.begin() + n);
                if (frame < 3 || (n && frame_sizes_printed < 12)) {
                    std::printf("frame %u track %u: BinkDoFrame=%u BinkGetTrackData=%u", frame, t.index, do_rc, n);
                    if (n) {
                        std::printf(" head=");
                        for (uint32_t k = 0; k < 16 && k < n; ++k) std::printf("%02x", packet[k]);
                    }
                    std::printf("\n");
                    ++frame_sizes_printed;
                }
            }
            if ((frame % 50) == 0) std::printf("progress: frame %u/%u\n", frame, limit);
            if (frame + 1 < limit) {
                g_step = "BinkNextFrame";
                api.next(handle);
                if (trace) std::printf("trace: frame %u after BinkNextFrame\n", frame);
            }
        }

        const std::filesystem::path out_dir = std::filesystem::temp_directory_path() / "leonida_bink_audio";
        std::filesystem::create_directories(out_dir, ec);
        const std::string stem = movie.stem().string();
        for (auto& t : tracks) {
            std::printf("track %u: calls=%u bytes=%u", t.index, t.calls, t.bytes);
            if (t.handle) {
                std::printf(" freq=%u channels=%u max=%u", t.handle->frequency, t.handle->channels,
                            t.handle->max_size);
            }
            std::printf("\n");
            if (!t.pcm.empty()) {
                const std::string base = stem + "_track" + std::to_string(t.index);
                const auto raw = out_dir / (base + ".pcm");
                {
                    std::ofstream f(raw, std::ios::binary);
                    f.write(reinterpret_cast<const char*>(t.pcm.data()), (std::streamsize)t.pcm.size());
                }
                const uint32_t rate = header.rates[t.index];
                const uint32_t channels = (header.flags[t.index] & 0x2000) ? 2 : 1;
                const uint32_t bits = 16;  // measured: BinkGetTrackData returns 16-bit PCM for both flag variants
                const double seconds = t.pcm.size() / (double)(rate * channels * (bits / 8));
                check(t.pcm.size() % (channels * (bits / 8)) == 0, "PCM byte count is a whole number of samples");
                const auto wav = out_dir / (base + ".wav");
                write_wav(wav, t.pcm, rate, bits, channels);
                std::printf("  wrote %s and %s  %u Hz, %u ch, %u-bit, %.3f s\n", raw.string().c_str(),
                            wav.string().c_str(), rate, channels, bits, seconds);
            }
            if (t.handle) api.close_track(t.handle);
        }

        api.close(handle);
        std::printf("%s\n", g_failures ? "RESULT FAIL" : "RESULT PASS");
        return g_failures ? 1 : 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
}
