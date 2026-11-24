// Optional Windows test: measure the Bink2Scarlett ABI the runtime needs, over every stream of the package's
// movies.rpf, through the package's own DLL mapped by rt::Loader.
//
//   LEONIDA_GAME_DIR=<package root>   (contains Bink2Scarlett.dll and movies.rpf)
//   LEONIDA_BINK_INDEX=<file.csv>     (tools/bink_index.py <package>/movies.rpf --out <file.csv>)
//   LEONIDA_BINK_LIMIT=<n>            (optional: only the first n streams of the index)
//
// Each stream is opened inside the archive with BinkSetFileOffset + BinkOpen(BINKFILEOFFSET), so nothing is
// extracted. For every stream the test reads the start of the HBINK and the BINKSUMMARY that BinkGetSummary writes,
// and keeps the word offsets whose value equals the stream header's (width, height, frame count, frame rate); an
// offset survives only if it matches on every stream. It also tallies the return values of BinkWait, BinkDoFrame and
// BinkCopyToBuffer. It prints offsets, counts and return values only: no frame, pixel or audio data.
//
// With either variable unset it prints "SKIP (no package/index)" and exits 77 (CTest skip).
#include <windows.h>

#include <algorithm>
#include <bitset>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "../shims/shims.h"
#include "../src/pe_loader.h"

namespace {

int g_failures = 0;

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

struct Stream {
    uint64_t offset = 0;
    uint32_t frames = 0, width = 0, height = 0, fps_num = 0, fps_den = 0, audio_tracks = 0;
};

// The stream header at `offset` (the same fields bink_decode_test reads from a carved file).
Stream read_stream(std::ifstream& f, uint64_t offset) {
    uint8_t h[0x2C]{};
    f.clear();
    f.seekg((std::streamoff)offset);
    f.read(reinterpret_cast<char*>(h), sizeof(h));
    if (f.gcount() != (std::streamsize)sizeof(h)) fatal("index offset is past the end of movies.rpf");
    if (h[0] != 'K' || h[1] != 'B' || h[2] != '2') fatal("index offset does not start a KB2 stream");
    Stream s;
    s.offset = offset;
    s.frames = read_u32(h + 0x08);
    s.width = read_u32(h + 0x14);
    s.height = read_u32(h + 0x18);
    s.fps_num = read_u32(h + 0x1C);
    s.fps_den = read_u32(h + 0x20);
    s.audio_tracks = read_u32(h + 0x28);
    if (!s.frames || !s.width || !s.height || !s.fps_num || !s.fps_den) fatal("stream header has a zero field");
    if (s.width > 16384 || s.height > 16384) fatal("stream header has an implausible frame size");
    return s;
}

// The "offset" column of tools/bink_index.py's CSV.
std::vector<uint64_t> read_index(const std::filesystem::path& path) {
    std::ifstream f(path);
    if (!f) fatal("cannot open LEONIDA_BINK_INDEX");
    std::string line;
    if (!std::getline(f, line)) fatal("LEONIDA_BINK_INDEX is empty");
    std::vector<std::string> head;
    for (std::stringstream ss(line); std::getline(ss, line, ',');) head.push_back(line);
    const auto col = std::find(head.begin(), head.end(), "offset");
    if (col == head.end()) fatal("LEONIDA_BINK_INDEX has no offset column");
    const size_t at = (size_t)(col - head.begin());
    std::vector<uint64_t> out;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::vector<std::string> cells;
        for (std::stringstream ss(line); std::getline(ss, line, ',');) cells.push_back(line);
        if (cells.size() <= at) fatal("LEONIDA_BINK_INDEX row has no offset");
        out.push_back(std::strtoull(cells[at].c_str(), nullptr, 10));
    }
    return out;
}

constexpr size_t kWords = 64;  // the first 256 bytes of each structure
using Words = std::vector<uint32_t>;
using Mask = std::bitset<kWords>;

// Copy up to kWords words from `p`; stops at the first unreadable word. No C++ objects here: SEH only.
size_t copy_words(const void* p, uint32_t* out) {
    size_t n = 0;
    __try {
        for (; n < kWords; ++n) out[n] = static_cast<const volatile uint32_t*>(p)[n];
    } __except (EXCEPTION_EXECUTE_HANDLER) {
    }
    return n;
}

Words snapshot(const void* p) {
    Words w(kWords, 0);
    w.resize(copy_words(p, w.data()));
    return w;
}

Mask equal_to(const Words& w, uint32_t value) {
    Mask m;
    for (size_t i = 0; i < w.size(); ++i) m[i] = w[i] == value;
    return m;
}

std::string offsets(const Mask& m) {
    std::string s;
    for (size_t i = 0; i < kWords; ++i)
        if (m[i]) {
            char b[16];
            std::snprintf(b, sizeof(b), "%s0x%02zx", s.empty() ? "" : " ", i * 4);
            s += b;
        }
    return s.empty() ? "(none)" : s;
}

std::string hex(size_t v) {
    char b[24];
    std::snprintf(b, sizeof(b), "%zx", v);
    return b;
}

// Bytes of a sentinel-filled buffer that a call replaced: the highest one is a lower bound on the struct size.
size_t written_extent(const std::vector<uint8_t>& buf, uint8_t sentinel) {
    for (size_t i = buf.size(); i-- > 0;)
        if (buf[i] != sentinel) return i + 1;
    return 0;
}

struct BinkApi {
    void* (*open)(const char*, uint32_t) = nullptr;
    uint32_t (*wait)(void*) = nullptr;
    uint32_t (*do_frame)(void*) = nullptr;
    uint32_t (*copy)(void*, void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) = nullptr;
    void (*next)(void*) = nullptr;
    void (*go_to)(void*, uint32_t, uint32_t) = nullptr;
    void (*close)(void*) = nullptr;
    void (*summary)(void*, void*) = nullptr;
    const char* (*get_error)() = nullptr;
    void (*set_file_offset)(uint64_t) = nullptr;
    void (*set_frame_rate)(uint32_t, uint32_t) = nullptr;  // applies to the next BinkOpen with kFrameRate
};

template <class F> void bind(rt::Module& m, const char* name, F& out) {
    auto it = m.exports.find(name);
    if (it == m.exports.end()) fatal(std::string("Bink2Scarlett.dll does not export ") + name);
    out = reinterpret_cast<F>(it->second);
}

constexpr uint32_t kCopyBgra32 = 3;     // the BinkCopyToBuffer format bink_decode_test measured as 32-bit BGRA
constexpr uint32_t kFileOffset = 0x20;  // BinkOpen flag: start at the offset BinkSetFileOffset gave
// BinkOpen flag: use the rate BinkSetFrameRate gave. Measured: of the 32 single-bit flags only 0x1000 changes the
// rate fields (RAD's Bink 1 header has BINKFRAMERATE at 0x00080000; that bit changes nothing here).
constexpr uint32_t kFrameRate = 0x1000;
constexpr size_t kSummaryBytes = 1024;
constexpr size_t kSummaryWritten = 144;  // BinkGetSummary writes bytes 0..143, every one, on all 515 streams

// One field, matched on every stream: the word offsets where the structure held the stream's value each time.
// hbink_at/summary_at are the measured offsets (docs/movies.md section 5); -1 when the structure has no such field.
struct Field {
    const char* name;
    int hbink_at, summary_at;
    Mask hbink, summary;
    uint32_t streams = 0;
};

bool has(const Mask& m, int byte_offset) { return byte_offset >= 0 && m[(size_t)byte_offset / 4]; }

}  // namespace

int main() {
    const std::string game_dir_arg = env("LEONIDA_GAME_DIR");
    const std::string index_arg = env("LEONIDA_BINK_INDEX");
    if (game_dir_arg.empty() || index_arg.empty()) {
        std::printf("SKIP (no package/index)\n");
        return 77;
    }
    const std::filesystem::path game_dir = game_dir_arg;
    const std::filesystem::path archive = game_dir / "movies.rpf";
    std::error_code ec;
    if (!std::filesystem::is_regular_file(game_dir / "Bink2Scarlett.dll", ec))
        fatal("LEONIDA_GAME_DIR does not contain Bink2Scarlett.dll");
    if (!std::filesystem::is_regular_file(archive, ec)) fatal("LEONIDA_GAME_DIR does not contain movies.rpf");

    std::vector<uint64_t> index = read_index(index_arg);
    if (!env("LEONIDA_BINK_LIMIT").empty())
        index.resize(std::min<size_t>(index.size(), std::strtoul(env("LEONIDA_BINK_LIMIT").c_str(), nullptr, 10)));
    if (index.empty()) fatal("LEONIDA_BINK_INDEX lists no stream");
    std::ifstream rpf(archive, std::ios::binary);
    if (!rpf) fatal("cannot open movies.rpf");

    try {
        rt::Loader loader(game_dir);
        rt::Module& dll = loader.load(game_dir / "Bink2Scarlett.dll");
        for (auto& r : dll.imports)
            if (r.binding == rt::Binding::Unimplemented && rt::lower(r.dll).find("xaudio") == std::string::npos)
                fatal("Bink2Scarlett.dll import " + r.dll + "!" + r.name + " is unimplemented");
        BinkApi api;
        bind(dll, "BinkOpen", api.open);
        bind(dll, "BinkWait", api.wait);
        bind(dll, "BinkDoFrame", api.do_frame);
        bind(dll, "BinkCopyToBuffer", api.copy);
        bind(dll, "BinkNextFrame", api.next);
        bind(dll, "BinkGoto", api.go_to);
        bind(dll, "BinkClose", api.close);
        bind(dll, "BinkGetSummary", api.summary);
        bind(dll, "BinkGetError", api.get_error);
        bind(dll, "BinkSetFileOffset", api.set_file_offset);
        bind(dll, "BinkSetFrameRate", api.set_frame_rate);

        // fps_num/fps_den in BINKSUMMARY: the playback rate (0x3c/0x40); the file's rate is at 0x34/0x38.
        Field fields[] = {{"width", 0x00, 0x28},   {"height", 0x04, 0x2c},  {"frames", 0x08, 0x48},
                          {"fps_num", 0x14, 0x3c}, {"fps_den", 0x18, 0x40}, {"audio_tracks", 0x38, -1}};
        for (Field& f : fields) f.hbink.set(), f.summary.set();
        // The current frame number: 1 after BinkOpen, 2 after BinkNextFrame, N after BinkGoto(N) (1-based).
        Mask frame_num;
        frame_num.set();
        uint32_t goto_streams = 0;
        std::map<std::string, uint32_t> returns;
        size_t hbink_min = kWords * 4, summary_min = kSummaryBytes, summary_max = 0;
        uint32_t opened = 0, sentinel_replaced = 0, summary_gaps = 0;  // streams with an unwritten byte below extent

        std::vector<uint8_t> frame, sum(kSummaryBytes);
        for (uint64_t offset : index) {
            const Stream s = read_stream(rpf, offset);
            api.set_file_offset(s.offset);
            void* bink = api.open(archive.string().c_str(), kFileOffset);
            if (!bink) {
                const char* e = api.get_error ? api.get_error() : nullptr;
                std::printf("FAIL BinkOpen at %llu: %s\n", (unsigned long long)s.offset, e && *e ? e : "(no text)");
                ++g_failures;
                continue;
            }
            ++opened;
            const Words open_words = snapshot(bink);
            hbink_min = std::min(hbink_min, open_words.size() * 4);

            const uint32_t values[] = {s.width, s.height, s.frames, s.fps_num, s.fps_den, s.audio_tracks};
            for (size_t i = 0; i < std::size(fields); ++i) {
                fields[i].hbink &= equal_to(open_words, values[i]);
                ++fields[i].streams;
            }
            Mask fn = equal_to(open_words, 1);

            // Two sentinels: a byte the call writes differs from at least one of them (a value can't equal both), so
            // a byte that kept the sentinel in both calls was not written. That gives the written bytes exactly,
            // not only the highest one; it says nothing about padding the structure may have past them.
            size_t extent = 0;
            std::vector<bool> written(kSummaryBytes, false);
            Words sum_words;
            for (uint8_t sentinel : {(uint8_t)0xCD, (uint8_t)0x5A}) {
                std::fill(sum.begin(), sum.end(), sentinel);
                api.summary(bink, sum.data());
                extent = std::max(extent, written_extent(sum, sentinel));
                for (size_t i = 0; i < kSummaryBytes; ++i) written[i] = written[i] || sum[i] != sentinel;
                sum_words.assign(kWords, 0);
                std::memcpy(sum_words.data(), sum.data(), kWords * 4);
            }
            summary_min = std::min(summary_min, extent);
            summary_max = std::max(summary_max, extent);
            if (std::count(written.begin(), written.begin() + extent, false)) ++summary_gaps;
            for (size_t i = 0; i < std::size(fields); ++i) fields[i].summary &= equal_to(sum_words, values[i]);

            frame.assign((size_t)s.width * 4 * s.height, 0xCD);
            auto decode = [&](const char* step) {
                const uint32_t w = api.wait(bink);
                const uint32_t d = api.do_frame(bink);
                const uint32_t c = api.copy(bink, frame.data(), s.width * 4, s.height, 0, 0, kCopyBgra32);
                ++returns[std::string(step) + ": BinkWait=" + std::to_string(w) + " BinkDoFrame=" +
                          std::to_string(d) + " BinkCopyToBuffer=" + std::to_string(c)];
            };
            decode("frame 1");
            if (std::count(frame.begin(), frame.end(), (uint8_t)0xCD) < (std::ptrdiff_t)frame.size())
                ++sentinel_replaced;
            api.next(bink);
            fn &= equal_to(snapshot(bink), 2);
            decode("frame 2");
            if (s.frames >= 4) {
                const uint32_t target = s.frames / 2 + 1;
                api.go_to(bink, target, 0);
                fn &= equal_to(snapshot(bink), target);
                decode("after BinkGoto");
                ++goto_streams;
            }
            frame_num &= fn;
            api.close(bink);
        }

        // Every stream of 1.0.2.1 runs at 30/1 fps, so the header alone can't tell the rate fields apart. Reopen
        // the first stream with a forced 24000/1001 playback rate: the fields that follow it are the playback rate,
        // the ones that keep 30/1 are the file's.
        Mask forced_num, forced_den, file_num, file_den, sum_forced_num, sum_forced_den, sum_file_num, sum_file_den;
        {
            const Stream s = read_stream(rpf, index.front());
            api.set_frame_rate(24000, 1001);
            api.set_file_offset(s.offset);
            void* bink = api.open(archive.string().c_str(), kFileOffset | kFrameRate);
            check(bink != nullptr, "BinkOpen with the frame-rate flag after BinkSetFrameRate(24000, 1001)");
            if (bink) {
                const Words w = snapshot(bink);
                forced_num = equal_to(w, 24000);
                forced_den = equal_to(w, 1001);
                file_num = equal_to(w, s.fps_num);
                file_den = equal_to(w, s.fps_den);
                std::fill(sum.begin(), sum.end(), (uint8_t)0xCD);
                api.summary(bink, sum.data());
                Words sw(kWords);
                std::memcpy(sw.data(), sum.data(), kWords * 4);
                sum_forced_num = equal_to(sw, 24000);
                sum_forced_den = equal_to(sw, 1001);
                sum_file_num = equal_to(sw, s.fps_num);
                sum_file_den = equal_to(sw, s.fps_den);
                api.close(bink);
            }
        }

        std::printf("streams: %zu in the index, %u opened, %u with BinkGoto checked\n", index.size(), opened,
                    goto_streams);
        std::printf("HBINK readable prefix: at least %zu bytes on every stream\n", hbink_min);
        std::printf("BINKSUMMARY: highest written byte +1 is %zu..%zu (of a %zu-byte buffer); %u streams leave a "
                    "byte below it unwritten\n", summary_min, summary_max, kSummaryBytes, summary_gaps);
        for (const Field& f : fields)
            std::printf("  %-12s HBINK %-24s BINKSUMMARY %s\n", f.name, offsets(f.hbink).c_str(),
                        offsets(f.summary).c_str());
        std::printf("  %-12s HBINK %s\n", "frame number", offsets(frame_num).c_str());
        std::printf("forced 24000/1001 on the first stream:\n");
        std::printf("  playback num HBINK %-24s BINKSUMMARY %s\n", offsets(forced_num).c_str(),
                    offsets(sum_forced_num).c_str());
        std::printf("  playback den HBINK %-24s BINKSUMMARY %s\n", offsets(forced_den).c_str(),
                    offsets(sum_forced_den).c_str());
        std::printf("  file num     HBINK %-24s BINKSUMMARY %s\n", offsets(file_num).c_str(),
                    offsets(sum_file_num).c_str());
        std::printf("  file den     HBINK %-24s BINKSUMMARY %s\n", offsets(file_den).c_str(),
                    offsets(sum_file_den).c_str());
        for (const auto& [what, n] : returns) std::printf("returns %s: %u streams\n", what.c_str(), n);

        check(opened == index.size(), "BinkOpen inside movies.rpf succeeds on every indexed stream");
        check(sentinel_replaced == opened, "BinkCopyToBuffer replaced the sentinel on every stream");
        for (const Field& f : fields) {
            char at[64];
            std::snprintf(at, sizeof(at), " is at HBINK +0x%02x", f.hbink_at);
            check(has(f.hbink, f.hbink_at), std::string(f.name) + at);
            if (f.summary_at >= 0) {
                std::snprintf(at, sizeof(at), " is at BINKSUMMARY +0x%02x", f.summary_at);
                check(has(f.summary, f.summary_at), std::string(f.name) + at);
            }
        }
        check(has(frame_num, 0x0c), "the 1-based current frame number is at HBINK +0x0c");
        check(summary_min == kSummaryWritten && summary_max == kSummaryWritten && summary_gaps == 0,
              "BinkGetSummary writes every byte of +0x00..+0x" + hex(kSummaryWritten - 1) + " and none after");
        check(has(forced_num, 0x14) && has(forced_den, 0x18), "BinkSetFrameRate's rate lands at HBINK +0x14/+0x18");
        check(has(sum_forced_num, 0x3c) && has(sum_forced_den, 0x40) && has(sum_file_num, 0x34) &&
                  has(sum_file_den, 0x38),
              "BINKSUMMARY keeps the file's rate at +0x34/+0x38 and the playback rate at +0x3c/+0x40");
        bool zero_returns = true;
        for (const auto& [what, n] : returns)
            zero_returns = zero_returns && what.find("BinkDoFrame=0 BinkCopyToBuffer=0") != std::string::npos;
        check(zero_returns, "BinkDoFrame and BinkCopyToBuffer return 0 on every call");
        std::printf("%s\n", g_failures ? "RESULT FAIL" : "RESULT PASS");
        return g_failures ? 1 : 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
}
