// bink_play: play a Bink 2 stream in a window through the package's own Bink2Scarlett.dll, mapped by rt::Loader (the
// same path bink_decode_test checks). A local viewing tool for movies carved from your own copy; nothing it reads or
// shows is written anywhere.
//
//   bink_play [NAME | movie.bk2 | archive.rpf@OFFSET] [--game-dir DIR] [--frames N] [--headless] [--loop]
//
// NAME is a movie's path in the game (Stinger/RSG_STINGER.bk2) or any unique part of it (stinger), looked up in the
// name table tools/bink_play.py keeps (%LOCALAPPDATA%\Leonida\bink_names.csv); if there is none yet, bink_play runs
// that tool once to build it. With no movie at all a windowed run plays the Rockstar stinger, so double-clicking
// the executable plays it. The game folder comes from --game-dir, LEONIDA_GAME_DIR, or the archive the name table
// was built from.
//
// A movie inside an archive plays where it is: `movies.rpf@OFFSET` hands BinkOpen the archive with
// BinkSetFileOffset, so nothing is extracted (tools/bink_play.py finds the offset from the movie's name). A relative
// archive path is taken from the game folder.
//
// --game-dir defaults to LEONIDA_GAME_DIR, the movie to LEONIDA_MOVIE. Without a package or a movie it prints
// "SKIP (no package/movie)" and exits 77 (CTest skip). --headless decodes without a window (as fast as it can) and
// prints the decode rate, so CTest can run it; --frames stops after N frames. Each headless decode starts from a
// frame filled with 0xA5 and fails if it leaves the frame untouched, so a pass means every frame wrote the buffer.
//
// Keys: Space pause/resume, Left/Right one second back/forward, Home restart, L loop on/off, Esc or closing the
// window quits. The frame is scaled to the window, keeping its aspect ratio.
#include <windows.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "../shims/shims.h"
#include "../src/pe_loader.h"

namespace {

// A windowed run is often started by double-click, where a console line vanishes with its window: say it in a box.
bool g_windowed = false;

[[noreturn]] void fatal(const std::string& what) {
    std::fprintf(stderr, "FAIL %s\n", what.c_str());
    if (g_windowed) MessageBoxA(nullptr, what.c_str(), "bink_play", MB_OK | MB_ICONERROR);
    std::exit(1);
}

std::string env(const char* name) {
    const char* v = std::getenv(name);
    return v ? v : "";
}

uint32_t read_u32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

struct MovieHeader {
    uint32_t frames = 0, width = 0, height = 0, fps_num = 0, fps_den = 0;
};

// The fixed Bink 2 header (docs/movies.md): frame count at 0x08, size at 0x14/0x18, frame rate at 0x1C/0x20.
MovieHeader read_movie_header(const std::filesystem::path& path, uint64_t offset) {
    std::ifstream f(path, std::ios::binary);
    if (!f) fatal("cannot open movie " + path.string());
    f.seekg((std::streamoff)offset);
    uint8_t h[0x2C]{};
    f.read(reinterpret_cast<char*>(h), sizeof(h));
    if (f.gcount() != (std::streamsize)sizeof(h)) fatal("movie is shorter than the 44-byte Bink 2 header");
    if (h[0] != 'K' || h[1] != 'B' || h[2] != '2' || h[3] < 'a' || h[3] > 'z')
        fatal("movie does not start with KB2[a-z]");
    MovieHeader out;
    out.frames = read_u32(h + 0x08);
    out.width = read_u32(h + 0x14);
    out.height = read_u32(h + 0x18);
    out.fps_num = read_u32(h + 0x1C);
    out.fps_den = read_u32(h + 0x20);
    if (!out.frames || !out.width || !out.height || !out.fps_num || !out.fps_den)
        fatal("movie header has a zero frame count, size or fps");
    if (out.width > 16384 || out.height > 16384) fatal("movie header has an implausible frame size");
    return out;
}

struct BinkApi {
    void* (*open)(const char*, uint32_t) = nullptr;
    uint32_t (*do_frame)(void*) = nullptr;
    uint32_t (*copy)(void*, void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) = nullptr;
    void (*next)(void*) = nullptr;
    void (*go_to)(void*, uint32_t, uint32_t) = nullptr;
    void (*close)(void*) = nullptr;
    const char* (*get_error)() = nullptr;
    void (*set_file_offset)(uint64_t) = nullptr;  // applies to the next BinkOpen
};

template <class F> void bind(rt::Module& m, const char* name, F& out) {
    auto it = m.exports.find(name);
    if (it == m.exports.end()) fatal(std::string("Bink2Scarlett.dll does not export ") + name);
    out = reinterpret_cast<F>(it->second);
}

constexpr uint32_t kCopyBgra32 = 3;  // the BinkCopyToBuffer format bink_decode_test measured as 32-bit BGRA
constexpr uint32_t kFileOffset = 0x20;  // BinkOpen flag: start at the offset BinkSetFileOffset gave

struct Player {
    BinkApi api;
    void* bink = nullptr;
    MovieHeader header;
    std::vector<uint8_t> frame;
    uint32_t index = 0;   // the frame in `frame`
    bool paused = false;
    bool loop = false;
    bool quit = false;
    bool seek_pending = false;
    uint32_t seek_to = 0;

    void decode() {
        // Headless, the frame is filled with a sentinel first, and a decode that leaves every byte of it fails: a
        // pass then means each decoded frame wrote the buffer (not that its pixels are right).
        constexpr uint8_t kSentinel = 0xa5;
        if (!g_windowed) std::fill(frame.begin(), frame.end(), kSentinel);
        api.do_frame(bink);
        api.copy(bink, frame.data(), header.width * 4, header.height, 0, 0, kCopyBgra32);
        if (!g_windowed && std::all_of(frame.begin(), frame.end(), [](uint8_t v) { return v == kSentinel; }))
            fatal("headless decode left the BGRA buffer untouched");
    }
    // Bink frame numbers in BinkGoto are 1-based.
    void go_to(uint32_t target) {
        index = std::min(target, header.frames - 1);
        api.go_to(bink, index + 1, 0);
        decode();
    }
    uint32_t fps_frames(double seconds) const {
        return (uint32_t)(seconds * header.fps_num / header.fps_den);
    }
};

Player* g_player = nullptr;

void paint(HWND hwnd) {
    PAINTSTRUCT ps;
    HDC dc = BeginPaint(hwnd, &ps);
    RECT rc;
    GetClientRect(hwnd, &rc);
    const int cw = rc.right - rc.left, ch = rc.bottom - rc.top;
    const Player& p = *g_player;
    // Fit the frame in the client area, keeping its aspect ratio, and black out the bars.
    int w = cw, h = (int)((int64_t)cw * p.header.height / p.header.width);
    if (h > ch) {
        h = ch;
        w = (int)((int64_t)ch * p.header.width / p.header.height);
    }
    const int x = (cw - w) / 2, y = (ch - h) / 2;
    HBRUSH black = (HBRUSH)GetStockObject(BLACK_BRUSH);
    RECT bars[] = {{0, 0, cw, y}, {0, y + h, cw, ch}, {0, y, x, y + h}, {x + w, y, cw, y + h}};
    for (const RECT& b : bars) FillRect(dc, &b, black);
    BITMAPINFO bi{};
    bi.bmiHeader.biSize = sizeof(bi.bmiHeader);
    bi.bmiHeader.biWidth = (LONG)p.header.width;
    bi.bmiHeader.biHeight = -(LONG)p.header.height;  // top-down
    bi.bmiHeader.biPlanes = 1;
    bi.bmiHeader.biBitCount = 32;
    bi.bmiHeader.biCompression = BI_RGB;
    SetStretchBltMode(dc, COLORONCOLOR);
    StretchDIBits(dc, x, y, w, h, 0, 0, (int)p.header.width, (int)p.header.height, p.frame.data(), &bi,
                  DIB_RGB_COLORS, SRCCOPY);
    EndPaint(hwnd, &ps);
}

LRESULT CALLBACK window_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    Player& p = *g_player;
    switch (msg) {
        case WM_PAINT: paint(hwnd); return 0;
        case WM_ERASEBKGND: return 1;
        case WM_KEYDOWN:
            switch (wp) {
                case VK_ESCAPE: DestroyWindow(hwnd); return 0;
                case VK_SPACE: p.paused = !p.paused; return 0;
                case 'L': p.loop = !p.loop; return 0;
                case VK_HOME: p.seek_pending = true; p.seek_to = 0; return 0;
                case VK_LEFT: {
                    const uint32_t back = p.fps_frames(1.0);
                    p.seek_pending = true;
                    p.seek_to = p.index > back ? p.index - back : 0;
                    return 0;
                }
                case VK_RIGHT:
                    p.seek_pending = true;
                    p.seek_to = std::min(p.index + p.fps_frames(1.0), p.header.frames - 1);
                    return 0;
            }
            break;
        case WM_DESTROY: p.quit = true; PostQuitMessage(0); return 0;
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

void set_title(HWND hwnd, const Player& p, const std::wstring& name, double rate) {
    wchar_t title[256];
    swprintf(title, 256, L"bink_play - %s - frame %u/%u - decode %.1f fps%s%s", name.c_str(), p.index + 1,
             p.header.frames, rate, p.paused ? L" - paused" : L"", p.loop ? L" - loop" : L"");
    SetWindowTextW(hwnd, title);
}

constexpr const char* kDefaultMovie = "Stinger/RSG_STINGER.bk2";

std::string lower_ascii(std::string v) {
    for (char& c : v) c = (char)std::tolower((unsigned char)c);
    std::replace(v.begin(), v.end(), '\\', '/');
    return v;
}

struct NamedMovie {
    std::string path;
    uint64_t offset = 0;
};

struct NameTable {
    std::filesystem::path archive;  // the archive the table was built from (its first line)
    std::vector<NamedMovie> movies;
};

std::filesystem::path name_table_path() {
    const std::string base = env("LOCALAPPDATA");
    return base.empty() ? std::filesystem::path() : std::filesystem::path(base) / "Leonida" / "bink_names.csv";
}

// tools/bink_play.py's table: "# <archive>|<size>|<mtime>", then a header and path,offset,width,height,frames rows.
// Stale when the archive's size changed. Paths in the table have no commas (they are game file paths).
bool read_name_table(NameTable& out) {
    const std::filesystem::path p = name_table_path();
    std::ifstream f(p);
    std::string line;
    if (p.empty() || !f || !std::getline(f, line) || line.rfind("# ", 0) != 0) return false;
    const std::string stamp = line.substr(2);
    const size_t bar1 = stamp.find('|'), bar2 = stamp.find('|', bar1 + 1);
    if (bar1 == std::string::npos || bar2 == std::string::npos) return false;
    out.archive = std::filesystem::path(stamp.substr(0, bar1));
    std::error_code ec;
    const auto size = std::filesystem::file_size(out.archive, ec);
    if (ec || std::to_string(size) != stamp.substr(bar1 + 1, bar2 - bar1 - 1)) return false;
    std::getline(f, line);  // header
    out.movies.clear();
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const size_t c1 = line.find(','), c2 = line.find(',', c1 + 1);
        if (c1 == std::string::npos || c2 == std::string::npos) continue;
        out.movies.push_back({line.substr(0, c1), std::strtoull(line.substr(c1 + 1, c2 - c1 - 1).c_str(), nullptr, 10)});
    }
    return !out.movies.empty();
}

// Builds the table with tools/bink_play.py (the repository's copy, two folders up from runtime/build), once.
bool build_name_table(const std::filesystem::path& game_dir) {
    wchar_t exe[MAX_PATH];
    GetModuleFileNameW(nullptr, exe, MAX_PATH);
    std::filesystem::path tool;
    for (std::filesystem::path d = std::filesystem::path(exe).parent_path(); !d.empty() && d != d.root_path();
         d = d.parent_path())
        if (std::filesystem::is_regular_file(d / "tools" / "bink_play.py")) {
            tool = d / "tools" / "bink_play.py";
            break;
        }
    if (tool.empty() || game_dir.empty()) return false;
    std::printf("naming the movies in movies.rpf (about half a minute, once)...\n");
    std::fflush(stdout);
    const std::wstring cmd = L"python \"" + tool.wstring() + L"\" --list --game-dir \"" + game_dir.wstring() + L"\"";
    STARTUPINFOW si{sizeof(si)};
    si.dwFlags = STARTF_USESTDHANDLES;
    HANDLE nul = CreateFileW(L"NUL", GENERIC_WRITE, FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, 0, nullptr);
    si.hStdOutput = nul;  // the list itself isn't wanted, only the table it leaves behind
    si.hStdError = GetStdHandle(STD_ERROR_HANDLE);
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    PROCESS_INFORMATION pi{};
    std::vector<wchar_t> line(cmd.begin(), cmd.end());
    line.push_back(0);
    const bool ok = CreateProcessW(nullptr, line.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr, &si, &pi);
    if (ok) {
        WaitForSingleObject(pi.hProcess, INFINITE);
        CloseHandle(pi.hThread);
        CloseHandle(pi.hProcess);
    }
    if (nul != INVALID_HANDLE_VALUE) CloseHandle(nul);
    return ok;
}

// The movie a NAME means: exactly that path (case-insensitive), else the only path containing it.
NamedMovie find_named(const NameTable& table, const std::string& name) {
    const std::string want = lower_ascii(name);
    std::vector<const NamedMovie*> part;
    for (const NamedMovie& m : table.movies) {
        const std::string have = lower_ascii(m.path);
        if (have == want) return m;
        if (have.find(want) != std::string::npos) part.push_back(&m);
    }
    if (part.size() == 1) return *part[0];
    std::string why = part.empty() ? "no movie is named like \"" + name + "\"" :
                                     std::to_string(part.size()) + " movies match \"" + name + "\"; be more specific:";
    for (size_t i = 0; i < part.size() && i < 20; ++i) why += "\n  " + part[i]->path;
    if (part.empty()) why += " (python tools\\bink_play.py --list shows the names)";
    fatal(why);
}

}  // namespace

int main(int argc, char** argv) {
    std::string game_dir_arg = env("LEONIDA_GAME_DIR"), movie_arg = env("LEONIDA_MOVIE");
    long max_frames = -1;
    bool headless = false, loop = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--game-dir" && i + 1 < argc) game_dir_arg = argv[++i];
        else if (a == "--frames" && i + 1 < argc) {
            char* end = nullptr;
            max_frames = std::strtol(argv[++i], &end, 10);
            if (!end || *end || max_frames <= 0) fatal("--frames takes a positive count");
        } else if (a == "--headless") headless = true;
        else if (a == "--loop") loop = true;
        else if (!a.empty() && a[0] != '-') movie_arg = a;
        else {
            std::fprintf(stderr, "usage: bink_play [NAME | movie.bk2 | archive.rpf@OFFSET] [--game-dir DIR] "
                                 "[--frames N] [--headless] [--loop]\n");
            return 2;
        }
    }
    g_windowed = !headless;
    // CTest's headless smoke run gives no movie: that is a skip, not a request for the default one.
    if (movie_arg.empty() && headless) {
        std::printf("SKIP (no package/movie)\n");
        return 77;
    }
    if (movie_arg.empty()) movie_arg = kDefaultMovie;

    // A name (not a file, not archive@OFFSET) goes through the name table, which also knows the game folder.
    std::error_code ec;
    const bool is_name = movie_arg.find('@') == std::string::npos && !std::filesystem::is_regular_file(movie_arg, ec);
    if (is_name) {
        NameTable table;
        if (!read_name_table(table)) {
            if (game_dir_arg.empty())
                fatal("bink_play needs the game folder once, to name the movies:\n\n"
                      "  bink_play " + movie_arg + " --game-dir <package folder>\n\n"
                      "After that, double-clicking bink_play.exe plays the stinger.");
            build_name_table(game_dir_arg);
            if (!read_name_table(table)) fatal("could not build the movie name table with tools\\bink_play.py");
        }
        const NamedMovie m = find_named(table, movie_arg);
        std::printf("%s\n", m.path.c_str());
        if (game_dir_arg.empty()) game_dir_arg = table.archive.parent_path().string();
        movie_arg = table.archive.string() + "@" + std::to_string(m.offset);
    }
    if (game_dir_arg.empty()) fatal("no game folder: pass --game-dir or set LEONIDA_GAME_DIR");

    const std::filesystem::path game_dir = game_dir_arg;
    std::filesystem::path movie = movie_arg;
    uint64_t offset = 0;
    if (const size_t at = movie_arg.rfind('@'); at != std::string::npos) {
        char* end = nullptr;
        offset = std::strtoull(movie_arg.c_str() + at + 1, &end, 0);
        if (at + 1 == movie_arg.size() || !end || *end) fatal("the movie's @OFFSET is not a number");
        movie = movie_arg.substr(0, at);
        if (movie.is_relative() && !std::filesystem::exists(movie)) movie = game_dir / movie;
    }
    if (!std::filesystem::is_regular_file(game_dir / "Bink2Scarlett.dll", ec))
        fatal("the game folder does not contain Bink2Scarlett.dll");
    if (!std::filesystem::is_regular_file(movie, ec)) fatal("the movie is not a file");

    Player player;
    g_player = &player;
    player.header = read_movie_header(movie, offset);
    player.loop = loop;
    const MovieHeader& h = player.header;
    std::printf("movie: %ux%u, %u frames, %u/%u fps\n", h.width, h.height, h.frames, h.fps_num, h.fps_den);

    try {
        rt::Loader loader(game_dir);
        rt::Module& dll = loader.load(game_dir / "Bink2Scarlett.dll");
        for (auto& r : dll.imports)
            if (r.binding == rt::Binding::Unimplemented && rt::lower(r.dll).find("xaudio") == std::string::npos)
                fatal("Bink2Scarlett.dll import " + r.dll + "!" + r.name + " is unimplemented");
        BinkApi& api = player.api;
        bind(dll, "BinkOpen", api.open);
        bind(dll, "BinkDoFrame", api.do_frame);
        bind(dll, "BinkCopyToBuffer", api.copy);
        bind(dll, "BinkNextFrame", api.next);
        bind(dll, "BinkGoto", api.go_to);
        bind(dll, "BinkClose", api.close);
        bind(dll, "BinkGetError", api.get_error);
        if (offset) {
            bind(dll, "BinkSetFileOffset", api.set_file_offset);
            api.set_file_offset(offset);
        }

        player.bink = api.open(movie.string().c_str(), offset ? kFileOffset : 0);
        if (!player.bink) {
            const char* e = api.get_error ? api.get_error() : nullptr;
            fatal(std::string("BinkOpen failed: ") + (e && *e ? e : "(no BinkGetError text)"));
        }
        player.frame.assign((size_t)h.width * 4 * h.height, 0);
        player.decode();

        using clock = std::chrono::steady_clock;
        const auto frame_time = std::chrono::duration<double>((double)h.fps_den / h.fps_num);
        const uint32_t last = max_frames > 0 ? (uint32_t)std::min<long>(max_frames, (long)h.frames) : h.frames;

        if (headless) {
            const auto start = clock::now();
            uint32_t decoded = 1;
            while (decoded < last) {
                api.next(player.bink);
                player.decode();
                ++decoded;
            }
            const double s = std::chrono::duration<double>(clock::now() - start).count();
            std::printf("decoded %u frames in %.2f s (%.1f fps; the movie runs at %.2f fps)\n", decoded, s,
                        s > 0 ? decoded / s : 0.0, (double)h.fps_num / h.fps_den);
            api.close(player.bink);
            std::printf("RESULT PASS\n");
            return 0;
        }

        WNDCLASSW wc{};
        wc.lpfnWndProc = window_proc;
        wc.hInstance = GetModuleHandleW(nullptr);
        wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
        wc.lpszClassName = L"LeonidaBinkPlay";
        RegisterClassW(&wc);
        // Start at up to 1280 wide (or the movie's size if smaller), the movie's aspect ratio.
        const int start_w = (int)std::min<uint32_t>(h.width, 1280);
        RECT r{0, 0, start_w, (int)((int64_t)start_w * h.height / h.width)};
        AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, FALSE);
        HWND hwnd = CreateWindowW(wc.lpszClassName, L"bink_play", WS_OVERLAPPEDWINDOW | WS_VISIBLE, CW_USEDEFAULT,
                                  CW_USEDEFAULT, r.right - r.left, r.bottom - r.top, nullptr, nullptr, wc.hInstance,
                                  nullptr);
        if (!hwnd) fatal("CreateWindow failed");
        const std::wstring name = movie.filename().wstring() + (offset ? L"@" + std::to_wstring(offset) : L"");

        auto due = clock::now() + frame_time;
        auto rate_start = clock::now();
        uint32_t rate_frames = 0, shown = 1;
        double rate = 0;
        while (!player.quit) {
            MSG msg;
            while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
                TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
            if (player.quit) break;
            if (player.seek_pending) {
                player.seek_pending = false;
                player.go_to(player.seek_to);
                InvalidateRect(hwnd, nullptr, FALSE);
                due = clock::now() + frame_time;
            }
            if (player.paused || clock::now() < due) {
                set_title(hwnd, player, name, rate);
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            if (player.index + 1 >= h.frames || (max_frames > 0 && shown >= last)) {
                if (!player.loop) break;
                player.go_to(0);
            } else {
                api.next(player.bink);
                player.decode();
                ++player.index;
            }
            ++shown;
            InvalidateRect(hwnd, nullptr, FALSE);
            UpdateWindow(hwnd);
            ++rate_frames;
            const double elapsed = std::chrono::duration<double>(clock::now() - rate_start).count();
            if (elapsed >= 1.0) {
                rate = rate_frames / elapsed;
                rate_frames = 0;
                rate_start = clock::now();
            }
            set_title(hwnd, player, name, rate);
            // Hold the movie's frame rate; when decoding falls behind, don't try to catch up in a burst.
            due += std::chrono::duration_cast<clock::duration>(frame_time);
            if (clock::now() > due + std::chrono::duration_cast<clock::duration>(frame_time * 4)) due = clock::now();
        }
        if (IsWindow(hwnd)) DestroyWindow(hwnd);
        api.close(player.bink);
        std::printf("played %u frames\n", shown);
        return 0;
    } catch (const std::exception& e) {
        fatal(e.what());
    }
}
