// A stand-in game executable for `gta6rt --run`: what a real title does in its first second, run through the
// loader instead of the Windows one. Test code, not game content. Run as
//   gta6rt --run tiny_game.exe -- hello 42 [tls] [eh] [stack] [game-file] [exit-code] [abort-mode]
// It checks its arguments, a static initializer, the heap, a file round trip, a second thread, and the
// xgameruntime.dll imports (answered by the shim registry: the stub DLL it links against would fail them), then
// calls exit() with 0 (42 with `exit-code`) when everything held. Two checks are opt-in, by argument (CTest's
// run_tiny_game_tls and run_tiny_game_eh): `tls`, thread_local on the game's own threads, which needs the loader's
// per-thread TLS (gta6rt starts main on a separate thread); and `eh`, a C++ exception caught inside the image.
// A third, `game-file` (run_tiny_game_dir), reads GAME:\tiny_game_probe.txt after moving the current directory.
// A fourth, `stack` (run_tiny_game_stack), checks that the entry thread and a thread created with dwStackSize 0
// both have 8 MB of stack: a driver compiles shaders on the game's threads (BUILD-1457). Only under gta6rt: run
// natively, the second thread gets this program's own 1 MB.
// An `abort-*` mode installs a SIGABRT disposition and calls abort(); tiny_game_abort_test compares the exact exit
// codes with a native run of the same image. No game files.
// On MSVC, `eh` needs more than the unwind tables #176 registers: _CxxThrowException finds the throwing image
// with RtlPcToFileHeader, which the loader redirects in the host MSVC runtime. Arguments, not environment variables,
// so scripts/run_wine_tests.py (which passes a test's command line but not its ENVIRONMENT) runs them too. stdout is
// unbuffered, so a crash still shows how far the run got.
#include <windows.h>

#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

extern "C" __declspec(dllimport) HRESULT __stdcall XGameRuntimeInitialize();
extern "C" __declspec(dllimport) void __stdcall XGameRuntimeUninitialize();
extern "C" __declspec(dllimport) uint32_t __stdcall XSystemGetDeviceType();

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
    std::printf("tiny_game: %s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++g_failures;
}

struct StaticInit {
    int value;
    StaticInit() : value(1234) {}
};
StaticInit g_static;

bool has_arg(int argc, char** argv, const char* name) {
    for (int i = 3; i < argc; ++i)
        if (std::strcmp(argv[i], name) == 0) return true;
    return false;
}

thread_local int t_counter = 5;

void abort_exit_handler(int) {
    std::_Exit(43);  // a handler that ends the run itself, as a real title's crash reporter may
}
void abort_crt_exit_handler(int) {
    std::exit(43);  // the same through the shimmed exit() import
}
void abort_return_handler(int) {}

// The reserved stack of the calling thread, in bytes, written to *out (a size_t).
DWORD WINAPI stack_reserve(void* out) {
    ULONG_PTR low = 0, high = 0;
    GetCurrentThreadStackLimits(&low, &high);
    *static_cast<size_t*>(out) = static_cast<size_t>(high - low);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("tiny_game: started, argc=%d\n", argc);
    check(argc >= 3 && std::strcmp(argv[1], "hello") == 0 && std::strcmp(argv[2], "42") == 0,
          "argv is the game's command line (\"hello\" \"42\"), not gta6rt's");
    check(std::strstr(GetCommandLineA(), "hello 42") != nullptr, "GetCommandLineA is the game's command line");
    check(g_static.value == 1234, "a static initializer ran before main");

    std::vector<int> v(1 << 20);
    for (size_t i = 0; i < v.size(); ++i) v[i] = int(i);
    check(v[123456] == 123456, "a 4 MiB heap allocation works");

    char tmp[MAX_PATH] = {};
    GetTempPathA(MAX_PATH, tmp);
    std::string path = std::string(tmp) + "leonida_tiny_game_" + std::to_string(GetCurrentProcessId()) + ".txt";
    bool file_ok = false;
    if (FILE* f = std::fopen(path.c_str(), "wb")) {
        std::fputs("leonida", f);
        std::fclose(f);
        char back[16] = {};
        if (FILE* r = std::fopen(path.c_str(), "rb")) {
            std::fread(back, 1, 7, r);
            std::fclose(r);
            file_ok = std::strcmp(back, "leonida") == 0;
        }
        std::remove(path.c_str());
    }
    check(file_ok, "a file round trip through the CRT works");

    long long sum = 0;
    std::thread worker([&] { for (int i = 1; i <= 1000; ++i) sum += i; });
    worker.join();
    check(sum == 500500, "a second thread ran and joined");

    check(XGameRuntimeInitialize() == S_OK, "XGameRuntimeInitialize is answered by the gta6rt shim (S_OK)");
    check(XSystemGetDeviceType() == 7, "XSystemGetDeviceType is answered by the gta6rt shim (Series X)");
    XGameRuntimeUninitialize();

    if (has_arg(argc, argv, "tls")) {
        int seen = -1;
        std::thread t([&] { seen = t_counter; t_counter = 9; });
        t.join();
        check(t_counter == 5 && seen == 5, "thread_local starts at its initializer on each thread");
    } else {
        std::printf("tiny_game: SKIP thread_local (argument tls)\n");
    }
    if (has_arg(argc, argv, "eh")) {
        bool caught = false;
        try {
            throw std::string("boom");
        } catch (const std::string& e) {
            caught = e == "boom";
        }
        check(caught, "a C++ exception is thrown and caught inside the image");
    } else {
        std::printf("tiny_game: SKIP C++ exception (argument eh)\n");
    }
    if (has_arg(argc, argv, "stack")) {
        // Windows gives a thread created with no size the reserve in the process's PE header, which is gta6rt's,
        // not this image's. The entry thread gets its size from run_entry.
        const size_t want = size_t{8} << 20;
        size_t entry = 0, created = 0;
        stack_reserve(&entry);
        if (HANDLE thread = CreateThread(nullptr, 0, stack_reserve, &created, 0, nullptr)) {
            WaitForSingleObject(thread, INFINITE);
            CloseHandle(thread);
        }
        std::printf("tiny_game: stack reserve: entry thread %zu bytes, CreateThread with dwStackSize 0 %zu bytes\n",
                    entry, created);
        check(entry >= want, "the entry thread has at least 8 MB of stack");
        check(created >= want, "a thread created with dwStackSize 0 has at least 8 MB of stack");
    } else {
        std::printf("tiny_game: SKIP stack reserve (argument stack)\n");
    }

    if (has_arg(argc, argv, "game-file")) {
        // The probe file sits in the game folder (check_game_dir.cmake writes it). Move the current directory away
        // first, as a title may: GAME: must still name the game folder, whether gta6rt was given a bare image name
        // or a relative --game-dir.
        SetCurrentDirectoryA(tmp);
        HANDLE file = CreateFileW(L"GAME:\\tiny_game_probe.txt", GENERIC_READ, FILE_SHARE_READ, nullptr,
                                  OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        char data[4] = {};
        DWORD count = 0;
        const bool read = file != INVALID_HANDLE_VALUE && ReadFile(file, data, sizeof(data), &count, nullptr) &&
                          count == sizeof(data) && std::memcmp(data, "game", sizeof(data)) == 0;
        if (file == INVALID_HANDLE_VALUE) std::printf("tiny_game: GAME: open error %lu\n", GetLastError());
        else CloseHandle(file);
        check(read, "GAME:\\tiny_game_probe.txt is read from the game folder after SetCurrentDirectory");
    } else {
        std::printf("tiny_game: SKIP GAME: file (argument game-file)\n");
    }

    if (has_arg(argc, argv, "abort-handler")) {
        std::signal(SIGABRT, abort_exit_handler);
        std::abort();
    }
    if (has_arg(argc, argv, "abort-crt-exit")) {
        std::signal(SIGABRT, abort_crt_exit_handler);
        std::abort();
    }
    if (has_arg(argc, argv, "abort-return")) {
        std::signal(SIGABRT, abort_return_handler);
        std::abort();
    }
    if (has_arg(argc, argv, "abort-ignore")) {
        std::signal(SIGABRT, SIG_IGN);
        std::abort();
    }
    if (has_arg(argc, argv, "abort-default")) {
        std::signal(SIGABRT, SIG_DFL);
        std::abort();
    }

    std::printf("tiny_game: %s\n", g_failures ? "FAILED" : "OK");
    std::fflush(stdout);
    // Reserve 42 for a successful guest test. The loader itself returns 1 on errors.
    std::exit(g_failures ? 1 : (has_arg(argc, argv, "exit-code") ? 42 : 0));
}
