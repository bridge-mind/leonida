// The process shim's command line round-trips: what GetCommandLineW says must parse with CommandLineToArgvW (the
// documented rules the CRT's argv parser shares) back to the __wargv the shim hands out, and __wargv must hold what
// main.cpp passed, whatever the system code page (RT-84). The WinMain line (lpCmdLine) is the game's too, and the
// command line and ExitProcess are served under kernelbase.dll and the API-set names as well as kernel32.dll.
// No game files.
#include <windows.h>
#include <shellapi.h>

#include <cstdio>
#include <cstring>
#include <cwchar>
#include <string>
#include <vector>

#include "process_shim.h"
#include "shims.h"

namespace {
int failures = 0;

wchar_t** wargv() {
    using Fn = wchar_t*** (__cdecl*)();
    return *reinterpret_cast<Fn>(rt::shims::find("ucrtbase.dll", "__p___wargv"))();
}

char** argv() {
    using Fn = char*** (__cdecl*)();
    return *reinterpret_cast<Fn>(rt::shims::find("ucrtbase.dll", "__p___argv"))();
}

int configure(bool wide, int mode) {
    using Fn = int(__cdecl*)(int);
    const auto fn = reinterpret_cast<Fn>(rt::shims::find(
        "ucrtbase.dll", wide ? "_configure_wide_argv" : "_configure_narrow_argv"));
    if (fn) return fn(mode);
    std::fprintf(stderr, "FAIL missing argv configuration shim\n");
    ++failures;
    return -1;
}

void round_trip(const std::vector<std::wstring>& args, const char* what) {
    rt::process::set_command_line(L"game.exe", args);
    if (configure(false, 1) || configure(true, 1)) {
        std::fprintf(stderr, "FAIL %s: unexpanded argv configuration\n", what);
        ++failures;
        return;
    }
    using Fn = LPWSTR(WINAPI*)();
    const LPWSTR line = reinterpret_cast<Fn>(rt::shims::find("kernel32.dll", "GetCommandLineW"))();
    int count = 0;
    LPWSTR* parsed = CommandLineToArgvW(line, &count);
    bool ok = parsed && count == int(args.size()) + 1;
    for (int i = 0; ok && i < count; ++i) ok = wargv()[i] && wcscmp(parsed[i], wargv()[i]) == 0;
    if (!ok) {
        std::fwprintf(stderr, L"FAIL %hs: GetCommandLineW [%ls] parses to %d argument(s)\n", what, line, count);
        ++failures;
    }
    LocalFree(parsed);
}

void check_argv_reset() {
    const int host_argc = __argc;
    char** const host_argv = __argv;
    wchar_t** const host_wargv = __wargv;
    using Argc = int*(__cdecl*)();
    const auto count = reinterpret_cast<Argc>(rt::shims::find("ucrtbase.dll", "__p___argc"));
    auto check = [](bool ok, const char* what) {
        if (!ok) { std::fprintf(stderr, "FAIL argv state: %s\n", what); ++failures; }
    };
    rt::process::set_command_line(L"first.exe", {L"one", L"two"});
    check(*count() == 0 && !argv() && !wargv(), "new run starts unconfigured");
    check(configure(false, 0) == 0 && configure(true, 0) == 0 && *count() == 0 && !argv() && !wargv(),
          "no-argument startup leaves the initial state alone");
    check(configure(false, 1) == 0 && *count() == 3 && argv() && !wargv(), "configure only narrow argv");
    char** const narrow = argv();
    check(configure(true, 1) == 0 && *count() == 3 && argv() == narrow && wargv(),
          "wide configuration retains the narrow table");
    wchar_t** const wide = wargv();
    check(configure(false, 0) == 0 && configure(true, 0) == 0 && *count() == 3 && argv() == narrow && wargv() == wide,
          "no-argument mode preserves configured counts and pointers");
    rt::process::set_command_line(L"second.exe", {L"new"});
    check(*count() == 0 && !argv() && !wargv(), "another run forgets both configured widths");
    check(configure(true, 1) == 0 && *count() == 2 && !argv() && wargv() &&
              std::wcscmp(wargv()[0], L"second.exe") == 0 && std::wcscmp(wargv()[1], L"new") == 0 && !wargv()[2],
          "the new run configures its own program name and arguments");
    check(__argc == host_argc && __argv == host_argv && __wargv == host_wargv, "host CRT argument globals are untouched");
}

// __wargv[1] must be `want` exactly, and the narrow argv[1] its ANSI code page text (the CRT's replacement for a
// character the code page lacks), not a crash.
void check_non_ascii(const std::wstring& want, const char* what) {
    round_trip({want}, what);
    if (!wargv()[1] || want != wargv()[1]) {
        std::fprintf(stderr, "FAIL %s: __wargv[1] is not the argument passed\n", what);
        ++failures;
    }
    char expected[64] = {};
    WideCharToMultiByte(CP_ACP, 0, want.c_str(), -1, expected, sizeof(expected), nullptr, nullptr);
    if (!argv()[1] || std::string(expected) != argv()[1]) {
        std::fprintf(stderr, "FAIL %s: argv[1] is not the code page %u text of the argument\n", what, GetACP());
        ++failures;
    }
}

// An executable can import GetCommandLineA/W and ExitProcess from kernel32.dll, kernelbase.dll or the API set its
// umbrella library names, and the loader matches the DLL name exactly: every name must hand out the same shim.
void check_import_names() {
    for (const char* dll : {"kernelbase.dll", "api-ms-win-core-processenvironment-l1-1-0.dll"})
        for (const char* name : {"GetCommandLineA", "GetCommandLineW"}) {
            void* fn = rt::shims::find(dll, name);
            if (!fn || fn != rt::shims::find("kernel32.dll", name)) {
                std::fprintf(stderr, "FAIL %s!%s is not the kernel32.dll shim\n", dll, name);
                ++failures;
            }
        }
    for (const char* dll : {"kernelbase.dll", "api-ms-win-core-processthreads-l1-1-0.dll"}) {
        void* fn = rt::shims::find(dll, "ExitProcess");
        if (!fn || fn != rt::shims::find("kernel32.dll", "ExitProcess")) {
            std::fprintf(stderr, "FAIL %s!ExitProcess is not the kernel32.dll shim\n", dll);
            ++failures;
        }
    }
    for (const char* dll : {"api-ms-win-crt-runtime-l1-1-0.dll", "ucrtbase.dll", "ucrtbased.dll"})
        for (const char* name : {"_get_narrow_winmain_command_line", "_get_wide_winmain_command_line"})
            if (!rt::shims::find(dll, name)) {
                std::fprintf(stderr, "FAIL %s!%s is not shimmed\n", dll, name);
                ++failures;
            }
}

// lpCmdLine as the CRT's WinMain startup gets it from _get_wide/_get_narrow_winmain_command_line: the game's command
// line after the program name and the blanks that follow it (the narrow one in the ANSI code page).
void check_winmain(const std::wstring& program, const std::vector<std::wstring>& args, const wchar_t* want,
                   const char* what) {
    rt::process::set_command_line(program, args);
    using Wide = wchar_t*(__cdecl*)();
    using Narrow = char*(__cdecl*)();
    for (const char* dll : {"api-ms-win-crt-runtime-l1-1-0.dll", "ucrtbase.dll"}) {
        const auto wide_fn = reinterpret_cast<Wide>(rt::shims::find(dll, "_get_wide_winmain_command_line"));
        const auto narrow_fn = reinterpret_cast<Narrow>(rt::shims::find(dll, "_get_narrow_winmain_command_line"));
        if (!wide_fn || !narrow_fn) continue;  // check_import_names has reported the missing shim
        const wchar_t* wide = wide_fn();
        const char* line = narrow_fn();
        char expected[256] = {};
        WideCharToMultiByte(CP_ACP, 0, want, -1, expected, sizeof(expected), nullptr, nullptr);
        if (!wide || std::wcscmp(wide, want) != 0) {
            std::fwprintf(stderr, L"FAIL %hs (%hs): the wide WinMain line is [%ls], want [%ls]\n", what, dll,
                          wide ? wide : L"(null)", want);
            ++failures;
        }
        if (!line || std::strcmp(line, expected) != 0) {
            std::fprintf(stderr, "FAIL %s (%s): the narrow WinMain line is [%s], want [%s]\n", what, dll,
                         line ? line : "(null)", expected);
            ++failures;
        }
    }
}

// msvcrt.dll's __getmainargs/__wgetmainargs (what a MinGW or old-MSVC CRT startup calls) hand out the game's argv and
// the process environment. The wide environment (_wenviron) does not exist until a wide environment function has run,
// and the narrow table is reallocated by _putenv, so both must be read at the call, not remembered by
// set_command_line (RT-136). `name=value` must be found in `env`, whatever was set before or after set_command_line.
template <typename C, typename Fn>
bool env_has(Fn getmainargs, const C* entry) {
    int argc = 0;
    C** argv = nullptr;
    C** env = nullptr;
    if (getmainargs(&argc, &argv, &env, 0, nullptr) != 0 || argc < 1 || !argv || !argv[0] || !env) return false;
    for (C** e = env; *e; ++e) {
        if constexpr (sizeof(C) == 1) {
            if (std::strcmp(reinterpret_cast<const char*>(*e), reinterpret_cast<const char*>(entry)) == 0) return true;
        } else {
            if (std::wcscmp(reinterpret_cast<const wchar_t*>(*e), reinterpret_cast<const wchar_t*>(entry)) == 0)
                return true;
        }
    }
    return false;
}

void check_getmainargs_environment() {
    using Narrow = int(__cdecl*)(int*, char***, char***, int, void*);
    using Wide = int(__cdecl*)(int*, wchar_t***, wchar_t***, int, void*);
    const auto narrow_fn = reinterpret_cast<Narrow>(rt::shims::find("msvcrt.dll", "__getmainargs"));
    const auto wide_fn = reinterpret_cast<Wide>(rt::shims::find("msvcrt.dll", "__wgetmainargs"));
    if (!narrow_fn || !wide_fn) {
        std::fprintf(stderr, "FAIL msvcrt.dll!__getmainargs/__wgetmainargs are not shimmed\n");
        ++failures;
        return;
    }
    // First use of the wide environment in this process: set_command_line runs before anything made _wenviron.
    // main() hosts start with only the narrow table; if a toolchain change made _wenviron exist, the first-call
    // checks below no longer prove the creation path, so say so rather than pass silently.
    if (_wenviron) std::printf("note: _wenviron already exists at startup; its lazy creation is not exercised\n");
    rt::process::set_command_line(L"game.exe", {L"-x"});
    if (_putenv_s("LEONIDA_RT136_A", "before") != 0 ||  // narrow only; the wide table is built on first use
        !env_has<wchar_t>(wide_fn, L"LEONIDA_RT136_A=before")) {
        std::fprintf(stderr, "FAIL __wgetmainargs: no environment (or one without LEONIDA_RT136_A) at the first call\n");
        ++failures;
    }
    if (!env_has<char>(narrow_fn, "LEONIDA_RT136_A=before")) {
        std::fprintf(stderr, "FAIL __getmainargs: no environment (or one without LEONIDA_RT136_A)\n");
        ++failures;
    }
    // A variable set after set_command_line reallocates the CRT's table; a pointer kept from before would dangle.
    if (_putenv_s("LEONIDA_RT136_B", "after") != 0 || _wputenv_s(L"LEONIDA_RT136_C", L"after") != 0) {
        std::fprintf(stderr, "FAIL _putenv_s/_wputenv_s did not set the test variables\n");
        ++failures;
    }
    if (!env_has<wchar_t>(wide_fn, L"LEONIDA_RT136_B=after") || !env_has<wchar_t>(wide_fn, L"LEONIDA_RT136_C=after")) {
        std::fprintf(stderr, "FAIL __wgetmainargs: an environment variable set after set_command_line is missing\n");
        ++failures;
    }
    if (!env_has<char>(narrow_fn, "LEONIDA_RT136_B=after") || !env_has<char>(narrow_fn, "LEONIDA_RT136_C=after")) {
        std::fprintf(stderr, "FAIL __getmainargs: an environment variable set after set_command_line is missing\n");
        ++failures;
    }
}
}  // namespace

int main() {
    round_trip({L"plain", L"with space"}, "plain and spaced arguments");
    round_trip({L"C:\\My Games\\", L"next"}, "a quoted argument ending in a backslash");
    round_trip({L"a\\\"b"}, "a backslash before a quote");
    round_trip({L"\\\\server\\share\\x y"}, "a UNC path with a space");

    // In code page 1252 but not 1253 (Greek), where path::string() used to throw (RT-84).
    check_non_ascii(L"caf\u00e9", "a Latin-1 argument");
    // Greek, Cyrillic and CJK together: no single-byte code page holds all of them, so this covers every PC.
    check_non_ascii(L"\u0394\u0416\u6F22 x", "an argument outside any single-byte code page");

    check_import_names();
    check_argv_reset();
    check_getmainargs_environment();
    check_winmain(L"game.exe", {L"-foo", L"with space"}, L"-foo \"with space\"", "arguments after a plain name");
    check_winmain(L"C:\\My Games\\game.exe", {L"-x"}, L"-x", "a quoted program name with a space");
    check_winmain(L"game.exe", {}, L"", "no arguments");
    // A byte above 0x7f in the narrow program name is part of the name, not the end of it.
    check_winmain(L"caf\u00e9.exe", {L"-x"}, L"-x", "a non-ASCII program name");

    // If the entry thread exits via ExitThread() without going through end_run(), run_entry() must preserve its exit
    // code (and claim g_exited before closing g_done).
    {
        auto exit_thread_entry = +[](void*) -> DWORD { ExitThread(42); };
        const int code = rt::process::run_entry(reinterpret_cast<void*>(exit_thread_entry), 0);
        if (code != 42) {
            std::fprintf(stderr, "FAIL run_entry with ExitThread(42) returned %d, want 42\n", code);
            ++failures;
        }
    }

    // The game thread that ends a run must park, not ExitProcess, even when run_entry has already returned and
    // cleared the run: main.cpp still prints "game exited with code N" and cleans up after it. On one CPU the
    // host preempts the game thread right after SetEvent(g_done), and Sleep lets the game thread run again before
    // the next run starts. A thread that exits the process here ends this test with code 77 before its last line.
    {
        auto returns_77 = +[](void*) -> DWORD { return 77; };
        DWORD_PTR process_mask = 0, system_mask = 0;
        GetProcessAffinityMask(GetCurrentProcess(), &process_mask, &system_mask);
        SetProcessAffinityMask(GetCurrentProcess(), process_mask & (~process_mask + 1));  // its lowest CPU
        for (int i = 0; i < 50; ++i) {
            const int code = rt::process::run_entry(reinterpret_cast<void*>(returns_77), 64 << 10);
            if (code != 77) {
                std::fprintf(stderr, "FAIL run %d of an entry point returning 77 returned %d\n", i, code);
                ++failures;
                break;
            }
            Sleep(1);
        }
        SetProcessAffinityMask(GetCurrentProcess(), process_mask);
    }

    if (failures) return 1;
    std::printf("process_cmdline: GetCommandLineW parses back to __wargv, non-ASCII arguments intact (code page %u); "
                "the WinMain line and the kernelbase/API-set names are the game's\n",
                GetACP());
    return 0;
}
