// Original synthetic guest. Name input is supplied independently of argv[0].
#include <windows.h>
#include <cstdio>
#include <cwchar>
#include <string>

extern "C" IMAGE_DOS_HEADER __ImageBase;

int wmain(int argc, wchar_t** argv) {
    if (argc != 2) return 20;
    BOOL substituted = FALSE;
    const UINT page = GetACP();
    const bool utf8 = page == CP_UTF8 || page == 54936;
    const DWORD flags = utf8 ? 0 : WC_NO_BEST_FIT_CHARS;
    const int size = WideCharToMultiByte(CP_ACP, flags, argv[1], -1, nullptr, 0,
                                         nullptr, utf8 ? nullptr : &substituted);
    if (!size) return 21;
    std::string name(static_cast<size_t>(size), '\0');
    if (!WideCharToMultiByte(CP_ACP, flags, argv[1], -1, name.data(), size,
                            nullptr, utf8 ? nullptr : &substituted)) return 22;
    const HMODULE self = reinterpret_cast<HMODULE>(&__ImageBase);
    // The name in upper case by the system's own table (every script, not only ASCII), the case Windows ignores.
    std::wstring upper(argv[1]);
    if (!LCMapStringEx(LOCALE_NAME_INVARIANT, LCMAP_UPPERCASE, argv[1], -1, upper.data(),
                       static_cast<int>(upper.size()) + 1, nullptr, nullptr, 0)) return 26;
    bool exact = false, ascii_case = false, acp_case = false;
    if (!substituted) {
        exact = GetModuleHandleA(name.c_str()) == self;
        std::wstring alias(argv[1]);
        for (wchar_t& c : alias) {
            if (c >= L'a' && c <= L'z') c = static_cast<wchar_t>(c - L'a' + L'A');
            else if (c >= L'A' && c <= L'Z') c = static_cast<wchar_t>(c - L'A' + L'a');
        }
        if (!WideCharToMultiByte(CP_ACP, flags, alias.c_str(), -1, name.data(), size,
                                nullptr, utf8 ? nullptr : &substituted)) return 24;
        ascii_case = GetModuleHandleA(name.c_str()) == self;
        // The same through the code page's own letters (GUEST_ДА.EXE for guest_да.exe), which Windows also ignores.
        BOOL upper_substituted = FALSE;
        if (!WideCharToMultiByte(CP_ACP, flags, upper.c_str(), -1, name.data(), size,
                                nullptr, utf8 ? nullptr : &upper_substituted)) return 25;
        acp_case = !upper_substituted && GetModuleHandleA(name.c_str()) == self;
    }
    // The wide lookups, the ones a title uses for its own file: as given, in the table's upper case, and
    // LoadLibraryW of a module already mapped, which Windows answers with that module.
    const bool self_w = GetModuleHandleW(argv[1]) == self;
    const bool case_w = GetModuleHandleW(upper.c_str()) == self;
    const HMODULE loaded = LoadLibraryW(argv[1]);
    const bool load_w = loaded == self;
    if (loaded) FreeLibrary(loaded);
    // A shim-served module named through a directory outside ASCII: under the loader, the shims answer for it by
    // basename whatever the directory is spelt in (the directory need not exist). Not meaningful natively.
    const wchar_t* shim_path = L"C:\\" L"漢" L"\\" L"xgameruntime.dll";
    const bool shim_dir_w = GetModuleHandleW(shim_path) != nullptr;
    const bool shim_dir_load_w = LoadLibraryW(shim_path) != nullptr;   // an owned module's handle is never freed
    std::printf("NAME_PROBE entered=1 acp=%u representable=%u self_a=%u ascii_case_a=%u acp_case_a=%u "
                "self_w=%u case_w=%u load_w=%u shim_dir_w=%u shim_dir_load_w=%u\n",
                page, !substituted ? 1u : 0u, exact ? 1u : 0u, ascii_case ? 1u : 0u, acp_case ? 1u : 0u,
                self_w ? 1u : 0u, case_w ? 1u : 0u, load_w ? 1u : 0u, shim_dir_w ? 1u : 0u,
                shim_dir_load_w ? 1u : 0u);
    const bool a_ok = substituted || (exact && ascii_case && acp_case);
    return a_ok && self_w && case_w && load_w ? 0 : 23;
}
