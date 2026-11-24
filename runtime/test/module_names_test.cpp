// Original synthetic fixtures only: native Windows loads and the real Loader share the same copied DLL bytes.
#include <windows.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "pe_loader.h"

namespace {
namespace fs = std::filesystem;
int checks = 0, failures = 0;

void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { ++failures; std::fprintf(stderr, "FAIL %s\n", label); }
}

std::string encoded(const std::wstring& text, UINT page, bool& exact) {
    BOOL substituted = FALSE;
    const UINT actual_page = page == CP_ACP ? GetACP() : page;
    const bool utf8 = actual_page == CP_UTF8 || actual_page == 54936;
    const DWORD flags = utf8 ? 0 : WC_NO_BEST_FIT_CHARS;
    const int size = WideCharToMultiByte(page, flags, text.c_str(), -1, nullptr, 0,
                                         nullptr, utf8 ? nullptr : &substituted);
    if (!size) { exact = false; return {}; }
    std::string result(static_cast<size_t>(size), '\0');
    if (!WideCharToMultiByte(page, flags, text.c_str(), -1, result.data(), size,
                            nullptr, utf8 ? nullptr : &substituted)) {
        exact = false;
        return {};
    }
    result.pop_back();
    const int wide_size = MultiByteToWideChar(page, 0, result.c_str(), -1, nullptr, 0);
    std::wstring roundtrip(wide_size > 0 ? static_cast<size_t>(wide_size) : 0, L'\0');
    if (wide_size) MultiByteToWideChar(page, 0, result.c_str(), -1, roundtrip.data(), wide_size);
    exact = !substituted && wide_size > 0 && text == roundtrip.c_str();
    return result;
}

std::string utf8(const fs::path& path) {
    bool exact = false;
    auto result = encoded(path.native(), CP_UTF8, exact);
    if (!exact) throw std::runtime_error("fixture UTF-8 conversion failed");
    return result;
}

struct Directory {
    fs::path path;
    Directory() : path(fs::temp_directory_path() /
                       ("leonida_module_names_" + std::to_string(GetCurrentProcessId()) + "_" +
                        std::to_string(GetTickCount64()))) {
        if (!fs::create_directory(path)) throw std::runtime_error("cannot create unique fixture directory");
    }
    ~Directory() { std::error_code ec; fs::remove_all(path, ec); }
};

struct NativeModule {
    HMODULE handle;
    explicit NativeModule(const fs::path& path) : handle(LoadLibraryW(path.c_str())) {}
    ~NativeModule() { if (handle) FreeLibrary(handle); }
};

using Add = int(__cdecl*)(int, int);

void names(const fs::path& source, const fs::path& root) {
    std::vector<std::wstring> names{L"Plain.dll", L"Uni-\u6f22.dll", L"Uni-\u5b57.dll"};
    std::wstring acp_name;
    for (const wchar_t c : {L'\u00e9', L'\u03b4', L'\u0436', L'\u6f22'}) {
        bool exact = false;
        const auto name = std::wstring(L"Acp-") + c + L".dll";
        encoded(name, CP_ACP, exact);
        if (exact) { acp_name = name; names.push_back(name); break; }
    }
    if (acp_name.empty()) std::puts("SKIP non-ASCII ACP alias: no round-tripping fixture character");
    rt::Loader loader(root);
    std::vector<rt::Module*> mapped;
    for (const auto& name : names) {
        const fs::path path = root / name;
        fs::copy_file(source, path);
        NativeModule native(path);
        check(native.handle != nullptr, "native Windows loads the renamed DLL");
        if (!native.handle) continue;
        const auto native_add = reinterpret_cast<Add>(GetProcAddress(native.handle, "tiny_add"));
        check(native_add && native_add(2, 3) == 5, "native renamed DLL resolves and executes its export");
        try {
            rt::Module& m = loader.load(path);
            check(m.base != nullptr, "mapped renamed DLL has an image base");
            check(m.name == utf8(path.filename()), "Module::name preserves the UTF-8 basename");
            std::string display;
            uint32_t offset = 0;
            check(loader.find_address(reinterpret_cast<uint64_t>(m.base), display, offset) &&
                      display == utf8(path.filename()) && offset == 0,
                  "address lookup returns the same UTF-8 display name");
            const auto add = reinterpret_cast<Add>(m.exports.at("tiny_add"));
            check(add(2, 3) == 5, "mapped renamed DLL resolves and executes its export");
            check(&loader.load(path) == &m, "repeated loads reuse the mapped module");

            std::wstring alias = name;
            for (auto& c : alias) {
                if (c >= L'a' && c <= L'z') c = static_cast<wchar_t>(c - L'a' + L'A');
                else if (c >= L'A' && c <= L'Z') c = static_cast<wchar_t>(c - L'A' + L'a');
            }
            NativeModule native_alias(root / alias);
            check(native_alias.handle == native.handle, "native ASCII-case path alias reuses the module");
            check(&loader.load(root / alias) == &m, "mapped ASCII-case path alias reuses the module");

            const fs::path shadow = root / "shadow";
            fs::create_directory(shadow);
            fs::copy_file(source, shadow / name);
            check(&loader.load(shadow / name) == &m, "the loader retains its existing basename cache rule");

            bool exact = false;
            const auto narrow = encoded(name, CP_ACP, exact);
            if (exact) {
                check(GetModuleHandleA(narrow.c_str()) == native.handle, "native ACP name lookup finds the DLL");
                check(loader.find_loaded_module(rt::lower(narrow)) == m.base, "mapped ACP name lookup finds the DLL");
                bool handled = false;
                check(loader.load_runtime_module(rt::lower(narrow), 0, handled) == m.base && handled,
                      "runtime ACP load reuses the mapped DLL");
                // The code page's own letters in upper case (as Windows ignores them), not only ASCII ones.
                std::wstring upper(name);
                LCMapStringEx(LOCALE_NAME_INVARIANT, LCMAP_UPPERCASE, name.c_str(), -1, upper.data(),
                              static_cast<int>(upper.size()) + 1, nullptr, nullptr, 0);
                bool upper_exact = false;
                const auto narrow_upper = encoded(upper, CP_ACP, upper_exact);
                if (upper_exact)
                    check(loader.find_loaded_module(rt::lower(narrow_upper)) == m.base,
                          "mapped ACP name lookup ignores case outside ASCII");
            }
            // The wide lookups the W entries use: as named, in the system table's upper case, and a run-time load
            // that returns the mapped module (what Windows does for LoadLibraryW of a loaded image).
            check(loader.find_loaded_module_w(name) == m.base, "mapped wide name lookup finds the DLL");
            std::wstring wide_upper(name);
            LCMapStringEx(LOCALE_NAME_INVARIANT, LCMAP_UPPERCASE, name.c_str(), -1, wide_upper.data(),
                          static_cast<int>(wide_upper.size()) + 1, nullptr, nullptr, 0);
            check(loader.find_loaded_module_w(wide_upper) == m.base, "mapped wide name lookup ignores case");
            bool handled_w = false;
            check(loader.load_runtime_module_w(name, 0, handled_w) == m.base && handled_w,
                  "runtime wide load reuses the mapped DLL");
            for (const auto* previous : mapped)
                check(previous != &m && previous->base != m.base, "distinct Unicode basenames remain distinct mappings");
            mapped.push_back(&m);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "renamed fixture: %s\n", e.what());
            check(false, "mapped renamed DLL completes all name checks");
        }
    }
    check(mapped.size() == names.size(), "every renamed fixture completes its mapped checks");

    if (!acp_name.empty()) {
        const auto fresh_name = std::wstring(L"Fresh-") + acp_name;
        fs::copy_file(source, root / fresh_name);
        bool exact = false, handled = false;
        const auto narrow = encoded(fresh_name, CP_ACP, exact);
        NativeModule native(root / fresh_name);
        check(exact && native.handle, "native control loads the fresh ACP-named DLL");
        void* loaded = loader.load_runtime_module(rt::lower(narrow), 0, handled);
        check(handled && loaded && loaded == loader.find_loaded_module(rt::lower(narrow)),
              "runtime ACP path conversion also loads a previously unmapped DLL");
    }
}

void errors(const fs::path& root) {
    for (const bool exists : {false, true}) {
        const fs::path path = root / (exists ? L"Bad-\u6f22.dll" : L"Missing-\u6f22.dll");
        if (exists) { std::ofstream file(path, std::ios::binary); file.put('x'); }
        rt::Loader loader(root);
        try {
            loader.load(path);
            check(false, "invalid Unicode-named input must fail");
        } catch (const std::exception& e) {
            const std::string expected = exists ? utf8(path) + ": not an MZ image" : "cannot open " + utf8(path);
            check(e.what() == expected, "Unicode path errors retain the path and original failure reason");
            check(loader.modules().empty(), "rejected Unicode-named input leaves no mapped module");
        }
    }
}
}  // namespace

int wmain(int argc, wchar_t** argv) {
    if (argc != 2) return 2;
    try {
        Directory directory;
        names(fs::absolute(argv[1]), directory.path);
        errors(directory.path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "fixture failure: %s\n", e.what());
        ++failures;
    }
    std::printf("module_names_test: %d checks, %d failures, ACP %u\n", checks, failures, GetACP());
    return failures ? 1 : 0;
}
