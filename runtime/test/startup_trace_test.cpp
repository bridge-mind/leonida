// Tests for the startup trace (gta6rt --trace FILE, shims/startup_trace.h).
//
// Everything is synthetic: the test writes a small x86-64 DLL image with two imports, one that binds to nothing
// (an unimplemented thunk) and kernel32!OutputDebugStringA (bound to the host), plus three functions in its .text:
//   0x1200  calls the unimplemented import
//   0x1210  calls OutputDebugStringA with the caller's rcx
//   0x1220  reads address 0 (an access violation the test catches)
// The loader maps it with the trace open, the test calls the three functions, closes the trace and checks the lines:
// map, unimplemented_call with the caller as module+RVA, debug_string with the text, exception with the faulting
// RVA and access kind. No game data and no GPU are involved.
// LEONIDA_TEST_DIR names the folder for the image and the trace, as for loader_test; the default is the temp folder.
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../shims/shims.h"
#include "../shims/startup_trace.h"
#include "../src/pe_loader.h"

namespace {

namespace fs = std::filesystem;

int g_checks = 0;
int g_failed = 0;

void check(bool ok, const std::string& what) {
    ++g_checks;
    std::printf("%s %s\n", ok ? "ok   " : "FAIL ", what.c_str());
    if (!ok) ++g_failed;
}

struct Import {
    const char* dll;
    const char* name;
};
const Import kImports[2] = {{"leonida_trace_probe.dll", "TraceProbeMissing"}, {"kernel32.dll", "OutputDebugStringA"}};
constexpr uint32_t kTextRva = 0x1000;
constexpr uint32_t kCallerBase = kTextRva + 0x200;  // caller i at kCallerBase + 0x10 * i
constexpr uint32_t kCallRetOffset = 10;             // sub rsp,0x28 (4) + call [rip+disp] (6)
constexpr uint32_t kFaultRva = kTextRva + 0x220;
constexpr const char* kImageName = "trace_image.bin";
constexpr const char* kDebugText = "leonida trace probe \"quoted\"";
constexpr const char* kAnsiText = "caf\xe9 probe";  // 0xE9: 'é' in code page 1252, never valid alone in UTF-8

uint32_t caller_rva(int i) { return kCallerBase + 0x10u * static_cast<uint32_t>(i); }
uint32_t return_rva(int i) { return caller_rva(i) + kCallRetOffset; }

std::vector<uint8_t> make_image() {
    std::vector<uint8_t> b(0x600, 0);
    auto* dos = reinterpret_cast<IMAGE_DOS_HEADER*>(b.data());
    dos->e_magic = IMAGE_DOS_SIGNATURE;
    dos->e_lfanew = 0x80;
    auto* nt = reinterpret_cast<IMAGE_NT_HEADERS64*>(b.data() + dos->e_lfanew);
    nt->Signature = IMAGE_NT_SIGNATURE;
    nt->FileHeader.Machine = IMAGE_FILE_MACHINE_AMD64;
    nt->FileHeader.NumberOfSections = 1;
    nt->FileHeader.SizeOfOptionalHeader = sizeof(IMAGE_OPTIONAL_HEADER64);
    nt->FileHeader.Characteristics = IMAGE_FILE_EXECUTABLE_IMAGE | IMAGE_FILE_DLL | IMAGE_FILE_LARGE_ADDRESS_AWARE;
    nt->OptionalHeader.Magic = IMAGE_NT_OPTIONAL_HDR64_MAGIC;
    nt->OptionalHeader.ImageBase = 0x180000000;
    nt->OptionalHeader.SizeOfHeaders = 0x200;
    nt->OptionalHeader.SizeOfImage = 0x2000;
    nt->OptionalHeader.Subsystem = IMAGE_SUBSYSTEM_WINDOWS_CUI;
    nt->OptionalHeader.NumberOfRvaAndSizes = IMAGE_NUMBEROF_DIRECTORY_ENTRIES;  // the loader reads no directory past it
    auto* sec = IMAGE_FIRST_SECTION(nt);
    std::memcpy(sec->Name, ".text", 6);
    sec->VirtualAddress = kTextRva;
    sec->Misc.VirtualSize = 0x400;
    sec->PointerToRawData = 0x200;
    sec->SizeOfRawData = 0x400;
    sec->Characteristics = IMAGE_SCN_CNT_CODE | IMAGE_SCN_MEM_EXECUTE | IMAGE_SCN_MEM_READ;

    uint8_t* text = b.data() + 0x200;
    for (int i = 0; i < 2; ++i) {
        const uint32_t name_off = 0x040 + 0x30u * i, lookup_off = 0x0A0 + 0x20u * i, iat_off = 0x0E0 + 0x20u * i,
                       hint_off = 0x120 + 0x40u * i;
        auto* d = reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR*>(text) + i;
        d->Name = kTextRva + name_off;
        d->OriginalFirstThunk = kTextRva + lookup_off;
        d->FirstThunk = kTextRva + iat_off;
        std::strcpy(reinterpret_cast<char*>(text + name_off), kImports[i].dll);
        *reinterpret_cast<uint64_t*>(text + lookup_off) = kTextRva + hint_off;
        std::strcpy(reinterpret_cast<char*>(text + hint_off + 2), kImports[i].name);

        const uint8_t code[] = {0x48, 0x83, 0xEC, 0x28,  // sub rsp, 0x28
                                0xFF, 0x15, 0,    0, 0, 0,  // call [rip + disp32]
                                0x48, 0x83, 0xC4, 0x28,  // add rsp, 0x28
                                0xC3};                   // ret
        uint8_t* fn = text + (caller_rva(i) - kTextRva);
        std::memcpy(fn, code, sizeof code);
        const int32_t disp = static_cast<int32_t>((kTextRva + iat_off) - return_rva(i));
        std::memcpy(fn + 6, &disp, 4);
    }
    const uint8_t fault[] = {0x48, 0x8B, 0x04, 0x25, 0, 0, 0, 0,  // mov rax, [0]
                             0xC3};                               // ret
    std::memcpy(text + (kFaultRva - kTextRva), fault, sizeof fault);
    auto& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    dir.VirtualAddress = kTextRva;
    dir.Size = 3 * sizeof(IMAGE_IMPORT_DESCRIPTOR);
    return b;
}

fs::path test_dir() {
    const char* env = std::getenv("LEONIDA_TEST_DIR");
    fs::path dir = (env && *env ? fs::path(env) : fs::temp_directory_path()) /
                   ("leonida_startup_trace_" + std::to_string(GetCurrentProcessId()));
    fs::create_directories(dir);
    return dir;
}

fs::path write_file(const fs::path& p, const std::vector<uint8_t>& bytes) {
    std::ofstream out(p, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!out) throw std::runtime_error("cannot write " + p.string());
    return p;
}

using VoidFn = uint64_t(__cdecl*)(const char*);

// Separate function: __try cannot share a frame with objects that have destructors.
bool call_faulting(void* fn) {
    __try {
        reinterpret_cast<uint64_t(__cdecl*)()>(fn)();
    } __except (GetExceptionCode() == EXCEPTION_ACCESS_VIOLATION ? EXCEPTION_EXECUTE_HANDLER
                                                                  : EXCEPTION_CONTINUE_SEARCH) {
        return true;
    }
    return false;
}

void raise_bad_dbg_print() {
    __try {
        ULONG_PTR args[2] = {5, 0x1000};  // invalid string pointer in DBG_PRINTEXCEPTION_C
        RaiseException(0x40010006, 0, 2, args);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
    }
}

std::string hex_rva(uint32_t rva) {
    char b[16];
    std::snprintf(b, sizeof b, "+0x%x", rva);
    return b;
}

std::vector<std::string> read_lines(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    std::vector<std::string> lines;
    for (std::string l; std::getline(in, l);) lines.push_back(l);
    return lines;
}

const std::string* find_line(const std::vector<std::string>& lines, const std::string& kind) {
    const std::string tag = "\"kind\":\"" + kind + "\"";
    for (auto& l : lines)
        if (l.find(tag) != std::string::npos) return &l;
    return nullptr;
}

bool has(const std::string* line, const std::string& what) { return line && line->find(what) != std::string::npos; }

void test_disabled_is_silent() {
    check(!rt::startup_trace::enabled(), "trace is off until open()");
    rt::startup_trace::event("never");  // must not crash with no file
    check(rt::startup_trace::quote("a\"b\\c\n\x01") == "\"a\\\"b\\\\c\\n\\u0001\"", "JSON string escaping");
}

// --trace runs before the loader reads anything, so it must never replace an existing file (an input such as the
// image itself could be named by mistake): open() refuses it with EEXIST and leaves its bytes as they were.
void test_existing_file_is_kept(const fs::path& dir) {
    const std::vector<uint8_t> bytes = {'M', 'Z', 0x90, 0x00, 'k', 'e', 'e', 'p'};
    const fs::path existing = write_file(dir / "existing_input.bin", bytes);
    int error = 0;
    check(!rt::startup_trace::open(existing.c_str(), &error), "open() refuses a file that already exists");
    check(error == EEXIST, "the refusal reports EEXIST");
    check(!rt::startup_trace::enabled(), "the trace stays off after a refused open()");
    std::ifstream in(existing, std::ios::binary);
    const std::vector<uint8_t> after((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    check(after == bytes, "the existing file's bytes survive");
}

// What a strict reader expects for OutputDebugStringA text: the ANSI code page converted to UTF-8.
std::string ansi_as_utf8(const char* a) {
    const int len = static_cast<int>(std::strlen(a));
    const int chars = MultiByteToWideChar(CP_ACP, 0, a, len, nullptr, 0);
    std::wstring w(static_cast<size_t>(chars), L'\0');
    MultiByteToWideChar(CP_ACP, 0, a, len, w.data(), chars);
    const int bytes = WideCharToMultiByte(CP_UTF8, 0, w.data(), chars, nullptr, 0, nullptr, nullptr);
    std::string s(static_cast<size_t>(bytes), '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.data(), chars, s.data(), bytes, nullptr, nullptr);
    return s;
}

bool valid_utf8(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    const std::string all((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return all.empty() ||
           MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, all.data(), static_cast<int>(all.size()), nullptr, 0) > 0;
}

void test_trace(const fs::path& dir) {
    const fs::path image = write_file(dir / kImageName, make_image());
    const fs::path trace = dir / "trace.jsonl";
    check(rt::startup_trace::open(trace.c_str()), "open the trace file");
    {
        rt::Loader loader(dir);
        rt::Module& m = loader.load(image);
        reinterpret_cast<VoidFn>(m.base + caller_rva(0))(nullptr);  // unimplemented import
        reinterpret_cast<VoidFn>(m.base + caller_rva(0))(nullptr);  // second call: no second event
        reinterpret_cast<VoidFn>(m.base + caller_rva(1))(kDebugText);
        reinterpret_cast<VoidFn>(m.base + caller_rva(1))(kAnsiText);
        raise_bad_dbg_print();
        OutputDebugStringA("after_bad_dbg_print");
        OutputDebugStringW(L"wide probe \u00e9\u4e2d");  // its ANSI fallback must not be a second, lossy line
        OutputDebugStringA("after_wide_probe");
        check(call_faulting(m.base + kFaultRva), "the access violation reached the test's handler");
        rt::startup_trace::event("exit", "\"code\":0");
    }
    rt::startup_trace::close();

    const auto lines = read_lines(trace);
    const std::string mod = std::string("\"") + kImageName;
    check(lines.size() >= 5, "the trace has at least 5 lines (got " + std::to_string(lines.size()) + ")");
    bool all_objects = true;
    for (auto& l : lines) all_objects &= !l.empty() && l.front() == '{' && l.back() == '}';
    check(all_objects, "every line is one JSON object");
    check(find_line(lines, "trace_open") == &lines.front(), "trace_open comes first");

    const std::string* map = find_line(lines, "map");
    check(has(map, "\"module\":" + mod + "\"") && has(map, "\"size\":\"0x2000\""), "map names the image and its size");

    int unimplemented = 0;
    for (auto& l : lines) unimplemented += l.find("\"kind\":\"unimplemented_call\"") != std::string::npos;
    const std::string* u = find_line(lines, "unimplemented_call");
    check(unimplemented == 1, "one unimplemented_call for two calls of the same import");
    check(has(u, "\"import\":\"leonida_trace_probe.dll!TraceProbeMissing\""), "unimplemented_call names the import");
    check(has(u, "\"caller\":" + mod + hex_rva(return_rva(0)) + "\""), "unimplemented_call caller is module+RVA");

    const std::string* d = find_line(lines, "debug_string");
    check(has(d, "\"text\":\"leonida trace probe \\\"quoted\\\"\""), "debug_string carries the escaped text");
    const std::string ansi = "\"text\":\"" + ansi_as_utf8(kAnsiText) + "\"";
    bool ansi_found = false;
    for (auto& l : lines) ansi_found |= l.find("\"kind\":\"debug_string\"") != std::string::npos && l.find(ansi) != std::string::npos;
    check(ansi_found, "an ANSI debug string with a byte above 0x7f is written as UTF-8");
    bool after_bad_found = false;
    for (auto& l : lines)
        after_bad_found |=
            l.find("\"kind\":\"debug_string\"") != std::string::npos && l.find("after_bad_dbg_print") != std::string::npos;
    check(after_bad_found, "an SEH fault inside the VEH does not leave in_handler stuck true");
    int wide_lines = 0, wide_utf8 = 0, after_wide = 0;
    for (auto& l : lines) {
        wide_lines += l.find("wide probe") != std::string::npos;
        wide_utf8 += l.find("\"text\":\"wide probe \xc3\xa9\xe4\xb8\xad\"") != std::string::npos;
        after_wide += l.find("\"text\":\"after_wide_probe\"") != std::string::npos;
    }
    check(wide_lines == 1 && wide_utf8 == 1, "OutputDebugStringW gives one debug_string, in UTF-8 (got " +
                                                 std::to_string(wide_lines) + ")");
    check(after_wide == 1, "an OutputDebugStringA after it is still logged");
    check(valid_utf8(trace), "the whole trace is valid UTF-8");

    const std::string* e = find_line(lines, "exception");
    check(has(e, "\"code\":\"0xc0000005\""), "exception code");
    check(has(e, "\"at\":" + mod + hex_rva(kFaultRva) + "\""), "exception address is module+RVA");
    check(has(e, "\"access\":\"read\"") && has(e, "\"address\":\"0x0\""), "access kind and faulting address");
    check(has(e, "\"stack\":[" + mod.substr(0) + hex_rva(kFaultRva)), "stack starts at the faulting instruction");

    check(find_line(lines, "exit") == &lines.back(), "exit comes last");
}

}  // namespace

int wmain() {
    try {
        const fs::path dir = test_dir();
        test_disabled_is_silent();
        test_existing_file_is_kept(dir);
        test_trace(dir);
        std::error_code ec;
        fs::remove_all(dir, ec);
    } catch (const std::exception& e) {
        std::printf("FAIL exception: %s\n", e.what());
        ++g_failed;
    }
    std::printf("%d check(s), %d failed\n", g_checks, g_failed);
    return g_failed ? 1 : 0;
}
