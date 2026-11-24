// End-to-end checks for rt::Loader against locally built fixtures. The default run needs no game files.
//
// The fixtures are an ordinary DLL (tiny_dll) and an EXE that imports it (tiny_exe). The test maps both through the
// public Loader API and checks the mapped headers, exports (by name and ordinal), import binding, recursive module
// loading/dedupe, section protections and forced base relocation. Negative inputs must throw std::runtime_error
// without leaving a module behind. The fixtures are passed as argv[1] and argv[2] by CTest. It also checks how the
// loader's reserved range is read back from LEONIDA_BASE_RESERVED (base_reservation.h).
//
// An optional final section reads LEONIDA_GAME_DIR and maps the 16 DLLs a retail package ships, without executing
// any of them or copying any file. It skips when the variable is unset; see test_optional_package.
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "base_reservation.h"
#include "pe_loader.h"

namespace {

namespace fs = std::filesystem;

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

const char* binding_name(rt::Binding b) {
    switch (b) {
        case rt::Binding::Shim: return "Shim";
        case rt::Binding::Host: return "Host";
        case rt::Binding::Module: return "Module";
        default: return "Unimplemented";
    }
}

std::vector<uint8_t> read_file(const fs::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path.string());
    return {std::istreambuf_iterator<char>(f), {}};
}

void write_file(const fs::path& path, const void* data, size_t size) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("cannot create " + path.string());
    f.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
    if (!f) throw std::runtime_error("cannot write " + path.string());
}

struct ImageInfo {
    uint64_t image_base = 0;
    uint32_t size_of_image = 0;
    uint32_t entry_rva = 0;
};

ImageInfo read_image_info(const fs::path& path) {
    const auto bytes = read_file(path);
    if (bytes.size() < sizeof(IMAGE_DOS_HEADER)) throw std::runtime_error("fixture is too small");
    const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(bytes.data());
    if (dos->e_magic != IMAGE_DOS_SIGNATURE) throw std::runtime_error("fixture has no MZ header");
    const size_t nt_offset = static_cast<size_t>(dos->e_lfanew);
    if (nt_offset + sizeof(IMAGE_NT_HEADERS64) > bytes.size()) throw std::runtime_error("fixture has no NT header");
    const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(bytes.data() + nt_offset);
    ImageInfo info;
    info.image_base = nt->OptionalHeader.ImageBase;
    info.size_of_image = nt->OptionalHeader.SizeOfImage;
    info.entry_rva = nt->OptionalHeader.AddressOfEntryPoint;
    return info;
}

void patch_machine(std::vector<uint8_t>& bytes, uint16_t machine) {
    auto* dos = reinterpret_cast<IMAGE_DOS_HEADER*>(bytes.data());
    auto* nt = reinterpret_cast<IMAGE_NT_HEADERS64*>(bytes.data() + dos->e_lfanew);
    nt->FileHeader.Machine = machine;
}

struct SectionView {
    std::string name;
    uint32_t rva = 0;
};

std::vector<SectionView> mapped_sections(const rt::Module& m) {
    std::vector<SectionView> out;
    const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(m.base);
    const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(m.base + dos->e_lfanew);
    const auto* sec = IMAGE_FIRST_SECTION(nt);
    for (unsigned i = 0; i < nt->FileHeader.NumberOfSections; ++i, ++sec) {
        SectionView view;
        char name[9] = {};
        std::memcpy(name, sec->Name, 8);
        view.name = name;
        view.rva = sec->VirtualAddress;
        out.push_back(std::move(view));
    }
    return out;
}

const SectionView* find_section(const std::vector<SectionView>& sections, const char* name) {
    for (const auto& s : sections)
        if (s.name == name) return &s;
    return nullptr;
}

uint32_t export_ordinal(const rt::Module& m, const char* wanted) {
    const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(m.base);
    const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(m.base + dos->e_lfanew);
    const auto& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT];
    if (!dir.Size) return 0;
    const auto* ed = reinterpret_cast<const IMAGE_EXPORT_DIRECTORY*>(m.base + dir.VirtualAddress);
    const auto* names = reinterpret_cast<const uint32_t*>(m.base + ed->AddressOfNames);
    const auto* ordinals = reinterpret_cast<const uint16_t*>(m.base + ed->AddressOfNameOrdinals);
    for (uint32_t i = 0; i < ed->NumberOfNames; ++i) {
        if (std::strcmp(reinterpret_cast<const char*>(m.base + names[i]), wanted) == 0)
            return ed->Base + ordinals[i];
    }
    return 0;
}

// Finds the IAT slot for one named import. The fixture toolchain emits OriginalFirstThunk, so names are still
// readable after the loader has overwritten FirstThunk with the bound addresses.
uint64_t* find_iat_slot(const rt::Module& m, const std::string& dll_lower, const char* name) {
    const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(m.base);
    const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(m.base + dos->e_lfanew);
    const auto& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    if (!dir.Size) return nullptr;
    for (auto* d = reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR*>(m.base + dir.VirtualAddress); d->Name; ++d) {
        if (rt::lower(reinterpret_cast<const char*>(m.base + d->Name)) != dll_lower) continue;
        if (!d->OriginalFirstThunk) continue;
        auto* lookup = reinterpret_cast<IMAGE_THUNK_DATA64*>(m.base + d->OriginalFirstThunk);
        auto* iat = reinterpret_cast<IMAGE_THUNK_DATA64*>(m.base + d->FirstThunk);
        for (; lookup->u1.AddressOfData; ++lookup, ++iat) {
            if (IMAGE_SNAP_BY_ORDINAL64(lookup->u1.Ordinal)) continue;
            const auto* by_name =
                reinterpret_cast<const IMAGE_IMPORT_BY_NAME*>(m.base + static_cast<uint32_t>(lookup->u1.AddressOfData));
            if (std::strcmp(reinterpret_cast<const char*>(by_name->Name), name) == 0) return &iat->u1.Function;
        }
    }
    return nullptr;
}

int count_imports(const rt::Module& m, const std::string& dll, const char* name, rt::Binding* binding = nullptr) {
    int count = 0;
    for (const auto& r : m.imports) {
        if (rt::lower(r.dll) == dll && r.name == name) {
            ++count;
            if (binding) *binding = r.binding;
        }
    }
    return count;
}

DWORD query_protection(const void* address, bool& ok) {
    MEMORY_BASIC_INFORMATION info{};
    ok = VirtualQuery(address, &info, sizeof(info)) != 0;
    return ok ? info.Protect : 0;
}

bool pointer_inside(const rt::Module& m, const void* p) {
    const auto address = reinterpret_cast<uintptr_t>(p);
    return address >= reinterpret_cast<uintptr_t>(m.base) &&
           address < reinterpret_cast<uintptr_t>(m.base) + m.size;
}

struct TempDir {
    fs::path path;

    TempDir() {
        path = fs::temp_directory_path() / ("leonida_pe_loader_test_" + std::to_string(GetCurrentProcessId()));
        std::error_code ec;
        fs::remove_all(path, ec);
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;
};

void test_fixture_dll(const fs::path& dll) {
    std::printf("\n-- fixture DLL: %s --\n", dll.string().c_str());
    const ImageInfo info = read_image_info(dll);

    rt::Loader loader(dll.parent_path());
    rt::Module& m = loader.load(dll);

    check(m.base != nullptr, "load(dll): base is non-null");
    check(m.size == info.size_of_image, "load(dll): size equals the file's SizeOfImage");
    check(m.size >= 0x1000, "load(dll): size covers at least one page");
    check(m.entry_rva == info.entry_rva && m.entry_rva < m.size,
          "load(dll): entry_rva matches AddressOfEntryPoint and is inside the image");
    check(m.preferred_base == info.image_base, "load(dll): preferred_base matches the file's ImageBase");
    check(loader.modules().size() == 1, "load(dll): exactly one module is mapped");
    std::printf("     base=%p preferred=%p size=%u entry=+%#x relocated=%s\n", static_cast<void*>(m.base),
                reinterpret_cast<void*>(static_cast<uintptr_t>(m.preferred_base)), m.size, m.entry_rva,
                m.relocated ? "yes" : "no");

    const auto add_it = m.exports.find("tiny_add");
    check(add_it != m.exports.end(), "exports contain tiny_add by name");
    if (add_it != m.exports.end()) {
        using AddFn = int(__cdecl*)(int, int);
        const auto add = reinterpret_cast<AddFn>(add_it->second);
        check(add(2, 3) == 5, "tiny_add(2, 3) through the resolved export returns 5");
    }

    const auto pid_it = m.exports.find("tiny_last_pid");
    check(pid_it != m.exports.end(), "exports contain tiny_last_pid by name");
    if (pid_it != m.exports.end()) {
        using LastPidFn = unsigned long(__cdecl*)();
        const auto last_pid = reinterpret_cast<LastPidFn>(pid_it->second);
        check(last_pid() == GetCurrentProcessId(),
              "tiny_add used the host GetCurrentProcessId bound through the IAT");
    }

    const auto value_it = m.exports.find("tiny_exported_value");
    check(value_it != m.exports.end(), "exports contain tiny_exported_value (data) by name");
    if (value_it != m.exports.end()) {
        check(*reinterpret_cast<const int*>(value_it->second) == 1234,
              "tiny_exported_value reads back as 1234");
    }

    const auto text_it = m.exports.find("tiny_text_ptr");
    check(text_it != m.exports.end(), "exports contain tiny_text_ptr (data) by name");
    if (text_it != m.exports.end()) {
        const auto* text = *reinterpret_cast<const char* const*>(text_it->second);
        check(pointer_inside(m, text), "tiny_text_ptr points inside the mapped image");
        if (pointer_inside(m, text))
            check(std::strcmp(text, "leonida tiny fixture") == 0,
                  "tiny_text_ptr points at the fixture string in .rdata");
    }

    const uint32_t ordinal = export_ordinal(m, "tiny_add");
    const std::string ordinal_key = "#" + std::to_string(ordinal);
    check(ordinal != 0, "tiny_add has an ordinal in the export directory");
    const auto ordinal_it = m.exports.find(ordinal_key);
    check(ordinal_it != m.exports.end() && add_it != m.exports.end() && ordinal_it->second == add_it->second,
          "export map has tiny_add's ordinal form pointing at the same address");
    std::printf("     tiny_add ordinal=%u key=%s\n", ordinal, ordinal_key.c_str());

    rt::Binding kernel32_binding = rt::Binding::Unimplemented;
    const int kernel32_matches = count_imports(m, "kernel32.dll", "GetCurrentProcessId", &kernel32_binding);
    check(kernel32_matches == 1, "import list has exactly one kernel32.dll!GetCurrentProcessId");
    check(kernel32_matches == 1 && kernel32_binding == rt::Binding::Host,
          "kernel32.dll!GetCurrentProcessId is bound to the host (Binding::Host)");
    std::printf("     kernel32.dll!GetCurrentProcessId -> %s\n",
                kernel32_matches == 1 ? binding_name(kernel32_binding) : "missing");

    uint64_t* const slot = find_iat_slot(m, "kernel32.dll", "GetCurrentProcessId");
    check(slot != nullptr, "the IAT slot for GetCurrentProcessId is present");
    if (slot) {
        const auto expected =
            reinterpret_cast<uint64_t>(GetProcAddress(GetModuleHandleA("kernel32.dll"), "GetCurrentProcessId"));
        check(expected != 0 && *slot == expected, "the IAT slot holds the host GetCurrentProcessId address");
    }

    const auto sections = mapped_sections(m);
    const SectionView* const rdata = find_section(sections, ".rdata");
    check(rdata != nullptr, "mapped image has a .rdata section");
    if (rdata) {
        bool queried = false;
        const DWORD protection = query_protection(m.base + rdata->rva, queried);
        check(queried, "VirtualQuery succeeds on the .rdata page");
        check(queried && protection == PAGE_READONLY, ".rdata is PAGE_READONLY, not writable or executable");
        std::printf("     .rdata protection=%#lx\n", protection);
    }

    const SectionView* const text = find_section(sections, ".text");
    check(text != nullptr, "mapped image has a .text section");
    if (text) {
        bool queried = false;
        const DWORD protection = query_protection(m.base + text->rva, queried);
        check(queried && protection == PAGE_EXECUTE_READ, ".text is PAGE_EXECUTE_READ (executable but not writable)");
        std::printf("     .text protection=%#lx\n", protection);
    }

    bool headers_queried = false;
    const DWORD headers_protection = query_protection(m.base, headers_queried);
    check(headers_queried && headers_protection == PAGE_READONLY, "PE headers are PAGE_READONLY");
}

void test_recursive_exe(const fs::path& exe, const fs::path& dll) {
    std::printf("\n-- fixture EXE: %s --\n", exe.string().c_str());
    const ImageInfo info = read_image_info(exe);

    rt::Loader loader(exe.parent_path());
    rt::Module& module = loader.load(exe);

    check(module.base != nullptr, "load(exe): base is non-null");
    check(module.size == info.size_of_image, "load(exe): size equals the file's SizeOfImage");
    check(module.entry_rva == info.entry_rva && module.entry_rva < module.size,
          "load(exe): entry_rva matches AddressOfEntryPoint and is inside the image");

    rt::Module* dependency = nullptr;
    for (auto* m : loader.modules()) {
        if (rt::lower(m->name) == rt::lower(dll.filename().string())) {
            check(dependency == nullptr, "only one module named tiny_dll.dll is loaded");
            dependency = m;
        }
    }
    check(dependency != nullptr, "load(exe) recursively maps the imported tiny_dll.dll");
    check(loader.modules().size() == 2, "load(exe) maps exactly the EXE and its tiny DLL");
    const auto& init = loader.initialisation_order();
    check(init.size() == 2 && init[0] == dependency && init[1] == &module,
          "initialisation_order() puts tiny_dll.dll before the EXE that imports it");
    std::printf("     modules=%zu\n", loader.modules().size());

    if (dependency) {
        check(dependency->base != nullptr, "recursive DLL mapping has a non-null base");
        const auto first_base = dependency->base;
        rt::Module& again = loader.load(dll);
        check(&again == dependency, "load(dll) returns the same Module* (module dedupe)");
        check(loader.load(dll).base == first_base, "module dedupe keeps the same base");
        std::printf("     tiny_dll base=%p\n", static_cast<void*>(dependency->base));

        rt::Binding add_binding = rt::Binding::Unimplemented;
        rt::Binding value_binding = rt::Binding::Unimplemented;
        const int add_matches = count_imports(module, "tiny_dll.dll", "tiny_add", &add_binding);
        const int value_matches = count_imports(module, "tiny_dll.dll", "tiny_exported_value", &value_binding);
        check(add_matches == 1 && add_binding == rt::Binding::Module,
              "exe import tiny_dll.dll!tiny_add is bound as Binding::Module");
        check(value_matches == 1 && value_binding == rt::Binding::Module,
              "exe import tiny_dll.dll!tiny_exported_value is bound as Binding::Module");
        std::printf("     tiny_dll.dll!tiny_add -> %s, tiny_dll.dll!tiny_exported_value -> %s\n",
                    add_matches == 1 ? binding_name(add_binding) : "missing",
                    value_matches == 1 ? binding_name(value_binding) : "missing");

        const auto export_it = dependency->exports.find("tiny_add");
        check(export_it != dependency->exports.end(), "the recursively loaded DLL exposes tiny_add");
        uint64_t* const slot = find_iat_slot(module, "tiny_dll.dll", "tiny_add");
        check(slot != nullptr, "the exe has an IAT slot for tiny_dll.dll!tiny_add");
        if (slot && export_it != dependency->exports.end()) {
            check(*slot == reinterpret_cast<uint64_t>(export_it->second),
                  "the exe IAT slot points at the recursively loaded tiny_add");
            using AddFn = int(__cdecl*)(int, int);
            check(reinterpret_cast<AddFn>(*slot)(2, 3) == 5, "calling the EXE's bound tiny_add(2, 3) returns 5");
        }
    }
}

// Loads target while occupied is held, so the preferred base cannot be used, then checks that every kind of
// relocated content still works: code, an exported integer, the relocated data pointer, and the host import.
void check_relocated_load(const fs::path& target, void* occupied, uint32_t expected_size, const char* how) {
    std::printf("     relocation via %s, occupied base=%p\n", how, occupied);
    rt::Loader loader(target.parent_path());
    rt::Module& m = loader.load(target);
    check(m.base != nullptr, "relocated load: base is non-null");
    check(m.relocated, "relocated load: Module::relocated is true");
    check(static_cast<void*>(m.base) != occupied, "relocated load: mapped away from the occupied base");
    check(m.size == expected_size, "relocated load: size still equals SizeOfImage");

    const auto add_it = m.exports.find("tiny_add");
    check(add_it != m.exports.end(), "relocated load: tiny_add is exported");
    if (add_it != m.exports.end()) {
        using AddFn = int(__cdecl*)(int, int);
        check(reinterpret_cast<AddFn>(add_it->second)(2, 3) == 5, "relocated load: tiny_add(2, 3) still returns 5");
    }

    const auto value_it = m.exports.find("tiny_exported_value");
    check(value_it != m.exports.end() && *reinterpret_cast<const int*>(value_it->second) == 1234,
          "relocated load: tiny_exported_value still reads 1234");

    const auto text_it = m.exports.find("tiny_text_ptr");
    check(text_it != m.exports.end(), "relocated load: tiny_text_ptr is exported");
    if (text_it != m.exports.end()) {
        const auto* text = *reinterpret_cast<const char* const*>(text_it->second);
        check(pointer_inside(m, text), "relocated load: tiny_text_ptr was relocated inside the image");
        if (pointer_inside(m, text))
            check(std::strcmp(text, "leonida tiny fixture") == 0,
                  "relocated load: relocated tiny_text_ptr still names the fixture string");
    }

    const auto pid_it = m.exports.find("tiny_last_pid");
    check(pid_it != m.exports.end(), "relocated load: tiny_last_pid is exported");
    if (pid_it != m.exports.end()) {
        using LastPidFn = unsigned long(__cdecl*)();
        check(reinterpret_cast<LastPidFn>(pid_it->second)() == GetCurrentProcessId(),
              "relocated load: the host GetCurrentProcessId import still works");
    }
    std::printf("     relocated base=%p\n", static_cast<void*>(m.base));
}

void test_forced_relocation(const fs::path& dll) {
    std::printf("\n-- forced relocation: %s --\n", dll.string().c_str());
    const ImageInfo info = read_image_info(dll);

    // Hold the fixture's own preferred base, so Loader::load must map the image elsewhere and apply its relocations.
    // If another mapping already owns that address, the loader's own VirtualAlloc at the preferred base will fail
    // for the same reason, so the original image is still loaded relocated.
    void* const hold =
        VirtualAlloc(reinterpret_cast<void*>(static_cast<uintptr_t>(info.image_base)), info.size_of_image,
                     MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    const auto occupied = reinterpret_cast<void*>(static_cast<uintptr_t>(info.image_base));
    if (hold) {
        check(true, "occupied the fixture's preferred base for the relocation test");
        check_relocated_load(dll, hold, info.size_of_image, "reserved preferred base");
        VirtualFree(hold, 0, MEM_RELEASE);
    } else {
        std::printf("     fixture's preferred base is already occupied; loading the original must relocate\n");
        check_relocated_load(dll, occupied, info.size_of_image, "pre-occupied preferred base");
    }
}

void test_rejections(const fs::path& dll, const fs::path& temp_dir) {
    std::printf("\n-- rejected inputs --\n");
    std::vector<uint8_t> not_pe(256, 0x4e);  // 'N', no MZ signature.
    const fs::path not_pe_path = temp_dir / "not_pe.bin";
    write_file(not_pe_path, not_pe.data(), not_pe.size());

    std::vector<uint8_t> truncated(32, 0);  // "MZ" but shorter than IMAGE_DOS_HEADER.
    truncated[0] = 'M';
    truncated[1] = 'Z';
    const fs::path truncated_path = temp_dir / "truncated_mz.bin";
    write_file(truncated_path, truncated.data(), truncated.size());

    auto wrong_machine_bytes = read_file(dll);
    patch_machine(wrong_machine_bytes, IMAGE_FILE_MACHINE_I386);
    const fs::path wrong_machine_path = temp_dir / "wrong_machine.dll";
    write_file(wrong_machine_path, wrong_machine_bytes.data(), wrong_machine_bytes.size());

    rt::Loader loader(temp_dir);
    auto expect_rejection = [&](const fs::path& path, const char* needle, const char* what) {
        bool threw = false;
        std::string message;
        try {
            loader.load(path);
        } catch (const std::runtime_error& e) {
            threw = true;
            message = e.what();
        }
        check(threw, what);
        if (threw) {
            const std::string suffix = std::string(what) + " mentions the reason";
            check(message.find(needle) != std::string::npos, suffix.c_str());
        }
        check(loader.modules().empty(), "a rejected image leaves no module behind");
    };

    expect_rejection(not_pe_path, "not an MZ image", "a buffer without an MZ signature is rejected");
    expect_rejection(truncated_path, "not an MZ image", "a file shorter than the DOS header is rejected");
    expect_rejection(wrong_machine_path, "not an x86-64 PE", "an i386 image is rejected");

    // The same loader must still work after the failures.
    rt::Module& good = loader.load(dll);
    check(good.base != nullptr && loader.modules().size() == 1, "a valid image still loads after the rejected ones");
}

// The image's .pdata as the file declares it.
const IMAGE_DATA_DIRECTORY& exception_directory(const rt::Module& m) {
    const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(m.base);
    const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(m.base + dos->e_lfanew);
    return nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXCEPTION];
}

// Every non-empty entry of a registered table, looked up by the system at its first byte, must come back as itself
// with this module's base. Returns the number of entries that don't.
uint32_t unwind_lookup_mismatches(const rt::Module& m) {
    const auto* table = static_cast<const RUNTIME_FUNCTION*>(m.function_table);
    uint32_t bad = 0;
    for (uint32_t i = 0; i < m.function_count; ++i) {
        if (table[i].BeginAddress == table[i].EndAddress) continue;
        DWORD64 image_base = 0;
        const auto* found =
            RtlLookupFunctionEntry(reinterpret_cast<DWORD64>(m.base) + table[i].BeginAddress, &image_base, nullptr);
        if (found != &table[i] || image_base != reinterpret_cast<DWORD64>(m.base)) ++bad;
    }
    return bad;
}

// Calls the fixture's fault probe with a null pointer: -1 means the image handled it, -2 that the fault escaped to
// here. Without registration the dispatcher can't unwind the image's frame, so the fault escapes or the process dies
// on a corrupt unwind, which ctest reports as a failure too. Kept apart because __try can't share a function with
// objects that have destructors.
int call_faulting(int(__cdecl* fault)(const volatile int*)) {
    __try {
        return fault(nullptr);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return -2;
    }
}

void test_unwind(const fs::path& dll) {
    std::printf("\n-- unwind registration: %s --\n", dll.string().c_str());
    rt::Loader loader(dll.parent_path());
    rt::Module& m = loader.load(dll);

    const auto& dir = exception_directory(m);
    check(dir.Size != 0, "the fixture DLL has an exception directory");
    check(m.function_table == m.base + dir.VirtualAddress && m.function_count == dir.Size / sizeof(RUNTIME_FUNCTION),
          "load registers the whole exception directory");
    check(m.function_count != 0 && unwind_lookup_mismatches(m) == 0,
          "RtlLookupFunctionEntry finds every entry in the mapped image");
    std::printf("     %u unwind entries\n", m.function_count);

    const auto it = m.exports.find("tiny_fault_caught");
    check(it != m.exports.end(), "exports contain tiny_fault_caught");
    if (it == m.exports.end()) return;
    using FaultFn = int(__cdecl*)(const volatile int*);
    const auto fault = reinterpret_cast<FaultFn>(it->second);
    const volatile int value = 42;
    check(fault(&value) == 42, "tiny_fault_caught reads through a valid pointer");
    check(call_faulting(fault) == -1,
          "an access violation inside the mapped image is caught by the image's own handler");
}

// True when the code at rva (or, through up to 4 `jmp rel32` thunks, the code they lead to) has an unwind entry.
// A jump thunk touches no stack, so it legitimately has none: libHttpClient.GDK.dll's entry point is one.
bool unwind_covers(const rt::Module& m, uint32_t rva) {
    for (int hops = 0; hops <= 4; ++hops) {
        DWORD64 image_base = 0;
        if (RtlLookupFunctionEntry(reinterpret_cast<DWORD64>(m.base) + rva, &image_base, nullptr)) return true;
        if (rva > m.size - 5 || m.base[rva] != 0xE9) return false;
        int32_t rel = 0;
        std::memcpy(&rel, m.base + rva + 1, sizeof(rel));
        const int64_t target = static_cast<int64_t>(rva) + 5 + rel;
        if (target < 0 || target >= m.size) return false;
        rva = static_cast<uint32_t>(target);
    }
    return false;
}

// Optional: the unwind tables of whichever package DLLs are in LEONIDA_GAME_DIR. Unlike test_optional_package it
// does not need all 16, so a folder with only some of them is still checked. Nothing in the images is executed.
void test_optional_package_unwind() {
    std::printf("\n-- optional package unwind tables --\n");
    char* env = nullptr;
    size_t env_size = 0;
    if (_dupenv_s(&env, &env_size, "LEONIDA_GAME_DIR") != 0 || !env) {
        std::printf("SKIP (no package): LEONIDA_GAME_DIR is not set\n");
        return;
    }
    const fs::path game_dir = env;
    std::free(env);
    if (!fs::is_directory(game_dir)) {
        std::printf("SKIP: LEONIDA_GAME_DIR is not a directory\n");
        return;
    }
    int found = 0;
    for (const auto& entry : fs::directory_iterator(game_dir)) {
        if (!entry.is_regular_file() || rt::lower(entry.path().extension().string()) != ".dll") continue;
        ++found;
        const std::string name = entry.path().filename().string();
        const std::string label = name + ": its whole exception directory is registered and every entry resolves";
        try {
            rt::Loader loader(game_dir);
            rt::Module& m = loader.load(entry.path());
            const auto& dir = exception_directory(m);
            const uint32_t bad = unwind_lookup_mismatches(m);
            const bool entry_covered = m.entry_rva == 0 || unwind_covers(m, m.entry_rva);
            std::printf("     %-36s unwind=%u mismatches=%u entry covered=%s\n", name.c_str(), m.function_count, bad,
                        entry_covered ? "yes" : "no");
            check(m.function_count == dir.Size / sizeof(RUNTIME_FUNCTION) && m.function_count != 0 && bad == 0 &&
                      entry_covered,
                  label.c_str());
        } catch (const std::exception& e) {
            std::printf("     %-36s load failed: %s\n", name.c_str(), e.what());
            check(false, label.c_str());
        }
    }
    if (!found) std::printf("SKIP: no DLL in LEONIDA_GAME_DIR\n");
}

// Optional integration check over the 16 DLLs the retail package ships. It reads only the file headers, section
// table and import directory: Loader never calls DllMain or an imported function. The package is not part of the
// repository; when LEONIDA_GAME_DIR is unset the test skips cleanly. When it is set, an incomplete directory is an
// error rather than a silent skip.
void test_optional_package() {
    std::printf("\n-- optional package integration --\n");
    char* env = nullptr;
    size_t env_size = 0;
    if (_dupenv_s(&env, &env_size, "LEONIDA_GAME_DIR") != 0 || !env) {
        std::printf("SKIP (no package): LEONIDA_GAME_DIR is not set\n");
        return;
    }
    const fs::path game_dir = env;
    std::free(env);

    if (game_dir.empty() || !fs::is_directory(game_dir)) {
        check(false, "LEONIDA_GAME_DIR names an existing directory");
        std::printf("     not a directory: %s\n", game_dir.string().c_str());
        return;
    }

    constexpr const char* kPackageDlls[] = {
        "Bink2Scarlett.dll",
        "XboxBvhBuilder.dll",
        "amd_fidelityfx_upscaler_xbox.dll",
        "amd_fidelityfx_loader_xbox.dll",
        "libHttpClient.GDK.dll",
        "XCurl.dll",
        "concrt140.dll",
        "msvcp140.dll",
        "msvcp140_1.dll",
        "msvcp140_2.dll",
        "msvcp140_atomic_wait.dll",
        "msvcp140_codecvt_ids.dll",
        "vccorlib140.dll",
        "vcomp140.dll",
        "vcruntime140.dll",
        "vcruntime140_1.dll",
    };
    constexpr size_t kPackageDllCount = sizeof(kPackageDlls) / sizeof(kPackageDlls[0]);
    constexpr uint64_t kPreferredBase = 0x180000000ull;

    size_t missing = 0;
    for (const char* name : kPackageDlls)
        if (!fs::is_regular_file(game_dir / name)) ++missing;
    if (missing) {
        check(false, "LEONIDA_GAME_DIR contains all 16 expected package DLLs");
        std::printf("     %zu of %zu expected DLLs are missing under %s:\n", missing, kPackageDllCount,
                    game_dir.string().c_str());
        for (const char* name : kPackageDlls) {
            const fs::path path = game_dir / name;
            if (!fs::is_regular_file(path)) std::printf("     missing: %s\n", path.string().c_str());
        }
        return;
    }

    std::printf("package: %s\n", game_dir.string().c_str());
    for (const char* name : kPackageDlls) {
        const fs::path path = game_dir / name;
        const std::string label = std::string(name) + ": maps at 0x180000000 with 0 unimplemented imports";
        try {
            rt::Loader loader(game_dir);
            rt::Module& m = loader.load(path);

            std::map<rt::Binding, int> counts;
            for (const auto& import : m.imports) counts[import.binding]++;
            const int shim = counts[rt::Binding::Shim];
            const int host = counts[rt::Binding::Host];
            const int module = counts[rt::Binding::Module];
            const int unimplemented = counts[rt::Binding::Unimplemented];

            std::printf("     %-36s base=%p%s size=%u entry=+%#x  shim=%d host=%d module=%d unimplemented=%d\n",
                        name, static_cast<void*>(m.base), m.relocated ? " (relocated)" : "", m.size, m.entry_rva,
                        shim, host, module, unimplemented);

            const bool ok = m.base != nullptr && m.preferred_base == kPreferredBase &&
                            m.base == reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(kPreferredBase)) &&
                            m.size != 0 && m.entry_rva != 0 && unimplemented == 0;
            check(ok, label.c_str());
        } catch (const std::exception& e) {
            std::printf("     %-36s load failed: %s\n", name, e.what());
            check(false, label.c_str());
        }
    }
}

// The child gta6rt relaunches reads its reserved range back from LEONIDA_BASE_RESERVED (base_reservation.h). Only
// arm64 Wine relaunches, so this is the one place a Windows run parses it.
void test_inherited_reservation() {
    namespace br = rt::base_reservation;
    wchar_t saved[64] = {};
    const DWORD saved_length = GetEnvironmentVariableW(br::kVariable, saved, 64);

    SetEnvironmentVariableW(br::kVariable, L"140000000:30000000");
    const auto r = br::inherited();
    check(r && r->base == 0x140000000ull && r->size == 0x30000000ull,
          "LEONIDA_BASE_RESERVED=140000000:30000000 is base 0x140000000, size 0x30000000");
    SetEnvironmentVariableW(br::kVariable, L"140000000:0");
    check(!br::inherited(), "LEONIDA_BASE_RESERVED with size 0 is no reservation");
    SetEnvironmentVariableW(br::kVariable, L"140000000");
    check(!br::inherited(), "LEONIDA_BASE_RESERVED without a size is no reservation");
    SetEnvironmentVariableW(br::kVariable, L"zz:30000000");
    check(!br::inherited(), "LEONIDA_BASE_RESERVED with a non-hex base is no reservation");
    SetEnvironmentVariableW(br::kVariable, nullptr);
    check(!br::inherited(), "no LEONIDA_BASE_RESERVED is no reservation");

    if (saved_length && saved_length < 64) SetEnvironmentVariableW(br::kVariable, saved);
}

void print_result() {
    if (g_failures == 0)
        std::printf("\npe_loader_test: OK (%d checks)\n", g_checks);
    else
        std::printf("\npe_loader_test: FAILED (%d checks, %d failure%s)\n", g_checks, g_failures,
                    g_failures == 1 ? "" : "s");
}

}  // namespace

int main(int argc, char** argv) {
    std::printf("pe_loader_test: rt::Loader against locally built fixtures (default run needs no game files)\n");
    if (argc != 3) {
        std::fprintf(stderr, "usage: pe_loader_test <tiny_dll.dll> <tiny_exe.exe>\n");
        return 2;
    }
    const fs::path dll = argv[1];
    const fs::path exe = argv[2];

    try {
        check(fs::exists(dll), "fixture DLL exists");
        check(fs::exists(exe), "fixture EXE exists");
        if (!fs::exists(dll) || !fs::exists(exe)) {
            print_result();
            return 1;
        }

        TempDir temp;
        test_fixture_dll(dll);
        test_recursive_exe(exe, dll);
        test_forced_relocation(dll);
        test_rejections(dll, temp.path);
        test_unwind(dll);
        test_inherited_reservation();
        test_optional_package();
        test_optional_package_unwind();
    } catch (const std::exception& e) {
        ++g_checks;
        ++g_failures;
        std::printf("FAIL unexpected exception: %s\n", e.what());
    }

    print_result();
    return g_failures ? 1 : 0;
}
