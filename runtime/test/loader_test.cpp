// Tests for the native PE loader. Every image here is built byte by byte, so nothing from the game is needed and
// nothing is read from the repo: the only files written go to a leonida_loader_test folder in the temporary
// directory, or in LEONIDA_TEST_DIR when that is set (for machines whose security software blocks the images there).
//
// What it pins down: a valid image maps exactly as its headers say, and a truncated or corrupt one is rejected with
// a message instead of being followed out of bounds. The truncation case covers every length of a valid image.
#include <windows.h>
#include <intrin.h>

#include <algorithm>
#include <atomic>
#include <chrono>
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
#include "free_image_base.h"

namespace {

int g_checks = 0;
int g_failed = 0;
unsigned g_files = 0;
std::filesystem::path g_dir;

// Sentinels that the synthesised TLS callback and DllMain stubs write, so a test can see execution happen.
uint32_t g_tls_ran = 0;
uint32_t g_main_ran = 0;

void pass(const char* what) {
    ++g_checks;
    std::printf("ok    %s\n", what);
}

void failed(const char* what, const std::string& why) {
    ++g_checks;
    ++g_failed;
    std::printf("FAIL  %s: %s\n", what, why.c_str());
}

// A minimal but valid x64 image: DOS stub, PE32+ headers, one .text section holding a recognisable byte pattern.
// No imports, exports or relocations unless a case adds them.
std::vector<uint8_t> make_image() {
    std::vector<uint8_t> bytes(0x400, 0);
    auto* dos = reinterpret_cast<IMAGE_DOS_HEADER*>(bytes.data());
    dos->e_magic = IMAGE_DOS_SIGNATURE;
    dos->e_lfanew = 0x80;
    auto* nt = reinterpret_cast<IMAGE_NT_HEADERS64*>(bytes.data() + dos->e_lfanew);
    nt->Signature = IMAGE_NT_SIGNATURE;
    nt->FileHeader.Machine = IMAGE_FILE_MACHINE_AMD64;
    nt->FileHeader.NumberOfSections = 1;
    nt->FileHeader.SizeOfOptionalHeader = sizeof(IMAGE_OPTIONAL_HEADER64);
    nt->OptionalHeader.Magic = IMAGE_NT_OPTIONAL_HDR64_MAGIC;
    nt->OptionalHeader.ImageBase = leonida_test::free_image_base(0x140000000);
    nt->OptionalHeader.AddressOfEntryPoint = 0x1000;
    nt->OptionalHeader.SizeOfHeaders = 0x200;
    nt->OptionalHeader.SizeOfImage = 0x2000;
    nt->OptionalHeader.Subsystem = IMAGE_SUBSYSTEM_WINDOWS_CUI;
    nt->OptionalHeader.NumberOfRvaAndSizes = IMAGE_NUMBEROF_DIRECTORY_ENTRIES;
    auto* sec = IMAGE_FIRST_SECTION(nt);
    std::memcpy(sec->Name, ".text", 6);
    sec->VirtualAddress = 0x1000;
    sec->Misc.VirtualSize = 0x200;
    sec->PointerToRawData = 0x200;
    sec->SizeOfRawData = 0x200;
    sec->Characteristics = IMAGE_SCN_CNT_CODE | IMAGE_SCN_MEM_EXECUTE | IMAGE_SCN_MEM_READ;
    for (int i = 0; i < 0x200; ++i) bytes[0x200 + i] = static_cast<uint8_t>(i);
    return bytes;
}

IMAGE_NT_HEADERS64* nt_of(std::vector<uint8_t>& bytes) {
    return reinterpret_cast<IMAGE_NT_HEADERS64*>(bytes.data() + 0x80);
}

// The section's bytes as they sit in the file, ready to compare against what was mapped.
const uint8_t* text_raw(const std::vector<uint8_t>& bytes) { return bytes.data() + 0x200; }

std::filesystem::path write_image(const std::vector<uint8_t>& bytes) {
    auto p = g_dir / (std::to_string(++g_files) + ".bin");
    std::ofstream out(p, std::ios::binary);
    if (!out) throw std::runtime_error("cannot write " + p.string());
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<long long>(bytes.size()));
    return p;
}

std::filesystem::path write_image_as(const std::vector<uint8_t>& bytes, const std::string& name) {
    auto p = g_dir / name;
    std::ofstream out(p, std::ios::binary);
    if (!out) throw std::runtime_error("cannot write " + p.string());
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<long long>(bytes.size()));
    return p;
}

// An image whose only export, "fwd", forwards to `target`: the address-table entry points at a string inside the
// export directory, which is how a forwarder is encoded.
std::vector<uint8_t> make_forwarder_image(const char* target, uint64_t image_base) {
    auto img = make_image();
    std::fill(img.begin() + 0x200, img.begin() + 0x400, uint8_t{0});  // real tables end in a zero descriptor
    nt_of(img)->OptionalHeader.ImageBase = leonida_test::free_image_base(image_base);
    auto* ed = reinterpret_cast<IMAGE_EXPORT_DIRECTORY*>(img.data() + 0x200);
    ed->Base = 1;
    ed->NumberOfFunctions = 1;
    ed->AddressOfFunctions = 0x1100;
    ed->NumberOfNames = 1;
    ed->AddressOfNames = 0x1120;
    ed->AddressOfNameOrdinals = 0x1140;
    *reinterpret_cast<uint32_t*>(img.data() + 0x300) = 0x1160;  // funcs[0]: inside the export directory
    *reinterpret_cast<uint32_t*>(img.data() + 0x320) = 0x1180;  // the export's name
    *reinterpret_cast<uint16_t*>(img.data() + 0x340) = 0;
    std::memcpy(img.data() + 0x360, target, std::strlen(target) + 1);
    std::memcpy(img.data() + 0x380, "fwd\0", 4);
    auto& dir = nt_of(img)->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT];
    dir.VirtualAddress = 0x1000;
    dir.Size = 0x200;  // covers the forwarder string at RVA 0x1160
    return img;
}

// An image importing `dll`!`symbol` by name.
std::vector<uint8_t> make_import_image(const char* dll, const char* symbol) {
    auto img = make_image();
    std::fill(img.begin() + 0x200, img.begin() + 0x400, uint8_t{0});
    auto* d = reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR*>(img.data() + 0x200);
    d->Name = 0x1060;  // past the zero descriptor that ends the table
    d->OriginalFirstThunk = 0x1100;
    d->FirstThunk = 0x1180;  // the address table the loader fills
    *reinterpret_cast<uint64_t*>(img.data() + 0x300) = 0x1120;  // hint/name
    *reinterpret_cast<uint16_t*>(img.data() + 0x320) = 0;
    std::memcpy(img.data() + 0x322, symbol, std::strlen(symbol) + 1);
    std::memcpy(img.data() + 0x260, dll, std::strlen(dll) + 1);
    auto& dir = nt_of(img)->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    dir.VirtualAddress = 0x1000;
    dir.Size = sizeof(IMAGE_IMPORT_DESCRIPTOR);
    return img;
}

void* host_symbol(const char* dll, const char* name) {
    HMODULE h = GetModuleHandleA(dll);
    return h ? reinterpret_cast<void*>(GetProcAddress(h, name)) : nullptr;
}

// A shim table for the forwarder test, so the test does not depend on the real xmem shim. Registering it at startup
// exercises the same registry a packaged module reaches.
int g_test_shim_target = 0;
void* test_shim_function() { return &g_test_shim_target; }
SHIM_TABLE(test_forwarder_shim, {"testshim.dll", "F", reinterpret_cast<void*>(&test_shim_function)});

// 0 when the image loaded, 1 when it was rejected. A crash here is a test failure, which is the point.
int load(const std::filesystem::path& p, std::string& error) {
    try {
        rt::Loader loader(g_dir);
        loader.load(p);
        return 0;
    } catch (const std::exception& e) {
        error = e.what();
        return 1;
    }
}

void expect_loads(const char* what, const std::vector<uint8_t>& bytes) {
    auto p = write_image(bytes);
    try {
        rt::Loader loader(g_dir);
        rt::Module& m = loader.load(p);
        if (!m.base) return failed(what, "the image mapped to a null base");
        if (m.size != 0x2000) return failed(what, "SizeOfImage came back as " + std::to_string(m.size));
        if (std::memcmp(m.base + 0x1000, text_raw(bytes), 0x200) != 0)
            return failed(what, "the mapped section doesn't match the file");
        pass(what);
    } catch (const std::exception& e) {
        failed(what, std::string("rejected a valid image: ") + e.what());
    }
}

void expect_rejects(const char* what, const std::vector<uint8_t>& bytes) {
    std::string error;
    if (load(write_image(bytes), error) == 0) return failed(what, "the image loaded");
    pass(what);
}

void test_valid_image() { expect_loads("a valid image maps with its section intact", make_image()); }

void test_entry_bounds() {
    for (bool is_dll : {false, true}) {
        for (uint32_t entry : {0u, 0x1000u, 0x1FFFu, 0x2000u, 0x2001u}) {
            auto img = make_image();
            if (is_dll) nt_of(img)->FileHeader.Characteristics |= IMAGE_FILE_DLL;
            nt_of(img)->OptionalHeader.AddressOfEntryPoint = entry;
            const std::string what = std::string(is_dll ? "DLL" : "EXE") + " entry RVA " + std::to_string(entry);
            if (entry < nt_of(img)->OptionalHeader.SizeOfImage) {
                expect_loads(what.c_str(), img);
            } else {
                std::string error;
                if (load(write_image(img), error) == 0)
                    failed(what.c_str(), "the image loaded");
                else if (error.find("the entry point is outside the image") == std::string::npos)
                    failed(what.c_str(), "unexpected error: " + error);
                else
                    pass(what.c_str());
            }
        }
    }
}

void test_header_bounds() {
    {
        auto img = make_image();
        nt_of(img)->OptionalHeader.ImageBase = 0x140000000;
        reinterpret_cast<IMAGE_DOS_HEADER*>(img.data())->e_lfanew = 0x10000;
        expect_rejects("e_lfanew past the end of the file", img);
    }
    {
        auto img = make_image();
        reinterpret_cast<IMAGE_DOS_HEADER*>(img.data())->e_lfanew = -1;
        expect_rejects("a negative e_lfanew", img);
    }
    {
        auto img = make_image();
        img.resize(0x80 + 0x40);  // cut inside the NT headers
        expect_rejects("NT headers truncated by the file size", img);
    }
    {
        auto img = make_image();
        nt_of(img)->OptionalHeader.SizeOfImage = 0;
        expect_rejects("SizeOfImage of 0", img);
    }
    {
        auto img = make_image();
        nt_of(img)->OptionalHeader.SizeOfHeaders = 0x4000;
        expect_rejects("SizeOfHeaders larger than SizeOfImage", img);
    }
    {
        auto img = make_image();
        nt_of(img)->FileHeader.SizeOfOptionalHeader = 0;
        expect_rejects("an optional header too small for PE32+", img);
    }
    {
        // A PE32 image: the magic says 32-bit, so reading it through a PE32+ header would be wrong.
        auto img = make_image();
        nt_of(img)->OptionalHeader.Magic = IMAGE_NT_OPTIONAL_HDR32_MAGIC;
        expect_rejects("a PE32 image in place of PE32+", img);
    }
    {
        auto img = make_image();
        nt_of(img)->FileHeader.Machine = IMAGE_FILE_MACHINE_I386;
        expect_rejects("an i386 machine type", img);
    }
    {
        auto img = make_image();
        nt_of(img)->FileHeader.NumberOfSections = 200;  // section table past the end of the file
        expect_rejects("a section table past the end of the file", img);
    }
}

void test_section_bounds() {
    {
        auto img = make_image();
        IMAGE_FIRST_SECTION(nt_of(img))->VirtualAddress = 0x1F00;  // 0x1F00 + 0x200 > SizeOfImage
        expect_rejects("a section that would be written past the image", img);
    }
    {
        auto img = make_image();
        IMAGE_FIRST_SECTION(nt_of(img))->PointerToRawData = 0x3F0;  // 0x3F0 + 0x200 > file size
        expect_rejects("a section that reads past the end of the file", img);
    }
}

void test_truncation() {
    // Every length of a valid image: no length below the full one may load, and none may crash.
    auto full = make_image();
    for (size_t len = 0; len < full.size(); ++len) {
        std::vector<uint8_t> img(full.begin(), full.begin() + static_cast<long long>(len));
        std::string error;
        if (load(write_image(img), error) == 0)
            return failed("every truncation of a valid image is rejected",
                          "a " + std::to_string(len) + "-byte image loaded");
    }
    pass("every truncation of a valid image is rejected");
}

void test_relocations() {
    // ImageBase 0 makes the loader relocate, which is what exercises the table walk.
    auto relocated = make_image();
    auto* nt = nt_of(relocated);
    nt->OptionalHeader.ImageBase = 0;
    auto& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_BASERELOC];
    dir.VirtualAddress = 0x1000;
    dir.Size = 0x0A;  // header (8) + one DIR64 entry (2)
    auto* block = reinterpret_cast<IMAGE_BASE_RELOCATION*>(relocated.data() + 0x200);
    block->VirtualAddress = 0x1000;
    block->SizeOfBlock = 0x0A;
    auto* entries = reinterpret_cast<uint16_t*>(block + 1);
    entries[0] = IMAGE_REL_BASED_DIR64 << 12;  // type DIR64, offset 0
    {
        // A correct table has to keep working: the pointer at 0x1000 moves by the difference from ImageBase.
        auto img = relocated;
        auto qword = [&] {
            uint64_t v;
            std::memcpy(&v, img.data() + 0x200, sizeof(v));
            return v;
        }();
        auto p = write_image(img);
        try {
            rt::Loader loader(g_dir);
            rt::Module& m = loader.load(p);
            uint64_t moved;
            std::memcpy(&moved, m.base + 0x1000, sizeof(moved));
            int64_t delta = reinterpret_cast<int64_t>(m.base) - 0;
            if (moved != qword + static_cast<uint64_t>(delta))
                failed("a valid relocation is applied", "the pointer wasn't moved by the load delta");
            else
                pass("a valid relocation is applied");
        } catch (const std::exception& e) {
            failed("a valid relocation is applied", e.what());
        }
    }
    {
        auto img = relocated;
        auto* d = &nt_of(img)->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_BASERELOC];
        d->VirtualAddress = 0x1FF8;  // the table itself runs past the image
        expect_rejects("a relocation table past the image", img);
    }
    {
        auto img = relocated;
        auto* b = reinterpret_cast<IMAGE_BASE_RELOCATION*>(img.data() + 0x200);
        b->SizeOfBlock = 2;  // smaller than its own header
        expect_rejects("a relocation block shorter than its header", img);
    }
    {
        auto img = relocated;
        auto* b = reinterpret_cast<IMAGE_BASE_RELOCATION*>(img.data() + 0x200);
        b->VirtualAddress = 0x1FFA;  // the entry's 8 bytes end past the image
        expect_rejects("a relocation entry past the image", img);
    }
}

void test_exports() {
    {
        auto img = make_image();
        auto& dir = nt_of(img)->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT];
        dir.VirtualAddress = 0x1FF0;
        dir.Size = 0x28;
        expect_rejects("an export directory past the image", img);
    }
    {
        // A directory whose address table runs off the end of the image.
        auto img = make_image();
        auto* ed = reinterpret_cast<IMAGE_EXPORT_DIRECTORY*>(img.data() + 0x200);
        ed->NumberOfFunctions = 4;
        ed->AddressOfFunctions = 0x1FF8;
        ed->NumberOfNames = 0;
        auto& dir = nt_of(img)->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT];
        dir.VirtualAddress = 0x1000;
        dir.Size = sizeof(IMAGE_EXPORT_DIRECTORY);
        expect_rejects("an export address table past the image", img);
    }
    {
        // A name whose ordinal doesn't index the address table.
        auto img = make_image();
        auto* ed = reinterpret_cast<IMAGE_EXPORT_DIRECTORY*>(img.data() + 0x200);
        ed->NumberOfFunctions = 1;
        ed->AddressOfFunctions = 0x1100;
        ed->NumberOfNames = 1;
        ed->AddressOfNames = 0x1120;
        ed->AddressOfNameOrdinals = 0x1140;
        auto* funcs = reinterpret_cast<uint32_t*>(img.data() + 0x300);
        funcs[0] = 0x1000;
        auto* name_rva = reinterpret_cast<uint32_t*>(img.data() + 0x320);
        *name_rva = 0x1160;
        auto* ord = reinterpret_cast<uint16_t*>(img.data() + 0x340);
        *ord = 7;  // past NumberOfFunctions
        std::memcpy(img.data() + 0x360, "f\0", 2);
        auto& dir = nt_of(img)->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT];
        dir.VirtualAddress = 0x1000;
        dir.Size = sizeof(IMAGE_EXPORT_DIRECTORY);
        expect_rejects("an export name with an ordinal past the address table", img);
    }
}

void test_empty_named_export() {
    const char* what = "an empty named export stays unresolved without hiding valid exports";
    auto img = make_forwarder_image("KERNEL32.GetCurrentProcessId", 0x180000000);
    auto* ed = reinterpret_cast<IMAGE_EXPORT_DIRECTORY*>(img.data() + 0x200);
    ed->NumberOfFunctions = 3;
    ed->NumberOfNames = 3;
    nt_of(img)->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT].Size = 0x1A0;
    auto* funcs = reinterpret_cast<uint32_t*>(img.data() + 0x300);
    funcs[1] = 0;
    funcs[2] = 0x11C0;  // a code export outside the export directory
    auto* names = reinterpret_cast<uint32_t*>(img.data() + 0x320);
    names[0] = 0x1188;
    names[1] = 0x1180;
    names[2] = 0x1190;
    auto* ords = reinterpret_cast<uint16_t*>(img.data() + 0x340);
    ords[0] = 1;
    ords[1] = 0;
    ords[2] = 2;
    std::memcpy(img.data() + 0x388, "empty\0", 6);
    std::memcpy(img.data() + 0x390, "valid\0", 6);
    auto provider = write_image_as(img, "empty_exports.dll");
    try {
        rt::Loader loader(g_dir);
        rt::Module& m = loader.load(provider);
        for (const char* sym : {"empty", "#2"}) {
            bool owns = false;
            if (m.exports.count(sym) || m.forwarders.count(sym) ||
                loader.resolve_module_export(m.base, sym, owns) || !owns)
                return failed(what, std::string(sym) + " did not remain absent from the mapped module");
        }
        for (const char* sym : {"valid", "#3", "fwd", "#1"}) {
            bool owns = false;
            void* expected = (std::strcmp(sym, "valid") == 0 || std::strcmp(sym, "#3") == 0)
                ? m.base + 0x11C0 : host_symbol("kernel32.dll", "GetCurrentProcessId");
            if (!expected || loader.resolve_module_export(m.base, sym, owns) != expected || !owns)
                return failed(what, std::string(sym) + " no longer resolves to its valid target");
        }
        auto importer = write_image_as(make_import_image("empty_exports.dll", "empty"), "empty_import.exe");
        rt::Module& importing = loader.load(importer);
        if (importing.imports.size() != 1 || importing.imports[0].binding != rt::Binding::Unimplemented)
            return failed(what, "an import of the empty slot did not bind as unimplemented");
        void* bound = *reinterpret_cast<void**>(importing.base + 0x1180);
        if (!bound || bound == m.base)
            return failed(what, "the empty import was not given a failure thunk");
        pass(what);
    } catch (const std::exception& e) {
        failed(what, e.what());
    }
}

void test_imports() {
    {
        auto img = make_image();
        auto& dir = nt_of(img)->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
        dir.VirtualAddress = 0x1FF0;
        dir.Size = sizeof(IMAGE_IMPORT_DESCRIPTOR);
        expect_rejects("an import table past the image", img);
    }
    {
        // A descriptor whose DLL name sits past the end of the allocation.
        auto img = make_image();
        auto* d = reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR*>(img.data() + 0x200);
        d->Name = 0x2000;  // exactly SizeOfImage: no byte of the name is inside the image
        d->FirstThunk = 0;
        d->OriginalFirstThunk = 0;
        auto& dir = nt_of(img)->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
        dir.VirtualAddress = 0x1000;
        dir.Size = sizeof(IMAGE_IMPORT_DESCRIPTOR);
        expect_rejects("an imported DLL name past the image", img);
    }
    {
        // A thunk table with no terminating zero entry: the walk has to stop at the end of the image.
        auto img = make_image();
        nt_of(img)->OptionalHeader.SizeOfImage = 0x1200;  // the image ends where the section does
        auto* d = reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR*>(img.data() + 0x200);
        std::memcpy(img.data() + 0x220, "xmem.dll\0", 9);
        d->Name = 0x1020;
        d->FirstThunk = 0x11F8;  // room for one entry before the image ends
        d->OriginalFirstThunk = 0x11F8;
        auto* thunk = reinterpret_cast<IMAGE_THUNK_DATA64*>(img.data() + 0x3F8);  // file offset of RVA 0x11F8
        thunk->u1.AddressOfData = 0x140;  // a name RVA that is also not where a hint sits
        auto& dir = nt_of(img)->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
        dir.VirtualAddress = 0x1000;
        dir.Size = sizeof(IMAGE_IMPORT_DESCRIPTOR);
        expect_rejects("an import thunk table that runs past the image", img);
    }
}

void test_import_name_rvas() {
    for (bool delay : {false, true}) {
        auto img = make_import_image("testshim.dll", "F");
        if (delay) {
            std::fill(img.begin() + 0x200, img.begin() + 0x240, uint8_t{0});
            auto* d = reinterpret_cast<IMAGE_DELAYLOAD_DESCRIPTOR*>(img.data() + 0x200);
            d->Attributes.AllAttributes = 1;
            d->DllNameRVA = 0x1060;
            d->ImportNameTableRVA = 0x1100;
            d->ImportAddressTableRVA = 0x1180;
            auto* nt = nt_of(img);
            nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT] = {};
            auto& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT];
            dir.VirtualAddress = 0x1000;
            dir.Size = 2 * sizeof(IMAGE_DELAYLOAD_DESCRIPTOR);
        }
        const char* valid = delay ? "a valid delay-import name RVA loads" : "a valid import name RVA loads";
        std::string error;
        if (load(write_image(img), error) != 0)
            failed(valid, error);
        else
            pass(valid);

        *reinterpret_cast<uint64_t*>(img.data() + 0x300) = (1ull << 32) | 0x1120;
        const char* invalid = delay ? "a delay-import name RVA with high bits is rejected"
                                    : "an import name RVA with high bits is rejected";
        error.clear();
        if (load(write_image(img), error) == 0)
            failed(invalid, "the image loaded");
        else if (error.find("an import name runs past the image") == std::string::npos)
            failed(invalid, error);
        else
            pass(invalid);
    }
}

void test_delay_imports() {
    {
        auto img = make_image();
        auto& dir = nt_of(img)->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT];
        dir.VirtualAddress = 0x1FF0;  // the descriptor itself runs past the image
        dir.Size = sizeof(IMAGE_DELAYLOAD_DESCRIPTOR);
        expect_rejects("a delay-import table past the image", img);
    }
    {
        // A descriptor whose DLL name sits past the end of the allocation.
        auto img = make_image();
        auto* d = reinterpret_cast<IMAGE_DELAYLOAD_DESCRIPTOR*>(img.data() + 0x200);
        d->DllNameRVA = 0x2000;  // exactly SizeOfImage
        auto& dir = nt_of(img)->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT];
        dir.VirtualAddress = 0x1000;
        dir.Size = sizeof(IMAGE_DELAYLOAD_DESCRIPTOR);
        expect_rejects("a delay-imported DLL name past the image", img);
    }
    {
        // A delay-import thunk table with no terminating zero entry.
        auto img = make_image();
        nt_of(img)->OptionalHeader.SizeOfImage = 0x1200;
        auto* d = reinterpret_cast<IMAGE_DELAYLOAD_DESCRIPTOR*>(img.data() + 0x200);
        std::memcpy(img.data() + 0x220, "xmem.dll\0", 9);
        d->DllNameRVA = 0x1020;
        d->ImportNameTableRVA = 0x11F8;  // one entry fits before the image ends
        d->ImportAddressTableRVA = 0x11F8;
        auto* thunk = reinterpret_cast<IMAGE_THUNK_DATA64*>(img.data() + 0x3F8);  // file offset of RVA 0x11F8
        thunk->u1.AddressOfData = 0xFFFF;  // non-zero, so the walk needs another entry and the image is over
        auto& dir = nt_of(img)->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT];
        dir.VirtualAddress = 0x1000;
        dir.Size = sizeof(IMAGE_DELAYLOAD_DESCRIPTOR);
        expect_rejects("a delay-import thunk table that runs past the image", img);
    }
}

// movabs rax, addr ; mov dword [rax], value ; mov eax, eax_value ; ret
void emit_store_stub(std::vector<uint8_t>& img, size_t off, uint64_t addr, uint32_t value, uint32_t eax_value) {
    size_t n = off;
    img[n++] = 0x48;
    img[n++] = 0xB8;
    for (int i = 0; i < 8; ++i) img[n++] = static_cast<uint8_t>(addr >> (8 * i));
    img[n++] = 0xC7;
    img[n++] = 0x00;
    for (int i = 0; i < 4; ++i) img[n++] = static_cast<uint8_t>(value >> (8 * i));
    img[n++] = 0xB8;
    for (int i = 0; i < 4; ++i) img[n++] = static_cast<uint8_t>(eax_value >> (8 * i));
    img[n++] = 0xC3;
}

// make_image() plus a TLS directory and an entry point, both aimed at real machine code: a stub that stores a
// sentinel in this process, so the test can tell a callback ran rather than only that it was parsed.
std::vector<uint8_t> make_init_image() {
    auto img = make_image();
    auto* nt = nt_of(img);
    nt->FileHeader.Characteristics |= IMAGE_FILE_DLL;
    nt->OptionalHeader.AddressOfEntryPoint = 0x1040;
    const uint64_t base = nt->OptionalHeader.ImageBase;  // the TLS callback addresses below are absolute
    auto va = [base](uint32_t rva) { return base + rva; };
    constexpr size_t text = 0x200;  // file offset of RVA 0x1000
    emit_store_stub(img, text + 0x00, reinterpret_cast<uint64_t>(&g_tls_ran), 1, 0);
    emit_store_stub(img, text + 0x40, reinterpret_cast<uint64_t>(&g_main_ran), 2, 1);
    std::memcpy(img.data() + text + 0x80, "TLS!", 4);
    auto* callbacks = reinterpret_cast<uint64_t*>(img.data() + text + 0x90);
    callbacks[0] = va(0x1000);
    callbacks[1] = 0;
    auto* tls = reinterpret_cast<IMAGE_TLS_DIRECTORY64*>(img.data() + text + 0xB0);
    tls->StartAddressOfRawData = va(0x1080);
    tls->EndAddressOfRawData = va(0x1084);
    tls->AddressOfIndex = va(0x10A0);
    tls->AddressOfCallBacks = va(0x1090);
    tls->SizeOfZeroFill = 8;
    auto& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_TLS];
    dir.VirtualAddress = 0x10B0;
    dir.Size = sizeof(IMAGE_TLS_DIRECTORY64);
    return img;
}

void expect_init_rejects(const char* what, const std::vector<uint8_t>& bytes) {
    auto p = write_image(bytes);
    try {
        rt::Loader loader(g_dir);
        loader.load(p);
        loader.initialise();
        failed(what, "the module initialised");
    } catch (const std::exception&) {
        pass(what);
    }
}

void test_initialise() {
    g_tls_ran = 0;
    g_main_ran = 0;
    auto p = write_image(make_init_image());
    try {
        rt::Loader loader(g_dir);
        rt::Module& m = loader.load(p);
        loader.initialise();
        if (g_tls_ran != 1)
            failed("initialise runs the TLS callback", "the callback never wrote its sentinel");
        else
            pass("initialise runs the TLS callback");
        if (g_main_ran != 2)
            failed("initialise calls DllMain for a DLL", "the entry point never wrote its sentinel");
        else
            pass("initialise calls DllMain for a DLL");
        // __declspec(thread) reads TEB->ThreadLocalStoragePointer[_tls_index] at gs:[58h], which is a different index
        // space from the TlsAlloc slots. The loader writes the OS-assigned module index (reserved through
        // gta6rt_tls_host.dll) into AddressOfIndex and puts the template copy in that TEB slot, so this reads exactly
        // what mapped __declspec(thread) code would.
        const uint32_t index = *reinterpret_cast<const uint32_t*>(m.base + 0x10A0);
        auto** module_slots = reinterpret_cast<void**>(__readgsqword(0x58));
        void* block = index == TLS_OUT_OF_INDEXES || index >= 64 || !module_slots ? nullptr : module_slots[index];
        if (index != m.tls_index)
            failed("the module TLS index is written to AddressOfIndex", "the image and the Module disagree");
        else
            pass("the module TLS index is written to AddressOfIndex");
        if (!block || std::memcmp(block, "TLS!", 4) != 0)
            failed("the TLS template is copied into the module TLS slot", "the TEB slot doesn't hold the template");
        else
            pass("the TLS template is copied into the module TLS slot");
    } catch (const std::exception& e) {
        failed("initialise runs a TLS callback and DllMain", e.what());
    }
}

void test_initialise_rejects() {
    {
        auto img = make_init_image();
        nt_of(img)->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_TLS].VirtualAddress = 0x1FF0;
        expect_init_rejects("a TLS directory past the image is rejected", img);
    }
    {
        auto img = make_init_image();
        auto* callbacks = reinterpret_cast<uint64_t*>(img.data() + 0x290);  // file offset of RVA 0x1090
        callbacks[0] = nt_of(img)->OptionalHeader.ImageBase + 0x9000;         // well past SizeOfImage
        expect_init_rejects("a TLS callback outside the image is rejected", img);
    }
    {
        g_main_ran = 0;
        auto img = make_init_image();
        emit_store_stub(img, 0x240, reinterpret_cast<uint64_t>(&g_main_ran), 3, 0);  // DllMain returns FALSE
        expect_init_rejects("a DllMain that returns FALSE is rejected", img);
    }
}

// The images below jump to host observers so notifications remain observable after the mappings are freed.
// The FLS destructor itself starts in mapped code: leaving it registered would call unmapped memory at exit.
void emit_jump_stub(std::vector<uint8_t>& img, size_t off, uint64_t target) {
    img[off++] = 0x48;
    img[off++] = 0xB8;  // movabs rax, target; jmp rax (preserves the notification arguments)
    std::memcpy(img.data() + off, &target, sizeof(target));
    off += sizeof(target);
    img[off++] = 0xFF;
    img[off] = 0xE0;
}

struct Notification {
    void* base;
    DWORD reason;
    bool tls;
    bool operator==(const Notification&) const = default;
};
std::vector<Notification> g_notifications;
struct LiveImage {
    void* base;
    uint32_t tls_index;
    void* tls_block;
};
std::vector<LiveImage> g_live_modules;
bool g_detach_state_live = true;
bool g_fail_attach = false;
bool g_use_fls = false;
DWORD g_fls_index = FLS_OUT_OF_INDEXES;
int g_fls_callbacks = 0;

// DllMain's lpvReserved for its last DLL_PROCESS_ATTACH, and whether one was seen.
void* g_attach_reserved = nullptr;
bool g_attach_seen = false;

void observe_notification(void* base, DWORD reason, void* reserved, bool tls) {
    g_notifications.push_back({base, reason, tls});
    if (reason == DLL_PROCESS_ATTACH && !tls) {
        g_attach_reserved = reserved;
        g_attach_seen = true;
    }
    // ~Loader unmaps the image after the detach, which is FreeLibrary's case: lpvReserved is NULL there.
    if (reason == DLL_PROCESS_DETACH && reserved) g_detach_state_live = false;
    if (reason != DLL_PROCESS_DETACH) return;
    for (const auto& m : g_live_modules) {
        MEMORY_BASIC_INFORMATION info{};
        if (!VirtualQuery(m.base, &info, sizeof(info)) || info.State != MEM_COMMIT)
            g_detach_state_live = false;
        if (m.tls_index != rt::kNoTlsIndex && m.tls_block) {
            auto** slots = reinterpret_cast<void**>(__readgsqword(0x58));
            if (slots[m.tls_index] != m.tls_block ||
                !VirtualQuery(m.tls_block, &info, sizeof(info)) || info.State != MEM_COMMIT ||
                std::memcmp(m.tls_block, "TLS!", 4) != 0)
                g_detach_state_live = false;
        }
    }
}

void NTAPI observe_tls(void* base, DWORD reason, void* reserved) {
    observe_notification(base, reason, reserved, true);
}

void NTAPI observe_fls(void* value) {
    if (value == reinterpret_cast<void*>(1)) ++g_fls_callbacks;
}

BOOL NTAPI observe_dllmain(void* base, DWORD reason, void* reserved) {
    observe_notification(base, reason, reserved, false);
    if (g_use_fls) {
        if (reason == DLL_PROCESS_ATTACH) {
            g_fls_index = FlsAlloc(reinterpret_cast<PFLS_CALLBACK_FUNCTION>(static_cast<uint8_t*>(base) + 0x1100));
            if (g_fls_index == FLS_OUT_OF_INDEXES || !FlsSetValue(g_fls_index, reinterpret_cast<void*>(1)))
                return FALSE;
        } else if (reason == DLL_PROCESS_DETACH && g_fls_index != FLS_OUT_OF_INDEXES) {
            if (!FlsFree(g_fls_index)) g_detach_state_live = false;
            g_fls_index = FLS_OUT_OF_INDEXES;
        }
    }
    return !(reason == DLL_PROCESS_ATTACH && g_fail_attach);
}

void reset_notifications() {
    g_notifications.clear();
    g_live_modules.clear();
    g_detach_state_live = true;
    g_fail_attach = false;
    g_use_fls = false;
    g_fls_callbacks = 0;
    g_attach_reserved = nullptr;
    g_attach_seen = false;
}

void expect_notifications(const char* what, const std::vector<Notification>& expected) {
    if (g_notifications == expected) pass(what);
    else failed(what, "unexpected TLS/DllMain notification count or order");
}

std::vector<uint8_t> make_lifecycle_image() {
    auto img = make_init_image();
    emit_jump_stub(img, 0x200, reinterpret_cast<uint64_t>(&observe_tls));
    emit_jump_stub(img, 0x240, reinterpret_cast<uint64_t>(&observe_dllmain));
    emit_jump_stub(img, 0x300, reinterpret_cast<uint64_t>(&observe_fls));
    return img;
}

void test_process_detach() {
    // A map-only load must remain free of image-code execution, including at destruction.
    reset_notifications();
    {
        rt::Loader loader(g_dir);
        loader.load(write_image(make_lifecycle_image()));
    }
    expect_notifications("a never-initialised image receives no detach notifications", {});

    for (bool fail_attach : {false, true}) {
        reset_notifications();
        g_fail_attach = fail_attach;
        g_use_fls = true;
        void* base = nullptr;
        bool rejected = false;
        {
            rt::Loader loader(g_dir);
            auto& m = loader.load(write_image(make_lifecycle_image()));
            base = m.base;
            g_live_modules = {{m.base, m.tls_index, m.tls_block}};
            try {
                loader.initialise();
                loader.initialise();  // successful attach is delivered once
            } catch (const std::runtime_error&) {
                rejected = true;
            }
        }
        if (rejected == fail_attach) pass("attach success/failure is preserved");
        else failed("attach success/failure is preserved", "unexpected initialise result");
        expect_notifications(fail_attach ? "a FALSE attach is detached once" : "TLS and DllMain detach once after attach",
                             {{base, DLL_PROCESS_ATTACH, true}, {base, DLL_PROCESS_ATTACH, false},
                              {base, DLL_PROCESS_DETACH, true}, {base, DLL_PROCESS_DETACH, false}});
        if (g_detach_state_live && g_fls_callbacks == 1 && g_fls_index == FLS_OUT_OF_INDEXES)
            pass("detach frees FLS while the callback image and static TLS are live");
        else
            failed("detach frees FLS while the callback image and static TLS are live", "FLS cleanup missing or state freed early");
        // Keep a failing negative-control run from crashing during process exit after reporting the regression.
        if (g_fls_index != FLS_OUT_OF_INDEXES) {
            FlsSetValue(g_fls_index, nullptr);
            FlsFree(g_fls_index);
            g_fls_index = FLS_OUT_OF_INDEXES;
        }
    }
    g_live_modules.clear();
    g_use_fls = false;
}

void test_process_detach_invalid_tls() {
    reset_notifications();
    g_use_fls = true;
    void* base = nullptr;
    {
        rt::Loader loader(g_dir);
        auto& m = loader.load(write_image(make_lifecycle_image()));
        base = m.base;
        loader.initialise();
        // The callback table is writable in a normal PE, but this synthetic image keeps it in executable .text.
        DWORD old_protection = 0;
        if (!VirtualProtect(m.base + 0x1090, sizeof(uint64_t), PAGE_EXECUTE_READWRITE, &old_protection))
            throw std::runtime_error("cannot make the synthetic TLS callback table writable");
        *reinterpret_cast<uint64_t*>(m.base + 0x1090) = reinterpret_cast<uint64_t>(m.base) + m.size;
        DWORD ignored = 0;
        if (!VirtualProtect(m.base + 0x1090, sizeof(uint64_t), old_protection, &ignored))
            throw std::runtime_error("cannot restore the synthetic TLS callback table protection");
    }
    expect_notifications("invalid TLS detach still calls DllMain for cleanup",
                         {{base, DLL_PROCESS_ATTACH, true}, {base, DLL_PROCESS_ATTACH, false},
                          {base, DLL_PROCESS_DETACH, false}});
    if (g_fls_callbacks == 1 && g_fls_index == FLS_OUT_OF_INDEXES)
        pass("invalid TLS detach still frees the mapped FLS callback");
    else
        failed("invalid TLS detach still frees the mapped FLS callback", "FLS cleanup missing");
    reset_notifications();
}

void test_process_detach_without_entry() {
    for (bool is_dll : {false, true}) {
        reset_notifications();
        auto img = make_lifecycle_image();
        if (is_dll) nt_of(img)->OptionalHeader.AddressOfEntryPoint = 0;
        else nt_of(img)->FileHeader.Characteristics &= ~IMAGE_FILE_DLL;
        void* base = nullptr;
        {
            rt::Loader loader(g_dir);
            auto& m = loader.load(write_image(img));
            base = m.base;
            loader.initialise();
        }
        expect_notifications(is_dll ? "a DLL without an entry point still detaches TLS" : "an EXE detaches TLS without calling its entry point",
                             {{base, DLL_PROCESS_ATTACH, true}, {base, DLL_PROCESS_DETACH, true}});
    }
    reset_notifications();
    auto first = make_lifecycle_image();
    auto second = make_image();
    nt_of(second)->OptionalHeader.ImageBase = leonida_test::free_image_base(0x180000000);
    nt_of(second)->FileHeader.Characteristics |= IMAGE_FILE_DLL;
    emit_jump_stub(second, 0x200, reinterpret_cast<uint64_t>(&observe_dllmain));
    void* first_base = nullptr;
    {
        rt::Loader loader(g_dir);
        first_base = loader.load(write_image(first)).base;
        loader.load(write_image(second));
        g_fail_attach = true;
        try {
            loader.initialise();
            failed("partial initialisation rejects the first DLL", "attach succeeded");
        } catch (const std::runtime_error&) {
            pass("partial initialisation rejects the first DLL");
        }
    }
    expect_notifications("partial initialisation does not detach an untouched later DLL",
                         {{first_base, DLL_PROCESS_ATTACH, true}, {first_base, DLL_PROCESS_ATTACH, false},
                          {first_base, DLL_PROCESS_DETACH, true}, {first_base, DLL_PROCESS_DETACH, false}});
    reset_notifications();
}

// DllMain docs (learn.microsoft.com/windows/win32/dlls/dllmain): "If fdwReason is DLL_PROCESS_ATTACH, lpvReserved is
// NULL for dynamic loads and non-NULL for static loads." Loader::load() from the host is the process-start graph (the
// executable and the DLLs it imports), a run-time LoadLibrary of a bundled DLL is a dynamic load.
void test_attach_reserved() {
    reset_notifications();
    try {
        rt::Loader loader(g_dir);
        loader.load(write_image(make_lifecycle_image()));
        loader.initialise();
        if (!g_attach_seen) failed("a static load's DllMain gets a non-NULL lpvReserved", "DllMain was not called");
        else if (!g_attach_reserved) failed("a static load's DllMain gets a non-NULL lpvReserved", "lpvReserved was NULL");
        else pass("a static load's DllMain gets a non-NULL lpvReserved");
    } catch (const std::exception& e) {
        failed("a static load's DllMain gets a non-NULL lpvReserved", e.what());
    }

    reset_notifications();
    try {
        write_image_as(make_lifecycle_image(), "attach_reserved_dynamic.dll");
        rt::Loader loader(g_dir);
        bool handled = false;
        void* h = loader.load_runtime_module("attach_reserved_dynamic.dll", 0, handled);
        if (!handled || !h) failed("a run-time load's DllMain gets a NULL lpvReserved", "the run-time load failed");
        else if (!g_attach_seen) failed("a run-time load's DllMain gets a NULL lpvReserved", "DllMain was not called");
        else if (g_attach_reserved) failed("a run-time load's DllMain gets a NULL lpvReserved", "lpvReserved was non-NULL");
        else pass("a run-time load's DllMain gets a NULL lpvReserved");
    } catch (const std::exception& e) {
        failed("a run-time load's DllMain gets a NULL lpvReserved", e.what());
    }
}

void test_process_detach_dependencies() {
    reset_notifications();
    auto dep = make_forwarder_image("KERNEL32.GetCurrentProcessId", 0x180000000);
    auto parent = make_import_image("detach_dep.dll", "fwd");
    for (auto* img : {&dep, &parent}) {
        nt_of(*img)->FileHeader.Characteristics |= IMAGE_FILE_DLL;
        nt_of(*img)->OptionalHeader.AddressOfEntryPoint = 0x11C0;
        emit_jump_stub(*img, 0x3C0, reinterpret_cast<uint64_t>(&observe_dllmain));
    }
    write_image_as(dep, "detach_dep.dll");
    void* parent_base = nullptr;
    void* dep_base = nullptr;
    {
        rt::Loader loader(g_dir);
        auto& m = loader.load(write_image_as(parent, "detach_parent.dll"));
        parent_base = m.base;
        for (const auto* mod : loader.modules())
            g_live_modules.push_back({mod->base, mod->tls_index, mod->tls_block});
        dep_base = loader.initialisation_order().front()->base;
        loader.initialise();
        loader.initialise();
    }
    expect_notifications("dependencies attach first and detach last",
                         {{dep_base, DLL_PROCESS_ATTACH, false}, {parent_base, DLL_PROCESS_ATTACH, false},
                          {parent_base, DLL_PROCESS_DETACH, false}, {dep_base, DLL_PROCESS_DETACH, false}});
    if (g_detach_state_live) pass("all dependency mappings remain live until notifications finish");
    else failed("all dependency mappings remain live until notifications finish", "a dependency was unmapped early");
    g_live_modules.clear();
}

// A dependency whose import table is corrupt fails while the parent binds it. Neither must stay findable by name:
// a later load or GetModuleHandle would otherwise get a mapped image whose imports were never bound.
void test_failed_dependency_not_registered() {
    const char* what = "a failed load leaves neither the module nor its failed dependency loaded";
    auto dep = make_import_image("kernel32.dll", "GetCurrentProcessId");
    reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR*>(dep.data() + 0x200)->Name = 0x5000;  // past SizeOfImage
    const auto dep_path = write_image_as(dep, "fail_dep.dll");
    const auto parent_path = write_image_as(make_import_image("fail_dep.dll", "x"), "fail_parent.dll");
    rt::Loader loader(g_dir);
    auto throws = [&](const std::filesystem::path& p) {
        try { loader.load(p); return false; } catch (const std::exception&) { return true; }
    };
    if (!throws(parent_path)) return failed(what, "the parent loaded over a corrupt dependency");
    if (loader.find_loaded_module("fail_dep.dll") || loader.find_loaded_module("fail_parent.dll"))
        return failed(what, "a failed module is still found by name");
    if (!throws(dep_path)) return failed(what, "a second load of the dependency returned the half-loaded module");
    if (!throws(parent_path)) return failed(what, "a second load of the parent returned the half-loaded module");
    pass(what);
}

void test_forwarded_exports() {
    {
        const char* what = "a forwarded export binds to its target instead of into the string";
        write_image_as(make_forwarder_image("KERNEL32.GetCurrentProcessId", 0x180000000), "prov.dll");
        auto importer = write_image_as(make_import_image("prov.dll", "fwd"), "importer.exe");
        try {
            rt::Loader loader(g_dir);
            rt::Module& m = loader.load(importer);
            void* expected = host_symbol("kernel32.dll", "GetCurrentProcessId");
            if (!expected) return failed(what, "GetCurrentProcessId is not loaded in this process");
            if (m.imports.size() != 1)
                return failed(what, "the importer has " + std::to_string(m.imports.size()) + " import record(s)");
            if (m.imports[0].binding != rt::Binding::Host)
                return failed(what, "the import did not bind through the forwarder to the host");
            if (*reinterpret_cast<void**>(m.base + 0x1180) != expected)
                return failed(what, "the address table does not hold the forwarded target");
            rt::Module* prov = nullptr;
            for (auto* mod : loader.modules())
                if (mod->name == "prov.dll") prov = mod;
            if (!prov) return failed(what, "the provider module was not loaded");
            if (prov->exports.count("fwd")) return failed(what, "the forwarder was stored as a code export");
            auto it = prov->forwarders.find("fwd");
            if (it == prov->forwarders.end() || it->second != "KERNEL32.GetCurrentProcessId")
                return failed(what, "the forwarder string was not recorded");
            pass(what);
        } catch (const std::exception& e) {
            failed(what, e.what());
        }
    }
    {
        const char* what = "a forwarder to a module that isn't there binds as unimplemented";
        write_image_as(make_forwarder_image("NOPE_MISSING_TARGET.f", 0x180000000), "prov.dll");
        auto importer = write_image_as(make_import_image("prov.dll", "fwd"), "importer.exe");
        try {
            rt::Loader loader(g_dir);
            rt::Module& m = loader.load(importer);
            if (m.imports.size() != 1 || m.imports[0].binding != rt::Binding::Unimplemented)
                return failed(what, "the import did not end up unimplemented");
            if (!*reinterpret_cast<void**>(m.base + 0x1180))
                return failed(what, "the address table was left null instead of a failure thunk");
            pass(what);
        } catch (const std::exception& e) {
            failed(what, e.what());
        }
    }
    {
        const char* what = "forwarders that point at each other stop instead of recursing";
        write_image_as(make_forwarder_image("prov_b.fwd", 0x180000000), "prov_a.dll");
        write_image_as(make_forwarder_image("prov_a.fwd", 0x190000000), "prov_b.dll");
        auto importer = write_image_as(make_import_image("prov_a.dll", "fwd"), "importer.exe");
        try {
            rt::Loader loader(g_dir);
            rt::Module& m = loader.load(importer);
            if (m.imports.size() != 1 || m.imports[0].binding != rt::Binding::Unimplemented)
                return failed(what, "a forwarder loop did not end as unimplemented");
            pass(what);
        } catch (const std::exception& e) {
            failed(what, e.what());
        }
    }
    {
        // Forwarder strings name the module without ".dll". The package ships Xbox builds of the CRT
        // (vcruntime140.dll, msvcp140.dll, ...) that the loader replaces with the host's for a direct import; a
        // forwarded one must get the host copy too, not map the bundled file.
        const char* what = "a forwarder to a CRT module without \".dll\" binds the host copy, not the bundled one";
        write_image_as(make_forwarder_image("KERNEL32.GetCurrentProcessId", 0x1a0000000), "vcruntime140.dll");
        write_image_as(make_forwarder_image("vcruntime140.memset", 0x180000000), "prov_crt.dll");
        auto importer = write_image_as(make_import_image("prov_crt.dll", "fwd"), "importer_crt.exe");
        HMODULE host_crt = LoadLibraryExA("vcruntime140.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        void* expected = host_crt ? reinterpret_cast<void*>(GetProcAddress(host_crt, "memset")) : nullptr;
        if (!expected) {
            std::printf("SKIP %s: no vcruntime140.dll in System32\n", what);
        } else {
            try {
                rt::Loader loader(g_dir);
                rt::Module& m = loader.load(importer);
                if (m.imports.size() != 1 || m.imports[0].binding != rt::Binding::Host)
                    return failed(what, "the forwarded CRT import did not bind to the host");
                if (*reinterpret_cast<void**>(m.base + 0x1180) != expected)
                    return failed(what, "the address table does not hold the host's memset");
                for (auto* mod : loader.modules())
                    if (rt::lower(mod->name) == "vcruntime140.dll")
                        return failed(what, "the bundled vcruntime140.dll was mapped");
                pass(what);
            } catch (const std::exception& e) {
                failed(what, e.what());
            }
        }
    }
    {
        const char* what = "a forwarder to a shimmed module binds to the shim";
        write_image_as(make_forwarder_image("testshim.F", 0x180000000), "prov_shim.dll");
        auto importer = write_image_as(make_import_image("prov_shim.dll", "fwd"), "importer_shim.exe");
        try {
            rt::Loader loader(g_dir);
            rt::Module& m = loader.load(importer);
            if (m.imports.size() != 1 || m.imports[0].binding != rt::Binding::Shim)
                return failed(what, "the import did not bind as a shim");
            if (*reinterpret_cast<void**>(m.base + 0x1180) != reinterpret_cast<void*>(&test_shim_function))
                return failed(what, "the address table does not hold the shim function");
            pass(what);
        } catch (const std::exception& e) {
            failed(what, e.what());
        }
    }
}

void test_delay_descriptor_fields() {
    {
        // Without dlattrRva the fields are 32-bit virtual addresses, which cannot address an x64 image.
        auto img = make_image();
        std::fill(img.begin() + 0x200, img.begin() + 0x400, uint8_t{0});
        auto* d = reinterpret_cast<IMAGE_DELAYLOAD_DESCRIPTOR*>(img.data() + 0x200);
        d->Attributes.AllAttributes = 0;
        d->DllNameRVA = 0x1060;  // past the zero descriptor that ends the table
        d->ImportAddressTableRVA = 0x1180;
        std::memcpy(img.data() + 0x260, "kernel32.dll\0", 13);
        auto& dir = nt_of(img)->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT];
        dir.VirtualAddress = 0x1000;
        dir.Size = sizeof(IMAGE_DELAYLOAD_DESCRIPTOR);
        expect_rejects("a delay-import descriptor without the RVA attribute", img);
    }
    {
        // Without a name table the IAT still holds delay thunks, not import names, so there is no symbol to bind.
        auto img = make_image();
        std::fill(img.begin() + 0x200, img.begin() + 0x400, uint8_t{0});
        auto* d = reinterpret_cast<IMAGE_DELAYLOAD_DESCRIPTOR*>(img.data() + 0x200);
        d->Attributes.AllAttributes = 1;  // dlattrRva
        d->DllNameRVA = 0x1060;  // past the zero descriptor that ends the table
        d->ImportNameTableRVA = 0;
        d->ImportAddressTableRVA = 0x1180;
        std::memcpy(img.data() + 0x260, "kernel32.dll\0", 13);
        auto& dir = nt_of(img)->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT];
        dir.VirtualAddress = 0x1000;
        dir.Size = sizeof(IMAGE_DELAYLOAD_DESCRIPTOR);
        expect_rejects("a delay-import descriptor without a name table", img);
    }
}

// make_image() plus an exception directory at RVA 0x1100 whose entries all share one minimal UNWIND_INFO at 0x1180
// (version 1, no flags, no prologue, no codes). Every RVA here lies inside the single .text section, so file offset
// = RVA - 0x1000 + 0x200.
constexpr uint32_t kPdataRva = 0x1100;
constexpr uint32_t kUnwindRva = 0x1180;

std::vector<uint8_t> make_unwind_image(const std::vector<RUNTIME_FUNCTION>& entries) {
    auto img = make_image();
    std::memset(img.data() + 0x200 + (kPdataRva - 0x1000), 0, 0x100);
    const uint8_t unwind_info[4] = {1, 0, 0, 0};
    std::memcpy(img.data() + 0x200 + (kUnwindRva - 0x1000), unwind_info, sizeof(unwind_info));
    std::memcpy(img.data() + 0x200 + (kPdataRva - 0x1000), entries.data(), entries.size() * sizeof(RUNTIME_FUNCTION));
    auto& dir = nt_of(img)->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXCEPTION];
    dir.VirtualAddress = kPdataRva;
    dir.Size = static_cast<DWORD>(entries.size() * sizeof(RUNTIME_FUNCTION));
    nt_of(img)->OptionalHeader.NumberOfRvaAndSizes = IMAGE_NUMBEROF_DIRECTORY_ENTRIES;
    return img;
}

// PE/COFF: NumberOfRvaAndSizes counts the data directories; the slots past it are not directories whatever they hold.
// With 9 directories the TLS slot (index 9) is not one, so bytes left in it must not make the image fail to load.
void test_tls_directory_count() {
    auto img = make_image();
    auto& tls = nt_of(img)->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_TLS];
    tls.VirtualAddress = 0xFFFF0000;  // would run past the image if it were read as a TLS directory
    tls.Size = sizeof(IMAGE_TLS_DIRECTORY64);
    nt_of(img)->OptionalHeader.NumberOfRvaAndSizes = IMAGE_DIRECTORY_ENTRY_TLS;
    expect_loads("a TLS slot past NumberOfRvaAndSizes is not read as a TLS directory", img);
    nt_of(img)->OptionalHeader.NumberOfRvaAndSizes = IMAGE_NUMBEROF_DIRECTORY_ENTRIES;
    expect_rejects("the same TLS slot within NumberOfRvaAndSizes is checked and refused", img);
}

void test_unwind_registration() {
    const std::vector<RUNTIME_FUNCTION> entries = {{0x1000, 0x1010, kUnwindRva}, {0x1010, 0x1020, kUnwindRva}};
    uintptr_t pc = 0;
    {
        auto img = make_unwind_image(entries);
        auto p = write_image(img);
        try {
            rt::Loader loader(g_dir);
            rt::Module& m = loader.load(p);
            pc = reinterpret_cast<uintptr_t>(m.base) + 0x1014;  // inside the second function
            DWORD64 image_base = 0;
            auto* found = RtlLookupFunctionEntry(pc, &image_base, nullptr);
            if (m.function_count != 2)
                failed("the exception directory is registered", "function_count is " + std::to_string(m.function_count));
            else if (!found)
                failed("the exception directory is registered", "RtlLookupFunctionEntry found nothing");
            else if (image_base != reinterpret_cast<DWORD64>(m.base) ||
                     reinterpret_cast<uint8_t*>(found) != m.base + kPdataRva + sizeof(RUNTIME_FUNCTION))
                failed("the exception directory is registered", "the lookup returned the wrong entry or image base");
            else
                pass("the exception directory is registered");
        } catch (const std::exception& e) {
            failed("the exception directory is registered", e.what());
        }
        DWORD64 image_base = 0;
        if (pc && RtlLookupFunctionEntry(pc, &image_base, nullptr))
            failed("the unwind table is removed with the loader", "the lookup still finds the freed image");
        else
            pass("the unwind table is removed with the loader");
    }
    {
        // The system loader drops a trailing partial entry rather than rejecting the image; so does this one.
        auto img = make_unwind_image(entries);
        nt_of(img)->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXCEPTION].Size += 5;
        auto p = write_image(img);
        try {
            rt::Loader loader(g_dir);
            rt::Module& m = loader.load(p);
            if (m.function_count != 2)
                failed("a trailing partial unwind entry is ignored", "function_count is " + std::to_string(m.function_count));
            else
                pass("a trailing partial unwind entry is ignored");
        } catch (const std::exception& e) {
            failed("a trailing partial unwind entry is ignored", e.what());
        }
    }
    // A zero-length entry appears in a shipped Windows image (symcryptk.dll), so it has to be accepted.
    expect_loads("a zero-length unwind entry is accepted",
                 make_unwind_image({{0x1000, 0x1000, kUnwindRva}, {0x1000, 0x1020, kUnwindRva}}));
    {
        auto img = make_unwind_image(entries);
        nt_of(img)->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXCEPTION].VirtualAddress = 0x1FF0;
        expect_rejects("an exception directory past the image", img);
    }
    expect_rejects("an unwind entry ending past the image",
                   make_unwind_image({{0x1000, 0x2001, kUnwindRva}}));
    expect_rejects("an unwind entry that ends before it starts",
                   make_unwind_image({{0x1010, 0x1000, kUnwindRva}}));
    expect_rejects("unwind entries out of order",
                   make_unwind_image({{0x1010, 0x1020, kUnwindRva}, {0x1000, 0x1010, kUnwindRva}}));
    expect_rejects("overlapping unwind entries",
                   make_unwind_image({{0x1000, 0x1018, kUnwindRva}, {0x1010, 0x1020, kUnwindRva}}));
    expect_rejects("unwind data past the image", make_unwind_image({{0x1000, 0x1010, 0x1FFE}}));
}

// An export whose address-table entry is past SizeOfImage (not a forwarder: outside the export directory) is
// refused instead of being stored as base + rva, a pointer outside the mapping.
// The address-to-module lookup the shim registry uses (on every C++ throw from a mapped image, and for trace and
// stop lines) runs on whatever thread asks, while another thread may be inside load(). It used to walk the Loader's
// module vector with no lock while load() appended to it. Without the fix this fails for certain only under
// AddressSanitizer; here it checks that every answer given during 300 loads is right.
void test_lookup_during_loads() {
    const char* what = "the address lookup is right while another thread loads modules";
    try {
        rt::Loader loader(g_dir);
        auto first = make_image();
        auto* first_nt = reinterpret_cast<IMAGE_NT_HEADERS64*>(first.data() + 0x80);
        first_nt->OptionalHeader.ImageBase = 0x200000000ull;
        rt::Module& anchor = loader.load(write_image(first));
        const uint64_t inside = reinterpret_cast<uint64_t>(anchor.base) + 0x1004;

        std::atomic<bool> stop{false};
        std::atomic<unsigned> lookups{0}, wrong{0};
        std::thread reader([&] {
            while (!stop.load()) {
                uint64_t base = 0;
                if (!rt::shims::mapped_image_base(inside, base) || base != reinterpret_cast<uint64_t>(anchor.base))
                    ++wrong;
                if (rt::shims::module_offset(inside).find("+0x1004") == std::string::npos) ++wrong;
                uint64_t none = 0;
                if (rt::shims::mapped_image_base(0x10, none)) ++wrong;  // in no image
                ++lookups;
            }
        });
        unsigned loaded = 0;
        for (unsigned i = 1; i <= 300; ++i) {
            auto img = make_image();
            auto* nt = reinterpret_cast<IMAGE_NT_HEADERS64*>(img.data() + 0x80);
            nt->OptionalHeader.ImageBase = 0x200000000ull + i * 0x10000ull;
            rt::Module& m = loader.load(write_image(img));
            uint64_t base = 0;
            if (rt::shims::mapped_image_base(reinterpret_cast<uint64_t>(m.base) + 0x1000, base) &&
                base == reinterpret_cast<uint64_t>(m.base))
                ++loaded;
        }
        while (lookups.load() < 1000) std::this_thread::yield();
        stop = true;
        reader.join();
        if (wrong.load())
            failed(what, std::to_string(wrong.load()) + " wrong answers in " + std::to_string(lookups.load()) + " lookups");
        else if (loaded != 300)
            failed(what, "only " + std::to_string(loaded) + " of 300 new modules were found by address");
        else
            pass(what);
    } catch (const std::exception& e) {
        failed(what, e.what());
    }
}

void test_export_bounds() {
    auto outside = make_forwarder_image("KERNEL32.GetCurrentProcessId", 0x180000000);
    nt_of(outside)->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT].Size = sizeof(IMAGE_EXPORT_DIRECTORY);
    *reinterpret_cast<uint32_t*>(outside.data() + 0x300) = 0x2000;  // funcs[0] == SizeOfImage
    expect_rejects("an export whose address is past the image", outside);
    auto last = make_forwarder_image("KERNEL32.GetCurrentProcessId", 0x180000000);
    nt_of(last)->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT].Size = sizeof(IMAGE_EXPORT_DIRECTORY);
    *reinterpret_cast<uint32_t*>(last.data() + 0x300) = 0x1FFF;  // the image's last byte: still inside
    expect_loads("an export at the image's last byte still loads", last);
}

// The folder the images go to: leonida_loader_test\<process id> in LEONIDA_TEST_DIR when that is set, otherwise in
// the temporary directory. A folder per run, because security software can hold the previous run's images for a
// while after they are written (so they can be neither deleted nor overwritten); the fixed parent name is what an
// exclusion names.
std::filesystem::path work_dir() {
    wchar_t* env = nullptr;
    size_t size = 0;
    std::filesystem::path base;
    if (_wdupenv_s(&env, &size, L"LEONIDA_TEST_DIR") == 0 && env && *env) base = env;
    std::free(env);
    if (base.empty()) base = std::filesystem::temp_directory_path();
    return base / "leonida_loader_test" / std::to_string(GetCurrentProcessId());
}

// Security software can scan each file that looks like an executable as it is written, and hold it while it does:
// measured with Trend Micro Maximum Security, opening such a file afterwards fails with "access denied" for a while
// (50 to 800 ms each), plain files are unaffected, and an excluded folder takes about 2 ms. Cases then fail at
// random and the truncation case, which writes over a thousand images, stalls for minutes. So the test first writes
// and reads back a few images, and stops with the reason if any is refused or they take too long. Empty when the
// folder behaves.
std::string written_images_blocked() {
    constexpr int probes = 8;
    constexpr auto slow = std::chrono::milliseconds(25);  // per image, on average
    const std::vector<uint8_t> image = make_image();
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < probes; ++i) {
        const auto path = write_image_as(image, "readback_check_" + std::to_string(i) + ".bin");
        const HANDLE file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                        FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file == INVALID_HANDLE_VALUE)
            return "opening one failed with Windows error " + std::to_string(GetLastError());
        std::vector<uint8_t> back(image.size() + 1);
        DWORD read = 0;
        const BOOL ok = ReadFile(file, back.data(), DWORD(back.size()), &read, nullptr);
        const DWORD error = GetLastError();
        CloseHandle(file);
        if (!ok) return "reading one failed with Windows error " + std::to_string(error);
        if (read != image.size() || !std::equal(image.begin(), image.end(), back.begin()))
            return "one read back as " + std::to_string(read) + " bytes that differ from the " +
                   std::to_string(image.size()) + " written";
    }
    const auto elapsed = std::chrono::steady_clock::now() - start;
    const auto each = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed) / probes;
    // Under Wine there is no such software, and its file I/O alone can average over 25 ms on a CI runner (38 and
    // 40 ms seen in the wine job): the refusals and contents above are still checked, the time isn't.
    const bool wine = GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "wine_get_version") != nullptr;
    if (each > slow && !wine)
        return "writing and reading one back took " + std::to_string(each.count()) + " ms on average (over " +
               std::to_string(slow.count()) + " ms means each is being scanned)";
    return {};
}

}  // namespace

int main() {
    g_dir = work_dir();
    std::error_code ec;
    std::filesystem::remove_all(g_dir, ec);
    std::filesystem::create_directories(g_dir, ec);
    if (ec) {
        std::printf("FAIL  cannot create %s: %s\n", g_dir.string().c_str(), ec.message().c_str());
        return 1;
    }

    try {
        if (const std::string why = written_images_blocked(); !why.empty()) {
            std::printf("FAIL  PE images written to %s are not readable straight away: %s.\n"
                        "      Security software that scans newly written executable files does this. Exclude %s\n"
                        "      from scanning, or set LEONIDA_TEST_DIR to a folder it does not scan.\n",
                        g_dir.string().c_str(), why.c_str(), g_dir.parent_path().string().c_str());
            std::filesystem::remove_all(g_dir, ec);
            return 1;
        }
        test_valid_image();
        test_entry_bounds();
        test_header_bounds();
        test_section_bounds();
        test_truncation();
        test_relocations();
        test_exports();
        test_empty_named_export();
        test_imports();
        test_import_name_rvas();
        test_delay_imports();
        test_initialise();
        test_initialise_rejects();
        test_process_detach();
        test_process_detach_invalid_tls();
        test_process_detach_without_entry();
        test_attach_reserved();
        test_process_detach_dependencies();
        test_failed_dependency_not_registered();
        test_delay_descriptor_fields();
        test_forwarded_exports();
        test_export_bounds();
        test_tls_directory_count();
        test_unwind_registration();
        test_lookup_during_loads();
    } catch (const std::exception& e) {
        // A write that fails (write_image throws) ends the run with the reason, instead of terminating the process.
        failed("writing the test images", e.what());
    }

    std::filesystem::remove_all(g_dir, ec);
    if (ec)  // not a failure of the loader; the next run uses a folder of its own anyway
        std::printf("note  could not remove %s: %s\n", g_dir.string().c_str(), ec.message().c_str());
    std::filesystem::remove(g_dir.parent_path(), ec);  // only if no other run's folder is left
    std::printf("\n%d checks, %d failed\n", g_checks, g_failed);
    return g_failed ? 1 : 0;
}
