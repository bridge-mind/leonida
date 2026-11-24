#define _CRT_SECURE_NO_WARNINGS

#include <windows.h>
#include <intrin.h>

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

#if __has_include("../src/pe_loader.h")
#include "../src/pe_loader.h"
#elif __has_include("pe_loader.h")
#include "pe_loader.h"
#endif

#include "free_image_base.h"

#if __has_include("../shims/shims.h")
#include "../shims/shims.h"
#elif __has_include("shims.h")
#include "shims.h"
#endif

namespace {

int g_checks = 0;
int g_failures = 0;
unsigned g_file_seq = 0;
std::filesystem::path g_temp_dir;

void pass(const char* what) {
    ++g_checks;
    std::printf("PASS %s\n", what);
}

void fail(const char* what, const std::string& why) {
    ++g_checks;
    ++g_failures;
    std::printf("FAIL %s: %s\n", what, why.c_str());
}

void check_condition(bool condition, const char* what, const std::string& why_failed = "") {
    if (condition) {
        pass(what);
    } else {
        fail(what, why_failed.empty() ? "condition not met" : why_failed);
    }
}

// ---------------------------------------------------------------------------
// Pure boundary validation logic tests (arithmetic, wrapping, null checks)
// ---------------------------------------------------------------------------

// Replicating require_range semantics to verify arithmetic wrapping invariants directly
bool check_range_safe(uint64_t rva, uint64_t len, uint32_t image_size) {
    if (rva > image_size || len > image_size - rva) {
        return false;
    }
    return true;
}

// Replicating read_string length and null termination validation directly
bool check_string_null_terminated(const char* buffer, size_t buffer_len) {
    return std::memchr(buffer, 0, buffer_len) != nullptr;
}

void test_range_wrapping_arithmetic() {
    std::printf("\n--- Direct Virtual Address Range Wrapping and Boundary Arithmetic ---\n");

    const uint32_t image_size = 0x2000;

    // Normal valid ranges
    check_condition(check_range_safe(0, 0x2000, image_size),
                    "Range covering entire image [0, 0x2000) is valid");
    check_condition(check_range_safe(0x1000, 0x200, image_size),
                    "Range [0x1000, 0x1200) inside image is valid");
    check_condition(check_range_safe(0x2000, 0, image_size),
                    "Zero-length range at exact image end [0x2000, 0x2000) is valid");

    // Boundary overflow without wrapping
    check_condition(!check_range_safe(0x2001, 0, image_size),
                    "RVA 0x2001 past image size is rejected");
    check_condition(!check_range_safe(0x1000, 0x1001, image_size),
                    "Range [0x1000, 0x2001) exceeding image size is rejected");
    check_condition(!check_range_safe(0x2000, 1, image_size),
                    "1-byte range at image end [0x2000, 0x2001) is rejected");

    // 32-bit integer wrapping cases:
    // If evaluated with 32-bit addition (uint32_t)(0xFFFFF000 + 0x2000) == 0x1000 <= 0x2000
    // check_range_safe uses uint64_t and image_size - rva to prevent wrapping bypass.
    const uint64_t wrap_rva_32 = 0xFFFFF000ull;
    const uint64_t wrap_len_32 = 0x2000ull;
    check_condition(!check_range_safe(wrap_rva_32, wrap_len_32, image_size),
                    "32-bit wrapping RVA 0xFFFFF000 + 0x2000 is rejected by rva > image_size");

    const uint64_t max_u32 = 0xFFFFFFFFull;
    check_condition(!check_range_safe(max_u32, 8, image_size),
                    "RVA 0xFFFFFFFF with length 8 is rejected");
    check_condition(!check_range_safe(0x1000, max_u32, image_size),
                    "RVA 0x1000 with length 0xFFFFFFFF is rejected by len > image_size - rva");

    // 64-bit integer wrapping cases:
    // rva + len would overflow 64-bit integer
    const uint64_t max_u64 = 0xFFFFFFFFFFFFFFFFull;
    check_condition(!check_range_safe(max_u64, 1, image_size),
                    "RVA 0xFFFFFFFFFFFFFFFF is rejected");
    check_condition(!check_range_safe(0x1000, max_u64, image_size),
                    "Length 0xFFFFFFFFFFFFFFFF is rejected");
    check_condition(!check_range_safe(max_u64 - 10, 20, image_size),
                    "64-bit wrapping sum (0xFFFFFFFFFFFFFFF5 + 20) is rejected");
}

void test_string_null_termination_direct() {
    std::printf("\n--- Direct String Null Termination Validation ---\n");

    // Valid null-terminated strings
    char terminated_early[] = "hello\0world";
    check_condition(check_string_null_terminated(terminated_early, sizeof(terminated_early)),
                    "String with early null terminator is accepted");

    char empty_string[] = "\0";
    check_condition(check_string_null_terminated(empty_string, 1),
                    "Empty null-terminated string at index 0 is accepted");

    char terminated_at_last_byte[16] = "123456789012345";
    terminated_at_last_byte[15] = '\0';
    check_condition(check_string_null_terminated(terminated_at_last_byte, 16),
                    "String null-terminated at exact buffer boundary is accepted");

    // Unterminated strings
    char unterminated[16];
    std::memset(unterminated, 'A', sizeof(unterminated));
    check_condition(!check_string_null_terminated(unterminated, sizeof(unterminated)),
                    "Buffer filled entirely with non-zero bytes is rejected as unterminated");

    char zero_length_buf[1] = {'X'};
    check_condition(!check_string_null_terminated(zero_length_buf, 0),
                    "Zero-length buffer scan rejects unterminated string");
}

// ---------------------------------------------------------------------------
// Synthetic PE Image Helpers (Zero Game Data)
// ---------------------------------------------------------------------------

std::vector<uint8_t> make_base_image(uint32_t image_size = 0x2000, uint32_t file_size = 0x400) {
    std::vector<uint8_t> bytes(file_size, 0);
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
    nt->OptionalHeader.SizeOfImage = image_size;
    nt->OptionalHeader.Subsystem = IMAGE_SUBSYSTEM_WINDOWS_CUI;
    nt->OptionalHeader.NumberOfRvaAndSizes = IMAGE_NUMBEROF_DIRECTORY_ENTRIES;

    auto* sec = IMAGE_FIRST_SECTION(nt);
    std::memcpy(sec->Name, ".text", 6);
    sec->VirtualAddress = 0x1000;
    sec->Misc.VirtualSize = 0x200;
    sec->PointerToRawData = 0x200;
    sec->SizeOfRawData = 0x200;
    sec->Characteristics = IMAGE_SCN_CNT_CODE | IMAGE_SCN_MEM_EXECUTE | IMAGE_SCN_MEM_READ;

    for (size_t i = 0x200; i < file_size && i < 0x400; ++i) {
        bytes[i] = static_cast<uint8_t>(i & 0xFF);
    }
    return bytes;
}

IMAGE_NT_HEADERS64* nt_header_of(std::vector<uint8_t>& bytes) {
    auto* dos = reinterpret_cast<IMAGE_DOS_HEADER*>(bytes.data());
    return reinterpret_cast<IMAGE_NT_HEADERS64*>(bytes.data() + dos->e_lfanew);
}

std::filesystem::path write_temp_image(const std::vector<uint8_t>& bytes) {
    auto path = g_temp_dir / ("pe_bound_" + std::to_string(++g_file_seq) + ".bin");
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to write test fixture to " + path.string());
    }
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    return path;
}

bool expect_loader_rejection(const char* test_name, const std::vector<uint8_t>& bytes,
                            const char* expected_err_substring = nullptr) {
    auto file_path = write_temp_image(bytes);
    try {
        rt::Loader loader(g_temp_dir);
        loader.load(file_path);
        fail(test_name, "expected image rejection, but Loader::load succeeded");
        return false;
    } catch (const std::exception& e) {
        const std::string msg = e.what();
        if (expected_err_substring && msg.find(expected_err_substring) == std::string::npos) {
            fail(test_name, "rejected with unexpected message: '" + msg +
                 "', expected substring: '" + expected_err_substring + "'");
            return false;
        }
        pass(test_name);
        return true;
    }
}

bool expect_loader_success(const char* test_name, const std::vector<uint8_t>& bytes) {
    auto file_path = write_temp_image(bytes);
    try {
        rt::Loader loader(g_temp_dir);
        rt::Module& m = loader.load(file_path);
        if (!m.base) {
            fail(test_name, "module mapped with null base");
            return false;
        }
        pass(test_name);
        return true;
    } catch (const std::exception& e) {
        fail(test_name, std::string("expected success, but load failed: ") + e.what());
        return false;
    }
}

// ---------------------------------------------------------------------------
// Import Table Null Termination Tests
// ---------------------------------------------------------------------------

void test_import_table_null_termination() {
    std::printf("\n--- Import Table and Descriptor Null Termination Tests ---\n");

    // Case 1: Import directory table has no terminating null IMAGE_IMPORT_DESCRIPTOR.
    // The descriptors run off the end of the image.
    {
        auto img = make_base_image(0x1200, 0x400);
        auto* nt = nt_header_of(img);
        auto& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
        // Descriptor sits at 0x11EC (20 bytes, ending at 0x1200 == SizeOfImage).
        // Next descriptor begins at 0x1200 and runs past image.
        dir.VirtualAddress = 0x11EC;
        dir.Size = sizeof(IMAGE_IMPORT_DESCRIPTOR) * 2;

        std::fill(img.begin() + 0x200, img.end(), uint8_t{0});
        std::memcpy(img.data() + 0x240, "xmem.dll\0", 9);

        // Put a valid descriptor at file offset 0x3EC (RVA 0x11EC)
        auto* d = reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR*>(img.data() + 0x3EC);
        d->Name = 0x1040;
        d->FirstThunk = 0x1060;
        d->OriginalFirstThunk = 0x1060;  // points to a zero thunk, so bind_thunks returns immediately

        expect_loader_rejection("Import descriptor list lacking null terminator descriptor",
                               img, "the import table runs past the image");
    }

    // Case 2: Import lookup thunk table (OriginalFirstThunk) has no terminating null entry.
    // Thunks run past the end of the image.
    {
        auto img = make_base_image(0x1200, 0x400);
        auto* nt = nt_header_of(img);
        auto& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
        dir.VirtualAddress = 0x1000;
        dir.Size = sizeof(IMAGE_IMPORT_DESCRIPTOR);

        std::fill(img.begin() + 0x200, img.end(), uint8_t{0});
        auto* d = reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR*>(img.data() + 0x200);
        std::memcpy(img.data() + 0x240, "xmem.dll\0", 9);
        d->Name = 0x1040;
        d->FirstThunk = 0x11F8;          // exactly 1 slot before image size 0x1200
        d->OriginalFirstThunk = 0x11F8;

        // Put a non-zero AddressOfData at RVA 0x11F8 (file offset 0x3F8)
        auto* thunk = reinterpret_cast<IMAGE_THUNK_DATA64*>(img.data() + 0x3F8);
        thunk->u1.AddressOfData = 0x1040;

        expect_loader_rejection("Import lookup table lacking null terminator thunk",
                               img, "an import lookup table runs past the image");
    }

    // Case 3: Imported DLL name string has no null terminator before image end.
    {
        auto img = make_base_image(0x1200, 0x400);
        auto* nt = nt_header_of(img);
        auto& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
        dir.VirtualAddress = 0x1000;
        dir.Size = sizeof(IMAGE_IMPORT_DESCRIPTOR);

        std::fill(img.begin() + 0x200, img.end(), uint8_t{0});
        auto* d = reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR*>(img.data() + 0x200);
        d->Name = 0x11F0;  // 16 bytes before image end 0x1200
        d->FirstThunk = 0;
        d->OriginalFirstThunk = 0;

        // Fill bytes from 0x11F0 to 0x1200 (file offset 0x3F0..0x3FF) with non-zero characters
        std::memset(img.data() + 0x3F0, 'D', 0x10);

        expect_loader_rejection("Imported DLL name missing null terminator within image",
                               img, "an imported DLL name isn't terminated inside the image");
    }

    // Case 4: Imported function name string (IMAGE_IMPORT_BY_NAME) has no null terminator.
    {
        auto img = make_base_image(0x1200, 0x400);
        auto* nt = nt_header_of(img);
        auto& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
        dir.VirtualAddress = 0x1000;
        dir.Size = sizeof(IMAGE_IMPORT_DESCRIPTOR);

        std::fill(img.begin() + 0x200, img.end(), uint8_t{0});
        auto* d = reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR*>(img.data() + 0x200);
        std::memcpy(img.data() + 0x240, "xmem.dll\0", 9);
        d->Name = 0x1040;
        d->FirstThunk = 0x1080;
        d->OriginalFirstThunk = 0x1080;

        // Thunk points to RVA 0x11F0 for IMAGE_IMPORT_BY_NAME
        auto* thunk = reinterpret_cast<IMAGE_THUNK_DATA64*>(img.data() + 0x280);
        thunk[0].u1.AddressOfData = 0x11F0;
        thunk[1].u1.AddressOfData = 0;  // thunk table is null-terminated

        // Hint at 0x11F0 (file offset 0x3F0) is 2 bytes; string starts at 0x11F2 (0x3F2)
        // Fill 0x3F2..0x3FF (14 bytes) with non-zero characters without null terminator
        std::memset(img.data() + 0x3F0, 0, 2);
        std::memset(img.data() + 0x3F2, 'F', 0xE);

        expect_loader_rejection("Imported function name missing null terminator within image",
                               img, "an import name isn't terminated inside the image");
    }

    // Case 5: Delay-import descriptor table lacking null terminator descriptor.
    {
        auto img = make_base_image(0x1200, 0x400);
        auto* nt = nt_header_of(img);
        auto& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT];
        // Descriptor at 0x11DC (32 bytes, ending at 0x11FC < 0x1200).
        // Next descriptor at 0x11FC runs past 0x1200.
        dir.VirtualAddress = 0x11DC;
        dir.Size = sizeof(IMAGE_DELAYLOAD_DESCRIPTOR) * 2;

        std::fill(img.begin() + 0x200, img.end(), uint8_t{0});
        std::memcpy(img.data() + 0x240, "xmem.dll\0", 9);

        auto* d = reinterpret_cast<IMAGE_DELAYLOAD_DESCRIPTOR*>(img.data() + 0x3DC);
        d->Attributes.RvaBased = 1;
        d->DllNameRVA = 0x1040;
        d->ImportAddressTableRVA = 0x1060;
        d->ImportNameTableRVA = 0x1060;  // points to zero thunk at 0x1060

        expect_loader_rejection("Delay-import descriptor table lacking null terminator",
                               img, "the delay-import table runs past the image");
    }

    // Case 6: Delay-imported DLL name string lacking null terminator.
    {
        auto img = make_base_image(0x1200, 0x400);
        auto* nt = nt_header_of(img);
        auto& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT];
        dir.VirtualAddress = 0x1000;
        dir.Size = sizeof(IMAGE_DELAYLOAD_DESCRIPTOR);

        std::fill(img.begin() + 0x200, img.end(), uint8_t{0});
        auto* d = reinterpret_cast<IMAGE_DELAYLOAD_DESCRIPTOR*>(img.data() + 0x200);
        d->Attributes.RvaBased = 1;
        d->ImportAddressTableRVA = 0x1080;
        d->DllNameRVA = 0x11F0;

        std::memset(img.data() + 0x3F0, 'Z', 0x10);  // no null terminator up to 0x1200

        expect_loader_rejection("Delay-imported DLL name lacking null terminator",
                               img, "a delay-imported DLL name isn't terminated inside the image");
    }
}

// ---------------------------------------------------------------------------
// Virtual Address Range Wrapping and Boundary Tests
// ---------------------------------------------------------------------------

void test_virtual_address_range_wrapping() {
    std::printf("\n--- Virtual Address Range Wrapping and Boundary Validation Tests ---\n");

    // Case 1: Section VirtualAddress wrapping around 32-bit boundary.
    // If VirtualAddress = 0xFFFFF000 and VirtualSize = 0x2000, 32-bit addition wraps to 0x1000.
    // The loader must reject this before mapping or copying memory.
    {
        auto img = make_base_image(0x2000, 0x400);
        auto* nt = nt_header_of(img);
        auto* sec = IMAGE_FIRST_SECTION(nt);
        sec->VirtualAddress = 0xFFFFF000u;
        sec->Misc.VirtualSize = 0x2000u;

        expect_loader_rejection("Section VirtualAddress 32-bit wrapping (0xFFFFF000 + 0x2000)",
                               img, "a section's data runs past the image");
    }

    // Case 2: Section PointerToRawData wrapping around 32-bit boundary.
    // PointerToRawData = 0xFFFFFF00 with SizeOfRawData = 0x200.
    {
        auto img = make_base_image(0x2000, 0x400);
        auto* nt = nt_header_of(img);
        auto* sec = IMAGE_FIRST_SECTION(nt);
        sec->PointerToRawData = 0xFFFFFF00u;
        sec->SizeOfRawData = 0x200u;

        expect_loader_rejection("Section PointerToRawData 32-bit overflow reads past file end",
                               img, "reads past the end of the file");
    }

    // Case 3: Relocation directory VirtualAddress wrapping around 32-bit boundary.
    {
        auto img = make_base_image(0x2000, 0x400);
        auto* nt = nt_header_of(img);
        nt->OptionalHeader.ImageBase = 0;  // Force relocation
        auto& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_BASERELOC];
        dir.VirtualAddress = 0xFFFFFFF0u;
        dir.Size = 0x20u;

        expect_loader_rejection("Relocation table directory VirtualAddress 32-bit wrapping",
                               img, "the relocation table runs past the image");
    }

    // Case 4: Base relocation block entry with offset wrapping past image boundary.
    // block->VirtualAddress = 0x1FFC, DIR64 entry writes 8 bytes crossing image size 0x2000.
    {
        auto img = make_base_image(0x2000, 0x400);
        auto* nt = nt_header_of(img);
        nt->OptionalHeader.ImageBase = 0;
        auto& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_BASERELOC];
        dir.VirtualAddress = 0x1000;
        dir.Size = 0x0A;

        auto* block = reinterpret_cast<IMAGE_BASE_RELOCATION*>(img.data() + 0x200);
        block->VirtualAddress = 0x1FFC;  // 0x1FFC + 8 = 0x2004 > 0x2000
        block->SizeOfBlock = 0x0A;
        auto* entries = reinterpret_cast<uint16_t*>(block + 1);
        entries[0] = static_cast<uint16_t>(IMAGE_REL_BASED_DIR64 << 12);

        expect_loader_rejection("Relocation entry DIR64 wrapping past image end",
                               img, "a relocation runs past the image");
    }

    // Case 5: Export directory tables with wrapping VirtualAddress pointers.
    // AddressOfFunctions, AddressOfNames, AddressOfNameOrdinals wrapping 32-bit space.
    {
        auto img = make_base_image(0x2000, 0x400);
        auto* nt = nt_header_of(img);
        auto& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT];
        dir.VirtualAddress = 0x1000;
        dir.Size = sizeof(IMAGE_EXPORT_DIRECTORY);

        auto* ed = reinterpret_cast<IMAGE_EXPORT_DIRECTORY*>(img.data() + 0x200);
        ed->NumberOfFunctions = 10;
        ed->AddressOfFunctions = 0xFFFFFF80u;  // wraps 32-bit

        expect_loader_rejection("Export AddressOfFunctions wrapping 32-bit space",
                               img, "the export address table runs past the image");
    }

    // Case 6: Export AddressOfNames wrapping 32-bit space.
    {
        auto img = make_base_image(0x2000, 0x400);
        auto* nt = nt_header_of(img);
        auto& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT];
        dir.VirtualAddress = 0x1000;
        dir.Size = sizeof(IMAGE_EXPORT_DIRECTORY);

        auto* ed = reinterpret_cast<IMAGE_EXPORT_DIRECTORY*>(img.data() + 0x200);
        ed->NumberOfFunctions = 1;
        ed->AddressOfFunctions = 0x1100;
        ed->NumberOfNames = 10;
        ed->AddressOfNames = 0xFFFFFFC0u;

        expect_loader_rejection("Export AddressOfNames wrapping 32-bit space",
                               img, "the export name table runs past the image");
    }

    // Case 7: Export AddressOfNameOrdinals wrapping 32-bit space.
    {
        auto img = make_base_image(0x2000, 0x400);
        auto* nt = nt_header_of(img);
        auto& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT];
        dir.VirtualAddress = 0x1000;
        dir.Size = sizeof(IMAGE_EXPORT_DIRECTORY);

        auto* ed = reinterpret_cast<IMAGE_EXPORT_DIRECTORY*>(img.data() + 0x200);
        ed->NumberOfFunctions = 1;
        ed->AddressOfFunctions = 0x1100;
        ed->NumberOfNames = 10;
        ed->AddressOfNames = 0x1120;
        ed->AddressOfNameOrdinals = 0xFFFFFFE0u;

        expect_loader_rejection("Export AddressOfNameOrdinals wrapping 32-bit space",
                               img, "the export ordinal table runs past the image");
    }

    // Case 8: TLS Directory virtual address wrapping.
    // StartAddressOfRawData preceding or wrapping.
    {
        auto img = make_base_image(0x2000, 0x400);
        auto* nt = nt_header_of(img);
        auto& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_TLS];
        dir.VirtualAddress = 0x1000;
        dir.Size = sizeof(IMAGE_TLS_DIRECTORY64);

        auto* tls = reinterpret_cast<IMAGE_TLS_DIRECTORY64*>(img.data() + 0x200);
        tls->StartAddressOfRawData = 0x140001100ull;
        tls->EndAddressOfRawData = 0x140001000ull;  // End precedes start

        expect_loader_rejection("TLS EndAddressOfRawData precedes StartAddressOfRawData",
                               img, "TLS template end precedes its start");
    }

    // Case 9: TLS total size wrapping 32-bit integer (> 0xFFFFFFFF).
    {
        auto img = make_base_image(0x2000, 0x400);
        auto* nt = nt_header_of(img);
        auto& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_TLS];
        dir.VirtualAddress = 0x1000;
        dir.Size = sizeof(IMAGE_TLS_DIRECTORY64);

        auto* tls = reinterpret_cast<IMAGE_TLS_DIRECTORY64*>(img.data() + 0x200);
        tls->StartAddressOfRawData = 0x140001000ull;
        tls->EndAddressOfRawData = 0x140001100ull;  // raw = 0x100
        tls->SizeOfZeroFill = 0xFFFFFFFFu;          // raw + zero > 0xFFFFFFFF

        expect_loader_rejection("TLS total size 64-bit integer overflow check",
                               img, "TLS block is too large");
    }

    // Case 10: Exception directory (.pdata) boundary and overlap checks.
    // Unwind entry EndAddress past image size.
    {
        auto img = make_base_image(0x2000, 0x400);
        auto* nt = nt_header_of(img);
        auto& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXCEPTION];
        dir.VirtualAddress = 0x1000;
        dir.Size = sizeof(RUNTIME_FUNCTION);

        auto* fn = reinterpret_cast<RUNTIME_FUNCTION*>(img.data() + 0x200);
        fn->BeginAddress = 0x1000;
        fn->EndAddress = 0x2001;  // exceeds m.size 0x2000
        fn->UnwindData = 0x1100;

        expect_loader_rejection("Exception entry EndAddress exceeding image size",
                               img, "covers code outside the image");
    }

    // Case 11: Exception entries out of order (wrapping / overlap).
    {
        auto img = make_base_image(0x2000, 0x400);
        auto* nt = nt_header_of(img);
        auto& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXCEPTION];
        dir.VirtualAddress = 0x1000;
        dir.Size = sizeof(RUNTIME_FUNCTION) * 2;

        auto* fn = reinterpret_cast<RUNTIME_FUNCTION*>(img.data() + 0x200);
        fn[0].BeginAddress = 0x1100;
        fn[0].EndAddress = 0x1200;
        fn[0].UnwindData = 0x1300;

        fn[1].BeginAddress = 0x1050;  // out of order, overlaps fn[0]
        fn[1].EndAddress = 0x1150;
        fn[1].UnwindData = 0x1300;

        expect_loader_rejection("Exception entries out of order or overlapping",
                               img, "is out of order or overlaps the one before it");
    }
}

// ---------------------------------------------------------------------------
// Boundary Stress and Exact Fits
// ---------------------------------------------------------------------------

void test_boundary_exact_fits() {
    std::printf("\n--- Boundary Stress and Exact Fit Scenarios ---\n");

    // Case 1: Valid image with section filling exactly up to SizeOfImage
    {
        auto img = make_base_image(0x2000, 0x400);
        auto* nt = nt_header_of(img);
        auto* sec = IMAGE_FIRST_SECTION(nt);
        sec->VirtualAddress = 0x1000;
        sec->Misc.VirtualSize = 0x1000;  // 0x1000 + 0x1000 == 0x2000 (exact image size)
        sec->SizeOfRawData = 0x200;
        sec->PointerToRawData = 0x200;

        expect_loader_success("Section ending at exact SizeOfImage boundary maps successfully", img);
    }

    // Case 2: Section VirtualAddress + VirtualSize exceeding SizeOfImage by exactly 1 byte
    {
        auto img = make_base_image(0x2000, 0x400);
        auto* nt = nt_header_of(img);
        auto* sec = IMAGE_FIRST_SECTION(nt);
        sec->VirtualAddress = 0x1000;
        sec->Misc.VirtualSize = 0x1001;  // 0x1000 + 0x1001 == 0x2001 > 0x2000
        sec->SizeOfRawData = 0x1001;
        sec->PointerToRawData = 0x200;
        img.resize(0x200 + 0x1001);

        expect_loader_rejection("Section exceeding SizeOfImage by 1 byte is rejected",
                               img, "a section's data runs past the image");
    }

    // Case 3: Empty string (null at first byte) in imported DLL name is rejected as outside or empty
    {
        auto img = make_base_image(0x2000, 0x400);
        auto* nt = nt_header_of(img);
        auto& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
        dir.VirtualAddress = 0x1000;
        dir.Size = sizeof(IMAGE_IMPORT_DESCRIPTOR);

        std::fill(img.begin() + 0x200, img.end(), uint8_t{0});
        auto* d = reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR*>(img.data() + 0x200);
        d->Name = 0x2000;  // exactly SizeOfImage: first byte is out of range

        expect_loader_rejection("Imported DLL name starting at exact SizeOfImage is outside image",
                               img, "an imported DLL name is outside the image");
    }
}

// HIGHLOW (type 3) relocations: a 32-bit absolute address, which the x64 Windows loader applies too. The image
// prefers base 0x10000, the lowest address VirtualAlloc hands out; the test holds that address so the loader has to
// map the image higher, which makes the delta positive. The relocation table sits at the start of .text (RVA 0x1000)
// and fixes up the dword at RVA 0x1100.
std::vector<uint8_t> make_highlow_image(uint32_t stored, uint16_t type = IMAGE_REL_BASED_HIGHLOW,
                                        uint64_t image_base = 0x10000) {
    auto img = make_base_image(0x2000, 0x400);
    std::fill(img.begin() + 0x200, img.end(), uint8_t{0});
    auto* nt = nt_header_of(img);
    nt->OptionalHeader.ImageBase = image_base;
    auto& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_BASERELOC];
    dir.VirtualAddress = 0x1000;
    dir.Size = sizeof(IMAGE_BASE_RELOCATION) + 2 * sizeof(uint16_t);
    auto* block = reinterpret_cast<IMAGE_BASE_RELOCATION*>(img.data() + 0x200);
    block->VirtualAddress = 0x1000;
    block->SizeOfBlock = dir.Size;
    auto* entries = reinterpret_cast<uint16_t*>(block + 1);
    entries[0] = static_cast<uint16_t>((type << 12) | 0x100);
    entries[1] = static_cast<uint16_t>(IMAGE_REL_BASED_ABSOLUTE << 12);  // padding, as linkers write it
    std::memcpy(img.data() + 0x300, &stored, sizeof(stored));           // RVA 0x1100
    return img;
}

void test_highlow_relocations() {
    std::printf("\n--- HIGHLOW (32-bit) base relocations ---\n");
    // Hold the preferred base. If something else already holds it, the image is relocated all the same.
    void* hold = VirtualAlloc(reinterpret_cast<void*>(0x10000), 0x2000, MEM_RESERVE, PAGE_NOACCESS);

    // Case 1: the dword names RVA 0x1100 of an image at 0x10000. After the move it must name RVA 0x1100 of the new
    // base when that fits in 32 bits; otherwise the load must be refused rather than truncated.
    {
        auto file_path = write_temp_image(make_highlow_image(0x10000 + 0x1100));
        try {
            rt::Loader loader(g_temp_dir);
            rt::Module& m = loader.load(file_path);
            const uint64_t want = reinterpret_cast<uint64_t>(m.base) + 0x1100;
            check_condition(m.relocated, "HIGHLOW image was relocated away from its held base");
            check_condition(want <= 0xFFFFFFFFull, "HIGHLOW image mapped low enough for a 32-bit address",
                            "mapped at " + std::to_string(reinterpret_cast<uint64_t>(m.base)) + " yet accepted");
            uint32_t got = 0;
            std::memcpy(&got, m.base + 0x1100, sizeof(got));
            check_condition(got == static_cast<uint32_t>(want), "HIGHLOW dword now names RVA 0x1100 of the new base",
                            "got " + std::to_string(got) + ", want " + std::to_string(want));
        } catch (const std::exception& e) {
            const std::string msg = e.what();
            // Only acceptable when the new base is too high for the 32-bit address.
            check_condition(msg.find("does not fit at this base") != std::string::npos,
                            "HIGHLOW image mapped above 4 GiB is refused with the 32-bit message", "threw: " + msg);
        }
    }

    // Case 2: a dword near the top of the 32-bit range overflows for any base above 0x10000: refused, not wrapped.
    expect_loader_rejection("HIGHLOW relocation past 4 GiB is refused, not truncated", make_highlow_image(0xFFFFFFF0u),
                            "does not fit at this base");

    // Case 3: other 32-bit forms (HIGH, type 1) are still refused by name.
    expect_loader_rejection("HIGH (type 1) relocation is still unsupported",
                            make_highlow_image(0, IMAGE_REL_BASED_HIGH), "unsupported relocation type 1");

    // Case 4: the fix-up itself, whatever the address space looks like. The preferred base is a region this test
    // holds, so the image lands elsewhere but usually near it; the delta is then small, and a dword in the middle of
    // the 32-bit range must move by exactly that delta. (A delta too large for it must be refused, as in case 2.)
    if (void* near_hold = VirtualAlloc(nullptr, 0x2000, MEM_RESERVE, PAGE_NOACCESS)) {
        const uint64_t preferred = reinterpret_cast<uint64_t>(near_hold);
        constexpr uint32_t stored = 0x80000000u;
        auto file_path = write_temp_image(make_highlow_image(stored, IMAGE_REL_BASED_HIGHLOW, preferred));
        try {
            rt::Loader loader(g_temp_dir);
            rt::Module& m = loader.load(file_path);
            const int64_t delta = static_cast<int64_t>(reinterpret_cast<uint64_t>(m.base) - preferred);
            const int64_t want = static_cast<int64_t>(stored) + delta;
            uint32_t got = 0;
            std::memcpy(&got, m.base + 0x1100, sizeof(got));
            check_condition(m.relocated && want >= 0 && want <= 0xFFFFFFFFll,
                            "HIGHLOW fix-up: relocated by a delta that fits", "delta " + std::to_string(delta));
            check_condition(got == static_cast<uint32_t>(want), "HIGHLOW fix-up: the dword moved by exactly the delta",
                            "got " + std::to_string(got) + ", want " + std::to_string(want) + ", delta " +
                                std::to_string(delta));
        } catch (const std::exception& e) {
            const std::string msg = e.what();
            check_condition(msg.find("does not fit at this base") != std::string::npos,
                            "HIGHLOW fix-up: only a too-large delta is refused", "threw: " + msg);
        }
        VirtualFree(near_hold, 0, MEM_RELEASE);
    }

    if (hold) VirtualFree(hold, 0, MEM_RELEASE);
}

// PE/COFF spec, optional header: NumberOfRvaAndSizes is "the number of data-directory entries in the remainder of
// the optional header". Entries past it are not directories, whatever bytes they hold.
void test_directory_count() {
    std::printf("\n--- NumberOfRvaAndSizes bounds the data directories ---\n");
    auto img = make_base_image(0x2000, 0x400);
    auto* nt = nt_header_of(img);
    nt->OptionalHeader.NumberOfRvaAndSizes = IMAGE_DIRECTORY_ENTRY_IMPORT;  // only the export entry is a directory
    nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT] = {0xFFFFF000u, sizeof(IMAGE_IMPORT_DESCRIPTOR)};
    nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT] = {0xFFFFF000u, 0x20};
    expect_loader_success("Stale entries past NumberOfRvaAndSizes are not read as directories", img);
}

// A forwarder "kernel32.#65537" names an ordinal no export table can hold (ordinals are 16 bits). It must not be
// truncated to ordinal 1 and bound to whatever kernel32 exports there.
void test_forwarder_ordinal_range() {
    std::printf("\n--- Forwarder ordinal range ---\n");
    auto dll = make_base_image(0x2000, 0x400);
    std::fill(dll.begin() + 0x200, dll.end(), uint8_t{0});
    auto* dnt = nt_header_of(dll);
    dnt->OptionalHeader.ImageBase = leonida_test::free_image_base(0x150000000ull);
    dnt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT] = {0x1000, 0x100};
    auto* ed = reinterpret_cast<IMAGE_EXPORT_DIRECTORY*>(dll.data() + 0x200);  // RVA 0x1000
    ed->Base = 1;
    ed->NumberOfFunctions = 1;
    ed->NumberOfNames = 1;
    ed->AddressOfFunctions = 0x1040;
    ed->AddressOfNames = 0x1050;
    ed->AddressOfNameOrdinals = 0x1060;
    const uint32_t forwarder_rva = 0x1080, name_rva = 0x10C0;
    std::memcpy(dll.data() + 0x240, &forwarder_rva, 4);
    std::memcpy(dll.data() + 0x250, &name_rva, 4);
    std::memcpy(dll.data() + 0x280, "kernel32.#65537", 16);
    std::memcpy(dll.data() + 0x2C0, "F", 2);
    {
        std::ofstream out(g_temp_dir / "fwd_ord.dll", std::ios::binary);
        out.write(reinterpret_cast<const char*>(dll.data()), static_cast<std::streamsize>(dll.size()));
    }

    auto exe = make_base_image(0x2000, 0x400);
    std::fill(exe.begin() + 0x200, exe.end(), uint8_t{0});
    auto* ent = nt_header_of(exe);
    ent->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT] = {0x1000, 2 * sizeof(IMAGE_IMPORT_DESCRIPTOR)};
    auto* d = reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR*>(exe.data() + 0x200);
    d->OriginalFirstThunk = 0x1040;
    d->Name = 0x1080;
    d->FirstThunk = 0x1060;
    const uint64_t hint_name = 0x1090;
    std::memcpy(exe.data() + 0x240, &hint_name, 8);
    std::memcpy(exe.data() + 0x260, &hint_name, 8);
    std::memcpy(exe.data() + 0x280, "fwd_ord.dll", 12);
    std::memcpy(exe.data() + 0x292, "F", 2);  // after the 2-byte hint

    auto path = write_temp_image(exe);
    try {
        rt::Loader loader(g_temp_dir);
        rt::Module& m = loader.load(path);
        const bool one = m.imports.size() == 1;
        check_condition(one && m.imports[0].binding == rt::Binding::Unimplemented,
                        "Forwarder to ordinal 65537 is not truncated to a host ordinal",
                        one ? "bound as " + std::to_string(static_cast<int>(m.imports[0].binding))
                            : std::to_string(m.imports.size()) + " imports");
    } catch (const std::exception& e) {
        fail("Forwarder to ordinal 65537 is not truncated to a host ordinal", e.what());
    }
}

void test_headers_and_reloc_tail_bounds() {
    std::printf("\n--- Headers vs SizeOfHeaders and Truncated Relocation Block Header ---\n");

    // 1. e_lfanew + sizeof(IMAGE_NT_HEADERS64) > SizeOfImage with SizeOfHeaders < e_lfanew.
    //    Previously passed map_image() checks and crashed dereferencing m->base + e_lfanew.
    {
        std::vector<uint8_t> bytes(0x2000, 0);
        auto* dos = reinterpret_cast<IMAGE_DOS_HEADER*>(bytes.data());
        dos->e_magic = IMAGE_DOS_SIGNATURE;
        dos->e_lfanew = 0x1000;
        auto* nt = reinterpret_cast<IMAGE_NT_HEADERS64*>(bytes.data() + dos->e_lfanew);
        nt->Signature = IMAGE_NT_SIGNATURE;
        nt->FileHeader.Machine = IMAGE_FILE_MACHINE_AMD64;
        nt->FileHeader.NumberOfSections = 0;
        nt->FileHeader.SizeOfOptionalHeader = sizeof(IMAGE_OPTIONAL_HEADER64);
        nt->OptionalHeader.Magic = IMAGE_NT_OPTIONAL_HDR64_MAGIC;
        nt->OptionalHeader.ImageBase = 0x140000000ull;
        nt->OptionalHeader.SectionAlignment = 0x1000;
        nt->OptionalHeader.FileAlignment = 0x200;
        nt->OptionalHeader.SizeOfImage = 0x1000;
        nt->OptionalHeader.SizeOfHeaders = 0x200;
        nt->OptionalHeader.NumberOfRvaAndSizes = IMAGE_NUMBEROF_DIRECTORY_ENTRIES;

        auto path = write_temp_image(bytes);
        bool threw = false;
        try {
            rt::Loader loader(g_temp_dir);
            loader.load(path);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        check_condition(threw, "NT headers past SizeOfHeaders/SizeOfImage are rejected cleanly");
    }

    // 2. SizeOfHeaders covers NT headers but truncates the section table.
    {
        auto bytes = make_base_image(0x2000, 0x400);
        auto* nt = nt_header_of(bytes);
        // e_lfanew(0x80) + sizeof(IMAGE_NT_HEADERS64)(0x108) = 0x188; section header ends at 0x1B0.
        nt->OptionalHeader.SizeOfHeaders = 0x190;
        auto path = write_temp_image(bytes);
        bool threw = false;
        try {
            rt::Loader loader(g_temp_dir);
            loader.load(path);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        check_condition(threw, "Section table extending past SizeOfHeaders is rejected cleanly");
    }

    // 3. Relocation directory with < 8 bytes remaining at the exact end of SizeOfImage (forced relocation).
    {
        void* occupied = VirtualAlloc(nullptr, 0x2000, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        check_condition(occupied != nullptr, "Reserved preferred base for truncated reloc test");
        if (occupied) {
            auto bytes = make_base_image(0x2000, 0x400);
            auto* nt = nt_header_of(bytes);
            nt->OptionalHeader.ImageBase = reinterpret_cast<uint64_t>(occupied);
            nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_BASERELOC] = {0x1FFC, 4};
            auto path = write_temp_image(bytes);
            bool threw = false;
            try {
                rt::Loader loader(g_temp_dir);
                loader.load(path);
            } catch (const std::runtime_error&) {
                threw = true;
            }
            VirtualFree(occupied, 0, MEM_RELEASE);
            check_condition(threw, "Truncated 4-byte relocation block at end of image is rejected cleanly");
        }
    }

    // 4. Valid 8-byte relocation block followed by 4 trailing bytes (remaining < sizeof(IMAGE_BASE_RELOCATION)).
    {
        void* occupied = VirtualAlloc(nullptr, 0x2000, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        check_condition(occupied != nullptr, "Reserved preferred base for trailing reloc bytes test");
        if (occupied) {
            auto bytes = make_base_image(0x2000, 0x1400);
            auto* nt = nt_header_of(bytes);
            nt->OptionalHeader.ImageBase = reinterpret_cast<uint64_t>(occupied);
            auto* sec = IMAGE_FIRST_SECTION(nt);
            sec->VirtualAddress = 0x1000;
            sec->Misc.VirtualSize = 0x1000;
            sec->PointerToRawData = 0x400;
            sec->SizeOfRawData = 0x1000;
            // Place a 12-byte reloc dir at RVA 0x1FF4 (raw offset 0x13F4): 8-byte block + 4 trailing bytes.
            nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_BASERELOC] = {0x1FF4, 12};
            IMAGE_BASE_RELOCATION hdr{};
            hdr.VirtualAddress = 0x1000;
            hdr.SizeOfBlock = sizeof(IMAGE_BASE_RELOCATION);
            std::memcpy(bytes.data() + 0x13F4, &hdr, sizeof(hdr));
            auto path = write_temp_image(bytes);
            bool threw = false;
            try {
                rt::Loader loader(g_temp_dir);
                loader.load(path);
            } catch (const std::runtime_error&) {
                threw = true;
            }
            VirtualFree(occupied, 0, MEM_RELEASE);
            check_condition(threw, "Trailing 4 bytes after valid relocation block are rejected cleanly");
        }
    }

    {
        auto bytes = make_base_image(0x2000, 0x400);
        IMAGE_FIRST_SECTION(nt_header_of(bytes))->VirtualAddress = 0;
        std::memcpy(bytes.data() + 0x200, bytes.data(), 0x200);
        reinterpret_cast<IMAGE_DOS_HEADER*>(bytes.data() + 0x200)->e_lfanew = 0x7FFF0000;
        expect_loader_rejection("Section below SizeOfHeaders is rejected", bytes, "section 0 overlaps the headers");
    }

    {
        auto bytes = make_base_image(0x2000, 0x400);
        auto* nt = nt_header_of(bytes);
        nt->OptionalHeader.ImageBase = 0;
        nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_BASERELOC] = {0x1000, 0x0A};
        auto* block = reinterpret_cast<IMAGE_BASE_RELOCATION*>(bytes.data() + 0x200);
        block->VirtualAddress = 0;
        block->SizeOfBlock = 0x0A;
        reinterpret_cast<uint16_t*>(block + 1)[0] = static_cast<uint16_t>(IMAGE_REL_BASED_DIR64 << 12 | 0x3C);
        expect_loader_rejection("Relocation inside the headers is rejected", bytes, "a relocation lands in the headers");
    }

    {
        auto bytes = make_base_image(0x400, 0x400);
        auto* nt = nt_header_of(bytes);
        nt->OptionalHeader.SectionAlignment = 0x200;
        nt->OptionalHeader.FileAlignment = 0x200;
        nt->OptionalHeader.AddressOfEntryPoint = 0x200;
        IMAGE_FIRST_SECTION(nt)->VirtualAddress = 0x200;
        expect_loader_success("Section starting exactly at SizeOfHeaders maps successfully", bytes);

        nt->OptionalHeader.ImageBase = 0;
        nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_BASERELOC] = {0x300, 0x0A};
        auto* block = reinterpret_cast<IMAGE_BASE_RELOCATION*>(bytes.data() + 0x300);
        block->VirtualAddress = 0;
        block->SizeOfBlock = 0x0A;
        reinterpret_cast<uint16_t*>(block + 1)[0] = static_cast<uint16_t>(IMAGE_REL_BASED_DIR64 << 12 | 0x200);
        expect_loader_success("DIR64 relocation at exactly SizeOfHeaders is accepted", bytes);
    }
}

}  // namespace

int main() {
    std::printf("test_pe_loader_bounds: PE Loader boundary checks, virtual address range wrapping, and import table null termination\n");

    const char* env_dir = std::getenv("LEONIDA_TEST_DIR");
    g_temp_dir = env_dir ? std::filesystem::path(env_dir)
                         : std::filesystem::temp_directory_path() /
                               ("leonida_pe_bounds_test_" + std::to_string(GetCurrentProcessId()));
    std::filesystem::create_directories(g_temp_dir);

    try {
        test_range_wrapping_arithmetic();
        test_string_null_termination_direct();
        test_import_table_null_termination();
        test_virtual_address_range_wrapping();
        test_boundary_exact_fits();
        test_highlow_relocations();
        test_directory_count();
        test_forwarder_ordinal_range();
        test_headers_and_reloc_tail_bounds();
    } catch (const std::exception& e) {
        std::printf("FATAL: Unhandled exception during test execution: %s\n", e.what());
        ++g_failures;
    }

    std::error_code ec;
    std::filesystem::remove_all(g_temp_dir, ec);

    std::printf("\n============================================================\n");
    if (g_failures == 0) {
        std::printf("test_pe_loader_bounds: ALL %d CHECKS PASSED\n", g_checks);
        return 0;
    } else {
        std::printf("test_pe_loader_bounds: FAILED (%d checks, %d failure%s)\n",
                    g_checks, g_failures, g_failures == 1 ? "" : "s");
        return 1;
    }
}
