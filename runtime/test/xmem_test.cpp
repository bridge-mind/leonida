// Checks the registered xmem.dll shim against the measured Xbox ABI.
//
// Every function is resolved through rt::shims::find, so the test exercises the same shim table and calling
// convention the runtime installs for xmem.dll imports instead of a private copy of the implementation. The checks
// cover what the measured call sites use: 64-byte-aligned allocations, XMemFree, working-set statistics with the
// size at +8, and XMemVirtualQuery's MEMORY_BASIC_INFORMATION prefix inside a zeroed 64-byte output.
// XMemTransferMemory follows the call shape in Microsoft's public samples: TRUE for any direction and size, with
// nothing committed, and one trace event per distinct (type, size). XMemVirtualAlloc accepts attribute words and console
// graphics protection bits and commits 2 MiB guest pages; the placeholder flags pass through; physical pages keep their
// contents across unmap and remap, alias when mapped twice, and scatter-map per address.
#include <windows.h>
#include <psapi.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>

#include "shims.h"
#include "startup_trace.h"

namespace {

using XMemAllocFn = void*(__stdcall*)(size_t, uint64_t);
using XMemFreeFn = void(__stdcall*)(void*, uint64_t);
using XMemVirtualQueryFn = SIZE_T(__stdcall*)(HANDLE, const void*, void*);
// BOOL XMemGetWorkingSetStatistics(UINT32 kind, stats*): the FSR DLL passes kind 0x20000001 at every call site.
using XMemGetWorkingSetStatisticsFn = BOOL(__stdcall*)(uint32_t, void*);
using XMemVirtualAllocFn = void*(WINAPI*)(void*, SIZE_T, DWORD, ULONGLONG, DWORD);
using XMemGetWriteWatchFn = UINT(WINAPI*)(DWORD, void*, SIZE_T, void**, ULONG_PTR*, ULONG*);
using XMemResetWriteWatchFn = UINT(WINAPI*)(void*, SIZE_T);
// BOOL XMemTransferMemory(XMEM_TRANSFER_TYPE, SIZE_T): the shape Microsoft's public NetRumble and NativeAOT-Xbox
// samples call (see runtime/shims/xmem.cpp).
using XMemTransferMemoryFn = BOOL(WINAPI*)(uint32_t, SIZE_T);
// The shapes measured at a title's call sites (see runtime/shims/xmem.cpp).
using XMemIsToolsMemAvailableFn = BOOL(WINAPI*)();
using XMemAllocatePhysicalPagesFn = BOOL(WINAPI*)(uint64_t, uint64_t*, uint64_t*);
using XMemFreePhysicalPagesFn = BOOL(WINAPI*)(uint64_t, const uint64_t*);
using XMemMapPhysicalPagesFn = void*(WINAPI*)(void*, uint64_t, const uint64_t*);
using XMemMapPhysicalPagesScatterFn = BOOL(WINAPI*)(void* const*, uint32_t, const uint64_t*);
constexpr uint32_t kMeasuredKind = 0x20000001;
constexpr DWORD kXmemLargePages = 0x20000000;
constexpr DWORD kXmem64kPages = 0x20400000;
constexpr SIZE_T kPage64k = 0x10000;
constexpr SIZE_T kPage2m = 0x200000;

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

bool aligned_64(const void* p) { return p && (reinterpret_cast<uintptr_t>(p) & 63u) == 0; }

void test_alloc(XMemAllocFn alloc, XMemFreeFn free_mem) {
    void* const zero = alloc(0, 0);
    check(zero != nullptr, "XMemAlloc(0, 0) returns non-null");
    check(aligned_64(zero), "XMemAlloc(0, 0) returns a 64-byte aligned block");

    bool zero_readable = false;
    if (zero) {
        auto* bytes = static_cast<uint8_t*>(zero);
        bytes[0] = 0xa5;
        zero_readable = bytes[0] == 0xa5;
    }
    check(zero_readable, "XMemAlloc(0, 0) block is writable and readable");

    void* const big = alloc(4096, 0x5e50000);
    check(big != nullptr, "XMemAlloc(4096, 0x5e50000) returns non-null");
    check(aligned_64(big), "XMemAlloc(4096, 0x5e50000) returns a 64-byte aligned block");

    bool pattern_ok = big != nullptr;
    if (big) {
        auto* bytes = static_cast<uint8_t*>(big);
        for (size_t i = 0; i < 4096; ++i) bytes[i] = static_cast<uint8_t>((i * 37u + 11u) & 0xffu);
        for (size_t i = 0; i < 4096; ++i) {
            if (bytes[i] != static_cast<uint8_t>((i * 37u + 11u) & 0xffu)) {
                pattern_ok = false;
                break;
            }
        }
    }
    check(pattern_ok, "XMemAlloc(4096, 0x5e50000) block holds a write/read pattern");

    free_mem(zero, 0);
    free_mem(big, 0x5e50000);
    check(true, "XMemFree accepts both blocks (no crash)");
}

void test_working_set(XMemGetWorkingSetStatisticsFn get_stats) {
    alignas(64) uint8_t stats[64];
    std::memset(stats, 0xa5, sizeof(stats));
    // The pointer is the second argument: the first (0x20000001 here, as the FSR DLL passes it) must never be written
    // through, which is what the one-argument form did.
    const BOOL ok = get_stats(kMeasuredKind, stats);
    check(ok != FALSE, "XMemGetWorkingSetStatistics(kind, 64-byte out) returns TRUE");

    uint64_t total_memory = 0;
    std::memcpy(&total_memory, stats + 0, sizeof(total_memory));
    check(ok != FALSE && total_memory != 0, "XMemGetWorkingSetStatistics writes a non-zero total memory at +0");

    MEMORYSTATUSEX ms{sizeof(ms)};
    if (GlobalMemoryStatusEx(&ms) && ms.ullTotalPhys != 0) {
        check(ok != FALSE && total_memory == ms.ullTotalPhys,
              "XMemGetWorkingSetStatistics total memory (+0) matches GlobalMemoryStatusEx ullTotalPhys");
    } else {
        check(ok != FALSE && total_memory == 12ULL * 1024 * 1024 * 1024,
              "XMemGetWorkingSetStatistics total memory (+0) falls back to 12 GiB title limit");
    }

    uint64_t working_set = 0;
    std::memcpy(&working_set, stats + 8, sizeof(working_set));
    check(ok != FALSE && working_set != 0, "XMemGetWorkingSetStatistics writes a non-zero size at +8");

    check(ok != FALSE && total_memory >= working_set,
          "XMemGetWorkingSetStatistics total memory (+0) >= working set (+8)");

    bool tail_untouched = true;
    for (size_t i = 16; i < sizeof(stats); ++i) {
        if (stats[i] != 0xa5) {
            tail_untouched = false;
            break;
        }
    }
    check(ok != FALSE && tail_untouched, "XMemGetWorkingSetStatistics leaves bytes 16..63 untouched");

    check(get_stats(kMeasuredKind, nullptr) == FALSE,
          "XMemGetWorkingSetStatistics(kind, nullptr) returns FALSE without crashing");
}

void test_virtual_query(XMemVirtualQueryFn query) {
    uint8_t local = 0;
    const void* const local_address = &local;

    alignas(64) uint8_t out[64];
    std::memset(out, 0xa5, sizeof(out));
    const SIZE_T written = query(GetCurrentProcess(), local_address, out);
    check(written != 0, "XMemVirtualQuery(current process, stack local) returns non-zero");

    MEMORY_BASIC_INFORMATION info{};
    std::memcpy(&info, out, sizeof(info));
    const uintptr_t base = reinterpret_cast<uintptr_t>(info.BaseAddress);
    const uintptr_t address = reinterpret_cast<uintptr_t>(local_address);
    const bool contains_local =
        base != 0 && info.RegionSize != 0 && address >= base && address - base < info.RegionSize;
    check(written != 0 && contains_local,
          "XMemVirtualQuery fills MEMORY_BASIC_INFORMATION with a region containing the local");

    bool tail_zero = true;
    for (size_t i = sizeof(info); i < sizeof(out); ++i) {
        if (out[i] != 0) {
            tail_zero = false;
            break;
        }
    }
    check(written != 0 && tail_zero, "XMemVirtualQuery leaves the 64-byte output tail zero");

    alignas(64) uint8_t bad_out[64];
    std::memset(bad_out, 0xa5, sizeof(bad_out));
    // Address 1 is queryable (it falls in the free region starting at 0), so use a non-canonical address that
    // VirtualQueryEx rejects.
    const void* const unmappable = reinterpret_cast<const void*>(~static_cast<uintptr_t>(0));
    const SIZE_T bad = query(GetCurrentProcess(), unmappable, bad_out);
    check(bad == 0, "XMemVirtualQuery on an invalid address returns 0 without crashing");
}

void test_virtual_alloc(XMemVirtualAllocFn virtual_alloc, XMemGetWriteWatchFn get_write_watch,
                       XMemResetWriteWatchFn reset_write_watch) {
    SYSTEM_INFO system_info{};
    GetSystemInfo(&system_info);
    const SIZE_T page_size = system_info.dwPageSize;
    check(page_size != 0, "GetSystemInfo reports a non-zero page size");
    if (page_size == 0) return;

    const SIZE_T region_size = page_size * 3;
    void* const base = virtual_alloc(nullptr, region_size, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, 0,
                                     PAGE_READWRITE);
    check(base != nullptr, "XMemVirtualAlloc creates a synthetic write-watch region");
    if (!base) return;

    void* watched[4]{};
    ULONG_PTR count = 4;
    ULONG granularity = 0;
    UINT status = get_write_watch(0, base, region_size, watched, &count, &granularity);
    check(status == NO_ERROR && count == 0, "new write-watch region starts with no dirty pages");
    check(status == NO_ERROR && granularity == page_size, "write-watch reports the host page granularity");

    auto* const bytes = static_cast<volatile uint8_t*>(base);
    bytes[0] = 0xa1;
    bytes[page_size * 2] = 0xb2;

    count = 4;
    status = get_write_watch(0, base, region_size, watched, &count, &granularity);
    bool saw_first_page = false;
    bool saw_third_page = false;
    bool all_addresses_are_page_aligned = count <= 4;
    for (ULONG_PTR i = 0; i < count && i < 4; ++i) {
        const uintptr_t address = reinterpret_cast<uintptr_t>(watched[i]);
        const uintptr_t start = reinterpret_cast<uintptr_t>(base);
        all_addresses_are_page_aligned &= address >= start && (address - start) % page_size == 0;
        saw_first_page |= address == start;
        saw_third_page |= address == start + page_size * 2;
    }
    check(status == NO_ERROR && count == 2 && saw_first_page && saw_third_page,
          "write-watch reports each written page by its relative address");
    check(all_addresses_are_page_aligned, "write-watch addresses are page aligned and within the region");

    count = 4;
    status = get_write_watch(0, base, region_size, watched, &count, &granularity);
    check(status == NO_ERROR && count == 2, "query-only write-watch leaves dirty pages recorded");

    status = reset_write_watch(base, region_size);
    check(status == NO_ERROR, "XMemResetWriteWatch clears the synthetic region");
    count = 4;
    status = get_write_watch(0, base, region_size, watched, &count, &granularity);
    check(status == NO_ERROR && count == 0, "reset write-watch query returns no dirty pages");

    bytes[page_size] = 0xc3;
    count = 4;
    status = get_write_watch(0, base, region_size, watched, &count, &granularity);
    check(status == NO_ERROR && count == 1 &&
              reinterpret_cast<uintptr_t>(watched[0]) == reinterpret_cast<uintptr_t>(base) + page_size,
          "write-watch records writes again after reset");

    check(VirtualFree(base, 0, MEM_RELEASE), "release the synthetic write-watch region");

    // 2 MiB guest pages: the region is 2 MiB-aligned, rounded up to whole 2 MiB pages and committed.
    void* const large_page = virtual_alloc(nullptr, kPage2m + page_size, MEM_RESERVE | MEM_COMMIT | kXmemLargePages,
                                           0x183, PAGE_READWRITE | PAGE_GRAPHICS_READWRITE | PAGE_GRAPHICS_COHERENT);
    check(large_page != nullptr, "XMemVirtualAlloc commits 2 MiB guest pages");
    if (large_page) {
        check((reinterpret_cast<uintptr_t>(large_page) & (kPage2m - 1)) == 0, "a 2 MiB-page region is 2 MiB-aligned");
        MEMORY_BASIC_INFORMATION info{};
        VirtualQuery(large_page, &info, sizeof(info));
        check(info.State == MEM_COMMIT && info.RegionSize >= 2 * kPage2m && info.Protect == PAGE_READWRITE,
              "a 2 MiB-page region is rounded up to whole pages, committed read-write without the graphics bits");
        static_cast<volatile uint8_t*>(large_page)[2 * kPage2m - 1] = 0x77;
        check(VirtualFree(large_page, 0, MEM_RELEASE), "release the 2 MiB-page region");
    }

    // The attribute word is accepted (every measured call passes a nonzero one) and the protection loses its graphics
    // bits, in the 64 KiB page mode as well.
    void* const attributed = virtual_alloc(nullptr, kPage64k, MEM_RESERVE | MEM_COMMIT | kXmem64kPages, 0x181,
                                           PAGE_READWRITE | PAGE_GRAPHICS_READWRITE);
    check(attributed != nullptr, "XMemVirtualAlloc accepts a nonzero attribute word and console protection bits");
    if (attributed) {
        MEMORY_BASIC_INFORMATION info{};
        VirtualQuery(attributed, &info, sizeof(info));
        check(info.State == MEM_COMMIT && info.Protect == PAGE_READWRITE,
              "the region is committed PAGE_READWRITE (graphics bits dropped)");
        static_cast<volatile uint8_t*>(attributed)[0] = 0x42;
        check(VirtualFree(attributed, 0, MEM_RELEASE), "release the attributed region");
    }

    // The retail executable's GPU pools: the attribute word is recorded, and the PAGE_GRAPHICS_* bits are dropped.
    MEMORY_BASIC_INFORMATION mbi{};
    void* const pool = virtual_alloc(nullptr, SIZE_T{8} << 30, MEM_RESERVE | 0x20400000, 0x21, PAGE_READWRITE);
    check(pool != nullptr, "attributes 0x21 with 64 KiB pages reserves 8 GiB");
    if (pool) {
        check(VirtualQuery(pool, &mbi, sizeof mbi) && mbi.State == MEM_RESERVE, "the 8 GiB pool is reserve-only");
        check(VirtualFree(pool, 0, MEM_RELEASE), "release the 8 GiB pool");
    }

    void* const noaccess = virtual_alloc(nullptr, 0x10000, MEM_RESERVE | 0x20400000, 0x3, 0x801);
    check(noaccess != nullptr, "attributes 0x3 with protect PAGE_GRAPHICS_NOACCESS|PAGE_NOACCESS succeeds");
    if (noaccess) {
        check(VirtualQuery(noaccess, &mbi, sizeof mbi) && mbi.AllocationProtect == PAGE_NOACCESS,
              "protect 0x801 becomes PAGE_NOACCESS");
        check(VirtualFree(noaccess, 0, MEM_RELEASE), "release the PAGE_NOACCESS pool");
    }

    void* const gpu = virtual_alloc(nullptr, 0x10000, MEM_RESERVE | MEM_COMMIT, 0x183, 0x22004);
    check(gpu != nullptr, "attributes 0x183 with protect 0x22004 succeeds");
    if (gpu) {
        check(VirtualQuery(gpu, &mbi, sizeof mbi) && mbi.Protect == PAGE_READWRITE, "protect 0x22004 becomes PAGE_READWRITE");
        static_cast<volatile uint8_t*>(gpu)[0] = 0x5a;
        check(static_cast<volatile uint8_t*>(gpu)[0] == 0x5a, "the 0x183 allocation is writable");
        check(VirtualFree(gpu, 0, MEM_RELEASE), "release the 0x183 allocation");
    }

    SetLastError(ERROR_SUCCESS);
    void* const replace_without_address =
        virtual_alloc(nullptr, kPage64k, MEM_RESERVE | MEM_REPLACE_PLACEHOLDER | kXmem64kPages, 0, PAGE_READWRITE);
    check(replace_without_address == nullptr && GetLastError() == ERROR_INVALID_PARAMETER,
          "MEM_REPLACE_PLACEHOLDER without an address is rejected");
}

// XMemAllocDefault / XMemFreeDefault share XMemAlloc's heap: a block from either is freed by either.
void test_default_alloc(XMemAllocFn alloc, XMemFreeFn free_mem, XMemAllocFn alloc_default,
                        XMemFreeFn free_default) {
    void* const a = alloc_default(100, 0x5e50000);
    check(aligned_64(a), "XMemAllocDefault returns a 64-byte aligned block");
    void* const zero = alloc_default(0, 0);
    check(zero != nullptr, "XMemAllocDefault(0) returns non-null");
    void* const b = alloc(100, 0);
    if (a) std::memset(a, 0x11, 100);
    if (b) std::memset(b, 0x22, 100);
    free_mem(a, 0x5e50000);
    free_default(b, 0);
    free_default(zero, 0);
    check(true, "XMemFree releases an XMemAllocDefault block and XMemFreeDefault an XMemAlloc block");
}

uint32_t read_u32(const void* base, size_t offset) {
    uint32_t value = 0;
    std::memcpy(&value, static_cast<const uint8_t*>(base) + offset, sizeof(value));
    return value;
}

void fill_page(void* page, uint32_t tag) {
    auto* words = static_cast<uint32_t*>(page);
    for (size_t i = 0; i < kPage64k / sizeof(uint32_t); i += 1024) words[i] = tag + static_cast<uint32_t>(i);
}

bool page_holds(const void* page, uint32_t tag) {
    for (size_t i = 0; i < kPage64k / sizeof(uint32_t); i += 1024) {
        if (read_u32(page, i * sizeof(uint32_t)) != tag + static_cast<uint32_t>(i)) return false;
    }
    return true;
}

bool reserved_only(const void* address) {
    MEMORY_BASIC_INFORMATION info{};
    return VirtualQuery(address, &info, sizeof(info)) && info.State == MEM_RESERVE;
}

struct PhysicalApi {
    XMemVirtualAllocFn virtual_alloc;
    XMemAllocatePhysicalPagesFn allocate;
    XMemFreePhysicalPagesFn free_pages;
    XMemMapPhysicalPagesFn map;
    XMemMapPhysicalPagesScatterFn scatter;
};

// Physical pages in a 64 KiB-page window, as the measured callers use them: reserve a window, allocate pages, map them,
// remap them elsewhere and in another order, unmap, free.
void test_physical_pages(const PhysicalApi& api) {
    constexpr uint64_t kPages = 4;
    auto* const window = static_cast<uint8_t*>(api.virtual_alloc(
        nullptr, kPages * kPage64k, MEM_RESERVE | kXmem64kPages, 0x23, PAGE_READWRITE | PAGE_GRAPHICS_READWRITE));
    check(window != nullptr, "XMemVirtualAlloc reserves a 64 KiB-page window");
    auto* const second = static_cast<uint8_t*>(api.virtual_alloc(
        nullptr, kPages * kPage64k, MEM_RESERVE | kXmem64kPages, 0x21, PAGE_READWRITE));
    check(second != nullptr, "XMemVirtualAlloc reserves a second window");
    if (!window || !second) return;
    check(reserved_only(window), "the window is reserved, not committed");

    uint64_t pages[kPages]{};
    uint64_t count = kPages;
    check(api.allocate(0x280, &count, pages) == TRUE && count == kPages,
          "XMemAllocatePhysicalPages allocates every page asked for and keeps the count");
    bool numbers_ok = true;
    for (uint64_t i = 0; i < kPages; ++i) {
        numbers_ok &= pages[i] != 0 && pages[i] % 16 == 0 && pages[i] < (uint64_t{1} << 25);
        for (uint64_t j = 0; j < i; ++j) numbers_ok &= pages[i] != pages[j];
    }
    check(numbers_ok, "page numbers are distinct non-zero multiples of 16 below 2^25");

    check(api.map(window, kPages, pages) == window, "XMemMapPhysicalPages returns the address it mapped");
    for (uint64_t i = 0; i < kPages; ++i) fill_page(window + i * kPage64k, 0x1000u * static_cast<uint32_t>(i + 1));
    {
        MEMORY_BASIC_INFORMATION info{};
        VirtualQuery(window, &info, sizeof(info));
        check(info.State == MEM_COMMIT && info.Type == MEM_MAPPED && info.Protect == PAGE_READWRITE,
              "a mapped page is a read-write view with the window's protection");
    }

    check(api.map(window, kPages, nullptr) == window, "XMemMapPhysicalPages with no page array unmaps");
    check(reserved_only(window) && reserved_only(window + 3 * kPage64k), "unmapped pages are reserved again");

    // Remap in reverse order into the other window: each page keeps its contents.
    uint64_t reversed[kPages];
    for (uint64_t i = 0; i < kPages; ++i) reversed[i] = pages[kPages - 1 - i];
    check(api.map(second, kPages, reversed) == second, "the pages map into another window");
    bool kept = true;
    for (uint64_t i = 0; i < kPages; ++i) {
        kept &= page_holds(second + i * kPage64k, 0x1000u * static_cast<uint32_t>(kPages - i));
    }
    check(kept, "a physical page keeps its contents across unmap and remap");

    // One page mapped at two addresses is one memory.
    check(api.map(window, 1, &pages[0]) == window, "a page already mapped elsewhere maps again");
    static_cast<volatile uint32_t*>(static_cast<void*>(window))[1] = 0xfeedf00d;
    check(read_u32(second + 3 * kPage64k, 4) == 0xfeedf00d, "both mappings of a page see the same memory");

    // Mapping over a mapped page replaces it; a 0 entry unmaps one page.
    const uint64_t replace[2] = {pages[1], 0};
    check(api.map(window, 2, replace) == window, "mapping over mapped pages replaces them");
    check(page_holds(window, 0x2000u) && reserved_only(window + kPage64k),
          "the first page now shows page 1 and a 0 entry left the second unmapped");

    // Scatter: entry i maps page i at address i, out of order; 0 unmaps.
    void* const addresses[3] = {window + 3 * kPage64k, window + 2 * kPage64k, window};
    const uint64_t scattered[3] = {pages[2], pages[3], 0};
    check(api.scatter(addresses, 3, scattered) == TRUE, "XMemMapPhysicalPagesScatter maps each page at its address");
    check(page_holds(window + 3 * kPage64k, 0x3000u) && page_holds(window + 2 * kPage64k, 0x4000u) &&
              reserved_only(window),
          "scattered pages land at their own addresses and a 0 entry unmaps");
    check(api.scatter(addresses, 3, nullptr) == TRUE && reserved_only(window + 3 * kPage64k),
          "XMemMapPhysicalPagesScatter with no page array unmaps");

    uint8_t outside = 0;
    SetLastError(ERROR_SUCCESS);
    check(api.map(reinterpret_cast<void*>((reinterpret_cast<uintptr_t>(&outside) + 0xffff) & ~uintptr_t{0xffff}), 1,
                  pages) == nullptr,
          "mapping outside any window fails");
    check(api.map(window + 1, 1, pages) == nullptr && GetLastError() == ERROR_INVALID_PARAMETER,
          "mapping at an address that is not 64 KiB-aligned fails");
    const uint64_t bogus = 0x1fffff0;  // the highest page number, far above the few this test allocates
    check(api.map(window, 1, &bogus) == nullptr, "mapping a page number that was never allocated fails");

    check(api.map(second, kPages, nullptr) == second, "unmap the second window");
    check(api.free_pages(kPages, pages) == TRUE, "XMemFreePhysicalPages frees the pages");
    check(api.free_pages(1, pages) == FALSE, "freeing a page twice fails");

    // Freed numbers are reused.
    uint64_t again[kPages]{};
    count = kPages;
    check(api.allocate(0x180, &count, again) == TRUE && count == kPages, "pages can be allocated again");
    bool reused = true;
    for (uint64_t i = 0; i < kPages; ++i) {
        bool found = false;
        for (uint64_t j = 0; j < kPages; ++j) found |= again[i] == pages[j];
        reused &= found;
    }
    check(reused, "freed page numbers are reused");
    check(api.map(window, 1, again) == window && read_u32(window, 0) == 0, "a fresh page maps and reads as zero");
    api.map(window, 1, nullptr);
    api.free_pages(kPages, again);

    uint64_t zero_count = 0;
    check(api.allocate(0x280, &zero_count, nullptr) == TRUE, "allocating zero pages succeeds");
    check(api.allocate(0x280, nullptr, pages) == FALSE, "a null count fails");
    // The windows were split into one placeholder per page; the process exit releases them.
}

// The placeholder flags pass through: reserve a placeholder, carve windows and committed memory out of it.
void test_placeholders(const PhysicalApi& api) {
    auto* const base = static_cast<uint8_t*>(
        api.virtual_alloc(nullptr, 6 * kPage64k, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, 7, PAGE_NOACCESS));
    check(base != nullptr, "XMemVirtualAlloc reserves a MEM_RESERVE_PLACEHOLDER region");
    if (!base) return;

    void* const window = api.virtual_alloc(base + kPage64k, 2 * kPage64k,
                                           MEM_RESERVE | MEM_REPLACE_PLACEHOLDER | kXmem64kPages, 0x27,
                                           PAGE_READWRITE | PAGE_GRAPHICS_READONLY);
    check(window == base + kPage64k, "a reserve-only MEM_REPLACE_PLACEHOLDER returns the address it replaced");

    auto* const committed = static_cast<uint8_t*>(
        api.virtual_alloc(base + 4 * kPage64k, kPage64k,
                          MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER | kXmem64kPages, 0x27, PAGE_READWRITE));
    check(committed == base + 4 * kPage64k, "a committing MEM_REPLACE_PLACEHOLDER returns the address it replaced");
    if (committed) {
        committed[0] = 0x5a;
        check(committed[0] == 0x5a, "the committed replacement is writable");
    }

    uint64_t pages[2]{};
    uint64_t count = 2;
    check(api.allocate(0x380, &count, pages) == TRUE && count == 2, "allocate two pages for the carved window");
    check(api.map(window, 2, pages) == window, "pages map into a window carved out of a placeholder");
    if (window) {
        fill_page(static_cast<uint8_t*>(window) + kPage64k, 0xabc0000u);
        check(page_holds(static_cast<uint8_t*>(window) + kPage64k, 0xabc0000u), "the carved window holds data");
    }
    check(api.map(window, 2, nullptr) == window, "unmap the carved window");
    check(api.free_pages(2, pages) == TRUE, "free the carved window's pages");
    check(reserved_only(base), "the rest of the placeholder is still reserved");
}

// gta6rt --trace: one "xmem_attributes" event per distinct raw attribute word and API, none with the trace off.
void test_attribute_trace(XMemAllocFn alloc, XMemFreeFn free_mem, XMemVirtualAllocFn virtual_alloc) {
    wchar_t temp[MAX_PATH];
    GetTempPathW(MAX_PATH, temp);
    const std::wstring path =
        std::wstring(temp) + L"xmem_trace_test_" + std::to_wstring(GetCurrentProcessId()) + L".jsonl";
    DeleteFileW(path.c_str());

    void* untraced = alloc(64, 0x1111);  // trace not open yet: nothing to write, nothing to crash
    check(untraced != nullptr, "XMemAlloc with the trace off still allocates");
    free_mem(untraced, 0x1111);

    check(rt::startup_trace::open(path.c_str()), "open the startup trace file");
    void* a = alloc(4096, 0x5e50000);
    void* b = alloc(8192, 0x5e50000);  // same word again: no second event
    void* c = alloc(16, 0x99);
    free_mem(a, 0x5e50000);
    free_mem(b, 0x5e50000);  // same word again: no second event
    free_mem(c, 0x99);
    free_mem(nullptr, 0x77);  // a free-only word is recorded too
    void* const v = virtual_alloc ? virtual_alloc(nullptr, 0x10000, MEM_RESERVE | MEM_COMMIT, 0x183, PAGE_READWRITE)
                                  : nullptr;
    if (v) VirtualFree(v, 0, MEM_RELEASE);
    rt::startup_trace::close();

    std::ifstream in(path);
    std::stringstream text;
    text << in.rdbuf();
    const std::string log = text.str();
    auto count = [&](const std::string& needle) {
        int n = 0;
        for (size_t at = log.find(needle); at != std::string::npos; at = log.find(needle, at + 1)) ++n;
        return n;
    };
    check(count("\"kind\":\"xmem_attributes\"") == 6, "six events: two XMemAlloc words, three XMemFree words and one XMemVirtualAlloc word");
    check(count("\"api\":\"XMemVirtualAlloc\",\"attributes\":\"0x183\",\"size\":65536") == 1,
          "XMemVirtualAlloc 0x183 is recorded once, with its size");
    check(count("\"api\":\"XMemAlloc\",\"attributes\":\"0x5e50000\",\"size\":4096") == 1,
          "XMemAlloc 0x5e50000 is recorded once, with the size of the first call");
    check(count("\"api\":\"XMemAlloc\",\"attributes\":\"0x99\",\"size\":16") == 1, "XMemAlloc 0x99 is recorded once");
    check(count("\"api\":\"XMemFree\",\"attributes\":\"0x5e50000\"") == 1, "XMemFree 0x5e50000 is recorded once");
    check(count("\"api\":\"XMemFree\",\"attributes\":\"0x77\"") == 1, "XMemFree 0x77 is recorded");
    check(count("\"attributes\":\"0x1111\"") == 0, "nothing was recorded while the trace was off");
    check(count("\"caller\":\"") == 6, "every event names its caller");
    DeleteFileW(path.c_str());
}

// XMemTransferMemory moves budget between the console's title and system partitions; a PC has neither, so the shim
// succeeds for every direction and size and moves (allocates) nothing. The XMEM_TRANSFER_TYPE values are not public,
// so the directions here are just the first few enum values and one far outside any plausible range.
void test_transfer_memory(XMemTransferMemoryFn transfer) {
    constexpr SIZE_T kMiB = 1024 * 1024;
    const uint32_t types[] = {0u, 1u, 2u, 0xffffffffu};
    // 2 MiB and 40 MiB are the sizes the samples pass ("Must be multiples of 2MB sizes"); 0, an odd size and SIZE_MAX
    // are the boundaries.
    const SIZE_T sizes[] = {0, 2 * kMiB, 40 * kMiB, 2 * kMiB + 1, (std::numeric_limits<SIZE_T>::max)()};

    PROCESS_MEMORY_COUNTERS_EX before{};
    const bool have_before = GetProcessMemoryInfo(GetCurrentProcess(),
                                                  reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&before), sizeof(before));
    bool all_true = true;
    bool last_error_kept = true;
    for (const uint32_t type : types) {
        for (const SIZE_T size : sizes) {
            SetLastError(0x2bad);
            const BOOL result = transfer(type, size);
            if (result != TRUE) {
                all_true = false;
                std::printf("  XMemTransferMemory(%u, %zu) -> %d\n", type, static_cast<size_t>(size), result);
            }
            last_error_kept &= GetLastError() == 0x2bad;
        }
    }
    check(all_true, "XMemTransferMemory returns TRUE (1) for every direction and size, 0 and SIZE_MAX included");
    check(last_error_kept, "XMemTransferMemory leaves the thread's last error alone");

    PROCESS_MEMORY_COUNTERS_EX after{};
    const bool have_after = GetProcessMemoryInfo(GetCurrentProcess(),
                                                 reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&after), sizeof(after));
    // 20 calls asked to move up to 40 MiB each; nothing may be committed for them. 1 MiB of slack covers the CRT's
    // own buffers (the one-time stderr line).
    check(have_before && have_after && after.PrivateUsage < before.PrivateUsage + kMiB,
          "XMemTransferMemory commits no memory for the transferred sizes");

    // A bool-returning declaration reads only AL: TRUE must be non-zero there as well.
    const auto as_bool = reinterpret_cast<bool(WINAPI*)(uint32_t, SIZE_T)>(reinterpret_cast<void*>(transfer));
    check(as_bool(0, 2 * kMiB), "XMemTransferMemory's result also reads as true through a bool return");

    // The enum is 32 bits: garbage in the upper half of RCX must not change anything.
    const auto wide = reinterpret_cast<BOOL(WINAPI*)(uint64_t, SIZE_T)>(reinterpret_cast<void*>(transfer));
    check(wide(0xdead000000000000ull, 2 * kMiB) == TRUE, "XMemTransferMemory ignores the upper half of RCX");
}

// gta6rt --trace: one "xmem_transfer" event per distinct (type, size), none with the trace off.
void test_transfer_trace(XMemTransferMemoryFn transfer) {
    wchar_t temp[MAX_PATH];
    GetTempPathW(MAX_PATH, temp);
    const std::wstring path =
        std::wstring(temp) + L"xmem_transfer_trace_test_" + std::to_wstring(GetCurrentProcessId()) + L".jsonl";
    DeleteFileW(path.c_str());

    check(transfer(7, 12345) == TRUE, "XMemTransferMemory with the trace off still returns TRUE");

    check(rt::startup_trace::open(path.c_str()), "open the startup trace file");
    transfer(0, 2097152);
    transfer(0, 2097152);  // same pair again: no second event
    transfer(0, 41943040);
    transfer(1, 2097152);
    rt::startup_trace::close();

    std::ifstream in(path);
    std::stringstream text;
    text << in.rdbuf();
    const std::string log = text.str();
    auto count = [&](const std::string& needle) {
        int n = 0;
        for (size_t at = log.find(needle); at != std::string::npos; at = log.find(needle, at + 1)) ++n;
        return n;
    };
    check(count("\"kind\":\"xmem_transfer\"") == 3, "three xmem_transfer events: one per distinct (type, size)");
    check(count("\"api\":\"XMemTransferMemory\",\"type\":0,\"size\":2097152") == 1,
          "type 0 / 2 MiB is recorded once although it was called twice");
    check(count("\"api\":\"XMemTransferMemory\",\"type\":0,\"size\":41943040") == 1, "type 0 / 40 MiB is recorded");
    check(count("\"api\":\"XMemTransferMemory\",\"type\":1,\"size\":2097152") == 1, "type 1 / 2 MiB is recorded");
    check(count("\"size\":12345") == 0, "nothing was recorded while the trace was off");
    check(count("\"caller\":\"") == 3, "every transfer event names its caller");
    DeleteFileW(path.c_str());
}

using HostVirtualAllocFn = void*(WINAPI*)(void*, SIZE_T, DWORD, DWORD);

// RT-217: a title's kernel32!VirtualAlloc carries the console's MEM_GRAPHICS / MEM_TITLE type bits, which the host
// rejects. The registered shim has to drop them and still allocate.
void test_title_virtual_alloc() {
    constexpr DWORD kMemGraphics = 0x10000000;
    constexpr DWORD kMemTitle = 0x40000000;
    const char* const dlls[] = {"kernel32.dll", "kernelbase.dll", "api-ms-win-core-memory-l1-1-0.dll"};
    for (const char* dll : dlls) {
        const auto alloc = reinterpret_cast<HostVirtualAllocFn>(rt::shims::find(dll, "VirtualAlloc"));
        char what[160];
        std::snprintf(what, sizeof what, "registry exposes %s!VirtualAlloc", dll);
        check(alloc != nullptr, what);
        if (!alloc) continue;

        void* const gfx = alloc(nullptr, 0x10000, MEM_RESERVE | MEM_COMMIT, 0x2004);  // PAGE_GRAPHICS_READWRITE
        check(gfx != nullptr, "title VirtualAlloc with protect 0x2004 succeeds");
        if (gfx) {
            MEMORY_BASIC_INFORMATION mbi{};
            check(VirtualQuery(gfx, &mbi, sizeof mbi) && mbi.Protect == PAGE_READWRITE,
                  "protect 0x2004 commits PAGE_READWRITE");
            VirtualFree(gfx, 0, MEM_RELEASE);
        }
        for (const DWORD bits : {kMemGraphics, kMemTitle, kMemGraphics | kMemTitle, DWORD{0}}) {
            void* const p = alloc(nullptr, 0x10000, MEM_RESERVE | MEM_COMMIT | bits, PAGE_READWRITE);
            std::snprintf(what, sizeof what, "%s!VirtualAlloc with type bits %#lx returns memory", dll,
                          static_cast<unsigned long>(bits));
            check(p != nullptr, what);
            if (!p) continue;
            static_cast<volatile uint8_t*>(p)[0] = 0x5a;
            check(static_cast<volatile uint8_t*>(p)[0] == 0x5a, "the allocation is committed and writable");
            check(VirtualFree(p, 0, MEM_RELEASE), "release the allocation");
        }
    }
}

void print_result() {
    if (g_failures == 0) {
        std::printf("xmem_test: OK (%d checks)\n", g_checks);
    } else {
        std::printf("xmem_test: FAILED (%d checks, %d failure%s)\n", g_checks, g_failures,
                    g_failures == 1 ? "" : "s");
    }
}

}  // namespace

int main() {
    std::printf("xmem_test: registered xmem.dll shim ABI\n");

    const auto alloc = reinterpret_cast<XMemAllocFn>(rt::shims::find("xmem.dll", "XMemAlloc"));
    const auto free_mem = reinterpret_cast<XMemFreeFn>(rt::shims::find("xmem.dll", "XMemFree"));
    const auto query = reinterpret_cast<XMemVirtualQueryFn>(rt::shims::find("xmem.dll", "XMemVirtualQuery"));
    const auto get_stats =
        reinterpret_cast<XMemGetWorkingSetStatisticsFn>(rt::shims::find("xmem.dll", "XMemGetWorkingSetStatistics"));
    const auto virtual_alloc =
        reinterpret_cast<XMemVirtualAllocFn>(rt::shims::find("xmem.dll", "XMemVirtualAlloc"));
    const auto get_write_watch =
        reinterpret_cast<XMemGetWriteWatchFn>(rt::shims::find("xmem.dll", "XMemGetWriteWatch"));
    const auto reset_write_watch =
        reinterpret_cast<XMemResetWriteWatchFn>(rt::shims::find("xmem.dll", "XMemResetWriteWatch"));
    const auto transfer =
        reinterpret_cast<XMemTransferMemoryFn>(rt::shims::find("xmem.dll", "XMemTransferMemory"));

    check(alloc != nullptr, "registry exposes xmem.dll!XMemAlloc");
    check(free_mem != nullptr, "registry exposes xmem.dll!XMemFree");
    check(query != nullptr, "registry exposes xmem.dll!XMemVirtualQuery");
    check(get_stats != nullptr, "registry exposes xmem.dll!XMemGetWorkingSetStatistics");
    check(virtual_alloc != nullptr, "registry exposes xmem.dll!XMemVirtualAlloc");
    check(get_write_watch != nullptr, "registry exposes xmem.dll!XMemGetWriteWatch");
    check(reset_write_watch != nullptr, "registry exposes xmem.dll!XMemResetWriteWatch");
    check(transfer != nullptr, "registry exposes xmem.dll!XMemTransferMemory");

    const auto alloc_default = reinterpret_cast<XMemAllocFn>(rt::shims::find("xmem.dll", "XMemAllocDefault"));
    const auto free_default = reinterpret_cast<XMemFreeFn>(rt::shims::find("xmem.dll", "XMemFreeDefault"));
    const auto tools_mem =
        reinterpret_cast<XMemIsToolsMemAvailableFn>(rt::shims::find("xmem.dll", "XMemIsToolsMemAvailable"));
    const PhysicalApi physical{
        virtual_alloc,
        reinterpret_cast<XMemAllocatePhysicalPagesFn>(rt::shims::find("xmem.dll", "XMemAllocatePhysicalPages")),
        reinterpret_cast<XMemFreePhysicalPagesFn>(rt::shims::find("xmem.dll", "XMemFreePhysicalPages")),
        reinterpret_cast<XMemMapPhysicalPagesFn>(rt::shims::find("xmem.dll", "XMemMapPhysicalPages")),
        reinterpret_cast<XMemMapPhysicalPagesScatterFn>(rt::shims::find("xmem.dll", "XMemMapPhysicalPagesScatter")),
    };
    check(alloc_default != nullptr, "registry exposes xmem.dll!XMemAllocDefault");
    check(free_default != nullptr, "registry exposes xmem.dll!XMemFreeDefault");
    check(tools_mem != nullptr, "registry exposes xmem.dll!XMemIsToolsMemAvailable");
    check(physical.allocate != nullptr, "registry exposes xmem.dll!XMemAllocatePhysicalPages");
    check(physical.free_pages != nullptr, "registry exposes xmem.dll!XMemFreePhysicalPages");
    check(physical.map != nullptr, "registry exposes xmem.dll!XMemMapPhysicalPages");
    check(physical.scatter != nullptr, "registry exposes xmem.dll!XMemMapPhysicalPagesScatter");

    if (!alloc || !free_mem || !query || !get_stats) {
        print_result();
        return 1;
    }

    test_alloc(alloc, free_mem);
    test_working_set(get_stats);
    test_virtual_query(query);
    if (virtual_alloc && get_write_watch && reset_write_watch) {
        test_virtual_alloc(virtual_alloc, get_write_watch, reset_write_watch);
    }
    test_attribute_trace(alloc, free_mem, virtual_alloc);
    if (transfer) {
        test_transfer_memory(transfer);
        test_transfer_trace(transfer);
    }

    test_title_virtual_alloc();

    if (alloc_default && free_default) test_default_alloc(alloc, free_mem, alloc_default, free_default);
    if (tools_mem) check(tools_mem() == FALSE, "XMemIsToolsMemAvailable returns FALSE (no devkit tools memory)");
    if (virtual_alloc && physical.allocate && physical.free_pages && physical.map && physical.scatter) {
        test_physical_pages(physical);
        test_placeholders(physical);
    }

    print_result();
    return g_failures ? 1 : 0;
}
