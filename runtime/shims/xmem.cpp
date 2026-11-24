// xmem.dll: the GDK's tracked allocator. Mapped onto the CRT's aligned heap and VirtualQueryEx.
//
// The measured ABI (docs from the clean-room spec) shows four imports:
//   XMemAlloc(size, attributes)
//   XMemFree(ptr, attributes)
//   XMemVirtualQuery(process, address, out64)
//   XMemGetWorkingSetStatistics(kind, out)   (kind 0x20000001 at all seven FSR call sites)
//
// ASSUMPTION: the attribute word is not decoded by the measured call sites (one passes 0x5e50000). Until a call
// site shows the alignment/type fields, every block is 64-byte aligned (covers cache lines and SIMD) and a size of
// zero is rounded up to one byte so the result is a stable, non-null pointer that XMemFree can release.
//
// The default allocators, the physical-page functions and XMemIsToolsMemAvailable follow argument shapes measured
// locally at a title's call sites (each function says which). The page functions are built on the Windows placeholder
// APIs (VirtualAlloc2, MapViewOfFile3), whose flags the title's own XMemVirtualAlloc calls already use.
#include <windows.h>
#include <intrin.h>
#include <malloc.h>
#include <psapi.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "shims.h"
#include "startup_trace.h"

namespace {

constexpr size_t kAlign = 64;
constexpr size_t kNoSize = static_cast<size_t>(-1);  // XMemFree has no size field

std::string caller_field(const void* caller) {
    return ",\"caller\":" + rt::startup_trace::quote(rt::startup_trace::where(reinterpret_cast<uint64_t>(caller)));
}

// gta6rt --trace FILE: the first XMemAlloc, XMemFree, XMemAllocDefault, XMemFreeDefault and XMemVirtualAlloc call for
// each distinct raw attribute word go to the startup trace as an "xmem_attributes" event (api, the word, the size for an
// allocation, the caller as module+RVA). The bits of the attribute word are not known yet; this records them from a
// real run without logging every allocation, which would fill the file. Off: one relaxed load per call.
void note_attributes(const char* api, uint64_t attributes, size_t size, const void* caller) {
    if (!rt::startup_trace::enabled()) return;
    static std::mutex mutex;
    static std::set<std::pair<std::string, uint64_t>> seen;
    {
        const std::lock_guard lock(mutex);
        if (!seen.insert({api, attributes}).second) return;
    }
    std::string fields = std::string("\"api\":\"") + api +
                         "\",\"attributes\":" + rt::startup_trace::quote(rt::startup_trace::hex(attributes));
    if (size != kNoSize) fields += ",\"size\":" + std::to_string(size);
    fields += caller_field(caller);
    rt::startup_trace::event("xmem_attributes", fields);
}

void* allocate_block(size_t size) { return _aligned_malloc(size ? size : 1, kAlign); }

void* __stdcall XMemAlloc(size_t size, uint64_t attributes) {
    note_attributes("XMemAlloc", attributes, size, _ReturnAddress());
    return allocate_block(size);
}

void __stdcall XMemFree(void* p, uint64_t attributes) {
    note_attributes("XMemFree", attributes, kNoSize, _ReturnAddress());
    _aligned_free(p);
}

// XMemAllocDefault / XMemFreeDefault: the GDK's own allocator behind XMemAlloc and XMemFree, which a title that hooks
// XMemAlloc forwards to. The measured title reaches each through one wrapper that is a bare tail jump to the import,
// so they take exactly XMemAlloc's and XMemFree's arguments. They share the heap: a block from either allocator may be
// released by either free.
void* __stdcall XMemAllocDefault(size_t size, uint64_t attributes) {
    note_attributes("XMemAllocDefault", attributes, size, _ReturnAddress());
    return allocate_block(size);
}

void __stdcall XMemFreeDefault(void* p, uint64_t attributes) {
    note_attributes("XMemFreeDefault", attributes, kNoSize, _ReturnAddress());
    _aligned_free(p);
}

// BOOL XMemIsToolsMemAvailable(void). The measured title calls it from one helper, which six out-of-memory paths reach
// only after an allocation has failed; the helper breaks into the debugger when the result is zero. Tools memory is
// the extra memory of a development kit. A PC has none, so the answer is FALSE, and the title's own breakpoint marks
// where it ran out of memory.
BOOL WINAPI XMemIsToolsMemAvailable() {
    static std::atomic<bool> warned{false};
    if (!warned.exchange(true, std::memory_order_relaxed)) {
        std::fprintf(stderr, "[rt] xmem: XMemIsToolsMemAvailable -> FALSE (no devkit tools memory; logged once)\n");
    }
    return FALSE;
}

// Measured call sites pass a zeroed 64-byte structure. Only the MEMORY_BASIC_INFORMATION-compatible prefix is
// known: callers read at least the base and size fields. Fill that prefix from VirtualQueryEx and zero the rest.
constexpr size_t kVirtualQueryOutSize = 64;
static_assert(sizeof(MEMORY_BASIC_INFORMATION) <= kVirtualQueryOutSize,
              "the 64-byte XMemVirtualQuery output must hold MEMORY_BASIC_INFORMATION");

// Returns the number of bytes written (the VirtualQuery convention) or 0 on failure. The exact return type and the
// meaning of every output field are not fully known; callers in the package only test for success.
SIZE_T __stdcall XMemVirtualQuery(HANDLE process, const void* address, void* out) {
    if (!out) return 0;
    std::memset(out, 0, kVirtualQueryOutSize);
    if (!address) return 0;

    static std::atomic<bool> warned{false};
    if (!warned.exchange(true, std::memory_order_relaxed)) {
        std::fprintf(stderr,
                     "[rt] xmem: XMemVirtualQuery: fields after MEMORY_BASIC_INFORMATION are unknown; "
                     "the 64-byte output is zero-filled there\n");
    }

    // A null handle means the current process; GetCurrentProcess() is the (HANDLE)-1 pseudo handle.
    HANDLE target = process ? process : GetCurrentProcess();
    MEMORY_BASIC_INFORMATION info{};
    SIZE_T written = VirtualQueryEx(target, address, &info, sizeof(info));
    if (!written) return 0;

    std::memcpy(out, &info, written < kVirtualQueryOutSize ? written : kVirtualQueryOutSize);
    return written;
}

// BOOL XMemGetWorkingSetStatistics(UINT32 kind, XMEM_WORKING_SET_STATISTICS* stats). Measured at all seven call sites
// in amd_fidelityfx_upscaler_xbox.dll (1.0.2.1; calls at RVAs 0x26b3, 0x2ea1, 0x39a9, 0x7452, 0x74d2, 0x7568,
// 0x75b1): ECX is the constant 0x20000001 (its meaning is not known) and RDX points at a stack buffer (0x80 bytes
// where the frame shows it). The caller reads a 64-bit size at +8 and treats a zero return as "no data". Taking the
// pointer from the first argument wrote to address 0x20000001.
// The measured FSR sites establish argument shape, kind and the 64-bit working-set read at +8; +0's title-total memory
// limit interpretation is inferred from other Series implementations (xcl, xbox-emulator) and title engine telemetry
// sizing memory pools (RT-220). Note that if GlobalMemoryStatusEx reports 0 physical memory, XMem falls back to the
// 12 GiB title limit, whereas XSystemGetAppMemoryUsageLimit remains 0.
BOOL __stdcall XMemGetWorkingSetStatistics(uint32_t /*kind*/, void* stats) {
    if (!stats) {
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true, std::memory_order_relaxed)) {
            std::fprintf(stderr, "[rt] xmem: XMemGetWorkingSetStatistics(nullptr) -> FALSE\n");
        }
        return FALSE;
    }

    PROCESS_MEMORY_COUNTERS pmc{};
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) return FALSE;

    MEMORYSTATUSEX ms{sizeof(ms)};
    uint64_t total_memory = 12ULL * 1024 * 1024 * 1024;
    if (GlobalMemoryStatusEx(&ms) && ms.ullTotalPhys != 0) {
        total_memory = ms.ullTotalPhys;
    }

    uint8_t* bytes = static_cast<uint8_t*>(stats);
    std::memset(bytes, 0, 16);
    std::memcpy(bytes + 0, &total_memory, sizeof(total_memory));
    uint64_t working_set = pmc.WorkingSetSize;
    std::memcpy(bytes + 8, &working_set, sizeof(working_set));
    return TRUE;
}

// The placeholder APIs (Windows 10 1803 and later) come from kernelbase.dll at run time, so neither gta6rt nor the
// tests link onecore.lib. A host without them fails the calls that need them with ERROR_NOT_SUPPORTED.
using VirtualAlloc2Fn = PVOID(WINAPI*)(HANDLE, PVOID, SIZE_T, ULONG, ULONG, MEM_EXTENDED_PARAMETER*, ULONG);
using MapViewOfFile3Fn = PVOID(WINAPI*)(HANDLE, HANDLE, PVOID, ULONG64, SIZE_T, ULONG, ULONG, MEM_EXTENDED_PARAMETER*,
                                        ULONG);
using UnmapViewOfFile2Fn = BOOL(WINAPI*)(HANDLE, PVOID, ULONG);

struct PlaceholderApi {
    VirtualAlloc2Fn virtual_alloc2 = nullptr;
    MapViewOfFile3Fn map_view3 = nullptr;
    UnmapViewOfFile2Fn unmap_view2 = nullptr;
    bool available() const { return virtual_alloc2 && map_view3 && unmap_view2; }
};

const PlaceholderApi& placeholder_api() {
    static const PlaceholderApi api = [] {
        PlaceholderApi resolved;
        if (const HMODULE kernelbase = GetModuleHandleW(L"kernelbase.dll")) {
            resolved.virtual_alloc2 = reinterpret_cast<VirtualAlloc2Fn>(GetProcAddress(kernelbase, "VirtualAlloc2"));
            resolved.map_view3 = reinterpret_cast<MapViewOfFile3Fn>(GetProcAddress(kernelbase, "MapViewOfFile3"));
            resolved.unmap_view2 =
                reinterpret_cast<UnmapViewOfFile2Fn>(GetProcAddress(kernelbase, "UnmapViewOfFile2"));
        }
        return resolved;
    }();
    return api;
}

// XMem allocation-type page bits select the guest page granularity and must not be forwarded as Win32 flags.
constexpr DWORD kXmemLargePages = 0x20000000;  // 2 MiB pages
constexpr DWORD kXmem64kPages = 0x20400000;    // 64 KiB pages
constexpr DWORD kXmemPageModeMask = kXmem64kPages;
constexpr SIZE_T kXmemPageSize64k = 0x10000;
constexpr SIZE_T kXmemPageSize2m = 0x200000;
constexpr DWORD kXmemHostAllocationTypeMask = MEM_COMMIT | MEM_RESERVE | MEM_REPLACE_PLACEHOLDER |
                                              MEM_RESERVE_PLACEHOLDER | MEM_RESET | MEM_TOP_DOWN | MEM_WRITE_WATCH |
                                              MEM_RESET_UNDO;

bool round_up_size(SIZE_T value, SIZE_T alignment, SIZE_T& rounded) {
    const SIZE_T max = (std::numeric_limits<SIZE_T>::max)();
    if (alignment == 0 || (alignment & (alignment - 1)) != 0 || value > max - (alignment - 1)) return false;
    rounded = (value + alignment - 1) & ~(alignment - 1);
    return true;
}

// The console's PAGE_GRAPHICS_* protection bits (winnt.h: PAGE_GRAPHICS_NOACCESS 0x800 .. PAGE_GRAPHICS_NOCACHE
// 0x40000). The host refuses them with ERROR_INVALID_PARAMETER, and the D3D12 layer reads guest memory through the
// CPU mapping, so only the CPU protection is applied. The retail executable passes 0x801 and 0x22004.
constexpr DWORD kPageGraphicsMask = 0x0007F800;

DWORD host_protect(DWORD p) {
    const DWORD cpu = p & ~kPageGraphicsMask;
    return cpu ? cpu : PAGE_NOACCESS;
}

// Physical pages and the windows they are mapped into. One mutex covers both tables and every map and unmap.
//
// A physical-page window is a reserve-only XMemVirtualAlloc in a guest page mode, or a reserve-only replacement of
// part of a placeholder. On the host it is a placeholder, the only kind of reservation a section view can later
// replace, and its protection is kept here: the map calls take none, so a mapped page gets its window's protection, as
// AWE's MapUserPhysicalPages does.
//
// ASSUMPTION: a title does not commit memory inside such a window with kernel32!VirtualAlloc. A placeholder refuses a
// plain commit; the measured windows are only ever passed to the map calls.
struct Window {
    SIZE_T size;
    DWORD protect;
};

struct Section {
    HANDLE handle;
    explicit Section(HANDLE h) : handle(h) {}
    Section(const Section&) = delete;
    Section& operator=(const Section&) = delete;
    ~Section() { CloseHandle(handle); }
};

struct PhysicalPage {
    std::shared_ptr<Section> section;  // a mapped view holds its own reference to the section
    uint64_t offset;
};

constexpr uint64_t kPhysicalPageSize = 0x10000;
// Page numbers are 4 KiB frame numbers of 64 KiB pages: the measured title stores them packed and assumes each is a
// non-zero multiple of 16 below 2^25, with 0 for an entry it unmaps. So a page number is slot * 16 with slot in
// [1, 2^21), and freed slots are reused so a long run never runs out.
constexpr uint64_t kFramesPerPage = kPhysicalPageSize / 0x1000;
constexpr uint64_t kPageSlotLimit = uint64_t{1} << 21;

std::mutex g_phys_mutex;
std::map<uintptr_t, Window> g_windows;              // base -> window
std::unordered_map<uint64_t, PhysicalPage> g_pages;  // page number -> backing
std::vector<uint64_t> g_free_slots;
uint64_t g_next_slot = 1;
std::unordered_map<uintptr_t, uint64_t> g_mapped;  // mapped 64 KiB address -> page number

// With g_phys_mutex held. A window replaces every window it overlaps: the title released those regions (with
// kernel32!VirtualFree, which the shim does not see) before the host could hand the addresses out again.
void record_window(void* base, SIZE_T size, DWORD protect) {
    const uintptr_t start = reinterpret_cast<uintptr_t>(base);
    const uintptr_t end = start + size;
    auto it = g_windows.lower_bound(start);
    if (it != g_windows.begin()) {
        const auto prev = std::prev(it);
        if (prev->first + prev->second.size > start) it = prev;
    }
    while (it != g_windows.end() && it->first < end) it = g_windows.erase(it);
    g_windows[start] = Window{size, protect};
}

// With g_phys_mutex held: the window holding [address, address + bytes), if one does.
const Window* find_window(uintptr_t address, uint64_t bytes) {
    auto it = g_windows.upper_bound(address);
    if (it == g_windows.begin()) return nullptr;
    --it;
    const uint64_t offset = address - it->first;
    if (offset >= it->second.size || bytes > it->second.size - offset) return nullptr;
    return &it->second;
}

// [address, address + size) as one placeholder of its own: split it out of the placeholder that holds it. Fails when
// the range is not inside one placeholder.
bool isolate_placeholder(void* address, SIZE_T size) {
    MEMORY_BASIC_INFORMATION info{};
    if (!VirtualQuery(address, &info, sizeof(info)) || info.State != MEM_RESERVE) {
        SetLastError(ERROR_INVALID_ADDRESS);
        return false;
    }
    if (info.AllocationBase == address && info.RegionSize == size) return true;
    return VirtualFree(address, size, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) != FALSE;
}

PVOID reserve_placeholder(void* address, SIZE_T size, SIZE_T alignment) {
    MEM_ADDRESS_REQUIREMENTS requirements{};
    MEM_EXTENDED_PARAMETER parameter{};
    ULONG parameter_count = 0;
    if (!address && alignment > kXmemPageSize64k) {
        requirements.Alignment = alignment;
        parameter.Type = MemExtendedParameterAddressRequirements;
        parameter.Pointer = &requirements;
        parameter_count = 1;
    }
    return placeholder_api().virtual_alloc2(GetCurrentProcess(), address, size, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                                            PAGE_NOACCESS, parameter_count ? &parameter : nullptr, parameter_count);
}

// PVOID XMemVirtualAlloc(address, size, allocationType, attributes, protect). All 23 calls of the measured title pass a
// nonzero attribute word; its bits are not decoded and nothing on the host depends on them, so it is accepted and only
// recorded (xmem_attributes). Its types cover committed 2 MiB and 64 KiB pages, reserve-only 64 KiB windows for physical pages, one
// MEM_RESERVE_PLACEHOLDER reservation and five MEM_REPLACE_PLACEHOLDER windows carved out of it, each then filled with
// XMemMapPhysicalPages. The guest page size sets the alignment and rounding of the region; the host's own pages stay
// 4 KiB (host large pages need SeLockMemoryPrivilege). A 2 MiB region the host places is reserved 2 MiB-aligned through
// VirtualAlloc2.
void* WINAPI XMemVirtualAlloc(void* address, SIZE_T size, DWORD allocation_type, ULONGLONG attributes, DWORD protect) {
    note_attributes("XMemVirtualAlloc", attributes, size, _ReturnAddress());
    if (size == 0) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return nullptr;
    }

    const DWORD page_mode = allocation_type & kXmemPageModeMask;
    if (page_mode != 0 && page_mode != kXmem64kPages && page_mode != kXmemLargePages) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return nullptr;
    }

    const DWORD unsupported = allocation_type & ~(kXmemHostAllocationTypeMask | kXmemPageModeMask);
    if (unsupported != 0) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return nullptr;
    }

    DWORD host_type = allocation_type & kXmemHostAllocationTypeMask;
    if (host_type == 0) host_type = MEM_RESERVE | MEM_COMMIT;
    const DWORD host_prot = host_protect(protect);

    const bool commit = (host_type & MEM_COMMIT) != 0;
    const bool new_placeholder = (host_type & MEM_RESERVE_PLACEHOLDER) != 0;
    const bool replace = (host_type & MEM_REPLACE_PLACEHOLDER) != 0;
    const bool window = page_mode != 0 && !commit && !new_placeholder;
    if ((new_placeholder || replace || window || page_mode == kXmemLargePages) && !placeholder_api().available()) {
        SetLastError(ERROR_NOT_SUPPORTED);
        return nullptr;
    }

    SYSTEM_INFO system_info{};
    GetSystemInfo(&system_info);
    SIZE_T page_granularity = system_info.dwPageSize;
    SIZE_T base_granularity = (host_type & MEM_RESERVE) ? system_info.dwAllocationGranularity : system_info.dwPageSize;
    if (page_mode == kXmem64kPages) page_granularity = base_granularity = kXmemPageSize64k;
    if (page_mode == kXmemLargePages) page_granularity = base_granularity = kXmemPageSize2m;
    if (new_placeholder || replace) {
        // Placeholders are split and replaced on allocation-granularity boundaries only.
        if (page_granularity < system_info.dwAllocationGranularity) page_granularity = system_info.dwAllocationGranularity;
        if (base_granularity < system_info.dwAllocationGranularity) base_granularity = system_info.dwAllocationGranularity;
    }
    if (page_granularity == 0 || (page_granularity & (page_granularity - 1)) != 0 || base_granularity == 0 ||
        (base_granularity & (base_granularity - 1)) != 0) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return nullptr;
    }

    if (address != nullptr) {
        const uintptr_t requested = reinterpret_cast<uintptr_t>(address);
        const uintptr_t max = (std::numeric_limits<uintptr_t>::max)();
        if (size > max - requested) {
            SetLastError(ERROR_INVALID_PARAMETER);
            return nullptr;
        }
        const uintptr_t requested_end = requested + size;
        const uintptr_t base = requested & ~(static_cast<uintptr_t>(base_granularity) - 1);
        if (requested_end > max - (static_cast<uintptr_t>(page_granularity) - 1)) {
            SetLastError(ERROR_INVALID_PARAMETER);
            return nullptr;
        }
        const uintptr_t end = (requested_end + static_cast<uintptr_t>(page_granularity) - 1) &
                              ~(static_cast<uintptr_t>(page_granularity) - 1);
        address = reinterpret_cast<void*>(base);
        size = static_cast<SIZE_T>(end - base);
    } else if (replace) {
        SetLastError(ERROR_INVALID_PARAMETER);  // a replacement names the placeholder it replaces
        return nullptr;
    } else if (page_granularity > system_info.dwPageSize && !round_up_size(size, page_granularity, size)) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return nullptr;
    }

    if (replace) {
        const std::lock_guard lock(g_phys_mutex);
        if (!isolate_placeholder(address, size)) return nullptr;
        if (window) {
            // A reserve-only replacement stays a placeholder: the physical pages are mapped into it.
            record_window(address, size, host_prot);
            return address;
        }
        return placeholder_api().virtual_alloc2(GetCurrentProcess(), address, size, host_type | MEM_RESERVE, host_prot,
                                                nullptr, 0);
    }
    if (new_placeholder || window) {
        void* const result = reserve_placeholder(address, size, base_granularity);
        if (result && window) {
            const std::lock_guard lock(g_phys_mutex);
            record_window(result, size, host_prot);
        }
        return result;
    }
    if (page_mode == kXmemLargePages && address == nullptr) {
        MEM_ADDRESS_REQUIREMENTS requirements{};
        requirements.Alignment = kXmemPageSize2m;
        MEM_EXTENDED_PARAMETER parameter{};
        parameter.Type = MemExtendedParameterAddressRequirements;
        parameter.Pointer = &requirements;
        return placeholder_api().virtual_alloc2(GetCurrentProcess(), nullptr, size, host_type | MEM_RESERVE, host_prot,
                                                &parameter, 1);
    }
    return VirtualAlloc(address, size, host_type, host_prot);
}

// With g_phys_mutex held. Unmaps the page at `address` if one is mapped there; the address becomes a placeholder again.
bool unmap_page(uintptr_t address) {
    const auto it = g_mapped.find(address);
    if (it == g_mapped.end()) return true;
    if (!placeholder_api().unmap_view2(GetCurrentProcess(), reinterpret_cast<void*>(address),
                                       MEM_PRESERVE_PLACEHOLDER)) {
        return false;
    }
    g_mapped.erase(it);
    return true;
}

// With g_phys_mutex held. Maps physical page `page` (0: unmap) at the 64 KiB-aligned `address`, replacing what was
// mapped there. The caller has found the window.
bool map_page(uintptr_t address, uint64_t page, DWORD protect) {
    if (!unmap_page(address)) return false;
    if (page == 0) return true;
    const auto it = g_pages.find(page);
    if (it == g_pages.end()) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return false;
    }
    void* const target = reinterpret_cast<void*>(address);
    if (!isolate_placeholder(target, kPhysicalPageSize)) return false;
    void* const view = placeholder_api().map_view3(it->second.section->handle, GetCurrentProcess(), target,
                                                   it->second.offset, kPhysicalPageSize, MEM_REPLACE_PLACEHOLDER,
                                                   protect, nullptr, 0);
    if (!view) return false;
    g_mapped[address] = page;
    return true;
}

// Two of the measured windows are reserved PAGE_NOACCESS for the CPU and the GPU alike; pages mapped there are still
// GPU memory, which the D3D12 layer reads on the CPU.
DWORD view_protect(const Window& window) {
    return window.protect == PAGE_NOACCESS ? PAGE_READWRITE : window.protect;
}

bool aligned_page_address(uintptr_t address) { return (address & (kPhysicalPageSize - 1)) == 0; }

// gta6rt --trace FILE: the first XMemAllocatePhysicalPages for each distinct attribute word goes to the startup trace
// as an "xmem_physical_pages" event with the page count and the caller.
void note_physical_attributes(uint64_t attributes, uint64_t count, const void* caller) {
    if (!rt::startup_trace::enabled()) return;
    static std::mutex mutex;
    static std::set<uint64_t> seen;
    {
        const std::lock_guard lock(mutex);
        if (!seen.insert(attributes).second) return;
    }
    rt::startup_trace::event("xmem_physical_pages",
                             "\"api\":\"XMemAllocatePhysicalPages\",\"attributes\":" +
                                 rt::startup_trace::quote(rt::startup_trace::hex(attributes)) +
                                 ",\"pages\":" + std::to_string(count) + caller_field(caller));
}

// BOOL XMemAllocatePhysicalPages(attributes, UINT64* pageCount, UINT64* pageArray). Measured at four call sites: RCX is
// a constant attribute word, RDX points at the number of 64 KiB pages (every caller computes it as the byte size
// rounded up and shifted right by 16), R8 at an array that receives one page number per page. The result is tested as
// a 32-bit BOOL. After success every caller compares the count with the one it asked for and frees the pages if they
// differ, so the count is in/out.
//
// Each call is backed by one pagefile-backed section of count * 64 KiB: committed (charged) at once, given physical
// memory as it is touched, zero-filled. Pages keep their contents while unmapped or mapped elsewhere, and one page can
// be mapped at several addresses. ASSUMPTION: the attribute word (memory type, perhaps the page size) is not decoded;
// every page is 64 KiB, which all four callers' arithmetic agrees with.
BOOL WINAPI XMemAllocatePhysicalPages(uint64_t attributes, uint64_t* page_count, uint64_t* page_array) {
    note_physical_attributes(attributes, page_count ? *page_count : 0, _ReturnAddress());
    if (!page_count || (*page_count && !page_array)) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return FALSE;
    }
    const uint64_t count = *page_count;
    if (count == 0) return TRUE;

    const std::lock_guard lock(g_phys_mutex);
    if (count > g_free_slots.size() + (kPageSlotLimit - g_next_slot)) {
        *page_count = 0;
        SetLastError(ERROR_NOT_ENOUGH_MEMORY);
        return FALSE;
    }
    const uint64_t bytes = count * kPhysicalPageSize;  // below 2^37: count is below 2^21
    const HANDLE handle = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE | SEC_COMMIT,
                                             static_cast<DWORD>(bytes >> 32), static_cast<DWORD>(bytes), nullptr);
    if (!handle) {
        *page_count = 0;
        return FALSE;
    }
    const auto section = std::make_shared<Section>(handle);
    for (uint64_t i = 0; i < count; ++i) {
        uint64_t slot;
        if (!g_free_slots.empty()) {
            slot = g_free_slots.back();
            g_free_slots.pop_back();
        } else {
            slot = g_next_slot++;
        }
        const uint64_t page = slot * kFramesPerPage;
        g_pages[page] = PhysicalPage{section, i * kPhysicalPageSize};
        page_array[i] = page;
    }
    return TRUE;
}

// BOOL XMemFreePhysicalPages(UINT64 pageCount, UINT64* pageArray). Measured at three call sites: RCX is the count (the
// one XMemAllocatePhysicalPages returned, or the sum of several allocations), RDX the array. The callers unmap the
// pages first; a page that is still mapped keeps its memory until it is unmapped. FALSE, with the entries before it
// freed, for a number that is not a live page.
BOOL WINAPI XMemFreePhysicalPages(uint64_t page_count, const uint64_t* page_array) {
    if (page_count && !page_array) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return FALSE;
    }
    const std::lock_guard lock(g_phys_mutex);
    for (uint64_t i = 0; i < page_count; ++i) {
        const auto it = g_pages.find(page_array[i]);
        if (it == g_pages.end()) {
            SetLastError(ERROR_INVALID_PARAMETER);
            return FALSE;
        }
        g_pages.erase(it);
        g_free_slots.push_back(page_array[i] / kFramesPerPage);
    }
    return TRUE;
}

// PVOID XMemMapPhysicalPages(PVOID address, UINT64 pageCount, UINT64* pageArray). Measured at twelve call sites: RCX
// is an address in a window XMemVirtualAlloc reserved, RDX a 64-bit count, R8 the pages to map there one after another,
// or null to unmap that many (four calls at shutdown). Callers test the 64-bit result for zero, so it is the address on
// success. A page number of 0 in the array unmaps that one page.
//
// Each page is its own view (MapViewOfFile3 into a 64 KiB placeholder), so any page can be remapped or unmapped later
// without touching its neighbours. A failure part-way leaves the pages before it mapped.
void* WINAPI XMemMapPhysicalPages(void* address, uint64_t page_count, const uint64_t* page_array) {
    const uintptr_t start = reinterpret_cast<uintptr_t>(address);
    if (!aligned_page_address(start) || page_count > (std::numeric_limits<uintptr_t>::max)() / kPhysicalPageSize ||
        start > (std::numeric_limits<uintptr_t>::max)() - page_count * kPhysicalPageSize) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return nullptr;
    }
    if (!placeholder_api().available()) {
        SetLastError(ERROR_NOT_SUPPORTED);
        return nullptr;
    }
    const std::lock_guard lock(g_phys_mutex);
    if (!page_array) {
        for (uint64_t i = 0; i < page_count; ++i) {
            if (!unmap_page(start + i * kPhysicalPageSize)) return nullptr;
        }
        return address;
    }
    const Window* const window = find_window(start, page_count * kPhysicalPageSize);
    if (!window) {
        SetLastError(ERROR_INVALID_ADDRESS);
        return nullptr;
    }
    const DWORD protect = view_protect(*window);
    for (uint64_t i = 0; i < page_count; ++i) {
        if (!map_page(start + i * kPhysicalPageSize, page_array[i], protect)) return nullptr;
    }
    return address;
}

// BOOL XMemMapPhysicalPagesScatter(PVOID* addresses, UINT32 pageCount, UINT64* pageArray). Measured at two call sites:
// RCX is an array of addresses, EDX a 32-bit count, R8 the page for each address (an entry of 0, or a null array,
// unmaps). The result is tested as a 32-bit BOOL. Like MapUserPhysicalPagesScatter: entry i maps pageArray[i] at
// addresses[i].
BOOL WINAPI XMemMapPhysicalPagesScatter(void* const* addresses, uint32_t page_count, const uint64_t* page_array) {
    if (page_count && !addresses) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return FALSE;
    }
    if (!placeholder_api().available()) {
        SetLastError(ERROR_NOT_SUPPORTED);
        return FALSE;
    }
    const std::lock_guard lock(g_phys_mutex);
    for (uint32_t i = 0; i < page_count; ++i) {
        const uintptr_t address = reinterpret_cast<uintptr_t>(addresses[i]);
        if (!aligned_page_address(address)) {
            SetLastError(ERROR_INVALID_PARAMETER);
            return FALSE;
        }
        const uint64_t page = page_array ? page_array[i] : 0;
        if (page == 0) {
            if (!unmap_page(address)) return FALSE;
            continue;
        }
        const Window* const window = find_window(address, kPhysicalPageSize);
        if (!window) {
            SetLastError(ERROR_INVALID_ADDRESS);
            return FALSE;
        }
        if (!map_page(address, page, view_protect(*window))) return FALSE;
    }
    return TRUE;
}

UINT WINAPI XMemGetWriteWatch(DWORD flags, void* base_address, SIZE_T region_size, void** addresses, ULONG_PTR* count,
                              ULONG* granularity) {
    return GetWriteWatch(flags, base_address, region_size, addresses, count, granularity);
}

UINT WINAPI XMemResetWriteWatch(void* base_address, SIZE_T region_size) {
    return ResetWriteWatch(base_address, region_size);
}

// BOOL XMemTransferMemory(XMEM_TRANSFER_TYPE type, SIZE_T size). Its reference page and the XMEM_TRANSFER_TYPE enum
// page (gaming/gdk/docs/reference/system/xmem/{functions/xmemtransfermemory,enums/xmem_transfer_type}) are NDA
// topics on Microsoft Learn, so the shape comes from public code built against the GDK's xmem.h: Microsoft's
// XBOX-MonoGame-NetRumble sample (console/NetRumble.Bootstrap/Main.cpp) and FNA-XNA/NativeAOT-Xbox (Bootstrap/Main.cpp)
// both call `if (!XMemTransferMemory(XMemTransferTitleToSystem, additionalKernelPoolMemory))` with a `const size_t`,
// commented "Transfer additional memory to kernel pool" and "Must be multiples of 2MB sizes". So: two arguments, the
// enum (32 bits) in ECX and the size in RDX, and a zero return is failure. The public XLaunchNewGame page adds that
// "Memory transferred to the system using XMemTransferMemory (NDA topic) is persisted", and the game-save debugging
// page suggests it for E_OUTOFMEMORY "to transfer some memory to the system partition".
//
// On the console it moves budget between the title's partition and the system's. A PC has one pool and no budget
// split, so there is nothing to move: every direction and size returns TRUE and nothing is allocated or reserved. Not
// known (the pages are NDA): the numeric XMEM_TRANSFER_TYPE values, and whether the console refuses or rounds a size
// that is not a multiple of 2 MiB. Refusing would stop a title at startup (both samples return -1), while succeeding
// changes nothing here. TRUE is 1 in EAX, so it also reads as true if the real return type is a one-byte bool.
//
// gta6rt --trace FILE: the first call for each distinct (type, size) pair goes to the startup trace as an
// "xmem_transfer" event with the caller as module+RVA, so a run with the executable records the values it passes.
BOOL WINAPI XMemTransferMemory(uint32_t type, SIZE_T size) {
    static std::atomic<bool> warned{false};
    if (!warned.exchange(true, std::memory_order_relaxed)) {
        std::fprintf(stderr,
                     "[rt] xmem: XMemTransferMemory(type %u, %zu bytes): a PC has no title/system memory partition; "
                     "returning TRUE without moving anything (logged once)\n",
                     type, static_cast<size_t>(size));
    }
    if (rt::startup_trace::enabled()) {
        static std::mutex mutex;
        static std::set<std::pair<uint32_t, uint64_t>> seen;
        bool first = false;
        {
            const std::lock_guard lock(mutex);
            first = seen.insert({type, static_cast<uint64_t>(size)}).second;
        }
        if (first) {
            rt::startup_trace::event(
                "xmem_transfer",
                "\"api\":\"XMemTransferMemory\",\"type\":" + std::to_string(type) +
                    ",\"size\":" + std::to_string(static_cast<uint64_t>(size)) + ",\"caller\":" +
                    rt::startup_trace::quote(
                        rt::startup_trace::where(reinterpret_cast<uint64_t>(_ReturnAddress()))));
        }
    }
    return TRUE;
}

// kernel32!VirtualAlloc as a title calls it. The console's allocation type has two bits of its own, MEM_GRAPHICS
// (0x10000000, memory the GPU reads) and MEM_TITLE (0x40000000, the title's own heap). On a PC the host rejects both
// with ERROR_INVALID_PARAMETER (measured on Windows 11 26200: VirtualAlloc(NULL, 64 KiB, MEM_COMMIT | MEM_RESERVE |
// 0x10000000, PAGE_READWRITE) returns NULL; so does 0x40000000), so an engine allocating its heaps this way gets
// NULL at boot. Both are hints about where the memory lives; the host has no such split, so they are dropped and
// the rest is passed through. The PAGE_GRAPHICS_* protection bits are dropped (host_protect). Not done: whether
// graphics memory has to come from an arena the D3D12 layer can see (#313).
constexpr DWORD kMemGraphics = 0x10000000;
constexpr DWORD kMemTitle = 0x40000000;

LPVOID WINAPI TitleVirtualAlloc(LPVOID address, SIZE_T size, DWORD allocation_type, DWORD protect) {
    return VirtualAlloc(address, size, allocation_type & ~(kMemGraphics | kMemTitle), host_protect(protect));
}

}  // namespace

#define TITLE_VIRTUAL_ALLOC(dll) {dll, "VirtualAlloc", reinterpret_cast<void*>(&TitleVirtualAlloc)}

// The classic libraries and the memory api sets a title's imports name.
SHIM_TABLE(title_memory,
    TITLE_VIRTUAL_ALLOC("kernel32.dll"),
    TITLE_VIRTUAL_ALLOC("kernelbase.dll"),
    TITLE_VIRTUAL_ALLOC("api-ms-win-core-memory-l1-1-0.dll"),
    TITLE_VIRTUAL_ALLOC("api-ms-win-core-memory-l1-1-1.dll"),
    TITLE_VIRTUAL_ALLOC("api-ms-win-core-memory-l1-1-2.dll"),
    TITLE_VIRTUAL_ALLOC("api-ms-win-core-memory-l1-1-3.dll"),
    TITLE_VIRTUAL_ALLOC("api-ms-win-core-memory-l1-1-4.dll"),
    TITLE_VIRTUAL_ALLOC("api-ms-win-core-memory-l1-1-5.dll"),
    TITLE_VIRTUAL_ALLOC("api-ms-win-core-memory-l1-1-6.dll"),
);

SHIM_TABLE(xmem,
    {"xmem.dll", "XMemAlloc", reinterpret_cast<void*>(&XMemAlloc)},
    {"xmem.dll", "XMemFree", reinterpret_cast<void*>(&XMemFree)},
    {"xmem.dll", "XMemVirtualQuery", reinterpret_cast<void*>(&XMemVirtualQuery)},
    {"xmem.dll", "XMemGetWorkingSetStatistics", reinterpret_cast<void*>(&XMemGetWorkingSetStatistics)},
    {"xmem.dll", "XMemVirtualAlloc", reinterpret_cast<void*>(&XMemVirtualAlloc)},
    {"xmem.dll", "XMemGetWriteWatch", reinterpret_cast<void*>(&XMemGetWriteWatch)},
    {"xmem.dll", "XMemResetWriteWatch", reinterpret_cast<void*>(&XMemResetWriteWatch)},
    {"xmem.dll", "XMemTransferMemory", reinterpret_cast<void*>(&XMemTransferMemory)},
    {"xmem.dll", "XMemAllocDefault", reinterpret_cast<void*>(&XMemAllocDefault)},
    {"xmem.dll", "XMemFreeDefault", reinterpret_cast<void*>(&XMemFreeDefault)},
    {"xmem.dll", "XMemIsToolsMemAvailable", reinterpret_cast<void*>(&XMemIsToolsMemAvailable)},
    {"xmem.dll", "XMemAllocatePhysicalPages", reinterpret_cast<void*>(&XMemAllocatePhysicalPages)},
    {"xmem.dll", "XMemFreePhysicalPages", reinterpret_cast<void*>(&XMemFreePhysicalPages)},
    {"xmem.dll", "XMemMapPhysicalPages", reinterpret_cast<void*>(&XMemMapPhysicalPages)},
    {"xmem.dll", "XMemMapPhysicalPagesScatter", reinterpret_cast<void*>(&XMemMapPhysicalPagesScatter)},
);
