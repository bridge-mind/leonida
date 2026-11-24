#include "pe_loader.h"

#include <windows.h>
#include <intrin.h>
#include <tlhelp32.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <system_error>
#include <vector>

#include "../shims/shims.h"
#include "../shims/startup_trace.h"
#include "base_reservation.h"

namespace rt {

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    return s;
}

namespace {

std::filesystem::path::string_type module_key(const std::filesystem::path& path) {
    // Keep identity in the platform's native path representation. Only the existing ASCII case folding is
    // applied here; representable names also retain their old narrow-byte alias below.
    auto key = path.filename().native();
    for (auto& c : key)
        if (c >= 'A' && c <= 'Z') c = static_cast<std::filesystem::path::value_type>(c + ('a' - 'A'));
    return key;
}

std::string narrow_module_key(const std::filesystem::path& path) {
    const auto filename = path.filename();
    try {
        const auto narrow = filename.string();
        // A lossy/default-character alias must not collapse two distinct Unicode basenames.
        if (std::filesystem::path(narrow).native() == filename.native()) return lower(narrow);
    } catch (const std::system_error&) {
        // Some Windows standard libraries throw when a basename is outside the current narrow code page.
        // It still has a native identity; it simply has no alias in the existing narrow lookup API.
    }
    return {};
}

// Windows matches module names case-insensitively with its own upcase table (RtlEqualUnicodeString); the
// ordinal ignore-case compare is that table.
bool same_module_name(const wchar_t* a, const wchar_t* b) {
    return CompareStringOrdinal(a, -1, b, -1, TRUE) == CSTR_EQUAL;
}

// A narrow (ANSI code page) module name widened for a case-insensitive compare; empty when it is not valid in the
// code page.
std::wstring acp_to_wide(const std::string& narrow) {
    if (narrow.empty()) return {};
    const int n = MultiByteToWideChar(CP_ACP, MB_ERR_INVALID_CHARS, narrow.data(), static_cast<int>(narrow.size()),
                                      nullptr, 0);
    if (n <= 0) return {};
    std::wstring wide(static_cast<size_t>(n), L'\0');
    MultiByteToWideChar(CP_ACP, MB_ERR_INVALID_CHARS, narrow.data(), static_cast<int>(narrow.size()), wide.data(), n);
    return wide;
}

std::string path_utf8(const std::filesystem::path& path) {
    const auto bytes = path.u8string();
    return {reinterpret_cast<const char*>(bytes.data()), bytes.size()};
}

std::vector<uint8_t> read_file(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path_utf8(p));
    return {std::istreambuf_iterator<char>(f), {}};
}

template <class T>
T* at(uint8_t* base, uint32_t rva) { return reinterpret_cast<T*>(base + rva); }

// Every RVA followed below comes out of the image itself, so each one is checked first. A truncated or corrupt
// image then fails with a message instead of reading, writing or protecting memory outside its allocation.
void require_range(const std::string& name, uint64_t rva, uint64_t len, uint32_t image_size, const char* what) {
    if (rva > image_size || len > image_size - rva)
        throw std::runtime_error(name + ": " + what + " runs past the image");
}

std::string read_string(const std::string& name, uint8_t* base, uint32_t image_size, uint32_t rva, const char* what) {
    if (rva >= image_size) throw std::runtime_error(name + ": " + what + " is outside the image");
    auto* s = reinterpret_cast<const char*>(base + rva);
    auto* end = static_cast<const char*>(std::memchr(s, 0, image_size - rva));
    if (!end) throw std::runtime_error(name + ": " + what + " isn't terminated inside the image");
    return {s, end};
}

// PE/COFF: NumberOfRvaAndSizes counts the data directories; entries past it are not directories, whatever they hold.
IMAGE_DATA_DIRECTORY directory(const IMAGE_NT_HEADERS64* nt, unsigned index) {
    return index < nt->OptionalHeader.NumberOfRvaAndSizes ? nt->OptionalHeader.DataDirectory[index] : IMAGE_DATA_DIRECTORY{};
}

// The TLS directory holds virtual addresses, which relocations have already moved to this mapping.
uint64_t rva_in(const Module& m, uint64_t va, const char* what) {
    const uint64_t base = reinterpret_cast<uint64_t>(m.base);
    if (va < base || va - base >= m.size) throw std::runtime_error(m.name + ": " + what + " is outside the image");
    return va - base;
}

IMAGE_TLS_DIRECTORY64* tls_directory(Module& m) {
    auto* nt = at<IMAGE_NT_HEADERS64>(m.base, reinterpret_cast<IMAGE_DOS_HEADER*>(m.base)->e_lfanew);
    // Through directory(), as every other directory: an entry past NumberOfRvaAndSizes is not a TLS directory.
    const auto dir = directory(nt, IMAGE_DIRECTORY_ENTRY_TLS);
    if (!dir.Size || !dir.VirtualAddress) return nullptr;
    require_range(m.name, dir.VirtualAddress, sizeof(IMAGE_TLS_DIRECTORY64), m.size, "the TLS directory");
    return at<IMAGE_TLS_DIRECTORY64>(m.base, dir.VirtualAddress);
}

// __declspec(thread) reads TEB->ThreadLocalStoragePointer[_tls_index] at gs:[58h]. That array is indexed by the
// loader-assigned per-module TLS index and is a different index space from TlsAlloc/TlsGetValue (TlsSlots). Only the OS
// loader can allocate a module index, so the runtime ships a minimal DLL whose only content is a TLS anchor
// (src/tls_host.cpp). Loading it makes the OS reserve an index and give every thread a block for it; the native loader
// reuses that index for the mapped image. When more than one manually mapped module uses static TLS at once, the loader
// clones gta6rt_tls_host.dll to a per-process temporary path on demand and loads it, so the OS assigns an independent
// module TLS index for each simultaneous module (up to kMaxMappedTlsModules).
constexpr DWORD kMaxModuleTlsSlots = 64;
constexpr size_t kMaxMappedTlsModules = 16;
void** module_tls_slots() { return reinterpret_cast<void**>(__readgsqword(0x58)); }

void __cdecl on_host_thread_event(DWORD reason);

struct TlsHostModule {
    HMODULE module = nullptr;
    DWORD index = TLS_OUT_OF_INDEXES;
};

std::filesystem::path tls_host_base_path() {
    std::wstring exe_path;
    DWORD size = MAX_PATH;
    while (true) {
        exe_path.resize(size);
        DWORD len = GetModuleFileNameW(nullptr, exe_path.data(), size);
        if (len == 0) return {};
        if (len < size - 1) {
            exe_path.resize(len);
            break;
        }
        size *= 2;
    }
    return std::filesystem::path(exe_path).parent_path() / L"gta6rt_tls_host.dll";
}

std::filesystem::path tls_host_clone_dir() {
    wchar_t buf[MAX_PATH];
    DWORD n = GetTempPathW(MAX_PATH, buf);
    if (n > 0 && n < MAX_PATH) return std::filesystem::path(buf);
    return std::filesystem::temp_directory_path();
}

struct TlsCloneRecord {
    HMODULE module = nullptr;
    std::filesystem::path path;
};

std::mutex g_tls_alloc_mutex;
std::array<TlsHostModule, kMaxMappedTlsModules> g_tls_hosts{};
std::array<std::atomic<Module*>, kMaxMappedTlsModules> g_tls_slot_owners{};
std::array<std::atomic<uint64_t>, kMaxMappedTlsModules> g_tls_slot_generations{};
std::atomic<uint64_t> g_next_tls_generation{1};
std::vector<TlsCloneRecord> g_tls_clones;

// Frees every TLS helper clone this process loaded and deletes its files. load() registers the wrapper with atexit;
// the --run path in main() must also call this directly before ExitProcess, which skips the atexit table. Call it
// only when no mapped code can run again: it unmaps the helper images.
void cleanup_tls_clones_internal() {
    for (const auto& rec : g_tls_clones) {
        if (rec.module) FreeLibrary(rec.module);
        if (!rec.path.empty()) DeleteFileW(rec.path.c_str());
    }
    g_tls_clones.clear();
}

void cleanup_tls_clones_at_exit() { cleanup_tls_clones_internal(); }

// The owner process named by "leonida_tls_host_<pid>_<slot>.dll"; 0 when the name does not carry one.
DWORD clone_owner_pid(const wchar_t* name) {
    constexpr wchar_t prefix[] = L"leonida_tls_host_";
    const wchar_t* p = name;
    for (size_t i = 0; i < sizeof(prefix) / sizeof(wchar_t) - 1; ++i)
        if (*p++ != prefix[i]) return 0;
    if (*p < L'0' || *p > L'9') return 0;
    unsigned long pid = 0;
    while (*p >= L'0' && *p <= L'9') {
        pid = pid * 10 + (*p - L'0');
        if (pid > 0xFFFFFFFFul) return 0;
        ++p;
    }
    return static_cast<DWORD>(pid);
}

bool process_alive(DWORD pid) {
    // PROCESS_QUERY_LIMITED_INFORMATION needs no handle rights a normal process could deny; a failure means the
    // process is gone. A still-mapped or still-open file is caught by the checks below even for a misjudged PID.
    HANDLE p = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
    if (!p) return false;
    CloseHandle(p);
    return true;
}

// Deletes clones left behind by processes that have exited. Deletion is scoped to files whose owner is gone: a
// clone of a live process is either still open by its copy step's handle or mapped as an image, so it is skipped
// (or DeleteFileW fails harmlessly), and a concurrent process between CopyFileW and LoadLibraryExW keeps its file.
void sweep_stale_tls_clones(const std::filesystem::path& dir) {
    WIN32_FIND_DATAW fd{};
    const std::filesystem::path pattern = dir / L"leonida_tls_host_*.dll";
    HANDLE h = FindFirstFileW(pattern.c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) return;
    do {
        if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;
        const std::filesystem::path path = dir / fd.cFileName;
        const DWORD owner = clone_owner_pid(fd.cFileName);
        if (owner && owner != GetCurrentProcessId() && process_alive(owner)) continue;
        // An exclusive open (no sharing) conflicts with any handle another process holds on the file, so a failure
        // means the copy->load sequence or a mapped image still owns it. Only unowned files reach DeleteFileW, and
        // that stays harmless for a mapped image.
        HANDLE probe =
            CreateFileW(path.c_str(), GENERIC_READ, 0, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (probe == INVALID_HANDLE_VALUE) continue;
        CloseHandle(probe);
        DeleteFileW(path.c_str());
    } while (FindNextFileW(h, &fd));
    FindClose(h);
}

TlsHostModule load_tls_host_image(const std::filesystem::path& path, bool hook_thread_events) {
    TlsHostModule host;
    if (path.empty()) return host;
    // The helper is linked with the static CRT (/MT), so it needs only system DLLs. Restricting the dependency search
    // to System32 keeps a game-supplied vcruntime next to the executable from being picked up.
    host.module = LoadLibraryExW(path.c_str(), nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!host.module) return host;

    auto* base = reinterpret_cast<uint8_t*>(host.module);
    auto* dos = reinterpret_cast<IMAGE_DOS_HEADER*>(base);
    auto* nt = at<IMAGE_NT_HEADERS64>(base, dos->e_lfanew);
    const auto& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_TLS];
    if (!dir.Size || !dir.VirtualAddress) {
        FreeLibrary(host.module);
        host.module = nullptr;
        return host;
    }
    auto* tls = at<IMAGE_TLS_DIRECTORY64>(base, dir.VirtualAddress);
    if (!tls->AddressOfIndex) {
        FreeLibrary(host.module);
        host.module = nullptr;
        return host;
    }
    host.index = *reinterpret_cast<DWORD*>(tls->AddressOfIndex);
    if (hook_thread_events) {
        // Only the primary helper reports thread attach/detach events; on_host_thread_event iterates all active slots.
        using SetThreadEvent = void(__cdecl*)(void(__cdecl*)(DWORD));
        if (auto set = reinterpret_cast<SetThreadEvent>(GetProcAddress(host.module, "gta6rt_tls_host_set_thread_event")))
            set(&on_host_thread_event);
    }
    return host;
}

// Must be called with g_tls_alloc_mutex held and WITHOUT g_tls_lock held (LoadLibraryExW takes the Windows loader lock).
TlsHostModule ensure_tls_host(size_t slot) {
    if (slot >= kMaxMappedTlsModules) return {};
    if (g_tls_hosts[slot].index != TLS_OUT_OF_INDEXES) return g_tls_hosts[slot];

    const std::filesystem::path primary = tls_host_base_path();
    if (primary.empty()) return {};

    if (slot == 0) {
        g_tls_hosts[0] = load_tls_host_image(primary, true);
        return g_tls_hosts[0];
    }

    // Ensure slot 0 is loaded first so thread attach/detach events are hooked.
    if (g_tls_hosts[0].index == TLS_OUT_OF_INDEXES) {
        g_tls_hosts[0] = load_tls_host_image(primary, true);
        if (g_tls_hosts[0].index == TLS_OUT_OF_INDEXES) return {};
    }

    // Clone gta6rt_tls_host.dll to a distinct path using CopyFileW (Wine 9 lacks CopyFile2 used by std::filesystem).
    const std::filesystem::path dir = tls_host_clone_dir();
    static bool swept = false;
    if (!swept) {
        swept = true;
        sweep_stale_tls_clones(dir);
        std::atexit(&cleanup_tls_clones_at_exit);
    }
    const std::wstring name =
        L"leonida_tls_host_" + std::to_wstring(GetCurrentProcessId()) + L"_" + std::to_wstring(slot) + L".dll";
    const std::filesystem::path clone_path = dir / name;
    // The staging name must not match the sweep pattern, which only covers final clone names ending in .dll.
    const std::filesystem::path staging = clone_path.wstring() + L".new";
    if (!CopyFileW(primary.c_str(), staging.c_str(), FALSE)) return {};

    // Open the guard before the clone exists under its final name and hold it across LoadLibraryExW: the rename
    // below needs DELETE access, which the handle's FILE_SHARE_DELETE grants, and no purge can remove the clone
    // while the handle is open (an exclusive open conflicts with it) or once the image is mapped. Copying straight
    // to the final name would leave a window where a concurrent sweep sees an unowned file.
    HANDLE guard = CreateFileW(staging.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE, nullptr,
                               OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (guard == INVALID_HANDLE_VALUE) {
        DeleteFileW(staging.c_str());
        return {};
    }
    if (!MoveFileExW(staging.c_str(), clone_path.c_str(), MOVEFILE_REPLACE_EXISTING)) {
        CloseHandle(guard);
        DeleteFileW(staging.c_str());
        return {};
    }
    TlsHostModule loaded = load_tls_host_image(clone_path, false);
    CloseHandle(guard);

    if (loaded.index == TLS_OUT_OF_INDEXES) {
        DeleteFileW(clone_path.c_str());
        return {};
    }
    for (size_t i = 0; i < kMaxMappedTlsModules; ++i) {
        if (i != slot && g_tls_hosts[i].index == loaded.index) {
            FreeLibrary(loaded.module);
            DeleteFileW(clone_path.c_str());
            return {};
        }
    }
    g_tls_clones.push_back({loaded.module, clone_path});
    g_tls_hosts[slot] = loaded;
    return g_tls_hosts[slot];
}

struct ThreadTlsSlot {
    Module* module = nullptr;
    uint64_t generation = 0;
    void* saved = nullptr;
    void* block = nullptr;
    uint32_t index = kNoTlsIndex;
};

// Per-thread installation state for each reserved TLS slot.
thread_local std::array<ThreadTlsSlot, kMaxMappedTlsModules> g_thread_tls_slots{};

// Orders the TLS owners' publication, their `initialised` flags and their retirement against the thread events. Shared
// in on_host_thread_event, exclusive only for those short writes (no allocation and no call that takes the Windows
// loader lock, which the events run under).
SRWLOCK g_tls_lock = SRWLOCK_INIT;
struct SharedTlsLock {
    SharedTlsLock() { AcquireSRWLockShared(&g_tls_lock); }
    ~SharedTlsLock() { ReleaseSRWLockShared(&g_tls_lock); }
    SharedTlsLock(const SharedTlsLock&) = delete;
    SharedTlsLock& operator=(const SharedTlsLock&) = delete;
};
struct ExclusiveTlsLock {
    ExclusiveTlsLock() { AcquireSRWLockExclusive(&g_tls_lock); }
    ~ExclusiveTlsLock() { ReleaseSRWLockExclusive(&g_tls_lock); }
    ExclusiveTlsLock(const ExclusiveTlsLock&) = delete;
    ExclusiveTlsLock& operator=(const ExclusiveTlsLock&) = delete;
};

// Puts the helper DLL's own block back in this thread's slot and frees the copy that replaced it. Uses only this
// thread's record, so it also works after the module that installed the copy is gone.
void release_thread_slot(size_t slot) {
    if (slot >= kMaxMappedTlsModules) return;
    ThreadTlsSlot& entry = g_thread_tls_slots[slot];
    if (!entry.module && !entry.block) return;
    if (entry.block) {
        void** slots = module_tls_slots();
        if (slots && entry.index < kMaxModuleTlsSlots && slots[entry.index] == entry.block)
            slots[entry.index] = entry.saved;
        VirtualFree(entry.block, 0, MEM_RELEASE);
    }
    entry = ThreadTlsSlot{};
}

// A block load() put into a thread that was already running (install_on_existing_threads). Such a thread has no
// g_thread_tls_slots record of it: the record is here, keyed by thread id, and the thread's own DLL_THREAD_DETACH or
// the module's ~Loader takes the block out again.
struct ExistingThreadBlock {
    DWORD thread = 0;
    size_t slot = 0;
    Module* module = nullptr;
    uint64_t generation = 0;   // the module's tls_generation: with the pointer, what identifies the module
    uint32_t index = kNoTlsIndex;
    void*** vector = nullptr;  // that thread's TEB->ThreadLocalStoragePointer field; the array itself can be replaced
    void* saved = nullptr;     // what the block replaced
    void* block = nullptr;
};

// Immortal, like the thread events that use them. `ended` holds the ids of threads whose DLL_THREAD_DETACH has run:
// Windows frees such a thread's TLS array next, so nothing may be put into it any more.
struct ExistingThreads {
    SRWLOCK lock = SRWLOCK_INIT;  // a leaf: nothing is called while it is held but VirtualFree and thread queries
    std::vector<ExistingThreadBlock> blocks;
    std::vector<DWORD> ended;
};
ExistingThreads& existing_threads() {
    static auto* state = new ExistingThreads;
    return *state;
}
struct ExistingThreadsLock {
    ExistingThreadsLock() { AcquireSRWLockExclusive(&existing_threads().lock); }
    ~ExistingThreadsLock() { ReleaseSRWLockExclusive(&existing_threads().lock); }
    ExistingThreadsLock(const ExistingThreadsLock&) = delete;
    ExistingThreadsLock& operator=(const ExistingThreadsLock&) = delete;
};

bool has_existing_thread_block(DWORD thread, const Module& m) {
    ExistingThreadsLock lock;
    for (const auto& b : existing_threads().blocks)
        if (b.thread == thread && b.module == &m && b.generation == m.tls_generation) return true;
    return false;
}

void install_thread_block(Module& m) {
    if (m.tls_slot >= kMaxMappedTlsModules) return;
    const size_t slot = m.tls_slot;
    ThreadTlsSlot& entry = g_thread_tls_slots[slot];
    if (entry.module == &m && entry.generation == m.tls_generation) return;
    if (has_existing_thread_block(GetCurrentThreadId(), m)) return;  // load() already gave this thread its copy
    if (entry.module) {
        // A copy left by a module whose Loader is gone (or from an earlier generation) is released; a live owner's is a conflict.
        if (entry.module == g_tls_slot_owners[slot].load() &&
            entry.generation == g_tls_slot_generations[slot].load()) {
            throw std::runtime_error("another manually mapped module's TLS is already installed in this thread slot");
        }
        release_thread_slot(slot);
    }

    const uint64_t total = static_cast<uint64_t>(m.tls_raw_size) + m.tls_zero_fill;
    if (!total) {
        // Callback-only TLS module (no data block): still record attachment in this slot so DLL_THREAD_DETACH runs.
        entry.module = &m;
        entry.generation = m.tls_generation;
        return;
    }
    if (m.tls_index == kNoTlsIndex) return;
    void** slots = module_tls_slots();
    if (!slots || m.tls_index >= kMaxModuleTlsSlots)
        throw std::runtime_error(m.name + ": cannot install TLS in this thread");

    void* block = VirtualAlloc(nullptr, static_cast<SIZE_T>(total), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!block) throw std::runtime_error(m.name + ": TLS block allocation failed");
    if (m.tls_raw_size) std::memcpy(block, m.tls_template, m.tls_raw_size);
    if (m.tls_zero_fill)
        std::memset(static_cast<uint8_t*>(block) + m.tls_raw_size, 0, m.tls_zero_fill);

    void* saved = slots[m.tls_index];
    slots[m.tls_index] = block;
    entry.module = &m;
    entry.generation = m.tls_generation;
    entry.saved = saved;
    entry.block = block;
    entry.index = m.tls_index;
}

void uninstall_thread_block(Module& m) {
    if (m.tls_slot >= kMaxMappedTlsModules) return;
    const size_t slot = m.tls_slot;
    ThreadTlsSlot& entry = g_thread_tls_slots[slot];
    if (entry.module != &m || entry.generation != m.tls_generation) return;
    if (m.tls_block == entry.block) {
        m.tls_block = nullptr;
        m.tls_saved_block = nullptr;
    }
    release_thread_slot(slot);
}

// `trace`: record each callback in the startup trace. Only the Loader's own calls (process attach and detach) pass
// true; the thread events run under the Windows loader lock and stay out of the trace.
void run_callbacks(Module& m, uint32_t reason, bool trace = false) {
    auto* tls = tls_directory(m);
    if (!tls || !tls->AddressOfCallBacks) return;
    const uint64_t table_rva = rva_in(m, tls->AddressOfCallBacks, "the TLS callback table");
    // The callback table has no length field: it ends at a zero entry, so every step is checked against the image.
    for (uint64_t i = 0;; ++i) {
        const uint64_t entry_rva = table_rva + i * sizeof(uint64_t);
        require_range(m.name, entry_rva, sizeof(uint64_t), m.size, "a TLS callback table entry");
        const uint64_t fn = *at<uint64_t>(m.base, static_cast<uint32_t>(entry_rva));
        if (!fn) break;
        auto callback =
            reinterpret_cast<PIMAGE_TLS_CALLBACK>(m.base + rva_in(m, fn, "a TLS callback"));
        if (trace && startup_trace::enabled())
            startup_trace::event("tls_callback", "\"module\":" + startup_trace::quote(m.name) + ",\"at\":" +
                                                     startup_trace::quote(startup_trace::where(reinterpret_cast<uint64_t>(callback))) +
                                                     ",\"reason\":" + std::to_string(reason));
        callback(m.base, static_cast<DWORD>(reason), nullptr);
    }
}

// ---- Threads that existed before load() ---------------------------------------------------------------------------
//
// When Windows loads a DLL with static TLS itself, every running thread gets a block for it (and no DLL_THREAD_ATTACH).
// The helper DLL gets exactly that, so in a thread that was already running the reserved slot holds the helper's
// one-byte block, and mapped __declspec(thread) code there would read and write past it. Such threads are not rare:
// the workers of the process thread pool predate most loads, and the task queue runs a title's callbacks on them.
// load() therefore gives each of them a copy of the template too, through the thread's TEB, which is ordinary memory
// of this process.
//
// What makes that safe:
//  - The slot is the helper's, and nothing reads the helper's variable, so swapping the pointer under a running
//    thread disturbs nothing; the thread cannot be in mapped code of a module whose load() has not returned.
//  - The swap is made with g_tls_lock held exclusively, so no thread is inside its attach or detach event meanwhile.
//  - A thread whose detach event has run is skipped (`ended`), and so is one with no TLS array or no helper block:
//    Windows is about to free, or never built, what the block would go into.
//  - The block comes out again in that thread's own detach event, before Windows frees the array, or in ~Loader.
// What is left:
//  - A thread that ran its detach event before the helper DLL was ever loaded and has not left the process yet is
//    not in `ended`; if Windows leaves a stale array pointer in its TEB for those last instructions, the write goes
//    to freed heap memory.
//  - Windows replaces a thread's TLS array (a new one, copied from the old) when some thread loads a DLL with static
//    TLS. If that happens during the swap, the write can land in the old array; the thread then still has the
//    helper's one-byte block, and the first mapped __declspec(thread) access on it writes past that block: heap
//    corruption, with nothing to show for it. The swap therefore reads the array pointer again after writing and
//    repeats on the new array (kSwapAttempts times); if the block is not in the thread's current array at the end,
//    it is taken back and the miss is logged. A replacement after that last look is not seen.
//  - TerminateThread frees a thread's TEB with no detach event. Every read and write of another thread's TEB or
//    TLS array here is made right after the thread was seen running and inside __try, so a TEB that has just gone
//    is a skipped thread, not a fault in load() or ~Loader. If the memory was already given to something else, the
//    write is not caught.
using QueryThreadFn = LONG(NTAPI*)(HANDLE, int, void*, ULONG, ULONG*);
struct ThreadBasicInfo {  // THREAD_BASIC_INFORMATION (winternl.h keeps it opaque), x64
    LONG exit_status;
    void* teb;
    void* process_id;
    void* thread_id;
    ULONG_PTR affinity;
    LONG priority;
    LONG base_priority;
};
static_assert(sizeof(ThreadBasicInfo) == 48);

QueryThreadFn query_thread_fn() {
    static const auto fn = reinterpret_cast<QueryThreadFn>(
        reinterpret_cast<void*>(GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "NtQueryInformationThread")));
    return fn;
}

// The TEB's ThreadLocalStoragePointer field of a thread that is still running; null otherwise.
void*** running_thread_tls_vector(HANDLE thread) {
    ThreadBasicInfo info{};
    const QueryThreadFn query = query_thread_fn();
    if (!query || query(thread, 0 /* ThreadBasicInformation */, &info, sizeof info, nullptr) < 0) return nullptr;
    if (info.exit_status != STILL_ACTIVE || !info.teb) return nullptr;
    return reinterpret_cast<void***>(static_cast<uint8_t*>(info.teb) + 0x58);
}

// Another thread's TLS array and one entry of it, read through its TEB. False when the memory is not there any more
// (the thread was terminated). No C++ objects in these three: they are __try frames.
bool read_thread_slot(void*** vector, uint32_t index, void**& slots, void*& value) noexcept {
    __try {
        slots = *vector;
        value = slots ? slots[index] : nullptr;
        return true;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        slots = nullptr;
        value = nullptr;
        return false;
    }
}

bool write_thread_slot(void** slots, uint32_t index, void* value) noexcept {
    __try {
        slots[index] = value;
        return true;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
}

// Puts `saved` back where `block` is, if it is still there.
void restore_thread_slot(void*** vector, uint32_t index, void* block, void* saved) noexcept {
    __try {
        void** slots = *vector;
        if (slots && slots[index] == block) slots[index] = saved;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
    }
}

// Takes a block out of its slot and frees it. The existing-threads lock is held; b.vector is null when the thread
// is gone.
void remove_existing_block(const ExistingThreadBlock& b) {
    if (b.vector && b.index < kMaxModuleTlsSlots) restore_thread_slot(b.vector, b.index, b.block, b.saved);
    VirtualFree(b.block, 0, MEM_RELEASE);
}

constexpr int kSwapAttempts = 4;
constexpr size_t kMaxEndedThreads = 1024;

// load(): every other running thread gets its own copy of m's template. Not fatal where it cannot be done.
void install_on_existing_threads(Module& m) {
    const uint64_t total = static_cast<uint64_t>(m.tls_raw_size) + m.tls_zero_fill;
    if (!total || m.tls_index >= kMaxModuleTlsSlots || m.tls_slot >= kMaxMappedTlsModules) return;
    // Stays ahead of the TLS lock: the first call resolves the function with GetModuleHandle and GetProcAddress, which
    // take the Windows loader lock, and a thread inside its attach event holds that lock while it waits for ours.
    if (!query_thread_fn()) return;

    // Outside the TLS lock: none of this may wait for a thread that is inside its attach event.
    // The handles are closed on every way out, also when a vector below cannot grow.
    struct Candidate {
        DWORD id = 0;
        HANDLE handle = nullptr;
    };
    struct Candidates {
        std::vector<Candidate> list;
        HANDLE snapshot = INVALID_HANDLE_VALUE;
        ~Candidates() {
            if (snapshot != INVALID_HANDLE_VALUE) CloseHandle(snapshot);
            for (const Candidate& c : list)
                if (c.handle) CloseHandle(c.handle);
        }
    } found;
    std::vector<Candidate>& candidates = found.list;
    // TH32CS_SNAPTHREAD lists the threads of every process; the loop keeps this one's.
    found.snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (found.snapshot == INVALID_HANDLE_VALUE) return;
    THREADENTRY32 entry{};
    entry.dwSize = sizeof entry;
    const DWORD process = GetCurrentProcessId(), self = GetCurrentThreadId();
    for (BOOL more = Thread32First(found.snapshot, &entry); more; more = Thread32Next(found.snapshot, &entry)) {
        if (entry.th32OwnerProcessID != process || entry.th32ThreadID == self) continue;
        candidates.push_back({entry.th32ThreadID, nullptr});  // the slot first, so the handle always has an owner
        candidates.back().handle = OpenThread(THREAD_QUERY_INFORMATION, FALSE, entry.th32ThreadID);
    }

    {
        ExclusiveTlsLock tls_lock;
        ExistingThreadsLock lock;
        ExistingThreads& state = existing_threads();
        for (const Candidate& c : candidates) {
            if (!c.handle) continue;
            if (std::find(state.ended.begin(), state.ended.end(), c.id) != state.ended.end()) continue;
            void*** vector = running_thread_tls_vector(c.handle);
            void** slots = nullptr;
            void* saved = nullptr;
            if (!vector || !read_thread_slot(vector, m.tls_index, slots, saved) || !slots || !saved) continue;
            // A copy an earlier module left in this slot of this thread goes first; what it replaced is kept.
            bool present = false;
            for (auto it = state.blocks.begin(); it != state.blocks.end();) {
                if (it->thread != c.id || it->slot != m.tls_slot) {
                    ++it;
                } else if (it->module == &m && it->generation == m.tls_generation) {
                    present = true;
                    break;
                } else {
                    if (saved == it->block) saved = it->saved;
                    it->vector = vector;
                    remove_existing_block(*it);
                    it = state.blocks.erase(it);
                }
            }
            if (present) continue;
            void* block = VirtualAlloc(nullptr, static_cast<SIZE_T>(total), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
            if (!block) continue;
            if (m.tls_raw_size) std::memcpy(block, m.tls_template, m.tls_raw_size);

            // Write, then look at the thread's array pointer again: if Windows has replaced the array meanwhile, the
            // write went into the old one, and it is made again in the new one.
            bool installed = false;
            for (int attempt = 0; attempt < kSwapAttempts && !installed; ++attempt) {
                if (!running_thread_tls_vector(c.handle) || !write_thread_slot(slots, m.tls_index, block)) break;
                void** now = nullptr;
                void* value = nullptr;
                if (!read_thread_slot(vector, m.tls_index, now, value) || !now) break;
                installed = now == slots && value == block;
                slots = now;
            }
            if (!installed) {
                restore_thread_slot(vector, m.tls_index, block, saved);
                VirtualFree(block, 0, MEM_RELEASE);
                static std::atomic<bool> logged{false};
                if (!logged.exchange(true))
                    std::fprintf(stderr, "[rt] loader: %s: a running thread did not take its TLS block (it ended, or "
                                         "its TLS array kept changing); mapped thread-local data is not valid on it\n",
                                 m.name.c_str());
                continue;
            }
            state.blocks.push_back({c.id, m.tls_slot, &m, m.tls_generation, m.tls_index, vector, saved, block});
        }
    }
}

// ~Loader, with g_tls_lock held exclusively: the blocks load() put into other threads come out. A thread that has
// ended without a detach event (TerminateThread) has no TEB to restore.
void remove_from_existing_threads(Module& m) {
    ExistingThreadsLock lock;
    ExistingThreads& state = existing_threads();
    for (auto it = state.blocks.begin(); it != state.blocks.end();) {
        if (it->module != &m || it->generation != m.tls_generation) {
            ++it;
            continue;
        }
        ExistingThreadBlock b = *it;
        it = state.blocks.erase(it);
        const HANDLE h = OpenThread(THREAD_QUERY_INFORMATION, FALSE, b.thread);
        if (!h || running_thread_tls_vector(h) != b.vector) b.vector = nullptr;
        if (h) CloseHandle(h);
        remove_existing_block(b);
    }
}

// DLL_THREAD_ATTACH: a record under this thread's id is from a thread that ended without a detach event.
void forget_ended_thread_id() {
    const DWORD self = GetCurrentThreadId();
    ExistingThreadsLock lock;
    ExistingThreads& state = existing_threads();
    state.ended.erase(std::remove(state.ended.begin(), state.ended.end(), self), state.ended.end());
    for (auto it = state.blocks.begin(); it != state.blocks.end();) {
        if (it->thread != self) {
            ++it;
            continue;
        }
        VirtualFree(it->block, 0, MEM_RELEASE);
        it = state.blocks.erase(it);
    }
}

// DLL_THREAD_DETACH, before this thread's own records are released: the blocks load() gave it come out, newest first,
// after the module's TLS callbacks have seen the event (Windows sends DLL_THREAD_DETACH to a DLL for a thread that
// was running before it was loaded, too).
void release_existing_blocks_of_this_thread() {
    const DWORD self = GetCurrentThreadId();
    std::vector<ExistingThreadBlock> mine;
    {
        ExistingThreadsLock lock;
        for (const auto& b : existing_threads().blocks)
            if (b.thread == self) mine.push_back(b);
    }
    for (auto it = mine.rbegin(); it != mine.rend(); ++it) {
        Module* m = it->module;
        if (g_tls_slot_owners[it->slot].load() != m || g_tls_slot_generations[it->slot].load() != it->generation ||
            !m->initialised)
            continue;
        try {
            run_callbacks(*m, DLL_THREAD_DETACH);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[rt] loader: thread detach: %s\n", e.what());
        }
    }
    ExistingThreadsLock lock;
    ExistingThreads& state = existing_threads();
    for (auto it = state.blocks.end(); it != state.blocks.begin();) {
        --it;
        if (it->thread != self) continue;
        remove_existing_block(*it);
        it = state.blocks.erase(it);
    }
    // Bounded: an id matters only until its thread has left the process, moments after this event, so the oldest
    // can go. (An id also leaves the list when a new thread gets it.)
    if (state.ended.size() >= kMaxEndedThreads) state.ended.erase(state.ended.begin(), state.ended.begin() + kMaxEndedThreads / 2);
    state.ended.push_back(self);
}

// Runs in gta6rt_tls_host.dll's DllMain for each thread the process starts or ends once the helper is loaded, under
// the Windows loader lock, on that thread. A new thread gets its own copy of each active TLS owner's template; an
// ending one gives each helper's block back before the Windows loader frees it. TLS callbacks get the thread events
// only after the module's DLL_PROCESS_ATTACH (initialise()), as the Windows loader orders them.
void __cdecl on_host_thread_event(DWORD reason) {
    SharedTlsLock lock;
    // The try/catch is per slot: a throw from one module's install or callbacks must not take the thread events
    // away from the other slots. Nothing may leave DllMain.
    if (reason == DLL_THREAD_ATTACH) {
        try {
            forget_ended_thread_id();
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[rt] loader: thread attach: %s\n", e.what());
        }
        for (size_t slot = 0; slot < kMaxMappedTlsModules; ++slot) {
            Module* m = g_tls_slot_owners[slot].load();
            if (!m) continue;
            try {
                install_thread_block(*m);
                if (m->initialised) run_callbacks(*m, DLL_THREAD_ATTACH);
            } catch (const std::exception& e) {
                // The thread runs on without this module's TLS; later slots still get theirs.
                std::fprintf(stderr, "[rt] loader: thread attach, slot %zu (%s): %s\n", slot, m->name.c_str(),
                             e.what());
            }
        }
    } else if (reason == DLL_THREAD_DETACH) {
        // The blocks load() gave this thread while it was already running come out first (they can sit on top of
        // an older module's block in the same slot).
        try {
            release_existing_blocks_of_this_thread();
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[rt] loader: thread detach: %s\n", e.what());
        }
        for (size_t i = kMaxMappedTlsModules; i-- > 0;) {
            Module* m = g_tls_slot_owners[i].load();
            if (m && g_thread_tls_slots[i].module == m &&
                g_thread_tls_slots[i].generation == m->tls_generation) {
                try {
                    if (m->initialised) run_callbacks(*m, DLL_THREAD_DETACH);
                } catch (const std::exception& e) {
                    std::fprintf(stderr, "[rt] loader: thread detach, slot %zu (%s): %s\n", i, m->name.c_str(),
                                 e.what());
                }
                // Released even when the callbacks threw: a leaked block would be handed to the OS loader as the
                // helper's own TLS block and freed with it at thread exit.
                uninstall_thread_block(*m);
            } else {
                release_thread_slot(i);  // installed by a module whose Loader is gone (or nothing to do)
            }
        }
    }
}

// The Xbox builds of the MSVC runtime import the same CRT the desktop does, so the host's own copies are used.
bool prefer_host_copy(const std::string& dll) {
    // The second row is the Debug builds of the same DLLs. A Debug build of a stand-in image (tiny_game) links them,
    // and without them here its C++ exceptions were not redirected below: a throw caught inside the image ended the
    // run with 0xE06D7363 (run_tiny_game_eh in a Debug build).
    static const char* names[] = {"vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll", "msvcp140_1.dll",
                                  "msvcp140_2.dll", "msvcp140_atomic_wait.dll", "msvcp140_codecvt_ids.dll",
                                  "concrt140.dll", "vccorlib140.dll", "vcomp140.dll",
                                  "vcruntime140d.dll", "vcruntime140_1d.dll", "msvcp140d.dll", "msvcp140_1d.dll",
                                  "msvcp140_2d.dll", "msvcp140d_atomic_wait.dll", "msvcp140d_codecvt_ids.dll",
                                  "concrt140d.dll", "vccorlib140d.dll", "vcomp140d.dll"};
    for (auto* n : names)
        if (dll == n) return true;
    return false;
}

// MSVC C++ exceptions in a mapped image. _CxxThrowException (vcruntime140.dll) puts the throwing image's base in the
// exception record, from RtlPcToFileHeader(pThrowInfo); the frame handler adds the ThrowInfo's RVAs (catchable types,
// copy constructor, destructor) to it. RtlPcToFileHeader only knows images the Windows loader mapped, so for ours the
// base is 0 and the exception can't be matched to a catch. The host copies the loader binds get their import of it
// redirected: the Windows answer first, then the modules a live Loader has mapped (shims::mapped_image_base).
using PcToFileHeaderFn = PVOID(NTAPI*)(PVOID, PVOID*);
std::atomic<PcToFileHeaderFn> g_host_pc_to_file_header{nullptr};

PVOID NTAPI pc_to_file_header(PVOID pc, PVOID* base) {
    if (PVOID found = g_host_pc_to_file_header.load()(pc, base)) return found;
    uint64_t mapped = 0;
    if (!shims::mapped_image_base(reinterpret_cast<uint64_t>(pc), mapped)) {
        *base = nullptr;
        return nullptr;
    }
    *base = reinterpret_cast<PVOID>(mapped);
    return *base;
}

// Points every import of RtlPcToFileHeader by name in a host module at pc_to_file_header, once per module. The
// module is a system DLL the Windows loader has already validated and bound, so its tables are read unchecked.
// The redirect stays for the process lifetime; with no live Loader it answers exactly as Windows does.
void redirect_pc_to_file_header(HMODULE host) {
    static std::mutex mutex;
    static std::vector<HMODULE> done;
    std::lock_guard lock(mutex);
    if (std::find(done.begin(), done.end(), host) != done.end()) return;
    done.push_back(host);

    auto* base = reinterpret_cast<uint8_t*>(host);
    auto* nt = at<IMAGE_NT_HEADERS64>(base, reinterpret_cast<IMAGE_DOS_HEADER*>(base)->e_lfanew);
    const auto& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    if (!dir.Size || !dir.VirtualAddress) return;
    for (auto* d = at<IMAGE_IMPORT_DESCRIPTOR>(base, dir.VirtualAddress); d->Name; ++d) {
        if (!d->OriginalFirstThunk) continue;  // no name table to match against
        auto* names = at<IMAGE_THUNK_DATA64>(base, d->OriginalFirstThunk);
        auto* iat = at<IMAGE_THUNK_DATA64>(base, d->FirstThunk);
        for (; names->u1.AddressOfData; ++names, ++iat) {
            if (IMAGE_SNAP_BY_ORDINAL64(names->u1.Ordinal)) continue;
            auto* by_name = at<IMAGE_IMPORT_BY_NAME>(base, static_cast<uint32_t>(names->u1.AddressOfData));
            if (std::strcmp(reinterpret_cast<const char*>(by_name->Name), "RtlPcToFileHeader") != 0) continue;
            auto* slot = reinterpret_cast<void**>(&iat->u1.Function);
            if (*slot == reinterpret_cast<void*>(&pc_to_file_header)) continue;
            PcToFileHeaderFn expected = nullptr;  // the first bound address is the Windows function
            g_host_pc_to_file_header.compare_exchange_strong(expected, reinterpret_cast<PcToFileHeaderFn>(*slot));
            DWORD old = 0;
            if (!VirtualProtect(slot, sizeof(void*), PAGE_READWRITE, &old)) continue;
            *slot = reinterpret_cast<void*>(&pc_to_file_header);
            VirtualProtect(slot, sizeof(void*), old, &old);
        }
    }
}

// The module a host import actually resolved into. GetProcAddress follows forwarders, so this can be another DLL
// than the one imported (on Windows vcruntime140.dll implements _CxxThrowException itself); that module's
// RtlPcToFileHeader import is the one the throw calls.
void redirect_pc_to_file_header_at(const void* resolved) {
    HMODULE owner = nullptr;
    if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           static_cast<LPCWSTR>(resolved), &owner) && owner)
        redirect_pc_to_file_header(owner);
}

}  // namespace

// Declared in pe_loader.h; the body needs the internal clone registry above.
void cleanup_tls_clones() { cleanup_tls_clones_internal(); }

namespace {
bool lookup_in_loader(void* ctx, uint64_t address, std::string& module, uint32_t& rva) {
    return static_cast<Loader*>(ctx)->find_address(address, module, rva);
}

void* bundled_find_loaded(void* ctx, const std::string& dll_lower) {
    return static_cast<Loader*>(ctx)->find_loaded_module(dll_lower);
}

void* bundled_load(void* ctx, const std::string& dll_lower, uint32_t flags, bool& handled) {
    return static_cast<Loader*>(ctx)->load_runtime_module(dll_lower, flags, handled);
}

void* bundled_get_proc(void* ctx, void* module, const std::string& sym, bool& owns_handle) {
    return static_cast<Loader*>(ctx)->resolve_module_export(module, sym, owns_handle);
}

bool bundled_owns_handle(void* ctx, void* module) {
    return static_cast<Loader*>(ctx)->owns_module_handle(module);
}

void* bundled_find_loaded_w(void* ctx, const std::wstring& basename) {
    return static_cast<Loader*>(ctx)->find_loaded_module_w(basename);
}

void* bundled_load_w(void* ctx, const std::wstring& basename, uint32_t flags, bool& handled) {
    return static_cast<Loader*>(ctx)->load_runtime_module_w(basename, flags, handled);
}

const shims::BundledModuleOps kBundledOps{
    &bundled_find_loaded,
    &bundled_load,
    &bundled_get_proc,
    &bundled_owns_handle,
    &bundled_find_loaded_w,
    &bundled_load_w,
};

// Marks what a run-time load maps as a dynamic load for as long as it is in scope (the caller holds the mutex).
struct RuntimeLoadScope {
    explicit RuntimeLoadScope(int& depth) : depth_(depth) { ++depth_; }
    ~RuntimeLoadScope() { --depth_; }
    RuntimeLoadScope(const RuntimeLoadScope&) = delete;
    RuntimeLoadScope& operator=(const RuntimeLoadScope&) = delete;
    int& depth_;
};
}  // namespace

Loader::Loader(std::filesystem::path game_dir) : game_dir_(std::move(game_dir)) {
    shims::add_module_lookup(this, &lookup_in_loader);
    shims::add_bundled_loader(this, &kBundledOps);
}

Loader::~Loader() {
    // Importers must release their resources before their dependencies. Keep every mapping, unwind table and TLS
    // block alive throughout the notifications: a detach handler can still call an imported function or free an
    // FLS slot whose destructor lives in another mapped image.
    for (auto it = init_order_.rbegin(); it != init_order_.rend(); ++it) {
        Module& m = **it;
        if (!m.initialised) continue;  // a map-only load never opts in to running image code
        set_initialised(m, false);  // under the TLS lock: thread events stop running its callbacks
        try {
            run_tls_callbacks(m, DLL_PROCESS_DETACH);
        } catch (const std::exception& e) {
            // A module may have changed its callback table since attach. A destructor cannot propagate the validation
            // error; report it and still let DllMain release resources before the image is freed.
            std::fprintf(stderr, "[rt] TLS process detach failed for %s: %s\n", m.name.c_str(), e.what());
        }
        auto* nt = at<IMAGE_NT_HEADERS64>(m.base, reinterpret_cast<IMAGE_DOS_HEADER*>(m.base)->e_lfanew);
        if ((nt->FileHeader.Characteristics & IMAGE_FILE_DLL) && m.entry_rva) {
            auto entry = reinterpret_cast<BOOL(NTAPI*)(void*, DWORD, void*)>(m.base + m.entry_rva);
            entry(m.base, DLL_PROCESS_DETACH, nullptr);  // explicit unload; the return value is ignored
        }
    }
    shims::remove_bundled_loader(this);
    shims::remove_module_lookup(this);
    for (auto* m : order_) {
        // TLS state must go away before the image and its TLS block are unmapped.
        if (m->tls_index != kNoTlsIndex && m->tls_slot < kMaxMappedTlsModules) {
            // Retired first, under the lock: no thread event installs it or runs its callbacks after this, and a
            // thread that still holds a copy frees it when it ends (on_host_thread_event).
            {
                ExclusiveTlsLock lock;
                Module* owner = m;
                if (g_tls_slot_owners[m->tls_slot].compare_exchange_strong(owner, nullptr))
                    g_tls_slot_generations[m->tls_slot].store(0);
                remove_from_existing_threads(*m);
            }
            uninstall_tls_for_thread(*m);
        }
        // Before the image is freed: a stale table would let a later stack walk read unmapped memory.
        if (m->function_table) RtlDeleteFunctionTable(static_cast<PRUNTIME_FUNCTION>(m->function_table));
        if (m->tls_block) VirtualFree(m->tls_block, 0, MEM_RELEASE);
        if (m->base) VirtualFree(m->base, 0, MEM_RELEASE);
        delete m;
    }
}

Module& Loader::load(const std::filesystem::path& image) {
    std::lock_guard lock(mutex_);
    const auto key = module_key(image);
    if (auto it = by_name_.find(key); it != by_name_.end()) return *it->second;
    const auto narrow_key = narrow_module_key(image);
    if (Module* existing = find_narrow_module(narrow_key)) return *existing;
    Module* mapped = nullptr;
    try {
        mapped = &map_image(image, narrow_key);
        mapped->dynamic_load = runtime_loads_ > 0;  // before bind_imports: its dependencies are marked the same way
        bind_imports(*mapped);  // loads every bundled module m imports first, so they land in init_order_ before m
        protect_sections(*mapped);
        register_unwind(*mapped);
    } catch (...) {
        // A failed load, nested ones (a dependency of a run-time load) included, must not stay findable by name:
        // a later lookup would get a half-mapped image with unbound imports. ~Loader still frees it through order_.
        by_name_.erase(key);
        throw;
    }
    Module& m = *mapped;
    init_order_.push_back(&m);
    if (startup_trace::enabled())
        startup_trace::event("map", "\"module\":" + startup_trace::quote(m.name) +
                                        ",\"base\":" + startup_trace::quote(startup_trace::hex(reinterpret_cast<uint64_t>(m.base))) +
                                        ",\"size\":" + startup_trace::quote(startup_trace::hex(m.size)) +
                                        ",\"relocated\":" + (m.relocated ? "true" : "false"));
    return m;
}

Module* Loader::find_narrow_module(const std::string& dll_lower) const {
    if (dll_lower.empty()) return nullptr;
    // library_loader and PE import names keep their existing narrow-byte contract: the exact compare below is
    // bytewise, and the key is never reinterpreted as UTF-8.
    for (const auto& entry : by_name_)
        if (entry.second->narrow_key == dll_lower) return entry.second;
    // The key is ASCII case-folded only. A name with letters outside ASCII (Cyrillic in code page 1251, say)
    // compares case-insensitively through the code page, as GetModuleHandleA does: GUEST_ДА.EXE finds guest_да.exe.
    // Widening the folded key is safe here: rt::lower() folds ASCII letters only, and no ANSI code page's multibyte
    // sequences contain ASCII letter bytes, so both sides widen to the same characters they were spelt with.
    if (const std::wstring wide = acp_to_wide(dll_lower); !wide.empty())
        for (const auto& entry : by_name_)
            if (!entry.second->narrow_key.empty() &&
                same_module_name(acp_to_wide(entry.second->narrow_key).c_str(), wide.c_str()))
                return entry.second;
    return nullptr;
}

Module* Loader::find_wide_module(const std::wstring& basename) const {
    if (basename.empty()) return nullptr;
    for (const auto& entry : by_name_)
        if (same_module_name(entry.first.c_str(), basename.c_str())) return entry.second;
    return nullptr;
}

void* Loader::find_loaded_module_w(const std::wstring& basename) const {
    std::lock_guard lock(mutex_);
    Module* m = find_wide_module(basename);
    return m ? static_cast<void*>(m->base) : nullptr;
}

void* Loader::find_loaded_module(const std::string& dll_lower) const {
    std::lock_guard lock(mutex_);
    Module* m = find_narrow_module(dll_lower);
    return m ? static_cast<void*>(m->base) : nullptr;
}

void* Loader::load_runtime_module(const std::string& dll_lower, uint32_t flags, bool& handled) {
    handled = false;
    if (dll_lower.empty() || prefer_host_copy(dll_lower)) return nullptr;

    std::lock_guard lock(mutex_);
    const RuntimeLoadScope dynamic(runtime_loads_);
    Module* existing = find_narrow_module(dll_lower);
    const std::filesystem::path candidate = game_dir_ / dll_lower;
    std::error_code ec;
    const bool is_bundled = existing || std::filesystem::is_regular_file(candidate, ec);
    if (!is_bundled) return nullptr;

    handled = true;
    return load_runtime_candidate(existing, candidate, flags, dll_lower);
}

void* Loader::load_runtime_module_w(const std::wstring& basename, uint32_t flags, bool& handled) {
    handled = false;
    if (basename.empty()) return nullptr;
    // A name the code page can spell takes the narrow path's host-copy rule as well.
    if (const std::string narrow = narrow_module_key(std::filesystem::path(basename)); !narrow.empty())
        if (prefer_host_copy(narrow)) return nullptr;

    std::lock_guard lock(mutex_);
    Module* existing = find_wide_module(basename);
    const std::filesystem::path candidate = game_dir_ / basename;
    std::error_code ec;
    if (!existing && !std::filesystem::is_regular_file(candidate, ec)) return nullptr;

    handled = true;
    return load_runtime_candidate(existing, candidate, flags, path_utf8(basename));
}

void* Loader::load_runtime_candidate(Module* existing, const std::filesystem::path& candidate, uint32_t flags,
                                     const std::string& display) {
    // Win32 DONT_RESOLVE_DLL_REFERENCES (0x1) forbids resolving dependencies or running TLS/DllMain; refuse it
    // explicitly instead of silently performing a normal resolving + initialising load.
    if ((flags & DONT_RESOLVE_DLL_REFERENCES) != 0) {
        std::fprintf(stderr,
                     "[rt] loader: LoadLibraryEx(%s, DONT_RESOLVE_DLL_REFERENCES) is not supported for bundled DLLs\n",
                     display.c_str());
        SetLastError(ERROR_NOT_SUPPORTED);
        return nullptr;
    }

    if (existing) {
        try {
            initialise();
            return static_cast<void*>(existing->base);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[rt] loader: run-time initialise of %s failed: %s\n", display.c_str(), e.what());
            SetLastError(ERROR_DLL_INIT_FAILED);
            return nullptr;
        }
    }

    Module* mapped = nullptr;
    try {
        Module& m = load(candidate);
        mapped = &m;
        initialise();
        std::fprintf(stderr, "[rt] loader: %s mapped from game_dir (run-time load)\n", m.name.c_str());
        return static_cast<void*>(m.base);
    } catch (const std::exception& e) {
        // load() drops a module that failed to map or bind (dependencies included); this also drops one whose
        // initialise() failed, so later GetModuleHandle/LoadLibrary calls don't see it. ~Loader frees it via order_.
        by_name_.erase(module_key(mapped ? mapped->path : candidate));
        std::fprintf(stderr, "[rt] loader: run-time load of %s failed: %s\n", display.c_str(), e.what());
        SetLastError(ERROR_DLL_INIT_FAILED);
        return nullptr;
    }
}

void* Loader::resolve_module_export(void* module_handle, const std::string& sym, bool& owns_handle) {
    owns_handle = false;
    if (!module_handle) return nullptr;
    std::lock_guard lock(mutex_);
    Module* target = nullptr;
    for (Module* m : order_) {
        if (m->base == module_handle) {
            target = m;
            break;
        }
    }
    if (!target) return nullptr;
    owns_handle = true;
    if (auto it = target->exports.find(sym); it != target->exports.end()) return it->second;
    if (auto it = target->forwarders.find(sym); it != target->forwarders.end()) {
        try {
            const RuntimeLoadScope dynamic(runtime_loads_);
            Binding binding = Binding::Unimplemented;
            void* fn = resolve_forwarder(it->second, binding, 0);
            if (binding == Binding::Module) initialise();
            return fn;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[rt] loader: forwarder %s from %s failed: %s\n", it->second.c_str(),
                         target->name.c_str(), e.what());
            return nullptr;
        }
    }
    return nullptr;
}

bool Loader::find_address(uint64_t address, std::string& module, uint32_t& rva) const {
    std::shared_lock lock(ranges_mutex_);
    for (const MappedRange& r : ranges_) {
        if (address >= r.base && address - r.base < r.size) {
            module = r.name;
            rva = static_cast<uint32_t>(address - r.base);
            return true;
        }
    }
    return false;
}

bool Loader::owns_module_handle(void* module_handle) const {
    if (!module_handle) return false;
    std::lock_guard lock(mutex_);
    for (const Module* m : order_)
        if (m->base == module_handle) return true;
    return false;
}

// x64 has no frame-pointer chain to fall back on: without its .pdata registered, an exception raised in (or passing
// through) a mapped image can't be dispatched past that frame, and the process dies. Images the system loader maps
// are found through its own module list; ours are VirtualAlloc'd, so they are registered as dynamic function tables.
//
// The table is checked before it is handed over, because RtlLookupFunctionEntry trusts it: it binary-searches the
// entries by BeginAddress and follows UnwindData without bounds checks. Measured over 3,780 x64 images in
// System32 (5,023,071 entries) and the package's FidelityFX DLLs: every table is sorted and non-overlapping and in
// bounds, and one shipped image (symcryptk.dll) has a zero-length entry (BeginAddress == EndAddress), so that is
// allowed. No entry in that set uses the indirect form (UnwindData bit 0).
void Loader::register_unwind(Module& m) {
    auto* nt = at<IMAGE_NT_HEADERS64>(m.base, reinterpret_cast<IMAGE_DOS_HEADER*>(m.base)->e_lfanew);
    const auto dir = directory(nt, IMAGE_DIRECTORY_ENTRY_EXCEPTION);
    // Like the system loader, a trailing partial entry is ignored.
    const uint32_t count = dir.Size / sizeof(RUNTIME_FUNCTION);
    if (!count) return;
    require_range(m.name, dir.VirtualAddress, static_cast<uint64_t>(count) * sizeof(RUNTIME_FUNCTION), m.size,
                  "the exception directory");
    auto* table = at<RUNTIME_FUNCTION>(m.base, dir.VirtualAddress);
    auto reject = [&](uint32_t i, const char* why) {
        throw std::runtime_error(m.name + ": unwind entry " + std::to_string(i) + " " + why);
    };
    uint32_t previous_end = 0;
    for (uint32_t i = 0; i < count; ++i) {
        const RUNTIME_FUNCTION& f = table[i];
        if (f.EndAddress < f.BeginAddress || f.EndAddress > m.size) reject(i, "covers code outside the image");
        if (f.BeginAddress < previous_end) reject(i, "is out of order or overlaps the one before it");
        // UNWIND_INFO starts with a 4-byte header; its codes and handler data are the unwinder's to read.
        const uint32_t info = f.UnwindData & ~1u;
        if (info > m.size || m.size - info < 4) reject(i, "has unwind data outside the image");
        previous_end = f.EndAddress;
    }
    if (!RtlAddFunctionTable(table, count, reinterpret_cast<DWORD64>(m.base)))
        throw std::runtime_error(m.name + ": RtlAddFunctionTable failed");
    m.function_table = table;
    m.function_count = count;
}

void Loader::initialise() {
    // Not order_: map_image pushes a module there before bind_imports maps its dependencies, so order_ lists an
    // importer ahead of the DLLs it imports. init_order_ is filled once a module's imports are all loaded.
    for (auto* m : init_order_) initialise(*m);
}

void Loader::initialise(Module& m) {
    if (m.initialised) return;

    // TLS was installed by load(); initialise() only delivers the events, so no slot is allocated or written twice.
    run_tls_callbacks(m, DLL_PROCESS_ATTACH);

    // Only a DLL gets DllMain. An EXE's entry point has a different signature and is the caller's to run.
    auto* nt = at<IMAGE_NT_HEADERS64>(m.base, reinterpret_cast<IMAGE_DOS_HEADER*>(m.base)->e_lfanew);
    if ((nt->FileHeader.Characteristics & IMAGE_FILE_DLL) && m.entry_rva) {
        auto entry = reinterpret_cast<BOOL(NTAPI*)(void*, DWORD, void*)>(m.base + m.entry_rva);
        // lpvReserved "is NULL for dynamic loads and non-NULL for static loads" (DllMain docs). What Windows points
        // it at is its own; a static load gets a CONTEXT captured here, so a DLL that reads through it reads memory.
        CONTEXT context{};
        if (!m.dynamic_load) RtlCaptureContext(&context);
        const BOOL ok = entry(m.base, DLL_PROCESS_ATTACH, m.dynamic_load ? nullptr : &context);
        if (startup_trace::enabled())
            startup_trace::event("dllmain", "\"module\":" + startup_trace::quote(m.name) +
                                                ",\"result\":" + std::to_string(ok));
        if (!ok) {
            set_initialised(m, true);  // it ran: a second DLL_PROCESS_ATTACH would be wrong
            throw std::runtime_error(m.name + ": DllMain returned FALSE");
        }
    }
    set_initialised(m, true);
}

// Under the TLS lock, which orders it against the thread events: they run a module's TLS callbacks only while this is
// true (from initialise() until ~Loader starts the process detach).
void Loader::set_initialised(Module& m, bool value) {
    ExclusiveTlsLock lock;
    m.initialised = value;
}

Module& Loader::map_image(const std::filesystem::path& path, const std::string& narrow_key) {
    auto file = read_file(path);
    const std::string name = path_utf8(path.filename());
    auto* dos = reinterpret_cast<IMAGE_DOS_HEADER*>(file.data());
    if (file.size() < sizeof(IMAGE_DOS_HEADER) || dos->e_magic != IMAGE_DOS_SIGNATURE)
        throw std::runtime_error(path_utf8(path) + ": not an MZ image");
    if (dos->e_lfanew < 0 || static_cast<uint64_t>(dos->e_lfanew) + sizeof(IMAGE_NT_HEADERS64) > file.size())
        throw std::runtime_error(name + ": the NT headers are outside the file");
    auto* nt = reinterpret_cast<IMAGE_NT_HEADERS64*>(file.data() + dos->e_lfanew);
    if (nt->Signature != IMAGE_NT_SIGNATURE || nt->FileHeader.Machine != IMAGE_FILE_MACHINE_AMD64)
        throw std::runtime_error(name + ": not an x86-64 PE");
    if (nt->FileHeader.SizeOfOptionalHeader < sizeof(IMAGE_OPTIONAL_HEADER64) ||
        nt->OptionalHeader.Magic != IMAGE_NT_OPTIONAL_HDR64_MAGIC)
        throw std::runtime_error(name + ": not a PE32+ image");
    auto& oh = nt->OptionalHeader;
    const uint64_t sections = static_cast<uint64_t>(dos->e_lfanew) + 4 + sizeof(IMAGE_FILE_HEADER) +
                              nt->FileHeader.SizeOfOptionalHeader;
    const uint64_t headers_end =
        sections + static_cast<uint64_t>(nt->FileHeader.NumberOfSections) * sizeof(IMAGE_SECTION_HEADER);
    if (headers_end > file.size())
        throw std::runtime_error(name + ": the section table runs past the end of the file");
    if (!oh.SizeOfImage || oh.SizeOfHeaders < headers_end || oh.SizeOfHeaders > oh.SizeOfImage ||
        oh.SizeOfHeaders > file.size())
        throw std::runtime_error(name + ": SizeOfHeaders and SizeOfImage don't fit the headers and file");
    if (oh.AddressOfEntryPoint && oh.AddressOfEntryPoint >= oh.SizeOfImage)
        throw std::runtime_error(name + ": the entry point is outside the image");

    auto* m = new Module;
    m->name = name;
    m->narrow_key = narrow_key;
    m->path = path;
    m->preferred_base = oh.ImageBase;
    m->size = oh.SizeOfImage;
    m->entry_rva = oh.AddressOfEntryPoint;
    by_name_[module_key(path)] = m;
    order_.push_back(m);

    // Inside the range gta6rt reserved for this image before the process started (base_reservation.h), else as usual.
    m->base = static_cast<uint8_t*>(base_reservation::commit_reserved(oh.ImageBase, oh.SizeOfImage));
    if (!m->base)
        m->base = static_cast<uint8_t*>(VirtualAlloc(reinterpret_cast<void*>(oh.ImageBase), oh.SizeOfImage,
                                                     MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    if (!m->base) {
        const auto relocs = directory(nt, IMAGE_DIRECTORY_ENTRY_BASERELOC);
        if (!relocs.Size)
            throw std::runtime_error(m->name + ": preferred base is taken and the image has no relocations");
        m->base = static_cast<uint8_t*>(VirtualAlloc(nullptr, oh.SizeOfImage, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
        if (!m->base) throw std::runtime_error(m->name + ": VirtualAlloc failed");
    }
    {
        // From here an address inside the image names it. The memory stays until ~Loader, which unregisters the
        // lookup first.
        std::unique_lock lock(ranges_mutex_);
        ranges_.push_back({reinterpret_cast<uint64_t>(m->base), m->size, m->name});
    }

    std::memcpy(m->base, file.data(), oh.SizeOfHeaders);
    auto* sec = IMAGE_FIRST_SECTION(nt);
    for (unsigned i = 0; i < nt->FileHeader.NumberOfSections; ++i, ++sec) {
        if (sec->VirtualAddress < oh.SizeOfHeaders)
            throw std::runtime_error(m->name + ": section " + std::to_string(i) + " overlaps the headers");
        uint32_t n = std::min(sec->SizeOfRawData, sec->Misc.VirtualSize ? sec->Misc.VirtualSize : sec->SizeOfRawData);
        if (!n) continue;
        if (sec->PointerToRawData > file.size() || n > file.size() - sec->PointerToRawData)
            throw std::runtime_error(m->name + ": section " + std::to_string(i) + " reads past the end of the file");
        require_range(m->name, sec->VirtualAddress, n, m->size, "a section's data");
        std::memcpy(m->base + sec->VirtualAddress, file.data() + sec->PointerToRawData, n);
    }

    int64_t delta = reinterpret_cast<int64_t>(m->base) - static_cast<int64_t>(oh.ImageBase);
    if (delta) {
        m->relocated = true;
        const auto dir = directory(nt, IMAGE_DIRECTORY_ENTRY_BASERELOC);
        if (!dir.Size) throw std::runtime_error(m->name + ": the image has no relocations and isn't at its base");
        require_range(m->name, dir.VirtualAddress, dir.Size, m->size, "the relocation table");
        auto* block = at<IMAGE_BASE_RELOCATION>(m->base, dir.VirtualAddress);
        auto* end = reinterpret_cast<uint8_t*>(block) + dir.Size;
        while (reinterpret_cast<uint8_t*>(block) < end) {
            const size_t remaining = static_cast<size_t>(end - reinterpret_cast<uint8_t*>(block));
            if (remaining < sizeof(IMAGE_BASE_RELOCATION) || block->SizeOfBlock < sizeof(IMAGE_BASE_RELOCATION) ||
                block->SizeOfBlock > remaining)
                throw std::runtime_error(m->name + ": a relocation block doesn't fit the relocation table");
            auto* entries = reinterpret_cast<uint16_t*>(block + 1);
            size_t count = (block->SizeOfBlock - sizeof(IMAGE_BASE_RELOCATION)) / 2;
            for (size_t i = 0; i < count; ++i) {
                uint16_t type = entries[i] >> 12, off = entries[i] & 0xFFF;
                if (type != IMAGE_REL_BASED_ABSOLUTE &&
                    static_cast<uint64_t>(block->VirtualAddress) + off < oh.SizeOfHeaders)
                    throw std::runtime_error(m->name + ": a relocation lands in the headers");
                if (type == IMAGE_REL_BASED_DIR64) {
                    require_range(m->name, static_cast<uint64_t>(block->VirtualAddress) + off, sizeof(uint64_t), m->size,
                                  "a relocation");
                    *at<uint64_t>(m->base, block->VirtualAddress + off) += delta;
                } else if (type == IMAGE_REL_BASED_HIGHLOW) {
                    // A 32-bit absolute address (code built /LARGEADDRESSAWARE:NO), which the x64 Windows loader
                    // also applies. Where the new address no longer fits in 32 bits, Windows would truncate it; the
                    // image would then jump or read through a wrong pointer, so the load is refused instead.
                    require_range(m->name, static_cast<uint64_t>(block->VirtualAddress) + off, sizeof(uint32_t),
                                  m->size, "a relocation");
                    auto* slot = at<uint32_t>(m->base, block->VirtualAddress + off);
                    const int64_t moved = static_cast<int64_t>(*slot) + delta;
                    if (moved < 0 || moved > 0xFFFFFFFFll)
                        throw std::runtime_error(m->name + ": a 32-bit (HIGHLOW) relocation does not fit at this base");
                    *slot = static_cast<uint32_t>(moved);
                } else if (type != IMAGE_REL_BASED_ABSOLUTE) {
                    throw std::runtime_error(m->name + ": unsupported relocation type " + std::to_string(type));
                }
            }
            block = reinterpret_cast<IMAGE_BASE_RELOCATION*>(reinterpret_cast<uint8_t*>(block) + block->SizeOfBlock);
        }
    }
    // The TLS directory and the exception directory hold virtual addresses, so both are read after relocation. TLS is
    // installed here, before bind_imports can return control to the caller and before any image code can run.
    initialize_tls(*m);
    parse_exports(*m);
    return *m;
}

// Static TLS for the thread that calls load(). The PE TLS directory only carries the template (Start..End) and the
// zero-fill size; it has no field naming the TLS block. The Windows loader allocates a block per module per thread and
// puts it in TEB->ThreadLocalStoragePointer[module index], which is what __declspec(thread) code reads via _tls_index.
// Only the OS loader can allocate that module index, so the loader reuses the index the OS assigned to
// gta6rt_tls_host.dll and maps a fresh clone of the helper for every additional simultaneous TLS module, each clone
// getting its own index. All kMaxMappedTlsModules slots can be live at once; a further module is rejected rather than
// silently sharing (and corrupting) a slot.
void Loader::initialize_tls(Module& m) {
    auto* tls = tls_directory(m);
    if (!tls) return;
    // A present but all-zero directory is treated like an absent one.
    if (!tls->StartAddressOfRawData && !tls->EndAddressOfRawData && !tls->AddressOfIndex && !tls->AddressOfCallBacks &&
        !tls->SizeOfZeroFill)
        return;
    if (tls->EndAddressOfRawData < tls->StartAddressOfRawData)
        throw std::runtime_error(m.name + ": TLS template end precedes its start");
    uint64_t raw = tls->EndAddressOfRawData - tls->StartAddressOfRawData;
    uint64_t total = raw + tls->SizeOfZeroFill;
    if (total > 0xFFFFFFFFull) throw std::runtime_error(m.name + ": TLS block is too large");
    const uint32_t raw_rva =
        raw ? static_cast<uint32_t>(rva_in(m, tls->StartAddressOfRawData, "the TLS template")) : 0;
    if (raw) require_range(m.name, raw_rva, raw, m.size, "the TLS template");
    uint64_t index_rva = 0;
    if (tls->AddressOfIndex) {
        index_rva = rva_in(m, tls->AddressOfIndex, "the TLS index");
        require_range(m.name, index_rva, sizeof(uint32_t), m.size, "the TLS index");
    }

    if (!module_tls_slots()) throw std::runtime_error(m.name + ": this thread has no TLS array");

    size_t slot = kMaxMappedTlsModules;
    DWORD index = TLS_OUT_OF_INDEXES;
    {
        std::lock_guard alloc_lock(g_tls_alloc_mutex);
        for (size_t i = 0; i < kMaxMappedTlsModules; ++i) {
            if (g_tls_slot_owners[i].load() == nullptr) {
                slot = i;
                break;
            }
        }
        if (slot >= kMaxMappedTlsModules)
            throw std::runtime_error(m.name + ": all " + std::to_string(kMaxMappedTlsModules) +
                                     " manually mapped static TLS slots are in use");

        // Load/clone gta6rt_tls_host.dll OUTSIDE ExclusiveTlsLock so LoadLibraryExW cannot deadlock with
        // on_host_thread_event (which runs under the Windows loader lock and takes SharedTlsLock).
        index = ensure_tls_host(slot).index;
        if (index == TLS_OUT_OF_INDEXES)
            throw std::runtime_error(m.name + ": gta6rt_tls_host.dll is missing or has no module TLS index");
        if (index >= kMaxModuleTlsSlots) throw std::runtime_error(m.name + ": reserved TLS index is out of range");

        // The fields are complete before a thread event can see the owner.
        ExclusiveTlsLock lock;
        Module* expected = nullptr;
        if (!g_tls_slot_owners[slot].compare_exchange_strong(expected, &m))
            throw std::runtime_error(m.name + ": static TLS slot was claimed concurrently");

        m.tls_slot = static_cast<uint32_t>(slot);
        m.tls_generation = g_next_tls_generation.fetch_add(1, std::memory_order_relaxed);
        g_tls_slot_generations[slot].store(m.tls_generation);
        m.tls_index = index;
        m.tls_template = raw ? m.base + raw_rva : nullptr;
        m.tls_raw_size = static_cast<uint32_t>(raw);
        m.tls_zero_fill = tls->SizeOfZeroFill;
        m.tls_block_size = static_cast<uint32_t>(total);
        // The index lives in a writable section and protect_sections() has not run yet, so the write needs no
        // VirtualProtect dance.
        if (index_rva) *reinterpret_cast<uint32_t*>(m.base + index_rva) = index;
    }
    try {
        install_thread_block(m);
    } catch (...) {
        ExclusiveTlsLock lock;
        m.tls_index = kNoTlsIndex;
        m.tls_slot = kNoTlsIndex;
        m.tls_generation = 0;
        g_tls_slot_generations[slot].store(0);
        g_tls_slot_owners[slot].store(nullptr);
        throw;
    }
    // The loading thread's copy is the one the Module records (and ~Loader frees on that thread).
    if (g_thread_tls_slots[slot].module == &m && g_thread_tls_slots[slot].generation == m.tls_generation) {
        m.tls_block = g_thread_tls_slots[slot].block;
        m.tls_saved_block = g_thread_tls_slots[slot].saved;
    }
    // Threads started from here on get their copy from the helper DLL's DLL_THREAD_ATTACH; the ones already running
    // get it now.
    install_on_existing_threads(m);
}

void Loader::install_tls_for_thread(Module& m) { install_thread_block(m); }

void Loader::uninstall_tls_for_thread(Module& m) { uninstall_thread_block(m); }

void Loader::run_tls_callbacks(Module& m, uint32_t reason) { run_callbacks(m, reason, /*trace=*/true); }

void Loader::parse_exports(Module& m) {
    auto* nt = at<IMAGE_NT_HEADERS64>(m.base, reinterpret_cast<IMAGE_DOS_HEADER*>(m.base)->e_lfanew);
    const auto dir = directory(nt, IMAGE_DIRECTORY_ENTRY_EXPORT);
    if (!dir.Size) return;
    require_range(m.name, dir.VirtualAddress, sizeof(IMAGE_EXPORT_DIRECTORY), m.size, "the export directory");
    auto* ed = at<IMAGE_EXPORT_DIRECTORY>(m.base, dir.VirtualAddress);
    require_range(m.name, ed->AddressOfFunctions, static_cast<uint64_t>(ed->NumberOfFunctions) * sizeof(uint32_t), m.size,
                  "the export address table");
    require_range(m.name, ed->AddressOfNames, static_cast<uint64_t>(ed->NumberOfNames) * sizeof(uint32_t), m.size,
                  "the export name table");
    require_range(m.name, ed->AddressOfNameOrdinals, static_cast<uint64_t>(ed->NumberOfNames) * sizeof(uint16_t), m.size,
                  "the export ordinal table");
    auto* funcs = at<uint32_t>(m.base, ed->AddressOfFunctions);
    auto* names = at<uint32_t>(m.base, ed->AddressOfNames);
    auto* ords = at<uint16_t>(m.base, ed->AddressOfNameOrdinals);
    // An export whose RVA lands inside the export directory is a forwarder string ("KERNEL32.Sleep",
    // "api-ms-win-core-file-l1-1-0.CreateFileW"), not code. Storing base + that RVA would hand callers a pointer
    // into the string, so forwarders are kept apart and followed when an import asks for one.
    const uint64_t dir_begin = dir.VirtualAddress, dir_end = dir_begin + dir.Size;
    auto add_export = [&](const std::string& key, uint32_t rva) {
        if (!rva) return;  // An unused address-table slot is absent even when a name refers to it.
        if (rva >= dir_begin && rva < dir_end)
            m.forwarders[key] = read_string(m.name, m.base, m.size, rva, "a forwarded export");
        else if (rva >= m.size)  // base + rva would point past the mapping, and a caller would jump there
            throw std::runtime_error(m.name + ": export " + key + " points outside the image");
        else
            m.exports[key] = m.base + rva;
    };
    for (uint32_t i = 0; i < ed->NumberOfFunctions; ++i)
        if (funcs[i]) add_export("#" + std::to_string(ed->Base + i), funcs[i]);
    for (uint32_t i = 0; i < ed->NumberOfNames; ++i) {
        if (ords[i] >= ed->NumberOfFunctions)
            throw std::runtime_error(m.name + ": an export name points past the address table");
        add_export(read_string(m.name, m.base, m.size, names[i], "an export name"), funcs[ords[i]]);
    }
}

// Follows a forwarder string to the module and export it names. Bundled modules are searched first, with or without
// the extension, because a forwarder may point at another mapped module; system DLLs go through LoadLibraryEx, whose
// own GetProcAddress already follows its forwarders. Errors are "not found": the caller decides what to bind.
void* Loader::resolve_forwarder(const std::string& spec, Binding& binding, int depth) {
    if (depth >= 8) return nullptr;  // a longer chain is a loop between modules, not a re-export
    auto dot = spec.rfind('.');
    if (dot == std::string::npos || dot == 0 || dot + 1 >= spec.size()) return nullptr;
    const std::string module = spec.substr(0, dot), symbol = spec.substr(dot + 1);

    // A forwarder can name a module the loader shims. Shims take precedence over a bundled or host copy, exactly as
    // they do for a direct import in resolve(); without this, a forwarded xmem or d3d12_xs call becomes a failure thunk.
    std::string shim_key = lower(module);
    if (shim_key.size() < 4 || shim_key.compare(shim_key.size() - 4, 4, ".dll") != 0) shim_key += ".dll";
    if (void* p = shims::find(shim_key, symbol)) {
        binding = Binding::Shim;
        return p;
    }

    const bool by_ordinal = symbol.size() > 1 && symbol[0] == '#';
    uint32_t ordinal = 0;
    if (by_ordinal) {
        for (size_t i = 1; i < symbol.size(); ++i) {
            if (symbol[i] < '0' || symbol[i] > '9') return nullptr;
            ordinal = ordinal * 10 + static_cast<uint32_t>(symbol[i] - '0');
            if (ordinal > 0xFFFF) return nullptr;  // checked after the digit, so "#65537" can't truncate to 1
        }
    }

    // shim_key, not the bare module name: forwarder strings usually leave out ".dll" ("MSVCP140.?foo"), and the host
    // CRT copies are named with it, so a forwarded CRT symbol must skip the bundled Xbox build as a direct import does.
    const std::filesystem::path candidates[] = {game_dir_ / module, game_dir_ / (module + ".dll")};
    for (const auto& candidate : candidates) {
        std::error_code ec;
        if (prefer_host_copy(shim_key) || !std::filesystem::exists(candidate, ec)) continue;
        Module& dep = load(candidate);
        if (auto it = dep.exports.find(symbol); it != dep.exports.end()) {
            binding = Binding::Module;
            return it->second;
        }
        if (auto it = dep.forwarders.find(symbol); it != dep.forwarders.end())
            return resolve_forwarder(it->second, binding, depth + 1);
        return nullptr;  // the bundled module exists and does not export it; don't look for a host namesake
    }

    HMODULE h = LoadLibraryExA(module.c_str(), nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!h) return nullptr;
    if (prefer_host_copy(shim_key)) redirect_pc_to_file_header(h);
    FARPROC p = GetProcAddress(h, by_ordinal ? MAKEINTRESOURCEA(static_cast<WORD>(ordinal)) : symbol.c_str());
    if (!p) return nullptr;
    if (prefer_host_copy(shim_key)) redirect_pc_to_file_header_at(reinterpret_cast<const void*>(p));
    binding = Binding::Host;
    return reinterpret_cast<void*>(p);
}

void* Loader::resolve(Module& m, const std::string& dll, const std::string& name, uint16_t ordinal, bool by_ordinal) {
    std::string key = lower(dll);
    std::string sym = by_ordinal ? "#" + std::to_string(ordinal) : name;

    if (void* p = shims::find(key, sym)) {
        m.imports.push_back({dll, sym, Binding::Shim});
        return p;
    }
    // Bundled game module (an Xbox build that ships beside the game)?
    auto bundled = game_dir_ / dll;
    if (!prefer_host_copy(key) && std::filesystem::exists(bundled)) {
        Module& dep = load(bundled);
        if (auto it = dep.exports.find(sym); it != dep.exports.end()) {
            m.imports.push_back({dll, sym, Binding::Module});
            return it->second;
        }
        if (auto it = dep.forwarders.find(sym); it != dep.forwarders.end()) {
            Binding binding = Binding::Unimplemented;
            if (void* p = resolve_forwarder(it->second, binding, 0)) {
                m.imports.push_back({dll, sym, binding});
                return p;
            }
        }
    }
    // Host Windows: system DLLs and API sets. LOAD_LIBRARY_SEARCH_SYSTEM32 keeps the game folder out of the search.
    if (HMODULE h = LoadLibraryExA(dll.c_str(), nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32)) {
        if (prefer_host_copy(key)) redirect_pc_to_file_header(h);  // before any mapped code can throw through it
        FARPROC p = GetProcAddress(h, by_ordinal ? MAKEINTRESOURCEA(ordinal) : name.c_str());
        if (p) {
            if (prefer_host_copy(key)) redirect_pc_to_file_header_at(reinterpret_cast<const void*>(p));
            m.imports.push_back({dll, sym, Binding::Host});
            return reinterpret_cast<void*>(p);
        }
    }
    m.imports.push_back({dll, sym, Binding::Unimplemented});
    return shims::make_unimplemented_thunk(dll, sym);
}

void Loader::bind_imports(Module& m) {
    auto* nt = at<IMAGE_NT_HEADERS64>(m.base, reinterpret_cast<IMAGE_DOS_HEADER*>(m.base)->e_lfanew);

    // Thunk tables have no length field: they end at a zero entry, so each step is checked against the image.
    auto bind_thunks = [&](const std::string& dll, uint64_t lookup_rva, uint64_t iat_rva) {
        for (;; lookup_rva += sizeof(IMAGE_THUNK_DATA64), iat_rva += sizeof(IMAGE_THUNK_DATA64)) {
            require_range(m.name, lookup_rva, sizeof(IMAGE_THUNK_DATA64), m.size, "an import lookup table");
            require_range(m.name, iat_rva, sizeof(IMAGE_THUNK_DATA64), m.size, "an import address table");
            auto* lookup = at<IMAGE_THUNK_DATA64>(m.base, static_cast<uint32_t>(lookup_rva));
            if (!lookup->u1.AddressOfData) return;
            bool by_ord = IMAGE_SNAP_BY_ORDINAL64(lookup->u1.Ordinal);
            uint16_t ord = by_ord ? (uint16_t)IMAGE_ORDINAL64(lookup->u1.Ordinal) : 0;
            std::string name;
            if (!by_ord) {
                require_range(m.name, lookup->u1.AddressOfData, sizeof(uint16_t), m.size, "an import name");
                uint32_t hint = static_cast<uint32_t>(lookup->u1.AddressOfData);
                name = read_string(m.name, m.base, m.size, hint + sizeof(uint16_t), "an import name");
            }
            at<IMAGE_THUNK_DATA64>(m.base, static_cast<uint32_t>(iat_rva))->u1.Function =
                reinterpret_cast<uint64_t>(resolve(m, dll, name, ord, by_ord));
        }
    };

    const auto dir = directory(nt, IMAGE_DIRECTORY_ENTRY_IMPORT);
    if (dir.Size) {
        for (uint64_t i = 0;; ++i) {
            uint64_t rva = dir.VirtualAddress + i * sizeof(IMAGE_IMPORT_DESCRIPTOR);
            require_range(m.name, rva, sizeof(IMAGE_IMPORT_DESCRIPTOR), m.size, "the import table");
            auto* d = at<IMAGE_IMPORT_DESCRIPTOR>(m.base, static_cast<uint32_t>(rva));
            if (!d->Name) break;
            std::string dll = read_string(m.name, m.base, m.size, d->Name, "an imported DLL name");
            uint32_t lookup = d->OriginalFirstThunk ? d->OriginalFirstThunk : d->FirstThunk;
            bind_thunks(dll, lookup, d->FirstThunk);
        }
    }
    // Delay imports are bound eagerly: simpler, and the full import surface shows up in --dry-run.
    const auto ddir = directory(nt, IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT);
    if (ddir.Size) {
        for (uint64_t i = 0;; ++i) {
            uint64_t rva = ddir.VirtualAddress + i * sizeof(IMAGE_DELAYLOAD_DESCRIPTOR);
            require_range(m.name, rva, sizeof(IMAGE_DELAYLOAD_DESCRIPTOR), m.size, "the delay-import table");
            auto* d = at<IMAGE_DELAYLOAD_DESCRIPTOR>(m.base, static_cast<uint32_t>(rva));
            if (!d->DllNameRVA) break;
            // x64 descriptors always carry dlattrRva (0x1). Without it the fields would be 32-bit virtual
            // addresses, which cannot express a 64-bit image base, so the descriptor is not usable as written.
            if (!d->Attributes.RvaBased)
                throw std::runtime_error(m.name + ": a delay-import descriptor without the RVA attribute");
            if (!d->ImportAddressTableRVA)
                throw std::runtime_error(m.name + ": a delay-import descriptor has no address table");
            std::string dll = read_string(m.name, m.base, m.size, d->DllNameRVA, "a delay-imported DLL name");
            // Without a name table the IAT still holds delay thunks, not import names, so there is no symbol to bind
            // eagerly. Every MSVC delay descriptor and every System32 DLL has one; reject the form instead of reading
            // thunk addresses as name RVAs.
            if (!d->ImportNameTableRVA)
                throw std::runtime_error(m.name + ": a delay-import descriptor has no name table");
            bind_thunks(dll, d->ImportNameTableRVA, d->ImportAddressTableRVA);
        }
    }
}

void Loader::protect_sections(Module& m) {
    auto* nt = at<IMAGE_NT_HEADERS64>(m.base, reinterpret_cast<IMAGE_DOS_HEADER*>(m.base)->e_lfanew);
    DWORD old;
    VirtualProtect(m.base, nt->OptionalHeader.SizeOfHeaders, PAGE_READONLY, &old);
    auto* sec = IMAGE_FIRST_SECTION(nt);
    for (unsigned i = 0; i < nt->FileHeader.NumberOfSections; ++i, ++sec) {
        bool x = sec->Characteristics & IMAGE_SCN_MEM_EXECUTE, w = sec->Characteristics & IMAGE_SCN_MEM_WRITE;
        DWORD prot = x ? (w ? PAGE_EXECUTE_READWRITE : PAGE_EXECUTE_READ) : (w ? PAGE_READWRITE : PAGE_READONLY);
        uint64_t size = sec->Misc.VirtualSize ? sec->Misc.VirtualSize : sec->SizeOfRawData;
        if (sec->VirtualAddress >= m.size) continue;
        size = std::min<uint64_t>(size, m.size - sec->VirtualAddress);
        if (size) VirtualProtect(m.base + sec->VirtualAddress, static_cast<SIZE_T>(size), prot, &old);
    }
    FlushInstructionCache(GetCurrentProcess(), m.base, m.size);
}

}  // namespace rt
