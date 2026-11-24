// Native PE mapper for Xbox Series (x86-64) images.
// Maps an image, applies relocations, binds every import through the shim registry, registers its unwind table and
// installs static TLS for the calling thread.
// Mapping never runs code from the image. Loader::initialise() delivers process attach to TLS callbacks and DllMain;
// ~Loader() delivers process detach before unmapping initialised modules. The TLS slot and template are set up by load().
// gta6rt_tls_host.dll must sit next to the executable; the loader reuses the module TLS index the OS assigns it.
#pragma once
#include <cstdint>
#include <filesystem>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <vector>

namespace rt {

enum class Binding { Shim, Host, Module, Unimplemented };

// No module TLS index has been assigned (matches TLS_OUT_OF_INDEXES without pulling <windows.h> into this header).
inline constexpr uint32_t kNoTlsIndex = 0xFFFFFFFFu;

struct ImportRecord {
    std::string dll;
    std::string name;   // "#123" for ordinals
    Binding binding;
};

struct Module {
    std::string name;        // UTF-8 basename for diagnostics and traces, not a lookup key
    std::string narrow_key;  // legacy lower-case narrow basename; empty when it cannot round-trip
    std::filesystem::path path;
    uint8_t* base = nullptr;
    uint64_t preferred_base = 0;
    uint32_t size = 0;
    uint32_t entry_rva = 0;
    bool relocated = false;
    bool initialised = false;

    // The image's exception directory (.pdata), registered with RtlAddFunctionTable by load() so that exception
    // dispatch and stack walks can unwind through its frames. Removed when the loader is destroyed.
    void* function_table = nullptr;
    uint32_t function_count = 0;

    // Mapped for a run-time LoadLibrary or a GetProcAddress forwarder (a dynamic load), not by a host load() (the
    // start-up graph, a static load). DllMain's DLL_PROCESS_ATTACH gets lpvReserved NULL only for a dynamic load.
    bool dynamic_load = false;

    std::vector<ImportRecord> imports;
    std::map<std::string, void*> exports;           // by name, plus "#ordinal"
    std::map<std::string, std::string> forwarders;  // same keys, holding "module.symbol" for re-exports

    // Static TLS (IMAGE_DIRECTORY_ENTRY_TLS). load() installs tls_block in the calling thread and writes tls_index to
    // the image's _tls_index. tls_index names TEB->ThreadLocalStoragePointer[tls_index], the array
    // __declspec(thread) reads; it is not a TlsAlloc slot. The loader reserves distinct indices by loading
    // gta6rt_tls_host.dll (and cloning it on demand for additional simultaneous TLS modules, up to 16 at a time).
    // Every thread started after load() gets its own copy through the helper's DLL_THREAD_ATTACH, and gives it back
    // at DLL_THREAD_DETACH. A thread that was already running gets its copy from load() itself, as it would from
    // Windows for a DLL Windows loads: a block, no DLL_THREAD_ATTACH, and DLL_THREAD_DETACH when it ends.
    uint32_t tls_index = kNoTlsIndex;
    uint32_t tls_slot = kNoTlsIndex;   // index into the loader's reserved TLS-host slot pool
    uint64_t tls_generation = 0;       // unique token per TLS initialisation, distinguishing reused slots/addresses
    void* tls_block = nullptr;         // calling thread's copy of the image's template
    void* tls_saved_block = nullptr;   // helper DLL block replaced in that thread, restored by ~Loader
    uint32_t tls_block_size = 0;       // raw template + zero fill
    const uint8_t* tls_template = nullptr;  // raw template in the mapped image
    uint32_t tls_raw_size = 0;
    uint32_t tls_zero_fill = 0;
};

class Loader {
public:
    // Also registers the module list with the shim registry, so an unimplemented import called under
    // LEONIDA_UNIMPLEMENTED=stop can name the module and RVA of its caller.
    explicit Loader(std::filesystem::path game_dir);
    Loader(const Loader&) = delete;
    Loader& operator=(const Loader&) = delete;
    // Delivers process detach to initialised modules in reverse dependency order before freeing any mappings or TLS.
    // Restores the TLS slot for the calling thread. Destroy the Loader on the thread that called load() while any
    // mapped TLS module is still live; all other threads executing mapped code must have finished first.
    ~Loader();

    // Loads an image and (recursively) the bundled modules it imports. Throws std::runtime_error on failure.
    // Static TLS is installed here, before any image code can run; initialise() does not install it again.
    Module& load(const std::filesystem::path& image);

    const std::vector<Module*>& modules() const { return order_; }

    // The mapped module `address` lies in, with its RVA there. Safe from any thread at any time, also while another
    // thread is in load(): it is what the shim registry asks for a caller's module (rt::shims::mapped_image_base on
    // every C++ throw from a mapped image, and the trace and stop lines), and it reads its own table, not modules().
    bool find_address(uint64_t address, std::string& module, uint32_t& rva) const;

    // The order initialise() runs modules in: each module after every module it imports (a cycle is broken at the
    // module that closed it). modules() is mapping order, which puts an importer before its imports.
    const std::vector<Module*>& initialisation_order() const { return init_order_; }

    // Runs each loaded module's TLS callbacks and then its DllMain(DLL_PROCESS_ATTACH), dependencies first.
    // Only DLLs get DllMain called: an EXE's entry point is the caller's to run. Throws on a corrupt TLS
    // directory, a TLS callback outside its image, or a DllMain that returns FALSE.
    // TLS was already installed by load(), so this is safe to call on the loading thread without a second install.
    void initialise();

    // Calls the image's IMAGE_TLS_DIRECTORY64 callbacks in order with (base, reason, nullptr). This is opt-in:
    // load() never runs image code; initialise() and ~Loader() use it for process attach/detach. No-op for
    // images without TLS callbacks. Once initialise() has run a module, the loader delivers DLL_THREAD_ATTACH and
    // DLL_THREAD_DETACH itself on threads started after load(), until ~Loader() starts the process detach; for a
    // thread that already existed, install its TLS first, then call this with DLL_THREAD_ATTACH/DETACH, and
    // uninstall after the detach callbacks.
    void run_tls_callbacks(Module& m, uint32_t reason);

    // Installs/removes this module's template copy in the calling thread's TEB slot. load() does this for its caller
    // and for every other thread running at the time, and the thread events for threads started after it, so on any
    // of those both are no-ops. They remain for a thread load() could not reach (pe_loader.cpp, "Threads that
    // existed before load()", lists the cases): install before mapped code runs there, uninstall before the thread
    // exits. Each module's copy sits in its own TEB slot, so one thread can hold copies of several modules. These do
    // not run TLS callbacks (use run_tls_callbacks with DLL_THREAD_ATTACH/DETACH for that).
    void install_tls_for_thread(Module& m);
    void uninstall_tls_for_thread(Module& m);

    // Run-time module lookup/loading used by the library_loader shim for bundled package DLLs.
    void* find_loaded_module(const std::string& dll_lower) const;
    void* load_runtime_module(const std::string& dll_lower, uint32_t flags, bool& handled);
    // The same by the module's native basename (any Unicode name, matched case-insensitively as Windows matches
    // module names), for the W entries of LoadLibrary / GetModuleHandle.
    void* find_loaded_module_w(const std::wstring& basename) const;
    void* load_runtime_module_w(const std::wstring& basename, uint32_t flags, bool& handled);
    void* resolve_module_export(void* module_handle, const std::string& sym, bool& owns_handle);
    bool owns_module_handle(void* module_handle) const;

private:
    void initialise(Module& m);
    static void set_initialised(Module& m, bool value);
    Module* find_narrow_module(const std::string& dll_lower) const;  // caller holds mutex_
    Module* find_wide_module(const std::wstring& basename) const;    // caller holds mutex_
    void* load_runtime_candidate(Module* existing, const std::filesystem::path& candidate, uint32_t flags,
                                 const std::string& display);        // caller holds mutex_
    Module& map_image(const std::filesystem::path& path, const std::string& narrow_key);
    void initialize_tls(Module& m);
    void bind_imports(Module& m);
    void* resolve(Module& m, const std::string& dll, const std::string& name, uint16_t ordinal, bool by_ordinal);
    void* resolve_forwarder(const std::string& spec, Binding& binding, int depth);
    void parse_exports(Module& m);
    void protect_sections(Module& m);
    void register_unwind(Module& m);

    mutable std::recursive_mutex mutex_;
    std::filesystem::path game_dir_;
    std::map<std::filesystem::path::string_type, Module*> by_name_;  // native basename, ASCII case folded
    std::vector<Module*> order_;       // mapping order; owns the modules
    std::vector<Module*> init_order_;  // dependencies first, see initialisation_order()
    int runtime_loads_ = 0;  // > 0 while a run-time load or forwarder maps modules (under mutex_): they are dynamic

    // What find_address reads: one entry per mapped image, added once its memory exists. Under ranges_mutex_, a leaf
    // lock: mutex_ is held while image code runs and the registry holds its own lock around the lookup, so the
    // lookup must not wait for either.
    struct MappedRange {
        uint64_t base = 0;
        uint32_t size = 0;
        std::string name;
    };
    mutable std::shared_mutex ranges_mutex_;  // shared in find_address: lookups on different threads do not queue
    std::vector<MappedRange> ranges_;
};

// Frees the TLS helper clones load() created and deletes their files (Module::tls_index background). load()
// registers this with atexit; a caller that ends with ExitProcess must invoke it directly, because ExitProcess
// skips the atexit table. Call it when no mapped code can run again: it unmaps the helper images.
void cleanup_tls_clones();

std::string lower(std::string s);

}  // namespace rt
