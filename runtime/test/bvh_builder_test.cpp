// Optional Windows test: map the package's XboxBvhBuilder.dll through rt::Loader, bring it up (Loader::initialise:
// TLS and DllMain), and construct an XboxBvhBuilder on a PC D3D12 device behind the Xbox-ABI device proxy.
// Needs the proxy's slot-62 shape (desc, w2, w3, riid, out). Results: docs/d3d12_vulkan.md, Appendix A.2.
// It records what the constructor does: how much of the object it writes, which device slots it calls, what it
// prints through OutputDebugString, and whether it gives up through TerminateProcess.
//
//   LEONIDA_GAME_DIR=<package root>   (contains XboxBvhBuilder.dll)
//   LEONIDA_BVH_FLAG=1                (optional: pass true as the constructor's bool argument)
//   LEONIDA_BVH_WARP=1                (optional: use the WARP adapter; the default on an NVIDIA adapter, RT-54)
//   LEONIDA_BVH_HARDWARE=1            (optional: keep an NVIDIA adapter instead of switching to WARP)
//   LEONIDA_BVH_RECORD62=1            (optional: print each CreateComputePipelineState's container shape, its ILDN
//                                      module name and digest, the call site (stack frames as offsets from the DLL's
//                                      image base with the enclosing function's start) and the name the builder
//                                      gives the returned PSO through SetName / SetPrivateData, RT-266)
//   LEONIDA_BVH_DXIL=1                (optional, with RECORD62: dx.op counts and validator error kinds, needs
//                                      dxcompiler.dll and dxil.dll on PATH)
//   LEONIDA_BVH_DEVICE=vulkan         (optional: the Vulkan-backed ID3D12Device from #119 instead of a PC device)
//
// Without LEONIDA_GAME_DIR it prints "SKIP (no package)" and exits 77 (CTest skip). Nothing from the DLL is written to disk.
//
// The exports are C++ members (demangled): XboxBvhBuilder::XboxBvhBuilder(ID3D12Device11*, bool) and ~XboxBvhBuilder().
// sizeof(XboxBvhBuilder) is not exported, so the object lives in a large buffer filled with a sentinel, and the
// highest byte the constructor changed is a lower bound for the size. OutputDebugStringA/W and TerminateProcess are
// redirected in this DLL's IAT only. Calls into the DLL run under SEH.
#include <windows.h>

#include <d3d12.h>
#include <dxcapi.h>
#include <dxgi1_6.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <set>
#include <string>
#include <vector>

#include "../shims/d3d12_xs_device.h"
#include "../shims/shims.h"
#include "../src/pe_loader.h"
#ifdef LEONIDA_VULKAN_DEVICE
#include "../shims/d3d12_com.h"
#endif

namespace {

int g_failures = 0;
uint64_t g_debug_strings = 0;
uint64_t g_terminate_calls = 0;
constexpr DWORD kTerminateCalled = 0xE0B71E00;  // raised instead of terminating, caught by the SEH guard

void check(bool condition, const std::string& what) {
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what.c_str());
    if (!condition) ++g_failures;
}

void note(const std::string& what) { std::printf("INFO %s\n", what.c_str()); }

std::string env(const char* name) {
    const char* v = std::getenv(name);
    return v ? v : "";
}

std::string hex(uint64_t v) {
    char b[32];
    std::snprintf(b, sizeof(b), "0x%llx", static_cast<unsigned long long>(v));
    return b;
}

void WINAPI debug_a(LPCSTR s) {
    ++g_debug_strings;
    std::printf("  dbg: %s%s", s ? s : "(null)", s && *s && s[std::strlen(s) - 1] == '\n' ? "" : "\n");
}
void WINAPI debug_w(LPCWSTR s) {
    ++g_debug_strings;
    std::printf("  dbg: %ls%s", s ? s : L"(null)", s && *s && s[wcslen(s) - 1] == L'\n' ? "" : "\n");
}
BOOL WINAPI terminate_stub(HANDLE, UINT exit_code) {
    ++g_terminate_calls;
    ULONG_PTR arg = exit_code;
    RaiseException(kTerminateCalled, EXCEPTION_NONCONTINUABLE, 1, &arg);
    return FALSE;
}

void** iat_slot(rt::Module& m, const char* want_dll, const char* want_name) {
    auto* dos = reinterpret_cast<IMAGE_DOS_HEADER*>(m.base);
    auto* nt = reinterpret_cast<IMAGE_NT_HEADERS64*>(m.base + dos->e_lfanew);
    auto& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    for (auto* d = reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR*>(m.base + dir.VirtualAddress); d->Name; ++d) {
        if (rt::lower(reinterpret_cast<const char*>(m.base + d->Name)) != want_dll) continue;
        auto* lookup = reinterpret_cast<IMAGE_THUNK_DATA64*>(m.base + (d->OriginalFirstThunk ? d->OriginalFirstThunk : d->FirstThunk));
        auto* iat = reinterpret_cast<IMAGE_THUNK_DATA64*>(m.base + d->FirstThunk);
        for (; lookup->u1.AddressOfData; ++lookup, ++iat) {
            if (IMAGE_SNAP_BY_ORDINAL64(lookup->u1.Ordinal)) continue;
            auto* n = reinterpret_cast<IMAGE_IMPORT_BY_NAME*>(m.base + static_cast<uint32_t>(lookup->u1.AddressOfData));
            if (std::string(want_name) == n->Name) return reinterpret_cast<void**>(&iat->u1.Function);
        }
    }
    return nullptr;
}

bool patch(void** slot, void* fn) {
    DWORD prot = 0;
    if (!slot || !VirtualProtect(slot, sizeof(void*), PAGE_READWRITE, &prot)) return false;
    *slot = fn;
    VirtualProtect(slot, sizeof(void*), prot, &prot);
    return true;
}

struct Call {
    bool faulted = false;
    DWORD code = 0;
    ULONG_PTR arg = 0;
    char cxx_type[128] = {};  // first catchable type of a C++ exception, e.g. ".?AVruntime_error@std@@"
    char cxx_what[256] = {};  // what() when that type is a std::exception
};

// MSVC x64 C++ exceptions (0xE06D7363): ExceptionInformation = {magic, object, ThrowInfo*, image base}; the
// ThrowInfo / CatchableTypeArray / CatchableType / TypeDescriptor chain uses 32-bit RVAs from that base.
void decode_cxx(const EXCEPTION_RECORD* r, Call* c) {
    if (r->ExceptionCode != 0xE06D7363 || r->NumberParameters < 4) return;
    auto* object = reinterpret_cast<void*>(r->ExceptionInformation[1]);
    auto* info = reinterpret_cast<const uint8_t*>(r->ExceptionInformation[2]);
    auto* base = reinterpret_cast<const uint8_t*>(r->ExceptionInformation[3]);
    if (!info || !base) return;
    const int32_t types_rva = *reinterpret_cast<const int32_t*>(info + 12);  // pCatchableTypeArray
    const uint8_t* types = base + types_rva;
    const int32_t count = *reinterpret_cast<const int32_t*>(types);
    bool is_std_exception = false;
    for (int32_t i = 0; i < count; ++i) {
        const uint8_t* ct = base + *reinterpret_cast<const int32_t*>(types + 4 + 4 * i);
        const char* name = reinterpret_cast<const char*>(base + *reinterpret_cast<const int32_t*>(ct + 4) + 16);
        if (i == 0) strncpy_s(c->cxx_type, name, _TRUNCATE);
        if (std::strcmp(name, ".?AVexception@std@@") == 0) is_std_exception = true;
    }
    if (std::strcmp(c->cxx_type, ".J") == 0 && object)  // `long`: an HRESULT thrown by value
        std::snprintf(c->cxx_what, sizeof(c->cxx_what), "long (HRESULT) 0x%08lx",
                      static_cast<unsigned long>(*static_cast<const long*>(object)));
    if (is_std_exception && object) {
        const char* what = static_cast<std::exception*>(object)->what();
        if (what) strncpy_s(c->cxx_what, what, _TRUNCATE);
    }
}

int filter(EXCEPTION_POINTERS* e, Call* c) {
    c->code = e->ExceptionRecord->ExceptionCode;
    if (e->ExceptionRecord->NumberParameters) c->arg = e->ExceptionRecord->ExceptionInformation[0];
    __try {
        decode_cxx(e->ExceptionRecord, c);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
    }
    return EXCEPTION_EXECUTE_HANDLER;
}

using Ctor = void* (*)(void* self, void* device, bool flag);
using Dtor = void (*)(void* self);

// Each guard returns a copy of `c`, not `c` itself: with `return c;` the local becomes the named return value, and
// clang-cl 19 and 20 crash (exit 139, in recoverAddrOfEscapedLocal) generating the __except filter that writes it,
// which stops the Linux cross-build (runtime/cmake/clang-cl-xwin.cmake). MSVC is fine either way.
Call guarded_ctor(Ctor fn, void* self, void* device, bool flag) {
    Call c;
    __try {
        fn(self, device, flag);
    } __except (filter(GetExceptionInformation(), &c)) {
        c.faulted = true;
    }
    return Call(c);
}

Call guarded_dtor(Dtor fn, void* self) {
    Call c;
    __try {
        fn(self);
    } __except (filter(GetExceptionInformation(), &c)) {
        c.faulted = true;
    }
    return Call(c);
}

// main's loader (#95) maps, sets up TLS and runs DllMain, but doesn't register the image's x64 unwind tables yet
// (#176 adds that). Both DLLs throw and catch C++ exceptions inside, which ends the process without them. Register
// .pdata here only when the loader hasn't, so the test keeps working once the loader does.
bool ensure_unwind_registered(rt::Module& m) {
    DWORD64 base = 0;
    if (RtlLookupFunctionEntry(reinterpret_cast<DWORD64>(m.base + m.entry_rva), &base, nullptr)) return true;
    auto* nt = reinterpret_cast<IMAGE_NT_HEADERS64*>(m.base + reinterpret_cast<IMAGE_DOS_HEADER*>(m.base)->e_lfanew);
    const auto& ex = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXCEPTION];
    return ex.Size && RtlAddFunctionTable(reinterpret_cast<PRUNTIME_FUNCTION>(m.base + ex.VirtualAddress),
                                          ex.Size / sizeof(RUNTIME_FUNCTION), reinterpret_cast<DWORD64>(m.base));
}

// Loader::initialise() runs the module's TLS setup and DllMain (#95). SEH faults inside the DLL are reported here;
// C++ exceptions (a DllMain returning FALSE, a corrupt TLS directory) go on to the caller's catch.
DWORD guarded_initialise(rt::Loader* loader) {
    DWORD code = 0;
    __try {
        loader->initialise();
    } __except ((code = GetExceptionCode()) == 0xE06D7363 ? EXCEPTION_CONTINUE_SEARCH : EXCEPTION_EXECUTE_HANDLER) {
    }
    return code == 0xE06D7363 ? 0 : code;
}

std::string describe(const Call& c) {
    if (!c.faulted) return "returned";
    if (c.code == kTerminateCalled) return "called TerminateProcess(exit code " + hex(c.arg) + ")";
    if (c.code == 0xE06D7363)
        return std::string("threw a C++ exception ") + (c.cxx_type[0] ? c.cxx_type : "(type unknown)") +
               (c.cxx_what[0] ? std::string(": \"") + c.cxx_what + "\"" : "");
    return "faulted with exception " + hex(c.code);
}

void* symbol(rt::Module& m, const char* name) {
    auto it = m.exports.find(name);
    return it == m.exports.end() ? nullptr : it->second;
}

// The D3D12 debug layer, when the Graphics Tools feature is installed, says why a PSO was rejected.
bool enable_debug_layer() {
    ID3D12Debug* debug = nullptr;
    if (FAILED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) return false;
    debug->EnableDebugLayer();
    debug->Release();
    return true;
}

void print_debug_messages(ID3D12Device* device) {
    ID3D12InfoQueue* q = nullptr;
    if (FAILED(device->QueryInterface(IID_PPV_ARGS(&q)))) return;
    const UINT64 n = q->GetNumStoredMessages();
    for (UINT64 i = 0; i < n && i < 32; ++i) {
        SIZE_T len = 0;
        q->GetMessage(i, nullptr, &len);
        std::vector<uint8_t> buf(len);
        auto* m = reinterpret_cast<D3D12_MESSAGE*>(buf.data());
        if (SUCCEEDED(q->GetMessage(i, m, &len))) std::printf("  d3d12 debug [%d]: %s\n", m->ID, m->pDescription);
    }
    q->ClearStoredMessages();
    q->Release();
}

// The first hardware adapter's PCI vendor ID, 0 when there is none.
UINT first_hardware_vendor() {
    IDXGIFactory4* factory = nullptr;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return 0;
    UINT vendor = 0;
    IDXGIAdapter1* adapter = nullptr;
    for (UINT i = 0; !vendor && factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 d{};
        adapter->GetDesc1(&d);
        if (!(d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) vendor = d.VendorId;
        adapter->Release();
    }
    factory->Release();
    return vendor;
}

ID3D12Device* create_pc_device(bool warp) {
    IDXGIFactory4* factory = nullptr;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return nullptr;
    IDXGIAdapter1* adapter = nullptr;
    if (warp) {
        factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter));
    } else {
        for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
            DXGI_ADAPTER_DESC1 d{};
            adapter->GetDesc1(&d);
            if (!(d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) break;
            adapter->Release();
            adapter = nullptr;
        }
    }
    ID3D12Device* device = nullptr;
    if (adapter) {
        DXGI_ADAPTER_DESC1 d{};
        adapter->GetDesc1(&d);
        std::printf("adapter: %ls\n", d.Description);
        D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device));
        adapter->Release();
    }
    factory->Release();
    return device;
}

void print_slot_hits(const void* proxy) {
    const unsigned n = rt::d3d12_xs::device_proxy_slot_count(proxy) + 32;
    std::string dev, heap;
    for (unsigned s = 0; s < n; ++s)
        if (auto h = rt::d3d12_xs::device_slot_hits(s)) dev += " " + std::to_string(s) + "x" + std::to_string(h);
    for (unsigned s = 0; s < 16; ++s)
        if (auto h = rt::d3d12_xs::heap_slot_hits(s)) heap += " " + std::to_string(s) + "x" + std::to_string(h);
    note("device slots hit (slot x count):" + (dev.empty() ? std::string(" none") : dev));
    note("descriptor-heap slots hit:" + (heap.empty() ? std::string(" none") : heap));
}

// Slot-62 recorder: the proxy's vtable is copied and slot 62 points here, so the raw argument words the DLL passes
// are printed before the proxy's own handler runs. Test-only; the proxy itself is unchanged.
using Slot62 = uint64_t(WINAPI*)(void*, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);
Slot62 g_slot62 = nullptr;
int g_slot62_logged = 0;

bool looks_like_iid(uint64_t p, const IID& iid) {
    __try {
        return p && std::memcmp(reinterpret_cast<const void*>(p), &iid, sizeof(IID)) == 0;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
}

uint64_t g_slot62_ok = 0, g_slot62_failed = 0;

// The shape of a DXBC container, for the record: its parts (four-character code and size), the DXIL part's shader
// model and DXIL version, the SFI0 feature flags, whether the hash is zero (unsigned), the ILDN module name and the
// first four bytes of the digest. Counts, flags and names only.
std::string container_shape(const void* bytecode, size_t length) {
    const auto* b = static_cast<const uint8_t*>(bytecode);
    auto u32 = [&](size_t o) {
        uint32_t v = 0;
        if (o + 4 <= length) std::memcpy(&v, b + o, 4);
        return v;
    };
    if (!b || length < 32 || std::memcmp(b, "DXBC", 4) != 0) return "not a DXBC container";
    bool zero_hash = true;
    for (int i = 4; i < 20; ++i) zero_hash &= b[i] == 0;
    const uint32_t parts = u32(28);
    std::string out = std::string(zero_hash ? "unsigned" : "signed") + ", " + std::to_string(parts) + " parts:";
    for (uint32_t i = 0; i < parts && 32 + 4 * size_t(i) + 4 <= length; ++i) {
        const uint32_t off = u32(32 + 4 * size_t(i));
        if (size_t(off) + 8 > length) break;
        const char fourcc[5] = {char(b[off]), char(b[off + 1]), char(b[off + 2]), char(b[off + 3]), 0};
        const uint32_t size = u32(off + 4);
        char item[64];
        std::snprintf(item, sizeof(item), " %s:%u", fourcc, size);
        out += item;
        const size_t data = size_t(off) + 8;
        if ((!std::strcmp(fourcc, "DXIL") || !std::strcmp(fourcc, "ILDB")) && size >= 24) {
            const uint32_t version = u32(data), dxil = u32(data + 12);
            std::snprintf(item, sizeof(item), "(sm %u.%u kind %u dxil %u.%u)", (version >> 4) & 0xf, version & 0xf,
                          version >> 16, (dxil >> 8) & 0xff, dxil & 0xff);
            out += item;
        } else if (!std::strcmp(fourcc, "SFI0") && size >= 8) {
            std::snprintf(item, sizeof(item), "(flags %#llx)",
                          static_cast<unsigned long long>(u32(data) | (uint64_t(u32(data + 4)) << 32)));
            out += item;
        } else if (!std::strcmp(fourcc, "ILDN") && size >= 4) {
            // ILDN: u16 flags, u16 name length, the debug name (NUL-padded): the module's PDB path, which names the
            // container (docs/embedded_shaders.md section 3). Only the file name is printed.
            const uint32_t name_length = u32(data) >> 16;
            if (size_t(data) + 4 + name_length <= length && name_length > 0) {
                std::string name(reinterpret_cast<const char*>(b + data + 4), name_length);
                name.erase(name.find_last_not_of('\0') + 1);
                name = name.substr(name.find_last_of("\\/") + 1);
                out += "(" + name + ")";
            }
        }
    }
    // The first four bytes of the container digest (bytes 4..19 of the header), to tell two creations of the same
    // container from two containers with the same module name.
    char digest[24];
    std::snprintf(digest, sizeof(digest), " digest %02x%02x%02x%02x", b[4], b[5], b[6], b[7]);
    return out + digest;
}

// The module a container belongs to: the file name (without ".pdb") in its ILDN part, a DXIL debug-name record
// (u16 flags, u16 name length, then the PDB path). Empty when there is no such part. The module names are the ones
// listed in docs/embedded_shaders.md section 3.
std::string container_module(const void* bytecode, size_t length) {
    const auto* b = static_cast<const uint8_t*>(bytecode);
    auto u32 = [&](size_t o) {
        uint32_t v = 0;
        if (o + 4 <= length) std::memcpy(&v, b + o, 4);
        return v;
    };
    if (!b || length < 32 || std::memcmp(b, "DXBC", 4) != 0) return {};
    const uint32_t parts = u32(28);
    for (uint32_t i = 0; i < parts && 32 + 4 * size_t(i) + 4 <= length; ++i) {
        const size_t off = u32(32 + 4 * size_t(i));
        if (off + 8 > length || std::memcmp(b + off, "ILDN", 4) != 0) continue;
        const size_t size = u32(off + 4), data = off + 8;
        if (size < 5 || data + size > length) return {};
        std::string path(reinterpret_cast<const char*>(b + data + 4), size - 4);
        path.resize(std::strlen(path.c_str()));
        const size_t slash = path.find_last_of("\\/");
        std::string name = slash == std::string::npos ? path : path.substr(slash + 1);
        if (name.size() > 4 && name.compare(name.size() - 4, 4, ".pdb") == 0) name.resize(name.size() - 4);
        return name;
    }
    return {};
}

// Every container the constructor has passed to slot 62 so far, to find one that is passed twice (RT-182). The bytes
// are compared, not a hash, and nothing is printed from them.
std::vector<std::vector<uint8_t>> g_seen_containers;
std::vector<std::string> g_seen_modules;
int g_repeat_first = 0, g_repeat_second = 0;  // creation numbers (1-based) of the first repeat found, 0 = none
int g_repeats = 0;

// LEONIDA_BVH_DXIL=1: what each shader uses, from dxcompiler.dll's disassembly in memory: the distinct dx.op names
// with call counts, and a few type/feature markers. Also dxil.dll's validator verdict. Names and counts only; the
// disassembly is not printed or written. The DLLs come from PATH or the Windows SDK, loaded at run time.
struct DxcTools {
    IDxcCompiler* compiler = nullptr;
    IDxcLibrary* library = nullptr;
    IDxcValidator* validator = nullptr;
};

DxcTools* dxc_tools() {
    static DxcTools tools;
    static bool tried = false;
    if (tried) return tools.compiler ? &tools : nullptr;
    tried = true;
    HMODULE compiler = LoadLibraryA("dxcompiler.dll");
    HMODULE validator = LoadLibraryA("dxil.dll");
    auto create = [](HMODULE m) {
        return m ? reinterpret_cast<DxcCreateInstanceProc>(GetProcAddress(m, "DxcCreateInstance")) : nullptr;
    };
    if (auto f = create(compiler)) {
        f(CLSID_DxcCompiler, IID_PPV_ARGS(&tools.compiler));
        f(CLSID_DxcLibrary, IID_PPV_ARGS(&tools.library));
    }
    if (auto f = create(validator)) f(CLSID_DxcValidator, IID_PPV_ARGS(&tools.validator));
    if (!tools.compiler || !tools.library) note("LEONIDA_BVH_DXIL: dxcompiler.dll not found on PATH");
    if (!tools.validator) note("LEONIDA_BVH_DXIL: dxil.dll not found on PATH (no validator verdict)");
    return tools.compiler && tools.library ? &tools : nullptr;
}

std::string dxil_usage(const void* bytecode, size_t length) {
    DxcTools* t = dxc_tools();
    if (!t) return "no dxcompiler";
    const std::vector<uint8_t> copy(static_cast<const uint8_t*>(bytecode), static_cast<const uint8_t*>(bytecode) + length);
    IDxcBlobEncoding* blob = nullptr;
    if (FAILED(t->library->CreateBlobWithEncodingFromPinned(copy.data(), static_cast<UINT32>(copy.size()), 0, &blob)))
        return "no blob";
    std::string out;
    if (t->validator) {
        IDxcOperationResult* result = nullptr;
        HRESULT status = E_FAIL;
        if (SUCCEEDED(t->validator->Validate(blob, DxcValidatorFlags_Default, &result)) && result)
            result->GetStatus(&status);
        out += std::string("validator ") + (SUCCEEDED(status) ? "ok" : "rejects (" + hex(uint32_t(status)) + ")");
        // The kinds of error with their counts. Quoted text (names, triples, instructions) is masked as 'X', and
        // anything after the message's first sentence is dropped, so nothing of the shader is printed.
        IDxcBlobEncoding* errors = nullptr;
        if (result && FAILED(status) && SUCCEEDED(result->GetErrorBuffer(&errors)) && errors) {
            const std::string m(static_cast<const char*>(errors->GetBufferPointer()), errors->GetBufferSize());
            std::vector<std::pair<std::string, int>> kinds;
            for (size_t p = m.find("error: "); p != std::string::npos; p = m.find("error: ", p + 1)) {
                std::string kind;
                bool quoted = false;
                for (size_t i = p + 7; i < m.size(); ++i) {
                    const char ch = m[i];
                    if (ch == '\n' || (ch == '.' && !quoted)) break;  // the first sentence; dots in names don't end it
                    if (ch == '\'') {
                        if (!quoted) kind += "'X'";
                        quoted = !quoted;
                    } else if (!quoted) {
                        kind += ch;
                    }
                }
                auto it = std::find_if(kinds.begin(), kinds.end(), [&](auto& k) { return k.first == kind; });
                if (it == kinds.end()) kinds.push_back({kind, 1}); else ++it->second;
            }
            for (auto& [kind, n] : kinds) out += "; " + kind + " x" + std::to_string(n);
            errors->Release();
        }
        if (result) result->Release();
    }
    IDxcBlobEncoding* text = nullptr;
    if (SUCCEEDED(t->compiler->Disassemble(blob, &text)) && text) {
        const std::string s(static_cast<const char*>(text->GetBufferPointer()), text->GetBufferSize());
        std::vector<std::pair<std::string, int>> ops;
        for (size_t p = s.find("@dx.op."); p != std::string::npos; p = s.find("@dx.op.", p + 1)) {
            const size_t line = s.rfind('\n', p);
            const size_t call = s.find(" call ", line == std::string::npos ? 0 : line);
            if (call == std::string::npos || call > p) continue;  // a declaration, not a call
            const size_t end = s.find_first_of(".(", p + 7);
            const std::string name = s.substr(p + 7, end - p - 7);
            auto it = std::find_if(ops.begin(), ops.end(), [&](auto& o) { return o.first == name; });
            if (it == ops.end()) ops.push_back({name, 1}); else ++it->second;
        }
        auto count = [&](const char* needle) {
            int n = 0;
            for (size_t p = s.find(needle); p != std::string::npos; p = s.find(needle, p + 1)) ++n;
            return n;
        };
        out += ", " + std::to_string(s.size() / 1024) + " KiB of text, lines " + std::to_string(count("\n"));
        out += ", i64 " + std::to_string(count(" i64 ")) + ", half " + std::to_string(count(" half ")) +
               ", double " + std::to_string(count(" double ")) + ", phi " + std::to_string(count(" phi ")) +
               ", switch " + std::to_string(count(" switch ")) + ", br " + std::to_string(count(" br ")) +
               ", alloca " + std::to_string(count(" alloca ")) + ", groupshared " + std::to_string(count("addrspace(3)")) +
               ", undef " + std::to_string(count(" undef")) + "; ops:";
        for (auto& [name, n] : ops) out += " " + name + "x" + std::to_string(n);
        text->Release();
    }
    blob->Release();
    return out;
}

// RT-266: where each creation comes from and which name the builder attaches to the PSO it gets back. The call
// site is the stack of the hook as offsets from the DLL's image base, each with the start of its function from the
// x64 unwind tables (RtlLookupFunctionEntry); the name is whatever the builder passes to the returned object's
// SetName (ID3D12Object slot 6) or SetPrivateData (slot 4) with the D3D debug-name GUIDs. The returned object gets a
// copy of its vtable with those two slots redirected (the creation number is looked up by object pointer).
uint64_t g_bvh_base = 0, g_bvh_size = 0;

std::string module_offset(uint64_t addr) {
    if (!g_bvh_base || addr < g_bvh_base || addr >= g_bvh_base + g_bvh_size) {
        char b[40];
        std::snprintf(b, sizeof b, "outside the DLL (%#llx)", (unsigned long long)addr);
        return b;
    }
    char b[96];
    DWORD64 image = 0;
    const RUNTIME_FUNCTION* rf = RtlLookupFunctionEntry(addr, &image, nullptr);
    if (rf)
        std::snprintf(b, sizeof b, "+%#llx (function +%#lx)", (unsigned long long)(addr - g_bvh_base),
                      (unsigned long)rf->BeginAddress);
    else
        std::snprintf(b, sizeof b, "+%#llx (no unwind entry)", (unsigned long long)(addr - g_bvh_base));
    return b;
}

void print_call_site(int number) {
    // Unwound by hand with the x64 tables: RtlCaptureStackBackTrace stops after the first DLL frame here.
    CONTEXT ctx;
    RtlCaptureContext(&ctx);
    unsigned printed = 0;
    for (int step = 0; step < 12 && printed < 6; ++step) {
        DWORD64 image = 0;
        const RUNTIME_FUNCTION* rf = RtlLookupFunctionEntry(ctx.Rip, &image, nullptr);
        if (!rf) break;
        void* handler_data = nullptr;
        DWORD64 frame = 0;
        RtlVirtualUnwind(UNW_FLAG_NHANDLER, image, ctx.Rip, const_cast<RUNTIME_FUNCTION*>(rf), &ctx, &handler_data,
                         &frame, nullptr);
        if (!ctx.Rip) break;
        const auto addr = static_cast<uint64_t>(ctx.Rip);
        if (addr < g_bvh_base || addr >= g_bvh_base + g_bvh_size) continue;  // this test's own frames
        std::printf("  slot62 #%d call site frame %u: %s\n", number, printed++, module_offset(addr).c_str());
    }
}

struct PsoName {
    int number;
    void* object;
    void** original_vtable;
    bool restored;
    void** vtable_copy;  // the hooked copy installed on the object; freed once the original is back
};
std::vector<PsoName> g_pso_names;
std::set<int> g_named;  // creation numbers that got a name
thread_local bool g_in_set_name = false;

// The newest live entry for the address (an address can be reused after a PSO is released).
PsoName* find_pso(void* object) {
    for (auto it = g_pso_names.rbegin(); it != g_pso_names.rend(); ++it)
        if (!it->restored && it->object == object) return &*it;
    return nullptr;
}

// Puts the object's own vtable back, so its destructor and every virtual call past the 9 copied slots see the
// original. Done as soon as the name is logged, and for the rest after the constructor returns.
void restore_pso_vtable(PsoName& p) {
    if (p.restored) return;
    p.restored = true;
    __try {
        *static_cast<void***>(p.object) = p.original_vtable;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
    }
    // The thunks that call this have already been dispatched through the copy and read only the original vtable
    // afterwards; find_pso skips restored entries, so nothing reads the copy past this point.
    delete[] p.vtable_copy;
    p.vtable_copy = nullptr;
    // Drops the reference hook_pso_names took; the builder still holds its own here, so this cannot free the object.
    using ReleaseFn = ULONG(STDMETHODCALLTYPE*)(void*);
    reinterpret_cast<ReleaseFn>(p.original_vtable[2])(p.object);
}

void restore_all_pso_vtables() {
    for (auto& p : g_pso_names) restore_pso_vtable(p);
}

std::string narrow(const wchar_t* w) {
    std::string s;
    for (; w && *w && s.size() < 200; ++w) s += *w < 0x80 ? char(*w) : '?';
    return s;
}

constexpr GUID kDebugNameW = {0x4cca5fd8, 0x921f, 0x42c8, {0x85, 0x66, 0x70, 0xca, 0xf2, 0xa9, 0xb7, 0x41}};
constexpr GUID kDebugNameA = {0x429b8c22, 0x9188, 0x4b0c, {0x87, 0x42, 0xac, 0xb0, 0xbf, 0x85, 0xc2, 0x00}};

HRESULT STDMETHODCALLTYPE pso_set_name(void* self, const wchar_t* name) {
    if (auto* p = find_pso(self)) {
        std::printf("  slot62 #%d name via SetName: \"%s\"\n", p->number, narrow(name).c_str());
        g_named.insert(p->number);
        using Fn = HRESULT(STDMETHODCALLTYPE*)(void*, const wchar_t*);
        g_in_set_name = true;  // the runtime's SetName may call SetPrivateData through the vtable
        const HRESULT hr = reinterpret_cast<Fn>(p->original_vtable[6])(self, name);
        g_in_set_name = false;
        restore_pso_vtable(*p);
        return hr;
    }
    return E_FAIL;
}

HRESULT STDMETHODCALLTYPE pso_set_private_data(void* self, const GUID& guid, UINT size, const void* data) {
    if (auto* p = find_pso(self)) {
        if (g_in_set_name) {
            using Fn = HRESULT(STDMETHODCALLTYPE*)(void*, const GUID&, UINT, const void*);
            return reinterpret_cast<Fn>(p->original_vtable[4])(self, guid, size, data);
        }
        char g[48];
        std::snprintf(g, sizeof g, "%08lx-%04x-%04x", guid.Data1, guid.Data2, guid.Data3);
        std::string value;
        if (data && size) {
            if (!std::memcmp(&guid, &kDebugNameW, sizeof(GUID)))
                value = narrow(std::wstring(static_cast<const wchar_t*>(data), size / 2).c_str());
            else if (!std::memcmp(&guid, &kDebugNameA, sizeof(GUID)))
                value = std::string(static_cast<const char*>(data), size);
        }
        std::printf("  slot62 #%d SetPrivateData guid %s size %u%s%s%s\n", p->number, g, size,
                    value.empty() ? "" : " name \"", value.c_str(), value.empty() ? "" : "\"");
        if (!value.empty()) g_named.insert(p->number);
        using Fn = HRESULT(STDMETHODCALLTYPE*)(void*, const GUID&, UINT, const void*);
        const HRESULT hr = reinterpret_cast<Fn>(p->original_vtable[4])(self, guid, size, data);
        if (!value.empty()) restore_pso_vtable(*p);
        return hr;
    }
    return E_FAIL;
}

void hook_pso_names(uint64_t out_ptr, int number) {
    void* object = nullptr;
    __try {
        if (out_ptr) object = *reinterpret_cast<void**>(out_ptr);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        object = nullptr;
    }
    if (!object) {
        std::printf("  slot62 #%d: no object at a5\n", number);
        return;
    }
    constexpr unsigned kSlots = 9;  // IUnknown, ID3D12Object, GetDevice, GetCachedBlob
    void** orig = *static_cast<void***>(object);
    auto** copy = new void*[kSlots];
    std::memcpy(copy, orig, kSlots * sizeof(void*));
    copy[4] = reinterpret_cast<void*>(&pso_set_private_data);
    copy[6] = reinterpret_cast<void*>(&pso_set_name);
    // Hold a reference until the vtable is restored, so the later write-back never lands in freed memory.
    using AddRefFn = ULONG(STDMETHODCALLTYPE*)(void*);
    reinterpret_cast<AddRefFn>(orig[1])(object);
    g_pso_names.push_back({number, object, orig, false, copy});
    *static_cast<void***>(object) = copy;
}

uint64_t WINAPI record_slot62(void* self, uint64_t a1, uint64_t a2, uint64_t a3, uint64_t a4, uint64_t a5, uint64_t a6) {
    const int n = g_slot62_logged++;
    if (n < 1) {
        std::printf("  slot62 args: a1=%#llx a2=%#llx a3=%#llx a4=%#llx a5=%#llx a6=%#llx\n",
                    (unsigned long long)a1, (unsigned long long)a2, (unsigned long long)a3, (unsigned long long)a4,
                    (unsigned long long)a5, (unsigned long long)a6);
        const uint64_t args[] = {a1, a2, a3, a4, a5, a6};
        for (int i = 0; i < 6; ++i)
            if (looks_like_iid(args[i], __uuidof(ID3D12PipelineState)))
                std::printf("  slot62: a%d points at IID_ID3D12PipelineState\n", i + 1);
    }
    const auto* desc = reinterpret_cast<const D3D12_COMPUTE_PIPELINE_STATE_DESC*>(a1);
    // Printed before the call: a driver that rejects the shader may end the process inside it.
    if (desc) {
        std::printf("  slot62 #%d container: %s\n", n + 1,
                    container_shape(desc->CS.pShaderBytecode, desc->CS.BytecodeLength).c_str());
        const std::string module = container_module(desc->CS.pShaderBytecode, desc->CS.BytecodeLength);
        std::printf("  slot62 #%d module: %s\n", n + 1, module.empty() ? "(no ILDN part)" : module.c_str());
        const auto* bytes = static_cast<const uint8_t*>(desc->CS.pShaderBytecode);
        std::vector<uint8_t> copy(bytes, bytes + desc->CS.BytecodeLength);
        int same_as = 0;
        for (size_t i = 0; i < g_seen_containers.size(); ++i)
            if (g_seen_containers[i] == copy) {
                same_as = static_cast<int>(i) + 1;
                break;
            }
        if (same_as) {
            std::printf("  slot62 #%d is byte-identical to #%d (module %s)\n", n + 1, same_as,
                        g_seen_modules[same_as - 1].c_str());
            if (!g_repeats) {
                g_repeat_first = same_as;
                g_repeat_second = n + 1;
            }
            ++g_repeats;
        }
        g_seen_containers.push_back(std::move(copy));
        g_seen_modules.push_back(module);
        if (env("LEONIDA_BVH_DXIL") == "1")
            std::printf("  slot62 #%d uses: %s\n", n + 1, dxil_usage(desc->CS.pShaderBytecode, desc->CS.BytecodeLength).c_str());
    }
    print_call_site(n + 1);
    const uint64_t r = g_slot62(self, a1, a2, a3, a4, a5, a6);
    const auto hr = static_cast<HRESULT>(static_cast<uint32_t>(r));
    (SUCCEEDED(hr) ? g_slot62_ok : g_slot62_failed)++;
    // The output slot is a5 (desc, two words, riid, out); the object it receives is the one the builder names.
    if (SUCCEEDED(hr)) hook_pso_names(a5, n + 1);
    std::printf("  slot62 #%d: CS bytecode %llu bytes, root signature %s, node mask %u, flags %#x -> hr %#010lx\n", n + 1,
                desc ? (unsigned long long)desc->CS.BytecodeLength : 0ull, desc && desc->pRootSignature ? "set" : "null",
                desc ? desc->NodeMask : 0u, desc ? (unsigned)desc->Flags : 0u, static_cast<unsigned long>(hr));
    return r;
}

void** g_proxy_vtable_copy = nullptr;
bool hook_slot62(void* proxy, unsigned slots) {
    void** vtable = *static_cast<void***>(proxy);
    g_proxy_vtable_copy = new void*[slots];
    std::memcpy(g_proxy_vtable_copy, vtable, slots * sizeof(void*));
    g_slot62 = reinterpret_cast<Slot62>(vtable[62]);
    g_proxy_vtable_copy[62] = reinterpret_cast<void*>(&record_slot62);
    *static_cast<void***>(proxy) = g_proxy_vtable_copy;
    return true;
}

constexpr const char* kCtor = "??0XboxBvhBuilder@@QEAA@PEAUID3D12Device11@@_N@Z";
constexpr const char* kDtor = "??1XboxBvhBuilder@@QEAA@XZ";
constexpr size_t kObjectBuffer = 1u << 20;
constexpr uint8_t kSentinel = 0xCD;

// The module's TLS index, whichever loader this builds against: #95 records a slot (has_tls_slot/tls_slot), #93
// a module TLS index (tls_index, kNoTlsIndex when there is none). ~0u means no TLS.
template <class M>
auto module_tls_index(const M& m, int) -> decltype(m.tls_index, uint32_t()) { return m.tls_index; }
template <class M>
auto module_tls_index(const M& m, long) -> decltype(m.tls_slot, uint32_t()) { return m.has_tls_slot ? m.tls_slot : ~0u; }

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);  // keep every line if the DLL takes the process down
    const std::string game_dir_arg = env("LEONIDA_GAME_DIR");
    if (game_dir_arg.empty()) {
        std::printf("SKIP (no package)\n");
        return 77;
    }
    const std::filesystem::path game_dir = game_dir_arg;
    const std::filesystem::path dll = game_dir / "XboxBvhBuilder.dll";
    std::error_code ec;
    if (!std::filesystem::is_regular_file(dll, ec)) {
        std::printf("FAIL LEONIDA_GAME_DIR does not contain XboxBvhBuilder.dll\n");
        return 1;
    }

    try {
        rt::Loader loader(game_dir);
        rt::Module& bvh = loader.load(dll);
        std::printf("mapped %s at %p, size %u, entry +%#x%s\n", bvh.name.c_str(), bvh.base, bvh.size,
                    bvh.entry_rva, bvh.relocated ? " (relocated)" : "");
        g_bvh_base = reinterpret_cast<uint64_t>(bvh.base);
        g_bvh_size = bvh.size;
        int shim = 0, host = 0, unimplemented = 0;
        for (auto& r : bvh.imports) {
            if (r.binding == rt::Binding::Shim) ++shim;
            if (r.binding == rt::Binding::Host) ++host;
            if (r.binding == rt::Binding::Unimplemented) {
                ++unimplemented;
                std::printf("  UNIMPLEMENTED %s!%s\n", r.dll.c_str(), r.name.c_str());
            }
            const std::string d = rt::lower(r.dll);
            if (d == "xmem.dll" || d == "d3d12_xs.dll")
                check(r.binding == rt::Binding::Shim, r.dll + "!" + r.name + " binds to the shim");
        }
        std::printf("imports: shim=%d host=%d unimplemented=%d\n", shim, host, unimplemented);
        check(unimplemented == 0, "no unimplemented imports");

        auto ctor = reinterpret_cast<Ctor>(symbol(bvh, kCtor));
        auto dtor = reinterpret_cast<Dtor>(symbol(bvh, kDtor));
        check(ctor && dtor, "constructor and destructor are exported");
        check(patch(iat_slot(bvh, "api-ms-win-core-debug-l1-1-0.dll", "OutputDebugStringA"), reinterpret_cast<void*>(&debug_a)) &&
                  patch(iat_slot(bvh, "api-ms-win-core-debug-l1-1-0.dll", "OutputDebugStringW"), reinterpret_cast<void*>(&debug_w)) &&
                  patch(iat_slot(bvh, "api-ms-win-core-processthreads-l1-1-0.dll", "TerminateProcess"),
                        reinterpret_cast<void*>(&terminate_stub)),
              "OutputDebugStringA/W and TerminateProcess redirected in this DLL's IAT");
        if (g_failures) return 1;

        check(ensure_unwind_registered(bvh), "x64 unwind tables registered (by the loader or here)");
        const DWORD init = guarded_initialise(&loader);
        check(init == 0, "Loader::initialise (TLS, DllMain) succeeds" + (init ? " (exception " + hex(init) + ")" : ""));
        if (init) return 1;
        const uint32_t tls = module_tls_index(bvh, 0);
        std::printf("TLS: %s index %u\n", tls != ~0u ? "loader" : "no", tls);

        ID3D12Device* pc = nullptr;
        const bool on_vulkan = env("LEONIDA_BVH_DEVICE") == "vulkan";
        if (on_vulkan) {
#ifdef LEONIDA_VULKAN_DEVICE
            // The Vulkan-backed ID3D12Device (#119) instead of the host's D3D12 device, behind the same proxy.
            const HRESULT vhr = rt::d3d12::create_vulkan_device_standalone(__uuidof(ID3D12Device), reinterpret_cast<void**>(&pc));
            check(SUCCEEDED(vhr) && pc, "Vulkan-backed D3D12 device created (" + hex(static_cast<uint32_t>(vhr)) + ")");
#else
            check(false, "LEONIDA_BVH_DEVICE=vulkan needs a build with the Vulkan SDK (gta6rt_d3d12_com)");
#endif
        } else {
            const bool debug_layer = enable_debug_layer();
            note(std::string("D3D12 debug layer ") + (debug_layer ? "enabled" : "not available (Graphics Tools not installed)"));
            // RT-54: the package's shaders are Xbox DXIL (target triple dxil-ms-xb2), which no PC driver has to
            // accept. NVIDIA's DXIL reader rejects one of them (the 49th CreateComputePipelineState, the only one
            // with a resource in a function signature) and can end the process inside the driver (0xC0000409)
            // instead of returning an error, so the constructor can't be measured there. AMD faults the same way
            // (0xC0000005 right after the first slot 62 call on an RX 7900 XTX, RT-89). WARP accepts all 53, so it is
            // the default on every hardware vendor; LEONIDA_BVH_HARDWARE=1 keeps the hardware device, to check a
            // newer driver.
            bool warp = env("LEONIDA_BVH_WARP") == "1";
            if (!warp && env("LEONIDA_BVH_HARDWARE") != "1") {
                if (const UINT vendor = first_hardware_vendor()) {
                    char id[8];
                    std::snprintf(id, sizeof id, "%04x", vendor);
                    note(std::string("hardware adapter (vendor ") + id + "): using WARP, since hardware DXIL readers "
                         "reject the DLL's Xbox DXIL shaders and can end the process (RT-54 NVIDIA, RT-89 AMD); "
                         "LEONIDA_BVH_HARDWARE=1 keeps the hardware device");
                }
                warp = true;
            }
            pc = create_pc_device(warp);
            check(pc != nullptr, "PC D3D12 device created");
        }
        if (!pc) return 1;
        if (!on_vulkan) {
            // The proxy answers ID3D12Device11 with a QueryInterface on the host device. The Windows 10 runtime
            // without the Agility SDK stops before ID3D12Device11, on hardware and WARP alike (#688).
            ID3D12Device11* host11 = nullptr;
            if (FAILED(pc->QueryInterface(IID_PPV_ARGS(&host11))) || !host11) {
                std::printf("SKIP (the host D3D12 runtime has no ID3D12Device11, e.g. Windows 10 without the Agility "
                            "SDK; LEONIDA_BVH_DEVICE=vulkan runs on the Vulkan-backed device)\n");
                pc->Release();
                return g_failures ? 1 : 0;
            }
            host11->Release();
        }
        pc->AddRef();  // kept for the debug messages; the proxy takes its own reference
        void* proxy = nullptr;
        const HRESULT hr = rt::d3d12_xs::create_device_proxy(pc, __uuidof(ID3D12Device11), &proxy);
        check(SUCCEEDED(hr) && proxy, "Xbox-ABI device proxy created as ID3D12Device11 (" + hex(static_cast<uint32_t>(hr)) + ")");
        if (!proxy) return 1;
#ifdef LEONIDA_VULKAN_DEVICE
        if (on_vulkan) rt::d3d12::set_device_outer(pc, static_cast<IUnknown*>(proxy));
#endif
        rt::d3d12_xs::reset_device_proxy_stats();
        if (env("LEONIDA_BVH_RECORD62") == "1")
            check(hook_slot62(proxy, rt::d3d12_xs::device_proxy_slot_count(proxy)), "slot 62 recorder installed");

        const bool flag = env("LEONIDA_BVH_FLAG") == "1";
        std::vector<uint8_t> object(kObjectBuffer, kSentinel);
        const ULONGLONG t0 = GetTickCount64();
        Call c = guarded_ctor(ctor, object.data(), proxy, flag);
        restore_all_pso_vtables();  // PSOs never named keep the copy until here (RT-266)
        const ULONGLONG ms = GetTickCount64() - t0;
        size_t last = 0, changed = 0;
        for (size_t i = 0; i < object.size(); ++i)
            if (object[i] != kSentinel) {
                last = i + 1;
                ++changed;
            }
        note(std::string("XboxBvhBuilder(device, ") + (flag ? "true" : "false") + ") " + describe(c) + " after " +
             std::to_string(ms) + " ms");
        note("object bytes written: up to offset " + hex(last) + " (" + std::to_string(changed) +
             " bytes differ from the sentinel): sizeof(XboxBvhBuilder) >= " + std::to_string(last));
        print_slot_hits(proxy);
        print_debug_messages(pc);
        note("slot 62 (Xbox CreateComputePipelineState) calls: " + std::to_string(rt::d3d12_xs::device_slot_hits(62)) +
             (g_slot62_logged ? ", succeeded " + std::to_string(g_slot62_ok) + ", failed " + std::to_string(g_slot62_failed)
                              : std::string()));
        if (g_slot62_logged) {
            note("slot 62 creations given a name by the builder: " + std::to_string(g_named.size()) + " of " +
                 std::to_string(g_slot62_logged));
            note("slot 62 creations: " + std::to_string(g_slot62_logged) + ", distinct containers: " +
                 std::to_string(g_slot62_logged - g_repeats) + ", repeated creations: " + std::to_string(g_repeats));
            if (g_repeats) {
                note("first repeat: creation #" + std::to_string(g_repeat_second) + " passes the same container as #" +
                     std::to_string(g_repeat_first) + ", module " + g_seen_modules[g_repeat_first - 1]);
                check(g_seen_modules[g_repeat_first - 1] == g_seen_modules[g_repeat_second - 1],
                      "the repeated container names the same module both times");
            }
        }
        note("OutputDebugString calls: " + std::to_string(g_debug_strings) + ", TerminateProcess calls: " +
             std::to_string(g_terminate_calls));
        check(!c.faulted || c.code == kTerminateCalled, "constructor returns or gives up through TerminateProcess, no fault");

        if (!c.faulted) {
            rt::d3d12_xs::reset_device_proxy_stats();
            c = guarded_dtor(dtor, object.data());
            note("~XboxBvhBuilder " + describe(c));
            print_slot_hits(proxy);
            check(!c.faulted, "destructor returns");
        }
        reinterpret_cast<IUnknown*>(proxy)->Release();
        pc->Release();
    } catch (const std::exception& e) {
        std::printf("FAIL %s\n", e.what());
        return 1;
    }

    std::printf("bvh_builder_test: %s (%d failure%s)\n", g_failures ? "FAILED" : "OK", g_failures,
                g_failures == 1 ? "" : "s");
    return g_failures ? 1 : 0;
}
