// CPU regression for explicit host-adapter selection. DXGI only enumerates names; D3D12 device creation is stubbed.
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "vk_test_device.h"

namespace {

unsigned device_calls = 0;
bool valid_feature_probe = false;

HRESULT WINAPI reject_device(IUnknown* adapter, D3D_FEATURE_LEVEL level, REFIID iid, void** output) {
    ++device_calls;
    valid_feature_probe = adapter && level == D3D_FEATURE_LEVEL_11_0 && iid == __uuidof(ID3D12Device) && !output;
    if (output) *output = nullptr;
    return E_NOINTERFACE;
}

FARPROC WINAPI device_lookup(HMODULE module, LPCSTR name) {
    if (name && std::string(name) == "D3D12CreateDevice")
        return reinterpret_cast<FARPROC>(reinterpret_cast<void*>(&reject_device));
    return GetProcAddress(module, name);
}

}  // namespace

// Keep the real Windows environment and DXGI APIs. Both ways of obtaining D3D12CreateDevice use the same stub.
#define D3D12CreateDevice reject_device
#define GetProcAddress device_lookup
#include "d3d12_copy_compare.h"
#undef GetProcAddress
#undef D3D12CreateDevice

namespace {

constexpr const char* variable = "LEONIDA_HOST_ADAPTER";

struct RestoreEnvironment {
    bool existed = false;
    std::string value;

    RestoreEnvironment() {
        SetLastError(ERROR_SUCCESS);
        const DWORD size = GetEnvironmentVariableA(variable, nullptr, 0);
        existed = size != 0 || GetLastError() != ERROR_ENVVAR_NOT_FOUND;
        if (size) {
            value.resize(size);
            value.resize(GetEnvironmentVariableA(variable, value.data(), size));
        }
    }

    ~RestoreEnvironment() { SetEnvironmentVariableA(variable, existed ? value.c_str() : nullptr); }
};

void require_feature_probe() {
    if (device_calls != 1 || !valid_feature_probe) {
        std::fprintf(stderr, "named reference did not probe its adapter with a null device output\n");
        std::_Exit(3);
    }
}

int child(const std::wstring& mode) {
    if (mode == L"--long-selection") {
        const std::string missing = "leonida_nonexistent_adapter_" + std::string(300, 'X');
        if (!SetEnvironmentVariableA(variable, missing.c_str())) return 2;
        copycmp::Ptr<ID3D12Device> host;
        std::string name;
        copycmp::create_host(host, name);  // Explicit selections that cannot be used must exit(1).
        std::fprintf(stderr, "create_host returned after an explicit nonexistent adapter selection\n");
        return 0;
    }
    if (mode == L"--unsupported-reference") {
        copycmp::Ptr<IDXGIFactory4> factory;
        copycmp::Ptr<IDXGIAdapter1> adapter;
        if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p))) ||
            FAILED(factory->EnumAdapters1(0, &adapter.p))) {
            std::puts("SKIP: DXGI has no enumerated adapter for the named-reference check");
            return 77;
        }
        const std::string selected = hostadapter::description(adapter.p);
        if (selected.empty() || !SetEnvironmentVariableA(variable, selected.c_str())) return 2;
        // The adapter is known to exist. The only unavailable part is the stubbed D3D12 feature-level probe.
        if (std::atexit(require_feature_probe) != 0) return 2;
        vktest::Device vk;
        IDXGIAdapter* reference = nullptr;
        std::string name;
        hostadapter::reference(factory.p, vk, false, &reference, name);
        if (reference) reference->Release();
        std::fprintf(stderr, "reference returned after selecting an adapter without D3D12 support\n");
        return 0;
    }
    return 2;
}

bool exits_fatally(const wchar_t* mode, bool allow_no_adapter) {
    std::vector<wchar_t> executable(32768);
    const DWORD length = GetModuleFileNameW(nullptr, executable.data(), static_cast<DWORD>(executable.size()));
    if (!length || length == executable.size()) return false;
    std::wstring command = L"\"" + std::wstring(executable.data(), length) + L"\" " + mode;
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process{};
    if (!CreateProcessW(executable.data(), command.data(), nullptr, nullptr, FALSE, 0,
                        nullptr, nullptr, &startup, &process)) {
        std::fprintf(stderr, "could not start child (Win32 error %lu)\n", GetLastError());
        return false;
    }
    CloseHandle(process.hThread);
    const DWORD wait = WaitForSingleObject(process.hProcess, 30000);
    DWORD code = 0;
    const bool exited = wait == WAIT_OBJECT_0 && GetExitCodeProcess(process.hProcess, &code);
    if (wait == WAIT_TIMEOUT) TerminateProcess(process.hProcess, 2);
    CloseHandle(process.hProcess);
    if (exited && code == 77 && allow_no_adapter) return true;
    if (!exited || code != 1) {
        std::fwprintf(stderr, L"%ls: expected fatal exit 1, got %lu (wait %#lx)\n", mode, code, wait);
        return false;
    }
    return true;
}

}  // namespace

int wmain(int argc, wchar_t** argv) {
    if (argc == 2) return child(argv[1]);
    RestoreEnvironment restore;
    unsigned failures = 0;
    const auto check_environment = [&](const char* input, const std::string& expected, const char* label) {
        if (!SetEnvironmentVariableA(variable, input) || hostadapter::wanted() != expected) {
            std::fprintf(stderr, "wanted(): %s was not preserved and lowercased\n", label);
            ++failures;
        }
    };
    check_environment(nullptr, "", "unset");
    check_environment("", "", "empty");
    check_environment("NvIdIa RtX", "nvidia rtx", "mixed case");
    for (size_t length : {127u, 128u, 300u}) {
        const std::string input(length, 'A');
        const std::string label = std::to_string(length) + " characters";
        check_environment(input.c_str(), std::string(length, 'a'), label.c_str());
    }
    if (!exits_fatally(L"--long-selection", false)) ++failures;
    if (!exits_fatally(L"--unsupported-reference", true)) ++failures;
    if (failures) return 1;
    std::puts("host adapter environment and explicit-selection checks passed (no D3D12 device created)");
    return 0;
}
