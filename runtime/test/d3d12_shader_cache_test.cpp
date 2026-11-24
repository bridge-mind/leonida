// Synthetic test for ID3D12Device9::CreateShaderCacheSession, ID3D12ShaderCacheSession
// (FindValue, StoreValue, SetDeleteOnDestroy, GetDesc), and ID3D12Device9::ShaderCacheControl (D3D-34).
// Compares argument validation, StoreValue/FindValue semantics, size queries, duplicate rejection,
// shared-session visibility, and SetDeleteOnDestroy against the host Windows D3D12 runtime when available.
// No game data.
#include "vk_test_device.h"
#include "d3d12_copy_compare.h"
#include "../shims/d3d12_com.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

template<typename T>
struct Ref {
    T* p = nullptr;
    ~Ref() { reset(); }
    void reset() { if (p) { p->Release(); p = nullptr; } }
    T* operator->() const { return p; }
};

int exercise_shader_cache(ID3D12Device9* dev, const char* label, bool is_vulkan_layer) {
    // {7c83f2a1-4b5e-4d19-9a32-1f2e3d4c5b6a}
    const GUID id1 = {0x7c83f2a1, 0x4b5e, 0x4d19, {0x9a, 0x32, 0x1f, 0x2e, 0x3d, 0x4c, 0x5b, 0x6a}};
    const GUID id2 = {0x1b74e9d0, 0x2c3a, 0x4f81, {0x8e, 0x11, 0x5a, 0x6b, 0x7c, 0x8d, 0x9e, 0x0f}};
    const GUID zero_id{};

    D3D12_SHADER_CACHE_SESSION_DESC desc{};
    desc.Identifier = id1;
    desc.Mode = D3D12_SHADER_CACHE_MODE_MEMORY;
    desc.Flags = D3D12_SHADER_CACHE_FLAG_NONE;
    desc.Version = 1;

    // 1. CreateShaderCacheSession argument validation & capability check (null output -> S_FALSE).
    // Passing nullptr for SAL _In_/_Inout_ parameters (pDesc, pKey, pValue, pValueSize) is undefined
    // behaviour on older in-box Windows D3D12 runtimes, so those null-pointer guards are exercised
    // only on the Vulkan layer.
    if (is_vulkan_layer) {
        EXPECT(dev->CreateShaderCacheSession(nullptr, __uuidof(ID3D12ShaderCacheSession), nullptr) == E_INVALIDARG);
    }
    EXPECT(dev->CreateShaderCacheSession(&desc, __uuidof(ID3D12ShaderCacheSession), nullptr) == S_FALSE);

    D3D12_SHADER_CACHE_SESSION_DESC bad = desc;
    bad.Identifier = zero_id;
    Ref<ID3D12ShaderCacheSession> bad_s;
    EXPECT(dev->CreateShaderCacheSession(&bad, IID_PPV_ARGS(&bad_s.p)) == E_INVALIDARG && !bad_s.p);
    EXPECT(dev->CreateShaderCacheSession(&bad, __uuidof(ID3D12ShaderCacheSession), nullptr) == E_INVALIDARG);

    bad = desc;
    bad.Mode = static_cast<D3D12_SHADER_CACHE_MODE>(2);
    EXPECT(dev->CreateShaderCacheSession(&bad, IID_PPV_ARGS(&bad_s.p)) == E_INVALIDARG && !bad_s.p);

    bad = desc;
    bad.Flags = static_cast<D3D12_SHADER_CACHE_FLAGS>(0x4);
    EXPECT(dev->CreateShaderCacheSession(&bad, IID_PPV_ARGS(&bad_s.p)) == E_INVALIDARG && !bad_s.p);

    void* wrong_iid = nullptr;
    EXPECT(dev->CreateShaderCacheSession(&desc, __uuidof(ID3D12Fence), &wrong_iid) == E_NOINTERFACE && !wrong_iid);

    // 2. StoreValue / FindValue / GetDesc / GetDevice.
    Ref<ID3D12ShaderCacheSession> s1;
    EXPECT(SUCCEEDED(dev->CreateShaderCacheSession(&desc, IID_PPV_ARGS(&s1.p))) && s1.p);
    const D3D12_SHADER_CACHE_SESSION_DESC got = s1->GetDesc();
    EXPECT(std::memcmp(&got.Identifier, &desc.Identifier, sizeof(GUID)) == 0);
    EXPECT(got.Mode == desc.Mode && got.Flags == desc.Flags && got.Version == desc.Version);
    Ref<ID3D12Device> parent;
    EXPECT(SUCCEEDED(s1->GetDevice(IID_PPV_ARGS(&parent.p))) && parent.p);

    const std::uint32_t k1 = 0x12345678u;
    const std::uint32_t v1 = 0xCAFEBABEu;
    std::uint32_t out_v = 0;
    UINT sz = sizeof(out_v);
    EXPECT(s1->FindValue(&k1, sizeof(k1), &out_v, &sz) == DXGI_ERROR_NOT_FOUND);
    EXPECT(sz == sizeof(out_v));
    if (is_vulkan_layer) {
        EXPECT(s1->FindValue(nullptr, sizeof(k1), &out_v, &sz) == E_INVALIDARG);
    }
    EXPECT(s1->FindValue(&k1, 0, &out_v, &sz) == E_INVALIDARG);
    if (is_vulkan_layer) {
        EXPECT(s1->FindValue(&k1, sizeof(k1), &out_v, nullptr) == E_INVALIDARG);
        EXPECT(s1->StoreValue(nullptr, sizeof(k1), &v1, sizeof(v1)) == E_INVALIDARG);
    }
    EXPECT(s1->StoreValue(&k1, 0, &v1, sizeof(v1)) == E_INVALIDARG);
    if (is_vulkan_layer) {
        EXPECT(s1->StoreValue(&k1, sizeof(k1), nullptr, sizeof(v1)) == E_INVALIDARG);
    }
    EXPECT(s1->StoreValue(&k1, sizeof(k1), &v1, 0) == E_INVALIDARG);
    EXPECT(s1->StoreValue(&k1, sizeof(k1), &v1, sizeof(v1)) == S_OK);
    EXPECT(s1->StoreValue(&k1, sizeof(k1), &v1, sizeof(v1)) == DXGI_ERROR_ALREADY_EXISTS);

    // Size query (pValue == nullptr) returns S_OK and writes stored size.
    sz = 0;
    EXPECT(s1->FindValue(&k1, sizeof(k1), nullptr, &sz) == S_OK && sz == sizeof(v1));
    sz = 2;
    EXPECT(s1->FindValue(&k1, sizeof(k1), nullptr, &sz) == S_OK && sz == sizeof(v1));
    // Non-null buffer too small returns DXGI_ERROR_MORE_DATA and writes required size.
    sz = 2;
    EXPECT(s1->FindValue(&k1, sizeof(k1), &out_v, &sz) == DXGI_ERROR_MORE_DATA && sz == sizeof(v1));
    // Oversized buffer returns S_OK, writes exact stored size, and copies payload.
    sz = 8;
    out_v = 0;
    EXPECT(s1->FindValue(&k1, sizeof(k1), &out_v, &sz) == S_OK && sz == sizeof(v1) && out_v == v1);

    // 3. Shared store across sessions with the same (Identifier, Mode, Version) + SetDeleteOnDestroy.
    Ref<ID3D12ShaderCacheSession> s2;
    EXPECT(SUCCEEDED(dev->CreateShaderCacheSession(&desc, IID_PPV_ARGS(&s2.p))) && s2.p);
    sz = sizeof(out_v);
    out_v = 0;
    EXPECT(s2->FindValue(&k1, sizeof(k1), &out_v, &sz) == S_OK && sz == sizeof(v1) && out_v == v1);

    // Different Identifier does not see k1.
    // Older in-box Windows D3D12 runtimes (e.g. Win11 23H2 d3d12.dll 10.0.22621.5415) AV (0xC0000005)
    // when releasing a D3D12_SHADER_CACHE_MODE_MEMORY session after SetDeleteOnDestroy(), so exercise
    // SetDeleteOnDestroy() on MODE_MEMORY only on the Vulkan layer; MODE_MEMORY still discards on final
    // release on both runtimes.
    D3D12_SHADER_CACHE_SESSION_DESC desc2 = desc;
    desc2.Identifier = id2;
    Ref<ID3D12ShaderCacheSession> other;
    EXPECT(SUCCEEDED(dev->CreateShaderCacheSession(&desc2, IID_PPV_ARGS(&other.p))) && other.p);
    sz = sizeof(out_v);
    EXPECT(other->FindValue(&k1, sizeof(k1), &out_v, &sz) == DXGI_ERROR_NOT_FOUND);
    if (is_vulkan_layer) other->SetDeleteOnDestroy();
    other.reset();

    // Releasing s1 keeps data alive while s2 is still open, then discards once s2 releases.
    if (is_vulkan_layer) s1->SetDeleteOnDestroy();
    s1.reset();
    sz = sizeof(out_v);
    out_v = 0;
    EXPECT(s2->FindValue(&k1, sizeof(k1), &out_v, &sz) == S_OK && sz == sizeof(v1) && out_v == v1);
    s2.reset();

    Ref<ID3D12ShaderCacheSession> s3;
    EXPECT(SUCCEEDED(dev->CreateShaderCacheSession(&desc, IID_PPV_ARGS(&s3.p))) && s3.p);
    sz = sizeof(out_v);
    EXPECT(s3->FindValue(&k1, sizeof(k1), &out_v, &sz) == DXGI_ERROR_NOT_FOUND);
    if (is_vulkan_layer) s3->SetDeleteOnDestroy();
    s3.reset();

    // 4. ShaderCacheControl invalid-argument checks (same on host and layer).
    EXPECT(dev->ShaderCacheControl(D3D12_SHADER_CACHE_KIND_FLAGS(0), D3D12_SHADER_CACHE_CONTROL_FLAG_ENABLE) ==
           E_INVALIDARG);
    EXPECT(dev->ShaderCacheControl(D3D12_SHADER_CACHE_KIND_FLAG_APPLICATION_MANAGED,
                                   D3D12_SHADER_CACHE_CONTROL_FLAGS(0)) == E_INVALIDARG);
    EXPECT(dev->ShaderCacheControl(D3D12_SHADER_CACHE_KIND_FLAG_APPLICATION_MANAGED,
                                   D3D12_SHADER_CACHE_CONTROL_FLAGS(0x8)) == E_INVALIDARG);
    EXPECT(dev->ShaderCacheControl(D3D12_SHADER_CACHE_KIND_FLAG_APPLICATION_MANAGED,
                                   D3D12_SHADER_CACHE_CONTROL_FLAG_ENABLE | D3D12_SHADER_CACHE_CONTROL_FLAG_DISABLE) ==
           E_INVALIDARG);

    std::printf("shader cache (%s): validation, StoreValue/FindValue, size queries, sharing and SetDeleteOnDestroy OK\n",
                label);
    return 0;
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    vktest::Device d;
    const auto status = d.initialize();
    if (status == vktest::DeviceStatus::unavailable) {
        std::fprintf(stderr, "SKIP: no usable Vulkan 1.3 device\n");
        return 77;
    }
    EXPECT(status == vktest::DeviceStatus::ready);

    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = d.instance_proc;
    info.instance = d.instance;
    info.physical = d.physical;
    info.device = d.device;
    info.queue = d.queue;
    info.queue_family = d.family;
    info.buffer_device_address = true;

    Ref<ID3D12Device9> device;
    EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p))));
    EXPECT(exercise_shader_cache(device.p, "Vulkan layer", true) == 0);

    // Layer-specific ShaderCacheControl CLEAR / DISABLE / ENABLE, default limits, DXGI_ERROR_CACHE_FULL,
    // version-mismatch handling, MEMORY teardown discard, and malformed-disk-file bounds checks.
    {
        const GUID id_ctrl = {0x4a12b3c4, 0x5d6e, 0x47f8, {0x90, 0xa1, 0xb2, 0xc3, 0xd4, 0xe5, 0xf6, 0x07}};
        D3D12_SHADER_CACHE_SESSION_DESC desc{};
        desc.Identifier = id_ctrl;
        desc.Mode = D3D12_SHADER_CACHE_MODE_MEMORY;
        desc.Version = 1;
        desc.MaximumInMemoryCacheEntries = 1;
        desc.MaximumInMemoryCacheSizeBytes = 8;
        Ref<ID3D12ShaderCacheSession> s;
        EXPECT(SUCCEEDED(device->CreateShaderCacheSession(&desc, IID_PPV_ARGS(&s.p))) && s.p);

        const std::uint32_t k1 = 1, k2 = 2;
        const std::uint64_t v1 = 0x1122334455667788ull;
        EXPECT(s->StoreValue(&k1, sizeof(k1), &v1, sizeof(v1)) == S_OK);
        // Full cache must return DXGI_ERROR_CACHE_FULL (not E_OUTOFMEMORY).
        EXPECT(s->StoreValue(&k2, sizeof(k2), &v1, sizeof(v1)) == DXGI_ERROR_CACHE_FULL);

        // Opening the same Identifier with a different Version while a session is live returns DXGI_ERROR_ALREADY_EXISTS.
        D3D12_SHADER_CACHE_SESSION_DESC v2_desc = desc;
        v2_desc.Version = 2;
        Ref<ID3D12ShaderCacheSession> conflict;
        EXPECT(device->CreateShaderCacheSession(&v2_desc, IID_PPV_ARGS(&conflict.p)) == DXGI_ERROR_ALREADY_EXISTS &&
               !conflict.p);

        // DISABLE must return DXGI_ERROR_NOT_CURRENTLY_AVAILABLE on CreateShaderCacheSession, FindValue, and StoreValue.
        EXPECT(device->ShaderCacheControl(D3D12_SHADER_CACHE_KIND_FLAG_APPLICATION_MANAGED,
                                          D3D12_SHADER_CACHE_CONTROL_FLAG_DISABLE) == S_OK);
        Ref<ID3D12ShaderCacheSession> disabled_s;
        EXPECT(device->CreateShaderCacheSession(&desc, IID_PPV_ARGS(&disabled_s.p)) ==
                   DXGI_ERROR_NOT_CURRENTLY_AVAILABLE &&
               !disabled_s.p);
        std::uint64_t out = 0;
        UINT sz = sizeof(out);
        EXPECT(s->FindValue(&k1, sizeof(k1), &out, &sz) == DXGI_ERROR_NOT_CURRENTLY_AVAILABLE);
        EXPECT(s->StoreValue(&k2, sizeof(k2), &v1, sizeof(v1)) == DXGI_ERROR_NOT_CURRENTLY_AVAILABLE);

        EXPECT(device->ShaderCacheControl(D3D12_SHADER_CACHE_KIND_FLAG_APPLICATION_MANAGED,
                                          D3D12_SHADER_CACHE_CONTROL_FLAG_ENABLE) == S_OK);
        sz = sizeof(out);
        EXPECT(s->FindValue(&k1, sizeof(k1), &out, &sz) == S_OK && out == v1);

        EXPECT(device->ShaderCacheControl(D3D12_SHADER_CACHE_KIND_FLAG_APPLICATION_MANAGED,
                                          D3D12_SHADER_CACHE_CONTROL_FLAG_CLEAR) == S_OK);
        sz = sizeof(out);
        EXPECT(s->FindValue(&k1, sizeof(k1), &out, &sz) == DXGI_ERROR_NOT_FOUND);
        s.reset();

        // Zero limits select documented defaults (1 KiB in-memory bytes, 128 entries) and MEMORY discards on final release.
        D3D12_SHADER_CACHE_SESSION_DESC def_desc{};
        def_desc.Identifier = id_ctrl;
        def_desc.Mode = D3D12_SHADER_CACHE_MODE_MEMORY;
        def_desc.Version = 3;
        Ref<ID3D12ShaderCacheSession> def_s;
        EXPECT(SUCCEEDED(device->CreateShaderCacheSession(&def_desc, IID_PPV_ARGS(&def_s.p))) && def_s.p);
        const std::vector<std::uint8_t> over_1k(1025, 0xAB);
        EXPECT(def_s->StoreValue(&k1, sizeof(k1), over_1k.data(), static_cast<UINT>(over_1k.size())) ==
               DXGI_ERROR_CACHE_FULL);
        const std::vector<std::uint8_t> fit_1k(512, 0xCD);
        EXPECT(def_s->StoreValue(&k1, sizeof(k1), fit_1k.data(), static_cast<UINT>(fit_1k.size())) == S_OK);
        def_s.reset();  // final release without SetDeleteOnDestroy must still discard MEMORY contents
        Ref<ID3D12ShaderCacheSession> reopened;
        EXPECT(SUCCEEDED(device->CreateShaderCacheSession(&def_desc, IID_PPV_ARGS(&reopened.p))) && reopened.p);
        sz = static_cast<UINT>(fit_1k.size());
        std::vector<std::uint8_t> read_buf(512);
        EXPECT(reopened->FindValue(&k1, sizeof(k1), read_buf.data(), &sz) == DXGI_ERROR_NOT_FOUND);
        reopened.reset();
    }

    // Cross-device process-wide sharing: two distinct ID3D12Device9 objects in the same process share the store.
    {
        const GUID id_shared = {0x99887766, 0x5544, 0x4321, {0xab, 0xcd, 0xef, 0x01, 0x23, 0x45, 0x67, 0x89}};
        D3D12_SHADER_CACHE_SESSION_DESC desc{};
        desc.Identifier = id_shared;
        desc.Mode = D3D12_SHADER_CACHE_MODE_MEMORY;
        desc.Version = 2;

        Ref<ID3D12Device9> device2;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device2.p))));
        Ref<ID3D12ShaderCacheSession> s_dev1, s_dev2;
        EXPECT(SUCCEEDED(device->CreateShaderCacheSession(&desc, IID_PPV_ARGS(&s_dev1.p))) && s_dev1.p);
        EXPECT(SUCCEEDED(device2->CreateShaderCacheSession(&desc, IID_PPV_ARGS(&s_dev2.p))) && s_dev2.p);

        const std::uint32_t k = 0x55AA1234u;
        const std::uint64_t v = 0xDEADBEEFCAFEBABEull;
        EXPECT(s_dev1->StoreValue(&k, sizeof(k), &v, sizeof(v)) == S_OK);
        std::uint64_t out = 0;
        UINT sz = sizeof(out);
        EXPECT(s_dev2->FindValue(&k, sizeof(k), &out, &sz) == S_OK && sz == sizeof(v) && out == v);
        s_dev1->SetDeleteOnDestroy();
        s_dev1.reset();
        s_dev2.reset();
        device2.reset();
    }

    // D3D12_SHADER_CACHE_MODE_DISK persistence across session + device teardown and cleanup on SetDeleteOnDestroy.
    {
        const auto prev_cwd = std::filesystem::current_path();
        const auto temp_dir = std::filesystem::temp_directory_path() / L"leonida_shader_cache_disk_test";
        std::error_code ec;
        std::filesystem::remove_all(temp_dir, ec);
        std::filesystem::create_directories(temp_dir);
        struct RestoreCwd {
            std::filesystem::path prev;
            std::filesystem::path dir;
            ~RestoreCwd() {
                std::error_code e;
                std::filesystem::current_path(prev, e);
                std::filesystem::remove_all(dir, e);
            }
        } restore_cwd{prev_cwd, temp_dir};
        std::filesystem::current_path(temp_dir);

        const GUID id_disk = {0x31415926, 0x5358, 0x4979, {0x83, 0x23, 0x84, 0x62, 0x64, 0x33, 0x83, 0x27}};
        D3D12_SHADER_CACHE_SESSION_DESC disk_desc{};
        disk_desc.Identifier = id_disk;
        disk_desc.Mode = D3D12_SHADER_CACHE_MODE_DISK;
        disk_desc.Flags = D3D12_SHADER_CACHE_FLAG_USE_WORKING_DIR;
        disk_desc.Version = 7;

        const std::uint32_t k = 0x0F1E2D3Cu;
        const std::uint64_t v = 0x0123456789ABCDEFull;
        {
            Ref<ID3D12Device9> dev_a;
            EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&dev_a.p))));
            Ref<ID3D12ShaderCacheSession> s_a;
            EXPECT(SUCCEEDED(dev_a->CreateShaderCacheSession(&disk_desc, IID_PPV_ARGS(&s_a.p))) && s_a.p);
            EXPECT(s_a->StoreValue(&k, sizeof(k), &v, sizeof(v)) == S_OK);
        }
        // After dev_a and s_a are destroyed, the cache file must exist on disk and reload into a brand-new device.
        bool found_file = false;
        for (const auto& entry : std::filesystem::directory_iterator(temp_dir)) {
            if (entry.is_regular_file() && entry.path().extension() == L".bin") found_file = true;
        }
        EXPECT(found_file);
        {
            Ref<ID3D12Device9> dev_b;
            EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&dev_b.p))));
            Ref<ID3D12ShaderCacheSession> s_b;
            EXPECT(SUCCEEDED(dev_b->CreateShaderCacheSession(&disk_desc, IID_PPV_ARGS(&s_b.p))) && s_b.p);
            std::uint64_t out = 0;
            UINT sz = sizeof(out);
            EXPECT(s_b->FindValue(&k, sizeof(k), &out, &sz) == S_OK && sz == sizeof(v) && out == v);
            // Opening with a different Version while s_b is live returns DXGI_ERROR_ALREADY_EXISTS.
            D3D12_SHADER_CACHE_SESSION_DESC bad_ver = disk_desc;
            bad_ver.Version = 8;
            Ref<ID3D12ShaderCacheSession> conflict;
            EXPECT(dev_b->CreateShaderCacheSession(&bad_ver, IID_PPV_ARGS(&conflict.p)) == DXGI_ERROR_ALREADY_EXISTS &&
                   !conflict.p);
        }
        // Reopening with a new Version after all sessions closed clears and recreates the disk cache.
        {
            D3D12_SHADER_CACHE_SESSION_DESC v8_desc = disk_desc;
            v8_desc.Version = 8;
            Ref<ID3D12Device9> dev_v8;
            EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&dev_v8.p))));
            Ref<ID3D12ShaderCacheSession> s_v8;
            EXPECT(SUCCEEDED(dev_v8->CreateShaderCacheSession(&v8_desc, IID_PPV_ARGS(&s_v8.p))) && s_v8.p);
            std::uint64_t out = 0;
            UINT sz = sizeof(out);
            EXPECT(s_v8->FindValue(&k, sizeof(k), &out, &sz) == DXGI_ERROR_NOT_FOUND);
            EXPECT(s_v8->StoreValue(&k, sizeof(k), &v, sizeof(v)) == S_OK);
            s_v8->SetDeleteOnDestroy();
        }
        // After SetDeleteOnDestroy + release, the file must be gone and a fresh session must miss.
        bool file_remains = false;
        for (const auto& entry : std::filesystem::directory_iterator(temp_dir)) {
            if (entry.is_regular_file() && entry.path().extension() == L".bin") file_remains = true;
        }
        EXPECT(!file_remains);
        // Corrupt/truncated disk file with huge claimed sizes must be rejected safely without huge allocation.
        {
            const std::filesystem::path bad_file =
                temp_dir / L"leonida_sc_31415926535849798323846264338327.bin";
            std::ofstream out(bad_file, std::ios::binary | std::ios::trunc);
            const std::uint32_t hdr[4] = {0x3143534Cu, 2u, 1u, 0u};
            const std::uint64_t ver = 7;
            const std::uint32_t huge_lens[2] = {0x7FFFFFFFu, 0x7FFFFFFFu};
            out.write(reinterpret_cast<const char*>(hdr), sizeof(hdr));
            out.write(reinterpret_cast<const char*>(&ver), sizeof(ver));
            out.write(reinterpret_cast<const char*>(huge_lens), sizeof(huge_lens));
            out.close();

            Ref<ID3D12Device9> dev_c;
            EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&dev_c.p))));
            Ref<ID3D12ShaderCacheSession> s_c;
            EXPECT(SUCCEEDED(dev_c->CreateShaderCacheSession(&disk_desc, IID_PPV_ARGS(&s_c.p))) && s_c.p);
            std::uint64_t val = 0;
            UINT sz = sizeof(val);
            EXPECT(s_c->FindValue(&k, sizeof(k), &val, &sz) == DXGI_ERROR_NOT_FOUND);
            s_c->SetDeleteOnDestroy();
        }
        // Inactive non-USE_WORKING_DIR disk file must be removed by ShaderCacheControl CLEAR.
        {
            const GUID id_inactive = {0x27182818, 0x2845, 0x4045, {0x92, 0x35, 0x36, 0x02, 0x87, 0x47, 0x13, 0x52}};
            D3D12_SHADER_CACHE_SESSION_DESC def_disk{};
            def_disk.Identifier = id_inactive;
            def_disk.Mode = D3D12_SHADER_CACHE_MODE_DISK;
            def_disk.Version = 1;
            {
                Ref<ID3D12ShaderCacheSession> s_in;
                EXPECT(SUCCEEDED(device->CreateShaderCacheSession(&def_disk, IID_PPV_ARGS(&s_in.p))) && s_in.p);
                EXPECT(s_in->StoreValue(&k, sizeof(k), &v, sizeof(v)) == S_OK);
            }
            EXPECT(device->ShaderCacheControl(D3D12_SHADER_CACHE_KIND_FLAG_APPLICATION_MANAGED,
                                              D3D12_SHADER_CACHE_CONTROL_FLAG_CLEAR) == S_OK);
            {
                Ref<ID3D12ShaderCacheSession> s_after;
                EXPECT(SUCCEEDED(device->CreateShaderCacheSession(&def_disk, IID_PPV_ARGS(&s_after.p))) && s_after.p);
                std::uint64_t val = 0;
                UINT sz = sizeof(val);
                EXPECT(s_after->FindValue(&k, sizeof(k), &val, &sz) == DXGI_ERROR_NOT_FOUND);
                s_after->SetDeleteOnDestroy();
            }
        }
        std::printf("shader cache (disk persistence + cross-device sharing + clear + bounds): OK\n");
    }

    // Compare the same scenarios against the host Windows D3D12 runtime when ID3D12Device9 is available.
    {
        copycmp::Ptr<ID3D12Device> host;
        std::string adapter_name;
        if (copycmp::create_host(host, adapter_name)) {
            Ref<ID3D12Device9> host9;
            if (SUCCEEDED(host.p->QueryInterface(IID_PPV_ARGS(&host9.p))) && host9.p) {
                EXPECT(exercise_shader_cache(host9.p, adapter_name.c_str(), false) == 0);
            }
        }
    }

    device.reset();
    if (d.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
        return 1;
    }
    std::printf("d3d12_shader_cache_test: passed (0 validation errors)\n");
    return 0;
}
