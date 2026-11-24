// Checks the clean-room root-signature flag fix against the real PC d3d12.dll.
//
// A synthetic empty root signature is serialized twice: untouched (Xbox flags 0x03000000) must be rejected by the
// PC serializer, and the copy produced by the shim helper (high byte cleared) must be accepted. The same is done
// for the legacy D3D12SerializeRootSignature path, plus a direct check that only bits 24-31 are cleared.
#include <windows.h>
#include <d3d12.h>

#include <cstdio>
#include <cstring>
#include <vector>

#include "d3d12_xs_rootsig.h"
#include "shims.h"

namespace {

int g_failures = 0;

void check(bool condition, const char* what) {
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

// Checks of the host serializer itself, not of the shim: Microsoft's d3d12.dll rejects the Xbox flag byte, Wine's
// (vkd3d) doesn't validate it. Runtime-dependent, as the host deserializer check below.
void check_host_rejects(HRESULT hr, const char* what) {
    if (FAILED(hr)) check(true, what);
    else std::printf("SKIP %s (this host's serializer accepts it; runtime-dependent)\n", what);
}

struct Blobs {
    ID3DBlob* blob = nullptr;
    ID3DBlob* error = nullptr;
    ~Blobs() {
        if (blob) blob->Release();
        if (error) error->Release();
    }
};

constexpr uint32_t kXboxFlags = 0x03000000u;

D3D12_ROOT_SIGNATURE_DESC make_legacy_desc(uint32_t flags) {
    D3D12_ROOT_SIGNATURE_DESC desc{};
    desc.NumParameters = 0;
    desc.pParameters = nullptr;
    desc.NumStaticSamplers = 0;
    desc.pStaticSamplers = nullptr;
    desc.Flags = static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(flags);
    return desc;
}

D3D12_ROOT_SIGNATURE_DESC1 make_legacy_desc1(uint32_t flags) {
    D3D12_ROOT_SIGNATURE_DESC1 desc{};
    desc.NumParameters = 0;
    desc.pParameters = nullptr;
    desc.NumStaticSamplers = 0;
    desc.pStaticSamplers = nullptr;
    desc.Flags = static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(flags);
    return desc;
}

D3D12_ROOT_SIGNATURE_DESC2 make_legacy_desc2(uint32_t flags) {
    D3D12_ROOT_SIGNATURE_DESC2 desc{};
    desc.NumParameters = 0;
    desc.pParameters = nullptr;
    desc.NumStaticSamplers = 0;
    desc.pStaticSamplers = nullptr;
    desc.Flags = static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(flags);
    return desc;
}

// Older in-box D3D12 runtimes (e.g. Windows 11 23H2, Windows 10 22H2) refuse
// root signature 1.2 entirely, so the checks of the host serializer's own 1.2
// behaviour are skipped there. The byte-level helper checks, the 1.2 → 1.1
// rewrite and the registered shim's 1.2 path always run.
bool supports_root_signature_1_2() {
    ID3D12Device* device = nullptr;
    HRESULT hr = D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device));
    if (FAILED(hr) || !device) return false;
    D3D12_FEATURE_DATA_ROOT_SIGNATURE feature{D3D_ROOT_SIGNATURE_VERSION_1_2};
    hr = device->CheckFeatureSupport(D3D12_FEATURE_ROOT_SIGNATURE, &feature, sizeof(feature));
    device->Release();
    return SUCCEEDED(hr) && feature.HighestVersion >= D3D_ROOT_SIGNATURE_VERSION_1_2;
}

void test_versioned_1_0() {
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC xbox{};
    xbox.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    xbox.Desc_1_0 = make_legacy_desc(kXboxFlags);

    {
        Blobs blobs;
        HRESULT hr = D3D12SerializeVersionedRootSignature(&xbox, &blobs.blob, &blobs.error);
        check_host_rejects(hr, "versioned 1.0 with Xbox flags 0x03000000 is rejected by the PC serializer");
    }

    D3D12_VERSIONED_ROOT_SIGNATURE_DESC cleared{};
    bool known = rt::d3d12_xs::copy_versioned_root_signature_desc_clear_xbox_flags(xbox, cleared);
    check(known, "versioned 1.0 descriptor version is recognised");
    check(cleared.Version == xbox.Version, "versioned 1.0 copy keeps the version");
    check(cleared.Desc_1_0.Flags == 0, "versioned 1.0 copy has the high flag byte cleared");
    check(xbox.Desc_1_0.Flags == static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(kXboxFlags),
          "versioned 1.0 original descriptor is unchanged");

    Blobs blobs;
    HRESULT hr = D3D12SerializeVersionedRootSignature(&cleared, &blobs.blob, &blobs.error);
    check(SUCCEEDED(hr), "versioned 1.0 copy with cleared flags is accepted by the PC serializer");
}

void test_versioned_1_1() {
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC xbox{};
    xbox.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
    xbox.Desc_1_1 = make_legacy_desc1(kXboxFlags);

    {
        Blobs blobs;
        HRESULT hr = D3D12SerializeVersionedRootSignature(&xbox, &blobs.blob, &blobs.error);
        check_host_rejects(hr, "versioned 1.1 with Xbox flags 0x03000000 is rejected by the PC serializer");
    }

    D3D12_VERSIONED_ROOT_SIGNATURE_DESC cleared{};
    bool known = rt::d3d12_xs::copy_versioned_root_signature_desc_clear_xbox_flags(xbox, cleared);
    check(known, "versioned 1.1 descriptor version is recognised");
    check(cleared.Desc_1_1.Flags == 0, "versioned 1.1 copy has the high flag byte cleared");
    check(xbox.Desc_1_1.Flags == static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(kXboxFlags),
          "versioned 1.1 original descriptor is unchanged");

    Blobs blobs;
    HRESULT hr = D3D12SerializeVersionedRootSignature(&cleared, &blobs.blob, &blobs.error);
    check(SUCCEEDED(hr), "versioned 1.1 copy with cleared flags is accepted by the PC serializer");
}

void test_versioned_1_2(bool supports_1_2) {
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC xbox{};
    xbox.Version = D3D_ROOT_SIGNATURE_VERSION_1_2;
    xbox.Desc_1_2 = make_legacy_desc2(kXboxFlags);

    // Byte-level: the helper recognises 1.2 and clears only the high byte.
    // These never touch the PC serializer, so they run regardless of runtime.
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC cleared{};
    bool known = rt::d3d12_xs::copy_versioned_root_signature_desc_clear_xbox_flags(xbox, cleared);
    check(known, "versioned 1.2 descriptor version is recognised");
    check(cleared.Desc_1_2.Flags == 0, "versioned 1.2 copy has the high flag byte cleared");
    check(xbox.Desc_1_2.Flags == static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(kXboxFlags),
          "versioned 1.2 original descriptor is unchanged");

    if (!supports_1_2) {
        std::printf("SKIP versioned 1.2 serializer checks (runtime HighestVersion < 1.2)\n");
        return;
    }

    {
        Blobs blobs;
        HRESULT hr = D3D12SerializeVersionedRootSignature(&xbox, &blobs.blob, &blobs.error);
        check_host_rejects(hr, "versioned 1.2 with Xbox flags 0x03000000 is rejected by the PC serializer");
    }

    Blobs blobs;
    HRESULT hr = D3D12SerializeVersionedRootSignature(&cleared, &blobs.blob, &blobs.error);
    check(SUCCEEDED(hr), "versioned 1.2 copy with cleared flags is accepted by the PC serializer");
}

// A 1.2 description with a root parameter and two static samplers, the samplers' Flags 0 unless the caller sets them.
struct Desc12 {
    D3D12_ROOT_PARAMETER1 param{};
    D3D12_STATIC_SAMPLER_DESC1 samplers[2]{};
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC desc{};
    Desc12() {
        param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        param.Constants = {3, 1, 4};
        param.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
        for (int i = 0; i < 2; ++i) {
            D3D12_STATIC_SAMPLER_DESC1& s = samplers[i];
            s.Filter = i ? D3D12_FILTER_ANISOTROPIC : D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT;
            s.AddressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
            s.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
            s.AddressW = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
            s.MipLODBias = 0.5f * float(i + 1);
            s.MaxAnisotropy = i ? 8u : 1u;
            s.ComparisonFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
            s.BorderColor = D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE;
            s.MinLOD = 1.0f;
            s.MaxLOD = 9.0f + float(i);
            s.ShaderRegister = 2u + i;
            s.RegisterSpace = 5u;
            s.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
        }
        desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_2;
        desc.Desc_1_2 = {1, &param, 2, samplers, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT};
    }
};

bool same_sampler(const D3D12_STATIC_SAMPLER_DESC& a, const D3D12_STATIC_SAMPLER_DESC1& b) {
    return a.Filter == b.Filter && a.AddressU == b.AddressU && a.AddressV == b.AddressV && a.AddressW == b.AddressW &&
           a.MipLODBias == b.MipLODBias && a.MaxAnisotropy == b.MaxAnisotropy &&
           a.ComparisonFunc == b.ComparisonFunc && a.BorderColor == b.BorderColor && a.MinLOD == b.MinLOD &&
           a.MaxLOD == b.MaxLOD && a.ShaderRegister == b.ShaderRegister && a.RegisterSpace == b.RegisterSpace &&
           a.ShaderVisibility == b.ShaderVisibility;
}

// The 1.2 → 1.1 rewrite the serializer shim uses on runtimes that stop at 1.1 (Windows 10, Windows 11 23H2).
void test_downgrade_1_2_to_1_1(bool supports_1_2) {
    Desc12 d;
    std::vector<D3D12_STATIC_SAMPLER_DESC> storage;
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC v11{};
    check(rt::d3d12_xs::downgrade_root_signature_1_2_to_1_1(d.desc, storage, v11),
          "1.2 without sampler flags is rewritten as 1.1");
    check(v11.Version == D3D_ROOT_SIGNATURE_VERSION_1_1 && v11.Desc_1_1.NumParameters == 1 &&
              v11.Desc_1_1.pParameters == &d.param && v11.Desc_1_1.NumStaticSamplers == 2 &&
              v11.Desc_1_1.pStaticSamplers == storage.data() &&
              v11.Desc_1_1.Flags == D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
          "the 1.1 rewrite keeps the parameters and flags and points at the converted samplers");
    check(storage.size() == 2 && same_sampler(storage[0], d.samplers[0]) && same_sampler(storage[1], d.samplers[1]),
          "every static sampler field survives the 1.1 rewrite");

    Blobs b11;
    check(SUCCEEDED(D3D12SerializeVersionedRootSignature(&v11, &b11.blob, &b11.error)),
          "the 1.1 rewrite is accepted by the PC serializer");
    if (supports_1_2) {
        Blobs b12;
        check(SUCCEEDED(D3D12SerializeVersionedRootSignature(&d.desc, &b12.blob, &b12.error)),
              "the 1.2 original is accepted by the PC serializer");
    } else {
        std::printf("SKIP the 1.2 original is accepted by the PC serializer (runtime HighestVersion < 1.2)\n");
    }

    D3D12_VERSIONED_ROOT_SIGNATURE_DESC empty12{};
    empty12.Version = D3D_ROOT_SIGNATURE_VERSION_1_2;
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC empty11{};
    check(rt::d3d12_xs::downgrade_root_signature_1_2_to_1_1(empty12, storage, empty11) &&
              empty11.Desc_1_1.NumStaticSamplers == 0 && empty11.Desc_1_1.pStaticSamplers == nullptr,
          "an empty 1.2 description is rewritten as an empty 1.1 one");

    for (D3D12_SAMPLER_FLAGS flag : {D3D12_SAMPLER_FLAG_UINT_BORDER_COLOR,
                                     D3D12_SAMPLER_FLAG_NON_NORMALIZED_COORDINATES}) {
        Desc12 f;
        f.samplers[1].Flags = flag;
        std::vector<D3D12_STATIC_SAMPLER_DESC> kept(1);
        D3D12_VERSIONED_ROOT_SIGNATURE_DESC untouched{};
        untouched.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
        const bool rewritten = rt::d3d12_xs::downgrade_root_signature_1_2_to_1_1(f.desc, kept, untouched);
        check(!rewritten && kept.size() == 1 && untouched.Version == D3D_ROOT_SIGNATURE_VERSION_1_0,
              flag == D3D12_SAMPLER_FLAG_UINT_BORDER_COLOR
                  ? "1.2 with a UINT border colour sampler flag is not rewritten, outputs untouched"
                  : "1.2 with a non-normalized coordinates sampler flag is not rewritten, outputs untouched");
    }

    D3D12_VERSIONED_ROOT_SIGNATURE_DESC v11in{};
    v11in.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
    check(!rt::d3d12_xs::downgrade_root_signature_1_2_to_1_1(v11in, storage, empty11),
          "a description that isn't 1.2 is not rewritten");
}

void test_legacy_entry_point() {
    D3D12_ROOT_SIGNATURE_DESC xbox = make_legacy_desc(kXboxFlags);

    {
        Blobs blobs;
        HRESULT hr = D3D12SerializeRootSignature(&xbox, D3D_ROOT_SIGNATURE_VERSION_1_0, &blobs.blob, &blobs.error);
        check_host_rejects(hr, "legacy serializer rejects flags 0x03000000");
    }

    D3D12_ROOT_SIGNATURE_DESC cleared{};
    rt::d3d12_xs::copy_root_signature_desc_clear_xbox_flags(xbox, cleared);
    check(cleared.Flags == 0, "legacy copy has the high flag byte cleared");
    check(xbox.Flags == static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(kXboxFlags), "legacy original is unchanged");

    Blobs blobs;
    HRESULT hr = D3D12SerializeRootSignature(&cleared, D3D_ROOT_SIGNATURE_VERSION_1_0, &blobs.blob, &blobs.error);
    check(SUCCEEDED(hr), "legacy copy with cleared flags is accepted by the PC serializer");
}

void test_preserves_low_bits_and_unknown_versions() {
    // Flags 0x03000045: clear the high byte but keep the valid low bits (input assembler + stream output).
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC xbox{};
    xbox.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    xbox.Desc_1_0 = make_legacy_desc(0x03000045u);

    D3D12_VERSIONED_ROOT_SIGNATURE_DESC cleared{};
    (void)rt::d3d12_xs::copy_versioned_root_signature_desc_clear_xbox_flags(xbox, cleared);
    check(cleared.Desc_1_0.Flags == static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(0x45u),
          "only the high flag byte is cleared; low flag bits are preserved");

    D3D12_VERSIONED_ROOT_SIGNATURE_DESC unknown = xbox;
    unknown.Version = static_cast<D3D_ROOT_SIGNATURE_VERSION>(0x7f);
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC forwarded{};
    bool known = rt::d3d12_xs::copy_versioned_root_signature_desc_clear_xbox_flags(unknown, forwarded);
    check(!known, "unknown descriptor version reports that it was not rewritten");
    const bool unchanged = forwarded.Version == unknown.Version &&
                           forwarded.Desc_1_0.NumParameters == unknown.Desc_1_0.NumParameters &&
                           forwarded.Desc_1_0.pParameters == unknown.Desc_1_0.pParameters &&
                           forwarded.Desc_1_0.NumStaticSamplers == unknown.Desc_1_0.NumStaticSamplers &&
                           forwarded.Desc_1_0.pStaticSamplers == unknown.Desc_1_0.pStaticSamplers &&
                           forwarded.Desc_1_0.Flags == unknown.Desc_1_0.Flags;
    check(unchanged, "unknown descriptor version is copied unchanged for forwarding");
}

void test_registered_shim(bool supports_1_2) {
    using SerializeFn =
        HRESULT(WINAPI*)(const D3D12_VERSIONED_ROOT_SIGNATURE_DESC*, ID3DBlob**, ID3DBlob**);
    auto fn = reinterpret_cast<SerializeFn>(
        rt::shims::find("d3d12_xs.dll", "D3D12SerializeVersionedRootSignature"));
    check(fn != nullptr, "shim registry exposes d3d12_xs!D3D12SerializeVersionedRootSignature");
    check(rt::shims::find("d3d12_xs.dll", "D3D12SerializeRootSignature") != nullptr,
          "shim registry exposes d3d12_xs!D3D12SerializeRootSignature");
    check(rt::shims::find("d3d12_xs.dll", "D3D12CreateRootSignatureDeserializer") != nullptr,
          "shim registry exposes d3d12_xs!D3D12CreateRootSignatureDeserializer");
    check(rt::shims::find("d3d12_xs.dll", "D3D12XboxCreateDevice") != nullptr,
          "shim registry exposes d3d12_xs!D3D12XboxCreateDevice");
    if (!fn) return;

    D3D12_VERSIONED_ROOT_SIGNATURE_DESC xbox{};
    xbox.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    xbox.Desc_1_0 = make_legacy_desc(kXboxFlags);

    Blobs blobs;
    HRESULT hr = fn(&xbox, &blobs.blob, &blobs.error);
    check(SUCCEEDED(hr), "registered shim clears the Xbox flag byte and the PC serializer accepts the result");
    check(xbox.Desc_1_0.Flags == static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(kXboxFlags),
          "registered shim leaves the caller's descriptor unchanged");

    // 1.2 serializes through the shim on every runtime: as 1.2 where the host takes it, rewritten as 1.1 where it
    // stops at 1.1.
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC xbox12{};
    xbox12.Version = D3D_ROOT_SIGNATURE_VERSION_1_2;
    xbox12.Desc_1_2 = make_legacy_desc2(kXboxFlags);

    Blobs blobs12;
    HRESULT hr12 = fn(&xbox12, &blobs12.blob, &blobs12.error);
    check(SUCCEEDED(hr12), "registered shim clears the Xbox flag byte for 1.2 and the PC serializer accepts");
    check(xbox12.Desc_1_2.Flags == static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(kXboxFlags),
          "registered shim leaves the caller's 1.2 descriptor unchanged");

    // With parameters and samplers but no sampler flags; read back through the host deserializer.
    Desc12 d;
    d.desc.Desc_1_2.Flags = static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(kXboxFlags | 0x1u);
    Blobs full;
    check(SUCCEEDED(fn(&d.desc, &full.blob, &full.error)),
          "registered shim serializes 1.2 with samplers and no sampler flags");
    if (full.blob) {
        ID3D12VersionedRootSignatureDeserializer* de = nullptr;
        const HRESULT hr_de = D3D12CreateVersionedRootSignatureDeserializer(
            full.blob->GetBufferPointer(), full.blob->GetBufferSize(), IID_PPV_ARGS(&de));
        check(SUCCEEDED(hr_de), "the shim's 1.2 blob deserializes");
        if (SUCCEEDED(hr_de) && de) {
            const D3D12_VERSIONED_ROOT_SIGNATURE_DESC* got = de->GetUnconvertedRootSignatureDesc();
            const D3D_ROOT_SIGNATURE_VERSION want =
                supports_1_2 ? D3D_ROOT_SIGNATURE_VERSION_1_2 : D3D_ROOT_SIGNATURE_VERSION_1_1;
            check(got && got->Version == want,
                  supports_1_2 ? "the shim's blob stays version 1.2 on a runtime that takes 1.2"
                               : "the shim's blob is version 1.1 on a runtime that stops at 1.1");
            if (got && got->Version == D3D_ROOT_SIGNATURE_VERSION_1_1) {
                const D3D12_ROOT_SIGNATURE_DESC1& g = got->Desc_1_1;
                check(g.Flags == D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT && g.NumParameters == 1 &&
                          g.pParameters[0].ParameterType == D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS &&
                          g.pParameters[0].Constants.ShaderRegister == 3 &&
                          g.pParameters[0].Constants.RegisterSpace == 1 &&
                          g.pParameters[0].Constants.Num32BitValues == 4 && g.NumStaticSamplers == 2 &&
                          same_sampler(g.pStaticSamplers[0], d.samplers[0]) &&
                          same_sampler(g.pStaticSamplers[1], d.samplers[1]),
                      "the 1.1 blob has the Xbox byte cleared and the same parameter and samplers");
            }
            de->Release();
        }
    }

    // Sampler flags have no 1.1 form: the host's own answer where it takes 1.2, a refusal where it doesn't.
    Desc12 flagged;
    flagged.samplers[0].BorderColor = D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE_UINT;
    flagged.samplers[0].Flags = D3D12_SAMPLER_FLAG_UINT_BORDER_COLOR;
    Blobs fb;
    const HRESULT hr_flagged = fn(&flagged.desc, &fb.blob, &fb.error);
    if (supports_1_2)
        check(SUCCEEDED(hr_flagged), "registered shim forwards 1.2 with sampler flags to a runtime that takes 1.2");
    else
        check(hr_flagged == E_INVALIDARG && !fb.blob,
              "registered shim refuses 1.2 with sampler flags when the runtime stops at 1.1");
}

void test_deserializer() {
    // The host deserializer checks the container hash and hands an Xbox high byte back in the flags; the shim
    // clears the byte on a copy, rehashes it and succeeds.
    D3D12_ROOT_SIGNATURE_DESC desc = make_legacy_desc(0);
    Blobs blobs;
    HRESULT hr = D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1_0, &blobs.blob, &blobs.error);
    check(SUCCEEDED(hr), "serialize a clean legacy root signature for the deserializer test");
    if (FAILED(hr) || !blobs.blob) return;

    const uint8_t* bytes = static_cast<const uint8_t*>(blobs.blob->GetBufferPointer());
    const SIZE_T size = blobs.blob->GetBufferSize();

    std::vector<uint8_t> tampered(bytes, bytes + size);
    uint32_t chunk_count = 0;
    std::memcpy(&chunk_count, bytes + 28, sizeof(chunk_count));
    bool found_rts0 = false;
    for (uint32_t i = 0; i < chunk_count; ++i) {
        uint32_t off = 0;
        std::memcpy(&off, bytes + 32 + i * 4, sizeof(off));
        if (off > size || size - off < 8 + 24) continue;
        if (std::memcmp(bytes + off, "RTS0", 4) != 0) continue;
        uint32_t flags = 0;
        std::memcpy(&flags, bytes + off + 8 + 20, sizeof(flags));
        check(flags == 0, "clean RTS0 flags are zero before tampering");
        std::memcpy(tampered.data() + off + 8 + 20, &kXboxFlags, sizeof(kXboxFlags));
        found_rts0 = true;
    }
    check(found_rts0, "RTS0 chunk found in the serialized blob");
    if (!found_rts0) return;

    // An Xbox-built blob was hashed with the high byte set: give the tampered copy a valid DXBC hash, as the
    // console serializer would. Checked first against the hash the PC serializer wrote.
    uint32_t hash[4];
    rt::d3d12_xs::dxbc_checksum(bytes, size, hash);
    check(std::memcmp(hash, bytes + 4, sizeof(hash)) == 0, "dxbc_checksum matches the PC serializer's hash");
    {
        // 0..7 root-constant parameters (24 bytes each) step the hashed length through all eight 8-byte slots
        // of the last MD5 block, covering both sides of the 56-byte tail boundary.
        D3D12_ROOT_PARAMETER params[7] = {};
        for (UINT i = 0; i < 7; ++i) {
            params[i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
            params[i].Constants.ShaderRegister = i;
            params[i].Constants.Num32BitValues = 1;
        }
        bool all_match = true;
        for (UINT n = 0; n <= 7; ++n) {
            D3D12_ROOT_SIGNATURE_DESC sized = make_legacy_desc(0);
            sized.NumParameters = n;
            sized.pParameters = n ? params : nullptr;
            Blobs sb;
            if (FAILED(D3D12SerializeRootSignature(&sized, D3D_ROOT_SIGNATURE_VERSION_1_0, &sb.blob, &sb.error)) ||
                !sb.blob) {
                all_match = false;
                continue;
            }
            const uint8_t* sbytes = static_cast<const uint8_t*>(sb.blob->GetBufferPointer());
            const SIZE_T ssize = sb.blob->GetBufferSize();
            uint32_t shash[4];
            rt::d3d12_xs::dxbc_checksum(sbytes, ssize, shash);
            if (std::memcmp(shash, sbytes + 4, sizeof(shash)) != 0) all_match = false;
        }
        check(all_match, "dxbc_checksum matches the PC serializer's hash for every tail length");
    }
    rt::d3d12_xs::dxbc_checksum(tampered.data(), tampered.size(), hash);
    std::memcpy(tampered.data() + 4, hash, sizeof(hash));

    std::vector<uint8_t> cleared;
    const bool rewritten = rt::d3d12_xs::copy_rts0_blob_clear_xbox_flags(tampered.data(), tampered.size(), cleared);
    check(rewritten, "blob helper reports a rewrite for the Xbox high byte");
    if (rewritten) {
        uint32_t cleared_flags = 0;
        for (uint32_t i = 0; i < chunk_count; ++i) {
            uint32_t off = 0;
            std::memcpy(&off, cleared.data() + 32 + i * 4, sizeof(off));
            if (off > cleared.size() || cleared.size() - off < 8 + 24) continue;
            if (std::memcmp(cleared.data() + off, "RTS0", 4) == 0) {
                std::memcpy(&cleared_flags, cleared.data() + off + 8 + 20, sizeof(cleared_flags));
            }
        }
        check(cleared_flags == 0, "blob helper cleared the Xbox high byte");
    }

    {
        ID3D12RootSignatureDeserializer* direct = nullptr;
        hr = D3D12CreateRootSignatureDeserializer(tampered.data(), tampered.size(), IID_PPV_ARGS(&direct));
        if (SUCCEEDED(hr) && direct) {
            // Some runtimes accept the high byte without clearing it, which is
            // exactly the bug the shim fixes. Newer runtimes reject it here.
            check(direct->GetRootSignatureDesc()->Flags ==
                      static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(kXboxFlags),
                  "host deserializer, when it accepts, keeps the Xbox high byte");
            direct->Release();
        } else {
            std::printf("host deserializer rejects the Xbox high byte (runtime-dependent)\n");
        }
    }

    using DeserializeFn = HRESULT(WINAPI*)(const void*, SIZE_T, REFIID, void**);
    auto deserialize = reinterpret_cast<DeserializeFn>(
        rt::shims::find("d3d12_xs.dll", "D3D12CreateRootSignatureDeserializer"));
    check(deserialize != nullptr, "shim registry exposes d3d12_xs!D3D12CreateRootSignatureDeserializer");
    if (!deserialize) return;

    ID3D12RootSignatureDeserializer* shim_de = nullptr;
    hr = deserialize(tampered.data(), tampered.size(), IID_PPV_ARGS(&shim_de));
    check(SUCCEEDED(hr), "shim deserializer clears the Xbox high byte and succeeds");
    if (SUCCEEDED(hr) && shim_de) {
        const D3D12_ROOT_SIGNATURE_DESC* d = shim_de->GetRootSignatureDesc();
        check(d->Flags == 0, "shim deserializer returns flags with the high byte cleared");
        shim_de->Release();
    }
}

}  // namespace

int main() {
    std::printf("root_sig_test: PC d3d12.dll root-signature flag handling\n");
    const bool supports_1_2 = supports_root_signature_1_2();
    test_versioned_1_0();
    test_versioned_1_1();
    test_versioned_1_2(supports_1_2);
    test_downgrade_1_2_to_1_1(supports_1_2);
    test_legacy_entry_point();
    test_preserves_low_bits_and_unknown_versions();
    test_registered_shim(supports_1_2);
    test_deserializer();
    std::printf("root_sig_test: %s (%d failure%s)\n", g_failures ? "FAILED" : "OK", g_failures,
                g_failures == 1 ? "" : "s");
    return g_failures ? 1 : 0;
}
