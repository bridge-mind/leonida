// Windows oracle for the RTS0 parser hypothesis (needs a real PC d3d12.lib).
//
// Builds synthetic versioned root-signature descriptors by hand (no game
// data), serializes each with Microsoft's D3D12SerializeVersionedRootSignature,
// and requires rt::rootsig::parse_rts0 to recover the same model. If the
// hypothesis layout in root_signature.cpp ever disagrees with the real
// serializer, this fails loudly here instead of corrupting draws silently.
// Versions 1.0, 1.1 and 1.2 are each checked with static samplers (1.2 only where the host serializes it).
// NOT portable: runs on Windows CI only, like root_sig_test.cpp.
#include <windows.h>
#include <d3d12.h>

#include <cstdint>
#include <cstdio>
#include <vector>

#include "root_signature.h"

namespace {

int g_failures = 0;

void check(bool condition, const char* what) {
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

// One line per assertion: "case: what", so a failure names the signature and the field instead of just "false".
void check_named(bool condition, const char* name, const char* what) {
    char line[256];
    std::snprintf(line, sizeof(line), "%s: %s", name, what);
    check(condition, line);
}

struct Blobs {
    ID3DBlob* blob = nullptr;
    ID3DBlob* error = nullptr;
    ~Blobs() {
        if (blob) blob->Release();
        if (error) error->Release();
    }
};

D3D12_VERSIONED_ROOT_SIGNATURE_DESC make_desc(
    const D3D12_ROOT_PARAMETER1* params, uint32_t nparams) {
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC desc{};
    desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
    desc.Desc_1_1.NumParameters = nparams;
    desc.Desc_1_1.pParameters = params;
    desc.Desc_1_1.NumStaticSamplers = 0;
    desc.Desc_1_1.pStaticSamplers = nullptr;
    desc.Desc_1_1.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;
    return desc;
}

bool parse_blob(ID3DBlob* blob, rt::rootsig::RootSignature& model) {
    const auto* bytes = static_cast<const uint8_t*>(blob->GetBufferPointer());
    const size_t size = blob->GetBufferSize();
    // Diagnostics first: if the layout hypothesis is wrong, the CI log shows
    // the real bytes instead of just a crash or a bare failure.
    std::printf("oracle blob: %zu bytes: ", size);
    for (size_t i = 0; i < 32 && i < size; ++i) {
        std::printf("%02x", bytes[i]);
    }
    std::printf("\n");
    if (GetEnvironmentVariableA("LEONIDA_ORACLE_DUMP", nullptr, 0)) {  // the whole blob as u32 words, for layouts
        for (size_t i = 0; i + 4 <= size; i += 4)
            std::printf("  +%3zu: %08x\n", i, unsigned(bytes[i] | bytes[i + 1] << 8 | bytes[i + 2] << 16 | bytes[i + 3] << 24));
    }
    std::fflush(stdout);
    const rt::rootsig::DecodeResult d = rt::rootsig::parse_rts0(bytes, size);
    if (!d.ok) {
        std::printf("oracle parse failed: %s\n", d.error.c_str());
        return false;
    }
    model = d.sig;
    return true;
}

// Parses and logs "case: parses"; false means the caller must not index the model (that segfaulted CI).
bool parse_checked(ID3DBlob* blob, const char* name, rt::rootsig::RootSignature& model) {
    const bool parsed = parse_blob(blob, model);
    check_named(parsed, name, "parses");
    return parsed;
}

// Serializes one synthetic desc and, on failure, prints the runtime's own complaint: that text is what tells a
// rejected flag apart from a rejected layout.
bool serialize_desc(const D3D12_VERSIONED_ROOT_SIGNATURE_DESC& desc, Blobs& out, const char* name) {
    const HRESULT hr = D3D12SerializeVersionedRootSignature(&desc, &out.blob, &out.error);
    check_named(hr == S_OK, name, "Microsoft serializer accepts it");
    if (hr != S_OK) {
        if (out.error != nullptr) {
            std::printf("serializer error: %s\n",
                        static_cast<const char*>(out.error->GetBufferPointer()));
        }
        return false;
    }
    return true;
}

// Every field of one static sampler, plus whether this root-signature version carries the 1.2 flags dword.
bool same_sampler(const rt::rootsig::StaticSampler& got, const D3D12_STATIC_SAMPLER_DESC1& want, bool flags) {
    return got.filter == uint32_t(want.Filter) && got.address_u == uint32_t(want.AddressU) &&
           got.address_v == uint32_t(want.AddressV) && got.address_w == uint32_t(want.AddressW) &&
           got.mip_lod_bias == want.MipLODBias && got.max_anisotropy == want.MaxAnisotropy &&
           got.comparison_func == uint32_t(want.ComparisonFunc) &&
           got.border_color == uint32_t(want.BorderColor) && got.min_lod == want.MinLOD &&
           got.max_lod == want.MaxLOD && got.shader_register == want.ShaderRegister &&
           got.register_space == want.RegisterSpace &&
           got.visibility == static_cast<rt::rootsig::Visibility>(want.ShaderVisibility) &&
           got.flags == (flags ? uint32_t(want.Flags) : 0u);
}

}  // namespace

int main() {
    using namespace rt::rootsig;

    // 4 constants (VS-visible) + 1 CBV: parses back to the same model and
    // packs to 6 dwords, exactly like the portable packer says for 1.1.
    D3D12_ROOT_PARAMETER1 params[2]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;
    params[0].Constants.ShaderRegister = 0;
    params[0].Constants.RegisterSpace = 0;
    params[0].Constants.Num32BitValues = 4;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[1].Descriptor.ShaderRegister = 2;
    params[1].Descriptor.RegisterSpace = 0;
    params[1].Descriptor.Flags = D3D12_ROOT_DESCRIPTOR_FLAG_NONE;

    const D3D12_VERSIONED_ROOT_SIGNATURE_DESC desc = make_desc(params, 2);
    Blobs blobs;
    const HRESULT hr =
        D3D12SerializeVersionedRootSignature(&desc, &blobs.blob, &blobs.error);
    check(hr == S_OK, "Microsoft serializer accepts the synthetic desc");
    if (hr != S_OK) {
        if (blobs.error != nullptr) {
            std::printf("serializer error: %s\n",
                        static_cast<const char*>(blobs.error->GetBufferPointer()));
        }
        std::printf("%d FAILURES\n", g_failures + 1);
        return 1;
    }

    RootSignature model;
    check(parse_blob(blobs.blob, model), "oracle blob parses");
    if (g_failures != 0) {
        // Never index into a model that failed to parse (that segfaulted CI).
        std::printf("%d FAILURES\n", g_failures);
        return 1;
    }
    check(model.params.size() == 2, "oracle model has 2 params");
    check(model.params[0].type == ParamType::Constants &&
              model.params[0].num_values == 4 &&
              model.params[0].visibility == Visibility::Vertex,
          "oracle constants match");
    check(model.params[1].type == ParamType::Descriptor &&
              model.params[1].descriptor_register == 2 &&
              model.params[1].descriptor_space == 0,
          "oracle CBV matches");
    const PackResult packed = pack(model);
    check(packed.ok && packed.dword_count == 6, "oracle model packs to 6 dwords");

    // The descriptor count and range offset are independent. The final SRV range is unbounded
    // and begins immediately after the fixed range, but the whole table still occupies one GPU address.
    {
        D3D12_DESCRIPTOR_RANGE1 ranges[] = {
            {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 2, 3, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DATA_STATIC, 0},
            {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, UINT32_MAX, 7, 1,
             D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND},
        };
        D3D12_ROOT_PARAMETER1 table_and_constants[2] = {{}, params[0]};
        auto& table = table_and_constants[0];
        table.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        table.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
        table.DescriptorTable = {2, ranges};
        const auto unbounded_desc = make_desc(table_and_constants, 2);
        Blobs unbounded;
        const HRESULT result =
            D3D12SerializeVersionedRootSignature(&unbounded_desc, &unbounded.blob, &unbounded.error);
        check(result == S_OK, "Microsoft serializer accepts an unbounded SRV range");
        RootSignature table_model;
        if (SUCCEEDED(result) && unbounded.blob &&
            parse_blob(unbounded.blob, table_model) && table_model.params.size() == 2 &&
            table_model.params[0].ranges.size() == 2) {
            const auto& fixed = table_model.params[0].ranges[0];
            const auto& tail = table_model.params[0].ranges[1];
            check(table_model.params[0].type == ParamType::Table &&
                      table_model.params[0].visibility == Visibility::Pixel &&
                      fixed.num_descriptors == 2 && fixed.offset_in_table == 0 &&
                      tail.range_type == D3D12_DESCRIPTOR_RANGE_TYPE_SRV &&
                      tail.num_descriptors == UINT32_MAX && tail.base_shader_register == 7 &&
                      tail.register_space == 1 && tail.flags == D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE &&
                      tail.offset_in_table == kRangeOffsetAppend,
                  "fixed and unbounded SRV ranges survive native serialization");
            const auto table_pack = pack(table_model);
            check(table_pack.ok && table_pack.d3d12_cost == 5 && table_pack.dword_count == 6 &&
                      table_pack.provenance.size() == 6 &&
                      table_pack.provenance[1].param_index == 0 && table_pack.provenance[2].param_index == 1,
                  "unbounded SRV table occupies two SGPRs before the following constants");
        } else {
            check(false, "native unbounded table parses with two ranges and two parameters");
        }
    }

    // Static samplers, one signature per version: every field round-trips through the real serializer, and 1.2
    // adds D3D12_STATIC_SAMPLER_DESC1::Flags (a 14th dword) to each sampler record. Each also carries a table and a
    // root descriptor, so the 1.1/1.2 descriptor and range flags sit between the samplers and the header.
    D3D12_STATIC_SAMPLER_DESC1 samplers[2]{};
    samplers[0].Filter = D3D12_FILTER_ANISOTROPIC;
    samplers[0].AddressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    samplers[0].AddressV = D3D12_TEXTURE_ADDRESS_MODE_MIRROR;
    samplers[0].AddressW = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    samplers[0].MipLODBias = -1.5f;
    samplers[0].MaxAnisotropy = 8;
    samplers[0].ComparisonFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
    samplers[0].BorderColor = D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE;
    samplers[0].MinLOD = 1.0f;
    samplers[0].MaxLOD = 9.0f;
    samplers[0].ShaderRegister = 4;
    samplers[0].RegisterSpace = 2;
    samplers[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    samplers[1].Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
    samplers[1].AddressU = samplers[1].AddressV = samplers[1].AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    samplers[1].MaxLOD = D3D12_FLOAT32_MAX;
    samplers[1].ShaderRegister = 7;
    samplers[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    samplers[1].BorderColor = D3D12_STATIC_BORDER_COLOR_OPAQUE_BLACK;
    D3D12_STATIC_SAMPLER_DESC samplers0[2]{};
    for (int i = 0; i < 2; ++i) {
        const D3D12_STATIC_SAMPLER_DESC1& s = samplers[i];
        samplers0[i] = {s.Filter, s.AddressU, s.AddressV, s.AddressW, s.MipLODBias, s.MaxAnisotropy,
                        s.ComparisonFunc, s.BorderColor, s.MinLOD, s.MaxLOD, s.ShaderRegister, s.RegisterSpace,
                        s.ShaderVisibility};
    }
    // Two ranges, so the range stride (20 bytes in 1.0, 24 with flags) is under test too. Every field differs from
    // its neighbours (an explicit table offset, nonzero registers and spaces, distinct flags), so a swapped or
    // shifted field read shows up instead of matching by accident.
    D3D12_DESCRIPTOR_RANGE range0[2] = {
        {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 3, 1, 0, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND},
        {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 2, 4, 2, 7}};
    D3D12_DESCRIPTOR_RANGE1 range1[2] = {
        {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 3, 1, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DATA_STATIC,
         D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND},
        {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 2, 4, 2, D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE, 7}};
    D3D12_ROOT_PARAMETER params0[2]{};
    params0[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params0[0].DescriptorTable = {2, range0};
    params0[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    params0[1].Descriptor = {5, 1};
    D3D12_ROOT_PARAMETER1 params1[2]{};
    params1[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params1[0].DescriptorTable = {2, range1};
    params1[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    params1[1].Descriptor = {5, 1, D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE};

    // Ranges as the model should hold them: 1.0 has no range flags (0 in the model); 1.1 and 1.2 use range1.
    const auto same_ranges = [&](const std::vector<DescriptorRange>& got, bool range_flags) {
        if (got.size() != 2) return false;
        for (int i = 0; i < 2; ++i) {
            const D3D12_DESCRIPTOR_RANGE1& want = range1[i];
            if (got[i].range_type != uint32_t(want.RangeType) || got[i].num_descriptors != want.NumDescriptors ||
                got[i].base_shader_register != want.BaseShaderRegister ||
                got[i].register_space != want.RegisterSpace ||
                got[i].flags != (range_flags ? uint32_t(want.Flags) : 0u) ||
                got[i].offset_in_table != want.OffsetInDescriptorsFromTableStart)
                return false;
        }
        return true;
    };
    const auto serialize = [&](const D3D12_VERSIONED_ROOT_SIGNATURE_DESC& d, Version version, const char* name,
                               bool sampler_flags) {
        const bool range_flags = version != Version::V1_0;
        Blobs out;
        if (!serialize_desc(d, out, name)) return;
        RootSignature got;
        if (!parse_checked(out.blob, name, got)) return;
        char what[160];
        std::snprintf(what, sizeof(what), "%s: version, a two-range table, a root SRV and two samplers", name);
        check(got.version == version && got.params.size() == 2 && got.params[0].type == ParamType::Table &&
                  got.params[0].ranges.size() == 2 && got.params[0].ranges[0].range_type == 0 &&
                  got.params[0].ranges[0].num_descriptors == 3 && got.params[0].ranges[1].range_type == 1 &&
                  got.params[0].ranges[1].num_descriptors == 2 &&
                  got.params[1].type == ParamType::Descriptor && got.params[1].descriptor_register == 5 &&
                  got.params[1].descriptor_space == 1 && got.static_samplers.size() == 2,
              what);
        if (got.static_samplers.size() != 2) return;
        std::snprintf(what, sizeof(what), "%s: every range field round-trips (flags before the table offset)", name);
        check(same_ranges(got.params[0].ranges, range_flags), what);
        std::snprintf(what, sizeof(what), "%s: every sampler field round-trips", name);
        check(same_sampler(got.static_samplers[0], samplers[0], sampler_flags) &&
                  same_sampler(got.static_samplers[1], samplers[1], sampler_flags),
              what);
        // The canonical cache codec keeps all of it, flags included.
        const std::vector<uint8_t> bytes = encode_canonical(got);
        const DecodeResult again = decode_canonical(bytes.data(), bytes.size());
        std::snprintf(what, sizeof(what), "%s: canonical codec round-trips the parsed model", name);
        check(again.ok && again.sig.version == version && again.sig.static_samplers.size() == 2 &&
                  again.sig.params.size() == 2 && same_ranges(again.sig.params[0].ranges, range_flags) &&
                  same_sampler(again.sig.static_samplers[0], samplers[0], sampler_flags) &&
                  same_sampler(again.sig.static_samplers[1], samplers[1], sampler_flags),
              what);
    };

    D3D12_VERSIONED_ROOT_SIGNATURE_DESC v10{};
    v10.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    v10.Desc_1_0 = {2, params0, 2, samplers0, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    serialize(v10, Version::V1_0, "1.0 with static samplers", false);

    D3D12_VERSIONED_ROOT_SIGNATURE_DESC v11{};
    v11.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
    v11.Desc_1_1 = {2, params1, 2, samplers0, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    serialize(v11, Version::V1_1, "1.1 with static samplers", false);

    // An empty signature (no parameters, no samplers) is legal and is the shape of a bindless compute signature.
    // It proves the header's count and offset words are believed when both are zero instead of being read as a
    // truncated table, and the canonical codec has a model to keep.
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC empty11{};
    empty11.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
    empty11.Desc_1_1 = {0, nullptr, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    {
        Blobs out;
        RootSignature got;
        if (serialize_desc(empty11, out, "1.1 empty") && parse_checked(out.blob, "1.1 empty", got)) {
            check_named(got.version == Version::V1_1 && got.params.empty() && got.static_samplers.empty() &&
                            got.flags == 0,
                        "1.1 empty", "version 1.1, no parameters, no samplers, flags 0");
            const std::vector<uint8_t> bytes = encode_canonical(got);
            const DecodeResult again = decode_canonical(bytes.data(), bytes.size());
            check_named(again.ok && again.sig.params.empty() && again.sig.static_samplers.empty() &&
                            again.sig.flags == 0,
                        "1.1 empty", "canonical codec round-trips the empty model");
        }
    }

    // Every root-signature flag that a signature with no parameters can legally carry, in one blob. A graphics
    // signature sets ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT and a bindless one sets the heap-direct bits, so a parser
    // that read the flags from the wrong header word would pick the other binding model without failing anywhere
    // else. There are no parameters because the DENY_* bits are invalid next to a parameter any of those stages
    // can see.
    const uint32_t every_pc_flag =
        D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT |
        D3D12_ROOT_SIGNATURE_FLAG_DENY_VERTEX_SHADER_ROOT_ACCESS |
        D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS |
        D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS |
        D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS |
        D3D12_ROOT_SIGNATURE_FLAG_DENY_PIXEL_SHADER_ROOT_ACCESS |
        D3D12_ROOT_SIGNATURE_FLAG_ALLOW_STREAM_OUTPUT |
        D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED |
        D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED;
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC flagged11{};
    flagged11.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
    flagged11.Desc_1_1 = {0, nullptr, 0, nullptr, static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(every_pc_flag)};
    {
        Blobs out;
        RootSignature got;
        if (serialize_desc(flagged11, out, "1.1 with every root-signature flag") &&
            parse_checked(out.blob, "1.1 with every root-signature flag", got)) {
            check_named(got.flags == every_pc_flag, "1.1 with every root-signature flag",
                        "the whole PC flag word round-trips (heap-direct bits included)");
            const std::vector<uint8_t> bytes = encode_canonical(got);
            const DecodeResult again = decode_canonical(bytes.data(), bytes.size());
            check_named(again.ok && again.sig.flags == every_pc_flag, "1.1 with every root-signature flag",
                        "canonical codec keeps the flag word");
        }
    }

    // A 1.0 signature with all three root view types and a table last. A root UAV opens most compute signatures,
    // 1.0 root views have no flags dword, and the table last means its payload cannot be found by a fixed stride.
    D3D12_DESCRIPTOR_RANGE view_range =
        {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 4, 0, 0, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND};
    D3D12_ROOT_PARAMETER view_params[4]{};
    view_params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    view_params[0].Descriptor = {2, 3};
    view_params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    view_params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    view_params[1].Descriptor = {5, 1};
    view_params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;
    view_params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    view_params[2].Descriptor = {7, 2};
    view_params[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    view_params[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    view_params[3].DescriptorTable = {1, &view_range};
    view_params[3].ShaderVisibility = D3D12_SHADER_VISIBILITY_GEOMETRY;
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC views10{};
    views10.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    views10.Desc_1_0 = {4, view_params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    {
        Blobs out;
        RootSignature got;
        if (serialize_desc(views10, out, "1.0 root CBV, SRV and UAV with a table last") &&
            parse_checked(out.blob, "1.0 root CBV, SRV and UAV with a table last", got)) {
            check_named(got.params.size() == 4 &&
                            got.params[0].type == ParamType::Descriptor &&
                            got.params[0].descriptor_register == 2 && got.params[0].descriptor_space == 3 &&
                            uint32_t(got.params[0].visibility) == uint32_t(D3D12_SHADER_VISIBILITY_ALL) &&
                            got.params[1].type == ParamType::Descriptor &&
                            got.params[1].descriptor_register == 5 && got.params[1].descriptor_space == 1 &&
                            uint32_t(got.params[1].visibility) == uint32_t(D3D12_SHADER_VISIBILITY_VERTEX) &&
                            got.params[2].type == ParamType::Descriptor &&
                            got.params[2].descriptor_register == 7 && got.params[2].descriptor_space == 2 &&
                            uint32_t(got.params[2].visibility) == uint32_t(D3D12_SHADER_VISIBILITY_PIXEL),
                        "1.0 root CBV, SRV and UAV with a table last",
                        "each root view type keeps its register, space and stage");
            check_named(got.params.size() == 4 && got.params[3].type == ParamType::Table &&
                            uint32_t(got.params[3].visibility) == uint32_t(D3D12_SHADER_VISIBILITY_GEOMETRY) &&
                            got.params[3].ranges.size() == 1 &&
                            got.params[3].ranges[0].range_type == 0 &&
                            got.params[3].ranges[0].num_descriptors == 4 &&
                            got.params[3].ranges[0].offset_in_table == kRangeOffsetAppend,
                        "1.0 root CBV, SRV and UAV with a table last",
                        "the table after three views keeps its range");
            const PackResult packed_views = pack(got);
            check_named(packed_views.ok && packed_views.dword_count == 8,
                        "1.0 root CBV, SRV and UAV with a table last",
                        "three views and a table use 8 user dwords in parameter order");
        }
    }

    // 64 constants is the largest legal root-constant parameter and exactly fills the 64-dword D3D12 budget. It is
    // legal D3D12 and impossible for the console's 32 user SGPRs, so the parse must recover all 64 and pack must
    // refuse the signature by name instead of truncating every draw at 32.
    D3D12_ROOT_PARAMETER1 huge_constants[1]{};
    huge_constants[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    huge_constants[0].Constants = {0, 1, 64};
    huge_constants[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC constants64{};
    constants64.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
    constants64.Desc_1_1 = {1, huge_constants, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    {
        Blobs out;
        RootSignature got;
        if (serialize_desc(constants64, out, "1.1 with 64 root constants") &&
            parse_checked(out.blob, "1.1 with 64 root constants", got)) {
            check_named(got.params.size() == 1 && got.params[0].type == ParamType::Constants &&
                            got.params[0].num_values == 64 && got.params[0].shader_register == 0 &&
                            got.params[0].register_space == 1,
                        "1.1 with 64 root constants",
                        "all 64 constant dwords and the non-zero space survive");
            const PackResult packed_huge = pack(got);
            check_named(!packed_huge.ok && packed_huge.error.find("spill") != std::string::npos,
                        "1.1 with 64 root constants",
                        "pack refuses 64 constants as spill instead of truncating");
        }
    }

    // All eight D3D12 shader-visibility values in one signature, with the constant registers in descending order.
    // The existing cases only reach ALL, VERTEX and PIXEL; a game with amplification and mesh shaders binds root
    // parameters to stages 6 and 7, which no oracle case had put through the serializer before.
    const D3D12_SHADER_VISIBILITY stages[8] = {
        D3D12_SHADER_VISIBILITY_ALL, D3D12_SHADER_VISIBILITY_VERTEX, D3D12_SHADER_VISIBILITY_HULL,
        D3D12_SHADER_VISIBILITY_DOMAIN, D3D12_SHADER_VISIBILITY_GEOMETRY, D3D12_SHADER_VISIBILITY_PIXEL,
        D3D12_SHADER_VISIBILITY_AMPLIFICATION, D3D12_SHADER_VISIBILITY_MESH};
    D3D12_ROOT_PARAMETER1 staged_params[8]{};
    for (int i = 0; i < 8; ++i) {
        staged_params[i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        staged_params[i].Constants = {static_cast<UINT>(7 - i), 0, 1};
        staged_params[i].ShaderVisibility = stages[i];
    }
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC staged11{};
    staged11.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
    staged11.Desc_1_1 = {8, staged_params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    {
        Blobs out;
        RootSignature got;
        if (serialize_desc(staged11, out, "1.1 with all eight visibilities") &&
            parse_checked(out.blob, "1.1 with all eight visibilities", got)) {
            bool stages_match = got.params.size() == 8;
            for (uint32_t i = 0; stages_match && i < 8; ++i) {
                stages_match = got.params[i].type == ParamType::Constants &&
                               got.params[i].num_values == 1 &&
                               got.params[i].shader_register == 7 - i &&
                               uint32_t(got.params[i].visibility) == uint32_t(stages[i]);
            }
            check_named(stages_match, "1.1 with all eight visibilities",
                        "every stage value keeps its parameter and the descending registers stay in order");
            const PackResult packed_stages = pack(got);
            check_named(packed_stages.ok && packed_stages.dword_count == 8,
                        "1.1 with all eight visibilities",
                        "eight single-dword constants pack in parameter order");
        }
    }

    // Every descriptor range type, with an unbounded SRV range and a non-zero base register and space on each.
    // Bindless signatures declare "all descriptors from here on" with NumDescriptors == UINT_MAX, and D3D12 forces
    // sampler ranges into a table of their own (the serializer rejects a mixed table), so the signature carries two
    // tables; a parser that treated UINT_MAX as an error or a signed count would lose the tail of every bindless
    // table.
    D3D12_DESCRIPTOR_RANGE1 resource_ranges[3] = {
        {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0xFFFFFFFFu, 0, 1,
         D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND},
        {D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 1, 2, 3, D3D12_DESCRIPTOR_RANGE_FLAG_NONE, 0},
        {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 8, 1, 2,
         static_cast<D3D12_DESCRIPTOR_RANGE_FLAGS>(D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE |
                                                   D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE),
         16}};
    D3D12_DESCRIPTOR_RANGE1 sampler_range = {
        D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, 4, 1, 4, D3D12_DESCRIPTOR_RANGE_FLAG_NONE,
        D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND};
    D3D12_ROOT_PARAMETER1 range_params[2]{};
    range_params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    range_params[0].DescriptorTable = {3, resource_ranges};
    range_params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    range_params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    range_params[1].DescriptorTable = {1, &sampler_range};
    range_params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC ranges11{};
    ranges11.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
    ranges11.Desc_1_1 = {2, range_params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    {
        Blobs out;
        RootSignature got;
        if (serialize_desc(ranges11, out, "1.1 with all four range types") &&
            parse_checked(out.blob, "1.1 with all four range types", got)) {
            if (got.params.size() != 2 || got.params[0].ranges.size() != 3 ||
                got.params[1].ranges.size() != 1) {
                check_named(false, "1.1 with all four range types", "two tables: three resources, one sampler");
            } else {
                const std::vector<DescriptorRange>& ranges = got.params[0].ranges;
                check_named(ranges[0].range_type == 0 && ranges[0].num_descriptors == 0xFFFFFFFFu &&
                                ranges[0].base_shader_register == 0 && ranges[0].register_space == 1 &&
                                ranges[0].flags == D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE &&
                                ranges[0].offset_in_table == kRangeOffsetAppend,
                            "1.1 with all four range types",
                            "unbounded SRV range keeps UINT_MAX, its space, flags and appended offset");
                check_named(ranges[1].range_type == 2 && ranges[1].num_descriptors == 1 &&
                                ranges[1].base_shader_register == 2 && ranges[1].register_space == 3 &&
                                ranges[1].flags == 0 && ranges[1].offset_in_table == 0,
                            "1.1 with all four range types",
                            "CBV range keeps base 2 space 3 and an explicit table offset of 0");
                check_named(ranges[2].range_type == 1 && ranges[2].num_descriptors == 8 &&
                                ranges[2].base_shader_register == 1 && ranges[2].register_space == 2 &&
                                ranges[2].flags ==
                                    uint32_t(D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE |
                                             D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE) &&
                                ranges[2].offset_in_table == 16,
                            "1.1 with all four range types",
                            "UAV range keeps the combined data and descriptor flags before offset 16");
                check_named(got.params[1].ranges[0].range_type == 3 &&
                                got.params[1].ranges[0].num_descriptors == 4 &&
                                got.params[1].ranges[0].base_shader_register == 1 &&
                                got.params[1].ranges[0].register_space == 4 &&
                                got.params[1].ranges[0].offset_in_table == kRangeOffsetAppend,
                            "1.1 with all four range types",
                            "the sampler table keeps its own range, base register and space");
                const std::vector<uint8_t> bytes = encode_canonical(got);
                const DecodeResult again = decode_canonical(bytes.data(), bytes.size());
                check_named(again.ok && again.sig.params.size() == 2 &&
                                again.sig.params[0].ranges.size() == 3 &&
                                again.sig.params[0].ranges[0].num_descriptors == 0xFFFFFFFFu &&
                                again.sig.params[0].ranges[2].flags ==
                                    uint32_t(D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE |
                                             D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE) &&
                                again.sig.params[1].ranges.size() == 1 &&
                                again.sig.params[1].ranges[0].range_type == 3,
                            "1.1 with all four range types",
                            "canonical codec keeps the unbounded count, the flags and the second table");
            }
        }
    }

    // A shadow sampler: comparison filter, MIRROR_ONCE addressing, a non-zero min LOD and GEOMETRY visibility. The
    // existing samplers are anisotropic and point-clamp, so no comparison filter or MIRROR_ONCE value reached the
    // parser; the single root constant beside it uses a non-zero register and space, which the first case cannot
    // distinguish from a swapped register/space read.
    D3D12_STATIC_SAMPLER_DESC1 shadow_sampler{};
    shadow_sampler.Filter = D3D12_FILTER_COMPARISON_ANISOTROPIC;
    shadow_sampler.AddressU = D3D12_TEXTURE_ADDRESS_MODE_MIRROR_ONCE;
    shadow_sampler.AddressV = D3D12_TEXTURE_ADDRESS_MODE_MIRROR_ONCE;
    shadow_sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_MIRROR_ONCE;
    shadow_sampler.MipLODBias = 0.25f;
    shadow_sampler.MaxAnisotropy = 16;
    shadow_sampler.ComparisonFunc = D3D12_COMPARISON_FUNC_GREATER_EQUAL;
    shadow_sampler.BorderColor = D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK;
    shadow_sampler.MinLOD = 0.5f;
    shadow_sampler.MaxLOD = 12.0f;
    shadow_sampler.ShaderRegister = 9;
    shadow_sampler.RegisterSpace = 5;
    shadow_sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_GEOMETRY;
    const D3D12_STATIC_SAMPLER_DESC shadow_sampler0 = {
        shadow_sampler.Filter, shadow_sampler.AddressU, shadow_sampler.AddressV, shadow_sampler.AddressW,
        shadow_sampler.MipLODBias, shadow_sampler.MaxAnisotropy, shadow_sampler.ComparisonFunc,
        shadow_sampler.BorderColor, shadow_sampler.MinLOD, shadow_sampler.MaxLOD, shadow_sampler.ShaderRegister,
        shadow_sampler.RegisterSpace, shadow_sampler.ShaderVisibility};
    D3D12_ROOT_PARAMETER1 shadow_params[1]{};
    shadow_params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    shadow_params[0].Constants = {3, 2, 1};
    shadow_params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC shadow11{};
    shadow11.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
    shadow11.Desc_1_1 = {1, shadow_params, 1, &shadow_sampler0, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    {
        Blobs out;
        RootSignature got;
        if (serialize_desc(shadow11, out, "1.1 with a comparison sampler") &&
            parse_checked(out.blob, "1.1 with a comparison sampler", got)) {
            check_named(got.params.size() == 1 && got.params[0].type == ParamType::Constants &&
                            got.params[0].num_values == 1 && got.params[0].shader_register == 3 &&
                            got.params[0].register_space == 2,
                        "1.1 with a comparison sampler",
                        "one constant at b3 space 2 keeps every field");
            check_named(got.static_samplers.size() == 1 &&
                            same_sampler(got.static_samplers[0], shadow_sampler, false),
                        "1.1 with a comparison sampler",
                        "comparison filter, MIRROR_ONCE, GEOMETRY and every other sampler field round-trip");
        }
    }

    // The 1.2 cases below need a host that serializes 1.2. Windows 10's built-in d3d12.dll stops at 1.1 and rejects
    // even an empty 1.2 description; their sampler flags have no 1.1 form, so they are skipped there.
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC empty12{};
    empty12.Version = D3D_ROOT_SIGNATURE_VERSION_1_2;
    Blobs probe;
    if (FAILED(D3D12SerializeVersionedRootSignature(&empty12, &probe.blob, &probe.error))) {
        std::printf("SKIP 1.2 with sampler flags, 1.2 with non-normalized coordinates (the host d3d12.dll can't "
                    "serialize root signature 1.2, e.g. Windows 10)\n");
        if (g_failures != 0) std::printf("%d FAILURES\n", g_failures);
        return g_failures == 0 ? 0 : 1;
    }

    // 1.2: a UINT border colour on one sampler, no flags on the other.
    samplers[0].BorderColor = D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE_UINT;
    samplers[0].Flags = D3D12_SAMPLER_FLAG_UINT_BORDER_COLOR;
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC v12{};
    v12.Version = D3D_ROOT_SIGNATURE_VERSION_1_2;
    v12.Desc_1_2 = {2, params1, 2, samplers, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    serialize(v12, Version::V1_2, "1.2 with sampler flags", true);

    // 1.2: non-normalized coordinates, which D3D12 allows only with point/linear filtering, no mips, no anisotropy,
    // no comparison and clamp or border addressing.
    samplers[0] = {};
    samplers[0].Filter = D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT;
    samplers[0].AddressU = samplers[0].AddressV = samplers[0].AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    samplers[0].ShaderRegister = 4;
    samplers[0].RegisterSpace = 2;
    samplers[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    samplers[0].BorderColor = D3D12_STATIC_BORDER_COLOR_OPAQUE_BLACK;
    samplers[0].Flags = D3D12_SAMPLER_FLAG_NON_NORMALIZED_COORDINATES;
    serialize(v12, Version::V1_2, "1.2 with non-normalized coordinates", true);

    if (g_failures != 0) std::printf("%d FAILURES\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
