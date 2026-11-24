// Portable D3D12 root-signature model + packer to user SGPRs (s0..s31).
//
// No d3d12.h, no windows.h: this compiles on Linux CI and in the Windows
// runtime alike. It models exactly what docs/d3d12_vulkan.md section 3.5
// needs: root parameters in order, static samplers (no dwords), the 64-dword
// D3D12 cost limit, and the 32-dword console user-SGPR budget. Anything that
// does not fit is refused loudly, never guessed (see pack()).
//
// Numbering follows D3D12 docs: descriptor tables cost 1 dword of the 64-dword
// limit, root views (CBV/SRV/UAV) cost 2, 32-bit constants cost their count.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace rt::rootsig {

// D3D12_ROOT_SIGNATURE_VERSION subset the layer handles. 1.2 is 1.1 plus a flags dword on each static sampler.
enum class Version : uint32_t { V1_0 = 1, V1_1 = 2, V1_2 = 3 };

// D3D12_ROOT_PARAMETER_TYPE subset (order matches D3D12 docs).
enum class ParamType : uint32_t {
    Constants = 0,  // 32BIT_CONSTANTS: NumValues dwords
    Descriptor = 1,  // CBV/SRV/UAV root descriptor: 2 dwords (64-bit VA)
    Table = 2,  // DESCRIPTOR_TABLE: 2 dwords (table GPU VA)
};

// D3D12_SHADER_VISIBILITY (ALL 0, VERTEX 1, HULL 2, DOMAIN 3,
// GEOMETRY 4, PIXEL 5, AMPLIFICATION 6, MESH 7).
enum class Visibility : uint32_t {
    All = 0,
    Vertex = 1,
    Geometry = 4,
    Pixel = 5,
    Amplification = 6,
    Mesh = 7,
};

// D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND: the range starts right after the previous one in its table.
constexpr uint32_t kRangeOffsetAppend = 0xFFFFFFFFu;

// One descriptor range inside a table (SRV/CBV/UAV/sampler + count), with every field of D3D12_DESCRIPTOR_RANGE1.
struct DescriptorRange {
    uint32_t range_type = 0;  // 0 SRV, 1 UAV, 2 CBV, 3 sampler (D3D12 order)
    uint32_t num_descriptors = 0;
    uint32_t base_shader_register = 0;
    uint32_t register_space = 0;
    // D3D12_DESCRIPTOR_RANGE_FLAGS, 1.1 and 1.2 only. A 1.0 range has no flags dword; it stays 0 here.
    uint32_t flags = 0;
    uint32_t offset_in_table = 0;  // OffsetInDescriptorsFromTableStart, or kRangeOffsetAppend
};

// Static sampler: no user-SGPR dwords; registered in the hash table instead.
struct StaticSampler {
    uint32_t filter = 0;
    uint32_t address_u = 0;
    uint32_t address_v = 0;
    uint32_t address_w = 0;
    float mip_lod_bias = 0.0f;
    uint32_t max_anisotropy = 0;
    uint32_t comparison_func = 0;
    uint32_t border_color = 0;  // 0 transparent-black .. 4 opaque-white
    float min_lod = 0.0f;
    float max_lod = 0.0f;
    uint32_t shader_register = 0;
    uint32_t register_space = 0;
    Visibility visibility = Visibility::All;
    // D3D12_SAMPLER_FLAGS (D3D12_STATIC_SAMPLER_DESC1, version 1.2 only): 1 UINT border colour, 2 non-normalized
    // coordinates. Earlier versions have no such field; it stays 0 and the codecs neither write nor read it.
    uint32_t flags = 0;
};

struct RootParameter {
    ParamType type = ParamType::Constants;
    Visibility visibility = Visibility::All;
    // Constants:
    uint32_t num_values = 0;  // shader register + space describe the slot
    uint32_t shader_register = 0;
    uint32_t register_space = 0;
    // Root descriptor:
    uint32_t descriptor_register = 0;
    uint32_t descriptor_space = 0;
    // Descriptor table:
    std::vector<DescriptorRange> ranges;
};

struct RootSignature {
    Version version = Version::V1_1;
    std::vector<RootParameter> params;  // at most 64 root parameters (D3D12)
    std::vector<StaticSampler> static_samplers;
    uint32_t flags = 0;  // low 16 bits: PC meaning; high byte: Xbox type bits
};

// Where one packed dword came from (provenance for diagnostics).
struct DwordProvenance {
    uint32_t param_index = 0;  // index into RootSignature::params
    uint32_t sub_index = 0;  // constant index, or 0 = VA lo, 1 = VA hi
};

struct PackResult {
    bool ok = false;
    std::string error;  // set when !ok; names the refusing parameter
    std::array<uint32_t, 32> dwords{};  // valid dwords [0, dword_count)
    uint32_t dword_count = 0;
    std::vector<DwordProvenance> provenance;  // same length as dword_count
    uint32_t d3d12_cost = 0;  // cost against the 64-dword D3D12 limit
};

constexpr uint32_t kMaxUserSgprs = 32;
constexpr uint32_t kMaxD3d12Cost = 64;

// Packs root parameters in order. Static samplers consume no dwords.
// Fails (ok == false) when the signature needs more than 32 user SGPRs
// (spill layout unknown: refuse, do not guess) or exceeds the 64-dword
// D3D12 cost limit, or holds more than 64 root parameters.
[[nodiscard]] PackResult pack(const RootSignature& sig);

// The console layout of root arguments in a stage's user SGPRs (docs/d3d12_vulkan.md section 3.5, D3D-448).
//
// Parameters go in root-signature order from the stage's first root-argument SGPR, each while it fits whole:
// a 32-bit constant takes one SGPR per value, a root CBV/SRV/UAV two (its 64-bit VA), a descriptor table ONE
// (the 32-bit VA of its first descriptor; the heaps live below 4 GiB). Static samplers take none.
//
// How each part is known:
//   - compute, base s4: measured. The console compiler's output (section 7) and the package's FidelityFX DLL, whose
//     29 compute containers carry their RTS0 next to the console-compiled program: base s4 plus the signature's span
//     in this layout equals the program's RSRC2 USER_SGPR count in 29 of 29, and the programs read the tables and
//     root CBVs at exactly these SGPRs (tools/rootsig_census.py --disasm). The game's compute census agrees (lowest
//     pointer s4, one-dword tables).
//   - vertex/pixel/geometry s8 and mesh s16: a hypothesis from the corpus census only (lowest pointer SGPR per stage);
//     no signature confirms them, and visibility is not modelled (every parameter takes its place in every stage).
// A parameter that does not fit goes to the console's spill array behind s[0:1] (compute, section 7) together with
// every parameter after it; that array is not modelled, so such a signature is refused, never truncated.
enum class Stage : uint32_t { Vertex = 0, Pixel = 1, Geometry = 2, Compute = 3, Mesh = 4 };

[[nodiscard]] constexpr uint32_t stage_base(Stage stage) noexcept {
    switch (stage) {
        case Stage::Compute:
            return 4;
        case Stage::Mesh:
            return 16;
        case Stage::Vertex:
        case Stage::Pixel:
        case Stage::Geometry:
            return 8;
    }
    return 8;
}

// User SGPRs one parameter takes in the console layout (see Stage).
[[nodiscard]] constexpr uint32_t stage_dwords(ParamType type, uint32_t num_values) noexcept {
    switch (type) {
        case ParamType::Constants:
            return num_values;
        case ParamType::Descriptor:
            return 2;
        case ParamType::Table:
            return 1;
    }
    return 0;
}

constexpr uint32_t kNoSgpr = 0xFFFFFFFFu;

struct StagePackResult {
    bool ok = false;
    std::string error;            // set when !ok; names the refusing parameter
    Stage stage = Stage::Compute;
    uint32_t base = 0;            // stage_base(stage)
    std::vector<uint32_t> first_sgpr;  // per parameter: its first user SGPR (kNoSgpr when the signature is refused)
    uint32_t end_sgpr = 0;        // one past the last SGPR used: the RSRC2 USER_SGPR count the layout implies
    uint32_t d3d12_cost = 0;      // cost against the 64-dword D3D12 limit
};

// Lays the parameters out for `stage` in the console layout. Fails like pack() on more than 64 parameters, a
// constant parameter of 0 values or the 64-dword cost limit, and on any parameter that would spill.
[[nodiscard]] StagePackResult pack_stage(const RootSignature& sig, Stage stage);

// D3D12 cost of one parameter against the 64-dword limit.
[[nodiscard]] constexpr uint32_t param_cost(ParamType type, uint32_t num_values) noexcept {
    switch (type) {
        case ParamType::Constants:
            return num_values;
        case ParamType::Descriptor:
            return 2;
        case ParamType::Table:
            return 1;
    }
    return 0;
}

// Canonical binary codec for caching root signatures (magic "LRS1",
// little-endian). This is Leonida's own layout, NOT the Microsoft RTS0
// format: it round-trips pack() inputs byte-exactly and is fully tested on
// every CI leg. Never write game blobs in this format and claim otherwise.
// Static samplers carry their flags dword only in version 1.2 signatures, so
// 1.0/1.1 encodings are unchanged. Codec version 2 stores all six range
// fields (version 1 stored only type and count, and is no longer read).
[[nodiscard]] std::vector<uint8_t> encode_canonical(const RootSignature& sig);

struct DecodeResult {
    bool ok = false;
    std::string error;
    RootSignature sig;
};

[[nodiscard]] DecodeResult decode_canonical(const uint8_t* data, size_t size);

// RTS0 chunk parser (payload layout per the maintainer: offset table).
//
// The input is a whole DXBC container (what D3D12SerializeVersionedRootSignature
// returns: "DXBC" magic, hash, size, chunk table). The parser walks the chunk
// table exactly like tools/rdna2spirv.py:input_signature does, finds the
// "RTS0" chunk and decodes its payload: version, counts and table offsets,
// then a parameter table of {d3d_type, visibility, offset} entries with
// payloads at their offsets (d3d types follow D3D12_ROOT_PARAMETER_TYPE
// numbering). A bare payload without the DXBC header is also accepted
// (unit-test path). Every deviation the oracle test finds is a hard failure
// to investigate, not a quirk to absorb.
// Versions 1 (1.0), 2 (1.1) and 3 (1.2): 1.1 and 1.2 add a flags dword to
// root descriptors and ranges; 1.2 adds one to each static sampler (13 dwords
// before, 14 after), as the PC serializer writes them (root_signature_oracle_test).
// A range record follows D3D12_DESCRIPTOR_RANGE1 order: type, count, base
// register, space, then Flags (1.1/1.2 only) BEFORE the table offset.
// Returns ok == false with a reason for unknown versions or truncation.
[[nodiscard]] DecodeResult parse_rts0(const uint8_t* data, size_t size);

}  // namespace rt::rootsig
