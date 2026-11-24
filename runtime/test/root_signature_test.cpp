// Portable tests for runtime/src/root_signature.{h,cpp} (no windows.h).
//
// Own main + PASS/FAIL lines like the other runtime tests; returns nonzero
// on any failure so CTest gates on it. Everything here is hand-built: no
// game data, no DXBC, no serializer needed.
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

rt::rootsig::RootParameter consts(uint32_t n) {
    rt::rootsig::RootParameter p;
    p.type = rt::rootsig::ParamType::Constants;
    p.num_values = n;
    return p;
}

rt::rootsig::RootParameter view() {
    rt::rootsig::RootParameter p;
    p.type = rt::rootsig::ParamType::Descriptor;
    return p;
}

rt::rootsig::RootParameter table() {
    rt::rootsig::RootParameter p;
    p.type = rt::rootsig::ParamType::Table;
    p.ranges.push_back({0, 8});
    return p;
}

void put_u32(std::vector<uint8_t>& out, uint32_t v) {
    out.push_back(static_cast<uint8_t>(v));
    out.push_back(static_cast<uint8_t>(v >> 8));
    out.push_back(static_cast<uint8_t>(v >> 16));
    out.push_back(static_cast<uint8_t>(v >> 24));
}

// Hand-built RTS0 payload in the offset-table layout the maintainer
// described: header, parameter table, then payloads at their offsets.
std::vector<uint8_t> build_payload() {
    std::vector<uint8_t> p;
    put_u32(p, 2);  // version 1.1
    put_u32(p, 2);  // 2 params
    put_u32(p, 24);  // params table offset
    put_u32(p, 0);  // 0 samplers
    put_u32(p, 0);  // samplers offset
    put_u32(p, 0x03000000u);  // Xbox flags
    // entry 0: constants at 48; entry 1: CBV descriptor at 60.
    put_u32(p, 1);
    put_u32(p, 0);
    put_u32(p, 48);
    put_u32(p, 2);
    put_u32(p, 4);
    put_u32(p, 60);
    put_u32(p, 7);  // register (ignored by packer)
    put_u32(p, 0);  // space
    put_u32(p, 3);  // 3 values
    put_u32(p, 0);  // register
    put_u32(p, 0);  // space
    put_u32(p, 0);  // 1.1 descriptor flags
    return p;
}

}  // namespace

int main() {
    using namespace rt::rootsig;

    // The public packer also validates callers that construct the model directly.
    {
        RootSignature sig;
        RootParameter invalid;
        invalid.type = static_cast<ParamType>(0xffffffffu);
        invalid.num_values = 1;
        sig.params = {invalid};
        const auto result = pack(sig);
        check(!result.ok && result.error.find("unknown") != std::string::npos,
              "invalid parameter enum cannot produce a successful zero-cost layout");
    }

    // Mixed signature: 4 constants + CBV + table + 1 static sampler.
    {
        RootSignature sig;
        sig.params = {consts(4), view(), table()};
        StaticSampler s;
        s.shader_register = 3;
        sig.static_samplers = {s};
        const PackResult r = pack(sig);
        check(r.ok, "mixed signature packs");
        check(r.dword_count == 8, "mixed signature uses 8 dwords");
        check(r.d3d12_cost == 4 + 2 + 1, "mixed signature costs 7");
        check(r.provenance.size() == 8 && r.provenance[0].param_index == 0 &&
                  r.provenance[0].sub_index == 0 && r.provenance[4].param_index == 1 &&
                  r.provenance[4].sub_index == 0 && r.provenance[5].sub_index == 1 &&
                  r.provenance[6].param_index == 2,
              "provenance tracks param and sub-dword");
    }

    // Exactly 32 dwords still fits.
    {
        RootSignature sig;
        for (int i = 0; i < 16; ++i) sig.params.push_back(view());
        const PackResult r = pack(sig);
        check(r.ok && r.dword_count == 32, "16 views fill 32 dwords");
    }

    // 34 dwords: spill is refused loudly, never truncated.
    {
        RootSignature sig;
        for (int i = 0; i < 17; ++i) sig.params.push_back(view());
        const PackResult r = pack(sig);
        check(!r.ok && r.error.find("spill") != std::string::npos,
              "17 views refused as spill");
    }

    // 65 constants exceed the 64-dword D3D12 cost limit.
    {
        RootSignature sig;
        sig.params = {consts(65)};
        const PackResult r = pack(sig);
        check(!r.ok && r.error.find("64-dword") != std::string::npos,
              "65 constants refused by cost limit");
    }
    // A count that wraps a 32-bit sum (1 + 0xFFFFFFFF) is still refused, not packed past dwords[32].
    {
        rt::rootsig::RootSignature sig;
        sig.params = {consts(1), consts(0xFFFFFFFFu)};
        const auto r = rt::rootsig::pack(sig);
        check(!r.ok && r.error.find("64-dword") != std::string::npos,
              "a constant count that wraps 32 bits is refused by the cost limit");
    }

    // Canonical codec round-trips byte-exactly, sampler included.
    {
        RootSignature sig;
        sig.version = Version::V1_0;
        sig.params = {consts(2), table()};
        sig.flags = 0x03000000u;  // Xbox high byte survives the model
        StaticSampler s;
        s.filter = 21;
        s.mip_lod_bias = 0.5f;
        sig.static_samplers = {s};
        const std::vector<uint8_t> bytes = encode_canonical(sig);
        const DecodeResult d = decode_canonical(bytes.data(), bytes.size());
        check(d.ok, "canonical round-trip decodes");
        check(d.sig.params.size() == 2 && d.sig.params[0].num_values == 2 &&
                  d.sig.params[1].ranges.size() == 1 &&
                  d.sig.static_samplers.size() == 1 && d.sig.flags == 0x03000000u &&
                  d.sig.static_samplers[0].filter == 21,
              "canonical round-trip preserves model");
    }

    // Canonical decoder controls: bad magic, truncation, trailing bytes.
    {
        const std::vector<uint8_t> bad = {0, 1, 2, 3};
        check(!decode_canonical(bad.data(), bad.size()).ok, "bad magic rejected");
        RootSignature sig;
        sig.params = {consts(1)};
        const std::vector<uint8_t> bytes = encode_canonical(sig);
        check(!decode_canonical(bytes.data(), bytes.size() - 1).ok,
              "truncated canonical rejected");
        std::vector<uint8_t> trailed = bytes;
        trailed.push_back(0);
        check(!decode_canonical(trailed.data(), trailed.size()).ok,
              "trailing bytes rejected");
    }

    // RTS0 hypothesis: a hand-built blob per the documented layout parses.
    // Hand-built RTS0 payload in the offset-table layout the maintainer
    // described: header, parameter table, then payloads at their offsets.
    {
        const std::vector<uint8_t> payload = build_payload();
        const DecodeResult d = parse_rts0(payload.data(), payload.size());
        check(d.ok, "hand-built RTS0 parses");
        check(d.sig.params.size() == 2 && d.sig.params[0].num_values == 3 &&
                  d.sig.flags == 0x03000000u,
              "RTS0 model matches hand-built blob");
        const PackResult r = pack(d.sig);
        check(r.ok && r.dword_count == 3 + 2, "parsed RTS0 packs to 5 dwords");
    }

    // RTS0 controls: unknown version, wild offsets and truncation fail loudly.
    {
        std::vector<uint8_t> blob;
        put_u32(blob, 9);  // unknown version
        put_u32(blob, 0);
        put_u32(blob, 0);
        put_u32(blob, 0);
        put_u32(blob, 0);
        put_u32(blob, 0);
        const DecodeResult d = parse_rts0(blob.data(), blob.size());
        check(!d.ok, "unknown RTS0 version rejected");
        std::vector<uint8_t> cut = {1, 0};
        check(!parse_rts0(cut.data(), cut.size()).ok, "truncated RTS0 rejected");
        std::vector<uint8_t> wild = build_payload();
        // Corrupt the params offset to point past the end.
        wild[8] = 0xFF;
        wild[9] = 0xFF;
        wild[10] = 0xFF;
        wild[11] = 0xFF;
        check(!parse_rts0(wild.data(), wild.size()).ok, "wild offset rejected");
    }

    // DXBC container walk: a hand-built container with one RTS0 chunk parses
    // through the chunk table (same path the MS serializer blob takes).
    {
        const std::vector<uint8_t> payload = build_payload();
        std::vector<uint8_t> blob;
        blob.insert(blob.end(), {'D', 'X', 'B', 'C'});
        blob.insert(blob.end(), 16, 0);  // hash
        put_u32(blob, 1);  // version
        put_u32(blob, 0);  // file size (patched below)
        put_u32(blob, 1);  // 1 chunk
        put_u32(blob, 36);  // chunk offset: 4 magic + 16 hash + 4 ver + 4 size + 4 count + 4 offset
        blob.insert(blob.end(), {'R', 'T', 'S', '0'});
        put_u32(blob, static_cast<uint32_t>(payload.size()));
        blob.insert(blob.end(), payload.begin(), payload.end());
        const uint32_t total = static_cast<uint32_t>(blob.size());
        blob[24] = static_cast<uint8_t>(total);
        blob[25] = static_cast<uint8_t>(total >> 8);
        blob[26] = static_cast<uint8_t>(total >> 16);
        blob[27] = static_cast<uint8_t>(total >> 24);
        const DecodeResult d = parse_rts0(blob.data(), blob.size());
        check(d.ok, "DXBC container walk finds RTS0");
        check(d.sig.params.size() == 2 && d.sig.params[0].num_values == 3,
              "walked RTS0 model matches payload");
    }

    // RTS0 descriptor tables: the payload is {num_ranges, ranges_offset} and the ranges sit at that offset, as the
    // PC serializer writes them (root_signature_oracle_test); here a 1.1 table with two 24-byte ranges placed after
    // a gap, so reading them inline after the count would fail.
    {
        std::vector<uint8_t> p;
        put_u32(p, 2);   // version 1.1
        put_u32(p, 1);   // 1 param
        put_u32(p, 24);  // params table offset
        put_u32(p, 0);   // 0 samplers
        put_u32(p, 0);
        put_u32(p, 0);   // flags
        put_u32(p, 0);   // entry 0: descriptor table
        put_u32(p, 5);   // pixel
        put_u32(p, 36);  // table payload
        put_u32(p, 2);   // 2 ranges
        put_u32(p, 52);  // at 52
        put_u32(p, 0xdeadbeefu);  // gap
        put_u32(p, 0xdeadbeefu);
        // {type, count, base register, space, flags, offset}: 1.1 puts the flags before the table offset.
        for (const uint32_t v : {0u, 4u, 0u, 0u, 8u, 0xffffffffu,      // SRV x4, DATA_STATIC, appended
                                 2u, 1u, 3u, 1u, 1u, 6u})              // CBV x1 at b3 space1, DESCRIPTORS_VOLATILE, offset 6
            put_u32(p, v);
        const DecodeResult d = parse_rts0(p.data(), p.size());
        check(d.ok && d.sig.params.size() == 1 && d.sig.params[0].type == ParamType::Table &&
                  d.sig.params[0].visibility == Visibility::Pixel && d.sig.params[0].ranges.size() == 2 &&
                  d.sig.params[0].ranges[0].range_type == 0 && d.sig.params[0].ranges[0].num_descriptors == 4 &&
                  d.sig.params[0].ranges[1].range_type == 2 && d.sig.params[0].ranges[1].num_descriptors == 1,
              "RTS0 table ranges read from their offset");
        check(d.ok && d.sig.params[0].ranges.size() == 2 && d.sig.params[0].ranges[0].flags == 8 &&
                  d.sig.params[0].ranges[0].offset_in_table == kRangeOffsetAppend &&
                  d.sig.params[0].ranges[1].base_shader_register == 3 &&
                  d.sig.params[0].ranges[1].register_space == 1 && d.sig.params[0].ranges[1].flags == 1 &&
                  d.sig.params[0].ranges[1].offset_in_table == 6,
              "RTS0 1.1 ranges: register, space, flags, then table offset");
        std::vector<uint8_t> wild = p;
        wild[40] = 0xF0;  // ranges offset far past the payload
        check(!parse_rts0(wild.data(), wild.size()).ok, "RTS0 ranges offset past the payload rejected");
    }

    // RTS0 1.0 ranges have no flags dword: 20 bytes, the table offset right after the space.
    {
        std::vector<uint8_t> p;
        put_u32(p, 1);   // version 1.0
        put_u32(p, 1);   // 1 param
        put_u32(p, 24);  // params table offset
        put_u32(p, 0);   // 0 samplers
        put_u32(p, 0);
        put_u32(p, 0);   // flags
        put_u32(p, 0);   // entry 0: descriptor table
        put_u32(p, 0);   // all stages
        put_u32(p, 36);  // table payload
        put_u32(p, 2);   // 2 ranges
        put_u32(p, 44);  // right after the payload
        for (const uint32_t v : {1u, 2u, 5u, 0u, 0xffffffffu,    // UAV x2 at u5, appended
                                 3u, 1u, 2u, 4u, 9u})            // sampler x1 at s2 space4, offset 9
            put_u32(p, v);
        const DecodeResult d = parse_rts0(p.data(), p.size());
        check(d.ok && d.sig.params.size() == 1 && d.sig.params[0].ranges.size() == 2 &&
                  d.sig.params[0].ranges[0].range_type == 1 && d.sig.params[0].ranges[0].num_descriptors == 2 &&
                  d.sig.params[0].ranges[0].base_shader_register == 5 &&
                  d.sig.params[0].ranges[0].offset_in_table == kRangeOffsetAppend &&
                  d.sig.params[0].ranges[0].flags == 0 && d.sig.params[0].ranges[1].range_type == 3 &&
                  d.sig.params[0].ranges[1].base_shader_register == 2 &&
                  d.sig.params[0].ranges[1].register_space == 4 &&
                  d.sig.params[0].ranges[1].offset_in_table == 9 && d.sig.params[0].ranges[1].flags == 0,
              "RTS0 1.0 ranges: no flags, table offset at +16");
        std::vector<uint8_t> cut = p;
        cut.resize(cut.size() - 4);
        check(!parse_rts0(cut.data(), cut.size()).ok, "RTS0 1.0 range without its offset rejected");
    }

    // The canonical codec keeps every range field, and refuses the old codec version that dropped them.
    {
        RootSignature sig;
        sig.params = {table()};
        sig.params[0].ranges.push_back({2, 1, 7, 3, 4, 12});
        const std::vector<uint8_t> bytes = encode_canonical(sig);
        const DecodeResult d = decode_canonical(bytes.data(), bytes.size());
        check(d.ok && d.sig.params.size() == 1 && d.sig.params[0].ranges.size() == 2 &&
                  d.sig.params[0].ranges[1].range_type == 2 && d.sig.params[0].ranges[1].num_descriptors == 1 &&
                  d.sig.params[0].ranges[1].base_shader_register == 7 &&
                  d.sig.params[0].ranges[1].register_space == 3 && d.sig.params[0].ranges[1].flags == 4 &&
                  d.sig.params[0].ranges[1].offset_in_table == 12,
              "canonical codec round-trips every range field");
        std::vector<uint8_t> v1 = bytes;
        v1[4] = 1;  // codec version 1 stored ranges as {type, count} only
        check(!decode_canonical(v1.data(), v1.size()).ok, "canonical codec version 1 refused");
    }

    // Version 1.2: the canonical codec carries the static-sampler flags, and only for 1.2, so 1.0/1.1 encodings are
    // byte-for-byte what they were before 1.2 existed.
    {
        RootSignature sig;
        sig.version = Version::V1_2;
        sig.params = {consts(1)};
        StaticSampler s;
        s.border_color = 5;  // D3D12_STATIC_BORDER_COLOR_OPAQUE_BLACK_UINT
        s.flags = 1;         // D3D12_SAMPLER_FLAG_UINT_BORDER_COLOR
        StaticSampler plain;
        plain.shader_register = 1;
        sig.static_samplers = {s, plain};
        const std::vector<uint8_t> v12 = encode_canonical(sig);
        const DecodeResult d = decode_canonical(v12.data(), v12.size());
        check(d.ok && d.sig.version == Version::V1_2 && d.sig.static_samplers.size() == 2 &&
                  d.sig.static_samplers[0].flags == 1 && d.sig.static_samplers[0].border_color == 5 &&
                  d.sig.static_samplers[1].flags == 0 && d.sig.static_samplers[1].shader_register == 1,
              "canonical 1.2 round-trips sampler flags");
        check(!decode_canonical(v12.data(), v12.size() - 4).ok, "canonical 1.2 without the last flags is truncated");
        sig.version = Version::V1_1;
        const std::vector<uint8_t> v11 = encode_canonical(sig);
        check(v11.size() + 2 * 4 == v12.size(), "canonical 1.1 has no sampler flags dword");
        const DecodeResult d11 = decode_canonical(v11.data(), v11.size());
        check(d11.ok && d11.sig.static_samplers[0].flags == 0, "canonical 1.1 decodes with flags 0");
    }

    // RTS0 version 3 (1.2): 1.1-style descriptor flags, then a 14-dword sampler whose last dword is its flags.
    {
        std::vector<uint8_t> p;
        put_u32(p, 3);   // version 1.2
        put_u32(p, 1);   // 1 param
        put_u32(p, 24);  // params table offset
        put_u32(p, 1);   // 1 sampler
        put_u32(p, 48);  // samplers offset
        put_u32(p, 0);   // flags
        put_u32(p, 2);   // entry 0: CBV
        put_u32(p, 0);
        put_u32(p, 36);
        put_u32(p, 9);   // register
        put_u32(p, 0);   // space
        put_u32(p, 4);   // descriptor flags (DATA_STATIC_WHILE_SET_AT_EXECUTE)
        const uint32_t sampler[14] = {0x15, 1, 2, 3, 0, 16, 4, 5, 0, 0x7f7fffffu, 3, 1, 5, 2};
        for (const uint32_t v : sampler) put_u32(p, v);
        const DecodeResult d = parse_rts0(p.data(), p.size());
        check(d.ok, "hand-built RTS0 1.2 parses");
        check(d.ok && d.sig.version == Version::V1_2 && d.sig.params.size() == 1 &&
                  d.sig.params[0].descriptor_register == 9 && d.sig.static_samplers.size() == 1 &&
                  d.sig.static_samplers[0].filter == 0x15 && d.sig.static_samplers[0].max_anisotropy == 16 &&
                  d.sig.static_samplers[0].border_color == 5 && d.sig.static_samplers[0].shader_register == 3 &&
                  d.sig.static_samplers[0].register_space == 1 &&
                  d.sig.static_samplers[0].visibility == Visibility::Pixel &&
                  d.sig.static_samplers[0].flags == 2,
              "RTS0 1.2 sampler fields and flags");
        std::vector<uint8_t> cut = p;
        cut.resize(cut.size() - 4);  // a 13-dword (1.0/1.1-sized) sampler in a 1.2 blob
        check(!parse_rts0(cut.data(), cut.size()).ok, "RTS0 1.2 sampler without flags rejected");
        std::vector<uint8_t> v4 = p;
        v4[0] = 4;
        check(!parse_rts0(v4.data(), v4.size()).ok, "RTS0 version 4 rejected");
    }

    // pack_stage: the console layout (D3D-448).
    {
        check(stage_base(Stage::Compute) == 4 && stage_base(Stage::Mesh) == 16 && stage_base(Stage::Pixel) == 8 &&
                  stage_base(Stage::Vertex) == 8 && stage_base(Stage::Geometry) == 8,
              "pack_stage: stage bases cs s4, ms s16, vs/ps/gs s8");
        check(stage_dwords(ParamType::Table, 0) == 1 && stage_dwords(ParamType::Descriptor, 0) == 2 &&
                  stage_dwords(ParamType::Constants, 3) == 3,
              "pack_stage: a table takes 1 dword, a root view 2, constants their count");

        // The shape of the package's FidelityFX compute signatures: two tables, then four root CBVs. The console
        // launched those programs with 14 user SGPRs and they read the tables at s4/s5 and the CBVs as s[6:7]..s[12:13].
        RootSignature fsr;
        fsr.params = {table(), table(), view(), view(), view(), view()};
        const StagePackResult r = pack_stage(fsr, Stage::Compute);
        check(r.ok && r.base == 4, "pack_stage: FidelityFX shape packs for compute");
        check(r.first_sgpr == std::vector<uint32_t>({4, 5, 6, 8, 10, 12}),
              "pack_stage: tables at s4, s5 and root CBVs at s6, s8, s10, s12");
        check(r.end_sgpr == 14, "pack_stage: ends at s14, the programs' USER_SGPR count");
        check(r.d3d12_cost == 10, "pack_stage: D3D12 cost 1 + 1 + 2 * 4");
        const PackResult old = pack(fsr);
        check(old.ok && old.dword_count == 12, "pack_stage: the s0 model spans 12 dwords for the same signature");

        // Static samplers take no SGPRs; the same parameters land at the stage's own base.
        RootSignature mixed;
        mixed.params = {consts(3), table(), view()};
        mixed.static_samplers.resize(2);
        const StagePackResult ps = pack_stage(mixed, Stage::Pixel);
        check(ps.ok && ps.first_sgpr == std::vector<uint32_t>({8, 11, 12}) && ps.end_sgpr == 14,
              "pack_stage: pixel stage from s8, static samplers free");
        const StagePackResult ms = pack_stage(mixed, Stage::Mesh);
        check(ms.ok && ms.first_sgpr == std::vector<uint32_t>({16, 19, 20}) && ms.end_sgpr == 22,
              "pack_stage: mesh stage from s16");

        // An empty signature is valid and uses nothing.
        const StagePackResult empty = pack_stage(RootSignature{}, Stage::Compute);
        check(empty.ok && empty.first_sgpr.empty() && empty.end_sgpr == 4, "pack_stage: empty signature");
    }

    // pack_stage refuses a spill instead of truncating: compute has 28 SGPRs above s4, mesh 16 above s16.
    {
        RootSignature exact;
        exact.params = {consts(28)};
        check(pack_stage(exact, Stage::Compute).ok, "pack_stage: 28 constants fill compute exactly");
        RootSignature over;
        over.params = {consts(27), view()};  // the view would need s31..s32
        const StagePackResult r = pack_stage(over, Stage::Compute);
        check(!r.ok && r.first_sgpr.empty() && r.error.find("root parameter 1") != std::string::npos &&
                  r.error.find("spill") != std::string::npos,
              "pack_stage: a parameter that does not fit whole is refused and named");
        RootSignature fits_old;
        fits_old.params = {consts(30)};  // fits pack()'s 32 from s0, not 28 from s4
        check(pack(fits_old).ok && !pack_stage(fits_old, Stage::Compute).ok,
              "pack_stage: compute budget is 28, not 32");
        RootSignature tables;
        tables.params.assign(20, table());  // one dword each: compute s4..s23; pack() needs 40
        check(pack_stage(tables, Stage::Compute).ok && !pack(tables).ok,
              "pack_stage: 20 one-dword tables fit compute, which pack() refuses (40 dwords of tables)");
        tables.params.assign(16, table());  // fills mesh s16..s31 exactly
        check(pack_stage(tables, Stage::Mesh).ok, "pack_stage: 16 tables fill the mesh stage exactly");
        tables.params.push_back(table());
        check(!pack_stage(tables, Stage::Mesh).ok, "pack_stage: a 17th table spills on mesh");
    }

    // pack_stage keeps pack()'s validation.
    {
        RootSignature zero;
        zero.params = {consts(0)};
        check(!pack_stage(zero, Stage::Compute).ok, "pack_stage: zero-value constants rejected");
        // Every parameter takes at least as many SGPRs as it costs, so the 32-SGPR budget refuses before the
        // 64-dword cost limit can; an over-cost signature is refused all the same.
        RootSignature costly;
        costly.params = {consts(40), consts(30)};
        check(!pack_stage(costly, Stage::Compute).ok, "pack_stage: over-cost signature refused");
        RootSignature many;
        many.params.assign(65, table());
        check(!pack_stage(many, Stage::Compute).ok, "pack_stage: more than 64 parameters rejected");
    }

    if (g_failures != 0) std::printf("%d FAILURES\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
