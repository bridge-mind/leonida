// Portable root-signature packer, see root_signature.h.
#include "root_signature.h"

#include <cstring>

namespace rt::rootsig {

PackResult pack(const RootSignature& sig) {
    PackResult out;
    if (sig.params.size() > 64) {
        out.error = "root signature has more than 64 root parameters";
        return out;
    }
    uint64_t cost = 0;  // 64-bit: a parameter of 0xFFFFFFFF constants must not wrap past the limits
    for (const RootParameter& p : sig.params) {
        if (p.type != ParamType::Constants && p.type != ParamType::Descriptor && p.type != ParamType::Table) {
            out.error = "unknown root parameter type";
            return out;
        }
        if (p.type == ParamType::Constants && p.num_values == 0) {
            out.error = "root constant parameter needs at least 1 dword";
            return out;
        }
        const uint32_t dwords = p.type == ParamType::Table          ? 2
                                : p.type == ParamType::Descriptor   ? 2
                                                                    : p.num_values;
        cost += param_cost(p.type, p.num_values);
        if (cost > kMaxD3d12Cost) {
            out.error = "root signature exceeds the 64-dword D3D12 cost limit";
            return out;
        }
        if (uint64_t(out.dword_count) + dwords > kMaxUserSgprs) {
            // Spill layout is unknown (see __XB_GetRootSignatureSpillPointer):
            // refuse loudly instead of silently truncating every draw.
            out.error = "root signature needs more than 32 user SGPRs (spill unsupported)";
            return out;
        }
        const uint32_t param_index =
            static_cast<uint32_t>(&p - sig.params.data());
        for (uint32_t i = 0; i < dwords; ++i) {
            out.dwords[out.dword_count] = 0;  // filled at bind time
            out.provenance.push_back({param_index, i});
            ++out.dword_count;
        }
    }
    out.d3d12_cost = static_cast<uint32_t>(cost);  // <= kMaxD3d12Cost here
    out.ok = true;
    return out;
}

StagePackResult pack_stage(const RootSignature& sig, Stage stage) {
    StagePackResult out;
    out.stage = stage;
    out.base = stage_base(stage);
    out.end_sgpr = out.base;
    if (sig.params.size() > 64) {
        out.error = "root signature has more than 64 root parameters";
        return out;
    }
    uint64_t cost = 0;  // 64-bit, as in pack()
    uint64_t next = out.base;
    std::vector<uint32_t> first;
    first.reserve(sig.params.size());
    for (size_t i = 0; i < sig.params.size(); ++i) {
        const RootParameter& p = sig.params[i];
        if (p.type == ParamType::Constants && p.num_values == 0) {
            out.error = "root constant parameter needs at least 1 dword";
            return out;
        }
        cost += param_cost(p.type, p.num_values);
        if (cost > kMaxD3d12Cost) {
            out.error = "root signature exceeds the 64-dword D3D12 cost limit";
            return out;
        }
        const uint64_t dwords = stage_dwords(p.type, p.num_values);
        if (next + dwords > kMaxUserSgprs) {
            // The console moves this parameter and every later one to the spill array behind s[0:1]; that array's
            // layout is not modelled here, so refuse rather than drop arguments.
            out.error = "root parameter " + std::to_string(i) + " does not fit in the user SGPRs from s" +
                        std::to_string(out.base) + " (spill array not modelled)";
            return out;
        }
        first.push_back(static_cast<uint32_t>(next));
        next += dwords;
    }
    out.first_sgpr = std::move(first);
    out.end_sgpr = static_cast<uint32_t>(next);
    out.d3d12_cost = static_cast<uint32_t>(cost);
    out.ok = true;
    return out;
}

namespace {

void put_u32(std::vector<uint8_t>& out, uint32_t v) {
    out.push_back(static_cast<uint8_t>(v));
    out.push_back(static_cast<uint8_t>(v >> 8));
    out.push_back(static_cast<uint8_t>(v >> 16));
    out.push_back(static_cast<uint8_t>(v >> 24));
}

void put_f32(std::vector<uint8_t>& out, float v) {
    uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(v));
    std::memcpy(&bits, &v, sizeof(bits));
    put_u32(out, bits);
}

class Reader {
public:
    Reader(const uint8_t* data, size_t size) : data_(data), size_(size) {}

    [[nodiscard]] bool u32(uint32_t& v) {
        if (pos_ + 4 > size_) return false;
        v = static_cast<uint32_t>(data_[pos_]) |
            (static_cast<uint32_t>(data_[pos_ + 1]) << 8) |
            (static_cast<uint32_t>(data_[pos_ + 2]) << 16) |
            (static_cast<uint32_t>(data_[pos_ + 3]) << 24);
        pos_ += 4;
        return true;
    }

    [[nodiscard]] bool f32(float& v) {
        uint32_t bits = 0;
        if (!u32(bits)) return false;
        std::memcpy(&v, &bits, sizeof(v));
        return true;
    }

    [[nodiscard]] bool at_end() const { return pos_ == size_; }

private:
    const uint8_t* data_ = nullptr;
    size_t size_ = 0;
    size_t pos_ = 0;
};

constexpr uint32_t kCanonicalMagic = 0x3153524Cu;  // "LRS1" little-endian
// 2: ranges store all six D3D12_DESCRIPTOR_RANGE1 fields (1 stored only type and count). Nothing persists these
// blobs across builds, so version 1 is simply refused.
constexpr uint32_t kCanonicalVersion = 2;

// D3D_ROOT_SIGNATURE_VERSION 1 (1.0), 2 (1.1), 3 (1.2): the numbering RTS0 and the canonical codec both store.
bool known_version(uint32_t v, Version& out) {
    if (v < 1 || v > 3) return false;
    out = static_cast<Version>(v);
    return true;
}

}  // namespace

std::vector<uint8_t> encode_canonical(const RootSignature& sig) {
    std::vector<uint8_t> out;
    put_u32(out, kCanonicalMagic);
    put_u32(out, kCanonicalVersion);
    put_u32(out, static_cast<uint32_t>(sig.version));
    put_u32(out, static_cast<uint32_t>(sig.params.size()));
    put_u32(out, static_cast<uint32_t>(sig.static_samplers.size()));
    put_u32(out, sig.flags);
    for (const RootParameter& p : sig.params) {
        put_u32(out, static_cast<uint32_t>(p.type));
        put_u32(out, static_cast<uint32_t>(p.visibility));
        put_u32(out, p.num_values);
        put_u32(out, p.shader_register);
        put_u32(out, p.register_space);
        put_u32(out, p.descriptor_register);
        put_u32(out, p.descriptor_space);
        put_u32(out, static_cast<uint32_t>(p.ranges.size()));
        for (const DescriptorRange& r : p.ranges) {
            put_u32(out, r.range_type);
            put_u32(out, r.num_descriptors);
            put_u32(out, r.base_shader_register);
            put_u32(out, r.register_space);
            put_u32(out, r.flags);
            put_u32(out, r.offset_in_table);
        }
    }
    for (const StaticSampler& s : sig.static_samplers) {
        put_u32(out, s.filter);
        put_u32(out, s.address_u);
        put_u32(out, s.address_v);
        put_u32(out, s.address_w);
        put_f32(out, s.mip_lod_bias);
        put_u32(out, s.max_anisotropy);
        put_u32(out, s.comparison_func);
        put_u32(out, s.border_color);
        put_f32(out, s.min_lod);
        put_f32(out, s.max_lod);
        put_u32(out, s.shader_register);
        put_u32(out, s.register_space);
        put_u32(out, static_cast<uint32_t>(s.visibility));
        if (sig.version == Version::V1_2) put_u32(out, s.flags);
    }
    return out;
}

DecodeResult decode_canonical(const uint8_t* data, size_t size) {
    DecodeResult out;
    if (data == nullptr && size != 0) {
        out.error = "null data with nonzero size";
        return out;
    }
    Reader r(data == nullptr ? reinterpret_cast<const uint8_t*>("") : data, size);
    uint32_t magic = 0, version = 0, nparams = 0, nsamplers = 0;
    if (!r.u32(magic) || magic != kCanonicalMagic) {
        out.error = "bad canonical magic";
        return out;
    }
    if (!r.u32(version) || version != kCanonicalVersion) {
        out.error = "unsupported canonical version";
        return out;
    }
    uint32_t ver = 0;
    if (!r.u32(ver) || !r.u32(nparams) || !r.u32(nsamplers) || !r.u32(out.sig.flags)) {
        out.error = "truncated canonical header";
        return out;
    }
    if (!known_version(ver, out.sig.version)) {
        out.error = "unknown root-signature version";
        return out;
    }
    if (nparams > 64) {
        out.error = "more than 64 root parameters";
        return out;
    }
    if (nsamplers > 4096) {
        out.error = "absurd sampler count";
        return out;
    }
    out.sig.params.reserve(nparams);
    for (uint32_t i = 0; i < nparams; ++i) {
        RootParameter p;
        uint32_t type = 0, vis = 0, nranges = 0;
        if (!r.u32(type) || !r.u32(vis) || !r.u32(p.num_values) || !r.u32(p.shader_register) ||
            !r.u32(p.register_space) || !r.u32(p.descriptor_register) ||
            !r.u32(p.descriptor_space) || !r.u32(nranges)) {
            out.error = "truncated canonical parameter";
            return out;
        }
        if (type > 2) {
            out.error = "unknown parameter type";
            return out;
        }
        p.type = static_cast<ParamType>(type);
        p.visibility = static_cast<Visibility>(vis);
        if (nranges > 1024) {
            out.error = "absurd range count";
            return out;
        }
        p.ranges.reserve(nranges);
        for (uint32_t j = 0; j < nranges; ++j) {
            DescriptorRange range;
            if (!r.u32(range.range_type) || !r.u32(range.num_descriptors) || !r.u32(range.base_shader_register) ||
                !r.u32(range.register_space) || !r.u32(range.flags) || !r.u32(range.offset_in_table)) {
                out.error = "truncated canonical range";
                return out;
            }
            p.ranges.push_back(range);
        }
        out.sig.params.push_back(std::move(p));
    }
    out.sig.static_samplers.reserve(nsamplers);
    for (uint32_t i = 0; i < nsamplers; ++i) {
        StaticSampler s;
        uint32_t vis = 0;
        if (!r.u32(s.filter) || !r.u32(s.address_u) || !r.u32(s.address_v) || !r.u32(s.address_w) ||
            !r.f32(s.mip_lod_bias) || !r.u32(s.max_anisotropy) || !r.u32(s.comparison_func) ||
            !r.u32(s.border_color) || !r.f32(s.min_lod) || !r.f32(s.max_lod) ||
            !r.u32(s.shader_register) || !r.u32(s.register_space) || !r.u32(vis) ||
            (out.sig.version == Version::V1_2 && !r.u32(s.flags))) {
            out.error = "truncated canonical sampler";
            return out;
        }
        s.visibility = static_cast<Visibility>(vis);
        out.sig.static_samplers.push_back(s);
    }
    if (!r.at_end()) {
        out.error = "trailing bytes after canonical root signature";
        return out;
    }
    out.ok = true;
    return out;
}

// RTS0 payload: u32 version (1 = 1.0, 2 = 1.1, 3 = 1.2), u32 num_params,
// u32 num_static_samplers, u32 flags, then per-parameter fields in d3d12.h
// order. Static samplers follow the field order of D3D12_STATIC_SAMPLER_DESC, and
// of D3D12_STATIC_SAMPLER_DESC1 (one more dword, Flags) in 1.2. Root descriptors
// and ranges have a flags dword from 1.1 on. Anything else (unknown versions,
// truncation) is a hard error.
// Payload decoder: the RTS0 chunk contents (see header for the hypothesis).
static DecodeResult parse_rts0_payload(const uint8_t* data, size_t size) {
    DecodeResult out;
    if (data == nullptr && size != 0) {
        out.error = "null data with nonzero size";
        return out;
    }
    if (data == nullptr) {
        out.error = "null RTS0 payload";
        return out;
    }
    // Random access into the payload: every (offset, byte-count) pair is
    // validated before any byte is touched. Offsets count from payload start.
    auto at = [&](size_t off, size_t bytes) -> const uint8_t* {
        if (bytes > size || off > size - bytes) return nullptr;
        return data + off;
    };
    auto u32_at = [&](size_t off, uint32_t& v) {
        const uint8_t* p = at(off, 4);
        if (p == nullptr) return false;
        v = static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
            (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
        return true;
    };
    auto f32_at = [&](size_t off, float& v) {
        uint32_t bits = 0;
        if (!u32_at(off, bits)) return false;
        static_assert(sizeof(bits) == sizeof(v));
        std::memcpy(&v, &bits, sizeof(v));
        return true;
    };
    // Header: version, num_params, params_offset, num_samplers,
    // samplers_offset, flags.
    uint32_t ver = 0, nparams = 0, params_off = 0, nsamplers = 0, samplers_off = 0, flags = 0;
    if (!u32_at(0, ver) || !u32_at(4, nparams) || !u32_at(8, params_off) ||
        !u32_at(12, nsamplers) || !u32_at(16, samplers_off) || !u32_at(20, flags)) {
        out.error = "truncated RTS0 header";
        return out;
    }
    if (!known_version(ver, out.sig.version)) {
        out.error = "unknown RTS0 version (not 1.0/1.1/1.2)";
        return out;
    }
    if (nparams > 64) {
        out.error = "RTS0 with more than 64 root parameters";
        return out;
    }
    if (nsamplers > 4096) {
        out.error = "RTS0 with absurd sampler count";
        return out;
    }
    const bool flags_v11 = ver >= 2;  // descriptor and range flags (1.1 and 1.2)
    out.sig.flags = flags;
    // Parameter table at params_off: num_params x {u32 d3d_type, u32
    // visibility, u32 offset}. d3d_type follows D3D12_ROOT_PARAMETER_TYPE
    // (0 table, 1 constants, 2/3/4 CBV/SRV/UAV root descriptors).
    constexpr size_t kEntryBytes = 12;
    if (nparams > 0 &&
        (params_off > size || nparams > (size - params_off) / kEntryBytes)) {
        out.error = "RTS0 parameter table overruns its payload";
        return out;
    }
    out.sig.params.reserve(nparams);
    for (uint32_t i = 0; i < nparams; ++i) {
        const size_t eoff = static_cast<size_t>(params_off) + i * kEntryBytes;
        uint32_t dtype = 0, vis = 0, poff = 0;
        if (!u32_at(eoff, dtype) || !u32_at(eoff + 4, vis) || !u32_at(eoff + 8, poff)) {
            out.error = "truncated RTS0 parameter entry";
            return out;
        }
        RootParameter p;
        p.visibility = static_cast<Visibility>(vis);
        if (dtype == 1) {
            p.type = ParamType::Constants;
            if (!u32_at(poff, p.shader_register) || !u32_at(poff + 4, p.register_space) ||
                !u32_at(poff + 8, p.num_values)) {
                out.error = "truncated RTS0 constants payload";
                return out;
            }
        } else if (dtype == 2 || dtype == 3 || dtype == 4) {
            p.type = ParamType::Descriptor;
            uint32_t dflags = 0;
            const size_t need = flags_v11 ? 12 : 8;
            if (poff > size || need > size - static_cast<size_t>(poff)) {
                out.error = "truncated RTS0 descriptor payload";
                return out;
            }
            if (!u32_at(poff, p.descriptor_register) || !u32_at(poff + 4, p.descriptor_space)) {
                out.error = "truncated RTS0 descriptor payload";
                return out;
            }
            if (flags_v11 && !u32_at(poff + 8, dflags)) {
                out.error = "truncated RTS0 descriptor flags";
                return out;
            }
            (void)dflags;  // descriptor flags do not affect the user-SGPR map
        } else if (dtype == 0) {
            p.type = ParamType::Table;
            // Table payload: {u32 num_ranges, u32 ranges_offset}; the ranges sit at ranges_offset (from the payload
            // start), like every other table in RTS0. Measured against the PC serializer (root_signature_oracle_test).
            uint32_t nranges = 0, ranges_off = 0;
            if (!u32_at(poff, nranges) || !u32_at(static_cast<size_t>(poff) + 4, ranges_off)) {
                out.error = "truncated RTS0 table payload";
                return out;
            }
            if (nranges > 1024) {
                out.error = "absurd RTS0 range count";
                return out;
            }
            // Range record, in D3D12_DESCRIPTOR_RANGE(1) order: {type, num, base_reg, space, offset} in 1.0, and
            // {type, num, base_reg, space, flags, offset} in 1.1 and 1.2: the flags come BEFORE the table offset
            // (measured against the PC serializer in root_signature_oracle_test).
            const size_t stride = flags_v11 ? 24 : 20;
            if (nranges > 0 && (ranges_off > size || nranges > (size - ranges_off) / stride)) {
                out.error = "RTS0 ranges overrun their payload";
                return out;
            }
            p.ranges.reserve(nranges);
            for (uint32_t j = 0; j < nranges; ++j) {
                const size_t roff = static_cast<size_t>(ranges_off) + j * stride;
                DescriptorRange range;
                const size_t offset_at = roff + (flags_v11 ? 20 : 16);
                if (!u32_at(roff, range.range_type) || !u32_at(roff + 4, range.num_descriptors) ||
                    !u32_at(roff + 8, range.base_shader_register) || !u32_at(roff + 12, range.register_space) ||
                    (flags_v11 && !u32_at(roff + 16, range.flags)) || !u32_at(offset_at, range.offset_in_table)) {
                    out.error = "truncated RTS0 range";
                    return out;
                }
                p.ranges.push_back(range);
            }
        } else {
            out.error = "unknown RTS0 parameter type";
            return out;
        }
        out.sig.params.push_back(std::move(p));
    }
    // Sampler records mirror D3D12_STATIC_SAMPLER_DESC field order; 1.2 appends Flags (D3D12_STATIC_SAMPLER_DESC1).
    const bool sampler_flags = ver == 3;
    const size_t kSamplerBytes = (sampler_flags ? 14 : 13) * 4;
    if (nsamplers > 0 &&
        (samplers_off > size || nsamplers > (size - samplers_off) / kSamplerBytes)) {
        out.error = "RTS0 samplers overrun their payload";
        return out;
    }
    out.sig.static_samplers.reserve(nsamplers);
    for (uint32_t i = 0; i < nsamplers; ++i) {
        const size_t soff = static_cast<size_t>(samplers_off) + i * kSamplerBytes;
        StaticSampler s;
        uint32_t vis = 0;
        if (!u32_at(soff, s.filter) || !u32_at(soff + 4, s.address_u) ||
            !u32_at(soff + 8, s.address_v) || !u32_at(soff + 12, s.address_w) ||
            !f32_at(soff + 16, s.mip_lod_bias) || !u32_at(soff + 20, s.max_anisotropy) ||
            !u32_at(soff + 24, s.comparison_func) || !u32_at(soff + 28, s.border_color) ||
            !f32_at(soff + 32, s.min_lod) || !f32_at(soff + 36, s.max_lod) ||
            !u32_at(soff + 40, s.shader_register) || !u32_at(soff + 44, s.register_space) ||
            !u32_at(soff + 48, vis) || (sampler_flags && !u32_at(soff + 52, s.flags))) {
            out.error = "truncated RTS0 sampler";
            return out;
        }
        s.visibility = static_cast<Visibility>(vis);
        out.sig.static_samplers.push_back(s);
    }
    out.ok = true;
    return out;
}

DecodeResult parse_rts0(const uint8_t* data, size_t size) {
    DecodeResult out;
    if ((data == nullptr && size != 0) || (data != nullptr && size < 4)) {
        out.error = "blob too small for a DXBC container";
        return out;
    }
    if (data == nullptr) {
        out.error = "null blob";
        return out;
    }
    // A bare payload (unit-test path) has no DXBC magic: decode directly.
    if (!(data[0] == 'D' && data[1] == 'X' && data[2] == 'B' && data[3] == 'C')) {
        return parse_rts0_payload(data, size);
    }
    // DXBC container walk (mirrors input_signature in tools/rdna2spirv.py):
    // +28: chunk count, +32: chunk offsets, each chunk is FourCC + u32 size.
    auto at = [&](size_t off, uint32_t& v) {
        if (off + 4 > size) return false;
        v = static_cast<uint32_t>(data[off]) | (static_cast<uint32_t>(data[off + 1]) << 8) |
            (static_cast<uint32_t>(data[off + 2]) << 16) | (static_cast<uint32_t>(data[off + 3]) << 24);
        return true;
    };
    uint32_t nchunks = 0;
    if (!at(28, nchunks)) {
        out.error = "truncated DXBC header";
        return out;
    }
    if (nchunks > 1024) {
        out.error = "absurd DXBC chunk count";
        return out;
    }
    for (uint32_t k = 0; k < nchunks; ++k) {
        uint32_t off = 0;
        // NOTE: off and payload_size are u32; promote to size_t BEFORE adding
        // or a hostile container wraps the check and turns it into an OOB read.
        if (!at(32 + 4 * static_cast<size_t>(k), off) ||
            static_cast<size_t>(off) + 8 > size) {
            out.error = "truncated DXBC chunk table";
            return out;
        }
        if (data[off] == 'R' && data[off + 1] == 'T' && data[off + 2] == 'S' && data[off + 3] == '0') {
            uint32_t payload_size = 0;
            if (static_cast<size_t>(off) + 8 > size) {
                out.error = "truncated RTS0 chunk header";
                return out;
            }
            payload_size = static_cast<uint32_t>(data[off + 4]) |
                           (static_cast<uint32_t>(data[off + 5]) << 8) |
                           (static_cast<uint32_t>(data[off + 6]) << 16) |
                           (static_cast<uint32_t>(data[off + 7]) << 24);
            if (static_cast<size_t>(off) + 8 + payload_size > size) {
                out.error = "RTS0 chunk overruns its container";
                return out;
            }
            return parse_rts0_payload(data + off + 8, payload_size);
        }
    }
    out.error = "no RTS0 chunk in DXBC container";
    return out;
}

}  // namespace rt::rootsig
