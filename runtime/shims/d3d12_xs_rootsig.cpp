#include "d3d12_xs_rootsig.h"

#include <cstring>

namespace rt::d3d12_xs {
namespace {

D3D12_ROOT_SIGNATURE_FLAGS clear_high_byte(D3D12_ROOT_SIGNATURE_FLAGS flags) {
    return static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(static_cast<uint32_t>(flags) & ~kXboxRootSignatureFlagHighByte);
}

uint32_t rotl(uint32_t x, int c) { return (x << c) | (x >> (32 - c)); }

void md5_block(uint32_t state[4], const uint32_t m[16]) {
    static const uint32_t k[64] = {
        0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
        0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
        0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
        0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
        0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
        0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
        0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
        0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391};
    static const int r[16] = {7, 12, 17, 22, 5, 9, 14, 20, 4, 11, 16, 23, 6, 10, 15, 21};
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    for (int i = 0; i < 64; ++i) {
        uint32_t f;
        int g;
        if (i < 16) { f = (b & c) | (~b & d); g = i; }
        else if (i < 32) { f = (d & b) | (~d & c); g = (5 * i + 1) % 16; }
        else if (i < 48) { f = b ^ c ^ d; g = (3 * i + 5) % 16; }
        else { f = c ^ (b | ~d); g = (7 * i) % 16; }
        const uint32_t t = d;
        d = c;
        c = b;
        b += rotl(a + f + k[i] + m[g], r[i / 16 * 4 + i % 4]);
        a = t;
    }
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
}

}  // namespace

void dxbc_checksum(const uint8_t* blob, size_t size, uint32_t out[4]) {
    // MD5 with the DXBC tail: the bit count goes first in the last block and (bits >> 2) | 1 last, in place of
    // MD5's 64-bit length (the d3dcompiler container hash; vkd3d-shader and Wine use the same).
    const uint8_t* data = blob + 20;
    const uint32_t n = static_cast<uint32_t>(size - 20);
    const uint32_t bits = n * 8;
    uint32_t state[4] = {0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476};
    uint32_t m[16];
    const uint32_t full = n / 64 * 64;
    for (uint32_t i = 0; i < full; i += 64) {
        std::memcpy(m, data + i, 64);
        md5_block(state, m);
    }
    const uint32_t left = n - full;
    uint8_t block[64] = {};
    if (left >= 56) {
        std::memcpy(block, data + full, left);
        block[left] = 0x80;
        std::memcpy(m, block, 64);
        md5_block(state, m);
        std::memset(m, 0, sizeof(m));
    } else {
        std::memcpy(block + 4, data + full, left);
        block[4 + left] = 0x80;
        std::memcpy(m, block, 64);
    }
    m[0] = bits;
    m[15] = (bits >> 2) | 1;
    md5_block(state, m);
    std::memcpy(out, state, 16);
}

bool copy_versioned_root_signature_desc_clear_xbox_flags(const D3D12_VERSIONED_ROOT_SIGNATURE_DESC& in,
                                                         D3D12_VERSIONED_ROOT_SIGNATURE_DESC& out) {
    out = in;
    switch (in.Version) {
        case D3D_ROOT_SIGNATURE_VERSION_1_0:
            out.Desc_1_0.Flags = clear_high_byte(in.Desc_1_0.Flags);
            return true;
        case D3D_ROOT_SIGNATURE_VERSION_1_1:
            out.Desc_1_1.Flags = clear_high_byte(in.Desc_1_1.Flags);
            return true;
        case D3D_ROOT_SIGNATURE_VERSION_1_2:
            out.Desc_1_2.Flags = clear_high_byte(in.Desc_1_2.Flags);
            return true;
        default:
            // Unknown version: forward unchanged.
            return false;
    }
}

bool downgrade_root_signature_1_2_to_1_1(const D3D12_VERSIONED_ROOT_SIGNATURE_DESC& in,
                                         std::vector<D3D12_STATIC_SAMPLER_DESC>& samplers,
                                         D3D12_VERSIONED_ROOT_SIGNATURE_DESC& out) {
    if (in.Version != D3D_ROOT_SIGNATURE_VERSION_1_2) return false;
    const D3D12_ROOT_SIGNATURE_DESC2& desc = in.Desc_1_2;
    if (desc.NumStaticSamplers != 0 && !desc.pStaticSamplers) return false;
    for (UINT i = 0; i < desc.NumStaticSamplers; ++i)
        if (desc.pStaticSamplers[i].Flags != D3D12_SAMPLER_FLAG_NONE) return false;

    samplers.resize(desc.NumStaticSamplers);
    for (UINT i = 0; i < desc.NumStaticSamplers; ++i) {
        const D3D12_STATIC_SAMPLER_DESC1& s = desc.pStaticSamplers[i];
        // D3D12_STATIC_SAMPLER_DESC is D3D12_STATIC_SAMPLER_DESC1 without the trailing Flags.
        samplers[i] = {s.Filter, s.AddressU, s.AddressV, s.AddressW, s.MipLODBias, s.MaxAnisotropy,
                       s.ComparisonFunc, s.BorderColor, s.MinLOD, s.MaxLOD, s.ShaderRegister, s.RegisterSpace,
                       s.ShaderVisibility};
    }
    out = {};
    out.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
    out.Desc_1_1.NumParameters = desc.NumParameters;
    out.Desc_1_1.pParameters = desc.pParameters;  // D3D12_ROOT_PARAMETER1 in both versions
    out.Desc_1_1.NumStaticSamplers = desc.NumStaticSamplers;
    out.Desc_1_1.pStaticSamplers = samplers.empty() ? nullptr : samplers.data();
    out.Desc_1_1.Flags = desc.Flags;
    return true;
}

void copy_root_signature_desc_clear_xbox_flags(const D3D12_ROOT_SIGNATURE_DESC& in, D3D12_ROOT_SIGNATURE_DESC& out) {
    out = in;
    out.Flags = clear_high_byte(in.Flags);
}

bool copy_rts0_blob_clear_xbox_flags(const void* data, size_t size, std::vector<uint8_t>& out) {
    out.clear();
    if (!data || size < 36) return false;

    const uint8_t* bytes = static_cast<const uint8_t*>(data);
    // The host deserializer only consumes the DXBC container the serializer
    // produces, so a non-DXBC blob is forwarded unchanged (no rewrite).
    if (std::memcmp(bytes, "DXBC", 4) != 0) return false;

    uint32_t chunk_count = 0;
    std::memcpy(&chunk_count, bytes + 28, sizeof(chunk_count));
    if (chunk_count == 0 || chunk_count > 16 || 32 + chunk_count * 4 > size) return false;

    for (uint32_t i = 0; i < chunk_count; ++i) {
        uint32_t off = 0;
        std::memcpy(&off, bytes + 32 + i * 4, sizeof(off));
        // Chunk header (FourCC + size) plus the fixed RTS0 header up to flags.
        if (off > size || size - off < 8 + 24) continue;
        if (std::memcmp(bytes + off, "RTS0", 4) != 0) continue;

        const uint8_t* payload = bytes + off + 8;
        uint32_t flags = 0;
        std::memcpy(&flags, payload + 20, sizeof(flags));
        const uint32_t cleared = static_cast<uint32_t>(clear_high_byte(
            static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(flags)));
        if (cleared == flags) return false;  // no Xbox high byte

        out.assign(bytes, bytes + size);
        std::memcpy(out.data() + off + 8 + 20, &cleared, sizeof(cleared));
        // The host deserializer checks the container hash (bytes 4..19), so a stale one is E_INVALIDARG.
        uint32_t hash[4];
        dxbc_checksum(out.data(), out.size(), hash);
        std::memcpy(out.data() + 4, hash, sizeof(hash));
        return true;
    }
    return false;
}

}  // namespace rt::d3d12_xs
