#pragma once

// Finds the translated SPIR-V for a shader the game hands to D3D12 (docs/d3d12_vulkan.md section 3.6).
//
// A pipeline's bytecode is a DXBC container. Its DXIL part holds an XDAT table whose entry 0x0D is an XBCH header
// followed by a zlib stream; the stream inflates to the FSSM blob, the native RDNA2 program the translator read
// (tools/fssm_unpack.py does the same in Python). The SHA-256 of that blob is the `fssm_sha256` field the translator
// writes into `spirv/index.json`, next to `<name>.spv`. This header needs no Windows, Vulkan or third-party code:
// it carries its own inflate (RFC 1950/1951), SHA-256 (FIPS 180-4) and a minimal JSON reader.

#include "spirv_words.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace leonida {

// ---------------------------------------------------------------------------------------------------------------
// zlib inflate

namespace detail {

class BitReader {
public:
    explicit BitReader(std::span<const std::uint8_t> in) : in_(in) {}

    std::uint32_t bits(int need) {
        std::uint64_t value = buffer_;
        while (count_ < need) {
            if (pos_ >= in_.size()) {
                failed_ = true;
                return 0;
            }
            value |= std::uint64_t(in_[pos_++]) << count_;
            count_ += 8;
        }
        buffer_ = value >> need;
        count_ -= need;
        return std::uint32_t(value & ((std::uint64_t(1) << need) - 1));
    }

    void align() {
        buffer_ = 0;
        count_ = 0;
    }

    bool byte(std::uint8_t& out) {
        if (pos_ >= in_.size()) return failed_ = true, false;
        out = in_[pos_++];
        return true;
    }

    std::size_t position() const { return pos_; }
    bool failed() const { return failed_; }

private:
    std::span<const std::uint8_t> in_;
    std::size_t pos_ = 0;
    std::uint64_t buffer_ = 0;
    int count_ = 0;
    bool failed_ = false;
};

struct Huffman {
    std::array<std::int16_t, 16> count{};
    std::array<std::int16_t, 288> symbol{};
};

// Canonical Huffman table from code lengths. Returns false for an over-subscribed set.
inline bool build_huffman(Huffman& h, const std::int16_t* lengths, int n) {
    h.count.fill(0);
    for (int i = 0; i < n; ++i) ++h.count[lengths[i]];
    if (h.count[0] == n) return true;  // no codes: valid only if never used
    int left = 1;
    for (int len = 1; len < 16; ++len) {
        left <<= 1;
        left -= h.count[len];
        if (left < 0) return false;
    }
    std::array<std::int16_t, 16> offsets{};
    for (int len = 1; len < 15; ++len) offsets[len + 1] = std::int16_t(offsets[len] + h.count[len]);
    for (int i = 0; i < n; ++i)
        if (lengths[i]) h.symbol[offsets[lengths[i]]++] = std::int16_t(i);
    return true;
}

inline int decode(BitReader& in, const Huffman& h) {
    int code = 0, first = 0, index = 0;
    for (int len = 1; len < 16; ++len) {
        code |= int(in.bits(1));
        const int count = h.count[len];
        if (code - count < first) return h.symbol[index + (code - first)];
        index += count;
        first += count;
        first <<= 1;
        code <<= 1;
        if (in.failed()) return -1;
    }
    return -1;
}

inline bool inflate_codes(BitReader& in, std::vector<std::uint8_t>& out, const Huffman& lengths,
                          const Huffman& distances, std::size_t limit) {
    static constexpr std::int16_t length_base[29] = {3,  4,  5,  6,  7,  8,  9,  10, 11,  13,  15,  17,  19,  23, 27,
                                                     31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
    static constexpr std::int16_t length_extra[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2,
                                                      2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
    static constexpr std::int16_t distance_base[30] = {1,    2,    3,    4,    5,    7,     9,     13,    17,  25,
                                                       33,   49,   65,   97,   129,  193,   257,   385,   513, 769,
                                                       1025, 1537, 2049, 3073, 4097, 6145,  8193,  12289, 16385, 24577};
    static constexpr std::int16_t distance_extra[30] = {0, 0, 0, 0, 1, 1, 2, 2,  3,  3,  4,  4,  5,  5,  6,
                                                        6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};
    for (;;) {
        int symbol = decode(in, lengths);
        if (symbol < 0 || in.failed()) return false;
        if (symbol < 256) {
            if (out.size() >= limit) return false;
            out.push_back(std::uint8_t(symbol));
            continue;
        }
        if (symbol == 256) return true;
        symbol -= 257;
        if (symbol >= 29) return false;
        const std::size_t length = std::size_t(length_base[symbol]) + in.bits(length_extra[symbol]);
        const int d = decode(in, distances);
        if (d < 0 || d >= 30) return false;
        const std::size_t distance = std::size_t(distance_base[d]) + in.bits(distance_extra[d]);
        if (in.failed() || distance > out.size() || out.size() + length > limit) return false;
        const std::size_t from = out.size() - distance;
        for (std::size_t i = 0; i < length; ++i) out.push_back(out[from + i]);
    }
}

} // namespace detail

// Inflates a zlib stream (RFC 1950: header, deflate data, Adler-32). Returns false for a malformed stream, a
// preset dictionary, a checksum mismatch, or output beyond `limit` bytes.
inline bool zlib_inflate(std::span<const std::uint8_t> in, std::vector<std::uint8_t>& out,
                         std::size_t limit = std::size_t(1) << 30) {
    out.clear();
    if (in.size() < 6) return false;
    const unsigned cmf = in[0], flg = in[1];
    if ((cmf & 0x0f) != 8 || (cmf >> 4) > 7 || ((cmf << 8) | flg) % 31 != 0 || (flg & 0x20)) return false;
    detail::BitReader reader(in.subspan(2));
    for (bool last = false; !last;) {
        last = reader.bits(1) != 0;
        const std::uint32_t type = reader.bits(2);
        if (reader.failed()) return false;
        if (type == 0) {  // stored
            reader.align();
            std::uint8_t b[4];
            for (auto& x : b)
                if (!reader.byte(x)) return false;
            const unsigned length = b[0] | (b[1] << 8), complement = b[2] | (b[3] << 8);
            if (length != (~complement & 0xffff) || out.size() + length > limit) return false;
            for (unsigned i = 0; i < length; ++i) {
                std::uint8_t x;
                if (!reader.byte(x)) return false;
                out.push_back(x);
            }
        } else if (type == 1) {  // fixed Huffman
            static const auto tables = [] {
                std::pair<detail::Huffman, detail::Huffman> t;
                std::int16_t lengths[288];
                for (int i = 0; i < 144; ++i) lengths[i] = 8;
                for (int i = 144; i < 256; ++i) lengths[i] = 9;
                for (int i = 256; i < 280; ++i) lengths[i] = 7;
                for (int i = 280; i < 288; ++i) lengths[i] = 8;
                detail::build_huffman(t.first, lengths, 288);
                for (int i = 0; i < 30; ++i) lengths[i] = 5;
                detail::build_huffman(t.second, lengths, 30);
                return t;
            }();
            if (!detail::inflate_codes(reader, out, tables.first, tables.second, limit)) return false;
        } else if (type == 2) {  // dynamic Huffman
            const int literal_count = int(reader.bits(5)) + 257, distance_count = int(reader.bits(5)) + 1;
            const int code_count = int(reader.bits(4)) + 4;
            if (literal_count > 286 || distance_count > 30) return false;
            static constexpr int order[19] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};
            std::int16_t lengths[320] = {};
            for (int i = 0; i < code_count; ++i) lengths[order[i]] = std::int16_t(reader.bits(3));
            detail::Huffman code_lengths;
            if (!detail::build_huffman(code_lengths, lengths, 19)) return false;
            for (int i = 0; i < literal_count + distance_count;) {
                const int symbol = detail::decode(reader, code_lengths);
                if (symbol < 0) return false;
                if (symbol < 16) {
                    lengths[i++] = std::int16_t(symbol);
                    continue;
                }
                std::int16_t value = 0;
                int repeat;
                if (symbol == 16) {
                    if (i == 0) return false;
                    value = lengths[i - 1];
                    repeat = 3 + int(reader.bits(2));
                } else if (symbol == 17) {
                    repeat = 3 + int(reader.bits(3));
                } else {
                    repeat = 11 + int(reader.bits(7));
                }
                if (i + repeat > literal_count + distance_count) return false;
                while (repeat--) lengths[i++] = value;
            }
            if (lengths[256] == 0) return false;  // no end-of-block code
            detail::Huffman literal, distance;
            if (!detail::build_huffman(literal, lengths, literal_count) ||
                !detail::build_huffman(distance, lengths + literal_count, distance_count))
                return false;
            if (!detail::inflate_codes(reader, out, literal, distance, limit)) return false;
        } else {
            return false;
        }
        if (reader.failed()) return false;
    }
    // Adler-32, big-endian, after the deflate data's last whole byte.
    const std::size_t at = 2 + reader.position();
    if (at + 4 > in.size()) return false;
    std::uint32_t a = 1, b = 0;
    for (const std::uint8_t x : out) {
        a = (a + x) % 65521;
        b = (b + a) % 65521;
    }
    const std::uint32_t expected = (std::uint32_t(in[at]) << 24) | (std::uint32_t(in[at + 1]) << 16) |
                                   (std::uint32_t(in[at + 2]) << 8) | in[at + 3];
    return ((b << 16) | a) == expected;
}

// ---------------------------------------------------------------------------------------------------------------
// SHA-256

inline std::string sha256_hex(std::span<const std::uint8_t> data) {
    static constexpr std::uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    std::uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                          0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    const auto rotr = [](std::uint32_t x, int n) { return (x >> n) | (x << (32 - n)); };
    std::vector<std::uint8_t> message(data.begin(), data.end());
    const std::uint64_t bits = std::uint64_t(data.size()) * 8;
    message.push_back(0x80);
    while (message.size() % 64 != 56) message.push_back(0);
    for (int i = 7; i >= 0; --i) message.push_back(std::uint8_t(bits >> (8 * i)));
    for (std::size_t block = 0; block < message.size(); block += 64) {
        std::uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = (std::uint32_t(message[block + 4 * i]) << 24) | (std::uint32_t(message[block + 4 * i + 1]) << 16) |
                   (std::uint32_t(message[block + 4 * i + 2]) << 8) | message[block + 4 * i + 3];
        for (int i = 16; i < 64; ++i) {
            const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        std::uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const std::uint32_t t1 = hh + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + k[i] + w[i];
            const std::uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            hh = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }
    static constexpr char hex[] = "0123456789abcdef";
    std::string out;
    for (const std::uint32_t word : h)
        for (int shift = 28; shift >= 0; shift -= 4) out.push_back(hex[(word >> shift) & 0xf]);
    return out;
}

// ---------------------------------------------------------------------------------------------------------------
// JSON (the subset the translator's index uses, which is all of JSON except number precision beyond double)

struct JsonObjectEntry;

struct Json {
    enum class Kind { null, boolean, number, string, array, object } kind = Kind::null;
    bool boolean = false;
    double number = 0;
    std::string string;
    std::vector<Json> array;
    // Declared through the incomplete JsonObjectEntry and destroyed out of line: a vector element type that holds a
    // Json by value cannot be instantiated while Json is still being defined. GCC tolerates it, Clang with libstdc++
    // does not (the pair<string, Json> form used to fail to build there). Structured bindings still work.
    std::vector<JsonObjectEntry> object;

    ~Json();
    const Json* find(std::string_view key) const;
};

struct JsonObjectEntry {
    std::string key;
    Json value;
};

inline Json::~Json() = default;

inline const Json* Json::find(std::string_view key) const {
    for (const auto& [k, v] : object)
        if (k == key) return &v;
    return nullptr;
}

class JsonReader {
public:
    explicit JsonReader(std::string_view text) : text_(text) {}

    bool parse(Json& out) {
        if (!value(out, 0)) return false;
        space();
        return pos_ == text_.size();
    }

private:
    void space() {
        while (pos_ < text_.size() && (text_[pos_] == ' ' || text_[pos_] == '\n' || text_[pos_] == '\r' ||
                                       text_[pos_] == '\t'))
            ++pos_;
    }

    bool literal(std::string_view word) {
        if (text_.substr(pos_, word.size()) != word) return false;
        pos_ += word.size();
        return true;
    }

    bool value(Json& out, int depth) {
        if (depth > 64) return false;
        space();
        if (pos_ >= text_.size()) return false;
        const char c = text_[pos_];
        if (c == '{') {
            out.kind = Json::Kind::object;
            ++pos_;
            space();
            if (pos_ < text_.size() && text_[pos_] == '}') return ++pos_, true;
            for (;;) {
                space();
                std::string key;
                if (!string(key)) return false;
                space();
                if (pos_ >= text_.size() || text_[pos_++] != ':') return false;
                Json item;
                if (!value(item, depth + 1)) return false;
                out.object.emplace_back(std::move(key), std::move(item));
                space();
                if (pos_ >= text_.size()) return false;
                if (text_[pos_] == ',') {
                    ++pos_;
                    continue;
                }
                return text_[pos_++] == '}';
            }
        }
        if (c == '[') {
            out.kind = Json::Kind::array;
            ++pos_;
            space();
            if (pos_ < text_.size() && text_[pos_] == ']') return ++pos_, true;
            for (;;) {
                Json item;
                if (!value(item, depth + 1)) return false;
                out.array.push_back(std::move(item));
                space();
                if (pos_ >= text_.size()) return false;
                if (text_[pos_] == ',') {
                    ++pos_;
                    continue;
                }
                return text_[pos_++] == ']';
            }
        }
        if (c == '"') {
            out.kind = Json::Kind::string;
            return string(out.string);
        }
        if (literal("true")) return out.kind = Json::Kind::boolean, out.boolean = true, true;
        if (literal("false")) return out.kind = Json::Kind::boolean, out.boolean = false, true;
        if (literal("null")) return out.kind = Json::Kind::null, true;
        // Numbers through strtod: Apple's libc++ does not provide the floating-point from_chars overload. The
        // token is already limited to JSON number characters, and the end check requires all of it to be consumed.
        const std::size_t start = pos_;
        while (pos_ < text_.size() && std::strchr("+-0123456789.eE", text_[pos_])) ++pos_;
        if (pos_ == start) return false;
        out.kind = Json::Kind::number;
        const std::string token(text_.substr(start, pos_ - start));
        char* end = nullptr;
        errno = 0;
        out.number = std::strtod(token.c_str(), &end);
        return errno != ERANGE && end == token.c_str() + token.size();
    }

    static int hex_digit(char c) {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    }

    bool string(std::string& out) {
        if (pos_ >= text_.size() || text_[pos_++] != '"') return false;
        while (pos_ < text_.size()) {
            const char c = text_[pos_++];
            if (c == '"') return true;
            if (c != '\\') {
                out.push_back(c);
                continue;
            }
            if (pos_ >= text_.size()) return false;
            const char e = text_[pos_++];
            switch (e) {
            case '"': case '\\': case '/': out.push_back(e); break;
            case 'b': out.push_back('\b'); break;
            case 'f': out.push_back('\f'); break;
            case 'n': out.push_back('\n'); break;
            case 'r': out.push_back('\r'); break;
            case 't': out.push_back('\t'); break;
            case 'u': {
                if (pos_ + 4 > text_.size()) return false;
                unsigned code = 0;
                for (int i = 0; i < 4; ++i) {  // exactly four hex digits
                    const int digit = hex_digit(text_[pos_++]);
                    if (digit < 0) return false;
                    code = code << 4 | unsigned(digit);
                }
                if (code < 0x80) {
                    out.push_back(char(code));
                } else if (code < 0x800) {
                    out.push_back(char(0xc0 | (code >> 6)));
                    out.push_back(char(0x80 | (code & 0x3f)));
                } else {  // surrogate pairs are kept as two 3-byte sequences; index names are ASCII
                    out.push_back(char(0xe0 | (code >> 12)));
                    out.push_back(char(0x80 | ((code >> 6) & 0x3f)));
                    out.push_back(char(0x80 | (code & 0x3f)));
                }
                break;
            }
            default: return false;
            }
        }
        return false;
    }

    std::string_view text_;
    std::size_t pos_ = 0;
};

// ---------------------------------------------------------------------------------------------------------------
// DXBC containers and the FSSM program

// The payload of one DXBC part ("DXIL", "PRIV", ...), or an empty span.
inline std::span<const std::uint8_t> dxbc_part(std::span<const std::uint8_t> container, const char (&fourcc)[5]) {
    const auto u32 = [&](std::size_t at) -> std::uint32_t {
        if (at + 4 > container.size()) return 0;
        std::uint32_t v;
        std::memcpy(&v, container.data() + at, 4);
        return v;
    };
    if (container.size() < 32 || std::memcmp(container.data(), "DXBC", 4) != 0) return {};
    const std::uint32_t parts = u32(28);
    if (parts > (container.size() - 32) / 4) return {};
    for (std::uint32_t i = 0; i < parts; ++i) {
        const std::uint32_t at = u32(32 + 4 * std::size_t(i));
        if (std::size_t(at) + 8 > container.size()) return {};
        const std::uint32_t size = u32(std::size_t(at) + 4);
        if (std::size_t(at) + 8 + size > container.size()) return {};
        if (std::memcmp(container.data() + at, fourcc, 4) == 0) return container.subspan(at + 8, size);
    }
    return {};
}

// The FSSM blob of a shader: inflated from the container's DXIL/XDAT/XBCH entry, or the bytecode itself when it is
// already a bare FSSM blob. Empty when neither applies.
inline std::vector<std::uint8_t> fssm_program(std::span<const std::uint8_t> bytecode) {
    if (bytecode.size() >= 4 && std::memcmp(bytecode.data(), "FSSM", 4) == 0)
        return {bytecode.begin(), bytecode.end()};
    const auto dxil = dxbc_part(bytecode, "DXIL");
    const auto u32 = [&](std::size_t at) -> std::uint32_t {
        if (at + 4 > dxil.size()) return 0;
        std::uint32_t v;
        std::memcpy(&v, dxil.data() + at, 4);
        return v;
    };
    std::size_t xdat = 0;
    for (; xdat + 4 <= dxil.size(); ++xdat)
        if (std::memcmp(dxil.data() + xdat, "XDAT", 4) == 0) break;
    if (xdat + 12 > dxil.size()) return {};
    // Some containers carry two different 0x0D payloads (195 of 26,503, tools/fssm_unpack.py). The translator's
    // fssm_sha256 is taken from the last one (fssm_unpack.py's default, which make_game.py uses), so this does too.
    const std::uint32_t count = u32(xdat + 4);
    std::span<const std::uint8_t> blob;
    for (std::uint32_t i = 0; i < count; ++i) {
        const std::size_t entry = xdat + 12 + 12 * std::size_t(i);
        if (entry + 12 > dxil.size()) return {};
        const std::uint32_t tag = u32(entry), offset = u32(entry + 4), size = u32(entry + 8);
        if (tag != 0x0d) continue;
        if (std::size_t(offset) + size > dxil.size() || size < 24) return {};
        blob = dxil.subspan(offset, size);
    }
    if (blob.empty() || std::memcmp(blob.data(), "XBCH", 4) != 0) return {};
    std::uint32_t unpacked;
    std::memcpy(&unpacked, blob.data() + 16, 4);
    std::vector<std::uint8_t> out;
    if (!zlib_inflate(blob.subspan(24), out, unpacked) || out.size() != unpacked) return {};
    return out;
}

// ---------------------------------------------------------------------------------------------------------------
// FSPX payloads: the FidelityFX DLL's compute programs (docs/fspx.md)

// The two entries of an FSPX payload the runtime reads: kind 0 (dispatch state) and kind 6 (the program).
struct FspxParts {
    std::span<const std::uint8_t> state, code;
};

// Kinds 0 and 6 of an FSPX payload, both empty when the payload is not one the DLL's reader accepts. Same rules as
// that reader and tools/fspx_unpack.py's parse_fspx: the total is clamped to the buffer, the directory must fit it,
// an entry whose range starts inside the directory or ends past the total is skipped, the last entry of a kind wins,
// and kind 0 and kind 6 must both be there and non-empty.
inline FspxParts fspx_parts(std::span<const std::uint8_t> payload) {
    const auto u32 = [&](std::size_t at) {
        std::uint32_t v;
        std::memcpy(&v, payload.data() + at, 4);
        return v;
    };
    if (payload.size() < 16 || std::memcmp(payload.data(), "FSPX", 4) != 0) return {};
    const std::uint64_t total = std::min<std::uint64_t>(u32(12), payload.size());
    const std::uint64_t end_dir = 16 + 16 * std::uint64_t(u32(8));
    if (total < end_dir) return {};
    std::uint64_t code_offset = 0, code_size = 0, state_offset = 0, state_size = 0;
    for (std::uint64_t at = 16; at < end_dir; at += 16) {
        const std::uint32_t kind = u32(std::size_t(at)), offset = u32(std::size_t(at) + 4),
                            size = u32(std::size_t(at) + 8);
        if (offset < end_dir || std::uint64_t(offset) + size > total) continue;  // the DLL skips it too
        if (kind == 6) code_offset = offset, code_size = size;
        if (kind == 0) state_offset = offset, state_size = size;
    }
    if (!state_size || !code_size) return {};
    return {payload.subspan(std::size_t(state_offset), std::size_t(state_size)),
            payload.subspan(std::size_t(code_offset), std::size_t(code_size))};
}

// The kind-6 program of an FSPX payload, or an empty span (fspx_parts).
inline std::span<const std::uint8_t> fspx_program(std::span<const std::uint8_t> payload) {
    return fspx_parts(payload).code;
}

// One pointer and size pair of the description device slot 67 receives (docs/fspx.md, "How the DLL reads it").
struct FspxPart {
    std::uint32_t kind;
    const void* data;
    std::uint64_t size;
};

// The largest entry fspx_payload_from_parts takes: far above the 1.0.2.1 DLL's largest (a 41,536-byte program), and
// small enough that the payload's u32 offsets cannot overflow.
inline constexpr std::uint64_t kFspxPartLimit = 16u << 20;

// An FSPX payload built from the pairs of a slot-67 description, in the layout fspx_parts() reads: the header, one
// directory entry per present pair and the entries' bytes after it, each at a 16-byte boundary. A pair is present
// when its pointer is non-null and its size non-zero; a pointer without a size or a size without a pointer, a size
// over kFspxPartLimit, a kind other than 0, 1, 6 and 7, the same kind twice, or a missing kind 0 or kind 6 gives an
// empty result, and nothing is read. Every pointer is read only for its own size.
inline std::vector<std::uint8_t> fspx_payload_from_parts(std::span<const FspxPart> parts) {
    bool seen[8] = {};
    std::vector<const FspxPart*> present;
    for (const FspxPart& part : parts) {
        if (part.kind > 7 || (part.kind != 0 && part.kind != 1 && part.kind != 6 && part.kind != 7)) return {};
        if (seen[part.kind]) return {};
        seen[part.kind] = true;
        if (!part.data && !part.size) continue;
        if (!part.data || !part.size || part.size > kFspxPartLimit) return {};
        present.push_back(&part);
    }
    const auto has = [&](std::uint32_t kind) {
        return std::any_of(present.begin(), present.end(), [&](const FspxPart* p) { return p->kind == kind; });
    };
    if (!has(0) || !has(6)) return {};
    const auto align = [](std::uint64_t v) { return (v + 15) & ~std::uint64_t(15); };
    std::uint64_t at = align(16 + 16 * std::uint64_t(present.size()));
    std::vector<std::uint64_t> offsets;
    for (const FspxPart* p : present) {
        offsets.push_back(at);
        at = align(at + p->size);
    }
    std::vector<std::uint8_t> out(std::size_t(at), 0);
    const auto put = [&](std::size_t where, std::uint32_t v) { std::memcpy(out.data() + where, &v, 4); };
    std::memcpy(out.data(), "FSPX", 4);
    put(4, 1);
    put(8, std::uint32_t(present.size()));
    put(12, std::uint32_t(at));
    for (std::size_t i = 0; i < present.size(); ++i) {
        put(16 + 16 * i, present[i]->kind);
        put(16 + 16 * i + 4, std::uint32_t(offsets[i]));
        put(16 + 16 * i + 8, std::uint32_t(present[i]->size));
        std::memcpy(out.data() + offsets[i], present[i]->data, std::size_t(present[i]->size));
    }
    return out;
}

// Whether `code` looks like a bare kind-6 program: the 36-byte prologue of three VOP3 words with opcode 0x180
// (bits 31..16 = 0xD580) at words 0, 3 and 6, which all 29 programs of the 1.0.2.1 DLL have (docs/fspx.md).
inline bool fspx_bare_program(std::span<const std::uint8_t> code) {
    if (code.size() < 36 || code.size() % 4) return false;
    for (const std::size_t word : {0u, 3u, 6u}) {
        std::uint32_t v;
        std::memcpy(&v, code.data() + 4 * word, 4);
        if (v >> 16 != 0xD580) return false;
    }
    return true;
}

// What a compute shader's bytecode is looked up by in the translator's index.
struct ComputeProgram {
    enum class Kind { none, fssm, fspx } kind = Kind::none;
    // The bytes whose SHA-256 is the index key (fssm_sha256 or fspx_code_sha256), and which the pipeline copies to
    // the GPU address arena as its code base: the program's code offsets are into them.
    std::vector<std::uint8_t> bytes;
    // An FSPX program from a container or a bare payload: its kind-0 entry, whose SHA-256 (fspx_state_sha256) picks
    // among translations of the same kind-6 bytes. Empty for a bare kind-6 program, which comes without it.
    std::vector<std::uint8_t> state;
};

// The program in a compute shader's bytecode:
//   - an FSSM blob, bare or in a DXBC container's DXIL part (fssm_program), the game's own shaders;
//   - the kind-6 program of an FSPX payload in a DXBC container's ACCC part, what the FidelityFX DLL passes to device
//     slot 62, or of a bare FSPX payload;
//   - a bare kind-6 program, what the DLL passes in its slot-67 description at +0x28 (with its exact size at +0x30).
// A DXBC container whose ACCC part is not a usable FSPX payload, and anything else, gives Kind::none.
inline ComputeProgram compute_program(std::span<const std::uint8_t> bytecode) {
    ComputeProgram out;
    out.bytes = fssm_program(bytecode);
    if (!out.bytes.empty()) {
        out.kind = ComputeProgram::Kind::fssm;
        return out;
    }
    FspxParts parts;
    if (bytecode.size() >= 4 && std::memcmp(bytecode.data(), "DXBC", 4) == 0)
        parts = fspx_parts(dxbc_part(bytecode, "ACCC"));
    else if (bytecode.size() >= 4 && std::memcmp(bytecode.data(), "FSPX", 4) == 0)
        parts = fspx_parts(bytecode);
    else if (fspx_bare_program(bytecode))
        parts.code = bytecode;
    if (parts.code.empty()) return out;
    out.kind = ComputeProgram::Kind::fspx;
    out.bytes.assign(parts.code.begin(), parts.code.end());
    out.state.assign(parts.state.begin(), parts.state.end());
    return out;
}

// ---------------------------------------------------------------------------------------------------------------
// The translator's spirv/index.json

// One element of a shader's DXBC input signature (ISG1), as tools/shader_meta.py records it. For a vertex shader
// the register is the Vulkan input location the translator gave it (docs/design.md, "Vertex fetch").
struct SignatureElement {
    std::string semantic;
    std::uint32_t semantic_index = 0;
    std::uint32_t register_index = 0;
    std::uint32_t system_value = 0;  // 0: an ordinary input fed by the input layout
    std::string component_type;      // "float", "uint", "sint"
};

struct ShaderEntry {
    std::string name;
    std::string stage;  // "vs", "ps", "cs", "ms", "gs"
    std::string fssm_sha256;
    std::string fspx_code_sha256;   // an FSPX program (docs/fspx.md): the SHA-256 of its kind-6 bytes
    std::string fspx_state_sha256;  // and of its kind-0 entry (dispatch state)
    std::size_t binding_count = 0;  // set-0 bindings the module declares
    std::uint32_t push_constant_bytes = 0;
    std::array<std::uint32_t, 3> local_size{1, 1, 1};
    // Workgroup memory the module declares: its LDS array plus what --wave64-emulate adds (SHADER-309). A compute
    // pipeline needs it at or under the device's maxComputeSharedMemorySize (32768 on some drivers).
    std::uint32_t lds_bytes = 0;
    bool uses_code_base = false;
    bool wave64 = false;
    // The subgroup size the translator assumed (64 for a wave64 shader, 32 otherwise); 0 when the entry says null,
    // as one translated with --wave64-emulate does, or has none.
    std::uint32_t subgroup_size = 0;
    // "workgroup" for a shader translated with --wave64-emulate: it uses no subgroup operation and is exact on any
    // subgroup size. Empty for a native translation.
    std::string wave64_exchange;
    bool emulated() const { return !wave64_exchange.empty(); }
    std::vector<SignatureElement> input_signature;  // vertex shaders: what the input layout must feed
    std::vector<std::uint32_t> inputs;              // pixel shaders: input locations declared
    std::vector<std::uint32_t> outputs;             // pixel/vertex shaders: output locations written
    // --f16-interp-fallback: param/attr index -> {low_half_location, high_half_location} (SHADER-378).
    std::map<std::uint32_t, std::array<std::uint32_t, 2>> f16_halves;
    // The S#s the shader builds from immediates (the index's literal_samplers): no heap slot names them, so a
    // pipeline that uses the shader registers them in the bindless set itself (docs/design.md, Literal samplers).
    std::vector<std::array<std::uint32_t, 4>> literal_samplers;
    // Mesh shaders: the output array sizes the SPIR-V declares (OutputVertices / OutputPrimitivesEXT), checked
    // against the device's maxMeshOutputVertices / maxMeshOutputPrimitives when a pipeline is created. 0: not given.
    std::uint32_t max_vertices = 0;
    std::uint32_t max_primitives = 0;
};

class ShaderLibrary {
public:
    // Reads <dir>/index.json, then the indexes of the programs make_game.py translates from the package's DLLs, each
    // when it is there:
    //   - <dir>/fspx/index.json, the FidelityFX DLL's FSPX programs, entries named "fspx/<name>" (docs/fspx.md);
    //   - <dir>/bvh/index.json, XboxBvhBuilder.dll's FSSM programs, entries named "bvh/<name>" and found by
    //     fssm_sha256 like the game's own shaders (docs/embedded_shaders.md).
    // Entries with neither fssm_sha256 nor fspx_code_sha256 cannot be found by content and are skipped. Both folders
    // are optional: a missing or unreadable index there leaves the main one, and the other folder's, loaded.
    bool load(const std::filesystem::path& dir, std::string* error = nullptr) {
        const auto fail = [&](const std::string& why) {
            if (error) *error = why;
            return false;
        };
        // Nothing here may throw into the D3D12 shim (only allocation can, and that is caught too).
        try {
            if (!load_index(dir, "", fail)) return false;
        } catch (...) {
            return fail("index.json could not be read");
        }
        for (const char* folder : {"fspx", "bvh"}) {
            try {
                const auto ignore = [](const std::string&) { return false; };
                const std::filesystem::path sub = dir / folder;
                if (std::ifstream(sub / "index.json", std::ios::binary))
                    load_index(sub, std::string(folder) + "/", ignore);
            } catch (...) {
            }
        }
        return true;
    }

    bool loaded() const { return !dir_.empty(); }
    std::size_t size() const { return count_; }
    // Entries loaded from <dir>/fspx/index.json. A corpus of RPF containers never reaches these: their programs
    // come from FidelityFX DLLs, so callers comparing reached entries with size() must leave them out.
    std::size_t fspx_size() const { return fspx_count_; }
    // Entries loaded from <dir>/bvh/index.json. No RPF container reaches these either: XboxBvhBuilder.dll passes
    // their containers itself.
    std::size_t bvh_size() const { return bvh_count_; }

    // The first entry for this program. Two containers can hold the same FSSM program with different input
    // signatures; `candidates` returns all of them.
    const ShaderEntry* find(std::string_view stage, std::string_view fssm_sha256) const {
        const auto all = candidates(stage, fssm_sha256);
        return all.empty() ? nullptr : all.front();
    }

    std::vector<const ShaderEntry*> candidates(std::string_view stage, std::string_view fssm_sha256) const {
        return lookup(entries_, stage, fssm_sha256);
    }

    // The entries of an FSPX program, by the SHA-256 of its kind-6 bytes (fspx_code_sha256).
    std::vector<const ShaderEntry*> fspx_candidates(std::string_view stage, std::string_view fspx_code_sha256) const {
        return lookup(fspx_entries_, stage, fspx_code_sha256);
    }

    // The entries for a program compute_program() found, by the key of its kind (docs/fspx.md, "Build and run-time
    // lookup"):
    //   - FSSM: by fssm_sha256, as before;
    //   - FSPX with its kind-0 entry (a container or a bare payload): by fspx_code_sha256, then only the entries
    //     whose fspx_state_sha256 is the SHA-256 of that kind-0 entry;
    //   - a bare kind-6 program, which comes without kind 0: by fspx_code_sha256 when all those entries have the
    //     same module; when their modules differ the program is ambiguous, the result is empty and `ambiguous` is
    //     set, instead of picking one of them.
    std::vector<const ShaderEntry*> program_candidates(std::string_view stage, const ComputeProgram& program,
                                                       bool* ambiguous = nullptr) const {
        if (ambiguous) *ambiguous = false;
        if (program.kind == ComputeProgram::Kind::none) return {};
        const std::string hash = sha256_hex(program.bytes);
        if (program.kind == ComputeProgram::Kind::fssm) return candidates(stage, hash);
        auto all = fspx_candidates(stage, hash);
        if (!program.state.empty()) {
            const std::string state = sha256_hex(program.state);
            std::erase_if(all, [&](const ShaderEntry* e) { return e->fspx_state_sha256 != state; });
            return all;
        }
        if (all.size() > 1) {
            const auto first = spirv(*all.front());
            for (std::size_t i = 1; i < all.size(); ++i) {
                if (first.empty() || spirv(*all[i]) != first) {
                    if (ambiguous) *ambiguous = true;
                    return {};
                }
            }
        }
        return all;
    }

    // The module's words, or empty when <dir>/<name>.spv is missing or not SPIR-V.
    std::vector<std::uint32_t> spirv(const ShaderEntry& entry) const {
        auto path = dir_;
        path += "/" + entry.name + ".spv";
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        return read_spirv_words(file, file.tellg());
    }

private:
    // A non-negative integer field, or 0 when it is missing, not a number or out of range.
    static std::uint32_t u32(const Json* value) {
        if (!value || value->kind != Json::Kind::number) return 0;
        const double n = value->number;
        return n >= 0 && n <= 4294967295.0 ? std::uint32_t(n) : 0;
    }

    using LookupKey = std::pair<std::string_view, std::string_view>;
    struct KeyLess {
        using is_transparent = void;
        bool operator()(LookupKey a, LookupKey b) const { return a < b; }
    };
    using Entries = std::map<std::pair<std::string, std::string>, std::vector<ShaderEntry>, KeyLess>;

    static std::vector<const ShaderEntry*> lookup(const Entries& entries, std::string_view stage, std::string_view key) {
        std::vector<const ShaderEntry*> out;
        const auto it = entries.find(LookupKey{stage, key});
        if (it != entries.end())
            for (const ShaderEntry& e : it->second) out.push_back(&e);
        return out;
    }

    // Reads <dir>/index.json. With an empty prefix it replaces what was loaded (the main index); with a prefix it
    // adds its entries under "<prefix><name>", the path of their modules relative to the main folder.
    template<typename Fail>
    bool load_index(const std::filesystem::path& dir, const std::string& prefix, Fail&& fail) {
        const auto index = dir / "index.json";
        std::ifstream file(index, std::ios::binary);
        if (!file) {
            const auto path = index.u8string();
            return fail("cannot open " + std::string(path.begin(), path.end()));
        }
        const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        constexpr std::string_view utf8_bom = "\xef\xbb\xbf";
        std::string_view json(text);
        if (json.starts_with(utf8_bom)) json.remove_prefix(utf8_bom.size());
        Json root;
        if (!JsonReader(json).parse(root) || root.kind != Json::Kind::object) return fail("index.json is not a JSON object");
        if (prefix.empty()) {
            dir_ = dir;
            entries_.clear();
            fspx_entries_.clear();
            count_ = 0;
            fspx_count_ = 0;
            bvh_count_ = 0;
        }
        for (const auto& [name, meta] : root.object) {
            const Json* hash = meta.find("fssm_sha256");
            const Json* code_hash = meta.find("fspx_code_sha256");
            const Json* stage = meta.find("stage");
            const bool by_fssm = hash && hash->kind == Json::Kind::string;
            const bool by_fspx = code_hash && code_hash->kind == Json::Kind::string;
            if ((!by_fssm && !by_fspx) || !stage || stage->kind != Json::Kind::string) continue;
            ShaderEntry e;
            e.name = prefix + name;
            e.stage = stage->string;
            if (by_fssm) e.fssm_sha256 = hash->string;
            if (by_fspx) e.fspx_code_sha256 = code_hash->string;
            if (const Json* s = meta.find("fspx_state_sha256"); s && s->kind == Json::Kind::string)
                e.fspx_state_sha256 = s->string;
            if (const Json* b = meta.find("bindings"); b && b->kind == Json::Kind::array) e.binding_count = b->array.size();
            e.push_constant_bytes = u32(meta.find("push_constant_bytes"));
            e.lds_bytes = u32(meta.find("lds_bytes"));
            if (const Json* l = meta.find("local_size"); l && l->kind == Json::Kind::array)
                for (std::size_t i = 0; i < l->array.size() && i < 3; ++i) e.local_size[i] = u32(&l->array[i]);
            if (const Json* c = meta.find("uses_code_base"); c && c->kind == Json::Kind::boolean) e.uses_code_base = c->boolean;
            if (const Json* w = meta.find("wave64"); w && w->kind == Json::Kind::boolean) e.wave64 = w->boolean;
            if (const Json* g = meta.find("subgroup_size"); g && g->kind == Json::Kind::number)
                e.subgroup_size = u32(g);
            if (const Json* x = meta.find("wave64_exchange"); x && x->kind == Json::Kind::string)
                e.wave64_exchange = x->string;
            if (const Json* s = meta.find("input_signature"); s && s->kind == Json::Kind::array) {
                for (const Json& item : s->array) {
                    SignatureElement element;
                    if (const Json* v = item.find("semantic"); v && v->kind == Json::Kind::string) element.semantic = v->string;
                    element.semantic_index = u32(item.find("semantic_index"));
                    element.register_index = u32(item.find("register"));
                    element.system_value = u32(item.find("system_value"));
                    if (const Json* v = item.find("component_type"); v && v->kind == Json::Kind::string)
                        element.component_type = v->string;
                    e.input_signature.push_back(std::move(element));
                }
            }
            if (const Json* in = meta.find("inputs"); in && in->kind == Json::Kind::array)
                for (const Json& item : in->array) e.inputs.push_back(u32(&item));
            if (const Json* o = meta.find("outputs"); o && o->kind == Json::Kind::array)
                for (const Json& item : o->array) e.outputs.push_back(u32(&item));
            if (const Json* fh = meta.find("f16_halves"); fh && fh->kind == Json::Kind::object) {
                for (const auto& [key, val] : fh->object) {
                    if (key.empty() || val.kind != Json::Kind::array || val.array.size() != 2) continue;
                    char* end = nullptr;
                    const unsigned long param = std::strtoul(key.c_str(), &end, 10);
                    if (!end || *end != '\0') continue;
                    e.f16_halves[std::uint32_t(param)] = {u32(&val.array[0]), u32(&val.array[1])};
                }
            }
            if (const Json* l = meta.find("literal_samplers"); l && l->kind == Json::Kind::array) {
                for (const Json& item : l->array) {
                    if (item.kind != Json::Kind::array || item.array.size() != 4) continue;
                    std::array<std::uint32_t, 4> words{};
                    for (std::size_t i = 0; i < 4; ++i) words[i] = u32(&item.array[i]);
                    e.literal_samplers.push_back(words);
                }
            }
            e.max_vertices = u32(meta.find("max_vertices"));
            e.max_primitives = u32(meta.find("max_primitives"));
            if (by_fspx) fspx_entries_[{e.stage, e.fspx_code_sha256}].push_back(e);
            if (by_fssm) entries_[{e.stage, e.fssm_sha256}].push_back(std::move(e));
            if (by_fspx && prefix == "fspx/") ++fspx_count_;
            if (prefix == "bvh/") ++bvh_count_;
            ++count_;
        }
        return true;
    }

    std::filesystem::path dir_;
    std::size_t count_ = 0;
    std::size_t fspx_count_ = 0;
    std::size_t bvh_count_ = 0;
    // Entries by (stage, fssm_sha256) and by (stage, fspx_code_sha256), in index order.
    Entries entries_;
    Entries fspx_entries_;
};

} // namespace leonida
