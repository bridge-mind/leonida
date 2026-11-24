// Shader library pieces without a GPU: SHA-256 (FIPS 180-4 vectors), zlib inflate (stored, fixed and dynamic
// Huffman blocks, compressed from synthetic data with Python's zlib), the JSON reader, FSSM extraction from a
// synthetic DXBC container, and index lookup by content hash. No game data.
#include "../src/shader_library.h"
#include "test_temp_dir.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <sstream>
#include <string>
#include <vector>

#define CHECK(condition) do {                                    \
    if (!(condition)) {                                          \
        std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
        return 1;                                                \
    }                                                            \
} while (0)

namespace {

std::vector<std::uint8_t> bytes(const std::string& s) { return {s.begin(), s.end()}; }

// The dynamic-Huffman sample: an LCG over a skewed alphabet (the same generator produced `dynamic` below).
std::vector<std::uint8_t> skewed(std::size_t n) {
    static constexpr char alphabet[] = "eeeeettaaoinshrdlu";
    std::vector<std::uint8_t> out;
    std::uint32_t x = 12345;
    for (std::size_t i = 0; i < n; ++i) {
        x = (x * 1103515245u + 12345u) & 0x7fffffffu;
        out.push_back(std::uint8_t(alphabet[(x >> 16) % 18]));
    }
    return out;
}

const std::vector<std::uint8_t> stored = {
    0x78, 0x01, 0x01, 0x28, 0x00, 0xd7, 0xff, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b,
    0x0c, 0x0d, 0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e,
    0x1f, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x29, 0xcc, 0x03, 0x0d};
const std::vector<std::uint8_t> fixed = {0x78, 0x01, 0xcb, 0x48, 0xcd, 0xc9, 0xc9, 0x57, 0x40, 0x26, 0x73,
                                         0x52, 0xf3, 0xf3, 0x32, 0x53, 0x12, 0x01, 0x7a, 0xaf, 0x09, 0x79};
const std::vector<std::uint8_t> dynamic = {
    0x78, 0xda, 0x1d, 0x92, 0xdb, 0x0a, 0xc0, 0x30, 0x08, 0x43, 0xbf, 0x55, 0x30, 0xa0, 0x20, 0x0a, 0x5e, 0xfe, 0x7f,
    0xe9, 0xf6, 0x54, 0x66, 0x6c, 0x4e, 0xb4, 0x13, 0xb6, 0x22, 0x26, 0x40, 0x08, 0xaa, 0xc0, 0xc3, 0x40, 0x0e, 0x17,
    0x1e, 0xe5, 0x09, 0x5f, 0x44, 0xc9, 0xab, 0x01, 0x10, 0x71, 0xa4, 0x40, 0x25, 0x25, 0xa2, 0xb1, 0x28, 0x31, 0x44,
    0x88, 0xcb, 0x62, 0x7c, 0x29, 0xda, 0x38, 0x43, 0x52, 0x78, 0xd2, 0x5b, 0x23, 0x47, 0x99, 0xb8, 0x61, 0xd8, 0x05,
    0x1f, 0xa4, 0x42, 0xf3, 0x26, 0x84, 0x17, 0xe4, 0xf0, 0x5e, 0x83, 0x81, 0x92, 0x35, 0x7a, 0x05, 0x2d, 0x5a, 0xce,
    0xb1, 0x3d, 0xab, 0xa5, 0xca, 0x0a, 0x95, 0x7b, 0x43, 0x11, 0xdd, 0xe0, 0x69, 0xc3, 0x5b, 0x37, 0x92, 0x2c, 0xd7,
    0x79, 0xd6, 0x92, 0xaf, 0x20, 0x43, 0x73, 0x61, 0x1f, 0x13, 0xcc, 0xa1, 0x55, 0xd9, 0x47, 0xd3, 0x89, 0x55, 0x6a,
    0x01, 0x06, 0x9a, 0x2e, 0x8b, 0x30, 0xba, 0x63, 0xaf, 0x56, 0xf7, 0x3c, 0x49, 0x0e, 0xd4, 0x90, 0xae, 0x2f, 0x65,
    0x70, 0xb7, 0xd9, 0xe7, 0x32, 0xcf, 0xef, 0x18, 0x3d, 0xd8, 0x59, 0xde, 0xc1, 0xf4, 0x63, 0xb2, 0x4b, 0x90, 0xb1,
    0x54, 0xff, 0x87, 0xa0, 0x53, 0x68, 0xdf, 0x57, 0xcc, 0x9c, 0x79, 0x3e, 0x8b, 0x74, 0xb8, 0xef, 0xac, 0x33, 0x25,
    0xc7, 0x44, 0x8c, 0x95, 0x29, 0x3d, 0xfb, 0x23, 0xee, 0xcb, 0xc8, 0x51, 0x48, 0x50, 0x4e, 0x93, 0x78, 0x38, 0xc4,
    0xbf, 0x67, 0x48, 0x8c, 0x37, 0x53, 0xb2, 0x5e, 0xa3, 0x83, 0xfc, 0x19, 0x30, 0xe5, 0x3f, 0xa1, 0x51, 0xb1, 0xe0,
    0x24, 0x8a, 0xb7, 0x20, 0x8d, 0xda, 0x79, 0xcb, 0xb2, 0xb7, 0x3a, 0x7e, 0xc9, 0xc4, 0x69, 0x5a, 0x06, 0x3a, 0xc5,
    0xfd, 0x33, 0x01, 0x33, 0x72, 0xb9, 0xc9, 0x09, 0x44, 0x38, 0xa4, 0xeb, 0xf7, 0x27, 0x6a, 0xe5, 0x11, 0x61, 0x99,
    0xe1, 0xa2, 0xc2, 0x75, 0x8d, 0x0d, 0xf6, 0x1e, 0x40, 0x6c, 0xeb, 0xbc, 0xfd, 0x73, 0xf1, 0x74, 0x81, 0x31, 0x73,
    0xf2, 0x81, 0x64, 0xff, 0xfb, 0x27, 0x60, 0xbf, 0xd1, 0x31, 0x02, 0xfe, 0x78, 0x7c, 0x3d, 0x44, 0x17, 0x72, 0x10,
    0xd3, 0xe8, 0xe7, 0x36, 0xe7, 0x77, 0x1f, 0x7a, 0xbd, 0xf9, 0x80};

void put32(std::vector<std::uint8_t>& out, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) out.push_back(std::uint8_t(v >> (8 * i)));
}

struct Payload {
    const std::vector<std::uint8_t>& zlib;
    std::uint32_t unpacked;
};

// A shader container shaped like the game's: DXBC with a DXIL part holding an XDAT table: a 16-byte 0x12 entry,
// then one 0x0D entry per payload, each XBCH + a zlib stream (tools/fssm_unpack.py documents the layout).
std::vector<std::uint8_t> container(std::initializer_list<Payload> payloads) {
    std::vector<std::vector<std::uint8_t>> blobs;
    for (const Payload& p : payloads) {
        std::vector<std::uint8_t> xbch = {'X', 'B', 'C', 'H'};
        put32(xbch, 1);
        put32(xbch, std::uint32_t(p.zlib.size()));
        put32(xbch, 0);
        put32(xbch, p.unpacked);
        put32(xbch, 0);
        xbch.insert(xbch.end(), p.zlib.begin(), p.zlib.end());
        blobs.push_back(std::move(xbch));
    }
    const std::uint32_t entries = 1 + std::uint32_t(blobs.size());
    std::vector<std::uint8_t> dxil(16, 0);  // stands in for the DxilProgramHeader stub
    const std::uint32_t xdat_at = std::uint32_t(dxil.size());
    dxil.insert(dxil.end(), {'X', 'D', 'A', 'T'});
    put32(dxil, entries);
    put32(dxil, 12);  // header size
    std::uint32_t at = xdat_at + 12 + entries * 12;
    put32(dxil, 0x12);  // a 16-byte hash entry, ignored
    put32(dxil, at);
    put32(dxil, 16);
    at += 16;
    for (const auto& blob : blobs) {
        put32(dxil, 0x0d);
        put32(dxil, at);
        put32(dxil, std::uint32_t(blob.size()));
        at += std::uint32_t(blob.size());
    }
    dxil.insert(dxil.end(), 16, 0xab);
    for (const auto& blob : blobs) dxil.insert(dxil.end(), blob.begin(), blob.end());

    std::vector<std::uint8_t> out = {'D', 'X', 'B', 'C'};
    out.resize(28, 0);  // hash, version, total size (unchecked)
    put32(out, 2);       // part count
    put32(out, 40);      // PRIV part offset
    put32(out, 40 + 8 + 4);
    out.insert(out.end(), {'P', 'R', 'I', 'V'});
    put32(out, 4);
    put32(out, 0);
    out.insert(out.end(), {'D', 'X', 'I', 'L'});
    put32(out, std::uint32_t(dxil.size()));
    out.insert(out.end(), dxil.begin(), dxil.end());
    return out;
}

// A synthetic kind-6 program (docs/fspx.md): the 36-byte prologue (three VOP3 words with opcode 0x180 and their
// literals), a filler word and s_endpgm.
std::vector<std::uint8_t> fspx_code(std::uint32_t seed) {
    std::vector<std::uint8_t> out;
    for (int i = 0; i < 3; ++i) {
        put32(out, 0xd5800000u | std::uint32_t(i));
        put32(out, 0x000000ffu);
        put32(out, seed + std::uint32_t(i));
    }
    put32(out, 0x7e000280u);  // v_mov_b32 v0, 0
    put32(out, 0xbf810000u);  // s_endpgm
    return out;
}

struct FspxEntry {
    std::uint32_t kind, offset, size;
};

// An FSPX payload: the header, the directory as given and the bytes laid out after it. `total` 0 means the payload's
// own size.
std::vector<std::uint8_t> fspx_payload(const std::vector<FspxEntry>& directory, const std::vector<std::uint8_t>& body,
                                       std::uint32_t total = 0) {
    std::vector<std::uint8_t> out = {'F', 'S', 'P', 'X'};
    put32(out, 1);
    put32(out, std::uint32_t(directory.size()));
    const std::uint32_t size = std::uint32_t(16 + 16 * directory.size() + body.size());
    put32(out, total ? total : size);
    for (const FspxEntry& e : directory) {
        put32(out, e.kind);
        put32(out, e.offset);
        put32(out, e.size);
        put32(out, 0);
    }
    out.insert(out.end(), body.begin(), body.end());
    return out;
}

// The usual four entries, kinds 0, 1, 6 and 7, after a four-entry directory (body offset 80).
std::vector<std::uint8_t> fspx_payload(const std::vector<std::uint8_t>& code) {
    std::vector<std::uint8_t> body(28, 0x11);                      // kind 0 at 80
    body.insert(body.end(), 20, 0x22);                             // kind 1 at 108
    body.insert(body.end(), code.begin(), code.end());             // kind 6 at 128
    body.insert(body.end(), 8, 0x77);                              // kind 7
    const auto n = std::uint32_t(code.size());
    return fspx_payload({{0, 80, 28}, {1, 108, 20}, {6, 128, n}, {7, 128 + n, 8}}, body);
}

// A DXBC container with an HASH part and an ACCC part holding `payload`, as the FidelityFX DLL's shaders have.
std::vector<std::uint8_t> accc_container(const std::vector<std::uint8_t>& payload) {
    std::vector<std::uint8_t> out = {'D', 'X', 'B', 'C'};
    out.resize(28, 0);
    put32(out, 2);
    put32(out, 40);
    put32(out, 40 + 8 + 20);
    out.insert(out.end(), {'H', 'A', 'S', 'H'});
    put32(out, 20);
    out.insert(out.end(), 20, 0x5a);
    out.insert(out.end(), {'A', 'C', 'C', 'C'});
    put32(out, std::uint32_t(payload.size()));
    out.insert(out.end(), payload.begin(), payload.end());
    return out;
}

void set32(std::vector<std::uint8_t>& out, std::size_t at, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) out[at + std::size_t(i)] = std::uint8_t(v >> (8 * i));
}

// FSPX: the kind-6 program from a DXBC container's ACCC part, a bare payload and a bare program; malformed
// directories are refused. Returns the failing line or 0.
int fspx_checks() {
    using namespace leonida;
    using Kind = ComputeProgram::Kind;
    const auto code = fspx_code(0x1000);
    const auto payload = fspx_payload(code);
    CHECK(payload.size() == 16 + 64 + 28 + 20 + code.size() + 8);
    CHECK(std::vector<std::uint8_t>(fspx_program(payload).begin(), fspx_program(payload).end()) == code);
    const auto wrapped = accc_container(payload);
    CHECK(fssm_program(wrapped).empty());  // what slot 62 gets today: no FSSM program
    for (const auto& input : {wrapped, payload, code}) {
        const ComputeProgram program = compute_program(input);
        CHECK(program.kind == Kind::fspx && program.bytes == code);
    }
    // The game's own shaders keep resolving as FSSM.
    auto fssm = bytes("FSSM and a native program");
    CHECK(compute_program(fssm).kind == Kind::fssm && compute_program(fssm).bytes == fssm);
    CHECK(compute_program(container({{fixed, 25}})).kind == Kind::fssm);

    // Refused: each is Kind::none, so the PSO keeps today's E_INVALIDARG.
    const auto refused = [](const std::vector<std::uint8_t>& input) {
        return fspx_program(input).empty() && compute_program(input).kind == Kind::none &&
               compute_program(accc_container(input)).kind == Kind::none;
    };
    const auto n = std::uint32_t(code.size());
    std::vector<std::uint8_t> body(48, 0x11);
    body.insert(body.end(), code.begin(), code.end());
    CHECK(refused(std::vector<std::uint8_t>(payload.begin(), payload.begin() + 15)));          // shorter than a header
    auto magic = payload;
    magic[3] = 'Y';
    CHECK(fspx_program(magic).empty() && compute_program(accc_container(magic)).kind == Kind::none);
    // Two entries end the directory at 48: kind 0 at 48 and kind 6 at 96 is a valid layout, the base of the rest.
    CHECK(compute_program(fspx_payload({{0, 48, 28}, {6, 96, n}}, body)).bytes == code);
    CHECK(refused(fspx_payload({{0, 48, 28}, {6, 48 + 48 + 1, n}}, body)));                  // kind 6 ends past the total
    CHECK(refused(fspx_payload({{0, 48, 28}, {6, 16, n}}, body)));                           // kind 6 inside the directory
    CHECK(refused(fspx_payload({{0, 48, 28}, {6, 0xffffff00u, 0x200}}, body)));              // offset + size wraps 32 bits
    CHECK(refused(fspx_payload({{0, 48, 28}, {6, 96, 0xffffffffu}}, body)));                 // size past the buffer
    CHECK(refused(fspx_payload({{0, 48, 28}, {7, 96, n}}, body)));                           // no kind 6
    CHECK(refused(fspx_payload({{1, 48, 28}, {6, 96, n}}, body)));                           // no kind 0
    CHECK(refused(fspx_payload({{0, 48, 28}, {6, 96, 0}}, body)));                           // kind 6 empty
    CHECK(refused(fspx_payload({{0, 48, 28}, {6, 96, n}}, body, 96 + n - 4)));               // total cuts kind 6
    auto long_directory = fspx_payload({{0, 48, 28}, {6, 96, n}}, body);
    set32(long_directory, 8, 0x10000000u);                                                   // directory past the total
    CHECK(refused(long_directory));
    set32(long_directory, 8, 0xffffffffu);                                                   // 16 + 16 * count wraps 32 bits
    CHECK(refused(long_directory));
    // The total word larger than the buffer is clamped, as the DLL does; the entries still fit, so it is accepted.
    CHECK(compute_program(fspx_payload({{0, 48, 28}, {6, 96, n}}, body, 0xffffffffu)).bytes == code);
    // The last entry of a kind wins, as in the DLL's reader.
    const auto other = fspx_code(0x2000);
    auto two = body;
    two.insert(two.end(), other.begin(), other.end());
    CHECK(compute_program(fspx_payload({{0, 64, 28}, {6, 112, n}, {6, 112 + n, n}}, two)).bytes == other);
    // A DXBC container without DXIL or ACCC, and bytes without the kind-6 prologue, are not programs.
    CHECK(compute_program(container({{dynamic, 601}})).kind == Kind::none);  // DXIL without a program, no ACCC
    auto no_prologue = code;
    no_prologue[14] ^= 0x01;  // word 3: bits 31..16 no longer 0xD580
    CHECK(compute_program(no_prologue).kind == Kind::none);
    CHECK(compute_program(std::vector<std::uint8_t>(code.begin(), code.begin() + 35)).kind == Kind::none);
    CHECK(compute_program(bytes("not a shader")).kind == Kind::none);
    return 0;
}

}  // namespace

int main() {
    using namespace leonida;

    // Test the shared file reader without passing malformed bytes to a driver.
    const std::vector<std::uint32_t> header = {0x07230203, 0x00010300, 0, 1, 0};
    const std::string header_bytes(reinterpret_cast<const char*>(header.data()), header.size() * sizeof(header.front()));
    {
        std::istringstream input(header_bytes);
        input.seekg(0, std::ios::end);
        CHECK(read_spirv_words(input, input.tellg()) == header);
    }
    for (const auto& bad : {std::string{}, header_bytes.substr(0, 16), header_bytes + 'x',
                            std::string("bad!") + header_bytes.substr(4)}) {
        std::istringstream input(bad);
        input.seekg(0, std::ios::end);
        CHECK(read_spirv_words(input, input.tellg()).empty());
    }
    {
        std::istringstream input(header_bytes);
        input.seekg(0, std::ios::end);
        CHECK(read_spirv_words(input, static_cast<std::streamoff>(header_bytes.size() + 4)).empty());
    }

    CHECK(sha256_hex({}) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    CHECK(sha256_hex(bytes("abc")) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    CHECK(sha256_hex(bytes("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")) ==
          "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");

    std::vector<std::uint8_t> out;
    CHECK(zlib_inflate(stored, out) && out.size() == 40);
    for (std::uint8_t i = 0; i < 40; ++i) CHECK(out[i] == i);
    CHECK(zlib_inflate(fixed, out) && out == bytes("hello hello hello leonida"));
    CHECK(zlib_inflate(dynamic, out) && out == skewed(600));
    CHECK(sha256_hex(out) == "9cd6848f765bff2a9f42a8b42f6164042dad67d3e9f82e0044ddc7d1ecdeebb2");
    auto broken = dynamic;
    broken.back() ^= 1;  // Adler-32
    CHECK(!zlib_inflate(broken, out));
    broken = dynamic;
    broken.resize(100);  // truncated
    CHECK(!zlib_inflate(broken, out));
    CHECK(!zlib_inflate(dynamic, out, 599));  // larger than the limit
    CHECK(!zlib_inflate(bytes("not zlib"), out));

    Json json;
    // "s" holds an escaped quote and a \u escape for U+00E9, which the reader must turn into UTF-8.
    const std::string text = "{\"a\": {\"stage\": \"cs\", \"n\": [1, 2.5, -3e2], \"ok\": true, "
                             "\"s\": \"x\\\"\\u00e9\", \"z\": null}}";
    const std::string expected_s = std::string("x\"") + char(0xc3) + char(0xa9);
    CHECK(JsonReader(text).parse(json));
    const Json* a = json.find("a");
    CHECK(a && a->find("stage")->string == "cs" && a->find("n")->array[2].number == -300 && a->find("ok")->boolean);
    CHECK(a->find("s")->string == expected_s && a->find("z")->kind == Json::Kind::null);
    CHECK(!JsonReader("{\"a\": }").parse(json));
    CHECK(!JsonReader("[1, 2").parse(json));
    CHECK(!JsonReader("{} extra").parse(json));
    // Malformed numbers and escapes are refused, never thrown.
    CHECK(!JsonReader("[-]").parse(json));
    CHECK(!JsonReader("[1e999]").parse(json));
    CHECK(!JsonReader("[1.2.3]").parse(json));
    CHECK(!JsonReader("\"\\uzzzz\"").parse(json));
    CHECK(!JsonReader("\"\\u0x41\"").parse(json));
    CHECK(!JsonReader("\"\\u00").parse(json));
    CHECK(JsonReader("\"\\u0041\"").parse(json) && json.string == "A");

    const auto shader = container({{dynamic, 600}});
    CHECK(dxbc_part(shader, "PRIV").size() == 4 && !dxbc_part(shader, "DXIL").empty() && dxbc_part(shader, "RTS0").empty());
    CHECK(fssm_program(shader) == skewed(600));
    auto fssm = bytes("FSSM and a native program");
    CHECK(fssm_program(fssm) == fssm);  // a bare FSSM blob passes through
    CHECK(fssm_program(container({{dynamic, 601}})).empty());  // stated size wrong
    CHECK(fssm_program(bytes("DXBC")).empty());
    // Truncated DXBC part table: must return empty immediately instead of reading out-of-bounds offset 0
    // (which would match "DXBC" at offset 0 or spin 0xFFFFFFFF iterations on a 32-byte header).
    {
        std::vector<std::uint8_t> trunc(32, 0);
        std::memcpy(trunc.data(), "DXBC", 4);
        trunc[4] = 4;   // if offset 0 is read as a part header, size=4 fits inside 32 bytes
        trunc[28] = 1;  // parts = 1, but 0 bytes of part-offset table follow
        CHECK(dxbc_part(trunc, "DXBC").empty());

        std::vector<std::uint8_t> huge_parts(32, 0);
        std::memcpy(huge_parts.data(), "DXBC", 4);
        huge_parts[28] = 0xff;
        huge_parts[29] = 0xff;
        huge_parts[30] = 0xff;
        huge_parts[31] = 0xff;  // parts = 0xFFFFFFFF with 0 bytes of part table
        CHECK(dxbc_part(huge_parts, "DXIL").empty());
        CHECK(fssm_program(huge_parts).empty());
    }
    // Two different 0x0D payloads: the last one is the program, as tools/fssm_unpack.py (and so fssm_sha256) has it.
    CHECK(fssm_program(container({{fixed, 25}, {dynamic, 600}})) == skewed(600));
    CHECK(fssm_program(container({{dynamic, 600}, {fixed, 25}})) == bytes("hello hello hello leonida"));

    // An index with one findable shader, one without a hash, and the module next to it.
    const testfiles::ScratchDir scratch("leonida_shader_library_test");
    const auto& dir = scratch.path();
    std::filesystem::create_directories(dir);
    const std::string hash = sha256_hex(skewed(600));
    const std::string hash2 = sha256_hex(bytes("another program"));
    {
        std::ofstream index(dir / "index.json", std::ios::binary);
        index << "\xef\xbb\xbf" "{\"cs_test\": {\"stage\": \"cs\", \"fssm_sha256\": \"" << hash
              << "\", \"bindings\": [2], \"push_constant_bytes\": 144, \"local_size\": [64, 1, 1], "
                 "\"uses_code_base\": true, \"wave64\": false},\n"
                 " \"vs_test\": {\"stage\": \"vs\", \"fssm_sha256\": \"" << hash << "\", \"input_signature\": ["
                 "{\"semantic\": \"COLOR\", \"semantic_index\": 0, \"register\": 0, \"mask\": 15, \"rw_mask\": 15, "
                 "\"system_value\": 0, \"component_type\": \"float\"}, "
                 "{\"semantic\": \"SV_VertexID\", \"semantic_index\": 0, \"register\": 2, \"mask\": 1, \"rw_mask\": 1, "
                 "\"system_value\": 6, \"component_type\": \"uint\"}]},\n"
                 " \"vs_same_program\": {\"stage\": \"vs\", \"fssm_sha256\": \"" << hash << "\", \"input_signature\": ["
                 "{\"semantic\": \"TEXCOORD\", \"semantic_index\": 3, \"register\": -1, \"system_value\": 0, "
                 "\"component_type\": \"float\"}]},\n"
                 " \"ps_out\": {\"stage\": \"ps\", \"fssm_sha256\": \"" << hash << "\", \"inputs\": [0, 1, 28, 29], "
                 "\"outputs\": [0, 2], \"f16_halves\": {\"0\": [28, 29], \"bad\": [1], \"7\": [20, 21]}},\n"
                 " \"ps_native\": {\"stage\": \"ps\", \"fssm_sha256\": \"" << hash2 << "\", \"wave64\": true, "
                 "\"subgroup_size\": 64},\n"
                 " \"cs_emulated\": {\"stage\": \"cs\", \"fssm_sha256\": \"" << hash2 << "\", \"wave64\": true, "
                 "\"subgroup_size\": null, \"wave64_exchange\": \"workgroup\", \"lds_bytes\": 33332},\n"
                 " \"ps_unhashed\": {\"stage\": \"ps\", \"bindings\": []}}";
        std::ofstream spv(dir / "cs_test.spv", std::ios::binary);
        const std::uint32_t words[5] = {0x07230203, 0x00010500, 0, 1, 0};
        spv.write(reinterpret_cast<const char*>(words), sizeof(words));
    }
    ShaderLibrary library;
    std::string error;
    CHECK(library.load(dir.string(), &error) && library.size() == 6);
    const ShaderEntry* vs = library.find("vs", hash);
    CHECK(vs && vs->input_signature.size() == 2 && vs->input_signature[0].semantic == "COLOR");
    CHECK(vs->input_signature[1].register_index == 2 && vs->input_signature[1].system_value == 6 &&
          vs->input_signature[1].component_type == "uint");
    // Two containers with the same program and different input signatures: both are kept, in index order, and an
    // out-of-range number (register -1) reads as 0 instead of wrapping.
    const auto vs_all = library.candidates("vs", hash);
    CHECK(vs_all.size() == 2 && vs_all[0] == vs && vs_all[1]->name == "vs_same_program");
    const std::string tagged_hash = hash + "-suffix";
    CHECK(library.candidates(std::string_view("vs-extra", 2),
                             std::string_view(tagged_hash).substr(0, hash.size())) == vs_all);
    CHECK(library.candidates("vs", tagged_hash).empty());
    CHECK(vs_all[1]->input_signature.size() == 1 && vs_all[1]->input_signature[0].semantic_index == 3 &&
          vs_all[1]->input_signature[0].register_index == 0);
    const ShaderEntry* ps = library.find("ps", hash);
    CHECK(ps && ps->outputs.size() == 2 && ps->outputs[1] == 2);
    CHECK(ps->inputs.size() == 4 && ps->inputs[0] == 0 && ps->inputs[3] == 29);
    CHECK(ps->f16_halves.size() == 2 && ps->f16_halves.at(0)[0] == 28 && ps->f16_halves.at(0)[1] == 29 &&
          ps->f16_halves.at(7)[0] == 20 && ps->f16_halves.at(7)[1] == 21);
    const ShaderEntry* entry = library.find("cs", sha256_hex(fssm_program(shader)));
    CHECK(entry && entry->name == "cs_test" && entry->binding_count == 1 && entry->push_constant_bytes == 144);
    CHECK(entry->local_size[0] == 64 && entry->uses_code_base && !entry->wave64);
    CHECK(!library.find("ms", hash));  // no mesh shader with this hash
    // The subgroup fields of the index (wave64 / subgroup_size / wave64_exchange): absent, native and emulated.
    CHECK(!entry->subgroup_size && entry->wave64_exchange.empty() && !entry->emulated());
    const ShaderEntry* native = library.find("ps", hash2);
    CHECK(native && native->wave64 && native->subgroup_size == 64 && !native->emulated());
    const ShaderEntry* emulated = library.find("cs", hash2);
    CHECK(emulated && emulated->wave64 && emulated->subgroup_size == 0 && emulated->wave64_exchange == "workgroup" &&
          emulated->emulated());
    CHECK(emulated->lds_bytes == 33332 && entry->lds_bytes == 0);  // workgroup bytes, emulation included (SHADER-309)
    CHECK(library.spirv(*entry).size() == 5);
    CHECK(!library.load((dir / "missing").string(), &error) && !error.empty());
    {
        std::ofstream index(dir / "index.json", std::ios::binary | std::ios::trunc);
        index << "{\"bad\": {\"stage\": \"cs\", \"fssm_sha256\": \"00\", \"push_constant_bytes\": -}}";
    }
    error.clear();
    CHECK(!library.load(dir.string(), &error) && !error.empty());  // refused, not thrown
    std::filesystem::remove_all(dir);

    if (const int line = fspx_checks()) return line;

    // FSPX entries: found by fspx_code_sha256, in the main index or in <dir>/fspx/index.json (named "fspx/<name>").
    {
        const auto code = fspx_code(0x1000), other = fspx_code(0x2000);
        const std::string code_hash = sha256_hex(code), other_hash = sha256_hex(other);
        std::filesystem::create_directories(dir / "fspx");
        {
            std::ofstream index(dir / "index.json", std::ios::binary);
            index << "{\"cs_game\": {\"stage\": \"cs\", \"fssm_sha256\": \"" << hash << "\"},\n"
                     " \"cs_fspx_main\": {\"stage\": \"cs\", \"fspx_code_sha256\": \"" << other_hash << "\"}}";
            std::ofstream sub(dir / "fspx" / "index.json", std::ios::binary);
            // fspx_payload's kind 0 is 28 bytes of 0x11.
            sub << "{\"upscaler_00\": {\"stage\": \"cs\", \"fspx_code_sha256\": \"" << code_hash
                << "\", \"fspx_state_sha256\": \"" << sha256_hex(std::vector<std::uint8_t>(28, 0x11))
                << "\", \"bindings\": [], \"local_size\": [8, 8, 1], \"wave64\": true}}";
            std::ofstream spv(dir / "fspx" / "upscaler_00.spv", std::ios::binary);
            const std::uint32_t words[5] = {0x07230203, 0x00010500, 0, 1, 0};
            spv.write(reinterpret_cast<const char*>(words), sizeof(words));
        }
        ShaderLibrary fspx;
        CHECK(fspx.load(dir.string(), &error) && fspx.size() == 3 && fspx.fspx_size() == 1);
        const auto found = fspx.fspx_candidates("cs", code_hash);
        CHECK(found.size() == 1 && found[0]->name == "fspx/upscaler_00" && found[0]->local_size[0] == 8 &&
              found[0]->wave64 && found[0]->fssm_sha256.empty());
        CHECK(fspx.spirv(*found[0]).size() == 5);  // <dir>/fspx/upscaler_00.spv
        // The key computed both ways finds the entry: from the container slot 62 gets, from the bare program.
        CHECK(fspx.program_candidates("cs", compute_program(accc_container(fspx_payload(code)))) == found);
        CHECK(fspx.program_candidates("cs", compute_program(code)) == found);
        CHECK(fspx.program_candidates("cs", compute_program(other)).at(0)->name == "cs_fspx_main");
        // The two keys do not mix: an FSPX hash is not an FSSM hash and the other way round.
        CHECK(fspx.candidates("cs", code_hash).empty() && fspx.fspx_candidates("cs", hash).empty());
        CHECK(fspx.find("cs", hash) && fspx.find("cs", hash)->name == "cs_game");
        CHECK(fspx.fspx_candidates("vs", code_hash).empty());
        CHECK(fspx.program_candidates("cs", compute_program(fspx_code(0x3000))).empty());  // not translated
        CHECK(fspx.program_candidates("cs", ComputeProgram{}).empty());
        // A broken fspx/index.json leaves the main index loaded; a missing one is not an error either.
        {
            std::ofstream sub(dir / "fspx" / "index.json", std::ios::binary | std::ios::trunc);
            sub << "{broken";
        }
        CHECK(fspx.load(dir.string(), &error) && fspx.size() == 2 && fspx.fspx_size() == 0 &&
              fspx.fspx_candidates("cs", code_hash).empty());
        std::filesystem::remove(dir / "fspx" / "index.json");
        CHECK(fspx.load(dir.string(), &error) && fspx.size() == 2 && fspx.fspx_size() == 0 && fspx.find("cs", hash));
        std::filesystem::remove_all(dir);
    }

    // XboxBvhBuilder.dll's programs (docs/embedded_shaders.md): FSSM in a DXIL container like the game's shaders,
    // translated into <dir>/bvh (entries named "bvh/<name>") and found by fssm_sha256, the game shaders' key.
    {
        const auto builder = container({{fixed, 25}});  // what the DLL passes to slot 62: the whole container
        const auto untranslated = container({{stored, 40}});
        const std::string builder_hash = sha256_hex(bytes("hello hello hello leonida"));
        const auto code = fspx_code(0x5000);
        const std::uint32_t words[5] = {0x07230203, 0x00010500, 0, 1, 0};
        const auto write = [&](const std::filesystem::path& file, const std::string& text) {
            std::ofstream out(file, std::ios::binary | std::ios::trunc);
            out << text;
        };
        const std::string game = "\"cs_game\": {\"stage\": \"cs\", \"fssm_sha256\": \"" + hash + "\"}";
        const std::string program_entry =
            "{\"stage\": \"cs\", \"fssm_sha256\": \"" + builder_hash +
            "\", \"bindings\": [], \"push_constant_bytes\": 144, \"local_size\": [64, 1, 1], \"lds_bytes\": 32768, "
            "\"uses_code_base\": true, \"wave64\": true, \"subgroup_size\": 64}";
        std::filesystem::create_directories(dir / "bvh");
        std::filesystem::create_directories(dir / "fspx");
        write(dir / "index.json", "{" + game + "}");
        write(dir / "fspx" / "index.json",
              "{\"upscaler_00\": {\"stage\": \"cs\", \"fspx_code_sha256\": \"" + sha256_hex(code) + "\"}}");
        // The second entry is a program whose metadata failed: no key, so nothing can find it and it is not counted.
        write(dir / "bvh" / "index.json",
              "{\"BuilderProgram\": " + program_entry + ",\n \"NoKey\": {\"error\": \"ValueError: synthetic\"}}");
        {
            std::ofstream spv(dir / "bvh" / "BuilderProgram.spv", std::ios::binary);
            spv.write(reinterpret_cast<const char*>(words), sizeof(words));
        }
        ShaderLibrary bvh;
        CHECK(bvh.load(dir.string(), &error) && bvh.size() == 3 && bvh.bvh_size() == 1 && bvh.fspx_size() == 1);
        const ComputeProgram program = compute_program(builder);
        CHECK(program.kind == ComputeProgram::Kind::fssm && sha256_hex(program.bytes) == builder_hash);
        const auto found = bvh.program_candidates("cs", program);
        CHECK(found.size() == 1 && found[0]->name == "bvh/BuilderProgram" && found[0] == bvh.find("cs", builder_hash));
        CHECK(found[0]->lds_bytes == 32768 && found[0]->wave64 && found[0]->subgroup_size == 64 &&
              found[0]->uses_code_base && found[0]->local_size[0] == 64 && found[0]->fspx_code_sha256.empty());
        CHECK(bvh.spirv(*found[0]).size() == 5);  // <dir>/bvh/BuilderProgram.spv
        // The other indexes are untouched: the game's shader and the FSPX program are found as before, each by its
        // own key, and a builder program is neither an FSPX program nor a shader of another stage.
        CHECK(bvh.find("cs", hash) && bvh.find("cs", hash)->name == "cs_game");
        CHECK(bvh.program_candidates("cs", compute_program(code)).at(0)->name == "fspx/upscaler_00");
        CHECK(bvh.fspx_candidates("cs", builder_hash).empty() && bvh.candidates("vs", builder_hash).empty());
        // A program the translator left out (Box16Rehydration today) has no entry: nothing is found, and no other
        // program's module stands in for it.
        CHECK(compute_program(untranslated).kind == ComputeProgram::Kind::fssm);
        CHECK(bvh.program_candidates("cs", compute_program(untranslated)).empty());
        // The same program in the game's index too: both entries, the game's first (index order).
        write(dir / "index.json", "{" + game + ",\n \"cs_same\": " + program_entry + "}");
        CHECK(bvh.load(dir.string(), &error) && bvh.size() == 4 && bvh.bvh_size() == 1);
        const auto both = bvh.candidates("cs", builder_hash);
        CHECK(both.size() == 2 && both[0]->name == "cs_same" && both[1]->name == "bvh/BuilderProgram");
        write(dir / "index.json", "{" + game + "}");
        // A broken bvh/index.json leaves the main index and the FSPX one loaded; a missing one is not an error
        // either, and neither is a broken fspx/index.json for the builder's programs.
        write(dir / "bvh" / "index.json", "{broken");
        CHECK(bvh.load(dir.string(), &error) && bvh.size() == 2 && bvh.bvh_size() == 0 && bvh.fspx_size() == 1 &&
              bvh.program_candidates("cs", program).empty() && bvh.find("cs", hash));
        std::filesystem::remove(dir / "bvh" / "index.json");
        CHECK(bvh.load(dir.string(), &error) && bvh.size() == 2 && bvh.bvh_size() == 0 && bvh.find("cs", hash));
        write(dir / "bvh" / "index.json", "{\"BuilderProgram\": " + program_entry + "}");
        write(dir / "fspx" / "index.json", "{broken");
        CHECK(bvh.load(dir.string(), &error) && bvh.size() == 2 && bvh.bvh_size() == 1 && bvh.fspx_size() == 0 &&
              bvh.program_candidates("cs", program).size() == 1);
        std::filesystem::remove_all(dir);
    }

    // The review case on #1911: one kind-6 program in two containers whose kind-0 entries differ (32x1x1 and
    // 64x1x1), translated to two different modules. A container picks its own entry by fspx_state_sha256; a bare
    // kind-6 program cannot tell them apart and is refused as ambiguous; with one module for the key it is accepted.
    {
        const auto code = fspx_code(0x4000);
        std::vector<std::uint8_t> state32(28, 0), state64(28, 0), state16(28, 0);
        state32[12] = 32;  // word 3: NumThreadsX
        state64[12] = 64;
        state16[12] = 16;
        const auto n = std::uint32_t(code.size());
        const auto wrapped = [&](const std::vector<std::uint8_t>& state) {
            std::vector<std::uint8_t> body = state;
            body.insert(body.end(), code.begin(), code.end());
            return accc_container(fspx_payload({{0, 48, 28}, {6, 48 + 28, n}}, body));
        };
        const auto module = [](std::uint32_t generator) {
            return std::vector<std::uint32_t>{0x07230203, 0x00010500, generator, 1, 0};
        };
        const auto write = [&](const std::string& name, const std::vector<std::uint32_t>& words) {
            std::ofstream spv(dir / "fspx" / (name + ".spv"), std::ios::binary);
            spv.write(reinterpret_cast<const char*>(words.data()), std::streamsize(words.size() * 4));
        };
        const auto index = [&](const std::vector<std::pair<std::string, const std::vector<std::uint8_t>*>>& entries) {
            std::ofstream main(dir / "index.json", std::ios::binary | std::ios::trunc);
            main << "{}";
            std::ofstream sub(dir / "fspx" / "index.json", std::ios::binary | std::ios::trunc);
            sub << "{";
            for (std::size_t i = 0; i < entries.size(); ++i)
                sub << (i ? ",\n" : "") << "\"" << entries[i].first << "\": {\"stage\": \"cs\", \"fspx_code_sha256\": \""
                    << sha256_hex(code) << "\", \"fspx_state_sha256\": \"" << sha256_hex(*entries[i].second) << "\"}";
            sub << "}";
        };
        std::filesystem::create_directories(dir / "fspx");
        write("pair_32", module(1));
        write("pair_64", module(2));
        index({{"pair_32", &state32}, {"pair_64", &state64}});
        ShaderLibrary pair;
        CHECK(pair.load(dir.string(), &error) && pair.size() == 2);
        CHECK(pair.fspx_candidates("cs", sha256_hex(code)).size() == 2);
        bool ambiguous = true;
        const auto by32 = pair.program_candidates("cs", compute_program(wrapped(state32)), &ambiguous);
        CHECK(by32.size() == 1 && by32[0]->name == "fspx/pair_32" && !ambiguous);
        const auto by64 = pair.program_candidates("cs", compute_program(wrapped(state64)), &ambiguous);
        CHECK(by64.size() == 1 && by64[0]->name == "fspx/pair_64" && !ambiguous);
        CHECK(pair.program_candidates("cs", compute_program(wrapped(state16)), &ambiguous).empty() && !ambiguous);
        CHECK(pair.program_candidates("cs", compute_program(code), &ambiguous).empty() && ambiguous);
        // Slot 67 with all four pairs (RT-99): the payload the proxy builds carries kind 0, so each program finds
        // its own translation, where the bare kind-6 program above is ambiguous.
        {
            const std::vector<std::uint8_t> kind1(20, 0x22), kind7(24, 0x77);
            for (const auto* state : {&state32, &state64}) {
                const FspxPart parts[] = {{0, state->data(), state->size()}, {1, kind1.data(), kind1.size()},
                                          {6, code.data(), code.size()}, {7, kind7.data(), kind7.size()}};
                const auto built = fspx_payload_from_parts(parts);
                const ComputeProgram program = compute_program(built);
                CHECK(program.kind == ComputeProgram::Kind::fspx && program.bytes == code && program.state == *state);
                const auto found = pair.program_candidates("cs", program, &ambiguous);
                CHECK(found.size() == 1 && !ambiguous &&
                      found[0]->name == (state == &state32 ? "fspx/pair_32" : "fspx/pair_64"));
            }
        }
        // The same two keys with one module (a duplicate container) is not ambiguous for the bare program.
        write("pair_64", module(1));
        CHECK(pair.program_candidates("cs", compute_program(code), &ambiguous).size() == 2 && !ambiguous);
        // One translation for the code key: the bare program takes it.
        index({{"pair_32", &state32}});
        CHECK(pair.load(dir.string(), &error) && pair.size() == 1);
        const auto bare = pair.program_candidates("cs", compute_program(code), &ambiguous);
        CHECK(bare.size() == 1 && bare[0]->name == "fspx/pair_32" && !ambiguous);
        std::filesystem::remove_all(dir);
    }

    // fspx_payload_from_parts: the slot-67 pairs as one payload, and every inconsistent pair refused before a read.
    {
        const auto code = fspx_code(0x5000);
        const std::vector<std::uint8_t> state(28, 0x11), kind1(20, 0x22), kind7(24, 0x77);
        const auto build = [&](std::initializer_list<FspxPart> parts) {
            return fspx_payload_from_parts(std::span<const FspxPart>(parts.begin(), parts.size()));
        };
        const auto full = build({{0, state.data(), 28}, {1, kind1.data(), 20}, {6, code.data(), code.size()},
                                 {7, kind7.data(), 24}});
        CHECK(!full.empty() && full.size() % 16 == 0);
        const FspxParts parsed = fspx_parts(full);
        CHECK(std::vector<std::uint8_t>(parsed.code.begin(), parsed.code.end()) == code);
        CHECK(std::vector<std::uint8_t>(parsed.state.begin(), parsed.state.end()) == state);
        // Kinds 1 and 7 are optional (null pointer and zero size), as in the DLL's reader.
        CHECK(compute_program(build({{0, state.data(), 28}, {1, nullptr, 0}, {6, code.data(), code.size()},
                                     {7, nullptr, 0}})).state == state);
        // A wild pointer with a size that is refused is never read; neither is anything after a refused pair.
        const void* wild = reinterpret_cast<const void*>(std::uintptr_t(0x10));
        CHECK(build({{0, state.data(), 28}, {6, code.data(), code.size()}, {7, wild, kFspxPartLimit + 1}}).empty());
        CHECK(build({{0, state.data(), 28}, {6, nullptr, code.size()}}).empty());          // size without pointer
        CHECK(build({{0, state.data(), 28}, {6, code.data(), 0}}).empty());                // pointer without size
        CHECK(build({{0, state.data(), 28}, {1, wild, 0}, {6, code.data(), code.size()}}).empty());
        CHECK(build({{0, nullptr, 0}, {6, code.data(), code.size()}}).empty());           // no kind 0
        CHECK(build({{0, state.data(), 28}, {7, kind7.data(), 24}}).empty());              // no kind 6
        CHECK(build({{0, state.data(), 28}, {6, code.data(), code.size()}, {6, code.data(), code.size()}}).empty());
        CHECK(build({{0, state.data(), 28}, {6, code.data(), code.size()}, {5, kind7.data(), 24}}).empty());
        CHECK(build({{0, state.data(), 28}, {6, code.data(), code.size()}, {99, kind7.data(), 24}}).empty());
    }

    std::printf("SHA-256, inflate (stored/fixed/dynamic), JSON, FSSM extraction, FSPX programs and index lookup OK\n");
    return 0;
}
