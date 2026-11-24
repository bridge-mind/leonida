// ImHex pattern for a PSIN metadata container (docs/psin.md; tools/psin_index.py and tools/psin_data.py read it).
//
// Layout, as measured in docs/psin.md:
//   0x00  "PSIN"
//   0x04  big-endian absolute offset of the first section
//   0x08  instance-data block table: rows of {u32 block type, u32 data size, aa x 8}, ended by "pppppppp" 8 bytes
//         short of a 16-byte boundary. The blocks follow in table order, each padded to 16 bytes, numbered from 1;
//         the last one ends at the first section. Their contents are shown as raw bytes: decoding them needs the
//         schema walk that psin_data.py does.
//   sections from the first-section offset to EOF: ASCII tag, big-endian total size (header included)
//     LSCH  struct headers (32 bytes) then all members (16 bytes); the header count is not stored, it is the one
//           count whose headers and members fill the payload exactly
//     LENM  enum headers (24 bytes) then all values (16 bytes); same rule
//     LSNS  u32 block count, {u32 kind, u32 byte count} per block, then the blocks: opaque 16-byte records
//     LRSC  "0DRP", u32 version, ...; the u32 at +0x10 is the root block number
//     CHKS  big-endian file size and a 32-bit value (not verified here)
//     LGID  empty
// Little-endian everywhere except the first-section offset and the section sizes. A truncated container shows what
// is whole and the rest as raw bytes.

#pragma description PSIN metadata container: block table, LSCH/LENM schema, LSNS, LRSC, CHKS (docs/psin.md)
#pragma endian little
#pragma magic [ 50 53 49 4E ] @ 0x00
#pragma array_limit 1048576
#pragma pattern_limit 8388608

import std.mem;
import std.core;

// ---- header and instance-data block table ---------------------------------------------------------------------

struct Header {
    char magic[4];
    be u32 first_section [[comment("big-endian absolute offset of the first section")]];
};

Header header @ 0x00;

fn count_rows(u128 limit) {
    u128 p = 0x08;
    u32 n = 0;
    while (p + 16 <= limit && std::mem::read_unsigned(p, 8) != 0x7070707070707070) {
        p += 16;
        n += 1;
    }
    return n;
};

u128 g_limit = header.first_section < std::mem::size() ? header.first_section : std::mem::size();
u32 g_rows = count_rows(g_limit);

struct BlockRow {
    u32 type [[comment("struct name hash from LSCH, a member type code below 0x100, or 0x100 + n")]];
    u32 size [[comment("data size in bytes")]];
    u8 fill[8] [[comment("aa x 8")]];
};

struct Block {
    u32 i = std::core::array_index();
    u8 data[rows[i].size] [[comment("block number i + 1; raw: see psin_data.py")]];
    u8 pad[(16 - rows[i].size % 16) % 16];
};

BlockRow rows[g_rows] @ 0x08;
char table_end[8] @ 0x08 + 16 * g_rows [[comment("\"pppppppp\"")]];
Block blocks[g_rows] @ 0x08 + 16 * g_rows + 8;

// ---- sections ---------------------------------------------------------------------------------------------------

fn header_count(u128 payload, u128 length, u32 header_size, u32 element_size, u32 count_at, u32 count_size) {
    if (length == 0) return 0;
    u32 elements = 0;
    u32 k = 1;
    while (header_size * k <= length) {
        elements += std::mem::read_unsigned(payload + header_size * (k - 1) + count_at, count_size);
        if (header_size * k + element_size * elements == length) return k;
        k += 1;
    }
    return 0xFFFFFFFF;  // no count fits: shown as raw bytes
};

fn sum_counts(u128 payload, u32 k, u32 header_size, u32 count_at, u32 count_size) {
    u32 total = 0;
    for (u32 i = 0, i < k, i += 1) {
        total += std::mem::read_unsigned(payload + header_size * i + count_at, count_size);
    }
    return total;
};

struct StructHeader {
    u32 name_hash;
    u32 signature [[comment("layout signature")]];
    u8 flags;
    u8 align_log2;
    u16 zero_0a;
    u32 zero_0c;
    u8 fill[8] [[comment("aa x 8")]];
    u32 size [[comment("byte size of the struct")]];
    u16 zero_1c;
    u16 member_count;
};

struct Member {
    u32 name_hash [[comment("0x100 = array element descriptor")]];
    u32 offset;
    u8 type;
    u8 sub;
    u16 index [[comment("for type 0x5c: position of its 0x100 element descriptor in this struct's members")]];
    u32 ref_hash [[comment("struct or enum name hash for the referring types, else 0")]];
};

struct EnumHeader {
    u32 name_hash;
    u32 signature;
    u8 fill[8] [[comment("ee x 8")]];
    u32 value_count;
    u32 zero_14;
};

struct EnumValue {
    s64 value;
    u64 name_hash [[comment("case-sensitive joaat in the low half")]];
};

struct LsnsDesc {
    u32 kind;
    u32 bytes;
};

struct LsnsRecord {
    u8 bytes[16] [[comment("opaque")]];
};

struct LsnsBlock {
    u32 i = std::core::array_index();
    LsnsRecord records[parent.descs[i].bytes / 16];
};

bool g_bad = false;

struct Section {
    char tag[4];
    be u32 size [[comment("big-endian total size, header included")]];
    u128 payload = $;
    if (size < 8 || addressof(tag) + size > std::mem::size()) {
        g_bad = true;
        u8 truncated[std::mem::size() - $] [[comment("section runs past the end of the file")]];
    } else {
        u32 length = size - 8;
        if (tag == "LSCH") {
            u32 k = header_count(payload, length, 32, 16, 30, 2);
            if (k == 0xFFFFFFFF) {
                u8 raw[length] [[comment("no struct count fills the payload")]];
            } else {
                StructHeader structs[k];
                Member members[sum_counts(payload, k, 32, 30, 2)];
            }
        } else if (tag == "LENM") {
            u32 k = header_count(payload, length, 24, 16, 16, 4);
            if (k == 0xFFFFFFFF) {
                u8 raw[length] [[comment("no enum count fills the payload")]];
            } else {
                EnumHeader enums[k];
                EnumValue values[sum_counts(payload, k, 24, 16, 4)];
            }
        } else if (tag == "LSNS" && length >= 4) {
            u32 block_count;
            if (4 + 8 * block_count <= length) {
                LsnsDesc descs[block_count];
                LsnsBlock lsns_blocks[block_count];
            }
            if ($ < payload + length) {
                u8 rest[payload + length - $] [[comment("does not fit the block table")]];
            }
        } else if (tag == "LRSC" && length >= 0x14 && std::mem::read_string(payload, 4) == "0DRP") {
            char rsc_magic[4];
            u32 rsc_version;
            u32 u32_08;
            u32 u32_0c;
            u32 root_block [[comment("block number of the root struct")]];
            u8 rest[length - 0x14];
        } else if (tag == "CHKS" && length >= 8) {
            be u32 checksum_size [[comment("big-endian file size")]];
            u8 checksum[4] [[comment("32-bit value, not verified")]];
            u8 rest[length - 8];
        } else {
            u8 raw[length];
        }
    }
};

Section sections[while(!g_bad && header.first_section + 8 <= std::mem::size() && std::mem::size() - $ >= 8)]
    @ header.first_section;
