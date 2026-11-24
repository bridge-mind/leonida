// ImHex pattern for one RBF0 resource: RAGE's binary form of an XML data file.
//
// The grammar is the one measured in docs/rbf0.md (section 3) and parsed by tools/rbf0_index.py (parse_rbf0):
//
//   0x00  "RBF0"
//   0x04  FB FF 01 00   optional prologue. With it, integers and floats are 8 bytes ("wide"); without it they are
//                       4 bytes ("narrow") and the records start at 0x04. What the four bytes mean is not established;
//                       the width follows them in every measured resource (docs/rbf0.md).
//   records, each a u8 descriptor index and a u8 type:
//     FF FF                        close the open element
//     FD FF                        text for the open element: u32 length, then that many bytes
//     index == descriptors so far  a new descriptor: u16 name length, then the UTF-8 name, before the payload
//     type 0x00  open an element: u16 0, u16 0, u16 attribute count
//     type 0x10  integer      0x20 true      0x30 false      0x40 float      0x50 three floats
//     type 0x60  string: u16 length, then UTF-8 bytes
//   The stream has no length field and no section table: it ends where the root element closes, and the pattern
//   stops there.
//
// Open an inflated, joined resource, not the archive. A resource larger than 64 KiB is split over several zlib
// streams, with no header on the continuations (docs/rbf0.md section 2); tools/rbf0_index.py joins them.

#pragma description RBF0 record stream (binary XML, docs/rbf0.md)
#pragma endian little
#pragma magic [ 52 42 46 30 ] @ 0x00
// The largest measured resource has 68,398 elements and about twice as many records.
#pragma array_limit 4194304
#pragma pattern_limit 33554432

import std.mem;
import std.io;

bool g_wide = std::mem::size() >= 8 && std::mem::read_unsigned(0x04, 4) == 0x0001FFFB;
u32 g_descriptors = 0;
u32 g_depth = 0;
bool g_done = false;

struct Header {
    char magic[4];
    if (g_wide) {
        u8 prologue[4] [[comment("FB FF 01 00: 8-byte integers and floats follow")]];
    }
};

struct Record {
    u8 index;
    u8 type;
    if (index == 0xFF) {
        if (g_depth > 0) g_depth -= 1;
        if (g_depth == 0) g_done = true;
    } else if (index == 0xFD) {
        u32 text_length;
        char text[text_length];
    } else {
        if (index == g_descriptors) {
            u16 name_length;
            char name[name_length] [[comment("new descriptor")]];
            g_descriptors += 1;
        }
        if (type == 0x00) {
            u16 word0;
            u16 word1;
            u16 attribute_count;
            g_depth += 1;
        } else if (type == 0x10) {
            if (g_wide) {
                u64 value_int;
            } else {
                u32 value_int;
            }
        } else if (type == 0x40) {
            if (g_wide) {
                double value_float;
            } else {
                float value_float;
            }
        } else if (type == 0x50) {
            if (g_wide) {
                double value_vec3[3];
            } else {
                float value_vec3[3];
            }
        } else if (type == 0x60) {
            u16 string_length;
            char value_string[string_length];
        }
    }
} [[format("format_record")]];

fn format_record(Record r) {
    if (r.index == 0xFF) return "close";
    if (r.index == 0xFD) return "text";
    return std::format("descriptor {} type 0x{:02X}", r.index, r.type);
};

Header header @ 0x00;
Record records[while(!g_done && !std::mem::eof())] @ sizeof(header);
