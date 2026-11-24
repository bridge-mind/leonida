// ImHex pattern for the header of an RPF9 archive.
//
// Only measured fields are named (docs/rpf.md "What is readable"; docs/story.md "Signatures actually observed"):
//   0x00  "9FPR"        RPF9 stored little-endian
//   0x04  data offset   u32, where the data area (zlib streams laid end to end) starts
//   0x08  u32           present in every header; meaning not established
//   0x0C  trailer size  u32, the size of the block that ends the file (docs/rpf.md, The trailer at EOF)
// Everything from 0x10 to the data offset is the rest of the header and the table of contents. The TOC is
// encrypted/packed: this pattern shows that range as opaque bytes and decodes nothing in it (no keys, no decryption).
//
// This describes the archive's own header at offset 0. Nested RPF9 headers occur inside the data area
// (docs/story.md); their offsets are not covered here.

#pragma description RPF9 archive header, opaque TOC and data area start (docs/rpf.md)
#pragma endian little
#pragma magic [ 39 46 50 52 ] @ 0x00

import std.mem;
import std.sys;

struct Rpf9Header {
    char magic[4] [[comment("\"9FPR\": RPF9 stored little-endian")]];
    u32 data_offset [[comment("start of the data area (docs/rpf.md)")]];
    u32 u32_08 [[comment("meaning not established")]];
    u32 trailer_size [[comment("size of the block at EOF (docs/rpf.md)")]];
};

Rpf9Header header @ 0x00;
std::assert(header.data_offset >= sizeof(header) && header.data_offset <= std::mem::size(),
            "the data offset is outside the file");
u8 header_rest_and_toc[header.data_offset - sizeof(header)] @ sizeof(header)
    [[comment("rest of the header and the table of contents: encrypted/packed, not decoded")]];
