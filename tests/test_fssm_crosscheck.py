"""FSSM container facts cross-checked against SCARAB, plus the duplicate-XBCH payload cases.

The two code-range facts are measured on the 26,503-container corpus; the blobs below are synthetic and
only reproduce the layout, so the test needs no game data:

  * code starts at `u32[0xC0] + 0x28`, after a 16-byte sub-header and a 24-byte two-record block. The
    words at `+0x10` are not instructions.
  * `code_end` at `0xE8` can be the `0xDEADBEEFDEADBEEF` "unset" sentinel (17/26,503 blobs). The code
    range must then come from the last `s_endpgm`, not from `code_end`.

`fssm_unpack` also has to keep both payloads of a container with two tag-0x0D (`XBCH`) entries:
195/26,503 containers have exactly that shape and the two payloads differ. `xdat_entries_all` keeps
them in directory order, `xdat_entries` keeps the last per tag for old callers, and `unpack()` defaults
to the last payload with `payload="first"` selecting the first.

    python -I tests/test_fssm_crosscheck.py
"""
import os
import struct
import sys
import unittest
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import fssm_unpack  # noqa: E402
import shader_disasm  # noqa: E402

CODE_OFF = 0x100
CODE_START = CODE_OFF + 0x28
CODE_SIZE = 0x40
CODE_END = CODE_OFF + CODE_SIZE
S_INST_PREFETCH = 0xBFA00003
S_ENDPGM = 0xBF810000
S_CODE_END = 0xBF9F0000
DEAD = 0xDEADBEEFDEADBEEF


def fssm(marker: int = 0x11, code_end: int = CODE_END) -> bytes:
    """A minimal FSSM record with the measured 0x28-byte prologue and one s_endpgm."""
    blob = bytearray(CODE_END)
    blob[0:4] = b"FSSM"
    struct.pack_into("<I", blob, 0x04, 0x101)
    struct.pack_into("<Q", blob, 0x08, len(blob) - 16)
    struct.pack_into("<Q", blob, 0x10, 6)
    struct.pack_into("<I", blob, 0xC0, CODE_OFF)
    struct.pack_into("<I", blob, 0xC8, CODE_SIZE)
    struct.pack_into("<Q", blob, 0xE8, code_end)
    # 16-byte sub-header, then the two 12-byte records seen in every blob (first dword 0xD580xxxx).
    struct.pack_into("<6I", blob, CODE_OFF + 0x10,
                     0xD5800000 | marker, 0x03FE0000, marker,
                     0xD5800000 | marker, 0x03FF0000, marker)
    struct.pack_into("<I", blob, CODE_START, S_INST_PREFETCH)
    struct.pack_into("<I", blob, CODE_START + 4, S_ENDPGM)
    for off in range(CODE_START + 8, CODE_END, 4):   # s_code_end padding to code_end
        struct.pack_into("<I", blob, off, S_CODE_END)
    return bytes(blob)


def xbch(payload: bytes) -> bytes:
    compressed = zlib.compress(payload)
    return (b"XBCH" + struct.pack("<5I", 3, len(compressed), 0, len(payload), 0)
            + compressed)


def dxbc_with(entries, hdr: int = 0x24) -> bytes:
    """A DXBC-shaped container with the given XDAT entries.

    The 195 duplicate containers use four entries and keep the header word at 0x24; the fourth
    entry therefore sits beyond the declared header size, as it does in the corpus.
    """
    n = len(entries)
    xdat = 0x18
    table = xdat + 12
    data = table + 12 * n
    payloads = [(tag, p if tag != 0x0D else xbch(p)) for tag, p in entries]
    dxil = bytearray(data + sum(len(p) for _, p in payloads))
    struct.pack_into("<I", dxil, 0x00, 0x10066)       # DxilProgramHeader.ProgramVersion (stage + SM6.6)
    struct.pack_into("<I", dxil, 0x04, len(dxil))     # DxilProgramHeader.Size
    dxil[0x08:0x0C] = b"DXIL"
    struct.pack_into("<I", dxil, 0x0C, 0x106)         # DXIL version, constant on the corpus
    struct.pack_into("<I", dxil, 0x10, len(dxil) - 12)
    struct.pack_into("<I", dxil, 0x14, 1)             # part count
    dxil[xdat:xdat + 4] = b"XDAT"
    struct.pack_into("<I", dxil, xdat + 4, n)
    struct.pack_into("<I", dxil, xdat + 8, hdr)
    off = data
    for i, (tag, payload) in enumerate(payloads):
        struct.pack_into("<III", dxil, table + 12 * i, tag, off, len(payload))
        dxil[off:off + len(payload)] = payload
        off += len(payload)

    chunk = 0x24
    out = bytearray(chunk + 8 + len(dxil))
    out[0:4] = b"DXBC"
    struct.pack_into("<I", out, 28, 1)                # chunk count
    struct.pack_into("<I", out, 32, chunk)            # first chunk offset
    out[chunk:chunk + 4] = b"DXIL"
    struct.pack_into("<I", out, chunk + 4, len(dxil))
    out[chunk + 8:chunk + 8 + len(dxil)] = dxil
    return bytes(out)


class CodeRangeTest(unittest.TestCase):
    def test_code_starts_after_the_0x28_prologue(self):
        blob = fssm()
        self.assertEqual(struct.unpack_from("<I", blob, CODE_OFF + 0x10)[0] & 0xFFFF0000, 0xD5800000)
        self.assertEqual(struct.unpack_from("<I", blob, CODE_START)[0], S_INST_PREFETCH)
        start, end, how = shader_disasm.code_range(blob)
        self.assertEqual((start, end, how), (CODE_START, CODE_START + 8, "header"))

    def test_code_end_dead_sentinel_does_not_hide_the_code(self):
        blob = fssm(code_end=DEAD)
        start, end, how = shader_disasm.code_range(blob)
        self.assertEqual((start, end, how), (CODE_START, CODE_START + 8, "header"))


class DuplicateXbchTest(unittest.TestCase):
    @staticmethod
    def container(first: bytes, second: bytes) -> bytes:
        return dxbc_with([(0x12, b"hash"), (0x0D, first), (0x0D, second), (0x07, b"stub")])

    def dxil(self, first: bytes, second: bytes) -> bytes:
        return fssm_unpack.parts(self.container(first, second))[b"DXIL"]

    def test_xdat_entries_all_preserves_both_payloads_in_order(self):
        first, second = fssm(0x11), fssm(0x22)
        self.assertNotEqual(first, second)
        entries = fssm_unpack.xdat_entries_all(self.dxil(first, second))
        self.assertEqual([tag for tag, _ in entries], [0x12, 0x0D, 0x0D, 0x07])
        # Entries hold the compressed XBCH payload: both must decompress to their own FSSM blob.
        blobs = [zlib.decompress(payload[24:]) for tag, payload in entries if tag == 0x0D]
        self.assertEqual(blobs, [first, second])

    def test_xdat_entries_keeps_the_last_payload_per_tag(self):
        first, second = fssm(0x11), fssm(0x22)
        entries = fssm_unpack.xdat_entries(self.dxil(first, second))
        self.assertEqual(sorted(entries), [0x07, 0x0D, 0x12])
        # The single-value view is kept for old callers: last entry wins, as before the fix.
        self.assertEqual(zlib.decompress(entries[0x0D][24:]), second)

    def test_unpack_selects_first_or_last_payload(self):
        first, second = fssm(0x11), fssm(0x22)
        container = self.container(first, second)
        self.assertEqual(fssm_unpack.unpack(container)[0], second)                 # default: last
        self.assertEqual(fssm_unpack.unpack(container, payload="last")[0], second)
        self.assertEqual(fssm_unpack.unpack(container, payload="first")[0], first)

    def test_unpack_rejects_an_unknown_payload_choice(self):
        with self.assertRaises(ValueError):
            fssm_unpack.unpack(self.container(fssm(0x11), fssm(0x22)), payload="middle")


if __name__ == "__main__":
    unittest.main()
