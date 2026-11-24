"""A hand-written listing without `; encoding:` bytes: instructions that need their own size are rejected cleanly.

tools/shader_ir.parse_file advances the PC by each instruction's encoding bytes. Without them (a listing written by
hand, like the fixtures in tests/semantic/fixtures/disasm) the PC of the next instruction is unknown, which
s_getpc_b64 and every PC-relative branch need. Until 2026-11-22 both raised TypeError in the translator; now they
raise rdna2spirv.Unsupported, which callers such as the batch translator count as an unsupported shader. Straight-line
code without encodings still translates. Standard library only; no GPU, no game files.
"""
import os
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.dont_write_bytecode = True
sys.path.insert(0, os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools"))

import rdna2spirv  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


def translate(lines):
    with tempfile.TemporaryDirectory() as tmp:
        os.mkdir(os.path.join(tmp, "disasm"))
        with open(os.path.join(tmp, "ps_noenc.dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        src = os.path.join(tmp, "disasm", "ps_noenc.s")
        with open(src, "w", encoding="utf-8") as f:
            f.write("; ps_noenc\n" + "".join(f"\t{ln}\n" for ln in lines)
                    + "\texp mrt0 v0, v0, v0, v0 done vm\n\ts_endpgm\n")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            return rdna2spirv.translate(src)


class MissingEncoding(unittest.TestCase):
    def test_getpc_is_unsupported_not_a_crash(self):
        with self.assertRaisesRegex(rdna2spirv.Unsupported, "s_getpc_b64 without encoding"):
            translate(["s_getpc_b64 s[0:1]", "v_mov_b32 v0, s0"])

    def test_branches_are_unsupported_not_a_crash(self):
        for br in ("s_branch 1", "s_cbranch_execz 1", "s_cbranch_scc0 2", "s_cbranch_vccnz 1"):
            with self.subTest(branch=br):
                with self.assertRaisesRegex(rdna2spirv.Unsupported, "branch without encoding"):
                    translate(["v_mov_b32 v0, 1", br, "v_mov_b32 v0, 2"])

    def test_straight_line_code_still_translates(self):
        self.assertIn("OpEntryPoint", translate(["v_mov_b32 v0, 1.0", "s_nop 0"]))


if __name__ == "__main__":
    unittest.main()
