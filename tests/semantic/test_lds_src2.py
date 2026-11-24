"""The DS `*_src2_*` forms (`ds_add_src2_u32`, `ds_max_src2_f64`, ...): GFX10.1 encodings the translator refuses.

These ops read their second operand from LDS instead of a VGPR. They exist in the GFX10.1 encoding the console's
gfx1013 shares, and not on gfx103x: llvm-mc 21.1.8 assembles all 31 forms below for `-mcpu=gfx1013` (DS opcodes
0x80-0x95 and 0xc0-0xd3) and rejects every one of them for `-mcpu=gfx1030` ("instruction not supported on this
GPU"), where it also decodes their bytes as an invalid encoding. They take one 16-bit `offset:`, not the
`offset0:`/`offset1:` pair of the two-address DS ops, which llvm-mc rejects on them ("not a valid operand"). The
address of the LDS operand is derived from that offset; neither the translator nor this file models it, and it has
not been measured on hardware.

`tools/rdna2spirv.py` has no rule for them and raises `Unsupported` naming the op, so a listing that holds one is
refused rather than translated as something else. `RefusedTests` pins that, with the plain `ds_add_u32` as the
control that the program reaches the translator. `EncodingTests` checks the gfx1013/gfx1030 split and the operand
form with llvm-mc, and is skipped when no llvm-mc with the AMDGPU target is found (LLVM_MC, as for the tools).

This replaces `test_lds_add_sub_src2_u32.py`, a stand-alone model of `ds_add_src2_u32`/`ds_sub_src2_u32` that never
ran the translator and wrote the ops with `offset0`/`offset1` (TEST-738, found by @Vonsek).

Standard library only; no GPU and no game files (llvm-mc optional).
"""
import os
import struct
import subprocess
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools"))

import support  # noqa: E402
from mc_command import mc_command  # noqa: E402
from rdna2spirv import Unsupported  # noqa: E402

SRC2_32 = [
    "ds_add_src2_u32", "ds_sub_src2_u32", "ds_rsub_src2_u32", "ds_inc_src2_u32", "ds_dec_src2_u32",
    "ds_min_src2_i32", "ds_max_src2_i32", "ds_min_src2_u32", "ds_max_src2_u32",
    "ds_and_src2_b32", "ds_or_src2_b32", "ds_xor_src2_b32", "ds_write_src2_b32",
    "ds_min_src2_f32", "ds_max_src2_f32", "ds_add_src2_f32",
]
SRC2_64 = [
    "ds_add_src2_u64", "ds_sub_src2_u64", "ds_rsub_src2_u64", "ds_inc_src2_u64", "ds_dec_src2_u64",
    "ds_min_src2_i64", "ds_max_src2_i64", "ds_min_src2_u64", "ds_max_src2_u64",
    "ds_and_src2_b64", "ds_or_src2_b64", "ds_xor_src2_b64", "ds_write_src2_b64",
    "ds_min_src2_f64", "ds_max_src2_f64",
]
SRC2 = SRC2_32 + SRC2_64

# Compute FSSM header: RSRC2 0x8088 (4 user SGPRs, workgroup id x, 128 bytes of LDS), 64x1x1 threads.
CS_FSSM = bytearray(0x148)
struct.pack_into("<4I", CS_FSSM, 0x138, 0x8088, 64, 1, 1)
CS_FSSM = bytes(CS_FSSM)


def translate(op_line):
    return support.translate_lines("cs_ds_src2", ["v_mov_b32_e32 v1, 0", op_line, "s_endpgm"], fssm=CS_FSSM)


class RefusedTests(unittest.TestCase):
    def test_plain_ds_add_reaches_the_translator(self):
        self.assertIn("OpEntryPoint", translate("ds_add_u32 v1, v1"))

    def test_every_src2_form_is_refused_by_name(self):
        for op in SRC2:
            for operands in ("v1", "v1 offset:4", "v1 offset:65535"):
                with self.subTest(op=op, operands=operands):
                    with self.assertRaises(Unsupported) as caught:
                        translate(f"{op} {operands}")
                    self.assertIn(op, str(caught.exception))


def _llvm_mc():
    """llvm-mc's argv if it runs and has an AMDGPU target that knows gfx1013, else None."""
    cmd = mc_command()
    try:
        r = subprocess.run(cmd + ["-triple=amdgcn", "-mcpu=gfx1013", "-show-encoding"], input="s_endpgm\n",
                           capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return cmd if r.returncode == 0 and "encoding" in r.stdout else None


MC = _llvm_mc()


def assemble(cpu, text):
    """(returncode, stdout, stderr) of llvm-mc assembling `text` for `cpu`, through a file (no BOM, no pipe)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "in.s")
        with open(path, "w", encoding="ascii", newline="\n") as f:
            f.write(text)
        r = subprocess.run(MC + ["-triple=amdgcn", f"-mcpu={cpu}", "-show-encoding", path],
                           capture_output=True, text=True, timeout=60)
    return r.returncode, r.stdout, r.stderr


def encodings(stdout):
    """The encoding bytes llvm-mc printed, one list per instruction."""
    out = []
    for line in stdout.splitlines():
        if "encoding: [" in line:
            body = line.split("encoding: [", 1)[1].split("]", 1)[0]
            out.append([int(b, 16) for b in body.split(",")])
    return out


@unittest.skipUnless(MC, "needs llvm-mc with the AMDGPU target (set LLVM_MC)")
class EncodingTests(unittest.TestCase):
    def test_gfx1013_has_every_form_with_a_16_bit_offset(self):
        for op in SRC2:
            with self.subTest(op=op):
                rc, out, err = assemble("gfx1013", f"{op} v1 offset:4660\n")
                self.assertEqual(rc, 0, err)
                (enc,) = encodings(out)
                word0 = int.from_bytes(bytes(enc[:4]), "little")
                self.assertEqual(word0 >> 26, 0b110110, "not a DS encoding")
                opcode = (word0 >> 18) & 0xFF
                wide = op in SRC2_64
                self.assertTrue(0xC0 <= opcode <= 0xD3 if wide else 0x80 <= opcode <= 0x95, hex(opcode))
                self.assertEqual(word0 & 0xFFFF, 4660, "offset is not the 16-bit field")

    def test_gfx1030_has_none_of_them(self):
        for op in SRC2:
            with self.subTest(op=op):
                rc, out, err = assemble("gfx1030", f"{op} v1\n")
                self.assertNotEqual(rc, 0)
                self.assertIn("not supported on this GPU", err)

    def test_offset0_offset1_is_not_their_operand_form(self):
        rc, out, err = assemble("gfx1013", "ds_add_src2_u32 v1 offset0:1 offset1:2\n")
        self.assertNotEqual(rc, 0)
        self.assertIn("not a valid operand", err)


if __name__ == "__main__":
    unittest.main()
