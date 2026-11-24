"""Semantic interpreter tests for RDNA2 v_pipeflush.

v_pipeflush (GFX10 VOP1) flushes the VALU pipeline and has no architectural effect: no register, memory, VCC, SCC
or EXEC changes. It is translated as nothing, like v_nop. Each case translates a synthetic pixel shader with
tools/rdna2spirv.py, runs it in tests/semantic/spirv_mini_interp.py and checks that the values around the
instruction come out unchanged.
"""
import os
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)

_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_pipeflush.s")
    with open(os.path.join(_TMP_DIR.name, "ps_pipeflush.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def translate(lines) -> str:
    if _TMP_DIR is None:
        setUpModule()
    asm = "; test_pipeflush\n" + "".join(f"\t{ln}\n" for ln in lines) + "\texp mrt0 v0, v1, v2, v3 done vm\n\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        return rdna2spirv.translate(_SRC_PATH)


class PipeflushTests(unittest.TestCase):
    PROLOGUE = ["v_mov_b32 v0, 0x40000001", "v_mov_b32 v1, 0x40000002", "s_mov_b32 s2, 7", "s_cmp_eq_u32 s2, 7",
                "v_cmp_eq_u32 vcc_lo, 0x40000001, v0"]
    EPILOGUE = ["s_cselect_b32 s3, 0x40000003, 0x40000000", "v_mov_b32 v2, s3",
                "v_cndmask_b32 v3, 0x40000000, 0x40000004, vcc_lo"]

    def test_values_survive(self):
        spv = translate(self.PROLOGUE + ["v_pipeflush"] + self.EPILOGUE)
        self.assertEqual(si.run_fragment(spv)["%out_mrt0"], (0x40000001, 0x40000002, 0x40000003, 0x40000004))

    def test_same_program_as_without_it(self):
        with_it = translate(self.PROLOGUE + ["v_pipeflush", "v_pipeflush_e64"] + self.EPILOGUE)
        without = translate(self.PROLOGUE + self.EPILOGUE)
        self.assertEqual(si.run_fragment(with_it), si.run_fragment(without))


if __name__ == "__main__":
    unittest.main()
