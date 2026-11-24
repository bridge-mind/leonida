"""Semantic interpreter tests for the GFX10 instructions translated as nothing.

s_icache_inv (instruction-cache invalidate), s_ttracedata / s_ttracedata_imm (thread-trace data), s_code_end (end of
code padding), s_version (ISA version marker), ds_nop (the DS no-operation), v_clrexcp (clears TRAPSTS exception
bits, which nothing translated reads), s_atc_probe / s_atc_probe_buffer (an address-translation-cache prefetch) and
s_incperflevel / s_decperflevel (performance-counter step) / s_wakeup (early wake of s_sleep waves) have no register, memory, SCC, VCC or EXEC effect, so the translator
emits nothing for them, as for s_nop or v_nop. Each case translates a synthetic pixel shader with
tools/rdna2spirv.py, runs it in tests/semantic/spirv_mini_interp.py and checks that values, SCC and VCC set before the
instruction come out unchanged, and that the program behaves as without it. The data-cache invalidations
(buffer_gl0_inv, buffer_gl1_inv, s_dcache_inv, s_gl1_inv) are about coherence between waves and stay Unsupported.
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
NOOP_LINES = ["s_icache_inv", "s_ttracedata", "s_ttracedata_imm 0x1", "s_code_end", "s_version 0x1", "ds_nop",
              "v_clrexcp", "s_atc_probe 0, s[0:1], 0x0", "s_atc_probe_buffer 0, s[0:3], 0x10",
              "s_incperflevel 0x1", "s_decperflevel 0x1", "s_wakeup"]
PROLOGUE = ["v_mov_b32 v0, 0x40000001", "v_mov_b32 v1, 0x40000002", "s_mov_b32 s2, 7", "s_cmp_eq_u32 s2, 7",
            "v_cmp_eq_u32 vcc_lo, 0x40000001, v0"]
EPILOGUE = ["s_cselect_b32 s3, 0x40000003, 0x40000000", "v_mov_b32 v2, s3",
            "v_cndmask_b32 v3, 0x40000000, 0x40000004, vcc_lo"]

_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_sopp_noops.s")
    with open(os.path.join(_TMP_DIR.name, "ps_sopp_noops.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def translate(lines) -> str:
    if _TMP_DIR is None:
        setUpModule()
    asm = "; test_sopp_noops\n" + "".join(f"\t{ln}\n" for ln in lines) + "\texp mrt0 v0, v1, v2, v3 done vm\n\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        return rdna2spirv.translate(_SRC_PATH)


class SoppNoopTests(unittest.TestCase):
    def test_each_leaves_values_scc_and_vcc_alone(self):
        for line in NOOP_LINES:
            with self.subTest(line=line):
                out = si.run_fragment(translate(PROLOGUE + [line] + EPILOGUE))["%out_mrt0"]
                self.assertEqual(out, (0x40000001, 0x40000002, 0x40000003, 0x40000004))

    def test_same_outputs_as_without_them(self):
        without = si.run_fragment(translate(PROLOGUE + EPILOGUE))
        self.assertEqual(si.run_fragment(translate(PROLOGUE + NOOP_LINES + EPILOGUE)), without)

    def test_data_cache_invalidations_stay_unsupported(self):
        for line in ("buffer_gl0_inv", "buffer_gl1_inv", "s_dcache_inv", "s_gl1_inv"):
            with self.subTest(line=line):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate(PROLOGUE + [line] + EPILOGUE)


if __name__ == "__main__":
    unittest.main()
