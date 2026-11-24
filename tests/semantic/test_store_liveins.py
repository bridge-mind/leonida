"""Store operands in the register analysis used by production translation."""
from pathlib import Path
import unittest

from support import mrt0_words, rdna2spirv, translate_lines
from shader_ir import parse_line


class StoreLiveIns(unittest.TestCase):
    def test_address_and_payload_are_sources(self):
        for op, reads in (("flat_store_dwordx2 v[0:1], v[4:5]", {0, 1, 4, 5}),
                          ("scratch_store_dwordx2 v0, v[4:5], off", {0, 4, 5})):
            with self.subTest(op=op):
                ins = parse_line(op)
                self.assertEqual(rdna2spirv.reg_reads_writes([ins]), (reads, set()))
                self.assertEqual(rdna2spirv.read_before_write_vgprs([ins]), reads)

    def test_branch_exports_live_barycentrics(self):
        fixture = Path(__file__).parent / "fixtures" / "ps_flat_store_livein.asm"
        program = fixture.read_text(encoding="ascii").splitlines()
        # gfx1013 llvm-mc -show-encoding. The taken branch skips END and an eight-byte store.
        scratch = ("scratch_store_dword v0, v4, off ; encoding: "
                   "[0x00,0x40,0x70,0xdc,0x00,0x04,0x7d,0x00]")
        for store in (program[4], scratch):
            with self.subTest(store=store):
                lines = program[:4] + [store] + program[5:]
                spv = translate_lines("ps_store_livein", lines)
                self.assertEqual(mrt0_words(spv, inputs={"%bary": (0.125, 0.25, 0.625)}),
                                 [0x3e800000, 0x3f200000] * 2)
