"""interp_ij_analysis: a write of the I/J register at instruction index 0 (or after the site in a loop) reaches it.

`reaching_write` tested predecessor blocks with `any(w for w in ws ...)`, so a write at instruction index 0
(falsy) never counted, and it seeded the search with the site's own block, so a write later in that block that
a loop back-edge brings round was never seen. Either way a computed barycentric pair was treated as pristine.
The programs are hand-built instruction lists; no game data, no GPU.
"""
import os
import sys
import unittest
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
sys.dont_write_bytecode = True
sys.path.insert(0, TOOLS_DIR)
import rdna2spirv  # noqa: E402


def make(op, args, pc, target=None):
    ins = {"op": op, "args": args, "pc": pc, "enc": "00000000"}
    if target is not None:
        ins["target"] = target
    return ins


def vg(r):
    return {"k": "vgpr", "r": r}


def attr():
    return {"k": "attr", "a": 0, "c": 0}


def interp_pair(pc):
    return [make("v_interp_p1_f32", [vg(2), vg(0), attr()], pc),
            make("v_interp_p2_f32", [vg(2), vg(1), attr()], pc + 4)]


def analyse(insns):
    g = SimpleNamespace(v_interp_copy={}, pixel_interpolants=None, frag_inputs={})
    return rdna2spirv.interp_ij_analysis(insns, g), insns


class ReachingWrite(unittest.TestCase):
    def test_write_at_index_zero_reaches_branch_target(self):
        insns = [make("v_mov_b32", [vg(0), {"k": "imm", "v": 0}], 0),
                 make("s_branch", [{"k": "imm", "v": 0}], 4, target=8)] + interp_pair(8) + \
                [make("s_endpgm", [], 16)]
        attrs, insns = analyse(insns)
        self.assertEqual(attrs, {0})
        self.assertIn("ij", insns[2])

    def test_write_at_index_one_still_reaches(self):
        insns = [make("s_nop", [], 0),
                 make("v_mov_b32", [vg(0), {"k": "imm", "v": 0}], 4),
                 make("s_branch", [{"k": "imm", "v": 0}], 8, target=12)] + interp_pair(12) + \
                [make("s_endpgm", [], 20)]
        attrs, _ = analyse(insns)
        self.assertEqual(attrs, {0})

    def test_no_write_stays_pristine(self):
        insns = [make("s_nop", [], 0),
                 make("s_branch", [{"k": "imm", "v": 0}], 4, target=8)] + interp_pair(8) + \
                [make("s_endpgm", [], 16)]
        attrs, _ = analyse(insns)
        self.assertEqual(attrs, set())

    def test_loop_back_edge_write_after_site(self):
        # block 0: interp pair, then v0 is rewritten and the loop branches back to the pair.
        insns = interp_pair(0) + [make("v_mov_b32", [vg(0), {"k": "imm", "v": 0}], 8),
                                  make("s_cbranch_scc1", [{"k": "imm", "v": 0}], 12, target=0),
                                  make("s_endpgm", [], 16)]
        attrs, _ = analyse(insns)
        self.assertEqual(attrs, {0})


if __name__ == "__main__":
    unittest.main()
