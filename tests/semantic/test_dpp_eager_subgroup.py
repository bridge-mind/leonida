"""DPP's ballot and shuffle run in every lane, also after SPIRV-Cross inlines expressions (SHADER-529).

SPIRV-Cross (MoltenVK) inlines a result that is used once into the expression that uses it. On the right of an
OpLogicalAnd/OpLogicalOr that becomes a short-circuiting && / ||, and in an arm of a scalar OpSelect a ternary, so
a subgroup op there runs only in the lanes that get that far. The DPP lowering used to give
`in_row && subgroupBallotBitExtract(subgroupBallot(active), src)` and `src_ok ? subgroupShuffle(v, src) : 0`: on a
Mac, row_shr:1's lane 1 found lane 0 (out of its row, so it skipped the ballot) inactive and kept its old value
(`cs_dpp_ops`). The translator now stores both results to Function variables (`Gen.eager`), and a store is a
statement every lane runs.

`conditional_subgroup_ops` follows each subgroup op's result through single-use expressions, the ones SPIRV-Cross
inlines, and reports one that lands on the right of a logical and/or or in a scalar select's arm. It is a model of
SPIRV-Cross's forwarding, not SPIRV-Cross: checked against SPIRV-Cross 1.3.239's MSL for the `cs_dpp*` GPU tests,
where it finds exactly the inlined ballots and shuffles before this change and none after. The values are checked
too, by running the translated DPP in tests/semantic/spirv_wave_interp.py.

Standard library only; no GPU, no LLVM, no SPIRV-Tools, no game files.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import support  # noqa: E402  (puts tools/ on the path)
import spirv_wave_interp as wi  # noqa: E402

# Group ops whose result depends only on their operands' values, not on which invocations run them.
_PURE = frozenset({"OpGroupNonUniformBallotBitExtract", "OpGroupNonUniformBallotBitCount",
                   "OpGroupNonUniformBallotFindLSB", "OpGroupNonUniformBallotFindMSB",
                   "OpGroupNonUniformInverseBallot"})


def _parse(text):
    insns = []
    for ln in text.splitlines():
        t = ln.split(";", 1)[0].split()
        if not t:
            continue
        if len(t) > 2 and t[1] == "=":
            insns.append((t[0], t[2], t[3:]))
        else:
            insns.append((None, t[0], t[1:]))
    return insns


def conditional_subgroup_ops(spvasm):
    """[(result id, opcode, where)] for each subgroup op SPIRV-Cross would inline into a conditional position."""
    insns = _parse(spvasm)
    types, uses = {}, {}
    for i, (res, op, args) in enumerate(insns):
        if res:
            types[res] = args[0] if args else None
        for pos, a in enumerate(args[1:] if res else args):     # a result's first operand is its type
            if a.startswith("%"):
                uses.setdefault(a, []).append((i, pos))
    found = []
    for res, op, _ in insns:
        if not (res and op.startswith("OpGroupNonUniform") and op not in _PURE):
            continue
        cur = res
        while len(uses.get(cur, ())) == 1:       # used more than once: SPIRV-Cross makes it a temporary
            i, pos = uses[cur][0]
            ures, uop, uargs = insns[i]
            if uop in ("OpLogicalAnd", "OpLogicalOr") and pos == 1:
                found.append((res, op, f"right of {uop} {ures}"))
                break
            if uop == "OpSelect" and pos in (1, 2) and types.get(uargs[1]) == "%bool":
                found.append((res, op, f"arm of OpSelect {ures}"))
                break
            if ures is None or uop == "OpPhi":   # a statement (store, branch) or a phi copy: every lane runs it
                break
            cur = ures
    return found


DPP_FORMS = [
    "row_shr:1 row_mask:0xf bank_mask:0xf",
    "row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1",
    "row_shl:3 row_mask:0xf bank_mask:0xf",
    "row_ror:5 row_mask:0xf bank_mask:0xf",
    "row_mirror row_mask:0xf bank_mask:0xf",
    "row_half_mirror row_mask:0x5 bank_mask:0xa",
    "row_share:2 row_mask:0xf bank_mask:0xf",
    "row_xmask:7 row_mask:0xf bank_mask:0xf",
    "quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf",
    "dpp8:[1,0,3,2,5,4,7,6]",
    "dpp8:[7,6,5,4,3,2,1,0] fi:1",
]


def dpp_vs(form, exec_off=None):
    """A vertex shader: v6 = 100, then v6 = v0 (the lane) + 10 through DPP `form`, with lane `exec_off` off in
    EXEC for the DPP; param0.x exports v6."""
    lines = ["v_mbcnt_lo_u32_b32 v0, -1, 0", "v_mov_b32 v1, 10", "v_mov_b32 v6, 100", "s_mov_b32 s10, exec_lo"]
    if exec_off is not None:
        lines.append(f"v_cmpx_ne_u32_e32 {exec_off}, v0")
    lines += [f"v_add_nc_u32_dpp v6, v0, v1 {form}", "s_mov_b32 exec_lo, s10", "v_mov_b32 v5, 0",
              "exp pos0 v5, v5, v5, v5 done", "exp param0 v6, v5, v5, v5", "s_endpgm"]
    return support.translate_lines("vs_dpp_eager", lines)


def dpp_reference(form, exec_off=None, wave=32):
    """v6 per lane after dpp_vs(form, exec_off), from the RDNA2 DPP rules (docs/design.md, Gen.dpp_insn)."""
    mods = form.split()
    get = lambda key, d: next((int(m.split(":")[1], 0) for m in mods if m.startswith(key + ":")), d)  # noqa: E731
    row_mask, bank_mask, bound, fi = get("row_mask", 0xF), get("bank_mask", 0xF), get("bound_ctrl", 0), get("fi", 0)
    active = [lane != exec_off for lane in range(wave)]
    out = []
    for lane in range(wave):
        loc, ok = lane & 15, True
        if mods[0].startswith("dpp8:"):
            sel = [int(x) for x in mods[0][6:-1].split(",")]
            src = (lane & ~7) | sel[lane & 7]
        elif mods[0].startswith("quad_perm:"):
            sel = [int(x) for x in mods[0][11:-1].split(",")]
            src = (lane & ~3) | sel[lane & 3]
        else:
            kind = mods[0].split(":")[0]
            n = get(kind, 0)
            if kind == "row_shr":
                ok, s = loc >= n, loc - n
            elif kind == "row_shl":
                ok, s = loc + n < 16, loc + n
            elif kind == "row_ror":
                s = (loc - n) & 15
            elif kind == "row_share":
                s = n
            elif kind == "row_xmask":
                s = loc ^ n
            elif kind == "row_mirror":
                s = 15 - loc
            else:                                                    # row_half_mirror
                s = (loc & 8) | (7 - (loc & 7))
            src = (lane & ~15) | (s & 15)
        dpp8 = mods[0].startswith("dpp8:")
        enabled = dpp8 or ((row_mask >> (lane >> 4)) & 1 and (bank_mask >> ((lane >> 2) & 3)) & 1)
        valid = ok and (active[src] or (dpp8 and fi))
        if not active[lane] or not enabled or (not valid and not dpp8 and not bound):
            out.append(100)                                          # not written: keeps its old value
        else:
            out.append((src if valid else 0) + 10)
    return out


class ConditionalSubgroupOpsModel(unittest.TestCase):
    """The checker against hand-written SPIR-V: it must find the old DPP pattern and leave the safe ones alone."""

    HEAD = "%b = OpGroupNonUniformBallot %v4uint_bal %c3 %act\n" \
           "%e = OpGroupNonUniformBallotBitExtract %bool %c3 %b %src\n"

    def test_ballot_on_the_right_of_and(self):
        found = conditional_subgroup_ops(self.HEAD + "%a = OpLogicalAnd %bool %in_row %e\nOpStore %x %a\n")
        self.assertEqual([(r, op) for r, op, _ in found], [("%b", "OpGroupNonUniformBallot")])

    def test_shuffle_in_a_select_arm(self):
        text = "%ok = OpINotEqual %bool %x %c0\n%s = OpGroupNonUniformShuffle %uint %c3 %v %src\n" \
               "%t = OpIAdd %uint %s %c1\n%r = OpSelect %uint %ok %t %c0\nOpStore %y %r\n"
        self.assertEqual([op for _, op, _ in conditional_subgroup_ops(text)], ["OpGroupNonUniformShuffle"])

    def test_safe_positions(self):
        for tail in ("%a = OpLogicalAnd %bool %e %in_row\nOpStore %x %a\n",          # left of &&: always runs
                     "%r = OpSelect %uint %e %c1 %c0\nOpStore %x %r\n",               # the condition: always runs
                     "OpStore %x %e\n%a = OpLogicalAnd %bool %in_row %e\n",          # two uses: a temporary
                     "OpStore %x %e\n"):
            with self.subTest(tail=tail.split("\n")[0]):
                self.assertEqual(conditional_subgroup_ops(self.HEAD + tail), [])

    def test_vector_select_is_not_a_ternary(self):
        text = "%s = OpGroupNonUniformShuffle %v2uint %c3 %v %src\n" \
               "%r = OpSelect %v2uint %cond2 %s %zero2\nOpStore %y %r\n"
        self.assertEqual(conditional_subgroup_ops(text), [])


class DppSubgroupOpsRunInEveryLane(unittest.TestCase):
    def test_no_dpp_subgroup_op_is_conditional(self):
        for form in DPP_FORMS:
            for exec_off in (None, 5):
                with self.subTest(form=form, exec_off=exec_off):
                    spv = dpp_vs(form, exec_off)
                    self.assertIn("OpGroupNonUniformShuffle", spv)
                    self.assertEqual(conditional_subgroup_ops(spv), [])

    def test_values(self):
        for form in DPP_FORMS:
            for exec_off in (None, 5, 16):
                with self.subTest(form=form, exec_off=exec_off):
                    res = wi.run_wave(dpp_vs(form, exec_off), wave=32,
                                      builtins={"VertexIndex": lambda lane: lane, "InstanceIndex": lambda lane: 0})
                    got = [o["%out_param0"][0] for o in res["outputs"]]
                    self.assertEqual(got, dpp_reference(form, exec_off))

    def test_reference_matches_cs_dpp_ops(self):
        # runtime/test/shaders/cs_dpp_ops.asm: out[0..7] = 100 10 11 12 13 14 15 16.
        self.assertEqual(dpp_reference("row_shr:1 row_mask:0xf bank_mask:0xf")[:8],
                         [100, 10, 11, 12, 13, 14, 15, 16])


if __name__ == "__main__":
    unittest.main()
