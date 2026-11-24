"""lane_masks.analyse: which registers hold a lane mask when an instruction runs.

    python -I tests/test_lane_masks.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import lane_masks as L  # noqa: E402


def s(r, n=1):
    return {"k": "sgpr", "r": r, "n": n}


def sp(t):
    return {"k": "special", "t": t}


def imm(v):
    return {"k": "imm", "v": v}


def v(r):
    return {"k": "vgpr", "r": r, "n": 1}


def ins(op, *args):
    return {"op": op, "args": list(args), "mods": []}


def masks_at_last(blocks, succ, block=-1):
    return set(L.analyse(blocks, succ)[0][id(blocks[block][-1])])


def test_straight_line():
    use = ins("s_bcnt1_i32_b64", s(8), s(6, 2))
    # A saved EXEC is a mask, in both registers of the pair.
    assert masks_at_last([[ins("s_mov_b64", s(6, 2), sp("exec")), use]], [[]]) == {6, 7}
    # A compare result is a mask, whether it lands in VCC or in a register pair.
    assert masks_at_last([[ins("v_cmp_gt_f32_e32", sp("vcc_lo"), imm(0), v(1)), use]], [[]]) == {"vcc"}
    assert masks_at_last([[ins("v_cmp_gt_f32_e64", s(6, 2), imm(0), v(1)), use]], [[]]) == {6, 7}
    # Masks combined with masks stay masks; saveexec leaves the old EXEC in its destination.
    assert masks_at_last([[ins("v_cmp_eq_u32_e64", s(2, 2), v(0), v(1)), ins("s_and_b64", s(6, 2), s(2, 2), sp("exec")),
                           ins("s_and_saveexec_b64", s(10, 2), s(6, 2)), use]], [[]]) == {2, 3, 6, 7, 10, 11}
    # A light mask built from loaded dwords is data, and so is a mask once data is mixed in or it is overwritten.
    assert masks_at_last([[ins("s_buffer_load_dword", s(6), s(0, 4), imm(8)), ins("s_lshr_b32", s(7), s(6), imm(3)),
                           use]], [[]]) == set()
    assert masks_at_last([[ins("s_mov_b64", s(6, 2), sp("exec")), ins("s_and_b64", s(6, 2), s(6, 2), s(20, 2)),
                           use]], [[]]) == set()
    assert masks_at_last([[ins("s_mov_b64", s(6, 2), sp("exec")), ins("s_load_dwordx4", s(4, 4), s(0, 2), imm(0)),
                           use]], [[]]) == set()
    # Compares that only set SCC, and v_cmpx, read their first operand: the mask in it survives.
    assert masks_at_last([[ins("s_mov_b64", s(6, 2), sp("exec")), ins("s_cmp_eq_u64", s(6, 2), imm(0)),
                           ins("v_cmpx_eq_u32_e32", s(6), v(1)), use]], [[]]) == {6, 7}
    # Carry out of a 64-bit add, and the carry/sign sdst of 64-bit MAD, are lane masks.
    assert masks_at_last([[ins("v_add_co_u32_e64", v(0), s(6, 2), v(1), v(2)), use]], [[]]) == {6, 7}
    assert masks_at_last([[ins("v_mad_u64_u32", {"k": "vgpr", "r": 0, "n": 2}, s(6, 2), v(2), v(3),
                               {"k": "vgpr", "r": 4, "n": 2}), use]], [[]]) == {6, 7}
    assert masks_at_last([[ins("v_mad_i64_i32", {"k": "vgpr", "r": 0, "n": 2}, sp("vcc"), v(2), v(3),
                               {"k": "vgpr", "r": 4, "n": 2}), use]], [[]]) == {"vcc"}
    assert masks_at_last([[ins("v_mad_u64_u32", {"k": "vgpr", "r": 0, "n": 2}, sp("null"), v(2), v(3),
                               {"k": "vgpr", "r": 4, "n": 2}), use]], [[]]) == set()


def test_joins():
    use = ins("s_ff1_i32_b64", s(8), s(6, 2))
    mask = ins("s_mov_b64", s(6, 2), sp("exec"))
    zero = ins("s_mov_b64", s(6, 2), imm(0))
    data = ins("s_load_dwordx2", s(6, 2), s(0, 2), imm(0))
    diamond = [[1, 2], [3], [3], []]
    # A mask on one path and a constant 0 on the other: a mask. A mask on one and data on the other: not.
    assert masks_at_last([[ins("s_nop")], [mask], [zero], [use]], diamond) == {6, 7}
    assert masks_at_last([[ins("s_nop")], [mask], [data], [use]], diamond) == set()
    # Constant on every path: plain data (the bits are the same either way).
    assert masks_at_last([[ins("s_nop")], [zero], [dict(zero)], [use]], diamond) == set()
    # A mask accumulated around a loop stays a mask; the loop body is seen with the entry state merged in.
    acc = ins("s_or_b64", s(6, 2), s(6, 2), sp("vcc"))
    cmp_ = ins("v_cmp_lt_u32_e32", sp("vcc_lo"), v(0), v(1))
    assert masks_at_last([[zero], [cmp_, acc], [use]], [[1], [1, 2], []]) == {6, 7, "vcc"}
    # Unreachable blocks get an answer too (nothing is a mask there).
    assert masks_at_last([[mask], [use]], [[], []]) == set()


def test_parked_masks():
    save = ins("s_mov_b64", s(0, 2), sp("exec"))
    park = [ins("v_writelane_b32", v(128), s(0), imm(0)), ins("v_writelane_b32", v(128), s(1), imm(1))]
    clobber = ins("s_load_dwordx2", s(0, 2), s(8, 2), imm(0))
    fetch = [ins("v_readlane_b32", s(0), v(128), imm(0)), ins("v_readlane_b32", s(1), v(128), imm(1))]
    use = ins("s_mov_b64", sp("exec"), s(0, 2))
    slots = {("lane", 128, 0), ("lane", 128, 1)}
    # A saved EXEC parked in two lanes of v128 and fetched back is a mask again; the slots stay known.
    assert masks_at_last([[save] + park + [clobber] + fetch + [use]], [[]]) == {0, 1} | slots
    # Data parked the same way stays data, and so does a mask fetched from a lane it was not parked in.
    assert masks_at_last([[clobber] + park + fetch + [use]], [[]]) == set()
    other_lane = ins("v_readlane_b32", s(0), v(128), imm(5))
    assert masks_at_last([[save, park[0], clobber, other_lane, use]], [[]]) == {("lane", 128, 0)}
    # Writing the VGPR any other way ends what was parked there.
    assert masks_at_last([[save] + park + [clobber, ins("v_mov_b32_e32", v(128), imm(0))] + fetch + [use]], [[]]) == set()
    # Parked on one path only: not a mask.
    assert masks_at_last([[save], park, [ins("s_nop")], [clobber] + fetch + [use]], [[1, 2], [3], [3], []]) == set()
    # The slots the translator has to keep per lane: those some v_readlane fetches a mask from.
    blocks = [[save] + park + [clobber] + fetch + [use]]
    assert L.parked(blocks, L.analyse(blocks, [[]])[0]) == [(128, 0), (128, 1)]
    blocks = [[clobber] + park + fetch + [use]]
    assert L.parked(blocks, L.analyse(blocks, [[]])[0]) == []


def test_masks_through_a_vgpr():
    use = ins("s_mov_b32", sp("exec_lo"), s(3))
    masks = [ins("s_mov_b32", s(3), sp("exec_lo")), ins("v_cmp_eq_u32_e64", s(47), v(1), v(2))]
    # An OR of two masks done in the vector unit and read back: still a mask, through v51.
    via = [ins("v_mov_b32_e32", v(51), s(3)), ins("v_or3_b32", v(51), s(3), s(47), v(51)),
           ins("v_readfirstlane_b32", s(3), v(51))]
    blocks = [masks + via + [use]]
    got, writes = L.analyse(blocks, [[]])
    assert set(got[id(use)]) == {3, 47, ("v", 51)}
    # Only the v_or3 is a mask write: the v_mov copy is an ordinary VALU write (SHADER-21 part 2).
    assert writes == {id(via[1])}
    # Mixed with data, or with the VGPR written by anything else in between, it is data.
    data = ins("s_buffer_load_dword", s(47), s(0, 4), imm(0))
    assert masks_at_last([[masks[0], data] + via + [use]], [[]]) == set()
    assert masks_at_last([masks + [via[0], ins("v_add_nc_u32_e32", v(51), v(51), v(2)), via[2], use]], [[]]) == {47}
    # A constant alone does not make a VGPR a mask (v_mov_b32 v, 0 is everywhere).
    blocks = [[ins("v_mov_b32_e32", v(51), imm(0)), ins("v_readfirstlane_b32", s(3), v(51)), use]]
    assert L.analyse(blocks, [[]]) == ({id(i): {} for i in blocks[0]}, set())


def test_vmov_mask_copy_is_exec_gated():
    # SHADER-21 part 2: a mask copied into a VGPR with v_mov_b32 is an ordinary
    # VALU write, so inactive lanes keep the old VGPR; only per-lane mask
    # computations (v_or3_b32 etc.) take the all-lanes store.
    cmp_ = ins("v_cmp_lt_u32_e64", sp("vcc_lo"), v(5), v(6))
    copy = ins("v_mov_b32_e32", v(2), sp("vcc_lo"))
    calc = ins("v_or3_b32", v(3), sp("vcc_lo"), sp("vcc_lo"), sp("vcc_lo"))
    use = ins("s_bcnt1_i32_b64", s(8), s(6, 2))
    blocks = [[cmp_, copy, calc, use]]
    got, writes = L.analyse(blocks, [[]])
    assert ("v", 2) in got[id(use)] and ("v", 3) in got[id(use)]
    assert writes == {id(calc)}


def test_bfm64_and_half_kinds():
    # s_bfm_b64 produces a 64-bit lane mask (low half in s6, high half in s7).
    use = ins("s_bcnt1_i32_b64", s(8), s(6, 2))
    got = L.analyse([[ins("s_bfm_b64", s(6, 2), imm(16), imm(8)), use]], [[]])[0][id(use)]
    assert got == {6: L.MASK_LO, 7: L.MASK_HI}
    # exec_lo / exec_hi preserve half identity on 32-bit copies.
    got2 = L.analyse([[ins("s_mov_b32", s(6), sp("exec_lo")), ins("s_mov_b32", s(7), sp("exec_hi")), use]], [[]])[0][id(use)]
    assert got2 == {6: L.MASK_LO, 7: L.MASK_HI}
    # vcc_lo / vcc_hi map to MASK_LO / MASK_HI when vcc holds a lane mask, and None when vcc is data.
    vcmp = ins("v_cmp_gt_u32_e32", sp("vcc_lo"), imm(16), v(0))
    got_vcc = L.analyse([[vcmp, ins("s_mov_b32", s(6), sp("vcc_lo")), ins("s_mov_b32", s(7), sp("vcc_hi")), use]], [[]])[0][id(use)]
    assert got_vcc == {"vcc": L.MASK, 6: L.MASK_LO, 7: L.MASK_HI}
    got_vcc_data = L.analyse([[ins("s_buffer_load_dword", sp("vcc_lo"), s(0, 4), imm(0)),
                               ins("s_mov_b32", s(6), sp("vcc_lo")), ins("s_mov_b32", s(7), sp("vcc_hi")), use]], [[]])[0][id(use)]
    assert got_vcc_data == {}
    # A separate write to vcc_hi ends vcc's life as a lane mask, so vcc_hi reads back as data.
    got_vcc_hi = L.analyse([[vcmp, ins("s_mov_b32", sp("vcc_hi"), imm(0x1234)),
                             ins("s_mov_b32", s(7), sp("vcc_hi")), use]], [[]])[0][id(use)]
    assert got_vcc_hi == {}
    # Merging low half and high half yields generic MASK (unknown half -> keep both halves).
    got3 = L.analyse([[ins("s_nop")], [ins("s_mov_b32", s(6), sp("exec_lo"))],
                      [ins("s_mov_b32", s(6), sp("exec_hi"))], [use]], [[1, 2], [3], [3], []])[0][id(use)]
    assert got3 == {6: L.MASK}


def test_b64_mask_imm_and_include_either():
    use64 = ins("s_bcnt1_i32_b64", s(8), s(6, 2))
    use32 = ins("s_bcnt1_i32_b32", s(8), s(6))
    vcmp = ins("v_cmp_eq_u32_e32", sp("vcc"), imm(0), v(0))
    # 64-bit bitwise op of a lane mask with a non-0/-1 immediate stores a per-lane mask via mask_src/mask_dst.
    got_and64 = L.analyse([[vcmp, ins("s_and_b64", s(6, 2), sp("vcc"), imm(0xFF)), use64]], [[]])[0][id(use64)]
    assert got_and64 == {"vcc": L.MASK, 6: L.MASK_LO, 7: L.MASK_HI}
    # 64-bit copy of a constant into vcc stores mask_src(imm) per lane into %vcc.
    got_mov_vcc = L.analyse([[ins("s_mov_b64", sp("vcc"), imm(0x3F)), use64]], [[]])[0][id(use64)]
    assert got_mov_vcc == {"vcc": L.MASK}
    # Plain 64-bit data combined with an immediate stays data; 32-bit mask + non-0/-1 immediate into an SGPR stays data.
    assert L.analyse([[ins("s_and_b64", s(6, 2), s(20, 2), imm(0xFF)), use64]], [[]])[0][id(use64)] == {}
    assert L.analyse([[ins("s_and_b32", s(6), sp("exec_lo"), imm(1)), use32]], [[]])[0][id(use32)] == {}
    # include_either=True exposes registers holding constant 0 / -1 (EITHER) alongside cur_masks.
    blocks = [[ins("s_mov_b32", s(6), sp("exec_lo")), ins("s_mov_b32", s(7), imm(0)), use64]]
    out, writes, either = L.analyse(blocks, [[]], include_either=True)
    assert out[id(use64)] == {6: L.MASK_LO}
    assert writes == set()
    assert either[id(use64)] == {7}


def test_writelane_sdwa_dpp_swaps_stores_and_div_scale():
    use_s = ins("s_bcnt1_i32_b32", s(8), s(6))
    use_v = ins("v_readfirstlane_b32", s(8), v(5))
    mk = ins("v_cmp_eq_u32_e64", s(6), v(0), v(1))

    # v_writelane_b32 writes one lane of v5, so v5 no longer holds a whole-VGPR mask ("v", 5),
    # while the parked slot ("lane", 5, 0) reflects the written scalar operand.
    wl_data = ins("v_writelane_b32", v(5), s(9), imm(0))
    got_wl = L.analyse([[mk, ins("v_or_b32_e32", v(5), s(6), s(6)), wl_data, use_v]], [[]])[0][id(use_v)]
    assert ("v", 5) not in got_wl
    wl_mask = ins("v_writelane_b32", v(5), s(6), imm(0))
    got_wl2 = L.analyse([[mk, ins("v_or_b32_e32", v(5), s(6), s(6)), wl_mask, use_v]], [[]])[0][id(use_v)]
    assert ("v", 5) not in got_wl2 and ("lane", 5, 0) in got_wl2

    # SDWA and DPP variants of VECTOR_BITWISE are not whole-dword per-lane mask computations:
    # they must neither mark ("v", dst) nor enter mask_writes.
    dpp_or = {"op": "v_or_b32_dpp", "args": [v(5), v(4), v(4)], "mods": ["row_shr:1", "row_mask:0x0"]}
    sdwa_or = {"op": "v_or_b32_sdwa", "args": [v(5), s(6), s(6)],
               "mods": ["dst_sel:BYTE_0", "dst_unused:UNUSED_PRESERVE", "src0_sel:BYTE_0", "src1_sel:BYTE_0"]}
    dpp_mov = {"op": "v_mov_b32_dpp", "args": [v(5), v(4)], "mods": ["row_shr:1"]}
    for op_ins in (dpp_or, sdwa_or, dpp_mov):
        blk = [mk, ins("v_mov_b32_e32", v(4), s(6)), op_ins, use_v]
        got, writes = L.analyse([blk], [[]])
        assert ("v", 5) not in got[id(use_v)], op_ins["op"]
        assert id(op_ins) not in writes, op_ins["op"]

    # v_swap_b32 writes both operands, invalidating parked/VGPR masks in a[1] as well as a[0].
    got_swap = L.analyse([[mk, ins("v_writelane_b32", v(5), s(6), imm(0)),
                           ins("v_swap_b32_e32", v(4), v(5)), use_v]], [[]])[0][id(use_v)]
    assert ("lane", 5, 0) not in got_swap and ("v", 5) not in got_swap

    # Relative-destination writes invalidate all registers at or above the base register.
    for rel_op in ("v_movreld_b32_e32", "v_movrelsd_b32_e32", "v_movrelsd_2_b32_e32"):
        got_rel = L.analyse([[mk, ins("v_writelane_b32", v(6), s(6), imm(0)),
                              ins("v_writelane_b32", v(3), s(6), imm(0)),
                              ins(rel_op, v(4), v(7)), use_v]], [[]])[0][id(use_v)]
        assert ("lane", 6, 0) not in got_rel and ("lane", 3, 0) in got_rel, rel_op
    got_swaprel = L.analyse([[mk, ins("v_writelane_b32", v(6), s(6), imm(0)),
                              ins("v_writelane_b32", v(2), s(6), imm(0)),
                              ins("v_swaprel_b32_e32", v(5), v(3)), use_v]], [[]])[0][id(use_v)]
    assert ("lane", 6, 0) not in got_swaprel and ("lane", 2, 0) in got_swaprel
    for srel_op in ("s_movreld_b32", "s_movreld_b64", "s_movrelsd_2_b32"):
        got_srel = L.analyse([[ins("s_mov_b32", s(2), sp("exec_lo")),
                               ins("s_mov_b32", s(6), sp("exec_lo")),
                               ins(srel_op, s(4), s(8)), use_s]], [[]])[0][id(use_s)]
        assert 6 not in got_srel and 2 in got_srel, srel_op

    # Store instructions, LDS-targeted buffer loads, and non-returning atomics do not write a[0].
    got_sstore = L.analyse([[mk, ins("s_store_dword", s(6), s(0, 2), imm(0)),
                             ins("s_buffer_store_dword", s(6), s(0, 4), imm(0)), use_s]], [[]])[0][id(use_s)]
    assert 6 in got_sstore
    for non_dst in (
        ins("global_store_dword", v(5), v(6), sp("off")),
        ins("buffer_store_dword", v(5), v(6), s(0, 4), imm(0)),
        ins("tbuffer_store_format_x", v(5), v(6), s(0, 4), imm(0)),
        ins("flat_store_dword", v(5), v(6)),
        ins("image_store", v(5), v(6), s(0, 8)),
        ins("ds_write_b32", v(5), v(6)),
        {"op": "buffer_load_dword", "args": [v(5), s(0, 4), imm(0)], "mods": ["offen", "lds"]},
        ins("global_atomic_add", v(5), v(6), sp("off")),
        ins("buffer_atomic_add", v(5), v(6), s(0, 4), imm(0)),
        ins("image_atomic_add", v(5), v(6), s(0, 8)),
        ins("ds_add_u32", v(5), v(6)),
    ):
        got_keep = L.analyse([[mk, ins("v_writelane_b32", v(5), s(6), imm(0)),
                               non_dst, use_v]], [[]])[0][id(use_v)]
        assert ("lane", 5, 0) in got_keep, non_dst["op"]
    # Returning atomics (glc / _rtn) DO write a[0] and must still invalidate it.
    for rtn_op in (
        {"op": "global_atomic_add", "args": [v(5), v(6), v(7), sp("off")], "mods": ["glc"]},
        {"op": "buffer_atomic_add", "args": [v(5), v(6), s(0, 4), imm(0)], "mods": ["glc"]},
        ins("ds_add_rtn_u32", v(5), v(6), v(7)),
    ):
        got_kill = L.analyse([[mk, ins("v_writelane_b32", v(5), s(6), imm(0)),
                               rtn_op, use_v]], [[]])[0][id(use_v)]
        assert ("lane", 5, 0) not in got_kill, rtn_op["op"]

    # v_div_scale_f32 / v_div_scale_f64 write a per-lane scale mask to a[1].
    for ds_op in ("v_div_scale_f32", "v_div_scale_f64"):
        got_ds = L.analyse([[ins(ds_op, v(0), s(6, 2), v(1), v(2), v(1)), use_s]], [[]])[0][id(use_s)]
        assert got_ds == {6: L.MASK_LO, 7: L.MASK_HI}, ds_op



def test_v_and_or_xnor_bfi_track_vector_masks():
    use = ins("s_and_b32", s(12), s(7), s(8))
    and_or = ins("v_and_or_b32", v(2), sp("vcc_lo"), sp("exec_lo"), imm(0))
    xnor = ins("v_xnor_b32", v(3), sp("vcc_lo"), sp("exec_lo"))
    bfi = ins("v_bfi_b32", v(4), sp("vcc_lo"), sp("exec_lo"), imm(-1))
    mixed = ins("v_bfi_b32", v(5), sp("exec_lo"), sp("exec_hi"), imm(0))
    blocks = [[
        ins("v_cmp_eq_u32_e32", sp("vcc_lo"), v(0), v(1)),
        and_or,
        xnor,
        bfi,
        mixed,
        ins("v_readfirstlane_b32", s(7), v(2)),
        ins("v_readfirstlane_b32", s(8), v(3)),
        ins("v_readfirstlane_b32", s(9), v(4)),
        ins("v_readfirstlane_b32", s(10), v(5)),
        use,
    ]]
    out, writes = L.analyse(blocks, [[]])
    for w in (and_or, xnor, bfi, mixed):
        assert id(w) in writes, w["op"]
    masks = out[id(use)]
    assert masks.get(("v", 2)) == L.MASK_LO
    assert masks.get(("v", 3)) == L.MASK_LO
    assert masks.get(("v", 4)) == L.MASK_LO
    assert masks.get(("v", 5)) == L.MASK
    assert masks.get(7) == L.MASK_LO
    assert masks.get(8) == L.MASK_LO
    assert masks.get(9) == L.MASK_LO
    assert masks.get(10) == L.MASK
    # Negative controls: any non-mask or non-0/-1 constant operand keeps dst as plain data.
    for bad in (
        ins("v_and_or_b32", v(6), sp("vcc_lo"), v(9), imm(0)),
        ins("v_and_or_b32", v(6), sp("vcc_lo"), sp("exec_lo"), imm(5)),
        ins("v_xnor_b32", v(6), sp("vcc_lo"), v(9)),
        ins("v_bfi_b32", v(6), v(9), sp("exec_lo"), imm(0)),
        ins("v_bfi_b32", v(6), sp("vcc_lo"), sp("exec_lo"), imm(3)),
    ):
        bad_use = ins("s_and_b32", s(12), s(7), sp("exec_lo"))
        out, writes = L.analyse([[
            ins("v_cmp_eq_u32_e32", sp("vcc_lo"), v(0), v(1)),
            bad,
            ins("v_readfirstlane_b32", s(7), v(6)),
            bad_use,
        ]], [[]])
        assert id(bad) not in writes, bad["op"]
        assert 7 not in out[id(bad_use)] and ("v", 6) not in out[id(bad_use)], bad["op"]

if __name__ == "__main__":
    test_straight_line()
    test_joins()
    test_parked_masks()
    test_masks_through_a_vgpr()
    test_vmov_mask_copy_is_exec_gated()
    test_bfm64_and_half_kinds()
    test_b64_mask_imm_and_include_either()
    test_writelane_sdwa_dpp_swaps_stores_and_div_scale()
    test_v_and_or_xnor_bfi_track_vector_masks()
    print("ok")
