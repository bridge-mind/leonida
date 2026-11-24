"""tools/sampler_source_census.py on hand-written listings and containers (no game files, no GPU)."""
import contextlib
import io
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import sampler_source_census as ssc  # noqa: E402

TOOL = os.path.join(ROOT, "tools", "sampler_source_census.py")
SFI0_SMP = 1 << 26
SAMPLE = ("image_sample v[0:3], v[4:5], s[20:27], s[28:31] dmask:0xf dim:SQ_RSRC_IMG_2D", 8)
END = ("s_endpgm", 4)


def enc(n):
    """A placeholder `; encoding:` of n bytes: the census only needs the length (for code offsets)."""
    return "; encoding: [" + ",".join(["0x00"] * n) + "]"


def listing(name, lines):
    """lines: (text, bytes) pairs -> a tab-indented listing as tools/shader_disasm.py writes it."""
    end = 0x100 + sum(n for _, n in lines)
    body = "".join(f"\t{text:<40}{enc(n)}\n" for text, n in lines)
    return f"; {name}.fssm code 0x100..0x{end:x}\n{body}"


def dxbc(flags):
    """A DXBC container holding only an SFI0 part with the given feature flags."""
    sfi0 = b"SFI0" + struct.pack("<I", 8) + struct.pack("<Q", flags)
    head = b"DXBC" + bytes(16) + struct.pack("<II", 1, 36 + len(sfi0)) + struct.pack("<II", 1, 36)
    return head + sfi0


class Work:
    """A temporary <work>/shaders/dxbc folder: carve index, containers, FSSM headers and disasm listings."""

    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.disasm = os.path.join(self.root, "disasm")
        os.makedirs(self.disasm)
        os.makedirs(os.path.join(self.root, "fssm"))
        self.index = []

    def add(self, name, stage, lines, flags=0, rsrc2=None, text=None, blob=None):
        self.index.append({"file": name + ".dxbc", "stage": stage})
        if flags is not None:
            with open(os.path.join(self.root, name + ".dxbc"), "wb") as f:
                f.write(dxbc(flags))
        with open(os.path.join(self.disasm, name + ".s"), "w", encoding="utf-8", newline="\n") as f:
            f.write(listing(name, lines) if text is None else text)
        if rsrc2 is not None or blob is not None:
            with open(os.path.join(self.root, "fssm", name + ".fssm"), "wb") as f:
                f.write(blob if blob is not None else bytes(0x138) + struct.pack("<I", rsrc2) + bytes(12))
        with open(os.path.join(self.root, "index.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(self.index, f)

    def close(self):
        self.tmp.cleanup()


# One program per class. Each ends in the same image_sample with its S# in s[28:31]. All values are made up.
USER_SGPRS = [SAMPLE, END]                               # the S# is user data as it stands
USER_TABLE = [                                           # 32-bit table pointer in s8, slot at a static offset
    ("s_mov_b32 s9, 0", 4),
    ("s_load_dwordx4 s[28:31], s[8:9], 0x40", 8),
    SAMPLE, END,
]
USER_HEAP = [                                            # sampler-heap base in s11, run-time index in s5
    ("s_mov_b32 s12, 0", 4),
    ("s_lshl_b32 s5, s5, 4", 4),
    ("s_load_dwordx4 s[28:31], s[11:12], s5", 8),
    SAMPLE, END,
]
LITERAL = [
    ("s_movk_i32 s28, 0x12", 4),
    ("s_mov_b32 s29, 0x300000", 8),
    ("s_mov_b32 s30, 0x5000000", 8),
    ("s_mov_b32 s31, 0", 4),
    SAMPLE, END,
]
CONSTANT_ADDRESS = [                                     # a fixed table: the address is two immediates
    ("s_mov_b32 s8, 0x7f001000", 8),
    ("s_mov_b32 s9, 0", 4),
    ("s_load_dwordx4 s[28:31], s[8:9], 0x40", 8),
    SAMPLE, END,
]
SYSTEM_POINTER = [                                       # the pointer sits above the user data
    ("s_load_dwordx4 s[28:31], s[40:41], 0x40", 8),
    SAMPLE, END,
]
CODE_RELATIVE = [                                        # a table next to the code
    ("s_getpc_b64 s[8:9]", 4),
    ("s_add_u32 s8, s8, 0x80", 8),
    ("s_addc_u32 s9, s9, 0", 4),
    ("s_load_dwordx4 s[28:31], s[8:9], 0x0", 8),
    SAMPLE, END,
]
NOT_FOLLOWED = [                                         # the table pointer is computed: no class fits
    ("s_mov_b32 s9, 0", 4),
    ("s_and_b32 s8, s8, s5", 4),
    ("s_load_dwordx4 s[28:31], s[8:9], 0x40", 8),
    SAMPLE, END,
]
FIXED_TABLE = [("s_mov_b32 s10, 0x7f001000", 8), ("s_mov_b32 s11, 0", 4)]   # a constant address in s[10:11]


def arms(first, second, before=(), after=()):
    """`before`, then `first` on one path and `second` on the other, then `after` and the sampling instruction."""
    size = sum(n for _, n in first) + 4, sum(n for _, n in second)
    return list(before) + [("s_cmp_eq_u32 s5, 0", 4), (f"s_cbranch_scc0 {size[0] // 4}", 4)] + list(first) + \
        [(f"s_branch {size[1] // 4}", 4)] + list(second) + list(after) + [SAMPLE, END]


def spilled(save, restore):
    """USER_TABLE with its S# saved by `save`, s[28:31] reused for a load from a system pointer, then `restore`."""
    return USER_TABLE[:2] + [(save.format(v=9 + d, s=28 + d), 4) for d in range(4)] + \
        [("s_load_dwordx4 s[28:31], s[40:41], 0x0", 8)] + \
        [(restore.format(v=9 + d, s=28 + d), 4) for d in range(4)] + [SAMPLE, END]


class SamplerSourceCensus(unittest.TestCase):
    def setUp(self):
        self.work = Work()

    def tearDown(self):
        self.work.close()

    def classes(self, lines, stage="ps", user=32):
        """[(class, detail)] of one hand-written program."""
        self.work.add("one", stage, lines)
        info = ssc.shader_ir.parse_file(os.path.join(self.work.disasm, "one.s"))
        return list(ssc.analyse(info["insns"], stage, user)[0].values())

    def test_user_sgprs(self):
        self.assertEqual(self.classes(USER_SGPRS), [("user_sgprs", None)])

    def test_user_pointer_static_and_dynamic(self):
        self.assertEqual(self.classes(USER_TABLE), [("user_pointer", "static")])
        self.assertEqual(self.classes(USER_HEAP), [("user_pointer", "dynamic")])

    def test_user_pointer_64bit_pair_and_offset_arithmetic(self):
        self.assertEqual(self.classes([
            ("s_add_u32 s8, s8, 0x100", 8),
            ("s_addc_u32 s9, s9, 0", 4),
            ("s_load_dwordx4 s[28:31], s[8:9], 0x40", 8),
            SAMPLE, END,
        ]), [("user_pointer", "static")])

    def test_literal(self):
        self.assertEqual(self.classes(LITERAL), [("literal", None)])
        # A constant that takes s_brev_b32 or s_not_b32 to build is still a literal, and says so.
        for op in ("s_brev_b32", "s_not_b32"):
            lines = LITERAL[:2] + [(f"{op} s30, 5", 4)] + LITERAL[3:]
            self.assertEqual(self.classes(lines), [("literal", f"with {op}")])
        self.assertEqual(ssc.UNARY["s_brev_b32"](5), 0xA0000000)
        self.assertEqual(ssc.UNARY["s_not_b32"](5), 0xFFFFFFFA)

    def test_other_base_forms(self):
        self.assertEqual(self.classes(CONSTANT_ADDRESS), [("other_base", "constant address")])
        self.assertEqual(self.classes(SYSTEM_POINTER), [("other_base", "system SGPR")])
        self.assertEqual(self.classes(CODE_RELATIVE), [("other_base", "code-relative address")])
        held = [("image_sample v[0:3], v[4:5], s[20:27], s[40:43] dmask:0xf dim:SQ_RSRC_IMG_2D", 8), END]
        self.assertEqual(self.classes(held), [("other_base", "system SGPR")])

    def test_the_user_count_decides_between_user_and_system(self):
        # The same code: s[8:9] is user data with 16 user SGPRs and a system pair with 8 (compute), and the
        # NGG s3 is never user data in a mesh shader.
        self.assertEqual(self.classes(USER_TABLE, "cs", 16), [("user_pointer", "static")])
        self.assertEqual(self.classes(USER_TABLE, "cs", 8), [("other_base", "system SGPR")])
        ngg = [("s_load_dwordx4 s[28:31], s[2:3], 0x40", 8), SAMPLE, END]
        self.assertEqual(self.classes(ngg, "ps"), [("user_pointer", "static")])
        self.assertEqual(self.classes(ngg, "ms"), [("undetermined", "pointer high word not followed")])

    def test_a_pointer_loaded_through_another_follows_the_first(self):
        def through(first):
            return first + [
                ("s_load_dwordx2 s[10:11], s[8:9], 0x8", 8),
                ("s_load_dwordx4 s[28:31], s[10:11], 0x40", 8),
                SAMPLE, END,
            ]
        self.assertEqual(self.classes(through([("s_mov_b32 s9, 0", 4)])), [("user_pointer", "static")])
        self.assertEqual(self.classes(through([("s_mov_b32 s8, 0x7f001000", 8), ("s_mov_b32 s9, 0", 4)])),
                         [("other_base", "constant address")])

    def test_a_buffer_load_follows_its_vsharp(self):
        self.assertEqual(self.classes([("s_buffer_load_dwordx4 s[28:31], s[4:7], 0x10", 8), SAMPLE, END]),
                         [("user_pointer", "static")])
        self.assertEqual(self.classes([("s_buffer_load_dwordx4 s[28:31], s[40:43], 0x10", 8), SAMPLE, END]),
                         [("other_base", "system SGPR")])
        self.assertEqual(self.classes([("s_buffer_load_dwordx4 s[28:31], s[30:33], 0x10", 8), SAMPLE, END]),
                         [("undetermined", "V# from different sources")])

    def test_the_second_half_of_a_wider_load(self):
        wide = [("s_mov_b32 s9, 0", 4), ("s_load_dwordx8 s[24:31], s[8:9], 0x40", 8), SAMPLE, END]
        self.assertEqual(self.classes(wide), [("user_pointer", "static")])

    def test_undetermined_is_reported_with_its_reason(self):
        self.assertEqual(self.classes(NOT_FOLLOWED), [("undetermined", "load base not followed")])
        self.assertEqual(self.classes([("s_and_b32 s28, s28, s5", 4), SAMPLE, END]),
                         [("undetermined", "a dword the dataflow does not follow")])
        self.assertEqual(self.classes([("s_mov_b32 s28, 1", 4), SAMPLE, END]),
                         [("undetermined", "sources of different classes: literal+user_sgprs")])
        high = [("s_mov_b32 s9, 0x7f", 4), ("s_load_dwordx4 s[28:31], s[8:9], 0x40", 8), SAMPLE, END]
        self.assertEqual(self.classes(high), [("undetermined", "pointer high word not followed")])

    def test_the_class_is_the_one_every_dword_has(self):
        # Two loads through the same user pointer make one S#: user data. One of them through a fixed table
        # makes the whole operand a fixed-table read; one the dataflow loses makes it undetermined.
        def halves(second):
            return [("s_mov_b32 s9, 0", 4)] + FIXED_TABLE + [("s_load_dwordx2 s[28:29], s[8:9], 0x0", 8)] + \
                second + [SAMPLE, END]
        self.assertEqual(self.classes(halves([("s_load_dwordx2 s[30:31], s[8:9], 0x8", 8)])),
                         [("user_pointer", "static")])
        self.assertEqual(self.classes(halves([("s_load_dwordx2 s[30:31], s[8:9], s5", 8)])),
                         [("user_pointer", "dynamic")])
        self.assertEqual(self.classes(halves([("s_load_dwordx2 s[30:31], s[10:11], 0x8", 8)])),
                         [("other_base", "constant address")])
        self.assertEqual(self.classes(halves([("s_and_b32 s31, s31, s5", 4)])),
                         [("undetermined", "a dword the dataflow does not follow")])
        self.assertEqual(self.classes(halves([("s_load_dwordx2 s[30:31], s[10:11], 0x8", 8),
                                              ("s_and_b32 s29, s29, s5", 4)])),
                         [("other_base", "constant address")])      # the fixed table wins over the lost dword

    def test_paths_that_bring_different_values_of_one_class(self):
        zero = [("s_mov_b32 s9, 0", 4)]
        slot = "s_load_dwordx4 s[28:31], s[8:9], {}"
        self.assertEqual(self.classes(arms([(slot.format("0x40"), 8)], [(slot.format("0x50"), 8)], zero)),
                         [("user_pointer", "static")])
        self.assertEqual(self.classes(arms([(slot.format("0x40"), 8)], [(slot.format("s5"), 8)], zero)),
                         [("user_pointer", "dynamic")])
        rest = LITERAL[1:4]
        self.assertEqual(self.classes(arms([("s_mov_b32 s28, 1", 4)], [("s_mov_b32 s28, 2", 4)], rest)),
                         [("literal", None)])
        self.assertEqual(self.classes(arms([("s_mov_b32 s28, 7", 4)], [("s_mov_b32 s28, 7", 4)], rest)),
                         [("literal", None)])

    def test_one_path_through_a_fixed_table_is_a_fixed_table(self):
        # Negative control for the join: the same two-armed program, one arm reading a constant address.
        user = ("s_load_dwordx4 s[28:31], s[8:9], 0x40", 8)
        fixed = ("s_load_dwordx4 s[28:31], s[10:11], 0x40", 8)
        before = [("s_mov_b32 s9, 0", 4)] + FIXED_TABLE
        self.assertEqual(self.classes(arms([user], [user], before)), [("user_pointer", "static")])
        self.assertEqual(self.classes(arms([user], [fixed], before)), [("other_base", "constant address")])
        self.assertEqual(self.classes(arms([fixed], [user], before)), [("other_base", "constant address")])
        # ... and an arm nothing can be said about leaves the instruction undetermined, not user data.
        lost = [("s_and_b32 s8, s8, s5", 4), user]
        self.assertEqual(self.classes(arms([user], lost, before)), [("undetermined", "load base not followed")])

    def test_paths_of_different_classes_are_not_given_one(self):
        # One path loads the S#, the other keeps the entry SGPRs; one path builds it, the other loads it.
        load = [("s_mov_b32 s9, 0", 4), ("s_load_dwordx4 s[28:31], s[8:9], 0x40", 8)]
        self.assertEqual(self.classes(arms(load, [])),
                         [("undetermined", "sources of different classes: user_pointer+user_sgprs")])
        self.assertEqual(self.classes(arms(load, LITERAL[:4])),
                         [("undetermined", "sources of different classes: literal+user_pointer")])

    def test_a_join_with_too_many_values_is_given_up(self):
        # A counter in s28: a constant that changes on every turn of the loop is dropped, not listed for ever.
        loop = LITERAL[1:4] + [("s_mov_b32 s28, 0", 4), ("s_add_u32 s28, s28, 1", 4), ("s_cmp_lt_u32 s28, s5", 4),
                               ("s_cbranch_scc1 0xfffd", 4), SAMPLE, END]
        self.assertEqual(self.classes(loop), [("undetermined", "a dword the dataflow does not follow")])

    def test_an_s_sharp_spilled_to_vgprs_keeps_its_class(self):
        whole = "v_mov_b32 v{v}, s{s}", "v_readfirstlane_b32 s{s}, v{v}"
        lane = "v_writelane_b32 v9, s{s}, {v}", "v_readlane_b32 s{s}, v9, {v}"
        self.assertEqual(self.classes(spilled(*whole)), [("user_pointer", "static")])
        self.assertEqual(self.classes(spilled(*lane)), [("user_pointer", "static")])
        # Negative controls: without the restore the registers hold the other load; a VGPR written in between,
        # or another lane read back, is not the spilled value.
        self.assertEqual(self.classes(spilled(whole[0], "s_nop")), [("other_base", "system SGPR")])
        lost = [("undetermined", "a dword the dataflow does not follow")]
        clobbered = spilled(*whole)
        clobbered.insert(7, ("v_add_nc_u32 v10, v10, v1", 4))
        self.assertEqual(self.classes(clobbered), lost)
        self.assertEqual(self.classes(spilled(lane[0], "v_readlane_b32 s{s}, v9, 0")), lost)
        self.assertEqual(self.classes(spilled(lane[0], "v_readlane_b32 s{s}, v10, {v}")), lost)
        rewritten = spilled(*lane)
        rewritten.insert(7, ("v_mov_b32 v9, 0", 4))
        self.assertEqual(self.classes(rewritten), lost)

    def test_a_spill_slot_one_path_does_not_write(self):
        # The save sits in one arm (the compiler's s_cbranch_execz skip): on the other path the VGPRs hold
        # something else, so the instruction is not given the class of the saved S#, but the reason names it.
        def program(first, second, table="s[8:9]"):
            save = [(f"v_mov_b32 v{9 + d}, s{28 + d}", 4) for d in range(4)]
            restore = [(f"v_readfirstlane_b32 s{28 + d}, v{9 + d}", 4) for d in range(4)]
            before = [("s_mov_b32 s9, 0", 4)] + FIXED_TABLE + [(f"s_load_dwordx4 s[28:31], {table}, 0x40", 8)]
            reuse = [("s_load_dwordx4 s[28:31], s[40:41], 0x0", 8)]
            return arms(save if first else [], save if second else [], before, reuse + restore)
        self.assertEqual(self.classes(program(True, True)), [("user_pointer", "static")])
        self.assertEqual(self.classes(program(True, False)),
                         [("undetermined", "spill slot a path does not save to (user_pointer where saved)")])
        self.assertEqual(self.classes(program(False, True)),
                         [("undetermined", "spill slot a path does not save to (user_pointer where saved)")])
        self.assertEqual(self.classes(program(False, False)),
                         [("undetermined", "a dword the dataflow does not follow")])
        # What was saved from a fixed table stays a fixed table, written on every path or not.
        self.assertEqual(self.classes(program(True, True, "s[10:11]")), [("other_base", "constant address")])
        self.assertEqual(self.classes(program(True, False, "s[10:11]")), [("other_base", "constant address")])

    def test_skips_on_exec_zero_can_be_left_out(self):
        # The compiler's shape: s_cbranch_execz jumps over a region that builds the S# when no lane is active.
        # With every edge followed the skipping path brings whatever s[28:31] held; with the skip left out the
        # region always runs. Only s_cbranch_execz is such a skip, and only when the region is fallen into.
        def skipping(branch, region):
            size = sum(n for _, n in region)
            return [("s_and_b32 s28, s28, s5", 4), (f"{branch} {size // 4}", 4)] + region + [SAMPLE, END]
        info = {}

        def both(lines, stage="ps"):
            self.work.add("one", stage, lines)
            info["insns"] = ssc.shader_ir.parse_file(os.path.join(self.work.disasm, "one.s"))["insns"]
            return [list(ssc.analyse(info["insns"], stage, 32, follow_skips=f)[0].values()) for f in (True, False)]
        lost = [("undetermined", "a dword the dataflow does not follow")]
        self.assertEqual(both(skipping("s_cbranch_execz", LITERAL[:4])), [lost, [("literal", None)]])
        self.assertEqual(both(skipping("s_cbranch_execz", USER_TABLE[:2])), [lost, [("user_pointer", "static")]])
        fixed = [("other_base", "constant address")]              # reported either way: three dwords are enough
        self.assertEqual(both(skipping("s_cbranch_execz", CONSTANT_ADDRESS[:3])), [fixed, fixed])
        self.assertEqual(both(skipping("s_cbranch_execnz", LITERAL[:4])), [lost, lost])
        self.assertEqual(both(skipping("s_cbranch_scc0", LITERAL[:4])), [lost, lost])
        # The census keeps the class of the first analysis and counts what the second one says next to it.
        self.work.add("one", "ps", skipping("s_cbranch_execz", USER_TABLE[:2]))
        self.work.add("two", "ps", skipping("s_cbranch_scc0", USER_TABLE[:2]))
        self.work.add("three", "ps", USER_TABLE)
        ps = ssc.census(self.work.disasm)["stages"]["ps"]
        self.assertEqual(ps["classes"]["undetermined"]["instructions"], 2)
        self.assertEqual(ps["classes"]["user_pointer"]["instructions"], 1)
        self.assertEqual(ps["undetermined_without_skips"], {"undetermined": 1, "user_pointer": 1})

    def test_a_vgpr_filled_by_a_vector_load_is_not_a_spill(self):
        # An S# that reaches the SGPRs through a vector memory load is not followed, and is reported as that.
        load = [(f"global_load_dword v{9 + d}, v1, s[8:9]", 8) for d in range(4)]
        restore = [(f"v_readfirstlane_b32 s{28 + d}, v{9 + d}", 4) for d in range(4)]
        loaded = [("undetermined", "restored from a VGPR a vector load wrote")]
        self.assertEqual(self.classes(load + restore + [SAMPLE, END]), loaded)
        copied = load + [("v_mov_b32 v20, v9", 4), ("v_readfirstlane_b32 s28, v20", 4)] + restore[1:]
        self.assertEqual(self.classes(copied + [SAMPLE, END]), loaded)
        # When another path saved a user S# in the same registers the reason names both.
        save = [(f"v_mov_b32 v{9 + d}, s{28 + d}", 4) for d in range(4)]
        self.assertEqual(self.classes(arms(save, load, USER_TABLE[:2], restore)),
                         [("undetermined", "spill slot a path fills with a vector load (user_pointer where saved)")])
        self.assertEqual(self.classes(arms(save[:3], load, USER_TABLE[:2], restore)), loaded)
        # A table pointer that came through a vector load is a base the dataflow does not follow.
        pointer = [load[0], ("v_readfirstlane_b32 s10, v9", 4), ("s_mov_b32 s11, 0", 4),
                   ("s_load_dwordx4 s[28:31], s[10:11], 0x40", 8), SAMPLE, END]
        self.assertEqual(self.classes(pointer), [("undetermined", "load base not followed")])

    def test_a_chain_of_loads_is_followed_once_per_load(self):
        # Each load takes its V# (four dwords) or its pointer (two) from the load before it. Following every
        # dword back on its own asked for the first load 4**depth times; the work has to grow with the chain.
        def calls(lines):
            self.work.add("one", "ps", lines)
            insns = ssc.shader_ir.parse_file(os.path.join(self.work.disasm, "one.s"))["insns"]
            count = [0]

            def profile(frame, event, arg):
                count[0] += event == "call" and frame.f_code.co_name == "load_source"
            sys.setprofile(profile)
            try:
                classes = list(ssc.analyse(insns, "ps", 32)[0].values())
            finally:
                sys.setprofile(None)
            return classes, count[0]
        wait = ("s_waitcnt lgkmcnt(0)", 4)
        for depth in (2, 3, 4, 5, 12, 40):
            vsharp = [("s_buffer_load_dwordx4 s[28:31], s[28:31], 0", 8), wait] * depth + [SAMPLE, END]
            pointer = [("s_load_dwordx2 s[8:9], s[8:9], 0", 8), wait] * depth + \
                [("s_load_dwordx4 s[28:31], s[8:9], 0x40", 8), SAMPLE, END]
            for lines, loads in ((vsharp, depth), (pointer, depth + 1)):
                classes, n = calls(lines)
                self.assertEqual(classes, [("user_pointer", "static")], depth)
                self.assertLessEqual(n, 2 * loads, f"{n} load_source calls for a chain of {loads} loads")
        # The same chains from a fixed table: what a load is remembered as is its own source, not user data.
        fixed = [("s_mov_b32 s8, 0x7f001000", 8), ("s_mov_b32 s9, 0", 4),
                 ("s_load_dwordx4 s[28:31], s[8:9], 0x40", 8)]
        chain = [("s_buffer_load_dwordx4 s[28:31], s[28:31], 0", 8), wait] * 12
        self.assertEqual(calls(fixed + chain + [SAMPLE, END])[0], [("other_base", "constant address")])
        self.assertEqual(calls(USER_TABLE[:2] + chain + [SAMPLE, END])[0], [("user_pointer", "static")])

    def test_a_pointer_chased_in_a_loop_ends_and_is_not_followed(self):
        # s[8:9] is the entry pointer or the one the load itself brought on the turn before: the load's address
        # depends on the load. That is given up, every time it is asked, and what it is given up as is not
        # remembered for the loads around it: the one before the loop stays user data.
        before = ("s_load_dwordx4 s[40:43], s[8:9], 0x80", 8)
        loop = [("s_load_dwordx2 s[8:9], s[8:9], 0", 8), ("s_cmp_lt_u32 s5, s6", 4), ("s_cbranch_scc1 0xfffc", 4),
                ("s_load_dwordx4 s[28:31], s[8:9], 0x40", 8), SAMPLE,
                ("image_sample v[0:3], v[4:5], s[20:27], s[40:43] dmask:0xf dim:SQ_RSRC_IMG_2D", 8), SAMPLE, END]
        lost = ("undetermined", "load base not followed")
        self.assertEqual(self.classes([before] + loop), [lost, ("user_pointer", "static"), lost])
        # A V# that is its own source through a loop, read after another load of the same cycle.
        ring = [("s_buffer_load_dwordx4 s[28:31], s[28:31], 0", 8),
                ("s_buffer_load_dwordx4 s[28:31], s[28:31], 0x10", 8),
                ("s_cmp_lt_u32 s5, s6", 4), ("s_cbranch_scc1 0xfffa", 4), SAMPLE, END]
        self.assertEqual(self.classes(ring), [("undetermined", "V# from different sources")])

    def test_only_the_sampler_operand_counts(self):
        # The T# comes from a constant address and the S# from user data: not a driver sampler table. A load
        # nobody samples with, and an image_load (no sampler operand), count nothing.
        lines = [
            ("s_mov_b32 s8, 0x7f001000", 8),
            ("s_mov_b32 s9, 0", 4),
            ("s_load_dwordx8 s[20:27], s[8:9], 0x0", 8),
            ("s_load_dwordx4 s[40:43], s[8:9], 0x40", 8),
            ("image_load v[0:3], v[4:5], s[20:27] dmask:0xf dim:SQ_RSRC_IMG_2D", 8),
            SAMPLE, END,
        ]
        self.assertEqual(self.classes(lines), [("user_sgprs", None)])

    def test_gather_and_get_lod_are_sampling_instructions(self):
        lines = USER_TABLE[:2] + [
            ("image_gather4_lz v[0:3], v[4:5], s[20:27], s[28:31] dmask:0x1 dim:SQ_RSRC_IMG_2D", 8),
            ("image_get_lod v[0:1], v[4:5], s[20:27], s[28:31] dmask:0x3 dim:SQ_RSRC_IMG_2D", 8),
            ("image_sample_l v[0:3], v[4:6], s[20:27], s[4:7] dmask:0xf dim:SQ_RSRC_IMG_2D", 8),
            END,
        ]
        self.assertEqual(self.classes(lines),
                         [("user_pointer", "static"), ("user_pointer", "static"), ("user_sgprs", None)])

    def test_statistics_bound_what_the_classes_cannot_decide(self):
        # "pointer + constant" and "table address + index" are one instruction: the constant is reported. The
        # second shader adds a larger one to a T# pointer, which is not the census's business.
        self.work.add("one", "ps", [
            ("s_add_u32 s8, s8, 0x7f001000", 8),
            ("s_mov_b32 s9, 0", 4),
            ("s_load_dwordx4 s[28:31], s[8:9], 0x40", 8),
            SAMPLE, END,
        ])
        self.work.add("two", "ps", [
            ("s_add_u32 s10, s10, 0x7fff0000", 8),
            ("s_mov_b32 s11, 0", 4),
            ("s_load_dwordx8 s[20:27], s[10:11], 0x800", 8),
            ("s_lshl3_add_u32 s8, s8, 0x30", 4),
            ("s_mov_b32 s9, 0", 4),
            ("s_load_dwordx4 s[28:31], s[8:9], 0x10", 8),
            SAMPLE, END,
        ])
        t = ssc.census(self.work.disasm)["total"]
        self.assertEqual(t["classes"]["user_pointer"], 2)
        self.assertEqual(t["max_pointer_constant"], 0x7f001000)
        self.assertEqual(t["max_static_offset"], 0x40)
        # The constant can sit in the offset operand as well, and a constant offset register is a static offset.
        self.work.add("three", "ps", [
            ("s_add_u32 s5, s5, 0x7f002000", 8),
            ("s_mov_b32 s9, 0", 4),
            ("s_load_dwordx4 s[28:31], s[8:9], s5", 8),
            SAMPLE,
            ("s_mov_b32 s6, 0x7f003000", 8),
            ("s_load_dwordx4 s[28:31], s[8:9], s6", 8),
            SAMPLE, END,
        ])
        t = ssc.census(self.work.disasm)["total"]
        self.assertEqual(t["classes"]["user_pointer"], 4)
        self.assertEqual(t["max_pointer_constant"], 0x7f002000)
        self.assertEqual(t["max_static_offset"], 0x7f003000)

    def test_census_counts_per_stage_and_sfi0_bit(self):
        self.work.add("a", "ps", USER_TABLE[:3] + USER_TABLE[2:])               # two instructions, one shader
        self.work.add("b", "ps", USER_HEAP + USER_HEAP[:-1], flags=SFI0_SMP)   # the copy after s_endpgm is dead
        self.work.add("c", "ps", LITERAL)
        self.work.add("d", "vs", CONSTANT_ADDRESS)
        self.work.add("e", "cs", USER_TABLE, rsrc2=16 << 1)
        self.work.add("f", "cs", USER_TABLE, rsrc2=4 << 1)
        self.work.add("g", "ms", [END])
        self.work.add("h", "ms", USER_SGPRS, flags=None)                       # no DXBC container: no SFI0
        out = ssc.census(self.work.disasm)
        self.assertEqual((out["shaders_read"], out["skipped"]), (8, {}))
        ps = out["stages"]["ps"]
        self.assertEqual((ps["shaders"], ps["shaders_sampling"], ps["instructions"]), (3, 3, 4))
        self.assertEqual(ps["unreachable_instructions"], 1)
        self.assertEqual(ps["classes"]["user_pointer"],
                         {"instructions": 3, "shaders": 2, "in_sampler_heap_shaders": 1})
        self.assertEqual(ps["classes"]["literal"],
                         {"instructions": 1, "shaders": 1, "in_sampler_heap_shaders": 0})
        self.assertEqual(ps["detail"], {"user_pointer": {"dynamic": {"instructions": 1, "shaders": 1},
                                                         "static": {"instructions": 2, "shaders": 1}}})
        self.assertEqual(out["stages"]["vs"]["detail"],
                         {"other_base": {"constant address": {"instructions": 1, "shaders": 1}}})
        cs = out["stages"]["cs"]
        self.assertEqual((cs["classes"]["user_pointer"]["instructions"], cs["classes"]["other_base"]["instructions"]),
                         (1, 1))
        ms = out["stages"]["ms"]
        self.assertEqual((ms["shaders"], ms["shaders_sampling"], ms["shaders_without_sfi0"]), (2, 1, 1))
        self.assertEqual(ps["shaders_without_sfi0"], 0)
        self.assertEqual(out["total"]["classes"], {"user_sgprs": 1, "user_pointer": 4, "literal": 1,
                                                   "other_base": 2, "undetermined": 0})
        self.assertEqual(out["total"]["instructions"], 8)
        self.assertEqual(ssc.census(self.work.disasm, jobs=2), out)

    def test_far_branch_and_code_that_is_not_contiguous(self):
        # A folded far branch (explicit target) over a run of data words: the block after the gap is reached
        # from the branch only, and nothing falls through into it from the block before the gap.
        def ins(pc, text, size=4):
            row = ssc.shader_ir.parse_line(f"\t{text:<40}{enc(size)}")
            row["pc"] = pc
            return row
        far = {"op": "s_branch", "args": [], "mods": [], "pc": 0x104, "enc": "00" * 20, "target": 0x200}
        insns = [ins(0x100, "s_mov_b32 s9, 0"), far,
                 ins(0x200, "s_load_dwordx4 s[28:31], s[8:9], 0x40", 8), ins(0x208, SAMPLE[0], 8),
                 ins(0x210, "s_endpgm")]
        blocks, succ = ssc.blocks_of(insns)
        self.assertEqual(([len(b) for b in blocks], succ), ([2, 3], [[1], []]))
        self.assertEqual(ssc.analyse(insns, "ps", 32)[0], {0x208: ("user_pointer", "static")})
        # Without the branch the second run is unreachable: its instruction is reported, not classified.
        cut = [insns[0], ins(0x104, "s_endpgm")] + insns[2:]
        uses, stats = ssc.analyse(cut, "ps", 32)
        self.assertEqual((uses, stats["unreachable"]), ({}, 1))

    def test_data_words_in_the_code_range_are_stepped_over(self):
        # A branch over one data word. The disassembler prints the instructions back to back, so the listing's
        # offsets are 4 short after the word and the branch would land inside the load; anchored to the blob
        # (the bytes here are placeholders, only their position counts) the load is where the branch goes.
        code = [("s_mov_b32 s9, 0", b"\x11" * 4), ("s_branch 1", b"\x12" * 4), (None, b"\xee" * 4),
                ("s_load_dwordx4 s[28:31], s[8:9], 0x40", b"\x13" * 8), (SAMPLE[0], b"\x14" * 8),
                ("s_endpgm", b"\x15" * 4)]
        text = "; data.fssm code 0x100..0x120\n" + "".join(
            f"\t{t:<40}; encoding: [{','.join(f'0x{b:02x}' for b in raw)}]\n" for t, raw in code if t)
        blob = bytes(0x100) + b"".join(raw for _, raw in code)
        self.work.add("data", "ps", None, text=text + "\t; invalid instruction encoding\n", blob=blob)
        self.work.add("plain", "ps", None, text=text, blob=blob)
        out = ssc.census(self.work.disasm)
        self.assertEqual(out["skipped"], {"branch target outside code": 1})
        self.assertEqual(out["skipped_sampling_instructions"], {"branch target outside code": 1})
        self.assertEqual(out["total"]["classes"]["user_pointer"], 1)
        self.assertEqual(out["stages"]["ps"]["shaders"], 1)

    def test_skipped_listings_are_counted_with_their_sampling_instructions(self):
        self.work.add("bad", "ps", USER_TABLE[:3] + [("s_cbranch_scc0 9", 4)])
        self.work.add("empty", "ps", [])
        self.work.add("nohdr", "cs", USER_TABLE)                      # compute without an FSSM header
        self.work.add("data", "ps", USER_TABLE)                       # data words, and no blob to anchor to
        with open(os.path.join(self.work.disasm, "data.s"), "a", encoding="utf-8", newline="\n") as f:
            f.write("\t; invalid instruction encoding\n")
        with open(os.path.join(self.work.disasm, "stray.s"), "w", encoding="utf-8", newline="\n") as f:
            f.write(listing("stray", USER_TABLE))
        out = ssc.census(self.work.disasm)
        self.assertEqual(out["skipped"], {"branch target outside code": 1, "embedded data words, no FSSM blob": 1,
                                          "no FSSM header": 1, "no instructions": 1, "not in the carve index": 1})
        self.assertEqual(out["skipped_sampling_instructions"],
                         {"branch target outside code": 1, "embedded data words, no FSSM blob": 1})
        self.assertEqual(out["stages"], {})

    def test_expect_none_turns_the_census_into_a_check(self):
        self.work.add("a", "ps", USER_TABLE)
        self.work.add("b", "ps", LITERAL)
        ok = [sys.executable, "-I", TOOL, self.work.disasm, "--expect-none", "other_base", "undetermined"]
        p = subprocess.run(ok, capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        # Negative control: one shader that reads its S# from a fixed address must fail the same check, and a
        # shader the dataflow cannot follow must fail it too instead of passing as "no driver table".
        self.work.add("c", "ps", CONSTANT_ADDRESS)
        p = subprocess.run(ok, capture_output=True, text=True)
        self.assertEqual(p.returncode, 1)
        self.assertEqual(p.stderr.strip(), "MISMATCH other_base: 1 instructions")
        self.work.add("d", "ps", NOT_FOLLOWED)
        p = subprocess.run(ok, capture_output=True, text=True)
        self.assertEqual(p.returncode, 1)
        self.assertEqual(p.stderr.split("\n")[:2],
                         ["MISMATCH other_base: 1 instructions", "MISMATCH undetermined: 1 instructions"])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(ssc.main([self.work.disasm, "--expect-none", "user_sgprs"]), 0)

    def test_cli_report_json_and_errors(self):
        self.work.add("a", "ps", USER_TABLE)
        self.work.add("b", "vs", CONSTANT_ADDRESS)
        js = os.path.join(self.work.root, "out", "census.json")
        p = subprocess.run([sys.executable, "-I", TOOL, self.work.disasm, "--json", js],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("vs other_base: constant address: 1 in 1 shaders", p.stdout)
        self.assertNotIn("a.s", p.stdout)                             # counts only, no shader names
        with open(js, "rb") as f:
            raw = f.read()
        self.assertNotIn(b"\r", raw)
        self.assertEqual(json.loads(raw)["total"]["classes"]["other_base"], 1)
        p = subprocess.run([sys.executable, "-I", TOOL, os.path.join(self.work.root, "nope")],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 2)
        self.assertTrue(p.stderr.startswith("ERROR:"), p.stderr)
        os.remove(os.path.join(self.work.root, "index.json"))
        p = subprocess.run([sys.executable, "-I", TOOL, self.work.disasm], capture_output=True, text=True)
        self.assertEqual(p.returncode, 2)
        self.assertTrue(p.stderr.startswith("ERROR: cannot read the carve index"), p.stderr)


if __name__ == "__main__":
    unittest.main()
