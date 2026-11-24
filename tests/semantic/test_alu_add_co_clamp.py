"""Semantic interpreter tests for VOP3 clamp on RDNA2 carry-out u32 add/sub ops.

Covers all six carry-out forms: v_add_co_u32, v_sub_co_u32, v_subrev_co_u32, v_add_co_ci_u32, v_sub_co_ci_u32,
and v_subrev_co_ci_u32. With VOP3 `clamp`, RDNA2 hardware (measured natively on gfx1035 and gfx1036, wave32 &
wave64) saturates D to 0xFFFFFFFF on add carry or 0 on sub borrow while still writing the real carry/borrow bit
into sdst (0 for inactive EXEC lanes). Without `clamp`, D wraps at 32 bits. No game files.
"""
import os
import random
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

M32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)

CO_OPS = ("v_add_co_u32", "v_sub_co_u32", "v_subrev_co_u32")
CO_CI_OPS = ("v_add_co_ci_u32", "v_sub_co_ci_u32", "v_subrev_co_ci_u32")
EDGE_PAIRS = (
    (0, 0),
    (0, 1),
    (1, 0),
    (5, 10),
    (10, 5),
    (5, 4),
    (0xFFFFFFFF, 0),
    (0, 0xFFFFFFFF),
    (0xFFFFFFFF, 1),
    (0xFFFFFFFF, 0xFFFFFFFF),
    (0x80000000, 0x80000000),
    (0x7FFFFFFF, 0x80000000),
)

_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_add_co_clamp.s")
    with open(os.path.join(_TMP_DIR.name, "ps_add_co_clamp.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def ref(op: str, x: int, y: int, cin: int, clamp: bool) -> tuple[int, int]:
    x, y = x & M32, y & M32
    if op.startswith("v_subrev_co"):
        x, y = y, x
    sub = "sub" in op
    c = (1 if cin else 0) if op.endswith("ci_u32") else 0
    if sub:
        full = x - y - c
        carry = 1 if full < 0 else 0
        res = 0 if (clamp and carry) else (full & M32)
    else:
        full = x + y + c
        carry = 1 if full > M32 else 0
        res = M32 if (clamp and carry) else (full & M32)
    return res, carry


def run_case(op: str, x: int, y: int, cin: int, clamp: bool, cd: str = "vcc_lo") -> tuple[int, int]:
    if _TMP_DIR is None:
        setUpModule()
    mod = " clamp" if clamp else ""
    lines = [
        f"v_mov_b32 v1, {x:#x}",
        f"v_mov_b32 v2, {y:#x}",
        f"s_mov_b32 vcc_lo, {-1 if cin else 0}",
    ]
    if op.endswith("ci_u32"):
        lines.append(f"{op}_e64 v0, {cd}, v1, v2, vcc_lo{mod}")
    else:
        lines.append(f"{op} v0, {cd}, v1, v2{mod}")
    if cd == "null":
        lines.append("v_mov_b32 v7, 0")
    else:
        lines.append(f"v_cndmask_b32_e64 v7, 0, 1, {cd}")
    lines += [
        "v_and_b32 v5, 0xffff, v0",
        "v_lshrrev_b32 v6, 16, v0",
        "v_or_b32 v5, 0x40000000, v5",
        "v_or_b32 v6, 0x40000000, v6",
        "v_or_b32 v7, 0x40000000, v7",
    ]
    asm = "; test_add_co_clamp\n" + "".join(f"\t{ln}\n" for ln in lines) + "\texp mrt0 v5, v6, v7, v5 done vm\n\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    lo, hi, cword = si.run_fragment(spv)["%out_mrt0"][:3]
    assert lo >> 16 == 0x4000 and hi >> 16 == 0x4000 and cword >> 16 == 0x4000, (hex(lo), hex(hi), hex(cword))
    return (lo & 0xFFFF) | ((hi & 0xFFFF) << 16), cword & 1


class AddCoClampTests(unittest.TestCase):
    def test_edge_pairs_all_six_ops(self):
        for op in CO_OPS:
            for clamp in (False, True):
                for x, y in EDGE_PAIRS:
                    self.assertEqual(run_case(op, x, y, 0, clamp), ref(op, x, y, 0, clamp),
                                     (op, hex(x), hex(y), clamp))
        for op in CO_CI_OPS:
            for clamp in (False, True):
                for cin in (0, 1):
                    for x, y in EDGE_PAIRS:
                        self.assertEqual(run_case(op, x, y, cin, clamp), ref(op, x, y, cin, clamp),
                                         (op, hex(x), hex(y), cin, clamp))

    def test_null_and_sgpr_carry_dests(self):
        for op in ("v_add_co_u32", "v_sub_co_u32", "v_add_co_ci_u32", "v_sub_co_ci_u32"):
            x, y, cin = (0xFFFFFFFF, 1, 1) if "add" in op else (0, 1, 1)
            want_d, want_c = ref(op, x, y, cin, True)
            got_d, _ = run_case(op, x, y, cin, True, cd="null")
            self.assertEqual(got_d, want_d, (op, "null"))
            self.assertEqual(run_case(op, x, y, cin, True, cd="s12"), (want_d, want_c), (op, "s12"))

    def test_inactive_exec_lane(self):
        if _TMP_DIR is None:
            setUpModule()
        lines = [
            "v_mov_b32 v0, 0x12345678",
            "v_mov_b32 v1, 0xffffffff",
            "v_mov_b32 v2, 0xffffffff",
            "s_mov_b32 vcc_lo, -1",
            "s_mov_b32 s10, exec_lo",
            "s_mov_b32 exec_lo, 0",
            "v_add_co_u32 v0, vcc_lo, v1, v2 clamp",
            "s_mov_b32 exec_lo, s10",
            "v_cndmask_b32_e64 v7, 0, 1, vcc_lo",
            "v_and_b32 v5, 0xffff, v0",
            "v_lshrrev_b32 v6, 16, v0",
            "v_or_b32 v5, 0x40000000, v5",
            "v_or_b32 v6, 0x40000000, v6",
            "v_or_b32 v7, 0x40000000, v7",
        ]
        asm = "; test_inactive\n" + "".join(f"\t{ln}\n" for ln in lines) + "\texp mrt0 v5, v6, v7, v5 done vm\n\ts_endpgm\n"
        with open(_SRC_PATH, "w", encoding="utf-8", newline="\n") as f:
            f.write(asm)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            spv = rdna2spirv.translate(_SRC_PATH)
        lo, hi, cword = si.run_fragment(spv)["%out_mrt0"][:3]
        self.assertEqual((lo & 0xFFFF) | ((hi & 0xFFFF) << 16), 0x12345678)
        self.assertEqual(cword & 1, 0)

    def test_seeded_random(self):
        rng = random.Random(20261122)
        for op in CO_OPS + CO_CI_OPS:
            for _ in range(12):
                x, y = rng.randrange(1 << 32), rng.randrange(1 << 32)
                cin = rng.randrange(2)
                for clamp in (False, True):
                    self.assertEqual(run_case(op, x, y, cin, clamp), ref(op, x, y, cin, clamp),
                                     (op, hex(x), hex(y), cin, clamp))


if __name__ == "__main__":
    unittest.main()
