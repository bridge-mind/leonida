"""CPU checks for tools/valu_fuzz.py (FUZZ-2, FUZZ-5): the reference models and the generators, no GPU.

The GPU side (vk_compute_test, rdna2spirv.py) is what the fuzzer compares against; these tests make sure the models it
compares with are right, so a mismatch is the translator's and not the fuzzer's.
"""
import os
import random
import shutil
import struct
import subprocess
import sys
import threading
from fractions import Fraction
from itertools import permutations
from pathlib import Path

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import valu_fuzz as vf  # noqa: E402


# ------------------------------------------------------------------ numbers

def test_f16_round_trips_every_finite_half():
    for h in range(0x10000):
        if (h & 0x7C00) == 0x7C00:
            continue
        v = vf.f16_value(h)
        bits, uncertain = vf.to_f16(v)
        if v == 0 or vf.is_f16_denorm(h):
            continue        # zeros come back as +0 (callers set the sign); denormals are uncertain by design
        assert not uncertain and bits == h, hex(h)


def test_to_f16_matches_pythons_round_to_nearest_even():
    rng = random.Random(1)
    for _ in range(20000):
        x = rng.uniform(-60000, 60000) * rng.choice((1, 1e-2, 1e-3))
        want = struct.unpack("<H", struct.pack("<e", x))[0]
        bits, uncertain = vf.to_f16(Fraction(x))
        if not uncertain:
            assert bits == want, (x, hex(bits), hex(want))


def test_to_f32_matches_pythons_round_to_nearest_even():
    rng = random.Random(2)
    for _ in range(20000):
        x = rng.uniform(-1e30, 1e30) * rng.choice((1, 1e-20, 1e-35))
        want = struct.unpack("<I", struct.pack("<f", x))[0]
        bits, uncertain = vf.to_f32(Fraction(x))
        if not uncertain:
            assert bits == want, (x, hex(bits), hex(want))


def test_one_is_not_zero():
    # a regression in the model itself: an unbiased exponent of 0 with a zero mantissa is +-1.0, not zero
    assert vf.to_f16(Fraction(1)) == (0x3C00, False)
    assert vf.to_f16(Fraction(-1)) == (0xBC00, False)
    assert vf.to_f32(Fraction(1)) == (0x3F800000, False)


def test_rtz():
    assert vf.to_f16(vf.f32_value(vf.f32_bits(-2.71828)), rtz=True) == (0xC16F, False)
    assert vf.to_f16(vf.f32_value(vf.f32_bits(-2.71828)))[0] == 0xC170      # nearest rounds away here


def test_signed_zeros():
    assert vf.f16_op("mul", 0xBC00, 0x0000) == (0x8000, False)    # -1 * +0 = -0
    assert vf.f16_op("add", 0x3C00, 0xBC00) == (0x0000, False)    # 1 + -1 = +0 under round to nearest
    assert vf.f16_op("add", 0x8000, 0x8000) == (0x8000, False)    # -0 + -0 = -0
    assert vf.f16_op("sub", 0x8000, 0x0000) == (0x8000, False)    # -0 - +0 = -0
    assert vf.f16_op("max", 0x0000, 0x8000)[1]                    # +0 vs -0: not promised, so uncertain


@pytest.mark.parametrize("operands,expected", [
    ((0x0000, 0x8000, 0x0000), 0x0000),  # +0, -0, +0
    ((0x8000, 0x0000, 0x8000), 0x8000),  # -0, +0, -0
    ((0xBC00, 0x0000, 0x8000), 0x8000),  # -1 < -0 < +0
    ((0x3C00, 0x0000, 0x8000), 0x0000),  # -0 < +0 < +1
])
def test_f16_med3_orders_signed_zeros(operands, expected):
    for args in dict.fromkeys(permutations(operands)):
        assert vf.f16_op("med3", *args) == (expected, False), args


@pytest.mark.parametrize("a,b,c,true,double", [
    (0xC281, 0xBA7F, 0xD175, 0xD121, 0xD120),
    (0x3C97, 0xBCD9, 0x50EB, 0x50BF, 0x50BE),
    (0x3846, 0x38EA, 0x5321, 0x532B, 0x532C),
])
def test_fma_is_rounded_once(a, b, c, true, double):
    # the FUZZ-2 finding: f32 Fma then f16 gives `double`; a correctly rounded f16 fma gives `true`
    assert vf.f16_op("fma", a, b, c) == (true, False)
    f32 = struct.unpack("<f", struct.pack("<f", float(vf.f16_value(a) * vf.f16_value(b) + vf.f16_value(c))))[0]
    assert vf.to_f16(Fraction(f32))[0] == double


def test_denormal_inputs_are_uncertain():
    assert vf.f16_op("add", 0x0001, 0x3C00)[1]
    assert vf.f16_op("mul", 0x3C00, 0x83FF)[1]


def test_i16_ops():
    assert vf.i16_op("v_ashrrev_i16", 4, 0x8000) == 0xF800
    assert vf.i16_op("v_lshrrev_b16", 20, 0x8000) == 0x0800          # shift amount is & 15
    assert vf.i16_op("v_max_i16", 0x7FFF, 0x8000) == 0x7FFF
    assert vf.i16_op("v_max_u16", 0x7FFF, 0x8000) == 0x8000
    assert vf.i16_op("v_sub_nc_u16", 0, 1) == 0xFFFF


# ------------------------------------------------------------------ f16 family

def test_f16_partial_writes_follow_the_contract():
    p = vf.F16Family(1, 0, 32)
    R = {r: vf.Reg(0xAAAA0000 | 0x3C00) for r in vf.POOL}
    d, s0, s1 = vf.POOL[1], vf.POOL[2], vf.POOL[3]
    p.step({"name": "v_add_f16", "d": d, "s": [s0, s1], "enc": "e32", "opsel": [0, 0, 0]}, R)
    assert R[d].v == 0xAAAA4000                              # 1 + 1 = 2 in the low half, high half kept
    p.step({"name": "v_add_nc_u16", "d": d, "s": [s0, s1], "enc": "e64", "opsel": [1, 1, 1]}, R)
    assert R[d].v == ((0xAAAA + 0xAAAA) & 0xFFFF) << 16 | 0x4000   # op_sel dst bit: high half, low kept
    p.step({"name": "v_cvt_f32_f16", "d": d, "s": [s0], "enc": "e32"}, R)
    assert R[d].v == 0x3F800000                              # 32-bit result: whole register


def test_f16_generator_is_deterministic_and_uses_gfx10_encodings():
    a = [p.asm() for p in vf.programs("f16", 7, 20, (32, 64))]
    b = [p.asm() for p in vf.programs("f16", 7, 20, (32, 64))]
    assert a == b
    text = "\n".join(a)
    for name in ("v_add_f16", "v_sub_f16", "v_mul_f16", "v_max_f16", "v_min_f16", "v_cvt_f32_f16"):
        assert f"{name}_e64" not in text or "op_sel" not in "".join(
            ln for ln in text.splitlines() if ln.startswith(f"{name}_e64"))    # GFX10: no op_sel on these
    assert "v_subrev_f16" not in text                                          # claimed elsewhere (#374)


def test_skip_op_leaves_an_op_out():
    for p in vf.programs("f16", 3, 30, (32,), skip=("v_fma_f16", "v_pk_fma_f16", "v_fmac_f16")):
        assert not any(op["name"] in ("v_fma_f16", "v_pk_fma_f16", "v_fmac_f16") for op in p.ops)


def _mem(**kw):
    """A MemFamily with its generated fields replaced by the caller's."""
    p = vf.MemFamily(1, 0, kw.pop("wave", 32))
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def _op(kind, n=1, off=0, r=None, soff=0, **kw):
    return {"kind": kind, "n": n, "off": off, "r": vf.POOL[0] if r is None else r, "soff": soff, "text": kind, **kw}


def test_word3_layouts():
    # GFX10: DST_SEL 0xfac, FORMAT 22 at [18:12], RESOURCE_LEVEL bit 24, OOB_SELECT [29:28]: the push-constant V# 0x31016fac
    assert vf.word3(3, "gfx10") == 0x31016FAC
    # GFX11 (Mesa gfx11-rsrc.json): FORMAT 6 bits [17:12], no RESOURCE_LEVEL, OOB_SELECT [29:28], TYPE [31:30] = 0
    assert vf.word3(3, "gfx11") == 0x30016FAC
    for mode in range(4):
        w = vf.word3(mode, "gfx11")
        assert (w >> 28) & 3 == mode and w >> 30 == 0 and (w >> 24) & 1 == 0 and (w >> 12) & 0x3F == 22
        assert vf.word3(mode, "gfx10") == w | 1 << 24


@pytest.mark.parametrize("args,want", [
    # RAW (3): offs + 4k + 4 > num_records is out of range, per dword (docs/design.md, RX 6800 and gfx1150 rows)
    ((3, 18, 0, 0, 0, 0, 3), True), ((3, 18, 0, 0, 0xc, 0xc, 1), False), ((3, 18, 0, 0, 8, 16, 0), False),
    ((3, 20, 0, 0, 13, 13, 1), False), ((3, 20, 0, 0, 12, 12, 1), True), ((3, 20, 0, 0, 12, 12, 2), False),
    # soffset is part of the checked offset in RAW mode only
    ((3, 24, 0, 0, 8, 16, 0), True), ((3, 24, 0, 0, 8, 24, 0), False),
    # STRUCTURED_WITH_OFFSET (0): idx >= N or off + 4k + 4 > stride
    ((0, 4, 16, 3, 12, 12, 0), True), ((0, 4, 16, 4, 0, 0, 0), False), ((0, 4, 16, 0, 12, 12, 1), False),
    # STRUCTURED (1): idx >= N only; the whole access goes through
    ((1, 4, 16, 3, 100, 100, 3), True), ((1, 4, 16, 4, 0, 0, 0), False), ((1, 0, 16, 0, 0, 0, 0), False),
    # DISABLED (2): N == 0 only
    ((2, 1, 0, 500, 100, 100, 0), True), ((2, 0, 0, 0, 0, 0, 0), False),
])
def test_buffer_dword_ok(args, want):
    assert vf.buffer_dword_ok(*args) is want


def test_out_of_range_loads_read_zero_and_stores_drop():
    # RAW V# of 2 slots and a bit: threads 0 and 1 are in range, 2 is cut after its first dword, the rest are out
    p = _mem(a_mode=3, a_n=2 * vf.SLOT + 4, a_stride=0, soff=0)
    r, r2 = vf.POOL[0], vf.POOL[4]
    p.ops = [_op("store_off", 2, 0, r), _op("load_off", 2, 0, r2)]
    regs, mem = p.model()
    assert regs[1][r2].v == regs[1][r].v and regs[1][r2 + 1].v == regs[1][r + 1].v      # in range
    assert regs[2][r2].v == regs[2][r].v and regs[2][r2 + 1].v == 0                    # straddles: per dword
    assert regs[5][r2].v == 0 and regs[5][r2 + 1].v == 0                               # out of range: 0
    assert mem[5 * vf.SLOT // 4] == vf.FILL_VK                                         # and its store was dropped
    assert mem[2 * vf.SLOT // 4 + 1] == vf.FILL_VK                                     # the straddling store's 2nd dword too


def test_structured_modes_check_the_index_only():
    r, r2 = vf.POOL[0], vf.POOL[4]
    p = _mem(b_mode=1, b_n=3, b_stride=vf.SLOT, soff=0)
    p.ops = [_op("store_idx", 4, 0, r), _op("load_idx", 4, 0, r2)]
    regs, mem = p.model()
    assert [regs[t][r2].v == regs[t][r].v for t in (0, 2)] == [True, True]
    assert all(regs[3][r2 + j].v == 0 for j in range(4))                               # index 3 >= num_records 3
    p = _mem(b_mode=0, b_n=100, b_stride=vf.SLOT, soff=0)
    p.ops = [_op("store_io", 1, 48, r), _op("load_io", 1, 48, r2)]                     # off = 48 + 4 * (t & 3) <= 60
    regs, mem = p.model()
    assert regs[3][r2].v == regs[3][r].v and mem[3 * vf.SLOT // 4 + (48 + 12) // 4] == regs[3][r].v


def test_scalar_loads_read_the_slot_of_the_waves_first_thread():
    for wave in (32, 64):
        p = _mem(wave=wave, a_mode=3, a_n=1 << 20, a_stride=0, s_stride=0, s_n=1 << 20, soff=0)
        r, r2 = vf.POOL[0], vf.POOL[4]
        p.ops = [_op("store_off", 1, 8, r), _op("sload", 1, 8, r2, imm=8), _op("sbuf", 1, 8, r2 + 1, imm=8)]
        regs, mem = p.model()
        for t in range(vf.THREADS):
            first = t - t % wave
            assert regs[t][r2].v == regs[first][r].v and regs[t][r2 + 1].v == regs[first][r].v


def test_s_buffer_load_bound_is_num_records_times_stride_and_ignores_oob_select():
    r2 = vf.POOL[4]
    for mode in range(4):
        # stride 4, 5 records = 20 bytes: the dword at 16 is in (nothing stored: the fill), the one at 20 is out
        p = _mem(s_stride=4, s_n=5, s_mode=mode, a_mode=3, a_n=1 << 20, a_stride=0, soff=0)
        p.ops = [_op("sbuf", 2, 16, r2, imm=16)]
        regs, _ = p.model()
        assert regs[0][r2].v == vf.FILL_VK and regs[0][r2 + 1].v == 0


def test_perturbed_models_differ():
    p = _mem(a_mode=3, a_n=2 * vf.SLOT + 4, a_stride=0, soff=0)
    p.ops = [_op("store_off", 2, 0, vf.POOL[0]), _op("load_off", 2, 0, vf.POOL[4])]
    good, _ = p.model()
    vf.PERTURB.add("whole_access")
    try:
        bad, _ = p.model()
    finally:
        vf.PERTURB.discard("whole_access")
    assert good[2][vf.POOL[4] + 1].v == 0 and good[2][vf.POOL[4]].v != 0
    assert bad[2][vf.POOL[4]].v == 0                                                   # all or nothing
    assert vf.PERTURB == set()


def test_mem_accesses_stay_in_the_threads_slot():
    for p in vf.programs("mem", 4, 40, (32,)):
        for op in p.ops:
            extra = 12 if op["kind"] in ("store_io", "load_io") else 0
            assert 0 <= op["off"] + op["soff"] and op["off"] + op["soff"] + extra + 4 * op["n"] <= vf.SLOT, op


def test_mem_generator_reaches_every_kind_and_mode():
    kinds, modes = set(), set()
    for p in vf.programs("mem", 1, 300, (32,)):
        kinds |= {op["kind"] for op in p.ops}
        modes |= {p.a_mode, p.b_mode, p.s_mode}
    assert kinds == set(vf.MEM_OPS) and modes == {0, 1, 2, 3}


def test_f16_pack_of_a_signaling_nan_half_is_uncertain_and_a_quiet_one_is_compared():
    assert vf.is_f16_snan(0x7C96) and vf.is_f16_snan(0xFC01) and not vf.is_f16_snan(0x7E96)
    assert not vf.is_f16_snan(0x7C00) and not vf.is_f16_snan(0x3C00)
    p = vf.F16Family(1, 0, 32)
    R = {r: vf.Reg(0x3C007E96) for r in vf.POOL}     # a quiet NaN low half
    d, s0, s1 = vf.POOL[1], vf.POOL[2], vf.POOL[3]
    p.step({"name": "v_pack_b32_f16", "d": d, "s": [s0, s1], "opsel": [0, 1]}, R)
    assert R[d].v == 0x3C007E96 and not R[d].ulo and not R[d].uhi


def test_compare_counts_the_words_it_does_not_judge():
    p = vf.F16Family(1, 0, 32)
    data = struct.pack(f"<{vf.THREADS * len(vf.POOL)}I", *([0] * (vf.THREADS * len(vf.POOL))))
    vf.compare(p, data)
    want, _ = p.model()
    assert p.uncertain_words == sum(1 for t in range(vf.THREADS) for r in vf.POOL if want[t][r].ulo or want[t][r].uhi)


def test_unknown_skip_op_is_a_usage_error(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["valu_fuzz.py", "--family", "mem", "--model-only", "--skip-op", "store_raw"])
    with pytest.raises(SystemExit) as exc:
        vf.main()
    assert exc.value.code == 2 and "unknown op store_raw" in capsys.readouterr().err


def test_f16_pack_of_a_nan_half_is_uncertain():
    p = vf.F16Family(1, 0, 32)
    R = {r: vf.Reg(0x3C007C96) for r in vf.POOL}     # high half 1.0, low half a signaling NaN
    d, s0, s1 = vf.POOL[1], vf.POOL[2], vf.POOL[3]
    p.step({"name": "v_pack_b32_f16", "d": d, "s": [s0, s1], "opsel": [0, 1]}, R)
    assert R[d].v == 0x3C007C96 and R[d].ulo and not R[d].uhi


# ------------------------------------------------------------------ the hardware kernels (text only, no GPU)

def test_hw_kernels_use_the_mnemonics_of_their_family():
    p = vf.programs("mem", 3, 1, (32,))[0]
    k10, k11 = "\n".join(p.hw_lines("gfx10")), "\n".join(p.hw_lines("gfx11"))
    assert "global_store_dword " in k10 and "global_store_b32 " in k11
    assert "dword" not in "\n".join(ln for ln in k11.splitlines() if ln.startswith(("buffer_", "s_load", "s_buffer")))
    assert "s_add_u32 s44, s2, 0x1000" in k11 and "s[2:3]" in k11          # the V#s are built from the output pointer
    assert f"{vf.word3(p.a_mode, 'gfx11'):#x}" in k11 and f"{vf.word3(p.a_mode, 'gfx10'):#x}" in k10


def test_hw_mem_kernel_waits_for_stores_and_invalidates_the_scalar_cache():
    p = _mem(soff=0)
    p.ops = [_op("store_off", 1, 0, vf.POOL[0]), _op("sload", 1, 0, vf.POOL[4], imm=0)]
    text = p.hw_body("gfx11")
    i_st = next(i for i, ln in enumerate(text) if ln.startswith("buffer_store"))
    i_ld = next(i for i, ln in enumerate(text) if ln.startswith("s_load"))
    assert "s_waitcnt_vscnt null, 0x0" in text[i_st:i_ld] and "s_dcache_inv" in text[i_st:i_ld]
    asm_text = "\n".join(p.asm_body())
    assert "s_dcache_inv" not in asm_text and "s_waitcnt_vscnt" not in asm_text       # the translator path is unchanged
    vf.HW_SYNC = "none"
    try:
        assert not any("vscnt" in ln or "dcache" in ln for ln in p.hw_body("gfx11"))
    finally:
        vf.HW_SYNC = "scalar"


def test_f16_16bit_integer_opsel_is_encoded_where_the_gfx11_assembler_has_no_spelling():
    p = vf.F16Family(1, 0, 32)
    op = {"name": "v_lshlrev_b16", "d": 22, "s": [20, 21], "enc": "e64", "opsel": [1, 0, 1],
          "text": "v_lshlrev_b16_e64 v22, v20, v21 op_sel:[1,0,1]"}
    p.ops = [op]
    t11, t10 = "\n".join(p.hw_body("gfx11")), "\n".join(p.hw_body("gfx10"))
    w0, w1 = vf.vop3_words(vf.GFX11_VOP3_OPCODE["v_lshlrev_b16"], 22, [20, 21], 0b1001)
    assert f".long {w0:#x}" in t11 and f".long {w1:#x}" in t11 and "v_lshlrev_b16_e64" not in t11.split("// ")[0]
    assert "v_lshlrev_b16_e64 v22, v20, v21 op_sel:[1,0,1]" in t10                  # GFX10 assembles it
    op["opsel"] = [0, 0, 0]
    op["text"] = "v_lshlrev_b16_e64 v22, v20, v21 op_sel:[0,0,0]"
    assert "v_lshlrev_b16_e64 v22, v20, v21" in p.hw_body("gfx11") and "op_sel" not in "\n".join(p.hw_body("gfx11"))


def _amd_tool(name):
    clang = os.environ.get("LEONIDA_AMD_CLANG")
    if not clang or not os.path.isfile(clang):
        return None
    tool = os.path.join(os.path.dirname(clang), name + (".exe" if os.name == "nt" else ""))
    return tool if os.path.isfile(tool) else None


@pytest.mark.skipif(_amd_tool("llvm-mc") is None, reason="AMD llvm-mc (next to LEONIDA_AMD_CLANG) not found")
def test_gfx11_opcode_table_matches_llvm_mc():
    from hw_oracle import asm

    mc = _amd_tool("llvm-mc")
    cmd = [mc, "-triple=amdgcn-amd-amdhsa", "-mcpu=gfx1150", "-show-encoding"]

    def assemble(text):
        return subprocess.run(cmd, input=text, capture_output=True, text=True)

    true16 = asm.needs_true16(lambda text: assemble(text).returncode == 0)
    for name, opcode in vf.GFX11_VOP3_OPCODE.items():
        text = f"{name} v20, v21, v22\n"
        r = assemble(asm.true16_spelling(text) if true16 else text)
        assert r.returncode == 0 and "encoding: [" in r.stdout, (name, r.stderr, r.stdout)
        got = [int(b, 16) for b in r.stdout.split("encoding: [")[1].split("]")[0].split(",")]
        w0, w1 = vf.vop3_words(opcode, 20, [21, 22], 0)
        want = [w0 & 255, w0 >> 8 & 255, w0 >> 16 & 255, w0 >> 24, w1 & 255, w1 >> 8 & 255, w1 >> 16 & 255, w1 >> 24]
        assert got == want, name


@pytest.mark.skipif(_amd_tool("clang") is None, reason="AMD clang (LEONIDA_AMD_CLANG) not found")
@pytest.mark.parametrize("family", sorted(vf.FAMILIES))
@pytest.mark.parametrize("arch", ["gfx1150", "gfx1030"])
def test_hw_kernels_assemble(family, arch):
    from hw_oracle import asm
    fam = asm.family(arch)
    cmd = [os.environ["LEONIDA_AMD_CLANG"], "--target=amdgcn-amd-amdhsa", f"-mcpu={arch}"]
    assembler = ["-x", "assembler", "-c", "-", "-o", os.devnull]

    def assemble(text, wave=32):
        return subprocess.run(cmd + (["-mwavefrontsize64"] if wave == 64 else []) + assembler,
                              input=text, capture_output=True, text=True)

    true16 = fam != "gfx10" and asm.needs_true16(lambda text: assemble(text).returncode == 0)
    for p in vf.programs(family, 2, 12, (32, 64)):
        k = asm.Kernel(p.hw_lines(fam), p.wave, vf.THREADS, p.out_words(), raw=True, vgprs=vf.HW_VGPRS, sgprs=vf.HW_SGPRS)
        text = asm.module_text([k], arch, wave=p.wave)[0]
        r = assemble(asm.true16_spelling(text) if true16 else text, p.wave)
        assert r.returncode == 0, (arch, p.name, r.stderr[-600:])


def test_oracle_hw_exits_77_without_the_tools(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("LEONIDA_AMD_CLANG", str(tmp_path / "no-such-clang.exe"))
    for family in sorted(vf.FAMILIES):
        monkeypatch.setattr(sys, "argv", ["valu_fuzz.py", "--family", family, "--oracle", "hw", "--programs", "2"])
        assert vf.main() == 77
    assert "hardware oracle unavailable" in capsys.readouterr().out


def test_oracle_hw_exits_77_on_an_rdna4_gpu(monkeypatch, capsys):
    from hw_oracle import runner

    class Rdna4:
        arch, device = "gfx1201", "fake"

        def close(self):
            pass

    monkeypatch.setattr(runner, "open_oracle", lambda **kw: Rdna4())
    for family in sorted(vf.FAMILIES):
        monkeypatch.setattr(sys, "argv", ["valu_fuzz.py", "--family", family, "--oracle", "hw", "--programs", "2"])
        assert vf.main() == 77
    assert "hardware oracle unavailable: gfx1201" in capsys.readouterr().out


def test_oracle_option_is_validated(monkeypatch):
    for argv in (["--oracle", "gpu"], ["--negative-control"], ["--oracle", "translator", "--negative-control"]):
        monkeypatch.setattr(sys, "argv", ["valu_fuzz.py", "--family", "mem", "--model-only", *argv])
        with pytest.raises(SystemExit) as exc:
            vf.main()
        assert exc.value.code == 2


# ------------------------------------------------------------------ assembling (needs llvm-mc)

def _mc():
    try:
        sys.path.insert(0, os.path.join(ROOT, "runtime", "test", "shaders"))
        import mc_command
        cmd = mc_command.mc_command()
        subprocess.run(cmd + ["--version"], capture_output=True, check=True)
        return cmd
    except Exception:
        return None


def _mc_accepts_f16_opsel(mc):
    """llvm-mc 21+ is needed for op_sel on 16-bit VOP3 (docs/testing.md); older builds reject the modifier.

    Probe a fixed instruction rather than a generated program, so a generator bug can never turn into a skip. The
    probe is the form llvm-mc 18 rejects (docs/testing.md): a 16-bit integer op, which GFX10 has only as op_sel VOP3.
    An f16 VOP2 op's VOP3 form takes no op_sel on GFX10 (tools/valu_fuzz.py), so no assembler accepts that as a probe.
    """
    probe = "v_ashrrev_i16_e64 v0, v1, v2 op_sel:[0,0,0]\n"
    r = subprocess.run(mc + ["-triple=amdgcn--amdpal", "-mcpu=gfx1013"], input=probe,
                       capture_output=True, text=True)
    return r.returncode == 0 and "error:" not in r.stderr


def test_f16_opsel_probe_distinguishes_old_assemblers():
    # A fake assembler that rejects everything stands in for LLVM < 21; one that accepts stdin stands in for 21+.
    reject = [sys.executable, "-c", "import sys; sys.exit(1)"]
    accept = [sys.executable, "-c", "import sys; sys.stdin.read()"]
    assert _mc_accepts_f16_opsel(reject) is False
    assert _mc_accepts_f16_opsel(accept) is True


@pytest.mark.skipif(_mc() is None, reason="llvm-mc with the AMDGPU target not found")
@pytest.mark.parametrize("family", sorted(vf.FAMILIES))
def test_programs_assemble_for_gfx1013(family):
    mc = _mc()
    if family == "f16" and not _mc_accepts_f16_opsel(mc):
        pytest.skip("llvm-mc 21+ is needed for op_sel on 16-bit VOP3 (docs/testing.md); set LLVM_MC")
    for p in vf.programs(family, 1, 6, (32, 64)):
        cmd = mc + ["-triple=amdgcn--amdpal", "-mcpu=gfx1013", "-show-encoding"]
        if p.wave == 64:
            cmd.append("-mattr=+wavefrontsize64,-wavefrontsize32")
        r = subprocess.run(cmd, input=p.asm(), capture_output=True, text=True)
        assert r.returncode == 0 and "error:" not in r.stderr, r.stderr[-800:]


def test_model_only_runs_every_family(capsys, monkeypatch):
    for family in sorted(vf.FAMILIES):
        monkeypatch.setattr(sys, "argv", ["valu_fuzz.py", "--family", family, "--programs", "4", "--model-only"])
        assert vf.main() == 0
    assert "programs generated and modelled" in capsys.readouterr().out


@pytest.mark.parametrize("extra", [
    ["--programs", "0"],
    ["--programs", "-5"],
    ["--max-report", "0"],
    ["--max-report", "-1"],
])
def test_cli_rejects_non_positive_programs_and_max_report(extra, capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["valu_fuzz.py", "--family", "f16", "--model-only", *extra])
    with pytest.raises(SystemExit) as exc:
        vf.main()
    assert exc.value.code == 2
    assert "--programs and --max-report must be at least 1" in capsys.readouterr().err


@pytest.mark.parametrize("bad", ["", "1", "a:b", "1:2:64:99", "1:-1", "1:2:16"])
def test_cli_rejects_invalid_show(bad, capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["valu_fuzz.py", "--family", "f16", "--show", bad])
    with pytest.raises(SystemExit) as exc:
        vf.main()
    assert exc.value.code == 2
    assert "--show expects SEED:INDEX[:WAVE] with INDEX >= 0 and WAVE in 32, 64" in capsys.readouterr().err


@pytest.mark.parametrize("spec", ["1:0", "1:2:32", "1:2:64"])
def test_cli_show_valid_formats(spec, capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["valu_fuzz.py", "--family", "f16", "--show", spec])
    assert vf.main() == 0
    assert "; model, thread 0:" in capsys.readouterr().out


# ------------------------------------------------------------------ runner (translator and GPU faked)

UNPINNED = "device: llvmpipe (LLVM 20.1.2, 256 bits) (subgroup 8..8)\ncompute subgroup size: 32 not available"


def _run_main(monkeypatch, capsys, tmp_path, msg, extra):
    """main() for two mem programs, every run returning all-zero words and vk_compute_test output `msg`."""
    def translate(progs, work, _emulate):
        (work / "spirv").mkdir(parents=True, exist_ok=True)
        for p in progs:
            (work / "spirv" / (p.name + ".spv")).touch()
        return {p.name: {} for p in progs}, ""

    def run_gpu(p, _spv, _build, _entry, _emulate, dump):
        dump.write_bytes(bytes(4 * p.out_words()))
        return 0, msg

    monkeypatch.setattr(vf, "translate", translate)
    monkeypatch.setattr(vf, "run_gpu", run_gpu)
    monkeypatch.setattr(sys, "argv", ["valu_fuzz.py", "--family", "mem", "--programs", "2", "--work", str(tmp_path),
                                      *extra])
    status = vf.main()
    return status, capsys.readouterr().out


def test_unpinned_wave32_is_a_skip_not_a_mismatch(monkeypatch, capsys, tmp_path):
    status, out = _run_main(monkeypatch, capsys, tmp_path, UNPINNED, ["--wave", "32"])
    assert status == 2, out                                   # nothing ran
    assert "MISMATCH" not in out
    assert "2 skipped" in out and "can't run it at 32 lanes" in out


def test_pinned_wave32_is_still_compared(monkeypatch, capsys, tmp_path):
    status, out = _run_main(monkeypatch, capsys, tmp_path, "compute subgroup size: 32", ["--wave", "32"])
    assert status == 1, out
    assert "MISMATCH" in out


def test_emulated_wave64_is_compared_at_any_subgroup_size(monkeypatch, capsys, tmp_path):
    status, out = _run_main(monkeypatch, capsys, tmp_path, UNPINNED, ["--wave", "64", "--emulate"])
    assert status == 1, out
    assert "MISMATCH" in out and "0 skipped" in out


def test_gpu_timeout_is_a_harness_error_and_next_program_runs(monkeypatch, capsys, tmp_path):
    build = tmp_path / "build"
    build.mkdir()
    (build / ("vk_compute_test" + (".exe" if os.name == "nt" else ""))).touch()
    progs = vf.programs("f16", 1, 2, (32,))
    attempted = []

    def translate(progs, work, _emulate):
        (work / "spirv").mkdir()
        for p in progs:
            (work / "spirv" / (p.name + ".spv")).touch()
        return {p.name: {} for p in progs}, ""

    def run(cmd, *, env, timeout, **kwargs):
        assert timeout == 0.25
        name = os.path.splitext(os.path.basename(cmd[1]))[0]
        attempted.append(name)
        if name == progs[0].name:
            raise subprocess.TimeoutExpired(cmd, timeout)
        with open(env["VKC_DUMP"], "wb") as dump:
            dump.write(bytes(4 * progs[1].out_words()))
        return subprocess.CompletedProcess(cmd, 0, "compute subgroup size: 32", "")

    def compare(prog, data):
        prog.uncertain_words = 0
        return []

    monkeypatch.setenv("LEONIDA_GPU_TEST_TIMEOUT", "0.25")
    monkeypatch.delenv("LEONIDA_PC_TAIL_UBO", raising=False)
    monkeypatch.setattr(vf, "translate", translate)
    monkeypatch.setattr(vf.subprocess, "run", run)
    monkeypatch.setattr(vf, "compare", compare)
    monkeypatch.setattr(sys, "argv", ["valu_fuzz.py", "--family", "f16", "--programs", "2", "--wave", "32",
                                      "--build", str(build), "--work", str(tmp_path)])

    assert vf.main() == 2
    assert attempted == [p.name for p in progs]
    out = capsys.readouterr().out
    assert f"ERROR {progs[0].name}" in out and "timed out after 0.25 s" in out
    assert "1/1 programs that ran match" in out and "1 harness errors" in out


def test_pc_tail_ubo_forwarded_to_translator_and_vk_compute_test(monkeypatch, tmp_path):
    cmds = []

    def fake_run(cmd, **kw):
        cmds.append((list(cmd), kw.get("env")))
        out_dir = tmp_path / "spirv"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "index.json").write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(vf.mc_command, "mc_command", lambda: ["llvm-mc"])
    monkeypatch.setattr(vf.subprocess, "run", fake_run)
    vf.translate([], tmp_path, False)
    vf.translate([], tmp_path, False, pc_tail=True)
    assert "--pc-tail-ubo" not in cmds[0][0]
    assert "--pc-tail-ubo" in cmds[1][0]

    build = tmp_path / "build"
    build.mkdir()
    (build / ("vk_compute_test" + (".exe" if os.name == "nt" else ""))).touch()
    monkeypatch.delenv("LEONIDA_PC_TAIL_UBO", raising=False)
    prog = vf.FAMILIES["mem"](1, 0, 32)
    vf.run_gpu(prog, tmp_path / "a.spv", str(build), {}, False, tmp_path / "a.bin")
    vf.run_gpu(prog, tmp_path / "a.spv", str(build), {}, False, tmp_path / "a.bin", pc_tail=True)
    assert "LEONIDA_PC_TAIL_UBO" not in cmds[2][1]
    assert cmds[3][1]["LEONIDA_PC_TAIL_UBO"] == "1"


def test_pc_tail_ubo_from_flag_or_env(monkeypatch, tmp_path):
    for extra, env_val, want in (
        ([], "", False),
        (["--pc-tail-ubo"], "", True),
        ([], "1", True),
        ([], "true", True),
        ([], "0", False),
    ):
        seen_t, seen_g = [], []

        def fake_translate(progs, work, _emulate, pc_tail=False):
            seen_t.append(pc_tail)
            (work / "spirv").mkdir(parents=True, exist_ok=True)
            for p in progs:
                (work / "spirv" / (p.name + ".spv")).touch()
            return {p.name: {} for p in progs}, ""

        def fake_run_gpu(p, _spv, _build, _entry, _emulate, dump, pc_tail=False):
            seen_g.append(pc_tail)
            return 77, "unsupported: test"

        monkeypatch.setenv("LEONIDA_PC_TAIL_UBO", env_val)
        monkeypatch.setattr(vf, "translate", fake_translate)
        monkeypatch.setattr(vf, "run_gpu", fake_run_gpu)
        monkeypatch.setattr(sys, "argv", ["valu_fuzz.py", "--family", "mem", "--programs", "1",
                                          "--wave", "32", "--work", str(tmp_path), *extra])
        vf.main()
        assert seen_t == [want] and seen_g == [want], (extra, env_val)


def test_parallel_assembly_preserves_wave_flags_and_inputs(monkeypatch, tmp_path):
    progs = vf.programs("f16", 1, 2, (32, 64))
    waves = {prog.asm(): prog.wave for prog in progs}
    barrier = None

    def fake_assemble(cmd, *, input, **kwargs):
        assert ("-mattr=+wavefrontsize64,-wavefrontsize32" in cmd) == (waves[input] == 64)
        if barrier:
            barrier.wait(timeout=10)
        return subprocess.CompletedProcess(cmd, 0, "\t.text\ns_endpgm\n", "")

    def fake_translate(cmd, **kwargs):
        output = Path(cmd[cmd.index("--out") + 1])
        assert len(list((output.parent / "disasm").glob("*.s"))) == len(progs)
        assert len(list((output.parent / "fssm").glob("*.fssm"))) == len(progs)
        output.mkdir()
        (output / "index.json").write_text("{}", encoding="utf-8", newline="\n")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(vf.mc_command, "mc_command", lambda: ["llvm-mc"])
    monkeypatch.setattr(vf.mc_command, "run", fake_assemble)
    monkeypatch.setattr(vf.subprocess, "run", fake_translate)
    serial = tmp_path / "serial"
    parallel = tmp_path / "parallel"
    vf.translate(progs, serial, False)
    barrier = threading.Barrier(2)
    vf.translate(progs, parallel, False, asm_jobs=2)
    for path in serial.rglob("*"):
        if path.is_file():
            assert path.read_bytes() == (parallel / path.relative_to(serial)).read_bytes()


def test_parallel_assembly_bounds_submitted_jobs(monkeypatch, tmp_path):
    programs = vf.programs("f16", 1, 5, (32, 64))
    submitted = 0
    collected = 0
    peak = 0

    class MeasuringExecutor(vf.ThreadPoolExecutor):
        def submit(self, fn, *args, **kwargs):
            nonlocal submitted, peak
            future = super().submit(fn, *args, **kwargs)
            submitted += 1
            peak = max(peak, submitted - collected)
            collect = future.result

            def result(*args, **kwargs):
                nonlocal collected
                collected += 1
                return collect(*args, **kwargs)

            future.result = result
            return future

    def fake_translate(cmd, **kwargs):
        output = Path(cmd[cmd.index("--out") + 1])
        output.mkdir()
        (output / "index.json").write_text("{}", encoding="utf-8", newline="\n")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(vf, "ThreadPoolExecutor", MeasuringExecutor)
    monkeypatch.setattr(vf.mc_command, "mc_command", lambda: ["llvm-mc"])
    monkeypatch.setattr(vf.mc_command, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, "s_endpgm\n", ""))
    monkeypatch.setattr(vf.subprocess, "run", fake_translate)
    vf.translate(programs, tmp_path, False, asm_jobs=2)
    assert submitted == collected == len(programs)
    assert peak == 4
