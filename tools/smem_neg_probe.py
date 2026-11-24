r"""SHADER-336: measure s_load / s_buffer_load with sOFF + IMM < 0 on an AMD GPU (tools/hw_oracle), and compare
with the translator's rule.

    python -I tools/smem_neg_probe.py                 # translator expectation only (no GPU)
    python -I tools/smem_neg_probe.py --oracle hw     # native AMD GPU, wave32 and wave64, plus the comparison

One raw kernel per wave size. Thread 0 seeds 64 dwords at byte 0x100 of the output buffer with 0x5eed0000 + X (X =
byte offset from 0x140, the "base" every load starts from; X is negative below it), then runs each load once and
writes its destination SGPRs to out[0..]. Every load either reads inside the seeded 0x100..0x1ff bytes, or (for
`s_buffer_load`) is clipped by a NUM_RECORDS of 0x100 bytes around the base, so no row can read outside the buffer
whatever the hardware rule turns out to be. Rows the assembler rejects (a negative `s_buffer_load` IMM) are encoded
by hand as `.long` words. Standard library only; no game files.

Exit codes: 0 every row matches the translator, 1 a mismatch, 2 harness error, 77 no hardware oracle.
Usage: `python -I tools/smem_neg_probe.py --help` lists the options.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "hw_oracle"))
sys.path.insert(0, HERE)

SEED, BASE, WORDS = 0x5EED0000, 0x140, 16
WORD3 = 0x31016FAC          # V# word3 of the RX 6800 measurement (docs/design.md, Scalar/buffer memory)


def seed(x):
    """The value seeded at byte BASE + x, or None (0) outside the seeded 0x100..0x1ff."""
    return (SEED + x) & 0xFFFFFFFF if -0x40 <= x < 0xC0 else None


def sbuf_raw(sdst, sbase, soff, imm):
    """s_buffer_load_dword sdst, s[sbase:sbase+3], s<soff> offset:imm with a 21-bit signed IMM (GFX10 SMEM)."""
    w0 = 0xF4200000 | (sdst << 6) | (sbase >> 1)
    w1 = ((soff & 0x7F) << 25) | (imm & 0x1FFFFF)
    return [f".long 0x{w0:08x}", f".long 0x{w1:08x}"]


# (name, lines, destination SGPRs, sOFF value or None, translator rule -> list of region offsets X or None)
#   s_load:        address = (base & ~3) + (sOFF & ~3) + sext21(IMM & ~3), 64-bit, no bound
#   s_buffer_load: off = (sOFF & ~3) + (IMM & ~3) as a 32-bit value; dword i is 0 when off + 4i + 4 > NUM_RECORDS
NUM_RECORDS = 0x100
ROWS = [
    ("sload_imm_m4",        ["s_load_dword s40, s[4:5], -0x4"], [40], None, [-4]),
    ("sload_imm_m40",       ["s_load_dword s41, s[4:5], -0x40"], [41], None, [-0x40]),
    ("sload_soff4_imm_m8",  ["s_load_dword s42, s[4:5], s20 offset:-0x8"], [42], 4, [-4]),
    ("sload_soff6_imm_m1",  ["s_load_dword s43, s[4:5], s20 offset:-0x1"], [43], 6, [0]),
    ("sload_soffc_imm_m20", ["s_load_dword s44, s[4:5], s20 offset:-0x20"], [44], 0xC, [-0x14]),
    ("sload_x2_soff4_m10",  ["s_load_dwordx2 s[46:47], s[4:5], s20 offset:-0x10"], [46, 47], 4, [-0xC, -8]),
    ("sbuf_imm_m4",         sbuf_raw(48, 12, 0x7D, -4), [48], None, None),     # soff 0x7d = null
    ("sbuf_soff8_imm_m4",   sbuf_raw(49, 12, 20, -4), [49], 8, None),
    ("sbuf_soffm4_imm8",    ["s_buffer_load_dword s50, s[12:15], s20 offset:0x8"], [50], 0xFFFFFFFC, None),
    ("sbuf_soffm4",         ["s_buffer_load_dword s51, s[12:15], s20"], [51], 0xFFFFFFFC, None),
]


def translator_sbuf(soff, imm):
    """The translator's s_buffer_load rule: a 32-bit offset (IMM kept as written, sOFF as is), bound-checked."""
    off = ((soff or 0) & ~3) + (imm & ~3)
    off &= 0xFFFFFFFF
    return [off] if off + 4 <= NUM_RECORDS else [None]


def expected():
    """Per row, the values the translator's rule gives (list of ints or None for 0)."""
    out = {}
    imms = {"sbuf_imm_m4": -4 & 0x1FFFFF, "sbuf_soff8_imm_m4": -4 & 0x1FFFFF, "sbuf_soffm4_imm8": 8, "sbuf_soffm4": 0}
    for name, _lines, regs, soff, xs in ROWS:
        if xs is None:
            # the translator reads an s_buffer_load IMM as unsigned (llvm-mc's 20-bit field): -4 is 0x1ffffc
            xs = translator_sbuf(soff, imms[name])
        out[name] = [seed(x) if x is not None else None for x in xs]
    return out


def kernel_lines():
    lines = ["s_mov_b32 s8, s2", "s_mov_b32 s9, s3", f"s_mov_b32 s10, 0x10000", f"s_mov_b32 s11, 0x{WORD3:08x}"]
    for i in range(0, 64, 4):                         # seed bytes 0x100..0x1ff
        for k in range(4):
            lines.append(f"v_mov_b32 v{1 + k}, 0x{(SEED + 0x100 + 4 * (i + k) - BASE) & 0xFFFFFFFF:08x}")
        lines.append(f"buffer_store_dwordx4 v[1:4], off, s[8:11], 0 offset:{0x100 + 4 * i}")
    lines += ["s_waitcnt vmcnt(0)", "s_waitcnt_vscnt null, 0x0", "s_dcache_inv", "s_waitcnt lgkmcnt(0)",
              f"s_add_u32 s4, s2, 0x{BASE:x}", "s_addc_u32 s5, s3, 0",
              f"s_add_u32 s12, s2, 0x{BASE:x}", "s_addc_u32 s13, s3, 0",
              f"s_mov_b32 s14, 0x{NUM_RECORDS:x}", f"s_mov_b32 s15, 0x{WORD3:08x}"]
    order = []
    for name, ls, regs, soff, _xs in ROWS:
        if soff is not None:
            lines.append(f"s_mov_b32 s20, 0x{soff & 0xFFFFFFFF:x}")
        lines += ls
        lines.append("s_waitcnt lgkmcnt(0)")
        order += regs
    for i, r in enumerate(order):
        lines.append(f"v_mov_b32 v1, s{r}")
        lines.append(f"buffer_store_dword v1, off, s[8:11], 0 offset:{4 * i}")
    lines += ["s_waitcnt vmcnt(0)", "s_endpgm"]
    return lines, order


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--oracle", choices=("translator", "hw"), default="translator")
    a = ap.parse_args(argv)
    want = expected()
    if a.oracle == "translator":
        for name, vals in want.items():
            print(f"{name:22} translator {' '.join('0' if v is None else hex(v) for v in vals)}")
        return 0
    import asm
    import runner
    try:
        oracle = runner.open_oracle()
    except runner.Unavailable as e:
        print(f"hardware oracle unavailable: {e}")
        return 77
    try:
        lines, order = kernel_lines()
        n = len(order)
        got = {}
        for wave in (32, 64):
            res = oracle.run([asm.Kernel(lines, wave=wave, threads=1, words=n, raw=True, vgprs=8, sgprs=64)])[0]
            if isinstance(res, str):
                print(f"wave{wave}: kernel rejected: {res}")
                return 2
            got[wave] = res
        print(f"device: {oracle.device} ({oracle.arch})")
    finally:
        oracle.close()
    bad, i = 0, 0
    for name, _ls, regs, _soff, _xs in ROWS:
        w32 = got[32][i:i + len(regs)]
        w64 = got[64][i:i + len(regs)]
        i += len(regs)
        exp = [0 if v is None else v for v in want[name]]
        ok = w32 == exp and w64 == exp
        bad += not ok
        hw = " ".join(hex(v) for v in w32) + ("" if w32 == w64 else "  (wave64 " + " ".join(hex(v) for v in w64) + ")")
        print(f"{name:22} hw {hw:26} translator {' '.join(hex(v) for v in exp):24} {'ok' if ok else 'DIFFERS'}")
    return 1 if bad else 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
