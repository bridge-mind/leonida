r"""SHADER-337: measure the y and z texel-offset fields of image_sample_lz_o on an AMD GPU (tools/hw_oracle).

    python -I tools/texoff_probe.py                # the rule's expectation only (no GPU)
    python -I tools/texoff_probe.py --oracle hw    # native AMD GPU, wave32 and wave64, compared with the rule

docs/design.md ("Texel offsets") has x measured on hardware (signed 6-bit field at bits [5:0], bits 6-7 ignored) and the
y and z signs only from a structural test. Here a raw kernel samples a linear R32_SFLOAT texture whose texel holds its
own index, with a point sampler that clamps to the last texel, so each result names the texel that was read:

    2D 64x64, texel (x, y) = 64 y + x:        y field = offset dword bits [13:8]
    3D 64x1x64, texel (x, 0, z) = 64 z + x:   z field = offset dword bits [21:16]

Each row starts at a texel and uses one offset dword. The values are chosen so that a signed and an unsigned reading
land on different texels (or one clamps), and some rows set the bits just above a field to see whether they are
ignored. The rule checked is the translator's: every field is sign-extended from 6 bits, the bits above it ignored
(tests/test_texel_offsets.py). Standard library only; the texture is built here, no game files.

Exit codes: 0 every row matches the rule, 1 a mismatch, 2 harness error, 77 no hardware oracle.
Usage: `python -I tools/texoff_probe.py --help` lists the options.
"""
import argparse
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "hw_oracle"))

W = 64
H2D = 64
D3D = 64
SQ_RSRC_IMG_3D = 10
CLAMP_LAST_TEXEL = 2


def sext6(v):
    v &= 0x3F
    return v - 64 if v & 0x20 else v


def clamp(v, n):
    return max(0, min(n - 1, v))


# (name, dim, start (x, y_or_z), offset dword)
ROWS = [
    # y field, 2D: from row 40 (0x20 signed is -32 -> 8, unsigned 32 -> clamps to 63; 0x3c is -4 -> 36 vs 60)
    ("y_0x20_from40", "2d", (5, 40), 0x20 << 8),
    ("y_0x3c_from40", "2d", (5, 40), 0x3C << 8),
    ("y_0x1f_from10", "2d", (5, 10), 0x1F << 8),     # +31 -> 41 either way (control)
    ("y_0x20_from10", "2d", (5, 10), 0x20 << 8),     # signed -32 clamps to 0, unsigned 32 -> 42
    ("y_0x21_from40", "2d", (5, 40), 0x21 << 8),     # -31 -> 9 vs 33 -> clamp 63
    ("y_bits14_15",   "2d", (5, 40), 0xC0 << 8),     # field 0 with bits 14-15 set: 40 if they are ignored
    ("y_0x3f_bit14",  "2d", (5, 40), 0x7F << 8),     # field 0x3f (-1) with bit 14: 39 if ignored
    ("y_with_x",      "2d", (20, 40), (0x3C << 8) | 0x3E),   # x -2, y -4 together
    # z field, 3D: same values on bits [21:16]
    ("z_0x20_from40", "3d", (5, 40), 0x20 << 16),
    ("z_0x3c_from40", "3d", (5, 40), 0x3C << 16),
    ("z_0x1f_from10", "3d", (5, 10), 0x1F << 16),
    ("z_0x20_from10", "3d", (5, 10), 0x20 << 16),
    ("z_0x21_from40", "3d", (5, 40), 0x21 << 16),
    ("z_bits22_23",   "3d", (5, 40), 0xC0 << 16),
    ("z_with_x_y",    "3d", (20, 40), (0x3C << 16) | (0x3F << 8) | 0x3E),   # y -1 on a 1-texel-high image clamps
]


def expected(dim, start, off):
    x0, b0 = start
    x = clamp(x0 + sext6(off), W)
    if dim == "2d":
        y = clamp(b0 + sext6(off >> 8), H2D)
        return float(y * W + x)
    z = clamp(b0 + sext6(off >> 16), D3D)
    return float(z * W + x)                          # y is clamped to the single row


def texture(dim):
    n = W * (H2D if dim == "2d" else D3D)
    return b"".join(struct.pack("<f", float(i)) for i in range(n))


def words(arch, dim):
    import tsharp
    if dim == "2d":
        return tsharp.image_words(arch, "R32_SFLOAT", W, H2D, 4)
    t = tsharp.image_words(arch, "R32_SFLOAT", W, 1, 4)
    t[3] = (t[3] & ~(0xF << 28)) | (SQ_RSRC_IMG_3D << 28)
    t[4] = D3D - 1                                   # 3D: DEPTH = depth - 1; a 256-byte row needs no custom pitch
    return t


def sampler_words():
    return [CLAMP_LAST_TEXEL | (CLAMP_LAST_TEXEL << 3) | (CLAMP_LAST_TEXEL << 6), 0, 0, 0]   # point, no mips


def kernel(arch, dim, rows):
    import tsharp
    t, s = words(arch, dim), sampler_words()
    uid = dim
    out = ["s_getpc_b64 s[4:5]", f".Lpc{uid}:", f"s_add_u32 s4, s4, .Ltex{uid}-.Lpc{uid}", "s_addc_u32 s5, s5, 0",
           "s_and_b32 s6, s4, 255", "s_lshr_b64 s[4:5], s[4:5], 8", "s_mov_b32 s8, s4", "s_and_b32 s9, s5, 255",
           f"s_or_b32 s9, s9, 0x{t[1]:08x}"]
    out += [f"s_mov_b32 s{8 + i}, 0x{t[i]:08x}" for i in range(2, 8)]
    out += [f"s_mov_b32 s{16 + i}, 0x{s[i]:08x}" for i in range(4)]
    depth = H2D if dim == "2d" else D3D
    for i, (_name, _d, (x0, b0), off) in enumerate(rows):
        u, v = (x0 + 0.5) / W, (b0 + 0.5) / depth
        out += [f"v_mov_b32 v1, 0x{off:08x}", f"v_mov_b32 v2, 0x{struct.unpack('<I', struct.pack('<f', u))[0]:08x}"]
        if dim == "2d":
            out += [f"v_mov_b32 v3, 0x{struct.unpack('<I', struct.pack('<f', v))[0]:08x}",
                    "image_sample_lz_o v8, v[1:3], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D"]
        else:
            out += ["v_mov_b32 v3, 0x3f000000",                                      # t = 0.5 (the one row)
                    f"v_mov_b32 v4, 0x{struct.unpack('<I', struct.pack('<f', v))[0]:08x}",
                    "image_sample_lz_o v8, v[1:4], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_3D"]
        out += ["s_waitcnt vmcnt(0)", "s_cmp_lg_u32 s6, 0", f"s_cbranch_scc0 .Lok{uid}{i}",
                f"v_mov_b32 v8, 0x{tsharp.POISON:08x}", f".Lok{uid}{i}:",
                f"v_mov_b32 v5, {4 * i}", "global_store_b32 v5, v8, s[2:3]"]
    out += ["s_waitcnt vmcnt(0)", "s_endpgm", ".p2align 8", f".Ltex{uid}:"]
    return out + tsharp._longs(texture(dim))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--oracle", choices=("rule", "hw"), default="rule")
    a = ap.parse_args(argv)
    if a.oracle == "rule":
        for name, dim, start, off in ROWS:
            print(f"{name:16} offset 0x{off:06x} rule {expected(dim, start, off):.0f}")
        return 0
    import asm
    import runner
    try:
        oracle = runner.open_oracle()
    except runner.Unavailable as e:
        print(f"hardware oracle unavailable: {e}")
        return 77
    try:
        print(f"device: {oracle.device}")
        bad = 0
        for dim in ("2d", "3d"):
            rows = [r for r in ROWS if r[1] == dim]
            got = {}
            for wave in (32, 64):
                res = oracle.run([asm.Kernel(kernel(oracle.arch, dim, rows), wave=wave, threads=1, words=len(rows),
                                             raw=True, vgprs=16, sgprs=24)])[0]
                if isinstance(res, str):
                    print(f"{dim} wave{wave}: kernel rejected: {res}")
                    return 2
                got[wave] = [struct.unpack("<f", struct.pack("<I", w))[0] for w in res]
            for i, (name, _d, start, off) in enumerate(rows):
                want = expected(dim, start, off)
                g32, g64 = got[32][i], got[64][i]
                ok = g32 == want and g64 == want
                bad += not ok
                extra = "" if g32 == g64 else f" (wave64 {g64:.0f})"
                print(f"{name:16} offset 0x{off:06x} hw {g32:6.0f}{extra:14} rule {want:6.0f}  {'ok' if ok else 'DIFFERS'}")
        return 1 if bad else 0
    finally:
        oracle.close()


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
