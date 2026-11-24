r"""Measure the triangle step of image_bvh_intersect_ray / image_bvh64_intersect_ray natively on RDNA2 (node types 0-3).

Builds the kernels of the RX 6800 measurement behind docs/design.md's Ray tracing row (and the rows of
tests/semantic/test_image_bvh_tri_hw.py): 22 raw kernels for tools/hw_oracle, 564 single-lane cases and 7 cases on 64
lanes (the 14 kernels of #4202, then the 8 tie kernels tri_t1..tri_t8: rays exactly on an edge or a vertex with x, y
or z dominant, both windings and both signs of d[kz], and equal |d| components). Each kernel writes hand-built 64-byte triangle nodes into the oracle's buffer and traces every case in both
forms: image_bvh_intersect_ray (32-bit node id, descriptor base = the node region) and image_bvh64_intersect_ray
(descriptor base 0, the 64-bit address as the id). The four words of each are compared with the measured rule
(rule_words below), with the translator (tools/rdna2spirv.py run on tests/semantic/spirv_mini_interp.py) and with
each other. Hand-built nodes and rays only; nothing here reads game files. Standard library only.

Usage:
    python -I tools/bvh_tri_probe.py                          # translator only (CPU): every case against the rule
    python -I tools/bvh_tri_probe.py --oracle hw              # the AMD GPU only, against the rule
    python -I tools/bvh_tri_probe.py --oracle translator,hw   # both, word for word, and the rule
    python -I tools/bvh_tri_probe.py --kernel tri_g1a         # only the named kernels (repeat --kernel)
    python -I tools/bvh_tri_probe.py --list                   # kernels and their cases
    python -I tools/bvh_tri_probe.py --write-kernels DIR      # every kernel's lines (.s) and cases (.json); no GPU
    python -I tools/bvh_tri_probe.py --json out.json          # every case's words
    python -I tools/bvh_tri_probe.py --self-test              # generator, safety checks and rule; no GPU

What a kernel may do on the GPU (the rules the measurement's kernels were reviewed under):
  * one kernel per dispatch (one oracle.run call each), one workgroup of 1 or 64 threads;
  * every node is placed by Kernel.add_node in whole 64-byte slots of the node region inside the oracle's own buffer
    (single-lane: +0x800..+0x1000 of 8 KiB; lane kernels: +0x4800..+0x5000 of 32 KiB), and a case names a slot, never
    a free-form id. A 4 KiB (12 KiB) guard band follows the node region and is never written;
  * the kernel checks at run time that the node region is 256-byte aligned (the 32-bit form's descriptor base is
    in 256-byte units); if not, it writes 0xBAADF00D to control word 3 and runs no BVH instruction;
  * node types 0..3 only: no box node (whose child ids would name memory), no type 6/7 (unknown semantics), no box
    extent rows; one descriptor, the one #3567 measured boxes with (w1 bit 31, w2 = 0xFFFFFFFF, w3 = 0x81000000);
  * results are checked for completeness: a control word written after the last case, no unwritten destination
    sentinel (0xBEEF0ijk), and no output that is all the launcher's prefill.

Assembler and launcher: tools/hw_oracle's runner, found as it finds them (LEONIDA_AMD_CLANG, LEONIDA_HIP_DLL,
LEONIDA_HW_ARCH; see tools/hw_oracle/runner.py and docs/shader_validation.md); without them --oracle hw exits 77.
The runner's constructor first runs its own three probe kernels (a v_mov kernel, and two 64-thread kernels that
store their lane ids into the oracle's buffer). The measurement (Windows 11, AMD 26.11.2, amdhip64_6.dll) had no
Windows AMD clang. It ran the runner's own two command lines inside WSL, Ubuntu's clang-21 and ld.lld-21, with each
Windows path passed as /mnt/<drive>/... (through a subclass of the runner, not through this tool):
    clang-21 --target=amdgcn-amd-amdhsa -mcpu=gfx1030 [-mwavefrontsize64] -mcode-object-version=5 -c k.s -o k.o
    ld.lld-21 -shared k.o -o k.co
amdhip64_6.dll refuses code object v6 (HIP error 209; the runner's probe falls back to v5 by itself). Doing the same
with this tool needs LEONIDA_AMD_CLANG to name a wrapper that runs the clang line that way, an ld.lld wrapper next to
it or on PATH, and LEONIDA_HW_ARCH=gfx1030, since no amdgpu-arch sits beside such a wrapper; that route was not tried
with this tool. The DRM launcher on Linux reads back at most 4096 dwords, so the two lane kernels (8192) are skipped
there.

Exit codes: 0 every compared word matched (and the two forms agreed); 1 a mismatch; 2 a CLI or harness error; 77 the
hardware oracle was asked for but is unavailable, or the GPU is not GFX10.
"""
import argparse
import gc
import hashlib
import json
import math
import os
import struct
import sys
import tempfile
import warnings
import zlib
from fractions import Fraction as Fr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.dont_write_bytecode = True
for _p in (HERE, os.path.join(ROOT, "tests", "semantic")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

INF, NAN = float("inf"), float("nan")
QNAN = 0x7FC00000
MISS = 0xFFFFFFFF
POISON = 0xBAADF00D
DONE = 0x600D0000
SENT = 0xBEEF0000
PREFILL = 0xDEAD0000
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)

# ---------------------------------------------------------------------------------------------------------------------
# the kernels (port of the measurement's lib/bvhk.py; the lines are the same, character for character)
# ---------------------------------------------------------------------------------------------------------------------
# Buffer layout (byte offsets from the oracle buffer start B = s[2:3]):
#   single-lane kernels (threads 1)                 lane kernels (threads 64)
#   [0x000, 0x400)  results: case i at 32*i          [0x0000, 0x4000) results: lane l, case i at 256*l + 32*i,
#                   (+0 bvh32 words, +16 bvh64)                        lane control at 256*l + 224
#   [0x400, 0x410)  control                          [0x4000, 0x4010) control
#   [0x800, 0x1000) node region, 32 slots of 64 B    [0x4800, 0x5000) node region
#   [0x1000, 0x2000) guard, never written            [0x5000, 0x8000) guard, never written
# Control words: [0] node region VA low, [1] high, [2] 0x600D0000 | uid once every case has run, [3] 0xBAADF00D if
# the node region was not 256-aligned (then no BVH instruction runs at all).
LAYOUT = {
    1: dict(res=0x000, ctl=0x400, node=0x800, node_end=0x1000, words=2048),
    64: dict(res=0x0000, ctl=0x4000, node=0x4800, node_end=0x5000, words=8192),
}
SLOT = 64
NSLOTS = 32
NODE_BYTES = 64                                 # a triangle node: 16 dwords
DESC = dict(w1=0x80000000, w2=0xFFFFFFFF, w3=0x81000000)


def f2u(x):
    if math.isnan(x):
        return 0x7FC00000 if math.copysign(1.0, x) > 0 else 0xFFC00000
    try:
        return struct.unpack("<I", struct.pack("<f", x))[0]
    except OverflowError:
        return 0x7F800000 if x > 0 else 0xFF800000


def u2f(w):
    return struct.unpack("<f", struct.pack("<I", w & 0xFFFFFFFF))[0]


def ray_words(tmax, org, d, inv=None):
    """[tmax, o.xyz, d.xyz, inv.xyz] as u32. inv defaults to 1/d with +-inf for a (signed) zero."""
    if inv is None:
        inv = [(INF if math.copysign(1.0, c) > 0 else -INF) if c == 0 else 1.0 / c for c in d]
    return [v & 0xFFFFFFFF if isinstance(v, int) else f2u(v) for v in (tmax, *org, *d, *inv)]


class Kernel:
    """The nodes and cases of one kernel (one dispatch), and its lines."""

    def __init__(self, label, threads=1, wave=32, nsa=False, lane_offsets=None):
        if threads not in LAYOUT:
            raise ValueError("threads must be 1 or 64")
        self.uid = zlib.crc32(label.encode("utf-8")) & 0xFFFF
        self.label, self.threads, self.wave, self.nsa = label, threads, wave, nsa
        self.L = LAYOUT[threads]
        self.lane_offsets = lane_offsets          # (dx, dy): org.x += (lane & 15) dx, org.y += (lane >> 4) dy
        self.slots = {}                           # slot -> 16 u32 words
        self.cases = []
        self.max_cases = 7 if threads == 64 else 32

    def add_node(self, words):
        """Place a 16-word node in the next free 64-byte slot of the node region; returns the slot."""
        if len(words) != 16:
            raise ValueError("a triangle node is 16 words")
        for s in range(NSLOTS):
            if s not in self.slots:
                self.slots[s] = [w & 0xFFFFFFFF for w in words]
                return s
        raise ValueError(f"kernel {self.label}: no free node slot")

    def add_case(self, label, slot, ntype, ray, meta=None):
        if not 0 <= ntype <= 3:
            raise ValueError(f"case {label}: node type {ntype}; this probe runs triangle nodes (0..3) only")
        if slot not in self.slots:
            raise ValueError(f"case {label}: slot {slot} holds no node")
        if len(self.cases) >= self.max_cases:
            raise ValueError(f"kernel {self.label}: more than {self.max_cases} cases")
        if len(ray) != 10:
            raise ValueError("ray is 10 words")
        self.cases.append(dict(label=label, slot=slot, type=ntype, ray=[w & 0xFFFFFFFF for w in ray], meta=meta or {}))
        return len(self.cases) - 1

    def node_off(self, slot):
        return self.L["node"] + SLOT * slot

    def node32(self, case):
        return (SLOT * case["slot"]) >> 3 | case["type"]

    def lines(self):
        L, u = self.L, self.uid
        w1 = DESC["w1"]
        out = [f"; kernel {u}: {self.label}",
               f"s_add_u32 s20, s2, 0x{L['node']:x}", "s_addc_u32 s21, s3, 0",        # node region VA
               f"s_add_u32 s24, s2, 0x{L['ctl']:x}", "s_addc_u32 s25, s3, 0",         # control block VA
               "v_mov_b32 v1, 0"]
        if self.threads == 64:
            out += ["v_lshlrev_b32 v1, 8, v0"]                                         # lane l -> 256 * l
        out += ["v_mov_b32 v16, s20", "v_mov_b32 v17, s21", "v_mov_b32 v18, 0", "v_mov_b32 v19, 0",
                "v_mov_b32 v13, 0",
                "global_store_dwordx4 v13, v[16:19], s[24:25]",
                "s_and_b32 s22, s20, 0xff", "s_cmp_lg_u32 s22, 0", f"s_cbranch_scc1 .Lbad{u}"]
        for s in sorted(self.slots):
            ws = self.slots[s]
            out.append(f"; node slot {s} at +0x{self.node_off(s):x}, {len(ws) * 4} bytes")
            for q in range(0, len(ws), 4):
                out += [f"v_mov_b32 v{16 + j}, 0x{ws[q + j]:08x}" for j in range(4)]
                out.append(f"global_store_dwordx4 v13, v[16:19], s[20:21] offset:{SLOT * s + 4 * q}")
        out += ["s_waitcnt_vscnt null, 0x0", "buffer_gl0_inv", "buffer_gl1_inv"]
        # descriptors: 32-bit form base = node region VA >> 8; 64-bit form base 0
        out += ["s_lshr_b64 s[14:15], s[20:21], 8", "s_mov_b32 s4, s14", "s_and_b32 s5, s15, 0xff",
                f"s_or_b32 s5, s5, 0x{w1:08x}", f"s_mov_b32 s6, 0x{DESC['w2']:08x}",
                f"s_mov_b32 s7, 0x{DESC['w3']:08x}",
                "s_mov_b32 s8, 0", f"s_mov_b32 s9, 0x{w1:08x}", f"s_mov_b32 s10, 0x{DESC['w2']:08x}",
                f"s_mov_b32 s11, 0x{DESC['w3']:08x}"]
        if self.threads == 64:
            out += ["v_mbcnt_lo_u32_b32 v14, -1, 0"]                                 # lane id as the wave sees it
            if self.wave == 64:
                out += ["v_mbcnt_hi_u32_b32 v14, -1, v14"]
        for i, c in enumerate(self.cases):
            out += self._case_lines(i, c)
        if self.threads == 64:
            out += ["v_mov_b32 v16, v14", "v_mov_b32 v17, v0", f"v_mov_b32 v18, 0x{DONE | u:08x}",
                    "v_mov_b32 v19, 0", "global_store_dwordx4 v1, v[16:19], s[2:3] offset:224"]
        out += ["v_mov_b32 v13, 0", f"v_mov_b32 v16, 0x{DONE | u:08x}",
                "global_store_dword v13, v16, s[24:25] offset:8",
                f"s_branch .Lend{u}",
                f".Lbad{u}:",
                "v_mov_b32 v13, 0", f"v_mov_b32 v16, 0x{POISON:08x}",
                "global_store_dword v13, v16, s[24:25] offset:12",
                f".Lend{u}:",
                "s_waitcnt_vscnt null, 0x0",
                "s_endpgm"]
        return out

    def _case_lines(self, i, c):
        out = [f"; case {i}: {c['label']}  (slot {c['slot']}, type {c['type']})"]
        out += [f"v_mov_b32 v{40 + j}, 0x{SENT | (i << 4) | j:08x}" for j in range(8)]
        r = c["ray"]
        out.append(f"v_mov_b32 v2, 0x{self.node32(c):x}")
        out += [f"v_mov_b32 v{3 + j}, 0x{r[j]:08x}" for j in range(10)]
        if self.lane_offsets:
            dx, dy = self.lane_offsets
            out += ["v_and_b32 v13, 15, v0", "v_cvt_f32_u32 v13, v13", f"v_fmac_f32 v4, 0x{f2u(dx):08x}, v13",
                    "v_lshrrev_b32 v13, 4, v0", "v_cvt_f32_u32 v13, v13", f"v_fmac_f32 v5, 0x{f2u(dy):08x}, v13"]
        a = "[" + ", ".join(f"v{2 + j}" for j in range(11)) + "]" if self.nsa else "v[2:12]"
        out.append(f"image_bvh_intersect_ray v[40:43], {a}, s[4:7]")
        off = SLOT * c["slot"]
        out += [f"s_add_u32 s12, s20, 0x{off:x}", "s_addc_u32 s13, s21, 0", "s_lshr_b64 s[12:13], s[12:13], 3",
                f"s_or_b32 s12, s12, 0x{c['type']:x}", "v_mov_b32 v24, s12", "v_mov_b32 v25, s13"]
        out += [f"v_mov_b32 v{26 + j}, v{3 + j}" for j in range(10)]
        a = "[" + ", ".join(f"v{24 + j}" for j in range(12)) + "]" if self.nsa else "v[24:35]"
        out.append(f"image_bvh64_intersect_ray v[44:47], {a}, s[8:11]")
        out += ["s_waitcnt vmcnt(0)",
                f"global_store_dwordx4 v1, v[40:43], s[2:3] offset:{32 * i}",
                f"global_store_dwordx4 v1, v[44:47], s[2:3] offset:{32 * i + 16}"]
        return out

    def program(self, asm):
        """The hw_oracle program (asm is tools/hw_oracle/asm.py)."""
        return asm.Kernel(self.lines(), wave=self.wave, threads=self.threads, words=self.L["words"], raw=True,
                          vgprs=48, sgprs=40)

    def lane_ray(self, case, lane):
        """The ray lane `lane` traces: the kernel's v_fmac_f32 of the lane offsets (exact for these values)."""
        r = list(case["ray"])
        if self.lane_offsets:
            dx, dy = self.lane_offsets
            r[1] = f2u(u2f(r[1]) + dx * (lane & 15))
            r[2] = f2u(u2f(r[2]) + dy * (lane >> 4))
        return r


def check_kernel(k):
    """The memory rules above, from the kernel's bookkeeping and its lines; raises ValueError on a violation."""
    L = k.L
    if not (L["node_end"] - L["node"] == NSLOTS * SLOT and L["node_end"] < 4 * L["words"] and L["node"] % 256 == 0):
        raise ValueError(f"{k.label}: layout")
    for s, ws in k.slots.items():
        if not (0 <= s < NSLOTS and len(ws) * 4 == NODE_BYTES):
            raise ValueError(f"{k.label}: node slot {s}")
    for c in k.cases:
        if c["slot"] not in k.slots or not 0 <= c["type"] <= 3:
            raise ValueError(f"{k.label}: case {c['label']}")
    # results: case i at 32 i (+16 for the 64-bit form); lane kernels give each lane 256 bytes, its control at +224
    end = 256 * k.threads if k.threads == 64 else 32 * len(k.cases)
    if end > L["ctl"] - L["res"] or (k.threads == 64 and 32 * len(k.cases) > 224):
        raise ValueError(f"{k.label}: results overlap the control block")
    lines = k.lines()
    bvh = [ln for ln in lines if ln.startswith("image_bvh")]
    if len(bvh) != 2 * len(k.cases):
        raise ValueError(f"{k.label}: {len(bvh)} BVH instructions for {len(k.cases)} cases")
    guard = lines.index(f"s_cbranch_scc1 .Lbad{k.uid}")
    if lines[guard - 2:guard] != ["s_and_b32 s22, s20, 0xff", "s_cmp_lg_u32 s22, 0"] or \
            guard > lines.index(next(ln for ln in lines if ln.startswith("image_bvh"))):
        raise ValueError(f"{k.label}: no alignment guard before the first BVH instruction")
    for ln in lines:
        if ln.startswith(("s_mov_b32 s6", "s_mov_b32 s10")) and not ln.endswith(f"0x{DESC['w2']:08x}"):
            raise ValueError(f"{k.label}: descriptor w2 `{ln}`")
        if ln.startswith(("s_mov_b32 s7", "s_mov_b32 s11")) and not ln.endswith(f"0x{DESC['w3']:08x}"):
            raise ValueError(f"{k.label}: descriptor w3 `{ln}`")
        if ln.startswith("global_store") and ln.split()[-1].startswith("offset:"):
            off = int(ln.split(":")[-1])
            base = ln.split(", ")[2].split()[0]
            hi = {"s[20:21]": L["node"] + off + 16, "s[24:25]": L["ctl"] + off + 4,
                  "s[2:3]": L["res"] + off + 16 + (k.threads - 1) * 256 * (k.threads == 64)}[base]
            limit = {"s[20:21]": L["node_end"], "s[24:25]": L["ctl"] + 16, "s[2:3]": L["ctl"]}[base]
            if hi > limit:
                raise ValueError(f"{k.label}: store past its region `{ln}`")
    return True


def results(k, output):
    """Per case {hw32, hw64} (lane kernels: a list of 64), and the control words."""
    L = k.L
    res = []
    for i, c in enumerate(k.cases):
        if k.threads == 1:
            b = (L["res"] + 32 * i) // 4
            res.append(dict(hw32=output[b:b + 4], hw64=output[b + 4:b + 8]))
        else:
            lanes = []
            for lane in range(k.threads):
                b = (L["res"] + 256 * lane + 32 * i) // 4
                lanes.append(dict(hw32=output[b:b + 4], hw64=output[b + 4:b + 8]))
            res.append(dict(lanes=lanes))
    return res, output[L["ctl"] // 4:L["ctl"] // 4 + 4]


def output_problem(k, output):
    """Why an output cannot be trusted, or None."""
    if not isinstance(output, list) or len(output) < k.L["words"]:
        return f"no output ({output!r:.200})"
    if all(w == PREFILL for w in output):
        return "the output is all prefill: the kernel did not run"
    res, ctl = results(k, output)
    if ctl[3] == POISON:
        return "the node region was not 256-byte aligned: no BVH instruction ran"
    if ctl[2] != DONE | k.uid:
        return f"control word 2 is 0x{ctl[2]:08x}, not 0x{DONE | k.uid:08x}: the kernel did not reach its end"
    for r in res:
        for x in r.get("lanes", [r]):
            if any(w & 0xFFFF0000 == SENT for w in x["hw32"] + x["hw64"]):
                return "a destination register was never written"
    return None


# ---------------------------------------------------------------------------------------------------------------------
# the cases (port of the measurement's tri/gen.py, the kernels that ran)
# ---------------------------------------------------------------------------------------------------------------------
# A closed fan of the four triangle node types around slot 2, all wound the same way, in the plane z = 3
FAN = [(0, 0, 3), (4, 0, 3), (0, 2, 3), (4, 2, 3), (-2, 3, 3)]
VSLOTS = {0: (0, 1, 2), 1: (1, 3, 2), 2: (2, 3, 4), 3: (2, 4, 0)}
REMAP = 0x09090909                              # every type: (V, W), what a builder writes
INSIDE = {0: (1, 0.25), 1: (3.5, 1.25), 2: (1.75, 2.125), 3: (-0.5, 2.0)}   # interior points, distinct weights
OUTSIDE = {0: (-0.25, 0.75, 0.5), 1: (0.75, -0.25, 0.5), 2: (0.75, 0.5, -0.25)}   # beyond the edge opposite vertex i
D_POS, D_NEG = (0.0, 0.0, 0.5), (0.0, 0.0, -0.5)
D_OBL, D_OBLN = (0.125, 0.0625, 0.5), (0.125, 0.0625, -0.5)
PX, PY = (2, 0, 1), (1, 2, 0)                   # cyclic: the dominant axis becomes x / y, orientation kept


def leaf(slots=FAN, remap=REMAP):
    """16 node words: vertex slots 0..3, slot 4 in words 12..14, the remap bytes in word 15."""
    w = [f2u(float(c)) for s in slots for c in s]
    if len(w) != 15:
        raise ValueError("five vertex slots")
    return w + [remap]


def bary(tri, p):
    (ax, ay), (bx, by), (cx, cy) = [(Fr(v[0]), Fr(v[1])) for v in tri]
    x, y = Fr(p[0]), Fr(p[1])
    det = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)
    b1 = ((x - ax) * (cy - ay) - (cx - ax) * (y - ay)) / det
    b2 = ((bx - ax) * (y - ay) - (x - ax) * (by - ay)) / det
    return (1 - b1 - b2, b1, b2)


def ray_at(p, d, z=3.0, t=8.0, tmax=100.0, perm=None, inv=None):
    """A ray that reaches (p, z) at parameter t along d. perm: cyclic axis permutation applied to the ray."""
    o = [p[0] - t * d[0], p[1] - t * d[1], z - t * d[2]]
    dd = list(d)
    if perm:
        o, dd = [o[i] for i in perm], [dd[i] for i in perm]
    return ray_words(tmax, o, dd, inv)


def permuted(slots, perm):
    return [tuple(s[i] for i in perm) for s in slots]


def mirror(slots):
    return [(-s[0], s[1], s[2]) for s in slots]


def outside_point(t, i, slots=FAN):
    P = [slots[s][:2] for s in VSLOTS[t]]
    b = OUTSIDE[i]
    return (sum(b[j] * P[j][0] for j in range(3)), sum(b[j] * P[j][1] for j in range(3)))


def prev_f(x):
    return u2f(f2u(x) - 1)


def next_f(x):
    return u2f(f2u(x) + 1)


def g1(variant="", **kw):
    """Slots and node types, both facings, axis and oblique rays, the cs_bvh triangle."""
    k = Kernel("tri_g1a" + variant, **kw)
    s_fan = k.add_node(leaf())
    s_mir = k.add_node(leaf(mirror(FAN)))
    for t in range(4):
        k.add_case(f"TRI{t} hit +z", s_fan, t, ray_at(INSIDE[t], D_POS))
        k.add_case(f"TRI{t} hit -z (back)", s_fan, t, ray_at(INSIDE[t], D_NEG))
    for t in range(4):
        k.add_case(f"TRI{t} miss (+z at TRI{(t + 1) % 4}'s point)", s_fan, t, ray_at(INSIDE[(t + 1) % 4], D_POS))
    for t in range(4):
        p = INSIDE[t]
        k.add_case(f"TRI{t} mirrored leaf hit +z", s_mir, t, ray_at((-p[0], p[1]), D_POS))
    for t in range(4):
        k.add_case(f"TRI{t} hit oblique +z", s_fan, t, ray_at(INSIDE[t], D_OBL))
        k.add_case(f"TRI{t} hit oblique -z", s_fan, t, ray_at(INSIDE[t], D_OBLN))
    # runtime/test/vk_compute_test.cpp "bvh" scene, triangle node 0x40: z = 10 triangle, the heap's 0xEE fill elsewhere
    cs = [f2u(c) for c in (0, 0, 10, 1, 0, 10, 0, 1, 10)] + [0xEEEEEEEE] * 6 + [0x09]
    k.add_case("cs_bvh scene triangle node 0x40", k.add_node(cs), 0,
               ray_words(100.0, (0.25, 0.25, 0.0), (0.0, 0.0, 1.0), (INF, INF, 1.0)))
    return k


def g1b():
    """Dominant axis x and y, kz ties, mirrored leaf along -z."""
    k = Kernel("tri_g1b")
    for perm, nm in ((PX, "x"), (PY, "y")):
        s = k.add_node(leaf(permuted(FAN, perm)))
        for t in (0, 2):
            k.add_case(f"TRI{t} dominant +{nm}", s, t, ray_at(INSIDE[t], D_POS, perm=perm))
            k.add_case(f"TRI{t} dominant -{nm}", s, t, ray_at(INSIDE[t], D_NEG, perm=perm))
    s = k.add_node(leaf())
    p = INSIDE[0]
    for d, nm in (((0.5, 0.0, 0.5), "|dx|=|dz|"), ((0.0, 0.5, 0.5), "|dy|=|dz|"), ((0.5, 0.5, 0.5), "|dx|=|dy|=|dz|")):
        k.add_case(f"TRI0 kz tie {nm}", s, 0, ray_at(p, d))
    k.add_case("TRI0 kz tie |dx|=|dy|>|dz|", s, 0, ray_at(p, (0.5, 0.5, 0.25), t=16.0))
    s_m = k.add_node(leaf(mirror(FAN)))
    for t in range(4):
        q = INSIDE[t]
        k.add_case(f"TRI{t} mirrored leaf hit -z", s_m, t, ray_at((-q[0], q[1]), D_NEG))
    return k


def g1c():
    """Misses with det < 0 and det > 0 for every type, outside by each of the three edges."""
    k = Kernel("tri_g1c")
    s_fan = k.add_node(leaf())
    s_mir = k.add_node(leaf(mirror(FAN)))
    k.add_case("control: TRI0 hit +z", s_fan, 0, ray_at(INSIDE[0], D_POS))
    k.add_case("control: TRI0 hit -z (back)", s_fan, 0, ray_at(INSIDE[0], D_NEG))
    for t in range(4):
        for i in range(3):
            p = outside_point(t, i)
            for d, nm in ((D_POS, "+z (det<0)"), (D_NEG, "-z (det>0)")):
                k.add_case(f"TRI{t} miss beyond edge opposite v{i} {nm}", s_fan, t, ray_at(p, d))
        p = outside_point(t, 0, mirror(FAN))
        k.add_case(f"TRI{t} mirrored leaf miss beyond edge opposite v0 +z (det>0)", s_mir, t, ray_at(p, D_POS))
    return k


def g2():
    """Remap byte: all 16 low nibbles per node type (the other three bytes 0x06), high nibbles on TRI0 both facings."""
    ks = []
    for half in (0, 1):
        k = Kernel(f"tri_g2{'ab'[half]}")
        for t in (2 * half, 2 * half + 1):
            for n in range(16):
                rm = sum((n if b == t else 0x06) << (8 * b) for b in range(4))
                k.add_case(f"TRI{t} byte{t}=0x{n:02x} others 0x06", k.add_node(leaf(remap=rm)), t,
                           ray_at(INSIDE[t], D_POS), meta=dict(byte=n))
        ks.append(k)
    k = Kernel("tri_g2c")
    for h in range(16):
        rm = ((h << 4) | 0x09) | (0x06 << 8) | (0x06 << 16) | (0x06 << 24)
        s = k.add_node(leaf(remap=rm))
        k.add_case(f"TRI0 byte0=0x{(h << 4) | 9:02x} +z", s, 0, ray_at(INSIDE[0], D_POS), meta=dict(byte=(h << 4) | 9))
        k.add_case(f"TRI0 byte0=0x{(h << 4) | 9:02x} -z (back)", s, 0, ray_at(INSIDE[0], D_NEG),
                   meta=dict(byte=(h << 4) | 9))
    ks.append(k)
    return ks


def g3():
    """t max, the sign of t, zero area, NaN and inf in the node (tri_g3a); the ray's own NaN, inf, zero and
    inconsistent inverse direction (tri_g3b)."""
    p = INSIDE[0]
    k = Kernel("tri_g3a")
    s = k.add_node(leaf())
    for tm, nm in ((100.0, "100 (control)"), (8.0, "8 = t"), (prev_f(8.0), "prev(8)"), (next_f(8.0), "next(8)"),
                   (7.0, "7"), (0.0, "0"), (-1.0, "-1"), (INF, "+inf")):
        k.add_case(f"TRI0 t=8 tmax={nm}", s, 0, ray_at(p, D_POS, tmax=tm))
    r = ray_at(p, D_POS)
    r[0] = QNAN
    k.add_case("TRI0 t=8 tmax=NaN", s, 0, r)
    k.add_case("TRI0 behind origin +z (t=-4)", s, 0, ray_at(p, D_POS, t=-4.0))
    k.add_case("TRI0 behind origin -z (t=-4, back)", s, 0, ray_at(p, D_NEG, t=-4.0))
    k.add_case("TRI0 origin on the plane (t=0)", s, 0, ray_at(p, D_POS, t=0.0))
    k.add_case("TRI0 t=2^-19", s, 0, ray_at(p, D_POS, t=2.0 ** -19))
    z = 3
    one = k.add_node(leaf([(1, 0.25, z)] * 3 + [FAN[3], FAN[4]]))
    k.add_case("zero area: three equal vertices, ray through them", one, 0, ray_at(p, D_POS))
    col = k.add_node(leaf([(0, 0.25, z), (2, 0.25, z), (4, 0.25, z), FAN[3], FAN[4]]))
    k.add_case("zero area: collinear, ray through the segment", col, 0, ray_at(p, D_POS))
    k.add_case("zero area: collinear, ray off the line", col, 0, ray_at((1, 1), D_POS))
    eon = k.add_node(leaf([(0, 0.25, 0), (4, 0.25, 0), (0, 0.25, 6), FAN[3], FAN[4]]))
    k.add_case("edge-on: ray in the triangle's plane, through it", eon, 0, ray_at(p, D_POS))
    for nm, vi, ci, val in (("v0.x NaN", 0, 0, QNAN), ("v0.z NaN", 0, 2, QNAN), ("v1.y NaN", 1, 1, QNAN),
                            ("v2.x +inf", 2, 0, 0x7F800000), ("v0.x -inf", 0, 0, 0xFF800000)):
        w = leaf()
        w[3 * vi + ci] = val
        k.add_case(f"TRI0 vertex {nm}", k.add_node(w), 0, ray_at(p, D_POS))
    w = leaf()
    w[9] = QNAN
    k.add_case("TRI0 NaN in slot 3 (not read by TRI0): control", k.add_node(w), 0, ray_at(p, D_POS))
    w = leaf()
    w[12] = QNAN
    k.add_case("TRI0 NaN in slot 4 word 12 (not read by TRI0): control", k.add_node(w), 0, ray_at(p, D_POS))

    k2 = Kernel("tri_g3b")
    s = k2.add_node(leaf())
    base = ray_at(p, D_POS)
    k2.add_case("TRI0 control", s, 0, base)

    def var(nm, idx, val):
        r = list(base)
        r[idx] = val & 0xFFFFFFFF
        k2.add_case(f"TRI0 ray {nm}", s, 0, r)
    var("o.x NaN", 1, QNAN)
    var("o.z NaN", 3, QNAN)
    var("o.x +inf", 1, 0x7F800000)
    var("d.x NaN", 4, QNAN)
    var("d.z NaN", 6, QNAN)
    var("inv.z NaN", 9, QNAN)
    var("inv.x NaN (axis the translator does not read)", 7, QNAN)
    var("inv.z = 4 (not 1/d.z = 2)", 9, f2u(4.0))
    var("inv.z = 1 (not 1/d.z)", 9, f2u(1.0))
    var("inv.x = 3 (not 1/d.x = inf)", 7, f2u(3.0))
    var("d.x = -0.0", 4, 0x80000000)
    r = list(base)
    r[4], r[7] = 0x80000000, 0xFF800000
    k2.add_case("TRI0 ray d.x = -0.0, inv.x = -inf", s, 0, r)
    r = list(base)
    r[4:10] = [0, 0, 0, 0x7F800000, 0x7F800000, 0x7F800000]
    k2.add_case("TRI0 ray d = 0, inv = +inf", s, 0, r)
    r = list(base)
    r[6], r[9] = 0x7F800000, 0
    k2.add_case("TRI0 ray d.z = +inf, inv.z = 0", s, 0, r)
    k2.add_case("TRI0 ray d.z = 1024 (t = 2^-8)", s, 0, ray_at(p, (0.0, 0.0, 1024.0), t=4.0 / 1024))
    k2.add_case("TRI0 ray d.z = 2^-20 (t = 2^22)", s, 0, ray_at(p, (0.0, 0.0, 2.0 ** -20), t=4.0 * 2 ** 20, tmax=INF))
    k2.add_case("TRI0 ray oblique, inv.z = 4 (not 1/d.z)", s, 0, ray_at(p, D_OBL, inv=(8.0, 16.0, 4.0)))
    k2.add_case("TRI0 ray oblique, inv.x = inv.y = 1 (not 1/d)", s, 0, ray_at(p, D_OBL, inv=(1.0, 1.0, 2.0)))
    return [k, k2]


def g4():
    """Shared-edge and shared-vertex tie-breaks on the closed fan."""
    k = Kernel("tri_g4a")
    s = k.add_node(leaf())
    for d, nm in ((D_POS, "+z"), (D_NEG, "-z"), (D_OBL, "oblique +z"), (D_OBLN, "oblique -z")):
        for t in range(4):
            k.add_case(f"fan vertex s2 (0,2), TRI{t}, {nm}", s, t, ray_at((0, 2), d))
    k2 = Kernel("tri_g4b")
    s = k2.add_node(leaf())
    edges = (("E01 (2,1)", (2, 1), (0, 1)), ("E03 (0,1)", (0, 1), (0, 3)), ("E12 (2,2)", (2, 2), (1, 2)),
             ("E23 (-1,2.5)", (-1, 2.5), (2, 3)))
    for d, nm in ((D_POS, "+z"), (D_NEG, "-z"), (D_OBL, "oblique +z")):
        for en, p, ts in edges:
            for t in ts:
                k2.add_case(f"shared edge {en}, TRI{t}, {nm}", s, t, ray_at(p, d))
    for nm, p, ts in (("outer vertex s0 (0,0)", (0, 0), (0, 3)), ("outer edge (2,0)", (2, 0), (0,)),
                      ("outer vertex s1 (4,0)", (4, 0), (0, 1)), ("outer vertex s3 (4,2)", (4, 2), (1, 2)),
                      ("outer vertex s4 (-2,3)", (-2, 3), (2,))):
        for t in ts:
            k2.add_case(f"{nm}, TRI{t}, +z", s, t, ray_at(p, D_POS))
    return [k, k2]


def g5_lanes(wave):
    """64 threads (two waves of 32, or one of 64), each lane its own origin."""
    k = Kernel(f"tri_lanes_w{wave}", threads=64, wave=wave, lane_offsets=(2.0 ** -6, 2.0 ** -6))
    s = k.add_node(leaf())
    for t in range(4):
        k.add_case(f"TRI{t} hit +z, origin + lane offset", s, t, ray_at(INSIDE[t], D_POS))
    k.add_case("TRI0 hit -z, origin + lane offset", s, 0, ray_at(INSIDE[0], D_NEG))
    k.add_case("TRI0 miss, origin + lane offset", s, 0, ray_at(INSIDE[1], D_POS))
    k.add_case("TRI1 hit oblique +z, origin + lane offset", s, 1, ray_at(INSIDE[1], D_OBL))
    return k


# ---------------------------------------------------------------------------------------------------------------------
# the tie cases (port of the measurement's ties/gen_ties.py and gen_bridge.py, the kernels tri_t1..t8 that ran): rays
# exactly on an edge or a vertex with x, y or z dominant, both windings, both signs of d[kz], and equal |d|
# components. Every kernel holds 32 cases, case i on its own slot i.
# ---------------------------------------------------------------------------------------------------------------------
TIE_SHARED = [("E01 (2,1)", (2, 1), (0, 1)), ("E03 (0,1)", (0, 1), (0, 3)), ("E12 (2,2)", (2, 2), (1, 2)),
              ("E23 (-1,2.5)", (-1, 2.5), (2, 3))]
TIE_VERTEX = [("fan vertex s2 (0,2)", (0, 2), (0, 1, 2, 3))]
TIE_OUTER = [("outer edge s0-s1 (2,0)", (2, 0), (0,)), ("outer vertex s0 (0,0)", (0, 0), (0,)),
             ("outer vertex s1 (4,0)", (4, 0), (0,))]
EQ_CLASSES = {"|dx|=|dy|>|dz|": ((0, 1), 2, 0.25), "|dx|=|dz|>|dy|": ((0, 2), 1, 0.125),
              "|dy|=|dz|>|dx|": ((1, 2), 0, 0.125), "|dx|=|dy|=|dz|": ((0, 1, 2), None, None)}
# the equal-|d| tie rows that ran (class, d, mirrored leaf, tie point), chosen by gen_ties.group_t4 so that every two
# possible dominant axes give different words on some row under the two registered rules
EQ_TIES = [("|dx|=|dy|>|dz|", (0.5, 0.5, 0.25), False, "E01 (2,1)"),
           ("|dx|=|dy|>|dz|", (0.5, -0.5, 0.25), False, "E12 (2,2)"),
           ("|dx|=|dy|>|dz|", (-0.5, 0.5, 0.25), True, "E03 (0,1)"),
           ("|dx|=|dy|>|dz|", (-0.5, -0.5, 0.25), False, "fan vertex s2 (0,2)"),
           ("|dx|=|dz|>|dy|", (0.5, 0.125, 0.5), False, "E12 (2,2)"),
           ("|dx|=|dz|>|dy|", (0.5, 0.125, -0.5), False, "E01 (2,1)"),
           ("|dx|=|dz|>|dy|", (-0.5, 0.125, 0.5), True, "E12 (2,2)"),
           ("|dx|=|dz|>|dy|", (-0.5, 0.125, -0.5), False, "E03 (0,1)"),
           ("|dy|=|dz|>|dx|", (0.125, 0.5, 0.5), False, "E01 (2,1)"),
           ("|dy|=|dz|>|dx|", (0.125, 0.5, -0.5), True, "E03 (0,1)"),
           ("|dy|=|dz|>|dx|", (0.125, -0.5, 0.5), False, "E01 (2,1)"),
           ("|dy|=|dz|>|dx|", (0.125, -0.5, -0.5), False, "E12 (2,2)"),
           ("|dx|=|dy|=|dz|", (0.5, 0.5, 0.5), False, "fan vertex s2 (0,2)"),
           ("|dx|=|dy|=|dz|", (0.5, 0.5, -0.5), True, "E03 (0,1)"),
           ("|dx|=|dy|=|dz|", (0.5, 0.5, -0.5), False, "E01 (2,1)"),
           ("|dx|=|dy|=|dz|", (0.5, -0.5, 0.5), False, "fan vertex s2 (0,2)"),
           ("|dx|=|dy|=|dz|", (-0.5, 0.5, 0.5), False, "E01 (2,1)"),
           ("|dx|=|dy|=|dz|", (-0.5, 0.5, 0.5), True, "E03 (0,1)")]
# measured rows of the kernels above, repeated as controls (kernel, case label)
TIE_CONTROLS = [("tri_g1a", "TRI0 hit +z"), ("tri_g4b", "shared edge E01 (2,1), TRI1, +z"),
                ("tri_g4b", "shared edge E01 (2,1), TRI0, +z"), ("tri_g4a", "fan vertex s2 (0,2), TRI2, -z"),
                ("tri_g4b", "shared edge E12 (2,2), TRI2, -z"), ("tri_g1b", "TRI0 dominant +x"),
                ("tri_g4b", "shared edge E23 (-1,2.5), TRI3, oblique +z"), ("tri_g1b", "TRI2 dominant -y"),
                ("tri_g4a", "fan vertex s2 (0,2), TRI1, oblique +z"),
                ("tri_g1c", "TRI1 miss beyond edge opposite v2 -z (det>0)"),
                ("tri_g4b", "outer vertex s0 (0,0), TRI0, +z"), ("tri_g1b", "TRI0 kz tie |dx|=|dy|=|dz|"),
                ("tri_g4b", "shared edge E03 (0,1), TRI0, -z"), ("tri_g1a", "TRI1 mirrored leaf hit +z"),
                ("tri_g4b", "shared edge E12 (2,2), TRI1, oblique +z"), ("tri_g1b", "TRI2 dominant +x")]
BRIDGE_CONTROLS = [("tri_g1a", "TRI1 mirrored leaf hit +z"), ("tri_g4b", "shared edge E03 (0,1), TRI0, -z"),
                   ("tri_g4a", "fan vertex s2 (0,2), TRI2, -z"), ("tri_g4b", "shared edge E12 (2,2), TRI1, oblique +z")]


def tie_rows(points, d, a=0.0, b=0.0, mirrored=False, rot=False, perm=None, tag=""):
    """[(label, node, type, ray)]: the fan turned 90 degrees in its plane ((x, y) -> (-y, x)) if rot, then mirrored
    (x -> -x, the other winding) if mirrored, lifted onto z = 3 + a x + b y, then axis-permuted (perm); a ray that
    reaches each tie point at t = 8."""
    def xy(p):
        x, y = (-p[1], p[0]) if rot else p
        return (-x, y) if mirrored else (x, y)
    pz = lambda x, y: 3 + a * x + b * y
    slots = [(*xy((x, y)), pz(*xy((x, y)))) for x, y, _ in FAN]
    node = leaf(permuted(slots, perm) if perm else slots)
    nm = ("rotated " if rot else "") + ("mirrored fan" if mirrored else "fan")
    if a or b:
        nm += f" on z=3{a:+g}x{b:+g}y"
    if perm:
        nm += f", axes cycled ({ {PX: 'z->x', PY: 'z->y'}[perm] })"
    out = []
    for pn, p, ts in points:
        q = xy(p)
        for t in ts:
            ray = ray_at(q, d, z=pz(*q), perm=perm)
            dd = [u2f(x) for x in ray[4:7]]
            out.append((f"{tag}{pn}, TRI{t}, {nm}, d=({dd[0]:g},{dd[1]:g},{dd[2]:g})", node, t, ray))
    return out


def eq_dirs(cls):
    tied, other, mag = EQ_CLASSES[cls]
    signs = [(1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1)] if len(tied) == 3 else \
        [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    out = []
    for sg in signs:
        d = [0.0, 0.0, 0.0]
        for ax, s in zip(tied, sg):
            d[ax] = 0.5 * s
        if other is not None:
            d[other] = mag
        out.append(tuple(d))
    return out


def tie_kernels(measured):
    """tri_t1..t7 (x / y dominant and equal |d|) and the z bridge tri_t8. measured: the kernels above, by label."""
    def ctl(kn, lab):
        k = measured[kn]
        c = next(c for c in k.cases if c["label"] == lab)
        return (f"control {kn}: {lab}", k.slots[c["slot"]], c["type"], c["ray"])
    body = []
    for perm, ax in ((PX, "x"), (PY, "y")):                    # T1: perpendicular fan, x / y dominant
        for mirrored in (False, True):
            for d, sn in ((D_POS, "+"), (D_NEG, "-")):
                pts = TIE_SHARED + TIE_VERTEX + ([] if (not mirrored and d == D_POS) else TIE_OUTER)
                body += tie_rows(pts, d, mirrored=mirrored, perm=perm, tag=f"T1 {sn}{ax}: ")
                body += tie_rows(TIE_SHARED[:1], d, mirrored=mirrored, rot=True, perm=perm, tag=f"T1 {sn}{ax}: ")
    for ax, a, b, mirrored, d in (("x", 1.0, 0.0, False, (0.125, -0.25, 0.5)),     # T2: tilted fans, oblique rays
                                  ("y", 1.0, 1.0, True, (-0.125, -0.25, -0.5))):
        body += tie_rows(TIE_SHARED + TIE_VERTEX, d, a=a, b=b, mirrored=mirrored, perm={"x": PX, "y": PY}[ax],
                         tag=f"T2 {ax}-dominant: ")
    i = 0                                                      # T3: equal |d|, no tie
    for cls in EQ_CLASSES:
        tied = EQ_CLASSES[cls][0]
        for d in eq_dirs(cls)[1::2]:
            t = i % 4
            i += 1
            body.append((f"T3 {cls}: TRI{t} interior hit, fan, d=({d[0]:g},{d[1]:g},{d[2]:g})", leaf(), t,
                         ray_at(INSIDE[t], d)))
        if len(tied) == 2 and tied != (0, 1):
            continue
        d = eq_dirs(cls)[3]
        body.append((f"T3 {cls}: TRI0 miss (at TRI1's point), fan, d=({d[0]:g},{d[1]:g},{d[2]:g})", leaf(), 0,
                     ray_at(INSIDE[1], d)))
    by_name = {p[0]: p for p in TIE_SHARED + TIE_VERTEX}       # T4: ties under equal |d|
    for cls, d, mirrored, pn in EQ_TIES:
        body += tie_rows([by_name[pn]], d, mirrored=mirrored, tag=f"T4 {cls}: ")
    ctls = [ctl(*c) for c in TIE_CONTROLS]
    ks, ci = [], 0
    for n in range(-(-len(body) // 31)):
        chunk = body[31 * n:31 * (n + 1)]
        cs = []
        while len(chunk) + len(cs) < 32:
            cs.append(ctls[ci % len(ctls)])
            ci += 1
        ks.append((f"tri_t{n + 1}", cs[:1] + chunk + cs[1:]))
    bridge = []                                                # tri_t8: z dominant, the cells #4202 never measured
    for d, sn in ((D_POS, "+z"), (D_NEG, "-z")):
        bridge += tie_rows(TIE_SHARED[:3] + TIE_VERTEX, d, mirrored=True, tag=f"T5 {sn}: ")
        bridge += tie_rows(TIE_SHARED[:1], d, mirrored=True, rot=True, tag=f"T5 {sn}: ")
        bridge += tie_rows(TIE_SHARED[:1], d, rot=True, tag=f"T5 {sn}: ")
    bctl = [ctl(*c) for c in BRIDGE_CONTROLS]
    ks.append(("tri_t8", bctl[:1] + bridge + bctl[1:]))
    out = []
    for label, rows in ks:
        k = Kernel(label)
        for lab, node, t, ray in rows:
            k.add_case(lab, k.add_node(node), t, ray)
        out.append(k)
    return out


def build_kernels():
    """The 22 kernels of the two measurements (14 of #4202, then the tie cases tri_t1..t8), in the order of
    tests/semantic/test_image_bvh_tri_hw.py."""
    for t, p in INSIDE.items():                  # each interior point is inside its own triangle only
        w = bary([FAN[s][:2] for s in VSLOTS[t]], p)
        assert all(x > 0 for x in w) and len(set(w)) == 3, (t, w)
        for u in range(4):
            if u != t:
                assert min(bary([FAN[s][:2] for s in VSLOTS[u]], p)) < 0, (t, u)
    ks = [g1(), g1("_nsa", nsa=True), g1("_w64", wave=64), g1b(), g1c(), *g2(), *g3(), *g4(), g5_lanes(32),
          g5_lanes(64)]
    ks += tie_kernels({k.label: k for k in ks})
    for k in ks:
        check_kernel(k)
    return ks


# ---------------------------------------------------------------------------------------------------------------------
# the measured rule (docs/design.md, Ray tracing row), binary32 rounding after every operation
# ---------------------------------------------------------------------------------------------------------------------
def f32(x):
    if math.isnan(x) or math.isinf(x):
        return x
    try:
        return struct.unpack("<f", struct.pack("<f", x))[0]
    except OverflowError:
        return math.copysign(INF, x)


def _add(a, b):
    return NAN if math.isinf(a) and math.isinf(b) and a != b else f32(a + b)


def _sub(a, b):
    return NAN if math.isinf(a) and math.isinf(b) and a == b else f32(a - b)


def _mul(a, b):
    return NAN if (a == 0 and math.isinf(b)) or (b == 0 and math.isinf(a)) else f32(a * b)


def rule_words(node, ntype, ray):
    """The four words RDNA2 returns for a triangle node (16 words) of type 0..3 and a ray (10 words), as u32.

    kz = the axis of the largest |d| (ties x, then y, then z); kx, ky the next two, swapped when d[kz] < 0. The shear
    is multiplied through by |d[kz]| (no division): A'' = A[kx] |d[kz]| - sign(d[kz]) d[kx] A[kz] for each vertex
    relative to the origin, which is |d[kz]| times the watertight test's sheared coordinate. U, V, W and det = U + V + W
    then carry |d[kz]|^2, and T'' = U A[kz] + V B[kz] + W C[kz] carries sign(d[kz]) |d[kz]|^3 T, so the words are
    k (T, det, b_i, b_j) with k = |d[kz]|^3: (sign(d[kz]) T'', |d[kz]| det, |d[kz]| b_i, |d[kz]| b_j). A miss puts
    copysign(inf, word 1) in word 0. Every zero is +0 and every NaN 0xFFC00000. A NaN anywhere in the ray, or an
    infinite origin, gives (inf, 1, 0, 0). The supplied inverse direction is read only for that NaN check. An edge
    P -> Q whose edge function is exactly 0 counts as inside when, with (tx, ty) = Q - P in the sheared frame with the
    d[kz] < 0 swap undone, ty == 0 ? sign(det) tx > 0 : sign(d[kz]) sign(det) ty > 0."""
    tmax = u2f(ray[0])
    o, d, inv = ([u2f(x) for x in ray[a:a + 3]] for a in (1, 4, 7))
    if any(math.isnan(x) for x in [tmax] + o + d + inv) or any(math.isinf(x) for x in o):
        return [0x7F800000, 0x3F800000, 0, 0]
    V = [[u2f(node[3 * s + c]) for c in range(3)] for s in VSLOTS[ntype]]
    ad = [abs(c) for c in d]
    if ad[0] >= ad[1]:
        kz, ky, kx = (0, 2, 1) if ad[0] >= ad[2] else (2, 1, 0)
    else:
        kz, ky, kx = (1, 0, 2) if ad[1] >= ad[2] else (2, 1, 0)
    s = -1.0 if d[kz] < 0 else 1.0
    if s < 0:
        kx, ky = ky, kx
    adz = abs(d[kz])
    px, py = _mul(s, d[kx]), _mul(s, d[ky])
    rel = [[_sub(V[j][c], o[c]) for c in range(3)] for j in range(3)]
    sh = [(_sub(_mul(r[kx], adz), _mul(px, r[kz])), _sub(_mul(r[ky], adz), _mul(py, r[kz])), r[kz]) for r in rel]
    (ax, ay, az), (bx, by, bz), (cx, cy, cz) = sh
    U = _sub(_mul(cx, by), _mul(cy, bx))
    Vv = _sub(_mul(ax, cy), _mul(ay, cx))
    W = _sub(_mul(bx, ay), _mul(by, ax))
    det = _add(U, _add(Vv, W))
    byte = (node[15] >> (8 * ntype)) & 0xFF
    pick = (U, Vv, W, U)                        # selector 3 reads U; bits 7:4 of the byte are ignored
    w = [_mul(s, _add(_add(_mul(U, az), _mul(Vv, bz)), _mul(W, cz)))]
    w += [_mul(adz, x) for x in (det, pick[byte & 3], pick[(byte >> 2) & 3])]
    hit = False
    if det == det and det != 0:
        sg = 1.0 if det > 0 else -1.0

        def passes(x, p, q):                    # a zero edge function: RDNA2's tie rule on the edge from p to q
            if x != 0:
                return x * sg > 0               # False for a NaN
            tx, ty = q[0] - p[0], q[1] - p[1]
            if s < 0:                           # in the frame with the kx/ky swap of d[kz] < 0 undone
                tx, ty = ty, tx
            return sg * tx > 0 if ty == 0 else s * sg * ty > 0
        a2, b2, c2 = (ax, ay), (bx, by), (cx, cy)
        hit = passes(U, b2, c2) and passes(Vv, c2, a2) and passes(W, a2, b2) and sg * w[0] >= 0
    out = [0xFFC00000 if x != x else 0 if x == 0 else f2u(x) for x in w]
    if not hit:
        out[0] = 0x7F800000 | (out[1] & 0x80000000)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# the translator (tools/rdna2spirv.py on tests/semantic/spirv_mini_interp.py)
# ---------------------------------------------------------------------------------------------------------------------
T_RAY, T_NODE = 0x1000, 0x2000


def _translate(work, form, nsa):
    import rdna2spirv
    lines = ["s_buffer_load_dwordx16 s[8:23], s[0:3], 0", "s_waitcnt lgkmcnt(0)"]
    if form == "64":
        lines += [f"v_mov_b32_e32 v{1 + i}, s{8 + i}" for i in range(12)] + ["s_mov_b32 s4, 0"]
        regs, op = [f"v{1 + i}" for i in range(12)], "image_bvh64_intersect_ray"
    else:
        lines += ["v_mov_b32_e32 v1, s8"] + [f"v_mov_b32_e32 v{2 + i}, s{10 + i}" for i in range(10)]
        lines += [f"s_mov_b32 s4, 0x{T_NODE >> 8:x}"]
        regs, op = [f"v{1 + i}" for i in range(11)], "image_bvh_intersect_ray"
    addr = "[" + ", ".join(regs) + "]" if nsa else f"v[1:{len(regs)}]"
    lines += [f"s_mov_b32 s5, 0x{DESC['w1']:08x}", f"s_mov_b32 s6, 0x{DESC['w2']:08x}",
              f"s_mov_b32 s7, 0x{DESC['w3']:08x}", f"{op} v[20:23], {addr}, s[4:7]",
              "exp mrt0 v20, v21, v22, v23 done vm", "s_endpgm"]
    name = f"ps_bvh_tri_probe_{form}{'_nsa' if nsa else ''}"
    os.makedirs(os.path.join(work, "disasm"), exist_ok=True)
    src = os.path.join(work, "disasm", name + ".s")
    with open(src, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"; {name}\n" + "".join(f"\t{ln}\n" for ln in lines))
    with open(os.path.join(work, name + ".dxbc"), "wb") as f:
        f.write(STUB_DXBC)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(src)
        gc.collect()
    return spv


def run_translator(kernels):
    """{kernel: [per case {"tr32", "tr64"} or {"lanes": [...]}]}; a case the interpreter refuses is a string."""
    import spirv_mini_interp
    out = {}
    with tempfile.TemporaryDirectory(prefix="bvh_tri_probe_") as work:
        spv = {(f, n): _translate(work, f, n) for f in ("32", "64") for n in (False, True)}

        def one(k, c, ray):
            r = {}
            for form in ("32", "64"):
                nid = (T_NODE >> 3 | c["type"]) if form == "64" else c["type"]
                mem = {T_RAY + 4 * i: w for i, w in enumerate([nid, 0, *ray, 0, 0, 0, 0])}
                mem.update({T_NODE + 4 * i: w for i, w in enumerate(k.slots[c["slot"]])})
                try:
                    r["tr" + form] = list(spirv_mini_interp.run_fragment(
                        spv[(form, k.nsa)], push_constants=[T_RAY, 0, 0x100, 0] + [0] * 28,
                        memory=mem)["%out_mrt0"])
                except Exception as e:       # noqa: BLE001  (UnsupportedConstruct: undefined in SPIR-V)
                    r["tr" + form] = f"{type(e).__name__}: {e}"
            return r
        for k in kernels:
            out[k.label] = [dict(lanes=[one(k, c, k.lane_ray(c, ln)) for ln in range(64)]) if k.threads == 64
                            else one(k, c, c["ray"]) for c in k.cases]
    return out


# ---------------------------------------------------------------------------------------------------------------------
# the GPU
# ---------------------------------------------------------------------------------------------------------------------
def run_hw(kernels, oracle, asm):
    """{kernel: [per case {"hw32", "hw64"} or {"lanes": [...]}], or a string saying why it was not run}. One
    oracle.run call (one dispatch) per kernel; the first untrustworthy output stops the run."""
    out = {}
    limit = getattr(oracle, "max_words", None) or (4096 if type(oracle).__name__ == "DrmOracle" else 16384)
    # the runner checks isinstance(program, asm.Kernel) against the asm module it imported itself (one loaded by file
    # path is another module object): build the programs with that one
    asm = getattr(sys.modules.get(type(oracle).__module__), "asm", None) or asm
    for k in kernels:
        check_kernel(k)
        if k.L["words"] > limit:
            out[k.label] = f"skipped: the launcher reads back at most {limit} dwords, this kernel {k.L['words']}"
            continue
        output = oracle.run([k.program(asm)], words=k.L["words"])[0]
        why = output_problem(k, output)
        if why:
            raise RuntimeError(f"{k.label}: {why}")
        out[k.label] = results(k, output)[0]
    return out


# ---------------------------------------------------------------------------------------------------------------------
# comparison and the command line
# ---------------------------------------------------------------------------------------------------------------------
def hx(ws):
    return " ".join(f"{w:08x}" for w in ws) if isinstance(ws, list) else str(ws)


def compare(kernels, tr, hw):
    """Rows of {kernel, case, lane, rule, tr32/tr64, hw32/hw64, status} and the number of failed rows."""
    rows, bad = [], 0
    for k in kernels:
        trk = tr.get(k.label) if tr else None
        hwk = hw.get(k.label) if hw else None
        if isinstance(hwk, str):
            print(f"{k.label}: {hwk}")
            hwk = None
        for i, c in enumerate(k.cases):
            lanes = range(64) if k.threads == 64 else [None]
            for ln in lanes:
                ray = k.lane_ray(c, ln) if ln is not None else c["ray"]
                want = rule_words(k.slots[c["slot"]], c["type"], ray)
                row = dict(kernel=k.label, case=c["label"], lane=ln, rule=want)
                for src, got in (("tr", trk), ("hw", hwk)):
                    if got is not None:
                        x = got[i]["lanes"][ln] if ln is not None else got[i]
                        row.update(x)
                vals = [row[x] for x in ("tr32", "tr64", "hw32", "hw64") if x in row]
                row["status"] = "OK" if all(v == want for v in vals) else "MISMATCH"
                if row["status"] != "OK":
                    bad += 1
                    print(f"[MISMATCH] {k.label}: {c['label']}" + (f", lane {ln}" if ln is not None else ""))
                    for x in ("rule", "tr32", "tr64", "hw32", "hw64"):
                        if x in row:
                            print(f"    {x:5} {hx(row[x])}")
                rows.append(row)
    return rows, bad


def self_test():
    ks = build_kernels()
    single = sum(len(k.cases) for k in ks if k.threads == 1)
    assert single == 564 and [len(k.cases) for k in ks if k.threads == 64] == [7, 7], single
    assert len({k.uid for k in ks}) == len(ks)
    for k in ks:                                  # a box node, type 6/7 or a free node id cannot be added
        for bad in (lambda: k.add_case("x", 0, 5, [0] * 10), lambda: k.add_case("x", 31, 0, [0] * 10)):
            n = len(k.cases)
            try:
                bad()
            except ValueError:
                pass
            else:
                raise AssertionError(f"{k.label}: an illegal case was accepted")
            del k.cases[n:]
    by = {k.label: k for k in ks}
    # hand-derived words of five measured rows (docs/design.md, Ray tracing row)
    g1a, g3b = by["tri_g1a"], by["tri_g3b"]
    for k, label, want in (
            (g1a, "TRI0 hit +z", [0xC1000000, 0xBF800000, 0xBE800000, 0xBE000000]),    # k = 1/8: -64/8, -8/8, ...
            (g1a, "TRI2 hit +z", [0xC0800000, 0xBF000000, 0xBE800000, 0xBD800000]),    # |det| 4: half of TRI0's
            (g3b, "TRI0 ray inv.x NaN (axis the translator does not read)", [0x7F800000, 0x3F800000, 0, 0]),
            (g3b, "TRI0 ray d.z = 1024 (t = 2^-8)", [0xCC000000, 0xD0000000, 0xCF000000, 0xCE800000]),   # k = 2^30
            (g3b, "TRI0 ray d = 0, inv = +inf", [0x7F800000, 0, 0, 0])):
        c = next(c for c in k.cases if c["label"] == label)
        got = rule_words(k.slots[c["slot"]], c["type"], c["ray"])
        assert got == want, (label, hx(got), hx(want))
    print(f"self-test OK: {len(ks)} kernels, {single} single-lane cases, 2 x 7 lane cases; rule checked on 5 rows")
    return 0


def write_kernels(kernels, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for k in kernels:
        with open(os.path.join(out_dir, k.label + ".s"), "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(k.lines()) + "\n")
        tab = dict(label=k.label, uid=k.uid, threads=k.threads, wave=k.wave, nsa=k.nsa, layout=k.L,
                   lane_offsets=k.lane_offsets, desc={x: f"0x{v:08x}" for x, v in DESC.items()},
                   nodes={str(s): [f"0x{w:08x}" for w in ws] for s, ws in sorted(k.slots.items())},
                   cases=[dict(c, ray=[f"0x{w:08x}" for w in c["ray"]], node32=f"0x{k.node32(c):x}",
                               rule=[f"0x{w:08x}" for w in rule_words(k.slots[c["slot"]], c["type"], c["ray"])])
                          for c in k.cases])
        with open(os.path.join(out_dir, k.label + ".json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(tab, f, indent=1)
        print(f"{k.label}: {len(k.cases)} cases, lines sha256 "
              f"{hashlib.sha256(chr(10).join(k.lines()).encode('utf-8')).hexdigest()[:16]}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--oracle", default="translator", help="translator (default), hw, or translator,hw")
    ap.add_argument("--kernel", action="append", default=[], help="run only this kernel (repeatable)")
    ap.add_argument("--list", action="store_true", help="list the kernels and their cases")
    ap.add_argument("--write-kernels", metavar="DIR", help="write every kernel's lines and cases; no GPU")
    ap.add_argument("--json", dest="json_out", help="write every case's words to this file")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    backends = tuple(dict.fromkeys(t.strip() for t in args.oracle.split(",")))
    if not backends or any(t not in ("translator", "hw") for t in backends):
        print("error: --oracle takes translator, hw or translator,hw", file=sys.stderr)
        return 2
    kernels = build_kernels()
    if args.kernel:
        names = [k.label for k in kernels]
        unknown = [n for n in args.kernel if n not in names]
        if unknown:
            print(f"error: unknown kernel(s) {', '.join(unknown)} (known: {', '.join(names)})", file=sys.stderr)
            return 2
        kernels = [k for k in kernels if k.label in args.kernel]
    if args.json_out is not None and (not args.json_out.strip() or os.path.isdir(args.json_out)):
        print(f"error: --json {args.json_out.strip() or '(empty)'} is not a file path", file=sys.stderr)
        return 2
    if args.list:
        for k in kernels:
            print(f"{k.label:15} threads={k.threads:2} wave={k.wave} nsa={str(k.nsa):5} cases={len(k.cases)}")
            for c in k.cases:
                print(f"    {c['label']}")
        return 0
    if args.write_kernels:
        return write_kernels(kernels, args.write_kernels)
    tr = run_translator(kernels) if "translator" in backends else None
    hw, device = None, None
    if "hw" in backends:
        from hw_oracle import asm, runner
        try:
            oracle = runner.open_oracle()
        except runner.Unavailable as e:
            print(f"hardware oracle unavailable: {e}", file=sys.stderr)
            return 77
        try:
            if asm.family(oracle.arch) != "gfx10":
                print(f"SKIP: {oracle.arch} is not GFX10 (RDNA1/RDNA2)", file=sys.stderr)
                return 77
            device = getattr(oracle, "device", oracle.arch)
            print(f"hardware oracle: {device}")
            try:
                hw = run_hw(kernels, oracle, asm)
            except RuntimeError as e:
                print(f"error: {e}", file=sys.stderr)
                return 2
        finally:
            oracle.close()
    rows, bad = compare(kernels, tr, hw)
    srcs = [x for x, v in (("translator", tr), ("hardware", hw)) if v is not None]
    print(f"summary: {len(rows)} results ({sum(r['lane'] is None for r in rows)} single-lane) of {len(kernels)} "
          f"kernels, {' and '.join(srcs)} against the rule, both forms: {len(rows) - bad} match, {bad} mismatch")
    if args.json_out is not None:
        parent = os.path.dirname(os.path.abspath(args.json_out))
        try:
            os.makedirs(parent, exist_ok=True)
            with open(args.json_out, "w", encoding="utf-8", newline="\n") as f:
                json.dump({"device": device, "rows": rows}, f, indent=1)
                f.write("\n")
        except OSError as e:
            print(f"error: cannot write --json {args.json_out}: {e}", file=sys.stderr)
            return 2
    return 1 if bad else 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
