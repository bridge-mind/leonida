r"""CPU-only differential fuzzer for wave and cross-lane operations (standard library only, no GPU, no llvm-mc).

tools/crosslane_fuzz.py (FUZZ-7) checks these instructions on a GPU. This runner checks the same family on the CPU, so it
can run anywhere the translator tests run, CI included: seeded straight-line compute programs go through

  * the reference: tools/cf_ref.WaveMachine (FUZZ-8's wave model: compares, v_cmpx, saveexec, mask logic, v_mbcnt),
    extended below with what it leaves out, the cross-lane and whole-mask instructions, each rule written next to
    it from the RDNA2 ISA;
  * the translator: rdna2spirv's SPIR-V executed by tests/semantic/spirv_mini_interp.run_workgroup, one invocation per
    lane, all lanes of the wave in one subgroup of the wave's size (wave64: AMD with requiredSubgroupSize = 64),

and every dword every lane stores is compared: four vector registers, two lane masks, VCC and the final EXEC as each
lane's bit, and four scalars.

Instructions: v_readlane_b32, v_readfirstlane_b32, v_writelane_b32, ds_bpermute_b32, v_permlane16_b32,
v_permlanex16_b32, s_bcnt0/1_i32_b32/b64, s_ff0/1_i32_b32/b64, s_not_b32/b64 (added here), and from cf_ref: v_cmp /
v_cmpx into lane masks, s_and/or/xor/andn2/orn2 masks, *_saveexec, s_mov_b64 / mask s_mov_b32, v_mbcnt_lo/hi;
v_cndmask_b32 on a mask and integer VALU ops from cpu_ref. Random EXEC: v_cmpx, saveexec, constants.

Lane masks and 64-bit data: the translator keeps a lane mask as each lane's own bit and data as its value
(tools/lane_masks.py). By default the generator keeps the two apart, so a default run is a regression check: the mask
registers only ever receive masks, and other immediates go straight to an instruction that reads them as a mask (EXEC,
a v_cndmask condition) or as data (s_bcnt, s_ff). Known gaps on main stay off unless asked for:
  --masks-as-data      mix lane masks and 64-bit data in one register: the open part of issue #210
  --permlane-partial   v_permlane16/x16 under a partial EXEC, where a source lane can be off: SHADER-74, #580

    python -I tools/wave_diff.py [--programs N] [--seed S] [--max-ops N] [--wave 32|64|both] [--masks-as-data]
    python -I tools/wave_diff.py --self-test       # canaries: a match, a caught sabotage, a counted skip

Exit code 0 when every program matches, 1 on a mismatch (lane, register, both words and the listing), 2 on harness
errors. Skips are counted with their reasons, never silent.
"""
import argparse
import gc
import os
import random
import struct
import sys
import tempfile
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)
SEM_DIR = os.path.join(os.path.dirname(HERE), "tests", "semantic")
if SEM_DIR not in sys.path:
    sys.path.insert(0, SEM_DIR)

import cf_ref  # noqa: E402
import cpu_ref  # noqa: E402
import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
from shader_ir import parse_file  # noqa: E402

M32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC]          # as tests/semantic/test_wave64_subgroup_ops.py
LANE_BYTES = 64                                  # each lane stores WORDS dwords at lane * 64

LANE = 1                                         # v1 = wave lane (v_mbcnt of -1)
VREGS = (8, 9, 10, 11)                           # vector scratch
SREGS = (12, 13, 14, 15)                         # plain scalars, never a mask
ADDR = 20
WORDS = VREGS + (12, 13, 14, 15, 16, 17, 18, 19)  # v12-v15: masks as bits, v16-v19: the scalars
CMP = ("eq", "ne", "lt", "le", "gt", "ge")
IMM_MASKS = (1, 2, 3, 5, 6, 7, 15, 16, 31, 32, 63, 64, -2, -3, -8, -16)


def fssm(threads):
    header = bytearray(0x148)
    struct.pack_into("<4I", header, 0x138, 0x8088, threads, 1, 1)
    return bytes(header)


# ---- reference --------------------------------------------------------------------------------------------------

class WaveRef(cf_ref.WaveMachine):
    """cf_ref's wave model plus the cross-lane and whole-mask instructions. Straight-line listings without encoding
    bytes get 4-byte placeholders (no branch needs the sizes)."""

    def __init__(self, info, wave):
        for i, ins in enumerate(info["insns"]):
            if not ins.get("enc"):
                if ins["op"].startswith(("s_branch", "s_cbranch")):
                    raise cf_ref.ProgramError("a branch needs encoding bytes")
                ins["enc"], ins["pc"] = "00000000", 4 * i
        super().__init__(info, wave)

    def _lane_sel(self, arg):
        return self._int_src(arg, 0, scalar=True) & (self.wave - 1)

    def _vsrc(self, ins, arg):
        if arg["k"] != "vgpr":
            raise cf_ref.Unsupported(ins["op"], self.pc, "source must be a VGPR")
        return [self._vgpr(arg["r"], lane) for lane in range(64)]

    def _count_src(self, op, src, width):
        if src["k"] == "imm":
            return src["v"] & ((1 << width) - 1)          # an inline constant sign-extends to the operand width
        if src["k"] == "sgpr" and src.get("n", 1) == width // 32:
            v = self.sgpr.get(src["r"], 0)
            return v | (self.sgpr.get(src["r"] + 1, 0) << 32) if width == 64 else v
        names = {64: {"vcc": "vcc", "exec": "exec"}, 32: {"vcc_lo": "vcc", "exec_lo": "exec"}}[width]
        if src["k"] == "special" and src["t"] in names:
            return getattr(self, names[src["t"]]) & ((1 << width) - 1)
        raise cf_ref.Unsupported(op, self.pc, "operand is not modelled")

    def step(self, ins):
        op, a = ins["op"], ins["args"]
        base = cpu_ref.base_op(op)
        if base == "s_waitcnt":
            return None
        if base in ("s_not_b32", "s_not_b64"):
            # sD = ~S0, SCC = sD != 0, on a lane mask of the wave's width.
            self._check_width(base, op)
            self._expect(ins, 2)
            r = ~self._read_mask(a[1]) & self.wmask
            self._write_mask(a[0], r)
            self.scc = 1 if r else 0
            return None
        if base[:-4] in ("s_bcnt0_i32", "s_bcnt1_i32", "s_ff0_i32", "s_ff1_i32") and base[-4:] in ("_b32", "_b64"):
            # bcnt1/0: how many bits are 1/0 (SCC = result != 0); ff1/0: the lowest bit that is 1/0, -1 if none
            # (SCC unchanged).
            self._expect(ins, 2)
            width = 64 if base.endswith("_b64") else 32
            v = self._count_src(op, a[1], width)
            if "0_i32" in base:
                v = ~v & ((1 << width) - 1)
            if "bcnt" in base:
                r = bin(v).count("1")
                self.scc = 1 if r else 0
            else:
                r = ((v & -v).bit_length() - 1) if v else M32
            self._write_scalar_dst(a[0], r)
            return None
        if base == "v_readlane_b32":
            # sD = vS of lane S1 mod the wave size (S1[5:0] in wave64, S1[4:0] in wave32), EXEC ignored.
            self._expect(ins, 3)
            self._write_scalar_dst(a[0], self._vsrc(ins, a[1])[self._lane_sel(a[2])])
            return None
        if base == "v_readfirstlane_b32":
            # sD = vS of the lowest lane that is on in EXEC; lane 0 when none is.
            self._expect(ins, 2)
            active = self._active_lanes()
            self._write_scalar_dst(a[0], self._vsrc(ins, a[1])[active[0] if active else 0])
            return None
        if base == "v_writelane_b32":
            # vD of lane S1 mod the wave size = S0; every other lane keeps its value. EXEC ignored.
            self._expect(ins, 3)
            self._write_vgpr(a[0], self._lane_sel(a[2]), self._int_src(a[1], 0, scalar=True))
            return None
        if base == "ds_bpermute_b32":
            # Backward permute: lane l reads vData of lane ((vAddr[l] + offset) / 4) mod 32 of its own 32-lane group
            # (a wave64 runs it as two wave32s), 0 if that lane is off in EXEC. All sources are read first.
            self._expect(ins, 3)
            mods = ins.get("mods") or []
            if any(not m.startswith("offset:") for m in mods):
                raise cf_ref.Unsupported(op, self.pc, "modifier is not modelled")
            off = next((int(m.split(":", 1)[1], 0) for m in mods), 0)
            addr, data = self._vsrc(ins, a[1]), self._vsrc(ins, a[2])
            out = {}
            for lane in self._active_lanes():
                src = (lane & 32) | ((((addr[lane] + off) & M32) >> 2) & 31)
                out[lane] = data[src] if (self.exec >> src) & 1 else 0
            for lane, value in out.items():
                self._write_vgpr(a[0], lane, value)
            return None
        if base in ("v_permlane16_b32", "v_permlanex16_b32"):
            # Lane l reads lane sel of a 16-lane row: its own row (permlane16) or the other row of its 32-lane half
            # (x16); sel is nibble (l & 7) of S1 for (l & 15) < 8, of S2 otherwise. Without op_sel (FI = 0) a
            # source lane that is off leaves vD unwritten, as measured on RDNA2 and RDNA3.5 (#374, PROC-19).
            self._expect(ins, 4)
            if ins.get("mods"):
                raise cf_ref.Unsupported(op, self.pc, "op_sel (FI/BOUND_CTRL) is not modelled")
            s1, s2 = self._int_src(a[2], 0, scalar=True), self._int_src(a[3], 0, scalar=True)
            data = self._vsrc(ins, a[1])
            out = {}
            for lane in self._active_lanes():
                within = lane & 15
                sel = ((s1 if within < 8 else s2) >> (4 * (within & 7))) & 15
                src = ((lane & ~15) ^ (16 if base == "v_permlanex16_b32" else 0)) | sel
                if (self.exec >> src) & 1:
                    out[lane] = data[src]
            for lane, value in out.items():
                self._write_vgpr(a[0], lane, value)
            return None
        return super().step(ins)


def run_model(lines, wave):
    """Every lane's WORDS after the program, from the reference."""
    with tempfile.TemporaryDirectory(prefix="wave-diff-") as tmp:
        path = os.path.join(tmp, "case.s")
        _write(path, lines, stores=False)
        m = WaveRef(parse_file(path), wave)
        m.run()
    return [[m._vgpr(r, lane) for r in WORDS] for lane in range(wave)]


# ---- generator --------------------------------------------------------------------------------------------------

class Gen:
    """One random program for one wave size. The lane-mask registers are SGPR pairs in wave64, single SGPRs in
    wave32."""

    def __init__(self, rng, wave, masks_as_data=False, permlane_partial=False):
        self.rng, self.wave, self.masks_as_data = rng, wave, masks_as_data
        self.permlane_partial, self.full_exec = permlane_partial, True
        self.w64 = wave == 64
        self.masks = ("s[6:7]", "s[8:9]") if self.w64 else ("s6", "s8")
        self.save = "s[10:11]" if self.w64 else "s10"
        self.vcc = "vcc" if self.w64 else "vcc_lo"
        self.exec = "exec" if self.w64 else "exec_lo"
        self.b = "b64" if self.w64 else "b32"
        self.lines = []

    def emit(self, line):
        self.lines.append(line)

    def v(self):
        return self.rng.choice(VREGS)

    def src(self):
        r = self.rng.random()
        if r < 0.75:
            return f"v{self.v()}"
        if r < 0.9:
            return str(self.rng.choice((0, 1, 2, 3, 5, 7, 16, 31, 32, 63, 64, -1, -16)))
        return f"s{self.rng.choice(SREGS)}"

    def mask_src(self, any_imm=False):
        """A lane-mask operand. Immediates other than 0 and -1 only where the instruction reads them straight away
        (any_imm), or with --masks-as-data: kept in a mask register they are data later read as a mask (#210)."""
        r = self.rng.random()
        if r < 0.7:
            return self.rng.choice(self.masks)
        if r < 0.85 or not (any_imm or self.masks_as_data):
            return self.rng.choice((self.vcc, self.exec, "-1", "0"))
        return str(self.rng.choice(IMM_MASKS))

    def half(self, mask, i):
        """SGPR i (0 = low, 1 = high) of a wave64 mask pair, or the wave32 mask itself."""
        return f"s{int(mask[2:-1].split(':')[i])}" if self.w64 else mask

    def prologue(self):
        rng = self.rng
        self.emit(f"v_mbcnt_lo_u32_b32 v{LANE}, -1, 0")
        if self.w64:
            self.emit(f"v_mbcnt_hi_u32_b32 v{LANE}, -1, v{LANE}")
        for v in VREGS:
            k = rng.randrange(4)
            if k == 0:
                self.emit(f"v_mad_u32_u24 v{v}, v{LANE}, {rng.choice((3, 5, 7, 13, 37))}, {rng.randrange(0, 65)}")
            elif k == 1:
                self.emit(f"v_xor_b32 v{v}, 0x{rng.getrandbits(32):08x}, v{LANE}")
            elif k == 2:
                self.emit(f"v_mul_u32_u24 v{v}, v{LANE}, v{LANE}")
            else:
                self.emit(f"v_lshlrev_b32 v{v}, {rng.randrange(0, 4)}, v{LANE}")
        for s in SREGS:
            self.emit(f"s_mov_b32 s{s}, {rng.randrange(0, 64)}")
        for m in self.masks:
            self.ballot(m)
        self.emit(f"v_cmp_{rng.choice(CMP)}_u32_e32 {self.vcc}, v{self.v()}, v{self.v()}")

    def ballot(self, dst=None):
        rng = self.rng
        dst = dst or (rng.choice(self.masks) if rng.random() < 0.8 else self.vcc)
        a = self.src() if rng.random() < 0.3 else f"v{self.v()}"
        self.emit(f"v_cmp_{rng.choice(CMP)}_{rng.choice(('u32', 'i32'))}_e64 {dst}, {a}, v{self.v()}")

    def op(self):
        rng = self.rng
        kinds = (["valu"] * 6 + ["ballot"] * 2 + ["vcc", "restore"] + ["mask"] * 2 + ["count"] * 2 +
                 ["cndmask"] * 2 + ["mbcnt"] * 2 + ["readlane"] * 3 + ["readfirstlane"] * 2 + ["writelane"] * 2 +
                 ["bpermute"] * 3 + ["exec"] * 2)
        if self.full_exec or self.permlane_partial:
            kinds += ["permlane"] * 3
        if self.masks_as_data:
            kinds += ["mask_as_data"] * 2
        k = rng.choice(kinds)
        d = self.v()
        if k == "valu":
            op = rng.choice(("v_add_nc_u32", "v_sub_nc_u32", "v_xor_b32", "v_and_b32", "v_or_b32", "v_max_u32",
                             "v_min_i32", "v_mul_lo_u32", "v_lshlrev_b32", "v_lshrrev_b32"))
            self.emit(f"{op} v{d}, {self.src()}, v{self.v()}")
        elif k == "ballot":
            self.ballot()
        elif k == "vcc":
            self.emit(f"v_cmp_{rng.choice(CMP)}_{rng.choice(('u32', 'i32'))}_e32 {self.vcc}, {self.src()}, "
                      f"v{self.v()}")
        elif k == "mask":
            op = rng.choice(("s_and", "s_or", "s_xor", "s_andn2", "s_orn2", "s_not", "s_mov"))
            dst = rng.choice(self.masks)
            if op in ("s_not", "s_mov"):
                self.emit(f"{op}_{self.b} {dst}, {self.mask_src()}")
            else:
                self.emit(f"{op}_{self.b} {dst}, {self.mask_src()}, {self.mask_src()}")
        elif k == "count":
            op = rng.choice(("s_bcnt1_i32", "s_ff1_i32", "s_bcnt0_i32", "s_ff0_i32"))
            self.emit(f"{op}_{self.b} s{rng.choice(SREGS)}, {self.mask_src(any_imm=True)}")
        elif k == "cndmask":
            r = rng.random()
            if r < 0.4:
                self.emit(f"v_cndmask_b32_e64 v{d}, {self.src()}, {self.src()}, {rng.choice(self.masks)}")
            elif r < 0.55:
                self.emit(f"v_cndmask_b32_e64 v{d}, {self.src()}, {self.src()}, {rng.choice(IMM_MASKS)}")
            else:
                self.emit(f"v_cndmask_b32_e32 v{d}, {self.src()}, v{self.v()}, {self.vcc}")
        elif k == "mbcnt":
            m = rng.choice(self.masks)
            if self.w64 and rng.random() < 0.5:
                self.emit(f"v_mbcnt_hi_u32_b32 v{d}, {self.half(m, 1)}, {self.src()}")
            else:
                self.emit(f"v_mbcnt_lo_u32_b32 v{d}, {self.half(m, 0)}, {self.src()}")
        elif k == "readlane":
            sel = str(rng.randrange(0, 64)) if rng.random() < 0.6 else f"s{rng.choice(SREGS)}"
            self.emit(f"v_readlane_b32 s{rng.choice(SREGS)}, v{self.v()}, {sel}")
        elif k == "readfirstlane":
            self.emit(f"v_readfirstlane_b32 s{rng.choice(SREGS)}, v{self.v()}")
        elif k == "writelane":
            sel = str(rng.randrange(0, 64)) if rng.random() < 0.6 else f"s{rng.choice(SREGS)}"
            val = f"s{rng.choice(SREGS)}" if rng.random() < 0.7 else str(rng.randrange(0, 65))
            self.emit(f"v_writelane_b32 v{d}, {val}, {sel}")
        elif k == "bpermute":
            off = f" offset:{rng.choice((4, 8, 64, 124, 128))}" if rng.random() < 0.3 else ""
            self.emit(f"ds_bpermute_b32 v{d}, v{self.v()}, v{self.v()}{off}")
            self.emit("s_waitcnt lgkmcnt(0)")
        elif k == "permlane":
            a, b = rng.sample(SREGS, 2)
            self.emit(f"s_mov_b32 s{a}, 0x{rng.getrandbits(32):08x}")
            self.emit(f"s_mov_b32 s{b}, 0x{rng.getrandbits(32):08x}")
            self.emit(f"{rng.choice(('v_permlane16_b32', 'v_permlanex16_b32'))} v{d}, v{self.v()}, s{a}, s{b}")
        elif k == "exec":
            r = rng.random()
            if r < 0.4:
                self.emit(f"v_cmpx_{rng.choice(CMP)}_u32_e32 {self.src()}, v{self.v()}")
            elif r < 0.8:
                op = rng.choice(("s_and_saveexec", "s_or_saveexec", "s_andn2_saveexec"))
                self.emit(f"{op}_{self.b} {rng.choice(self.masks)}, {rng.choice(self.masks)}")
            else:
                self.emit(f"s_and_{self.b} {self.exec}, {self.exec}, {self.mask_src(any_imm=True)}")
            self.full_exec = False
        elif k == "restore":
            self.emit(f"s_mov_{self.b} {self.exec}, -1")
            self.full_exec = True
        elif k == "mask_as_data":
            self.emit(f"v_mov_b32 v{d}, {self.half(rng.choice(self.masks), rng.randrange(2))}")

    def observe(self):
        """Every lane on, then the masks as each lane's bit and the scalars as values."""
        self.emit(f"s_mov_{self.b} {self.save}, {self.exec}")
        self.emit(f"s_mov_{self.b} {self.exec}, -1")
        for i, m in enumerate(self.masks + (self.save, self.vcc)):
            self.emit(f"v_cndmask_b32_e64 v{12 + i}, 0, 1, {m}")
        for i, s in enumerate(SREGS):
            self.emit(f"v_mov_b32 v{16 + i}, s{s}")


def generate_program(rng, n_ops, wave=64, masks_as_data=False, permlane_partial=False):
    g = Gen(rng, wave, masks_as_data, permlane_partial)
    g.prologue()
    for _ in range(n_ops):
        g.op()
    g.observe()
    return g.lines


# ---- translator side --------------------------------------------------------------------------------------------

def _write(path, lines, stores):
    with open(path, "w", encoding="utf-8") as f:
        f.write("; wave_diff case  code 0x0..0x1000\n")
        for line in lines:
            f.write("\t" + line + "\n")
        if stores:
            f.write(f"\tv_lshlrev_b32 v{ADDR}, 6, v{LANE}\n")
            for k, r in enumerate(WORDS):
                f.write(f"\tv_add_nc_u32 v{ADDR + 1}, {4 * k}, v{ADDR}\n")
                f.write(f"\tbuffer_store_dword v{r}, v{ADDR + 1}, s[0:3], 0 offen\n")
        f.write("\ts_endpgm\n")


def translate(name, lines, wave):
    with tempfile.TemporaryDirectory(prefix="wave-diff-") as tmp:
        for d in ("disasm", "fssm"):
            os.mkdir(os.path.join(tmp, d))
        path = os.path.join(tmp, "disasm", name + ".s")
        _write(path, lines, stores=True)
        with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        with open(os.path.join(tmp, "fssm", name + ".fssm"), "wb") as f:
            f.write(fssm(wave))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            spv = rdna2spirv.translate(path)
            gc.collect()
        return spv


def run_spirv(spv, wave):
    """Every lane's WORDS from one workgroup of `wave` invocations in one subgroup of that size."""
    memory = si.run_workgroup(
        spv, wave, wave,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
        push_constants=VSHARP + [0] * 28, memory={})
    return [[memory.get(lane * LANE_BYTES + 4 * k) for k in range(len(WORDS))] for lane in range(wave)]


def compare(model, spirv):
    for lane, (m_words, s_words) in enumerate(zip(model, spirv)):
        for k, (m, s) in enumerate(zip(m_words, s_words)):
            if s is None:
                return f"lane {lane} v{WORDS[k]}: the SPIR-V never stored it"
            if m & M32 != s & M32:
                return f"lane {lane} v{WORDS[k]}: expected 0x{m & M32:08x}, actual 0x{s & M32:08x}"
    return None


def run_one(name, lines, wave=64):
    """("match" | "mismatch" | "skip", detail)."""
    try:
        model = run_model(lines, wave)
    except cpu_ref.Unsupported as e:
        return "skip", f"reference: {e}"          # a ProgramError is a harness error, reported by run_suite
    try:
        spv = translate(name, lines, wave)
    except rdna2spirv.Unsupported as e:
        return "skip", f"translator unsupported: {e}"
    try:
        out = run_spirv(spv, wave)
    except si.UnsupportedConstruct as e:
        return "skip", f"mini_interp unsupported: {e}"
    bad = compare(model, out)
    return ("match", "") if bad is None else ("mismatch", bad)


def run_suite(programs, seed, max_ops, waves=(64,), masks_as_data=False, report=None, permlane_partial=False):
    """`report`, when given, is called with (wave, program index, status, detail) for every mismatch and error."""
    counts = {"pass": 0, "mismatch": 0, "skip": {}, "harness-error": 0}
    first = None
    rng = random.Random(seed)
    for i in range(programs):
        wave = waves[i % len(waves)]
        name = f"cs_wdiff_{i:04d}"
        lines = generate_program(rng, rng.randrange(1, max_ops + 1), wave, masks_as_data, permlane_partial)
        try:
            status, detail = run_one(name, lines, wave)
        except Exception as e:  # noqa: BLE001 - harness and translator crashes must be loud
            status, detail = "error", f"{type(e).__name__}: {e}"
        if status == "match":
            counts["pass"] += 1
            continue
        if status == "skip":
            reason = detail.split(" at pc")[0]
            counts["skip"][reason] = counts["skip"].get(reason, 0) + 1
            continue
        counts["mismatch" if status == "mismatch" else "harness-error"] += 1
        if report:
            report(wave, i, status, detail)
        if first is None:
            first = f"{name} (wave{wave}, seed {seed} program {i}): {detail}\n" + "\n".join("\t" + x for x in lines)
    return counts, first


def observed(body, wave):
    """`body` followed by the observation epilogue: a program the comparison can read back."""
    g = Gen(random.Random(0), wave)
    g.lines = list(body)
    g.observe()
    return g.lines


def self_test():
    lines = observed(["v_mbcnt_lo_u32_b32 v1, -1, 0", "v_mbcnt_hi_u32_b32 v1, -1, v1", "v_mad_u32_u24 v8, v1, 3, 1",
                      "v_readlane_b32 s12, v8, 40", "v_writelane_b32 v9, s12, 5",
                      "v_cmp_gt_u32_e64 s[6:7], 20, v1", "s_bcnt1_i32_b64 s13, s[6:7]"], 64)
    status, detail = run_one("cs_canary_ok", lines, 64)
    if status != "match":
        print(f"canary FAILED: hand-written program did not match ({status}: {detail})")
        return False
    model = run_model(lines, 64)
    if model[0][WORDS.index(16)] != 121 or model[5][WORDS.index(9)] != 121 or model[0][WORDS.index(17)] != 20:
        print("canary FAILED: the reference's readlane/writelane/ballot values are not the hand-derived ones")
        return False
    spv = translate("cs_canary_bad", lines, 64)
    bad = spv.replace("OpBitwiseAnd %uint", "OpBitwiseOr %uint", 1)
    if bad == spv or compare(model, run_spirv(bad, 64)) is None:
        print("canary FAILED: a sabotaged translation was not flagged")
        return False
    status, _ = run_one("cs_canary_skip", lines[:2] + ["v_permlane16_b32 v8, v1, s12, s13 op_sel:[1,0]"], 64)
    if status != "skip":
        print(f"canary FAILED: permlane with op_sel was not a counted skip ({status})")
        return False
    print("canaries: match ok, reference values hand-checked, sabotage detected, skip counted")
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description="CPU differential fuzzer for wave operations: reference vs translated SPIR-V.")
    ap.add_argument("--programs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-ops", type=int, default=12)
    ap.add_argument("--wave", choices=("32", "64", "both"), default="both")
    ap.add_argument("--masks-as-data", action="store_true", help="also mix lane masks and 64-bit data (#210)")
    ap.add_argument("--permlane-partial", action="store_true",
                    help="also v_permlane16/x16 under a partial EXEC (FI = 0, source off: SHADER-74, #580)")
    ap.add_argument("--report-all", action="store_true", help="print one line per mismatch, not only the first")
    ap.add_argument("--self-test", action="store_true", help="run the harness canaries and exit")
    a = ap.parse_args(argv)
    if a.programs < 1 or a.max_ops < 1:
        ap.error("--programs and --max-ops must be at least 1")
    if a.self_test:
        return 0 if self_test() else 1
    waves = (32, 64) if a.wave == "both" else (int(a.wave),)
    report = (lambda w, i, st, d: print(f"  wave{w} program {i}: {st}: {d}", flush=True)) if a.report_all else None
    counts, first = run_suite(a.programs, a.seed, a.max_ops, waves, a.masks_as_data, report, a.permlane_partial)
    skips = sum(counts["skip"].values())
    print(f"{a.programs} programs (seed {a.seed}, wave {a.wave}): {counts['pass']} pass, {counts['mismatch']} "
          f"mismatch, {skips} skipped, {counts['harness-error']} harness errors")
    for reason, n in sorted(counts["skip"].items(), key=lambda x: -x[1]):
        print(f"  skip x{n}: {reason}")
    if first is not None:
        print("FIRST FAILURE:\n" + first)
        return 1 if counts["mismatch"] else 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
