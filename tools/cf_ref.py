"""Wave-level CPU reference for RDNA2 control flow (standard library only).

tools/cpu_ref.py executes straight-line code. This module extends its Machine with what control flow needs and
nothing else: branches, vector and scalar compares, `v_cmpx`, the saveexec family, 64-bit mask moves and the
export. It is the reference side of tools/cf_diff.py and, like cpu_ref, is a model written from the ISA document
(AMD "RDNA 2" Instruction Set Architecture, sections 3 (program flow), 4.4 (EXEC), 6 (SOP*), 8 (VOPC)), not
from the translator.

Model:
  * One wave of 32 or 64 lanes, all in one Machine (cpu_ref's 64-lane mode). A wave32 program runs with
    EXEC[63:32] = 0, so lanes 32-63 never execute, and its masks (VCC, EXEC, compare results) are 32 bits.
  * PC is a byte offset. A branch target is next_pc + 4 * simm16 (sign-extended), as the hardware computes it.
    The listing must carry `; encoding:` bytes so instruction sizes (and so branch targets) are known.
  * Compares write 0 for inactive lanes (VOPC behaviour). On RDNA2 `v_cmpx_*` writes EXEC only, never VCC or
    an SGPR; EXEC = EXEC & compare.
  * SCC: s_cmp_* set it to the comparison; s_and/or/xor/andn2/orn2/nand/nor/xnor set it to result != 0;
    s_*_saveexec set it to (new EXEC != 0); s_add_u32/s_sub_u32 to the carry/borrow; s_add_i32/s_sub_i32 to
    signed overflow; s_cselect reads it and leaves it alone.
  * In wave32 mode VCCZ and EXECZ test the low 32 bits.
  * `exp` records, for every lane whose EXEC bit is set, the four source VGPRs. Lanes that are off at the export
    write nothing, which is the hardware's per-lane valid mask for an `exp ... done vm`.

Everything else (memory, LDS, images, cross-lane ops, 16-bit and float ops beyond cpu_ref's) raises
cpu_ref.Unsupported with the mnemonic and PC.

API:
    run(listing_path, wave=32 | 64, max_steps=...) -> {
        "wave": 32 | 64,
        "exports": {lane: (w0, w1, w2, w3)},      # lanes that were active at the export
        "exec": final EXEC, "vcc": final VCC, "scc": final SCC,
        "sgpr": {index: u32}, "steps": int,
    }
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import cpu_ref  # noqa: E402
from shader_ir import parse_file  # noqa: E402

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF
Unsupported = cpu_ref.Unsupported
ProgramError = cpu_ref.ProgramError

_CMP = {
    "eq": lambda a, b: a == b, "ne": lambda a, b: a != b, "lg": lambda a, b: a != b,
    "lt": lambda a, b: a < b, "le": lambda a, b: a <= b, "gt": lambda a, b: a > b, "ge": lambda a, b: a >= b,
}
_SOP2_LOGIC = {
    "and": lambda a, b, m: a & b, "or": lambda a, b, m: a | b, "xor": lambda a, b, m: a ^ b,
    "andn2": lambda a, b, m: a & ~b & m, "orn2": lambda a, b, m: (a | ~b) & m,
    "nand": lambda a, b, m: ~(a & b) & m, "nor": lambda a, b, m: ~(a | b) & m, "xnor": lambda a, b, m: ~(a ^ b) & m,
}
# s_<op>_saveexec_b32/b64: D = EXEC; EXEC = f(S0, EXEC). (ISA section 6.4, SOP1.)
_SAVEEXEC = {
    "and": lambda s, e, m: s & e, "or": lambda s, e, m: s | e, "xor": lambda s, e, m: s ^ e,
    "andn2": lambda s, e, m: s & ~e & m, "orn2": lambda s, e, m: (s | ~e) & m,
    "nand": lambda s, e, m: ~(s & e) & m, "nor": lambda s, e, m: ~(s | e) & m, "xnor": lambda s, e, m: ~(s ^ e) & m,
    "andn1": lambda s, e, m: ~s & e & m, "orn1": lambda s, e, m: (~s | e) & m,
}
_BRANCHES = {"s_branch", "s_cbranch_scc0", "s_cbranch_scc1", "s_cbranch_vccz", "s_cbranch_vccnz",
             "s_cbranch_execz", "s_cbranch_execnz"}


def _signed(v, bits=32):
    v &= (1 << bits) - 1
    return v - (1 << bits) if v >> (bits - 1) else v


class WaveMachine(cpu_ref.Machine):
    def __init__(self, info, wave=32, max_steps=200000):
        if wave not in (32, 64):
            raise ValueError("wave must be 32 or 64")
        super().__init__(info, {"lanes": 64, "exec": (1 << wave) - 1, "max_steps": max_steps})
        self.wave = wave
        self.wmask = (1 << wave) - 1
        self.exports = {}
        self.by_pc = {}
        for i, ins in enumerate(self.insns):
            if not ins.get("enc"):
                raise ProgramError(f"instruction at index {i} ({ins['op']}) has no encoding bytes; "
                                   "branch targets need instruction sizes")
            self.by_pc[ins["pc"]] = i

    # -- masks ---------------------------------------------------------------------------------------------------
    def _read_mask(self, arg):
        """A wave-sized lane mask from an SGPR (pair), VCC, EXEC or an inline constant."""
        k = arg["k"]
        if k == "sgpr":
            lo = self.sgpr.get(arg["r"], 0)
            if arg.get("n", 1) >= 2:
                return ((self.sgpr.get(arg["r"] + 1, 0) << 32) | lo) & self.wmask
            return lo & self.wmask if self.wave == 32 else lo
        if k == "special":
            t = arg["t"]
            if t == "vcc":
                return self.vcc & self.wmask
            if t == "vcc_lo":
                return self.vcc & MASK32
            if t == "vcc_hi":
                return (self.vcc >> 32) & MASK32
            if t == "exec":
                return self.exec & self.wmask
            if t == "exec_lo":
                return self.exec & MASK32
            if t == "exec_hi":
                return 0 if self.wave == 32 else (self.exec >> 32) & MASK32
            if t == "null":
                return 0
        if k == "imm":
            return arg["v"] & self.wmask
        raise Unsupported(arg.get("t", k), self.pc, "mask operand is not modelled")

    def _write_mask(self, dst, value):
        k = dst["k"]
        if k == "sgpr":
            self.sgpr[dst["r"]] = value & MASK32
            if dst.get("n", 1) >= 2:
                self.sgpr[dst["r"] + 1] = (value >> 32) & MASK32
            return
        if k == "special":
            t = dst["t"]
            if t == "vcc":
                self.vcc = (value & MASK64) if self.wave == 64 else ((self.vcc & ~MASK32 & MASK64) | (value & MASK32))
                return
            if t == "vcc_lo":
                self.vcc = (self.vcc & ~MASK32 & MASK64) | (value & MASK32)
                return
            if t == "vcc_hi":
                self.vcc = (self.vcc & MASK32) | ((value & MASK32) << 32)
                return
            if t == "exec":
                self.exec = value & MASK64
                return
            if t == "exec_lo":
                self.exec = (self.exec & ~MASK32 & MASK64) | (value & MASK32)
                return
            if t == "exec_hi":
                if self.wave == 64:
                    self.exec = (self.exec & MASK32) | ((value & MASK32) << 32)
                return
            if t == "null":
                return
        raise Unsupported(dst.get("t", k), self.pc, "mask destination is not modelled")

    def _wide(self, base):
        return base.endswith("_b64")

    def _check_width(self, base, op):
        if self._wide(base) != (self.wave == 64):
            raise Unsupported(op, self.pc, f"{'64' if self._wide(base) else '32'}-bit mask op in a wave{self.wave} "
                                           "program is not modelled")

    def _active_lanes(self):
        m = self.exec & self.wmask
        return [i for i in range(self.wave) if (m >> i) & 1]

    def _read_special(self, name):
        if name == "exec_hi" and self.wave == 32:
            return 0
        return super()._read_special(name)

    def _write_scalar_dst(self, dst, value):
        if dst.get("k") == "special" and dst.get("t") == "exec_hi" and self.wave == 32:
            return
        super()._write_scalar_dst(dst, value)

    # -- execution -----------------------------------------------------------------------------------------------
    def run(self):
        if not self.insns:
            raise ProgramError("listing has no instructions")
        i = 0
        while not self.halted:
            if i >= len(self.insns):
                raise ProgramError("execution ran past the last instruction without s_endpgm")
            if self.steps >= self.max_steps:
                raise ProgramError(f"step limit {self.max_steps} reached at pc 0x{self.insns[i]['pc']:x}")
            ins = self.insns[i]
            self.steps += 1
            self.pc = ins["pc"]
            self.next_pc = ins["pc"] + len(ins["enc"]) // 2
            target = self.step(ins)
            nxt = self.next_pc if target is None else target
            if not self.halted:
                if nxt not in self.by_pc:
                    raise ProgramError(f"branch from pc 0x{self.pc:x} to 0x{nxt:x}, which is not an instruction")
                i = self.by_pc[nxt]
        return {
            "wave": self.wave,
            "exports": dict(self.exports),
            "exec": self.exec & self.wmask, "vcc": self.vcc & self.wmask, "scc": self.scc,
            "sgpr": dict(sorted(self.sgpr.items())), "steps": self.steps,
        }

    def step(self, ins):
        op = ins["op"]
        base = cpu_ref.base_op(op)
        a = ins["args"]
        if op == "exp":
            return self._export(ins)
        mods = ins.get("mods") or []
        if mods and base not in cpu_ref.FLOAT_OPS:
            raise Unsupported(op, self.pc, f"modifier(s) {', '.join(mods)} are not modelled")
        if base in _BRANCHES:
            return self._branch(base, a)
        if base in ("v_add_nc_u32", "v_sub_nc_u32", "v_subrev_nc_u32", "v_mul_lo_u32"):
            self._expect(ins, 3)
            for lane in self._active_lanes():
                x, y = self._int_src(a[1], lane), self._int_src(a[2], lane)
                r = {"v_add_nc_u32": x + y, "v_sub_nc_u32": x - y, "v_subrev_nc_u32": y - x,
                     "v_mul_lo_u32": x * y}[base]
                self._write_vgpr(a[0], lane, r)
            return None
        if base in ("v_mbcnt_lo_u32_b32", "v_mbcnt_hi_u32_b32"):
            self._expect(ins, 3)
            for lane in self._active_lanes():
                m = self._int_src(a[1], lane)
                if base == "v_mbcnt_lo_u32_b32":
                    below = m & ((1 << min(lane, 32)) - 1)
                else:
                    below = m & ((1 << max(lane - 32, 0)) - 1)
                self._write_vgpr(a[0], lane, below.bit_count() + self._int_src(a[2], lane))
            return None
        if base.startswith(("v_cmp_", "v_cmpx_")):
            return self._vcmp(ins, base)
        if base.startswith("s_cmp_"):
            self._expect(ins, 2)
            cc, _, ty = base[len("s_cmp_"):].partition("_")
            if cc not in _CMP or ty not in ("u32", "i32"):
                raise Unsupported(op, self.pc, "compare is not modelled")
            x, y = self._int_src(a[0], 0, scalar=True), self._int_src(a[1], 0, scalar=True)
            if ty == "i32":
                x, y = _signed(x), _signed(y)
            self.scc = 1 if _CMP[cc](x, y) else 0
            return None
        if base.endswith("_saveexec_b32") or base.endswith("_saveexec_b64"):
            self._check_width(base, op)
            self._expect(ins, 2)
            kind = base[2:].split("_saveexec")[0]
            if kind not in _SAVEEXEC:
                raise Unsupported(op, self.pc, "saveexec variant is not modelled")
            s0 = self._read_mask(a[1])
            old = self.exec & self.wmask
            self._write_mask(a[0], old)
            new = _SAVEEXEC[kind](s0, old, self.wmask) & self.wmask
            self.exec = new if self.wave == 64 else (self.exec & ~MASK32) | new
            self.scc = 1 if new else 0
            return None
        parts = base.split("_")
        if len(parts) == 3 and parts[0] == "s" and parts[1] in _SOP2_LOGIC and parts[2] in ("b32", "b64"):
            self._expect(ins, 3)
            mask_op = self._touches_mask(a)
            if parts[2] == "b64" or mask_op:
                if parts[2] == "b64" or any(x["k"] == "special" and x["t"] in ("exec", "vcc") for x in a):
                    self._check_width(base, op)
                mask = MASK64 if parts[2] == "b64" else MASK32
                x, y = self._read_mask(a[1]) & mask, self._read_mask(a[2]) & mask
                r = _SOP2_LOGIC[parts[1]](x, y, mask) & mask
                self._write_mask(a[0], r)
            else:
                x, y = self._int_src(a[1], 0, scalar=True), self._int_src(a[2], 0, scalar=True)
                r = _SOP2_LOGIC[parts[1]](x, y, MASK32) & MASK32
                self._write_scalar_dst(a[0], r)
            self.scc = 1 if r else 0
            return None
        if base == "s_mov_b64":
            if self.wave != 64:
                raise Unsupported(op, self.pc, "s_mov_b64 in a wave32 program is not modelled")
            self._expect(ins, 2)
            src = a[1]
            v = (src["v"] & MASK64) if src["k"] == "imm" else self._read_mask(src)
            self._write_mask(a[0], v)
            return None
        if base == "s_mov_b32" and self._touches_mask(a):
            if any(x["k"] == "special" and x["t"] in ("exec", "vcc") for x in a):
                self._check_width(base, op)
            self._expect(ins, 2)
            v = a[1]["v"] & MASK32 if a[1]["k"] == "imm" else self._read_mask(a[1]) & MASK32
            self._write_mask(a[0], v)
            return None
        if base in ("s_sub_u32", "s_add_i32", "s_sub_i32"):
            self._expect(ins, 3)
            x, y = self._int_src(a[1], 0, scalar=True), self._int_src(a[2], 0, scalar=True)
            if base == "s_sub_u32":
                r, self.scc = (x - y) & MASK32, 1 if y > x else 0
            else:
                r = (x + y if base == "s_add_i32" else x - y) & MASK32
                sx, sy, sr = _signed(x), _signed(y), _signed(r)
                if base == "s_add_i32":
                    self.scc = 1 if (sx >= 0) == (sy >= 0) and (sr >= 0) != (sx >= 0) else 0
                else:
                    self.scc = 1 if (sx >= 0) != (sy >= 0) and (sr >= 0) != (sx >= 0) else 0
            self._write_scalar_dst(a[0], r)
            return None
        if base == "s_cselect_b32":
            self._expect(ins, 3)
            x, y = self._int_src(a[1], 0, scalar=True), self._int_src(a[2], 0, scalar=True)
            self._write_scalar_dst(a[0], x if self.scc else y)
            return None
        if base == "v_cndmask_b32" and self.wave == 32 and len(a) == 3:
            # VOP2 form: the implicit condition is VCC_LO in wave32 (cpu_ref models a wave64 VCC).
            for lane in self._active_lanes():
                bit = (self.vcc >> lane) & 1
                self._write_vgpr(a[0], lane, self._int_src(a[2], lane) if bit else self._int_src(a[1], lane))
            return None
        super().step(ins)
        return None

    @staticmethod
    def _touches_mask(args):
        return any(x["k"] == "special" and x["t"] in ("exec", "exec_lo", "exec_hi", "vcc", "vcc_lo", "vcc_hi")
                   for x in args)

    def _branch(self, base, a):
        if len(a) != 1 or a[0]["k"] not in ("imm", "label"):
            raise Unsupported(base, self.pc, "branch operand is not an immediate offset")
        off = a[0]["v"] if a[0]["k"] == "imm" else None
        if off is None:
            raise Unsupported(base, self.pc, "symbolic branch labels are not modelled")
        target = self.next_pc + 4 * _signed(off, 16)
        cond = {
            "s_branch": True,
            "s_cbranch_scc0": self.scc == 0, "s_cbranch_scc1": self.scc == 1,
            "s_cbranch_vccz": (self.vcc & self.wmask) == 0, "s_cbranch_vccnz": (self.vcc & self.wmask) != 0,
            "s_cbranch_execz": (self.exec & self.wmask) == 0, "s_cbranch_execnz": (self.exec & self.wmask) != 0,
        }[base]
        return target if cond else None

    def _export(self, ins):
        a = ins["args"]
        bad_mods = [m for m in (ins.get("mods") or []) if m not in ("done", "vm")]
        if bad_mods:
            raise Unsupported("exp", self.pc, f"modifier(s) {', '.join(bad_mods)} are not modelled")
        if len(a) != 5 or a[0].get("t") != "mrt0" or any(x["k"] != "vgpr" for x in a[1:]):
            raise Unsupported("exp", self.pc, "only `exp mrt0 vA, vB, vC, vD` is modelled")
        for lane in self._active_lanes():
            self.exports[lane] = tuple(self._vgpr(x["r"], lane) for x in a[1:])
        return None

    def _vcmp(self, ins, base):
        op, a = ins["op"], ins["args"]
        cmpx = base.startswith("v_cmpx_")
        body = base[len("v_cmpx_"):] if cmpx else base[len("v_cmp_"):]
        cc, _, ty = body.partition("_")
        if cc not in _CMP or ty not in ("u32", "i32"):
            raise Unsupported(op, self.pc, "compare is not modelled")
        srcs = a if cmpx else a[1:]
        if cmpx and len(a) == 3:
            raise Unsupported(op, self.pc, "an RDNA2 v_cmpx has no scalar destination")
        if len(srcs) != 2:
            raise Unsupported(op, self.pc, f"expected 2 compare sources, got {len(srcs)}")
        mask = 0
        for lane in self._active_lanes():
            x, y = self._int_src(srcs[0], lane), self._int_src(srcs[1], lane)
            if ty == "i32":
                x, y = _signed(x), _signed(y)
            if _CMP[cc](x, y):
                mask |= 1 << lane
        if cmpx:
            self.exec = mask if self.wave == 64 else (self.exec & ~MASK32) | mask
            return None
        dst = a[0]
        if self.wave == 32 and dst["k"] == "special" and dst["t"] == "vcc":
            dst = {"k": "special", "t": "vcc_lo"}
        if self.wave == 64 and dst["k"] == "sgpr" and dst.get("n", 1) < 2:
            raise Unsupported(op, self.pc, "a wave64 compare needs an SGPR pair destination")
        self._write_mask(dst, mask)
        return None


def run(listing_path, wave=32, max_steps=200000):
    cpu_ref._require_header(listing_path)
    return WaveMachine(parse_file(listing_path), wave=wave, max_steps=max_steps).run()
