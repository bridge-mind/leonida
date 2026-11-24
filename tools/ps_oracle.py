r"""Run hand-written RDNA2 pixel-shader ISA on an AMD GPU: a hardware oracle for what only a pixel shader can do.

tools/hw_oracle launches compute kernels; v_interp_*, the parameter data the SPI writes into LDS, exports and helper
lanes need a real draw. This tool gets one through Vulkan's VK_KHR_pipeline_binary (the AMD Windows driver exposes
it, with uncompressed data):

  1. vk_harness builds a pipeline from a template vertex + pixel shader (SPIR-V) and writes its binary
     (VKH_BINARY_OUT). On AMD that binary is one PAL ELF: `.text` holds `_amdgpu_gs_main` (the vertex stage) and
     `_amdgpu_ps_main`, the `.note` holds the PAL metadata (msgpack): register counts and the SPI setup
     (SPI_PS_INPUT_ENA/ADDR, the parameter layout, the export format), which the driver programs from the metadata.
  2. The ISA under test is assembled with AMD clang (gfx1036 or the --mcpu given) and replaces `_amdgpu_ps_main`;
     `.text` grows when it is longer, and the pixel stage's VGPR/SGPR counts in the metadata are raised to cover the
     registers it names.
  3. vk_harness builds the pipeline from the patched binary (VKH_BINARY_IN), draws one full-screen triangle into an
     RGBA32F target and reads it back, so every pixel's mrt0 export comes back as four raw dwords.

The template decides what the hardware sets up before the code runs: which barycentrics are in VGPRs, which user
SGPRs hold what (the PS sets M0 from one of them for v_interp), how many parameters the SPI writes into LDS and in
what format, and the export format of mrt0 (32_ABGR for the harness' RGBA32F target). --show-template prints the
template's own ISA and that part of its metadata, so the code under test can follow the same conventions.
Templates: `const` (no inputs), `interp2` (two smooth vec4 varyings: location 0 = (u, v, 0, 1) and location 1 =
(u, v, 0, 0) with u, v = 0..2 across the screen, as scripts/run_gpu_tests.py's vs_tri), or --vs/--ps files
(.spv, or .spvasm assembled with spirv-as). --format picks another color target (the driver then picks the export
format for it, as for a game pipeline); the readback is that target's texels as stored. The selected format,
including the default RGBA32F, takes precedence over a VKH_COLOR_FORMAT inherited from the shell.

Checked on an AMD Radeon iGPU (gfx1036, Windows driver 26.10.1): the driver accepts a patched binary and the patched
code runs as the pixel shader (no checksum over the code). The pixel stage there is compiled as wave64.

On Linux, RADV (Mesa) returns one binary per stage instead (radv_shader_serialize's layout, see the RADV section
below); the pixel stage's code, sizes and register counts are found by their values and patched there. RADV takes
GFX10 (RDNA1/RDNA2) devices only (GFX11+ is refused: no v_interp), --set is PAL-only, and Mesa may merge a template's
varyings (interp2 becomes one parameter, attr0 = (2u, 2v, 0, 1)). Checked on gfx1036 with Mesa 26.3.1.

Usage:
    python -I tools/ps_oracle.py --build <runtime build> --template interp2 --show-template
    python -I tools/ps_oracle.py --build <runtime build> --template interp2 --asm body.s [--size 8 8] [--json out.json]

Select the AMD GPU with LEONIDA_VK_DEVICE (e.g. radeon). Standard library only; needs vk_harness from the runtime
build, AMD clang (LEONIDA_AMD_CLANG, as tools/hw_oracle), llvm-objdump (llvm-mc on RADV) next to it for
--show-template, and spirv-as on PATH for the templates. Exit codes: 0 ran, 1 the GPU run failed, 2 usage or a
bad input, 77 the device or a tool is missing (no VK_KHR_pipeline_binary, neither a PAL nor a RADV binary, a RADV
GPU other than GFX10, no clang).
"""
import argparse
import collections
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from report_paths import same_report_path  # noqa: E402

PS_SYMBOL = "_amdgpu_ps_main"
S_CODE_END = struct.pack("<I", 0xBF9F0000)
W64_BIT = 0x2000                       # s_version's UC_VERSION_W64_BIT


class Unavailable(Exception):
    """A tool or a device feature is missing: exit 77."""


# ---- msgpack, the subset PAL metadata uses ----

class F32(float):
    """A float that was (and is written back as) msgpack float32."""


def mp_load(b, i=0):
    """(value, next index). Maps become dicts (key order kept), floats keep their width (F32 / float)."""
    t = b[i]
    if t <= 0x7F:
        return t, i + 1
    if t >= 0xE0:
        return t - 0x100, i + 1
    if 0x80 <= t <= 0x8F or t in (0xDE, 0xDF):
        n, i = (t & 0x0F, i + 1) if t <= 0x8F else (struct.unpack_from(">H" if t == 0xDE else ">I", b, i + 1)[0],
                                                     i + (3 if t == 0xDE else 5))
        d = {}
        for _ in range(n):
            k, i = mp_load(b, i)
            v, i = mp_load(b, i)
            d[k] = v
        return d, i
    if 0x90 <= t <= 0x9F or t in (0xDC, 0xDD):
        n, i = (t & 0x0F, i + 1) if t <= 0x9F else (struct.unpack_from(">H" if t == 0xDC else ">I", b, i + 1)[0],
                                                     i + (3 if t == 0xDC else 5))
        out = []
        for _ in range(n):
            v, i = mp_load(b, i)
            out.append(v)
        return out, i
    if 0xA0 <= t <= 0xBF or t in (0xD9, 0xDA, 0xDB):
        if t <= 0xBF:
            n, i = t & 0x1F, i + 1
        else:
            w = {0xD9: 1, 0xDA: 2, 0xDB: 4}[t]
            n, i = int.from_bytes(b[i + 1:i + 1 + w], "big"), i + 1 + w
        return b[i:i + n].decode("utf-8"), i + n
    if t in (0xC4, 0xC5, 0xC6):
        w = {0xC4: 1, 0xC5: 2, 0xC6: 4}[t]
        n = int.from_bytes(b[i + 1:i + 1 + w], "big")
        return bytes(b[i + 1 + w:i + 1 + w + n]), i + 1 + w + n
    if t == 0xC0:
        return None, i + 1
    if t in (0xC2, 0xC3):
        return t == 0xC3, i + 1
    if t == 0xCA:
        return F32(struct.unpack_from(">f", b, i + 1)[0]), i + 5
    if t == 0xCB:
        return struct.unpack_from(">d", b, i + 1)[0], i + 9
    ints = {0xCC: ">B", 0xCD: ">H", 0xCE: ">I", 0xCF: ">Q", 0xD0: ">b", 0xD1: ">h", 0xD2: ">i", 0xD3: ">q"}
    if t in ints:
        return struct.unpack_from(ints[t], b, i + 1)[0], i + 1 + struct.calcsize(ints[t])
    raise ValueError(f"msgpack type 0x{t:02x} at {i} is not supported")


def _mp_len(n, small, small_max, wide):
    if n <= small_max:
        return bytes([small | n])
    for code, fmt, top in wide:
        if n < top:
            return bytes([code]) + struct.pack(fmt, n)
    raise ValueError("too long")


def mp_dump(v):
    if v is None:
        return b"\xc0"
    if v is True or v is False:
        return b"\xc3" if v else b"\xc2"
    if isinstance(v, int):
        if 0 <= v <= 0x7F:
            return bytes([v])
        if -32 <= v < 0:
            return bytes([v + 0x100])
        for code, fmt, lo, hi in ((0xCC, ">B", 0, 1 << 8), (0xCD, ">H", 0, 1 << 16), (0xCE, ">I", 0, 1 << 32),
                                  (0xCF, ">Q", 0, 1 << 64), (0xD0, ">b", -(1 << 7), 0), (0xD1, ">h", -(1 << 15), 0),
                                  (0xD2, ">i", -(1 << 31), 0), (0xD3, ">q", -(1 << 63), 0)):
            if lo <= v < hi:
                return bytes([code]) + struct.pack(fmt, v)
        raise ValueError(f"integer {v} out of range")
    if isinstance(v, F32):
        return b"\xca" + struct.pack(">f", v)
    if isinstance(v, float):
        return b"\xcb" + struct.pack(">d", v)
    if isinstance(v, str):
        s = v.encode("utf-8")
        if len(s) <= 31:
            return bytes([0xA0 | len(s)]) + s
        return _mp_len(len(s), 0, -1, ((0xD9, ">B", 1 << 8), (0xDA, ">H", 1 << 16), (0xDB, ">I", 1 << 32))) + s
    if isinstance(v, (bytes, bytearray)):
        return _mp_len(len(v), 0, -1, ((0xC4, ">B", 1 << 8), (0xC5, ">H", 1 << 16), (0xC6, ">I", 1 << 32))) + bytes(v)
    if isinstance(v, list):
        return _mp_len(len(v), 0x90, 15, ((0xDC, ">H", 1 << 16), (0xDD, ">I", 1 << 32))) + b"".join(map(mp_dump, v))
    if isinstance(v, dict):
        head = _mp_len(len(v), 0x80, 15, ((0xDE, ">H", 1 << 16), (0xDF, ">I", 1 << 32)))
        return head + b"".join(mp_dump(k) + mp_dump(x) for k, x in v.items())
    raise TypeError(f"cannot write {type(v).__name__} as msgpack")


# ---- the PAL ELF of a pipeline binary ----

_SH = "<IIQQQQIIQQ"


class Elf:
    """An ELF64 with no program headers (what the AMD driver's pipeline binary is): sections kept in file order,
    rewritten with any section's data replaced."""

    def __init__(self, data):
        data = bytes(data)
        if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
            raise Unavailable("the pipeline binary is not a 64-bit little-endian ELF (not an AMD PAL binary)")
        if struct.unpack_from("<H", data, 0x12)[0] != 0xE0:
            raise Unavailable("the pipeline binary is not an AMDGPU ELF")
        if struct.unpack_from("<H", data, 0x38)[0]:
            raise ValueError("an ELF with program headers is not supported")
        self.head = bytearray(data[:64])
        shoff = struct.unpack_from("<Q", data, 0x28)[0]
        shentsize, shnum, self.shstrndx = struct.unpack_from("<HHH", data, 0x3A)
        self.hdrs = [list(struct.unpack_from(_SH, data, shoff + i * shentsize)) for i in range(shnum)]
        self.data = [bytearray(data[h[4]:h[4] + h[5]]) if h[1] != 8 else bytearray() for h in self.hdrs]
        strtab = self.data[self.shstrndx]
        self.names = [strtab[h[0]:strtab.index(b"\0", h[0])].decode() for h in self.hdrs]

    def index(self, name):
        if name not in self.names:
            raise ValueError(f"no {name} section")
        return self.names.index(name)

    def symbols(self):
        """[(name, value, size, symtab entry index)] of .symtab."""
        st = self.index(".symtab")
        strs = self.data[self.hdrs[st][6]]
        out = []
        for i in range(len(self.data[st]) // 24):
            name, _info, _other, _shndx, value, size = struct.unpack_from("<IBBHQQ", self.data[st], i * 24)
            out.append((strs[name:strs.index(b"\0", name)].decode(), value, size, i))
        return out

    def set_symbol_size(self, entry, size):
        struct.pack_into("<Q", self.data[self.index(".symtab")], entry * 24 + 16, size)

    def to_bytes(self):
        order = sorted((i for i in range(1, len(self.hdrs)) if self.hdrs[i][1] != 8), key=lambda i: self.hdrs[i][4])
        out = bytearray(self.head)
        for i in order:
            self.hdrs[i][4] = len(out)
            self.hdrs[i][5] = len(self.data[i])
            out += self.data[i]
        while len(out) % 8:
            out += b"\0"
        struct.pack_into("<Q", out, 0x28, len(out))
        for h in self.hdrs:
            out += struct.pack(_SH, *h)
        return bytes(out)

    # The PAL metadata note: one NT_AMDGPU_METADATA (32) with owner "AMDGPU" and a msgpack desc.
    def metadata(self):
        note = self.data[self.index(".note")]
        namesz, descsz, typ = struct.unpack_from("<III", note, 0)
        if typ != 32 or note[12:12 + namesz].rstrip(b"\0") != b"AMDGPU":
            raise Unavailable("no AMDGPU metadata note (not a PAL binary)")
        at = 12 + (namesz + 3) // 4 * 4
        return mp_load(note, at)[0]

    def set_metadata(self, meta):
        desc = mp_dump(meta)
        name = b"AMDGPU\0"
        note = struct.pack("<III", len(name), len(desc), 32) + name + b"\0" * (-len(name) % 4) + desc
        self.data[self.index(".note")] = bytearray(note + b"\0" * (-len(note) % 4))


def read_binaries(path):
    """vk_harness's VKH_BINARY_OUT file -> [(key size, 32 key bytes, data)]."""
    with open(path, "rb") as f:
        d = f.read()
    n, at, out = struct.unpack_from("<I", d, 0)[0], 4, []
    for _ in range(n):
        ks = struct.unpack_from("<I", d, at)[0]
        key = d[at + 4:at + 36]
        size = struct.unpack_from("<Q", d, at + 36)[0]
        out.append((ks, key, d[at + 44:at + 44 + size]))
        at += 44 + size
    return out


def write_binaries(path, bins):
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(bins)))
        for ks, key, data in bins:
            f.write(struct.pack("<I", ks) + key + struct.pack("<Q", len(data)) + data)


def ps_stage(meta):
    """The pixel hardware stage's metadata map (amdpal.pipelines[0].hardware_stages[.ps])."""
    try:
        return meta["amdpal.pipelines"][0][".hardware_stages"][".ps"]
    except (KeyError, IndexError, TypeError):
        raise Unavailable("the PAL metadata has no pixel hardware stage") from None


_REG = re.compile(r"(?<![\w.])([vs])(?:(\d+)|\[\d+:(\d+)\])(?![\w.])")


def regs_used(lines):
    """(VGPRs, SGPRs) the lines name: one past the highest v<N> / v[a:b] and s<N> / s[a:b]."""
    top = {"v": 0, "s": 0}
    for line in lines:
        for kind, single, last in _REG.findall(line.split(";", 1)[0]):
            top[kind] = max(top[kind], int(last or single) + 1)
    return top["v"], top["s"]


def set_meta(pipe, path, value):
    """Set one value in the pipeline's metadata map: path is '/'-separated keys from amdpal.pipelines[0] (list
    indices as numbers), e.g. ".graphics_registers/.spi_ps_input_cntl/0/.fp16_interp_mode"."""
    keys = [int(k) if k.isdigit() else k for k in path.split("/") if k]
    node = pipe
    for k in keys[:-1]:
        node = node[k]
    if keys[-1] not in node if isinstance(node, dict) else not 0 <= keys[-1] < len(node):
        raise ValueError(f"no metadata entry {path!r}")
    old = node[keys[-1]]
    if isinstance(old, F32):
        value = F32(value)
    node[keys[-1]] = value


def patch_ps(elf_bytes, code, lines=(), sets=()):
    """The ELF with `_amdgpu_ps_main` replaced by `code` (raw instruction bytes), padded with s_code_end, the pixel
    stage's VGPR/SGPR counts raised to cover the registers `lines` name, and each (path, value) of `sets` written
    into the metadata (set_meta). `_amdgpu_ps_main` must be the last function of .text (it is in the AMD driver's
    binaries)."""
    elf = Elf(elf_bytes)
    syms = {s[0]: s for s in elf.symbols()}
    if PS_SYMBOL not in syms:
        raise Unavailable(f"no {PS_SYMBOL} in the pipeline binary")
    _, start, _, entry = syms[PS_SYMBOL]
    if any(v > start for n, v, _s, _e in syms.values() if n and n != PS_SYMBOL):
        raise ValueError(f"{PS_SYMBOL} is not the last function of .text")
    ti = elf.index(".text")
    code = bytes(code) + S_CODE_END * ((-len(code) % 64) // 4 + 64)      # the hardware prefetches past the end
    elf.data[ti] = bytearray(elf.data[ti][:start]) + bytearray(code)
    elf.set_symbol_size(entry, len(code) - 64 * 4)
    v, s = regs_used(lines)
    meta = elf.metadata()
    stage = ps_stage(meta)
    changed = bool(sets)
    for path, value in sets:
        set_meta(meta["amdpal.pipelines"][0], path, value)
    if v > stage.get(".vgpr_count", 0) or s > stage.get(".sgpr_count", 0):
        stage[".vgpr_count"] = max(stage.get(".vgpr_count", 0), v)
        stage[".sgpr_count"] = max(stage.get(".sgpr_count", 0), s)
        changed = True
    if changed:
        elf.set_metadata(meta)
    return elf.to_bytes()


def ps_code(elf_bytes):
    """(start, bytes) of `_amdgpu_ps_main` in .text (to the end of .text, s_code_end padding included)."""
    elf = Elf(elf_bytes)
    syms = {s[0]: s for s in elf.symbols()}
    start = syms[PS_SYMBOL][1]
    return start, bytes(elf.data[elf.index(".text")][start:])


def is_wave64(elf_bytes):
    """The pixel stage's wave size from its first instruction, s_version (UC_VERSION_W64_BIT)."""
    _, code = ps_code(elf_bytes)
    word = struct.unpack_from("<I", code, 0)[0]
    if word >> 23 != 0x161:          # SOPK opcode 1 (s_version) in GFX10: 0xB080xxxx
        return None
    return bool(word & W64_BIT)


# ---- RADV (Mesa) stage binaries ----
# RADV hands back one binary per stage, as radv_shader_serialize writes it (Mesa 26.3, radv_pipeline_cache.c; the
# structs in radv_shader.h): uint32 type (0 = legacy), struct ac_shader_config (ac_binary.h: num_sgprs, num_vgprs
# first, rsrc1 later), struct radv_shader_info (its size changes between Mesa versions; wave_size, a byte, stage,
# next_stage sit together in it), uint32 total_size (padded to 8), then code_size, exec_size, ir_size, disasm_size,
# stats_size, debug_info_size, then the data: stats, code, ir, disasm, debug info. Debug strings may follow
# total_size. RADV builds an imported binary with radv_shader_create_uncached (no cache lookup) and programs
# SPI_SHADER_PGM_RSRC1_PS from config.rsrc1, whose VGPRS field is (num_vgprs - 1) / (wave32 ? 8 : 4)
# (radv_postprocess_binary_config). Only the two leading config fields are read at a fixed place; the rest is found
# from values that must agree with each other, so a Mesa with another layout is refused instead of misread.

RadvLayout = collections.namedtuple("RadvLayout", "code code_size exec_size sizes_at total_at vgprs sgprs rsrc1_at "
                                                  "wave is_ps")
MESA_FRAGMENT, MESA_NONE = 4, 0xFF
# Mesa's chip names (amd_family.c) of GFX10 with their LLVM targets (ac_get_llvm_processor_name)
RADV_GFX10 = {"NAVI10": "gfx1010", "NAVI12": "gfx1011", "NAVI14": "gfx1012", "GFX1013": "gfx1013",
              "NAVI21": "gfx1030", "NAVI22": "gfx1031", "NAVI23": "gfx1032", "VANGOGH": "gfx1033",
              "NAVI24": "gfx1034", "REMBRANDT": "gfx1035", "RAPHAEL_MENDOCINO": "gfx1036"}


def radv_layout(data):
    """Where the code, its sizes and the register counts are in one RADV stage binary (RadvLayout), or Unavailable."""
    u32 = lambda at: struct.unpack_from("<I", data, at)[0]   # noqa: E731
    if len(data) < 64 or u32(0) != 0:
        raise Unavailable("not a legacy RADV shader binary")
    hits = []
    for at in range(8, len(data) - 23, 4):
        code, exe, ir, dis, stats, dbg = struct.unpack_from("<6I", data, at)
        end = at + 24 + stats + code + ir + dis + dbg
        if code and not code % 4 and not exe % 4 and 0 < exe <= code and end <= len(data):
            hits += [(at, t) for t in (at - 4, at - 8) if u32(t) == end]
    if len(hits) != 1:
        raise Unavailable("the RADV binary's code sizes were not found (an unknown Mesa layout)")
    sizes_at, total_at = hits[0]
    code_size, exec_size, stats = (u32(sizes_at), u32(sizes_at + 4), u32(sizes_at + 16))
    stage = [i for i in range(8, total_at - 3)
             if data[i] in (32, 64) and data[i + 2:i + 4] == bytes([MESA_FRAGMENT, MESA_NONE])]
    sgprs, vgprs = u32(4), u32(8)
    code_at = sizes_at + 24 + stats
    if len(stage) != 1:      # another stage (the vertex one): nothing in it is patched
        return RadvLayout(code_at, code_size, exec_size, sizes_at, total_at, vgprs, sgprs, None, None, False)
    wave = data[stage[0]]
    granule = 8 if wave == 32 else 4
    # rsrc1: VGPRS = (num_vgprs - 1) / granule and FLOAT_MODE = config.float_mode, an earlier config word
    rsrc1 = [k for k in range(12, min(total_at, 96), 4)
             if u32(k) & 0x3F == (vgprs - 1) // granule and u32(k) >> 12 & 0xFF
             and any(u32(j) == u32(k) >> 12 & 0xFF for j in range(12, k, 4))]
    if len(rsrc1) != 1:
        raise Unavailable("the RADV binary's rsrc1 was not found (an unknown Mesa layout)")
    return RadvLayout(code_at, code_size, exec_size, sizes_at, total_at, vgprs, sgprs, rsrc1[0], wave, True)


def radv_ps_index(bins):
    """Which of RADV's stage binaries is the pixel stage."""
    ps = [i for i, (_ks, _key, data) in enumerate(bins) if radv_layout(data).is_ps]
    if len(ps) != 1:
        raise Unavailable(f"{len(ps)} of the {len(bins)} RADV binaries are a pixel stage (expected one)")
    return ps[0]


def radv_arch(device):
    """The LLVM target of a RADV device name ('... (RADV RAPHAEL_MENDOCINO)'), Unavailable when it is not GFX10."""
    m = re.search(r"\(RADV ([A-Z0-9_]+)\)", device or "")
    if not m:
        raise Unavailable(f"the pipeline came back as several binaries and the device ({device or '?'}) is not RADV")
    if m.group(1) not in RADV_GFX10:
        raise Unavailable(f"unsupported architecture: {m.group(1)} ({device}); on RADV this tool runs GFX10 "
                          "(RDNA1/RDNA2) pixel shaders only (GFX11+ interpolates with lds_param_load, not v_interp)")
    return RADV_GFX10[m.group(1)]


def patch_radv_ps(data, code, lines=()):
    """The RADV pixel-stage binary with its code replaced by `code` (raw instruction bytes, padded with s_code_end),
    the sizes fixed and num_vgprs/num_sgprs and rsrc1's VGPRS raised to cover the registers `lines` name."""
    lay = radv_layout(data)
    if not lay.is_ps:
        raise ValueError("not the pixel-stage binary")
    padded = bytes(code) + S_CODE_END * ((-len(code) % 64) // 4 + 64)    # the hardware prefetches past the end
    out = bytearray(data[:lay.code]) + padded + data[lay.code + lay.code_size:]
    put = lambda at, v: struct.pack_into("<I", out, at, v)               # noqa: E731
    put(lay.total_at, struct.unpack_from("<I", data, lay.total_at)[0] + len(padded) - lay.code_size)
    put(lay.sizes_at, len(padded))
    put(lay.sizes_at + 4, len(code))
    v, s = regs_used(lines)
    granule = 8 if lay.wave == 32 else 4
    vgprs = max(lay.vgprs, -(-v // granule) * granule)
    put(8, vgprs)
    put(4, max(lay.sgprs, s))
    rsrc1 = struct.unpack_from("<I", data, lay.rsrc1_at)[0]
    put(lay.rsrc1_at, rsrc1 & ~0x3F | (vgprs - 1) // granule)
    return bytes(out)


# ---- templates ----

VS_TRI = """\
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %main "main" %vid %pos %out0 %out1
               OpDecorate %vid BuiltIn VertexIndex
               OpDecorate %pos BuiltIn Position
               OpDecorate %out0 Location 0
               OpDecorate %out1 Location 1
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %uint = OpTypeInt 32 0
        %int = OpTypeInt 32 1
      %float = OpTypeFloat 32
         %v4 = OpTypeVector %float 4
       %p_in = OpTypePointer Input %int
      %p_out = OpTypePointer Output %v4
         %u3 = OpConstant %uint 3
        %arr = OpTypeArray %v4 %u3
      %p_arr = OpTypePointer Function %arr
       %p_v4 = OpTypePointer Function %v4
        %fm1 = OpConstant %float -1
         %f0 = OpConstant %float 0
         %f1 = OpConstant %float 1
         %f2 = OpConstant %float 2
         %f3 = OpConstant %float 3
         %p0 = OpConstantComposite %v4 %fm1 %fm1 %f0 %f1
         %p1 = OpConstantComposite %v4 %f3 %fm1 %f0 %f1
         %p2 = OpConstantComposite %v4 %fm1 %f3 %f0 %f1
         %pa = OpConstantComposite %arr %p0 %p1 %p2
         %a0 = OpConstantComposite %v4 %f0 %f0 %f0 %f1
         %a1 = OpConstantComposite %v4 %f2 %f0 %f0 %f1
         %a2 = OpConstantComposite %v4 %f0 %f2 %f0 %f1
         %aa = OpConstantComposite %arr %a0 %a1 %a2
         %b0 = OpConstantComposite %v4 %f0 %f0 %f0 %f0
         %b1 = OpConstantComposite %v4 %f2 %f0 %f0 %f0
         %b2 = OpConstantComposite %v4 %f0 %f2 %f0 %f0
         %ba = OpConstantComposite %arr %b0 %b1 %b2
        %vid = OpVariable %p_in Input
        %pos = OpVariable %p_out Output
       %out0 = OpVariable %p_out Output
       %out1 = OpVariable %p_out Output
       %main = OpFunction %void None %fn
          %e = OpLabel
         %vp = OpVariable %p_arr Function
         %va = OpVariable %p_arr Function
         %vb = OpVariable %p_arr Function
               OpStore %vp %pa
               OpStore %va %aa
               OpStore %vb %ba
          %i = OpLoad %int %vid
         %xp = OpAccessChain %p_v4 %vp %i
         %xv = OpLoad %v4 %xp
               OpStore %pos %xv
         %yp = OpAccessChain %p_v4 %va %i
         %yv = OpLoad %v4 %yp
               OpStore %out0 %yv
         %zp = OpAccessChain %p_v4 %vb %i
         %zv = OpLoad %v4 %zp
               OpStore %out1 %zv
               OpReturn
               OpFunctionEnd
"""

_PS_HEAD = """\
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" {ios}
               OpExecutionMode %main OriginUpperLeft
               OpDecorate %out Location 0
{decor}       %void = OpTypeVoid
         %fn = OpTypeFunction %void
      %float = OpTypeFloat 32
         %v4 = OpTypeVector %float 4
       %p_in = OpTypePointer Input %v4
      %p_out = OpTypePointer Output %v4
        %out = OpVariable %p_out Output
"""

TEMPLATES = {
    # mrt0 = a constant: no inputs at all (exports, EXEC, helper lanes)
    "const": _PS_HEAD.format(ios="%out", decor="") + """\
         %f1 = OpConstant %float 1
          %c = OpConstantComposite %v4 %f1 %f1 %f1 %f1
       %main = OpFunction %void None %fn
          %e = OpLabel
               OpStore %out %c
               OpReturn
               OpFunctionEnd
""",
    # mrt0 = location 0 + location 1: two smooth vec4 parameters, perspective barycentrics
    "interp2": _PS_HEAD.format(ios="%in0 %in1 %out", decor="""\
               OpDecorate %in0 Location 0
               OpDecorate %in1 Location 1
""") + """\
        %in0 = OpVariable %p_in Input
        %in1 = OpVariable %p_in Input
       %main = OpFunction %void None %fn
          %e = OpLabel
          %a = OpLoad %v4 %in0
          %b = OpLoad %v4 %in1
          %s = OpFAdd %v4 %a %b
               OpStore %out %s
               OpReturn
               OpFunctionEnd
""",
}


TEMPLATE_CONST_INT = """\
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %out
               OpExecutionMode %main OriginUpperLeft
               OpDecorate %out Location 0
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
        %int = OpTypeInt 32 %int_kind
         %v4 = OpTypeVector %int 4
      %p_out = OpTypePointer Output %v4
        %out = OpVariable %p_out Output
         %i1 = OpConstant %int 1
          %c = OpConstantComposite %v4 %i1 %i1 %i1 %i1
       %main = OpFunction %void None %fn
          %e = OpLabel
               OpStore %out %c
               OpReturn
               OpFunctionEnd
"""


def spirv(path_or_text, out, work):
    """A .spv as is; SPIR-V assembly (a .spvasm path or template text) through spirv-as."""
    if not path_or_text.lstrip().startswith("Op") and path_or_text.endswith(".spv"):
        return path_or_text
    src = path_or_text
    if not path_or_text.lstrip().startswith("Op"):
        with open(path_or_text, encoding="utf-8") as f:
            src = f.read()
    asm_path = os.path.join(work, os.path.basename(out) + "asm")
    with open(asm_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(src)
    try:
        r = subprocess.run(["spirv-as", "--target-env", "vulkan1.2", asm_path, "-o", out], capture_output=True,
                           text=True)
    except OSError:
        raise Unavailable("spirv-as is not on PATH") from None
    if r.returncode:
        raise ValueError(f"spirv-as: {r.stderr.strip()}")
    return out


# ---- running ----

def harness(build):
    for name in ("vk_harness.exe", "vk_harness"):
        p = os.path.join(build, name)
        if os.path.exists(p):
            return p
    raise Unavailable(f"no vk_harness in {build}")


# --format: the color target (VkFormat number, bytes per texel, component kind). The driver picks the export format
# (spi_shader_col_format) for it, as for a game pipeline with that render target.
FORMATS = {
    "rgba32f": (109, 16, "float"), "rgba32u": (107, 16, "uint"), "rgba32i": (108, 16, "sint"),
    "rg32f": (103, 8, "float"), "r32f": (100, 4, "float"), "r32u": (98, 4, "uint"),
    "rgba16f": (97, 8, "float"), "rgba16unorm": (91, 8, "float"), "rgba16snorm": (92, 8, "float"),
    "rgba16u": (95, 8, "uint"), "rgba16i": (96, 8, "sint"), "rg16f": (83, 4, "float"), "rg16unorm": (77, 4, "float"),
    "rgba8unorm": (37, 4, "float"), "rgba8snorm": (38, 4, "float"), "rgba8u": (41, 4, "uint"),
    "rgb10a2unorm": (64, 4, "float"),
}


def run_harness(exe, vs, ps, out_raw, size, env_extra, fmt="rgba32f"):
    env = dict(os.environ, **env_extra)
    number, nbytes, _ = FORMATS[fmt]
    env["VKH_COLOR_FORMAT"] = f"{number}:{nbytes}"
    r = subprocess.run([exe, vs, ps, out_raw, str(size[0]), str(size[1])], capture_output=True, text=True, env=env)
    if r.returncode == 77:
        raise Unavailable(r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "vk_harness: unsupported")
    return r


def assemble(lines, mcpu, wave64, work):
    """Raw instruction bytes of the lines, through AMD clang (an amdpal object, its .text)."""
    from hw_oracle import runner
    try:
        clang = runner.find_clang()
    except runner.Unavailable as e:
        raise Unavailable(str(e)) from None
    s = os.path.join(work, "ps.s")
    o = os.path.join(work, "ps.o")
    with open(s, "w", encoding="utf-8", newline="\n") as f:
        f.write(".text\n" + "".join(f"  {ln}\n" for ln in lines))
    flags = ["-mwavefrontsize64"] if wave64 else []
    r = subprocess.run([clang, "--target=amdgcn-amd-amdpal", f"-mcpu={mcpu}", *flags, "-c", s, "-o", o],
                       capture_output=True, text=True)
    if r.returncode:
        raise ValueError("the assembler rejected the code:\n" + r.stderr.strip())
    with open(o, "rb") as f:
        obj = Elf(f.read())
    if any(n.startswith(".rel") and n.endswith(".text") for n in obj.names):
        raise ValueError("the code needs relocations (a symbol it does not define); use plain branches and literals")
    return bytes(obj.data[obj.index(".text")])


def read_lines(path):
    lines = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.split(";", 1)[0].split("//", 1)[0].strip()
            if ln:
                lines.append(ln)
    return lines


def pixels(raw_path, size, texel_bytes=16):
    """Each texel of the readback as its little-endian dwords (one dword for a 4-byte format)."""
    with open(raw_path, "rb") as f:
        d = f.read()
    n = size[0] * size[1]
    if len(d) != texel_bytes * n:
        raise RuntimeError(f"readback is {len(d)} bytes, expected {texel_bytes * n}")
    w = texel_bytes // 4
    return [list(struct.unpack_from(f"<{w}I", d, texel_bytes * i)) for i in range(n)]


class Oracle:
    """One template pipeline, built once; run() patches its pixel shader and draws."""

    def __init__(self, build, vs=None, ps=None, template="interp2", size=(8, 8), mcpu="gfx1036", work=None,
                 fmt="rgba32f"):
        if fmt not in FORMATS:
            raise ValueError(f"unknown format {fmt!r} (known: {', '.join(FORMATS)})")
        self.fmt, self.texel_bytes = fmt, FORMATS[fmt][1]
        self.exe = harness(build)
        self.size = tuple(size)
        self.mcpu = mcpu
        self._tmp = None if work else tempfile.TemporaryDirectory(prefix="ps_oracle_")
        self.work = work or self._tmp.name
        os.makedirs(self.work, exist_ok=True)
        self.vs = spirv(vs or VS_TRI, os.path.join(self.work, "vs.spv"), self.work)
        if ps is None and template not in TEMPLATES:
            raise ValueError(f"unknown template {template!r} (known: {', '.join(TEMPLATES)})")
        text = TEMPLATES[template] if ps is None else ps
        if ps is None and FORMATS[fmt][2] != "float":
            # an integer target takes an integer output (the template's float result, bitcast)
            if template != "const":
                raise ValueError(f"--format {fmt}: only the const template has an integer output")
            text = TEMPLATE_CONST_INT.replace("%int_kind", "1" if FORMATS[fmt][2] == "sint" else "0")
        self.ps = spirv(text, os.path.join(self.work, "ps.spv"), self.work)
        out_bin = os.path.join(self.work, "template.bin")
        r = run_harness(self.exe, self.vs, self.ps, os.path.join(self.work, "template.raw"), self.size,
                        {"VKH_BINARY_OUT": out_bin}, fmt)
        if r.returncode or not os.path.exists(out_bin):
            raise RuntimeError("vk_harness could not build the template:\n" + r.stderr.strip()[-800:])
        self.device = next((m.group(1) for m in re.finditer(r"device[:=]\s*(.+)", r.stderr)), "")
        self.bins = read_binaries(out_bin)
        if len(self.bins) == 1:          # the AMD driver: one PAL ELF
            self.backend, self.ps_index = "pal", 0
            self.elf = self.bins[0][2]
            self.meta = Elf(self.elf).metadata()
            self.wave64 = is_wave64(self.elf)
        else:                            # RADV: one binary per stage
            self.arch = radv_arch(self.device)
            self.backend, self.ps_index = "radv", radv_ps_index(self.bins)
            self.wave64 = radv_layout(self.bins[self.ps_index][2]).wave == 64
        self.template_pixels = pixels(os.path.join(self.work, "template.raw"), self.size, self.texel_bytes)

    def show(self):
        """The template's pixel-shader ISA (llvm-objdump) and the SPI part of its metadata."""
        from hw_oracle import runner
        if self.backend == "radv":
            return self._show_radv(runner)
        path = os.path.join(self.work, "template.elf")
        with open(path, "wb") as f:
            f.write(self.elf)
        start, code = ps_code(self.elf)
        lines = [f"pixel stage: wave{64 if self.wave64 else 32}, {len(code)} bytes from 0x{start:x}"]
        try:
            objdump = os.path.join(os.path.dirname(runner.find_clang()), "llvm-objdump" + (".exe" if os.name == "nt" else ""))
            r = subprocess.run([objdump, "-d", f"--mcpu={self.mcpu}", f"--start-address=0x{start:x}", path],
                               capture_output=True, text=True)
            lines += [ln for ln in r.stdout.splitlines() if ln.strip() and "s_code_end" not in ln]
        except (runner.Unavailable, OSError) as e:
            lines.append(f"(no disassembly: {e})")
        pipe = self.meta["amdpal.pipelines"][0]
        regs = pipe.get(".graphics_registers", {})
        keep = {k: v for k, v in regs.items() if k.startswith((".spi_ps", ".spi_shader_col", ".spi_baryc",
                                                                ".spi_interp", ".ps_", ".db_shader"))}
        lines.append("graphics registers: " + json.dumps(keep, indent=1, default=str))
        lines.append("pixel stage: " + json.dumps(ps_stage(self.meta), indent=1, default=str))
        return "\n".join(lines)

    def _show_radv(self, runner):
        """The RADV template's pixel-shader ISA (llvm-mc) and register counts."""
        data = self.bins[self.ps_index][2]
        lay = radv_layout(data)
        code = data[lay.code:lay.code + lay.exec_size]
        lines = [f"pixel stage ({self.arch}): wave{lay.wave}, {lay.exec_size} bytes of code, {lay.vgprs} VGPRs, "
                 f"{lay.sgprs} SGPRs, rsrc1 0x{struct.unpack_from('<I', data, lay.rsrc1_at)[0]:08x}"]
        try:
            mc = os.path.join(os.path.dirname(runner.find_clang()), "llvm-mc" + (".exe" if os.name == "nt" else ""))
            mc = mc if os.path.exists(mc) else shutil.which("llvm-mc") or mc
            r = subprocess.run([mc, "--disassemble", "-arch=amdgcn", f"-mcpu={self.mcpu}",
                                f"-mattr=+wavefrontsize{lay.wave}"], input=" ".join(f"0x{b:02x}" for b in code),
                               capture_output=True, text=True)
            lines += [ln for ln in r.stdout.splitlines() if ln.strip() and not ln.lstrip().startswith(".")]
        except (runner.Unavailable, OSError) as e:
            lines.append(f"(no disassembly: {e})")
        lines.append("(RADV keeps the SPI setup in its own structs; RADV_DEBUG=shaders prints it with the ISA)")
        return "\n".join(lines)

    def run(self, lines, sets=()):
        """Every pixel's mrt0 words (row-major, size[0] x size[1]) with the pixel shader replaced by `lines` and the
        metadata changes `sets` ((path, value) pairs, see set_meta) applied."""
        if sets and self.backend == "radv":
            raise ValueError("--set edits PAL metadata; RADV binaries have none (use a template that sets it up)")
        code = assemble(lines, self.mcpu, self.wave64, self.work)
        bins = list(self.bins)
        ks, key, data = bins[self.ps_index]
        bins[self.ps_index] = (ks, key, patch_ps(data, code, lines, sets) if self.backend == "pal"
                               else patch_radv_ps(data, code, lines))
        path = os.path.join(self.work, "patched.bin")
        write_binaries(path, bins)
        raw = os.path.join(self.work, "patched.raw")
        if os.path.exists(raw):
            os.remove(raw)
        r = run_harness(self.exe, self.vs, self.ps, raw, self.size, {"VKH_BINARY_IN": path}, self.fmt)
        if r.returncode or not os.path.exists(raw):
            raise RuntimeError(f"vk_harness exited {r.returncode} with the patched binary:\n" + r.stderr.strip()[-800:])
        return pixels(raw, self.size, self.texel_bytes)

    def close(self):
        if self._tmp:
            self._tmp.cleanup()


def summarize(px, size):
    """Distinct mrt0 words with how many pixels and the first pixel (x, y) that has them."""
    seen = {}
    for i, w in enumerate(px):
        k = tuple(w)
        if k not in seen:
            seen[k] = [0, (i % size[0], i // size[0])]
        seen[k][0] += 1
    return seen


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--build", required=True, help="runtime build folder with vk_harness")
    ap.add_argument("--template", default="interp2", help=f"one of {', '.join(TEMPLATES)} (default interp2)")
    ap.add_argument("--vs", help="vertex shader (.spv or SPIR-V assembly) instead of the built-in one")
    ap.add_argument("--ps", help="template pixel shader (.spv or SPIR-V assembly) instead of --template")
    ap.add_argument("--asm", help="the pixel-shader ISA to run (one instruction per line, ; comments)")
    ap.add_argument("--size", nargs=2, type=int, default=(8, 8), metavar=("W", "H"))
    ap.add_argument("--format", default="rgba32f", choices=sorted(FORMATS),
                    help="the color target (default rgba32f); the driver picks the export format for it")
    ap.add_argument("--mcpu", default="gfx1036")
    ap.add_argument("--show-template", action="store_true", help="print the template's ISA and SPI setup")
    ap.add_argument("--set", action="append", default=[], metavar="PATH=JSON",
                    help="change the pipeline metadata, e.g. .graphics_registers/.spi_ps_input_cntl/0/.fp16_interp_mode=true")
    ap.add_argument("--json", help="write {device, wave, size, pixels} here")
    a = ap.parse_args(argv)
    if not (a.asm or a.show_template):
        ap.error("nothing to do: give --asm and/or --show-template")
    if not all(0 < n <= 4096 for n in a.size):
        ap.error("--size must be 1..4096")
    if a.json:                       # the report is written after the run, over whatever it names
        for flag, path in (("--asm", a.asm), ("--ps", a.ps), ("--vs", a.vs)):
            if path and same_report_path(a.json, path):
                ap.error(f"--json would overwrite the {flag} input: {a.json}")
    try:
        lines = read_lines(a.asm) if a.asm else None
        sets = []
        for item in a.set:
            path, eq, val = item.partition("=")
            if not eq:
                raise ValueError(f"--set {item!r}: expected PATH=JSON")
            sets.append((path, json.loads(val)))
        o = Oracle(a.build, a.vs, a.ps, a.template, a.size, a.mcpu, fmt=a.format)
    except Unavailable as e:
        print(f"ps_oracle unavailable: {e}", file=sys.stderr)
        return 77
    except (OSError, ValueError, RuntimeError) as e:
        print(f"ps_oracle: {e}", file=sys.stderr)
        return 2
    try:
        print(f"pixel stage wave{64 if o.wave64 else 32}{' on ' + o.device if o.device else ''}")
        if a.show_template:
            print(o.show())
        if lines is None:
            return 0
        try:
            px = o.run(lines, sets)
        except (ValueError, KeyError, IndexError, TypeError) as e:
            print(f"ps_oracle: {e}", file=sys.stderr)
            return 2
        except RuntimeError as e:
            print(f"ps_oracle: {e}", file=sys.stderr)
            return 1
        for words, (count, xy) in summarize(px, a.size).items():
            print(f"{count:6d} px, first at {xy}: " + " ".join(f"0x{w:08x}" for w in words))
        if a.json:
            with open(a.json, "w", encoding="utf-8", newline="\n") as f:
                json.dump({"device": o.device, "wave": 64 if o.wave64 else 32, "size": list(a.size), "pixels": px},
                          f, indent=1)
        return 0
    finally:
        o.close()


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
