"""Shared fixtures for the translator tests.

Nothing here touches game data: shaders are tiny hand-written assembly programs, assembled with llvm-mc into the
same ``; encoding:`` listing format that tools/shader_disasm.py produces, next to a synthetic empty DXBC container
(rdna2spirv.translate() looks for ``<disasm dir>/../<name>.dxbc`` to read the input signature).
"""
import glob
import os
import re
import shutil
import struct
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import mc_command  # noqa: E402
import rdna2spirv  # noqa: E402


def _repair_stdin_handle():
    """Windows, under pytest's fd capture: give the children the tests start a stdin that is open.

    The capture replaces fd 0 with os.dup2(), which closes the handle fd 0 had, while the process's
    STD_INPUT_HANDLE keeps that handle's number. subprocess starts from GetStdHandle(), not from fd 0, so a child
    started without stdin=/input= fails with "[WinError 6] The handle is invalid" or, when the number has been
    reused meanwhile, gets whatever file now has it. Which one depends on what ran before, so the failures move
    from run to run. With -s or --capture=sys, and without pytest, the two handles agree and nothing is done.
    """
    if sys.platform != "win32":
        return
    import ctypes
    import msvcrt
    kernel32 = ctypes.WinDLL("kernel32")
    kernel32.GetStdHandle.restype = ctypes.c_void_p
    kernel32.GetStdHandle.argtypes = [ctypes.c_uint32]
    kernel32.SetStdHandle.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
    std_input = 0xFFFFFFF6                              # STD_INPUT_HANDLE, (DWORD)-10
    handle = kernel32.GetStdHandle(std_input)
    try:
        fd0 = msvcrt.get_osfhandle(0)
    except OSError:
        fd0 = None
    if not handle or handle == fd0:                     # no stdin at all (subprocess copes), or not redirected
        return
    # A NUL of our own, never closed: fd 0's handle would go stale again the next time the capture swaps fd 0.
    kernel32.SetStdHandle(std_input, msvcrt.get_osfhandle(os.open(os.devnull, os.O_RDONLY)))


_repair_stdin_handle()      # before the llvm-mc probe below, which is the first child


def _has_amdgpu(mc):
    """True if this llvm-mc was built with the AMDGPU target (the official Windows release archives are not)."""
    try:
        r = subprocess.run(mc + ["--version"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return False
    return "amdgcn" in r.stdout


def _find_llvm_mc():
    candidates = []
    env = os.environ.get("LLVM_MC")
    if env:
        candidates.append(mc_command.split_command(env))
    found = shutil.which("llvm-mc")
    if found:
        candidates.append([found])
    # Debian/Ubuntu install it versioned, out of PATH.
    candidates += [[p] for p in sorted(glob.glob("/usr/lib/llvm-*/bin/llvm-mc"), reverse=True)]
    # Homebrew's llvm and llvm@N are keg-only, never linked onto PATH.
    for prefix in ("/opt/homebrew/opt", "/usr/local/opt"):
        candidates += [[p] for p in sorted(glob.glob(prefix + "/llvm*/bin/llvm-mc"), reverse=True)]
    # Default location of the official Windows installer, which doesn't add LLVM to PATH unless asked to.
    candidates += [[p] for p in glob.glob(r"C:\Program Files\LLVM\bin\llvm-mc.exe")]
    for c in candidates:
        if c and _has_amdgpu(c):
            return c
    return None


LLVM_MC = _find_llvm_mc()
HAVE_SPIRV = shutil.which("spirv-as") and shutil.which("spirv-val")

requires_tools = pytest.mark.skipif(
    not (LLVM_MC and HAVE_SPIRV),
    reason=("needs an llvm-mc built with the amdgcn target, and spirv-as/spirv-val on PATH; "
            f"LLVM_MC={os.environ.get('LLVM_MC')!r}, found llvm-mc: {LLVM_MC!r}, "
            f"spirv-as/val: {bool(HAVE_SPIRV)}"),
)

# 32-byte DXBC header: magic, digest, version, size, chunk count = 0. Gives an empty input signature.
EMPTY_DXBC = b"DXBC" + bytes(16) + struct.pack("<III", 1, 32, 0)



def make_dxbc(chunks):
    """DXBC container holding ``chunks`` = [(fourcc, payload bytes)]. The digest is left zeroed (never read)."""
    head = 32 + 4 * len(chunks)
    offsets, body = [], b""
    for fourcc, payload in chunks:
        offsets.append(head + len(body))
        body += fourcc + struct.pack("<I", len(payload)) + payload
    return (b"DXBC" + bytes(16) + struct.pack("<III", 1, head + len(body), len(chunks))
            + b"".join(struct.pack("<I", o) for o in offsets) + body)


def signature_chunk(fourcc, elements):
    """ISG1/OSG1 chunk. elements = [(register, system value, mask)] or [(register, system value, mask, sem_idx)].
    System value 1 is SV_Position, 2 is SV_ClipDistance, 3 is SV_CullDistance, 0 is a plain varying."""
    out = struct.pack("<II", len(elements), 8)
    for item in elements:
        reg, sysv, mask = item[:3]
        sem = item[3] if len(item) > 3 else 0
        # stream, name offset, semantic index, system value, component type (3 = float), register, masks, padding
        out += struct.pack("<6I", 0, 0, sem, sysv, 3, reg) + bytes([mask, mask, 0, 0]) + struct.pack("<I", 0)
    return fourcc, out


def mesh_psv0_chunk(threads=(128, 1, 1), max_vertices=64, max_primitives=64, topology=2, group_shared=0):
    """PSV0 chunk of a mesh shader: only the runtime-info fields rdna2spirv.mesh_psv() reads (stage 13 = mesh,
    MSInfo vertex/primitive limits, output topology 2 = triangles, NumThreads X/Y/Z)."""
    r = bytearray(48)
    struct.pack_into("<I", r, 0, group_shared)
    struct.pack_into("<HH", r, 12, max_vertices, max_primitives)
    r[24] = 13
    r[27] = topology
    struct.pack_into("<III", r, 36, *threads)
    return b"PSV0", struct.pack("<I", len(r)) + bytes(r)


# DXBC the hand-written gs_tri / ms_tri samples in runtime/test/shaders need (their header comments say which).
GS_TRI_DXBC = make_dxbc([signature_chunk(b"ISG1", [(0, 1, 0xF), (1, 0, 0xF)]),      # SV_Position reg 0, TEXCOORD0 reg 1
                         signature_chunk(b"OSG1", [(0, 1, 0xF), (1, 0, 0xF)])])
MS_TRI_DXBC = make_dxbc([mesh_psv0_chunk()])                                          # 128 threads, 64/64, triangles
MS_CLIP_DXBC = make_dxbc([mesh_psv0_chunk(),
                          signature_chunk(b"OSG1", [(0, 1, 0xF), (1, 2, 0x1)])])      # SV_Position + SV_ClipDistance.x
SAMPLE_DXBC = {"gs_tri": GS_TRI_DXBC, "ms_tri": MS_TRI_DXBC, "ms_clip": MS_CLIP_DXBC, "ms_wave_info_s3": MS_TRI_DXBC,
               "ms_w64x_ballot": MS_TRI_DXBC, "ms_f16_pack": MS_TRI_DXBC,      # scripts/run_gpu_tests.py, ms_tri's PSV0
               "gs_w64_a_ring": GS_TRI_DXBC, "gs_w64_c_lane_id": GS_TRI_DXBC,  # build.py gives every gs_* gs_tri's
               "ms_v5_workgroup": make_dxbc([mesh_psv0_chunk(threads=(32, 1, 1), max_vertices=3,
                                                           max_primitives=1, group_shared=128)])}

# spirv-val errors on valid-looking output, by sample.
# CullPrimitiveEXT used to need a waiver for SPIRV-Tools ≤2025.1 (#5980) when emitted as OpTypeArray of
# bool; the translator now uses the Block+bool-member form (also allowed by VUID-07036), which those
# validators accept. Keep this map for unrelated known false positives.
KNOWN_VAL_ISSUES = {}

# Compute shaders take their system registers from the FSSM header (rdna2spirv.translate reads it from
# <disasm dir>/../fssm/<name>.fssm): word 0x138 is COMPUTE_PGM_RSRC2, words 0x13C/0x140/0x144 the thread-group size.
# 0x8088 = 4 user SGPRs, TGID_X, 512 B of LDS; group 64x1x1 (the header the cs_* samples in runtime/test/shaders use).
DEFAULT_CS_HEADER = (0x8088, (64, 1, 1))


class Shader:
    def __init__(self, workdir, name, asm, cs_header=DEFAULT_CS_HEADER, dxbc=EMPTY_DXBC):
        self.name = name
        disasm = workdir / "disasm"
        disasm.mkdir(exist_ok=True)
        (workdir / f"{name}.dxbc").write_bytes(dxbc)
        if name.startswith("cs_"):
            rsrc2, (lx, ly, lz) = cs_header
            (workdir / "fssm").mkdir(exist_ok=True)
            (workdir / "fssm" / f"{name}.fssm").write_bytes(bytes(0x138) + struct.pack("<4I", rsrc2, lx, ly, lz))
        cmd = LLVM_MC + ["-triple=amdgcn--amdpal", "-mcpu=gfx1013"]
        for ln in asm.splitlines():
            s = ln.strip()
            if not s:
                continue
            if not s.startswith(";"):
                break
            m = re.match(r";\s*mattr:\s*(\S+)", s)
            if m:
                cmd.append(f"-mattr={m.group(1)}")
                break
        cmd.append("-show-encoding")
        r = subprocess.run(cmd, input=asm, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        self.path = disasm / f"{name}.s"
        self.path.write_text(f"; {name}  synthetic\n{r.stdout}")
        self.workdir = workdir

    def translate(self):
        return rdna2spirv.translate(str(self.path))

    def validate(self, spvasm, known_val_issue=None):
        """Assemble and run spirv-val. known_val_issue: a substring of a spirv-val error that is a validator problem,
        not a translator one; the test is then reported as xfail with that text, so it turns into a plain pass once
        a newer spirv-tools no longer reports it. (spirv-val stops at its first error: later ones stay hidden.)"""
        sa = self.workdir / f"{self.name}.spvasm"
        sv = self.workdir / f"{self.name}.spv"
        sa.write_text(spvasm)
        r = subprocess.run(["spirv-as", str(sa), "-o", str(sv), "--target-env", "vulkan1.2"],
                           capture_output=True, text=True)
        assert r.returncode == 0, "spirv-as: " + r.stderr
        val = ["spirv-val", "--target-env", "vulkan1.2"]
        r = subprocess.run(val + [str(sv)], capture_output=True, text=True)
        if r.returncode != 0 and "Offset can only be used with OpImage*Gather" in r.stderr:
            # image_sample*_o (sample ps_texoff.asm): a non-constant Offset outside gather is VK_KHR_maintenance8, and
            # rdna2spirv.py validates with this flag (docs/design.md, "Texel offsets"). A spirv-val from before
            # maintenance8 prints its usage text for the flag it doesn't know: nothing can be validated there.
            r = subprocess.run(val + ["--allow-offset-texture-operand", str(sv)], capture_output=True, text=True)
            if r.returncode != 0 and "USAGE" in r.stdout + r.stderr:
                pytest.skip("this spirv-val has no --allow-offset-texture-operand (VK_KHR_maintenance8)")
        if r.returncode != 0 and known_val_issue and known_val_issue in r.stderr:
            pytest.xfail("known spirv-val issue (" + known_val_issue + "): " + r.stderr.strip().splitlines()[0])
        assert r.returncode == 0, "spirv-val: " + r.stderr


@pytest.fixture
def make_shader(tmp_path):
    """make_shader(asm, name="ps_test") -> Shader. The name's prefix ("ps_", "vs_", "cs_") selects the stage.
    For cs_ shaders, cs_header=(RSRC2, (x, y, z)) sets the synthetic FSSM header; dxbc=<bytes> replaces the
    empty DXBC container (see make_dxbc)."""
    def make(asm, name="ps_test", **kw):
        return Shader(tmp_path, name, asm, **kw)
    return make
