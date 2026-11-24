"""Assemble, translate, and run the synthetic RDNA2->SPIR-V GPU test suite on Vulkan.

Usage:
    python -I scripts/run_gpu_tests.py [--llvm-mc CMD] [--build DIR] [--work DIR] [--filter SUBSTR] [--require-pass]

--llvm-mc (or LLVM_MC) is a path or a whole command, e.g. "wsl -d Ubuntu -e /usr/lib/llvm-21/bin/llvm-mc".
"""
import argparse
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHADER_DIR = os.path.join(ROOT, "runtime", "test", "shaders")

TRANSLATOR = os.path.join(ROOT, "tools", "rdna2spirv.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))
import mc_command  # noqa: E402
sys.path.insert(0, SHADER_DIR)
from build import CLIP_CASES, clip_container, shader_source
from check import (Skip, clip_test, fmt_1010102_test, fmt_d16_xyz_test, fmt_int_narrow_test, fmt_sint32_test,
                   fmt_xy_xyz_test, tbuffer_store_test)
from check import compute_unsupported   # vk_compute_test's exit 77 over a feature the module needs
from compute_fixture import compute_header
from check import SAMPLER_CASES, sampler_test   # ps_ssharp: the S#/T# mip fields (vk_sampler_test)
from check import (CMPX_CLASS_WORDS, CVT_F32_SAT_CASES, CVT_PK_U8_WORDS, PK_MUL_LO_WORDS, GRADIENT_TESTS, TEST_PROGRAMS, TEXTURE_MIP_DEPENDENCIES,
                   TEXTURE_MIP_TEST, GATHER_LOD_TEST, MIP_TEST_PASSES, TEXTURE_CONSTANTS, USER_SGPRS_ENV, USER_SGPRS_WORDS, Skip, mesh_v5_test, not_found_message,
                   pick_build_dir, start_failure, table_fetch_test, texture_gradient_a16_test, texture_mip_test)
from check import SIGNED_MIP_TEST, UINT_MIP_TEST
from check import HEAP_INIT_EXPECTED, heap_init_env   # cs_heap_init: vk_compute_test's VKC_HEAP_INIT
from check import judged_words, words_match   # dump words not judged on some devices (cs_signed_zero_nan)
from check import TEXOFF_ARGS, texoff_problem   # ps_texoff's expected words and arguments live in check.py
from check import TEXOFF_FALLBACK, texoff_env   # LEONIDA_TEXOFF_FALLBACK=1: texel offsets without maintenance8
from check import gpu_test_timeout, timed_out   # per-program time limit (LEONIDA_GPU_TEST_TIMEOUT)
from check import failure   # "exit <status>: <last lines>", check.py's detail for a program that exited non-zero
from check import run_readback
from check import MESH_COMPUTE, mesh_env   # LEONIDA_MESH_COMPUTE=1: mesh shaders as compute + a generated VS
from check import (WAVE32_LANE_DUMPS, WAVE32_LANE_ENV, bitmask_perm_narrow_skip, narrow_subgroup_skip,
                   unpinned_subgroup_skip, vcmpx_forms_narrow_skip, wave32_lane_skip)   # as in check.py
from check import PIXEL_VERTEX, VINT_ENV, interp_f16_high_problem, ij_computed_problem, mandelbrot_problem   # pixel producers and checks
from check import barycentric_problem
from check import NAN_PAYLOAD_TEST, nan_payload_problem
from check import GET_LOD_TESTS, STORE_WORDS, get_lod_test, store_words_problem   # TEST-184: image atomics and queries
from check import wide_pixel_skip   # ps_w64x_readlane: judged only where the fragment stage was required 64-wide
from check import FMA_FUSED_ENV, judge_fma_fused   # cs_fma_fused: strict under LEONIDA_FMA=khr, fused/split under glsl
from check import bary_persp_problem   # ps_persp_bary's perspective-correct reference


def make_dxbc(chunks: list[tuple[bytes, bytes]]) -> bytes:
    """Build a minimal DXBC container from (fourcc, payload) pairs."""
    n = len(chunks)
    hdr_size = 32 + 4 * n
    offsets = []
    body = bytearray()
    for tag, payload in chunks:
        offsets.append(hdr_size + len(body))
        body += tag + struct.pack("<I", len(payload)) + payload
    total = hdr_size + len(body)
    hdr = bytearray(b"DXBC") + bytes(16) + struct.pack("<III", 1, total, n)
    for off in offsets:
        hdr += struct.pack("<I", off)
    return bytes(hdr) + bytes(body)


def make_sig_chunk(elements: list[tuple[int, int, int, int, int]]) -> bytes:
    """Build ISG1/OSG1 payload: elements of (sysv, ctype, reg, mask, rw_mask)."""
    out = bytearray(struct.pack("<II", len(elements), 8))
    for sysv, ctype, reg, mask, rw in elements:
        e = bytearray(32)
        struct.pack_into("<6I", e, 0, 0, 0, 0, sysv, ctype, reg)
        e[24] = mask
        e[25] = rw
        out += e
    return bytes(out)


def make_ms_psv0(threads_x: int, maxv: int, maxp: int, topology: int = 2, group_shared: int = 0) -> bytes:
    """Build PSV0 payload for a mesh shader (stage 13)."""
    r = bytearray(48)
    struct.pack_into("<I", r, 0, group_shared)
    struct.pack_into("<HH", r, 12, maxv, maxp)
    r[24] = 13  # mesh stage
    r[27] = topology  # 2 = triangles
    struct.pack_into("<III", r, 36, threads_x, 1, 1)
    return struct.pack("<I", len(r)) + bytes(r)


def assemble_to_s(llvm_mc: str, asm_path: str, s_path: str, name: str) -> None:
    """Assemble raw RDNA2 .asm with llvm-mc -show-encoding into .s format for rdna2spirv.py."""
    with open(asm_path, encoding="utf-8") as source:
        raw = source.read()
    cmd = mc_command.mc_command(llvm_mc) + ["-triple=amdgcn--amdpal", "-mcpu=gfx1013"]
    for ln in raw.splitlines():
        s = ln.strip()
        if not s:
            continue
        if not s.startswith(";"):
            break
        m = re.match(r";\s*mattr:\s*(\S+)", s)
        if m:
            cmd.append(f"-mattr={m.group(1)}")
            break
    if name in CLIP_CASES:
        cmd.append(f"-defsym=PARAM_SLOT={CLIP_CASES[name]}")
    cmd.append("-show-encoding")
    p = mc_command.run(   # a missing llvm-mc: a short message naming LLVM_MC/--llvm-mc, not a traceback
        cmd,
        flag="--llvm-mc",
        input=raw,
        capture_output=True,
        text=True,
    )
    if p.returncode != 0:
        raise RuntimeError(f"llvm-mc failed on {name}:\n{p.stderr}")
    lines = [f"; {name}  code 0x0..0x0 (synthetic), gfx1013, 0 invalid words"]
    for ln in p.stdout.splitlines():
        if ln.startswith("\t.") or ln.strip().startswith("."):
            continue
        if ln.startswith("\t"):
            lines.append(ln)
    with open(s_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def resolve_assembler(choice: str) -> str:
    """Resolve --assembler choice ('auto', 'spirv-as', or 'python') to a concrete assembler name."""
    if choice in ("spirv-as", "python"):
        return choice
    if choice == "auto":
        return "spirv-as" if shutil.which("spirv-as") else "python"
    raise ValueError(f"unknown assembler: {choice!r}")


def spirv_as(asm_text: str, out_spv: str, assembler: str = "spirv-as") -> None:
    if assembler == "python":
        import spirv_asm
        try:
            spv_bytes = spirv_asm.assemble(asm_text, target_env="vulkan1.2")
        except spirv_asm.SpirvAsmError as exc:
            raise RuntimeError(f"spirv_asm failed for {out_spv}:\n{exc}") from None
        with open(out_spv, "wb") as f:
            f.write(spv_bytes)
        return
    p = subprocess.run(["spirv-as", "--target-env", "vulkan1.2", "-o", out_spv, "-"],
                       input=asm_text, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"spirv-as failed for {out_spv}:\n{p.stderr}")


VS_TRI_SPVASM = """\
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %main "main" %gl_VertexIndex %gl_Position %out0 %out1
               OpDecorate %gl_VertexIndex BuiltIn VertexIndex
               OpDecorate %gl_Position BuiltIn Position
               OpDecorate %out0 Location 0
               OpDecorate %out1 Location 1
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %uint = OpTypeInt 32 0
        %int = OpTypeInt 32 1
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
   %ptr_in_i = OpTypePointer Input %int
  %ptr_out_v = OpTypePointer Output %v4float
     %uint_3 = OpConstant %uint 3
     %arr_v4 = OpTypeArray %v4float %uint_3
 %ptr_fn_arr = OpTypePointer Function %arr_v4
 %ptr_fn_v4  = OpTypePointer Function %v4float
        %fm1 = OpConstant %float -1
         %f0 = OpConstant %float 0
         %f1 = OpConstant %float 1
         %f2 = OpConstant %float 2
         %f3 = OpConstant %float 3
      %pos_0 = OpConstantComposite %v4float %fm1 %fm1 %f0 %f1
      %pos_1 = OpConstantComposite %v4float %f3 %fm1 %f0 %f1
      %pos_2 = OpConstantComposite %v4float %fm1 %f3 %f0 %f1
    %pos_arr = OpConstantComposite %arr_v4 %pos_0 %pos_1 %pos_2
      %uv0_0 = OpConstantComposite %v4float %f0 %f0 %f0 %f1
      %uv0_1 = OpConstantComposite %v4float %f2 %f0 %f0 %f1
      %uv0_2 = OpConstantComposite %v4float %f0 %f2 %f0 %f1
    %uv0_arr = OpConstantComposite %arr_v4 %uv0_0 %uv0_1 %uv0_2
      %uv1_0 = OpConstantComposite %v4float %f0 %f0 %f0 %f0
      %uv1_1 = OpConstantComposite %v4float %f2 %f0 %f0 %f0
      %uv1_2 = OpConstantComposite %v4float %f0 %f2 %f0 %f0
    %uv1_arr = OpConstantComposite %arr_v4 %uv1_0 %uv1_1 %uv1_2
%gl_VertexIndex = OpVariable %ptr_in_i Input
%gl_Position = OpVariable %ptr_out_v Output
       %out0 = OpVariable %ptr_out_v Output
       %out1 = OpVariable %ptr_out_v Output
       %main = OpFunction %void None %fn
      %entry = OpLabel
      %p_var = OpVariable %ptr_fn_arr Function
      %0_var = OpVariable %ptr_fn_arr Function
      %1_var = OpVariable %ptr_fn_arr Function
               OpStore %p_var %pos_arr
               OpStore %0_var %uv0_arr
               OpStore %1_var %uv1_arr
        %idx = OpLoad %int %gl_VertexIndex
       %p_pt = OpAccessChain %ptr_fn_v4 %p_var %idx
       %p_vl = OpLoad %v4float %p_pt
               OpStore %gl_Position %p_vl
       %0_pt = OpAccessChain %ptr_fn_v4 %0_var %idx
       %0_vl = OpLoad %v4float %0_pt
               OpStore %out0 %0_vl
       %1_pt = OpAccessChain %ptr_fn_v4 %1_var %idx
       %1_vl = OpLoad %v4float %1_pt
               OpStore %out1 %1_vl
               OpReturn
               OpFunctionEnd
"""


def make_fs_passthru(loc: int) -> str:
    return f"""\
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %in_v %out_v
               OpExecutionMode %main OriginUpperLeft
               OpDecorate %in_v Location {loc}
               OpDecorate %out_v Location 0
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
   %ptr_in_v = OpTypePointer Input %v4float
  %ptr_out_v = OpTypePointer Output %v4float
       %in_v = OpVariable %ptr_in_v Input
      %out_v = OpVariable %ptr_out_v Output
       %main = OpFunction %void None %fn
      %entry = OpLabel
          %v = OpLoad %v4float %in_v
               OpStore %out_v %v
               OpReturn
               OpFunctionEnd
"""


def read_raw_rgba32f(path: str) -> list[tuple[float, float, float, float]]:
    with open(path, "rb") as source:
        data = source.read()
    n = len(data) // 16 if len(data) % 16 == 0 else 0   # a partial pixel means a malformed readback: no frame
    return [struct.unpack_from("<4f", data, 16 * i) for i in range(n)]


def read_raw_u32(path: str) -> list[tuple[int, int, int, int]]:
    with open(path, "rb") as source:
        data = source.read()
    n = len(data) // 16 if len(data) % 16 == 0 else 0   # a partial pixel means a malformed readback: no frame
    return [struct.unpack_from("<4I", data, 16 * i) for i in range(n)]


UNSUPPORTED = 77   # vk_harness: the device lacks a stage the test draws with (geometry or mesh shaders)


def unsupported_skip(msg: str) -> str:
    """The SKIP detail for a harness that exited UNSUPPORTED, quoting its "unsupported: ..." line."""
    m = re.search(r"unsupported: .*", msg)
    return f"SKIP ({m.group(0) if m else 'unsupported device'})"


def verdict_line(rc: int, msg: str) -> str:
    """The detail of a row whose test program judges itself and prints its verdict last: that line. A program that
    exited non-zero gets its exit status in front, as in check.py, so one that died before its verdict (exit -11 on
    Linux, 3221225477 = 0xC0000005 on Windows) does not read as a FAIL with only its device or extension line, or
    with no detail at all."""
    return failure(rc, msg, 1) if rc else msg.splitlines()[-1] if msg else ""


def helper_result(rc: int, msg: str) -> tuple[bool | None, str]:
    """(ok, detail) for a helper that judges its own output (vk_store_test): 77 is a SKIP that names the missing
    feature, not a failure (TEST-469); otherwise the exit code decides and verdict_line gives the detail."""
    if rc == UNSUPPORTED:
        return None, unsupported_skip(msg)
    return rc == 0, verdict_line(rc, msg)


def compute_result(rc: int, msg: str, ok: bool, detail: str) -> tuple[bool | None, str]:
    """(ok, detail) for a vk_compute_test run the caller judges itself, unless the module needs a feature the device
    lacks (check.compute_unsupported: exit 77 with an "unsupported: ..." line), which is a SKIP quoting it."""
    skip = compute_unsupported(rc, msg)
    return (None, skip) if skip else (ok, detail)


def run_cmd(cmd: list[str], env: dict | None = None, gpu: bool = True, *,
            cwd: str | None = None) -> tuple[int, str]:
    # A hung GPU test program fails its test instead of stalling the suite (check.py, LEONIDA_GPU_TEST_TIMEOUT).
    # gpu=False (the translator over the whole suite) runs without a limit.
    limit = gpu_test_timeout() if gpu else None
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=limit, cwd=cwd)
    except subprocess.TimeoutExpired:
        return 1, timed_out(cmd[0], limit)
    except OSError as e:   # e.g. Smart App Control blocked a fresh build: fail this test, keep running the others
        return 1, start_failure(cmd[0], e)
    return p.returncode, (p.stdout + "\n" + p.stderr).strip()


def bary_fallback() -> bool:
    """LEONIDA_BARY_FALLBACK=1 (or 0): translate raw barycentrics to an ordinary vec3 input and draw it through
    tools/bary_fallback_gs.py. Unset, it is on where tools/vk_subgroup_info.py reports a driver with a known
    BaryCoordKHR bug: Mesa ANV (SHADER-925) and AMD's Windows driver on gfx90c iGPUs (#5933)."""
    v = (os.environ.get("LEONIDA_BARY_FALLBACK") or "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return _device_barycentric_fallback() is True


def f16_interp_fallback() -> bool:
    """LEONIDA_F16_INTERP_FALLBACK=1 (or 0): translate with --f16-interp-fallback, so ps_interp_f16_high reads the f16
    halves its producer exports decoded instead of PerVertexKHR inputs. Unset, it is on where the device can't
    compile PerVertexKHR: tools/vk_subgroup_info.py reports per_vertex_inputs false (MoltenVK)."""
    v = (os.environ.get("LEONIDA_F16_INTERP_FALLBACK") or "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return _device_per_vertex() is False


_PER_VERTEX = []
_BARYCENTRIC_FALLBACK = []


def _device_per_vertex():
    """per_vertex_inputs from tools/vk_subgroup_info.py for the device the tests run on (cached), or None."""
    if not _PER_VERTEX:
        answer = None
        try:
            r = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "tools", "vk_subgroup_info.py")],
                               capture_output=True, text=True, timeout=60)
            answer = json.loads(r.stdout.strip().splitlines()[-1]).get("per_vertex_inputs")
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            pass
        _PER_VERTEX.append(answer)
    return _PER_VERTEX[0]


def _device_barycentric_fallback():
    """barycentric_fallback from tools/vk_subgroup_info.py (Mesa ANV, SHADER-925; AMD Windows on gfx90c, #5933),
    cached per suite."""
    if not _BARYCENTRIC_FALLBACK:
        answer = None
        try:
            r = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "tools", "vk_subgroup_info.py")],
                               capture_output=True, text=True, timeout=60)
            answer = json.loads(r.stdout.strip().splitlines()[-1]).get("barycentric_fallback")
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            pass
        _BARYCENTRIC_FALLBACK.append(answer)
    return _BARYCENTRIC_FALLBACK[0]


def w64_emulated() -> bool:
    # Wave64 emulation on: the suite runs every wave64 check without asking for a 64-wide
    # subgroup, matching check.py (W64_EMULATED). Accepts "1"/"true"/"yes"/"on" and tolerates
    # the trailing space cmd.exe's `set VAR=1 &` leaves in the value.
    return (os.environ.get("LEONIDA_W64_EMULATE") or "").strip().lower() in ("1", "true", "yes", "on")
def pc_tail_ubo() -> bool:
    # LEONIDA_PC_TAIL_UBO=1: translate with --pc-tail-ubo (s0..s31 in 128 push-constant bytes, the tail in a uniform
    # block at set 1), for devices whose maxPushConstantsSize is 128. The test programs read the same variable
    # (runtime/test/vk_pc_tail.h) and bind the block; same accepted values as LEONIDA_W64_EMULATE.
    return (os.environ.get("LEONIDA_PC_TAIL_UBO") or "").strip().lower() in ("1", "true", "yes", "on")
def wavemask_check(vk_compute: str, spv: str, env: dict | None, emulated: bool) -> tuple[bool | None, str]:
    """cs_wavemask, as in check.py: (passed, detail), passed None for a skip. The native build needs a 64-wide
    subgroup and skips where the device has none; with LEONIDA_W64_EMULATE=1 it was translated with
    --wave64-emulate and runs on any subgroup size, so it asks for none."""
    cmd = [vk_compute, spv, "1", "wavemask"] + ([] if emulated else ["--require-wave64"])
    rc, msg = run_cmd(cmd, env=env)
    if skip := compute_unsupported(rc, msg):
        return None, skip
    if not emulated and rc == 77:
        return None, "SKIP (no 64-wide compute subgroup on this device)"
    if not emulated and rc == 0 and "compute subgroup size: 64" not in msg.splitlines():
        return False, "64-wide subgroup was not confirmed (rebuild vk_compute_test)"
    return rc == 0, verdict_line(rc, msg)
def dump_words(msg: str) -> list[int] | None:
    """The eight words of vk_compute_test's "out[0..7] = ..." line, or None unless it is eight decimal words (as in
    check.py): trailing text, a missing word or a non-number is a malformed dump, not a result."""
    m = re.search(r"^out\[0\.\.7\] = ((?:[0-9]+ ){7}[0-9]+)$", msg, re.MULTILINE)
    return [int(x) for x in m.group(1).split()] if m else None


def subgroup_pin_skip(msg: str, cs_env: dict | None) -> Skip | None:
    """A Skip when the runner asked for a subgroup size (VKC_SUBGROUP_SIZE) and vk_compute_test says this device
    cannot pin it, so the dispatched words are not the ones the fixture was written for. Only for the dumps in
    WAVE32_PIN_DUMPS: everything else the runner judges at the device's own width, which is how cs_store and the
    cs_w32_* fixtures keep being tested on a 64-only device (check.py is conservative and skips all of them)."""
    if cs_env and "VKC_SUBGROUP_SIZE" in cs_env:
        return unpinned_subgroup_skip(msg)
    return None


# Dumps whose expected words depend on the wave width but are not lane-mask dumps (those are WAVE32_LANE_DUMPS):
# cs_barrier_loop's trip count is 4 waves of 32, so at 64 lanes it writes a different pattern and the words would
# report a failure that is really an unpinnable wave size (GCN3/GCN4 expose 64..64 only).
WAVE32_PIN_DUMPS = ("cs_barrier_loop",)


def judge_cs_dump(rc: int, msg: str, want: list[int], *, wave64: bool, partial: bool, emulated: bool,
                  hexwords: bool = False) -> tuple[bool | None, str]:
    """Judge one vk_compute_test "dump" run: (True, words), (False, why), or (None, why) when the device can't run it.
    wave64: it was run with --require-wave64, so exit 77 is a skip and the 64-wide confirmation must be printed.
    partial: a group smaller than a wave, run at subgroup size 64 without full subgroups; skipped when the device
    reports no subgroup size of 64. emulated: translated with --wave64-emulate, runs on any subgroup size."""
    if rc == 77:   # the program refused the module and said why ("unsupported: <why>"): a skip with that reason, in every mode
        why = next((ln for ln in msg.splitlines() if ln.startswith("unsupported:")), None)
        if why:
            return None, "SKIP (" + why[len("unsupported:"):].strip() + ")"
    no64 = partial and not emulated and not any(int(x) >= 64 for x in re.findall(r"subgroup \d+\.\.(\d+)", msg))
    if (wave64 and rc == 77) or (no64 and rc == 0):
        return None, "SKIP (no 64-wide compute subgroup on this device)"
    if rc == 0 and wave64 and "compute subgroup size: 64" not in msg.splitlines():
        return False, "64-wide subgroup was not confirmed (rebuild vk_compute_test)"
    words = dump_words(msg)
    if rc != 0 or words is None:
        return False, msg
    vals = words[:len(want)]
    return words_match(vals, want), " ".join((hex if hexwords else str)(x) for x in vals)


def check_pixel_const(raw_path: str, want: tuple[int, int, int, int]) -> tuple[bool, str]:
    px = read_raw_u32(raw_path)
    if len(px) != 64 * 64:
        return False, f"expected 4096 pixels, got {len(px)}"
    bad = [(i, p) for i, p in enumerate(px) if p != want]
    if bad:
        i, p = bad[0]
        return False, f"{len(bad)}/4096 mismatches; first at ({i%64},{i//64}): got {[hex(x) for x in p]}, want {[hex(x) for x in want]}"
    return True, "4096/4096 pixels exact"


def frame_problem(px) -> str:
    """Why a 64x64 frame can't be judged: a short frame or a NaN passes every tolerance test (check.py rejects both)."""
    if len(px) != 64 * 64:
        return f"expected 4096 pixels, got {len(px)}"
    return "" if all(math.isfinite(v) for p in px for v in p) else "non-finite values in the frame"


def check_bary(raw_path: str) -> tuple[bool, str]:
    problem = barycentric_problem(read_raw_rgba32f(raw_path), 1e-4)
    return not problem, problem or "4096/4096 pixels match"


def check_bitmask_perm(raw_path: str, msg: str = "") -> tuple[bool | None, str]:
    px = read_raw_u32(raw_path)
    if skip := bitmask_perm_narrow_skip(msg, px):
        return None, str(skip)
    if problem := frame_problem(px):
        return False, problem
    bad = 0
    first = ""
    judged = 0
    for i, (lane, v1, v2, v3) in enumerate(px):
        judged += 1
        w1 = lane ^ 9
        w2 = ((lane & 5) | 16) ^ 1
        if v1 != w1 or v2 != w2 or v3 != 0:
            if not bad:
                first = f"at ({i%64},{i//64}) lane={lane}: got ({v1},{v2},{v3}), want ({w1},{w2},0)"
            bad += 1
    return (bad == 0 and judged > 0), (first if bad else f"{judged}/{judged} pixels exact")


# Pixel shaders that read memory: vk_harness puts this V# in s[0:3] and seeds the heap it points at, as
# tests/semantic/test_value_semantics.py does for the interpreter (same table as check.py).
PIXEL_ENV = {"ps_vcc_x2": {"VKH_SGPRS": "0x1000,0,16,0", "VKH_HEAP": "0x1000=0x11223344,0x1004=0x55667788"},
             # the 14 v_cmpx forms: f32 and 64-bit operands from memory (derivations in the .asm header)
             "ps_vcmpx_forms": {"VKH_SGPRS": "0x1000,0,96,0",
                                "VKH_HEAP": "0x1000=0x7fc10001,0x1004=0x3f800000,0x1008=0x40000000,0x100c=0x80000000,"
                                            "0x1010=0,0x1014=0xff800000,0x1018=0,0x101c=0,0x1020=0x80000000,0x1024=0,"
                                            "0x1028=0x1,0x102c=0,0x1030=0x7,0x1034=0x80000000,0x1038=0x7,0x103c=0,"
                                            "0x1040=0x1,0x1044=0x5,0x1048=0x80000000,0x104c=0x5,0x1050=0x7,0x1054=0x5,"
                                            "0x1058=0x7,0x105c=0x5"},
             # 32-bit integer compares: operands from memory (derivations in the .asm header)
             "ps_vcmp_int32": {"VKH_SGPRS": "0x1000,0,32,0",
                               "VKH_HEAP": "0x1000=0x1,0x1004=0,0x1008=0x80000000,0x100c=0x12345678,0x1010=0x3,"
                                           "0x1014=0x5,0x1018=0,0x101c=0"},
             # float NaN / fract rows measured on RDNA2 hardware (see the .asm headers)
             "ps_fnan_med3": {"VKH_SGPRS": "0x1000,0,32,0",
                              "VKH_HEAP": "0x1000=0x7fc10001,0x1004=0x40e00000,0x1008=0x40a00000,"
                                          "0x100c=0x7fc10001,0x1010=0x40600000,0x1014=0xc0200000"},
             "ps_fnan_fract": {"VKH_SGPRS": "0x1000,0,16,0",
                               "VKH_HEAP": "0x1000=0xaedbe6ff,0x1004=0x00008800,0x1008=0x7fc10001,0x100c=0xc0200000"},
             "ps_fnan_med3_s2": {"VKH_SGPRS": "0x1000,0,16,0",
                                 "VKH_HEAP": "0x1000=0x40a00000,0x1004=0x40e00000,0x1008=0x7fc10001,0x100c=0xc0200000"},
             "ps_f16_fnan_med3": {"VKH_SGPRS": "0x1000,0,16,0",
                                  "VKH_HEAP": "0x1000=0x00004500,0x1004=0x00004700,0x1008=0x00007e01,0x100c=0x0000c100"},
             # f32 VOPC with a NaN, 64-bit integer VOPC: inputs from memory (derivations in the .asm headers)
             "ps_vcmp_f32_nan": {"VKH_SGPRS": "0x1000,0,16,0",
                                 "VKH_HEAP": "0x1000=0x7fc10001,0x1004=0x3f800000,0x1008=0x40000000"},
             "ps_vcmp_64": {"VKH_SGPRS": "0x1000,0,64,0",
                            "VKH_HEAP": "0x1000=0x80000000,0x1004=0,0x1008=1,0x100c=0,0x1010=0xffffffff,"
                                        "0x1014=0xffffffff,0x1018=1,0x101c=0,0x1020=1,0x1024=0,0x1028=0x80000000,"
                                        "0x102c=0,0x1030=7,0x1034=5,0x1038=7,0x103c=5"},
             # the ps_vint_* integer fixtures share one heap, h0..h7 (see their .asm headers)
             "ps_vint_logic3": VINT_ENV,
             "ps_vint_bits": VINT_ENV,
             "ps_vint_mul24": VINT_ENV,
             "ps_vint_minmax": VINT_ENV,
             "ps_vint_minmax3": VINT_ENV}


def check_flat(raw_path: str) -> tuple[bool, str]:
    """ps_flat with vs_ids: the integer IDs come from the first (provoking) vertex, the UVs still interpolate."""
    with open(raw_path, "rb") as source:
        data = source.read()
    if len(data) != 64 * 64 * 16:
        return False, f"expected a 64x64 frame, got {len(data)} bytes"
    px = list(struct.iter_unpack("<2I2f", data))
    if problem := frame_problem(px):
        return False, problem
    w = int(len(px) ** 0.5)
    wrong = sum((a, b) != (0x101, 0x102) for a, b, _, _ in px)
    err = max(max(abs(u - (i % w + 0.5) / w), abs(v - (i // w + 0.5) / w)) for i, (_, _, u, v) in enumerate(px))
    ok = wrong == 0 and err < 1e-6
    return ok, (f"{len(px)} pixels: flat IDs exact, uv within {err:.1g}" if ok else
                f"{wrong} pixels without the first vertex's value, uv off by {err:.3g}")


def check_wqm(raw_path: str) -> tuple[bool, str]:
    """ps_wqm.asm: per pixel (wqm, own, wqm, own); wqm iff x%4 < 2 and y%4 >= 2, own iff x%4 == 1 and y%4 == 2."""
    px = read_raw_u32(raw_path)
    if problem := frame_problem(px):
        return False, problem
    w = int(len(px) ** 0.5)
    wrong = sum(tuple(p) != (a, b, a, b)
                for i, p in enumerate(px)
                for a, b in [(int(i % w % 4 < 2 and i // w % 4 >= 2), int(i % w % 4 == 1 and i // w % 4 == 2))])
    return wrong == 0, "whole-quad mask exact on every pixel" if not wrong else f"{wrong} of {len(px)} pixels differ"


def check_uniform_branch(raw_path: str) -> tuple[bool, str]:
    px = read_raw_rgba32f(raw_path)
    u32 = read_raw_u32(raw_path)
    """ps_uniform_branch / ps_scc_mask32: s5 = 8 wherever the lane's own bit is set; a lane with bit 0 gets 8 when
    another lane of its wave has the bit (the wave-wide vote), else 7. All three kinds must occur, as in check.py:
    without a (8, bit 0) pixel the vote is not shown, and a translator that decides per lane would pass."""
    # Only z (the barycentric i) is judged as a float; a NaN i passes both range tests below.
    if problem := frame_problem([p[2:3] for p in px]):
        return False, problem
    bad = 0
    n8 = 0
    n7 = 0
    n80 = 0
    first = ""
    for i, ((_, _, bary_i, _), (s5, bit, _, _)) in enumerate(zip(px, u32)):
        n80 += s5 == 8 and bit == 0
        if bit == 1:
            if bary_i <= 0.27 or s5 != 8:
                if not bad:
                    first = f"bit=1 at ({i%64},{i//64}): s5={s5}, i={bary_i:.4f}"
                bad += 1
        elif bit == 0:
            if bary_i > 0.2701 or s5 not in (7, 8):
                if not bad:
                    first = f"bit=0 at ({i%64},{i//64}): s5={s5}, i={bary_i:.4f}"
                bad += 1
        else:
            if not bad:
                first = f"unexpected bit={bit}"
            bad += 1
        if s5 == 8:
            n8 += 1
        elif s5 == 7:
            n7 += 1
    if bad:
        return False, first
    detail = f"s5==8: {n8} ({n80} of them with bit 0), s5==7: {n7} ({len(u32)} px)"
    if not n80:
        return False, "no bit=0 pixel got s5=8: the branch was decided per lane, not by the wave; " + detail
    return (n8 - n80 > 0 and n7 > 0), detail


def check_d16_texture(raw_path: str) -> tuple[bool, str]:
    """Point-sample the 8x8 UNORM gradient and round each component to binary16 before widening."""
    data = Path(raw_path).read_bytes()
    if len(data) != 64 * 64 * 16:
        return False, f"expected a 64x64 RGBA32F frame, got {len(data)} bytes"
    levels = [struct.unpack("<e", struct.pack("<e", i * 32 / 255.0))[0] for i in range(8)]
    bad = 0
    first = ""
    for i, pixel in enumerate(struct.iter_unpack("<4f", data)):
        x, y = i % 64, i // 64
        expected = (levels[x // 8], levels[y // 8], levels[4], 1.0)
        if pixel != expected:
            if not bad:
                first = f"at ({x},{y}): got {pixel}, want {expected}"
            bad += 1
    return (False, f"{bad}/4096 mismatches; {first}") if bad else (True, "4096/4096 RGBA pixels exact")


def check_grad(raw_path: str, want_b: float, want_a: float) -> tuple[bool, str]:
    px = read_raw_rgba32f(raw_path)
    if problem := frame_problem(px):
        return False, problem
    bad = 0
    first = ""
    for i, (r, g, b, a) in enumerate(px):
        wx = ((i % 64) + 0.5) / 64.0
        wy = ((i // 64) + 0.5) / 64.0
        if abs(r - wx) > 1e-3 or abs(g - wy) > 1e-3 or abs(b - want_b) > 1e-3 or abs(a - want_a) > 1e-3:
            if not bad:
                first = f"at ({i%64},{i//64}): got ({r:.4f},{g:.4f},{b:.4f},{a:.4f}), want ({wx:.4f},{wy:.4f},{want_b},{want_a})"
            bad += 1
    return (bad == 0), (first if bad else "4096/4096 pixels exact")


def check_clip_half(raw_path: str, want_b: float, want_a: float) -> tuple[bool, str]:
    """Check 64x64 output with ClipDistance[0] = NDC x: left half (x < 32) clipped to clear (-1),
    right half (x >= 32) covered (2048/4096 pixels) with the gradient."""
    px = read_raw_rgba32f(raw_path)
    if problem := frame_problem(px):
        return False, problem
    covered = 0
    bad = 0
    first = ""
    for i, (r, g, b, a) in enumerate(px):
        x, y = i % 64, i // 64
        is_clear = (r == -1.0 and g == -1.0 and b == -1.0 and a == -1.0)
        if not is_clear:
            covered += 1
        if x < 32:
            if not is_clear:
                if not bad:
                    first = f"unclipped pixel at ({x},{y}): got ({r:.4f},{g:.4f},{b:.4f},{a:.4f})"
                bad += 1
        else:
            wx = (x + 0.5) / 64.0
            wy = (y + 0.5) / 64.0
            if is_clear or abs(r - wx) > 1e-3 or abs(g - wy) > 1e-3 or abs(b - want_b) > 1e-3 or abs(a - want_a) > 1e-3:
                if not bad:
                    first = f"at ({x},{y}): got ({r:.4f},{g:.4f},{b:.4f},{a:.4f}), want ({wx:.4f},{wy:.4f},{want_b},{want_a})"
                bad += 1
    ok = (bad == 0 and covered == 2048)
    detail = f"{covered}/4096 pixels covered (expected 2048)" if ok else f"covered={covered}/2048; {first}"
    return ok, detail


# Fixtures run by another harness, which needs what these tools do not have (a D3D12 device, the layer's root
# arguments and code base). They are still assembled and translated here, so a translator regression shows, but no
# checker runs them: the test program that owns them judges them.
OTHER_HARNESS = {"ps_m15": "d3d12_translated_draw_test", "vs_m15": "d3d12_translated_draw_test",
                 "ps_m2": "d3d12_m2_textured_test", "vs_m2": "d3d12_m2_textured_test", "ps_m2_uoff": "d3d12_m2_textured_test",
                 "ps_m2_literal": "d3d12_m2_textured_test", "ps_m2_array": "d3d12_m2_formats_test",
                 "ps_m3": "d3d12_m3_frame_test", "vs_m3": "d3d12_m3_frame_test",
                 "ps_m3_post": "d3d12_m3_frame_test", "ps_m3_query": "d3d12_m3_frame_test",
                 "cs_m4_fill": "d3d12_m4_compute_test", "cs_m4_indirect": "d3d12_m4_compute_test",
                 "cs_buf_typed": "d3d12_buffer_uav_test", "cs_buf_counter": "d3d12_buffer_uav_test",
                 "ps_live_width": "w64_graphics.py"}


# Shaders a selected test draws with, built even when the filter leaves them out (#192).
DRAWN_WITH = {GATHER_LOD_TEST: "vs_ids.asm", "ps_flat": "vs_ids.asm", "ps_persp_bary": "vs_persp.asm", "ps_texture_sample": "vs_ids.asm", "ps_cube_sample": "vs_ids.asm",
              "ps_texture_sample_d16": "vs_ids.asm", "ps_sample_c_lz": "vs_ids.asm", "ps_sample_c_lz_a16": "vs_ids.asm",
              "ps_table_fetch": "vs_ids.asm", "ps_sample_c": "vs_ids.asm", "ps_sample_c_b": "vs_ids.asm",
              **{name: "vs_ids.asm" for name in TEXTURE_CONSTANTS}, "ps_ssharp": "vs_ids.asm", "ps_imgatomic": "vs_ids.asm",
              **{name: "vs_ids.asm" for name in GRADIENT_TESTS}, "vs_clip": "ps_clip_color.asm",
              "ps_interp_f16_high": "vs_f16_pack.asm", "ms_f16_pack": "ps_interp_f16_high.asm",
              "ps_interp_f32_mods": "vs_interp_consts.asm", "ps_interp_mov_mods": "vs_interp_consts.asm",
              SIGNED_MIP_TEST: "vs_ids.asm", UINT_MIP_TEST: "vs_ids.asm",
              "vs_base_ids": "ps_flat.asm"}
DRAWN_WITH.update({name: "vs_ids.asm" for name in GET_LOD_TESTS})


def with_draw_shaders(asm_files: list[str]) -> list[str]:
    build = list(asm_files)
    for f in asm_files:
        extra = DRAWN_WITH.get(shader_source(os.path.splitext(f)[0]))
        if extra and extra not in build:
            build.append(extra)
    if TEXTURE_MIP_TEST + ".asm" in asm_files:
        for name in TEXTURE_MIP_DEPENDENCIES:
            if name + ".asm" not in build:
                build.append(name + ".asm")
    return build


REQUIRED_TEST_PROGRAMS = ("vk_compute_test", "vk_store_test", "vk_harness")
BUILD_CONFIGS = ("RelWithDebInfo", "Release")


def find_test_programs(build: str, ext: str) -> str | None:
    """Return the directory containing all required GPU test programs.

    Single-configuration generators (Ninja, Makefiles) put executables directly
    in the build directory. Visual Studio and other multi-configuration
    generators put them in a configuration subdirectory.
    """
    candidates = [build, *(os.path.join(build, config) for config in BUILD_CONFIGS)]
    for directory in candidates:
        if all(os.path.isfile(os.path.join(directory, name + ext)) for name in REQUIRED_TEST_PROGRAMS):
            return directory
    return None


def native_folder(path: str, ansi: str = "mbcs") -> str | None:
    """`path` spelled so that the native programs this runner starts can open files under it, or None.

    On Windows the test programs and the Vulkan SDK tools (spirv-val, spirv-as) read their command line in the ANSI
    code page, so a character outside it, as in a Cyrillic or CJK user name under %TEMP%, reaches them as '?'. Such a
    folder is replaced by its 8.3 short name, which is plain ASCII, when the volume keeps one. Elsewhere, and for a
    path the code page holds, `path` comes back as it is. `ansi` is the codec of that code page (a parameter so a
    test can pick a narrow one).
    """
    def fits(p: str) -> bool:
        try:
            p.encode(ansi)
            return True
        except UnicodeEncodeError:
            return False

    if os.name != "nt" or fits(path):
        return path
    import ctypes
    buf = ctypes.create_unicode_buffer(32768)
    n = ctypes.windll.kernel32.GetShortPathNameW(path, buf, len(buf))
    short = buf.value if 0 < n < len(buf) else ""
    return short if short and fits(short) else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llvm-mc", default=os.environ.get("LLVM_MC", "llvm-mc"),
                    help="llvm-mc path, or a whole command such as 'wsl -d Ubuntu -e /usr/lib/llvm-21/bin/llvm-mc'")
    ap.add_argument("--build", default=os.path.join(ROOT, "runtime", "build"),
                    help="CMake build folder; with the Visual Studio generator the RelWithDebInfo/Release/... "
                         "subfolder that has the test programs is found by itself")
    ap.add_argument("--work", default=None)
    ap.add_argument("--filter", default="")
    ap.add_argument("--require-pass", action="store_true", help="exit 1 if no GPU test passes")
    ap.add_argument("--assembler", choices=("auto", "spirv-as", "python"), default="auto",
                    help="SPIR-V assembler: auto (use spirv-as if on PATH, else tools/spirv_asm.py), "
                         "spirv-as, or python")
    ap.add_argument("--no-validate", action="store_true",
                    help="skip spirv-val during translation (automatic when --assembler resolves to python "
                         "and spirv-val is not on PATH)")
    ap.add_argument("--disasm-dir", default=None,
                    help="folder of pre-disassembled <name>.s listings (e.g. tests/semantic/fixtures/disasm); "
                         "skips llvm-mc and runs only fixtures present there")
    a = ap.parse_args()
    # Some test programs run with their working directory in <work>/spirv (vk_texture_test), so a relative path
    # given here would point somewhere else for them.
    a.build = os.path.abspath(a.build)
    if a.work:
        a.work = os.path.abspath(a.work)
    if a.disasm_dir is not None:
        a.disasm_dir = os.path.abspath(a.disasm_dir)
        if not os.path.isdir(a.disasm_dir):
            print(f"error: --disasm-dir is not a directory: {a.disasm_dir}", file=sys.stderr)
            return 2

    ext = ".exe" if os.name == "nt" else ""
    required = tuple(t + ext for t in TEST_PROGRAMS)
    a.build = pick_build_dir(a.build, required)
    missing = [t for t in required if not os.path.isfile(os.path.join(a.build, t))]
    if missing:
        sys.exit(not_found_message(a.build, missing))
    requested_build = a.build
    resolved_build = find_test_programs(requested_build, ext)
    if resolved_build is None:
        searched = [requested_build, *(os.path.join(requested_build, config) for config in BUILD_CONFIGS)]
        details = []
        for directory in searched:
            missing = [name + ext for name in REQUIRED_TEST_PROGRAMS
                       if not os.path.isfile(os.path.join(directory, name + ext))]
            details.append(f"  {directory}: {', '.join(missing)}")
        sys.exit("Required GPU test programs were not found together in one build directory.\n"
                 "Missing files by searched directory:\n" + "\n".join(details) + "\n"
                 "Build them with cmake -S runtime -B runtime/build, then cmake --build runtime/build.\n"
                 "For Visual Studio, use: cmake --build <build-dir> --config RelWithDebInfo "
                 "--target vk_compute_test vk_store_test vk_harness.\n"
                 "CMake only builds them when it finds Vulkan. If it printed \"Vulkan SDK not found\", install the "
                 "Vulkan headers (vulkan-headers on Arch) and configure again.")
    a.build = resolved_build

    work = a.work or tempfile.mkdtemp(prefix="leonida_gpu_tests_")
    try:
        try:
            os.makedirs(work, exist_ok=True)
        except OSError as exc:      # the same report run_suite gives for its own folders (#4544)
            print(f"error: cannot prepare GPU test directory {work}: {exc.strerror or exc}", file=sys.stderr)
            return 2
        native = native_folder(work)
        if native is None:
            print(f"error: the work folder {ascii(work)} has characters outside this PC's ANSI code page and no 8.3 "
                  "short name, so the test programs and the Vulkan SDK tools can't open files in it. Pass --work "
                  "with a folder whose path they can, such as C:\\leonida_gpu_work.", file=sys.stderr)
            return 2
        return run_suite(a, native)
    finally:
        if a.work is None:
            shutil.rmtree(work, ignore_errors=True)


def _w64_graphics_only() -> frozenset:
    try:
        import w64_graphics
    except ImportError:
        return frozenset()
    return frozenset(w64_graphics.FIXTURES)


W64_GRAPHICS_ONLY = _w64_graphics_only()


def run_suite(a, work: str) -> int:
    assembler = resolve_assembler(getattr(a, "assembler", "auto"))
    explicit_no_validate = bool(getattr(a, "no_validate", False))
    no_validate = explicit_no_validate or (assembler == "python" and shutil.which("spirv-val") is None)
    disasm_src = getattr(a, "disasm_dir", None)

    asm_files = sorted({f for f in os.listdir(SHADER_DIR) if f.endswith(".asm")} | {n + ".asm" for n in CLIP_CASES})
    # The wave64 pixel fixtures of runtime/test/shaders/w64_graphics.py (tiers A-C, #1541) are checked by that
    # runner, not here: they need a 64-wide fragment subgroup or the halves-aware lowering it drives.
    asm_files = [f for f in asm_files if f[:-4] not in W64_GRAPHICS_ONLY]
    if disasm_src is not None:
        available = {os.path.splitext(f)[0] for f in os.listdir(disasm_src) if f.endswith(".s")}
        asm_files = [
            f for f in asm_files
            if all(os.path.splitext(p)[0] in available for p in with_draw_shaders([f]))
        ]
        if not asm_files:
            print(f"error: --disasm-dir {disasm_src!r} matched no GPU test cases", file=sys.stderr)
            return 2
    if a.filter:
        asm_files = [f for f in asm_files if a.filter in f]
        if not asm_files:
            print(f"error: --filter {a.filter!r} matched no GPU test cases", file=sys.stderr)
            return 2
    build_files = with_draw_shaders(asm_files)

    fssm_dir = os.path.join(work, "fssm")
    disasm_dir = os.path.join(work, "disasm")
    spirv_dir = os.path.join(work, "spirv")
    raw_dir = os.path.join(work, "raw")
    for directory in (work, fssm_dir, disasm_dir, spirv_dir, raw_dir):
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError as exc:
            print(f"error: cannot prepare GPU test directory {directory}: {exc.strerror or exc}", file=sys.stderr)
            return 2

    # 1. Create synthetic DXBC / FSSM metadata + assemble .asm -> .s
    disasm_files = []
    for fname in build_files:
        name = os.path.splitext(fname)[0]
        if name in CLIP_CASES and name != "vs_clip_undeclared":
            dxbc = clip_container()
        elif name == "gs_tri":
            dxbc = make_dxbc([
                (b"ISG1", make_sig_chunk([(1, 3, 0, 0xF, 0xF), (0, 3, 1, 0xF, 0xF)])),
                (b"OSG1", make_sig_chunk([(1, 3, 0, 0xF, 0xF), (0, 3, 1, 0xF, 0xF)])),
            ])
        elif name in ("ms_tri", "ms_w64x_ballot", "ms_wave_info_s3", "ms_f16_pack"):
            dxbc = make_dxbc([(b"PSV0", make_ms_psv0(128, 64, 64, 2))])
        elif name == "ms_v5_workgroup":
            dxbc = make_dxbc([(b"PSV0", make_ms_psv0(32, 3, 1, group_shared=128))])
        elif name == "ms_clip":
            dxbc = make_dxbc([
                (b"PSV0", make_ms_psv0(128, 64, 64, 2)),
                # OSG1: SV_Position (sysv=1) reg 0 xyzw, SV_ClipDistance (sysv=2) reg 1 x
                (b"OSG1", make_sig_chunk([(1, 3, 0, 0xF, 0xF), (2, 3, 1, 0x1, 0xE)])),
            ])
        else:
            dxbc = make_dxbc([])
        with open(os.path.join(work, f"{name}.dxbc"), "wb") as f:
            f.write(dxbc)
        if name.startswith("cs_"):
            with open(os.path.join(fssm_dir, f"{name}.fssm"), "wb") as f:
                f.write(compute_header(os.path.join(SHADER_DIR, fname)))
        fname = shader_source(name) + ".asm"
        disasm_path = os.path.join(disasm_dir, f"{name}.s")
        if disasm_src is not None:
            try:
                shutil.copyfile(os.path.join(disasm_src, f"{name}.s"), disasm_path)
            except shutil.SameFileError:
                # A replay can use the work folder's existing listings as its source.
                pass
        else:
            assemble_to_s(a.llvm_mc, os.path.join(SHADER_DIR, fname), disasm_path, name)
        disasm_files.append(disasm_path)
        # A failed translation must not run an old binary left in a reused --work directory.
        Path(spirv_dir, f"{name}.spv").unlink(missing_ok=True)

    # 2. Translate this run's inputs only. The translator looks for metadata one level above the input directory.
    with tempfile.TemporaryDirectory(prefix="disasm-", dir=work) as selected_disasm:
        for path in disasm_files:
            shutil.copyfile(path, os.path.join(selected_disasm, os.path.basename(path)))
        cmd = [sys.executable, "-I", TRANSLATOR, selected_disasm, "--out", spirv_dir]
        if assembler == "python":
            cmd += ["--assembler", "python"]
        if no_validate:
            if not explicit_no_validate:
                print("note: spirv-val not found on PATH; translating with --no-validate", flush=True)
            cmd.append("--no-validate")
        if w64_emulated():   # wave64 compute/mesh exact on any subgroup size
            cmd.append("--wave64-emulate")
        if bary_fallback():   # raw barycentrics from an interpolated input, fed by a pass-through GS
            cmd.append("--bary-fallback")
        if f16_interp_fallback():   # *_f16 interps from decoded-half inputs, no PerVertexKHR (MoltenVK)
            cmd.append("--f16-interp-fallback")
        if TEXOFF_FALLBACK:   # image_sample*_o: offset folded into the coordinate, no VK_KHR_maintenance8
            cmd.append("--texel-offset-fallback")
        if MESH_COMPUTE:   # mesh shaders as compute writing to a buffer, drawn by tools/mesh_compute_vs.py
            cmd.append("--mesh-compute")
        if pc_tail_ubo():   # 128 push-constant bytes; heap base and code base from a uniform block at set 1
            cmd.append("--pc-tail-ubo")
        if os.environ.get("LEONIDA_FMA") in ("glsl", "khr", "exact"):   # GLSL Fma, OpFmaKHR or emulated (docs/design.md)
            cmd += ["--fma", os.environ["LEONIDA_FMA"]]
        rc, out = run_cmd(cmd, gpu=False)
    print("=== rdna2spirv.py ===")
    print(out)
    if rc != 0:
        return 1
    if bary_fallback():
        import bary_fallback_gs
        for gs in bary_fallback_gs.build_for_index(spirv_dir):
            print("bary fallback GS:", os.path.basename(gs))

    if MESH_COMPUTE:
        import mesh_compute_vs
        for vs_spv in mesh_compute_vs.build_for_index(spirv_dir):
            print("mesh compute VS:", os.path.basename(vs_spv))

    # 3. Build helper SPIR-V shaders
    vs_tri = os.path.join(spirv_dir, "vs_tri.spv")
    fs_passthru = os.path.join(spirv_dir, "fs_passthru.spv")
    fs_loc1 = os.path.join(spirv_dir, "fs_loc1.spv")
    spirv_as(VS_TRI_SPVASM, vs_tri, assembler=assembler)
    spirv_as(make_fs_passthru(0), fs_passthru, assembler=assembler)
    spirv_as(make_fs_passthru(1), fs_loc1, assembler=assembler)

    index = {}
    idx_path = os.path.join(spirv_dir, "index.json")
    if os.path.isfile(idx_path):
        with open(idx_path, encoding="utf-8") as source:
            index = json.load(source)

    ext = ".exe" if os.name == "nt" else ""
    vk_compute = os.path.join(a.build, "vk_compute_test" + ext)
    vk_store = os.path.join(a.build, "vk_store_test" + ext)
    vk_harness = os.path.join(a.build, "vk_harness" + ext)
    vk_bindless = os.path.join(a.build, "vk_bindless_test" + ext)

    results: list[tuple[str, bool, str]] = []
    for fname in asm_files:
        name = os.path.splitext(fname)[0]
        spv = os.path.join(spirv_dir, f"{name}.spv")
        if name == "ps_ij_computed" and (bary_fallback() or f16_interp_fallback()):
            # Before the .spv check: without PerVertexKHR inputs the translator refuses a computed I/J pair, so there
            # is no .spv (the f16 fallback is on by default where PerVertexKHR doesn't compile: MoltenVK).
            results.append((name, None, "SKIP (computed I/J pairs need PerVertexKHR inputs: "
                                        "no --bary-fallback / --f16-interp-fallback)"))
            continue
        if not os.path.isfile(spv):
            results.append((name, False, "missing .spv (translation failed)"))
            continue
        sg = index.get(name, {}).get("subgroup_size")
        cs_env = dict(os.environ, VKC_SUBGROUP_SIZE=str(sg)) if sg else None
        gfx_env = dict(os.environ, VKH_SUBGROUP_SIZE=str(sg)) if sg else None

        if name in CLIP_CASES:
            detail = clip_test(a.build, spirv_dir, name)
            results.append((name, None if isinstance(detail, Skip) else not detail, detail or "four pixel masks exact"))
        elif name == "cs_workgroup_yz":
            rc, msg = run_cmd([vk_compute, spv, "2x3x4", "workgroup_yz"], env=cs_env)
            results.append((name,) + compute_result(rc, msg, rc == 0 and "2x3x4 workgroup Y/Z: 0 mismatches"
                                                    in msg.splitlines(), msg))
        elif name == "cs_lds":
            rc, msg = run_cmd([vk_compute, spv, "4", "mirror64"], env=cs_env)
            results.append((name,) + compute_result(rc, msg, rc == 0, verdict_line(rc, msg)))
        elif name == "cs_exec64":
            rc, msg = run_cmd([vk_compute, spv, "1", "exec64"], env=cs_env)
            results.append((name,) + compute_result(rc, msg, rc == 0, verdict_line(rc, msg)))
        elif name in ("cs_bvh", "cs_bvh32"):   # the 64-bit and the 32-bit node pointer: one scene, the same words
            rc, msg = run_cmd([vk_compute, spv, "1", "bvh"], env=cs_env)
            results.append((name,) + compute_result(rc, msg, rc == 0, verdict_line(rc, msg)))
        elif name in STORE_WORDS:
            # TEST-184: image atomics (cs_imgatomic, cs_imgatomic3d) and image_get_resinfo (cs_resinfo). vk_store_test
            # binds the images and prints the r32 ones after the dispatch; the expected words are check.py's.
            rc, msg = run_cmd([vk_store, spv, "words"])
            if rc == UNSUPPORTED:
                results.append((name, None, unsupported_skip(msg)))
            elif rc != 0:
                results.append((name, False, msg))
            else:
                problem = store_words_problem(msg, name)
                results.append((name, not problem, problem or "all 64 words of the uint image, the sint image and "
                                "the volume exact"))
        elif name in GET_LOD_TESTS:
            # TEST-184: image_get_lod on vk_texture_test's two-mip texture (two sampler LOD ranges) and on its 2D array.
            problem = get_lod_test(a.build, spirv_dir, name)
            ok = None if isinstance(problem, Skip) else not problem
            results.append((name, ok, problem or f"4096 pixels in each of {len(GET_LOD_TESTS[name])} draw(s): the "
                            "clamped and the calculated LOD within 1/16"))
        elif name in ("cs_bounds", "cs_smem_bounds", "cs_smem_stride_align", "cs_smem_sload_neg", "cs_smem_sconst_off", "cs_buf_tail", "cs_oob_select", "cs_oob_tail",
                      "cs_buf_swizzle", "cs_cmpx_class_f32", "cs_cvt_pk_u8", "cs_pk_mul_lo_const", "cs_buf_load_sub", "cs_buf_fmt8"):
            # num_records bounds check (cs_bounds.asm): the out-of-bounds store is dropped and the out-of-bounds
            # load reads 0, so thread 0's four output words are fixed. One thread, no wave64 requirement.
            # cs_cmpx_class_f32 and cs_cvt_pk_u8 are per-lane and need no particular subgroup width either.
            # The other four pin the SMEM and raw MUBUF rules measured on an RX 6800 (derivations in each .asm).
            rc, msg = run_cmd([vk_compute, spv, "1", "dump"], env=cs_env)
            want = {
                "cs_bounds": [0x12345678, 0x12345678, 0, 0xEEEEEEEE],
                "cs_smem_bounds": [0, 0x5EED0010, 0x5EED000C, 0, 0x5EED000C, 0x5EED0010, 0, 0],
                "cs_smem_stride_align": [0x5EED0010, 0, 0, 0x5EED0000, 0x5EED0000, 0x5EED0000, 0x5EED0000,
                                         0x5EED0004],
                # never run with a translator that zero-extends the s_load immediate (reads 4 GiB past the heap)
                "cs_smem_sload_neg": [0x5EED0004, 0x5EED0000, 0x5EED0004, 0x5EED0004, 0x5EED0008, 0x5EED000C,
                                      0xEEEEEEEE, 0xEEEEEEEE],
                # SHADER-328: a constant SGPR offset gives what the run-time one gives (no 32-bit wrap)
                "cs_smem_sconst_off": [0, 0x5EED0018, 0x5EED0018, 0x5EED0014],
                "cs_buf_tail": [0, 0x5EED000C, 0x5EED0008, 0x5EED000C, 0, 0, 0x5EED0010, 0],
                "cs_oob_select": [0x5EED000C, 0, 0x5EED0014, 0, 0x5EED0018, 0xEEEEEEEE, 0, 0],
                "cs_oob_tail": [0, 0, 0x5EED0010, 0, 0xBEEF0000, 0xEEEEEEEE, 0, 0x5EED0010],
                # swizzled MUBUF addressing (measured natively on gfx1036) and ADD_TID_ENABLE (lanes 0 and 1 store)
                "cs_buf_swizzle": [0x5EED0024, 0x5EED0044, 0x5EED004C, 0x5EED0008, 0x5EED000C, 0, 0xBEEF0000,
                                   0xBEEF0001],
                "cs_cmpx_class_f32": CMPX_CLASS_WORDS,
                "cs_cvt_pk_u8": CVT_PK_U8_WORDS,
                "cs_pk_mul_lo_const": PK_MUL_LO_WORDS,
                # TEST-510: sub-dword MUBUF loads (RAW end check by element size) and GFX10_FORMAT_8_UINT
                "cs_buf_load_sub": [0x80, 0xFFFFFF80, 0x1FF, 0xFFFF807F, 0xAB01, 0xCD, 0, 0],
                "cs_buf_fmt8": [0x7F, 0x80, 0xFF, 0x01, 0, 0, 0xFF, 1],
            }[name]
            words = dump_words(msg)
            vals = words[:len(want)] if words else []
            results.append((name,) + compute_result(rc, msg, rc == 0 and vals == want,
                                                    " ".join(hex(x) for x in vals) if vals else msg))
        elif name == "cs_fmt_1010102":
            problem = fmt_1010102_test(a.build, spirv_dir, cs_env)
            results.append((name, not problem, problem or "64/64 dumped words exact"))
        elif name == "cs_fmt_sint32":
            problem = fmt_sint32_test(a.build, spirv_dir, cs_env)
            results.append((name, not problem, problem or "64/64 dumped words exact"))
        elif name == "cs_fmt_int_narrow":
            problem = fmt_int_narrow_test(a.build, spirv_dir, cs_env)
            results.append((name, not problem, problem or "136/136 dumped words exact"))
        elif name == "cs_fmt_xy_xyz":
            # TEST-185: xy/xyz format loads and stores, words measured natively on a Radeon 680M (the .asm header)
            problem = fmt_xy_xyz_test(a.build, spirv_dir, cs_env)
            results.append((name, not problem, problem or "128/128 dumped words exact"))
        elif name == "cs_tbuffer_store":
            # tbuffer_store_format_*: the instruction's format, words measured natively on a Radeon 680M (the .asm header)
            problem = tbuffer_store_test(a.build, spirv_dir, cs_env)
        elif name == "cs_fmt_d16_xyz":
            # d16_xyz / d16_xy typed-buffer loads and d16_xyz stores (cs_fmt_d16_xyz.asm header)
            problem = fmt_d16_xyz_test(a.build, spirv_dir, cs_env)
            results.append((name, not problem, problem or "80/80 dumped words exact"))
            results.append((name, not problem, problem or "128/128 dumped words exact"))
        elif name in ("cs_global_neg_offset", "cs_global_mem_ext"):
            # Negative global_load/global_store offsets (cs_global_neg_offset.asm): 40 groups give a 0x2800-byte
            # heap holding both the right addresses and the 4096-lower ones. One thread, no wave64 requirement.
            # cs_global_mem_ext.asm: global/buffer atomics, 8/16-bit global stores and d16 loads in the same heap.
            rc, msg = run_cmd([vk_compute, spv, "40", "dump"], env=cs_env)
            want = {"cs_global_neg_offset": [0x5EED1FFC, 0x8BAD1800, 0x5EED1FF0, 0x5EED1FF4, 0x9E, 0xFFFF8BAD,
                                             0x57023FFC, 0xEEEEEEEE],
                    "cs_global_mem_ext": [0x5, 0xC, 0x977, 0xFFF00000, 0xEDCB5678, 0xEF332211, 0x887766BE, 0xFF801234]}[name]
            ok, detail = judge_cs_dump(rc, msg, want, wave64=False, partial=False, emulated=False, hexwords=True)
            results.append((name, ok, detail))
        elif name == "cs_heap_init":
            # vk_compute_test copies VKC_HEAP_INIT's file into the heap; the shader copies its words at 0x100 to 0.
            rc, msg = run_cmd([vk_compute, spv, "1", "dump"], env=dict(cs_env or os.environ, **heap_init_env()))
            words = dump_words(msg)
            vals = words[:len(HEAP_INIT_EXPECTED)] if words else []
            results.append((name,) + compute_result(rc, msg, rc == 0 and vals == HEAP_INIT_EXPECTED,
                                                    " ".join(hex(x) for x in vals) if vals else msg))
        elif name == "cs_user_sgprs":
            # vk_compute_test seeds s4..s7 from VKC_SGPRS; the shader stores them back.
            rc, msg = run_cmd([vk_compute, spv, "1", "dump"], env=dict(cs_env or os.environ, **USER_SGPRS_ENV))
            words = dump_words(msg)
            vals = words[:len(USER_SGPRS_WORDS)] if words else []
            results.append((name,) + compute_result(rc, msg, rc == 0 and vals == USER_SGPRS_WORDS,
                                                    " ".join(hex(x) for x in vals) if vals else msg))
        elif name == "cs_f64_add":
            rc, msg = run_cmd([vk_compute, spv, "1", "dump"], env=cs_env)
            if rc == 77:   # Float64, or Int64 where the module declares both: quote whichever it names
                results.append((name, None, compute_unsupported(rc, msg) or "SKIP (no shaderFloat64 on this device)"))
            else:
                ok, detail = judge_cs_dump(rc, msg, [0, 0x40040000], wave64=False, partial=False, emulated=False,
                                           hexwords=True)
                results.append((name, ok, detail))
        elif name == "cs_div_fmas_f64":
            # One-rounding binary64 FMA: strict where the translator emitted OpFmaKHR on doubles (shaderFmaFloat64,
            # --fma khr); under the default GLSL Fma the split result (0) is allowed, as cs_fma_fused allows it.
            rc, msg = run_cmd([vk_compute, spv, "1", "dump"], env=cs_env)
            if rc == UNSUPPORTED:      # no shaderFloat64, or (--fma khr) no shaderFmaFloat64
                results.append((name, None, unsupported_skip(msg)))
            else:
                words = (dump_words(msg) or [])[:2] if rc == 0 else []
                strict = "shaderFmaFloat64" in ((index.get(name) or {}).get("features") or ())
                fused, split = [0, 0x3c300000], [0, 0]
                ok = rc == 0 and (words == fused or (not strict and words == split))
                detail = " ".join(hex(x) for x in words) if words else msg
                if ok and words == split:
                    detail += " (split: the driver rounds GLSL Fma on doubles twice; --fma khr fixes it)"
                results.append((name, ok, detail))
        elif name == "cs_cvt_f32_sat":
            # #300's saturating float -> int conversions: one dispatch per input word in s4 (VKC_SGPRS).
            wrong, skip = [], None
            for bits, unsigned, signed in CVT_F32_SAT_CASES:
                rc, msg = run_cmd([vk_compute, spv, "1", "dump"],
                                  env=dict(cs_env or os.environ, VKC_SGPRS=f"{bits:#010x}"))
                if skip := compute_unsupported(rc, msg):   # the module, not this input: no other input can run either
                    break
                words = dump_words(msg)
                if rc != 0 or not words or words[:2] != [unsigned, signed]:
                    got = " ".join(hex(x) for x in words[:2]) if words else msg.splitlines()[-1] if msg else "no output"
                    wrong.append(f"{bits:#010x} gave {got}, want {unsigned:#x} {signed:#x}")
            n = len(CVT_F32_SAT_CASES)
            if skip:
                results.append((name, None, skip))
            else:
                results.append((name, not wrong, f"{n}/{n} inputs exact" if not wrong
                                else f"{len(wrong)}/{n} inputs wrong, first {wrong[0]}"))
        elif name == "cs_fma_fused":
            # Single-rounded f32 FMA (+ f16 round-to-odd): strict under LEONIDA_FMA=khr (OpFmaKHR, exit 77 if
            # unsupported), fused-or-split f32 allowed under default --fma glsl (e.g. Mesa RADV splits 5/7).
            rc, msg = run_cmd([vk_compute, spv, "1", "dump"], env=dict(cs_env or os.environ, **FMA_FUSED_ENV))
            if rc == UNSUPPORTED:
                results.append((name, None, unsupported_skip(msg)))
            elif rc != 0:
                results.append((name, False, msg))
            else:
                # TEST-1147: judge by the mode the translator recorded in index.json, as check.py does.
                # --fma exact has no KHR capability, so capability/env alone used to accept split words here.
                meta = index.get("cs_fma_fused") or {}
                mode = meta.get("fma") or ""
                if mode:
                    # The recorded mode wins over a stale LEONIDA_FMA, as in check.py.
                    strict = mode in ("khr", "exact")
                else:
                    strict = ("shaderFmaFloat32" in (meta.get("features") or ())
                              or os.environ.get("LEONIDA_FMA") in ("khr", "exact"))
                ok, detail = judge_fma_fused(dump_words(msg), khr=strict)
                results.append((name, ok, detail))
        elif name == "cs_wavemask":
            # Whole-wave mask test, as in check.py: needs a 64-wide subgroup, skipped where the device has none.
            # With LEONIDA_W64_EMULATE=1 it was translated with --wave64-emulate and runs on any subgroup size.
            results.append((name,) + wavemask_check(vk_compute, spv, cs_env, w64_emulated()))
        elif name in ("cs_store", "cs_signed_zero_nan", "cs_smem_soffset_imm", "cs_w32_lane_wrap", "cs_movrelsd_2", "cs_w32_permute", "cs_w32_sincos", "cs_w32_buffer_sub16", "cs_w32_buffer_store_sub", "cs_w32_fcmpswap", "cs_w32_permlane_fi", "cs_exec_const", "cs_w32_mask_imm", "cs_w64_mask_imm", "cs_w32_lds_u8_d16", "cs_w32_lds_subdword", "cs_w32_lds_wide", "cs_w32_int_ext", "cs_w32_salu64", "cs_w32_misc_ext", "cs_w32_new_ops", "cs_wave64_ballot", "cs_wm_half_ops", "cs_wm_half_cmp", "cs_wm_bfm64", "cs_wm_scc_rfl", "cs_wqm_w64", "cs_wqm_w32", "cs_wqm_w64_half",
                      "cs_dpp", "cs_dpp_row", "cs_dpp_mask", "cs_dpp8", "cs_dpp_ops", "cs_dpp_bank", "cs_dpp_fi",
                      "cs_dpp_bound", "cs_dpp_w64",
                      "cs_w64_mask_detect",
                      "cs_w64x_readlane", "cs_w64x_writelane", "cs_w64x_readfirstlane", "cs_w64x_bpermute", "cs_w64x_permute",
                      "cs_w64x_branch", "cs_w64x_ballot", "cs_w64x_wg128_branches", "cs_w64x_lds_raw",
                      "cs_w64x_lds_war", "cs_w64x_partial", "cs_w64x_sbarrier", "cs_w64x_load_ballot_load",
                      "cs_w64x_mask_reuse", "cs_w64x_mask_reuse2", "cs_w64x_chain_barrier", "cs_barrier_divergent",
                      "cs_bpermute32", "cs_w64x_partial_scmp", "cs_w32_partial_scmp", "cs_barrier_early_end",
                      "cs_barrier_loop", "cs_w64x_andn2_exec", "cs_w64x_saveexec_b64", "cs_lds_b64_cmpst", "cs_w32_andn2_exec",
                      "cs_mask_vgpr_exec"):
            cs_dump_wants = {
                "cs_store": [0x400080FF, 0xE00003FF, 0x3C004000, 0xEEEEEEEE, 0x40003C00, 0xBC003800, 0x40FF8000, 0x3FF00000],
                # base + soffset + imm (cs_smem_soffset_imm.asm); out[4..7] are seeds the shader leaves alone
                "cs_smem_soffset_imm": [0x5EED0024, 0x5EED0028, 0x5EED002C, 0x5EED0030,
                                        0x5EED0010, 0x5EED0014, 0x5EED0018, 0x5EED001C],
                # SHADER-52: wave32 lane select & 31 (40 → 8)
                "cs_w32_lane_wrap": [25, 16, 777, 777],
                # v_movrelsd_2_b32: VGPR[D + M0[25:16]] = VGPR[S0 + M0[9:0]] (cs_movrelsd_2.asm)
                "cs_movrelsd_2": [11, 33, 33, 55, 1717, 77, 1515, 1717],
                # SHADER-90: ds_permute_b32 (cs_w32_permute.asm): scatter, unwritten lane = 0, EXEC gates the write
                "cs_w32_permute": [218, 29, 8, 211, 0, 15, 99, 15],
                # integer ops with CPU-only checks until now (cs_w32_int_ext.asm)
                "cs_w32_int_ext": [1, 27, 1, 0x33332222, 0xC0000000, 0x4000, 18, 0x20304050],
                # 64-bit scalar data ops and v_mad_u64_u32 (cs_w32_salu64.asm)
                # more ops with CPU-only checks until now (cs_w32_misc_ext.asm)
                # ops merged on 2026-11-22 with CPU-only checks (cs_w32_new_ops.asm)
                # plain 8/16-bit MUBUF stores, one crossing a dword (cs_w32_buffer_store_sub.asm)
                "cs_w32_buffer_store_sub": [0x1122AB44, 0xCDEF7788, 0x34AABBCC, 0xDDDDDD12, 0x0102035A, 0x05061234, 0xAB, 0xFFFFCDEF],
                # 8/16-bit MUBUF d16 loads and stores (cs_w32_buffer_sub16.asm)
                "cs_w32_buffer_sub16": [0xBEEF00AA, 0x00AA1234, 0xBEEFFFAA, 0xFFAA1234, 0xBEEF8899, 0x88991234, 0x1122CD44, 0xABCD7788],
                # ds_cmpst_f32 and buffer_atomic_fcmpswap: float compare (cs_w32_fcmpswap.asm)
                "cs_w32_fcmpswap": [0, 0x3F800000, 0x7FC00000, 0x3F800000, 0x80000000, 0x40400000, 0x3F800000, 0x7FC00000],
                "cs_w32_new_ops": [0x55, 0x8001, 0xC0000000, 0x80014000, 0x4000FFFF, 0x77, 0x99, 0x1234],
                "cs_w32_misc_ext": [0x0FFF0FFF, 0xE2C4A688, 0x0007FFFF, 0x7FFF8000, 0xFFFE0002, 0xFFFF0005, 0x509, 0xCAFEF00D],
                "cs_w32_salu64": [0x1E6A2C48, 0x1234, 0xFFFFFFFF, 0x100, 47, 48, 0xFFFF8000, 0xFFFFFFFE],
                # SHADER-85: ds_read_u8/i8_d16[_hi] (cs_w32_lds_u8_d16.asm header)
                "cs_w32_lds_u8_d16": [0xBEEF00FF, 0x00FF1234, 0xBEEFFF80, 0xFF801234, 0xBEEF007F, 0x007F1234, 0xBEEF0080, 0xFFFF1234],
                # 8/16-bit and 64/96/128-bit LDS accesses (cs_w32_lds_subdword.asm / cs_w32_lds_wide.asm headers)
                "cs_w32_lds_subdword": [0x11B2D444, 0xA1B2C3D4, 0xFFFFFFB2, 0x000000B2, 0x00000011, 0xFFFFA1B2, 0x0000A1B2, 0xFFFFC3D4],
                "cs_w32_lds_wide": [0xCAFEF00D, 0xFEEDFACE, 0x600DD00D, 0x00000000, 0x8BADF00D, 0xDEADC0DE, 0x12345678, 0x9ABCDEF0],
                # ds_cmpst_b32 and the b64 2-address forms (cs_lds_b64_cmpst.asm header)
                "cs_lds_b64_cmpst": [0x11110001, 0x22220002, 0x33330003, 0x44440004,
                                     0xCCCC000C, 0x44440004, 0xBBBB000B, 0x3579BADC],
                # constant EXEC masks (cs_exec_const.asm): each lane takes its own bit of the constant
                "cs_exec_const": [16, 8, 2, 16, 0, 16],
                # constant lane masks: lane L takes bit L (cs_w32_mask_imm.asm / cs_w64_mask_imm.asm headers)
                "cs_w32_mask_imm": [2056, 20, 2, 6152, 2050, 4112, 258, 32],
                "cs_w64_mask_imm": [2056, 1084, 828, 2052, 319, 2104, 516, 2064],
                "cs_dpp": [1, 0, 3, 2, 5, 4, 7, 6],
                "cs_dpp_row": [0, 0, 1, 2, 3, 4, 5, 6],
                "cs_dpp_mask": [1, 0, 3, 2, 0xEEEEEEEE, 0xEEEEEEEE, 0xEEEEEEEE, 0xEEEEEEEE],
                "cs_dpp8": [7, 6, 5, 4, 3, 2, 1, 0],
                "cs_dpp_ops": [100, 10, 11, 12, 13, 14, 15, 16],
                "cs_dpp_bound": [10, 10, 11, 12, 13, 14, 15, 16],
                "cs_dpp_bank": [0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFFFD, 0xFFFFFFFC, 7, 7, 7, 7],
                "cs_dpp_fi": [1, 2, 3, 0x7777, 5, 6, 7, 0x7777],
                "cs_dpp_w64": [1, 0, 3, 2, 5, 4, 7, 6],
                # SHADER-99: -0 and NaN survive (SignedZeroInfNanPreserve, cs_signed_zero_nan.asm); 0 in out[4..7]
                # without it
                "cs_signed_zero_nan": [0x80000000, 0x80000000, 0x80008000, 0x00008000, 1, 1, 1, 0x00007C00],
                "cs_wave64_ballot": [32, 16, 24, 40, 8, 48],
                "cs_wm_half_ops": [32, 16, 16, 0, 16],
                "cs_wm_half_cmp": [1, 0, 1, 1],
                "cs_wm_bfm64": [16, 8, 8, 40],
                "cs_wm_scc_rfl": [55, 48],
                "cs_w64_mask_detect": [48],
                "cs_wqm_w64": [20, 4, 12, 8, 1, 0, 15728895, 20],       # cs_wqm_w64.asm / cs_wqm_w32.asm headers
                "cs_wqm_w32": [12, 1, 4, 0, 8, 15728880, 4, 32],
                "cs_wqm_w64_half": [11, 14, 0, 1, 6, 8, 12, 2],
                # cross-half tests (derivations in each .asm); need exact wave64 or its emulation
                "cs_w64x_readlane": [121, 121, 16, 97, 16, 100, 100, 121],
                "cs_w64x_writelane": [4660, 77, 99, 2748, 0, 0, 4660, 99],
                "cs_w64x_readfirstlane": [140, 140, 163, 132, 100, 105, 133, 48],
                "cs_w64x_bpermute": [36, 316, 253, 218, 435, 225, 0, 442],
                "cs_w64x_permute": [218, 442, 435, 239, 0, 99, 239, 232],      # SHADER-90, cs_w64x_permute.asm
                "cs_w64x_branch": [0, 0, 0, 0, 5, 5, 9, 9],
                "cs_w64x_mask_reuse": [8, 40, 8, 40, 2, 20, 10, 64],
                "cs_w64x_mask_reuse2": [10, 20, 8, 8, 5, 2, 132, 131],
                "cs_w64x_chain_barrier": [564, 563, 500, 627],
                "cs_w64x_ballot": [2053, 2061, 2069, 2077, 2085, 2093, 2101, 2109],
                "cs_w64x_wg128_branches": [121, 121, 343, 343, 14, 31, 19812, 22728],
                "cs_w64x_lds_raw": [132, 137, 100, 108, 131, 101, 149, 118],
                "cs_w64x_lds_war": [1032, 1063, 1000, 1031, 2032, 2063, 2000, 2031],
                "cs_w64x_partial": [40, 40, 528, 0, 0, 532, 539, 524820],
                "cs_w64x_partial_scmp": [0, 1, 0, 1, 1, 1, 1, 0],     # s_cmp mask, -1 in a partial wave
                "cs_w32_partial_scmp": [0, 1, 1, 0, 1, 1, 1, 1],      # the same in a 20-thread wave32
                "cs_w32_andn2_exec": [1, 8, 16, 2064, 2068, 3, 20, 1544],   # s_andn2_saveexec/wrexec_b32 (.asm header)
                "cs_w32_sincos": [1, 1, 1, 1, 1, 1],                  # SHADER-88 (cs_w32_sincos.asm): angles in turns
                # SHADER-74: v_permlane16/x16 FI / BOUND_CTRL with the source lane off in EXEC (cs_w32_permlane_fi.asm)
                "cs_w32_permlane_fi": [1000, 0, 4, 1, 1000, 0, 52, 49],
                "cs_w64x_sbarrier": [605, 605, 686, 686, 3032, 3013, 3032, 3031],
                "cs_w64x_load_ballot_load": [43, 74, 75, 42, 42, 18, 41, 32],
                "cs_w64x_andn2_exec": [1, 16, 32, 4144, 2052, 3, 8, 1544],   # s_andn2_saveexec/wrexec_b64 (.asm header)
                "cs_w64x_saveexec_b64": [6160, 12304, 5120, 2088, 6168, 8192, 7176, 1837171],   # .asm header
                "cs_bpermute32": [0, 50, 162, 99, 36, 218, 183, 15],     # wave32: needs 32 lanes
                # SHADER-91: s_barrier sites in different wave-uniform branches meet (cs_barrier_divergent.asm)
                "cs_barrier_divergent": [127, 126, 125, 124, 123, 122, 121, 120],
                # SHADER-91: other waves end without s_barrier (cs_barrier_early_end.asm); a loop whose trip count
                # differs per wave (cs_barrier_loop.asm)
                "cs_barrier_early_end": [127, 126, 125, 124, 123, 122, 121, 120],
                "cs_barrier_loop": [128, 96, 64, 32, 128, 96, 64, 0xEEEEEEEE],
                # SHADER-21 part 2: a mask copied to a VGPR with v_mov_b32 respects EXEC —
                # inactive lanes keep the old VGPR (cs_mask_vgpr_exec.asm)
                "cs_mask_vgpr_exec": [0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
                                      0xDEADBEEF, 0xDEADBEEF, 0xDEADBEEF, 0xDEADBEEF],
            }
            want = cs_dump_wants[name]
            # Every dump but cs_store / cs_smem_soffset_imm / cs_w32_lane_wrap / cs_movrelsd_2 / cs_w32_permute /
            # cs_w32_lds_u8_d16 / cs_w32_sincos / cs_w32_int_ext / cs_w32_salu64 / cs_w32_misc_ext / cs_w32_new_ops / cs_signed_zero_nan, the wave32
            # cs_wqm_w32 / cs_bpermute32 and the DPP tests is a whole-wave (wave64) mask test: ask for a 64-wide subgroup and skip,
            # as check.py does, where the device has none
            # (vk_compute_test exits 77) instead of reporting a failure. With LEONIDA_W64_EMULATE=1 they were
            # translated with --wave64-emulate and run on any subgroup size.
            # cs_w64x_partial and cs_w64x_partial_scmp (40-thread groups) can't have full subgroups: they ask for
            # size 64 without them.
            emulated = w64_emulated()
            partial = name in ("cs_w64x_partial", "cs_w64x_partial_scmp")
            wave64 = name not in ("cs_store", "cs_signed_zero_nan", "cs_smem_soffset_imm", "cs_wqm_w32", "cs_w32_lane_wrap", "cs_movrelsd_2",
                                  "cs_w32_permute", "cs_barrier_divergent", "cs_barrier_early_end", "cs_barrier_loop",
                                  "cs_w32_lds_u8_d16", "cs_w32_int_ext", "cs_w32_salu64", "cs_w32_misc_ext", "cs_w32_new_ops",
                                  "cs_w32_lds_subdword", "cs_w32_lds_wide",
                                  "cs_bpermute32", "cs_w32_partial_scmp", "cs_w32_permlane_fi", "cs_w32_sincos", "cs_w32_buffer_sub16", "cs_w32_buffer_store_sub", "cs_w32_fcmpswap",
                                  "cs_lds_b64_cmpst", "cs_exec_const", "cs_w32_mask_imm", "cs_mask_vgpr_exec",
                                  "cs_dpp", "cs_dpp_row", "cs_dpp_mask", "cs_dpp8",
                                  "cs_dpp_ops", "cs_dpp_bound", "cs_dpp_bank", "cs_dpp_fi", "cs_w32_andn2_exec") and not emulated and not partial
            if partial and not emulated:
                cs_env = dict(os.environ, VKC_SUBGROUP_SIZE="64")
            if name in WAVE32_LANE_DUMPS:   # 32 lanes even without index.json, as check.py requests (#774)
                cs_env = dict(os.environ, **WAVE32_LANE_ENV)
            rc, msg = run_cmd([vk_compute, spv, "1", "dump"] + (["--require-wave64"] if wave64 else []), env=cs_env)
            skip = None
            if name in WAVE32_LANE_DUMPS:
                if rc == UNSUPPORTED:
                    results.append((name, None, unsupported_skip(msg)))
                    continue
                if rc == 0:
                    skip = wave32_lane_skip(msg)
            if skip is None and name in WAVE32_PIN_DUMPS:
                # the device may not be able to run the wave size this fixture was written for (a 64-only device
                # can't pin 32, so cs_barrier_loop's words mean nothing); check.py skips it too
                skip = subgroup_pin_skip(msg, cs_env)
            if skip:   # e.g. Mesa llvmpipe's 8-wide subgroups: these 32-lane masks can't come out right
                ok, detail = None, str(skip)
            else:
                want = judged_words(name, msg, want)   # cs_signed_zero_nan on Apple and llvmpipe (check.py)
                ok, detail = judge_cs_dump(rc, msg, want, wave64=wave64, partial=partial, emulated=emulated,
                                           hexwords=name in ("cs_store", "cs_smem_soffset_imm", "cs_w32_lds_u8_d16",
                                                             "cs_w32_lds_subdword", "cs_w32_lds_wide",
                                                             "cs_lds_b64_cmpst", "cs_signed_zero_nan"))
                if ok and None in want:
                    unjudged = ", ".join(str(i) for i, w in enumerate(want) if w is None)
                    detail += f" (words {unjudged} not judged on this device: UNJUDGED_WORDS in check.py)"
            results.append((name, ok, detail))
        elif name == "cs_imgstore":
            rc, msg = run_cmd([vk_store, spv])
            results.append((name, *helper_result(rc, msg)))
        elif name in ("cs_imgload", "cs_imgload_d16", "cs_sample_lz_int_d16"):
            rc, msg = run_cmd([vk_store, spv, "load" if name == "cs_imgload" else "load-d16"])
            results.append((name, *helper_result(rc, msg)))
        elif name in MIP_TEST_PASSES:
            problem = texture_mip_test(a.build, spirv_dir, name)
            ok = None if isinstance(problem, Skip) else not problem
            results.append((name, ok, problem or "two passes, 4096 RGBA32F pixels each exact; wrong mip and UV rejected"))
        elif name == "ps_table_fetch":
            problem = table_fetch_test(a.build, spirv_dir)
            results.append((name, None if isinstance(problem, Skip) else not problem,
                            problem or "4096 exact pixels each for SINT, UINT and descriptor miss"))
        elif name in GRADIENT_TESTS:
            problem = texture_gradient_a16_test(a.build, spirv_dir, name)
            ok = None if isinstance(problem, Skip) else not problem
            results.append((name, ok, problem or "4096 RGBA pixels matched"))
        elif name in ("ps_texture_sample", "ps_cube_sample", "ps_texture_sample_d16", "ps_sample_c_lz",
                      "ps_sample_c_lz_a16", "ps_sample_c", "ps_sample_c_b"):
            # Bindless texture tests: vk_texture_test finds vs_ids.spv and the shader in its working directory.
            tool = os.path.join(a.build, "vk_texture_test" + ext)
            raw = os.path.join(raw_dir, f"{name}.raw")
            if name == "ps_texture_sample_d16":
                cmd = [tool, os.path.join(spirv_dir, "vs_ids.spv"), spv, raw, "0", "0", "0", "0", "0"]
            else:
                mode = {"ps_cube_sample": ["cube"], "ps_sample_c_lz": ["dref"],
                        "ps_sample_c_lz_a16": ["dref_a16"], "ps_sample_c": ["dref_c"],
                        "ps_sample_c_b": ["dref_c_b"]}.get(name, [])
                cmd = [tool, *mode, raw]
            rc, msg = run_readback(raw, run_cmd, cmd, cwd=spirv_dir)
            if rc == UNSUPPORTED:   # e.g. fewer push constant bytes than the texture tests need
                results.append((name, None, unsupported_skip(msg)))
            elif name == "ps_texture_sample_d16" and rc == 0:
                ok, detail = check_d16_texture(raw)
                results.append((name, ok, detail))
            else:
                results.append((name, rc == 0 and "PASSED" in msg, verdict_line(rc, msg)))
        elif name in TEXTURE_CONSTANTS:
            expected = TEXTURE_CONSTANTS[name]
            tool = os.path.join(a.build, "vk_texture_test" + ext)
            raw = os.path.join(raw_dir, f"{name}.raw")
            rc, msg = run_readback(raw, run_cmd, [tool, os.path.join(spirv_dir, "vs_ids.spv"), spv, raw, "0", "92",
                                                "c00000", "14500000", "0"], cwd=spirv_dir)
            bad = 4096
            if rc == 0 and os.path.isfile(raw):
                data = Path(raw).read_bytes()
                bad = 4096 if len(data) != 64 * 64 * 16 else sum(q != expected for q in struct.iter_unpack("<4f", data))
            if rc == UNSUPPORTED:
                results.append((name, None, unsupported_skip(msg)))
            else:
                detail = ("4096/4096 pixels exact" if bad == 0 else
                          f"{bad} pixels differ from {expected}: {verdict_line(0, msg)}")
                # After a non-zero exit no frame was compared: the exit status, not a pixel count.
                results.append((name, rc == 0 and bad == 0, verdict_line(rc, msg) if rc else detail))
        elif name in ("ps_tex", "ps_imgatomic"):
            # A texture through the descriptor hash table and the bindless set (ps_tex), or image atomics in a pixel
            # shader through the set's r32ui storage arrays (ps_imgatomic, drawn with vs_ids). vk_bindless_test
            # compares the result itself, and exits 77 where the device lacks a feature the set or shader needs.
            vs = os.path.join(spirv_dir, "vs_ids.spv") if name == "ps_imgatomic" else vs_tri
            rc, msg = run_cmd([vk_bindless, vs, spv])
            lines = msg.splitlines()
            if rc == 77:
                reason = next((ln for ln in lines if ln.startswith("unsupported:")), "unsupported device")
                results.append((name, None, f"SKIP ({reason})"))
            else:
                results.append((name, rc == 0, verdict_line(rc, msg)))
        elif name == "ps_ssharp":
            # The S#/T# mip fields through vulkan_sampler() and the view range (check.py's SAMPLER_CASES).
            problem = sampler_test(a.build, spirv_dir)
            results.append((name, None if isinstance(problem, Skip) else not problem,
                            problem or f"{len(SAMPLER_CASES)} S#/T# cases, 4096 pixels each exact"))
        elif name in ("ps_texoff", "ps_texoff_implicit", "ps_texoff_forms"):
            # Texel offsets: needs vk_texture_test's texture, sampler and descriptor table (vk_harness binds none),
            # and VK_KHR_maintenance8 for the non-constant Offset; the tool exits 77 without it.
            raw = os.path.join(raw_dir, f"{name}.raw")
            rc, msg = run_readback(raw, run_cmd, [os.path.join(a.build, "vk_texture_test" + ext), vs_tri, spv, raw,
                                                *TEXOFF_ARGS], env=texoff_env())
            if rc == UNSUPPORTED:
                results.append((name, None, unsupported_skip(msg)))
            elif rc != 0:
                results.append((name, False, msg))
            else:
                with open(raw, "rb") as f:
                    problem, covered = texoff_problem(f.read(), msg, name=name)
                results.append((name, not problem, problem or f"4096/4096 pixels exact ({covered})"))
        elif name in OTHER_HARNESS:
            results.append((name, None, f"SKIP (run by {OTHER_HARNESS[name]})"))
        elif name.startswith("ps_"):
            raw = os.path.join(raw_dir, f"{name}.raw")
            vs = os.path.join(spirv_dir, "vs_ids.spv") if name == "ps_flat" else vs_tri   # check.py draws ps_flat with vs_ids
            if name in PIXEL_VERTEX:
                # --bary-fallback alone has no PerVertexKHR inputs, so the f16 halves can't be interpolated; with
                # --f16-interp-fallback too (on by itself without fragmentShaderBarycentric) it reads decoded halves.
                if name == "ps_interp_f16_high" and bary_fallback() and not f16_interp_fallback():
                    results.append((name, None, "SKIP (per-vertex f16 interpolants need barycentrics)"))
                    continue
                vs = os.path.join(spirv_dir, PIXEL_VERTEX[name] + ".spv")
            env = dict(gfx_env or os.environ, **PIXEL_ENV[name]) if name in PIXEL_ENV else gfx_env
            bary_gs = os.path.join(spirv_dir, f"{name}.bary_gs.spv")
            if bary_fallback() and os.path.isfile(bary_gs):   # --bary-fallback PS: barycentrics from this GS
                env = dict(env or os.environ, VKH_GS=bary_gs, VKH_GS_HELPER="1")
            rc, msg = run_readback(raw, run_cmd, [vk_harness, vs, spv, raw], env=env)
            if rc == 77:   # vk_harness: the device lacks a feature this shader needs (e.g. barycentrics)
                results.append((name, None, unsupported_skip(msg)))
                continue
            if rc != 0:
                results.append((name, False, msg))
                continue
            const_wants = {
                "ps_clip_color": (0x3E800000, 0x3F000000, 0x3F400000, 0x3F800000),
                "ps_exp_color_mask": (0x40000000, 0x40800000, 0x40000000, 0x40800000),   # .asm header
                "ps_ops": (0x48804300, 5, 7, 0xBFFFFFFC),
                "ps_ops2": (0xABCD4400, 0x46001234, 0xBFFFFFFA, 7),
                "ps_ops3": (0xBFFFFFFA, 0x44660066, 0x44AABBCC, 16),
                "ps_ops4": (0x34000010, 0x00120800, 0xBEEF0000, 0x12343C00),
                "ps_scc": (1, 1, 3, 0x80000000),
                "ps_sdwa": (0x000000F1, 0xBFFEFFF2, 0x3C003344, 0xBFF70000),
                "ps_sdwa_dword_hi": (0x00004000, 0x00000003, 0x00004700, 0x0000C000),   # 16-bit SDWA, dst_sel:DWORD
                "ps_legacy_mul": (0, 0, 0, 0x40C00000),        # same expectation as check.py
                "ps_mul_legacy_denorm": (0, 0, 0, 0x80000000),  # mode0 input FTZ + DX9 +0 override; output FTZ keeps sign
                "ps_mul_legacy_denorm_mods": (0, 0, 0, 0x40400000),  # product FTZ before omod; omod -0 -> +0
                "ps_ops5": (0x12340010, 0xBEEF0800, 0xAAAAFFFF, 0xDEADBC00),   # from the .asm headers
                "ps_ops6": (0xF800FFFF, 0x01000010, 0x00800800, 0x8000FFF0),
                "ps_ops7": (0x00FF8077, 0xFF0000FF, 0x89ABCDEF, 0x00070003),
                "ps_ops8": (0x80000000, 0xBFFFFFFF, 0x000F100E, 0x0F0F0F1F),
                "ps_f32_mods": (0xC0256A01, 0x3F800000, 0x41E00000, 0x3FC00000),
                "ps_vcc_x2": (0x11223344, 0x55667788, 0, 0),
                "ps_vcmpx_forms": (0xF05B2A48, 0xF05E8A12, 0xB174BE18, 0x15550000),   # 14 v_cmpx forms (.asm header)
                "ps_vcmp_int32": (0x963E52E8, 0x984DCEE8, 0xB9458EE8, 0xB73612E8),   # u32/i32 compares (.asm header)
                "ps_med3_mods": (0x3F800000, 0x40400000, 0x40C00000, 0x3F400000),
                "ps_scc_minmax": (0x00030205, 0x00030207, 0x00030209, 0x00030203),
                "ps_scc_logic32": (0x0fff0fff, 0x000f000f, 0xf00ff00f, 0x36),
                "ps_scc_logic64": (0x0fff0fff, 0xff000000, 0xff0000ff, 0x16),
                "ps_fma_literal": (0x41B80000, 0x41800000, 0x40E00000, 0x41A40000),
                "ps_f64_mods": (0x40040000, 0x3FF00000, 0x3FE40000, 0x3FF00000),   # needs shaderFloat64
                "ps_f64_round": (0xC12AF665, 0x3FE7FFFF, 0xBFFFFFFF, 0xC12AF666),  # needs shaderFloat64
                "ps_outmods_f16": (0xAAAA3C00, 0xAAAA3E00, 0x7C000000, 0x3C000000),
                # f16 op_sel / partial writes / inline constants: RDNA2 hardware words (see the .asm headers)
                "ps_f16_med3_opsel": (0xAAAA4100, 0xAAAA4200, 0xAAAA3E00, 0x40005555),
                "ps_f16_med3_mods": (0x41005555, 0x3C005555, 0x41003800, 0xAAAABA00),
                "ps_f16_vop2": (0xAAAA4000, 0xAAAAC000, 0xAAAA3A00, 0x42003F00),
                "ps_f16_vop2b": (0xAAAA3E00, 0xAAAAB800, 0xAAAA3C00, 0x42003400),
                "ps_f16_vop1": (0xAAAA4000, 0xAAAABC00, 0xAAAAC000, 0xAAAA8000),
                "ps_f16_vop1b": (0xAAAA3400, 0xAAAA4000, 0xAAAAB400, 0xAAAA3800),
                "ps_f16_const": (0xAAAA3D00, 0x38005555, 0xAAAA3800, 0x3F000000),
                "ps_f16_const2": (0x42003D00, 0x3D004200, 0xAAAABC00, 3),
                "ps_f16_pack": (0x34003C00, 0x38003400, 0x34000000, 0x44004200),
                "ps_f16_pack_mods": (0x4400B400, 0xC4003400, 0x4400C200, 0xB400B400),
                "ps_sdwa_f16_const": (0xAAAA3800, 0xAAAA1C00, 0xAAAA3E00, 0xAAAA73F0),   # measured on gfx1036
                "ps_sff_imm": (2, 1, 20, 16),                  # SHADER-46: s_ff0/s_ff1 of an immediate
                "ps_fnan_med3": (0x40A00000, 0xC0200000, 0xC0200000, 0x40600000),   # measured on RDNA2 hardware
                "ps_fnan_fract": (0x3F7FFFFF, 0x00003BFF, 0xC0200000, 0xC0200000),
                # v_med3 with a NaN in S2 or in two operands, by the same measured rule (derivations in the .asm headers)
                "ps_fnan_med3_s2": (0x40A00000, 0x40A00000, 0xC0200000, 0x40E00000),
                "ps_f16_fnan_med3": (0xAAAA4500, 0xAAAA4500, 0xAAAAC100, 0xAAAA4700),
                # TEST-175: docs/design.md rules that were checked on the CPU only (derivations in the .asm headers)
                "ps_cmp_nlg": (0x40000037, 0x40000007, 0x40000001, 0x3F800000),
                "ps_far_branch": (0x00000007, 0x00000013, 0x00000051, 0x0000000B),   # far branches (.asm header)
                "ps_vcmp_f16": (0x48A12080, 0x4CE9B1A2, 0x4BA72C98, 0x4FEFBDBA),   # 16-bit float v_cmp e32 (.asm header)
                "ps_vcmp_f16_e64": (0x44C11100, 0x4CF171C1, 0x47C71D18, 0x4FF77DD9),   # same, e64 with swapped operands
                "ps_vcmpx_f16": (0x68A12080, 0x6CE9B1A2, 0x6BA72C98, 0x6FEFBDBA),   # 16-bit float v_cmpx e32 (.asm header)
                "ps_vcmpx_f16_e64": (0x64C11100, 0x6CF171C1, 0x67C71D18, 0x6FF77DD9),   # same, e64 with swapped operands
                "ps_cvt_off": (0xBF000000, 0x3EE00000, 0xBE800000, 0x3F800000),
                "ps_fmac_f16_mods": (0xAAAA3C00, 0xAAAA3A00, 0xAAAA0000, 0xAAAA4400),
                "ps_vop3_const_mods": (0xC0000000, 0xBF800000, 0x3F800000, 0xAAAABC00),
                "ps_cndmask_sign_mods": (0xFF800001, 0x7F800001, 0x80000001, 0x7F800001),
                "ps_interp_f32_mods": (0x3F800000, 0x00000000, 0x3F400000, 0x3F800000),   # drawn with vs_interp_consts
                "ps_interp_mov_mods": (0x3F800000, 0x3FC00000, 0x40000000, 0x3F400000),
                # v_mad_u32_u16 / v_mad_i32_i16 and v_subrev_f16, which had CPU checks only (.asm headers)
                "ps_valu16_mad": (0x00000002, 0x00030002, 0x00008000, 0x00000004),
                "ps_valu16_mad_sel": (0x00008003, 0xDB880000, 0xFFFFFFFF, 0x80000000),   # measured on gfx1036
                "ps_valu16_subrev": (0xAAAABC00, 0xAAAAC000, 0xAAAA3C00, 0x42003800),
                "ps_vcmp_f32_nan": (0x0000376C, 0x0000FD5B, 0x0000D36C, 0x00000FE7),   # from the .asm headers
                "ps_vcmp_64": (0x0000B3C4, 0x000091E6, 0x00006787, 0x00007C93),
                # 32-bit integer VALU ops on VINT_ENV's heap (derivations in the .asm headers)
                "ps_vint_logic3": (0xFFFFFFFE, 0xBBB05070, 0x1D3B5977, 0x123F5F7E),
                "ps_vint_bits": (0xFFFFFFFE, 0x40000000, 0x1E6A2C48, 0x000001D5),
                "ps_vint_mul24": (0x13345678, 0xFFFFFFFD, 0x00007FFF, 0x0000FFFF),
                "ps_vint_minmax": (0x7FFFFFFE, 0x80000001, 0x80000001, 0x7FFFFFFE),
                "ps_vint_minmax3": (0xF0F0F0F0, 0xAB800000, 0xF0F0F0F0, 0xAB800000),
                "ps_w64x_readlane": (0x40000079, 0x40000010, 0x40000064, 0x400000BE),   # same expectation as check.py
            }
            if skip := wide_pixel_skip(name, msg):   # a 64-wide pixel stage the device could not give: nothing to judge
                results.append((name, None, str(skip)))
                continue
            if name in const_wants:
                if name == "ps_vcmpx_forms" and (skip := vcmpx_forms_narrow_skip(msg)):
                    ok, detail = None, str(skip)
                else:
                    ok, detail = check_pixel_const(raw, const_wants[name])
            elif name == NAN_PAYLOAD_TEST:
                problem = nan_payload_problem(read_raw_u32(raw), msg)
                ok, detail = not problem, problem or "4096/4096 pixels match the driver-aware NaN payload check"
            elif name == "ps_bary":
                ok, detail = check_bary(raw)
            elif name == "ps_persp_bary":
                problem = bary_persp_problem(read_raw_rgba32f(raw))
                ok, detail = not problem, problem or "perspective-correct on every covered pixel"
            elif name == "ps_bitmask_perm":
                m = re.search(r"subgroup \d+\.\.(\d+)", msg)
                if m and int(m.group(1)) < 32:
                    ok, detail = None, f"SKIP (subgroup {m.group(1)} is narrower than 32 lanes)"
                else:
                    ok, detail = check_bitmask_perm(raw, msg)
            elif name in ("ps_uniform_branch", "ps_scc_mask32"):   # same expected pattern (see the .asm headers)
                ok, detail = check_uniform_branch(raw)
            elif name == "ps_wqm":
                ok, detail = check_wqm(raw)
            elif name == "ps_flat":
                ok, detail = check_flat(raw)
            elif name == "ps_interp_f16_high":
                # A missing or partial readback is an empty frame for this test, not a crash of the suite (as #3442).
                problem = interp_f16_high_problem(read_raw_u32(raw) if os.path.isfile(raw) else [])
                ok, detail = not problem, problem or "f16 low/high halves and f32 control within 2e-3"
            elif name == "ps_ij_computed":
                problem = ij_computed_problem(read_raw_u32(raw) if os.path.isfile(raw) else [])
                ok, detail = not problem, problem or "computed I/J pair words and v/2 within 1e-6"
            elif name == "ps_mandelbrot":
                px = read_raw_u32(raw) if os.path.isfile(raw) else []
                problem = frame_problem(px) or mandelbrot_problem(px)
                ok, detail = not problem, problem or f"{len(px)}/{len(px)} iteration counts match the float32 model"
            else:
                ok, detail = False, "no checker defined"
            results.append((name, ok, detail))
        elif name == "gs_tri":
            raw = os.path.join(raw_dir, f"{name}.raw")
            env = dict(gfx_env or os.environ, VKH_GS=spv)
            rc, msg = run_readback(raw, run_cmd, [vk_harness, vs_tri, fs_loc1, raw], env=env)
            if rc == UNSUPPORTED:
                results.append((name, None, unsupported_skip(msg)))
            elif rc != 0:
                results.append((name, False, msg))
            else:
                ok, detail = check_grad(raw, 0.0, 0.0)
                results.append((name, ok, detail))
        elif name == "ms_v5_workgroup":
            rc, msg = run_cmd(["spirv-val", "--target-env", "vulkan1.2", spv])
            if rc:
                results.append((name, False, msg))
                continue
            problem = mesh_v5_test(a.build, spirv_dir, Path(raw_dir) / f"{name}.raw")
            results.append((name, None if isinstance(problem, Skip) else not problem,
                            problem or "24/24 workgroup coordinates exact (2 x 3 x 4)"))
        elif name in ("ms_tri", "ms_w64x_ballot", "ms_wave_info_s3"):
            raw = os.path.join(raw_dir, f"{name}.raw")
            env = mesh_env(spirv_dir, name, gfx_env)
            rc, msg = run_readback(raw, run_cmd, [vk_harness, spv, fs_passthru, raw], env=env)
            # ms_w64x_ballot needs exact wave64 (a 64-wide subgroup, or --wave64-emulate): skip it elsewhere, as
            # the wave64 compute dumps are.
            no64 = (name == "ms_w64x_ballot" and not w64_emulated() and
                    not any(int(x) >= 64 for x in re.findall(r"subgroup \d+\.\.(\d+)", msg)))
            if rc == UNSUPPORTED:
                results.append((name, None, unsupported_skip(msg)))
            elif rc != 0:
                results.append((name, False, msg))
            elif no64:
                results.append((name, None, "SKIP (no 64-wide subgroup on this device)"))
            else:
                ok, detail = check_grad(raw, 0.5, 1.0)
                results.append((name, ok, detail))
        elif name == "ms_f16_pack":   # mesh producer of packed f16 params, drawn with ps_interp_f16_high
            raw = os.path.join(raw_dir, f"{name}.raw")
            env = mesh_env(spirv_dir, name, gfx_env)   # VKH_MESH=1, or the --mesh-compute pass and its VS
            fs = os.path.join(spirv_dir, "ps_interp_f16_high.spv")
            if bary_fallback() and not f16_interp_fallback():
                results.append((name, None, "SKIP (per-vertex f16 interpolants need barycentrics)"))
                continue
            rc, msg = run_readback(raw, run_cmd, [vk_harness, spv, fs, raw], env=env)
            if rc == UNSUPPORTED:
                results.append((name, None, unsupported_skip(msg)))
            elif rc != 0:
                results.append((name, False, msg))
            else:
                # A missing or partial readback is an empty frame for this test, not a crash of the suite (as #3442).
                problem = interp_f16_high_problem(read_raw_u32(raw) if os.path.isfile(raw) else [])
                results.append((name, not problem, problem or "f16 low/high halves and f32 control within 2e-3"))
        elif name == "ms_clip":
            raw = os.path.join(raw_dir, f"{name}.raw")
            env = mesh_env(spirv_dir, name, gfx_env)
            rc, msg = run_readback(raw, run_cmd, [vk_harness, spv, fs_passthru, raw], env=env)
            if rc == UNSUPPORTED:
                results.append((name, None, unsupported_skip(msg)))
            elif rc != 0:
                results.append((name, False, msg))
            else:
                ok, detail = check_clip_half(raw, 0.5, 1.0)
                results.append((name, ok, detail))
        elif name == "vs_base_ids":
            # Zero-based v0 / v3 (the hardware's VS input VGPRs): drawn with firstVertex 16 and firstInstance 7,
            # ps_flat must still see vertex 0 of instance 0 (0x101), as check.py checks.
            raw = os.path.join(raw_dir, f"{name}.raw")
            env = dict(gfx_env or os.environ, VKH_DRAW="16,7")
            rc, msg = run_readback(raw, run_cmd, [vk_harness, spv, os.path.join(spirv_dir, "ps_flat.spv"), raw], env=env)
            if rc == UNSUPPORTED:
                results.append((name, None, unsupported_skip(msg)))
            elif rc != 0:
                results.append((name, False, msg))
            else:
                ok, detail = check_flat(raw)
                results.append((name, ok, detail))

    if a.filter and not results:   # the filter matched only inputs/partners: say so rather than "0/0 passed"
        names = [os.path.splitext(f)[0] for f in asm_files]
        partners = {os.path.splitext(f)[0] for f in DRAWN_WITH.values()} | set(TEXTURE_MIP_DEPENDENCIES)
        shown = ", ".join(names[:5]) + (", ..." if len(names) > 5 else "")
        role = "partners built for other tests" if partners.issuperset(names) else "inputs without a GPU test row"
        hint = ("; cs_bench_* shaders are benchmark inputs, run them with the benchmark tool"
                if all(n.startswith("cs_bench_") for n in names) else "")
        print(f"error: --filter {a.filter!r} selected {shown} but no runnable GPU test cases ({role}){hint}",
              file=sys.stderr)
        return 2

    print("\n=== GPU TEST RESULTS ===")
    passed = skipped = 0
    for name, ok, detail in results:
        ok = None if isinstance(detail, Skip) else ok   # a check.py helper's "this device can't run it" (cs_fmt_*)
        tag = "SKIP" if ok is None else "PASS" if ok else "FAIL"
        passed += ok is True
        skipped += ok is None
        print(f"  {tag:4s}  {name:<20s}  {detail}")
    print(f"\n{passed}/{len(results)} passed" + (f", {skipped} skipped" if skipped else ""))
    if passed + skipped != len(results):
        return 1
    if getattr(a, "require_pass", False) and not passed:
        print("error: --require-pass: no GPU tests passed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
