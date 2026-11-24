"""Build the hand-written test shaders in this folder into SPIR-V.

    python3 -I runtime/test/shaders/build.py <out dir> [name ...]

Each .asm is RDNA2 assembly. It is assembled with llvm-mc (LLVM 21+ with the AMDGPU target; set LLVM_MC to its
path or to a whole command, as for the shaders stage). Listings, synthetic FSSM headers and DXBC containers are
staged in a temporary directory. After validation, the build writes <out>/spirv/<name>.spv and the translator
metadata.

Compute shaders can override the default header with leading "; group: X,Y,Z" and "; rsrc2: 0x..." comments.
A shader whose header has "; fssm-at: 0xOFFSET" and "; fssm-data: 0xWORD ..." comments also gets a synthetic
FSSM program (the magic, zeros up to OFFSET, then the words): it is published as <out>/fssm/<name>.fssm and its
SHA-256 goes into the index as fssm_sha256, which is what the layer looks a PSO's shader up by (the translator
hashes <work>/fssm/<name>.fssm for the game's programs, and compute shaders here already). A test builds the shader container from that file. The words
are also what a shader reads through s_getpc_b64 (the code base) at that offset.
The stage comes from the name's prefix. Mesh shaders (ms_) also need a DXBC container with their PSV0, which this
doesn't make; geometry shaders (gs_) get gs_tri's signatures (gs_container).
The comment at the top of each .asm says which test program to run on the result and what it should print.
With LEONIDA_FMA=khr (or glsl, or exact), the fused multiply-adds are translated with --fma khr (OpFmaKHR, which needs
VK_KHR_shader_fma; the test programs enable it when the device has it, else they exit 77), --fma glsl, or --fma exact
(a single rounding built from plain f32 ops, for a driver that splits GLSL Fma and has no VK_KHR_shader_fma).
With LEONIDA_BARY_FALLBACK=1, pixel shaders that read raw barycentrics are translated with --bary-fallback, and
each gets a pass-through geometry shader, <name>.bary_gs.spv, that check.py draws it with (tools/bary_fallback_gs.py).
With the variable unset, build.py/check.py follow the Mesa ANV probe automatically.
Under that or LEONIDA_F16_INTERP_FALLBACK=1, ps_ij_computed isn't built: its computed I/J pair needs PerVertexKHR
inputs, and check.py reports it as SKIP.
With LEONIDA_PC_TAIL_UBO=1, every shader is translated with --pc-tail-ubo (s0..s31 in 128 push-constant bytes, the
heap base and code base in a uniform block at set 1), for devices whose maxPushConstantsSize is 128; set it for
check.py too, whose test programs then bind that block (runtime/test/vk_pc_tail.h).
"""
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "..", "..", "..", "tools")
TRANSLATOR = os.path.join(TOOLS, "rdna2spirv.py")
sys.path.insert(0, TOOLS)
import mc_command  # noqa: E402

MC = mc_command.mc_command()
RSRC2, GROUP = 0x8088, (64, 1, 1)
# Not built under LEONIDA_BARY_FALLBACK=1 or LEONIDA_F16_INTERP_FALLBACK=1 (see main).
NEEDS_PER_VERTEX_INPUTS = ("ps_ij_computed",)
sys.path.insert(0, HERE)  # this script is also run with python -I
from compute_fixture import compute_header, shift_user_data, user_data_shift  # noqa: E402


def barycentric_fallback_override():
    v = (os.environ.get("LEONIDA_BARY_FALLBACK") or "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return None


def barycentric_fallback():
    override = barycentric_fallback_override()
    if override is not None:
        return override
    try:
        r = subprocess.run([sys.executable, "-I", os.path.join(TOOLS, "vk_subgroup_info.py")],
                           capture_output=True, text=True, timeout=60)
        answer = json.loads(r.stdout.strip().splitlines()[-1])
        return answer.get("barycentric_fallback") is True
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return False


def asm_mattr(path):
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if not s:
            continue
        if not s.startswith(";"):
            break
        m = re.match(r";\s*mattr:\s*(\S+)", s)
        if m:
            return m.group(1)
    return None


CLIP_CASES = {"vs_clip": 1, "vs_clip_param4": 4, "vs_clip_undeclared": 1}


def asm_fssm_fields(path):
    """(offset, words) of the "fssm-at" and "fssm-data" header comments, (None, []) without them."""
    at, words = None, []
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if s and not s.startswith(";"):
            break
        m = re.match(r";\s*fssm-at:\s*(0x[0-9a-fA-F]+|\d+)", s)
        if m:
            at = int(m.group(1), 0)
        m = re.match(r";\s*fssm-data:\s*(.*)", s)
        if m:
            words += [int(w, 0) for w in m.group(1).split()]
    return at, words


def asm_fssm(path):
    """The synthetic FSSM program a vs_/ps_ shader's header asks for, or None."""
    at, words = asm_fssm_fields(path)
    if at is None or not words or at < 4 or at % 4:
        return None
    return b"FSSM" + bytes(at - 4) + b"".join(struct.pack("<I", w) for w in words)


def cs_fssm_extra(path):
    """A compute shader's "fssm-at"/"fssm-data" words (at 0x148 or later, after compute_header's header), or None.
    Two compute shaders with the same group size and RSRC2 would otherwise share one FSSM hash, which is what the
    layer finds a PSO's shader by."""
    at, words = asm_fssm_fields(path)
    if at is None or not words or at < 0x148 or at % 4:
        return None
    return at, b"".join(struct.pack("<I", w) for w in words)


def shader_source(name):
    if name == "ps_mullit":
        return "../../../tests/semantic/fixtures/ps_mullit"
    return "vs_clip" if name in CLIP_CASES else name


def clip_container():
    # One OSG1 element: stream 0, SV_ClipDistance0.xy, float, logical register 2.
    name = b"SV_ClipDistance\0"
    payload = struct.pack("<8I2B6x", 1, 8, 0, 40, 0, 2, 3, 2, 3, 0) + name
    return (b"DXBC" + bytes(16) + struct.pack("<4I", 1, 44 + len(payload), 1, 36)
            + b"OSG1" + struct.pack("<I", len(payload)) + payload)


def gs_container():
    # ISG1 and OSG1 of the gs_* fixtures, as scripts/run_gpu_tests.py makes them for gs_tri: SV_Position (system value
    # 1) in register 0 and TEXCOORD0 in register 1, float4 (component type 3), no names.
    sig = struct.pack("<II", 2, 8) + b"".join(struct.pack("<6I2B6x", 0, 0, 0, sv, 3, reg, 0xF, 0xF)
                                              for sv, reg in ((1, 0), (0, 1)))
    chunk = struct.pack("<I", len(sig)) + sig
    return (b"DXBC" + bytes(16) + struct.pack("<5I", 1, 40 + 2 * (4 + len(chunk)), 2, 40, 44 + len(chunk))
            + b"ISG1" + chunk + b"OSG1" + chunk)


def translate_tests(work, names):
    disasm, fssm, spirv = (work / d for d in ("disasm", "fssm", "spirv"))
    disasm.mkdir()
    fssm.mkdir()
    shift = user_data_shift()   # LEONIDA_USER_DATA_SHIFT: compute fixtures for a D3D12 PSO (compute_fixture.py)
    for n in names:
        asm_path = os.path.join(HERE, shader_source(n) + ".asm")
        raw = Path(asm_path).read_text(encoding="utf-8")
        if shift and n.startswith("cs_"):
            raw = shift_user_data(raw, shift)
        mattr = asm_mattr(asm_path)
        cmd = MC + ["-triple=amdgcn--amdpal", "-mcpu=gfx1013"]
        if n in CLIP_CASES:
            cmd.append(f"-defsym=PARAM_SLOT={CLIP_CASES[n]}")
        if mattr:
            cmd.append(f"-mattr={mattr}")
        cmd.append("-show-encoding")
        # Feed the text on stdin: an LLVM_MC command that runs in WSL cannot open a Windows path argument.
        r = mc_command.run(cmd, input=raw, capture_output=True, text=True)
        if r.returncode or "error:" in r.stderr:
            sys.exit(f"{n}.asm did not assemble:\n{r.stderr}")
        # rdna2spirv.py rejects assembler directives such as the "\t.text" llvm-mc prints first.
        code = "".join(ln for ln in r.stdout.splitlines(keepends=True) if not ln.lstrip().startswith("."))
        (disasm / (n + ".s")).write_text(f"; {n}  hand-written test\n{code}", encoding="utf-8", newline="\n")
        program = None if n.startswith("cs_") else asm_fssm(asm_path)
        if program:   # where the translator looks for a program to hash, as in the game's work folder
            (fssm / (n + ".fssm")).write_bytes(program)
        if n.startswith("cs_"):
            program = bytes(compute_header(asm_path, shift))
            extra = cs_fssm_extra(asm_path)
            if extra:
                program += bytes(extra[0] - len(program)) + extra[1]
            (fssm / (n + ".fssm")).write_bytes(program)
        elif n in CLIP_CASES and n != "vs_clip_undeclared":
            (work / (n + ".dxbc")).write_bytes(clip_container())
        elif n.startswith("gs_"):
            (work / (n + ".dxbc")).write_bytes(gs_container())
        else:
            (work / (n + ".dxbc")).write_bytes(bytes(32))  # chunk count 0
    cmd = [sys.executable, "-I", TRANSLATOR, disasm, "--out", spirv]
    if os.environ.get("LEONIDA_W64_EMULATE") == "1":   # wave64 on 32-wide subgroups (compute shaders only)
        cmd.append("--wave64-emulate")
    if barycentric_fallback():   # raw barycentrics without VK_KHR_fragment_shader_barycentric
        cmd.append("--bary-fallback")
    if os.environ.get("LEONIDA_FMA") in ("glsl", "khr", "exact"):   # GLSL Fma (default), OpFmaKHR, or emulated
        cmd += ["--fma", os.environ["LEONIDA_FMA"]]
    if os.environ.get("LEONIDA_F16_INTERP_FALLBACK") == "1":   # f16 interpolation without PerVertexKHR (MoltenVK)
        cmd.append("--f16-interp-fallback")
    if os.environ.get("LEONIDA_TEXOFF_FALLBACK") == "1":   # texel offsets without VK_KHR_maintenance8
        cmd.append("--texel-offset-fallback")
    # 128-byte push constants: the tail in a set-1 uniform block. The values the test programs accept (vk_pc_tail.h).
    if (os.environ.get("LEONIDA_PC_TAIL_UBO") or "").strip().lower() in ("1", "true", "yes", "on"):
        cmd.append("--pc-tail-ubo")
    return subprocess.run(cmd).returncode


def main():
    if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
        print(__doc__)
        return 0
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    # The CMake test targets name the spirv-as/spirv-val folder here instead of passing a whole PATH, which the Visual
    # Studio generator's command line can't carry intact (a PATH entry ending in a backslash escapes the next quote).
    if os.environ.get("LEONIDA_SPIRV_TOOLS"):
        os.environ["PATH"] = os.environ["LEONIDA_SPIRV_TOOLS"] + os.pathsep + os.environ.get("PATH", "")
    out = Path(sys.argv[1])
    available = sorted({p.stem for p in Path(HERE).glob("*.asm")} | CLIP_CASES.keys())
    names = sys.argv[2:] or available
    # mullit_readback.py uses the CPU semantic fixture as an explicit build target.
    unknown = sorted(set(names) - set(available) - {"ps_mullit"})
    if unknown:
        sys.exit("Unknown test shaders: " + ", ".join(unknown))
    skipped = [n for n in names if n.startswith("ms_")]
    if skipped:
        print("not built (need a DXBC container with signatures):", ", ".join(skipped))
    names = [n for n in names if n not in skipped]
    # A computed I/J pair needs PerVertexKHR inputs, which both fallbacks take away: the translator refuses it there,
    # and check.py and run_gpu_tests.py report it as SKIP. Leave it out instead of losing the whole batch.
    if barycentric_fallback() or os.environ.get("LEONIDA_F16_INTERP_FALLBACK") == "1":
        per_vertex = [n for n in names if n in NEEDS_PER_VERTEX_INPUTS]
        if per_vertex:
            print("not built (computed I/J pairs need PerVertexKHR inputs, off under the fallback):",
                  ", ".join(per_vertex))
        names = [n for n in names if n not in per_vertex]
    if not names:
        return 0
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="test-shaders-", dir=out) as tmp:
        work = Path(tmp)
        status = translate_tests(work, names)
        if status:
            return status
        built = work / "spirv"
        index = built / "index.json"
        validated = json.loads(index.read_text(encoding="utf-8")) if index.exists() else {}
        missing = [n for n in names if n not in validated or not (built / (n + ".spv")).is_file()]
        if missing:
            sys.exit("Translation did not produce validated SPIR-V for: " + ", ".join(missing))
        spirv = out / "spirv"
        spirv.mkdir(exist_ok=True)
        published_index = spirv / index.name
        combined = json.loads(published_index.read_text(encoding="utf-8")) if published_index.exists() else {}
        combined.update(validated)
        index.write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8", newline="\n")
        for n in validated:
            for suffix in (".spv", ".spvasm"):
                artifact = built / (n + suffix)
                artifact.replace(spirv / artifact.name)
            sidecar = work / "fssm" / (n + ".fssm")
            published = not n.startswith("cs_") or cs_fssm_extra(os.path.join(HERE, n + ".asm"))
            if published and sidecar.is_file():
                (out / "fssm").mkdir(exist_ok=True)
                sidecar.replace(out / "fssm" / sidecar.name)
        index.replace(published_index)
        sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "tools"))
        import spirv_stubs
        spirv_stubs.generate_for_index(str(spirv), combined)
        if barycentric_fallback():
            # The pass-through geometry shader each --bary-fallback PS is drawn with (<name>.bary_gs.spv).
            import bary_fallback_gs
            bary_fallback_gs.build_for_index(str(spirv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
