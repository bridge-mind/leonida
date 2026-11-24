r"""Run assembled RDNA2 programs through the translator on a real Vulkan GPU (standard library only).

The differential fuzzers that judge the translator in tests/semantic/spirv_mini_interp.py (Python floats and integers)
cannot see what a driver does with the SPIR-V: its denormal, NaN and rounding rules, and what it makes of an input
the SPIR-V leaves undefined. This module runs the same programs on the GPU: llvm-mc assembles each one for gfx1013
as a compute program of one wave (32 or 64 lanes, every lane the same), rdna2spirv.py translates them in one go,
runtime/build/vk_compute_test executes each (LEONIDA_VK_DEVICE picks the device) and lane 0's stored words come back.

    results = run_programs(texts, words=4, wave=32, build="runtime/build")

`rewrite=False` leaves the lines as they are (they already spell wave64); `uniform=False` skips the check that every lane gave lane 0's words (programs that seed VCC with data); `per_lane=True` returns every lane's words
(a list of `wave` lists) for programs whose lanes differ, instead of lane 0's. `texts` are the instruction lines of each program (strings, one instruction per line). Each program gets the prologue
(EXEC all ones, the lane id saved in v20) and an epilogue that stores v8..v(8+words-1) of every lane at lane *
words * 4 through the V# in s[0:3]. The result list has, per program, a list of `words` ints (lane 0) or a str: why
there is no result (llvm-mc rejected it, the translator refused it, the device cannot run it, the lanes disagree).
wave64_text() rewrites the wave32 spellings of lane masks (vcc_lo, s20, s21) as the 64-bit ones.

opaque=True (the default of run_programs) keeps the driver from folding the program: a `v_mov_b32 vN, K` seed becomes
`v_xor_b32 vN, K ^ FILL, vF`, where vF is read from the output buffer, which vk_compute_test fills with 0xEEEEEEEE.
The program computes the same values at run time, but a compiler that sees literal operands folds them itself, and
the AMD Windows driver (26.10.1) folds an ordered compare of a literal NaN to true in a wave32 pipeline, which says
nothing about what the translated shader does with a NaN it is given.
"""
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SHADERS = os.path.join(os.path.dirname(HERE), "runtime", "test", "shaders")
LANE_ID_REG = 20        # v20 holds the lane id for the epilogue; programs must leave it alone
ADDR_REG = 21           # v21 is the store address


def wave64_text(line, masks=("s20", "s21")):
    """The wave32 line with its lane masks as 64-bit pairs: vcc_lo -> vcc, s20 -> s[20:21], s21 -> s[22:23]."""
    out = re.sub(r"\bs_mov_b32 vcc_lo, ", "s_mov_b64 vcc, ", line)
    out = re.sub(r"\bvcc_lo\b", "vcc", out)
    for m in masks:
        n = int(m[1:])
        base = 20 if n == 20 else 22
        out = re.sub(rf"\b{m}\b", f"s[{base}:{base + 1}]", out)
    return out


FILL = 0xEEEEEEEE       # what vk_compute_test puts in the output buffer before the dispatch
FILL_REG = 22
SEED_RE = re.compile(r"^v_mov_b32(?:_e32)? v(\d+), (0x[0-9a-fA-F]+|\d+)$")


def opaque_seed(line):
    """`v_mov_b32 vN, K` as the same value computed from the buffer fill; other lines unchanged."""
    m = SEED_RE.match(line)
    if not m or int(m.group(1)) in (LANE_ID_REG, ADDR_REG, FILL_REG):
        return line
    return f"v_xor_b32_e32 v{m.group(1)}, 0x{(int(m.group(2), 0) & 0xFFFFFFFF) ^ FILL:x}, v{FILL_REG}"


def program_text(name, lines, words, wave, opaque=False, rewrite=True):
    """The compute program: prologue, the lines, the stores."""
    head = [f"; {name}", f"; group: {wave},1,1", "; rsrc2: 0x8088"]
    if wave == 64:
        head.append("; mattr: +wavefrontsize64,-wavefrontsize32")
        if rewrite:
            lines = [wave64_text(ln) for ln in lines]
    out = head + ["s_mov_b64 exec, -1" if wave == 64 else "s_mov_b32 exec_lo, -1", f"v_mov_b32 v{LANE_ID_REG}, v0",
                  f"v_lshlrev_b32 v{ADDR_REG}, {(4 * words).bit_length() - 1}, v{LANE_ID_REG}"]
    if opaque:
        out += [f"buffer_load_dword v{FILL_REG}, v{ADDR_REG}, s[0:3], 0 offen", "s_waitcnt vmcnt(0)"]
        lines = [opaque_seed(ln) for ln in lines]
    out += list(lines)
    for k in range(words):
        out.append(f"buffer_store_dword v{8 + k}, v{ADDR_REG}, s[0:3], 0 offen" + (f" offset:{4 * k}" if k else ""))
    out.append("s_endpgm")
    return "\n".join(out) + "\n"


def run_programs(texts, words=4, wave=32, build=None, work_parent=None, retries=2, opaque=True, rewrite=True,
                 per_lane=False, uniform=True):
    """Per program lane 0's `words` words, or a str (see the module docstring). words must be 1, 2, 4 or 8."""
    if words not in (1, 2, 4, 8):
        raise ValueError("words must be 1, 2, 4 or 8")
    if wave not in (32, 64):
        raise ValueError("wave must be 32 or 64")
    if SHADERS not in sys.path:
        sys.path.insert(0, SHADERS)
    import mc_command
    from compute_fixture import compute_header, find_test_program, unpinned_subgroup
    build = build or os.path.join(os.path.dirname(HERE), "runtime", "build")
    folder = find_test_program(build, "vk_compute_test")   # the folder holding it (a Visual Studio config subfolder)
    if not folder:
        raise FileNotFoundError(f"vk_compute_test not found in {build}; build it first "
                                "(cmake --build <build dir> --target vk_compute_test)")
    exe = os.path.join(folder, "vk_compute_test" + (".exe" if os.name == "nt" else ""))
    work = tempfile.mkdtemp(prefix="vk_fuzz_", dir=work_parent)
    try:
        disasm, fssm, spirv = (os.path.join(work, d) for d in ("disasm", "fssm", "spirv"))
        for d in (disasm, fssm):
            os.mkdir(d)
        mc = mc_command.mc_command() + ["-triple=amdgcn--amdpal", "-mcpu=gfx1013", "-show-encoding"]
        if wave == 64:
            mc.append("-mattr=+wavefrontsize64,-wavefrontsize32")
        names, results = [], []
        for i, lines in enumerate(texts):
            name = f"cs_vkfuzz_{i}"
            text = program_text(name, lines, words, wave, opaque, rewrite)
            path = os.path.join(work, name + ".asm")
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)
            r = mc_command.run(mc, input=text, capture_output=True, text=True)
            if r.returncode or "error:" in r.stderr:
                names.append(None)
                results.append("llvm-mc: " + (r.stderr.strip().splitlines() or ["failed"])[0][:100])
                continue
            code = "".join(ln for ln in r.stdout.splitlines(keepends=True) if not ln.lstrip().startswith("."))
            with open(os.path.join(disasm, name + ".s"), "w", encoding="utf-8", newline="\n") as f:
                f.write(f"; {name}\n{code}")
            with open(os.path.join(fssm, name + ".fssm"), "wb") as f:
                f.write(compute_header(path))
            names.append(name)
            results.append(None)
        subprocess.run([sys.executable, "-I", os.path.join(HERE, "rdna2spirv.py"), disasm, "--out", spirv],
                       capture_output=True, text=True)
        index = {}
        index_path = os.path.join(spirv, "index.json")
        if os.path.isfile(index_path):
            with open(index_path, encoding="utf-8-sig") as f:
                index = json.load(f)
        for i, name in enumerate(names):
            if name is None:
                continue
            spv = os.path.join(spirv, name + ".spv")
            if name not in index or not os.path.isfile(spv):
                results[i] = "translator: not translated"
                continue
            dump = os.path.join(work, name + ".bin")
            env = dict(os.environ, VKC_OUT_WORDS=str(words * wave), VKC_DUMP=dump)
            args = [exe, spv, "1", "dump"]
            if wave == 64:
                args.append("--require-wave64")
            else:
                env["VKC_SUBGROUP_SIZE"] = "32"
            for _ in range(retries + 1):
                if os.path.isfile(dump):
                    os.remove(dump)
                r = subprocess.run(args, capture_output=True, text=True, env=env)
                if r.returncode in (0, 77):
                    break
            msg = (r.stdout + r.stderr).strip()
            if r.returncode == 77:
                results[i] = "vk_compute_test: " + (msg.splitlines()[-1] if msg else "unsupported")
            elif wave == 32 and unpinned_subgroup(msg):
                results[i] = f"the device can't run 32-lane subgroups ({unpinned_subgroup(msg)} lanes)"
            elif r.returncode or not os.path.isfile(dump) or os.path.getsize(dump) != 4 * words * wave:
                results[i] = f"vk_compute_test exit {r.returncode}"
            else:
                with open(dump, "rb") as f:
                    flat = struct.unpack(f"<{words * wave}I", f.read())
                lanes = [list(flat[k:k + words]) for k in range(0, words * wave, words)]
                odd = [k for k, w in enumerate(lanes) if w != lanes[0]]
                if per_lane:
                    results[i] = lanes
                elif not uniform:
                    results[i] = lanes[0]
                else:
                    results[i] = lanes[0] if not odd else f"lanes {odd[:3]} differ from lane 0 on identical inputs"
        return results
    finally:
        if not os.environ.get("VK_FUZZ_KEEP"):
            shutil.rmtree(work, ignore_errors=True)


def judge(progs, hw, verdict, wave=32, words=4, build=None, work_parent=None, opaque=True, uniform=True):
    """Run progs (lists of lines) on the Vulkan GPU and judge each against hw[i] (words, or a str: why there are none)
    with verdict(lines, hw_words, got_words) -> (class, detail). Returns ({class: n}, [(i, detail, lines)] for the
    class "mismatch")."""
    got = run_programs(progs, words, wave, build, work_parent, opaque=opaque, uniform=uniform)
    counts, bad = {}, []
    for i, lines in enumerate(progs):
        if isinstance(hw[i], str) or isinstance(got[i], str):
            why = "skip: " + (("hardware: " + hw[i]) if isinstance(hw[i], str) else got[i])
            counts[why] = counts.get(why, 0) + 1
            continue
        klass, detail = verdict(lines, hw[i], got[i])
        counts[klass] = counts.get(klass, 0) + 1
        if klass == "mismatch":
            bad.append((i, detail, lines))
    return counts, bad


def all_skipped(counts):
    """True when a Vulkan run compared no program (every entry is a skip, or none ran)."""
    return not any(n > 0 and not k.startswith("skip") for k, n in counts.items())


def exact_verdict(lines, hw, got, n_obs=4):
    """match, or mismatch naming the first word that differs (the observation lines end the program)."""
    if len(hw) != len(got):
        return "mismatch", f"word count differs: hardware {len(hw)}, translated on the GPU {len(got)}"
    for k in range(len(hw)):
        if hw[k] != got[k]:
            obs = lines[-n_obs + k] if k < n_obs and len(lines) >= n_obs else ""
            parts = obs.split(", ", 1)
            reg = parts[1] if len(parts) == 2 else f"word {k}"
            return "mismatch", f"word {k} ({reg}): hardware 0x{hw[k]:08x}, translated on the GPU 0x{got[k]:08x}"
    return "match", ""
