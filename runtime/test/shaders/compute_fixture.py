"""Synthetic FSSM compute headers shared by both GPU test runners."""
import os
import re
import struct
from pathlib import Path


def compute_header(asm_path, shift=0):
    # Default: four user SGPRs (V#), TGID_X, 512 bytes LDS, 64 threads. `shift`: see shift_user_data.
    rsrc2, group = 0x8088, (64, 1, 1)
    for line in Path(asm_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if not line.startswith(";"):
            break
        size = re.match(r";\s*group:\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", line)
        if size:
            group = tuple(map(int, size.groups()))
        elif line[1:].lstrip().startswith("rsrc2:"):
            rsrc2 = int(line.split(":", 1)[1].strip(), 0)
    header = bytearray(0x148)
    struct.pack_into("<4I", header, 0x138, shifted_rsrc2(rsrc2, shift), *group)
    return header


SGPR_LIMIT = 106  # s0..s105 on GFX10
_SGPR = re.compile(r"\bs(\d+)\b")
_SGPR_RANGE = re.compile(r"\bs\[(\d+):(\d+)\]")


def shift_user_data(text, shift):
    """The asm text with every scalar register moved up by `shift` (s[0:3] -> s[4:7] for 4), comments untouched.

    The fixtures here read their user data from s0 (the standalone harnesses push it there). In a D3D12 compute PSO
    the console layout starts the root arguments at s4 (rt::rootsig::pack_stage, D3D-448), so a D3D12 test that
    reuses such a fixture builds it with LEONIDA_USER_DATA_SHIFT=4: the same program, its user data 4 SGPRs up, and
    shifted_rsrc2 moves the user count and the thread-group ids that follow it. Named registers (vcc, exec, m0,
    ttmp) don't move. Raises ValueError when a register would pass s105."""
    if shift == 0:
        return text

    def one(m):
        n = int(m.group(1)) + shift
        if n >= SGPR_LIMIT:
            raise ValueError(f"s{m.group(1)} + {shift} is past s{SGPR_LIMIT - 1}")
        return f"s{n}"

    def pair(m):
        lo, hi = int(m.group(1)) + shift, int(m.group(2)) + shift
        if hi >= SGPR_LIMIT:
            raise ValueError(f"s[{m.group(1)}:{m.group(2)}] + {shift} is past s{SGPR_LIMIT - 1}")
        return f"s[{lo}:{hi}]"

    out = []
    for line in text.splitlines(keepends=True):
        code, sep, comment = line.partition(";")
        code = _SGPR.sub(one, _SGPR_RANGE.sub(pair, code))
        out.append(code + sep + comment)
    return "".join(out)


def shifted_rsrc2(rsrc2, shift):
    """COMPUTE_PGM_RSRC2 with USER_SGPR [5:1] raised by `shift` (the enabled thread-group ids follow it)."""
    user = ((rsrc2 >> 1) & 31) + shift
    if not 0 <= user <= 31:
        raise ValueError(f"USER_SGPR {user} does not fit RSRC2's 5-bit field")
    return (rsrc2 & ~(31 << 1)) | (user << 1)


def user_data_shift():
    """LEONIDA_USER_DATA_SHIFT as an int (0 when unset); see shift_user_data."""
    text = (os.environ.get("LEONIDA_USER_DATA_SHIFT") or "0").strip()
    try:
        shift = int(text)
    except ValueError:
        raise ValueError(f"LEONIDA_USER_DATA_SHIFT must be a whole number, not {text!r}") from None
    if not 0 <= shift <= 27:
        raise ValueError(f"LEONIDA_USER_DATA_SHIFT {shift} is outside 0..27")
    return shift


def unpinned_subgroup(out):
    """The lane count vk_compute_test was asked for with VKC_SUBGROUP_SIZE and could not pin on this device, or None.

    vk_compute_test then prints "compute subgroup size: N not available" and runs at whatever size the driver picks
    (#774); Mesa llvmpipe's only size is 8. A wave32 program's words from such a run say nothing about the translator,
    so the fuzzers skip it, as check.py's unpinned_subgroup_skip does for its lane-mask dumps. Output without the line
    (the size was pinned, none was asked for, or an older vk_compute_test) gives None."""
    m = re.search(r"^compute subgroup size: (\d+) not available$", out, re.MULTILINE)
    return int(m.group(1)) if m else None


def find_test_program(build, name):
    """The folder holding test program `name`: `build` itself (Ninja, Makefiles) or the RelWithDebInfo/Release
    subfolder a Visual Studio build puts it in, as run_gpu_tests.py's find_test_programs looks. None if not built.
    A Debug configuration isn't searched (neither does run_gpu_tests.py): GPU fuzzing wants an optimized harness."""
    exe = name + (".exe" if os.name == "nt" else "")
    for folder in (build, os.path.join(build, "RelWithDebInfo"), os.path.join(build, "Release")):
        if os.path.isfile(os.path.join(folder, exe)):
            return folder
    return None
