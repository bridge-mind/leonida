"""Run XboxBvhBuilder.dll's translated compute programs on the GPU with synthetic inputs (TEST-594).

docs/embedded_shaders.md: the DLL carries 52 console compute programs, and the translator takes all but one; none had
been run on a GPU. This translates them from your own package and runs each one through vk_compute_test:

    python -I tools/bvh_gpu_smoke.py --game-dir <package> [--work DIR] [--build runtime/build]
                                     [--max-shared-memory 32768] [--table-qword 0x2000] [--include-rehydration]
                                     [--json report.json]

Each program reads its root arguments from a table at s[0:1], which in vk_compute_test is the heap base (Xbox VA 0).
The heap starts as that table, 64-bit words that all hold --table-qword, then zeros up to --heap-bytes (seeded with
VKC_HEAP_INIT). So every root descriptor points at a zeroed region past the table and every root constant reads
--table-qword or 0. That is enough to run each program, not to give it meaningful work: most leave early, and the
report says how many words each one changed. A run passes when it exits 0, the Khronos validation layer (when
installed) reports no error, and a second run writes the same heap bytes.

--checked also runs the programs whose effect reads straight off their code with inputs chosen for them, and compares
every byte they write (CHECKED below). Five take a count and write that many 16- or 8-byte entries from thread index 0
up; 32 threads per group, so one group with a count of 20 also shows that entries past the count are left alone.
PrefixSum64 scans 64 words in place into their exclusive prefix sum and stores the total.

The three rehydration programs (Box16Rehydration, Box32Rehydration, LeafRehydration) belong to TEST-376, which checks
them against tools/bvh_rehydrate.py; --include-rehydration runs them too.

--max-shared-memory is passed to rdna2spirv.py: set it to the device's maxComputeSharedMemorySize (32768 on AMD's
Windows driver; the translator's default is 65536). Over it, a program's LDS array shrinks to what the program uses,
which the wave64 bitonic sorts need on a 32 KiB device.

Nothing from the package is written outside --work. The report holds module names, statuses and counts only.
"""
import argparse
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REHYDRATION = ("Box16Rehydration", "Box32Rehydration", "LeafRehydration")
TABLE_QWORDS = 512   # a 4 KiB root table


def heap_image(table_qword, heap_bytes):
    """The initial heap: TABLE_QWORDS words of table_qword, then zeros up to heap_bytes."""
    table = struct.pack(f"<{TABLE_QWORDS}Q", *([table_qword] * TABLE_QWORDS))
    if heap_bytes < len(table):
        raise ValueError(f"--heap-bytes must be at least {len(table)}")
    return table + bytes(heap_bytes - len(table))


def changed_words(initial, after):
    """How many 32-bit words of the heap differ from the initial image."""
    n = min(len(initial), len(after)) // 4
    a = struct.unpack_from(f"<{n}I", initial)
    b = struct.unpack_from(f"<{n}I", after)
    return sum(1 for x, y in zip(a, b) if x != y)


def run_program(vkc, spv, init, dump, heap_words, validation, timeout):
    """One dispatch of one workgroup. Returns (status, last line of output): ok, 'exit N', validation, timeout."""
    env = dict(os.environ, VKC_OUT_WORDS=str(heap_words), VKC_HEAP_INIT=str(init), VKC_DUMP=str(dump))
    if validation:
        env["VK_INSTANCE_LAYERS"] = "VK_LAYER_KHRONOS_validation"
    try:
        p = subprocess.run([str(vkc), str(spv), "1", "dump"], capture_output=True, text=True, env=env,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        return "timeout", ""
    out = (p.stdout or "") + (p.stderr or "")
    last = (out.strip().splitlines() or [""])[-1][:160]
    if p.returncode != 0:
        return f"exit {p.returncode}", last
    if "Validation Error" in out:
        return "validation", next((line for line in out.splitlines() if "VUID" in line), last)[:160]
    return "ok", last


def smoke(spirv_dir, vkc, *, table_qword, heap_bytes, validation, timeout, include_rehydration, runner=run_program):
    """Run every program in spirv_dir twice. Returns [{name, status, words_changed, detail}] in name order."""
    image = heap_image(table_qword, heap_bytes)
    results = []
    with tempfile.TemporaryDirectory(prefix="bvh_gpu_smoke_") as tmp:
        tmp = Path(tmp)
        init = tmp / "heap.bin"
        init.write_bytes(image)
        for spv in sorted(Path(spirv_dir).glob("*.spv")):
            name = spv.stem
            if name in REHYDRATION and not include_rehydration:
                continue
            first, second = tmp / (name + ".1.bin"), tmp / (name + ".2.bin")
            status, detail = runner(vkc, spv, init, first, heap_bytes // 4, validation, timeout)
            changed = None
            if status == "ok":
                again, detail2 = runner(vkc, spv, init, second, heap_bytes // 4, validation, timeout)
                a = first.read_bytes() if first.is_file() else b""
                b = second.read_bytes() if second.is_file() else b""
                if again != "ok":
                    status, detail = again, detail2
                elif not a or a != b:
                    status, detail = "nondeterministic", "the two runs wrote different heap bytes"
                else:
                    status = "pass"
                changed = changed_words(image, a) if a else None
            results.append({"name": name, "status": status, "words_changed": changed, "detail": detail})
    return results


# ---- checked runs ----
#
# Root-table fields as each program reads them (offsets in the table at s[0:1]; addresses are 64-bit, counts and
# values 32-bit), and what it writes. FILL marks the bytes a program must leave alone.
DST, SRC, SUMS = 0x2000, 0x6000, 0x4000   # regions past the 4 KiB table
FILL = 0xAA
COUNT = 20                          # within one 32-thread group


def _u32(heap, at):
    return struct.unpack_from("<I", heap, at)[0]


def _table(fields):
    table = bytearray(TABLE_QWORDS * 8)
    for offset, size, value in fields:
        struct.pack_into("<Q" if size == 8 else "<I", table, offset, value)
    return bytes(table)


def _expect_entries(heap, base, stride, count, entry):
    """Problems with `count` entries of `stride` bytes at `base` (entry(i) gives the expected bytes) and the FILL after
    them, up to one more group's worth of entries."""
    problems = []
    for i in range(count):
        got = bytes(heap[base + i * stride: base + (i + 1) * stride])
        if got != entry(i):
            problems.append(f"entry {i}: {got.hex()} != {entry(i).hex()}")
    tail = bytes(heap[base + count * stride: base + 32 * stride])
    if tail != bytes([FILL]) * len(tail):
        problems.append(f"bytes past entry {count - 1} changed")
    return problems


# PrefixSum64's input: 64 words large enough that the sums wrap past 2^32.
PREFIX_INPUT = [(0x01000193 * (i + 1) ^ (i << 20)) & 0xFFFFFFFF for i in range(64)]


def _prefix_setup(heap):
    struct.pack_into("<64I", heap, DST, *PREFIX_INPUT)


def _prefix_expect(heap):
    """The exclusive prefix sum of the 64 words, in place, and their total (mod 2^32) as group 0's block sum."""
    problems, run = [], 0
    for i, x in enumerate(PREFIX_INPUT):
        got = _u32(heap, DST + 4 * i)
        if got != run:
            problems.append(f"element {i}: {got:#x} != {run:#x}")
        run = (run + x) & 0xFFFFFFFF
    if _u32(heap, SUMS) != run:
        problems.append(f"block sum: {_u32(heap, SUMS):#x} != {run:#x}")
    if bytes(heap[DST + 256:DST + 512]) != bytes([FILL]) * 256:
        problems.append("bytes past the 64 elements changed")
    return problems


def _source_pattern(i):
    return struct.pack("<4I", 0x10000 + 4 * i, 0x20000 + 4 * i, 0x30000 + 4 * i, 0x40000 + 4 * i)


CHECKED = {
    # zero COUNT 16-byte entries at the address in table+0x0 (count at +0x30)
    "RadixReset": ([(0x0, 8, DST), (0x30, 4, COUNT)],
                   lambda h: _expect_entries(h, DST, 16, COUNT, lambda i: bytes(16))),
    # COUNT 8-byte entries (-FLT_MAX, +FLT_MAX), the empty k-DOP bounds, at table+0x0 (count at +0x1c)
    "ResetKDOP": ([(0x0, 8, DST), (0x1c, 4, COUNT)],
                  lambda h: _expect_entries(h, DST, 8, COUNT, lambda i: struct.pack("<2I", 0xFF7FFFFF, 0x7F7FFFFF))),
    # entry i is base + 4i .. base + 4i + 3, at table+0x8 (count at +0x14, base at +0x18)
    "ResetToIndexBuffer": ([(0x8, 8, DST), (0x14, 4, COUNT), (0x18, 4, 1000)],
                           lambda h: _expect_entries(h, DST, 16, COUNT,
                                                     lambda i: struct.pack("<4I", *(1000 + 4 * i + k for k in range(4))))),
    # COUNT 16-byte entries copied from table+0x0 to table+0x8 (count at +0x14)
    "CopyCounters": ([(0x0, 8, SRC), (0x8, 8, DST), (0x14, 4, COUNT)],
                     lambda h: _expect_entries(h, DST, 16, COUNT, _source_pattern)),
    # the value at +0x18, four times per 16-byte entry, at table+0x8 (count at +0x14)
    "ResetCounterValues": ([(0x8, 8, DST), (0x14, 4, COUNT), (0x18, 4, 0xC0FFEE11)],
                           lambda h: _expect_entries(h, DST, 16, COUNT, lambda i: struct.pack("<4I", *[0xC0FFEE11] * 4))),
    # one group scans 64 words (32 lanes, two words each) in place at table+0x0 into their exclusive prefix sum; with
    # bit 0 of +0x10 set, lane 31 stores the group's total at table+0x8 + 4 * group
    "PrefixSum64": ([(0x0, 8, DST), (0x8, 8, SUMS), (0x10, 4, 1)], _prefix_expect, _prefix_setup),
}


def checked_heap(fields, heap_bytes):
    """The table with `fields`, FILL everywhere past it, and the source pattern at SRC."""
    heap = bytearray(_table(fields)) + bytearray([FILL]) * (heap_bytes - TABLE_QWORDS * 8)
    for i in range(32):
        heap[SRC + 16 * i: SRC + 16 * (i + 1)] = _source_pattern(i)
    return bytes(heap)


def run_checked(spirv_dir, vkc, *, heap_bytes, validation, timeout, runner=run_program):
    """Each CHECKED program with its inputs; [{name, status, detail}] with status pass, mismatch or a run failure."""
    results = []
    with tempfile.TemporaryDirectory(prefix="bvh_gpu_checked_") as tmp:
        tmp = Path(tmp)
        for name, (fields, expect, *setup) in CHECKED.items():
            spv = Path(spirv_dir) / (name + ".spv")
            if not spv.is_file():
                results.append({"name": name, "status": "missing", "detail": f"{spv.name} not translated"})
                continue
            init, dump = tmp / (name + ".in"), tmp / (name + ".out")
            heap = bytearray(checked_heap(fields, heap_bytes))
            for prepare in setup:
                prepare(heap)
            init.write_bytes(bytes(heap))
            status, detail = runner(vkc, spv, init, dump, heap_bytes // 4, validation, timeout)
            if status == "ok":
                problems = expect(dump.read_bytes()) if dump.is_file() else ["no heap dump"]
                status, detail = ("pass", "") if not problems else ("mismatch", "; ".join(problems[:3]))
            results.append({"name": name, "status": status, "detail": detail})
    return results


def translate(game_dir, work, max_shared_memory):
    """The pipeline docs/embedded_shaders.md documents, into work/dxbc and work/spirv. Returns the spirv folder."""
    dll = Path(game_dir) / "XboxBvhBuilder.dll"
    if not dll.is_file():
        raise SystemExit(f"error: {dll} not found")
    dxbc, spirv = Path(work) / "dxbc", Path(work) / "spirv"
    steps = [
        [HERE / "pe_shader_containers.py", str(dll), "--extract", str(dxbc)],
        [HERE / "fssm_unpack.py", str(dxbc)],
        # rdna2spirv.py reads each program's stage from dxbc/index.json, two folders above the disassembly
        [HERE / "shader_disasm.py", str(dxbc / "fssm"), "--out", str(dxbc / "disasm"), "--mcpu", "gfx1013"],
        [HERE / "rdna2spirv.py", str(dxbc / "disasm"), "--out", str(spirv),
         "--max-shared-memory", str(max_shared_memory)],
    ]
    for step in steps:
        p = subprocess.run([sys.executable, "-I", *map(str, step)], capture_output=True, text=True)
        if p.returncode != 0:
            raise SystemExit(f"error: {Path(step[0]).name} failed:\n{p.stdout}{p.stderr}")
    return spirv


def find_vk_compute_test(build):
    exe = "vk_compute_test.exe" if os.name == "nt" else "vk_compute_test"
    for c in (Path(build) / exe, Path(build) / "Release" / exe, Path(build) / "RelWithDebInfo" / exe):
        if c.is_file():
            return c
    raise SystemExit(f"error: {exe} not found in {build} (build the runtime first)")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run XboxBvhBuilder.dll's translated programs on the GPU (TEST-594).")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--game-dir", help="your package folder (holds XboxBvhBuilder.dll)")
    src.add_argument("--spirv", help="a folder of already translated programs (*.spv)")
    ap.add_argument("--work", help="work folder for the translation (default: a temporary folder)")
    ap.add_argument("--build", default=str(HERE.parent / "runtime" / "build"), help="folder with vk_compute_test")
    ap.add_argument("--max-shared-memory", type=int, default=32768,
                    help="the device's maxComputeSharedMemorySize, passed to rdna2spirv.py (default 32768)")
    ap.add_argument("--table-qword", type=lambda v: int(v, 0), default=0x2000,
                    help="the value of every root-table word (default 0x2000: past the table, inside the heap)")
    # 16 MiB keeps BuildKDOP (~2 MiB) and RadixCount (~8 MiB at default --table-qword 0x2000) inside the heap.
    ap.add_argument("--heap-bytes", type=int, default=16 << 20, help="initial heap image size (default 16 MiB)")
    ap.add_argument("--timeout", type=float, default=60.0, help="seconds per dispatch")
    ap.add_argument("--no-validation", action="store_true", help="don't enable VK_LAYER_KHRONOS_validation")
    ap.add_argument("--include-rehydration", action="store_true", help="also run the TEST-376 programs")
    ap.add_argument("--checked", action="store_true", help="also run CHECKED's programs and compare their output")
    ap.add_argument("--json", help="write the per-program results here")
    a = ap.parse_args(argv)
    if not math.isfinite(a.timeout) or a.timeout <= 0:
        ap.error("--timeout must be finite and greater than zero")
    if a.heap_bytes % 4 or a.heap_bytes < TABLE_QWORDS * 8:
        ap.error(f"--heap-bytes must be a multiple of 4 and at least {TABLE_QWORDS * 8}")
    if not 0 <= a.table_qword < 1 << 64:
        ap.error("--table-qword must fit in 64 bits")
    vkc = find_vk_compute_test(a.build)
    if a.spirv:
        spirv = Path(a.spirv)
    else:
        work = Path(a.work) if a.work else Path(tempfile.mkdtemp(prefix="bvh_gpu_work_"))
        spirv = translate(a.game_dir, work, a.max_shared_memory)
    results = smoke(spirv, vkc, table_qword=a.table_qword, heap_bytes=a.heap_bytes, validation=not a.no_validation,
                    timeout=a.timeout, include_rehydration=a.include_rehydration)
    if not results:
        print(f"error: no translated programs in {spirv}", file=sys.stderr)
        return 2
    for r in results:
        words = "-" if r["words_changed"] is None else str(r["words_changed"])
        print(f"{r['name']:28} {r['status']:16} {words:>8}  {r['detail'] if r['status'] != 'pass' else ''}")
    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    working = sum(1 for r in results if r["status"] == "pass" and r["words_changed"] and
                  r["words_changed"] > 0)
    print(f"{counts.get('pass', 0)} of {len(results)} pass; {working} changed heap words beyond running; "
          + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()) if k != "pass"))
    checked = []
    if a.checked:
        checked = run_checked(spirv, vkc, heap_bytes=a.heap_bytes, validation=not a.no_validation, timeout=a.timeout)
        for r in checked:
            print(f"checked {r['name']:20} {r['status']:10} {r['detail']}")
        print(f"checked: {sum(r['status'] == 'pass' for r in checked)} of {len(checked)} write exactly what they should")
    if a.json:
        Path(a.json).write_text(json.dumps({"max_shared_memory": a.max_shared_memory, "table_qword": a.table_qword,
                                            "heap_bytes": a.heap_bytes, "results": results, "checked": checked},
                                           indent=1), encoding="utf-8", newline="\n")
    ok = counts.get("pass", 0) == len(results) and all(r["status"] == "pass" for r in checked)
    return 0 if ok else 1


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
