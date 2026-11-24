"""Repo-friendly, optional GPU test: byte-exact Leonida COPY/COMPUTE tiled-buffer replay
AND actual VK_SHARING_MODE_CONCURRENT on *every* sparse VkBuffer created by the layer.

Runs the original synthetic `d3d12_sparse_queue_family_test` target.
Invokes the SDK's VK_LAYER_LUNARG_api_dump in a short-lived isolated directory, then reruns the test
under VK_LAYER_KHRONOS_validation alone (api_dump's vkDestroyDevice can crash on AMD Windows, with both loaded
more often than alone: see died_in_vk_destroy_device);
no private-game files, asset paths, device memory dumps, or committed logs.

  python -I check_sparse_queue_family.py path/to/d3d12_sparse_queue_family_test.exe
  python -I check_sparse_queue_family.py path/to/baseline.exe --expect-baseline-exclusive

By default the baseline fails (exit 1) and patched runtime passes (exit 0).
If the API dump layer is absent the result skips (exit 77, CTest SKIP_RETURN_CODE).
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Mapping


SPARSE_FLAGS = "VkBufferCreateFlags = 7 (VK_BUFFER_CREATE_SPARSE_BINDING_BIT"
MODE = re.compile(r"sharingMode:\s+VkSharingMode = (VK_SHARING_MODE_\w+)")
COUNT = re.compile(r"queueFamilyIndexCount:\s+uint32_t = (\d+)")
INDICES = re.compile(r"pQueueFamilyIndices\[(\d+)\]:\s+const uint32_t = (\d+)")
# Each of the two runs gets 30 s, so both together stay inside the CTest TIMEOUT of 65 s (runtime/CMakeLists.txt).
RUN_TIMEOUT = 30
ACCESS_VIOLATION = 0xC0000005   # STATUS_ACCESS_VIOLATION, as Windows reports it in the exit code
CASES = ["LAYER_MAP_COMPUTE_COPY_COMPUTE", "LAYER_MAP_COPY_COPY_COMPUTE", "LAYER_MAP_DIRECT_COPY_COMPUTE"]   # sorted
FINAL = re.compile(r"^LAYER_FINAL PASS$", re.M)
CALL_HEAD = re.compile(r"^(vk\w+)\(", re.M)
CASE = re.compile(r"^RESULT (LAYER_MAP_(?:DIRECT|COPY|COMPUTE)_COPY_COMPUTE) PASS mapped_mismatch=0/65536 unmapped_nonzero=0/65536$", re.M)


def check_trace(path: Path, expect_exclusive: bool) -> tuple[int, int]:
    rows = path.read_text(encoding="utf-8", errors="replace").splitlines()
    records = 0
    for i, line in enumerate(rows):
        if SPARSE_FLAGS not in line:
            continue
        end = i + 1
        while end < len(rows) and "VkBufferCreateFlags =" not in rows[end] and not CALL_HEAD.match(rows[end]):
            end += 1
        tail = "\n".join(rows[i + 1:end])
        mode, count = MODE.search(tail), COUNT.search(tail)
        if not mode or not count:
            raise AssertionError(f"incomplete API dump sparse VkBuffer at line {i+1}")
        sharing, n = mode.group(1), int(count.group(1))
        indices = [(int(slot), int(family)) for slot, family in INDICES.findall(tail)]
        if expect_exclusive:
            if sharing != "VK_SHARING_MODE_EXCLUSIVE" or n != 0 or indices:
                raise AssertionError(f"baseline negative control not reproduced at line {i+1}")
        elif (sharing != "VK_SHARING_MODE_CONCURRENT" or n < 2 or len(indices) != n
              or len({slot for slot, _ in indices if slot < n}) != n
              or len({family for _, family in indices}) != n):
            raise AssertionError(f"incomplete or non-concurrent sparse buffer at line {i+1}: "
                                 f"{sharing}, count={n}, indices={indices}")
        records += 1
    if records < 6 or records % 2:
        raise AssertionError(f"insufficient sparse VkBuffer create-info records: {records}")
    return records // 2, records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=Path, help="Compiled Leonida d3d12_sparse_queue_family_test binary")
    parser.add_argument("--expect-baseline-exclusive", action="store_true",
                        help="negative-control mode: prove buggy EXCLUSIVE create infos instead of requiring fix")
    args = parser.parse_args()
    exe = args.executable.resolve()
    if not exe.is_file():
        print(f"ERROR: missing executable {exe}", file=sys.stderr)
        return 2
    with tempfile.TemporaryDirectory(prefix="leonida_sparse_trace_") as td:
        path = Path(td)
        dump_name = path / "sparse_create_info.log"
        (path / "vk_layer_settings.txt").write_text(
            "lunarg_api_dump.file = true\n"
            f"lunarg_api_dump.log_filename = {dump_name}\n"
            "lunarg_api_dump.output_format = text\n"
            "lunarg_api_dump.pre_dump = true\n"
            "lunarg_api_dump.detailed = true\n",
            encoding="utf-8",
        )
        dump_env, validation_env = pass_envs(os.environ, path)
        # Two runs, one layer each. SDK 1.4.363's api_dump reads the VkDevice's dispatch key after the call down the
        # chain has freed the device (fix: LunarG/VulkanTools#2754), so on the AMD Windows driver its vkDestroyDevice
        # can crash with 0xC0000005. Validation below it makes that likelier (every run on some RDNA2 iGPUs), and an
        # RX 9070 crashes now and then with api_dump alone, so this run allows exactly that crash.
        dump_out, skip = run_pass(exe, dump_env, path, trace_dir=path)
        if skip:
            return skip
        # Check the trace before the validation run: a skip there must not hide a contract the trace has broken.
        candidates = list(path.glob("sparse_create_info.log*"))
        if not candidates:
            print("SKIP 77: Vulkan SDK VK_LAYER_LUNARG_api_dump unavailable or did not write trace")
            return 77
        calls, infos = check_trace(candidates[0], args.expect_baseline_exclusive)
        validation_out, skip = run_pass(exe, validation_env, path)
        if skip:
            return skip
        errors = [line for line in validation_out.splitlines() if "Validation Error" in line or "VUID-" in line]
        if errors:
            raise AssertionError(f"Khronos validation reported {len(errors)} error line(s); first: {errors[0][:300]}")
        kinds = sorted(CASE.findall(dump_out))
        print(f"GPU PASS: {len(kinds)} COPY/COMPUTE sparse queue cases, mapped 65,536 bytes exact, unmapped 65,536 zeros")
        print(f"VK API CONTRACT PASS: {calls} sparse VkBuffer creations / {infos} create-info records = "
              f"{'EXCLUSIVE negative-control' if args.expect_baseline_exclusive else 'CONCURRENT with 2+ unique families'}")
        print("Khronos validation run (no api_dump): the same GPU cases pass with no validation errors reported")
        return 0


def pass_envs(base: Mapping[str, str], settings_dir: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Environments for the api_dump run and the validation run; each loads exactly one of the two SDK layers."""
    dump = dict(base)
    # VK_INSTANCE_LAYERS makes the loader insert api_dump without the runtime asking for it.
    dump["VK_INSTANCE_LAYERS"] = "VK_LAYER_LUNARG_api_dump"
    dump["LEONIDA_VK_VALIDATION"] = "0"
    dump["VK_LAYER_SETTINGS_PATH"] = str(settings_dir)
    validation = dict(base)
    validation.pop("VK_INSTANCE_LAYERS", None)
    validation["LEONIDA_VK_VALIDATION"] = "1"
    return dump, validation


def died_in_vk_destroy_device(trace_dir: Path) -> bool:
    """True when the last call in the api_dump trace is vkDestroyDevice (pre_dump writes it before the call).

    The trace can't tell api_dump's own read of the freed device from a driver crash in the same call. The validation
    run can: it destroys the device without api_dump and has to exit 0.
    """
    traces = list(trace_dir.glob("sparse_create_info.log*"))
    if len(traces) != 1:
        return False
    calls = CALL_HEAD.findall(traces[0].read_text(encoding="utf-8", errors="replace"))
    return bool(calls) and calls[-1] == "vkDestroyDevice"


def run_pass(exe: Path, env: Mapping[str, str], cwd: Path, trace_dir: Path | None = None) -> tuple[str, int]:
    """Run the GPU test once: (output, 77) for a skip, (output, 0) when all three cases passed.

    trace_dir (the api_dump run only) allows an access violation in vkDestroyDevice, once every result is printed.
    """
    run = subprocess.run([str(exe)], cwd=cwd, env=dict(env), capture_output=True,
                         text=True, encoding="utf-8", errors="replace", timeout=RUN_TIMEOUT)
    out = run.stdout + "\n" + run.stderr
    if run.returncode == 77:
        print("SKIP 77: Leonida Vulkan device / sparse features unavailable")
        return out, 77
    # The loader refuses the instance when a layer named in VK_INSTANCE_LAYERS (or the validation layer
    # LEONIDA_VK_VALIDATION asks for) is not installed: that is the "layer absent" skip, not a layer failure.
    if run.returncode != 0 and "vkCreateInstance" in out and "failed" in out:
        print("SKIP 77: VK_LAYER_LUNARG_api_dump or VK_LAYER_KHRONOS_validation unavailable (vkCreateInstance failed)")
        return out, 77
    teardown_crash = (trace_dir is not None and run.returncode == ACCESS_VIOLATION
                      and died_in_vk_destroy_device(trace_dir))
    if run.returncode != 0 and not teardown_crash:
        print(out[-7000:], file=sys.stderr)
        raise AssertionError(f"Leonida GPU test failed exit={run.returncode}")
    kinds = sorted(CASE.findall(out))
    if kinds != CASES or len(FINAL.findall(out)) != 1:
        raise AssertionError(f"GPU byte verification missing: cases={kinds}; tail={out[-2000:]}")
    if teardown_crash:
        print("NOTE: the api_dump run crashed in vkDestroyDevice (0xC0000005) after every case passed; "
              "the validation run still has to exit 0")
    return out, 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, subprocess.TimeoutExpired) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
