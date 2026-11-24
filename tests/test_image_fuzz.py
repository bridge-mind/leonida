"""CPU checks for tools/image_fuzz.py: the oracle, the texel mapping and the generated shader, no GPU.

The GPU side (vk_image_fuzz, rdna2spirv.py) is what the fuzzer compares against; these tests make sure the
oracle it compares with is right, so a mismatch is the translator's and not the fuzzer's.
"""
import json
import os
from pathlib import Path
import random
import signal
import struct
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import image_fuzz as imf  # noqa: E402


def test_empty_format_selection_is_a_usage_error(tmp_path):
    work = tmp_path / "work"
    result = subprocess.run([sys.executable, "-I", imf.__file__, "--formats", " , ", "--work", str(work)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 2, result.stderr
    assert "--formats must select at least one format" in result.stderr
    assert "Traceback" not in result.stderr
    assert not work.exists()


def test_work_that_is_a_file_is_a_usage_error(tmp_path):
    work = tmp_path / "work.txt"
    work.write_text("keep me", encoding="utf-8")
    result = subprocess.run([sys.executable, "-I", imf.__file__, "--formats", "R8G8B8A8_UNORM", "--programs", "1",
                             "--work", str(work)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 2, result.stderr
    assert "ERROR: --work is not a folder" in result.stderr
    assert "Traceback" not in result.stderr
    assert work.read_text(encoding="utf-8") == "keep me"


def test_work_under_a_file_is_a_usage_error(tmp_path):
    blocker = tmp_path / "blocker.txt"
    blocker.write_text("x", encoding="utf-8")
    result = subprocess.run([sys.executable, "-I", imf.__file__, "--formats", "R8G8B8A8_UNORM", "--programs", "1",
                             "--work", str(blocker / "work")], capture_output=True, text=True, timeout=60)
    assert result.returncode == 2, result.stderr
    assert "ERROR: --work is not a folder" in result.stderr
    assert "Traceback" not in result.stderr
    assert blocker.read_text(encoding="utf-8") == "x"


def test_work_that_is_a_dangling_symlink_is_a_usage_error(tmp_path):
    link = tmp_path / "link"
    try:
        os.symlink(tmp_path / "missing", link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("cannot create a symlink here")
    result = subprocess.run([sys.executable, "-I", imf.__file__, "--formats", "R8G8B8A8_UNORM", "--programs", "1",
                             "--work", str(link)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 2, result.stderr
    assert "ERROR: --work is not a folder" in result.stderr
    assert "Traceback" not in result.stderr


def test_format_codes_match_the_translator_table():
    with open(os.path.join(imf.TOOLS, "gfx10_formats.json"), encoding="utf-8") as f:
        by_name = {name: int(code) for code, name in json.load(f).items()}
    gfx10_names = {
        "R8G8B8A8_UNORM": "GFX10_FORMAT_8_8_8_8_UNORM",
        "R16G16B16A16_SFLOAT": "GFX10_FORMAT_16_16_16_16_FLOAT",
        "R32G32B32A32_SFLOAT": "GFX10_FORMAT_32_32_32_32_FLOAT",
        "R32_SFLOAT": "GFX10_FORMAT_32_FLOAT",
        "R8G8B8A8_UINT": "GFX10_FORMAT_8_8_8_8_UINT",
        "R8G8B8A8_SINT": "GFX10_FORMAT_8_8_8_8_SINT",
    }
    assert set(gfx10_names) == set(imf.FORMATS)
    for name, (_, code, bpp, model) in imf.FORMATS.items():
        assert by_name[gfx10_names[name]] == code, name
        assert bpp in (4, 8, 16) and model in ("unorm8", "half4", "float4", "float1", "uint8", "sint8"), name


def test_unorm_oracle_divides_by_255():
    texels = bytes([0, 1, 127, 255, 10, 20, 30, 40])
    got = imf.oracle("R8G8B8A8_UNORM", texels, 2, 1)
    assert got[0] == [0.0, imf.f32(1 / 255.0), imf.f32(127 / 255.0), 1.0]
    assert got[1] == [imf.f32(v / 255.0) for v in (10, 20, 30, 40)]


def test_half_oracle_reads_four_binary16():
    wanted = [-2.0, -0.0, 0.5, 65504.0]
    texels = b"".join(struct.pack("<e", v) for v in wanted)
    got = imf.oracle("R16G16B16A16_SFLOAT", texels, 1, 1)[0]
    assert got == wanted


def test_float_oracle_and_missing_components():
    texels = struct.pack("<4f", 1.5, -2.25, 1024.0, float("inf"))
    got = imf.oracle("R32G32B32A32_SFLOAT", texels, 1, 1)[0]
    assert got == [1.5, -2.25, 1024.0, float("inf")]
    one = struct.pack("<f", -7.0)
    assert imf.oracle("R32_SFLOAT", one, 1, 1)[0] == [-7.0, 0.0, 0.0, 1.0]


@pytest.mark.parametrize("size", imf.SIZES)
def test_texel_mapping_over_every_size(size):
    for p in (0, 1, 17, 63):
        assert imf.texel_of_pixel(p, p, size, size) == (p * size // 64, p * size // 64)
    assert imf.texel_of_pixel(63, 0, size, size)[0] == size - 1


def test_pixel_shader_is_tabbed_and_carries_the_size():
    asm = imf.pixel_shader(16, 8)
    code = [line for line in asm.splitlines() if line and not line.startswith(";")]
    assert all(line.startswith("\t") for line in code), code
    assert "\timage_load v[4:7], v[2:3], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D" in asm
    assert "\tv_mul_f32_e32 v2, 16.0, v2" in asm and "\tv_mul_f32_e32 v3, 8.0, v3" in asm


@pytest.mark.parametrize("mask,want", [
    (0x1, [10.0, 2.0, 3.0, 4.0]),
    (0x2, [20.0, 2.0, 3.0, 4.0]),
    (0x5, [10.0, 30.0, 3.0, 4.0]),
    (0xa, [20.0, 40.0, 3.0, 4.0]),
    (0xf, [10.0, 20.0, 30.0, 40.0]),
])
def test_dmask_packs_selected_components_and_preserves_unwritten_registers(mask, want):
    assert imf.masked_components([10.0, 20.0, 30.0, 40.0], mask) == want


@pytest.mark.parametrize("mask", range(1, 16))
def test_pixel_shader_uses_the_requested_dmask(mask):
    asm = imf.pixel_shader(16, 8, mask)
    assert f"dmask:0x{mask:x}" in asm
    for reg, value in enumerate((1.0, 2.0, 3.0, 4.0), 4):
        assert f"v_mov_b32_e32 v{reg}, {value}" in asm


def test_dmask_schedule_covers_each_format_and_keeps_full_mask_first():
    formats = tuple(imf.FORMATS)
    pairs = {(formats[i % len(formats)], imf.DMASKS[(i // len(formats)) % len(imf.DMASKS)])
             for i in range(len(formats) * len(imf.DMASKS))}
    assert imf.DMASKS[0] == 0xf
    assert pairs == {(fmt, mask) for fmt in formats for mask in range(1, 16)}


@pytest.mark.parametrize("fmt,want,kind", [
    ("R8G8B8A8_UINT", [0, 127, 128, 255], "u32"),
    ("R8G8B8A8_SINT", [0, 127, -128, -1], "i32"),
])
def test_integer_oracle_and_conversion(fmt, want, kind):
    texels = bytes([0, 127, 128, 255]) * 64
    assert imf.oracle(fmt, texels, 8, 8)[0] == want
    asm = imf.pixel_shader(8, 8, 0x5, imf.FORMATS[fmt][3])
    assert f"v_cvt_f32_{kind}_e32 v4, v4" in asm
    assert f"v_cvt_f32_{kind}_e32 v5, v5" in asm
    assert f"v_cvt_f32_{kind}_e32 v6, v6" not in asm
    got = [want[0], want[2], 3.0, 4.0] * (imf.DRAW * imf.DRAW)
    assert not imf.compare(fmt, texels, 8, 8, got, 0x5)
    got[1] = want[1]
    assert imf.compare(fmt, texels, 8, 8, got, 0x5)


def test_compare_detects_wrong_dmask_order_and_unwritten_register():
    width = height = 8
    texels = struct.pack("<4f", 10.0, 20.0, 30.0, 40.0) * (width * height)
    got = [10.0, 30.0, 3.0, 4.0] * (imf.DRAW * imf.DRAW)
    assert imf.compare("R32G32B32A32_SFLOAT", texels, width, height, got, 0x5) == []
    got[1] = 20.0  # wrong component instead of B
    got[6] = 30.0  # second pixel: unwritten register changed
    problems = imf.compare("R32G32B32A32_SFLOAT", texels, width, height, got, 0x5)
    assert [(p[0], p[1]) for p in problems] == [(0, 0), (1, 0)]


def test_tsharp_word1_carries_the_format_and_the_size():
    rng = random.Random(3)
    for name, (_, code, bpp, _) in imf.FORMATS.items():
        texels, tsharp = imf.make_texels(name, 8, 16, rng)
        assert len(texels) == 8 * 16 * bpp
        assert tsharp[1] == code << 20
        assert tsharp[2] == 7 | (15 << 16)


def test_compare_distinguishes_signed_zero():
    """Vonsek's case: +0 where -0 is expected must be a mismatch, with the pixel's coordinates."""
    width = height = 8
    texels = b"".join(struct.pack("<4f", -0.0, 0.25, 0.5, 1.0) for _ in range(width * height))
    expected = imf.oracle("R32G32B32A32_SFLOAT", texels, width, height)
    got = []
    for y in range(imf.DRAW):
        for x in range(imf.DRAW):
            tx, ty = imf.texel_of_pixel(x, y, width, height)
            got.extend(expected[ty * width + tx])
    assert imf.compare("R32G32B32A32_SFLOAT", texels, width, height, got) == []
    got[(7 * imf.DRAW + 5) * 4] = 0.0  # the same texel, red +0 instead of -0
    problems = imf.compare("R32G32B32A32_SFLOAT", texels, width, height, got)
    assert [(p[0], p[1]) for p in problems] == [(5, 7)]


def f32_add_ulps(value, n):
    return struct.unpack("<f", struct.pack("<I", struct.unpack("<I", struct.pack("<f", value))[0] + n))[0]


def drawn(texel_values, width, height, dmask=0xf):
    """The 64x64 RGBA32F readback for per-texel component lists, packed by dmask over the sentinels."""
    got = []
    for y in range(imf.DRAW):
        for x in range(imf.DRAW):
            tx, ty = imf.texel_of_pixel(x, y, width, height)
            got.extend(imf.masked_components(texel_values[ty * width + tx], dmask))
    return got


def every_unorm8_byte():
    """An 8x8 R8G8B8A8_UNORM texture holding each of the 256 byte values once."""
    return bytes(range(256)), 8, 8


def test_compare_accepts_the_drivers_one_ulp_unorm8_conversion(monkeypatch):
    """llvmpipe converts UNORM8 as c * f32(1/255): one ulp above the exact c / 255 for 126 of the 256 values (27 gives
    0x3dd8d8da, not 0x3dd8d8d9). The conversion is the driver's, so it matches, and the summary counts it."""
    monkeypatch.setattr(imf, "unorm8_near", 0, raising=False)
    texels, width, height = every_unorm8_byte()
    recip = imf.f32(1 / 255.0)
    product = [[imf.f32(texels[i * 4 + c] * recip) for c in range(4)] for i in range(width * height)]
    exact = imf.oracle("R8G8B8A8_UNORM", texels, width, height)
    off = sum(p != e for pv, ev in zip(product, exact) for p, e in zip(pv, ev))
    assert off == 126
    assert struct.pack("<f", product[27 // 4][27 % 4]) == struct.pack("<I", 0x3DD8D8DA)
    assert imf.compare("R8G8B8A8_UNORM", texels, width, height, drawn(product, width, height)) == []
    assert imf.unorm8_near == 126 * (imf.DRAW // width) * (imf.DRAW // height)   # every texel is drawn 64 times


@pytest.mark.parametrize("wrong", ["two_ulps", "divide_by_256", "negative_zero"])
def test_compare_still_rejects_unorm8_values_that_are_not_the_quotient(wrong):
    texels, width, height = every_unorm8_byte()
    values = imf.oracle("R8G8B8A8_UNORM", texels, width, height)
    if wrong == "two_ulps":
        values = [[f32_add_ulps(v, 2) for v in texel] for texel in values]
    elif wrong == "divide_by_256":
        values = [[imf.f32(texels[i * 4 + c] / 256.0) for c in range(4)] for i in range(width * height)]
    else:   # byte 0 comes back as -0.0
        values[0] = [-0.0] + values[0][1:]
    assert imf.compare("R8G8B8A8_UNORM", texels, width, height, drawn(values, width, height))


def test_the_one_ulp_allowance_is_only_for_unorm8_texel_components():
    """Float formats stay bit exact, and so do the sentinels in the registers a UNORM8 dmask leaves unwritten."""
    width = height = 8
    texels = b"".join(struct.pack("<4f", 0.1, 0.25, 0.5, 1.0) for _ in range(width * height))
    values = [[f32_add_ulps(v, 1) for v in texel] for texel in imf.oracle("R32G32B32A32_SFLOAT", texels, width, height)]
    assert imf.compare("R32G32B32A32_SFLOAT", texels, width, height, drawn(values, width, height))

    texels, width, height = every_unorm8_byte()
    got = drawn(imf.oracle("R8G8B8A8_UNORM", texels, width, height), width, height, dmask=0x1)
    assert imf.compare("R8G8B8A8_UNORM", texels, width, height, got, 0x1) == []
    got[1] = f32_add_ulps(got[1], 1)   # pixel (0, 0): the sentinel 2.0 in the second register, one ulp off
    assert [(p[0], p[1]) for p in imf.compare("R8G8B8A8_UNORM", texels, width, height, got, 0x1)] == [(0, 0)]


def test_compare_treats_any_nan_as_equal():
    """The bit comparison keeps the old policy: NaN payloads are not modelled."""
    width = height = 8
    texels = b"".join(struct.pack("<4f", float("nan"), 1.0, 2.0, 3.0) for _ in range(width * height))
    expected = imf.oracle("R32G32B32A32_SFLOAT", texels, width, height)
    other_nan = struct.unpack("<f", struct.pack("<I", 0x7FC00001))[0]
    got = []
    for y in range(imf.DRAW):
        for x in range(imf.DRAW):
            tx, ty = imf.texel_of_pixel(x, y, width, height)
            value = list(expected[ty * width + tx])
            value[0] = other_nan
            got.extend(value)
    assert imf.compare("R32G32B32A32_SFLOAT", texels, width, height, got) == []


def run_case_with_harness_result(monkeypatch, tmp_path, returncode, stdout, stderr=""):
    """run_case with the translator stubbed out and vk_image_fuzz answering `returncode`/`stdout`/`stderr`."""
    build = tmp_path / "build"
    build.mkdir()
    (build / (imf.HARNESS + (".exe" if os.name == "nt" else ""))).touch()
    monkeypatch.setattr(imf, "assemble", lambda *_args: None)
    monkeypatch.setattr(imf, "translate", lambda work: str(tmp_path / "spirv"))
    monkeypatch.setattr(imf.subprocess, "run",
                        lambda cmd, **_kw: subprocess.CompletedProcess(cmd, returncode, stdout, stderr))
    return imf.run_case("R8G8B8A8_UNORM", 4, 4, random.Random(1), str(tmp_path), str(build), False, False)


def test_unsupported_device_skips_with_the_harness_reason(monkeypatch, tmp_path, capsys):
    # TEST-149: a 128-byte push constant device exits 77 with its own reason, which the skip line quotes.
    result = run_case_with_harness_result(
        monkeypatch, tmp_path, 77, "device: X\nunsupported: need 144 push constant bytes, device has 128\n")
    assert result is None
    assert "SKIP (unsupported: need 144 push constant bytes, device has 128)" in capsys.readouterr().out


def test_rejected_fragment_translation_is_reported_before_the_harness(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    build = tmp_path / "build"
    build.mkdir()
    harness = build / (imf.HARNESS + (".exe" if os.name == "nt" else ""))
    harness.touch()
    monkeypatch.setattr(imf, "assemble", lambda *_args: None)

    def run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd[0] == str(harness):
            return subprocess.CompletedProcess(cmd, 77, "unsupported: no Vulkan 1.2 device", "")
        out = work / "spirv"
        out.mkdir()
        (out / "vs_ids.spv").touch()
        return subprocess.CompletedProcess(cmd, 0, "2 shaders: 1 translated, 1 pass spirv-val\n  1 s_trap", "")

    monkeypatch.setattr(imf.subprocess, "run", run)
    with pytest.raises(SystemExit, match=r"(?s)ps_image_load\.spv.*s_trap"):
        imf.run_case("R8G8B8A8_UNORM", 4, 4, random.Random(1), str(work), str(build), False, False)


def test_unsupported_without_a_reason_line_still_skips(monkeypatch, tmp_path, capsys):
    assert run_case_with_harness_result(monkeypatch, tmp_path, 77, "device: X\n") is None
    assert "SKIP (the device does not support the set)" in capsys.readouterr().out


def test_a_harness_error_is_a_failure_not_a_skip(monkeypatch, tmp_path, capsys):
    assert run_case_with_harness_result(monkeypatch, tmp_path, 1, "device: X\nvkCreateDevice failed\n") is False
    out = capsys.readouterr().out
    assert "FAIL (vk_image_fuzz exit 1: vkCreateDevice failed)" in out and "SKIP" not in out
    assert "    device: X" in out  # the earlier lines are printed too


def test_a_crash_names_the_signal_and_prints_stderr(monkeypatch, tmp_path, capsys):
    # #1429: ANV segfaulted in vkCreateGraphicsPipelines and the fuzzer printed only "FAIL (harness error)".
    stderr = "validation: pipeline layout ok\ncreating graphics pipeline\n"
    assert run_case_with_harness_result(monkeypatch, tmp_path, -signal.SIGSEGV, "device: X\n", stderr) is False
    out = capsys.readouterr().out
    assert "FAIL (vk_image_fuzz killed by SIGSEGV: creating graphics pipeline)" in out
    assert "  vk_image_fuzz stderr:\n    validation: pipeline layout ok\n    creating graphics pipeline\n" in out
    assert "harness error" not in out


def test_a_silent_crash_still_says_how_the_harness_ended(monkeypatch, tmp_path, capsys):
    assert run_case_with_harness_result(monkeypatch, tmp_path, 0xC0000005, "") is False
    out = capsys.readouterr().out
    assert "FAIL (vk_image_fuzz exit 0xc0000005 (access violation), no output)" in out


def test_a_long_stderr_prints_only_its_tail(monkeypatch, tmp_path, capsys):
    stderr = "".join(f"line {i}\n" for i in range(50))
    assert run_case_with_harness_result(monkeypatch, tmp_path, 3, "", stderr) is False
    out = capsys.readouterr().out
    assert f"stderr (last {imf.STDERR_TAIL} of 50 lines):" in out
    assert "    line 49" in out and f"    line {50 - imf.STDERR_TAIL}\n" in out
    assert f"    line {49 - imf.STDERR_TAIL}\n" not in out


@pytest.mark.parametrize("returncode,want", [
    (1, "exit 1"),
    (2, "exit 2"),
    (-signal.SIGABRT, "killed by SIGABRT"),
    (-99, "killed by signal 99"),
    (0xC0000409, "exit 0xc0000409 (stack buffer overrun)"),
    (0xC0001234, "exit 0xc0001234"),
])
def test_exit_status(returncode, want):
    assert imf.exit_status(returncode) == want


def test_self_test_runs():
    r = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "tools", "image_fuzz.py"), "--self-test"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "self-test: ok" in r.stdout


def build_dir_main_uses(monkeypatch, argv):
    """The build folder main() hands run_case for `argv`, with run_case stubbed to a pass."""
    seen = []
    monkeypatch.setattr(imf, "run_case", lambda *args, **_kw: seen.append(args[5]) or True)
    monkeypatch.setattr(sys, "argv", ["image_fuzz.py", "--programs", "1"] + argv)
    assert imf.main() == 0
    return seen[0]


def test_default_build_is_runtime_build_on_every_platform(monkeypatch, tmp_path):
    # It was runtime/build/linux, which only a Linux build following docs/testing.md has: on Windows (and on Linux
    # following docs/shader_validation.md) every run stopped with "vk_image_fuzz not found" unless --build was given.
    monkeypatch.setattr(imf, "find_test_program", lambda build, name: None)
    assert build_dir_main_uses(monkeypatch, ["--work", str(tmp_path)]) == os.path.join(imf.ROOT, "runtime", "build")


def test_visual_studio_config_subfolder_is_found(monkeypatch, tmp_path):
    build = tmp_path / "build"
    (build / "RelWithDebInfo").mkdir(parents=True)
    (build / "RelWithDebInfo" / (imf.HARNESS + (".exe" if os.name == "nt" else ""))).touch()
    got = build_dir_main_uses(monkeypatch, ["--build", str(build), "--work", str(tmp_path / "w")])
    assert got == str(build / "RelWithDebInfo")


def test_translate_forwards_pc_tail_ubo(monkeypatch, tmp_path):
    cmds = []

    def fake_run(cmd, **_kw):
        cmds.append(list(cmd))
        out = cmd[cmd.index("--out") + 1]           # translate() now checks the two programs were emitted
        os.makedirs(out, exist_ok=True)
        for name in ("vs_ids.spv", "ps_image_load.spv"):
            open(os.path.join(out, name), "wb").close()
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(imf.subprocess, "run", fake_run)
    imf.translate(str(tmp_path))
    imf.translate(str(tmp_path), pc_tail=True)
    assert "--pc-tail-ubo" not in cmds[0]
    assert "--pc-tail-ubo" in cmds[1]


def test_run_case_passes_pc_tail_to_translator_and_harness(monkeypatch, tmp_path):
    build = tmp_path / "build"
    build.mkdir()
    (build / (imf.HARNESS + (".exe" if os.name == "nt" else ""))).touch()
    seen_translate = []
    seen_env = []
    monkeypatch.setattr(imf, "assemble", lambda *_args: None)
    monkeypatch.setattr(imf, "translate", lambda work, pc_tail=False: seen_translate.append(pc_tail) or str(tmp_path / "spirv"))

    def fake_run(cmd, **kw):
        seen_env.append(kw.get("env"))
        return subprocess.CompletedProcess(cmd, 77, "unsupported: test\n", "")

    monkeypatch.setattr(imf.subprocess, "run", fake_run)
    imf.run_case("R8G8B8A8_UNORM", 4, 4, random.Random(1), str(tmp_path), str(build), False, False, pc_tail=True)
    assert seen_translate == [True]
    assert seen_env[0]["LEONIDA_PC_TAIL_UBO"] == "1"


def test_main_enables_pc_tail_ubo_from_flag_or_env(monkeypatch, tmp_path):
    for argv, env_val, want in (
        ([], "", False),
        (["--pc-tail-ubo"], "", True),
        ([], "1", True),
        ([], "true", True),
        ([], "0", False),
    ):
        seen = []
        monkeypatch.setenv("LEONIDA_PC_TAIL_UBO", env_val)
        monkeypatch.setattr(imf, "run_case", lambda *args, **kw: seen.append(kw.get("pc_tail", False)) or True)
        monkeypatch.setattr(sys, "argv", ["image_fuzz.py", "--programs", "1", "--work", str(tmp_path)] + argv)
        assert imf.main() == 0
        assert seen == [want], (argv, env_val)
