"""SHADER-10: the hand-written ps_w64_* and vs_w64_* fixtures get the tier their header names, and the census counts
tiers.

The fixtures (runtime/test/shaders/{ps,vs}_w64_{a,b,c}_*.asm) are the GPU tests of w64_graphics.py: each pixel fixture
was drawn on an AMD Radeon 880M with the fragment subgroup size required to 64 and to 32, each vertex fixture (TEST-251)
on an RX 9070 XT and a Ryzen 9800X3D iGPU (RADV) with the VS stage at wave64 and at wave32 (RADV_PERFTEST=gewave32).
Tier A and B fixtures gave identical words (and the words their header derives), tier C fixtures differed, so this
test pins the translator's detector to what the hardware showed. The permute fixtures (TEST-319: ps_w64_a_permlane,
_a_swizzle, _a_bpermute) gave the words their header derives on an RTX 5070 (32 only), and every fixture's
RTX 5070 run matched (tiers A, B) or differed from (tier C) a 64-lane RX 580 (GCN, RADV) reference.
Needs llvm-mc with the AMDGPU target and spirv-as / spirv-val, like the other tests that build shaders; the
pure-Python parts run without them.
"""
import json
import os
import struct
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SHADERS = os.path.join(ROOT, "runtime", "test", "shaders")
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, SHADERS)

import wave64_halves  # noqa: E402
import w64_graphics  # noqa: E402
from conftest import LLVM_MC, requires_tools  # noqa: E402

EXPECT = {
    "ps_w64_a_skip": ("A", [], []),
    "ps_w64_a_one_lane": ("A", [], []),
    "ps_w64_a_permlane": ("A", [], []),
    "ps_w64_a_swizzle": ("A", [], []),
    "ps_w64_a_bpermute": ("A", [], []),
    "ps_w64_a_uniform_vote": ("A", [], []),

    "ps_w64_a_discard": ("A", [], []),
    "ps_w64_b_waterfall": ("B", [], ["waterfall"]),
    "ps_w64_b_readfirstlane": ("B", [], ["readfirstlane-uniform"]),
    "ps_w64_b_readlane_uniform": ("B", [], ["readlane-uniform"]),
    "ps_w64_b_readlane_uniform_lane": ("B", [], ["readlane-uniform"]),
    "ps_w64_b_ballot_atomic": ("B", [], ["aggregated-atomic", "elected-lane"]),
    "ps_w64_c_waterfall_count": ("C", ["stale-scalar"], None),
    "ps_w64_c_bcnt_export": ("C", ["lane-count"], None),
    "ps_w64_c_readlane": ("C", ["cross-lane"], None),
    "ps_w64_c_lane_id": ("C", ["lane-id"], None),
    "ps_w64_c_vote_stale": ("C", ["stale-scalar", "unsafe-vote-branch"], None),
    "ps_w64_c_vote_else": ("C", ["unsafe-vote-branch"], None),
    "ps_w64_c_scc_vote": ("C", ["cross-lane"], None),
    "ps_w64_c_execz_sgpr": ("C", ["stale-scalar", "unsafe-execz-branch"], None),
    "vs_w64_a_skip": ("A", [], []),
    "vs_w64_b_waterfall": ("B", [], ["waterfall"]),
    "vs_w64_b_readlane_uniform": ("B", [], ["readfirstlane-uniform", "readlane-uniform"]),
    "vs_w64_b_ballot_atomic": ("B", [], ["aggregated-atomic", "elected-lane"]),
    "vs_w64_c_bcnt_export": ("C", ["lane-count"], None),
    "vs_w64_c_readlane": ("C", ["cross-lane"], None),
    "vs_w64_c_lane_id": ("C", ["lane-id"], None),
    "gs_w64_a_ring": ("A", [], []),
    "gs_w64_c_lane_id": ("C", ["lane-id"], None),
}


def test_every_fixture_has_a_tier_and_a_gpu_check():
    asm = {f[:-4] for f in os.listdir(SHADERS) if f.startswith(("ps_w64_", "vs_w64_", "gs_w64_")) and f.endswith(".asm")}
    assert asm == set(EXPECT) == set(w64_graphics.FIXTURES)
    for name, (tier, _, _) in EXPECT.items():
        assert w64_graphics.FIXTURES[name][0] == tier


def test_tier_c_reference_rejects_changes_to_invariant_words(tmp_path, monkeypatch):
    name = "ps_w64_c_lane_id"
    (tmp_path / (name + ".spv")).touch()
    expected = w64_graphics.FIXTURES[name][1](0)
    reference = tuple(0 if value is None else value for value in expected)
    (tmp_path / (name + ".ref64.raw")).write_bytes(struct.pack("<4I", *reference))
    monkeypatch.setenv("W64G_REF", str(tmp_path))
    monkeypatch.delenv("W64G_SAVE_REF", raising=False)

    words = [tuple(value + (expected[i] is None) for i, value in enumerate(reference))]
    monkeypatch.setattr(w64_graphics, "draw", lambda _tools, _spirv, _name, size, _tmp: (words, None, size == 32))
    assert w64_graphics.check(None, tmp_path, name) == ""

    stable = next(i for i, value in enumerate(expected) if value is not None)
    words[0] = tuple(value + (i == stable) for i, value in enumerate(words[0]))
    msg = w64_graphics.check(None, tmp_path, name)     # wording varies by which comparison caught it
    assert "partition" in msg and "differs" in msg, msg


@requires_tools
def test_fixtures_are_classified_as_their_headers_say(tmp_path):
    env = dict(os.environ)
    if LLVM_MC:
        env["LLVM_MC"] = " ".join(LLVM_MC)
    r = subprocess.run([sys.executable, "-I", os.path.join(SHADERS, "build.py"), str(tmp_path), "vs_ids", *EXPECT],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    index = json.loads((tmp_path / "spirv" / "index.json").read_text(encoding="utf-8"))
    for name, (tier, reasons, idioms) in EXPECT.items():
        m = index[name]
        assert m["wave64"] is True and m["wave64_emulated"] is True
        assert m["wave64_graphics"] == tier, (name, m.get("wave64_graphics_reasons"))
        assert m.get("wave64_graphics_reasons", []) == reasons, name
        if idioms is not None:
            assert m.get("wave64_graphics_idioms", []) == idioms, name
    assert "wave64_graphics" not in index["vs_ids"]            # a wave32 shader carries no tier


def test_census_counts_tiers_without_names():
    index = {
        "a": {"stage": "ps", "wave64": True, "wave64_graphics": "A"},
        "b": {"stage": "ps", "wave64": True, "wave64_graphics": "B", "wave64_graphics_idioms": ["waterfall"]},
        "c": {"stage": "vs", "wave64": True, "wave64_graphics": "C", "wave64_graphics_reasons": ["lane-count"]},
        "d": {"stage": "ps", "wave64": False},
        "e": {"stage": "cs", "wave64": True},
        "f": {"stage": "gs", "wave64": True},
    }
    c = wave64_halves.census(index)
    assert c["tiers"] == {"ps": {"A": 1, "B": 1}, "vs": {"C": 1}}
    assert c["reasons"] == {"lane-count": 1} and c["idioms"] == {"waterfall": 1}
    assert c["no_analysis"] == 1                                 # the gs entry was translated before the analysis


def test_census_cli_prints_counts_only(tmp_path, capsys):
    path = tmp_path / "index.json"
    path.write_text(json.dumps({"secret_shader_name": {"stage": "ps", "wave64": True, "wave64_graphics": "C",
                                                        "wave64_graphics_reasons": ["lane-id"]}}))
    assert wave64_halves.main(["--census", str(path)]) == 0
    out = capsys.readouterr().out
    assert "ps: C=1" in out and "lane-id x1" in out and "secret_shader_name" not in out


def test_a_bad_census_is_one_line_not_a_traceback(tmp_path, capsys):
    """--census used to traceback: FileNotFoundError, PermissionError on a folder, JSONDecodeError for a file that is
    not JSON and AttributeError for a JSON list, so the tier census could not say what was wrong with its input."""
    not_json = tmp_path / "junk.json"
    not_json.write_text("RSC8 not an index", encoding="utf-8")
    a_list = tmp_path / "list.json"
    a_list.write_text("[]", encoding="utf-8")
    a_number = tmp_path / "num.json"
    a_number.write_text("7", encoding="utf-8")
    folder = tmp_path / "folder"
    folder.mkdir()
    for path in (tmp_path / "missing.json", folder, not_json, a_list, a_number):
        capsys.readouterr()
        code = wave64_halves.main(["--census", str(path)])
        out = capsys.readouterr()
        assert code == 2, f"{path.name}: exit {code}, {out.err!r}"
        assert "Traceback" not in out.err, out.err
        assert len(out.err.strip().splitlines()) == 1 and out.err.startswith("ERROR: "), out.err
        assert out.out == "", f"{path.name} printed a census: {out.out!r}"


def test_an_empty_index_still_reports_no_tier(tmp_path, capsys):
    path = tmp_path / "index.json"
    path.write_text("{}", encoding="utf-8")
    assert wave64_halves.main(["--census", str(path)]) == 0
    out = capsys.readouterr()
    assert out.out.strip() == "no wave64 VS/PS/GS entry with a wave64_graphics tier" and out.err == ""


def test_expected_words_of_the_gpu_fixtures():
    assert w64_graphics._a_skip(1 + 64 * 3) == (7 * 3 + 1, 1, 2000 + 3, 2)
    assert w64_graphics._a_skip(6 + 64 * 2) == (1006, 0, 2000 + 2 + 9, 3)
    assert w64_graphics._b_waterfall(5 + 64 * 6) == (17 * ((5 & 3) + 4 * (6 & 3)) + 5, (5 & 3) + 4 * (6 & 3), 0, 0)
    assert w64_graphics._b_readfirstlane(10) == (133, 277, 0, 0)
    assert w64_graphics._a_discard(29 + 64 * 3) == (0xBF800000,) * 4                 # x < 30: discarded
    assert w64_graphics._a_discard(30 + 64 * 49) == (7 * 49 + 30, 1, 2049, 3)
    assert w64_graphics._a_discard(31 + 64 * 1) == (0xBF800000,) * 4                 # x % 4 == 3, y odd
    assert w64_graphics._vs_a_skip(1 + 64 * 3) == (7 * 3 + 1, 1, 2000 + 3, 2)        # no whole-quad word in r1
    assert w64_graphics._vs_a_skip(6 + 64 * 2) == (1006, 0, 2000 + 2 + 9, 3)
    assert w64_graphics._vs_b_readlane_uniform(70) == (6 + 123, 200, 6 + 123, 277)
    assert w64_graphics._vs_c_lane_id(700) == (None, None, 700, 0)


def test_vs_probe_and_radv_perftest(monkeypatch):
    monkeypatch.setenv("RADV_PERFTEST", "gewave32,nggc")
    assert w64_graphics._radv_perftest(64) == "nggc"                # the user's other flags stay, gewave32 goes
    assert w64_graphics._radv_perftest(32) == "nggc,gewave32"
    assert all(w64_graphics.FIXTURES[p][0] == "C" and p.startswith(s) for s, p in w64_graphics.PROBES.items())
    assert w64_graphics.VS_RECORDS * 4 >= 0x100 + 4 * w64_graphics.VS_VERTICES   # the atomic's slots end below the records


def test_live_lane_need_skips_on_a_narrow_stage_and_runs_on_a_wide_one(tmp_path, monkeypatch):
    # The skip is keyed on the measured live width, not on the reported subgroup size or on whether a size could be required.
    name = "ps_w64_a_one_lane"
    (tmp_path / (name + ".spv")).write_bytes(b"")
    words = [w64_graphics.FIXTURES[name][1](i) for i in range(64 * 64)]
    monkeypatch.setattr(w64_graphics, "draw", lambda tools, spirv, n, size, tmp: (words, None, False if size else None))
    monkeypatch.setitem(w64_graphics.NEEDS_LIVE_LANES, name, 32)
    monkeypatch.setattr(w64_graphics, "live_width", lambda tools, spirv, tmp: 16)
    skip = w64_graphics.check("tools", str(tmp_path), name)
    assert isinstance(skip, w64_graphics.Skip) and "16 live lanes" in skip and "needs 32" in skip
    monkeypatch.setattr(w64_graphics, "live_width", lambda tools, spirv, tmp: 32)
    assert w64_graphics.check("tools", str(tmp_path), name) == ""
    monkeypatch.setattr(w64_graphics, "live_width", lambda tools, spirv, tmp: None)   # probe could not run: judged
    assert w64_graphics.check("tools", str(tmp_path), name) == ""
