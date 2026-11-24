"""tools/user_sgpr_census.py on hand-written index.json data (no game files, no GPU)."""
import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest import mock

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import report_paths  # noqa: E402
import user_sgpr_census as usc  # noqa: E402


def reads(*pairs):
    return [{"sgpr": s, "kind": k} for s, k in pairs]


INDEX = {
    "a.ps": {"stage": "ps", "user_sgpr_reads": reads((8, "ptr32_lo"), (9, "const32"))},
    "b.ps": {"stage": "ps", "user_sgpr_reads": reads((1, "const32"), (8, "ptr32_lo"), (11, "ptr32_lo"))},
    "c.ps": {"stage": "ps", "user_sgpr_reads": reads((9, "ptr32_lo"))},
    "d.cs": {"stage": "cs", "user_sgpr_reads": reads((4, "ptr32_lo"), (5, "const32"))},
    "e.gs": {"stage": "gs", "user_sgpr_reads": reads((2, "const32"), (8, "ptr64_lo"), (9, "ptr64_hi"))},
    "f.vs": {"stage": "vs", "user_sgpr_reads": []},
}


def test_lowest_pointer_per_stage():
    s = usc.census(INDEX)
    assert s["ps"]["shaders"] == 3
    assert s["ps"]["lowest_pointer_sgpr"] == {"s8": 2, "s9": 1}
    assert s["ps"]["most_common_lowest_pointer_sgpr"] == 8
    assert s["ps"]["lowest_pointer_kind"] == {"ptr32_lo": 3}
    assert s["cs"]["most_common_lowest_pointer_sgpr"] == 4
    assert s["gs"]["lowest_pointer_kind"] == {"ptr64_lo": 1}
    assert s["vs"]["most_common_lowest_pointer_sgpr"] is None


def test_tie_most_common_prefers_smaller_sgpr_regardless_of_insertion_order(tmp_path, capsys):
    """Equal highest counts must pick the smaller SGPR, not depend on dict insertion order."""
    forward = {"x": {"stage": "ps", "user_sgpr_reads": reads((8, "ptr32_lo"))},
               "y": {"stage": "ps", "user_sgpr_reads": reads((9, "ptr32_lo"))}}
    reversed_order = {"y": forward["y"], "x": forward["x"]}

    a = usc.census(forward)
    b = usc.census(reversed_order)
    assert a["ps"]["most_common_lowest_pointer_sgpr"] == 8
    assert b["ps"]["most_common_lowest_pointer_sgpr"] == 8

    # --expect must agree for both orderings.
    p = tmp_path / "index.json"
    p.write_text(json.dumps(reversed_order), encoding="utf-8")
    assert usc.main([str(p), "--expect", "ps=8"]) == 0
    assert usc.main([str(p), "--expect", "ps=9"]) == 1
    capsys.readouterr()


def test_const_reads_are_not_pointers():
    s = usc.census({"x": {"stage": "ps", "user_sgpr_reads": reads((0, "const32"), (3, "const32"))}})
    assert s["ps"]["lowest_pointer_sgpr"] == {}
    assert s["ps"]["shaders_reading_kind"] == {"const32": 1}


def model_two_dword_tables_from_s0():
    """Shaders packed as runtime/src/root_signature.h does today: tables are 64-bit VAs from s0."""
    return {
        "t0": {"stage": "ps", "user_sgpr_reads": reads((0, "ptr64_lo"), (1, "ptr64_hi"), (2, "ptr64_lo"), (3, "ptr64_hi"))},
        "t1": {"stage": "ps", "user_sgpr_reads": reads((0, "ptr64_lo"), (1, "ptr64_hi"), (2, "const32"))},
    }


def model_one_dword_tables_from_s8():
    """Shaders packed with a stage base (PS s8) and one-dword tables: 32-bit VAs, high half implicit zero."""
    return {
        "t0": {"stage": "ps", "user_sgpr_reads": reads((8, "ptr32_lo"), (9, "ptr32_lo"))},
        "t1": {"stage": "ps", "user_sgpr_reads": reads((1, "const32"), (8, "ptr32_lo"), (9, "const32"), (10, "ptr32_lo"))},
    }


def test_the_two_packing_models_are_told_apart():
    a = usc.census(model_two_dword_tables_from_s0())["ps"]
    b = usc.census(model_one_dword_tables_from_s8(), base={"ps": 8})["ps"]
    # two dwords per table: pointers two SGPRs apart, never a 32-bit pointer, high halves read
    assert a["pointer_gap"] == {2: 1}
    assert a["after_ptr32"] == {}
    assert a["most_common_lowest_pointer_sgpr"] == 0
    assert a["shaders_reading_kind"].get("ptr64_hi") == 2
    # one dword per table from s8: pointers adjacent, no high halves, a system read below the base
    assert b["pointer_gap"] == {1: 1, 2: 1}
    # t0: s8 -> s9 is a pointer, s9 -> s10 not read; t1: s8 -> s9 const32, s10 -> s11 not read
    assert b["after_ptr32"] == {"const32": 1, "not read": 2, "pointer": 1}
    assert b["most_common_lowest_pointer_sgpr"] == 8
    assert "ptr64_hi" not in b["shaders_reading_kind"]
    assert b["below_base"] == {"s1:const32": 1}


@pytest.mark.parametrize("output_kind", ["same-path", "relative-alias", "hard-link", "folding-volume-case"])
def test_json_report_refuses_to_overwrite_index(tmp_path, capsys, output_kind):
    assert usc.same_report_path is report_paths.same_report_path
    index = tmp_path / "index.json"
    original = json.dumps(INDEX).encode("utf-8")
    index.write_bytes(original)
    output = index
    if output_kind == "relative-alias":
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        output = subdir / ".." / index.name
    elif output_kind == "hard-link":
        output = tmp_path / "alias.json"
        try:
            output.hardlink_to(index)
        except OSError as err:
            pytest.skip(f"hard links unavailable: {err}")
    elif output_kind == "folding-volume-case":
        output = tmp_path / "INDEX.JSON"

    probe = mock.Mock(return_value=True)
    ctx = (mock.patch.multiple(report_paths, path_key=mock.Mock(side_effect=str),
                               same_name_on_folding_volume=probe)
           if output_kind == "folding-volume-case" else contextlib.nullcontext())
    with ctx, (mock.patch.object(os.path, "samefile", side_effect=OSError("not same"))
               if output_kind == "folding-volume-case" else contextlib.nullcontext()):
        with pytest.raises(SystemExit) as exc:
            usc.main([str(index), "--json", str(output)])
    if output_kind == "folding-volume-case":
        probe.assert_called_once()

    assert exc.value.code == 2
    assert "--json must not overwrite the input index" in capsys.readouterr().err
    assert index.read_bytes() == original


def test_json_report_accepts_a_separate_output(tmp_path):
    index = tmp_path / "index.json"
    output = tmp_path / "summary.json"
    index.write_text(json.dumps(INDEX), encoding="utf-8")

    assert usc.main([str(index), "--json", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["ps"]["lowest_pointer_sgpr"] == {"s8": 2, "s9": 1}
    assert json.loads(index.read_text(encoding="utf-8")) == INDEX


def test_expect_passes_and_fails(tmp_path, capsys):
    p = tmp_path / "index.json"
    p.write_text(json.dumps(INDEX), encoding="utf-8")
    assert usc.main([str(p), "--expect", "ps=8", "cs=4"]) == 0
    assert usc.main([str(p), "--expect", "ps=0"]) == 1
    assert "expected s0, got s8" in capsys.readouterr().err


def test_cli_rejects_malformed_stage_assignment():
    with tempfile.TemporaryDirectory() as tmp:
        index = Path(tmp, "index.json")
        report = Path(tmp, "report.json")
        index.write_text(json.dumps(INDEX), encoding="utf-8")
        tool = os.path.join(ROOT, "tools", "user_sgpr_census.py")
        for option in ("--base", "--expect"):
            for assignment in ("ps", "ps=bad", "=8", "ps=-1"):
                result = subprocess.run(
                    [sys.executable, "-I", tool, str(index), "--json", str(report), option, assignment],
                    capture_output=True, text=True,
                )
                assert result.returncode == 2, result.stderr
                assert "expected STAGE=SGPR" in result.stderr
                assert "Traceback" not in result.stderr
                assert not report.exists()


TOOL = os.path.join(ROOT, "tools", "user_sgpr_census.py")


def run_cli(*args):
    # A piped Python child writes in the locale encoding (the ANSI code page on Windows): read it back in that
    # encoding, or a non-ASCII temp path in the error line no longer matches.
    return subprocess.run([sys.executable, "-I", TOOL, *map(str, args)], capture_output=True, text=True,
                          errors="replace")


def assert_one_error_line(result, *fragments):
    assert result.returncode == 2, result.stdout + result.stderr
    assert "Traceback" not in result.stderr
    lines = result.stderr.splitlines()
    assert len(lines) == 1 and lines[0].startswith("ERROR: "), result.stderr
    for fragment in fragments:
        assert fragment in lines[0]
    assert result.stdout == ""


@pytest.mark.parametrize("kind", ["missing", "folder", "empty", "not-json", "not-utf8", "nested-too-deep"])
def test_cli_unreadable_index_is_one_error_line(tmp_path, kind):
    index = tmp_path / "index.json"
    if kind == "folder":
        index.mkdir()
    elif kind != "missing":
        index.write_bytes({"empty": b"", "not-json": b"{not json", "not-utf8": b"\xff\xfe{}",
                           "nested-too-deep": b"[" * 100000}[kind])
    report = tmp_path / "report.json"

    result = run_cli(index, "--json", report)

    assert_one_error_line(result, "cannot read", str(index))
    assert not report.exists()


WRONG_SHAPES = [
    ([], "expected a JSON object of shader -> metadata entries"),
    ("text", "expected a JSON object of shader -> metadata entries"),
    ({"a.ps": []}, "a.ps: each shader entry must be a JSON object"),
    ({"a.ps": {"stage": 3}}, "a.ps: 'stage' must be a string"),
    ({"a.ps": {"stage": "ps", "user_sgpr_reads": {"sgpr": 8}}}, "a.ps: 'user_sgpr_reads' must be a JSON array"),
    ({"a.ps": {"stage": "ps", "user_sgpr_reads": [8]}}, "a.ps: each read needs 'sgpr' and 'kind' fields"),
    ({"a.ps": {"stage": "ps", "user_sgpr_reads": [{"sgpr": 8}]}}, "a.ps: each read needs 'sgpr' and 'kind' fields"),
    ({"a.ps": {"stage": "ps", "user_sgpr_reads": [{"kind": "const32"}]}},
     "a.ps: each read needs 'sgpr' and 'kind' fields"),
    ({"a.ps": {"stage": "ps", "user_sgpr_reads": [{"sgpr": "8", "kind": "ptr32_lo"}]}},
     "a.ps: each read 'sgpr' must be a nonnegative integer"),
    ({"a.ps": {"stage": "ps", "user_sgpr_reads": [{"sgpr": -1, "kind": "ptr32_lo"}]}},
     "a.ps: each read 'sgpr' must be a nonnegative integer"),
    ({"a.ps": {"stage": "ps", "user_sgpr_reads": [{"sgpr": True, "kind": "ptr32_lo"}]}},
     "a.ps: each read 'sgpr' must be a nonnegative integer"),
    ({"a.ps": {"stage": "ps", "user_sgpr_reads": [{"sgpr": 8.5, "kind": "ptr32_lo"}]}},
     "a.ps: each read 'sgpr' must be a nonnegative integer"),
    ({"a.ps": {"stage": "ps", "user_sgpr_reads": [{"sgpr": 8, "kind": None}]}},
     "a.ps: each read kind must be a string"),
    # A good entry first: the whole index is checked, not only its first shader.
    ({**INDEX, "z.ps": {"stage": "ps", "user_sgpr_reads": [{"sgpr": 8, "kind": ["ptr32_lo"]}]}},
     "z.ps: each read kind must be a string"),
]


@pytest.mark.parametrize("document,message", WRONG_SHAPES, ids=[str(i) for i in range(len(WRONG_SHAPES))])
def test_wrongly_shaped_index_is_one_error_line(tmp_path, capsys, document, message):
    index = tmp_path / "index.json"
    report = tmp_path / "report.json"
    index.write_text(json.dumps(document), encoding="utf-8")

    assert usc.main([str(index), "--json", str(report), "--base", "ps=8", "--expect", "ps=8"]) == 2

    captured = capsys.readouterr()
    assert captured.err == f"ERROR: {index}: {message}\n"
    assert captured.out == ""
    assert not report.exists()


def test_cli_wrongly_shaped_index_has_no_traceback(tmp_path):
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"a.ps": {"stage": "ps", "user_sgpr_reads": [{"sgpr": 8}, {"sgpr": 9, "kind": "x"}]}}),
                     encoding="utf-8")

    assert_one_error_line(run_cli(index), "a.ps: each read needs 'sgpr' and 'kind' fields")


def test_entries_without_stage_or_reads_are_still_counted(tmp_path, capsys):
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"a": {}, "b": {"user_sgpr_reads": None}, "c.ps": {"stage": "ps"},
                                 "d.ps": {"stage": "ps", "user_sgpr_reads": reads((8, "ptr32_lo"))}}),
                     encoding="utf-8")
    report = tmp_path / "report.json"

    assert usc.main([str(index), "--json", str(report)]) == 0

    summary = json.loads(report.read_text(encoding="utf-8"))
    assert (summary["?"]["shaders"], summary["ps"]["shaders"]) == (2, 2)
    assert summary["ps"]["lowest_pointer_sgpr"] == {"s8": 1}
    assert capsys.readouterr().err == ""


def test_load_index_accepts_utf8_bom(tmp_path):
    index = tmp_path / "index.json"
    index.write_bytes(b"\xef\xbb\xbf" + json.dumps(INDEX).encode("utf-8"))
    assert usc.load_index(index) == INDEX
    report = tmp_path / "report.json"
    result = run_cli(index, "--json", report)
    assert result.returncode == 0, result.stderr
    assert json.loads(report.read_text(encoding="utf-8"))["ps"]["lowest_pointer_sgpr"] == {"s8": 2, "s9": 1}


@pytest.mark.parametrize("kind", ["missing-folder", "folder", "under-a-file"])
def test_cli_unwritable_report_is_one_error_line(tmp_path, kind):
    index = tmp_path / "index.json"
    index.write_text(json.dumps(INDEX), encoding="utf-8")
    if kind == "missing-folder":
        report = tmp_path / "missing" / "report.json"
    elif kind == "folder":
        report = tmp_path / "report.json"
        report.mkdir()
    else:
        blocker = tmp_path / "blocker"
        blocker.write_bytes(b"keep")
        report = blocker / "report.json"

    result = run_cli(index, "--json", report)

    assert result.returncode == 2, result.stdout + result.stderr
    assert "Traceback" not in result.stderr
    lines = result.stderr.splitlines()
    assert len(lines) == 1 and lines[0].startswith(f"ERROR: cannot write {report}: "), result.stderr
    assert "ps: 3 shaders" in result.stdout          # the census itself ran and was printed
    assert json.loads(index.read_text(encoding="utf-8")) == INDEX
    if kind == "under-a-file":
        assert blocker.read_bytes() == b"keep"


# --pairs (D3D-318): the VS and PS of one technique, named alike apart from the stage token.

def test_technique_names():
    assert usc.technique("!VS_Water_0123abcd_Wrapped", "vs") == "Water"
    assert usc.technique("#PS_Water_89abcdef_Wrapped", "ps") == "Water"
    assert usc.technique("PS_Water", "ps") == "Water"
    assert usc.technique("!VS_Water_0123abcd_Wrapped", "ps") is None   # the token must match the stage
    assert usc.technique("standard_0123abcd_Wrapped", "vs") is None


def root_block(params, stage, omit_invisible):
    """User-SGPR reads of `stage` for root parameters [(kind, {stages that can see it})] packed from s8, one dword
    each: in root order with a hole for an invisible parameter (one block), or without it (per-stage omission)."""
    out, sgpr = [], 8
    for kind, stages in params:
        if stage in stages:
            out.append((sgpr, kind))
        if stage in stages or not omit_invisible:
            sgpr += 1
    return reads(*out)


# A, both stages; B, pixel only; C and D, both stages.
PARAMS = [("ptr32_lo", {"vs", "ps"}), ("const32", {"ps"}), ("ptr32_lo", {"vs", "ps"}), ("const32", {"vs", "ps"})]


def technique_index(omit_invisible):
    return {f"{prefix}{stage.upper()}_Fog_{h}_Wrapped": {"stage": stage, "user_sgpr_reads":
                                                         root_block(PARAMS, stage, omit_invisible)}
            for prefix, stage, h in (("!", "vs", "0123abcd"), ("#", "ps", "89abcdef"))}


def test_one_block_in_root_order_agrees():
    pc = usc.pair_census(technique_index(omit_invisible=False))
    assert pc["techniques_with_vs_and_ps"] == 1 and pc["combinations"] == 1
    # VS s8 A, s10 C, s11 D; PS s8 A, s9 B, s10 C, s11 D: three shared reads, all the same kind.
    assert (pc["shared_same_kind"], pc["shared_different_kind"]) == (3, 0)
    assert pc["combinations_with_a_one_sided_sgpr_below_a_shared_one"] == 1
    assert (pc["above_one_sided_same_kind"], pc["above_one_sided_different_kind"]) == (2, 0)


def test_per_stage_omission_disagrees():
    pc = usc.pair_census(technique_index(omit_invisible=True))
    # VS s8 A, s9 C, s10 D against PS s8 A, s9 B, s10 C, s11 D: C and D sit one SGPR lower in the VS.
    assert (pc["shared_same_kind"], pc["shared_different_kind"]) == (1, 2)


def test_unpaired_shaders_only_feed_the_control():
    index = {
        "!VS_Sky_0123abcd_Wrapped": {"stage": "vs", "user_sgpr_reads": reads((8, "ptr32_lo"), (9, "const32"))},
        "#PS_Rain_89abcdef_Wrapped": {"stage": "ps", "user_sgpr_reads": reads((8, "ptr32_lo"), (9, "ptr32_lo"))},
        "c.cs": {"stage": "cs", "user_sgpr_reads": reads((8, "ptr32_lo"))},
    }
    pc = usc.pair_census(index)
    assert pc["techniques_with_vs_and_ps"] == 0 and pc["combinations"] == 0
    assert pc["control_all_vs_ps"] == {"combinations": 1, "shared_same_kind": 1, "shared_different_kind": 1}


def test_index_saved_with_a_utf8_bom_is_read(tmp_path, capsys):
    """A carve index edited on Windows can carry a BOM; it is still the same JSON."""
    p = tmp_path / "index.json"
    p.write_bytes(b"\xef\xbb\xbf" + json.dumps(INDEX).encode("utf-8"))
    assert usc.main([str(p)]) == 0
    out = capsys.readouterr()
    assert "ps: 3 shaders" in out.out
    assert "ERROR" not in out.err


def test_bom_does_not_hide_broken_json(tmp_path, capsys):
    """Skipping the BOM still leaves a strict JSON parse: a broken index is one ERROR line and exit 2."""
    p = tmp_path / "index.json"
    p.write_bytes(b"\xef\xbb\xbf{\"a.ps\": }")
    assert usc.main([str(p)]) == 2
    assert "ERROR: cannot read" in capsys.readouterr().err


def test_pairs_cli_writes_the_report_and_no_names():
    with tempfile.TemporaryDirectory() as tmp:
        index = Path(tmp, "index.json")
        report = Path(tmp, "report.json")
        index.write_text(json.dumps(technique_index(omit_invisible=False)), encoding="utf-8")
        result = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "tools", "user_sgpr_census.py"),
                                 str(index), "--pairs", "--json", str(report)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert "VS/PS pairs: 1 techniques with both stages" in result.stdout
        assert "Fog" not in result.stdout
        assert json.loads(report.read_text(encoding="utf-8"))["vs_ps_pairs"]["shared_different_kind"] == 0
