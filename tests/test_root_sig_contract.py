"""Contract tests for tools/root_sig_contract.py (no game data, no GPU).

Run from the repository root:  python -m pytest tests/test_root_sig_contract.py
Everything is hand-built: synthetic root signatures and synthetic index
entries, so these run on every CI leg including Windows without llvm-mc.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import root_sig_contract as rsc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "root_sig_contract.py")

ROOT_SIG = {
    "version": "1.1",
    "flags": 0,
    "params": [
        {"type": "constants", "count": 4, "visibility": "all"},
        {"type": "cbv", "visibility": "vertex"},
        {"type": "table", "ranges": [{"kind": "srv", "count": 8}],
         "visibility": "pixel"},
    ],
}


def test_pack_layout_order_and_provenance():
    dwords, error = rsc.pack_layout(ROOT_SIG)
    assert error == ""
    assert len(dwords) == 8
    assert [d["param"] for d in dwords] == [0, 0, 0, 0, 1, 1, 2, 2]
    assert [d["sub"] for d in dwords] == [0, 1, 2, 3, 0, 1, 0, 1]
    assert dwords[0]["ptype"] == "constants"
    assert dwords[4]["ptype"] == "descriptor"
    assert dwords[6]["ptype"] == "table"


def test_pack_layout_static_samplers_free():
    sig = dict(ROOT_SIG, static_samplers=[{"filter": 21}])
    dwords, error = rsc.pack_layout(sig)
    assert error == "" and len(dwords) == 8


def test_pack_layout_spill_refused():
    sig = {"version": "1.1", "flags": 0,
           "params": [{"type": "cbv"} for _ in range(17)]}
    dwords, error = rsc.pack_layout(sig)
    assert dwords == [] and "spill" in error


def test_pack_layout_cost_limit():
    sig = {"version": "1.1", "flags": 0,
           "params": [{"type": "constants", "count": 65}]}
    dwords, error = rsc.pack_layout(sig)
    assert dwords == [] and "64-dword" in error


def test_pack_layout_unknown_type():
    sig = {"version": "1.1", "flags": 0, "params": [{"type": "ufo"}]}
    dwords, error = rsc.pack_layout(sig)
    assert dwords == [] and "unknown type" in error


def test_cli_refuses_boolean_root_constant_count(tmp_path):
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps({"synthetic": {
        "user_sgpr_reads": [{"sgpr": 0, "kind": "const32"}],
    }}), encoding="utf-8")
    sig_path = tmp_path / "sig.json"
    sig_path.write_text(json.dumps({
        "params": [{"type": "constants", "count": True}],
    }), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-I", TOOL, str(index_path), "--rootsig", str(sig_path)],
        capture_output=True, text=True)
    assert result.returncode == 1
    assert "root signature refused: param 0: constants need an integer count >= 1" in result.stdout


def test_reconcile_compatible():
    dwords, _ = rsc.pack_layout(ROOT_SIG)
    reads = [{"sgpr": 0, "kind": "const32"},
             {"sgpr": 4, "kind": "ptr64_lo"},
             {"sgpr": 5, "kind": "ptr64_hi"},
             {"sgpr": 6, "kind": "tsharp"}]
    assert rsc.reconcile(reads, dwords) == []


@pytest.mark.parametrize(("base", "model"), [(0, []), (4, ["--model", "console"])])
def test_cli_checks_explicit_pointer_pairs(tmp_path, base, model):
    sig_path = tmp_path / "sig.json"
    sig_path.write_text(json.dumps({"params": [{"type": "cbv"}, {"type": "srv"}]}),
                        encoding="utf-8", newline="\n")
    index_path = tmp_path / "index.json"
    for hi, status in ((base + 1, 0), (base + 3, 1)):
        pair = [base, hi]
        reads = [{"sgpr": base, "kind": "ptr64_lo", "pair": pair},
                 {"sgpr": hi, "kind": "ptr64_hi", "pair": pair}]
        index_path.write_text(json.dumps({"fixture": {"stage": "cs", "user_sgpr_reads": reads}}),
                              encoding="utf-8", newline="\n")
        result = subprocess.run([sys.executable, "-I", TOOL, index_path, "--rootsig", sig_path, *model],
                                capture_output=True, text=True)
        assert result.returncode == status, result.stdout + result.stderr
        if status:
            assert f"pair {pair} does not name both halves of one root address" in result.stdout


@pytest.mark.parametrize(("sgpr", "kind", "needle"), [
    (0, "ptr64_lo", "incompatible"),  # s0 is a constant dword, not a VA
    (31, "const32", "8 dwords"),  # layout ends at s7
    (4, "ptr64_hi", "paired lo"),  # s4 is sub 0 of the CBV pair
    (5, "ptr64_lo", "high half"),  # s5 is the high half of the CBV pair
    (7, "ptr64_lo", "high half"),  # s7 is the high half of the table pair
    (0, "nope", "unknown read kind"),
    (-1, "tsharp", "8 dwords"),  # negative index must not alias last table dword
    (-9, "tsharp", "8 dwords"),  # beyond layout must not crash
])
def test_reconcile_contradictions(sgpr, kind, needle):
    dwords, _ = rsc.pack_layout(ROOT_SIG)
    issues = rsc.reconcile([{"sgpr": sgpr, "kind": kind}], dwords)
    assert len(issues) == 1 and needle in issues[0]


def test_cluster_groups_shapes():
    index = {
        "a": {"user_sgpr_reads": [{"sgpr": 0, "kind": "const32"}]},
        "b": {"user_sgpr_reads": [{"sgpr": 0, "kind": "const32"}]},
        "c": {"user_sgpr_reads": [{"sgpr": 4, "kind": "ptr64_lo"}]},
    }
    ranked = rsc.cluster(index)
    assert ranked[0][1] == 2 and ranked[1][1] == 1


def test_cluster_ties_and_example_names_are_stable():
    index = {
        "z_const": {"user_sgpr_reads": [{"sgpr": 0, "kind": "const32"}]},
        "y_ptr": {"user_sgpr_reads": [{"sgpr": 2, "kind": "ptr64_lo"}]},
        "a_const": {"user_sgpr_reads": [{"sgpr": 0, "kind": "const32"}]},
        "b_ptr": {"user_sgpr_reads": [{"sgpr": 2, "kind": "ptr64_lo"}]},
    }
    ranked = rsc.cluster(index)
    assert ranked == rsc.cluster(dict(reversed(list(index.items()))))
    assert [row[2] for row in ranked] == [["a_const", "z_const"], ["b_ptr", "y_ptr"]]


def test_cli_hard_fail_on_contradiction(tmp_path):
    index = {"bad": {"user_sgpr_reads": [{"sgpr": 31, "kind": "const32"}]}}
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    sig_path = tmp_path / "sig.json"
    sig_path.write_text(json.dumps(ROOT_SIG), encoding="utf-8")
    r = subprocess.run(
        [sys.executable, "-I", TOOL, str(index_path),
         "--rootsig", str(sig_path)],
        capture_output=True, text=True)
    assert r.returncode == 1
    assert "CONTRADICTION bad" in r.stdout


@pytest.mark.parametrize(("sgpr", "status", "diagnostic"), [
    (4, 0, "1 compatible"),
    (5, 1, "s5 (ptr32_lo) reads the high half of the address"),
    (6, 0, "1 compatible"),
    (7, 1, "s7 (ptr32_lo) reads the high half of the address"),
])
def test_cli_32bit_pointer_must_use_low_address_slot(
    tmp_path: Path, sgpr: int, status: int, diagnostic: str,
) -> None:
    index = {"shader": {"user_sgpr_reads": [{"sgpr": sgpr, "kind": "ptr32_lo"}]}}
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    sig_path = tmp_path / "sig.json"
    sig_path.write_text(json.dumps(ROOT_SIG), encoding="utf-8")
    result = subprocess.run([sys.executable, "-I", TOOL, str(index_path), "--rootsig", str(sig_path)],
                            capture_output=True, text=True)
    assert result.returncode == status, result.stdout + result.stderr
    assert diagnostic in result.stdout


@pytest.mark.parametrize("bad_index", ["not json", "[]", '{"a": 1}'])
def test_bad_index_shape_is_one_error_not_a_traceback(tmp_path, bad_index):
    index_path = tmp_path / "index.json"
    index_path.write_text(bad_index, encoding="utf-8")
    sig_path = tmp_path / "sig.json"
    sig_path.write_text(json.dumps(ROOT_SIG), encoding="utf-8")
    r = subprocess.run([sys.executable, "-I", TOOL, str(index_path), "--rootsig", str(sig_path)],
                       capture_output=True, text=True)
    assert r.returncode == 2
    assert "ERROR:" in r.stderr
    assert "Traceback" not in r.stderr


@pytest.mark.parametrize("bad_sig", ["not json", "[]"])
def test_bad_rootsig_shape_is_one_error_not_a_traceback(tmp_path, bad_sig):
    index_path = tmp_path / "index.json"
    index_path.write_text("{}", encoding="utf-8")
    sig_path = tmp_path / "sig.json"
    sig_path.write_text(bad_sig, encoding="utf-8")
    r = subprocess.run([sys.executable, "-I", TOOL, str(index_path), "--rootsig", str(sig_path)],
                       capture_output=True, text=True)
    assert r.returncode == 2
    assert "ERROR:" in r.stderr
    assert "Traceback" not in r.stderr


def test_missing_files_are_one_error(tmp_path):
    r = subprocess.run([sys.executable, "-I", TOOL, str(tmp_path / "missing_index.json"),
                        "--rootsig", str(tmp_path / "missing_sig.json")],
                       capture_output=True, text=True)
    assert r.returncode == 2
    assert "ERROR: cannot read" in r.stderr
    assert "Traceback" not in r.stderr


@pytest.mark.parametrize("bad_params", ["abc", {"type": "cbv"}, [1], ["cbv"]])
def test_bad_param_shapes_are_one_error_not_a_traceback(tmp_path, bad_params):
    index_path = tmp_path / "index.json"
    index_path.write_text("{}", encoding="utf-8")
    sig_path = tmp_path / "sig.json"
    sig_path.write_text(json.dumps({"params": bad_params}), encoding="utf-8")
    r = subprocess.run([sys.executable, "-I", TOOL, str(index_path), "--rootsig", str(sig_path)],
                       capture_output=True, text=True)
    assert r.returncode == 2
    assert "ERROR:" in r.stderr
    assert "Traceback" not in r.stderr


@pytest.mark.parametrize("bad_reads", [
    3, {"sgpr": 0}, ["x"], [{"kind": "const32"}], [{"sgpr": 0}],
    [{"sgpr": 0, "kind": ["const32"]}], [{"sgpr": 0, "kind": {"a": 1}}],
])
def test_bad_read_shapes_are_one_error_not_a_traceback(tmp_path, bad_reads):
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps({"s": {"user_sgpr_reads": bad_reads}}), encoding="utf-8")
    sig_path = tmp_path / "sig.json"
    sig_path.write_text(json.dumps(ROOT_SIG), encoding="utf-8")
    r = subprocess.run([sys.executable, "-I", TOOL, str(index_path), "--rootsig", str(sig_path)],
                       capture_output=True, text=True)
    assert r.returncode == 2
    assert "ERROR:" in r.stderr
    assert "Traceback" not in r.stderr


def test_cli_hard_fail_on_negative_sgpr(tmp_path):
    index = {"bad": {"user_sgpr_reads": [{"sgpr": -1, "kind": "tsharp"}]}}
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    sig_path = tmp_path / "sig.json"
    sig_path.write_text(json.dumps(ROOT_SIG), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-I", TOOL, str(index_path),
         "--rootsig", str(sig_path)],
        capture_output=True, text=True)
    assert result.returncode == 1
    assert "CONTRADICTION bad" in result.stdout
    assert "s-1 read but layout has 8 dwords" in result.stdout


@pytest.mark.parametrize("sgpr", [True, False, 1.0, "1", None, [], {}])
def test_reconcile_rejects_noninteger_sgpr(sgpr):
    dwords, _ = rsc.pack_layout(ROOT_SIG)
    # Including a real index catches bool/int and float/int key collisions.
    reads = [{"sgpr": 1, "kind": "const32"},
             {"sgpr": sgpr, "kind": "const32"}]
    assert rsc.reconcile(reads, dwords) == [
        f"invalid SGPR index {sgpr!r}: expected an integer"]


def test_cluster_preserves_invalid_index_distinctions():
    index = {
        "valid": {"user_sgpr_reads": [{"sgpr": 1, "kind": "const32"}]},
        "invalid": {"user_sgpr_reads": [
            {"sgpr": 1, "kind": "const32"},
            {"sgpr": True, "kind": "const32"},
            {"sgpr": "1", "kind": "const32"},
            {"sgpr": [], "kind": "const32"}]},
    }
    ranked = rsc.cluster(index)
    assert len(ranked) == 2
    assert sum(count for _, count, _ in ranked) == 2

    # Valid SGPRs must keep numeric order even when an invalid index is present.
    index["mixed"] = {"user_sgpr_reads": [
        {"sgpr": 10, "kind": "const32"},
        {"sgpr": "bad", "kind": "const32"},
        {"sgpr": 2, "kind": "const32"}]}
    mixed_shape = next(shape for shape, _, names in rsc.cluster(index)
                       if names == ["mixed"])
    assert mixed_shape == ((2, "const32"), (10, "const32"),
                           ("invalid:'bad'", "const32"))


def test_cli_rejects_noninteger_sgprs_with_cluster(tmp_path):
    index = {"bad": {"user_sgpr_reads": [
        {"sgpr": 1, "kind": "const32"},
        {"sgpr": True, "kind": "const32"},
        {"sgpr": "1", "kind": "const32"},
        {"sgpr": 1.0, "kind": "const32"},
        {"sgpr": [], "kind": "const32"}]}}
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    sig_path = tmp_path / "sig.json"
    sig_path.write_text(json.dumps(ROOT_SIG), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-I", TOOL, str(index_path),
         "--rootsig", str(sig_path), "--cluster"],
        capture_output=True, text=True)
    assert result.returncode == 1
    assert "CONTRADICTION bad" in result.stdout
    assert "invalid SGPR index" in result.stdout
    assert "read-set shapes" in result.stdout
    assert "Traceback" not in result.stderr


# --model console: rt::rootsig::pack_stage's layout (D3D-448). The package's FidelityFX compute signatures have this
# shape (two tables, then four root CBVs); their programs read the tables at s4/s5 as 32-bit pointers and copy each
# root CBV's 64-bit VA as two dwords. Hand-written here: shapes and register numbers only.
FSR_SHAPE = {"version": "1.1", "flags": 0, "params": [
    {"type": "table", "ranges": [{"kind": "srv", "count": 1}]},
    {"type": "table", "ranges": [{"kind": "uav", "count": 1}]},
    {"type": "cbv"}, {"type": "cbv"}, {"type": "cbv"}, {"type": "cbv"}]}
FSR_READS = [{"sgpr": 4, "kind": "ptr32_lo"}, {"sgpr": 5, "kind": "ptr32_lo"},
             {"sgpr": 6, "kind": "const32"}, {"sgpr": 7, "kind": "const32"}]


def test_console_layout_compute():
    dwords, error = rsc.pack_layout(FSR_SHAPE, "cs")
    assert error == ""
    assert len(dwords) == 14  # s0..s3 driver, then 10 dwords: the programs' USER_SGPR count
    assert [d["ptype"] for d in dwords[:4]] == ["driver"] * 4
    assert [(d["param"], d["sub"]) for d in dwords[4:]] == [
        (0, 0), (1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (4, 0), (4, 1), (5, 0), (5, 1)]


def test_console_layout_stage_bases():
    for stage, base in (("ps", 8), ("vs", 8), ("gs", 8), ("ms", 16)):
        dwords, error = rsc.pack_layout(ROOT_SIG, stage)
        assert error == "" and len(dwords) == base + 4 + 2 + 1, stage   # constants 4, cbv 2, table 1
    dwords, error = rsc.pack_layout(ROOT_SIG, "hs")
    assert dwords == [] and "no console base" in error


def test_console_layout_spill_refused():
    sig = {"version": "1.1", "flags": 0, "params": [{"type": "table"} for _ in range(17)]}
    assert rsc.pack_layout(sig, "ms")[1] != ""                     # 17 one-dword tables past s31
    sig["params"].pop()
    assert rsc.pack_layout(sig, "ms")[1] == ""                     # 16 fill s16..s31
    dwords, error = rsc.pack_layout({"params": [{"type": "constants", "count": 29}]}, "cs")
    assert dwords == [] and "param 0" in error and "spill" in error


def test_console_reconcile_fsr_reads():
    console, _ = rsc.pack_layout(FSR_SHAPE, "cs")
    assert rsc.reconcile(FSR_READS, console) == []
    s0, _ = rsc.pack_layout(FSR_SHAPE)
    issues = rsc.reconcile(FSR_READS, s0)
    assert any("s5 (ptr32_lo) reads the high half" in i for i in issues)


def test_whole_address_copy_needs_both_halves():
    console, _ = rsc.pack_layout(FSR_SHAPE, "cs")
    one_half = [{"sgpr": 6, "kind": "const32"}]
    assert any("incompatible" in i for i in rsc.reconcile(one_half, console))


def test_driver_block_reads():
    console, _ = rsc.pack_layout(FSR_SHAPE, "cs")
    assert rsc.reconcile([{"sgpr": 1, "kind": "const32"}], console) == []          # a system value
    issues = rsc.reconcile([{"sgpr": 0, "kind": "ptr64_lo"}], console)
    assert any("driver block" in i for i in issues)                                   # a spill pointer


def test_cli_console_model_uses_each_shaders_stage(tmp_path):
    index = {"a.cs": {"stage": "cs", "user_sgpr_reads": FSR_READS},
             "b.ps": {"stage": "ps", "user_sgpr_reads": [{"sgpr": 8, "kind": "ptr32_lo"}]}}
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    sig_path = tmp_path / "sig.json"
    sig_path.write_text(json.dumps(FSR_SHAPE), encoding="utf-8")
    console = subprocess.run([sys.executable, "-I", TOOL, str(index_path), "--rootsig", str(sig_path),
                              "--model", "console"], capture_output=True, text=True)
    assert console.returncode == 0, console.stdout + console.stderr
    assert "2 shaders: 2 compatible, 0 contradictory" in console.stdout
    plain = subprocess.run([sys.executable, "-I", TOOL, str(index_path), "--rootsig", str(sig_path)],
                           capture_output=True, text=True)
    assert plain.returncode == 1 and "CONTRADICTION" in plain.stdout
