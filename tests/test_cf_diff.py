"""Tests for tools/cf_diff.py and tools/cf_ref.py, the FUZZ-8 control-flow fuzzer (no game data, no GPU, no
llvm-mc).

    python -m pytest tests/test_cf_diff.py -v
"""
import os
import random
import subprocess
import sys
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "cf_diff.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))

import cf_diff  # noqa: E402
import cf_ref  # noqa: E402
import rdna2spirv  # noqa: E402

# Condition true for lanes 4 and 20 only (v3 = lane * 5 & 15), so a lane's own bit and the wave's mask disagree.
FEW_LANES = ("eq", "u32", 4, 3)

# Wave32 shapes whose SCC comes from a 32-bit logic op on lane masks. Until 2026-11-22 the translator set that SCC
# from the lane's own bit instead of the whole mask, and every one of these mismatched.
SCC_MASK_CASES = {
    "s_and_b32 + s_cbranch_scc0": [("sccmask", FEW_LANES, "and", "s_cbranch_scc0", [("probe", 11)])],
    "s_xor_b32 + s_cbranch_scc1": [("sccmask", FEW_LANES, "xor", "s_cbranch_scc1", [("probe", 11)])],
    "s_xor_b32 + s_cselect_b32": [("csel", FEW_LANES, "xor", 11)],
    "loop closed by s_andn2_b32 exec + s_cbranch_scc1": [
        ("vloop", (0, 3, True), None, [("probe", 9), ("op", "s_add_u32 s10, s10, 1")])],
}


@pytest.mark.parametrize("count", ["0", "-1"])
def test_empty_program_batch_is_a_usage_error(count, capsys, monkeypatch):
    def unexpected_suite(*args, **kwargs):
        pytest.fail("started a suite with no programs")

    monkeypatch.setattr(cf_diff, "run_suite", unexpected_suite)
    with pytest.raises(SystemExit) as exc:
        cf_diff.main(["--programs", count])
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "--programs must be at least 1" in captured.err
    assert captured.out == ""


def test_a_run_where_every_program_was_skipped_is_not_a_success(capsys, monkeypatch):
    monkeypatch.setattr(cf_diff, "run_case", lambda *args, **kwargs: ("skip", "unsupported form"))
    assert cf_diff.main(["--programs", "2", "--seed", "1"]) == 2
    assert "no program ran: every one was skipped" in capsys.readouterr().out


@pytest.mark.parametrize("statuses, code", [
    (["skip", "skip"], 2),            # nothing was compared
    (["skip", "mismatch"], 1),        # a mismatch outranks "nothing else ran"
    (["match", "skip"], 0),           # one wave compared and matched
])
def test_a_single_program_run_exits_like_the_suite(statuses, code, capsys, monkeypatch):
    results = iter(statuses)
    monkeypatch.setattr(cf_diff, "run_case", lambda *args, **kwargs: (next(results), "detail"))
    assert cf_diff.main(["--program", "3", "--seed", "1", "--wave", "both"]) == code
    assert ("no program ran: every one was skipped" in capsys.readouterr().out) == (code == 2)


def test_generation_is_deterministic():
    assert cf_diff.program_for(5, 3, 32, 3) == cf_diff.program_for(5, 3, 32, 3)
    assert cf_diff.program_for(5, 3, 32, 3) != cf_diff.program_for(5, 4, 32, 3)


def test_branch_offsets_land_on_their_labels():
    e = cf_diff.Emitter(32)
    e.program([("if", FEW_LANES, True, [("probe", 3)]), ("sloop", 2, [("probe", 4)])])
    lines = e.assemble()
    pcs, pc = [], 0
    for _, size in lines:
        pcs.append(pc)
        pc += size
    for (text, size), at in zip(lines, pcs):
        if text.startswith(("s_branch", "s_cbranch")):
            target = at + 4 + 4 * int(text.split()[1])
            assert target in pcs, text


def test_reference_needs_encodings():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "x.s")
        with open(p, "w", encoding="utf-8") as f:
            f.write("; x  code 0x0..0x8\n\ts_branch 0\n\ts_endpgm\n")
        with pytest.raises(cf_ref.ProgramError):
            cf_ref.run(p)


def test_reference_wave32_saveexec_and_restore():
    body = [("if", FEW_LANES, True, [("op", "v_add_nc_u32 v9, 1, v9")])]
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "x.s")
        with open(p, "w", encoding="utf-8") as f:
            f.write(cf_diff.listing_text(cf_diff.build(body, 32), "ps_x"))
        r = cf_ref.run(p, wave=32)
    assert sorted(r["exports"]) == list(range(32))
    assert [lane for lane in range(32) if r["exports"][lane][1] == 1] == [4, 20]
    assert r["exec"] == 0xFFFFFFFF


@pytest.mark.parametrize("wave", (32, 64))
@pytest.mark.parametrize("structured", (True, False), ids=("structured", "dispatch-loop"))
def test_small_fixed_seed_run_is_clean(wave, structured):
    counts, failures = cf_diff.run_suite(6, seed=20261122, waves=(wave,), structured=structured)
    assert counts["harness-error"] == 0, failures
    assert counts["mismatch"] == 0, failures
    assert counts["pass"] == 6, counts


@pytest.mark.parametrize("case", sorted(SCC_MASK_CASES))
@pytest.mark.parametrize("structured", (True, False), ids=("structured", "dispatch-loop"))
def test_scc_of_wave32_mask_ops_is_wave_wide(case, structured):
    status, detail = cf_diff.run_case("ps_scc_mask", SCC_MASK_CASES[case], 32, structured)
    assert status == "match", detail


@pytest.mark.parametrize("case", sorted(SCC_MASK_CASES))
def test_scc_mask_cases_catch_a_per_lane_scc(case, monkeypatch):
    """The cases above are sensitive: with the old per-lane SCC they mismatch."""
    monkeypatch.setattr(rdna2spirv.Gen, "scc_logic", lambda self, a, r: self.scc_nonzero(r))
    status, _ = cf_diff.run_case("ps_scc_mask", SCC_MASK_CASES[case], 32)
    assert status == "mismatch"


def test_minimize_keeps_the_mismatch(monkeypatch):
    monkeypatch.setattr(rdna2spirv.Gen, "scc_logic", lambda self, a, r: self.scc_nonzero(r))
    body = [("probe", 1), ("op", "v_add_nc_u32 v8, 3, v8")] + SCC_MASK_CASES["s_and_b32 + s_cbranch_scc0"]
    small = cf_diff.minimize(body, 32)
    assert len(small) < len(body)
    assert cf_diff.run_case("ps_scc_mask", small, 32)[0] == "mismatch"


def test_self_test_passes():
    assert cf_diff.self_test() is True


def test_no_arguments_prints_usage():
    proc = subprocess.run([sys.executable, "-I", TOOL], capture_output=True, text=True, timeout=5)
    assert proc.returncode == 2
    assert "usage:" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_generator_covers_every_shape():
    kinds = set()

    def walk(nodes):
        for n in nodes:
            kinds.add(n[0])
            for x in n[1:]:
                if isinstance(x, list):
                    walk(x)
    for i in range(200):
        walk(cf_diff.generate(random.Random(i), 32))
    assert {"if", "ifelse", "cmpx", "saveexec", "vccz", "sif", "sloop", "vloop",
            "irr", "sccmask", "csel", "probe"} <= kinds


@pytest.mark.parametrize("op_name", ("andn2", "orn2", "xor", "nand", "nor", "xnor"))
@pytest.mark.parametrize("wave", (32, 64))
def test_saveexec_variants_match(op_name, wave):
    body = [("saveexec", FEW_LANES, op_name, "s_cbranch_scc0", [("probe", 13), ("op", "v_add_nc_u32 v9, 1, v9")])]
    status, detail = cf_diff.run_case(f"ps_saveexec_{op_name}", body, wave, structured=True)
    assert status == "match", detail
