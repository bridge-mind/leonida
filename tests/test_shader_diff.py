"""Tests for tools/shader_diff.py (no game data, no GPU, no llvm-mc).

Run from the repository root:
    python -m pytest tests/test_shader_diff.py -v
"""
import os
import random
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import shader_diff as sd


@pytest.mark.parametrize("seed", [3179, 1, 99])
def test_integer_operands_never_read_float_results(seed):
    # Seed 3179 exhausts the clean register pool before an integer add. Floating
    # results may differ in NaN payload, so their bits cannot feed integer checks.
    # Seeds 1 and 99 check the general invariant without requiring exhaustion.
    lines, writers = sd.generate_program(random.Random(seed), 150)
    float_results = set()
    recovered = False
    for line in lines:
        op = line.split()[0]
        regs = [int(value) for value in re.findall(r"\bv(\d+)\b", line)]
        if not regs:
            continue
        dst, *sources = regs
        if op in sd._FLOAT_OPS:
            float_results.add(dst)
        else:
            assert not (set(sources) & float_results), line
            if op == "v_mov_b32" and float_results == set(sd.SCRATCH):
                recovered = True
            float_results.discard(dst)
    if seed == 3179:
        assert recovered, "fixture must reach the exhausted-pool boundary"
    for reg, (op, index) in writers.items():
        assert lines[index - 1].startswith(f"{op.split('/')[0]} v{reg},")
        assert not any(line.split()[1] == f"v{reg}," for line in lines[index:])
    assert (lines, writers) == sd.generate_program(random.Random(seed), 150)




@pytest.mark.parametrize("seed", [1, 17, 99])
def test_program_generation_is_deterministic(seed):
    lines, writers = sd.generate_program(random.Random(seed), 6)
    assert (lines, writers) == sd.generate_program(random.Random(seed), 6)
    dst = int(lines[-1].split()[1].strip("v,"))
    assert dst in sd.EXPORT_REGS
    assert writers[dst] == ("v_mov_b32/sgpr-copy", len(lines))
    for reg in sd.EXPORT_REGS:
        op, index = writers[reg]
        assert lines[index - 1].startswith(f"{op.split('/')[0]} v{reg},")
        assert not any(line.split()[1] == f"v{reg}," for line in lines[index:])


@pytest.mark.parametrize("seed, n_ops, expected_reg, seed_only", [
    pytest.param(24, 6, 10, True, id="preserve-three-alu-results"),
    pytest.param(3, 18, 11, False, id="replace-oldest-last-alu-write"),
])
def test_scalar_copy_destination_from_write_history(seed, n_ops, expected_reg, seed_only):
    lines, writers = sd.generate_program(random.Random(seed), n_ops)
    history = {f"v{reg},": [] for reg in sd.EXPORT_REGS}
    for index, line in enumerate(lines[:-1], 1):
        op, dst, operands = line.split(maxsplit=2)
        if dst in history:
            history[dst].append((op, index))

    selected = history[f"v{expected_reg},"]
    others = [writes for dst, writes in history.items() if dst != f"v{expected_reg},"]
    for writes in history.values():
        assert writes[0][0] == "v_mov_b32"
        assert all(op != "v_mov_b32" for op, _ in writes[1:])
    assert all(len(writes) >= 2 for writes in others)
    if seed_only:
        assert len(selected) == 1
    else:
        assert len(selected) >= 2
        # Another output received an ALU result earlier but was rewritten later.
        assert any(writes[1][1] < selected[1][1] for writes in others)
        assert all(selected[-1][1] < writes[-1][1] for writes in others)

    op, dst, src = lines[-1].split()
    assert (op, dst) == ("v_mov_b32", f"v{expected_reg},")
    assert src.startswith("s") and src[1:].isdigit()
    assert writers[expected_reg] == ("v_mov_b32/sgpr-copy", len(lines))
    for reg in sd.EXPORT_REGS:
        if reg != expected_reg:
            assert writers[reg] == history[f"v{reg},"][-1]


def test_run_one_observes_scalar_copy(monkeypatch):
    lines, writers = sd.generate_program(random.Random(17), 1)
    assert sd.run_one("ps_scalar_copy", lines, writers) == ("match", "")
    write_listing = sd._write_listing
    opcode, dst, src = lines[-1].split()
    assert opcode == "v_mov_b32" and src.startswith("s")

    def corrupt_scalar_copy(path, body, with_export):
        if with_export:
            # Only the translated listing gets the wrong scalar (seeds are under 256).
            body = body[:-1] + [f"{opcode} {dst} 12345"]
        write_listing(path, body, with_export)

    monkeypatch.setattr(sd, "_write_listing", corrupt_scalar_copy)
    status, detail = sd.run_one("ps_scalar_copy", lines, writers)
    assert status == "mismatch", detail
    reg = int(dst.strip("v,"))
    assert f"lane word {sd.EXPORT_REGS.index(reg)}" in detail
    assert f"last writer v_mov_b32/sgpr-copy at body line {len(lines)} for v{reg}" in detail


def test_compare_pair_flags_known_bad():
    bad = sd.compare_pair([1, 2, 3, 4], [1, 2, 3, 5])
    assert bad is not None and "lane word 3" in bad
    assert sd.compare_pair([1, 2, 3, 4], [1, 2, 3, 4]) is None


@pytest.mark.parametrize("op", ["v_add_f32", "v_add_f32_e32", "v_add_f32_e64"])
def test_compare_pair_both_nan_matches(op):
    qnan = 0x7FC00001
    writers = {reg: (op, 1) for reg in sd.EXPORT_REGS}
    assert sd.compare_pair([qnan] * 4, [0x7FC00002] * 4, writers=writers) is None
    bad = sd.compare_pair([qnan] * 4, [0x3F800000] * 4, writers=writers)
    assert bad is not None and "only one side is NaN" in bad
    assert f"last writer {op} at body line 1" in bad


@pytest.mark.parametrize("changed_word, replacement, width, expected", [
    (1, 0x7FC00000, 4, ("lane word 1", "0xffffffff", "0x7fc00000", "v_mov_b32 at body line 4")),
    (2, 3, 4, ("lane word 2", "v_mov_b32 at body line 5")),
    (None, None, 3, ("shape mismatch: cpu has 4 words, spv has 3",)),
])
def test_run_one_reports_the_first_actual_mismatch(monkeypatch, changed_word, replacement, width, expected):
    lines = ["v_mov_b32 v8, 0x7f800000", "v_mov_b32 v12, 0xff800000",
             "v_add_f32 v8, v8, v12", "v_mov_b32 v9, 0xffffffff",
             "v_mov_b32 v10, 1", "v_mov_b32 v11, 2"]
    writers = {8: ("v_add_f32", 3), 9: ("v_mov_b32", 4), 10: ("v_mov_b32", 5), 11: ("v_mov_b32", 6)}
    assert sd.run_one("ps_mixed_outputs", lines, writers) == ("match", "")
    run_fragment = sd.run_fragment

    def mutate_export(spvasm):
        outputs = run_fragment(spvasm)
        words = list(outputs["%out_mrt0"])
        words[0] = 0x7FC00001  # Different payload for a floating-point NaN is allowed.
        if changed_word is not None:
            words[changed_word] = replacement
        outputs["%out_mrt0"] = words[:width]
        return outputs

    monkeypatch.setattr(sd, "run_fragment", mutate_export)
    status, detail = sd.run_one("ps_mixed_outputs", lines, writers)
    assert status == "mismatch"
    for fragment in expected:
        assert fragment in detail
    assert "v_add_f32 at body line 3" not in detail


def test_unsupported_op_is_a_counted_skip():
    status, detail = sd.run_one(
        "ps_probe_skip", ["v_interp_p1_f32 v8, v9, v10"],
        {8: ("v_interp_p1_f32", 1)})
    assert status == "skip"
    assert detail


def test_small_fixed_seed_run_is_clean():
    counts, first_failure = sd.run_suite(5, seed=20261121, max_ops=4)
    assert counts["mismatch"] == 0, first_failure
    assert counts["harness-error"] == 0, first_failure
    assert counts["pass"] == 5, counts


def test_cli_reports_the_number_of_programs_run(capsys):
    assert sd.main(["--programs", "5", "--limit", "2"]) == 0
    assert capsys.readouterr().out.startswith("2 programs (seed 1): 2 pass,")


def test_self_test_passes():
    assert sd.self_test() is True
