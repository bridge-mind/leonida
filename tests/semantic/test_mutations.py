"""Mutation tests: sabotage a private copy of tools/ in the ways a real translator regression would, and require
that tests/semantic/test_ps_semantics.py then FAILS for the shader that exercises the sabotaged rule.

Each mutation is an exact-text replacement that must match exactly once (so a refactor that moves the code makes
this test fail loudly instead of silently mutating nothing). The repository's own tools/ is never modified.
Each baseline run on the unmutated copy must pass first.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TRANSLATOR = "rdna2spirv.py"

# (label, file, old text, new text, test that must fail)
MUTATIONS = [
    ("SDWA sext() ignored", TRANSLATOR,
     'f = "OpBitFieldSExtract" if a.get("sext") else "OpBitFieldUExtract"',
     'f = "OpBitFieldUExtract"',
     "test_ps_sdwa"),
    ("SDWA source field offset dropped", TRANSLATOR,
     'self.emit(f"{r} = {f} %uint {v} {self.cu(off)} {self.cu(width)}")',
     'self.emit(f"{r} = {f} %uint {v} {self.cu(0)} {self.cu(width)}")',
     "test_ps_sdwa"),
    ("v_min3/max3 min and max swapped", TRANSLATOR,
     'f = ("S" if op.endswith("i32") else "U") + ("Min" if "min" in op else "Max")',
     'f = ("S" if op.endswith("i32") else "U") + ("Max" if "min" in op else "Min")',
     "test_ps_ops"),
    ("v_bfe_i32 extracted as unsigned", TRANSLATOR,
     'f = "OpBitFieldSExtract" if op.endswith("i32") else "OpBitFieldUExtract"\n            self.emit(f"{r} = {f} %uint {x} {om} {wc}")',
     'f = "OpBitFieldUExtract"\n            self.emit(f"{r} = {f} %uint {x} {om} {wc}")',
     "test_ps_ops"),
    ("v_pk_fma_f16 multiplies src0 by itself", TRANSLATOR,
     'for e_, v_ in zip(ex, (vx, vy, srcs[2])):',
     'for e_, v_ in zip(ex, (vx, vx, srcs[2])):',
     "test_ps_ops"),
    ("v_fma_f16 ignores source op_sel half", TRANSLATOR,
     'self.emit(f"{f} = OpCompositeExtract %float {v2} {1 if hi else 0}")',
     'self.emit(f"{f} = OpCompositeExtract %float {v2} 0")',
     "test_ps_ops2"),
    ("v_subrev_co_ci_u32 drops operand swap", TRANSLATOR,
     '            if op.startswith("v_subrev_co"):\n                x, y = y, x',
     '            if False:\n                x, y = y, x',
     "test_ps_ops2"),
    ("v_cmpx_eq_u64 deactivates active lane", TRANSLATOR,
     '                self.emit(f"{both} = OpLogicalAnd %bool {eq} {self.active_b()}")\n                self.emit(f"OpStore %active {both}")',
     '                self.emit(f"{both} = OpLogicalAnd %bool {eq} {self.active_b()}")\n                self.emit("OpStore %active %false")',
     "test_ps_ops2"),
    ("v_mul_i32_i24 zero-extends instead of sign-extending", TRANSLATOR,
     'self.emit(f"{xs} = OpBitFieldSExtract %uint {x} {self.cu(0)} {self.cu(24)}")',
     'self.emit(f"{xs} = OpBitFieldUExtract %uint {x} {self.cu(0)} {self.cu(24)}")',
     "test_ps_ops3"),
    ("v_ffbh_i32 off by one from MSB", TRANSLATOR,
     'self.emit(f"{sub} = OpISub %int {self.as_int(self.cu(31))} {f}")',
     'self.emit(f"{sub} = OpISub %int {self.as_int(self.cu(30))} {f}")',
     "test_ps_ops3"),
    ("v_pk_lshlrev_b16 reads low shift amount for both halves", TRANSLATOR,
     'self.emit(f"{half} = OpShiftRightLogical %uint {sh_src} {self.cu(16)}")',
     'self.emit(f"{half} = OpShiftRightLogical %uint {sh_src} {self.cu(0)}")',
     "test_ps_ops4"),
    ("v_sin/v_cos omits turns-to-radians 2*pi factor", TRANSLATOR,
     'self.emit(f"{arg} = OpFMul %float {t} {self.fconst(6.283185307179586)}")',
     'self.emit(f"{arg} = OpFMul %float {t} {self.fconst(1.0)}")',
     "test_ps_ops4"),
    ("v_mul_legacy_f32 drops zero-times-anything guard", TRANSLATOR,
     # since SHADER-340 v_mul_legacy_f32 flushes its inputs and product; the guard is the OpSelect on either zero
     '            self.emit(f"{either} = OpLogicalOr %bool {zx} {zy}")\n'
     '            self.emit(f"{r} = OpSelect %float {either} %f0 {self.flush_f32(m)}")',
     '            self.emit(f"{either} = OpLogicalOr %bool {zx} {zy}")\n'
     '            self.emit(f"{r} = OpCopyObject %float {self.flush_f32(m)}")',
     "test_ps_legacy_mul"),
]


def run_semantics(tools_dir):
    # The child finds test_ps_semantics through PYTHONPATH, not cwd=HERE: Windows refuses a working directory longer
    # than 258 characters (WinError 267), which a checkout in a deep folder reaches.
    path = os.pathsep.join(filter(None, [HERE, os.environ.get("PYTHONPATH")]))
    env = dict(os.environ, LEONIDA_TOOLS_DIR=tools_dir, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=path)
    p = subprocess.run([sys.executable, "-m", "unittest", "test_ps_semantics", "-v"], env=env,
                       capture_output=True, text=True, timeout=300)
    return p.returncode, p.stdout + p.stderr


def failed_tests(output):
    return set(re.findall(r"^(?:FAIL|ERROR): (test_\w+)", output, re.M))


def apply_mutation(path, old, new):
    """Replace old with new in path exactly once, keeping the file's line endings.

    A checkout made before .gitattributes pinned `eol=lf` (or a branch applied through a CRLF-translating
    editor) stores tools/ with CRLF line endings, while every anchor here is written with LF. Matching the
    anchors against the raw bytes reports the multiline anchors as missing, so normalize both sides to LF
    and restore the file's original line-ending style on the way out.
    """
    with open(path, encoding="utf-8", newline="") as f:
        raw = f.read()
    sep = "\r\n" if "\r\n" in raw else "\n"
    text = raw.replace("\r\n", "\n")
    if text.count(old) != 1:
        raise AssertionError(f"mutation anchor must occur exactly once: {old!r}")
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text.replace(old, new).replace("\n", sep))


class MutationTests(unittest.TestCase):
    def make_copy(self, tmp):
        dst = os.path.join(tmp, "tools")
        shutil.copytree(os.path.join(ROOT, "tools"), dst, ignore=shutil.ignore_patterns("__pycache__", "rpf9dump"))
        return dst

    def test_baseline_copy_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, out = run_semantics(self.make_copy(tmp))
            self.assertEqual(rc, 0, out)

    def test_each_mutation_is_caught(self):
        for label, fname, old, new, expect in MUTATIONS:
            with self.subTest(mutation=label), tempfile.TemporaryDirectory() as tmp:
                tools = self.make_copy(tmp)
                path = os.path.join(tools, fname)
                apply_mutation(path, old, new)
                rc, out = run_semantics(tools)
                self.assertNotEqual(rc, 0, f"mutation not detected: {label}\n{out}")
                self.assertIn(expect, failed_tests(out), f"{label}: expected {expect} to fail\n{out}")

    def test_anchors_match_crlf_translator_copies(self):
        # Regression: on a CRLF checkout of tools/ the four multiline anchors (v_bfe_i32, v_subrev_co_ci_u32,
        # v_cmpx_eq_u64, v_mul_legacy_f32) do not occur in the raw bytes and test_each_mutation_is_caught
        # stops before any mutation runs. apply_mutation must find every anchor in a CRLF copy, mutate it
        # once, keep the file on CRLF and leave it valid Python.
        with open(os.path.join(ROOT, "tools", TRANSLATOR), encoding="utf-8", newline="") as f:
            text = f.read().replace("\r\n", "\n")
        crlf_text = text.replace("\n", "\r\n")
        for label, fname, old, new, expect in MUTATIONS:
            if fname != TRANSLATOR:
                continue
            with self.subTest(mutation=label), tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, TRANSLATOR)
                with open(path, "w", encoding="utf-8", newline="") as f:
                    f.write(crlf_text)
                apply_mutation(path, old, new)
                with open(path, encoding="utf-8", newline="") as f:
                    mutated = f.read()
                # the copy stays on CRLF: no bare LF survives, and the injected mutation text followed suit
                self.assertNotIn("\n", mutated.replace("\r\n", ""))
                self.assertIn(new.replace("\n", "\r\n"), mutated)
                self.assertNotIn(old.replace("\n", "\r\n"), mutated)
                compile(mutated, path, "exec")


if __name__ == "__main__":
    unittest.main()
