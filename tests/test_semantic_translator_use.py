"""Every tests/semantic/test_*.py actually runs tools/rdna2spirv.py.

    python -I tests/test_semantic_translator_use.py

A translator regression test that reimplements the answer in Python and never calls the translator
passes whatever the translator does, so it is not a regression test.  This guard reads the test
files (it does not import them) and requires each one to reach ``rdna2spirv``:

* a direct ``import rdna2spirv`` / ``from rdna2spirv import ...``;
* an import of a helper in tests/semantic (``support``, ``regen_fixtures``, another test module)
  that itself reaches the translator, followed transitively;
* a subprocess/runpy test or a meta-test whose own string names ``rdna2spirv.py``.

``ALLOWLIST`` is deliberately short and reviewed: it holds the interpreter self-tests, the fixture
drift check, and the files whose subject is hand-written SPIR-V.  A new entry needs a reason and a
name that says why the file must not call the translator.

This checks that a file *reaches* the translator, not that every assertion depends on it: a file that
imports ``support`` and asserts nothing about its output still passes.  The pattern it is meant to
stop is the one that imports nothing from the repository at all and models the instruction itself.
"""
import ast
import os
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEMANTIC = os.path.join(ROOT, "tests", "semantic")

# Files that legitimately do not invoke the translator.  Keep this list short; every entry is a
# promise that the file tests something other than translation (or tests the translator's input).
ALLOWLIST = {
    "test_spirv_tools_required.py": "meta-test: fails CI when spirv-as/spirv-val are missing, so translator tests cannot skip quietly",
    "test_interp.py": "self-test of spirv_mini_interp.py against hand-written SPIR-V",
    "test_spirv_wave_interp.py": "self-test of spirv_wave_interp.py against hand-written SPIR-V",
    "test_fixtures.py": "checks fixtures/disasm/*.s against regen_fixtures.py; the listing is the subject",
    "test_alu_frexp.py": "hand-written SPIR-V run by spirv_mini_interp.py; no translator output",
    "test_alu_ldexp.py": "hand-written SPIR-V run by spirv_mini_interp.py; no translator output",
}

# Debt from before this guard: files that model the answer instead of running the translator.  They are
# expected to be replaced, so an entry here never has to be kept in step with another PR: fixing the file,
# or deleting it, simply removes it from the debt.  Do not move a file here to silence the guard.
DEBT = {
    "test_valu_div_fixup.py": "private oracle whose rules RDNA2 disagrees with; TEST-545 replaces it",
    "test_alu_cndmask_b64.py": "private oracle merged before this guard (2026-11-22); should translate a listing",
    "test_int_to_float.py": "private oracle merged before this guard (2026-11-22); should translate a listing",
    "test_lds_add_sub_src2_u32.py": "private oracle merged before this guard (2026-11-22); should translate a listing",
    "test_salu_add_sub_co_u32.py": "private oracle merged before this guard (2026-11-22); should translate a listing",
    "test_salu_and_saveexec_b64.py": "private oracle merged before this guard (2026-11-22); should translate a listing",
    "test_salu_andn2_saveexec_b64.py": "private oracle merged before this guard (2026-11-22); should translate a listing",
    "test_salu_nand_nor_saveexec_b64.py": "private oracle merged before this guard (2026-11-22); should translate a listing",
    "test_salu_or_saveexec_b64.py": "private oracle merged before this guard (2026-11-22); should translate a listing",
    "test_salu_orn2_saveexec_b64.py": "private oracle merged before this guard (2026-11-22); should translate a listing",
    "test_salu_xnor_saveexec_b64.py": "private oracle merged before this guard (2026-11-22); should translate a listing",
    "test_salu_xor_saveexec_b64.py": "private oracle merged before this guard (2026-11-22); should translate a listing",
    "test_valu_alignbyte_b32.py": "private oracle merged before this guard (2026-11-22); should translate a listing",
    "test_valu_cvt_f32_f64.py": "private oracle merged before this guard (2026-11-22); should translate a listing",
    "test_valu_dot4_int.py": "private oracle merged before this guard (2026-11-22); should translate a listing",
    "test_valu_dot8_int.py": "private oracle merged before this guard (2026-11-22); should translate a listing",
    "test_valu_fma_f64.py": "private oracle merged before this guard (2026-11-22); should translate a listing",
    "test_valu_fract_f16.py": "private oracle merged before this guard (2026-11-22); should translate a listing",
    "test_valu_fract_f64.py": "private oracle merged before this guard (2026-11-22); should translate a listing",
    "test_valu_frexp_exp_i32_f64.py": "private oracle merged before this guard (2026-11-22); should translate a listing",
    "test_valu_frexp_f64.py": "private oracle merged before this guard (2026-11-22); should translate a listing",
    "test_valu_frexp_mant_f64.py": "private oracle merged before this guard (2026-11-22); should translate a listing",
    "test_valu_lerp_u8.py": "private oracle merged before this guard (2026-11-22); should translate a listing",
    "test_valu_minmax_f64.py": "private oracle merged before this guard (2026-11-22); should translate a listing",
    "test_valu_mul_f64.py": "private oracle merged before this guard (2026-11-22); should translate a listing",
    "test_valu_mul_hi.py": "private oracle merged before this guard (2026-11-22); should translate a listing",
    "test_valu_pack_b32_f16.py": "private oracle merged before this guard (2026-11-22); should translate a listing",
    "test_valu_perm_b32.py": "private oracle merged before this guard (2026-11-22); should translate a listing",
    "test_valu_pk_add_f16.py": "private oracle merged before this guard (2026-11-22); should translate a listing",
    "test_valu_pk_fma_f16.py": "private oracle merged before this guard (2026-11-22); should translate a listing",
    "test_valu_pk_minmax_f16.py": "private oracle merged before this guard (2026-11-22); should translate a listing",
    "test_valu_pk_mul_f16.py": "private oracle merged before this guard (2026-11-22); should translate a listing",
    "test_valu_rcp_f16.py": "private oracle merged before this guard (2026-11-22); should translate a listing",
    "test_valu_rcp_f64.py": "private oracle merged before this guard (2026-11-22); should translate a listing",
    "test_valu_rsq_f16.py": "private oracle merged before this guard (2026-11-22); should translate a listing",
    "test_valu_rsq_f64.py": "private oracle merged before this guard (2026-11-22); should translate a listing",
    "test_valu_sad_u32.py": "private oracle merged before this guard (2026-11-22); should translate a listing",
    "test_valu_sqrt_f16.py": "private oracle merged before this guard (2026-11-22); should translate a listing",
    "test_valu_sqrt_f64.py": "private oracle merged before this guard (2026-11-22); should translate a listing",
    "test_valu_trig_f64.py": "private oracle merged before this guard (2026-11-22); should translate a listing",
}


def _parse(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return ast.parse(f.read(), path)


def _imports(tree):
    """(module, bound name, line) for every import, at any nesting depth, relative ones included.

    The bound name is what the module body can use: ``import support`` binds ``support``,
    ``from support import translate_lines`` binds ``translate_lines``.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, alias.asname or alias.name.split(".")[0], node.lineno
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level > 1:      # from ..x import y: outside tests/semantic
                continue
            for alias in node.names:
                yield module, alias.asname or alias.name, node.lineno


def _used_names(tree):
    """Every name loaded in the module body: an import that is never referenced cannot reach anything."""
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}


def _docstring_literals(tree):
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                found.add(id(first.value))
    return found


def _process_names_translator(tree):
    """The line of a non-docstring string naming rdna2spirv.py, when the file also drives a process."""
    imports_process = any(module.split(".")[0] in ("subprocess", "runpy") for module, _, _ in _imports(tree))
    if not imports_process:
        return None
    docstrings = _docstring_literals(tree)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and "rdna2spirv.py" in node.value and id(node) not in docstrings):
            return node.lineno
    return None


def _first_line(path):
    """A useful line number for a file that never reaches the translator: its first import, else 1."""
    try:
        tree = _parse(path)
    except SyntaxError:
        return 1
    lines = [line for _, _, line in _imports(tree)]
    return min(lines) if lines else 1


def _reaches(path, seen, semantic_dir):
    """True when `path` imports (and uses) the translator, a helper that does, or names it to a subprocess."""
    path = os.path.normpath(path)
    if path in seen:
        return False
    seen = seen | {path}
    tree = _parse(path)
    used = _used_names(tree) | {"*"}      # a star import has no name to match
    if _process_names_translator(tree) is not None:
        return True
    for module, binding, _ in _imports(tree):
        if binding not in used:
            continue                      # imported and never referenced: it cannot reach the translator
        if module.split(".")[0] == "rdna2spirv":
            return True
        helper = os.path.join(semantic_dir, (module.split(".")[0] or binding) + ".py")
        if os.path.isfile(helper) and _reaches(helper, seen, semantic_dir):
            return True
    return False


def offenders(semantic_dir=SEMANTIC, allowlist=None, debt=None):
    """[(relative path, line, reason)] for test files that do not reach rdna2spirv and are not allowed."""
    allowlist = ALLOWLIST if allowlist is None else allowlist
    debt = DEBT if debt is None else debt
    found = []
    for name in sorted(os.listdir(semantic_dir)):
        if not (name.startswith("test_") and name.endswith(".py")):
            continue
        if name in allowlist or name in debt:
            continue
        path = os.path.join(semantic_dir, name)
        if _reaches(path, set(), semantic_dir):
            continue
        found.append((os.path.join("tests", "semantic", name), _first_line(path),
                      "imports no rdna2spirv and no helper that does"))
    return found


class TranslatorUseTests(unittest.TestCase):
    def test_every_semantic_test_reaches_the_translator(self):
        bad = offenders()
        if bad:
            self.fail(
                "translator regression tests that never run tools/rdna2spirv.py:\n"
                + "\n".join(f"  {path}:{line}: {why}" for path, line, why in bad)
                + "\n\nCall support.translate_fixture/translate_lines/translate_listing (or import "
                  "rdna2spirv) so the test observes the translator, or add a reviewed ALLOWLIST entry "
                  "in tests/test_semantic_translator_use.py saying why the file must not.")

    def test_allowlist_entries_are_real_and_still_needed(self):
        """A stale entry would silently exempt a file that now calls the translator."""
        for name, why in ALLOWLIST.items():
            path = os.path.join(SEMANTIC, name)
            self.assertTrue(os.path.isfile(path), f"ALLOWLIST names a missing file: {name}")
            self.assertTrue(why.strip(), f"ALLOWLIST entry {name} has no reason")
            self.assertFalse(_reaches(path, set(), SEMANTIC),
                             f"{name} now reaches rdna2spirv; remove its ALLOWLIST entry")

    def test_detector_on_a_synthetic_tree(self):
        """A guard that accepts everything cannot pass: check direct, helper, process and silent files."""
        with tempfile.TemporaryDirectory() as tmp:
            files = {
                "support.py": "import rdna2spirv\nrdna2spirv.translate\n",
                "test_direct.py": "import rdna2spirv\nrdna2spirv.translate\n",
                "test_helper.py": "import support\nsupport.translate_lines\n",
                "test_nested.py": "def f():\n    import support\n    return support.translate_lines\n",
                "test_process.py": "import subprocess\nCMD = 'tools/rdna2spirv.py'\n",
                "test_silent.py": "import spirv_mini_interp\n",
                "test_unused.py": "import support\n",   # an unused import reaches nothing
            }
            for name, text in files.items():
                with open(os.path.join(tmp, name), "w", encoding="utf-8", newline="\n") as f:
                    f.write(text)
            self.assertEqual(offenders(tmp, allowlist={}, debt={}), [
                (os.path.join("tests", "semantic", "test_silent.py"), 1,
                 "imports no rdna2spirv and no helper that does"),
                (os.path.join("tests", "semantic", "test_unused.py"), 1,
                 "imports no rdna2spirv and no helper that does")])


if __name__ == "__main__":
    unittest.main()
