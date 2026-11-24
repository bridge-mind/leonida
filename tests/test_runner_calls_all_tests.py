"""The __main__ runner of every standard-library test file actually runs its top-level test functions.

    python -I tests/test_runner_calls_all_tests.py

tests/test_ci_registration.py sorts tests/test_*.py into CI jobs: a file without a PYTEST_FILE_MARKERS match but
with a __main__ guard runs in the standard-library loop as ``python -I tests/<file>.py``, and pytest never
imports it.  When such a runner forgets to call one of the file's top-level ``def test_*`` functions, those
checks run nowhere while the run still prints ok and exits 0 -- tests/test_cfg_structure.py carried a dead
test_depth() this way (PR #3122 fixes it), and tests/test_feature_scan.py had the sibling bug of swallowing its
only unrouted assert (PR #3051).  This is the runner counterpart of tests/test_ci_registration.py: it reuses
that file's classification straight from the workflow (so it cannot drift from ci.yml) and fails for every
stdlib-mode file whose __main__ block neither names each top-level test function nor dispatches the whole set
through the ``globals()`` loop some runners use.  Intentional non-calls live in RUNNER_SKIPS with their reason,
so a new dead test cannot hide behind a documented one.

Only the tracked test files and the workflow are read; no game files, no build tools, no pytest.
"""
import ast
import os
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_ci_registration as registration  # noqa: E402

# Top-level test functions a stdlib-mode runner deliberately does not call.  Every entry needs a reason; a name
# neither listed here nor reached by the runner fails the check below.
RUNNER_SKIPS = {
    "tests/test_cfg_structure.py": {
        # One-line wrapper around _check_random_graphs(), which the runner already calls; listing it would run
        # the 18,000-graph sweep twice.
        "test_random_graphs",
        # Dead on upstream/main at 16282101: its four S.depth() asserts run nowhere.  PR #3122 adds the call;
        # drop this entry once that merges and the runner calls test_depth() first.
        "test_depth",
    },
    "tests/test_feature_scan.py": {
        # Its __main__ runner calls every test_* through run_tests([... for ... in globals().items() ...]), a
        # comprehension dispatches_tests() does not follow, so each top-level test is listed here.
        "test_feature_scan",
        "test_short_signature",
        "test_part_payload_outside_container_is_unreadable",
        "test_bad_shaders_folder_is_one_line",
        "test_out_inside_corpus_is_a_usage_error",
        "test_duplicate_shader_names_keep_file_stages",
        "test_v_perm_b32_selectors",
        "test_typed_buffers_and_psv0",
        "test_a_test_that_raises_fails_the_run_and_the_rest_still_run",
        "test_a_failed_check_fails_only_its_own_test",
        "test_out_via_directory_alias_into_corpus_is_a_usage_error",
        "test_out_hard_link_to_carved_index_is_a_usage_error",
    },
}


def runner_block(tree):
    """The module's real ``if __name__ == "__main__":`` node, or None (a bare mention satisfies the loop's grep)."""
    for node in tree.body:
        if isinstance(node, ast.If) and "__main__" in ast.dump(node.test) and "__name__" in ast.dump(node.test):
            return node
    return None


def dispatches_tests(block):
    """True when the runner runs its tests through a loop over module globals and calls the loop variable."""
    for node in ast.walk(block):
        if not isinstance(node, ast.For):
            continue
        targets = {t.id for t in ast.walk(node.target) if isinstance(t, ast.Name)}
        if not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                   and n.func.id in ("globals", "locals", "vars") for n in ast.walk(node)):
            continue
        if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in targets
               for n in ast.walk(node)):
            return True
    return False


def uncalled_tests(source):
    """Top-level test functions the file's __main__ runner never runs; None when there is no real guard."""
    tree = ast.parse(source)
    block = runner_block(tree)
    if block is None:
        return None                       # test_ci_registration already fails files the loop runs without a guard
    if dispatches_tests(block):
        return []
    names = {n.id for n in ast.walk(block) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(block) if isinstance(n, ast.Attribute)}
    return [node.name for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test") and node.name not in names]


class RunnerCallsTestsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rules = registration.ci_rules(registration.ROOT)
        cls.files = [f for f in registration.claimed_test_files(registration.ROOT)
                     if registration.job_of(registration.ROOT, f, cls.rules) == "stdlib"]

    def test_the_scan_sees_the_stdlib_files(self):
        # A classification that collapses to nothing must not turn the check below into a silent no-op.
        self.assertGreaterEqual(len(self.files), 100,
                                f"only {len(self.files)} stdlib-mode test files found: {self.files[:5]}")
        self.assertIn("tests/test_cfg_structure.py", self.files)

    def test_every_stdlib_runner_calls_its_test_functions(self):
        dead = []
        for rel in self.files:
            missing = uncalled_tests(registration.read(registration.ROOT, rel))
            if not missing:               # None (no guard) and [] (every test reached) both pass
                continue
            skips = RUNNER_SKIPS.get(rel, ())
            dead += [f"  {rel}: {name}() is defined but its __main__ runner never calls it, so the python -I loop "
                     f"runs it nowhere and still prints ok; call it, or add it to RUNNER_SKIPS with a reason"
                     for name in missing if name not in skips]
        self.assertEqual(dead, [], "dead top-level tests in stdlib-mode runners:\n" + "\n".join(dead))

    def test_a_dropped_call_is_flagged(self):
        # Bite check on a file this PR does not touch: losing one runner call must be reported, not absorbed.
        source = registration.read(registration.ROOT, "tests/test_lane_masks.py")
        self.assertEqual(uncalled_tests(source), [])
        sabotaged = source.replace("    test_straight_line()", "    # dropped for the bite check", 1)
        self.assertEqual(uncalled_tests(sabotaged), ["test_straight_line"])


if __name__ == "__main__":
    unittest.main()
