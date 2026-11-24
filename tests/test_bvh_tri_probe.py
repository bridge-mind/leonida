"""CPU-only unit tests for tools/bvh_tri_probe.py: case generation, the rule, the kernel text and the GPU path with a
stand-in oracle. No GPU, no assembler.

    python -I tests/test_bvh_tri_probe.py
"""
import hashlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from contextlib import redirect_stderr, redirect_stdout

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(REPO, "tools")
PROBE = os.path.join(TOOLS, "bvh_tri_probe.py")

sys.dont_write_bytecode = True
for _p in (TOOLS, os.path.join(REPO, "tests", "semantic")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bvh_tri_probe as probe  # noqa: E402
import test_image_bvh_tri_hw as table  # noqa: E402  (the measured words; a module, so no test class is collected twice)
from hw_oracle import asm as hw_asm, runner as hw_runner  # noqa: E402

# sha256 of the lines of five kernels: the text the RX 6800 ran (the module the in-tree asm.module_text builds from
# these lines hashed to the reviewed specs' module_sha256). A change to a kernel's text must be deliberate.
PINNED = {
    "tri_g1a": "048d468da740e304d01924b1ea7a19441ae3c8be7804b2bf6e75a6f64d58ca12",
    "tri_g3b": "1cdababaf441f0f9bfdd7b5dbcdd4c4088bbe5f134e404bf2ac93607ef19013d",
    "tri_lanes_w64": "bf80be19903a096d0bbc3ac74e177b64f0202fb9ad66ac26cc4a9adda4f7def4",
    "tri_t1": "e6fe057700dd38cb26fa1f4fde137cb385720f09c9c63a1746ef21df9a54e536",
    "tri_t8": "6b2dfb6dd701898ba9990c31b14f077a52024a151c255a1cd04444c5cdc48c4d",
}


def kernels():
    return {k.label: k for k in probe.build_kernels()}


class FakeOracle:
    """Returns, for each kernel, the output buffer the hardware wrote in the measurement, rebuilt from the words of
    tests/semantic/test_image_bvh_tri_hw.py (single-lane kernels; a lane kernel gets the rule's words)."""
    arch = "gfx1030"

    def __init__(self, ks, plant=None, prefill_only=False, no_done=False):
        self.ks, self.plant, self.prefill_only, self.no_done = ks, plant, prefill_only, no_done
        self.words = {(r[0], r[1]): list(r[6]) for r in table.ROWS}
        self.calls = []

    def run(self, programs, words):
        self.calls.append(len(programs))
        k = self.ks[programs[0].lines[0].split(": ", 1)[1]]
        out = [probe.PREFILL] * words
        if self.prefill_only:
            return [out]
        for i, c in enumerate(k.cases):
            for ln in range(k.threads):
                w = self.words.get((k.label, c["label"])) if k.threads == 1 else \
                    probe.rule_words(k.slots[c["slot"]], c["type"], k.lane_ray(c, ln))
                if self.plant == (k.label, i):
                    w = [w[0] ^ 1] + w[1:]
                b = (k.L["res"] + 256 * ln * (k.threads == 64) + 32 * i) // 4
                out[b:b + 8] = w + w
        ctl = k.L["ctl"] // 4
        out[ctl + 2] = 0 if self.no_done else probe.DONE | k.uid
        out[ctl + 3] = 0
        return [out]


class Generation(unittest.TestCase):
    def test_self_test(self):
        proc = subprocess.run([sys.executable, "-I", PROBE, "--self-test"], capture_output=True, text=True,
                              check=False)
        self.assertEqual(proc.returncode, 0, proc.stderr or proc.stdout)
        self.assertIn("self-test OK", proc.stdout)

    def test_cases_are_the_measured_rows(self):
        ks = kernels()
        got = []
        for k in ks.values():
            if k.threads == 1:
                got += [(k.label, c["label"], k.slots[c["slot"]], c["type"], c["ray"]) for c in k.cases]
        want = [(kn, lb, table.NODES[n] + [w15], t, list(ray)) for kn, lb, n, w15, t, ray, _ in table.ROWS]
        self.assertEqual(len(got), 564)
        self.assertEqual(got, want)
        for kn, lb, n, w15, t, ray, _ in table.LANE_ROWS:
            label, lane = lb.rsplit(", lane ", 1)
            k = ks[kn]
            c = next(c for c in k.cases if c["label"] == label)
            self.assertEqual((k.slots[c["slot"]], c["type"], k.lane_ray(c, int(lane))),
                             (table.NODES[n] + [w15], t, list(ray)), lb)

    def test_rule_gives_the_measured_words(self):
        for kn, lb, n, w15, t, ray, hw in table.ROWS + table.LANE_ROWS:
            with self.subTest(kernel=kn, case=lb):
                self.assertEqual(probe.rule_words(table.NODES[n] + [w15], t, list(ray)), list(hw))

    def test_translator_against_the_rule(self):
        # the full translator check is tests/semantic/test_image_bvh_tri_hw.py; here one kernel through the tool's path
        k = kernels()["tri_g4a"]
        rows, bad = probe.compare([k], probe.run_translator([k]), None)
        self.assertEqual((len(rows), bad), (16, 0))
        self.assertTrue(all("tr32" in r and "tr64" in r for r in rows))


class KernelText(unittest.TestCase):
    def test_pinned_kernels(self):
        ks = kernels()
        for label, sha in PINNED.items():
            self.assertEqual(hashlib.sha256("\n".join(ks[label].lines()).encode("utf-8")).hexdigest(), sha, label)

    def test_safety_rules(self):
        for k in kernels().values():
            with self.subTest(kernel=k.label):
                lines = k.lines()
                self.assertTrue(probe.check_kernel(k))
                bvh = [ln for ln in lines if ln.startswith("image_bvh")]
                self.assertEqual(len(bvh), 2 * len(k.cases))
                # every 32-bit form names a node of a placed slot, type 0..3, through the node region's base
                ids = [int(ln.split(", 0x")[1], 16) for ln in lines if ln.startswith("v_mov_b32 v2, 0x")]
                self.assertEqual(ids, [k.node32(c) for c in k.cases])
                for i in ids:
                    self.assertIn(i & 7, range(4))
                    self.assertIn(((i & ~7) << 3) // probe.SLOT, k.slots)
                self.assertLessEqual(k.L["node"] + probe.SLOT * (max(k.slots) + 1), k.L["node_end"])
                self.assertLess(k.L["node_end"], 4 * k.L["words"])
                self.assertEqual(sorted({ln for ln in lines if ln.startswith(("s_mov_b32 s6", "s_mov_b32 s7"))}),
                                 ["s_mov_b32 s6, 0xffffffff", "s_mov_b32 s7, 0x81000000"])
                self.assertLess(lines.index(f"s_cbranch_scc1 .Lbad{k.uid}"), lines.index(bvh[0]))
                prog = k.program(hw_asm)
                self.assertTrue(prog.raw)
                self.assertEqual((prog.words, prog.threads, prog.wave), (k.L["words"], k.threads, k.wave))

    def test_illegal_nodes_and_cases_are_refused(self):
        k = probe.Kernel("t")
        s = k.add_node(probe.leaf())
        for ntype in (4, 5, 6, 7):
            with self.assertRaises(ValueError):
                k.add_case("box or unknown node", s, ntype, [0] * 10)
        with self.assertRaises(ValueError):
            k.add_case("a slot with no node", s + 1, 0, [0] * 10)
        with self.assertRaises(ValueError):
            k.add_node([0] * 32)                 # a box32-sized node
        for _ in range(probe.NSLOTS - 1):
            k.add_node(probe.leaf())
        with self.assertRaises(ValueError):
            k.add_node(probe.leaf())              # past the node region

    def test_check_kernel_catches_a_bad_kernel(self):
        k = kernels()["tri_g1a"]
        k.cases[0]["type"] = 5
        with self.assertRaises(ValueError):
            probe.check_kernel(k)
        k = kernels()["tri_g1a"]
        with unittest.mock.patch.object(probe, "DESC", dict(probe.DESC, w3=0x80000000)):
            lines = k.lines()
        with unittest.mock.patch.object(k, "lines", return_value=lines), self.assertRaises(ValueError):
            probe.check_kernel(k)


class HardwarePath(unittest.TestCase):
    def test_measured_outputs_match(self):
        ks = kernels()
        single = [k for k in ks.values() if k.threads == 1]
        o = FakeOracle(ks)
        rows, bad = probe.compare(single, None, probe.run_hw(single, o, hw_asm))
        self.assertEqual((len(rows), bad), (564, 0))
        self.assertEqual(o.calls, [1] * len(single))                  # one kernel per dispatch

    def test_a_different_word_is_a_mismatch(self):
        ks = kernels()
        k = [ks["tri_g3a"]]
        with redirect_stdout(io.StringIO()):
            _, bad = probe.compare(k, None, probe.run_hw(k, FakeOracle(ks, plant=("tri_g3a", 5)), hw_asm))
        self.assertEqual(bad, 1)

    def test_untrusted_outputs_stop_the_run(self):
        ks = kernels()
        for kw in (dict(prefill_only=True), dict(no_done=True)):
            with self.subTest(**kw), self.assertRaises(RuntimeError):
                probe.run_hw([ks["tri_g1a"]], FakeOracle(ks, **kw), hw_asm)

    def test_lane_kernels_skipped_on_a_small_launcher(self):
        ks = kernels()
        o = FakeOracle(ks)
        o.max_words = 4096                                            # the DRM launcher's limit
        hw = probe.run_hw([ks["tri_g1a"], ks["tri_lanes_w32"]], o, hw_asm)
        self.assertIn("skipped", hw["tri_lanes_w32"])
        self.assertEqual(o.calls, [1])

    def test_missing_oracle_exits_77(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith("LEONIDA_HW_") and k != "LEONIDA_AMD_CLANG"}
        env["LEONIDA_AMD_CLANG"] = os.path.join(REPO, "does_not_exist_clang")
        proc = subprocess.run([sys.executable, "-I", PROBE, "--oracle", "hw", "--kernel", "tri_g1a"], env=env,
                              capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 77, proc.stderr or proc.stdout)


class CommandLine(unittest.TestCase):
    def test_bad_arguments_rejected_before_open_oracle(self):
        for argv in (["--oracle", "gpu"], ["--oracle", ""], ["--oracle", "hw", "--kernel", "tri_g9"]):
            with self.subTest(argv=argv):
                err = io.StringIO()
                with unittest.mock.patch.object(hw_runner, "open_oracle",
                                                side_effect=AssertionError("open_oracle called")), \
                        redirect_stderr(err):
                    self.assertEqual(probe.main(argv), 2)
                self.assertIn("error:", err.getvalue())

    def test_json_directory_is_named_as_typed(self):
        # every other diagnostic in this tool prints the path the user typed; repr() doubled each backslash
        with tempfile.TemporaryDirectory() as tmp:
            err = io.StringIO()
            with redirect_stderr(err):
                self.assertEqual(probe.main(["--json", tmp]), 2)
            self.assertEqual(err.getvalue(), f"error: --json {tmp} is not a file path\n")

    def test_json_empty_is_a_named_error(self):
        # an empty --json value cannot be echoed as a path; the message says (empty) instead of showing ''
        err = io.StringIO()
        with redirect_stderr(err):
            self.assertEqual(probe.main(["--json", "  "]), 2)
        self.assertEqual(err.getvalue(), "error: --json (empty) is not a file path\n")

    def test_write_kernels(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            self.assertEqual(probe.main(["--write-kernels", tmp, "--kernel", "tri_g4b"]), 0)
            with open(os.path.join(tmp, "tri_g4b.s"), "rb") as f:
                data = f.read()
            self.assertNotIn(b"\r\n", data)
            self.assertEqual(data.decode("utf-8"), "\n".join(kernels()["tri_g4b"].lines()) + "\n")
            self.assertTrue(os.path.isfile(os.path.join(tmp, "tri_g4b.json")))


if __name__ == "__main__":
    unittest.main()
