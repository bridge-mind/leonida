"""tools/ps_sweep.py on a synthetic index and synthetic outputs: no GPU, game data or Vulkan tools needed.

    python -I tests/test_ps_sweep.py
"""

import contextlib
import io
import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import ps_sweep  # noqa: E402


def meta(bary=False, bindings=(), flow="structured", stage="ps", **extra):
    m = {"stage": stage, "features": ["fragmentShaderBarycentric"] if bary else [], "bindings": list(bindings),
         "control_flow": flow}
    m.update(extra)
    return m


def synthetic_index():
    index = {}
    for i in range(60):
        index[f"a{i}"] = meta()
    for i in range(40):
        index[f"b{i}"] = meta(bary=True, bindings=(0, 1, 2))
    for i in range(30):
        index[f"c{i}"] = meta(bindings=(0, 1, 2, 5), flow="dispatch loop")
    for i in range(3):
        index[f"d{i}"] = meta(bindings=(0, 1, 2, 3), flow="straight line")
    for i in range(20):
        index[f"v{i}"] = meta(stage="vs")
    return index


TARGET_FLOATS = 32 * 32 * 4


def write_raw(path, values):
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(values)}f", *values))
        f.write(struct.pack("<f", -1.0) * (TARGET_FLOATS - len(values)))


class Sampling(unittest.TestCase):
    def test_strata_and_pixel_shaders_only(self):
        ps = {k: v for k, v in synthetic_index().items() if v["stage"] == "ps"}
        self.assertEqual(len(ps), 133)
        picked, per = ps_sweep.pick_sample(ps, 40, 7)
        self.assertEqual(set(per), {(False, "none", "structured"), (True, "low", "structured"),
                                    (False, "high", "dispatch loop"), (False, "low", "straight line")})
        self.assertEqual(len(picked), len(set(picked)))
        self.assertTrue(all(name in ps for name in picked))
        # every stratum is represented, a stratum smaller than the floor is taken whole
        self.assertEqual(per[(False, "low", "straight line")], (3, 3))
        self.assertTrue(all(sampled >= min(10, size) for size, sampled in per.values()))
        self.assertEqual(sum(sampled for _, sampled in per.values()), len(picked))

    def test_fixed_seed_is_stable_and_seed_matters(self):
        ps = {k: v for k, v in synthetic_index().items() if v["stage"] == "ps"}
        first = ps_sweep.pick_sample(ps, 40, 7)[0]
        self.assertEqual(first, ps_sweep.pick_sample(ps, 40, 7)[0])
        self.assertNotEqual(first, ps_sweep.pick_sample(ps, 40, 8)[0])

    def test_sample_command_writes_numbers_only_to_stdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            spirv, out = os.path.join(tmp, "spirv"), os.path.join(tmp, "out")
            os.makedirs(spirv)
            with open(os.path.join(spirv, "index.json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump(synthetic_index(), f)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                ps_sweep.main(["sample", "--spirv", spirv, "--out", out, "--n", "40"])
            self.assertNotIn("a1", buf.getvalue().replace("sampled", ""))
            self.assertIn("population 133", buf.getvalue())
            with open(os.path.join(out, "sample.json"), encoding="utf-8") as f:
                self.assertEqual(len(json.load(f)), int(buf.getvalue().split("sampled ")[1].split()[0]))


class VertexShader(unittest.TestCase):
    def test_outputs_match_inputs_and_flat_locations(self):
        glsl = ps_sweep.vs_glsl([0, 2, 5], {2})
        self.assertIn("layout(location=0) out vec4 o0;", glsl)
        self.assertIn("layout(location=2) flat out vec4 o2;", glsl)
        self.assertIn("layout(location=5) out vec4 o5;", glsl)
        self.assertNotIn("o1", glsl)
        self.assertIn("gl_VertexIndex", glsl)

    def test_no_inputs(self):
        glsl = ps_sweep.vs_glsl([], set())
        self.assertNotIn("out vec4", glsl)
        self.assertIn("gl_Position", glsl)

    def test_f16_halves_are_decoded_from_the_params_own_words(self):
        # --f16-interp-fallback: the PS reads param 1's halves at locations 26 and 27 and the raw param itself is not an
        # input, so a default translation would read param 1's words per vertex at location 1.
        glsl = ps_sweep.vs_glsl([3, 26, 27], set(), {"1": [26, 27]})
        raw1 = ps_sweep._raw_glsl(1)
        self.assertIn(f"vec4 r = {raw1};", glsl)
        self.assertIn("o26 = vec4(unpackHalf2x16(w.x)[0], unpackHalf2x16(w.y)[0], unpackHalf2x16(w.z)[0], "
                      "unpackHalf2x16(w.w)[0]);", glsl)
        self.assertIn("o27 = vec4(unpackHalf2x16(w.x)[1],", glsl)
        self.assertIn(f"o3 = {ps_sweep._raw_glsl(3)};", glsl)
        self.assertNotIn(f"o26 = {ps_sweep._raw_glsl(26)}", glsl)
        self.assertNotIn("o1 =", glsl)

    def test_without_halves_the_text_is_unchanged(self):
        self.assertEqual(ps_sweep.vs_glsl([0, 2], {2}), ps_sweep.vs_glsl([0, 2], {2}, {}))
        self.assertNotIn("unpackHalf2x16", ps_sweep.vs_glsl([0, 2], {2}, None))


class GeometryShaderLookup(unittest.TestCase):
    def test_job_env_drops_inherited_per_job_variables(self):
        base = {"PATH": "p", "VKS_GS": "stale.spv", "VKS_TEX": "1", "VKS_NEED_MAINTENANCE8": "1"}
        env = ps_sweep.job_env(base, 9, "z", {}, None)
        for k in ps_sweep.JOB_VARS:
            self.assertNotIn(k, env)
        self.assertEqual((env["PATH"], env["VKS_TIMEOUT_S"]), ("p", "9"))
        env = ps_sweep.job_env(base, 9, "t", {"needs_maintenance8": True}, "g.spv")
        self.assertEqual((env["VKS_GS"], env["VKS_TEX"], env["VKS_NEED_MAINTENANCE8"]), ("g.spv", "1", "1"))

    def test_gs_is_the_bary_gs_file_beside_the_pixel_shader(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(ps_sweep.gs_for(d, "x1"))
            open(os.path.join(d, "x1.bary_gs.spv"), "wb").close()
            self.assertEqual(ps_sweep.gs_for(d, "x1"), os.path.join(d, "x1.bary_gs.spv"))
            self.assertIsNone(ps_sweep.gs_for(d, "x2"))


class Classification(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def raw(self, name, values):
        path = os.path.join(self.tmp.name, name)
        write_raw(path, values)
        return path

    @staticmethod
    def rec(code):
        return {"i": 0, "code": code, "reason": "", "targets": 1}

    def classify(self, a, b):
        return ps_sweep.classify_pair(self.rec(0), self.rec(0), self.raw("a", a), self.raw("b", b))

    def test_ok_within_tolerance(self):
        self.assertEqual(self.classify([0.5, 1.0, 2.0, -1.0], [0.5004, 1.0004, 2.0, -1.0]), ("OK on both", ""))

    def test_differs_beyond_tolerance_with_tier(self):
        self.assertEqual(self.classify([0.5, 1.0], [0.5, 1.5]), ("differs beyond tolerance", "gross (>0.1)"))
        self.assertEqual(self.classify([0.5, 1.0], [0.5, 1.05]), ("differs beyond tolerance", "moderate (>1e-2)"))
        self.assertEqual(self.classify([0.5, 1.0], [0.5, 1.005]), ("differs beyond tolerance", "small (>1e-3)"))

    def test_nan_and_zero_on_one_side_only(self):
        nan = float("nan")
        self.assertEqual(self.classify([0.5, 1.0], [0.5, nan]), ("NaN or zero on one GPU only", "nan"))
        self.assertEqual(self.classify([0.5, 1.0], [0.0, 0.0]), ("NaN or zero on one GPU only", "zero"))
        self.assertEqual(self.classify([0.5, 1.0], [-1.0, -1.0]), ("NaN or zero on one GPU only", "unwritten"))
        self.assertEqual(self.classify([nan, 1.0], [nan, 1.0]), ("OK on both", ""))   # NaN on both is not a difference

    def test_non_finite_values_compare_per_component(self):
        nan, inf = float("nan"), float("inf")
        gross = ("differs beyond tolerance", "gross (>0.1)")
        # NaN in different components of the two outputs
        self.assertEqual(self.classify([nan, 1.0], [1.0, nan]), gross)
        # an infinity against a finite value, and against the opposite infinity
        self.assertEqual(self.classify([inf, 1.0], [1.0, 1.0]), gross)
        self.assertEqual(self.classify([inf, 1.0], [-inf, 1.0]), gross)
        # the same non-finite value at the same component matches
        self.assertEqual(self.classify([inf, nan, 1.0], [inf, nan, 1.0]), ("OK on both", ""))

    def test_not_translated_is_its_own_class(self):
        rec = {"code": ps_sweep.NOT_TRANSLATED, "reason": "not translated"}
        ok = {"code": 0, "reason": ""}
        self.assertEqual(ps_sweep.classify_pair(rec, ok, "x", "y"), ("not translated", ""))
        self.assertEqual(ps_sweep.classify_pair(ok, rec, "x", "y"), ("not translated", ""))

    def test_failures_and_timeouts(self):
        # A non-success code takes precedence; neither path should even be opened.
        for a, b, expected in ((0, 77, ("fails on one GPU", "b: missing device feature")),
                               (4, 0, ("fails on one GPU", "a: exit 4")),
                               (77, 5, ("fails on both", "77/5")),
                               (3, 0, ("timeout", "")), (0, 3, ("timeout", ""))):
            with self.subTest(a=a, b=b):
                result = ps_sweep.classify_pair(self.rec(a), self.rec(b), self.tmp.name, self.tmp.name)
                self.assertEqual(result, expected)


class CompareCommand(unittest.TestCase):
    def test_readback_shapes(self):
        one = struct.pack("<f", 0.25) * TARGET_FLOATS
        two = one * 2
        cases = [
            # recorded counts, raw outputs (None = absent), expected classification and detail
            (2, 2, two, two, "OK on both", ""),
            (8, 8, one * 8, one * 8, "OK on both", ""),
            (2, 2, two, two[:-4] + struct.pack("<f", 1.0), "differs beyond tolerance", "gross (>0.1)"),
            (1, 1, b"", b"", "invalid readback", "a: wrong output size"),
            (1, 1, one[:-4], one[:-4], "invalid readback", "a: wrong output size"),
            (1, 1, one, one[:-4], "invalid readback", "b: wrong output size"),
            (1, 1, two, two, "invalid readback", "a: wrong output size"),
            (1, 1, one, one + b"\x00", "invalid readback", "b: wrong output size"),
            (2, 2, one, one, "invalid readback", "a: wrong output size"),
            (1, 2, one, one, "invalid readback", "target counts differ"),
            (None, 1, one, one, "invalid readback", "invalid target count"),
            (0, 0, b"", b"", "invalid readback", "invalid target count"),
            (9, 9, one * 9, one * 9, "invalid readback", "invalid target count"),
            (1, True, one, one, "invalid readback", "invalid target count"),
            (1, 1, one, None, "invalid readback", "b: missing output"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            sample = [{"name": str(i), "stratum": [False, "none", "structured"]} for i in range(len(cases))]
            (out / "sample.json").write_text(json.dumps(sample), encoding="utf-8", newline="\n")
            for side, tag in enumerate(("a", "b")):
                folder = out / f"{tag}_z"
                folder.mkdir()
                with (folder / "results_0.jsonl").open("w", encoding="utf-8", newline="\n") as f:
                    for i, case in enumerate(cases):
                        rec = {"i": i, "code": 0, "reason": ""}
                        if case[side] is not None:
                            rec["targets"] = case[side]
                        f.write(json.dumps(rec) + "\n")
                        data = case[2 + side]
                        if data is not None:
                            (folder / f"{i}.raw").write_bytes(data)
            with contextlib.redirect_stdout(io.StringIO()):
                ps_sweep.main(["compare", "--out", tmp])
            rows = json.loads((out / "classes_z.json").read_text(encoding="utf-8"))
            self.assertEqual([(r["class"], r["sub"]) for r in rows], [case[-2:] for case in cases])

    def test_rerun_overrides_and_counts(self):
        with tempfile.TemporaryDirectory() as out:
            with open(os.path.join(out, "sample.json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump([{"name": "x", "stratum": [False, "none", "structured"]},
                           {"name": "y", "stratum": [False, "none", "structured"]}], f)
            for tag, codes in (("a", (0, 3)), ("b", (0, 0))):
                d = os.path.join(out, f"{tag}_z")
                os.makedirs(d)
                with open(os.path.join(d, "results_0.jsonl"), "w", encoding="utf-8", newline="\n") as f:
                    for i, c in enumerate(codes):
                        f.write(json.dumps({"i": i, "code": c, "reason": "", "targets": 1}) + "\n")
                write_raw(os.path.join(d, "0.raw"), [0.25] * 8)
                write_raw(os.path.join(d, "1.raw"), [0.25] * 8)
            # shader 1 timed out on a while another process was starting: its re-run is clean
            re = os.path.join(out, "a_z_re")
            os.makedirs(re)
            with open(os.path.join(re, "results_0.jsonl"), "w", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps({"i": 1, "code": 0, "reason": "", "targets": 1}) + "\n")
            write_raw(os.path.join(re, "1.raw"), [0.25] * 8)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                ps_sweep.main(["compare", "--out", out, "--tags", "a", "b"])
            self.assertIn("OK on both: 2", buf.getvalue())
            self.assertNotIn("timeout", buf.getvalue())

    def test_exact_readbacks_and_rerun(self):
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            clear = struct.pack("<f", -1) * (ps_sweep.W * ps_sweep.H * 4)
            zero = bytes(len(clear))
            neg_zero = zero[:-4] + struct.pack("<f", -0.0)
            # Entries are (declared targets, payload). None means no result record.
            pairs = [
                ((2, clear + zero), (2, clear + zero)),
                ((1, clear), (1, clear)),
                ((1, zero), (1, neg_zero)),
                ((1, zero), (1, None)),
                ((1, zero[:-1]), (1, zero[:-1])),
                ((1, zero), (2, zero)),
                ((1, clear), None),
                ((1, zero + b"x"), (1, zero + b"x")),
                ((1, clear), (1, zero)),
            ]
            (out / "sample.json").write_text(json.dumps([{}] * len(pairs)), encoding="utf-8", newline="\n")
            for side, tag in enumerate(("a", "b")):
                folder = out / f"{tag}_z"
                folder.mkdir()
                with (folder / "results_0.jsonl").open("w", encoding="utf-8", newline="\n") as log:
                    for i, pair in enumerate(pairs):
                        if pair[side] is None:
                            continue
                        n, data = pair[side]
                        rec = {"i": i, "code": 0, "targets": n}
                        if side == 1:
                            rec["samples_passed"] = 0
                        print(json.dumps(rec), file=log)
                        if data is not None:
                            (folder / f"{i}.raw").write_bytes(data)
            rerun = out / "a_z_re"
            rerun.mkdir()
            (rerun / "8.raw").write_bytes(zero)
            (rerun / "results_8.jsonl").write_text('{"i":8,"code":3,"targets":1}\n', encoding="utf-8", newline="\n")
            with contextlib.redirect_stdout(io.StringIO()):
                ps_sweep.main(["compare", "--exact", "--out", tmp])
            report = json.loads((out / "exact_z.json").read_text(encoding="utf-8"))
            self.assertEqual([r["class"] for r in report["rows"]], [
                "non-clear exact", "clear-only exact", "mismatch", "missing", "incompatible",
                "incompatible", "missing", "incompatible", "missing",
            ])
            self.assertEqual(report["rows"][0]["targets"], ["clear-only exact", "non-clear exact"])
            self.assertEqual(report["targets"], {"non-clear exact": 1, "clear-only exact": 2, "mismatch": 1})
            self.assertEqual(report["rows"][-1]["codes"], [3, 0])
            self.assertEqual(report["rows"][0]["samples_passed"], [None, 0])
            self.assertEqual(report["rows"][-1]["samples_passed"], [None, 0])
            self.assertNotIn("samples_passed", report["rows"][6])

    def test_incomplete_depth_measurements(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = {"depth": True, "code": 0, "dir": tmp, "samples_passed": 0}
            self.assertEqual(ps_sweep.depth_measurements(0, rec, rec), {"depth_class": "missing"})
            with open(os.path.join(tmp, "0.raw.depth"), "wb") as plane:
                plane.write(b"short")
            self.assertEqual(ps_sweep.depth_measurements(0, rec, rec), {"depth_class": "incompatible"})
            with open(os.path.join(tmp, "0.raw.depth"), "wb") as plane:
                plane.write(bytes(ps_sweep.W * ps_sweep.H * 4))
            self.assertEqual(ps_sweep.depth_measurements(0, rec, rec), {"depth_class": "incompatible"})
            self.assertEqual(ps_sweep.depth_measurements(0, {"code": 0}, rec), {"depth_class": "missing"})

    @unittest.skipUnless(os.environ.get("PS_SWEEP_TEST_EXE"), "requires a Vulkan worker")
    def test_integer_texture_fetch(self):
        from pathlib import Path
        import subprocess

        worker = os.environ["PS_SWEEP_TEST_EXE"]
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            vs = ps_sweep.make_vs("glslangValidator", {}, tmp)
            src = folder / "fetch.frag"
            fs = folder / "fetch.spv"
            raw = folder / "draw.raw"
            draw = [worker, vs, str(fs), str(raw), "32", "32", "1"]
            def compile_shader(text):
                src.write_text(text, encoding="utf-8", newline="\n")
                subprocess.run(["glslangValidator", "-V", str(src), "-o", str(fs)], check=True)

            for mode, prefix, expected in (("sint", "i", (-126, 34, -8, 77)), ("uint", "u", (130, 34, 248, 77))):
                source = ("#version 450\n#extension GL_EXT_samplerless_texture_functions : require\n"
                          f"layout(set=0,binding=0) uniform {prefix}texture2D tex[1];\n"
                          "layout(location=0) out vec4 color;\n"
                          "void main() {color = vec4(texelFetch(tex[0], ivec2(2,3), 0));}\n")
                compile_shader(source)
                env = ps_sweep.job_env(os.environ, 20, mode, {}, None)
                subprocess.run(draw, env=env, check=True)
                self.assertEqual(raw.read_bytes(), struct.pack("<4f", *expected) * 1024)
                raw.unlink()
                env["VKS_TEX"] = "uint" if mode == "sint" else "sint"
                self.assertEqual(subprocess.run(draw, env=env).returncode, 77)
                self.assertFalse(raw.exists())
                for other in (source.replace("tex[1];", "tex[2];").replace("tex[0],", "tex[1],"),
                              source.replace("2,3", "8,3"), source.replace("), 0)", "), 1)"),
                              source.replace("binding=0", "binding=3"),
                              source.replace("ivec2(2,3)", "ivec2(gl_FragCoord.xy)"),
                              source.replace("layout(location=0)", "layout(set=0,binding=0) uniform texture2D other;\n"
                                             "layout(location=0)").replace(";}"," + texelFetch(other, ivec2(0), 0);}")):
                    compile_shader(other)
                    self.assertEqual(subprocess.run([worker, "--screen-integer", str(fs)]).returncode, 77)

            compile_shader(source.replace("utexture2D", "texture2D"))
            env = ps_sweep.job_env(os.environ, 20, "t", {}, None)
            subprocess.run(draw, env=env, check=True)
            for rgba in struct.iter_unpack("<4f", raw.read_bytes()):
                for component, byte in zip(rgba, (64, 96, 64, 255)):
                    self.assertAlmostEqual(component, byte / 255, places=6)
            self.assertEqual(raw.stat().st_size, 1024 * 16)

    @unittest.skipUnless(os.environ.get("PS_SWEEP_TEST_EXE"), "requires a precise-query Vulkan worker")
    def test_clear_draws_with_opposite_coverage(self):
        from pathlib import Path
        import subprocess

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            for tag, statement in (("covered", "color = vec4(-1);"), ("discarded", "discard;")):
                folder = out / tag
                folder.mkdir()
                src = folder / "control.frag"
                src.write_text("#version 450\nlayout(location=0) out vec4 color;\nvoid main() {" + statement + "}\n",
                               encoding="utf-8", newline="\n")
                subprocess.run(["glslangValidator", "-V", str(src), "-o", str(folder / "control.spv")], check=True)
                (folder / "index.json").write_text('{"control":{"stage":"ps"}}', encoding="utf-8", newline="\n")
                if tag == "covered":
                    ps_sweep.main(["sample", "--spirv", str(folder), "--out", tmp])
                ps_sweep.main(["run", "--spirv", str(folder), "--out", tmp, "--tag", tag, "--occlusion",
                               "--exe", os.environ["PS_SWEEP_TEST_EXE"]])
            compare = ["compare", "--out", tmp, "--tags", "covered", "discarded"]
            ps_sweep.main(compare)
            ps_sweep.main(compare + ["--exact"])
            exact = json.loads((out / "exact_z.json").read_text())["rows"][0]
            ordinary = json.loads((out / "classes_z.json").read_text())[0]
            self.assertEqual(exact["class"], "clear-only exact")
            self.assertEqual(exact["samples_passed"], [ps_sweep.W * ps_sweep.H, 0])
            self.assertEqual(ordinary["samples_passed"], exact["samples_passed"])

    @unittest.skipUnless(os.environ.get("PS_SWEEP_TEST_EXE"), "requires a Vulkan worker with D32 readback")
    def test_equal_color_different_depth(self):
        from pathlib import Path
        import subprocess

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            for tag, code in (("clear", "gl_FragDepth = 1.0;"), ("near", "gl_FragDepth = 0.25;"),
                              ("discarded", "discard;")):
                folder = out / tag
                folder.mkdir()
                src = folder / "control.frag"
                src.write_text("#version 450\nlayout(location=0) out vec4 color;\nvoid main(){color=vec4(-1);"
                               + code + "}\n", encoding="utf-8", newline="\n")
                subprocess.run(["glslangValidator", "-V", str(src), "-o", str(folder / "control.spv")], check=True)
                (folder / "index.json").write_text('{"control":{"stage":"ps"}}', encoding="utf-8", newline="\n")
                if tag == "clear":
                    ps_sweep.main(["sample", "--spirv", str(folder), "--out", tmp])
                ps_sweep.main(["run", "--spirv", str(folder), "--out", tmp, "--tag", tag, "--two-clear",
                               "--exe", os.environ["PS_SWEEP_TEST_EXE"]])
            pixels = ps_sweep.W * ps_sweep.H
            for tag, differing, written in (("near", pixels, pixels), ("discarded", 0, 0)):
                args = ["compare", "--out", tmp, "--tags", "clear", tag]
                ps_sweep.main(args)
                ps_sweep.main(args + ["--exact"])
                ordinary = json.loads((out / "classes_z.json").read_text())[0]
                exact = json.loads((out / "exact_z.json").read_text())["rows"][0]
                self.assertEqual(exact["targets"], ["clear-only exact"])
                for row in (ordinary, exact):
                    self.assertEqual(row["depth_different_pixels"], differing)
                    self.assertEqual(row["depth_nonclear_pixels"], [0, differing])
                    self.assertEqual(row["samples_passed"], [pixels, written])
                    self.assertEqual(row["depth_clears"], [1, 0])
                    self.assertEqual(row["zero_depth_different_pixels"], pixels)
                    self.assertEqual(row["depth_control"], [
                        {"identical": pixels, "follow_clear": 0, "other": 0},
                        {"identical": written, "follow_clear": pixels - written, "other": 0},
                    ])
                self.assertEqual(ordinary["class"], "depth mismatch")
                self.assertEqual(exact["class"], "mismatch")

    def test_help_warns_about_nvidia_resets(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), self.assertRaises(SystemExit):
            ps_sweep.main(["--help"])
        self.assertIn("TDR", buf.getvalue())
        self.assertIn("display", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
