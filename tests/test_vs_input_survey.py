"""vs_input_survey: NGG markers, input VGPRs read before written, and the VS families, on hand-written listings.

    python -I tests/test_vs_input_survey.py

Every input is built here (index.json and five tiny .s listings). No game data.
"""
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import vs_input_survey as V  # noqa: E402

failures = 0


def check(cond, what):
    global failures
    print(("PASS " if cond else "FAIL ") + what)
    failures += not cond


LISTINGS = {
    # vertex id only, legacy exports
    "a_vs": "\tv_cvt_f32_u32_e32 v1, v0\n\texp pos0 v1, v1, v1, v1 done\n\texp param0 v1, v1, v1, v1\n\ts_endpgm\n",
    # vertex and instance id (v3 read before v3 is written)
    "b_vs": "\tv_add_nc_u32_e32 v1, v0, v3\n\tv_mov_b32_e32 v3, 0\n\texp pos0 v1, v1, v1, v3 done\n"
            "\texp param0 v1, v1, v1, v1\n\ts_endpgm\n",
    # fetch call, then the fetched attributes from v4
    "c_vs": "\ts_swappc_b64 s[4:5], s[4:5]\n\tv_fma_f32 v1, v4, v5, v6\n\texp pos0 v1, v1, v1, v1 done\n"
            "\texp param0 v4, v5, v6, v1\n\ts_endpgm\n",
    # constant position: reads neither
    "d_vs": "\tv_mov_b32_e32 v1, 1.0\n\texp pos0 v1, v1, v1, v1 done\n\texp param0 v1, v1, v1, v1\n\ts_endpgm\n",
    # a pixel shader, which the survey must skip
    "e_ps": "\tv_mov_b32_e32 v1, v0\n\ts_endpgm\n",
}


def build(root):
    dx = os.path.join(root, "dxbc")
    os.makedirs(os.path.join(dx, "disasm"))
    index = []
    for name, text in LISTINGS.items():
        with open(os.path.join(dx, "disasm", name + ".s"), "w", encoding="utf-8") as f:
            f.write(f"; {name}\n" + text)
        index.append({"name": name, "file": name + ".dxbc", "stage": name.split("_")[1]})
    with open(os.path.join(dx, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f)
    return dx


def test_vs_input_survey():
    """Runs under pytest (CI's translator-tests job) and as a plain script."""
    with tempfile.TemporaryDirectory() as tmp:
        dx = build(tmp)
        r = V.survey(dx)
        check(r["vertex_shaders"] == 4, "only the four vertex shaders are surveyed")
        m = r["markers"]
        check(m["exp prim"] == 0 and m["s_sendmsg"] == 0 and m["v_mbcnt"] == 0, "no NGG markers in legacy listings")
        check(m["exp pos0"] == 4 and m["exp param"] == 4 and m["fetch call"] == 1, "exports and the fetch call counted")
        check(r["input_vgprs_read"].get("v0") == 2 and r["input_vgprs_read"].get("v3") == 1,
              "v0 read on entry by two shaders, v3 by one (before it is written)")
        check("v1" not in r["input_vgprs_read"], "v1 is written before it is read, so it is not an input")
        check(r["input_vgprs_read"].get("v4") == 1, "the fetched attribute v4 counts as read on entry")
        check(r["families"] == {"reads v0 (vertex id)": 1, "reads v0 (vertex id) and v3 (instance id)": 1,
                                "fetch call, attributes from v4 up": 1, "reads neither v0 nor a fetch call": 1},
              "each shader lands in exactly one family")

        with open(os.path.join(dx, "disasm", "a_vs.s"), "a", encoding="utf-8") as f:
            f.write("\texp prim v0, off, off, off done\n")
        out = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "scripts", "vs_input_survey.py"), dx],
                             capture_output=True, text=True)
        check(out.returncode == 0 and "exp prim 1" in out.stdout and "NGG markers present" in out.stdout,
              "a primitive export flips the verdict away from legacy")
        check("a_vs" not in out.stdout and "c_vs" not in out.stdout, "the summary prints counts, not shader names")

    # bad inputs and outputs: one error line, exit 2, no traceback, and index.json is never overwritten
    tool = os.path.join(ROOT, "scripts", "vs_input_survey.py")

    def cli(*args):
        return subprocess.run([sys.executable, "-I", tool, *args], capture_output=True, text=True)

    with tempfile.TemporaryDirectory() as tmp:
        d = os.path.join(tmp, "dxbc")
        os.mkdir(d)
        plain = os.path.join(tmp, "plain")
        with open(plain, "w") as f:
            f.write("x")
        idx = os.path.join(d, "index.json")
        cases = {
            "missing folder": [os.path.join(tmp, "nope")],
            "a file as the folder": [plain],
            "folder without index.json": [d],
        }
        for what, args in cases.items():
            r = cli(*args)
            check(r.returncode == 2 and "Traceback" not in r.stderr and len(r.stderr.strip().splitlines()) == 1,
                  f"{what}: one error line, exit 2")
        for what, body in (("junk bytes", b"\xff\xfe\x80"), ("empty file", b""), ("not a list", b'{"a": 1}'),
                           ("a list of non-objects", b"[1, 2]"),
                           ("a vs entry without file", b'[{"stage": "vs"}]'),
                           ("a vs entry with a non-string file", b'[{"stage": "vs", "file": 5}]')):
            with open(idx, "wb") as f:
                f.write(body)
            r = cli(d)
            check(r.returncode == 2 and "Traceback" not in r.stderr and len(r.stderr.strip().splitlines()) == 1,
                  f"index.json {what}: one error line, exit 2")
        with open(idx, "w") as f:
            f.write("[]")
        r = cli(d, "--json", os.path.join(plain, "sub", "r.json"))
        check(r.returncode == 2 and "Traceback" not in r.stderr, "--json under a regular file: one error, exit 2")
        r = cli(d, "--json", idx)
        with open(idx) as f:
            check(r.returncode == 2 and f.read() == "[]", "--json equal to index.json is refused and leaves it alone")
    assert failures == 0, f"{failures} vs_input_survey check(s) failed"


if __name__ == "__main__":
    try:
        test_vs_input_survey()
    except AssertionError:
        pass
    print(f"test_vs_input_survey: {'OK' if not failures else 'FAILED'} ({failures} failure{'s' if failures != 1 else ''})")
    sys.exit(1 if failures else 0)
