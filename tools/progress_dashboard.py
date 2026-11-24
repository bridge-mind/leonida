"""Progress dashboard: what works, what changed and what blocks next, from outcomes the project already produces.

    python -I tools/progress_dashboard.py --out <folder outside the repository> [--work <work>] [--run-tests]
                                          [--junit FILE ...] [--github] [--login NAME]

Writes, into --out only (a folder inside the repository is refused, so nothing here can be committed by accident):
    progress.html                 one self-contained page (no network, no packages; open it from disk)
    snapshots/progress_<utc>.json the data behind it; a snapshot equal to the previous one is not written again
    history.json                  one summary line per snapshot, for the trend chart
    unit_results.json             the last --run-tests outcome, kept so a later run can show what changed since

Sources, each optional and reported as missing rather than counted as zero when absent:
    unit tests      --run-tests runs tests/test_*.py the way CI selects them (standard-library files with a
                    __main__ block), one process each; the outcome and the commit it ran on are kept
    ctest           runtime/build/Testing/Temporary/LastTest.log, and any --junit file (CTest or pytest flavour)
    gpu tests       --gpu-log: saved output of runtime/test/shaders/check.py, one panel group per GPU
    contract        the translator contract table in docs/design.md, by how each row says it was confirmed
    imports         the measured import tables in docs/runtime_imports.md
    corpus          --work: counts per stage from <work>/shaders/dxbc/index.json and <work>/shaders/spirv/index.json
    github          --github, read-only through `gh`: every pull request into main (merged = implemented, open =
                    partial), latest CI runs on main, the D3D- table of #313 checked against those merges, the
                    coordination issue's section sizes and, with --login, that account's claims and pull requests

No game data leaves the inputs: the page and the JSON carry counts, pass/fail, error classes and the project's
own document text; never a shader name, a carved file name or a path into the game copy.
Usage: `python -I tools/progress_dashboard.py --help` lists the options.
"""
import argparse
import base64
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from file_mode import published_mode  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = os.path.join(ROOT, "tools", "progress_dashboard.html")
LOGO = os.path.join(ROOT, "assets", "logo.png")
SCHEMA = "leonida.progress/1"
STATUSES = ("verified", "implemented", "partial", "failed", "missing", "blocked", "unknown")
PYTEST_MARKERS = re.compile(r"^import pytest|^ *HAVE_NUMPY =", re.M)   # the CI selection rule (ci.yml)
DEFAULT_REPO = "ItsNotRexed/Leonida"


# ----------------------------------------------------------------------------------------------- helpers
def now_utc():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0)


def iso(t):
    if t is None:
        return None
    if isinstance(t, (int, float)):
        t = dt.datetime.fromtimestamp(t, dt.timezone.utc)
    return t.replace(microsecond=0).astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def parse_iso(s):
    if not s:
        return None
    try:
        d = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    return d if d.tzinfo else d.astimezone()


def read_utf8(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def save_text(path, text):
    # A fixed `path + ".tmp"` blocks refreshes when that name is taken by a stale directory
    # or a concurrent writer. Publish from a unique temp in the same directory instead.
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + ".", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.chmod(tmp, published_mode(path))          # mkstemp creates 0600
        os.replace(tmp, path)
    except BaseException:
        try:
            os.chmod(tmp, 0o600)                     # a read-only temp cannot be unlinked on Windows
            os.unlink(tmp)
        except OSError:
            pass
        raise


def dump_json(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=1) + "\n"


def run(cmd, cwd=None, timeout=60):
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    return r.stdout.decode("utf-8", "replace")


HOME = os.path.expanduser("~")


def scrub(text, limit=240):
    """One line of tool output with every absolute path replaced: a path inside the repository becomes its relative
    path, any other path (the work folder, the home folder, a temp folder) becomes <path>, file name included, since
    a file name outside the repository can be a carved shader's name."""
    if not text:
        return ""
    line = text.strip().splitlines()[-1] if text.strip() else ""

    def shown(m):
        path = m.group(0)
        if os.name != "nt" and re.match(r"(?i)[a-z]:[\\/]", path):
            # A drive-letter path is relative to the current folder elsewhere, so realpath would put it inside the
            # repository and print it whole (user name and file name included): it is never one of ours here.
            return "<path>"
        if inside(path, ROOT):
            rel = os.path.relpath(os.path.realpath(path), os.path.realpath(ROOT)).replace(os.sep, "/")
            return "." if rel == "." else rel
        return "<path>"

    # Match the known checkout prefix before the generic patterns stop at whitespace.
    if os.name == "nt":
        root = r"[\\/]".join(re.escape(part) for part in re.split(r"[\\/]", ROOT))
        checkout = rf"(?i:(?<![\w/\\]){root})"
    else:
        checkout = rf"(?<![\w.:/~]){re.escape(ROOT)}"
    # Consume a same-prefix sibling too, so shown() can redact its entire path.
    checkout_path = rf"{checkout}[^\s'\"]*"
    drive_path = r"(?i:\b[a-z]:[\\/][^\s'\"]*)"
    posix_path = r"(?<![\w.:/~])/[^\s/'\"][^\s'\"]*"
    line = re.sub(rf"{checkout_path}|{drive_path}|{posix_path}", shown, line)
    if HOME and len(HOME) > 3:
        line = line.replace(HOME, "~")
    line = re.sub(r"~[\\/][^\s'\"]*", "<path>", line)
    return line[:limit]


def inside(child, parent):
    child, parent = os.path.realpath(child), os.path.realpath(parent)
    try:
        return os.path.commonpath([child, parent]) == parent
    except ValueError:      # different drives on Windows
        return False


class Collector:
    """Gathers components (one per cell), metrics (ratios with an explicit denominator) and source records."""

    def __init__(self):
        self.components = []
        self.ids = set()
        self.metrics = []
        self.sources = []
        self.notes = []
        self.panels = {}

    def panel(self, pid, title, unit, weighted=False):
        self.panels[pid] = {"id": pid, "title": title, "unit": unit, "weighted": weighted}

    def source(self, sid, path, status, records=None, note=None):
        rel = None
        if path:
            rel = os.path.relpath(path, ROOT).replace("\\", "/") if inside(path, ROOT) else "<outside>/" + os.path.basename(path)
        self.sources.append({k: v for k, v in {"id": sid, "path": rel, "status": status, "records": records, "note": note}.items()
                             if v is not None})

    def add(self, cid, panel, group, name, status, detail=None, weight=None, **attrs):
        assert status in STATUSES, status
        if cid in self.ids:
            n = 2
            while f"{cid}#{n}" in self.ids:
                n += 1
            cid = f"{cid}#{n}"
        self.ids.add(cid)
        c = {"id": cid, "panel": panel, "group": group, "name": name, "status": status}
        if detail:
            c["detail"] = detail
        if weight is not None:
            c["weight"] = weight
        c.update({k: v for k, v in attrs.items() if v not in (None, "", [], {})})
        self.components.append(c)

    def metric(self, mid, panel, title, num, den, formula, note=None):
        val = None if num is None or not den else round(num / den, 6)
        self.metrics.append({k: v for k, v in {"id": mid, "panel": panel, "title": title, "numerator": num, "denominator": den,
                                                 "value": val, "formula": formula, "note": note,
                                                 "definition": hashlib.sha256(formula.encode("utf-8")).hexdigest()[:10]}.items()
                             if v is not None or k in ("value", "numerator", "denominator")})


# ----------------------------------------------------------------------------------------------- git
def git_info():
    head = run(["git", "rev-parse", "HEAD"], cwd=ROOT)
    if head is None:
        return {"available": False}
    info = {"available": True, "head": head.strip()}
    info["head_time"] = iso(int((run(["git", "log", "-1", "--format=%ct"], cwd=ROOT) or "0").strip() or 0))
    info["branch"] = (run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT) or "").strip()
    status = run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT) or ""
    info["modified_files"] = len([l for l in status.splitlines() if l.strip()])
    up = run(["git", "rev-parse", "--verify", "--quiet", "origin/main"], cwd=ROOT)
    if up:
        info["origin_main"] = up.strip()
        counts = run(["git", "rev-list", "--left-right", "--count", "HEAD...origin/main"], cwd=ROOT)
        if counts:
            ahead, behind = counts.split()
            info["ahead"], info["behind"] = int(ahead), int(behind)
    return info


def changed_since(commit):
    """Files changed between `commit` and HEAD plus uncommitted changes, or None when git cannot tell."""
    out = run(["git", "diff", "--name-only", commit, "--"], cwd=ROOT)
    if out is None:
        return None
    return {l.strip() for l in out.splitlines() if l.strip()}


def last_commit_time(path):
    out = run(["git", "log", "-1", "--format=%ct", "--", path], cwd=ROOT)
    return int(out.strip()) if out and out.strip() else None


# ----------------------------------------------------------------------------------------------- unit tests
def _walk_error(exc: OSError):
    # os.walk skips a directory it cannot list unless onerror raises; the dashboard must not report an
    # incomplete test selection or code digest as complete. main() turns the OSError into one error line.
    raise exc


def ci_test_files():
    """tests/test_*.py that CI runs one by one: standard-library files with a __main__ block (ci.yml)."""
    found = []
    tests = os.path.join(ROOT, "tests")
    for base, dirs, files in os.walk(tests, onerror=_walk_error):
        dirs[:] = sorted(d for d in dirs if d not in ("semantic", "__pycache__", "fixtures"))
        for name in sorted(files):
            if not (name.startswith("test_") and name.endswith(".py")):
                continue
            path = os.path.join(base, name)
            text = read_utf8(path)
            if PYTEST_MARKERS.search(text) or "__main__" not in text:
                continue
            found.append(os.path.relpath(path, ROOT).replace("\\", "/"))
    return found


def run_one_test(rel, timeout):
    t0 = time.monotonic()
    try:
        r = subprocess.run([sys.executable, "-I", os.path.join(ROOT, rel)], cwd=ROOT, capture_output=True, timeout=timeout)
        out = (r.stdout + r.stderr).decode("utf-8", "replace")
        skipped = re.search(r"^OK \(skipped=(\d+)\)\r?$", out, re.M)
        res = "pass" if r.returncode == 0 else "fail"
        tail = "" if res == "pass" else scrub(out)
        ran = re.search(r"Ran (\d+) tests?", out)
        cases = int(ran.group(1)) if ran else None
        return {"test": rel, "result": res, "seconds": round(time.monotonic() - t0, 2), "detail": tail,
                "cases": cases, "all_skipped": bool(skipped and cases == int(skipped.group(1)))}
    except subprocess.TimeoutExpired:
        return {"test": rel, "result": "fail", "seconds": timeout, "detail": f"timed out after {timeout} s"}


CODE_DIRS = ("tools", "scripts", "runtime/test/shaders")


def file_digest(rel):
    try:
        with open(os.path.join(ROOT, rel), "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except OSError:
        return None


def code_digest():
    """Hash make_game.py and the Python files in CODE_DIRS by content, including uncommitted edits."""
    h = hashlib.sha256()
    rels = ["make_game.py"]
    for d in CODE_DIRS:
        for base, dirs, files in os.walk(os.path.join(ROOT, d), onerror=_walk_error):
            dirs[:] = sorted(x for x in dirs if x not in ("__pycache__", "bin", "obj", "target", "build") and not x.startswith("."))
            rels += [os.path.relpath(os.path.join(base, f), ROOT).replace("\\", "/") for f in sorted(files) if f.endswith(".py")]
    for rel in rels:
        h.update(rel.encode("utf-8") + b"\0" + (file_digest(rel) or "-").encode("ascii"))
    return h.hexdigest()[:16]


def run_unit_tests(files, jobs, timeout):
    before = code_digest()
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        results = list(pool.map(lambda f: run_one_test(f, timeout), files))
    for r in results:
        r["digest"] = file_digest(r["test"])
    g = git_info()
    after = code_digest()
    return {"ran_at": iso(now_utc()), "commit": g.get("head"), "modified_files": g.get("modified_files"),
            "code_digest": after if after == before else None,   # code edited during the run: nothing is current
            "python": sys.version.split()[0], "results": sorted(results, key=lambda r: r["test"])}


def collect_unit(col, out_dir, args):
    col.panel("tests", "Unit tests (CI selection)", "test files")
    path = os.path.join(out_dir, "unit_results.json")
    if args.run_tests:
        print(f"running unit tests ({len(args.test_files)} selected, {args.jobs} at a time)...")
        data = run_unit_tests(args.test_files, args.jobs, args.test_timeout)
        save_text(path, dump_json(data))
    elif os.path.isfile(path):
        data = json.loads(read_utf8(path))
    else:
        col.source("unit_results", None, "missing", note="no unit-test outcome yet: run with --run-tests")
        return
    col.source("unit_results", path, "ok", len(data["results"]), note=f"ran {data['ran_at']} on {str(data.get('commit'))[:10]}")
    code_now = code_digest()
    code_same = data.get("code_digest") is not None and data["code_digest"] == code_now
    passed = current = 0
    for r in data["results"]:
        test = r["test"]
        if r["result"] != "pass":
            fresh = "failed"
        elif not r.get("digest") or data.get("code_digest") is None:
            fresh = "unknown"
        elif code_same and file_digest(test) == r["digest"]:
            fresh = "current"
        else:
            fresh = "revalidation_required"
        status = {"pass": "verified", "fail": "failed"}.get(r["result"], "unknown")
        if r["result"] == "pass" and r.get("all_skipped"):
            status = "unknown"
        if status == "verified" and fresh == "revalidation_required":
            status = "partial"
        passed += r["result"] == "pass"
        current += status == "verified"
        name = test.split("/", 1)[-1]
        col.add("test." + name, "tests", group_of_test(name), name, status, detail=r.get("detail") or None,
                freshness=fresh, seconds=r.get("seconds"), cases=r.get("cases"), ran_on=str(data.get("commit"))[:10])
    n = len(data["results"])
    col.metric("tests.passed", "tests", "Unit-test files passing (last run)", passed, n, "files whose last run exited 0 / files run")
    col.metric("tests.current", "tests", "Unit-test files passing on the current code", current, n,
               "files that passed with no change since to themselves, make_game.py or Python files under "
               + ", ".join(d + "/" for d in CODE_DIRS) + " / files run")


def group_of_test(name):
    stem = name[len("test_"):-3] if name.startswith("test_") else name
    head = stem.split("_")[0]
    families = {"docs": "docs", "i18n": "docs", "rdna2spirv": "translator", "spirv": "translator", "f16": "translator",
                "f32": "translator", "valu": "translator", "cf": "translator", "rpf9": "archives", "rpf9dump": "archives",
                "rbf0": "formats", "psin": "formats", "world": "world", "anim": "animation", "tex": "textures",
                "bcpack": "textures", "ttpf": "textures", "audio": "audio", "bink": "audio", "make": "make_game",
                "gamedata": "gamedata", "navmesh": "world", "story": "world", "d3d12": "runtime", "vk": "runtime"}
    return families.get(head, "other")


# ----------------------------------------------------------------------------------------------- ctest / junit
def parse_junit(xml):
    """[(name, result, seconds, detail)] from CTest (--output-junit) or pytest (--junitxml) XML."""
    root = xml if isinstance(xml, ET.Element) else ET.fromstring(xml)
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    out = []
    for suite in suites:
        for tc in suite.iter("testcase"):
            status = tc.get("status")
            st = (status or "").lower()
            fail = tc.find("failure") if tc.find("failure") is not None else tc.find("error")
            if fail is not None or st == "fail":
                res = "fail"
            elif tc.find("skipped") is not None or st in ("notrun", "disabled"):
                res = "skip"
            else:
                res = "pass"
            name, cls = tc.get("name") or "?", tc.get("classname")
            if cls and cls != name:
                name = f"{cls}::{name}"
            try:
                secs = float(tc.get("time")) if tc.get("time") else None
            except ValueError:
                secs = None
            detail = scrub((fail.get("message") or "") if fail is not None else "")
            out.append((name, res, secs, detail))
    return out


def parse_lasttest(text: str) -> list[tuple[str, str, float | None]]:
    """[(name, result, seconds)] from CTest's Testing/Temporary/LastTest.log."""
    out, name, secs = [], None, None
    captured = False
    nested_runs = 0
    lines = text.splitlines()
    for index, line in enumerate(lines):
        next_line = lines[index + 1] if index + 1 < len(lines) else ""
        if captured:
            # CTest writes child output verbatim, so it can contain complete nested logs.
            if line.startswith("Start testing: ") and next_line == "-" * 58:
                nested_runs += 1
            elif nested_runs and line.startswith("End testing: "):
                nested_runs -= 1
            elif not nested_runs and line == "<end of output>" and next_line.startswith("Test time ="):
                captured = False
            continue
        m = re.match(r"^\d+/\d+ Test: (.+)$", line)
        if m:
            name, secs = m.group(1).strip(), None
            continue
        if not name:
            continue
        if line == "Output:":
            captured = True
            continue
        m = re.match(r"^Test time =\s*([\d.]+) sec", line)
        if m:
            secs = float(m.group(1))
            continue
        if line.startswith("Test Passed"):
            res = "pass"
        elif line == "Test Pass Reason:" and next_line:
            res = "skip" if next_line.startswith("Skip regular expression found in output.") else "pass"
        elif line.startswith("Test Failed") or line == "Test Fail Reason:":
            res = "fail"
        elif line.startswith("Test Not Run"):
            res = "skip"
        else:
            continue
        out.append((name, res, secs))
        name = None
    return out


def collect_ctest(col, args):
    col.panel("runtime", "Runtime tests (CTest)", "tests")
    runtime_changed = last_commit_time("runtime")
    seen = 0
    log = os.path.join(ROOT, "runtime", "build", "Testing", "Temporary", "LastTest.log")
    sets = []
    if os.path.isfile(log):
        sets.append(("ctest", log, [(n, r, s, "") for n, r, s in parse_lasttest(read_utf8(log))], True))
        col.source("ctest_lasttest", log, "ok", len(sets[-1][2]))
    else:
        col.source("ctest_lasttest", log, "missing", note="no runtime/build CTest run on this machine")
    for path in args.junit or []:
        if not os.path.isfile(path):
            col.source("junit", path, "missing")
            continue
        try:
            root = ET.parse(path).getroot()
        except (ET.ParseError, LookupError, ValueError) as e:
            col.source("junit", path, "invalid", note=str(e))
            continue
        # CTest writes a single testsuite with a disabled count and hostname. pytest wraps its suite in testsuites.
        is_ctest = root.tag == "testsuite" and {"disabled", "hostname"} <= root.attrib.keys()
        rows = parse_junit(root)
        sets.append(("junit:" + os.path.basename(path), path, rows, is_ctest))
        col.source("junit", path, "ok", len(rows))
    for label, path, rows, is_ctest in sets:
        ran = os.path.getmtime(path)
        for name, res, secs, detail in rows:
            if res == "fail":
                fresh, status = "failed", "failed"
            elif res == "skip":
                fresh, status = "unknown", "unknown"
            elif is_ctest and runtime_changed and ran < runtime_changed:
                fresh, status = "revalidation_required", "partial"
            else:
                fresh, status = "current", "verified"
            seen += 1
            col.add(f"ctest.{label}.{name}", "runtime", label, name, status, detail=detail or None, freshness=fresh, seconds=secs,
                    ran_at=iso(ran))
    if seen:
        ok = sum(1 for c in col.components if c["panel"] == "runtime" and c["status"] == "verified")
        col.metric("runtime.current", "runtime", "Runtime tests passing on the current code", ok, seen,
                   "passing tests, excluding CTest reports older than the last commit touching runtime/ / tests recorded")


# ----------------------------------------------------------------------------------------------- GPU tests
GPU_LINE = re.compile(r"^(\S+)\s+(ok|SKIP \(.*\)|.+)$")
GPU_SUMMARY = re.compile(r"^(\d+) of (\d+) pass, (\d+) skipped")


def parse_gpu_log(text):
    """[(shader test, result, detail)] from the output of runtime/test/shaders/check.py (one line per test,
    "<name> ok", "<name> SKIP (why)" or "<name> <what came back>", then "N of M pass, K skipped")."""
    rows, summary = [], None
    for line in text.splitlines():
        m = GPU_SUMMARY.match(line.strip())
        if m:
            summary = tuple(int(x) for x in m.groups())
            continue
        m = GPU_LINE.match(line.rstrip())
        # check.py prints f"{name:20} {result}": the result starts right after the name padded to 20 columns,
        # which tells result lines from the other lines it prints (the build folder it picked, warnings)
        if not m or not re.fullmatch(r"[A-Za-z0-9_.]+", m.group(1)) or m.start(2) != max(len(m.group(1)), 20) + 1:
            continue
        name, rest = m.group(1), m.group(2).strip()
        if rest == "ok":
            rows.append((name, "pass", ""))
        elif rest.startswith("SKIP"):
            rows.append((name, "skip", rest))
        else:
            rows.append((name, "fail", scrub(rest)))
    return rows, summary


def collect_gpu(col, logs):
    col.panel("gpu", "GPU tests (hand-written RDNA2 programs)", "test programs")
    if not logs:
        col.source("gpu_logs", None, "missing", note="pass --gpu-log <saved check.py output>[=<GPU, driver, OS>]")
        return
    changed = max(t for t in (last_commit_time("runtime/test/shaders"), last_commit_time("tools/rdna2spirv.py"), 0) if t is not None)
    for spec in logs:
        path, _, label = spec.partition("=")
        if not os.path.isfile(path):
            col.source("gpu_log", path, "missing")
            continue
        rows, summary = parse_gpu_log(read_utf8(path))
        label = label or os.path.splitext(os.path.basename(path))[0]
        note = None
        if summary and summary[1] != len(rows):
            note = f"summary says {summary[1]} tests, {len(rows)} lines parsed"
        col.source("gpu_log", path, "ok" if rows else "invalid", len(rows), note=note)
        ran = os.path.getmtime(path)
        for name, res, detail in rows:
            if res == "fail":
                status, fresh = "failed", "failed"
            elif res == "skip":
                status, fresh = "unknown", "unknown"
            elif changed and ran < changed:
                status, fresh = "partial", "revalidation_required"
            else:
                status, fresh = "verified", "current"
            col.add(f"gpu.{label}.{name}", "gpu", label, name, status, detail=detail or None, freshness=fresh, ran_at=iso(ran))
    tested = [c for c in col.components if c["panel"] == "gpu" and c["status"] != "unknown"]
    if tested:
        col.metric("gpu.current", "gpu", "GPU test programs passing on the current translator", sum(c["status"] == "verified" for c in tested),
                   len(tested), "programs ok after the last commit touching runtime/test/shaders or tools/rdna2spirv.py / programs run (skips excluded)")


# ----------------------------------------------------------------------------------------------- docs
def md_tables(text):
    """[(heading, header_cells, rows, first_line)] for every pipe table in a Markdown text."""
    lines, out, heading, i = text.splitlines(), [], None, 0
    sep = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("#"):
            heading = ln.lstrip("#").strip()
        if ln.lstrip().startswith("|") and i + 1 < len(lines) and sep.match(lines[i + 1]):
            header = cells(ln)
            rows, j = [], i + 2
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                rows.append(cells(lines[j]))
                j += 1
            out.append((heading, header, rows, i + 1))
            i = j
            continue
        i += 1
    return out


def cells(line):
    s = line.strip().strip("|")
    parts, buf, tick = [], [], False
    for ch in s:
        if ch == "`":
            tick = not tick
        if ch == "|" and not tick:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf).strip())
    return parts


def plain(s):
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)
    return re.sub(r"\*\*|`", "", s).strip()


# How a contract row says it was confirmed. The first matching rule decides; the matched words are kept as evidence.
CONFIRMATION = [
    ("not_gpu", "partial", re.compile(r"not yet GPU-(?:verified|checked)|Unmeasured|still needs an? [^.]{0,40}(?:GPU|part)|no GPU measurement", re.I)),
    ("gpu", "verified", re.compile(r"GPU-checked|GPU-verified|measured on (?:an? )?(?:gfx\d+|AMD|NVIDIA|Intel|Apple|RX |RTX |GeForce|Radeon)|"
                                   r"run_gpu_tests|on an? (?:AMD|NVIDIA|Intel|Apple) [^.]{0,30}(?:GPU|Radeon|GeForce|RTX|RX|Arc|M\d)", re.I)),
    ("reference", "implemented", re.compile(r"Source: (?:LLVM|Mesa)|LLVM `|Mesa `|llvm-mc|ACO|RADV", re.I)),
    ("corpus", "implemented", re.compile(r"\bcorpus\b|of the 26,\d{3}|over (?:all|the) [\d,]+ shaders", re.I)),
    ("cpu", "implemented", re.compile(r"CPU-checked|CPU reference|unit test|tests/semantic|exact-arithmetic model", re.I)),
]
OPEN = re.compile(r"\bstill open\b|\bremains? open\b|\bopen items?\b", re.I)


def classify_contract(text):
    for kind, status, rx in CONFIRMATION:
        m = rx.search(text)
        if m:
            return kind, status, m.group(0)
    return "unstated", "unknown", None


def collect_contract(col):
    col.panel("contract", "Translator contract (docs/design.md)", "contract rows")
    path = os.path.join(ROOT, "docs", "design.md")
    if not os.path.isfile(path):
        col.source("contract", path, "missing")
        return
    tables = [t for t in md_tables(read_utf8(path)) if t[0] and "contract" in t[0].lower()]
    if not tables:
        col.source("contract", path, "invalid", note="no table under a 'contract' heading")
        return
    rows = tables[0][2]
    counts = {}
    for k, row in enumerate(rows):
        if len(row) < 2:
            continue
        piece, text = plain(row[0]), plain(" ".join(row[1:]))
        kind, status, words = classify_contract(text)
        if status == "verified" and OPEN.search(text):
            status = "partial"
            kind += "+open"
        counts[kind] = counts.get(kind, 0) + 1
        col.add(f"contract.{k:03d}", "contract", kind_label(kind), piece[:120], status,
                detail=(f"confirmation: “{words}”" if words else "no confirmation method stated in the row"),
                line=tables[0][3] + 2 + k, open_items=bool(OPEN.search(text)) or None)
    col.source("contract", path, "ok", len(rows))
    gpu = sum(1 for c in col.components if c["panel"] == "contract" and c["status"] == "verified")
    col.metric("contract.gpu", "contract", "Contract rows confirmed on a GPU, nothing left open", gpu, len(rows),
               "rows naming a GPU check and no open item / rows in the contract table")
    stated = sum(1 for c in col.components if c["panel"] == "contract" and c["status"] != "unknown")
    col.metric("contract.stated", "contract", "Contract rows that state how they were confirmed", stated, len(rows),
               "rows matching a confirmation rule (GPU, LLVM/Mesa reference, corpus count, CPU test) / rows")


def kind_label(kind):
    return {"gpu": "GPU-checked", "gpu+open": "GPU-checked, items open", "not_gpu": "Waiting for a GPU measurement",
            "reference": "LLVM/Mesa reference", "corpus": "Corpus count", "cpu": "CPU test only",
            "unstated": "Confirmation not stated"}.get(kind, kind)


def collect_imports(col):
    col.panel("imports", "Runtime imports (docs/runtime_imports.md)", "imports")
    path = os.path.join(ROOT, "docs", "runtime_imports.md")
    if not os.path.isfile(path):
        col.source("imports", path, "missing")
        return
    n = unimpl = 0
    for heading, header, rows, _ in md_tables(read_utf8(path)):
        h = [plain(x).lower() for x in header]
        if "binding" in h and "import" in h:
            bi, ii, ci = h.index("binding"), h.index("import"), h.index("consumer") if "consumer" in h else 0
            for row in rows:
                if len(row) <= max(bi, ii):
                    continue
                binding = plain(row[bi]).lower()
                status = "missing" if "unimpl" in binding or binding in ("none", "missing") else "implemented"
                unimpl += status == "missing"
                n += 1
                col.add(f"import.{plain(row[ci])}.{plain(row[ii])}", "imports", plain(row[ci]), plain(row[ii]), status,
                        detail=f"binding: {binding}")
        elif "unimplemented" in h and "image" in h:
            ui, ii = h.index("unimplemented"), h.index("image")
            for row in rows:
                if len(row) > ui and plain(row[ui]).replace(",", "").isdigit():
                    k = int(plain(row[ui]).replace(",", ""))
                    col.add(f"image.{plain(row[ii])}", "imports", "Images (unimplemented imports)", plain(row[ii]),
                            "verified" if k == 0 else "missing", detail=f"{k} unimplemented import(s), as measured in the document")
    col.source("imports", path, "ok", n)
    if n:
        col.metric("imports.bound", "imports", "Measured imports with a binding", n - unimpl, n,
                   "imports bound to a shim, the host or a shipped module / imports listed")


# ----------------------------------------------------------------------------------------------- corpus (counts only)
def collect_corpus(col, work):
    col.panel("corpus", "Shader corpus (your own run; counts only)", "shaders, aggregated per stage", weighted=True)
    if not work:
        col.source("corpus", None, "missing", note="pass --work <work> to count your own translation run")
        return
    # make_game.py writes <work>/shaders/dxbc and <work>/shaders/spirv; the shaders folder itself is accepted too
    shaders = os.path.join(work, "shaders") if os.path.isdir(os.path.join(work, "shaders", "dxbc")) else work
    carved_p, spirv_p = os.path.join(shaders, "dxbc", "index.json"), os.path.join(shaders, "spirv", "index.json")
    if not os.path.isfile(carved_p):
        col.source("corpus", None, "missing", note="no <work>/shaders/dxbc/index.json (run make_game.py's shaders stage)")
        return
    carved = json.loads(read_utf8(carved_p))
    entries = carved if isinstance(carved, list) else list(carved.values())
    per_stage = {}
    for e in entries:
        st = (e.get("stage") if isinstance(e, dict) else None) or "unknown"
        per_stage.setdefault(st, [0, 0])[0] += 1
    validated_total = 0
    if os.path.isfile(spirv_p):
        spirv = json.loads(read_utf8(spirv_p))
        for meta in spirv.values():
            if isinstance(meta, dict) and "error" not in meta:
                st = meta.get("stage") or "unknown"
                per_stage.setdefault(st, [0, 0])[1] += 1
                validated_total += 1
        col.source("corpus_spirv", "<work>/shaders/spirv/index.json", "ok", len(spirv))
    else:
        col.source("corpus_spirv", "<work>/shaders/spirv/index.json", "missing", note="translation not finished")
    col.source("corpus_carved", "<work>/shaders/dxbc/index.json", "ok", len(entries))
    for st, (n_carved, n_valid) in sorted(per_stage.items()):
        stage = str(st)
        if n_valid:
            col.add(f"corpus.{stage}.validated", "corpus", stage, f"{stage}: pass spirv-val", "implemented",
                    detail=f"{n_valid:,} of {n_carved:,} carved", weight=n_valid, count=n_valid)
        rest = max(n_carved - n_valid, 0)
        if rest:
            col.add(f"corpus.{stage}.not_validated", "corpus", stage, f"{stage}: not translated or rejected by spirv-val",
                    "failed", detail=f"{rest:,} of {n_carved:,} carved", weight=rest, count=rest)
    col.metric("corpus.validated", "corpus", "Carved shaders that translate and pass spirv-val", validated_total, len(entries),
               "entries in <work>/shaders/spirv/index.json without an error / entries in <work>/shaders/dxbc/index.json",
               note="validation is not execution: nothing here says a shader's output is right")
    col.notes.append("corpus: counts come from your own work folder; no shader name or path is stored")


# ----------------------------------------------------------------------------------------------- github (read-only)
def gh_json(args, timeout=60):
    out = run(["gh"] + args, timeout=timeout)
    if out is None:
        return None
    try:
        return json.loads(out)
    except ValueError:
        return None


def d3d_state(owner, pr, status_text, merged_prs=None, open_prs=None):
    """State of one D3D- row. With the repository's merged and open pull-request numbers, a PR the table still shows
    as open but already merged counts as merged, and one closed without merging is dropped; without them the
    table's own words decide."""
    refs = re.findall(r"#(\d+)((?:\s+(?:merged|closed|on main))?)", pr)
    said_merged = {int(r) for r, suffix in refs if "merged" in suffix or "on main" in suffix}
    bare = [int(r) for r, suffix in refs if not suffix.strip()]
    text_open = "PR open" in status_text or "draft" in status_text.lower()
    if merged_prs is not None and open_prs is not None:
        merged = said_merged | {r for r in bare if r in merged_prs}
        still_open = [r for r in bare if r in open_prs]
        text_open = text_open and not refs
    else:
        merged, still_open = said_merged, bare
    if still_open or text_open:
        return "partial", "pull request open"
    if merged:
        return "implemented", "merged on main"
    if owner and owner not in ("—", "-", ""):
        return "unknown", "claimed, no code yet"
    return "missing", "unclaimed"


PR_LIMIT = 5000
ITEM_ID = re.compile(r"\b([A-Z][A-Z0-9]{0,9})-(\d+)\b")
NOT_ITEMS = {"SHA", "UTF", "IEEE", "ISO", "RFC", "CRC", "CVE", "MD", "BC", "X", "R", "G", "B"}


def item_ids(title):
    """Coordination item IDs named in a pull-request title ("D3D-86: ...", "... (SHADER-339)"), in order, once each.
    A prefix ending in a digit (D3D12-, OPTIONS2-) or a known non-item word (SHA-256, UTF-8) is not an item."""
    out = []
    for prefix, num in ITEM_ID.findall(title or ""):
        if prefix in NOT_ITEMS or prefix[-1].isdigit():
            continue
        if f"{prefix}-{num}" not in out:
            out.append(f"{prefix}-{num}")
    return out


def d3d_item_ids(pulls):
    """D3D- item IDs named by the titles of these pull requests."""
    return {i for p in pulls for i in item_ids(p.get("title")) if i.startswith("D3D-")}


def collect_pulls(col, repo, merged, open_prs, limit=PR_LIMIT):
    """Every pull request into main: merged is IMPLEMENTED, open is PARTIAL; closed without merging is left out.
    Grouped by the item family its title names (D3D, SHADER, BUILD, ...)."""
    col.panel("merged", f"Pull requests into main ({repo}): merged = implemented", "pull requests")
    note = f"{len(merged)} merged, {len(open_prs)} open"
    if len(merged) >= limit or len(open_prs) >= limit:
        note += f"; the {limit}-PR limit was reached, older pull requests are not counted"
    col.source("github_prs", None, "ok", len(merged) + len(open_prs), note=note)
    items_merged, items_any = set(), set()
    for p, status in [(p, "implemented") for p in merged] + [(p, "partial") for p in open_prs]:
        ids = item_ids(p.get("title"))
        items_any.update(ids)
        if status == "implemented":
            items_merged.update(ids)
        group = (ids[0].split("-")[0] + " items") if ids else "no item ID"
        if status == "implemented":
            detail = f"merged on main {p.get('mergedAt') or ''}".strip()
        else:
            detail = "pull request open" + (" (draft)" if p.get("isDraft") else "")
        col.add(f"pr.{p['number']}", "merged", group, f"#{p['number']} {p.get('title') or ''}"[:120], status, detail=detail,
                number=p["number"], items=ids, merged_at=p.get("mergedAt") if status == "implemented" else None,
                author=(p.get("author") or {}).get("login"))
    col.metric("merged.prs", "merged", "Pull requests into main that are merged", len(merged), len(merged) + len(open_prs),
               "pull requests merged into main / merged + open (closed without merging are left out)")
    if items_any:
        col.metric("merged.items", "merged", "Coordination items with a merged pull request", len(items_merged), len(items_any),
                   "item IDs (D3D-, SHADER-, BUILD-, ...) named by a merged PR title / item IDs named by a merged or open PR title")
    latest = sorted(merged, key=lambda p: p.get("mergedAt") or "", reverse=True)[:30]
    return [{"number": p["number"], "title": (p.get("title") or "")[:120], "merged_at": p.get("mergedAt"),
             "items": item_ids(p.get("title")), "author": (p.get("author") or {}).get("login")} for p in latest]


def collect_github(col, repo, login):
    col.panel("d3d12", "D3D12 → Vulkan layer items (#313)", "D3D- items")
    col.panel("ci", "CI on main", "workflows")
    if not shutil.which("gh"):
        col.source("github", None, "missing", note="gh is not installed")
        return {}
    summary = {"repo": repo}
    # every pull request into main, so a merge shows up on the next run even before a tracking table is edited
    merged = gh_json(["pr", "list", "-R", repo, "--state", "merged", "--base", "main", "--limit", str(PR_LIMIT),
                      "--json", "number,title,mergedAt,author"], timeout=300)
    open_prs = gh_json(["pr", "list", "-R", repo, "--state", "open", "--base", "main", "--limit", str(PR_LIMIT),
                        "--json", "number,title,isDraft,author"], timeout=300)
    merged_set = open_set = None
    if merged is None or open_prs is None:
        col.panel("merged", f"Pull requests into main ({repo}): merged = implemented", "pull requests")
        col.source("github_prs", None, "invalid", note="gh pr list failed (not logged in?)")
    else:
        summary["merged_latest"] = collect_pulls(col, repo, merged, open_prs)
        merged_set, open_set = {p["number"] for p in merged}, {p["number"] for p in open_prs}
    runs = gh_json(["run", "list", "-R", repo, "--branch", "main", "--limit", "40",
                    "--json", "workflowName,conclusion,status,headSha,createdAt,url"])
    if runs is None:
        col.source("github_ci", None, "invalid", note="gh run list failed (not logged in?)")
    else:
        latest, pending = {}, {}
        for r in runs:      # newest first: the first completed run per workflow decides; a newer one still running is noted
            if r["status"] == "completed":
                latest.setdefault(r["workflowName"], r)
            elif r["workflowName"] not in latest:
                pending.setdefault(r["workflowName"], r)
        for name in sorted(set(latest) | set(pending)):
            r, newer = latest.get(name), pending.get(name)
            if r is None:
                status, what, r = "unknown", f"no completed run in the last 40; one {newer['status']}", newer
            else:
                status, what = ("verified" if r["conclusion"] == "success" else "failed"), r["conclusion"]
            col.add(f"ci.{name}", "ci", "latest completed run per workflow", name, status,
                    detail=f"{what} on {r['headSha'][:10]}" + (f"; a newer run is {newer['status']}" if newer and newer is not r else ""),
                    at=r["createdAt"], url=r.get("url"))
        col.source("github_ci", None, "ok", len(latest))
    issue = gh_json(["issue", "view", "313", "-R", repo, "--json", "body,updatedAt"])
    if issue:
        n = 0
        table_ids, table_done = set(), set()
        for heading, header, rows, _ in md_tables(issue["body"]):
            h = [plain(x).lower() for x in header]
            if h[:1] != ["id"] or "pr" not in h:
                continue
            for row in rows:
                if not plain(row[0]).startswith("D3D-") or len(row) < len(h):
                    continue
                owner, pr, st = plain(row[h.index("owner")]), plain(row[h.index("pr")]), plain(row[h.index("status")]) if "status" in h else ""
                status, why = d3d_state(owner, pr, st, merged_set, open_set)
                table_ids.add(plain(row[0]))
                if status == "implemented":
                    table_done.add(plain(row[0]))
                n += 1
                col.add(f"d3d.{plain(row[0])}", "d3d12", why, f"{plain(row[0])} {plain(row[1])}"[:120], status,
                        detail=(st or why)[:200], owner=owner, prs=pr[:120])
        col.source("github_313", None, "ok", n, note=f"issue updated {issue.get('updatedAt')}")
        if n:
            # the table lists only items still in progress; a merged item leaves it, so the merges are counted where they are recorded
            lists_read = merged_set is not None
            merged_ids = d3d_item_ids(merged) if lists_read else set()
            open_ids = d3d_item_ids(open_prs) if lists_read else set()
            pr_done = (merged_ids - table_ids) - open_ids    # rows in #313 keep the table's state; the rest need a merged PR and no open one
            col.metric("d3d12.merged", "d3d12", "D3D- items merged on main" + ("" if lists_read else " (partial)"),
                       len(table_done | pr_done), len(table_ids | merged_ids | open_ids),
                       "#313 rows whose state is implemented (d3d_state), plus D3D- IDs not in #313 that a merged PR title names "
                       "and no open PR title names / all known D3D- IDs (#313 rows plus IDs named by any merged or open PR title)",
                       note=None if lists_read else "partial: the merged or open PR list could not be read, "
                                                     "so only the #313 rows are counted")
    coord = gh_json(["issue", "list", "-R", repo, "--state", "open", "--search", "Coordinating who works on what in:title",
                     "--json", "number,title,updatedAt"])
    if coord:
        num = coord[0]["number"]
        body = gh_json(["issue", "view", str(num), "-R", repo, "--json", "body"]) or {}
        text = body.get("body", "")
        sizes = {}
        for sec in ("Up for grabs", "Ready for review or a GPU test", "Needs a rebase on `main`"):
            m = re.search(r"### " + re.escape(sec) + r"(.*?)(?=\n### |\Z)", text, re.S)
            if m:
                part = m.group(1)
                sizes[plain(sec)] = (len(re.findall(r"^\|\s*(?:⭐\s*)?[A-Z]+-\d+", part, re.M)) if sec == "Up for grabs"
                                     else len(set(re.findall(r"#(\d+)", part))))
        summary["coordination"] = {"issue": num, "updated": coord[0]["updatedAt"], "sizes": sizes}
        col.source("github_coordination", None, "ok", len(sizes), note=f"issue #{num}")
        if login:
            # claims made in the issue this one continues still count (its own first paragraph says so)
            prev = re.search(r"continues #(\d+)", text)
            claims = []
            for n in [num] + ([int(prev.group(1))] if prev else []):
                comments = gh_json(["issue", "view", str(n), "-R", repo, "--json", "comments"]) or {}
                claims += [{"at": c.get("createdAt"), "issue": n, "text": plain(c["body"].splitlines()[0])[:160]}
                           for c in comments.get("comments", [])
                           if c.get("author", {}).get("login") == login and "claiming" in c["body"].lower()]
            prs = gh_json(["pr", "list", "-R", repo, "--author", login, "--state", "all", "--limit", "50",
                           "--json", "number,title,state,isDraft,updatedAt"]) or []
            summary["mine"] = {"login": login, "claims": claims,
                               "pull_requests": [{"number": p["number"], "title": p["title"][:120], "state": p["state"],
                                                  "draft": p.get("isDraft"), "updated": p.get("updatedAt")} for p in prs]}
    return summary


# ----------------------------------------------------------------------------------------------- snapshot + page
def counts_by_panel(components):
    out = {}
    for c in components:
        p = out.setdefault(c["panel"], {s: 0 for s in STATUSES})
        p[c["status"]] += 1
    return out


def build(args, out_dir):
    col = Collector()
    g = git_info()
    collect_unit(col, out_dir, args)
    collect_ctest(col, args)
    collect_gpu(col, args.gpu_log)
    collect_contract(col)
    collect_imports(col)
    collect_corpus(col, args.work)
    gh = collect_github(col, args.repo, args.login) if args.github else {}
    if not args.github:
        col.source("github", None, "skipped", note="pass --github to read pull requests into main, CI, #313 and the coordination issue")
    snap = {
        "schema": SCHEMA, "generated_at": iso(now_utc()), "git": g,
        "panels": [col.panels[k] for k in ("tests", "runtime", "gpu", "contract", "imports", "corpus", "merged", "d3d12", "ci") if k in col.panels],
        "counts": counts_by_panel(col.components), "components": col.components, "metrics": col.metrics,
        "sources": col.sources, "notes": col.notes, "github": gh,
    }
    body = {k: v for k, v in snap.items() if k != "generated_at"}
    snap["content_hash"] = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return snap


def summary_line(snap):
    return {"generated_at": snap["generated_at"], "head": (snap["git"].get("head") or "")[:10], "content_hash": snap["content_hash"][:16],
            "metrics": {m["id"]: {"v": m["value"], "n": m["numerator"], "d": m["denominator"], "f": m["definition"]} for m in snap["metrics"]}}


def logo_src():
    """The project logo (assets/logo.png) as a data: URI, so the page stays one file that opens from disk; empty when
    the asset is absent, and the page then shows no image."""
    try:
        with open(LOGO, "rb") as f:
            data = f.read()
    except OSError:
        return ""
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii") if data[:8] == b"\x89PNG\r\n\x1a\n" else ""


def render_page(snap, history):
    template = read_utf8(TEMPLATE)
    payload = json.dumps({"snapshot": snap, "history": history}, ensure_ascii=False, sort_keys=True)
    payload = payload.replace("</", "<\\/")       # a document string can never close the script element
    return template.replace("/*__PROGRESS_LOGO__*/", logo_src()).replace("/*__PROGRESS_DATA__*/null", payload)


def write_outputs(snap, out_dir):
    os.makedirs(os.path.join(out_dir, "snapshots"), exist_ok=True)
    hist_path = os.path.join(out_dir, "history.json")
    history = json.loads(read_utf8(hist_path)) if os.path.isfile(hist_path) else []
    new = not history or history[-1]["content_hash"] != snap["content_hash"][:16]
    if new:
        stamp = snap["generated_at"].replace("-", "").replace(":", "")
        save_text(os.path.join(out_dir, "snapshots", f"progress_{stamp}.json"), dump_json(snap))
        history.append(summary_line(snap))
        save_text(hist_path, dump_json(history))
    save_text(os.path.join(out_dir, "progress.html"), render_page(snap, history))
    return new


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output folder, outside the repository")
    ap.add_argument("--work", help="make_game.py work folder, for corpus counts (numbers only)")
    ap.add_argument("--run-tests", action="store_true", help="run the unit tests now (CI selection) and keep the outcome")
    ap.add_argument("--tests", metavar="REGEX", help="with --run-tests, only the test files matching REGEX")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2), help="test processes at a time")
    ap.add_argument("--test-timeout", type=int, default=300, help="seconds per test file")
    ap.add_argument("--junit", nargs="*", metavar="FILE", help="CTest --output-junit or pytest --junitxml files")
    ap.add_argument("--gpu-log", nargs="*", metavar="FILE[=LABEL]",
                    help="saved output of runtime/test/shaders/check.py; LABEL names the GPU, driver and OS")
    ap.add_argument("--github", action="store_true", help="read pull requests into main, CI runs, #313 and the coordination issue through gh")
    ap.add_argument("--repo", default=DEFAULT_REPO, help=f"GitHub repository (default {DEFAULT_REPO})")
    ap.add_argument("--login", help="with --github, show this account's claims and pull requests")
    args = ap.parse_args(argv)
    if args.jobs < 1:
        ap.error("--jobs must be a positive integer")
    if args.test_timeout < 1:
        ap.error("--test-timeout must be a positive integer")
    if args.run_tests and args.tests:
        try:
            args.tests = re.compile(args.tests)
        except (re.error, OverflowError, RecursionError) as exc:
            ap.error(f"--tests: invalid regular expression: {exc}")
    out_dir = os.path.abspath(args.out)
    if inside(out_dir, ROOT):
        print(f"--out must be outside the repository ({ROOT}): the outputs are never meant to be committed", file=sys.stderr)
        return 2
    if args.work and not os.path.isdir(args.work):
        print(f"--work {args.work} is not a folder", file=sys.stderr)
        return 2
    if args.run_tests:
        args.test_files = [f for f in ci_test_files() if not args.tests or re.search(args.tests, f)]
        if args.tests and not args.test_files:
            ap.error("--tests did not match any test files")
    if os.path.exists(out_dir) and not os.path.isdir(out_dir):
        print(f"--out {args.out} is not a directory", file=sys.stderr)
        return 2
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as exc:
        print(f"--out {args.out}: {exc.strerror or exc}", file=sys.stderr)
        return 2
    try:
        snap = build(args, out_dir)
    except OSError as exc:
        print(f"error: {exc.filename or ''}: {exc.strerror or exc}", file=sys.stderr)
        return 2
    new = write_outputs(snap, out_dir)
    n = len(snap["components"])
    print(f"{n} units, {len(snap['metrics'])} metrics -> {os.path.join(out_dir, 'progress.html')}"
          + ("" if new else " (no change since the last snapshot)"))
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
