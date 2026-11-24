r"""Run the translator regression suite (tests/semantic) one process per test file, several files at a time.

    python -I scripts/run_semantic_tests.py tests/semantic          # one file per core
    python -I scripts/run_semantic_tests.py tests/semantic -j 4     # four at a time
    python -I scripts/run_semantic_tests.py tests/semantic -v       # also print the output of the files that pass

It runs the files `python -m unittest discover -s tests/semantic` finds (`test*.py`, and packages below it), each as
`python -m unittest <module>` from that folder, so a file sees the same import path as under discover. The files can
run side by side because each one writes only into its own temporary folder; a new test must do the same. With -j 1,
or on a single core, it runs that discover command itself. Prints the output of every failing file and one summary
line, and exits 1 if a file fails or none is found. Standard library only.
"""
import argparse
import concurrent.futures
import fnmatch
import os
import re
import subprocess
import sys
import time


def test_modules(directory):
    """Dotted names of the test modules unittest discover finds under `directory` (it enters packages only)."""
    names = []
    for base, dirs, files in os.walk(directory):
        dirs[:] = sorted(d for d in dirs if os.path.isfile(os.path.join(base, d, "__init__.py")))
        prefix = os.path.relpath(base, directory).replace(os.sep, ".")
        for name in sorted(files):
            if fnmatch.fnmatchcase(name, "test*.py"):
                names.append(name[:-3] if prefix == "." else f"{prefix}.{name[:-3]}")
    return names


def run_module(directory, module, verbose):
    """(return code, tests run, output) of `python -m unittest <module>` run from `directory`."""
    cmd = [sys.executable, "-m", "unittest", *(["-v"] if verbose else []), module]
    p = subprocess.run(cmd, cwd=directory, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                       errors="replace")
    ran = re.search(r"^Ran (\d+) tests? in", p.stdout, re.M)
    return p.returncode, int(ran.group(1)) if ran else 0, p.stdout


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("directory", help="the suite's folder, tests/semantic")
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 1, help="files at a time (default: cores)")
    ap.add_argument("-v", "--verbose", action="store_true", help="unittest -v, and print passing files' output")
    a = ap.parse_args(argv)
    directory = os.path.abspath(a.directory)
    if a.jobs <= 1:
        return subprocess.call([sys.executable, "-m", "unittest", "discover", "-s", directory,
                                *(["-v"] if a.verbose else [])])
    modules = test_modules(directory)
    if not modules:
        print(f"no test*.py files in {directory}")
        return 1
    # Largest files first: they tend to run longest, and one started last would set the wall time alone.
    modules.sort(key=lambda m: -os.path.getsize(os.path.join(directory, *m.split(".")) + ".py"))
    start, total, failed = time.monotonic(), 0, []
    with concurrent.futures.ThreadPoolExecutor(a.jobs) as pool:
        jobs = {pool.submit(run_module, directory, m, a.verbose): m for m in modules}
        for job in concurrent.futures.as_completed(jobs):
            code, ran, out = job.result()
            total += ran
            if code:
                failed.append(jobs[job])
                print(f"==== FAILED: {jobs[job]}\n{out}", flush=True)
            elif a.verbose:
                print(f"==== {jobs[job]}\n{out}", flush=True)
    took = time.monotonic() - start
    summary = f"{len(modules)} files, {total} tests, {len(failed)} files failed in {took:.0f} s with {a.jobs} jobs"
    print(summary + (": " + ", ".join(sorted(failed)) if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
