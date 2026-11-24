"""A process pool gets at most 61 workers on Windows, in every tool that takes --jobs for one.

    python -I tests/test_pool_workers.py

Above 61 workers concurrent.futures.ProcessPoolExecutor raises `ValueError: max_workers must be <= 61` on Windows
and multiprocessing.Pool hangs (tools/pool_workers.py has the measurement). awc_opus.py, awc_shorts.py,
world_lights_paths.py, world_build.py, anim_clips.py and anim_glb.py (--coverage) handed --jobs to a pool as it was.

Each tool is run through its own code three ways: with the limit patched in and the pool class replaced by one
that records its size and runs the work in this process (so the cap is checked on every platform); the two audio
tools from the command line with --jobs 62; and, on Windows, with 62 jobs and the real pool. Every input is built
here: an empty folder, a file of zeros, or the synthetic archives of tests/test_anim_clips.py and
tests/test_anim_glb.py.
"""
import argparse
import concurrent.futures
import multiprocessing
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
sys.path.insert(0, TOOLS)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import anim_clips  # noqa: E402
import anim_glb  # noqa: E402
import awc_opus  # noqa: E402
import awc_shorts  # noqa: E402
import pool_workers  # noqa: E402
import world_build  # noqa: E402
import world_lights_paths  # noqa: E402
from test_anim_clips import two_sequence_dictionary, write_rpf  # noqa: E402
from test_anim_glb import joaat32, part_resource, skeleton_resource, write_archive  # noqa: E402

WINDOWS_LIMIT = 61
SIZES = []          # the size each recording pool was created with


class RecordingExecutor:
    """Stands in for ProcessPoolExecutor: notes its size and runs each job at once, in this process."""

    def __init__(self, max_workers=None):
        SIZES.append(max_workers)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, fn, *args, **kwargs):
        future = concurrent.futures.Future()
        future.set_result(fn(*args, **kwargs))
        return future

    def map(self, fn, jobs):
        return [fn(job) for job in jobs]


class RecordingPool:
    """Stands in for multiprocessing.Pool the same way."""

    def __init__(self, processes=None):
        SIZES.append(processes)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def imap_unordered(self, fn, jobs, chunksize=1):
        return [fn(job) for job in jobs]


def quiet(fn, *args, **kwargs):
    with open(os.devnull, "w", encoding="utf-8") as sink, mock.patch.object(sys, "stdout", sink):
        return fn(*args, **kwargs)


class ProcessWorkersTest(unittest.TestCase):
    def test_limit_is_the_platforms(self):
        self.assertEqual(pool_workers.MAX_PROCESS_WORKERS, WINDOWS_LIMIT if os.name == "nt" else None)

    def test_jobs_are_lowered_only_above_the_limit(self):
        with mock.patch.object(pool_workers, "MAX_PROCESS_WORKERS", WINDOWS_LIMIT):
            self.assertEqual([pool_workers.process_workers(n) for n in (1, 2, 60, 61, 62, 64, 4096)],
                             [1, 2, 60, 61, 61, 61, 61])
        with mock.patch.object(pool_workers, "MAX_PROCESS_WORKERS", None):
            self.assertEqual([pool_workers.process_workers(n) for n in (1, 61, 62, 4096)], [1, 61, 62, 4096])


class ToolPoolSizeTest(unittest.TestCase):
    """Each tool, asked for more jobs than the limit, creates its pool at the limit and still does the work."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name
        SIZES.clear()
        patcher = mock.patch.object(pool_workers, "MAX_PROCESS_WORKERS", WINDOWS_LIMIT)
        patcher.start()
        self.addCleanup(patcher.stop)

    def path(self, name):
        return os.path.join(self.root, name)

    def test_awc_opus(self):
        os.mkdir(self.path("banks"))
        with mock.patch.object(awc_opus, "ProcessPoolExecutor", RecordingExecutor):
            for jobs, size in ((2, 2), (61, 61), (62, 61), (200, 61)):
                self.assertEqual(quiet(awc_opus.main, [self.path("banks"), self.path("ogg"), "--jobs", str(jobs)]), 0)
                self.assertEqual(SIZES.pop(), size)

    def test_awc_shorts(self):
        os.mkdir(self.path("banks"))
        with mock.patch.object(awc_shorts, "ProcessPoolExecutor", RecordingExecutor):
            for jobs, size in ((2, 2), (62, 61), (200, 61)):
                args = [self.path("banks"), self.path("shorts.csv"), "--jobs", str(jobs)]
                self.assertEqual(quiet(awc_shorts.main, args), 0)
                self.assertEqual(SIZES.pop(), size)

    def test_world_lights_paths(self):
        with open(self.path("zeros.rpf"), "wb") as f:
            f.write(bytes(4096))
        one = world_lights_paths.scan(self.path("zeros.rpf"), jobs=1)
        with mock.patch.object(world_lights_paths, "ProcessPoolExecutor", RecordingExecutor):
            for jobs, size in ((2, 2), (62, 61), (200, 61)):
                self.assertEqual(world_lights_paths.scan(self.path("zeros.rpf"), jobs=jobs), one)
                self.assertEqual(SIZES.pop(), size)

    def test_world_build(self):
        with mock.patch.object(concurrent.futures, "ProcessPoolExecutor", RecordingExecutor):
            for workers, size in ((2, 2), (62, 61), (200, 61)):
                self.assertEqual(list(world_build._run_jobs(abs, iter(range(-300, 0)), workers)),
                                 list(range(300, 0, -1)))
                self.assertEqual(SIZES.pop(), size)

    def test_anim_clips(self):
        rpf = self.path("synthetic.rpf")
        write_rpf(rpf, [two_sequence_dictionary("first@dict"), two_sequence_dictionary("second@dict")])
        rows, one = anim_clips.scan(rpf, jobs=1)
        self.assertEqual(len(rows), 2)                  # two work items, so jobs > 1 does open a pool
        self.assertEqual(SIZES, [])
        with mock.patch.object(multiprocessing, "Pool", RecordingPool):
            for jobs, size in ((2, 2), (62, 61), (200, 61)):
                _, many = anim_clips.scan(rpf, jobs=jobs)
                self.assertEqual(many["counts"], one["counts"])
                self.assertEqual(SIZES.pop(), size)


    def test_anim_glb_coverage(self):
        packs, anim = self.path("packs.rpf"), self.path("anim.rpf")
        skin = [[(0, 200), (1, 55)], [(1, 255)], [(0, 128), (1, 127)], [(1, 255)]]
        palette = [joaat32("skel_c"), joaat32("skel_a")]
        skeleton = write_archive(packs, [[skeleton_resource()], list(part_resource(palette, "u8x4", skin))])[0]
        write_rpf(anim, [two_sequence_dictionary("first@dict"), two_sequence_dictionary("second@dict")])

        def coverage(jobs):
            args = argparse.Namespace(archive=packs, skeleton=(skeleton, None), anim=anim, jobs=jobs)
            return anim_glb.coverage(args)["counts"]

        one = coverage(1)
        self.assertEqual((one["dictionaries"], one["bone_tracks_in_skeleton"]), (2, 6))
        self.assertEqual(SIZES, [])
        with mock.patch.object(concurrent.futures, "ProcessPoolExecutor", RecordingExecutor):
            for jobs, size in ((2, 2), (61, 61), (62, 61), (200, 61)):
                self.assertEqual(coverage(jobs), one)
                self.assertEqual(SIZES.pop(), size)


class CommandLineTest(unittest.TestCase):
    """The audio tools default --jobs to half the CPU count and make_game.py hands its own to awc_opus.py."""

    def run_tool(self, tool, *args):
        return subprocess.run([sys.executable, "-I", os.path.join(TOOLS, tool), *args], capture_output=True,
                              text=True, encoding="utf-8", errors="replace")

    def test_62_jobs_is_not_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            banks = os.path.join(d, "banks")
            os.mkdir(banks)
            for tool, out in (("awc_opus.py", os.path.join(d, "ogg")), ("awc_shorts.py", os.path.join(d, "s.csv"))):
                with self.subTest(tool=tool):
                    r = self.run_tool(tool, banks, out, "--jobs", "62")
                    self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
                    self.assertNotIn("Traceback", r.stderr)
                    self.assertIn("0 banks", r.stdout)


@unittest.skipUnless(os.name == "nt", "the 61-worker limit is Windows's; elsewhere 62 workers always worked")
class RealPoolOnWindowsTest(unittest.TestCase):
    """62 jobs through the real ProcessPoolExecutor, which starts a worker per job it is given, not all 61."""

    def test_world_build_run_jobs(self):
        self.assertEqual(list(world_build._run_jobs(abs, [-1, -2, -3], 62)), [1, 2, 3])

    def test_world_lights_paths_scan(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "zeros.rpf")
            with open(path, "wb") as f:
                f.write(bytes(4096))
            self.assertEqual(world_lights_paths.scan(path, jobs=62), world_lights_paths.scan(path, jobs=1))


if __name__ == "__main__":
    unittest.main()
