"""How many worker processes one pool can have on this platform.

Windows waits on a pool's workers with WaitForMultipleObjects, which takes at most 64 handles. Above 61 workers
concurrent.futures.ProcessPoolExecutor raises `ValueError: max_workers must be <= 61`, and multiprocessing.Pool
prints `ValueError: need at most 63 handles` from its result thread and never returns (measured on Windows 11,
Python 3.12.5: 62 workers were killed after 60 s, 61 finished in 0.8 s). A tool whose --jobs defaults to half the
CPU count, or is handed make_game.py's, reaches that on a large machine with no option given.

A pool is therefore created with process_workers(jobs): the same number everywhere else, 61 at most on Windows.
Threads have no such limit. tools/world_census.py rejects --jobs above the limit instead (its MAX_JOBS).
"""
import os

MAX_PROCESS_WORKERS = 61 if os.name == "nt" else None


def process_workers(jobs):
    """`jobs`, lowered to the most worker processes one pool can have here."""
    return jobs if MAX_PROCESS_WORKERS is None else min(jobs, MAX_PROCESS_WORKERS)
