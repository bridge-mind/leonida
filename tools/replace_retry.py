"""os.replace() that rides out the short windows in which Windows refuses to rename a file it has just seen written.

On Windows, os.replace raises PermissionError (winerror 5 or 32) while another process holds the source or the
destination open without share-delete: a virus scanner, the search indexer, a cloud-sync client on Documents. The
hold lasts milliseconds to a few seconds, so the rename is tried again for a bounded time and only then re-raised.
Elsewhere, and for every other error, this is os.replace.
"""
import os
import time

RETRY_SECONDS = 5.0
FIRST_DELAY = 0.01
MAX_DELAY = 0.25
_TRANSIENT_WINERRORS = (5, 32)    # ERROR_ACCESS_DENIED, ERROR_SHARING_VIOLATION


def _transient(error):
    return isinstance(error, PermissionError) and getattr(error, "winerror", None) in _TRANSIENT_WINERRORS


def replace(src, dst, *, seconds=None):
    """os.replace(src, dst), retried with a doubling delay for `seconds` (default RETRY_SECONDS) on a transient
    Windows PermissionError; the last error is raised once the time is up."""
    deadline = time.monotonic() + (RETRY_SECONDS if seconds is None else seconds)
    delay = FIRST_DELAY
    while True:
        try:
            return os.replace(src, dst)
        except PermissionError as e:
            if not _transient(e) or time.monotonic() >= deadline:
                raise
        time.sleep(min(delay, max(0.0, deadline - time.monotonic())))
        delay = min(delay * 2, MAX_DELAY)
