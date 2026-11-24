"""Do two report paths name the same file? One answer for the tools that refuse an output that would overwrite
an input or another output.

    from report_paths import same_report_path        # after sys.path.insert(0, <this folder>), as -I needs

`same_report_path(first, second)` is true when the two paths resolve to one name (`realpath`, then `normcase`, and on a
case-insensitive macOS volume also case-folded and with composed and decomposed accents treated alike) or when both
exist and `os.path.samefile` says they are one file (a hard link, another spelling the name rules missed). Neither
path has to exist: two report paths that don't exist yet still collide when they differ only in case on a volume that
ignores case. Where the name rules cannot tell, `same_name_on_folding_volume` asks the volume itself with one
scratch folder: a Linux mount that folds case (vfat, exFAT, CIFS) answers neither `normcase` nor `pathconf`.
Standard library only; no game data.
"""
import os
import posixpath
import sys
import tempfile
import unicodedata

_PC_CASE_SENSITIVE = 11  # <sys/unistd.h> on macOS; os.pathconf_names doesn't list it


def case_insensitive_volume(path):
    """True when the volume that will hold `path` ignores letter case, as macOS's default APFS volume does.
    os.path.normcase folds case only on Windows, so on macOS `R.json` and `r.json` would pass as two reports."""
    if sys.platform != "darwin":
        return False
    while not os.path.exists(path):
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    try:
        return os.pathconf(path, _PC_CASE_SENSITIVE) == 0
    except (OSError, ValueError):
        return True  # the macOS default; folding can only refuse a pair, never overwrite one


def path_key(path):
    """A name two spellings of one path share: realpath, normcase, and on a case-insensitive volume case-folded NFD."""
    return _resolved_key(os.path.realpath(path))


def _resolved_key(path):
    """`path_key` for a path `os.path.realpath` has already resolved."""
    # Use the path rules of the platform we are resolving for. On macOS os.path is
    # posixpath and keeps case; on Windows ntpath lowercases. Resolving a simulated
    # macOS path through ntpath would hide the case-sensitive branch under test.
    path = posixpath.normcase(path) if sys.platform == "darwin" else os.path.normcase(path)
    if case_insensitive_volume(path):
        # APFS and HFS+ also treat composed and decomposed accents as one name.
        return unicodedata.normalize("NFD", path.casefold())
    return path


def same_name_on_folding_volume(first, second):
    """True when two names that differ only by case are one file on the volume that holds them.

    os.path.normcase folds case only on Windows and the `pathconf` probe above answers only on macOS, so a Linux
    mount that ignores case (vfat, exFAT, CIFS) would pass `R.json` and `r.json` as two reports and let the second
    one overwrite the first. Ask the volume itself: create one spelling in a scratch folder beside the reports and
    look for the other. Names that differ by more than case are never one file, so that costs nothing.
    """
    return _same_resolved_name(os.path.realpath(first), os.path.realpath(second))


def _same_resolved_name(first, second):
    """`same_name_on_folding_volume` for paths `os.path.realpath` has already resolved."""
    first_dir, first_name = os.path.split(first)
    second_dir, second_name = os.path.split(second)
    if first_name.casefold() != second_name.casefold():
        return False
    try:
        if not os.path.samefile(first_dir, second_dir):
            return False
        with tempfile.TemporaryDirectory(prefix=".leonida-report-", dir=first_dir) as probe:
            with open(os.path.join(probe, first_name), "xb"):
                return os.path.exists(os.path.join(probe, second_name))
    except OSError:
        return False


def same_report_path(first, second):
    # Resolve each path once: the key and the volume probe both want the resolved name, and a guard runs once
    # per input file, so a second realpath per path is the whole cost of the check (see #6220's review).
    first, second = os.path.realpath(first), os.path.realpath(second)
    if _resolved_key(first) == _resolved_key(second):
        return True
    try:
        return os.path.samefile(first, second)
    except OSError:
        pass
    return _same_resolved_name(first, second)
