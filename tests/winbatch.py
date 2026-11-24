"""Write a one-line Windows batch file (a fake tool for a test) that cmd.exe can run from any user folder.

cmd.exe decodes a batch file with the console's OEM code page, not UTF-8, so a path with non-ASCII letters (a
user folder named in Cyrillic, say) written as UTF-8 becomes "The system cannot find the path specified".
Standard library only.
"""


def batch_path(path):
    """A spelling of path that cmd.exe reads correctly from a batch file: as is when ASCII, else its 8.3 short name
    (ASCII). Falls back to the long path where the volume has 8.3 names turned off."""
    if path.isascii():
        return path
    import ctypes
    buf = ctypes.create_unicode_buffer(32768)
    n = ctypes.windll.kernel32.GetShortPathNameW(path, buf, len(buf))
    return buf.value if 0 < n < len(buf) else path


def _encodes(char):
    try:
        char.encode("oem")
        return True
    except UnicodeEncodeError:
        return False


def write_python_cmd(cmd_path, python, script):
    """Write cmd_path as a batch file that runs `python script` with the batch file's arguments.

    Raises unittest.SkipTest when a path has no 8.3 name and letters outside the OEM code page (Cyrillic on a
    cp850 machine with 8.3 names off, say): cmd.exe couldn't run such a file in any encoding."""
    line = f'@"{batch_path(python)}" "{batch_path(script)}" %*\n'
    try:
        data = line.encode("oem")   # the code page cmd.exe reads a batch file with
    except UnicodeEncodeError:
        import unittest
        # Windows' "oem" codec doesn't report a useful position, so name the letters it can't encode here.
        letters = "".join(sorted({c for c in line if not _encodes(c)}))
        raise unittest.SkipTest(f"{letters!r} in a path has no 8.3 name and is outside the OEM code page,"
                                " so cmd.exe can't run a batch file that names it; use an ASCII TEMP") from None
    with open(cmd_path, "wb") as f:
        f.write(data)
    return cmd_path
