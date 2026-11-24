"""Permission bits for a file published through a private staging file.

tempfile.NamedTemporaryFile creates its file 0600, so a writer that stages next to the destination and then
os.replace()s it has to give the staged file the bits the destination would have had. Those are the bits of the
file it replaces, or, for a new file, what open() would give one in that directory.

The umask is not read with os.umask(0) / os.umask(mask): that sets process-wide state, so a thread creating a
file between the two calls gets mask 0 (#750 and #728 review). Instead a probe file is created with mode 0666 and
its bits are read back, which applies the current umask (and a directory's default ACL) without changing anything.
"""
import os
import secrets
import stat


def new_file_mode(directory):
    """The permission bits open() would give a new file in `directory`, found without touching the umask."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    for _ in range(100):
        probe = os.path.join(directory, f".mode-{secrets.token_hex(8)}.probe")
        try:
            fd = os.open(probe, flags, 0o666)
        except FileExistsError:
            continue
        try:
            return stat.S_IMODE(os.fstat(fd).st_mode)
        finally:
            os.close(fd)
            try:
                os.remove(probe)
            except OSError:
                pass
    raise FileExistsError(f"no free probe name in {directory}")


def published_mode(path):
    """The permission bits a file published at `path` should get: those of the file it replaces, or a new file's."""
    try:
        return stat.S_IMODE(os.stat(path).st_mode)
    except FileNotFoundError:
        return new_file_mode(os.path.dirname(os.path.abspath(path)))
