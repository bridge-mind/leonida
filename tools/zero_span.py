"""Classify the bytes between two carved streams without copying them.

The archive scanners count the bytes they could not read as either zero padding or unread data: a gap counts as
padding only when every byte in it is zero. Slicing an mmap copies the slice, so `gap = m[last:s]` allocates the
whole gap at once. On a large archive a gap can run to gigabytes (story000.rpf, about 20 GB, cannot be indexed on
a 16 GB machine that way), so the gap is compared against zeros one window at a time instead, and the first
non-zero window ends the check.

The range is normalized with Python's own slice rules, so the counts and verdicts match `buf[start:end]` for
negative and out-of-range endpoints. That is the right contract here because callers address the buffer through
slices rather than raw file offsets.
"""

WINDOW = 1 << 22                 # 4 MiB: the most a check allocates, besides the shared zero block
_ZEROS = bytes(WINDOW)


def all_zero(buf, start: int, end: int) -> bool:
    """True when buf[start:end] is empty or all zero bytes. Reads at most WINDOW bytes at a time."""
    start, end, _ = slice(start, end).indices(len(buf))
    p = start
    while p < end:
        q = min(end, p + WINDOW)
        if buf[p:q] != _ZEROS[:q - p]:
            return False
        p = q
    return True


def gap_split(buf, start: int, end: int) -> tuple[int, int]:
    """(zero_padding, unread) bytes for the gap buf[start:end], counted as the scanners always have: all of
    it is padding when every byte is zero, otherwise all of it is unread. The range is clipped to the buffer
    the way a slice is, so the counts match len(buf[start:end])."""
    start, end, _ = slice(start, end).indices(len(buf))
    n = max(0, end - start)
    return (n, 0) if all_zero(buf, start, end) else (0, n)
