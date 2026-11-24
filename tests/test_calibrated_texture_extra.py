"""Expanded stdlib-only unit tests for tools/calibrated_texture.py.

Covers native loader wiring, array validation, zig/distance/accumulate helpers,
tile address mapping, BC1/BC3/BC4/BC5/BC7 block rebuilders, LocalDecoder chunk
decoding/caching/surface/DDS export paths, error handling, and CLI integration
using synthetic pools/profiles and unittest.mock (no game data).

    python -I tests/test_calibrated_texture_extra.py -v
"""
from __future__ import annotations

import bisect
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import ctypes
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock
import zlib

ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT / "tools"
TESTS_DIR = ROOT / "tests"
for _p in (str(TESTS_DIR), str(TOOLS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bcpack  # noqa: E402
import bcpack_tables  # noqa: E402
import bcpack_tables_bptc as _bptc  # noqa: E402
import test_bcpack as tb  # noqa: E402
import test_ttpf_mips as ttm  # noqa: E402
import tex_headers  # noqa: E402
import ttpf_mips as ttpf  # noqa: E402
from test_calibrated_texture import synthetic_profile  # noqa: E402


# ---------------------------------------------------------------------------
# Minimal pure-stdlib NumPy shim for environments without NumPy installed.
# Only used when importlib.util.find_spec("numpy") is None.
# ---------------------------------------------------------------------------

class _FakeDType:
    """Minimal NumPy-like dtype descriptor supporting integers, bool, float, str, and Vn."""

    _SPEC = {
        "bool": ("b", 1, "?", False, False),
        "b1": ("b", 1, "?", False, False),
        "int8": ("i", 1, "b", True, False),
        "i1": ("i", 1, "b", True, False),
        "uint8": ("u", 1, "B", False, False),
        "u1": ("u", 1, "B", False, False),
        "int16": ("i", 2, "<h", True, False),
        "i2": ("i", 2, "<h", True, False),
        "<i2": ("i", 2, "<h", True, False),
        "uint16": ("u", 2, "<H", False, False),
        "u2": ("u", 2, "<H", False, False),
        "<u2": ("u", 2, "<H", False, False),
        "int32": ("i", 4, "<i", True, False),
        "i4": ("i", 4, "<i", True, False),
        "<i4": ("i", 4, "<i", True, False),
        "uint32": ("u", 4, "<I", False, False),
        "u4": ("u", 4, "<I", False, False),
        "<u4": ("u", 4, "<I", False, False),
        "int64": ("i", 8, "<q", True, False),
        "i8": ("i", 8, "<q", True, False),
        "<i8": ("i", 8, "<q", True, False),
        "uint64": ("u", 8, "<Q", False, False),
        "u8": ("u", 8, "<Q", False, False),
        "<u8": ("u", 8, "<Q", False, False),
        "float64": ("f", 8, "<d", True, True),
        "f8": ("f", 8, "<d", True, True),
        "str": ("U", 4, None, False, False),
        "object": ("O", 8, None, False, False),
    }

    def __init__(self, spec):
        if isinstance(spec, _FakeDType):
            self.name = spec.name
            self.kind = spec.kind
            self.itemsize = spec.itemsize
            self._struct = spec._struct
            self.signed = spec.signed
            self.is_float = spec.is_float
            return
        if spec is bool:
            key = "bool"
        elif spec is int:
            key = "int64"
        elif spec is float:
            key = "float64"
        elif spec is str:
            key = "str"
        else:
            key = str(spec)
        if key.startswith("V") and key[1:].isdigit():
            n = int(key[1:])
            self.name = key
            self.kind = "V"
            self.itemsize = n
            self._struct = None
            self.signed = False
            self.is_float = False
            return
        if key not in self._SPEC:
            raise TypeError(f"Unsupported fake dtype: {spec!r}")
        kind, size, fmt, signed, is_float = self._SPEC[key]
        self.name = key
        self.kind = kind
        self.itemsize = size
        self._struct = struct.Struct(fmt) if fmt else None
        self.signed = signed
        self.is_float = is_float

    def cast_val(self, v):
        if self.kind == "b":
            return bool(v)
        if self.kind == "V":
            if isinstance(v, (bytes, bytearray)):
                b = bytes(v)
            elif isinstance(v, int) and v == 0:
                b = b"\x00" * self.itemsize
            else:
                raise TypeError(f"Cannot store {v!r} in {self.name}")
            if len(b) != self.itemsize:
                raise ValueError(f"Expected {self.itemsize} bytes, got {len(b)}")
            return b
        if self.kind in "iu":
            iv = int(v)
            bits = self.itemsize * 8
            mask = (1 << bits) - 1
            iv &= mask
            if self.signed:
                sign = 1 << (bits - 1)
                iv = (iv ^ sign) - sign
            return iv
        if self.kind == "f":
            return float(v)
        return v

    def __eq__(self, other):
        if isinstance(other, _FakeDType):
            return (self.kind, self.itemsize) == (other.kind, other.itemsize)
        return NotImplemented

    def __repr__(self):
        return f"_FakeDType({self.name!r})"


def _normalize_dtype(dtype):
    return dtype if isinstance(dtype, _FakeDType) else _FakeDType(dtype)


def _promote_dtype(a, b):
    da = _normalize_dtype(a)
    db = _normalize_dtype(b)
    if da == db:
        return da
    if da.kind == "b":
        return db
    if db.kind == "b":
        return da
    if da.kind == "f" or db.kind == "f":
        return _FakeDType("float64")
    if da.kind == db.kind:
        return da if da.itemsize >= db.itemsize else db
    u_dt, i_dt = (da, db) if da.kind == "u" else (db, da)
    if u_dt.itemsize >= i_dt.itemsize:
        if u_dt.itemsize == 8:
            return _FakeDType("uint64")
        return _FakeDType(f"int{max(i_dt.itemsize, u_dt.itemsize * 2) * 8}")
    return i_dt


def _infer_dtype(flat_vals):
    if not flat_vals:
        return _FakeDType("int64")
    kinds = set()
    for v in flat_vals:
        if isinstance(v, (bytes, bytearray)):
            kinds.add(("V", len(v)))
        elif isinstance(v, bool):
            kinds.add("b")
        elif isinstance(v, int):
            kinds.add("i")
        elif isinstance(v, float):
            kinds.add("f")
        elif isinstance(v, str):
            kinds.add("U")
        else:
            kinds.add("O")
    if len(kinds) == 1:
        k = next(iter(kinds))
        if isinstance(k, tuple) and k[0] == "V":
            return _FakeDType(f"V{k[1]}")
        if k == "b":
            return _FakeDType("bool")
        if k == "i":
            lo = min(flat_vals)
            hi = max(flat_vals)
            if -0x8000000000000000 <= lo and hi <= 0x7FFFFFFFFFFFFFFF:
                return _FakeDType("int64")
            return _FakeDType("uint64")
        if k == "f":
            return _FakeDType("float64")
        if k == "U":
            return _FakeDType("str")
    if kinds <= {"b", "i"}:
        return _FakeDType("int64")
    if kinds <= {"b", "i", "f"}:
        return _FakeDType("float64")
    return _FakeDType("object")


def _flatten_nested(obj):
    if isinstance(obj, _FakeArray):
        return obj.shape, obj._flat_list(), obj.dtype
    if isinstance(obj, (list, tuple)):
        if not obj:
            return (0,), [], None
        sub_shapes = []
        items = []
        sub_dtypes = []
        for elem in obj:
            if isinstance(elem, (list, tuple, _FakeArray)):
                sh, fl, dt = _flatten_nested(elem)
                sub_shapes.append(sh)
                items.extend(fl)
                if dt is not None:
                    sub_dtypes.append(dt)
            else:
                sub_shapes.append(())
                items.append(elem)
        if len(set(sub_shapes)) != 1:
            raise ValueError("inhomogeneous shape")
        shape = (len(obj),) + sub_shapes[0]
        dt = sub_dtypes[0] if sub_dtypes and all(d == sub_dtypes[0] for d in sub_dtypes) else None
        return shape, items, dt
    return (), [obj], None


def _shape_strides(shape):
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return tuple(strides)


def _broadcast_shapes(s1, s2):
    n = max(len(s1), len(s2))
    p1 = (1,) * (n - len(s1)) + s1
    p2 = (1,) * (n - len(s2)) + s2
    out = []
    for a, b in zip(p1, p2):
        if a == b:
            out.append(a)
        elif a == 1:
            out.append(b)
        elif b == 1:
            out.append(a)
        else:
            raise ValueError(f"operands could not be broadcast together with shapes {s1} {s2}")
    return tuple(out)


def _broadcast_index(out_coord, target_shape):
    pad = len(out_coord) - len(target_shape)
    res = []
    for c, dim in zip(out_coord[pad:], target_shape):
        res.append(0 if dim == 1 else c)
    return tuple(res)


def _unravel(flat_idx, shape):
    if not shape:
        return ()
    coords = [0] * len(shape)
    rem = flat_idx
    for ax in range(len(shape) - 1, -1, -1):
        dim = shape[ax]
        coords[ax] = rem % dim
        rem //= dim
    return tuple(coords)


def _ravel_coord(coord, strides):
    return sum(c * s for c, s in zip(coord, strides))


class _FakeCtypesView:
    def __init__(self, arr):
        self._arr = arr

    @property
    def data(self):
        return ctypes.addressof(self._arr._buf)


class _FakeFlat:
    def __init__(self, arr):
        self._arr = arr

    def __getitem__(self, idx):
        return self._arr.ravel()[idx]

    def __setitem__(self, idx, value):
        indices = idx._flat_list() if isinstance(idx, _FakeArray) else list(idx)
        vals = value._flat_list() if isinstance(value, _FakeArray) else (
            list(value) if isinstance(value, (list, tuple)) else [value] * len(indices)
        )
        if len(vals) == 1 and len(indices) != 1:
            vals = vals * len(indices)
        for pos, val in zip(indices, vals):
            self._arr._set_flat(int(pos), val)


class _FakeArray:
    """Pure-Python contiguous ndarray subset backed by a live ctypes byte buffer."""

    def __init__(self, shape, dtype, flat_vals=None, raw_bytes=None):
        self.shape = tuple(shape)
        self.dtype = _normalize_dtype(dtype)
        self.size = math.prod(self.shape) if self.shape else 1
        self.ndim = len(self.shape)
        self._strides = _shape_strides(self.shape)
        total_bytes = self.size * self.dtype.itemsize
        if self.dtype.kind in "UO":
            self._buf = None
            self._obj_data = list(flat_vals) if flat_vals is not None else [None] * self.size
            return
        self._obj_data = None
        self._buf = (ctypes.c_ubyte * max(1, total_bytes))()
        if raw_bytes is not None:
            if len(raw_bytes) != total_bytes:
                raise ValueError(f"Expected {total_bytes} raw bytes, got {len(raw_bytes)}")
            if total_bytes:
                ctypes.memmove(self._buf, raw_bytes, total_bytes)
        elif flat_vals is not None:
            for i, val in enumerate(flat_vals):
                self._set_flat(i, val)

    def _get_flat(self, i):
        if i < 0:
            i += self.size
        if not 0 <= i < self.size:
            raise IndexError(i)
        if self._obj_data is not None:
            return self._obj_data[i]
        sz = self.dtype.itemsize
        if self.dtype.kind == "V":
            return bytes(self._buf[i * sz : (i + 1) * sz])
        return self.dtype._struct.unpack_from(self._buf, i * sz)[0]

    def _set_flat(self, i, val):
        if i < 0:
            i += self.size
        if not 0 <= i < self.size:
            raise IndexError(i)
        if self._obj_data is not None:
            self._obj_data[i] = val
            return
        coerced = self.dtype.cast_val(val)
        sz = self.dtype.itemsize
        if self.dtype.kind == "V":
            for b_i, byte in enumerate(coerced):
                self._buf[i * sz + b_i] = byte
        else:
            self.dtype._struct.pack_into(self._buf, i * sz, coerced)

    def _flat_list(self):
        return [self._get_flat(i) for i in range(self.size)]

    def _get_coord(self, coord):
        return self._get_flat(_ravel_coord(coord, self._strides))

    def _set_coord(self, coord, val):
        self._set_flat(_ravel_coord(coord, self._strides), val)

    @property
    def ctypes(self):
        return _FakeCtypesView(self)

    @property
    def flat(self):
        return _FakeFlat(self)

    @property
    def T(self):
        return self.transpose(*reversed(range(self.ndim)))

    def __len__(self):
        if self.ndim == 0:
            raise TypeError("len() of unsized object")
        return self.shape[0]

    def __iter__(self):
        if self.ndim == 0:
            raise TypeError("iteration over a 0-d array")
        for i in range(self.shape[0]):
            yield self[i]

    def __int__(self):
        if self.size != 1:
            raise TypeError("only length-1 arrays can be converted to Python scalars")
        return int(self._get_flat(0))

    def __index__(self):
        if self.size != 1 or self.dtype.kind not in "iu":
            raise TypeError("only integer scalar arrays can be converted to a scalar index")
        return int(self._get_flat(0))

    def __bool__(self):
        if self.size != 1:
            raise ValueError("The truth value of an array with more than one element is ambiguous")
        return bool(self._get_flat(0))

    def copy(self):
        if self._obj_data is not None:
            return _FakeArray(self.shape, self.dtype, flat_vals=list(self._obj_data))
        return _FakeArray(self.shape, self.dtype, raw_bytes=self.tobytes())

    def tobytes(self):
        if self._obj_data is not None:
            raise TypeError("cannot convert object/str array to bytes")
        n = self.size * self.dtype.itemsize
        return bytes(memoryview(self._buf)[:n])

    def tolist(self):
        def _build(offset, shape):
            if not shape:
                return self._get_flat(offset)
            step = math.prod(shape[1:]) if len(shape) > 1 else 1
            return [_build(offset + i * step, shape[1:]) for i in range(shape[0])]
        return _build(0, self.shape)

    def astype(self, dtype):
        dt = _normalize_dtype(dtype)
        return _FakeArray(self.shape, dt, flat_vals=self._flat_list())

    def ravel(self):
        return _FakeArray((self.size,), self.dtype, raw_bytes=self.tobytes()) if self._buf is not None else _FakeArray((self.size,), self.dtype, flat_vals=self._flat_list())

    def reshape(self, *newshape):
        if len(newshape) == 1 and isinstance(newshape[0], (tuple, list)):
            newshape = tuple(newshape[0])
        shape = list(newshape)
        if -1 in shape:
            neg_idx = shape.index(-1)
            known = math.prod(d for d in shape if d != -1)
            shape[neg_idx] = self.size // known
        shape = tuple(shape)
        if math.prod(shape) != self.size:
            raise ValueError(f"cannot reshape array of size {self.size} into shape {shape}")
        if self._buf is not None:
            return _FakeArray(shape, self.dtype, raw_bytes=self.tobytes())
        return _FakeArray(shape, self.dtype, flat_vals=self._flat_list())

    def transpose(self, *axes):
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        if not axes:
            axes = tuple(reversed(range(self.ndim)))
        new_shape = tuple(self.shape[a] for a in axes)
        vals = []
        for idx in range(self.size):
            coord = _unravel(idx, new_shape)
            src = [0] * self.ndim
            for new_ax, old_ax in enumerate(axes):
                src[old_ax] = coord[new_ax]
            vals.append(self._get_coord(tuple(src)))
        return _FakeArray(new_shape, self.dtype, flat_vals=vals)

    def any(self):
        return any(bool(v) for v in self._flat_list())

    def all(self):
        return all(bool(v) for v in self._flat_list())

    def max(self):
        vals = self._flat_list()
        if not vals:
            raise ValueError("zero-size array to reduction operation maximum which has no identity")
        return max(vals)

    def min(self):
        vals = self._flat_list()
        if not vals:
            raise ValueError("zero-size array to reduction operation minimum which has no identity")
        return min(vals)

    # --- Indexing ---
    def _parse_key(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        # Fast path: single boolean mask matching array shape or leading axis
        if len(key) == 1 and isinstance(key[0], _FakeArray) and key[0].dtype.kind == "b":
            return ("bool_mask", key[0])
        if len(key) == 1 and isinstance(key[0], (list, tuple)) and key[0] and all(isinstance(v, bool) for v in key[0]):
            return ("bool_mask", _fake_asarray(key[0], dtype="bool"))

        # General multi-axis key supporting int, slice, None, and integer fancy indexers
        items = list(key)
        non_none = [x for x in items if x is not None]
        while len(non_none) < self.ndim:
            items.append(slice(None))
            non_none.append(slice(None))
        return ("axes", items)

    def __getitem__(self, key):
        kind, parsed = self._parse_key(key)
        if kind == "bool_mask":
            mask = parsed
            if mask.shape == self.shape:
                vals = [v for v, m in zip(self._flat_list(), mask._flat_list()) if m]
                return _FakeArray((len(vals),), self.dtype, flat_vals=vals)
            if mask.ndim == 1 and len(mask) == self.shape[0]:
                tail_shape = self.shape[1:]
                tail_size = math.prod(tail_shape) if tail_shape else 1
                vals = []
                rows = 0
                for r, keep in enumerate(mask._flat_list()):
                    if keep:
                        rows += 1
                        base = r * tail_size
                        for off in range(tail_size):
                            vals.append(self._get_flat(base + off))
                return _FakeArray((rows,) + tail_shape, self.dtype, flat_vals=vals)
            raise IndexError("boolean index did not match indexed array")

        # Separate axes into newaxis, slice, int, or fancy integer array
        fancy_positions = []
        axis = 0
        axis_specs = []
        for item in parsed:
            if item is None:
                axis_specs.append(("newaxis", None))
                continue
            dim = self.shape[axis]
            if isinstance(item, slice):
                axis_specs.append(("slice", range(*item.indices(dim))))
            elif isinstance(item, int) or (isinstance(item, _FakeArray) and item.ndim == 0 and item.dtype.kind in "iu"):
                idx = int(item)
                if idx < 0:
                    idx += dim
                if not 0 <= idx < dim:
                    raise IndexError(idx)
                axis_specs.append(("int", idx))
            else:
                arr = item if isinstance(item, _FakeArray) else _fake_asarray(item)
                if arr.dtype.kind not in "iu":
                    raise IndexError("arrays used as indices must be of integer type")
                fancy_positions.append((len(axis_specs), dim, arr))
                axis_specs.append(("fancy", arr))
            axis += 1

        if not fancy_positions:
            out_shape = []
            ranges = []
            for sp_kind, sp_val in axis_specs:
                if sp_kind == "newaxis":
                    out_shape.append(1)
                    ranges.append([0])
                elif sp_kind == "slice":
                    out_shape.append(len(sp_val))
                    ranges.append(list(sp_val))
                else:
                    ranges.append([sp_val])
            if not out_shape:
                coord = tuple(r[0] for r in ranges)
                return self._get_coord(coord)
            vals = []
            total = math.prod(out_shape)
            for idx in range(total):
                oc = _unravel(idx, tuple(out_shape))
                src = []
                oi = 0
                for sp_kind, sp_val in axis_specs:
                    if sp_kind == "newaxis":
                        oi += 1
                    elif sp_kind == "slice":
                        src.append(sp_val[oc[oi]])
                        oi += 1
                    else:
                        src.append(sp_val)
                vals.append(self._get_coord(tuple(src)))
            return _FakeArray(tuple(out_shape), self.dtype, flat_vals=vals)

        # Broadcast all fancy arrays together
        bshape = ()
        for _, _, arr in fancy_positions:
            bshape = _broadcast_shapes(bshape, arr.shape)
        contiguous_fancy = [p for p, _, _ in fancy_positions] == list(
            range(fancy_positions[0][0], fancy_positions[0][0] + len(fancy_positions))
        )
        # Build output shape
        out_shape = []
        fancy_inserted = False
        slice_ranges = {}
        for pos, (sp_kind, sp_val) in enumerate(axis_specs):
            if sp_kind == "newaxis":
                out_shape.append(1)
            elif sp_kind == "slice":
                out_shape.append(len(sp_val))
                slice_ranges[pos] = list(sp_val)
            elif sp_kind == "fancy":
                if contiguous_fancy:
                    if not fancy_inserted:
                        out_shape.extend(bshape)
                        fancy_inserted = True
                else:
                    if not fancy_inserted:
                        out_shape = list(bshape) + out_shape
                        fancy_inserted = True

        # Map each output coordinate to source coordinate
        total = math.prod(out_shape) if out_shape else 1
        vals = []
        b_start = 0 if not contiguous_fancy else sum(
            1 for p, (k, _) in enumerate(axis_specs[: fancy_positions[0][0]]) if k in ("newaxis", "slice")
        )
        b_len = len(bshape)
        for flat_i in range(total):
            oc = _unravel(flat_i, tuple(out_shape))
            b_coord = oc[b_start : b_start + b_len]
            src = []
            oi = 0 if contiguous_fancy else b_len
            for pos, (sp_kind, sp_val) in enumerate(axis_specs):
                if sp_kind == "newaxis":
                    oi += 1
                elif sp_kind == "slice":
                    src.append(slice_ranges[pos][oc[oi]])
                    oi += 1
                elif sp_kind == "int":
                    src.append(sp_val)
                elif sp_kind == "fancy":
                    if pos == fancy_positions[0][0] and contiguous_fancy:
                        oi += b_len
                    dim = self.shape[len(src)]
                    fc = _broadcast_index(b_coord, sp_val.shape)
                    idx_v = int(sp_val._get_coord(fc)) if sp_val.ndim else int(sp_val._get_flat(0))
                    if idx_v < 0:
                        idx_v += dim
                    if not 0 <= idx_v < dim:
                        raise IndexError(idx_v)
                    src.append(idx_v)
            vals.append(self._get_coord(tuple(src)))
        if not out_shape:
            return vals[0]
        return _FakeArray(tuple(out_shape), self.dtype, flat_vals=vals)

    def __setitem__(self, key, value):
        kind, parsed = self._parse_key(key)
        if kind == "bool_mask":
            mask = parsed
            rhs = value if isinstance(value, _FakeArray) else _fake_asarray(value, dtype=self.dtype)
            if mask.shape == self.shape:
                k = 0
                for i, m in enumerate(mask._flat_list()):
                    if m:
                        v = rhs._get_flat(0) if rhs.size == 1 else rhs._get_flat(k)
                        self._set_flat(i, v)
                        k += 1
                return
            if mask.ndim == 1 and len(mask) == self.shape[0]:
                tail_shape = self.shape[1:]
                tail_size = math.prod(tail_shape) if tail_shape else 1
                n_true = sum(1 for m in mask._flat_list() if m)
                target_shape = (n_true,) + tail_shape
                b_shape = _broadcast_shapes(target_shape, rhs.shape)
                if b_shape != target_shape:
                    raise ValueError(f"cannot broadcast {rhs.shape} to {target_shape}")
                t_idx = 0
                for r, keep in enumerate(mask._flat_list()):
                    if keep:
                        base = r * tail_size
                        for off in range(tail_size):
                            tc = (t_idx,) + _unravel(off, tail_shape)
                            rc = _broadcast_index(tc, rhs.shape)
                            self._set_flat(base + off, rhs._get_coord(rc) if rhs.ndim else rhs._get_flat(0))
                        t_idx += 1
                return
            raise IndexError("boolean index did not match indexed array")

        # Concrete axis assignment (ints and slices)
        slices = []
        sub_shape = []
        axis = 0
        for item in parsed:
            if item is None:
                continue
            dim = self.shape[axis]
            if isinstance(item, slice):
                rng = list(range(*item.indices(dim)))
                slices.append(rng)
                sub_shape.append(len(rng))
            else:
                idx = int(item)
                if idx < 0:
                    idx += dim
                if not 0 <= idx < dim:
                    raise IndexError(idx)
                slices.append([idx])
            axis += 1
        rhs = value if isinstance(value, _FakeArray) else _fake_asarray(value, dtype=self.dtype)
        sub_shape = tuple(sub_shape)
        if not sub_shape:
            coord = tuple(r[0] for r in slices)
            self._set_coord(coord, rhs._get_flat(0) if rhs.size == 1 else rhs)
            return
        total = math.prod(sub_shape)
        for fi in range(total):
            sc = _unravel(fi, sub_shape)
            dst = []
            si = 0
            for rng in slices:
                if len(rng) == 1 and len(sub_shape) < len(slices):
                    dst.append(rng[0])
                else:
                    dst.append(rng[sc[si]])
                    si += 1
            # Re-derive dst accurately:
            dst = []
            si = 0
            for ax_i, item in enumerate(parsed):
                if item is None:
                    continue
                if isinstance(item, slice):
                    dst.append(slices[ax_i][sc[si]])
                    si += 1
                else:
                    dst.append(slices[ax_i][0])
            rc = _broadcast_index(sc, rhs.shape)
            self._set_coord(tuple(dst), rhs._get_coord(rc) if rhs.ndim else rhs._get_flat(0))

    # --- Operators ---
    def _binop(self, other, fn, out_dtype=None):
        rhs = other if isinstance(other, _FakeArray) else _fake_asarray(other)
        bshape = _broadcast_shapes(self.shape, rhs.shape)
        dt = _normalize_dtype(out_dtype) if out_dtype is not None else _promote_dtype(self.dtype, rhs.dtype)
        total = math.prod(bshape) if bshape else 1
        vals = []
        for i in range(total):
            oc = _unravel(i, bshape)
            a_val = self._get_coord(_broadcast_index(oc, self.shape)) if self.ndim else self._get_flat(0)
            b_val = rhs._get_coord(_broadcast_index(oc, rhs.shape)) if rhs.ndim else rhs._get_flat(0)
            vals.append(fn(a_val, b_val))
        return _FakeArray(bshape, dt, flat_vals=vals)

    def _rbinop(self, other, fn, out_dtype=None):
        lhs = other if isinstance(other, _FakeArray) else _fake_asarray(other)
        return lhs._binop(self, fn, out_dtype=out_dtype)

    def _ibop(self, other, fn):
        res = self._binop(other, fn, out_dtype=self.dtype)
        for i, v in enumerate(res._flat_list()):
            self._set_flat(i, v)
        return self

    def __add__(self, o): return self._binop(o, lambda a, b: a + b)
    def __radd__(self, o): return self._rbinop(o, lambda a, b: a + b)
    def __iadd__(self, o): return self._ibop(o, lambda a, b: a + b)
    def __sub__(self, o): return self._binop(o, lambda a, b: a - b)
    def __rsub__(self, o): return self._rbinop(o, lambda a, b: a - b)
    def __mul__(self, o): return self._binop(o, lambda a, b: a * b)
    def __rmul__(self, o): return self._rbinop(o, lambda a, b: a * b)
    def __floordiv__(self, o): return self._binop(o, lambda a, b: a // b)
    def __mod__(self, o): return self._binop(o, lambda a, b: a % b)
    def __imod__(self, o): return self._ibop(o, lambda a, b: a % b)
    def __lshift__(self, o): return self._binop(o, lambda a, b: a << b)
    def __rlshift__(self, o):
        # Preserve self's integer dtype when shifting a scalar 1 << uint64_arr
        rhs_arr = o if isinstance(o, _FakeArray) else _fake_asarray(o, dtype=self.dtype)
        return rhs_arr._binop(self, lambda a, b: a << b, out_dtype=self.dtype)
    def __rshift__(self, o): return self._binop(o, lambda a, b: a >> b)
    def __and__(self, o): return self._binop(o, lambda a, b: a & b)
    def __or__(self, o): return self._binop(o, lambda a, b: a | b)
    def __ior__(self, o): return self._ibop(o, lambda a, b: a | b)
    def __xor__(self, o): return self._binop(o, lambda a, b: a ^ b)
    def __lt__(self, o): return self._binop(o, lambda a, b: a < b, out_dtype="bool")
    def __le__(self, o): return self._binop(o, lambda a, b: a <= b, out_dtype="bool")
    def __gt__(self, o): return self._binop(o, lambda a, b: a > b, out_dtype="bool")
    def __ge__(self, o): return self._binop(o, lambda a, b: a >= b, out_dtype="bool")
    def __eq__(self, o): return self._binop(o, lambda a, b: a == b, out_dtype="bool")
    def __ne__(self, o): return self._binop(o, lambda a, b: a != b, out_dtype="bool")

    def __neg__(self):
        return _FakeArray(self.shape, self.dtype, flat_vals=[-v for v in self._flat_list()])

    def __invert__(self):
        if self.dtype.kind == "b":
            return _FakeArray(self.shape, self.dtype, flat_vals=[not v for v in self._flat_list()])
        return _FakeArray(self.shape, self.dtype, flat_vals=[~v for v in self._flat_list()])

    def __matmul__(self, other):
        rhs = other if isinstance(other, _FakeArray) else _fake_asarray(other)
        dt = _promote_dtype(self.dtype, rhs.dtype)
        if self.ndim == 2 and rhs.ndim == 1 and self.shape[1] == rhs.shape[0]:
            m, k = self.shape
            r_vals = rhs._flat_list()
            s_vals = self._flat_list()
            out = [sum(s_vals[i * k + j] * r_vals[j] for j in range(k)) for i in range(m)]
            return _FakeArray((m,), dt, flat_vals=out)
        raise ValueError(f"Unsupported matmul shapes {self.shape} @ {rhs.shape}")


def _fake_asarray(values, dtype=None):
    if isinstance(values, _FakeArray):
        if dtype is None or values.dtype == _normalize_dtype(dtype):
            return values
        return values.astype(dtype)
    shape, flat, sub_dt = _flatten_nested(values)
    dt = _normalize_dtype(dtype) if dtype is not None else (sub_dt or _infer_dtype(flat))
    return _FakeArray(shape, dt, flat_vals=flat)


def _fake_array(values, dtype=None):
    arr = _fake_asarray(values, dtype=dtype)
    return arr.copy()


def _fake_ascontiguousarray(values, dtype=None):
    return _fake_asarray(values, dtype=dtype).copy()


def _fake_zeros(shape, dtype="float64"):
    sh = (int(shape),) if isinstance(shape, int) else tuple(shape)
    dt = _normalize_dtype(dtype)
    n = math.prod(sh) if sh else 1
    if dt.kind == "V":
        return _FakeArray(sh, dt, flat_vals=[b"\x00" * dt.itemsize] * n)
    return _FakeArray(sh, dt, flat_vals=[0] * n)


def _fake_empty(shape, dtype="float64"):
    return _fake_zeros(shape, dtype=dtype)


def _fake_frombuffer(buf, dtype="uint8"):
    raw = bytes(buf)
    dt = _normalize_dtype(dtype)
    if len(raw) % dt.itemsize != 0:
        raise ValueError("buffer size must be a multiple of element size")
    return _FakeArray((len(raw) // dt.itemsize,), dt, raw_bytes=raw)


def _fake_arange(start, stop=None, step=1, dtype=None):
    if stop is None:
        start, stop = 0, start
    vals = list(range(int(start), int(stop), int(step)))
    return _FakeArray((len(vals),), dtype or "int64", flat_vals=vals)


def _fake_any(arr):
    return _fake_asarray(arr).any()


def _fake_all(arr):
    return _fake_asarray(arr).all()


def _fake_unique(arr):
    vals = sorted(set(_fake_asarray(arr)._flat_list()))
    dt = arr.dtype if isinstance(arr, _FakeArray) else _infer_dtype(vals)
    return _FakeArray((len(vals),), dt, flat_vals=vals)


def _fake_searchsorted(a, v, side="left"):
    edges = _fake_asarray(a)._flat_list()
    queries = _fake_asarray(v)
    fn = bisect.bisect_right if side == "right" else bisect.bisect_left
    res = [fn(edges, x) for x in queries._flat_list()]
    return _FakeArray(queries.shape, "int64", flat_vals=res)


def _fake_roll(a, shift):
    arr = _fake_asarray(a)
    vals = arr._flat_list()
    n = len(vals)
    if n == 0:
        return arr.copy()
    s = shift % n
    rolled = vals[-s:] + vals[:-s] if s else list(vals)
    return _FakeArray(arr.shape, arr.dtype, flat_vals=rolled)


def _fake_column_stack(tup):
    cols = [_fake_asarray(c) for c in tup]
    n = cols[0].shape[0]
    dt = cols[0].dtype
    for c in cols[1:]:
        dt = _promote_dtype(dt, c.dtype)
    flat = []
    for r in range(n):
        for c in cols:
            flat.append(c._get_flat(r))
    return _FakeArray((n, len(cols)), dt, flat_vals=flat)


class _FakeBitwiseOr:
    @staticmethod
    def reduce(arr, axis=0):
        a = _fake_asarray(arr)
        if axis < 0:
            axis += a.ndim
        out_shape = tuple(d for i, d in enumerate(a.shape) if i != axis)
        dim = a.shape[axis]
        total = math.prod(out_shape) if out_shape else 1
        vals = []
        for idx in range(total):
            oc = _unravel(idx, out_shape)
            acc = 0
            for k in range(dim):
                src = list(oc)
                src.insert(axis, k)
                acc |= a._get_coord(tuple(src))
            vals.append(acc)
        return _FakeArray(out_shape, a.dtype, flat_vals=vals)


def _make_fake_numpy():
    import types
    mod = types.ModuleType("numpy")
    mod.array = _fake_array
    mod.asarray = _fake_asarray
    mod.ascontiguousarray = _fake_ascontiguousarray
    mod.zeros = _fake_zeros
    mod.empty = _fake_empty
    mod.frombuffer = _fake_frombuffer
    mod.arange = _fake_arange
    mod.any = _fake_any
    mod.all = _fake_all
    mod.unique = _fake_unique
    mod.searchsorted = _fake_searchsorted
    mod.roll = _fake_roll
    mod.column_stack = _fake_column_stack
    mod.bitwise_or = _FakeBitwiseOr()
    mod.ndarray = _FakeArray
    return mod


def _load_calibrated_texture():
    """Load tools/calibrated_texture.py using real NumPy if present, else the stdlib shim."""
    if importlib.util.find_spec("numpy") is not None:
        import calibrated_texture
        return calibrated_texture, False
    fake_np = _make_fake_numpy()
    spec = importlib.util.spec_from_file_location(
        "calibrated_texture", str(TOOLS_DIR / "calibrated_texture.py")
    )
    mod = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {"numpy": fake_np}):
        spec.loader.exec_module(mod)
    return mod, True


ct, _USING_FAKE_NP = _load_calibrated_texture()
np = ct.np


# ---------------------------------------------------------------------------
# Synthetic RDS5 / ETPF pool and BCPack stream builders (no game data)
# ---------------------------------------------------------------------------

def write_custom_pool(path, records, chunks):
    """Write a synthetic RDS5/ETPF pool with arbitrary `(codec, payload, virt_size)` chunks."""
    count = len(chunks) + 2
    table_end = 24 + (count - 1) * 8
    etpf = ttpf.align(table_end, 256)
    area = len(records) * 24
    metadata = 16 + area + len(records) * 32
    padded = ttpf.align(metadata, 4096)

    source = etpf + padded
    virt = padded
    entries = []
    packed_chunks = []
    for codec, payload, vsize_chunk in chunks:
        assert vsize_chunk > 0 and vsize_chunk % 4096 == 0
        if codec == 0:
            assert len(payload) == vsize_chunk
            pad = 0
        else:
            pad = (-len(payload)) % 256
        entries.append((source // 256) | (codec << 26) | ((virt // 4096) << 30) | (pad << 52))
        packed_chunks.append((source, payload))
        source += len(payload) + pad
        virt += vsize_chunk

    vsize = virt
    entries.append((source // 256) | ((vsize // 4096) << 30))

    header = bytearray(24)
    header[0:4] = b"RDS5"
    struct.pack_into("<I", header, 4, count)
    struct.pack_into("<Q", header, 8, vsize | (22 << 56))
    struct.pack_into("<I", header, 16, (count + 31) // 32)
    struct.pack_into("<I", header, 20, 0)

    blob = bytearray(source)
    blob[0:24] = header
    for i, e in enumerate(entries):
        struct.pack_into("<Q", blob, 24 + i * 8, e)
    blob[etpf : etpf + 4] = b"ETPF"
    struct.pack_into("<III", blob, etpf + 4, 634, len(records), area)
    for i, rec in enumerate(records):
        blob[etpf + 16 + i * 16 : etpf + 16 + (i + 1) * 16] = rec
        struct.pack_into("<Q", blob, etpf + 16 + len(records) * 16 + i * 8, 0x1000 + i)
    for src, payload in packed_chunks:
        blob[src : src + len(payload)] = payload
    Path(path).write_bytes(bytes(blob))
    return padded


def _bc3_profile_from_tables():
    alpha = [a0 | (a1 << 3) for a0, a1 in bcpack_tables.BC3_ALPHA_PAIR]
    endpoints = {"-512": [0, 0, 0, 128, 0, 0, 0, 128]}
    for k, vals in (bcpack_tables.BC3_HISTORY_CONFIRMED | bcpack_tables.BC3_HISTORY_PROVISIONAL).items():
        a0, a1, r0, g0, b0, r1, g1, b1 = vals
        endpoints[str(k - 256)] = [r0, g0, b0, a0, r1, g1, b1, a1]
    return {
        "endpoints": endpoints,
        "color": list(bcpack_tables.BC3_COLOUR_QUADRANT),
        "alpha": alpha,
    }


def _bc1_profile_with_confirmed():
    prof = synthetic_profile()
    for entry, vals in bcpack_tables.BC1_HISTORY_CONFIRMED.items():
        prof["bc1"]["endpoints"][256 - entry] = list(vals)
    return prof


def _empty_bc7_section():
    return {
        "labels": [[0, 0, 0, 0] for _ in range(1024)],
        "anchors": [1] * 128,
        "seeds": {
            "base": [[[0, 8] for _ in range(56)] for _ in range(256)],
            "secondary": [[[0, 8] for _ in range(56)] for _ in range(256)],
            "early": [[[0, 8] for _ in range(56)] for _ in range(256)],
            "refined": [[[0, 8] for _ in range(56)] for _ in range(256)],
        },
        "adjustments": {},
    }


def _encode_bc4_from_lanes(lanes):
    return tb.encode_lanes(lanes, 4)


def _encode_bc5_from_lanes(lanes):
    return tb.encode_lanes(lanes, 5)


def _encode_bc3_from_lanes(lanes):
    return tb.encode_lanes(lanes, 3)


def _encode_bc6h_solid(nblocks):
    # Self-contained mode-11 (1-region, 4-bit index) zero-delta blocks.
    # For di=0, iref[0]=8; shift anchor pixel 0 by -8 so index 0 clears the anchor bit.
    lanes = [[0] * (2 * nblocks) for _ in range(16)]
    ix0 = tb._pack_ix_pair(tb._zig(-8), 0, 0)
    for i in range(nblocks):
        lanes[15][2 * i] = 10  # mode 11
        lanes[2][2 * i] = tb._zig(100 + (i & 15))
        lanes[11][2 * i] = ix0
    return tb.encode_lanes(lanes, 6)


def _encode_bc7_mode6_solid(nblocks):
    blk = ttm.solid_bc7(180, 90, 45)
    return tb.encode_bc7_mode6_blocks(blk * nblocks)


@contextmanager
def _fake_native_context(*, unpack_fn=None, rebuild_fn=None):
    """Patch ct.native() with a mock library supporting ctypes pointer reads/writes."""
    lib = mock.MagicMock()
    if unpack_fn is None:
        lib.unpack.side_effect = lambda *args: -1
    else:
        lib.unpack.side_effect = unpack_fn
    if rebuild_fn is not None:
        lib.rebuild.side_effect = rebuild_fn
    else:
        lib.rebuild.side_effect = lambda *args: None
    with mock.patch.object(ct, "native", return_value=lib):
        yield lib


def _read_u8_buf(ptr, n):
    return bytes((ctypes.c_ubyte * n).from_address(ptr)) if n else b""


def _write_u8_buf(ptr, data):
    if data:
        ctypes.memmove(ptr, data, len(data))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNativeLoader(unittest.TestCase):
    def setUp(self):
        ct.native.cache_clear()
        self.addCleanup(ct.native.cache_clear)

    def test_platform_library_names_and_ctypes_signatures(self):
        cases = (
            ("win32", "bcpack_calibration.dll"),
            ("darwin", "libbcpack_calibration.dylib"),
            ("linux", "libbcpack_calibration.so"),
            ("freebsd14", "libbcpack_calibration.so"),
        )
        expected_dir = Path(ct.__file__).with_name("bcpack_calibration") / "target/release"
        for plat, filename in cases:
            with self.subTest(platform=plat):
                ct.native.cache_clear()
                fake_lib = mock.MagicMock()
                with mock.patch.object(ct.sys, "platform", plat), \
                        mock.patch.object(ct.ctypes, "CDLL", return_value=fake_lib) as cdll:
                    got = ct.native()
                self.assertIs(got, fake_lib)
                cdll.assert_called_once_with(str(expected_dir / filename))
                self.assertEqual(
                    fake_lib.unpack.argtypes,
                    [
                        ctypes.c_char_p,
                        ctypes.c_size_t,
                        ctypes.c_char_p,
                        ctypes.c_size_t,
                        ctypes.c_size_t,
                        ctypes.c_void_p,
                    ],
                )
                self.assertIs(fake_lib.unpack.restype, ctypes.c_ssize_t)
                self.assertEqual(
                    fake_lib.rebuild.argtypes,
                    [ctypes.c_void_p, ctypes.c_size_t] + [ctypes.c_void_p] * 5,
                )
                self.assertIsNone(fake_lib.rebuild.restype)

    def test_native_caches_until_cleared_and_propagates_oserror(self):
        fake_lib = mock.MagicMock()
        with mock.patch.object(ct.ctypes, "CDLL", return_value=fake_lib) as cdll:
            self.assertIs(ct.native(), fake_lib)
            self.assertIs(ct.native(), fake_lib)
            self.assertEqual(cdll.call_count, 1)
            ct.native.cache_clear()
            self.assertIs(ct.native(), fake_lib)
            self.assertEqual(cdll.call_count, 2)

        ct.native.cache_clear()
        with mock.patch.object(ct.ctypes, "CDLL", side_effect=OSError("missing dll")) as cdll:
            with self.assertRaises(OSError):
                ct.native()
            with self.assertRaises(OSError):
                ct.native()
            self.assertEqual(cdll.call_count, 2)


class TestArrayAndMathHelpers(unittest.TestCase):
    def test_array_validation_and_contiguous_conversion(self):
        arr = ct.array([[0, 255], [-1, 128]], (2, 2), "int16", -1, 255)
        self.assertEqual(arr.shape, (2, 2))
        self.assertEqual(arr.dtype.kind, "i")
        self.assertEqual(arr.tobytes(), struct.pack("<4h", 0, 255, -1, 128))

        # Unsigned boundary 0..65535
        u = ct.array([0, 65535], (2,), "uint16", 0, 65535)
        self.assertEqual(u.tobytes(), struct.pack("<2H", 0, 65535))

        # Transposed non-contiguous input becomes contiguous in row-major order.
        src = np.array([[1, 2], [3, 4]]).T
        contig = ct.array(src, (2, 2), "uint8", 0, 10)
        self.assertEqual(contig.tobytes(), bytes([1, 3, 2, 4]))

        # Invalid shape, type, and out-of-range values
        for bad_val, shape, dt, lo, hi in (
            ([1, 2, 3], (2, 2), "int32", 0, 10),
            ([1.0, 2.0], (2,), "int32", 0, 10),
            ([True, False], (2,), "uint8", 0, 1),
            (["1", "2"], (2,), "uint8", 0, 10),
            ([-2, 0], (2,), "int16", -1, 10),
            ([0, 256], (2,), "uint8", 0, 255),
        ):
            with self.subTest(bad_val=bad_val, shape=shape):
                with self.assertRaisesRegex(ValueError, "Invalid calibration field"):
                    ct.array(bad_val, shape, dt, lo, hi)

    def test_unzig_matches_bcpack(self):
        vals = list(range(256))
        got = ct.unzig(np.array(vals, dtype="int32")).tolist()
        self.assertEqual(got, [bcpack._unzig(v) for v in vals])
        self.assertEqual(ct.unzig(0), 0)
        self.assertEqual(ct.unzig(1), -1)
        self.assertEqual(ct.unzig(2), 1)
        self.assertEqual(ct.unzig(255), -128)

    def test_distance_exhaustive_inter8_and_lane_selection(self):
        pairs = [(a, b) for a in range(16) for b in range(16)]
        # Shape (256, 2, 2): put pair in lane 1 and zero in lane 0.
        syms = [[[0, a], [0, b]] for a, b in pairs]
        arr = np.array(syms, dtype="uint8")
        self.assertEqual(ct.distance(arr, 0).tolist(), [0] * 256)
        self.assertEqual(ct.distance(arr, 1).tolist(), [bcpack._inter8(a, b) for a, b in pairs])

    def test_accumulate_chains_zero_and_external_roots(self):
        # Empty input
        empty_r = ct.accumulate(np.array([], dtype="int32"), np.zeros((0, 2), dtype="int32"))
        self.assertEqual(empty_r.shape, (0,))

        # Block 0: d=0 (-512); Block 1->0->-512; Block 2->1->0; Block 3->2->1->0 (depth 3);
        # Block 4: d=10 (external -> root = 4 - 10 = -6); Block 5->4 (root = -6).
        distances = np.array([0, 1, 1, 1, 10, 1, 2, 0], dtype="int32")
        residuals = np.array(
            [[10, 1], [2, 3], [4, 5], [6, 7], [100, 200], [5, -10], [7, 8], [9, 10]],
            dtype="int32",
        )
        roots = ct.accumulate(distances, residuals)
        self.assertEqual(roots.tolist(), [-512, -512, -512, -512, -6, -6, -6, -512])
        self.assertEqual(
            residuals.tolist(),
            [[10, 1], [12, 4], [16, 9], [22, 16], [100, 200], [105, 190], [107, 208], [9, 10]],
        )

    def test_addresses_matches_detile_across_4k_and_64k_tiles(self):
        ct.addresses.cache_clear()
        for tile in (4096, 65536):
            for fmt, bs in ((71, 8), (98, 16)):
                with self.subTest(tile=tile, bs=bs):
                    lay = ttpf.surface_layout(256, 192, fmt, 8, tile)
                    bw, bh = lay["block_w"], lay["block_h"]
                    for mip in (lay["mips"][0], lay["mips"][-1]):
                        w, h = mip["blocks_w"], mip["blocks_h"]
                        x0, y0 = mip["tile_x"], mip["tile_y"]
                        got = ct.addresses(tile, bs, bw, bh, w, h, x0, y0).tolist()
                        # Compare against ttpf.detile by placing unique byte markers at each offset.
                        # Note detile checks offset within mip["region"]; fill a synthetic region with
                        # 8-byte / 16-byte tagged offsets.
                        region = bytearray(mip["region"])
                        for off in range(0, len(region), bs):
                            struct.pack_into("<Q", region, off, off)
                        detiled = ttpf.detile(lay, mip, bytes(region))
                        expected = [
                            struct.unpack_from("<Q", detiled, i * bs)[0] for i in range(w * h)
                        ]
                        self.assertEqual(got, expected)
        info_before = ct.addresses.cache_info()
        ct.addresses(4096, 8, 32, 16, 4, 4, 0, 0)
        ct.addresses(4096, 8, 32, 16, 4, 4, 0, 0)
        info_after = ct.addresses.cache_info()
        self.assertEqual(info_after.hits, info_before.hits + 1)


class TestBlockRebuildersParity(unittest.TestCase):
    def test_bc1_blocks_parity_chains_and_missing_history(self):
        prof = _bc1_profile_with_confirmed()
        seeds = ct.array(prof["bc1"]["endpoints"], (256, 6), "int32", -1, 63)
        selectors = ct.array(prof["bc1"]["selectors"], (4, 256), "uint32", 0, 0xFFFFFFFF)

        # Block 0: external confirmed entry 1 (d=255 -> 256+0-255=1 -> age=255)
        # Block 1: d=1 -> chains onto Block 0 (external root)
        # Block 2: d=0 -> self-contained with negative endpoint wrap
        # Block 3: d=1 -> chains onto Block 2
        lanes = [
            [tb._zig(-1), tb._zig(1), 0, tb._zig(1)],  # l0: -1 mod 256 = 255
            [0, 0, 0, 0],
            [tb._zig(2), tb._zig(-3), tb._zig(-1), tb._zig(4)],
            [tb._zig(-4), tb._zig(5), tb._zig(-2), tb._zig(10)],
            [tb._zig(1), tb._zig(2), tb._zig(-3), tb._zig(7)],
            [tb._zig(-1), tb._zig(3), tb._zig(-5), tb._zig(2)],
            [tb._zig(6), tb._zig(-2), tb._zig(-1), tb._zig(8)],
            [tb._zig(0), tb._zig(1), tb._zig(-2), tb._zig(3)],
            [0, 17, 128, 255],
            [1, 33, 64, 200],
            [2, 99, 150, 10],
            [3, 210, 44, 77],
        ]
        sy = np.ascontiguousarray(np.asarray(lanes, dtype="uint8").T)
        got = ct.bc1_blocks(sy, (seeds, selectors))
        expected = bcpack._bc1_blocks(lanes, 4, None)
        self.assertEqual(got, expected)

        # Missing external history (provisional entry 6 -> age 250 has -1 in seeds) raises ValueError.
        bad_seeds = ct.array([[-1] * 6 for _ in range(256)], (256, 6), "int32", -1, 63)
        lanes_ext = [[tb._zig(-6)] if k == 0 else [0] for k in range(12)]
        sy_ext = np.ascontiguousarray(np.asarray(lanes_ext, dtype="uint8").T)
        with self.assertRaisesRegex(ValueError, "Missing BC1 endpoint history"):
            ct.bc1_blocks(sy_ext, (bad_seeds, selectors))

        # Self-contained stream (d=0) succeeds even with -1 in all external seeds.
        lanes_self = [[k * 3 % 32] for k in range(12)]
        lanes_self[0] = [0]
        lanes_self[1] = [0]
        sy_self = np.ascontiguousarray(np.asarray(lanes_self, dtype="uint8").T)
        self.assertEqual(ct.bc1_blocks(sy_self, (bad_seeds, selectors)), bcpack._bc1_blocks(lanes_self, 1, None))

    def test_bc7_blocks_invokes_native_rebuild(self):
        sy = np.array([[i for i in range(16)] for _ in range(4)], dtype="uint8")  # 2 blocks
        labels = ct.array([[1, 2, 0, 0] for _ in range(1024)], (1024, 4), "int16", -1, 255)
        anchors = ct.array([7] * 128, (128,), "uint16", 0, 65535)
        seed = ct.array([[[0, 8] for _ in range(56)] for _ in range(256)], (256, 56, 2), "uint8", 0, 255)

        b0 = bytes(range(16))
        b1 = bytes(range(16, 32))
        seen = {}

        def fake_rebuild(s_ptr, count, lab_ptr, seed_ptr, anc_ptr, out_ptr, ok_ptr):
            seen["args"] = (s_ptr, count, lab_ptr, seed_ptr, anc_ptr, out_ptr, ok_ptr)
            seen["s_bytes"] = _read_u8_buf(s_ptr, count * 32)
            _write_u8_buf(out_ptr, b0 + b1)
            _write_u8_buf(ok_ptr, bytes([1, 0]))

        with _fake_native_context(rebuild_fn=fake_rebuild) as lib:
            out, ok = ct.bc7_blocks(sy, (labels, anchors), seed)
        self.assertEqual(lib.rebuild.call_count, 1)
        self.assertEqual(seen["args"][1], 2)
        self.assertEqual(seen["args"][2], labels.ctypes.data)
        self.assertEqual(seen["args"][3], seed.ctypes.data)
        self.assertEqual(seen["args"][4], anchors.ctypes.data)
        self.assertEqual(seen["s_bytes"], sy.tobytes())
        self.assertEqual(out.tobytes(), b0 + b1)
        self.assertEqual(ok.tolist(), [True, False])

        # Zero blocks
        empty_sy = np.zeros((0, 16), dtype="uint8")
        with _fake_native_context() as lib:
            out0, ok0 = ct.bc7_blocks(empty_sy, (labels, anchors), seed)
        self.assertEqual(out0.shape, (0,))
        self.assertEqual(ok0.shape, (0,))
        self.assertEqual(lib.rebuild.call_args[0][1], 0)

    def test_scalar_bc4_parity_and_missing_slot(self):
        table = dict(bcpack_tables.BC4_HISTORY_PROVISIONAL)
        # Block 0: d=0; Block 1: d=1; Block 2: external slot 25 (d=256+2-25=233 -> unzig(-23))
        d2 = tb._zig(-23)
        lanes = [
            [0, tb._zig(1), d2],
            [0, 0, 0],
            [tb._zig(30), tb._zig(-50), tb._zig(-100)],
            [tb._zig(-120), tb._zig(45), tb._zig(20)],
            [0, 12, 33], [1, 15, 44], [2, 20, 55], [3, 25, 63],
            [4, 30, 7], [5, 35, 8], [6, 40, 9], [7, 45, 10],
        ]
        sy = np.ascontiguousarray(np.asarray(lanes, dtype="uint8").T)
        got = ct.scalar_bc4(sy, table)
        expected = bcpack._bc4_blocks(lanes, 3, set())
        self.assertEqual(got, expected)

        with self.assertRaises(KeyError):
            ct.scalar_bc4(sy, {})

    def test_local_bc5_parity_and_missing_slot(self):
        # Block 0: confirmed entry 182 (d=256+0-182=74)
        # Block 1: d=1 (chains onto Block 0)
        # Block 2: provisional entry 3 (d=256+2-3=255 -> -1 mod 256)
        # Block 3: distance=0 (seed 128)
        # local_bc5 uses compatibility endpoint ordering where s[:, 0, 2:4] holds
        # (r0, r1) and s[:, 1, 2:4] holds (g0, g1), while selector lanes 4..11
        # match bcpack._bc5_blocks identically.
        d0a, d0b = tb._deint4(tb._zig(74) & 255)
        d1a, d1b = tb._deint4(tb._zig(1) & 255)
        d2a, d2b = tb._deint4(tb._zig(-1) & 255)
        lanes = [[] for _ in range(12)]
        lanes[0] = [d0a, d0b, d1a, d1b, d2a, d2b, 0, 0]
        lanes[1] = [0] * 8
        lanes[2] = [tb._zig(10), tb._zig(-20), tb._zig(5), tb._zig(-15), tb._zig(30), tb._zig(40), tb._zig(-60), tb._zig(70)]
        lanes[3] = [tb._zig(-5), tb._zig(25), tb._zig(-10), tb._zig(35), tb._zig(-40), tb._zig(12), tb._zig(80), tb._zig(-90)]
        for ln in range(4, 12):
            lanes[ln] = [(ln * 7 + k * 5) & 63 for k in range(8)]
        sy = np.ascontiguousarray(np.asarray(lanes, dtype="uint8").T)
        got = ct.local_bc5(sy)

        # Verify selector bytes (bytes 2..7 and 10..15 of each 16-byte block) match bcpack._bc5_blocks,
        # and endpoint bytes match the compatibility ordering against BC5_HISTORY_* tables.
        ref = bcpack._bc5_blocks(lanes, 4, set())
        self.assertEqual(len(got), 64)
        h182 = bcpack_tables.BC5_HISTORY_CONFIRMED[182]
        h3 = bcpack_tables.BC5_HISTORY_PROVISIONAL[3]
        # Block 0: base h182 reshaped (2, 2) -> [[h182[0], h182[1]], [h182[2], h182[3]]]
        b0_ep = (
            (h182[0] + 10) % 256,
            (h182[1] - 5) % 256,
            (h182[2] - 20) % 256,
            (h182[3] + 25) % 256,
        )
        # Block 1: chains onto Block 0
        b1_ep = (
            (b0_ep[0] + 5) % 256,
            (b0_ep[1] - 10) % 256,
            (b0_ep[2] - 15) % 256,
            (b0_ep[3] + 35) % 256,
        )
        # Block 2: base h3
        b2_ep = (
            (h3[0] + 30) % 256,
            (h3[1] - 40) % 256,
            (h3[2] + 40) % 256,
            (h3[3] + 12) % 256,
        )
        # Block 3: d=0 -> seed (128, 128, 128, 128)
        b3_ep = ((128 - 60) % 256, (128 + 80) % 256, (128 + 70) % 256, (128 - 90) % 256)
        for bi, ep in enumerate((b0_ep, b1_ep, b2_ep, b3_ep)):
            off = bi * 16
            self.assertEqual((got[off], got[off + 1], got[off + 8], got[off + 9]), ep)
            self.assertEqual(got[off + 2 : off + 8], ref[off + 2 : off + 8])
            self.assertEqual(got[off + 10 : off + 16], ref[off + 10 : off + 16])

        # Unknown history entry 1 (neither confirmed nor provisional) raises KeyError.
        u0a, u0b = tb._deint4(tb._zig(-1) & 255)
        bad_lanes = [[u0a, u0b] if k == 0 else [0, 0] for k in range(12)]
        bad_sy = np.ascontiguousarray(np.asarray(bad_lanes, dtype="uint8").T)
        with self.assertRaises(KeyError):
            ct.local_bc5(bad_sy)

    def test_local_bc3_parity_and_missing_slot(self):
        profile = _bc3_profile_from_tables()
        # Block 0: d=0 (slot -512)
        # Block 1: d=1 (chains onto Block 0)
        # Block 2: provisional entry 26 (d=256+2-26=232 -> unzig(-24) & 255)
        d0a, d0b = 0, 0
        d1a, d1b = tb._deint4(tb._zig(1) & 255)
        d2a, d2b = tb._deint4(tb._zig(-24) & 255)
        lanes = [[] for _ in range(12)]
        lanes[0] = [d0a, d0b, d1a, d1b, d2a, d2b]
        lanes[1] = [0] * 6
        lanes[2] = [tb._zig(20), tb._zig(-30), tb._zig(5), tb._zig(-10), tb._zig(40), tb._zig(-50)]
        for ln in range(3, 7):
            lanes[ln] = [(ln * 11 + k * 7) & 63 for k in range(6)]
        lanes[7] = [tb._zig(7), tb._zig(-3), tb._zig(4), tb._zig(-2), tb._zig(12), tb._zig(-8)]
        lanes[8] = [tb._zig(-15), tb._zig(22), tb._zig(3), tb._zig(-9), tb._zig(18), tb._zig(25)]
        lanes[9] = [tb._zig(9), tb._zig(-12), tb._zig(-1), tb._zig(6), tb._zig(-14), tb._zig(11)]
        lanes[10] = [12, 45, 78, 123, 200, 251]
        lanes[11] = [33, 67, 101, 144, 188, 229]

        sy = np.ascontiguousarray(np.asarray(lanes, dtype="uint8").T)
        got = ct.local_bc3(sy, profile)
        expected = bcpack._bc3_blocks(lanes, 3, set())
        self.assertEqual(got, expected)

        with self.assertRaises(KeyError):
            ct.local_bc3(sy, {"endpoints": {}, "color": profile["color"], "alpha": profile["alpha"]})


class TestLocalDecoder(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def _open_pool(self, name, records, chunks):
        path = self.root / name
        base = write_custom_pool(path, records, chunks)
        pool = ttpf.Pool(str(path))
        self.addCleanup(pool.close)
        return pool, base

    def test_init_rejects_bad_calibration_versions_and_isolates_cache(self):
        pool, base = self._open_pool(
            "raw.ttpf",
            [ttm.pack_etpf_record(64, 64, 1, 0, 71, 1, 1, 4, 4096)],
            [(0, b"\xAA" * 4096, 4096)],
        )
        for bad in (None, [], "v1", 1, {}, {"version": 0}, {"version": 2}, {"version": "1"}):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, "Unsupported calibration version"):
                    ct.LocalDecoder(pool, bad)

        dec1 = ct.LocalDecoder(pool, {"version": 1})
        dec2 = ct.LocalDecoder(pool, {"version": 1})
        self.assertEqual(dec1.edges.tolist(), [base, base + 4096])
        b1, m1 = dec1.read_chunk(0, 8, "base")
        self.assertIsNone(m1)
        self.assertEqual(len(b1), 512)
        self.assertEqual(dec1.read_chunk.cache_info().currsize, 1)
        self.assertEqual(dec2.read_chunk.cache_info().currsize, 0)

    def test_seed_validation_precision_and_caching(self):
        pool, _ = self._open_pool(
            "dummy.ttpf",
            [ttm.pack_etpf_record(64, 64, 1, 0, 98, 1, 1, 4, 4096)],
            [(0, b"\x00" * 4096, 4096)],
        )
        cal = {"version": 1, "bc7": _empty_bc7_section()}
        dec = ct.LocalDecoder(pool, cal)
        s1 = dec.seed("base")
        s2 = dec.seed("base")
        self.assertIs(s1, s2)
        with self.assertRaises(KeyError):
            dec.seed("nonexistent")

        cal["bc7"]["seeds"]["bad_prec"] = [[[0, 9] for _ in range(56)] for _ in range(256)]
        with self.assertRaisesRegex(ValueError, "Invalid history precision"):
            dec.seed("bad_prec")

    def test_symbols_native_fallback_compatibility_and_errors(self):
        # Build a valid 64x64 1-mip BC1 BCPack chunk (512 blocks = 4096 bytes)
        layout = ttpf.surface_layout(64, 64, 71, 1, 4096)
        linear = ttm.solid_chain(layout, 71)[0]
        tiled = bytearray(layout["size"])
        ttm.retile_to(layout, layout["mips"][0], linear, tiled)
        packed_bc1 = tb.encode_bc1_blocks(bytes(tiled))
        chunk_sha = hashlib.sha256(packed_bc1).hexdigest()

        # Chunk 0: valid BC1 BCPack
        # Chunk 1: valid BC1 BCPack with an extra trailing 0x00 byte so decode_lanes consumes < packed
        # Chunk 2: valid BC4 BCPack (unused compatibility branch when codec != 1)
        bc4_lanes = [[0] * 512 for _ in range(12)]
        packed_bc4 = _encode_bc4_from_lanes(bc4_lanes)
        pool, _ = self._open_pool(
            "syms.ttpf",
            [ttm.pack_etpf_record(64, 64, 1, 0, 71, 1, 1, 4, 4096)],
            [
                (1, packed_bc1, 4096),
                (1, packed_bc1 + b"\x00", 4096),
                (4, packed_bc4, 4096),
            ],
        )

        # 1. Native unpack succeeds (returns packed) -> bcpack.decode_lanes is NOT called.
        dec_nat = ct.LocalDecoder(pool, {"version": 1})
        expected_lanes, _ = bcpack.decode_lanes(packed_bc1, 1, 4096)
        expected_sy = np.ascontiguousarray(np.asarray(expected_lanes, dtype="uint8").T)

        def fake_unpack(raw, packed, widths_bytes, nlanes, nsym, sy_ptr):
            self.assertEqual(raw, packed_bc1)
            self.assertEqual(packed, len(packed_bc1))
            self.assertEqual(widths_bytes, bytes(w - 1 for w in bcpack.LANE_WIDTHS[1]))
            self.assertEqual(nlanes, 12)
            self.assertEqual(nsym, 512)
            _write_u8_buf(sy_ptr, expected_sy.tobytes())
            return packed

        with _fake_native_context(unpack_fn=fake_unpack), \
                mock.patch.object(bcpack, "decode_lanes", side_effect=AssertionError("unexpected fallback")):
            got_sy = dec_nat.symbols(0)
        self.assertEqual(got_sy.tobytes(), expected_sy.tobytes())

        # 2. Native unpack returns -1 -> falls back to bcpack.decode_lanes.
        with _fake_native_context():
            got_fallback = dec_nat.symbols(0)
        self.assertEqual(got_fallback.tobytes(), expected_sy.tobytes())

        # 3. Trailing unstripped zero byte makes consumed != packed -> ValueError.
        with _fake_native_context():
            with self.assertRaisesRegex(ValueError, "Entropy decode failed for chunk 1"):
                dec_nat.symbols(1)

        # 4. Compatibility rotation on codec 1 rotates selected flat positions left by 1.
        cycle = [1, 5, 13]
        dec_rot = ct.LocalDecoder(pool, {"version": 1, "compatibility": {chunk_sha: cycle}})
        with _fake_native_context():
            rotated = dec_rot.symbols(0)
        flat_orig = expected_sy.ravel().tolist()
        flat_rot = rotated.ravel().tolist()
        self.assertEqual(flat_rot[1], flat_orig[5])
        self.assertEqual(flat_rot[5], flat_orig[13])
        self.assertEqual(flat_rot[13], flat_orig[1])
        for idx in range(len(flat_orig)):
            if idx not in cycle:
                self.assertEqual(flat_rot[idx], flat_orig[idx])

        # 5. Compatibility rotation errors (duplicate, out-of-range, negative, non-int including bool).
        for bad_cycle in ([0, 0], [-1], [expected_sy.size], [True], [1.0], ["0"]):
            with self.subTest(bad_cycle=bad_cycle):
                dec_bad = ct.LocalDecoder(
                    pool, {"version": 1, "compatibility": {chunk_sha: bad_cycle}}
                )
                with _fake_native_context():
                    with self.assertRaisesRegex(ValueError, "Invalid symbol rotation"):
                        dec_bad.symbols(0)

        # 6. Compatibility does NOT run for non-BC1 codecs, empty cycle, or unknown sha.
        bc4_sha = hashlib.sha256(packed_bc4).hexdigest()
        dec_bc4 = ct.LocalDecoder(pool, {"version": 1, "compatibility": {bc4_sha: [0, 0]}})
        with _fake_native_context():
            sy4 = dec_bc4.symbols(2)
        self.assertEqual(sy4.shape, (512, 12))

        dec_noop = ct.LocalDecoder(
            pool,
            {"version": 1, "compatibility": {chunk_sha: [], "0" * 64: [0, 1]}},
        )
        with _fake_native_context():
            self.assertEqual(dec_noop.symbols(0).tobytes(), expected_sy.tobytes())

    def test_read_chunk_errors_and_all_codecs(self):
        # Raw (codec 0), zlib (codec 8), and bad-length zlib chunk
        raw_bytes = bytes((i & 0xFF) for i in range(4096))
        zlib_bytes = zlib.compress(raw_bytes)
        bad_zlib = zlib.compress(raw_bytes[:2048])

        # Codec 6 (BC6H) and Codec 7 (BC7)
        packed_bc6h = _encode_bc6h_solid(256)
        packed_bc7 = _encode_bc7_mode6_solid(256)

        pool, _ = self._open_pool(
            "codecs.ttpf",
            [ttm.pack_etpf_record(64, 64, 1, 0, 71, 1, 1, 4, 4096)],
            [
                (0, raw_bytes, 4096),
                (8, zlib_bytes, 4096),
                (8, bad_zlib, 4096),
                (6, packed_bc6h, 4096),
                (7, packed_bc7, 4096),
            ],
        )
        cal = {
            "version": 1,
            "bc6h": {},
            "bc7": _empty_bc7_section(),
        }
        dec = ct.LocalDecoder(pool, cal)

        # Size/block-size/codec mismatches (including unknown codec 99)
        with self.assertRaisesRegex(ValueError, "Block size and codec disagree"):
            dec.read_chunk(0, 7, "base")  # 4096 % 7 != 0
        with self.assertRaisesRegex(ValueError, "Block size and codec disagree"):
            dec.read_chunk(3, 8, "base")  # codec 6 is 16-byte blocks, not 8
        orig_chunks = list(pool.chunks)
        try:
            pool.chunks = orig_chunks + [(0, 4096, 0, 4096, 99, 0)]
            with self.assertRaisesRegex(ValueError, "Block size and codec disagree"):
                dec.read_chunk(len(orig_chunks), 16, "base")
        finally:
            pool.chunks = orig_chunks

        # Raw & zlib
        b_raw, m_raw = dec.read_chunk(0, 8, "base")
        self.assertIsNone(m_raw)
        self.assertEqual(b_raw.tobytes(), raw_bytes)
        b_z, m_z = dec.read_chunk(1, 16, "base")
        self.assertIsNone(m_z)
        self.assertEqual(b_z.tobytes(), raw_bytes)
        with self.assertRaisesRegex(ValueError, "Invalid raw chunk length"):
            dec.read_chunk(2, 8, "base")

        # BC6H delegates to bcpack._bc6h_blocks
        with _fake_native_context():
            b6, m6 = dec.read_chunk(3, 16, "base")
        self.assertIsNone(m6)
        self.assertEqual(b6.tobytes(), bcpack.decode(packed_bc6h, 6, 4096))

        # BC7 profile cascades: early (1 call), base (early stop if all valid), base->secondary, refined->3
        call_seeds = []

        def rebuild_early_stop(s_ptr, count, lab_ptr, seed_ptr, anc_ptr, out_ptr, ok_ptr):
            call_seeds.append(_read_u8_buf(seed_ptr, 8))
            _write_u8_buf(out_ptr, b"\x11" * (count * 16))
            _write_u8_buf(ok_ptr, b"\x01" * count)

        with _fake_native_context(rebuild_fn=rebuild_early_stop):
            b7_base, m7_base = dec.read_chunk(4, 16, "base")
            self.assertEqual(len(call_seeds), 1)
            self.assertTrue(m7_base.all())
            self.assertEqual(b7_base.tobytes(), b"\x11" * 4096)

            call_seeds.clear()
            dec.read_chunk(4, 16, "early")
            self.assertEqual(len(call_seeds), 1)

        # Partial fill across base -> secondary -> refined
        cal2 = {"version": 1, "bc7": _empty_bc7_section()}
        # Mark first byte of each seed so mocked rebuild distinguishes them
        cal2["bc7"]["seeds"]["base"][0][0] = [1, 8]
        cal2["bc7"]["seeds"]["secondary"][0][0] = [2, 8]
        cal2["bc7"]["seeds"]["refined"][0][0] = [3, 8]
        dec2 = ct.LocalDecoder(pool, cal2)
        seen_tags = []

        def rebuild_cascading(s_ptr, count, lab_ptr, seed_ptr, anc_ptr, out_ptr, ok_ptr):
            tag = _read_u8_buf(seed_ptr, 1)[0]
            seen_tags.append(tag)
            out_buf = bytearray(count * 16)
            ok_buf = bytearray(count)
            for idx in range(count):
                if (tag == 1 and idx % 3 == 0) or (tag == 2 and idx % 3 == 1):
                    out_buf[idx * 16 : (idx + 1) * 16] = bytes([tag]) * 16
                    ok_buf[idx] = 1
                elif tag == 3 and idx == 2:
                    out_buf[idx * 16 : (idx + 1) * 16] = bytes([3]) * 16
                    ok_buf[idx] = 1
            _write_u8_buf(out_ptr, bytes(out_buf))
            _write_u8_buf(ok_ptr, bytes(ok_buf))

        with _fake_native_context(rebuild_fn=rebuild_cascading):
            blocks, valid = dec2.read_chunk(4, 16, "refined")
        self.assertEqual(seen_tags, [1, 2, 3])
        self.assertTrue(bool(valid[0]) and bool(valid[1]) and bool(valid[2]))
        self.assertFalse(bool(valid[5]))  # 5 % 3 == 2, != 2
        self.assertEqual(blocks.tobytes()[:48], b"\x01" * 16 + b"\x02" * 16 + b"\x03" * 16)

    def test_surface_bounds_multichunk_and_bc7_adjustments(self):
        # Chunk 0: 4096 bytes raw (bs=16, 256 blocks)
        # Chunk 1: 4096 bytes BC7 (bs=16, 256 blocks)
        c0 = b"".join(bytes([i & 0xFF]) * 16 for i in range(256))
        c1 = _encode_bc7_mode6_solid(256)
        pool, base = self._open_pool(
            "surface.ttpf",
            [ttm.pack_etpf_record(64, 64, 1, 0, 98, 1, 1, 4, 4096)],
            [(0, c0, 4096), (7, c1, 4096)],
        )
        cal = {"version": 1, "bc7": _empty_bc7_section()}
        dec = ct.LocalDecoder(pool, cal)

        # Empty address array returns empty bytes cleanly
        self.assertEqual(dec.surface(np.array([], dtype="int64"), 16, "base"), b"")

        # Out of bounds
        for bad_at in ([base - 16], [base + 8192], [base, base + 10000]):
            with self.subTest(bad_at=bad_at):
                with self.assertRaisesRegex(ValueError, "Texture exceeds virtual pool"):
                    dec.surface(np.array(bad_at, dtype="int64"), 16, "base")

        # Non-monotonic multi-chunk lookup where unreferenced blocks in chunk 1 remain unresolved
        def rebuild_partial(s_ptr, count, lab_ptr, seed_ptr, anc_ptr, out_ptr, ok_ptr):
            seed_bytes = _read_u8_buf(seed_ptr, 256 * 56 * 2)
            # Check if adjustment patched seed[7, 3] = (99, 8)
            patched = seed_bytes[(7 * 56 + 3) * 2 : (7 * 56 + 3) * 2 + 2] == bytes([99, 8])
            out_buf = bytearray(count * 16)
            ok_buf = bytearray(count)
            for idx in range(count):
                if idx == 10:
                    out_buf[idx * 16 : (idx + 1) * 16] = b"\xAA" * 16
                    ok_buf[idx] = 1
                elif idx == 20 and patched:
                    out_buf[idx * 16 : (idx + 1) * 16] = b"\xBB" * 16
                    ok_buf[idx] = 1
            _write_u8_buf(out_ptr, bytes(out_buf))
            _write_u8_buf(ok_ptr, bytes(ok_buf))

        with _fake_native_context(rebuild_fn=rebuild_partial):
            # Request block 10 of chunk 1 (resolved), block 5 of chunk 0, block 0 of chunk 0
            at_ok = np.array([base + 4096 + 10 * 16, base + 5 * 16, base], dtype="int64")
            got_ok = dec.surface(at_ok, 16, "base")
            self.assertEqual(got_ok, b"\xAA" * 16 + bytes([5]) * 16 + bytes([0]) * 16)

            # Request unresolved block 20 with profile != "refined" -> Missing calibrated blocks
            at_miss = np.array([base + 4096 + 10 * 16, base + 4096 + 20 * 16], dtype="int64")
            with self.assertRaisesRegex(ValueError, "Missing calibrated blocks"):
                dec.surface(at_miss, 16, "base")

            # Refined profile: compute exact sha256 key expected by LocalDecoder.surface
            sy1 = dec.symbols(1)
            miss_ix = np.array([20], dtype="<i8")
            adj_key = hashlib.sha256(sy1.tobytes() + miss_ix.tobytes()).hexdigest()

            # Bad fitted endpoint bounds
            for bad_adj in (
                [(256, 0, 0)],
                [(-1, 0, 0)],
                [(0, 24, 0)],
                [(0, -1, 0)],
                [(0, 0, 256)],
                [(0, 0, -1)],
            ):
                cal["bc7"]["adjustments"][adj_key] = bad_adj
                with self.assertRaisesRegex(ValueError, "Invalid fitted endpoint"):
                    dec.surface(at_miss, 16, "refined")

            # Valid bounds that still fail to resolve block 20 -> Incomplete calibrated endpoints
            cal["bc7"]["adjustments"][adj_key] = [(7, 3, 42)]
            with self.assertRaisesRegex(ValueError, "Incomplete calibrated endpoints"):
                dec.surface(at_miss, 16, "refined")

            # Valid adjustment resolves block 20 without mutating cached refined seed!
            cal["bc7"]["adjustments"][adj_key] = [(7, 3, 99)]
            got_ref = dec.surface(at_miss, 16, "refined")
            self.assertEqual(got_ref, b"\xAA" * 16 + b"\xBB" * 16)
            self.assertEqual(dec.seed("refined").tobytes()[(7 * 56 + 3) * 2 : (7 * 56 + 3) * 2 + 2], b"\x00\x08")

    def test_dds_end_to_end_codecs_and_layouts(self):
        # 1. BC3 (fmt 77, codec 3), BC4 (fmt 80, codec 4), BC5 (fmt 83, codec 5),
        #    BC6H (fmt 95, codec 6), BC7 (fmt 98, codec 7) across 4K/64K, partial mips,
        #    non-square, all-tail, single-mip, and multi-chunk spans.
        for fmt, codec, w, h, mips, mode in (
            (77, 3, 64, 32, 6, 4),    # BC3, 4K tile, non-square
            (80, 4, 16, 16, 5, 4),    # BC4, 4K tile, all-tail (first_tail == 0)
            (83, 5, 64, 64, 1, 4),    # BC5, 4K tile, single-mip
            (95, 6, 64, 64, 5, 4),    # BC6H, 4K tile
            (98, 7, 16, 16, 1, 0),    # BC7, 64K tile, small single-mip
        ):
            with self.subTest(fmt=fmt, codec=codec, dims=(w, h, mips)):
                tile = ttpf.tile_bytes(mode)
                lay = ttpf.surface_layout(w, h, fmt, mips, tile)
                linear = ttm.solid_chain(lay, fmt)
                tiled = bytearray(lay["size"])
                for m in lay["mips"]:
                    ttm.retile_to(lay, m, linear[m["index"]], tiled)

                eb = lay["element_bytes"]
                nblocks = lay["size"] // eb
                if codec == 3:
                    inv_pair = {p: i for i, p in enumerate(bcpack_tables.BC3_ALPHA_PAIR)}
                    inv_quad = {v: i for i, v in enumerate(bcpack_tables.BC3_COLOUR_QUADRANT)}
                    lanes = [[] for _ in range(12)]
                    prev = (128, 128, 0, 0, 0, 0, 0, 0)
                    for i in range(nblocks):
                        blk = tiled[i * 16 : (i + 1) * 16]
                        a0, a1 = blk[0], blk[1]
                        abits = int.from_bytes(blk[2:8], "little")
                        c0, c1, r0, r1, r2, r3 = struct.unpack_from("<HHBBBB", blk, 8)
                        ep = (
                            a0, a1,
                            (c0 >> 11) & 31, (c0 >> 5) & 63, c0 & 31,
                            (c1 >> 11) & 31, (c1 >> 5) & 63, c1 & 31,
                        )
                        mods = bcpack._BC3_MODS
                        deltas = [((ep[k] - prev[k] + mods[k] // 2) % mods[k]) - mods[k] // 2 for k in range(8)]
                        # Use d=0 on block 0, d=1 afterwards
                        if i == 0:
                            lanes[0].extend([0, 0])
                        else:
                            d1a, d1b = tb._deint4(tb._zig(1) & 255)
                            lanes[0].extend([d1a, d1b])
                        lanes[1].extend([0, 0])
                        lanes[2].extend([tb._zig(deltas[0]), tb._zig(deltas[1])])
                        for r in range(4):
                            row = (abits >> (12 * r)) & 0xFFF
                            lanes[3 + r].extend([
                                inv_pair[(row & 7, (row >> 3) & 7)],
                                inv_pair[((row >> 6) & 7, (row >> 9) & 7)],
                            ])
                        lanes[7].extend([tb._zig(deltas[2]), tb._zig(deltas[5])])
                        lanes[8].extend([tb._zig(deltas[3]), tb._zig(deltas[6])])
                        lanes[9].extend([tb._zig(deltas[4]), tb._zig(deltas[7])])
                        qa = (r0 & 15) | ((r1 & 15) << 4)
                        qb = (r0 >> 4) | ((r1 >> 4) << 4)
                        qc = (r2 & 15) | ((r3 & 15) << 4)
                        qd = (r2 >> 4) | ((r3 >> 4) << 4)
                        lanes[10].extend([inv_quad[qa], inv_quad[qb]])
                        lanes[11].extend([inv_quad[qc], inv_quad[qd]])
                        prev = ep
                    packed_chunk = _encode_bc3_from_lanes(lanes)
                elif codec == 4:
                    inv_pair = {p: i for i, p in enumerate(bcpack_tables.BC4_PAIR)}
                    lanes = [[] for _ in range(12)]
                    for i in range(nblocks):
                        blk = tiled[i * 8 : (i + 1) * 8]
                        a0, a1 = blk[0], blk[1]
                        bits = int.from_bytes(blk[2:8], "little")
                        lanes[0].append(0)
                        lanes[1].append(0)
                        lanes[2].append(tb._zig(((a0 - 128 + 128) % 256) - 128))
                        lanes[3].append(tb._zig(((a1 - 128 + 128) % 256) - 128))
                        for r in range(4):
                            row = (bits >> (12 * r)) & 0xFFF
                            lanes[4 + 2 * r].append(inv_pair[(row & 7, (row >> 3) & 7)])
                            lanes[4 + 2 * r + 1].append(inv_pair[((row >> 6) & 7, (row >> 9) & 7)])
                    packed_chunk = _encode_bc4_from_lanes(lanes)
                elif codec == 5:
                    inv_pair = {p: i for i, p in enumerate(bcpack_tables.BC5_PAIR)}
                    lanes = [[] for _ in range(12)]
                    for i in range(nblocks):
                        blk = tiled[i * 16 : (i + 1) * 16]
                        r0, r1 = blk[0], blk[1]
                        br = int.from_bytes(blk[2:8], "little")
                        g0, g1 = blk[8], blk[9]
                        bg = int.from_bytes(blk[10:16], "little")
                        lanes[0].extend([0, 0])
                        lanes[1].extend([0, 0])
                        # Compatibility endpoint ordering: half 0 -> (r0, r1), half 1 -> (g0, g1)
                        lanes[2].extend([tb._zig(((r0 - 128 + 128) % 256) - 128), tb._zig(((g0 - 128 + 128) % 256) - 128)])
                        lanes[3].extend([tb._zig(((r1 - 128 + 128) % 256) - 128), tb._zig(((g1 - 128 + 128) % 256) - 128)])
                        for ch, bval in enumerate((br, bg)):
                            for r in range(4):
                                row = (bval >> (12 * r)) & 0xFFF
                                lanes[4 + 4 * ch + r].extend([
                                    inv_pair[(row & 7, (row >> 3) & 7)],
                                    inv_pair[((row >> 6) & 7, (row >> 9) & 7)],
                                ])
                    packed_chunk = _encode_bc5_from_lanes(lanes)
                elif codec == 6:
                    # Use raw solid BC6H blocks with mode 0x0F? Simpler: encode mode-11 blocks and
                    # verify decoded DDS payload matches bcpack.decode(packed_chunk, 6, ...).
                    packed_chunk = _encode_bc6h_solid(nblocks)
                    decoded_bc6 = bcpack.decode(packed_chunk, 6, lay["size"])
                    linear = {m["index"]: ttpf.detile(lay, m, decoded_bc6[m["offset"] : m["offset"] + m["region"]]) for m in lay["mips"]}
                elif codec == 7:
                    # Ensure padding blocks are valid mode-6 for encoder
                    pad_blk = (1 << 6).to_bytes(16, "little")
                    for off in range(0, len(tiled), 16):
                        if tiled[off : off + 16] == b"\x00" * 16:
                            tiled[off : off + 16] = pad_blk
                    packed_chunk = tb.encode_bc7_mode6_blocks(bytes(tiled))

                pool, base = self._open_pool(
                    f"pool_{fmt}.ttpf",
                    [ttm.pack_etpf_record(w, h, mips, 0, fmt, 1, 1, mode, 4096)],
                    [(codec, packed_chunk, lay["size"])],
                )
                cal = {
                    "version": 1,
                    "bc3": _bc3_profile_from_tables(),
                    "bc4": {"custom": {str(k): list(v) for k, v in bcpack_tables.BC4_HISTORY_PROVISIONAL.items()}},
                    "bc6h": {},
                    "bc7": _empty_bc7_section(),
                    "entries": {Path(pool.path).name: {"0": "custom"}},
                }

                def rebuild_from_tiled(s_ptr, count, lab_ptr, seed_ptr, anc_ptr, out_ptr, ok_ptr, _t=bytes(tiled)):
                    _write_u8_buf(out_ptr, _t[: count * 16])
                    _write_u8_buf(ok_ptr, b"\x01" * count)

                dec = ct.LocalDecoder(pool, cal)
                with _fake_native_context(rebuild_fn=rebuild_from_tiled):
                    dds = dec.dds(0)
                    # Partial mips when mips > 1
                    if mips > 1:
                        dds_part = dec.dds(0, mips=2)
                        hdr_p = tex_headers.parse_dds_header(dds_part)
                        self.assertEqual(hdr_p["mips"], 2)
                        self.assertEqual(dds_part[148:], linear[0] + linear[1])

                hdr = tex_headers.parse_dds_header(dds)
                self.assertEqual(hdr["width"], w)
                self.assertEqual(hdr["height"], h)
                self.assertEqual(hdr["mips"], mips)
                self.assertEqual(hdr["dxgi_format"], fmt)
                self.assertEqual(hdr["array_size"], 1)
                self.assertEqual(dds[148:], b"".join(linear[i] for i in range(mips)))

    def test_dds_multichunk_and_2d_array_and_validation_errors(self):
        # 2D texture spanning 2 chunks (raw + zlib) plus 2D array and invalid descriptors
        lay = ttpf.surface_layout(64, 64, 71, 5, 4096)  # size = 8192 = 2 * 4096 pages
        lin = ttm.solid_chain(lay, 71)
        tiled = bytearray(lay["size"])
        for m in lay["mips"]:
            ttm.retile_to(lay, m, lin[m["index"]], tiled)

        # 2-slice 2D array using slice-major layout expected by LocalDecoder.dds
        arr_lay = ttpf.surface_layout(32, 32, 71, 4, 4096)  # 4096 bytes per slice
        s0 = bytearray(4096)
        s1 = bytearray(4096)
        lin0 = ttm.solid_chain(arr_lay, 71, color_fn=lambda x, y, bw, bh: (255, 0, 0))
        lin1 = ttm.solid_chain(arr_lay, 71, color_fn=lambda x, y, bw, bh: (0, 255, 0))
        for m in arr_lay["mips"]:
            ttm.retile_to(arr_lay, m, lin0[m["index"]], s0)
            ttm.retile_to(arr_lay, m, lin1[m["index"]], s1)

        p_dummy = self.root / "multi.ttpf"
        chunk0 = bytes(tiled[:4096])
        chunk1 = zlib.compress(bytes(tiled[4096:]))
        chunk2 = bytes(s0 + s1)
        base0 = ttpf.align(16 + 7 * 24 + 7 * 32, 4096)
        # Keep records in non-decreasing base order so Pool.span() maps rec0 and rec1 accurately.
        rec_unmapped = ttm.pack_etpf_record(32, 32, 1, 0, 71, 1, 1, 4, 0)
        rec0 = ttm.pack_etpf_record(64, 64, 5, 0, 71, 1, 1, 4, base0)
        rec_bad_fmt = ttm.pack_etpf_record(32, 32, 1, 0, 28, 1, 1, 4, base0)
        rec_bad_dim = ttm.pack_etpf_record(32, 32, 1, 0, 71, 6, 2, 18, base0)
        rec_huge = ttm.pack_etpf_record(16384, 16384, 1, 0, 98, 2, 1, 4, base0)
        rec1 = ttm.pack_etpf_record(32, 32, 4, 0, 71, 2, 1, 4, base0 + 8192)
        rec_cross = ttm.pack_etpf_record(256, 256, 7, 1, 71, 1, 1, 4, base0 + 8192)

        write_custom_pool(
            p_dummy,
            [rec_unmapped, rec0, rec_bad_fmt, rec_bad_dim, rec_huge, rec1, rec_cross],
            [(0, chunk0, 4096), (8, chunk1, 4096), (0, chunk2, 8192)],
        )
        pool = ttpf.Pool(str(p_dummy))
        self.addCleanup(pool.close)
        dec = ct.LocalDecoder(pool, {"version": 1})

        # Multi-chunk 2D texture (index 1)
        dds0 = dec.dds(1)
        self.assertEqual(dds0[148:], b"".join(lin[i] for i in range(5)))

        # 2D array (depth=2, index 5)
        dds1 = dec.dds(5)
        hdr1 = tex_headers.parse_dds_header(dds1)
        self.assertEqual((hdr1["array_size"], hdr1["mips"]), (2, 4))
        expected_arr = b"".join(lin0[i] for i in range(4)) + b"".join(lin1[i] for i in range(4))
        self.assertEqual(dds1[148:], expected_arr)

        # Error paths
        with self.assertRaisesRegex(ValueError, "supported BCn 2D texture or array"):
            dec.dds(2)
        with self.assertRaisesRegex(ValueError, "supported BCn 2D texture or array"):
            dec.dds(3)
        for bad_mips in (0, -1, 6):
            with self.assertRaisesRegex(ValueError, "Mip count outside descriptor range"):
                dec.dds(1, mips=bad_mips)
        with self.assertRaisesRegex(ValueError, "Texture has no mapped allocation"):
            dec.dds(0)
        with self.assertRaisesRegex(ValueError, "Texture crosses its allocation boundary"):
            dec.dds(6)
        with self.assertRaisesRegex(ValueError, "DDS exceeds output limit"):
            dec.dds(4)

    def test_dds_no_tail_64k_tile_layout(self):
        # 512x256 BC7 (fmt 98, bs=16, 64K tile -> bw=64, bh=64) with 2 mips:
        # mip 0 is 128x64 blocks (2 tiles), mip 1 is 64x32 blocks (> bw//2=32 -> 1 tile), so tail == 2 == levels.
        lay = ttpf.surface_layout(512, 256, 98, 2, 65536)
        self.assertEqual(lay["first_tail"], 2)
        lin0 = bytes([0xAA] * (128 * 64 * 16))
        lin1 = bytes([0xBB] * (64 * 32 * 16))
        tiled = bytearray(lay["size"])
        ttm.retile_to(lay, lay["mips"][0], lin0, tiled)
        ttm.retile_to(lay, lay["mips"][1], lin1, tiled)
        pool, _ = self._open_pool(
            "notail.ttpf",
            [ttm.pack_etpf_record(512, 256, 2, 0, 98, 1, 1, 0, 4096)],
            [(8, zlib.compress(bytes(tiled)), lay["size"])],
        )
        dec = ct.LocalDecoder(pool, {"version": 1})
        dds = dec.dds(0)
        self.assertEqual(dds[148:], lin0 + lin1)

    def test_compact_dds_layout_uses_tex_headers_tail_and_mips(self):
        # ASSET-568: NPOT chains and a single-mip sub-tile must use the measured compact
        # layout's first_tail/mips, not locally re-derived grids (the latter packed a
        # lone mip into the tail and misplaced the NPOT standalone subresources).
        for width, height, fmt, levels, tile in ((65, 65, 98, 7, 4096), (32, 32, 71, 1, 4096)):
            with self.subTest(size=(width, height, fmt, levels, tile)):
                lay = ttpf.surface_layout(width, height, fmt, levels, tile)
                bw, bh, grids, offsets = ct.compact_dds_layout(width, height, fmt, levels, tile)
                self.assertEqual((bw, bh), (lay["block_w"], lay["block_h"]))
                self.assertEqual(grids, [(m["blocks_w"], m["blocks_h"]) for m in lay["mips"]])
                stride = tile if lay["first_tail"] < levels else 0
                for m in lay["mips"]:
                    i = m["index"]
                    if i < lay["first_tail"]:
                        want = (m["offset"], 0, 0, stride)
                    else:
                        want = (0, m["tile_x"], m["tile_y"], stride)
                    self.assertEqual(offsets[i], want)
                if levels == 1:
                    self.assertEqual(lay["first_tail"], 1)
                    self.assertEqual(offsets[0], (0, 0, 0, 0))

    def test_all_supported_formats_and_entries_fallback(self):
        self.assertEqual(ct.LocalDecoder.formats, frozenset((71, 72, 77, 78, 80, 83, 95, 96, 98, 99)))
        base0 = ttpf.align(16 + 10 * 24 + 10 * 32, 4096)
        records = [
            ttm.pack_etpf_record(16, 16, 1, 0, fmt, 1, 1, 4, base0)
            for fmt in sorted(ct.LocalDecoder.formats)
        ]
        pool, _ = self._open_pool("all_fmts.ttpf", records, [(0, b"\x5A" * 4096, 4096)])
        dec = ct.LocalDecoder(
            pool,
            {"version": 1, "entries": {"other.ttpf": {"0": "early"}, "all_fmts.ttpf": {"9": "custom"}}},
        )
        seen_profiles = []
        orig_surface = dec.surface

        def spy_surface(at, bs, profile):
            seen_profiles.append(profile)
            return orig_surface(at, bs, profile)

        with mock.patch.object(dec, "surface", side_effect=spy_surface):
            for idx, fmt in enumerate(sorted(ct.LocalDecoder.formats)):
                dds = dec.dds(idx)
                hdr = tex_headers.parse_dds_header(dds)
                bs = 8 if fmt in (71, 72, 80) else 16
                self.assertEqual(hdr["dxgi_format"], fmt)
                self.assertEqual(hdr["pitch_linear"], 4 * 4 * bs)
                self.assertEqual(dds[148:], b"\x5A" * (16 * bs))
        self.assertEqual(seen_profiles, ["base"] * 9 + ["custom"])

    def test_invalid_bc1_and_bc7_config_shapes(self):
        bc1_packed = tb.encode_bc1_blocks(b"\x00" * 4096)
        bc7_packed = _encode_bc7_mode6_solid(256)
        pool, _ = self._open_pool(
            "bad_cfg.ttpf",
            [ttm.pack_etpf_record(64, 64, 1, 0, 71, 1, 1, 4, 4096)],
            [(1, bc1_packed, 4096), (7, bc7_packed, 4096)],
        )
        bad_bc1 = ct.LocalDecoder(
            pool, {"version": 1, "bc1": {"endpoints": [[0] * 6], "selectors": [[0] * 256] * 4}}
        )
        with _fake_native_context(), self.assertRaisesRegex(ValueError, "Invalid calibration field"):
            bad_bc1.read_chunk(0, 8, "base")

        bad_bc7 = ct.LocalDecoder(
            pool, {"version": 1, "bc7": {"labels": [[0, 0, 0, 0]], "anchors": [0] * 128}}
        )
        with _fake_native_context(), self.assertRaisesRegex(ValueError, "Invalid calibration field"):
            bad_bc7.read_chunk(1, 16, "base")


class TestExportFileAndCli(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.pool_path = self.root / "pool.ttpf"
        self.lay, self.linear, _ = ttm.build_bc1_bcpack_pool(self.pool_path)
        self.profile = self.root / "profile.json"
        self.profile.write_text(json.dumps(synthetic_profile()), encoding="utf-8")

    def test_export_file_atomic_link_replace_and_cleanup(self):
        pool = ttpf.Pool(str(self.pool_path))
        self.addCleanup(pool.close)
        out = self.root / "out.dds"

        # 1. Create new file (force=False)
        with _fake_native_context():
            rc = ct.export_file(pool, 0, out, None, False, self.profile)
        self.assertEqual(rc, 0)
        data = out.read_bytes()
        self.assertEqual(data[:4], b"DDS ")
        self.assertEqual(data[148:], b"".join(self.linear[i] for i in sorted(self.linear)))
        # No leftover temporary files
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["out.dds", "pool.ttpf", "profile.json"])

        # 2. Existing target with force=False raises FileExistsError, keeps target intact, cleans tmp
        out.write_bytes(b"SENTINEL")
        with _fake_native_context():
            with self.assertRaises(FileExistsError):
                ct.export_file(pool, 0, out, None, False, self.profile)
        self.assertEqual(out.read_bytes(), b"SENTINEL")
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["out.dds", "pool.ttpf", "profile.json"])

        # 3. Existing target with force=True replaces target atomically and cleans tmp
        with _fake_native_context():
            rc_force = ct.export_file(pool, 0, out, 2, True, self.profile)
        self.assertEqual(rc_force, 0)
        replaced = out.read_bytes()
        self.assertEqual(tex_headers.parse_dds_header(replaced)["mips"], 2)
        self.assertEqual(replaced[148:], self.linear[0] + self.linear[1])
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["out.dds", "pool.ttpf", "profile.json"])

        # 4. Cache is cleared even if dds() fails
        clear_calls = []

        with _fake_native_context(), \
                mock.patch.object(ct.LocalDecoder, "dds", side_effect=ValueError("boom")):
            orig_init = ct.LocalDecoder.__init__

            def tracked_init(self_dec, p, c):
                orig_init(self_dec, p, c)
                orig_cc = self_dec.read_chunk.cache_clear
                self_dec.read_chunk.cache_clear = lambda: (clear_calls.append(1), orig_cc())[1]

            with mock.patch.object(ct.LocalDecoder, "__init__", tracked_init):
                with self.assertRaisesRegex(ValueError, "boom"):
                    ct.export_file(pool, 0, self.root / "unused.dds", None, False, self.profile)
        self.assertEqual(len(clear_calls), 1)

        # 5. Missing or invalid profile JSON
        with self.assertRaises(FileNotFoundError):
            ct.export_file(pool, 0, self.root / "u.dds", None, False, self.root / "missing.json")
        bad_json = self.root / "bad.json"
        bad_json.write_text("{not json", encoding="utf-8")
        with self.assertRaises(json.JSONDecodeError):
            ct.export_file(pool, 0, self.root / "u.dds", None, False, bad_json)

    def test_ttpf_mips_cli_calibration_paths(self):
        out = self.root / "cli.dds"

        def run_cli(*argv):
            out_buf, err_buf = io.StringIO(), io.StringIO()
            with mock.patch.dict(sys.modules, {"calibrated_texture": ct}), \
                    _fake_native_context(), \
                    redirect_stdout(out_buf), redirect_stderr(err_buf):
                rc = ttpf.main(list(argv))
            return rc, out_buf.getvalue(), err_buf.getvalue()

        # Success
        rc, _, err = run_cli("export", str(self.pool_path), "0", str(out), "--calibration", str(self.profile))
        self.assertEqual(rc, 0, err)
        dds = out.read_bytes()
        self.assertEqual(dds[:4], b"DDS ")
        self.assertEqual(dds[148:], b"".join(self.linear[i] for i in sorted(self.linear)))

        # Refuses overwrite without --force
        out.write_bytes(b"KEEP_ME")
        rc2, _, err2 = run_cli("export", str(self.pool_path), "0", str(out), "--calibration", str(self.profile))
        self.assertEqual(rc2, 2)
        self.assertIn("calibrated export failed", err2)
        self.assertNotIn("Traceback", err2)
        self.assertEqual(out.read_bytes(), b"KEEP_ME")

        # Overwrites with --force and --mips
        rc3, _, err3 = run_cli(
            "export", str(self.pool_path), "0", str(out),
            "--calibration", str(self.profile), "--mips", "1", "--force",
        )
        self.assertEqual(rc3, 0, err3)
        self.assertEqual(tex_headers.parse_dds_header(out.read_bytes())["mips"], 1)

        # Rejects --calibration + --history together
        rc4, _, err4 = run_cli(
            "export", str(self.pool_path), "0", str(out),
            "--calibration", str(self.profile), "--history", str(self.profile), "--force",
        )
        self.assertEqual(rc4, 2)
        self.assertIn("choose either --calibration or --history", err4)

        # Bad profile version returns exit code 2 cleanly
        self.profile.write_text('{"version": 99}', encoding="utf-8")
        rc5, _, err5 = run_cli(
            "export", str(self.pool_path), "0", str(out),
            "--calibration", str(self.profile), "--force",
        )
        self.assertEqual(rc5, 2)
        self.assertIn("Unsupported calibration version", err5)
        self.assertNotIn("Traceback", err5)


if __name__ == "__main__":
    unittest.main()
