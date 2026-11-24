"""Narrow integer typed-buffer formats (8/16-bit UINT and SINT) in tools/rdna2spirv.py, against words measured natively.

Every expected word below was read back from an AMD Radeon RDNA2 iGPU (gfx1036, Ryzen 9 9950X3D, Windows AMD
driver, through tools/hw_oracle with AMD clang 23 from rocm-sdk-core 10.0.0), in wave32 and wave64 alike, with the
same V# and instruction the translated shader gets here:

- loads: `buffer_load_format_xyzw` and `_d16_xyzw` from four source dwords (SOURCE) with DST_SEL XYZW and X011. UINT
  zero-extends and SINT sign-extends each channel; missing channels repeat the format's own (c mod n); DST_SEL 1 is
  integer 1; d16 packs each result as a 16-bit integer.
- stores: `buffer_store_format_x/xy/xyzw` of four value sets into a buffer filled with 0xDEAD0000, with the V#
  selectors the runtime gives each format (gfx10_descriptors.h: X001, XY01, XYZW). UINT saturates to the channel
  (unsigned), SINT clamps the signed 32-bit value to [-2^(w-1), 2^(w-1) - 1]; a 1-2 byte element keeps the rest
  of its dword. `buffer_store_format_d16_xyzw` (D16_STORES) reads each 16-bit half as a signed integer for SINT and
  clamps it the same way, so 0xff80 stores -128 into 8_SINT.

Formats 27, 60 and 69 (16_16_UINT, 8_8_8_8_UINT, 16_16_16_16_UINT) were decoded before; they are the control.
"""
from __future__ import annotations

import os
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.dont_write_bytecode = True
sys.path.insert(0, os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools"))
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
BASE = 0x1000
SOURCE = (0x7F01FF80, 0x8000FFFF, 0x12345678, 0x9ABCDEF0)
SEL_XYZW, SEL_X011 = 4 | 5 << 3 | 6 << 6 | 7 << 9, 4 | 0 << 3 | 1 << 6 | 1 << 9
RUNTIME_SEL = {1: 4 | 0 << 3 | 0 << 6 | 1 << 9, 2: 4 | 5 << 3 | 0 << 6 | 1 << 9, 4: SEL_XYZW}
CHANNELS = {5: 1, 6: 1, 11: 1, 12: 1, 18: 2, 19: 2, 27: 2, 28: 2, 60: 4, 61: 4, 69: 4, 70: 4}
VALUES = {"big": (300, 0xFFFFFF38, 0x7FFFFFFF, 0x80000000), "edge": (0x7F, 0x80, 0xFFFFFF80, 0xFFFFFF7F),
          "edge16": (0x7FFF, 0x8000, 0xFFFF8000, 0xFFFF7FFF), "small": (5, 0xFFFFFFFF, 0, 0x101)}

LOADS = {  # format: (xyzw with DST_SEL XYZW, xyzw with DST_SEL X011, d16_xyzw's two registers)
    5: ((0x00000080, 0x00000080, 0x00000080, 0x00000080), (0x00000080, 0x00000000, 0x00000001, 0x00000001), (0x00800080, 0x00800080)),  # 8_UINT
    6: ((0xffffff80, 0xffffff80, 0xffffff80, 0xffffff80), (0xffffff80, 0x00000000, 0x00000001, 0x00000001), (0xff80ff80, 0xff80ff80)),  # 8_SINT
    11: ((0x0000ff80, 0x0000ff80, 0x0000ff80, 0x0000ff80), (0x0000ff80, 0x00000000, 0x00000001, 0x00000001), (0xff80ff80, 0xff80ff80)),  # 16_UINT
    12: ((0xffffff80, 0xffffff80, 0xffffff80, 0xffffff80), (0xffffff80, 0x00000000, 0x00000001, 0x00000001), (0xff80ff80, 0xff80ff80)),  # 16_SINT
    18: ((0x00000080, 0x000000ff, 0x00000080, 0x000000ff), (0x00000080, 0x00000000, 0x00000001, 0x00000001), (0x00ff0080, 0x00ff0080)),  # 8_8_UINT
    19: ((0xffffff80, 0xffffffff, 0xffffff80, 0xffffffff), (0xffffff80, 0x00000000, 0x00000001, 0x00000001), (0xffffff80, 0xffffff80)),  # 8_8_SINT
    27: ((0x0000ff80, 0x00007f01, 0x0000ff80, 0x00007f01), (0x0000ff80, 0x00000000, 0x00000001, 0x00000001), (0x7f01ff80, 0x7f01ff80)),  # 16_16_UINT
    28: ((0xffffff80, 0x00007f01, 0xffffff80, 0x00007f01), (0xffffff80, 0x00000000, 0x00000001, 0x00000001), (0x7f01ff80, 0x7f01ff80)),  # 16_16_SINT
    60: ((0x00000080, 0x000000ff, 0x00000001, 0x0000007f), (0x00000080, 0x00000000, 0x00000001, 0x00000001), (0x00ff0080, 0x007f0001)),  # 8_8_8_8_UINT
    61: ((0xffffff80, 0xffffffff, 0x00000001, 0x0000007f), (0xffffff80, 0x00000000, 0x00000001, 0x00000001), (0xffffff80, 0x007f0001)),  # 8_8_8_8_SINT
    69: ((0x0000ff80, 0x00007f01, 0x0000ffff, 0x00008000), (0x0000ff80, 0x00000000, 0x00000001, 0x00000001), (0x7f01ff80, 0x8000ffff)),  # 16_16_16_16_UINT
    70: ((0xffffff80, 0x00007f01, 0xffffffff, 0xffff8000), (0xffffff80, 0x00000000, 0x00000001, 0x00000001), (0x7f01ff80, 0x8000ffff)),  # 16_16_16_16_SINT
}
STORES = {  # (format, value set, components): the two dwords at the element, 0xDEAD0000 before the store
    (5, 'big', 1): (0xdead00ff, 0xdead0000),  # 8_UINT
    (5, 'big', 2): (0xdead00ff, 0xdead0000),  # 8_UINT
    (5, 'big', 4): (0xdead00ff, 0xdead0000),  # 8_UINT
    (5, 'edge', 1): (0xdead007f, 0xdead0000),  # 8_UINT
    (5, 'edge', 2): (0xdead007f, 0xdead0000),  # 8_UINT
    (5, 'edge', 4): (0xdead007f, 0xdead0000),  # 8_UINT
    (5, 'edge16', 1): (0xdead00ff, 0xdead0000),  # 8_UINT
    (5, 'edge16', 2): (0xdead00ff, 0xdead0000),  # 8_UINT
    (5, 'edge16', 4): (0xdead00ff, 0xdead0000),  # 8_UINT
    (5, 'small', 1): (0xdead0005, 0xdead0000),  # 8_UINT
    (5, 'small', 2): (0xdead0005, 0xdead0000),  # 8_UINT
    (5, 'small', 4): (0xdead0005, 0xdead0000),  # 8_UINT
    (6, 'big', 1): (0xdead007f, 0xdead0000),  # 8_SINT
    (6, 'big', 2): (0xdead007f, 0xdead0000),  # 8_SINT
    (6, 'big', 4): (0xdead007f, 0xdead0000),  # 8_SINT
    (6, 'edge', 1): (0xdead007f, 0xdead0000),  # 8_SINT
    (6, 'edge', 2): (0xdead007f, 0xdead0000),  # 8_SINT
    (6, 'edge', 4): (0xdead007f, 0xdead0000),  # 8_SINT
    (6, 'edge16', 1): (0xdead007f, 0xdead0000),  # 8_SINT
    (6, 'edge16', 2): (0xdead007f, 0xdead0000),  # 8_SINT
    (6, 'edge16', 4): (0xdead007f, 0xdead0000),  # 8_SINT
    (6, 'small', 1): (0xdead0005, 0xdead0000),  # 8_SINT
    (6, 'small', 2): (0xdead0005, 0xdead0000),  # 8_SINT
    (6, 'small', 4): (0xdead0005, 0xdead0000),  # 8_SINT
    (11, 'big', 1): (0xdead012c, 0xdead0000),  # 16_UINT
    (11, 'big', 2): (0xdead012c, 0xdead0000),  # 16_UINT
    (11, 'big', 4): (0xdead012c, 0xdead0000),  # 16_UINT
    (11, 'edge', 1): (0xdead007f, 0xdead0000),  # 16_UINT
    (11, 'edge', 2): (0xdead007f, 0xdead0000),  # 16_UINT
    (11, 'edge', 4): (0xdead007f, 0xdead0000),  # 16_UINT
    (11, 'edge16', 1): (0xdead7fff, 0xdead0000),  # 16_UINT
    (11, 'edge16', 2): (0xdead7fff, 0xdead0000),  # 16_UINT
    (11, 'edge16', 4): (0xdead7fff, 0xdead0000),  # 16_UINT
    (11, 'small', 1): (0xdead0005, 0xdead0000),  # 16_UINT
    (11, 'small', 2): (0xdead0005, 0xdead0000),  # 16_UINT
    (11, 'small', 4): (0xdead0005, 0xdead0000),  # 16_UINT
    (12, 'big', 1): (0xdead012c, 0xdead0000),  # 16_SINT
    (12, 'big', 2): (0xdead012c, 0xdead0000),  # 16_SINT
    (12, 'big', 4): (0xdead012c, 0xdead0000),  # 16_SINT
    (12, 'edge', 1): (0xdead007f, 0xdead0000),  # 16_SINT
    (12, 'edge', 2): (0xdead007f, 0xdead0000),  # 16_SINT
    (12, 'edge', 4): (0xdead007f, 0xdead0000),  # 16_SINT
    (12, 'edge16', 1): (0xdead7fff, 0xdead0000),  # 16_SINT
    (12, 'edge16', 2): (0xdead7fff, 0xdead0000),  # 16_SINT
    (12, 'edge16', 4): (0xdead7fff, 0xdead0000),  # 16_SINT
    (12, 'small', 1): (0xdead0005, 0xdead0000),  # 16_SINT
    (12, 'small', 2): (0xdead0005, 0xdead0000),  # 16_SINT
    (12, 'small', 4): (0xdead0005, 0xdead0000),  # 16_SINT
    (18, 'big', 1): (0xdead00ff, 0xdead0000),  # 8_8_UINT
    (18, 'big', 2): (0xdeadffff, 0xdead0000),  # 8_8_UINT
    (18, 'big', 4): (0xdeadffff, 0xdead0000),  # 8_8_UINT
    (18, 'edge', 1): (0xdead007f, 0xdead0000),  # 8_8_UINT
    (18, 'edge', 2): (0xdead807f, 0xdead0000),  # 8_8_UINT
    (18, 'edge', 4): (0xdead807f, 0xdead0000),  # 8_8_UINT
    (18, 'edge16', 1): (0xdead00ff, 0xdead0000),  # 8_8_UINT
    (18, 'edge16', 2): (0xdeadffff, 0xdead0000),  # 8_8_UINT
    (18, 'edge16', 4): (0xdeadffff, 0xdead0000),  # 8_8_UINT
    (18, 'small', 1): (0xdead0005, 0xdead0000),  # 8_8_UINT
    (18, 'small', 2): (0xdeadff05, 0xdead0000),  # 8_8_UINT
    (18, 'small', 4): (0xdeadff05, 0xdead0000),  # 8_8_UINT
    (19, 'big', 1): (0xdead007f, 0xdead0000),  # 8_8_SINT
    (19, 'big', 2): (0xdead807f, 0xdead0000),  # 8_8_SINT
    (19, 'big', 4): (0xdead807f, 0xdead0000),  # 8_8_SINT
    (19, 'edge', 1): (0xdead007f, 0xdead0000),  # 8_8_SINT
    (19, 'edge', 2): (0xdead7f7f, 0xdead0000),  # 8_8_SINT
    (19, 'edge', 4): (0xdead7f7f, 0xdead0000),  # 8_8_SINT
    (19, 'edge16', 1): (0xdead007f, 0xdead0000),  # 8_8_SINT
    (19, 'edge16', 2): (0xdead7f7f, 0xdead0000),  # 8_8_SINT
    (19, 'edge16', 4): (0xdead7f7f, 0xdead0000),  # 8_8_SINT
    (19, 'small', 1): (0xdead0005, 0xdead0000),  # 8_8_SINT
    (19, 'small', 2): (0xdeadff05, 0xdead0000),  # 8_8_SINT
    (19, 'small', 4): (0xdeadff05, 0xdead0000),  # 8_8_SINT
    (27, 'big', 1): (0x0000012c, 0xdead0000),  # 16_16_UINT
    (27, 'big', 2): (0xffff012c, 0xdead0000),  # 16_16_UINT
    (27, 'big', 4): (0xffff012c, 0xdead0000),  # 16_16_UINT
    (27, 'edge', 1): (0x0000007f, 0xdead0000),  # 16_16_UINT
    (27, 'edge', 2): (0x0080007f, 0xdead0000),  # 16_16_UINT
    (27, 'edge', 4): (0x0080007f, 0xdead0000),  # 16_16_UINT
    (27, 'edge16', 1): (0x00007fff, 0xdead0000),  # 16_16_UINT
    (27, 'edge16', 2): (0x80007fff, 0xdead0000),  # 16_16_UINT
    (27, 'edge16', 4): (0x80007fff, 0xdead0000),  # 16_16_UINT
    (27, 'small', 1): (0x00000005, 0xdead0000),  # 16_16_UINT
    (27, 'small', 2): (0xffff0005, 0xdead0000),  # 16_16_UINT
    (27, 'small', 4): (0xffff0005, 0xdead0000),  # 16_16_UINT
    (28, 'big', 1): (0x0000012c, 0xdead0000),  # 16_16_SINT
    (28, 'big', 2): (0xff38012c, 0xdead0000),  # 16_16_SINT
    (28, 'big', 4): (0xff38012c, 0xdead0000),  # 16_16_SINT
    (28, 'edge', 1): (0x0000007f, 0xdead0000),  # 16_16_SINT
    (28, 'edge', 2): (0x0080007f, 0xdead0000),  # 16_16_SINT
    (28, 'edge', 4): (0x0080007f, 0xdead0000),  # 16_16_SINT
    (28, 'edge16', 1): (0x00007fff, 0xdead0000),  # 16_16_SINT
    (28, 'edge16', 2): (0x7fff7fff, 0xdead0000),  # 16_16_SINT
    (28, 'edge16', 4): (0x7fff7fff, 0xdead0000),  # 16_16_SINT
    (28, 'small', 1): (0x00000005, 0xdead0000),  # 16_16_SINT
    (28, 'small', 2): (0xffff0005, 0xdead0000),  # 16_16_SINT
    (28, 'small', 4): (0xffff0005, 0xdead0000),  # 16_16_SINT
    (60, 'big', 1): (0x000000ff, 0xdead0000),  # 8_8_8_8_UINT
    (60, 'big', 2): (0x0000ffff, 0xdead0000),  # 8_8_8_8_UINT
    (60, 'big', 4): (0xffffffff, 0xdead0000),  # 8_8_8_8_UINT
    (60, 'edge', 1): (0x0000007f, 0xdead0000),  # 8_8_8_8_UINT
    (60, 'edge', 2): (0x0000807f, 0xdead0000),  # 8_8_8_8_UINT
    (60, 'edge', 4): (0xffff807f, 0xdead0000),  # 8_8_8_8_UINT
    (60, 'edge16', 1): (0x000000ff, 0xdead0000),  # 8_8_8_8_UINT
    (60, 'edge16', 2): (0x0000ffff, 0xdead0000),  # 8_8_8_8_UINT
    (60, 'edge16', 4): (0xffffffff, 0xdead0000),  # 8_8_8_8_UINT
    (60, 'small', 1): (0x00000005, 0xdead0000),  # 8_8_8_8_UINT
    (60, 'small', 2): (0x0000ff05, 0xdead0000),  # 8_8_8_8_UINT
    (60, 'small', 4): (0xff00ff05, 0xdead0000),  # 8_8_8_8_UINT
    (61, 'big', 1): (0x0000007f, 0xdead0000),  # 8_8_8_8_SINT
    (61, 'big', 2): (0x0000807f, 0xdead0000),  # 8_8_8_8_SINT
    (61, 'big', 4): (0x807f807f, 0xdead0000),  # 8_8_8_8_SINT
    (61, 'edge', 1): (0x0000007f, 0xdead0000),  # 8_8_8_8_SINT
    (61, 'edge', 2): (0x00007f7f, 0xdead0000),  # 8_8_8_8_SINT
    (61, 'edge', 4): (0x80807f7f, 0xdead0000),  # 8_8_8_8_SINT
    (61, 'edge16', 1): (0x0000007f, 0xdead0000),  # 8_8_8_8_SINT
    (61, 'edge16', 2): (0x00007f7f, 0xdead0000),  # 8_8_8_8_SINT
    (61, 'edge16', 4): (0x80807f7f, 0xdead0000),  # 8_8_8_8_SINT
    (61, 'small', 1): (0x00000005, 0xdead0000),  # 8_8_8_8_SINT
    (61, 'small', 2): (0x0000ff05, 0xdead0000),  # 8_8_8_8_SINT
    (61, 'small', 4): (0x7f00ff05, 0xdead0000),  # 8_8_8_8_SINT
    (69, 'big', 1): (0x0000012c, 0x00000000),  # 16_16_16_16_UINT
    (69, 'big', 2): (0xffff012c, 0x00000000),  # 16_16_16_16_UINT
    (69, 'big', 4): (0xffff012c, 0xffffffff),  # 16_16_16_16_UINT
    (69, 'edge', 1): (0x0000007f, 0x00000000),  # 16_16_16_16_UINT
    (69, 'edge', 2): (0x0080007f, 0x00000000),  # 16_16_16_16_UINT
    (69, 'edge', 4): (0x0080007f, 0xffffffff),  # 16_16_16_16_UINT
    (69, 'edge16', 1): (0x00007fff, 0x00000000),  # 16_16_16_16_UINT
    (69, 'edge16', 2): (0x80007fff, 0x00000000),  # 16_16_16_16_UINT
    (69, 'edge16', 4): (0x80007fff, 0xffffffff),  # 16_16_16_16_UINT
    (69, 'small', 1): (0x00000005, 0x00000000),  # 16_16_16_16_UINT
    (69, 'small', 2): (0xffff0005, 0x00000000),  # 16_16_16_16_UINT
    (69, 'small', 4): (0xffff0005, 0x01010000),  # 16_16_16_16_UINT
    (70, 'big', 1): (0x0000012c, 0x00000000),  # 16_16_16_16_SINT
    (70, 'big', 2): (0xff38012c, 0x00000000),  # 16_16_16_16_SINT
    (70, 'big', 4): (0xff38012c, 0x80007fff),  # 16_16_16_16_SINT
    (70, 'edge', 1): (0x0000007f, 0x00000000),  # 16_16_16_16_SINT
    (70, 'edge', 2): (0x0080007f, 0x00000000),  # 16_16_16_16_SINT
    (70, 'edge', 4): (0x0080007f, 0xff7fff80),  # 16_16_16_16_SINT
    (70, 'edge16', 1): (0x00007fff, 0x00000000),  # 16_16_16_16_SINT
    (70, 'edge16', 2): (0x7fff7fff, 0x00000000),  # 16_16_16_16_SINT
    (70, 'edge16', 4): (0x7fff7fff, 0x80008000),  # 16_16_16_16_SINT
    (70, 'small', 1): (0x00000005, 0x00000000),  # 16_16_16_16_SINT
    (70, 'small', 2): (0xffff0005, 0x00000000),  # 16_16_16_16_SINT
    (70, 'small', 4): (0xffff0005, 0x01010000),  # 16_16_16_16_SINT
}


D16_VALUES = {"ewann": (0xFF80FF80, 0x8000FF80), "edge": (0x7FFF0080, 0xFF7F0005), "small": (0x00010005, 0xFFFF0000)}
D16_STORES = {  # (format, register pair): buffer_store_format_d16_xyzw v[2:3], the two dwords at the element
    (5, 'ewann'): (0xDEAD00FF, 0xDEAD0000),
    (5, 'edge'): (0xDEAD0080, 0xDEAD0000),
    (5, 'small'): (0xDEAD0005, 0xDEAD0000),
    (6, 'ewann'): (0xDEAD0080, 0xDEAD0000),
    (6, 'edge'): (0xDEAD007F, 0xDEAD0000),
    (6, 'small'): (0xDEAD0005, 0xDEAD0000),
    (11, 'ewann'): (0xDEADFF80, 0xDEAD0000),
    (11, 'edge'): (0xDEAD0080, 0xDEAD0000),
    (11, 'small'): (0xDEAD0005, 0xDEAD0000),
    (12, 'ewann'): (0xDEADFF80, 0xDEAD0000),
    (12, 'edge'): (0xDEAD0080, 0xDEAD0000),
    (12, 'small'): (0xDEAD0005, 0xDEAD0000),
    (18, 'ewann'): (0xDEADFFFF, 0xDEAD0000),
    (18, 'edge'): (0xDEADFF80, 0xDEAD0000),
    (18, 'small'): (0xDEAD0105, 0xDEAD0000),
    (19, 'ewann'): (0xDEAD8080, 0xDEAD0000),
    (19, 'edge'): (0xDEAD7F7F, 0xDEAD0000),
    (19, 'small'): (0xDEAD0105, 0xDEAD0000),
    (27, 'ewann'): (0xFF80FF80, 0xDEAD0000),
    (27, 'edge'): (0x7FFF0080, 0xDEAD0000),
    (27, 'small'): (0x00010005, 0xDEAD0000),
    (28, 'ewann'): (0xFF80FF80, 0xDEAD0000),
    (28, 'edge'): (0x7FFF0080, 0xDEAD0000),
    (28, 'small'): (0x00010005, 0xDEAD0000),
    (60, 'ewann'): (0xFFFFFFFF, 0xDEAD0000),
    (60, 'edge'): (0xFF05FF80, 0xDEAD0000),
    (60, 'small'): (0xFF000105, 0xDEAD0000),
    (61, 'ewann'): (0x80808080, 0xDEAD0000),
    (61, 'edge'): (0x80057F7F, 0xDEAD0000),
    (61, 'small'): (0xFF000105, 0xDEAD0000),
    (69, 'ewann'): (0xFF80FF80, 0x8000FF80),
    (69, 'edge'): (0x7FFF0080, 0xFF7F0005),
    (69, 'small'): (0x00010005, 0xFFFF0000),
    (70, 'ewann'): (0xFF80FF80, 0x8000FF80),
    (70, 'edge'): (0x7FFF0080, 0xFF7F0005),
    (70, 'small'): (0x00010005, 0xFFFF0000),
}


def run(fmt: int, body: str, dst_sel: int, memory: dict) -> tuple:
    """Translate a pixel shader whose V# (s[0:3], raw, 256 bytes) points at BASE with this format and selectors."""
    with tempfile.TemporaryDirectory() as tmp:
        os.mkdir(os.path.join(tmp, "disasm"))
        src = os.path.join(tmp, "disasm", "ps_narrow_int.s")
        with open(os.path.join(tmp, "ps_narrow_int.dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        with open(src, "w", encoding="utf-8") as f:
            f.write("; test_buffer_format_narrow_int\n"
                    f"\ts_mov_b32 s0, {BASE:#x}\n\ts_mov_b32 s1, 0\n\ts_mov_b32 s2, 0x100\n"
                    f"\ts_mov_b32 s3, {fmt << 12 | dst_sel | 1 << 24 | 3 << 28:#x}\n\tv_mov_b32 v1, 0\n" + body)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            spv = rdna2spirv.translate(src)
    return si.run_fragment(spv, memory=memory)["%out_mrt0"]


def load(fmt: int, dst_sel: int, d16: bool = False) -> tuple:
    regs, dst = ("v[2:3]", "v2, v3, v2, v3") if d16 else ("v[2:5]", "v2, v3, v4, v5")
    body = (f"\tbuffer_load_format_{'d16_' if d16 else ''}xyzw {regs}, v1, s[0:3], 0 offen\n"
            f"\ts_waitcnt vmcnt(0)\n\texp mrt0 {dst} done vm\n\ts_endpgm\n")
    return run(fmt, body, dst_sel, {BASE + 4 * i: w for i, w in enumerate(SOURCE)})


def store(fmt: int, values: tuple, n: int, channels: int) -> tuple:
    memory = {BASE: 0xDEAD0000, BASE + 4: 0xDEAD0000}
    movs = "".join(f"\tv_mov_b32 v{2 + i}, {v:#x}\n" for i, v in enumerate(values[:n]))
    regs = "v2" if n == 1 else f"v[2:{1 + n}]"
    body = (movs + f"\tbuffer_store_format_{'xyzw'[:n]} {regs}, v1, s[0:3], 0 offen\n"
            "\texp mrt0 v2, v2, v2, v2 done vm\n\ts_endpgm\n")
    run(fmt, body, RUNTIME_SEL[channels], memory)
    return memory[BASE], memory[BASE + 4]


def store_d16(fmt: int, pair: tuple, channels: int) -> tuple:
    """buffer_store_format_d16_xyzw of two registers (x | y << 16, z | w << 16): each half is a 16-bit integer, signed
    for SINT, then saturated or clamped like a full-width store."""
    memory = {BASE: 0xDEAD0000, BASE + 4: 0xDEAD0000}
    body = (f"\tv_mov_b32 v2, {pair[0]:#x}\n\tv_mov_b32 v3, {pair[1]:#x}\n"
            "\tbuffer_store_format_d16_xyzw v[2:3], v1, s[0:3], 0 offen\n\texp mrt0 v2, v2, v2, v2 done vm\n\ts_endpgm\n")
    run(fmt, body, RUNTIME_SEL[channels], memory)
    return memory[BASE], memory[BASE + 4]


class NarrowIntegerLoads(unittest.TestCase):
    def test_measured_words(self):
        for fmt, (xyzw, x011, d16) in LOADS.items():
            with self.subTest(fmt=fmt):
                self.assertEqual(load(fmt, SEL_XYZW), xyzw, "DST_SEL XYZW")
                self.assertEqual(load(fmt, SEL_X011), x011, "DST_SEL X011: 0 and integer 1")
                self.assertEqual(load(fmt, SEL_XYZW, d16=True)[:2], d16, "d16")


class NarrowIntegerStores(unittest.TestCase):
    def test_measured_words(self):
        for (fmt, values, n), want in STORES.items():
            with self.subTest(fmt=fmt, values=values, n=n):
                self.assertEqual(store(fmt, VALUES[values], n, CHANNELS[fmt]), want)

    def test_measured_d16_words(self):
        # A d16 SINT half is sign-extended before the clamp: 0xff80 stores -128 (0x80) into 8_SINT, not 127.
        for (fmt, pair), want in D16_STORES.items():
            with self.subTest(fmt=fmt, pair=pair):
                self.assertEqual(store_d16(fmt, D16_VALUES[pair], CHANNELS[fmt]), want)


if __name__ == "__main__":
    unittest.main()
