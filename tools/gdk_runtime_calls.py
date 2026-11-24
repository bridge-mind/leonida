"""List the GDK runtime classes, interfaces and vtable slots a PE image calls through xgameruntime.lib's thunks.

Usage:
    python -I tools/gdk_runtime_calls.py <image> [<image> ...] [--json] [--out DIR]
                                       [--table FILE] [--runtime-source FILE]

A title linked with xgameruntime.lib imports no GDK function. Each function of the library is a thunk that calls a
helper with a class id and an interface id (the helper initializes the runtime once and calls QueryApiImpl), then
calls one slot of the returned object's vtable with the function's own arguments after `this`, then Release (slot 2)
(docs/runtime_imports.md, "Class layouts, measured"). This finds those thunks in the image's code without running
it, and names each slot from tools/gdk_runtime_classes.json (tools/gen_gdk_runtime_classes.py makes it from
xgameruntime-docs). Run on the game executable, the classes and slots it prints are the runtime surface to serve
before the first boot.

How it finds them (x86-64 images only):
1. Every 16-byte in-memory copy of a class or interface id the table knows, in the data sections (in the code
   sections when the data sections have none).
2. Every RIP-relative `lea reg,[id]` in executable sections; the first call after it is a helper candidate. A
   candidate is a helper when, after it returns, one or more vtable slots past IUnknown are called (`mov r,[obj]`,
   `mov r,[r+disp]` and a call through it, the CFG `call __guard_dispatch_icall` form or `call [r+disp]`, followed
   through register copies).
3. Every call to a helper is a thunk, whatever ids it passes: an id the table doesn't know is reported as unknown.
   The ids are the RIP-relative addresses loaded into the helper's first two argument registers (followed through
   register copies): rcx holds the class id, rdx the interface id (the class id when rdx holds none). A helper call
   whose rcx holds no address is listed as not followed, with the calls after which no slot call is found.
4. After the helper call, each slot call up to Release, the next helper call or the end of the function is
   recorded. Function bounds come from the exception directory (.pdata), or a byte window without one.
5. Whether anything in the image reaches each thunk function: a call, tail jump, RIP-relative address load or
   8-byte pointer in data, directly or through an incremental-link `jmp` stub, from code that is itself reached
   when it is another thunk. An incremental link doesn't remove unreferenced functions, so an image can carry
   thunks nothing calls (libHttpClient.GDK.dll calls 15 of its 404).

The interface's slot table is the methods of its inheritance chain, base first, from slot 3. An id that is a class
id but not an interface id is read as the class's implemented interface. The argument count is an estimate: the
highest of rdx/r8/r9 (xmm1-3) written, or [rsp+20h+8k] stored, between the previous call and the slot call.

Output: per image and class, the class and interface ids, the slots called with their method names and whether
their thunks are referenced, the thunk count, and whether the runtime serves the class (a `constexpr GUID
k...Class` with its id in one of runtime/shims/xgameruntime*.cpp: xgameruntime.cpp and the classes it hands on to
xgameruntime_user.cpp, _gamesave.cpp, _store.cpp, _offline.cpp; the class only, not each slot); then the known ids present in the image that
no thunk reaches. Text by default, --json for JSON. --out DIR writes both, as <image file name>.gdk_calls.txt and
.json (x.dll-2.gdk_calls.txt for a second input named x.dll), and prints one summary line per image instead.

Exit status 1 when an image (or the class table) can't be read, or an image is cut off before the end of its
sections (the report of what is there is still written); 0 otherwise. Reads the files only; never runs them.
Standard library only.
"""
from __future__ import annotations

import argparse
import bisect
import json
import os
import re
import struct
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe_inspect import DIR_EXCEPTION, MACHINES, PeFormatError, PeImage  # noqa: E402
from report_paths import same_report_path  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TABLE = os.path.join(HERE, "gdk_runtime_classes.json")
DEFAULT_RUNTIME_SOURCE = os.path.join(HERE, "..", "runtime", "shims")   # every xgameruntime*.cpp in it

IUNKNOWN_SLOTS = 3          # QueryInterface, AddRef, Release
RELEASE = 2
NO_PDATA_WINDOW = 256       # bytes scanned after a helper call when no .pdata entry covers it
BACK_WINDOW = 64            # bytes searched before a helper call for the instructions that load its arguments
LEA_TO_CALL = 8             # instructions from an id's lea to the helper call it feeds

# ---- A small x86-64 instruction decoder: lengths, ModRM operands and branch targets ----

RIP = 16                    # base register value of a RIP-relative operand
XMM = 32                    # register number of xmm0 in writes()
RAX, RCX, RDX, RSP, R8, R9 = 0, 1, 2, 4, 8, 9
VOLATILE = frozenset((0, 1, 2, 8, 9, 10, 11))
ARG_INDEX = {RDX: 1, R8: 2, R9: 3, XMM + 1: 1, XMM + 2: 2, XMM + 3: 3}

_PREFIXES = frozenset((0x26, 0x2E, 0x36, 0x3E, 0x64, 0x65, 0x66, 0x67, 0xF0, 0xF2, 0xF3))
_ALU = [b + i for b in range(0, 0x40, 8) for i in range(4)]
_M1_MODRM = frozenset(_ALU + [0x63, 0x69, 0x6B, *range(0x80, 0x90), 0xC0, 0xC1, 0xC6, 0xC7, *range(0xD0, 0xD4),
                              *range(0xD8, 0xE0), 0xF6, 0xF7, 0xFE, 0xFF])
_M1_INVALID = frozenset((0x06, 0x07, 0x0E, 0x16, 0x17, 0x1E, 0x1F, 0x27, 0x2F, 0x37, 0x3F, 0x60, 0x61, 0x82, 0x9A,
                         0xCE, 0xD4, 0xD5, 0xD6, 0xEA))
_M1_IMM8 = frozenset([b + 4 for b in range(0, 0x40, 8)] + [0x6A, 0x6B, 0x80, 0x83, 0xA8, *range(0xB0, 0xB8), 0xC0,
                                                           0xC1, 0xC6, 0xCD, 0xE4, 0xE5, 0xE6, 0xE7])
_M1_IMMZ = frozenset([b + 5 for b in range(0, 0x40, 8)] + [0x68, 0x69, 0x81, 0xA9, 0xC7])
_M1_REL8 = frozenset([*range(0x70, 0x80), 0xE0, 0xE1, 0xE2, 0xE3, 0xEB])
_M2_NO_MODRM = frozenset((0x05, 0x06, 0x07, 0x08, 0x09, 0x0B, 0x0E, *range(0x30, 0x38), 0x77, 0xA0, 0xA1, 0xA2,
                          0xA8, 0xA9, 0xAA, *range(0xC8, 0xD0)))
_M2_IMM8 = frozenset((0x0F, 0x70, 0x71, 0x72, 0x73, 0xA4, 0xAC, 0xBA, 0xC2, 0xC4, 0xC5, 0xC6))
_M2_INVALID = frozenset((0x04, 0x0A, 0x0C, 0x24, 0x25, 0x26, 0x27, 0x36, 0x39, 0x3B, 0x3C, 0x3D, 0x3E, 0x3F))
_VEX_IMM8 = frozenset((0x70, 0x71, 0x72, 0x73, 0xC2, 0xC4, 0xC5, 0xC6))

# Opcodes (0x1xx = 0F xx) that write their ModRM reg operand, or their r/m operand when it is a register.
_WRITE_REG = frozenset((0x02, 0x03, 0x0A, 0x0B, 0x12, 0x13, 0x1A, 0x1B, 0x22, 0x23, 0x2A, 0x2B, 0x32, 0x33, 0x63,
                        0x69, 0x6B, 0x86, 0x87, 0x8A, 0x8B, 0x8D, *range(0x140, 0x150), 0x1AF, 0x1B6, 0x1B7, 0x1B8,
                        0x1BC, 0x1BD, 0x1BE, 0x1BF, 0x1C0, 0x1C1))
_WRITE_RM = frozenset((0x00, 0x01, 0x08, 0x09, 0x10, 0x11, 0x18, 0x19, 0x20, 0x21, 0x28, 0x29, 0x30, 0x31, 0x86,
                       0x87, 0x88, 0x89, 0x8F, 0xC0, 0xC1, 0xD0, 0xD1, 0xD2, 0xD3, *range(0x190, 0x1A0), 0x1C0,
                       0x1C1))
_WRITE_XMM = frozenset((0x110, 0x112, 0x114, 0x115, 0x116, 0x128, 0x12A, *range(0x151, 0x160), 0x16E, 0x16F))
_STORE = frozenset((0x88, 0x89, 0xC6, 0xC7, 0x111, 0x129, 0x17F, 0x1D6))


class Insn:
    """One decoded instruction. `op` is the opcode byte, 0x100|b for 0F b, 0x200|b / 0x300|b for 0F 38 / 0F 3A
    and 0x1000|map<<8|b under VEX/EVEX. `reg`/`rm` carry their REX bits; `base` is RIP for a RIP-relative operand
    and None for a register operand; `target` is a branch target or a RIP-relative operand's RVA."""
    __slots__ = ("rva", "size", "op", "rex", "reg", "mod", "rm", "base", "index", "disp", "target")

    def __init__(self, rva, size, op, rex, reg, mod, rm, base, index, disp, target):
        self.rva, self.size, self.op, self.rex = rva, size, op, rex
        self.reg, self.mod, self.rm, self.base, self.index, self.disp, self.target = (
            reg, mod, rm, base, index, disp, target)

    @property
    def end(self) -> int:
        return self.rva + self.size

    @property
    def wide(self) -> bool:
        return bool(self.rex & 8)

    def is_call(self) -> bool:
        return self.op == 0xE8 or (self.op == 0xFF and self.mod is not None and self.reg & 7 == 2)

    def ends_flow(self) -> bool:
        """ret, or an unconditional jump."""
        return (self.op in (0xC3, 0xC2, 0xE9, 0xEB)
                or (self.op == 0xFF and self.mod is not None and self.reg & 7 in (4, 5)))

    def call_key(self):
        """What a direct call reaches: ("rel", rva) for call rel32, ("mem", rva) for call [rip+x]."""
        if self.op == 0xE8:
            return ("rel", self.target)
        if self.op == 0xFF and self.base == RIP and self.reg & 7 == 2:
            return ("mem", self.target)
        return None

    def memory_at(self, base: int) -> bool:
        """A memory operand [base+disp] with no index register."""
        return self.mod is not None and self.mod != 3 and self.base == base and self.index is None


def decode(buf, off: int, base_rva: int) -> Insn | None:
    """Decode the instruction at buf[off] (buf starts at base_rva). None for an invalid or cut-off instruction."""
    n = len(buf)
    p = off
    p66 = p67 = False
    while p < n and buf[p] in _PREFIXES:
        p66 |= buf[p] == 0x66
        p67 |= buf[p] == 0x67
        p += 1
    if p >= n or p - off > 14:
        return None
    rex = 0
    b = buf[p]
    if 0x40 <= b <= 0x4F:
        rex = b
        p += 1
        if p >= n:
            return None
        b = buf[p]
    p += 1
    modrm = False
    imm = rel = 0
    if b in (0xC4, 0xC5, 0x62):                       # VEX / EVEX: always so in 64-bit mode
        if b == 0xC5:
            if p >= n:
                return None
            rex, vmap = 0x40 | (0 if buf[p] & 0x80 else 4), 1
            p += 1
        else:
            if p + (2 if b == 0x62 else 1) >= n:
                return None
            b1, b2 = buf[p], buf[p + 1]
            vmap = b1 & (0x07 if b == 0x62 else 0x1F)
            rex = (0x40 | (0 if b1 & 0x80 else 4) | (0 if b1 & 0x40 else 2) | (0 if b1 & 0x20 else 1)
                   | (8 if b2 & 0x80 else 0))
            p += 3 if b == 0x62 else 2
        if p >= n:
            return None
        b = buf[p]
        p += 1
        op = 0x1000 | (vmap << 8) | b
        modrm = not (vmap == 1 and b == 0x77)
        imm = 1 if vmap == 3 or (vmap == 1 and b in _VEX_IMM8) else 0
    elif b == 0x0F:
        if p >= n:
            return None
        b = buf[p]
        p += 1
        if b in (0x38, 0x3A):
            if p >= n:
                return None
            op = (0x200 if b == 0x38 else 0x300) | buf[p]
            p += 1
            modrm = True
            imm = 1 if b == 0x3A else 0
        else:
            if b in _M2_INVALID:
                return None
            op = 0x100 | b
            if 0x80 <= b <= 0x8F:
                rel = 4
            elif b not in _M2_NO_MODRM:
                modrm = True
                imm = 1 if b in _M2_IMM8 else 0
    else:
        if b in _M1_INVALID:
            return None
        op = b
        modrm = b in _M1_MODRM
        if b in _M1_IMM8:
            imm = 1
        elif b in _M1_IMMZ:
            imm = 2 if (p66 and not (rex & 8)) else 4
        elif b in _M1_REL8:
            rel = 1
        elif b in (0xE8, 0xE9):
            rel = 4
        elif 0xB8 <= b <= 0xBF:
            imm = 8 if rex & 8 else (2 if p66 else 4)
        elif b in (0xC2, 0xCA):
            imm = 2
        elif b == 0xC8:
            imm = 3
        elif 0xA0 <= b <= 0xA3:
            imm = 4 if p67 else 8

    reg = mod = rm = mbase = index = None
    disp = 0
    rip = False
    if modrm:
        if p >= n:
            return None
        m = buf[p]
        p += 1
        mod, reg, low = m >> 6, ((m >> 3) & 7) | ((rex & 4) << 1), m & 7
        dsize = 1 if mod == 1 else 4 if mod == 2 else 0
        if mod == 3:
            rm = low | ((rex & 1) << 3)
        elif low == 4:
            if p >= n:
                return None
            s = buf[p]
            p += 1
            idx = ((s >> 3) & 7) | ((rex & 2) << 2)
            index = None if idx == 4 else idx
            if s & 7 == 5 and mod == 0:
                dsize = 4                               # [index*scale + disp32], no base
            else:
                mbase = (s & 7) | ((rex & 1) << 3)
        elif low == 5 and mod == 0:
            mbase, dsize, rip = RIP, 4, True
        else:
            mbase = low | ((rex & 1) << 3)
        if p + dsize > n:
            return None
        if dsize == 1:
            disp = struct.unpack_from("<b", buf, p)[0]
        elif dsize == 4:
            disp = struct.unpack_from("<i", buf, p)[0]
        p += dsize
        if op in (0xF6, 0xF7) and reg & 7 in (0, 1):    # test r/m, imm
            imm = 1 if op == 0xF6 else (2 if (p66 and not (rex & 8)) else 4)
    target = None
    if rel:
        if p + rel > n:
            return None
        delta = struct.unpack_from("<b" if rel == 1 else "<i", buf, p)[0]
        p += rel
        target = base_rva + p + delta
    p += imm
    if p > n or p - off > 15:
        return None
    if rip:
        target = base_rva + p + disp
    return Insn(base_rva + off, p - off, op, rex, reg, mod, rm, mbase, index, disp, target)


def writes(i: Insn) -> list[int]:
    """Registers the instruction writes (general registers 0-15, xmm n as XMM+n), for the forms that matter here."""
    op = i.op
    if i.mod is None:
        low = (op & 7) | ((i.rex & 1) << 3)
        if 0xB0 <= op <= 0xBF or 0x58 <= op <= 0x5F:
            return [low]
        if 0x91 <= op <= 0x97:
            return [RAX, low]
        if op == 0x99:
            return [RDX]
        if op == 0x98:
            return [RAX]
        return []
    out = []
    if op in _WRITE_REG:
        out.append(i.reg)
    elif op in _WRITE_XMM:
        out.append(XMM + i.reg)
    if i.mod == 3:
        sub = i.reg & 7
        if (op in _WRITE_RM or (op in (0x80, 0x81, 0x83) and sub != 7) or (op in (0xC6, 0xC7) and sub == 0)
                or (op in (0xF6, 0xF7) and sub in (2, 3)) or (op in (0xFE, 0xFF) and sub in (0, 1))):
            out.append(i.rm)
    if op in (0xF6, 0xF7) and i.reg & 7 >= 4:          # mul/imul/div/idiv
        out += [RAX, RDX]
    return out


def stack_argument(i: Insn) -> int | None:
    """The argument number (5th = 4 after `this`) of a store to the outgoing stack area [rsp+20h+8k], else None."""
    if (i.op in _STORE and i.memory_at(RSP) and i.disp >= 0x20 and i.disp % 8 == 0
            and not (i.op in (0xC6, 0xC7) and i.reg & 7)):
        return 4 + (i.disp - 0x20) // 8
    return None


# ---- The class table and the runtime shim ----

def guid_bytes(guid: str) -> bytes:
    return uuid.UUID(guid).bytes_le


class RuntimeTable:
    """tools/gdk_runtime_classes.json: class and interface ids, inheritance and method order."""

    def __init__(self, doc: dict):
        self.source = doc.get("source", "")
        self.classes: dict = doc["classes"]
        self.interfaces: dict = doc["interfaces"]
        self.class_by_id = {c["id"]: name for name, c in self.classes.items()}
        self.interface_by_id = {i["id"]: name for name, i in self.interfaces.items()}

    @classmethod
    def load(cls, path: str) -> "RuntimeTable":
        with open(path, encoding="utf-8-sig") as f:
            return cls(json.load(f))

    def ids(self) -> set[str]:
        return set(self.class_by_id) | set(self.interface_by_id)

    def label(self, guid: str) -> str:
        """What an id names: 'XUserImpl', 'IXUserImpl5', 'XThreadingImpl/IXThreadingImpl', or 'unknown'."""
        names = [n for n in (self.class_by_id.get(guid), self.interface_by_id.get(guid)) if n]
        return "/".join(names) or "unknown"

    def class_of(self, guid: str) -> str | None:
        if guid in self.class_by_id:
            return self.class_by_id[guid]
        iface = self.interface_by_id.get(guid)
        return self.interfaces[iface]["class"] if iface else None

    def interface_for(self, guid: str) -> str | None:
        """The interface whose slot table names the slots called through this id."""
        if guid in self.interface_by_id:
            return self.interface_by_id[guid]
        cls = self.class_by_id.get(guid)
        if cls and self.classes[cls]["implements"]:
            return self.classes[cls]["implements"][0]
        return None

    def chain(self, iface: str) -> list[str]:
        out, seen = [], set()
        while iface and iface in self.interfaces and iface not in seen:
            seen.add(iface)
            out.append(iface)
            iface = self.interfaces[iface]["inherits"]
        return out[::-1]

    def slot_table(self, iface: str) -> tuple[list, list[str]]:
        """(method names from slot 3 on, None where unknown), and the chain's interfaces without a layout."""
        names, no_layout = [], []
        for name in self.chain(iface):
            methods = self.interfaces[name]["methods"]
            if methods is None:
                no_layout.append(name)
            else:
                names.extend(methods)
        return names, no_layout

    def method(self, iface: str | None, slot: int) -> tuple[str | None, str | None]:
        """(method name or None, the interface it was read from when that is not `iface`)."""
        if not iface:
            return None, None
        names, _ = self.slot_table(iface)
        k = slot - IUNKNOWN_SLOTS
        if 0 <= k < len(names):
            return names[k], None
        # Past the requested interface: the object is the class's, so its implemented interface may name the slot
        # when the requested one is on that interface's chain (an id that is both the class's and its base's).
        cls = self.interfaces[iface]["class"]
        for impl in self.classes.get(cls, {}).get("implements", []):
            if impl != iface and iface in self.chain(impl):
                more, _ = self.slot_table(impl)
                if 0 <= k < len(more):
                    return more[k], impl
        return None, None


GUID_CONSTANT = re.compile(
    r"constexpr\s+GUID\s+k(\w*?)Class\s*=\s*\{\s*(0x[0-9a-fA-F]+)\s*,\s*(0x[0-9a-fA-F]+)\s*,\s*(0x[0-9a-fA-F]+)\s*,"
    r"\s*\{([^}]*)\}\s*\}")


def read_runtime_source(path: str) -> str:
    """The text of a shim source file, or of every xgameruntime*.cpp in a folder (OSError when there is none)."""
    if os.path.isdir(path):
        names = sorted(n for n in os.listdir(path) if n.startswith("xgameruntime") and n.endswith(".cpp"))
        if not names:
            raise FileNotFoundError(2, "no xgameruntime*.cpp in the folder", path)
        texts = []
        for n in names:
            with open(os.path.join(path, n), encoding="utf-8", errors="replace") as f:
                texts.append(f.read())
        return "\n".join(texts)
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def served_classes(source: str) -> dict[str, str]:
    """{class id: constant name} for each `constexpr GUID k<Name>Class = {...}` in the runtime shim's source."""
    out = {}
    for m in GUID_CONSTANT.finditer(source):
        try:
            tail = [int(x, 16) for x in m.group(5).split(",") if x.strip()]
            if len(tail) != 8:
                continue
            guid = uuid.UUID(fields=(int(m.group(2), 16), int(m.group(3), 16), int(m.group(4), 16), tail[0],
                                     tail[1], int.from_bytes(bytes(tail[2:]), "big")))
        except ValueError:          # not a GUID after all (a value out of range, a non-hex byte)
            continue
        out[str(guid)] = f"k{m.group(1)}Class"
    return out


# ---- Scanning an image ----

class Code:
    """The image's executable sections and .pdata function bounds."""

    def __init__(self, img: PeImage):
        self.img = img
        self.sections = []                  # (rva, bytes): file bytes of each executable section, mapped span only
        for s in img.sections:
            if s["exec"]:
                data = img.section_data(s)
                span = s["vsize"] or s["rawsize"]
                self.sections.append((s["rva"], data[:span]))
        self.begins, self.ends = [], []
        rva, size = img.directory(DIR_EXCEPTION)
        at = img.offset(rva) if rva and size >= 12 else None
        if at is not None:
            count = min(size, len(img.data) - at) // 12
            entries = sorted((b, e) for b, e, _ in struct.iter_unpack("<III", img.data[at:at + count * 12]) if e > b)
            self.begins = [b for b, _ in entries]
            self.ends = [e for _, e in entries]

    def view(self, rva: int):
        """(bytes, section rva) of the executable section holding rva, or (None, 0)."""
        for base, data in self.sections:
            if base <= rva < base + len(data):
                return data, base
        return None, 0

    def decode(self, rva: int) -> Insn | None:
        data, base = self.view(rva)
        return None if data is None else decode(data, rva - base, base)

    def function(self, rva: int) -> tuple[int, int] | None:
        k = bisect.bisect_right(self.begins, rva) - 1
        if k >= 0 and self.begins[k] <= rva < self.ends[k]:
            return self.begins[k], self.ends[k]
        return None

    def run(self, start: int, stop: int) -> list[Insn] | None:
        """The instructions from start up to stop, or None when decoding doesn't land on stop exactly."""
        out = []
        rva = start
        while rva < stop:
            i = self.decode(rva)
            if i is None:
                return None
            out.append(i)
            rva = i.end
        return out if rva == stop else None


def guid_at(img: PeImage, rva: int) -> str | None:
    at = img.offset(rva)
    if at is None or at + 16 > len(img.data):
        return None
    return str(uuid.UUID(bytes_le=bytes(img.data[at:at + 16])))


def find_ids(img: PeImage, ids: set[str]) -> dict[int, str]:
    """{rva: id} for every in-memory copy of a known id in the data sections, or in the code sections when the data
    sections have none (a linker can merge .rdata into .text)."""
    found = {}
    patterns = {guid_bytes(g): g for g in ids}
    for code in (False, True):
        for s in img.sections:
            if s["exec"] != code:
                continue
            data = img.section_data(s)[:s["vsize"] or s["rawsize"]]
            for pattern, guid in patterns.items():
                at = data.find(pattern)
                while at >= 0:
                    found[s["rva"] + at] = guid
                    at = data.find(pattern, at + 1)
        if found:
            break
    return found


LEA_RIP = re.compile(rb"[\x48\x4c]\x8d[\x05\x0d\x15\x1d\x25\x2d\x35\x3d]")


def lea_sites(code: Code, targets: dict[int, str]):
    """RVAs of `lea r64,[rip+x]` instructions whose x is one of targets."""
    for base, data in code.sections:
        for m in LEA_RIP.finditer(data):
            at = m.start()
            if at + 7 > len(data):
                break
            if base + at + 7 + struct.unpack_from("<i", data, at + 3)[0] in targets:
                yield base + at


def branches(code: Code, opcode: bytes):
    """(source rva, target rva) of every byte sequence that reads as `opcode rel32` (E8 call, E9 jmp, FF 15 call
    [rip+x]). Some are not instructions; callers match the targets against addresses they know."""
    size = len(opcode) + 4
    unpack = struct.Struct("<i").unpack_from
    for base, data in code.sections:
        at = data.find(opcode)
        while 0 <= at <= len(data) - size:
            yield base + at, base + at + size + unpack(data, at + size - 4)[0]
            at = data.find(opcode, at + 1)


def references(img: PeImage, code: Code, starts: set[int]) -> dict[int, int]:
    """{function start: references} for the thunk functions at `starts`: calls, tail jumps, RIP-relative address
    loads and 8-byte pointers in data, each also through an incremental-link `jmp` stub (a jmp to the function from
    outside every .pdata function). A reference from inside another of these functions counts only when that one
    is referenced itself (every thunk calls the XThreadAssertNotTimeSensitive thunk, live or not). 0 means nothing
    that is reached reaches it."""
    def source_function(source: int) -> int | None:
        func = code.function(source)
        return func[0] if func else None

    stubs: dict[int, int] = {}
    outside = dict.fromkeys(starts, 0)
    inside: dict[int, set[int]] = {s: set() for s in starts}

    def add(func: int, source: int | None) -> None:
        if source in inside and source != func:
            inside[func].add(source)
        elif source not in inside:
            outside[func] += 1

    for source, target in branches(code, b"\xe9"):
        if target in outside:
            func = source_function(source)
            if func is None:
                stubs[source] = target                 # not inside any function: an incremental-link stub
            else:
                add(target, func)                      # a tail call
    owner = {s: s for s in starts} | stubs
    for source, target in branches(code, b"\xe8"):
        if target in owner:
            add(owner[target], source_function(source))
    for base, data in code.sections:
        for m in LEA_RIP.finditer(data):
            at = m.start()
            if at + 7 <= len(data):
                func = owner.get(base + at + 7 + struct.unpack_from("<i", data, at + 3)[0])
                if func is not None:
                    add(func, source_function(base + at))
    by_va = {img.image_base + addr: func for addr, func in owner.items()}
    for s in img.sections:
        if s["exec"]:
            continue
        data = img.section_data(s)
        for value in memoryview(data[:len(data) // 8 * 8]).cast("Q"):
            func = by_va.get(value)
            if func is not None:
                outside[func] += 1
    # Propagate: a thunk referenced by a referenced thunk is referenced.
    count = dict(outside)
    changed = True
    while changed:
        changed = False
        for func, sources in inside.items():
            live = sum(1 for s in sources if count[s] > 0)
            if outside[func] + live != count[func]:
                count[func] = outside[func] + live
                changed = True
    return count


def call_after(code: Code, lea_rva: int) -> Insn | None:
    """The first call within LEA_TO_CALL instructions of an id's lea, unless control flow leaves first."""
    rva = lea_rva
    for _ in range(LEA_TO_CALL):
        i = code.decode(rva)
        if i is None or i.ends_flow():
            return None
        if i.is_call():
            return i
        rva = i.end
    return None


def call_arguments(code: Code, call: int) -> list[int | None]:
    """The RIP-relative addresses passed in rcx, rdx, r8 and r9 to the call at `call`, in that order (None for a
    register that holds none). The instructions before it are decoded from its function's start when .pdata covers
    it and the start is near; else (or when that finds none) from each byte in BACK_WINDOW from which decoding lands
    on the call, the first run that loads an address into rcx or rdx winning."""
    func = code.function(call)
    starts = [func[0]] if func and call - func[0] <= 4096 else []
    starts += range(max(func[0] if func else 0, call - BACK_WINDOW), call)
    for start in starts:
        run = code.run(start, call)
        if run is not None:
            found = argument_addresses(run)
            if found[0] is not None or found[1] is not None:
                return found
    return [None] * 4


def argument_addresses(run: list[Insn]) -> list[int | None]:
    """The RIP-relative addresses in rcx, rdx, r8 and r9 at the end of `run` (None where there isn't one)."""
    state: dict[int, int] = {}
    for i in run:
        if i.is_call():
            for r in VOLATILE:
                state.pop(r, None)
            continue
        if i.ends_flow():
            state.clear()
            continue
        if i.op == 0x8D and i.wide and i.base == RIP:
            state[i.reg] = i.target
            continue
        if i.wide and i.mod == 3 and i.op in (0x8B, 0x89):      # mov r64, r64
            dst, src = (i.reg, i.rm) if i.op == 0x8B else (i.rm, i.reg)
            if src in state:
                state[dst] = state[src]
            else:
                state.pop(dst, None)
            continue
        for r in writes(i):
            state.pop(r, None)
    return [state.get(r) for r in (RCX, RDX, R8, R9)]


def slot_calls(code: Code, start: int, helpers: set) -> tuple[list[tuple[int, int, int]], set]:
    """[(slot, estimated argument count, call rva)] of the vtable calls after a helper call returns at `start`, and
    the keys of the direct calls that dispatched them (the CFG dispatch routine)."""
    func = code.function(start)
    stop = func[1] if func else start + NO_PDATA_WINDOW
    vtables: set[int] = set()            # registers holding a vtable pointer
    methods: dict[int, int] = {}         # register -> slot of the method pointer it holds
    written: set[int] = set()            # argument registers written since the last call
    stack: set[int] = set()              # stack arguments stored since the last call
    found, dispatchers = [], set()
    rva = start
    while rva < stop:
        i = code.decode(rva)
        if i is None:
            break
        rva = i.end
        if i.is_call():
            slot = None
            if i.op == 0xFF and i.mod != 3 and i.base in vtables and i.index is None and i.disp % 8 == 0:
                slot = i.disp // 8                                  # call [vtable+disp]
            elif i.op == 0xFF and i.mod == 3:
                slot = methods.get(i.rm)                            # call reg
            elif i.call_key() is not None and RAX in methods:
                slot = methods[RAX]                                 # CFG: call __guard_dispatch_icall (target in rax)
                dispatchers.add(i.call_key())
            if slot is None and i.call_key() in helpers:
                break                                               # the next thunk's helper call
            if slot is not None:
                args = max([ARG_INDEX[r] for r in written if r in ARG_INDEX] + list(stack) + [0])
                found.append((slot, args, i.rva))
                if slot == RELEASE:
                    break
            for r in VOLATILE:
                vtables.discard(r)
                methods.pop(r, None)
            written.clear()
            stack.clear()
            continue
        if i.ends_flow():
            if not func:
                break
            vtables.clear()
            methods.clear()
            continue
        k = stack_argument(i)
        if k is not None:
            stack.add(k)
        dests = writes(i)
        written.update(dests)
        load = (i.op == 0x8B and i.wide and i.mod != 3 and i.index is None
                and i.base is not None and i.base not in (RIP, RSP))
        from_vtable = load and i.base in vtables and i.disp >= 0 and i.disp % 8 == 0
        copy = i.wide and i.mod == 3 and i.op in (0x8B, 0x89)       # mov r64, r64 carries what its source held
        if copy:
            src = i.rm if i.op == 0x8B else i.reg
            src_vtable, src_method = src in vtables, methods.get(src)
        for r in dests:
            vtables.discard(r)
            methods.pop(r, None)
        if copy:
            if src_vtable:
                vtables.add(dests[0])
            if src_method is not None:
                methods[dests[0]] = src_method
        elif from_vtable:
            methods[i.reg] = i.disp // 8                            # mov r, [vtable+disp]
        elif load and i.disp == 0:
            vtables.add(i.reg)                                      # mov r, [object]
    return found, dispatchers


def scan(img: PeImage, table: RuntimeTable) -> dict:
    """Thunk sites of the image: {"helpers": [...], "sites": [...], "ids_present": {id: [rva, ...]}}."""
    known = find_ids(img, table.ids())
    present: dict[str, list[int]] = {}
    for rva, guid in sorted(known.items()):
        present.setdefault(guid, []).append(rva)
    result = {"helpers": [], "sites": [], "unmatched": [], "ids_present": present, "pdata_functions": 0}
    if img.machine != 0x8664:
        result["skipped"] = f"{MACHINES.get(img.machine, hex(img.machine))} code is not scanned (x86-64 only)"
        return result
    code = Code(img)
    result["pdata_functions"] = len(code.begins)

    # A helper: a call fed by a known id's lea and followed by a vtable call past IUnknown.
    candidates: dict[int, Insn] = {}
    for lea in lea_sites(code, known):
        call = call_after(code, lea)
        key = call.call_key() if call else None
        if key is not None:
            candidates.setdefault(call.rva, call)
    helpers, dispatchers = set(), set()
    for call in candidates.values():
        calls, used = slot_calls(code, call.end, set())
        dispatchers |= used
        if any(slot >= IUNKNOWN_SLOTS for slot, _, _ in calls):
            helpers.add(call.call_key())
    helpers -= dispatchers           # a CFG dispatch routine is never the helper
    result["helpers"] = sorted(helpers)

    near = {t for kind, t in helpers if kind == "rel"}
    helper_calls = {source for source, target in branches(code, b"\xe8") if target in near} if near else set()
    far = {t for kind, t in helpers if kind == "mem"}
    if far:
        helper_calls.update(source for source, target in branches(code, b"\xff\x15") if target in far)

    for call_rva in sorted(helper_calls):
        call = code.decode(call_rva)
        if call is None or call.call_key() not in helpers:
            continue
        # The class id is the one in rcx, the interface id the one in rdx (the class id when rdx holds none).
        rcx, rdx = (None if a is None else guid_at(img, a) for a in call_arguments(code, call_rva)[:2])
        class_id, interface_id = rcx, rdx or rcx
        slots = [c for c in slot_calls(code, call.end, helpers)[0] if c[0] >= IUNKNOWN_SLOTS] if rcx else []
        func = code.function(call_rva)
        site = {"call": call_rva, "function": func[0] if func else None, "class_id": class_id,
                "interface_id": interface_id, "slots": slots}
        result["sites" if slots else "unmatched"].append(site)

    starts = {s["function"] for s in result["sites"] if s["function"] is not None}
    refs = references(img, code, starts) if starts else {}
    for s in result["sites"]:
        s["references"] = refs.get(s["function"]) if s["function"] is not None else None
    return result


def ranges(numbers) -> str:
    """'3-7, 9, 12-14' for [3, 4, 5, 6, 7, 9, 12, 13, 14]."""
    out, run = [], []
    for n in sorted(set(numbers)):
        if run and n == run[-1] + 1:
            run.append(n)
            continue
        if run:
            out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else str(run[0]))
        run = [n]
    if run:
        out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else str(run[0]))
    return ", ".join(out)


def report(path: str, table: RuntimeTable, served: dict[str, str] | None) -> dict:
    """The JSON report of one image. Raises PeFormatError or OSError when it can't be read."""
    with open(path, "rb") as f:
        data = f.read()
    try:
        img = PeImage(data)
        found = scan(img, table)
    except struct.error as e:
        raise PeFormatError(f"truncated or corrupt table: {e}") from None
    cut = [s["name"] for s in img.sections if s["rawsize"] and s["rawptr"] + s["rawsize"] > len(data)]

    classes: dict[str, dict] = {}
    for site in found["sites"]:
        cid, iid = site["class_id"], site["interface_id"]
        refs = site["references"]
        cls = classes.setdefault(cid, {
            "class": table.class_of(cid), "class_id": cid,
            "served": None if served is None else cid in served,
            "thunks": 0, "referenced_thunks": 0, "interfaces": {}})
        cls["thunks"] += 1
        if cls["referenced_thunks"] is not None:
            cls["referenced_thunks"] = None if refs is None else cls["referenced_thunks"] + (refs > 0)
        names_from = table.interface_for(iid)
        iface = cls["interfaces"].setdefault(iid, {
            "interface": table.label(iid), "interface_id": iid, "slots_named_from": names_from,
            "without_layout": table.slot_table(names_from)[1] if names_from else [], "thunks": 0, "slots": {}})
        iface["thunks"] += 1
        where = f"{site['function'] if site['function'] is not None else site['call']:#x}"
        for slot, args, _ in site["slots"]:
            method, via = table.method(names_from, slot)
            entry = iface["slots"].setdefault(slot, {"slot": slot, "method": method, "named_from": via,
                                                     "args_estimate": set(), "thunk_rvas": {}})
            entry["args_estimate"].add(args)
            entry["thunk_rvas"][where] = refs

    out_classes = []
    for cid, cls in sorted(classes.items(), key=lambda kv: (kv[1]["class"] or "~", kv[0])):
        ifaces = []
        for iface in sorted(cls["interfaces"].values(), key=lambda i: i["interface"]):
            slots = []
            for _, s in sorted(iface["slots"].items()):
                refs = list(s["thunk_rvas"].values())
                slots.append(dict(s, args_estimate=sorted(s["args_estimate"]), thunk_rvas=sorted(s["thunk_rvas"]),
                                  referenced=None if None in refs else sum(r > 0 for r in refs)))
            known = all(s["referenced"] is not None for s in slots)
            ifaces.append(dict(iface, slots=slots, slot_list=ranges(s["slot"] for s in slots),
                               referenced_slot_list=ranges(s["slot"] for s in slots if s["referenced"])
                               if known else None))
        out_classes.append(dict(cls, interfaces=ifaces))

    reached = {table.class_of(c) for c in classes} | set(classes)
    unreached = []
    for name, cls in sorted(table.classes.items()):
        if name in reached:
            continue
        ids = [cls["id"]] + [i["id"] for i in table.interfaces.values() if i["class"] == name]
        hits = [{"id": g, "names": table.label(g), "copies": len(found["ids_present"][g])}
                for g in dict.fromkeys(ids) if g in found["ids_present"]]
        if hits:
            unreached.append({"class": name, "class_id": cls["id"], "ids": hits})

    unmatched = [{"call": f"{s['call']:#x}", "class": table.class_of(s["class_id"]) if s["class_id"] else None,
                  "class_id": s["class_id"], "interface_id": s["interface_id"]} for s in found["unmatched"]]
    refs = [s["references"] for s in found["sites"]]
    return {
        "file": os.path.basename(path),
        "size": len(data),
        "machine": MACHINES.get(img.machine, hex(img.machine)),
        "is_dll": bool(img.characteristics & 0x2000),
        "pdata_functions": found["pdata_functions"],
        "cut_off_sections": cut,
        "skipped": found.get("skipped"),
        "helpers": [f"{kind} {rva:#x}" for kind, rva in found["helpers"]],
        "thunks": len(found["sites"]),
        "referenced_thunks": None if None in refs else sum(r > 0 for r in refs),
        "classes": out_classes,
        "helper_calls_without_slot_call": unmatched,
        "unreached_classes": unreached,
    }


def served_text(flag) -> str:
    return {True: "served by the runtime", False: "NOT served by the runtime", None: "served: unknown"}[flag]


def referenced_text(count, total) -> str:
    if count is None:
        return "references unknown"
    if count == total:
        return "referenced"
    return "unreferenced" if count == 0 else f"{count} referenced"


def to_text(r: dict) -> str:
    kind = "DLL" if r["is_dll"] else "EXE"
    out = [f"== {r['file']} ({r['machine']} {kind}, {r['size']:,} bytes, {r['pdata_functions']:,} .pdata functions)"]
    if r["cut_off_sections"]:
        out.append(f"WARNING: cut off inside {', '.join(r['cut_off_sections'])}: the file ends before the section "
                   "data, so this report is incomplete")
    if r["skipped"]:
        out.append(r["skipped"])
    out.append(f"QueryApiImpl helper(s): {', '.join(r['helpers']) or 'none found'}; {r['thunks']} thunk(s) reach "
               f"{len(r['classes'])} runtime class(es)")
    if r["thunks"]:
        if r["referenced_thunks"] is None:
            out.append("References to the thunks are unknown (no .pdata function bounds).")
        else:
            out.append(f"{r['referenced_thunks']} of the {r['thunks']} thunks are referenced (called, jumped to, "
                       "address taken or stored) in this image; an unreferenced one is present but nothing in the "
                       "image reaches it (an incremental link does not remove unreferenced functions).")
        out.append("args~N: the argument count after `this`, estimated from the registers and stack slots written "
                   "before the call.")
    width = max([len(s["method"] or "unknown") + (len(s["named_from"]) + 7 if s["named_from"] else 0)
                 for c in r["classes"] for i in c["interfaces"] for s in i["slots"]] + [40])
    for cls in r["classes"]:
        out.append("")
        out.append(f"{cls['class'] or 'unknown class'}  {cls['class_id']}  {served_text(cls['served'])}  "
                   f"{cls['thunks']} thunk(s), {referenced_text(cls['referenced_thunks'], cls['thunks'])}")
        for iface in cls["interfaces"]:
            names = iface["slots_named_from"]
            source = "" if names is None or names in iface["interface"].split("/") else f", slots named from {names}"
            out.append(f"  interface {iface['interface']}  {iface['interface_id']}{source}: "
                       f"slots {iface['slot_list']}")
            if iface["without_layout"]:
                out.append(f"    (no documented layout for {', '.join(iface['without_layout'])}: its own methods, "
                           "if any, are named in its base's list; slots past the list are unknown)")
            if iface["interface_id"] != cls["class_id"]:
                out.append("    (requested with an interface id other than the class id)")
            for s in iface["slots"]:
                name = s["method"] or "unknown"
                if s["named_from"]:
                    name += f" (from {s['named_from']})"
                args = "/".join(str(a) for a in s["args_estimate"])
                out.append(f"    slot {s['slot']:>3}  {name:<{width}}  args~{args:<4} {len(s['thunk_rvas'])} thunk(s), "
                           f"{referenced_text(s['referenced'], len(s['thunk_rvas']))}")
    todo = [c for c in r["classes"] if c["served"] is not True]
    if todo:
        out.append("")
        out.append("Not served by the runtime (class as interface: slots called; the referenced ones):"
                   if all(c["served"] is False for c in todo) else
                   "Served status unknown (class as interface: slots called; the referenced ones):")
        for c in todo:
            for iface in c["interfaces"]:
                ref = iface["referenced_slot_list"]
                ref = "unknown" if ref is None else (ref or "none")
                out.append(f"  {c['class'] or c['class_id']} as {iface['interface']}: {iface['slot_list']}; "
                           f"referenced: {ref}")
    if r["helper_calls_without_slot_call"]:
        out.append("")
        out.append("Helper calls not followed to a slot (no class id read from rcx, or no slot call found after "
                   f"them): {len(r['helper_calls_without_slot_call'])}")
        for u in r["helper_calls_without_slot_call"]:
            if u["class_id"] is None:
                out.append(f"  {u['call']}  class id not read; interface {u['interface_id'] or 'not read'}")
            else:
                out.append(f"  {u['call']}  {u['class'] or 'unknown class'} {u['class_id']} / {u['interface_id']}")
    if r["unreached_classes"]:
        out.append("")
        out.append(f"Known runtime ids present but reached by no thunk ({len(r['unreached_classes'])} classes):")
        for u in r["unreached_classes"]:
            ids = ", ".join(f"{h['names']} {h['id']}" for h in u["ids"])
            out.append(f"  {u['class']}: {ids}")
    return "\n".join(out) + "\n"


def summary(r: dict) -> str:
    todo = [c["class"] or c["class_id"] for c in r["classes"] if c["served"] is False]
    live = [c["class"] or c["class_id"] for c in r["classes"] if c["served"] is False and c["referenced_thunks"]]
    ref = "?" if r["referenced_thunks"] is None else r["referenced_thunks"]
    text = (f"{r['file']:<32} thunks={r['thunks']} referenced={ref} classes={len(r['classes'])} "
            f"not-served={','.join(todo) or '-'} not-served-referenced={','.join(live) or '-'}")
    if r["cut_off_sections"]:
        text += "  CUT OFF"
    return text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("image", nargs="+")
    ap.add_argument("--json", action="store_true", help="print JSON instead of text")
    ap.add_argument("--out", help="write <image>.gdk_calls.txt and .json here and print a summary line per image")
    ap.add_argument("--table", default=DEFAULT_TABLE, help="the class table (default: %(default)s)")
    ap.add_argument("--runtime-source", default=DEFAULT_RUNTIME_SOURCE,
                    help="a shim source, or a folder of xgameruntime*.cpp, whose `constexpr GUID k...Class` ids count "
                         "as served (default: %(default)s)")
    a = ap.parse_args(argv)

    try:
        table = RuntimeTable.load(a.table)
    except (OSError, ValueError, KeyError, TypeError) as e:
        print(f"cannot read the class table {a.table}: {e}", file=sys.stderr)
        return 1
    try:
        served = served_classes(read_runtime_source(a.runtime_source))
    except OSError as e:
        print(f"warning: cannot read {a.runtime_source} ({e.strerror or e}); 'served' is unknown", file=sys.stderr)
        served = None
    if a.out:
        if os.path.isfile(a.out):
            print(f"cannot create output directory {a.out}: path is a file, not a directory", file=sys.stderr)
            return 1
        try:
            os.makedirs(a.out, exist_ok=True)
        except OSError as e:
            print(f"cannot create output directory {a.out}: {e.strerror or e}", file=sys.stderr)
            return 1

    rc = 0
    reports, used = [], set()
    for path in a.image:
        try:
            r = report(path, table, served)
        except (PeFormatError, OSError) as e:
            print(f"{path}: not a readable PE image ({e})", file=sys.stderr)
            rc = 1
            continue
        if r["cut_off_sections"]:
            print(f"{path}: the file ends inside {', '.join(r['cut_off_sections'])}; the report is incomplete",
                  file=sys.stderr)
            rc = 1
        reports.append(r)
        if not a.out:
            continue
        name, n = r["file"], 1
        while name.lower() in used:
            n += 1
            name = f"{r['file']}-{n}"
        used.add(name.lower())
        if n > 1:
            print(f"{path}: another input is also named {r['file']}; report written as {name}.gdk_calls.txt/.json",
                  file=sys.stderr)
        stem = os.path.join(a.out, name + ".gdk_calls")
        json_path = stem + ".json"
        txt_path = stem + ".txt"
        inputs = list(a.image) + [a.table]
        if os.path.isfile(a.runtime_source):
            inputs.append(a.runtime_source)
        collision = next(
            (out_path for out_path in (json_path, txt_path)
             if any(same_report_path(inp, out_path) for inp in inputs)),
            None,
        )
        if collision:
            print(f"{path}: report {collision} would overwrite input file", file=sys.stderr)
            rc = 1
            continue
        try:
            with open(json_path, "w", encoding="utf-8", newline="\n") as f:
                json.dump({"table": table.source, **r}, f, indent=1)
                f.write("\n")
            with open(txt_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(to_text(r))
        except OSError as e:
            print(f"{path}: cannot write report ({e.strerror or e})", file=sys.stderr)
            rc = 1
            continue
        print(summary(r))

    if not a.out:
        if a.json:
            print(json.dumps({"table": table.source, "images": reports}, indent=1))
        else:
            print(f"(class table: {table.source})")
            for r in reports:
                print()
                print(to_text(r), end="")
    return rc


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
