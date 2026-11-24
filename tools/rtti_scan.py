"""Recover C++ class names, hierarchies and vtables from an x64 MSVC PE via RTTI.

x64 RTTI uses image-relative (RVA) fields:
  TypeDescriptor             { u64 pVFTable; u64 spare; char name[] = ".?AVClass@ns@@" }
  CompleteObjectLocator      { u32 signature(=1); u32 offset; u32 cdOffset; u32 rvaTypeDescriptor;
                               u32 rvaClassHierarchy; u32 rvaSelf }
  ClassHierarchyDescriptor   { u32 signature; u32 attributes; u32 numBaseClasses; u32 rvaBaseClassArray }
  BaseClassDescriptor        { u32 rvaTypeDescriptor; u32 numContainedBases; i32 mdisp, pdisp, vdisp; u32 attrs;
                               u32 rvaClassHierarchy }
A vtable is preceded (at -8) by a pointer to its CompleteObjectLocator.

Usage:
    python -I rtti_scan.py <image> [--out file.json]
Output: classes (demangled-ish name, type descriptor RVA, base classes) and vtables (RVA, class, offset in the
complete object, number of slots that point into executable sections).
"""
import argparse
import json
import os
import re
import struct
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from file_mode import published_mode  # noqa: E402



IMAGE_SCN_MEM_EXECUTE = 0x20000000
PE32PLUS_MAGIC = 0x020B
MAX_PE_IMAGE_SIZE = 0x80000000  # 2 GiB (PE32+ spec maximum)


def load_pe64_image(data: bytes) -> tuple[int, bytes, list[dict]]:
    """Parse a 64-bit PE32+ image using only the Python standard library.

    Returns (image_base, memory_mapped_image, sections) where each section dict contains
    'name', 'virtual_address', 'virtual_size', 'raw_ptr', 'raw_size', and 'characteristics'.
    """
    if len(data) < 0x40 or data[:2] != b"MZ":
        raise ValueError("not a PE image: missing MZ header")
    (e_lfanew,) = struct.unpack_from("<I", data, 0x3C)
    if e_lfanew + 24 > len(data) or data[e_lfanew : e_lfanew + 4] != b"PE\x00\x00":
        raise ValueError("not a PE image: missing PE signature")

    _machine, num_sections, _ts, _sym, _nsym, opt_size, _chars = struct.unpack_from(
        "<HHIIIHH", data, e_lfanew + 4
    )
    opt_off = e_lfanew + 24
    if opt_size < 64 or opt_off + opt_size > len(data):
        raise ValueError("truncated PE optional header")
    (magic,) = struct.unpack_from("<H", data, opt_off)
    if magic != PE32PLUS_MAGIC:
        raise ValueError(f"unsupported PE optional header magic 0x{magic:04X} (expected PE32+ 0x020B)")

    (image_base,) = struct.unpack_from("<Q", data, opt_off + 24)
    size_of_image, size_of_headers = struct.unpack_from("<II", data, opt_off + 56)

    sec_table_off = opt_off + opt_size
    if sec_table_off + num_sections * 40 > len(data):
        raise ValueError("truncated PE section table")

    sections: list[dict] = []
    mapped_size = size_of_image
    for i in range(num_sections):
        sec_off = sec_table_off + i * 40
        raw_name, vsize, vaddr, rsize, rptr, _rel, _ln, _nrel, _nln, chars = struct.unpack_from(
            "<8sIIIIIIHHI", data, sec_off
        )
        span = vsize if vsize > 0 else rsize
        end = vaddr + max(vsize, rsize)
        if end > mapped_size:
            mapped_size = end
        sections.append({
            "name": raw_name.split(b"\x00", 1)[0].decode("latin-1"),
            "virtual_address": vaddr,
            "virtual_size": span,
            "raw_ptr": rptr,
            "raw_size": rsize,
            "characteristics": chars,
        })

    if mapped_size <= 0 or mapped_size > MAX_PE_IMAGE_SIZE:
        raise ValueError(f"invalid PE mapped image size: {mapped_size}")

    img = bytearray(mapped_size)
    hdr_copy = min(size_of_headers, len(data), mapped_size)
    if hdr_copy > 0:
        img[0:hdr_copy] = data[0:hdr_copy]

    for sec in sections:
        rptr = sec["raw_ptr"]
        rsize = sec["raw_size"]
        vaddr = sec["virtual_address"]
        if rsize <= 0 or rptr >= len(data) or vaddr >= mapped_size:
            continue
        copy_len = min(rsize, len(data) - rptr, mapped_size - vaddr)
        if copy_len > 0:
            img[vaddr : vaddr + copy_len] = data[rptr : rptr + copy_len]

    return image_base, bytes(img), sections


_CV = {"A": (), "B": ("const",), "C": ("volatile",), "D": ("const", "volatile")}
_BUILTIN = {"X": "void", "C": "signed char", "D": "char", "E": "unsigned char", "F": "short", "G": "unsigned short",
            "H": "int", "I": "unsigned int", "J": "long", "K": "unsigned long", "M": "float", "N": "double",
            "O": "long double", "_D": "__int8", "_E": "unsigned __int8", "_F": "__int16", "_G": "unsigned __int16",
            "_H": "__int32", "_I": "unsigned __int32", "_J": "__int64", "_K": "unsigned __int64", "_L": "__int128",
            "_M": "unsigned __int128", "_N": "bool", "_Q": "char8_t", "_S": "char16_t", "_U": "char32_t",
            "_W": "wchar_t"}
_OPERATOR = {"2": " new", "3": " delete", "4": "=", "5": ">>", "6": "<<", "7": "!", "8": "==", "9": "!=", "A": "[]",
             "B": "", "C": "->", "D": "*", "E": "++", "F": "--", "G": "-", "H": "+", "I": "&", "J": "->*", "K": "/",
             "L": "%", "M": "<", "N": "<=", "O": ">", "P": ">=", "Q": ",", "R": "()", "S": "~", "T": "^", "U": "|",
             "V": "&&", "W": "||", "X": "*=", "Y": "+=", "Z": "-=", "_0": "/=", "_1": "%=", "_2": ">>=",
             "_3": "<<=", "_4": "&=", "_5": "|=", "_6": "^=", "_U": " new[]", "_V": " delete[]"}
_POINTER = {"P": " *", "Q": " * const", "R": " * volatile", "S": " * const volatile", "A": " &", "B": " & volatile"}
_QUALIFIERS = re.compile(r"(?: const| volatile)*$")
_SLOT = "\0"    # in a function or array type: where a further `*`, `&` or qualifier goes, `void (*|)(int)`
_LOCAL_SCOPE = re.compile(r"\?(?:\d|[B-P][A-P]*@)\?")
# How a type descriptor's name ends: '@@'; a local scope's function ('@?...Z@', '_E@' when noexcept); a back-reference.
_NAME_END = re.compile(rb"(?:@@|@\?[\x21-\x7e]*(?:Z|_E)@|@\d+@)\x00$")


class _TypeName:
    """A reader for the part of MSVC's name mangling that a class name can contain: scopes, templates and their
    arguments, back-references, anonymous namespaces and names local to a function. A form it does not know
    raises ValueError (or runs off the end: IndexError); it never guesses."""

    def __init__(self, text):
        self.s, self.i = text, 0
        self.names, self.params = [], []     # the back-reference tables: name pieces, function parameter types

    def peek(self):
        return self.s[self.i:self.i + 1]

    def next(self):
        c = self.s[self.i]
        self.i += 1
        return c

    def eat(self, text):
        if self.s.startswith(text, self.i):
            self.i += len(text)
            return True
        return False

    def number(self):
        negative = self.eat("?")
        c = self.next()
        if c.isdigit():
            value = int(c) + 1
        else:
            value = 0
            while c != "@":
                if not "A" <= c <= "P":
                    raise ValueError("bad number")
                value = value * 16 + ord(c) - ord("A")
                c = self.next()
        return -value if negative else value

    def memorize(self, name):
        if name not in self.names and len(self.names) < 10:
            self.names.append(name)

    def simple(self):
        end = self.s.index("@", self.i)
        name = self.s[self.i:end]
        if not name or name[0] in "?$" or name[0].isdigit():
            raise ValueError("bad name")
        self.i = end + 1
        self.memorize(name)
        return name

    def piece(self):
        """One piece of a scope chain."""
        c = self.peek()
        if c.isdigit():
            self.i += 1
            return self.names[int(c)]
        if self.s.startswith("?$", self.i):
            name = self.template()
            self.memorize(name)
            return name
        if self.eat("?A"):
            self.i = self.s.index("@", self.i) + 1
            self.memorize("`anonymous namespace'")
            return "`anonymous namespace'"
        if _LOCAL_SCOPE.match(self.s, self.i):
            self.i += 1
            ordinal = self.number()
            self.i += 1
            return f"`{self.symbol()}'::`{ordinal}'"
        return self.simple()

    def qualified(self):
        pieces = [self.piece()]
        while not self.eat("@"):
            pieces.append(self.piece())
        return "::".join(reversed(pieces))

    def template(self):
        """`?$name@arguments@`. A template name has back-reference tables of its own."""
        self.i += 2
        outer = self.names, self.params
        self.names, self.params = [], []
        name = self.operator() if self.peek() == "?" else self.simple()
        if name == "operator":
            raise ValueError("a conversion operator template is not read")
        arguments = []
        while not self.eat("@"):
            if self.eat("$$$V") or self.eat("$$V") or self.eat("$$Z") or self.eat("$S"):
                continue                                   # an empty pack, or the boundary between two packs
            if self.eat("$M"):                             # a value of a deduced (`auto`) type: the type, the value
                self.type()
                if not self.eat("0"):
                    raise ValueError("unknown template argument")
                arguments.append(str(self.number()))
            elif self.eat("$0"):
                arguments.append(str(self.number()))
            elif self.eat("$$Y"):
                arguments.append(self.qualified())
            else:
                arguments.append(self.type().replace(_SLOT, ""))
        self.names, self.params = outer
        return f"{name}<{', '.join(arguments)}>"

    def operator(self):
        self.i += 1
        code = self.next()
        if code == "_":
            code += self.next()
        if code in ("0", "1"):
            return code                                    # constructor, destructor: named after the class
        if code not in _OPERATOR:
            raise ValueError("unknown special name")
        return "operator" + _OPERATOR[code]

    def symbol(self):
        """A whole decorated function name, as a local scope holds one. Returns its qualified name; the signature
        is read, to find where it ends, and dropped."""
        if not self.eat("?"):
            raise ValueError("not a symbol")
        c = self.peek()
        if c.isdigit():
            self.i += 1
            name = self.names[int(c)]
        elif self.s.startswith("?$", self.i):
            name = self.template()
        elif c == "?":
            name = self.operator()
        else:
            name = self.simple()
        scope = []
        while not self.eat("@"):
            scope.append(self.piece())
        if name[0] in "01":
            if not scope:
                raise ValueError("a constructor outside a class")
            name = ("~" if name[0] == "1" else "") + scope[0] + name[1:]
        kind = self.next()
        if kind in "ABEFIJMNQRUV":                         # a non-static member: `this` qualifiers
            while self.peek() in ("E", "F", "G", "H", "I"):
                self.i += 1
            self.cv()
        elif kind not in "CDKLSTYZ":
            raise ValueError("not a plain function")
        result = self.signature()[0]
        if name == "operator":                             # a conversion: named after the type it returns
            name += " " + result.replace(_SLOT, "")
        return "::".join(scope[::-1] + [name])

    def cv(self):
        return _CV[self.next()]

    def sizes(self):
        """An array's dimensions: how many, then each."""
        count = self.number()
        sizes = [self.number() for _ in range(count)]
        if count < 1 or min(sizes) < 0:
            raise ValueError("bad array dimensions")
        return "".join(f"[{n}]" for n in sizes)

    @staticmethod
    def declare(text, declarator, keep=False):
        """The type `text` with `declarator` (` *`, ` (*)(int)`, ` [4]`, ...) applied. It goes after a plain type,
        and inside the brackets of a function or array type: a pointer to `void (*)(int)` is `void (**)(int)`.
        `keep` leaves the slot after a `*` or `&`; a function or array declarator brings its own."""
        if _SLOT in text:
            return text.replace(_SLOT, declarator.lstrip() + (_SLOT if keep else ""))
        return text + declarator

    @staticmethod
    def qualify(text, qualifiers):
        """`text` with `const`/`volatile` after it. A const pointer is marked twice, by its own code and by the
        qualifiers of whatever refers to it, so one that is already there is not repeated."""
        head, slot, tail = text.partition(_SLOT)
        present = _QUALIFIERS.search(head).group(0).split()
        return head + "".join(f" {q}" for q in qualifiers if q not in present) + slot + tail

    def signature(self):
        """Calling convention, return type, parameters, exception specification: (return type, parameters,
        ` noexcept` or nothing)."""
        if not "A" <= self.next() <= "W":
            raise ValueError("unknown calling convention")
        result = "" if self.eat("@") else self.type()
        parameters = []
        if self.eat("X"):
            parameters.append("void")
        else:
            while not self.eat("@"):
                if self.eat("Z"):
                    parameters.append("...")
                    break
                if self.peek().isdigit():
                    parameters.append(self.params[int(self.next())])
                    continue
                start = self.i
                parameters.append(self.type().replace(_SLOT, ""))
                if self.i - start > 1 and len(self.params) < 10:
                    self.params.append(parameters[-1])
        if self.eat("_E"):
            return result, parameters, " noexcept"
        if not self.eat("Z"):
            raise ValueError("no exception specification")
        return result, parameters, ""

    def function(self, declarator):
        result, parameters, noexcept = self.signature()
        return self.declare(result, f" {declarator}({', '.join(parameters)}){noexcept}")

    def indirect(self, suffix):
        """What a pointer or reference refers to, with `suffix` (` *`, ` &`, ...) after it."""
        unaligned = ()
        while self.peek() in ("E", "F", "I"):              # __ptr64, __unaligned, __restrict
            if self.next() == "F":
                unaligned = ("__unaligned",)
        if self.eat("6"):
            return self.function(f"({suffix.strip()}{_SLOT})")
        if self.eat("$A"):                                 # a C++/CX handle
            suffix = suffix.replace("*", "^")
        qualifiers = self.cv()
        if self.eat("Y"):
            sizes = self.sizes()
            return self.declare(self.qualify(self.type(), qualifiers + unaligned), f" ({suffix.strip()}{_SLOT}){sizes}")
        return self.declare(self.qualify(self.type(), qualifiers + unaligned), suffix, keep=True)

    def type(self):
        if self.eat("$$Q"):
            return self.indirect(" &&")
        if self.eat("$$T"):
            return "std::nullptr_t"
        if self.eat("$$A6"):
            return self.function("")
        if self.eat("$$C") or self.eat("?"):               # a qualified type passed by value
            qualifiers = self.cv()
            return self.qualify(self.type(), qualifiers)
        if self.eat("$$BY"):                               # an array passed as a template argument
            sizes = self.sizes()
            return self.declare(self.type(), " " + sizes)
        c = self.next()
        if c == "_":
            c += self.next()
        if c in _BUILTIN:
            return _BUILTIN[c]
        if c in ("T", "U", "V"):
            return self.qualified()
        if c == "W":
            if not "0" <= self.next() <= "7":
                raise ValueError("bad enum")
            return self.qualified()
        if c in _POINTER:
            return self.indirect(_POINTER[c])
        raise ValueError("unknown type")


def simple_name(mangled: str) -> str:
    """'.?AVFoo@bar@@' -> 'bar::Foo', '.?AV?$Foo@H@bar@@' -> 'bar::Foo<int>'. The `class`/`struct`/`enum` keywords,
    calling conventions and the signature of a function that a local class sits in are left out. A name with a
    form this does not read (a member pointer or an array as a template argument, say) is returned as it is."""
    m = re.match(r"\.\?A(?:[TUV]|W[0-7]?)", mangled)
    if not m or not mangled.endswith("@"):
        return mangled
    reader = _TypeName(mangled[m.end():])
    try:
        name = reader.qualified()
    except (ValueError, IndexError, KeyError, RecursionError):
        return mangled
    return name.replace(_SLOT, "") if reader.i == len(reader.s) else mangled


def scan_pe64_rtti(data: bytes) -> dict:
    """Recover x64 MSVC TypeDescriptors, class hierarchies, and vtables from raw PE32+ bytes."""
    base, img, sections = load_pe64_image(data)
    exec_ranges = [
        (s["virtual_address"], min(s["virtual_address"] + s["virtual_size"], len(img)))
        for s in sections
        if s["characteristics"] & IMAGE_SCN_MEM_EXECUTE
    ]

    def is_exec(rva):
        return any(lo <= rva < hi for lo, hi in exec_ranges)

    def u32(rva):
        return struct.unpack_from("<I", img, rva)[0]

    # 1. Type descriptors: '.?AV' / '.?AU' / '.?AW' / '.?AT' strings, 16 bytes after the descriptor start. A name
    # ends with '@@', except where its outermost scope is a function (a local class: '...@?1??f@@YAXXZ@') or a
    # back-reference ('...@1@'); any other text that ends with one '@' is not counted.
    tds = {}
    for m in re.finditer(rb"\.\?A[TUVW][\x21-\x7e]{1,2000}?@\x00", img):
        rva = m.start() - 16
        if not _NAME_END.search(m.group(0)):
            continue
        if rva >= 0:
            tds[rva] = m.group(0)[:-1].decode("latin-1")

    # 2. Complete object locators: signature 1 and rvaSelf == own RVA.
    cols = {}
    for m in re.finditer(rb"\x01\x00\x00\x00", img):
        rva = m.start()
        if rva % 4 or rva + 24 > len(img):
            continue
        sig, off, cd, rtd, rchd, rself = struct.unpack_from("<6I", img, rva)
        if rself == rva and rtd in tds:
            cols[rva] = {"offset": off, "td": rtd, "chd": rchd}

    # 3. Class hierarchies.
    classes = {}
    chd_ok = set()
    for rva, col in cols.items():
        td = col["td"]
        if td in chd_ok:
            continue
        chd = col["chd"]
        valid_chd = False
        attrs, bases = 0, []
        if chd > 0:
            try:
                sig, attrs, nbase, rarr = struct.unpack_from("<4I", img, chd)
                if sig == 0 and rarr > 0:
                    for i in range(1, min(nbase, 256)):
                        bcd = u32(rarr + 4 * i)
                        btd = u32(bcd)
                        if btd in tds:
                            bases.append(simple_name(tds[btd]))
                    valid_chd = True
                else:
                    attrs = 0
            except struct.error:
                attrs, bases = 0, []
        if valid_chd or td not in classes:
            classes[td] = {"name": simple_name(tds[td]), "mangled": tds[td], "td_rva": td,
                           "multiple": bool(attrs & 1), "virtual": bool(attrs & 2), "bases": bases}
        if valid_chd:
            chd_ok.add(td)

    # 4. Vtables: a qword equal to (image base + COL rva), followed by code pointers.
    vtables = []
    col_vas = {base + rva: rva for rva in cols}
    for sec in sections:
        if sec["characteristics"] & IMAGE_SCN_MEM_EXECUTE:
            continue
        lo = sec["virtual_address"]
        hi = min(lo + sec["virtual_size"], len(img))
        for p in range((lo + 7) & ~7, hi - 8, 8):
            q = struct.unpack_from("<Q", img, p)[0]
            if q in col_vas:
                vt = p + 8
                n = 0
                while vt + 8 * (n + 1) <= hi:
                    f = struct.unpack_from("<Q", img, vt + 8 * n)[0] - base
                    if not (0 <= f < len(img) and is_exec(f)):
                        break
                    n += 1
                col = cols[col_vas[q]]
                vtables.append({"rva": vt, "class": classes.get(col["td"], {}).get("name", "?"),
                                "offset_in_object": col["offset"], "slots": n})

    return {
        "type_descriptors": len(tds),
        "classes": sorted(classes.values(), key=lambda c: c["name"]),
        "vtables": sorted(vtables, key=lambda v: v["rva"]),
    }


def write_report(path, text):
    """Write the report next to its destination and swap it in, so a write that fails part-way (a full disk, a
    locked range) leaves the previous report as it was instead of an empty or truncated file. The result keeps
    the previous report's permission bits, or gets the usual ones for a new file."""
    parent = os.path.dirname(os.path.abspath(path))
    tmp = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=parent, prefix=".rtti-",
                                         suffix=".tmp", delete=False) as f:
            tmp = f.name
            f.write(text)
        os.chmod(tmp, published_mode(path))
        os.replace(tmp, path)
    except BaseException:
        if tmp is not None:
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--out")
    a = ap.parse_args()
    if not os.path.exists(a.image):
        print(f"ERROR: {a.image}: does not exist", file=sys.stderr)
        return 2
    path = a.out or re.sub(r"\.(exe|dll)$", "", a.image, flags=re.I) + ".rtti.json"
    same_path = os.path.normcase(os.path.realpath(a.image)) == os.path.normcase(os.path.realpath(path))
    try:
        same_path = same_path or os.path.samefile(a.image, path)
    except OSError:
        pass  # A missing or inaccessible path is handled by the normal read/write error path.
    if same_path:
        ap.error("output must not overwrite the source image")
    try:
        with open(a.image, "rb") as f:
            data = f.read()
        scanned = scan_pe64_rtti(data)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    out = {"image": a.image, **scanned}
    try:
        write_report(path, json.dumps(out, indent=1))
    except OSError as exc:
        print(f"ERROR: {path}: {exc.strerror or exc}", file=sys.stderr)
        return 1
    print(f"{a.image}: {out['type_descriptors']} type descriptors, {len(out['classes'])} classes with RTTI, {len(out['vtables'])} vtables -> {path}")
    for c in out["classes"][:5]:
        print("  ", c["name"], "<-", ", ".join(c["bases"][:4]))
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
