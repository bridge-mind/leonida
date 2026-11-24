"""Unit tests for tools/gdk_runtime_calls.py and tools/gen_gdk_runtime_classes.py.

The image is built here: a .text section of hand-assembled xgameruntime.lib-style thunks (the CFG form that loads the
slot into rax and calls a dispatch routine, also with the slot copied into rax from another register, `call
[rax+disp8]`, and `call [r10+disp32]` with a stack argument), a .rdata section with the class and interface ids, and
.pdata function bounds. The ids and method names come from the
committed tools/gdk_runtime_classes.json; the docs tree for the generator is a made-up one. No game files.

    python -I tests/test_gdk_runtime_calls.py
"""
import contextlib
import io
import json
import os
import shutil
import struct
import sys
import tempfile
import unittest
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS_DIR = os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import gdk_runtime_calls as grc  # noqa: E402
import gen_gdk_runtime_classes as gen  # noqa: E402

THREADING = "073b7dcb-1fcf-4030-94be-e3c9eb623428"      # XThreadingImpl, also IXThreadingImpl
NETWORKING = "37e56907-2f10-41e8-b72f-36edb185331a"     # XNetworkingImpl
NETWORKING2 = "bf2346b2-39af-4658-b5ea-44713c7e83b3"    # IXNetworkingImpl2
PACKAGE = "af406016-e850-4aa8-a88d-2f3dcb9dac7e"        # XPackageImpl
PACKAGE3 = "e2a4734b-2f4a-456d-aa8f-d065e04fb209"       # IXPackageImpl3
STORE = "0dd112ac-7c24-448c-b92b-3960fb5bd30c"          # XStoreImpl: present, no thunk
UNKNOWN_CLASS = "11111111-2222-3333-4444-555555555555"
UNKNOWN_IFACE = "66666666-7777-8888-9999-aaaaaaaaaaaa"

IMAGE_BASE = 0x180000000
TEXT_RVA, TEXT_RAW, TEXT_SIZE = 0x1000, 0x400, 0x1000
RDATA_RVA, RDATA_RAW, RDATA_SIZE = 0x3000, 0x1400, 0x1000
GUIDS = {THREADING: 0x3000, NETWORKING: 0x3010, NETWORKING2: 0x3020, PACKAGE: 0x3030, PACKAGE3: 0x3040,
         STORE: 0x3050, UNKNOWN_CLASS: 0x3060, UNKNOWN_IFACE: 0x3070}
CACHE_RVA = 0x3090          # zeroed .rdata: a global a thunk passes in r8, not an id
PDATA_RVA = 0x3800
LABELS = {}                 # the assembled functions' RVAs, filled by build_image()
SERVED_SOURCE = ("constexpr GUID kThreadingClass = {0x073b7dcb, 0x1fcf, 0x4030, "
                 "{0x94, 0xbe, 0xe3, 0xc9, 0xeb, 0x62, 0x34, 0x28}};\n")


class Asm:
    """Bytes at an RVA, with labels and rel8/rel32 fields resolved at the end."""

    def __init__(self, base: int):
        self.base, self.buf, self.labels, self.fixups = base, bytearray(), {}, []

    def here(self) -> int:
        return self.base + len(self.buf)

    def label(self, name: str) -> None:
        self.labels[name] = self.here()

    def raw(self, text: str) -> None:
        self.buf += bytes.fromhex(text)

    def rel(self, text: str, target, size: int = 4) -> None:
        """An instruction ending in a displacement to `target` (a label or an RVA), counted from its end."""
        self.raw(text)
        self.fixups.append((len(self.buf), size, target))
        self.buf += bytes(size)

    def finish(self) -> bytes:
        for pos, size, target in self.fixups:
            t = self.labels[target] if isinstance(target, str) else target
            self.buf[pos:pos + size] = struct.pack("<b" if size == 1 else "<i", t - (self.base + pos + size))
        return bytes(self.buf)


def query(a: Asm, out_disp: int, class_id: str, iface_id: str | None, done: str) -> None:
    """lea r8,[rsp+out]; the id leas; call helper; test eax,eax; js done. iface_id None: mov rdx,rcx."""
    a.raw(f"4C8D4424{out_disp:02X}")                              # lea r8,[rsp+out]
    if iface_id:
        a.rel("488D15", GUIDS[iface_id])                         # lea rdx,[iid]
    a.rel("488D0D", GUIDS[class_id])                             # lea rcx,[clsid]
    if not iface_id:
        a.raw("488BD1")                                          # mov rdx,rcx
    a.rel("E8", "helper")
    a.raw("85C0")                                                # test eax,eax
    a.rel("78", done, 1)                                         # js done


def build_code(a: Asm) -> list[tuple[int, int]]:
    """The functions; returns their (begin, end) for .pdata."""
    funcs = []

    def begin(name):
        a.label(name)
        return a.here()

    def end(start):
        funcs.append((start, a.here()))
        a.raw("CC" * (-len(a.buf) % 16))

    s = begin("entry")
    a.raw("4883EC28")                                            # sub rsp,28h
    a.rel("E8", "thunk1")
    a.rel("E8", "thunk2")
    a.rel("E8", "stub3")                                         # through an incremental-link style jmp stub
    a.raw("4883C428C3")                                          # add rsp,28h; ret
    end(s)

    s = begin("helper")
    a.raw("33C0C3")                                              # xor eax,eax; ret
    end(s)
    s = begin("dispatch")
    a.raw("FFE0")                                                # jmp rax (the CFG dispatch routine)
    end(s)

    # thunk1, CFG form: XThreadingImpl (class id as interface id), slot 12 = XTaskQueueCreate(mode, mode, &queue)
    s = begin("thunk1")
    a.raw("4883EC38")
    query(a, 0x28, THREADING, None, "done1")
    a.raw("488B4C2428" "488B01")                                 # mov rcx,[rsp+28h]; mov rax,[rcx]
    a.raw("BA01000000" "41B802000000" "4C8D4C2430")              # mov edx,1; mov r8d,2; lea r9,[rsp+30h]
    a.raw("488B4060")                                            # mov rax,[rax+60h]
    a.rel("E8", "dispatch")
    a.raw("488B4C2428" "488B01" "488B4010")                      # Release: slot 2
    a.rel("E8", "dispatch")
    a.label("done1")
    a.raw("4883C438C3")
    end(s)

    # thunk2, call [rax+disp8]: XNetworkingImpl as IXNetworkingImpl2, slot 15 = XNetworkingGetConnectivityHint(&hint)
    s = begin("thunk2")
    a.raw("53" "4883EC30" "488BD9")                              # push rbx; sub rsp,30h; mov rbx,rcx
    query(a, 0x28, NETWORKING, NETWORKING2, "done2")
    a.raw("488B4C2428" "488B01" "488BD3")                        # mov rcx,[rsp+28h]; mov rax,[rcx]; mov rdx,rbx
    a.raw("FF5078")                                              # call [rax+78h]
    a.raw("488B4C2428" "488B01" "FF5010")                        # Release
    a.label("done2")
    a.raw("4883C4305BC3")
    end(s)

    # thunk3, call [r10+disp32] and a stack argument: XPackageImpl as IXPackageImpl3, slot 40 =
    # XPackageRegisterPackageInstalled(queue, context, callback, &token)
    s = begin("thunk3")
    a.raw("4883EC48")
    query(a, 0x38, PACKAGE, PACKAGE3, "done3")
    a.raw("488B4C2438" "4C8B11")                                 # mov rcx,[rsp+38h]; mov r10,[rcx]
    a.raw("33D2" "4533C0" "4533C9")                              # xor edx,edx; xor r8d,r8d; xor r9d,r9d
    a.raw("48C744242000000000")                                  # mov qword ptr [rsp+20h],0
    a.raw("41FF9240010000")                                      # call [r10+140h]
    a.raw("488B4C2438" "4C8B11" "41FF5210")                      # Release through r10
    a.label("done3")
    a.raw("4883C448C3")
    end(s)

    # thunk4: ids the table doesn't know, slot 5, no arguments; nothing calls it. It calls thunkA first, as the
    # library's thunks call the XThreadAssertNotTimeSensitive thunk.
    s = begin("thunk4")
    a.raw("4883EC38")
    a.rel("E8", "thunkA")
    query(a, 0x28, UNKNOWN_CLASS, UNKNOWN_IFACE, "done4")
    a.raw("488B4C2428" "488B01" "488B4028")                      # mov rax,[rax+28h]
    a.rel("E8", "dispatch")
    a.raw("488B4C2428" "488B01" "488B4010")
    a.rel("E8", "dispatch")
    a.label("done4")
    a.raw("4883C438C3")
    end(s)

    # thunkA: XThreadingImpl slot 29 = XThreadAssertNotTimeSensitive, the slot load as disp32; only thunk4 calls it.
    s = begin("thunkA")
    a.raw("4883EC38")
    query(a, 0x28, THREADING, THREADING, "doneA")
    a.raw("488B4C2428" "488B01" "488B80E8000000")                # mov rax,[rax+0E8h]
    a.rel("E8", "dispatch")
    a.raw("488B4C2428" "488B01" "488B4010")
    a.rel("E8", "dispatch")
    a.label("doneA")
    a.raw("4883C438C3")
    end(s)

    # thunkB: XThreadingImpl slot 16 = XTaskQueueDispatch(queue, port, timeout). rdx holds no id, r8 a global (the
    # object's cache), and the method pointer reaches rax through r11.
    s = begin("thunkB")
    a.raw("4883EC38")
    a.rel("4C8D05", CACHE_RVA)                                   # lea r8,[cache]
    a.rel("488D0D", GUIDS[THREADING])                            # lea rcx,[clsid]
    a.raw("33D2")                                                # xor edx,edx
    a.rel("E8", "helper")
    a.raw("85C0")
    a.rel("78", "doneB", 1)
    a.rel("488B0D", CACHE_RVA)                                   # mov rcx,[cache]
    a.raw("488B01" "4C8B9880000000")                             # mov rax,[rcx]; mov r11,[rax+80h]
    a.raw("33D2" "41B801000000" "4183C9FF")                      # xor edx,edx; mov r8d,1; or r9d,-1
    a.raw("498BC3")                                              # mov rax,r11
    a.rel("E8", "dispatch")
    a.rel("488B0D", CACHE_RVA)
    a.raw("488B01" "488B4010")                                   # Release
    a.rel("E8", "dispatch")
    a.label("doneB")
    a.raw("4883C438C3")
    end(s)

    # thunkC: a helper call whose ids come from a register, not a RIP-relative lea: listed, not followed.
    s = begin("thunkC")
    a.raw("4883EC38" "4C8D442428" "488BCB" "488BD3")              # lea r8,[rsp+28h]; mov rcx,rbx; mov rdx,rbx
    a.label("callC")
    a.rel("E8", "helper")
    a.raw("85C0")
    a.rel("78", "doneC", 1)
    a.raw("488B4C2428" "488B01" "488B4018")                      # slot 3
    a.rel("E8", "dispatch")
    a.raw("488B4C2428" "488B01" "488B4010")
    a.rel("E8", "dispatch")
    a.label("doneC")
    a.raw("4883C438C3")
    end(s)

    a.label("stub3")                                           # outside every function, like an ILT entry
    a.rel("E9", "thunk3")
    return funcs


def build_image(with_pdata: bool = True) -> bytes:
    opt_size = 240
    buf = bytearray(RDATA_RAW + RDATA_SIZE)
    buf[0:2] = b"MZ"
    struct.pack_into("<I", buf, 0x3C, 0x40)
    buf[0x40:0x44] = b"PE\0\0"
    struct.pack_into("<HHIIIHH", buf, 0x44, 0x8664, 2, 0, 0, 0, opt_size, 0x2022)
    opt = 0x58
    struct.pack_into("<H", buf, opt, 0x20B)
    struct.pack_into("<I", buf, opt + 16, TEXT_RVA)
    struct.pack_into("<Q", buf, opt + 24, IMAGE_BASE)
    struct.pack_into("<II", buf, opt + 32, 0x1000, 0x200)
    struct.pack_into("<II", buf, opt + 56, 0x4000, TEXT_RAW)        # SizeOfImage, SizeOfHeaders
    struct.pack_into("<H", buf, opt + 68, 2)
    struct.pack_into("<I", buf, opt + 108, 16)

    a = Asm(TEXT_RVA)
    funcs = build_code(a)
    code = a.finish()
    LABELS.update(a.labels)
    assert len(code) <= TEXT_SIZE
    buf[TEXT_RAW:TEXT_RAW + len(code)] = code
    for guid, rva in GUIDS.items():
        at = RDATA_RAW + rva - RDATA_RVA
        buf[at:at + 16] = uuid.UUID(guid).bytes_le
    if with_pdata:
        at = RDATA_RAW + PDATA_RVA - RDATA_RVA
        for i, (b, e) in enumerate(funcs):
            struct.pack_into("<III", buf, at + i * 12, b, e, 0x3F00)
        struct.pack_into("<II", buf, opt + 112 + 3 * 8, PDATA_RVA, len(funcs) * 12)

    sections = opt + opt_size
    for i, (name, rva, size, raw, chars) in enumerate(((b".text", TEXT_RVA, TEXT_SIZE, TEXT_RAW, 0x60000020),
                                                       (b".rdata", RDATA_RVA, RDATA_SIZE, RDATA_RAW, 0x40000040))):
        at = sections + i * 40
        buf[at:at + 8] = name.ljust(8, b"\0")
        struct.pack_into("<IIII", buf, at + 8, size, rva, size, raw)
        struct.pack_into("<I", buf, at + 36, chars)
    return bytes(buf)


def by_class(report: dict) -> dict:
    return {c["class"] or c["class_id"]: c for c in report["classes"]}


def slots_of(cls: dict) -> dict:
    return {(i["interface"], s["slot"]): s for i in cls["interfaces"] for s in i["slots"]}


class ImageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.table = grc.RuntimeTable.load(grc.DEFAULT_TABLE)
        cls.path = cls.write("thunks.dll", build_image())
        cls.served = grc.served_classes(SERVED_SOURCE)
        cls.report = grc.report(cls.path, cls.table, cls.served)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    @classmethod
    def write(cls, name: str, data: bytes) -> str:
        path = os.path.join(cls.tmp, name)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def test_helper_and_thunk_count(self):
        self.assertEqual(self.report["helpers"], [f"rel {LABELS['helper']:#x}"])
        self.assertEqual(self.report["thunks"], 6)
        # thunkC passes its ids in registers: listed, with no class read.
        self.assertEqual(self.report["helper_calls_without_slot_call"],
                         [{"call": f"{LABELS['callC']:#x}", "class": None, "class_id": None,
                           "interface_id": None}])

    def test_classes_slots_and_names(self):
        classes = by_class(self.report)
        self.assertEqual(set(classes), {"XThreadingImpl", "XNetworkingImpl", "XPackageImpl", UNKNOWN_CLASS})
        threading = slots_of(classes["XThreadingImpl"])
        self.assertEqual(threading[("XThreadingImpl/IXThreadingImpl", 12)]["method"], "XTaskQueueCreate")
        self.assertEqual(threading[("XThreadingImpl/IXThreadingImpl", 12)]["args_estimate"], [3])
        self.assertEqual(threading[("XThreadingImpl/IXThreadingImpl", 29)]["method"], "XThreadAssertNotTimeSensitive")
        self.assertEqual(threading[("XThreadingImpl/IXThreadingImpl", 29)]["args_estimate"], [0])
        # thunkB: no id in rdx, so the interface is the class id (not the global r8 points at); slot through r11.
        self.assertEqual([i["interface_id"] for i in classes["XThreadingImpl"]["interfaces"]], [THREADING])
        dispatch = threading[("XThreadingImpl/IXThreadingImpl", 16)]
        self.assertEqual((dispatch["method"], dispatch["args_estimate"]), ("XTaskQueueDispatch", [3]))

        networking = classes["XNetworkingImpl"]
        self.assertEqual([i["interface_id"] for i in networking["interfaces"]], [NETWORKING2])
        hint = slots_of(networking)[("IXNetworkingImpl2", 15)]
        self.assertEqual((hint["method"], hint["args_estimate"]), ("XNetworkingGetConnectivityHint", [1]))

        package = slots_of(classes["XPackageImpl"])[("IXPackageImpl3", 40)]
        self.assertEqual((package["method"], package["args_estimate"]), ("XPackageRegisterPackageInstalled", [4]))

    def test_unknown_ids_are_reported(self):
        unknown = by_class(self.report)[UNKNOWN_CLASS]
        self.assertIsNone(unknown["class"])
        self.assertEqual(unknown["interfaces"][0]["interface"], "unknown")
        self.assertEqual(unknown["interfaces"][0]["interface_id"], UNKNOWN_IFACE)
        slot = unknown["interfaces"][0]["slots"][0]
        self.assertEqual((slot["slot"], slot["method"], slot["args_estimate"]), (5, None, [0]))

    def test_served_flag(self):
        classes = by_class(self.report)
        self.assertIs(classes["XThreadingImpl"]["served"], True)
        self.assertIs(classes["XPackageImpl"]["served"], False)
        unserved = grc.report(self.path, self.table, None)
        self.assertIsNone(by_class(unserved)["XThreadingImpl"]["served"])

    def test_references(self):
        classes = by_class(self.report)
        threading = slots_of(classes["XThreadingImpl"])
        self.assertEqual(threading[("XThreadingImpl/IXThreadingImpl", 12)]["referenced"], 1)
        # thunkA is called only by thunk4, which nothing calls.
        self.assertEqual(threading[("XThreadingImpl/IXThreadingImpl", 29)]["referenced"], 0)
        self.assertEqual(threading[("XThreadingImpl/IXThreadingImpl", 16)]["referenced"], 0)
        self.assertEqual(classes["XPackageImpl"]["referenced_thunks"], 1)       # through the jmp stub
        self.assertEqual(classes[UNKNOWN_CLASS]["referenced_thunks"], 0)
        self.assertEqual(self.report["referenced_thunks"], 3)

    def test_unreached_known_ids(self):
        unreached = {u["class"]: u for u in self.report["unreached_classes"]}
        self.assertEqual(set(unreached), {"XStoreImpl"})
        self.assertEqual(unreached["XStoreImpl"]["ids"][0]["id"], STORE)

    def test_without_pdata_the_byte_window_finds_the_same_slots(self):
        path = self.write("no_pdata.dll", build_image(with_pdata=False))
        r = grc.report(path, self.table, self.served)
        self.assertEqual(r["pdata_functions"], 0)
        self.assertEqual({(c["class"], s["slot"]) for c in r["classes"] for i in c["interfaces"] for s in i["slots"]},
                         {(c["class"], s["slot"]) for c in self.report["classes"] for i in c["interfaces"]
                          for s in i["slots"]})
        self.assertIsNone(r["referenced_thunks"])

    def test_main_writes_text_and_json_with_lf(self):
        out = os.path.join(self.tmp, "out")
        source = self.write("xgameruntime.cpp", SERVED_SOURCE.encode())
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            rc = grc.main([self.path, "--out", out, "--runtime-source", source])
        self.assertEqual(rc, 0)
        self.assertIn("not-served-referenced=XNetworkingImpl,XPackageImpl\n", stdout.getvalue())
        with open(os.path.join(out, "thunks.dll.gdk_calls.json"), "rb") as f:
            raw = f.read()
        self.assertNotIn(b"\r\n", raw)
        self.assertEqual(json.loads(raw)["thunks"], 6)
        with open(os.path.join(out, "thunks.dll.gdk_calls.txt"), "rb") as f:
            text = f.read().decode("utf-8")
        self.assertNotIn("\r\n", text)
        self.assertIn("slot  12  XTaskQueueCreate", text)
        self.assertIn("XPackageImpl as IXPackageImpl3: 40; referenced: 40", text)
        self.assertIn("class id not read; interface not read", text)

    def test_main_names_repeated_inputs_and_refuses_a_file_as_out(self):
        out = os.path.join(self.tmp, "out-twice")
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            self.assertEqual(grc.main([self.path, self.path, "--out", out]), 0)
            self.assertEqual(grc.main([self.path, "--out", self.path]), 1)
        self.assertTrue(os.path.isfile(os.path.join(out, "thunks.dll-2.gdk_calls.json")))
        self.assertIn("report written as thunks.dll-2.gdk_calls.txt/.json", stderr.getvalue())
        self.assertIn("path is a file, not a directory", stderr.getvalue())

    def test_main_refuses_out_that_overwrites_input(self):
        aliased_dir = os.path.join(self.tmp, "alias-out")
        os.makedirs(aliased_dir, exist_ok=True)
        aliased = os.path.join(aliased_dir, "thunks.dll.gdk_calls.json")
        shutil.copyfile(grc.DEFAULT_TABLE, aliased)
        with open(aliased, "rb") as f:
            original = f.read()
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            rc = grc.main([self.path, "--table", aliased, "--out", aliased_dir])
        self.assertEqual(rc, 1)
        self.assertIn("would overwrite input file", stderr.getvalue())
        with open(aliased, "rb") as f:
            self.assertEqual(f.read(), original)
        self.assertFalse(os.path.exists(os.path.join(aliased_dir, "thunks.dll.gdk_calls.txt")))

    def test_the_default_source_is_every_runtime_shim(self):
        # The classes xgameruntime.cpp hands on to its sibling files count as served: XUserImpl, XGameSaveImpl and
        # XStoreImpl live in xgameruntime_user.cpp, _gamesave.cpp and _store.cpp.
        served = grc.served_classes(grc.read_runtime_source(grc.DEFAULT_RUNTIME_SOURCE))
        names = set(served.values())
        for name in ("kThreadingClass", "kUserClass", "kSaveClass", "kStoreClass"):
            self.assertIn(name, names)
        self.assertEqual(served.get("704c3f58-e629-4cc2-b197-30511b996fe2"), "kSaveClass")
        # xgameruntime.cpp serves XGameRuntimeFeatureImpl; the ids its feature table only names (k...ClassId) don't count.
        self.assertEqual(served.get("8836fe87-edb9-4fe3-8dad-05f0d2cd5b40"), "kGameRuntimeFeatureClass")
        self.assertEqual(served.get("1b339674-328d-4283-a200-3171f18d3639"), "kLauncherClass")   # XLauncherImpl, xgameruntime_offline.cpp
        self.assertNotIn("a4f1aee2-4bf1-4485-b008-a7c26d52ac27", served)   # XAppCaptureImpl
        empty = os.path.join(self.tmp, "no-shims")
        os.makedirs(empty, exist_ok=True)
        with self.assertRaises(OSError):
            grc.read_runtime_source(empty)

    def test_main_json_to_stdout(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            rc = grc.main([self.path, "--json", "--runtime-source", os.path.join(self.tmp, "missing.cpp")])
        self.assertEqual(rc, 0)
        self.assertIn("'served' is unknown", stderr.getvalue())
        doc = json.loads(stdout.getvalue())
        self.assertEqual(doc["images"][0]["file"], "thunks.dll")
        self.assertIsNone(by_class(doc["images"][0])["XThreadingImpl"]["served"])

    def test_truncated_image(self):
        data = build_image()
        cut = self.write("cut.dll", data[:TEXT_RAW + 0x100])
        r = grc.report(cut, self.table, self.served)
        self.assertEqual(r["cut_off_sections"], [".text", ".rdata"])
        self.assertEqual(r["thunks"], 0)                 # the ids are gone with .rdata
        stderr, stdout = io.StringIO(), io.StringIO()
        with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(stdout):
            self.assertEqual(grc.main([cut]), 1)
            self.assertEqual(grc.main([self.write("head.dll", data[:0x100])]), 1)
            self.assertEqual(grc.main([self.write("text.txt", b"not a PE image")]), 1)
        self.assertIn("incomplete", stderr.getvalue())
        self.assertIn("not a readable PE image", stderr.getvalue())


class DecoderTest(unittest.TestCase):
    def test_lengths(self):
        cases = {
            "488D0D00000000": 7,          # lea rcx,[rip+0]
            "41FF9240010000": 7,          # call [r10+140h]
            "48C744242000000000": 9,      # mov qword ptr [rsp+20h],0
            "E800000000": 5,              # call rel32
            "660F1F440000": 6,            # nop word ptr [rax+rax]
            "C5F877": 3,                  # vzeroupper
            "48B80000000000000000": 10,   # mov rax,imm64
            "F6C101": 3,                  # test cl,1
            "F7C100000000": 6,            # test ecx,imm32
            "66F7C10000": 5,              # test cx,imm16
            "6648F7C100000000": 8,        # test rcx,imm32 (REX.W overrides 66h)
            "66C7C10000": 5,              # mov cx,imm16
            "6648C7C100000000": 8,        # mov rcx,imm32 (REX.W overrides 66h)
            "4C8B11": 3,                  # mov r10,[rcx]
            "0F8400000000": 6,            # je rel32
        }
        for text, size in cases.items():
            with self.subTest(text=text):
                insn = grc.decode(bytes.fromhex(text), 0, 0x1000)
                self.assertIsNotNone(insn)
                self.assertEqual(insn.size, size)

    def test_operands(self):
        lea = grc.decode(bytes.fromhex("488D1510000000"), 0, 0x1000)
        self.assertEqual((lea.reg, lea.base, lea.target), (grc.RDX, grc.RIP, 0x1017))
        call = grc.decode(bytes.fromhex("41FF9240010000"), 0, 0)
        self.assertEqual((call.base, call.disp, call.is_call()), (10, 0x140, True))
        self.assertIsNone(grc.decode(bytes.fromhex("48"), 0, 0))          # cut off after a REX prefix


class TableTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = grc.RuntimeTable.load(grc.DEFAULT_TABLE)

    def test_second_interface_starts_at_slot_3(self):
        self.assertEqual(self.table.method("IXUserGamertagImpl", 3), ("XUserGetGamertag", None))
        self.assertEqual(self.table.interface_for("cef4fac0-7676-4a94-a119-4c43f9eb5b74"), "IXUserGamertagImpl")

    def test_unknown_entries_keep_their_slot(self):
        self.assertEqual(self.table.method("IXPackageImpl3", 20), (None, None))
        self.assertEqual(self.table.method("IXPackageImpl3", 22), ("XPackageUnregisterPackageInstalled", None))
        self.assertEqual(self.table.method("IXPackageImpl3", 40), ("XPackageRegisterPackageInstalled", None))

    def test_interfaces_without_layout(self):
        names, no_layout = self.table.slot_table("IXSystemImpl5")
        self.assertEqual(no_layout, ["IXSystemImpl2", "IXSystemImpl5"])
        self.assertEqual(self.table.method("IXSystemImpl5", 8), ("XSystemAllowFullDownloadBandwidth", None))
        self.assertEqual(self.table.method("IXSystemImpl5", 9), (None, None))
        self.assertIsNone(self.table.interfaces["IXSystemImpl5"]["methods"])

    def test_class_id_used_as_interface_id(self):
        self.assertEqual(self.table.interface_for("03f0fe74-fdd9-4e5c-b630-f9339c47acc5"), "IXDisplayImpl")
        # XSystemImpl's class id is also IXSystemImpl's; a slot past it is named from the class's interface.
        self.assertEqual(self.table.interface_for("e349bd1a-fc20-4e40-b99c-4178cc6b409f"), "IXSystemImpl")
        self.assertEqual(self.table.method("IXSystemImpl", 6), ("XSystemHandleTrack", "IXSystemImpl5"))

    def test_served_classes(self):
        other = ("constexpr GUID kOther = {0x1, 0x2, 0x3, {0, 0, 0, 0, 0, 0, 0, 0}};\n"         # not a k...Class
                 "constexpr GUID kBadClass = {0x1, 0x2, 0x3, {0x100, 0, 0, 0, 0, 0, 0, 0}};\n")   # not a GUID
        self.assertEqual(grc.served_classes(SERVED_SOURCE + other), {THREADING: "kThreadingClass"})

    def test_ranges(self):
        self.assertEqual(grc.ranges([3, 4, 5, 9, 12, 13]), "3-5, 9, 12-13")


class GeneratorTest(unittest.TestCase):
    PAGES = {
        "README.md": "# XFooImpl\n\nClass Id: `aaaaaaaa-0000-0000-0000-000000000001`\n\n"
                     "Implements: [IXFooImpl3](IXFooImpl3.md), [IXFooExtraImpl](IXFooExtraImpl.md)\n",
        "IXFooImpl.md": "# IXFooImpl\n\nInterface Id: `aaaaaaaa-0000-0000-0000-000000000001`\n\nInherits: IUnknown\n\n"
                        "Layout:\n- [XFooFirst](https://example.invalid/first)\n- *unknown*\n"
                        "- XFooPrivate (no docs)\n- *unknown* (perhaps an older [XFooFirst](https://example.invalid))\n"
                        "\nA closing paragraph.\n",
        "IXFooImpl2.md": "# IXFooImpl2\n\nInterface Id: `aaaaaaaa-0000-0000-0000-000000000002`\n\n"
                         "Inherits: [IXFooImpl](IXFooImpl.md)\n\nNo layout is known for this one.\n",
        "IXFooImpl3.md": "# IXFooImpl3\n\nInterface Id: `AAAAAAAA-0000-0000-0000-000000000003`\n\n"
                         "Inherits: [IXFooImpl2](IXFooImpl2.md)\n\nLayout:\n- [XFooThird](https://example.invalid)\n",
        "IXFooExtraImpl.md": "# IXFooExtraImpl\n\nInterface Id: `aaaaaaaa-0000-0000-0000-0000000000ee`\n\n"
                             "Inherits: IUnknown\n\nLayout:\n- [XFooExtra](https://example.invalid)\n",
    }

    def setUp(self):
        self.root = tempfile.mkdtemp()
        folder = os.path.join(self.root, "COM", "XFooImpl")
        os.makedirs(folder)
        for name, text in self.PAGES.items():
            with open(os.path.join(folder, name), "w", encoding="utf-8", newline="\n") as f:
                f.write(text)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_parse(self):
        doc = gen.build(self.root)
        self.assertEqual(doc["classes"]["XFooImpl"],
                         {"id": "aaaaaaaa-0000-0000-0000-000000000001", "implements": ["IXFooImpl3", "IXFooExtraImpl"]})
        ifaces = doc["interfaces"]
        self.assertEqual(ifaces["IXFooImpl"]["methods"], ["XFooFirst", None, "XFooPrivate", None])
        self.assertIsNone(ifaces["IXFooImpl2"]["methods"])
        self.assertEqual(ifaces["IXFooImpl3"]["id"], "aaaaaaaa-0000-0000-0000-000000000003")
        self.assertEqual(ifaces["IXFooImpl3"]["inherits"], "IXFooImpl2")
        self.assertIsNone(ifaces["IXFooExtraImpl"]["inherits"])
        self.assertEqual(gen.check(doc["classes"], ifaces), [])
        table = grc.RuntimeTable(doc)
        self.assertEqual(table.method("IXFooImpl3", 7), ("XFooThird", None))
        self.assertEqual(table.method("IXFooExtraImpl", 3), ("XFooExtra", None))
        self.assertEqual(table.slot_table("IXFooImpl3")[1], ["IXFooImpl2"])

    def test_main_writes_lf_json_and_reports_broken_links(self):
        os.remove(os.path.join(self.root, "COM", "XFooImpl", "IXFooExtraImpl.md"))
        out = os.path.join(self.root, "table.json")
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(gen.main([self.root, "--out", out]), 0)
        self.assertIn("IXFooExtraImpl, which has no page", stderr.getvalue())
        with open(out, "rb") as f:
            raw = f.read()
        self.assertNotIn(b"\r\n", raw)
        self.assertIn("GPL-3.0", json.loads(raw)["licence"])
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(gen.main([os.path.join(self.root, "missing"), "--out", out]), 1)


if __name__ == "__main__":
    unittest.main()
