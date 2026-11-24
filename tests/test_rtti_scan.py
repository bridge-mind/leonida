"""Tests for tools/rtti_scan.py covering synthetic MSVC RTTI structures.

    python -I tests/test_rtti_scan.py

Tests synthetic MSVC TypeDescriptors, CompleteObjectLocators, ClassHierarchyDescriptors,
BaseClassDescriptors, and vtable slot recovery against synthetic PE images.
No game data or proprietary binaries are used.
"""
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import rtti_scan

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "rtti_scan.py")


def make_synthetic_pe(text_bytes: bytes = b"", rdata_bytes: bytes = b"", image_base: int = 0x140000000) -> bytes:
    """Build a minimal valid 64-bit PE image in memory with .text and .rdata sections."""
    sec_align = 0x1000
    file_align = 0x200

    def align_up(x: int, a: int) -> int:
        return (x + a - 1) & ~(a - 1)

    text_len = max(align_up(len(text_bytes), sec_align), sec_align)
    text_data = text_bytes.ljust(text_len, b"\xc3")

    rdata_len = max(align_up(len(rdata_bytes), sec_align), sec_align)
    rdata_data = rdata_bytes.ljust(rdata_len, b"\x00")

    text_rva = 0x1000
    rdata_rva = text_rva + text_len
    size_of_image = rdata_rva + rdata_len

    dos = b"MZ" + b"\x00" * 0x3A + struct.pack("<I", 0x80)
    dos = dos.ljust(0x80, b"\x00")
    pe_sig = b"PE\x00\x00"
    file_hdr = struct.pack("<HHIIIHH", 0x8664, 2, 0, 0, 0, 240, 0x0022)
    opt_hdr = struct.pack(
        "<HBBIIIIIQIIHHHHHHIIIIHHQQQQII",
        0x020B, 14, 0, text_len, rdata_len, 0, text_rva, text_rva,
        image_base, sec_align, file_align, 6, 0, 0, 0, 6, 0, 0,
        size_of_image, 0x400, 0, 3, 0,
        0x100000, 0x1000, 0x100000, 0x1000, 0, 16
    ) + b"\x00" * (16 * 8)

    sec_text = struct.pack(
        "<8sIIIIIIHHI",
        b".text\x00\x00\x00", text_len, text_rva, text_len, 0x400,
        0, 0, 0, 0, 0x60000020
    )
    sec_rdata = struct.pack(
        "<8sIIIIIIHHI",
        b".rdata\x00\x00", rdata_len, rdata_rva, rdata_len, 0x400 + text_len,
        0, 0, 0, 0, 0x40000040
    )
    headers = (dos + pe_sig + file_hdr + opt_hdr + sec_text + sec_rdata).ljust(0x400, b"\x00")
    return headers + text_data + rdata_data


class SimpleNameTest(unittest.TestCase):
    def test_class_in_single_namespace(self):
        self.assertEqual(rtti_scan.simple_name(".?AVFoo@bar@@"), "bar::Foo")

    def test_class_in_nested_namespaces(self):
        self.assertEqual(rtti_scan.simple_name(".?AVRenderer@Graphics@Engine@@"), "Engine::Graphics::Renderer")

    def test_top_level_class(self):
        self.assertEqual(rtti_scan.simple_name(".?AVRootWidget@@"), "RootWidget")

    def test_struct_prefix_u(self):
        self.assertEqual(rtti_scan.simple_name(".?AUConfig@Settings@Core@@"), "Core::Settings::Config")

    def test_union_prefix_w(self):
        self.assertEqual(rtti_scan.simple_name(".?AWVariantPayload@Data@@"), "Data::VariantPayload")

    def test_non_mangled_strings_returned_unmodified(self):
        self.assertEqual(rtti_scan.simple_name("RegularClassName"), "RegularClassName")
        self.assertEqual(rtti_scan.simple_name(".?AVIncompleteString"), ".?AVIncompleteString")
        self.assertEqual(rtti_scan.simple_name(""), "")

    # The expected names below are what dbghelp's UnDecorateSymbolName prints for these hand-written names, without
    # the class/struct/enum keywords, `__cdecl`, `__ptr64` and a local class's function signature.
    def test_template_arguments_are_not_scopes(self):
        for mangled, name in (
            (".?AV?$Pool@H@mem@@", "mem::Pool<int>"),      # was reported as mem::H::?$Pool
            (".?AV?$Map@_KPEBD@util@@", "util::Map<unsigned __int64, char const *>"),
            (".?AV?$Flags@W4Mode@gfx@@@gfx@@", "gfx::Flags<gfx::Mode>"),
            (".?AV?$Fixed@$0BA@$0?0$00@math@@", "math::Fixed<16, -1, 1>"),
            (".?AV?$Tuple@$$V@core@@", "core::Tuple<>"),
            (".?AV?$Tuple@HN$$QEAM@core@@", "core::Tuple<int, double, float &&>"),
            (".?AVInner@?$Outer@_N@core@@", "core::Outer<bool>::Inner"),
        ):
            with self.subTest(mangled=mangled):
                self.assertEqual(rtti_scan.simple_name(mangled), name)

    def test_back_references(self):
        # Inside a template the digits count that template's own names: 0 is Pair, 1 Node, 2 scene.
        self.assertEqual(rtti_scan.simple_name(".?AV?$Pair@VNode@scene@@VEdge@2@@scene@@"),
                         "scene::Pair<scene::Node, scene::Edge>")
        # A template that was named once is one back-reference: 2 is List<int>.
        self.assertEqual(rtti_scan.simple_name(".?AV?$Wrap@VIterator@?$List@H@core@@V123@@core@@"),
                         "core::Wrap<core::List<int>::Iterator, core::List<int>::Iterator>")

    def test_pointer_reference_array_and_function_arguments(self):
        for mangled, name in (
            (".?AV?$Callback@$$A6AXAEBUEvent@ui@@H@Z@ui@@", "ui::Callback<void (ui::Event const &, int)>"),
            (".?AV?$Sink@P6AXPEAX0@_E@io@@", "io::Sink<void (*)(void *, void *) noexcept>"),
            (".?AV?$Span@AEAY03$$CBH@io@@", "io::Span<int const (&)[4]>"),
            (".?AV?$Ref@AEBQEAUNode@scene@@@scene@@", "scene::Ref<scene::Node * const &>"),
            # A pointer to a function pointer or to an array goes inside the brackets, not after them.
            (".?AV?$Cb@PEAP6AXH@Z@lab@@", "lab::Cb<void (**)(int)>"),
            (".?AV?$Cb@QEAP6AXH@Z@lab@@", "lab::Cb<void (** const)(int)>"),
            (".?AV?$Cb@P6AP6AXH@ZXZ@lab@@", "lab::Cb<void (*(*)(void))(int)>"),
            (".?AV?$Cb@AEAY03P6AXH@Z@lab@@", "lab::Cb<void (*(&)[4])(int)>"),
            (".?AV?$Cb@PEAY03$$CBH@lab@@", "lab::Cb<int const (*)[4]>"),
            (".?AV?$Arr@$$BY01H@lab@@", "lab::Arr<int [2]>"),
        ):
            with self.subTest(mangled=mangled):
                self.assertEqual(rtti_scan.simple_name(mangled), name)

    def test_anonymous_namespace_and_local_classes(self):
        for mangled, name in (
            (".?AVWorker@?A0x1234abcd@jobs@@", "jobs::`anonymous namespace'::Worker"),
            (".?AV<lambda_1>@?1??Run@Task@jobs@@QEAAXH@Z@", "`jobs::Task::Run'::`2'::<lambda_1>"),
            (".?AV<lambda_1>@?0???0Task@jobs@@QEAA@XZ@", "`jobs::Task::Task'::`1'::<lambda_1>"),
            (".?AV<lambda_1>@?0???RTask@jobs@@QEBAHH@Z@", "`jobs::Task::operator()'::`1'::<lambda_1>"),
            # A conversion operator is named after the type it returns: two of them in one class stay apart.
            (".?AVX@?0???BObj@@QEBAHXZ@", "`Obj::operator int'::`1'::X"),
            (".?AVX@?0???BObj@@QEBAP6AXH@ZXZ@", "`Obj::operator void (*)(int)'::`1'::X"),
            (".?AV<lambda_2>@?5???$Each@H@List@core@@SA_NPEBVNode@2@0@Z@", "`core::List::Each<int>'::`6'::<lambda_2>"),
            (".?AV?$Holder@V<lambda_1>@?1??Run@Task@jobs@@QEAAXH@Z@H@jobs@@",
             "jobs::Holder<`jobs::Task::Run'::`2'::<lambda_1>, int>"),
        ):
            with self.subTest(mangled=mangled):
                self.assertEqual(rtti_scan.simple_name(mangled), name)

    def test_enum_and_union(self):
        self.assertEqual(rtti_scan.simple_name(".?AW4Mode@gfx@@"), "gfx::Mode")
        self.assertEqual(rtti_scan.simple_name(".?ATBits@gfx@@"), "gfx::Bits")

    def test_a_form_that_is_not_read_stays_mangled(self):
        # No made-up scope chain: a pointer to a member function, a template cut short, a name that never ends,
        # an array with a negative number of dimensions or a negative size.
        for mangled in (".?AV?$Bind@P8Task@jobs@@EAAXXZ@jobs@@", ".?AV?$Pool@@", ".?AV?$Pool@H@", ".?AV9@@",
                        ".?AV?$Arr@$$BY?0H@lab@@", ".?AV?$Arr@$$BY0?1H@lab@@", ".?AV?$Arr@AEAY?0H@lab@@",
                        ".?AV" + "?$A@V" * 5000 + "@"):
            with self.subTest(mangled=mangled[:40]):
                self.assertEqual(rtti_scan.simple_name(mangled), mangled)


class RttiScanCoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.image_base = 0x140000000

    def tearDown(self):
        self.tmp.cleanup()

    def run_tool(self, *args):
        return subprocess.run([sys.executable, "-I", TOOL, *args], capture_output=True, text=True, errors="replace")

    def test_type_descriptor_detection_and_parsing(self):
        rdata = bytearray(0x1000)
        # TypeDescriptor 1 at RVA 0x2000 (offset 0): 16 bytes prefix + mangled name
        rdata[0:16] = b"\x00" * 16
        rdata[16:32] = b".?AVPlayer@@\x00\x00\x00\x00"

        # TypeDescriptor 2 at RVA 0x2040 (offset 0x40)
        rdata[0x40:0x50] = b"\x00" * 16
        rdata[0x50:0x70] = b".?AUWeapon@Item@@\x00\x00\x00"

        # TypeDescriptor 3 at RVA 0x2080 (offset 0x80)
        rdata[0x80:0x90] = b"\x00" * 16
        rdata[0x90:0xB0] = b".?AWRenderState@@\x00"

        pe_bytes = make_synthetic_pe(text_bytes=b"\x90" * 0x10, rdata_bytes=bytes(rdata))
        img_path = os.path.join(self.tmp.name, "td_test.exe")
        with open(img_path, "wb") as f:
            f.write(pe_bytes)

        out_json = os.path.join(self.tmp.name, "out.json")
        p = self.run_tool(img_path, "--out", out_json)
        self.assertEqual(p.returncode, 0, p.stderr)

        with open(out_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["type_descriptors"], 3)
        self.assertEqual(len(data["classes"]), 0)  # No COLs yet

    def test_single_inheritance_hierarchy_and_vtable(self):
        rdata = bytearray(0x1000)

        # TD Animal at RVA 0x2000
        rdata[0x000:0x010] = b"\x00" * 16
        rdata[0x010:0x028] = b".?AVAnimal@@\x00\x00\x00\x00"

        # TD Dog at RVA 0x2040
        rdata[0x040:0x050] = b"\x00" * 16
        rdata[0x050:0x068] = b".?AVDog@@\x00\x00\x00\x00\x00"

        # BCD Animal at RVA 0x2080 (first field is rvaTypeDescriptor)
        rdata[0x080:0x09C] = struct.pack("<IIiiiII", 0x2000, 0, 0, 0, 0, 0, 0)
        # BCD Dog at RVA 0x0A0 (RVA 0x20A0)
        rdata[0x0A0:0x0BC] = struct.pack("<IIiiiII", 0x2040, 1, 0, 0, 0, 0, 0)

        # BaseClassArray at RVA 0x20C0 (offsets [Dog, Animal])
        rdata[0x0C0:0x0C8] = struct.pack("<II", 0x20A0, 0x2080)

        # ClassHierarchyDescriptor Dog at RVA 0x20D0: attrs=0 (single), nbase=2, rarr=0x20C0
        rdata[0x0D0:0x0E0] = struct.pack("<4I", 0, 0, 2, 0x20C0)

        # CompleteObjectLocator Dog at RVA 0x20E0:
        # sig=1, offset=0, cdOffset=0, rvaTD=0x2040, rvaCHD=0x20D0, rvaSelf=0x20E0
        rdata[0x0E0:0x0F8] = struct.pack("<6I", 1, 0, 0, 0x2040, 0x20D0, 0x20E0)

        # Vtable pointer at RVA 0x20F8 (preceding vtable at 0x2100)
        rdata[0x0F8:0x100] = struct.pack("<Q", self.image_base + 0x20E0)
        # Vtable slots at RVA 0x2100: 3 code pointers into .text (0x1000..0x1020), then null
        rdata[0x100:0x120] = struct.pack(
            "<QQQQ",
            self.image_base + 0x1000,
            self.image_base + 0x1010,
            self.image_base + 0x1020,
            0
        )

        pe_bytes = make_synthetic_pe(text_bytes=b"\xc3" * 0x100, rdata_bytes=bytes(rdata))
        img_path = os.path.join(self.tmp.name, "single_inherit.exe")
        with open(img_path, "wb") as f:
            f.write(pe_bytes)

        out_json = os.path.join(self.tmp.name, "single_inherit.json")
        p = self.run_tool(img_path, "--out", out_json)
        self.assertEqual(p.returncode, 0, p.stderr)

        with open(out_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["type_descriptors"], 2)
        self.assertEqual(len(data["classes"]), 1)
        cls_info = data["classes"][0]
        self.assertEqual(cls_info["name"], "Dog")
        self.assertEqual(cls_info["mangled"], ".?AVDog@@")
        self.assertEqual(cls_info["td_rva"], 0x2040)
        self.assertFalse(cls_info["multiple"])
        self.assertFalse(cls_info["virtual"])
        self.assertEqual(cls_info["bases"], ["Animal"])

        self.assertEqual(len(data["vtables"]), 1)
        vt_info = data["vtables"][0]
        self.assertEqual(vt_info["rva"], 0x2100)
        self.assertEqual(vt_info["class"], "Dog")
        self.assertEqual(vt_info["offset_in_object"], 0)
        self.assertEqual(vt_info["slots"], 3)

    def test_class_local_to_a_function_and_template_base(self):
        """A class declared inside a function has a name that ends with the function's `Z@`, not `@@`: its type
        descriptor used to be missed, and with it the class and its vtable."""
        local = b".?AVJob@?1??Run@Task@jobs@@QEAAXH@Z@"
        base = b".?AV?$Base@H@jobs@@"
        rdata = bytearray(0x1000)
        rdata[0x010:0x010 + len(base)] = base                 # TD at RVA 0x2000
        rdata[0x050:0x050 + len(local)] = local               # TD at RVA 0x2040
        rdata[0x210:0x21A] = b".?AVNoise@"                    # one '@' after a plain name: not a descriptor name
        rdata[0x080:0x09C] = struct.pack("<IIiiiII", 0x2000, 0, 0, 0, 0, 0, 0)     # BCD of the base
        rdata[0x0A0:0x0BC] = struct.pack("<IIiiiII", 0x2040, 1, 0, 0, 0, 0, 0)     # BCD of the local class
        rdata[0x0C0:0x0C8] = struct.pack("<II", 0x20A0, 0x2080)
        rdata[0x0D0:0x0E0] = struct.pack("<4I", 0, 0, 2, 0x20C0)
        rdata[0x0E0:0x0F8] = struct.pack("<6I", 1, 0, 0, 0x2040, 0x20D0, 0x20E0)
        rdata[0x0F8:0x100] = struct.pack("<Q", self.image_base + 0x20E0)
        rdata[0x100:0x110] = struct.pack("<QQ", self.image_base + 0x1000, 0)

        scanned = rtti_scan.scan_pe64_rtti(make_synthetic_pe(text_bytes=b"\xc3" * 0x100, rdata_bytes=bytes(rdata)))

        self.assertEqual(scanned["type_descriptors"], 2)
        self.assertEqual([(c["name"], c["mangled"], c["bases"]) for c in scanned["classes"]],
                         [("`jobs::Task::Run'::`2'::Job", local.decode(), ["jobs::Base<int>"])])
        self.assertEqual([(v["rva"], v["class"], v["slots"]) for v in scanned["vtables"]],
                         [(0x2100, "`jobs::Task::Run'::`2'::Job", 1)])

    def test_multiple_and_virtual_inheritance(self):
        rdata = bytearray(0x1000)

        # Base A TD at 0x2000
        rdata[0x000:0x010] = b"\x00" * 16
        rdata[0x010:0x028] = b".?AVBaseA@@\x00\x00\x00\x00\x00"

        # Base B TD at 0x2040
        rdata[0x040:0x050] = b"\x00" * 16
        rdata[0x050:0x068] = b".?AVBaseB@@\x00\x00\x00\x00\x00"

        # Derived TD at 0x2080
        rdata[0x080:0x090] = b"\x00" * 16
        rdata[0x090:0x0B0] = b".?AVDerived@@\x00\x00\x00"

        # BCDs at 0x20C0, 0x20E0, 0x2100
        rdata[0x0C0:0x0DC] = struct.pack("<IIiiiII", 0x2080, 2, 0, 0, 0, 0, 0)
        rdata[0x0E0:0x0FC] = struct.pack("<IIiiiII", 0x2000, 0, 0, 0, 0, 0, 0)
        rdata[0x100:0x11C] = struct.pack("<IIiiiII", 0x2040, 0, 0, 0, 0, 0, 0)

        # BaseClassArray at 0x2120: [Derived, BaseA, BaseB]
        rdata[0x120:0x12C] = struct.pack("<3I", 0x20C0, 0x20E0, 0x2100)

        # CHD Derived at 0x2130: attrs=3 (bit 0 multiple, bit 1 virtual), nbase=3, rarr=0x2120
        rdata[0x130:0x140] = struct.pack("<4I", 0, 3, 3, 0x2120)

        # Primary COL at 0x2140: offset=0
        rdata[0x140:0x158] = struct.pack("<6I", 1, 0, 0, 0x2080, 0x2130, 0x2140)

        # Secondary COL at 0x2160: offset=8
        rdata[0x160:0x178] = struct.pack("<6I", 1, 8, 0, 0x2080, 0x2130, 0x2160)

        # Vtable 1 ptr at 0x21F8 -> vtable at 0x2200 (2 slots)
        rdata[0x1F8:0x200] = struct.pack("<Q", self.image_base + 0x2140)
        rdata[0x200:0x218] = struct.pack("<QQQ", self.image_base + 0x1000, self.image_base + 0x1008, 0)

        # Vtable 2 ptr at 0x2218 -> vtable at 0x2220 (1 slot)
        rdata[0x218:0x220] = struct.pack("<Q", self.image_base + 0x2160)
        rdata[0x220:0x230] = struct.pack("<QQ", self.image_base + 0x1010, 0)

        pe_bytes = make_synthetic_pe(text_bytes=b"\xc3" * 0x100, rdata_bytes=bytes(rdata))
        img_path = os.path.join(self.tmp.name, "multi_virtual.exe")
        with open(img_path, "wb") as f:
            f.write(pe_bytes)

        out_json = os.path.join(self.tmp.name, "multi_virtual.json")
        p = self.run_tool(img_path, "--out", out_json)
        self.assertEqual(p.returncode, 0, p.stderr)

        with open(out_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(len(data["classes"]), 1)
        cls_info = data["classes"][0]
        self.assertEqual(cls_info["name"], "Derived")
        self.assertTrue(cls_info["multiple"])
        self.assertTrue(cls_info["virtual"])
        self.assertEqual(cls_info["bases"], ["BaseA", "BaseB"])

        self.assertEqual(len(data["vtables"]), 2)
        v0, v1 = data["vtables"][0], data["vtables"][1]
        self.assertEqual(v0["offset_in_object"], 0)
        self.assertEqual(v0["slots"], 2)
        self.assertEqual(v1["offset_in_object"], 8)
        self.assertEqual(v1["slots"], 1)

    def test_vtable_slots_stop_at_the_end_of_their_section(self):
        rdata = bytearray(0x1000)
        rdata[0x10:0x1D] = b".?AVProbe@@\x00\x00"
        # The vtable's first slot ends at .rdata's declared VirtualSize (0x200).
        struct.pack_into("<6I", rdata, 0x40, 1, 0, 0, 0x2000, 0, 0x2040)
        struct.pack_into("<3Q", rdata, 0x1F0,
                         self.image_base + 0x2040, self.image_base + 0x1000, self.image_base + 0x1008)
        pe = bytearray(make_synthetic_pe(rdata_bytes=rdata))
        section_virtual_size = 0x80 + 24 + 240 + 40 + 8
        struct.pack_into("<I", pe, section_virtual_size, 0x200)
        vtables = rtti_scan.scan_pe64_rtti(pe)["vtables"]
        self.assertEqual(vtables, [{"rva": 0x21F8, "class": "Probe", "offset_in_object": 0, "slots": 1}])

        # The same second code pointer is a real slot when the section extends to include it.
        struct.pack_into("<I", pe, section_virtual_size, 0x208)
        self.assertEqual(rtti_scan.scan_pe64_rtti(pe)["vtables"][0]["slots"], 2)

    def test_col_validation_rejects_corrupted_locators(self):
        rdata = bytearray(0x1000)

        # Valid TD at 0x2000
        rdata[0x000:0x010] = b"\x00" * 16
        rdata[0x010:0x028] = b".?AVTestObj@@\x00\x00\x00"

        # Corrupted COL 1: rvaSelf mismatch (rself != rva) at 0x2040
        rdata[0x040:0x058] = struct.pack("<6I", 1, 0, 0, 0x2000, 0x0, 0x9999)

        # Corrupted COL 2: unknown TD at 0x2060 (rtd = 0x5555 not in tds)
        rdata[0x060:0x078] = struct.pack("<6I", 1, 0, 0, 0x5555, 0x0, 0x2060)

        # Corrupted COL 3: signature mismatch (sig = 0) at 0x2080
        rdata[0x080:0x098] = struct.pack("<6I", 0, 0, 0, 0x2000, 0x0, 0x2080)

        # Corrupted COL 4: unaligned (placed at 0x20A1)
        rdata[0x0A1:0x0B9] = struct.pack("<6I", 1, 0, 0, 0x2000, 0x0, 0x20A1)

        pe_bytes = make_synthetic_pe(text_bytes=b"\xc3" * 0x10, rdata_bytes=bytes(rdata))
        img_path = os.path.join(self.tmp.name, "corrupt_col.exe")
        with open(img_path, "wb") as f:
            f.write(pe_bytes)

        out_json = os.path.join(self.tmp.name, "corrupt_col.json")
        p = self.run_tool(img_path, "--out", out_json)
        self.assertEqual(p.returncode, 0, p.stderr)

        with open(out_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        # None of the corrupt COLs should produce classes or vtables
        self.assertEqual(data["type_descriptors"], 1)
        self.assertEqual(len(data["classes"]), 0)
        self.assertEqual(len(data["vtables"]), 0)

    def test_chd_out_of_bounds_recovers_gracefully(self):
        rdata = bytearray(0x1000)

        # Valid TD at 0x2000
        rdata[0x000:0x010] = b"\x00" * 16
        rdata[0x010:0x028] = b".?AVRobustClass@@\x00"

        # Valid COL at 0x2040 pointing to an invalid/out-of-bounds CHD RVA (0x99000)
        rdata[0x040:0x058] = struct.pack("<6I", 1, 0, 0, 0x2000, 0x99000, 0x2040)

        pe_bytes = make_synthetic_pe(text_bytes=b"\xc3" * 0x10, rdata_bytes=bytes(rdata))
        img_path = os.path.join(self.tmp.name, "oob_chd.exe")
        with open(img_path, "wb") as f:
            f.write(pe_bytes)

        out_json = os.path.join(self.tmp.name, "oob_chd.json")
        p = self.run_tool(img_path, "--out", out_json)
        self.assertEqual(p.returncode, 0, p.stderr)

        with open(out_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(len(data["classes"]), 1)
        cls_info = data["classes"][0]
        self.assertEqual(cls_info["name"], "RobustClass")
        # Should fall back to empty bases and attributes
        self.assertEqual(cls_info["bases"], [])
        self.assertFalse(cls_info["multiple"])
        self.assertFalse(cls_info["virtual"])

    def test_null_or_header_chd_does_not_read_dos_header_as_hierarchy(self):
        rdata = bytearray(0x1000)
        rdata[0x000:0x010] = b"\x00" * 16
        rdata[0x010:0x028] = b".?AVNullChd@@\x00\x00\x00"
        # COL at 0x2040 with chd=0 (null RVA)
        rdata[0x040:0x058] = struct.pack("<6I", 1, 0, 0, 0x2000, 0, 0x2040)

        pe = bytearray(make_synthetic_pe(text_bytes=b"\xc3" * 0x10, rdata_bytes=bytes(rdata)))
        # Populate standard non-zero DOS header fields at offset 4..16 (sig!=0, attrs=3, nbase=4, rarr=0x2080)
        struct.pack_into("<3I", pe, 4, 3, 4, 0x2080)
        # Put a fake BCD array at 0x2080..0x2090 pointing to TD 0x2000 so a bogus read at RVA 0 would find bases
        struct.pack_into("<4I", pe, 0x400 + 0x1000 + 0x080, 0x2090, 0x2090, 0x2090, 0x2090)
        struct.pack_into("<I", pe, 0x400 + 0x1000 + 0x090, 0x2000)

        got = rtti_scan.scan_pe64_rtti(bytes(pe))
        self.assertEqual(len(got["classes"]), 1)
        cls_info = got["classes"][0]
        self.assertFalse(cls_info["multiple"])
        self.assertFalse(cls_info["virtual"])
        self.assertEqual(cls_info["bases"], [])

    def test_later_col_recovers_hierarchy_after_earlier_col_with_bad_chd(self):
        rdata = bytearray(0x1000)
        # Base TD at 0x2000, Derived TD at 0x2040
        rdata[0x010:0x020] = b".?AVBase@@\x00\x00\x00\x00\x00\x00"
        rdata[0x050:0x060] = b".?AVDerived@@\x00\x00\x00"
        # BCDs at 0x2080 (Derived) and 0x20A0 (Base)
        rdata[0x080:0x09C] = struct.pack("<IIiiiII", 0x2040, 1, 0, 0, 0, 0, 0)
        rdata[0x0A0:0x0BC] = struct.pack("<IIiiiII", 0x2000, 0, 0, 0, 0, 0, 0)
        # BaseClassArray at 0x20C0, valid CHD at 0x20D0 (attrs=1 multiple, nbase=2)
        rdata[0x0C0:0x0C8] = struct.pack("<2I", 0x2080, 0x20A0)
        rdata[0x0D0:0x0E0] = struct.pack("<4I", 0, 1, 2, 0x20C0)
        # First COL at 0x20E0 has out-of-bounds CHD (0x99000); second COL at 0x2100 has valid CHD (0x20D0)
        rdata[0x0E0:0x0F8] = struct.pack("<6I", 1, 0, 0, 0x2040, 0x99000, 0x20E0)
        rdata[0x100:0x118] = struct.pack("<6I", 1, 8, 0, 0x2040, 0x20D0, 0x2100)

        pe_bytes = make_synthetic_pe(text_bytes=b"\xc3" * 0x10, rdata_bytes=bytes(rdata))
        got = rtti_scan.scan_pe64_rtti(pe_bytes)
        by_name = {c["name"]: c for c in got["classes"]}
        self.assertTrue(by_name["Derived"]["multiple"])
        self.assertEqual(by_name["Derived"]["bases"], ["Base"])

    def test_nonnull_chd_with_bad_signature_or_null_base_array_is_refused(self):
        for name, signature, array_rva in (("bad signature", 7, 0x20A0), ("null array", 0, 0)):
            with self.subTest(name=name):
                rdata = bytearray(0x1000)
                mangled = b".?AVInvalidHierarchy@@\0"
                rdata[0x10:0x10 + len(mangled)] = mangled
                rdata[0x40:0x58] = struct.pack("<6I", 1, 0, 0, 0x2000, 0x2080, 0x2040)
                # A non-null CHD isolates signature validation from the null-RVA guard.
                # nbase=1 also ensures a null array cannot be rejected accidentally by an out-of-bounds read.
                rdata[0x80:0x90] = struct.pack("<4I", signature, 3, 1, array_rva)
                rdata[0xA0:0xA4] = struct.pack("<I", 0x20B0)
                rdata[0xB0:0xCC] = struct.pack("<IIiiiII", 0x2000, 0, 0, 0, 0, 0, 0)
                pe = make_synthetic_pe(text_bytes=b"\xc3" * 0x10, rdata_bytes=bytes(rdata))
                got = rtti_scan.scan_pe64_rtti(pe)
                self.assertEqual(len(got["classes"]), 1)
                cls_info = got["classes"][0]
                self.assertEqual(cls_info["bases"], [])
                self.assertFalse(cls_info["multiple"])
                self.assertFalse(cls_info["virtual"])

    def test_unaligned_section_start_does_not_scan_before_section_for_vtables(self):
        rdata = bytearray(0x1000)
        rdata[0x010:0x020] = b".?AVAligned@@\x00\x00\x00"
        # Valid COL at 0x2040
        rdata[0x040:0x058] = struct.pack("<6I", 1, 0, 0, 0x2000, 0, 0x2040)
        # Put a COL pointer at RVA 0x2100 and a code pointer at 0x2108. A third section will start at 0x2104,
        # so the COL pointer begins four bytes before that section.
        rdata[0x100:0x110] = struct.pack("<2Q", self.image_base + 0x2040, self.image_base + 0x1000)
        pe = bytearray(make_synthetic_pe(text_bytes=b"\xc3" * 0x10, rdata_bytes=bytes(rdata)))
        sec2_off = 0x80 + 24 + 240 + 40
        # Shrink .rdata VirtualSize to 0x100 (covers 0x2000..0x2100, excluding 0x2100) and add a 3rd section at 0x2104
        struct.pack_into("<I", pe, sec2_off + 8, 0x100)
        struct.pack_into("<H", pe, 0x80 + 6, 3)  # num_sections = 3
        sec3 = struct.pack(
            "<8sIIIIIIHHI",
            b".data\x00\x00\x00", 0x100, 0x2104, 0x100, 0x400 + 0x1000 + 0x104,
            0, 0, 0, 0, 0x40000040,
        )
        pe[sec2_off + 40 : sec2_off + 80] = sec3
        # Also write the stale COL pointer into the end of .rdata raw bytes at RVA 0x2100 by letting .rdata raw_size
        # cover 0x2100 while its virtual_size stops at 0x2100, and .data starts at 0x2104.
        got = rtti_scan.scan_pe64_rtti(bytes(pe))
        self.assertEqual(got["vtables"], [])


    def test_default_output_filename(self):
        rdata = bytearray(0x1000)
        rdata[0:16] = b"\x00" * 16
        rdata[16:32] = b".?AVSample@@\x00\x00\x00\x00\x00"

        pe_bytes = make_synthetic_pe(text_bytes=b"\x90" * 0x10, rdata_bytes=bytes(rdata))
        img_path = os.path.join(self.tmp.name, "test_sample.exe")
        with open(img_path, "wb") as f:
            f.write(pe_bytes)

        # Run without --out
        p = self.run_tool(img_path)
        self.assertEqual(p.returncode, 0, p.stderr)

        expected_json = os.path.join(self.tmp.name, "test_sample.rtti.json")
        self.assertTrue(os.path.isfile(expected_json))
        with open(expected_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["type_descriptors"], 1)

    def test_report_must_not_overwrite_the_input_image(self):
        image = os.path.join(self.tmp.name, "source.exe")
        original = make_synthetic_pe(text_bytes=b"\xc3", rdata_bytes=b".?AVSample@@\x00")
        hardlink = os.path.join(self.tmp.name, "source-link.exe")

        for label, output in [
            ("identical", image),
            ("alternate spelling", os.path.join(self.tmp.name, ".", "source.exe")),
            ("hard link", hardlink),
        ]:
            with self.subTest(label):
                with open(image, "wb") as f:
                    f.write(original)
                if os.path.exists(hardlink):
                    os.remove(hardlink)
                if label == "hard link":
                    try:
                        os.link(image, hardlink)
                    except (OSError, NotImplementedError) as e:
                        self.skipTest(f"hard links unavailable: {e}")
                p = self.run_tool(image, "--out", output)
                self.assertNotEqual(p.returncode, 0)
                self.assertIn("output must not overwrite the source image", p.stderr)
                with open(image, "rb") as f:
                    self.assertEqual(f.read(), original)

        distinct = os.path.join(self.tmp.name, "report.json")
        p = self.run_tool(image, "--out", distinct)
        self.assertEqual(p.returncode, 0, p.stderr)
        with open(image, "rb") as f:
            self.assertEqual(f.read(), original)
        with open(distinct, encoding="utf-8") as f:
            self.assertEqual(json.load(f)["image"], image)

    def test_output_it_cannot_write_is_one_line(self):
        img_path = os.path.join(self.tmp.name, "unwritable.exe")
        with open(img_path, "wb") as f:
            f.write(make_synthetic_pe(text_bytes=b"\xc3" * 16, rdata_bytes=b"\x00" * 16))

        for label, out in [
            ("missing folder", os.path.join(self.tmp.name, "no_dir", "out.json")),
            ("folder as file", self.tmp.name),
        ]:
            with self.subTest(label):
                p = self.run_tool(img_path, "--out", out)
                self.assertEqual(p.returncode, 1, p.stdout)
                self.assertNotIn("Traceback", p.stderr)
                self.assertTrue(p.stderr.startswith(f"ERROR: {out}: "), p.stderr)
                self.assertEqual(len(p.stderr.strip().splitlines()), 1, p.stderr)
                self.assertEqual(p.stdout, "")

    def test_failed_write_keeps_previous_report(self):
        # A write that fails after the file opens (full disk, locked range) used to leave a 0-byte report.
        from unittest import mock
        out = os.path.join(self.tmp.name, "kept.rtti.json")
        with open(out, "w", encoding="utf-8") as f:
            f.write('{"previous": true}')
        real = tempfile.NamedTemporaryFile

        def failing(*args, **kwargs):
            handle = real(*args, **kwargs)

            def fail_write(text):
                handle.file.write(text[:8])
                raise OSError(28, "No space left on device")

            handle.write = fail_write
            return handle

        for label, patch in [
            ("write", mock.patch.object(rtti_scan.tempfile, "NamedTemporaryFile", side_effect=failing)),
            ("replace", mock.patch.object(rtti_scan.os, "replace", side_effect=PermissionError(13, "busy"))),
        ]:
            with self.subTest(label), patch, self.assertRaises(OSError):
                rtti_scan.write_report(out, '{"new": true}')
            with open(out, encoding="utf-8") as f:
                self.assertEqual(f.read(), '{"previous": true}')
            self.assertEqual(sorted(os.listdir(self.tmp.name)), ["kept.rtti.json"])

        rtti_scan.write_report(out, '{"new": true}')
        with open(out, encoding="utf-8") as f:
            self.assertEqual(f.read(), '{"new": true}')
        self.assertEqual(sorted(os.listdir(self.tmp.name)), ["kept.rtti.json"])

    @unittest.skipUnless(os.name == "posix", "POSIX permission bits")
    def test_report_keeps_permission_bits(self):
        # NamedTemporaryFile creates 0600; the swapped-in report must not lose group/other read (#750 review).
        out = os.path.join(self.tmp.name, "perm.rtti.json")
        old = os.umask(0o022)
        try:
            rtti_scan.write_report(out, "{}")
            self.assertEqual(os.stat(out).st_mode & 0o777, 0o644)
            os.chmod(out, 0o640)
            rtti_scan.write_report(out, "{}")
            self.assertEqual(os.stat(out).st_mode & 0o777, 0o640)
            os.umask(0o027)
            os.remove(out)
            rtti_scan.write_report(out, "{}")
            self.assertEqual(os.stat(out).st_mode & 0o777, 0o640)
        finally:
            os.umask(old)

    def test_new_report_is_writable(self):
        # On Windows os.chmod only sets the read-only flag: a new report must not come out read-only.
        out = os.path.join(self.tmp.name, "w.rtti.json")
        rtti_scan.write_report(out, "{}")
        self.assertTrue(os.access(out, os.W_OK))
        rtti_scan.write_report(out, '{"again": 1}')
        with open(out, encoding="utf-8") as f:
            self.assertEqual(f.read(), '{"again": 1}')

    def test_invalid_pe_images_rejected_cleanly(self):
        good = bytearray(make_synthetic_pe(text_bytes=b"\xc3" * 16, rdata_bytes=b"\x00" * 16))
        bad_pe32 = bytearray(good)
        struct.pack_into("<H", bad_pe32, 0x80 + 24, 0x010B)  # PE32 (32-bit), not PE32+
        bad_sig = bytearray(good)
        bad_sig[0x80:0x84] = b"NOPE"
        bad_sec = bytearray(good)
        struct.pack_into("<H", bad_sec, 0x80 + 6, 999)  # section count exceeds file size

        for label, payload in [
            ("empty", b""),
            ("not MZ", b"NO" + b"\x00" * 128),
            ("bad PE signature", bytes(bad_sig)),
            ("32-bit PE32", bytes(bad_pe32)),
            ("truncated section table", bytes(bad_sec)),
        ]:
            with self.subTest(label):
                with self.assertRaises(ValueError):
                    rtti_scan.load_pe64_image(payload)

        # CLI returns exit code 1 with clear error message on invalid PE.
        bad_path = os.path.join(self.tmp.name, "not_a_pe.exe")
        with open(bad_path, "wb") as f:
            f.write(b"NOT_A_PE")
        p = self.run_tool(bad_path)
        self.assertEqual(p.returncode, 1)
        self.assertIn("ERROR:", p.stderr)


if __name__ == "__main__":
    unittest.main()
