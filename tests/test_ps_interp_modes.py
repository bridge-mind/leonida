"""The PS interpolation-mode decoder reads PSV0 elements, ISG1 read masks, FSSM read methods
and classifies the single barycentric pair; malformed containers are refused.

Covers tools/ps_interp_modes.py (SHADER-181).  All containers here are synthetic: the PSV0
layout is the one dxc writes (lib/DxilContainer/DxilPipelineStateValidation.cpp), the
interpolation-mode numbers are DXIL::InterpolationMode ("must match D3D_INTERPOLATION_MODE")
and the class names come from the SPI_PS_INPUT_ENA fields (Mesa gfx103.json).
"""
import contextlib
import errno
import io
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
TOOL = os.path.join(TOOLS, "ps_interp_modes.py")
sys.path.insert(0, TOOLS)

import ps_interp_modes  # noqa: E402


def psv0(elements, inputs=None, strings=b"\0TEXCOORD\0", semidx=(0,), resources=0):
    """Build a PSV0 chunk payload.  elements are (row, cols, kind, ctype, mode) tuples."""
    if inputs is None:
        inputs = len(elements)
    heap = strings.ljust((len(strings) + 3) & ~3, b"\0")
    body = b"".join(struct.pack("<II6B2B", 1, 0, 1, row, 0x40 | cols, kind, ctype, mode, 0, 0)
                    for row, cols, kind, ctype, mode in elements)
    runtime = b"\0" * 24
    r1 = bytes([0, 0, 0, 0, inputs, 0, 0, 0]) + b"\0" * 12
    out = struct.pack("<I", len(runtime) + len(r1)) + runtime + r1
    out += struct.pack("<I", resources)
    out += struct.pack("<I", len(heap)) + heap
    out += struct.pack("<I", len(semidx)) + b"".join(struct.pack("<I", s) for s in semidx)
    out += struct.pack("<I", 16) + body
    return out


def dxbc_container(chunks):
    """Wrap {fourcc: payload} into a DXBC container."""
    total = 32 + 4 * len(chunks)
    offsets = []
    body = b""
    for fourcc, payload in chunks.items():
        offsets.append(total + len(body))
        body += fourcc + struct.pack("<I", len(payload)) + payload
        while len(body) % 4:
            body += b"\0"
    header = b"DXBC" + b"\0" * 16 + struct.pack("<III", 1, total + len(body), len(chunks))
    return header + b"".join(struct.pack("<I", o) for o in offsets) + body

def isg1(elements):
    """ISG1 chunk payload; elements are (register, rw, name) with the repo's 32-byte layout."""
    names = [n.encode() + b"\0" for _, _, n in elements]
    first = 8 + 32 * len(elements)
    out = struct.pack("<II", len(elements), 8)
    for i, (reg, rw, name) in enumerate(elements):
        noff = first + sum(len(n) for n in names[:i])
        out += struct.pack("<6I", 0, noff, 0, 0, 3, reg)
        out += bytes([0xF, rw, 0, 0, 0, 0, 0, 0])
    return out + b"".join(names)


def fssm(records=()):
    """Synthetic FSSM blob: magic + zero header + input records at 0x138."""
    data = bytearray(0x138 + 8 * len(records) + 16)
    data[:4] = b"FSSM"
    for i, (semantic, mask, flags) in enumerate(records):
        base = 0x138 + 8 * i
        data[base] = 6
        data[base + 1] = semantic
        data[base + 2] = mask
        data[base + 3] = i
        data[base + 6] = i
        data[base + 7] = flags
    return bytes(data)


class Psv0InputsTest(unittest.TestCase):
    def test_elements_decode_with_rows_and_modes(self):
        c = dxbc_container({b"PSV0": psv0([(0, 4, 0, 3, 2), (1, 2, 0, 6, 2), (2, 4, 3, 3, 4)])})
        rows = ps_interp_modes.psv0_inputs(c)
        self.assertEqual([(r["row"], r["cols"], r["kind"], r["ctype"], r["mode"]) for r in rows],
                         [(0, 4, 0, 3, 2), (1, 2, 0, 6, 2), (2, 4, 3, 3, 4)])

    def test_semantic_name_comes_from_the_string_table(self):
        c = dxbc_container({b"PSV0": psv0([(0, 4, 0, 3, 2)])})
        self.assertEqual(psv0_inputs_name(c), "TEXCOORD")

    def test_truncated_chunks_are_refused(self):
        good = dxbc_container({b"PSV0": psv0([(0, 4, 0, 3, 2)])})
        # shorten every prefix boundary: runtime-info size word (<4 bytes), element size, semantic table, strings, resources
        for cut in (2, 4, 28, 36, 44, 52, 60):
            with self.subTest(cut=cut):
                self.assertRaises(ValueError, truncate_and_parse, good, cut)

    def test_v0_runtime_info_without_r1_is_refused(self):
        # runtime_info_size = 24 (PSVRuntimeInfo0 only, no PSVRuntimeInfo1): offset r1+4 (byte 32) lands inside the
        # following resource/string tables and must not be read as SigInputElements.
        heap = b"\0TEXCOORD\0\0\0"
        elem = struct.pack("<II6B2B", 1, 0, 1, 0, 0x44, 0, 3, 2, 0, 0)
        v0_only = (struct.pack("<I", 24) + (b"\0" * 24)
                   + struct.pack("<I", 0)
                   + struct.pack("<I", len(heap)) + heap
                   + struct.pack("<I", 1) + struct.pack("<I", 0)
                   + struct.pack("<I", 16) + elem)
        c = dxbc_container({b"PSV0": v0_only})
        self.assertRaises(ValueError, ps_interp_modes.psv0_inputs, c)

    def test_element_size_below_documented_16_is_refused(self):
        payload = bytearray(psv0([(0, 4, 0, 3, 2)]))
        # the element-size word is the last u32 before the elements
        payload[-18] = 8
        c = dxbc_container({b"PSV0": bytes(payload)})
        self.assertRaises(ValueError, ps_interp_modes.psv0_inputs, c)

    def test_missing_chunk_is_refused(self):
        self.assertRaises(ValueError, ps_interp_modes.psv0_inputs, dxbc_container({b"ISG1": b"\0" * 8}))


def psv0_inputs_name(container):
    return ps_interp_modes.psv0_inputs(container)[0]["name"]


def truncate_and_parse(container, size):
    chunks = ps_interp_modes.chunk_table(container)
    off, _ = chunks[b"PSV0"]
    data = container[:off + 8 + size]
    return ps_interp_modes.psv0_inputs(data)


class InterpClassTest(unittest.TestCase):
    def test_every_d3d_mode_maps_to_its_spi_ps_input_ena_bit(self):
        # mode -> (class, ENA bit), bit numbers from Mesa gfx103.json SPI_PS_INPUT_ENA
        expect = {2: ("persp_center", 0x2), 3: ("persp_centroid", 0x4), 6: ("persp_sample", 0x1),
                  4: ("linear_center", 0x20), 5: ("linear_centroid", 0x40),
                  7: ("linear_sample", 0x10), 1: ("flat", 0)}
        for mode, (cls, bit) in expect.items():
            self.assertEqual(ps_interp_modes.interp_class(0, mode), (cls, bit))
        self.assertEqual(ps_interp_modes.interp_class(3, 4), ("linear_center", 0x20))

    def test_system_value_kinds_are_system_inputs(self):
        self.assertEqual(ps_interp_modes.interp_class(13, 1), ("system", 0))
        self.assertEqual(ps_interp_modes.interp_class(12, 1), ("system", 0))

    def test_unknown_mode_is_other(self):
        self.assertEqual(ps_interp_modes.interp_class(0, 0), ("other", 0))
        self.assertEqual(ps_interp_modes.interp_class(0, 8), ("other", 0))


class ReadMethodTest(unittest.TestCase):
    def test_flag_bits_decode_to_read_methods(self):
        self.assertEqual(ps_interp_modes.read_method(0x01), {"mov": True, "p1p2": False,
                                                             "param16": False, "f16_ops": False})
        self.assertEqual(ps_interp_modes.read_method(0x02), {"mov": False, "p1p2": True,
                                                             "param16": False, "f16_ops": False})
        self.assertEqual(ps_interp_modes.read_method(0x1A), {"mov": False, "p1p2": True,
                                                             "param16": True, "f16_ops": True})
        self.assertEqual(ps_interp_modes.read_method(0x09), {"mov": True, "p1p2": False,
                                                             "param16": True, "f16_ops": False})


class DisasmFactsTest(unittest.TestCase):
    def test_v_interp_lines_decode_to_kinds_and_ij(self):
        text = ("\tv_interp_p1_f32_e32 v2, v0, attr2.x\n"
                "\tv_interp_p2_f32_e32 v2, v1, attr2.x\n"
                "\tv_interp_mov_f32 v5, p0, attr3.x\n"
                "\tv_interp_p1ll_f16 v6, v0, attr1.x\n")
        per, ij = ps_interp_modes.disasm_facts(text)
        self.assertEqual(per, {2: {"p1p2"}, 3: {"mov"}, 1: {"f16"}})
        self.assertEqual(ij, {"v0", "v1"})

    def test_mov_p0_is_not_an_ij_operand(self):
        per, ij = ps_interp_modes.disasm_facts("\tv_interp_mov_f32 v5, p10, attr0.x\n")
        self.assertEqual(per, {0: {"mov"}})
        self.assertEqual(ij, set())


class PairClassTest(unittest.TestCase):
    def rows(self, *specs):
        return [{"kind": k, "mode": m, "rw": rw} for k, m, rw in specs]

    def test_perspective_generic_wins_the_pair(self):
        rows = self.rows((3, 4, 0xF), (0, 2, 0xF))
        self.assertEqual(ps_interp_modes.pair_class(rows, [], {0: {"p1p2"}}, {"v0"}), "persp_center")

    def test_live_position_alone_interpolates_linear(self):
        rows = self.rows((3, 4, 0xF), (0, 2, 0))
        self.assertEqual(ps_interp_modes.pair_class(rows, [], {0: {"p1p2"}}, {"v0"}), "linear_center")

    def test_mode5_position_is_linear_centroid(self):
        rows = self.rows((3, 5, 0xF))
        self.assertEqual(ps_interp_modes.pair_class(rows, [], {0: {"p1p2"}}, {"v0"}), "linear_centroid")

    def test_position_plus_flat_row_stays_flat(self):
        # a shader that reads SV_Position and a Constant row: the interpolation reads the
        # flat row (P0 semantics, any pair) and the position comes from the system VGPRs
        rows = self.rows((3, 5, 0xF), (0, 1, 0xF))
        self.assertEqual(ps_interp_modes.pair_class(rows, [], {1: {"p1p2"}}, {"v0"}), "flat_only")

    def test_constant_rows_only_is_flat(self):
        rows = self.rows((0, 1, 0xF))
        self.assertEqual(ps_interp_modes.pair_class(rows, [], {0: {"p1p2"}}, {"v0"}), "flat_only")

    def test_no_interpolation_at_all(self):
        self.assertEqual(ps_interp_modes.pair_class(self.rows((3, 4, 0xF)), [], {}, set()), "none")

    def test_raw_barycentric_read_without_v_interp(self):
        self.assertEqual(ps_interp_modes.pair_class(self.rows(), [], {}, {"v0"}), "raw_bary")


class ShaderRowTest(unittest.TestCase):
    def test_end_to_end_row_from_synthetic_container(self):
        c = dxbc_container({
            b"PSV0": psv0([(0, 4, 3, 3, 4), (1, 4, 0, 3, 2)]),
            b"ISG1": isg1([(0, 0xF, "SV_Position"), (1, 0xF, "TEXCOORD0")]),
        })
        with tempfile.TemporaryDirectory() as tmp:
            dx = os.path.join(tmp, "a.dxbc")
            fs = os.path.join(tmp, "fssm", "a.fssm")
            ds = os.path.join(tmp, "disasm", "a.s")
            os.mkdir(os.path.join(tmp, "fssm"))
            os.mkdir(os.path.join(tmp, "disasm"))
            with open(dx, "wb") as f: f.write(c)
            with open(fs, "wb") as f: f.write(fssm([(0, 0xF, 0x02), (1, 0xF, 0x02)]))
            with open(ds, "w") as f: f.write("v_interp_p1_f32_e32 v2, v0, attr1.x\n")
            row = ps_interp_modes.shader_row(dx, fs, ds)
        self.assertEqual(row["pair_class"], "persp_center")
        self.assertEqual([i["class"] for i in row["inputs"]], ["linear_center", "persp_center"])
        self.assertEqual([i["rw"] for i in row["inputs"]], [15, 15])
        self.assertEqual([t["p1p2"] for t in row["table"]], [True, True])
        self.assertEqual(row["usage"], {"1": ["p1p2"]})

    def test_unparsable_psv0_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            dx = os.path.join(tmp, "b.dxbc")
            with open(dx, "wb") as f: f.write(dxbc_container({b"PSV0": psv0([])[:20]}))
            self.assertIsNone(ps_interp_modes.shader_row(dx, None, None))

    def test_component_packed_same_row_preserves_each_element_rw_and_name(self):
        # Two inputs packed into register row 1: TEXCOORD0.xy (start_col=0, cols=2, mask=0x3, rw=0x3, read) and
        # TEXCOORD1.zw (start_col=2, cols=2, mask=0xC, rw=0x0, unread), with noff=0 in PSV0 so names come from ISG1.
        elems = [
            struct.pack("<II6B2B", 0, 0, 1, 0, 0x40 | 4, 3, 3, 4, 0, 0),
            struct.pack("<II6B2B", 0, 0, 1, 1, 0x40 | (0 << 4) | 2, 0, 3, 2, 0, 0),
            struct.pack("<II6B2B", 0, 0, 1, 1, 0x40 | (2 << 4) | 2, 0, 3, 2, 0, 0),
        ]
        runtime = b"\0" * 24 + bytes([0, 0, 0, 0, 3, 0, 0, 0]) + b"\0" * 12
        psv_payload = (struct.pack("<I", len(runtime)) + runtime
                       + struct.pack("<I", 0) + struct.pack("<I", 4) + b"\0\0\0\0"
                       + struct.pack("<I", 1) + struct.pack("<I", 0)
                       + struct.pack("<I", 16) + b"".join(elems))
        names = [b"SV_Position\0", b"TEXCOORD0\0", b"TEXCOORD1\0"]
        first = 8 + 32 * 3
        isg_payload = struct.pack("<II", 3, 8)
        specs = [(0, 0xF, 0xF), (1, 0x3, 0x3), (1, 0xC, 0x0)]
        for i, (reg, mask, rw) in enumerate(specs):
            noff = first + sum(len(n) for n in names[:i])
            isg_payload += struct.pack("<6I", 0, noff, i, 0, 3, reg) + bytes([mask, rw, 0, 0, 0, 0, 0, 0])
        isg_payload += b"".join(names)
        c = dxbc_container({b"PSV0": psv_payload, b"ISG1": isg_payload})
        with tempfile.TemporaryDirectory() as tmp:
            dx = os.path.join(tmp, "packed.dxbc")
            ds = os.path.join(tmp, "packed.s")
            with open(dx, "wb") as f: f.write(c)
            with open(ds, "w") as f: f.write("v_interp_p1_f32_e32 v2, v0, attr1.x\n")
            row = ps_interp_modes.shader_row(dx, None, ds)
        self.assertEqual([i["rw"] for i in row["inputs"]], [15, 3, 0])
        self.assertEqual([i["name"] for i in row["inputs"]], ["SV_Position", "TEXCOORD0", "TEXCOORD1"])
        self.assertEqual(row["pair_class"], "persp_center")

    def test_multi_row_element_aggregates_rw_across_spanned_registers(self):
        # A 2-row input (rows=2, start row=0) read only in register 1 must still report its live rw mask.
        elem = struct.pack("<II6B2B", 1, 0, 2, 0, 0x44, 6, 3, 2, 0, 0)
        runtime = b"\0" * 24 + bytes([0, 0, 0, 0, 1, 0, 0, 0]) + b"\0" * 12
        heap = b"\0SV_ClipDistance\0\0\0\0"
        psv_payload = (struct.pack("<I", len(runtime)) + runtime
                       + struct.pack("<I", 0) + struct.pack("<I", len(heap)) + heap
                       + struct.pack("<I", 2) + struct.pack("<II", 0, 1)
                       + struct.pack("<I", 16) + elem)
        c = dxbc_container({b"PSV0": psv_payload,
                            b"ISG1": isg1([(0, 0, "SV_ClipDistance"), (1, 0x3, "SV_ClipDistance")])})
        with tempfile.TemporaryDirectory() as tmp:
            dx = os.path.join(tmp, "multirow.dxbc")
            with open(dx, "wb") as f: f.write(c)
            row = ps_interp_modes.shader_row(dx, None, None)
        self.assertEqual(row["inputs"][0]["rw"], 3)


class TallyTest(unittest.TestCase):
    def test_census_counts_modes_and_pairs(self):
        rows = [
            {"inputs": [{"kind": 0, "mode": 2, "rw": 15, "class": "persp_center"},
                        {"kind": 3, "mode": 4, "rw": 15, "class": "linear_center"}],
             "table": [{"mov": False, "p1p2": True, "param16": False, "f16_ops": False, "mask": 15, "flags": 2}],
             "usage": {"0": ["p1p2"]}, "ij": ["v0", "v1"], "pair_class": "persp_center"},
            {"inputs": [{"kind": 3, "mode": 4, "rw": 15, "class": "linear_center"}],
             "table": [], "usage": {}, "ij": [], "pair_class": "linear_center"},
        ]
        t = ps_interp_modes.tally(rows)
        self.assertEqual(t["shaders"], 2)
        self.assertEqual(t["inputs"], 3)
        self.assertEqual(t["live_inputs"], 3)
        self.assertEqual(t["mode_by_kind"], {"kind0 mode2": 1, "kind3 mode4": 2})
        self.assertEqual(t["live_non_persp_center_inputs"], 2)
        self.assertEqual(t["shaders_with_non_persp_center"], 2)
        self.assertEqual(t["pair_class"], {"persp_center": 1, "linear_center": 1})
        self.assertEqual(t["flag_vs_usage"], {"mov=0 p1p2=1 f16=0 -> p1p2": 1})


class CliTest(unittest.TestCase):
    def test_directory_census_with_stage_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.mkdir(os.path.join(tmp, "fssm"))
            os.mkdir(os.path.join(tmp, "disasm"))
            with open(os.path.join(tmp, "ps1.dxbc"), "wb") as f:
                f.write(dxbc_container({b"PSV0": psv0([(0, 4, 0, 3, 2)]),
                                        b"ISG1": isg1([(0, 0xF, "TEXCOORD0")])}))
            with open(os.path.join(tmp, "fssm", "ps1.fssm"), "wb") as f:
                f.write(fssm([(0, 0xF, 0x02)]))
            with open(os.path.join(tmp, "disasm", "ps1.s"), "w") as f:
                f.write("v_interp_p1_f32_e32 v2, v0, attr0.x\n")
            with open(os.path.join(tmp, "vs1.dxbc"), "wb") as f:
                f.write(dxbc_container({b"PSV0": psv0([(0, 4, 0, 3, 2)])}))
            with open(os.path.join(tmp, "index.json"), "w") as f:
                json.dump([{"name": "ps1", "file": "ps1.dxbc", "stage": "ps"},
                           {"name": "vs1", "file": "vs1.dxbc", "stage": "vs"}], f)
            out = subprocess.run([sys.executable, "-I", TOOL, tmp, "--json"],
                                 capture_output=True, text=True, check=True)
            doc = json.loads(out.stdout)
        self.assertEqual(len(doc["files"]), 1)
        self.assertEqual(doc["tally"]["pair_class"], {"persp_center": 1})
        self.assertEqual(doc["tally"]["mode_by_kind"], {"kind0 mode2": 1})

    def write_two_shaders(self, tmp):
        """Two synthetic PS containers; the census tallies them without FSSM/disasm sidecars."""
        for name in ("ps1", "ps2"):
            with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
                f.write(dxbc_container({b"PSV0": psv0([(0, 4, 0, 3, 2)]),
                                        b"ISG1": isg1([(0, 0xF, "TEXCOORD0")])}))

    def test_limit_below_one_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.write_two_shaders(tmp)
            for bad in ("0", "-1"):
                with self.subTest(limit=bad):
                    p = subprocess.run([sys.executable, "-I", TOOL, tmp, "--json", "--limit", bad],
                                       capture_output=True, text=True)
                    self.assertEqual(p.returncode, 2)
                    self.assertIn("must be at least 1", p.stderr)
                    self.assertEqual(p.stdout, "")

    def test_positive_limit_still_caps_the_census(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.write_two_shaders(tmp)
            p = subprocess.run([sys.executable, "-I", TOOL, tmp, "--json", "--limit", "1"],
                               capture_output=True, text=True, check=True)
            self.assertEqual(len(json.loads(p.stdout)["files"]), 1)

    def test_directory_named_dxbc_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.write_two_shaders(tmp)
            os.mkdir(os.path.join(tmp, "bogus.dxbc"))
            p = subprocess.run([sys.executable, "-I", TOOL, tmp, "--json"],
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertEqual(len(json.loads(p.stdout)["files"]), 2)

    def test_scan_io_failures_do_not_emit_a_partial_census(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.write_two_shaders(tmp)
            failing_path = os.path.join(tmp, "ps2.dxbc")
            error = PermissionError(errno.EACCES, "permission denied", failing_path)
            original = ps_interp_modes.shader_row

            def fail_second(path, fssm_path, disasm_path):
                if path == failing_path:
                    raise error
                return original(path, fssm_path, disasm_path)

            for target, options, path, diagnostic in (
                ("files_of", {"side_effect": OSError(errno.EIO, "I/O error", tmp)}, tmp, "cannot list"),
                ("shader_row", {"side_effect": fail_second}, failing_path, "cannot read"),
            ):
                with self.subTest(target=target):
                    out, err = io.StringIO(), io.StringIO()
                    with mock.patch.object(ps_interp_modes, target, **options), contextlib.redirect_stdout(out), \
                            contextlib.redirect_stderr(err):
                        self.assertEqual(ps_interp_modes.main([tmp, "--json"]), 2)
                    self.assertEqual(out.getvalue(), "")
                    self.assertEqual(len(err.getvalue().splitlines()), 1)
                    self.assertIn(diagnostic, err.getvalue())
                    self.assertIn(path, err.getvalue())

    def test_missing_path_is_an_error(self):
        out = subprocess.run([sys.executable, "-I", TOOL, "Z:/nope"],
                             capture_output=True, text=True)
        self.assertEqual(out.returncode, 2)
        self.assertIn("does not exist", out.stderr)


if __name__ == "__main__":
    unittest.main()
