"""Unit tests for tools/isa_coverage.py (RDNA2 ISA coverage of the translator).

The fixtures are our own text: a few made-up instruction entries in the schema of AMD's machine-readable ISA XML and
a few lines of translator-shaped source. No game data and no AMD file is needed. One test reads the real
amdgpu_isa_rdna2.xml when LEONIDA_RDNA2_ISA_XML points at it (download line in docs/isa_coverage.md) and skips
otherwise.

    python -I tests/test_isa_coverage.py
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS_DIR = os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import isa_coverage as ic  # noqa: E402

TRANSLATOR = os.path.join(TOOLS_DIR, "rdna2spirv.py")
REAL_XML = os.environ.get(ic.ENV_XML)


def _ins(name, *encodings):
    encs = "".join(f"<InstructionEncoding><EncodingName>{e}</EncodingName></InstructionEncoding>" for e in encodings)
    return (f"<Instruction><InstructionName>{name}</InstructionName>"
            f"<InstructionEncodings>{encs}</InstructionEncodings></Instruction>")


FIXTURE_XML = (
    "<Spec><Document><License>MIT</License><SchemaVersion>1.1.1</SchemaVersion></Document>"
    "<ISA><Architecture><ArchitectureName>AMD RDNA 2</ArchitectureName></Architecture><Instructions>"
    + _ins("V_ADD_F32", "ENC_VOP2", "VOP3_INST_LITERAL", "VOP2_VOP_SDWA", "VOP2_VOP_DPP16")
    + _ins("V_FANCY_F32", "ENC_VOP1")
    + _ins("S_MOV_B32", "ENC_SOP1", "SOP1_INST_LITERAL")
    + _ins("S_DENORM_MODE", "ENC_SOPP")
    + _ins("V_CMP_LT_F32", "ENC_VOPC", "VOP3_SDST_ENC")
    + _ins("S_NOP", "ENC_SOPP")
    + _ins("GLOBAL_LOAD_DWORD", "ENC_FLAT_GLBL")
    + _ins("SCRATCH_LOAD_DWORD", "ENC_FLAT_SCRATCH")
    + _ins("IMAGE_LOAD", "ENC_MIMG", "MIMG_NSA1")
    + _ins("S_ENDPGM", "ENC_SOPP")
    + _ins("EXP", "ENC_EXP")
    + _ins("S_AND_SAVEEXEC_B64", "ENC_SOP1")
    + _ins("S_XOR_SAVEEXEC_B64", "ENC_SOP1")
    + _ins("V_SHR_HI", "ENC_VOP1")
    + _ins("V_SHR_LO", "ENC_VOP1")
    + "</Instructions></ISA></Spec>")

FIXTURE_SRC = '''
NOOPS = {"s_nop", "s_endpgm"}
TABLE = {"v_add_f32": "OpFAdd", "v_made_up_f32": "OpFAdd"}


def base_op(op):
    return op


class Gen:
    LOGIC = {"and": 1, "or": 2}

    def _insn(self, ins):
        op = base_op(ins["op"])
        if op in ("s_denorm_mode",):
            raise Unsupported(op)
        if op != "s_gone_b32":
            pass
        if op == "s_mov_b32":
            pass
        elif op.startswith("v_cmp_"):
            pass
        elif op in ("image_load", "v_add_u32_e64"):
            if bad:
                raise Unsupported(op + " form")
        elif op == "exp":
            pass
        elif op.endswith("_saveexec_b64") and op[2:].split("_saveexec")[0] in self.LOGIC:
            pass
        elif op.endswith("_hi") and flag:
            pass
        elif op.endswith("_lo"):
            raise Unsupported(op)
        else:
            raise Unsupported(op)

    def analysis(self, op):
        return op.startswith("global_")
'''


def _fixture(text, suffix):
    f = tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False, encoding="utf-8")
    f.write(text)
    f.close()
    return f.name


class LlvmProbeTests(unittest.TestCase):
    def test_backend_failure_is_not_instruction_support(self) -> None:
        diagnostic = "llvm-mc: error: unable to get target for 'amdgcn'"
        failure = subprocess.CompletedProcess(["llvm-mc"], 1, "", diagnostic)
        with patch.object(ic.subprocess, "run", return_value=failure):
            outcome = ic.probe_llvm_mc("llvm-mc", ["s_endpgm"])["s_endpgm"]
        self.assertEqual(outcome, f"LLVM probe failed (exit 1): {diagnostic}")

    def test_reports_success_and_known_instruction_diagnostics(self) -> None:
        probes = (
            ("s_endpgm", 0, "", "LLVM: valid on gfx1013"),
            ("v_add_f32", 1, "<stdin>:1:1: error: too few operands for instruction", "LLVM: valid on gfx1013"),
            ("v_bogus", 1, "<stdin>:1:1: error: invalid instruction", "unknown to LLVM"),
            ("v_wmma_f32", 1, "<stdin>:1:1: error: instruction not supported on this GPU",
             "LLVM: other GPU only (not gfx1013)"),
        )
        for name, code, diagnostic, expected in probes:
            with self.subTest(name=name):
                outcome = subprocess.CompletedProcess(["llvm-mc"], code, "", diagnostic)
                with patch.object(ic.subprocess, "run", return_value=outcome):
                    self.assertEqual(ic.probe_llvm_mc("llvm-mc", [name]), {name: expected})


class LlvmStallTests(unittest.TestCase):
    def test_stalled_llvm_mc_is_one_error_not_a_traceback(self) -> None:
        stalled = subprocess.TimeoutExpired(["llvm-mc"], 30)
        with patch.object(ic.subprocess, "run", side_effect=stalled) as run:
            probe = ic.probe_llvm_mc("llvm-mc", ["s_endpgm", "v_add_f32"])
        self.assertEqual(list(probe), ["error"])
        self.assertIn("did not answer", probe["error"])
        self.assertEqual(run.call_count, 1)      # the remaining names are not each given the full timeout again

    def test_main_reports_the_stall_next_to_each_name(self) -> None:
        xml, src = _fixture(FIXTURE_XML, ".xml"), _fixture(FIXTURE_SRC, ".py")
        self.addCleanup(os.unlink, xml)
        self.addCleanup(os.unlink, src)
        stalled = subprocess.TimeoutExpired(["llvm-mc"], 30)
        out = io.StringIO()
        with patch.object(ic.subprocess, "run", side_effect=stalled), contextlib.redirect_stdout(out):
            code = ic.main(["--isa", xml, "--translator", src, "--llvm-mc", "llvm-mc"])
        self.assertEqual(code, 0)
        self.assertIn("did not answer", out.getvalue())


class FixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.xml = _fixture(FIXTURE_XML, ".xml")
        cls.src = _fixture(FIXTURE_SRC, ".py")
        cls.isa = ic.read_isa(cls.xml)
        cls.scan = ic.scan_translator(cls.src)
        cls.rep = ic.build_report(cls.isa, cls.scan, ic.accepted_names(cls.src))

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls.xml)
        os.unlink(cls.src)

    def test_reads_metadata_and_encoding_groups(self):
        self.assertEqual(self.isa["architecture"], "AMD RDNA 2")
        self.assertEqual(self.isa["license"], "MIT")
        ins = self.isa["instructions"]
        self.assertEqual(ins["v_add_f32"], ["VOP2", "VOP3", "DPP/SDWA"])
        self.assertEqual(ins["v_cmp_lt_f32"], ["VOPC", "VOP3"])
        self.assertEqual(ins["global_load_dword"], ["GLOBAL"])
        self.assertEqual(ins["scratch_load_dword"], ["SCRATCH"])
        self.assertEqual(ins["image_load"], ["MIMG"])

    def test_classes(self):
        c = self.rep["class"]
        self.assertEqual(c["v_add_f32"], "handled-exact")        # a table key
        self.assertEqual(c["s_mov_b32"], "handled-exact")        # an == test
        self.assertEqual(c["s_nop"], "handled-exact")            # a set
        self.assertEqual(c["v_cmp_lt_f32"], "handled-family")    # op.startswith in _insn
        self.assertEqual(c["s_denorm_mode"], "refused")          # a test whose whole body raises Unsupported
        self.assertEqual(c["v_fancy_f32"], "unknown")
        # op.startswith("global_") sits in a helper that is not the dispatch, so it must not count.
        self.assertEqual(c["global_load_dword"], "unknown")
        self.assertEqual(self.rep["totals"], {"handled-exact": 6, "handled-family": 2, "refused": 1, "unknown": 6})

    def test_suffix_families(self):
        c = self.rep["class"]
        self.assertEqual(c["s_and_saveexec_b64"], "handled-family")
        self.assertEqual(c["s_xor_saveexec_b64"], "unknown")
        self.assertEqual(c["v_shr_hi"], "unknown")
        self.assertEqual(c["v_shr_lo"], "unknown")
        self.assertEqual(self.rep["suffixes"], {"_saveexec_b64": 1})
        self.assertIn("Suffix families the dispatch tests with op.endswith", ic.markdown(self.rep))

    def test_partial_refusal_is_flagged_but_handled(self):
        self.assertEqual(self.rep["class"]["image_load"], "handled-exact")
        self.assertIn("image_load", self.rep["partial"])

    def test_names_that_are_not_rdna2(self):
        names = {e["name"]: e for e in self.rep["not_in_rdna2"]}
        self.assertEqual(sorted(names), ["s_gone_b32", "v_add_u32_e64", "v_made_up_f32"])
        self.assertTrue(names["v_add_u32_e64"]["never_matches_after_base_op"])
        self.assertEqual(names["v_add_u32_e64"]["stem"], "v_add_u32")
        self.assertFalse(names["v_made_up_f32"]["never_matches_after_base_op"])

    def test_not_equal_is_not_a_refusal(self):
        # `if op != "x": raise Unsupported` guards x, it does not refuse it.
        src = _fixture('def f(op):\n    if op != "s_setreg_imm32_b32":\n        raise Unsupported(op)\n', ".py")
        try:
            self.assertEqual(ic.scan_translator(src)["refused"], [])
        finally:
            os.unlink(src)

    def test_accepted_names_keeps_tuple_comparators_when_refusal_uses_tuple(self):
        # Python shares the ast.Load() singleton across every tuple/list comparator in the AST, so putting
        # non-expr nodes into accepted_names()'s skip set used to drop every other `op in (...)` tuple in the file.
        accepted = ic.accepted_names(self.src)
        self.assertIn("image_load", accepted)
        self.assertIn("v_add_u32_e64", accepted)
        self.assertNotIn("s_denorm_mode", accepted)
        src = _fixture(
            'def f(op):\n'
            '    if op in ("v_fancy_f32",):\n'
            '        raise Unsupported(op)\n'
            '    elif op in ("v_fancy_f32", "s_mov_b32"):\n'
            '        return 1\n',
            ".py",
        )
        try:
            rep = ic.build_report(self.isa, ic.scan_translator(src), ic.accepted_names(src))
            self.assertEqual(rep["class"]["v_fancy_f32"], "handled-exact")
            self.assertNotIn("v_fancy_f32", rep["refused"])
        finally:
            os.unlink(src)

    def test_negated_and_guarded_comparisons_are_not_refusals(self):
        src = _fixture(
            'def f(op, bad):\n'
            '    if not (op == "s_setreg_imm32_b32"):\n'
            '        raise Unsupported(op)\n'
            '    if op in ("v_qsad_pk_u16_u8", "v_mqsad_pk_u16_u8") and bad:\n'
            '        raise Unsupported(op)\n',
            ".py",
        )
        try:
            scan = ic.scan_translator(src)
            self.assertEqual(scan["refused"], [])
            self.assertEqual(scan["partial"], ["v_mqsad_pk_u16_u8", "v_qsad_pk_u16_u8"])
            self.assertTrue({"v_qsad_pk_u16_u8", "v_mqsad_pk_u16_u8"} <= ic.accepted_names(src))
        finally:
            os.unlink(src)

    def test_groups_count_each_encoding(self):
        g = self.rep["groups"]
        self.assertEqual(g["VOP2"]["handled-exact"], 1)
        self.assertEqual(g["VOP3"]["total"], 2)              # v_add_f32 literal form + v_cmp_lt_f32 sdst form
        self.assertEqual(g["DPP/SDWA"]["total"], 1)
        self.assertEqual(g["GLOBAL"]["unknown"], 1)

    def test_outputs(self):
        text = ic.markdown(self.rep, list_unknown=True)
        self.assertIn("| VOP2 | 1 | 1 | 0 | 0 | 0 |", text)
        self.assertIn("`v_made_up_f32`", text)
        self.assertIn("`v_fancy_f32`", text)
        self.assertEqual(json.loads(json.dumps(self.rep))["totals"], self.rep["totals"])

    def test_cli(self):
        import contextlib
        import io
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(ic.main(["--isa", self.xml, "--translator", self.src, "--json"]), 0)
        self.assertEqual(json.loads(out.getvalue())["instructions"], 15)


class TranslatorScanTests(unittest.TestCase):
    """The real translator: facts that must keep holding for the scan to mean anything. No XML needed."""

    @classmethod
    def setUpClass(cls):
        cls.scan = ic.scan_translator(TRANSLATOR)

    def test_finds_well_known_rules(self):
        for name in ("v_mov_b32", "s_mov_b32", "v_cndmask_b32", "v_add_f32", "s_endpgm", "exp"):
            self.assertIn(name, self.scan["exact"], name)
        self.assertIn("v_cmp_", self.scan["prefixes"])
        self.assertIn("s_denorm_mode", self.scan["refused"])

    def test_dispatch_size_sanity(self):
        # A scan that silently stops seeing the if/elif chain would report almost everything unknown.
        self.assertGreater(len(self.scan["exact"]), 300)

    def test_accepted_names_matches_non_refused_exact(self):
        accepted = ic.accepted_names(TRANSLATOR)
        self.assertEqual(set(self.scan["exact"]) - accepted, set(self.scan["refused"]))


@unittest.skipUnless(REAL_XML and os.path.isfile(REAL_XML), f"set {ic.ENV_XML} to amdgpu_isa_rdna2.xml")
class RealIsaTests(unittest.TestCase):
    # Names the translator matches that AMD's RDNA2 XML does not list. Each one was looked at (docs/isa_coverage.md);
    # a name that is not here and not in the XML is a typo or a wrong-generation name and fails the test.
    KNOWN_NOT_IN_XML = {
        "s_delay_alu": "RDNA3 instruction, kept in NOOPS so RDNA3 text does not trip the translator",
        "s_get_waveid_in_workgroup": "gfx1013 per llvm-mc, missing from the XML",
        "v_mac_f32": "gfx1013 per llvm-mc, missing from the XML",
        "v_mad_f32": "gfx1013 per llvm-mc, missing from the XML",
        "v_mad_legacy_f32": "gfx1013 per llvm-mc, missing from the XML",
        "v_madak_f32": "gfx1013 per llvm-mc, missing from the XML",
        "v_madmk_f32": "gfx1013 per llvm-mc, missing from the XML",
        "v_mac_legacy_f32": "gfx1013 per llvm-mc (VOP2 v_mac_legacy_f32_e32), missing from the XML",
        "v_add_u16": "internal: v_add_nc_u16 is re-dispatched under this GFX9 spelling",
        "v_sub_u16": "internal: v_sub_nc_u16 is re-dispatched under this GFX9 spelling",
        "v_add_i16": "internal: v_add_nc_i16 is re-dispatched under this name",
        "v_sub_i16": "internal: v_sub_nc_i16 is re-dispatched under this name",
        "s_atc_probe": "gfx1013 per llvm-mc, missing from the XML (a no-op in NOOPS)",
        "s_atc_probe_buffer": "gfx1013 per llvm-mc, missing from the XML (a no-op in NOOPS)",
    }

    @classmethod
    def setUpClass(cls):
        cls.rep = ic.build_report(ic.read_isa(REAL_XML), ic.scan_translator(TRANSLATOR),
                                  ic.accepted_names(TRANSLATOR))

    def test_translator_names_exist_in_rdna2(self):
        stray = sorted(e["name"] for e in self.rep["not_in_rdna2"] if e["name"] not in self.KNOWN_NOT_IN_XML)
        self.assertEqual(stray, [], "translator matches names that are not RDNA2 instructions")

    def test_no_name_ends_in_a_stripped_suffix(self):
        dead = [e["name"] for e in self.rep["not_in_rdna2"] if e["never_matches_after_base_op"]]
        self.assertEqual(dead, [])

    def test_every_prefix_matches_something(self):
        self.assertEqual(self.rep["prefixes_without_rdna2_match"], [])

    def test_saveexec_family_is_handled(self):
        stems = ("and", "andn1", "andn2", "nand", "nor", "or", "orn1", "orn2", "xnor", "xor")
        for w in ("b32", "b64"):
            for k in stems:
                self.assertEqual(self.rep["class"][f"s_{k}_saveexec_{w}"], "handled-family", (k, w))


if __name__ == "__main__":
    unittest.main()
