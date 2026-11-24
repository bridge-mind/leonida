"""Bad-input errors for the index/carve/audio/coverage tools: a message, not a traceback.

    python -I tests/test_bad_input_errors.py

All inputs are built here; no game file or asset is used.
"""
import os
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# A machine-readable ISA spec of our own making, small enough to hand-write: one RDNA 2 instruction,
# enough for isa_coverage to reach its report with no game data involved.
MINIMAL_ISA_XML = """<?xml version="1.0"?>
<isa>
  <ISA>
    <Architecture><ArchitectureName>AMD RDNA 2</ArchitectureName></Architecture>
    <Instructions>
      <Instruction>
        <InstructionName>V_ADD_F32</InstructionName>
        <InstructionEncodings>
          <InstructionEncoding><EncodingName>ENC_VOP3_INST</EncodingName></InstructionEncoding>
        </InstructionEncodings>
      </Instruction>
    </Instructions>
  </ISA>
</isa>
"""


def run(tool, *args):
    return subprocess.run([sys.executable, "-I", os.path.join(ROOT, "tools", tool), *args],
                          capture_output=True, text=True)


class BadInputTest(unittest.TestCase):
    def assert_error(self, p, text):
        self.assertNotEqual(p.returncode, 0, p.stdout)
        self.assertIn(text.lower(), p.stderr.lower())
        self.assertNotIn("traceback", p.stderr.lower())

    def assert_error_exit_2(self, p, path):
        """An argparse rejection: exit 2, the path and a reason on stderr, no traceback."""
        self.assert_error(p, path)
        self.assertEqual(p.returncode, 2, p.stderr)

    def test_empty_archive_in_scan_tools(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty = os.path.join(tmp, "empty.rpf")
            open(empty, "wb").close()
            for tool in ("anim_index.py", "rbf0_index.py", "restofworld_index.py"):
                with self.subTest(tool=tool):
                    out = os.path.join(tmp, "out.json")
                    p = run(tool, empty, out)
                    self.assert_error(p, "empty file")
                    self.assertFalse(os.path.exists(out), "a failed run must not write an index")

    def test_empty_and_short_archive_in_carve(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty = os.path.join(tmp, "empty.rpf")
            open(empty, "wb").close()
            self.assert_error(run("rpf9_carve.py", empty, os.path.join(tmp, "out")), "empty file")

            short = os.path.join(tmp, "short.rpf")
            with open(short, "wb") as f:
                f.write(b"RPF9")
            self.assert_error(run("rpf9_carve.py", short, os.path.join(tmp, "out")), "too short")

    def test_carve_start_at_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "empty-data.rpf")
            data = bytearray(0x40)
            data[0:4] = b"9FPR"
            struct.pack_into("<I", data, 4, 0x40)
            with open(arc, "wb") as f:
                f.write(data)
            p = run("rpf9_carve.py", arc, os.path.join(tmp, "out"), "--start", "64")
            self.assert_error(p, "outside the file")

    def test_carve_and_gamedata_index_reject_a_non_rpf9_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "notrpf.rpf")
            with open(arc, "wb") as f:
                f.write(b"MZ" + bytes(0x100))
            self.assert_error(run("rpf9_carve.py", arc, os.path.join(tmp, "out")), "not an RPF9 archive")
            self.assert_error(run("gamedata_index.py", arc), "not an RPF9 archive")

    def test_audio_config_with_no_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            bank = os.path.join(tmp, "empty.rec")
            with open(bank, "wb") as f:
                f.write(bytes(64))
            self.assert_error(run("audio_config.py", bank), "no audio-config records")

    def test_isa_coverage_isa_file_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "no-such.xml")
            self.assert_error_exit_2(run("isa_coverage.py", "--isa", missing), missing)

    def test_isa_coverage_isa_path_is_a_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assert_error_exit_2(run("isa_coverage.py", "--isa", tmp), tmp)

    def test_isa_coverage_isa_xml_malformed(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = os.path.join(tmp, "bad.xml")
            with open(bad, "w", encoding="utf-8", newline="\n") as f:
                f.write("nope")
            self.assert_error_exit_2(run("isa_coverage.py", "--isa", bad), bad)

    def test_isa_coverage_isa_xml_with_no_instructions(self):
        """Well-formed XML that is not the ISA spec: zero instructions is one error line, not a ZeroDivisionError."""
        with tempfile.TemporaryDirectory() as tmp:
            for i, content in enumerate(("<a/>", "<isa><ISA/></isa>",
                                         "<isa><Document><SchemaVersion>1</SchemaVersion></Document></isa>")):
                with self.subTest(content=content):
                    xml = os.path.join(tmp, f"no-instructions-{i}.xml")
                    with open(xml, "w", encoding="utf-8", newline="\n") as f:
                        f.write(content)
                    p = run("isa_coverage.py", "--isa", xml)
                    self.assert_error_exit_2(p, xml)
                    self.assertIn("no instructions", p.stderr.lower())

    def test_isa_coverage_translator_file_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            isa = os.path.join(tmp, "isa.xml")
            with open(isa, "w", encoding="utf-8", newline="\n") as f:
                f.write(MINIMAL_ISA_XML)
            missing = os.path.join(tmp, "no-such-translator.py")
            self.assert_error_exit_2(run("isa_coverage.py", "--isa", isa, "--translator", missing), missing)

    def test_isa_coverage_translator_not_utf8(self):
        with tempfile.TemporaryDirectory() as tmp:
            isa = os.path.join(tmp, "isa.xml")
            with open(isa, "w", encoding="utf-8", newline="\n") as f:
                f.write(MINIMAL_ISA_XML)
            latin = os.path.join(tmp, "latin1.py")
            with open(latin, "wb") as f:
                f.write(b"# caf\xe9: 0xff is never a UTF-8 lead byte\n\xff\n")
            self.assert_error_exit_2(run("isa_coverage.py", "--isa", isa, "--translator", latin), latin)


if __name__ == "__main__":
    unittest.main()
