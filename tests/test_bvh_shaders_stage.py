"""make_game.py's BVH builder step (bvh_shaders) and the index key the runtime finds its programs by (fssm_sha256).

    python -I tests/test_bvh_shaders_stage.py

Zero game data: the "DLL" is synthetic bytes around hand-built DXBC containers whose XBCH record is a zlib stream of
a made-up FSSM blob, and llvm-mc and the translator are stood in for the way tests/test_fspx_shaders_stage.py does.
The extracting, unpacking and disassembling tools are the repository's own, run as make_game.py runs them.
"""
import hashlib
import io
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
import make_game  # noqa: E402
import rdna2spirv  # noqa: E402
import shader_meta  # noqa: E402
from test_pe_shader_containers import container, ildn, xdat  # noqa: E402

TOOLS = ["pe_shader_containers.py", "fssm_unpack.py", "shader_disasm.py", "rdna2spirv.py"]
ENDPGM = b"\x00\x00\x81\xbf"


def fssm_blob(seed: int) -> bytes:
    """A made-up FSSM blob shader_disasm.py accepts: the code section offset at 0xC0, code 0x28 bytes into it, a
    few filler words that differ per seed and the closing s_endpgm."""
    head = bytearray(b"FSSM" + b"\x01\x01" + bytes(0xF0 - 6))
    struct.pack_into("<I", head, 0xC0, 0xC8)
    return bytes(head) + b"".join(struct.pack("<I", 0x7E000280 + seed + k) for k in range(3)) + ENDPGM


def builder_container(name: str, blob: bytes) -> bytes:
    """DXBC { SFI0, ILDN = the module's PDB path, DXIL = program header + XDAT { 0x12, 0x0D = XBCH + zlib(blob),
    0x07 = bitcode } }: the parts of the builder's containers the tools read (docs/embedded_shaders.md section 2)."""
    packed = zlib.compress(blob)
    xbch = b"XBCH" + struct.pack("<IIIII", 1, len(packed), 0, len(blob), 0) + packed
    prefix = b"e\x00\x05\x00" + struct.pack("<I", 24) + b"DXIL" + b"\x05\x01\x00\x00" + struct.pack("<II", 0x70, 0x668)
    dxil = prefix + xdat([(0x12, bytes(16)), (0x0D, xbch), (0x07, b"BC\xc0\xde" + bytes(8))], len(prefix))
    return container([(b"SFI0", bytes(8)), (b"ILDN", ildn(name)), (b"DXIL", dxil)])


def dll_like(*containers: bytes) -> bytes:
    """Containers between unrelated bytes, as a DLL holds them."""
    return b"MZ" + b"\0" * 126 + b"".join(c + b"\xcc" * 12 for c in containers)


def tool(name: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-I", os.path.join(ROOT, "tools", name), *args],
                          capture_output=True, text=True)


class IndexKey(unittest.TestCase):
    """fssm_sha256 is the SHA-256 of the FSSM blob fssm_unpack.py inflates from the container's XBCH record: the
    bytes runtime/src/shader_library.h's fssm_program() inflates from the container the DLL passes to slot 62."""

    def test_the_extracted_layout_gives_the_translator_the_key_and_the_stage(self):
        blobs = {"SortKeys": fssm_blob(1), "WriteNodes": fssm_blob(2)}
        with tempfile.TemporaryDirectory() as t:
            dll = os.path.join(t, "builder.dll")
            with open(dll, "wb") as f:
                f.write(dll_like(*(builder_container(name, blob) for name, blob in blobs.items())))
            out = os.path.join(t, "bvh")
            r = tool("pe_shader_containers.py", dll, "--extract", out)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            r = tool("fssm_unpack.py", out, "--carve-index", os.path.join(out, "index.json"))
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            os.makedirs(os.path.join(out, "disasm"))
            rdna2spirv._STAGES = None        # stage_of() reads <out>/index.json once per process
            self.addCleanup(setattr, rdna2spirv, "_STAGES", None)
            for name, blob in blobs.items():
                with open(os.path.join(out, "fssm", name + ".fssm"), "rb") as f:
                    self.assertEqual(f.read(), blob)
                listing = os.path.join(out, "disasm", name + ".s")
                meta = shader_meta.collect(listing, name, [[]], [[]])
                self.assertEqual(meta.get("fssm_sha256"), hashlib.sha256(blob).hexdigest())
                self.assertNotIn("fspx_code_sha256", meta)
                self.assertEqual(rdna2spirv.stage_of(name, listing), "cs")

    def test_a_container_without_an_xbch_record_gets_no_program(self):
        """Such a container is extracted and then counted by the step as not translated, never dropped."""
        bare = container([(b"SFI0", bytes(8)), (b"ILDN", ildn("NoProgram")), (b"DXIL", b"\0" * 24)])
        with tempfile.TemporaryDirectory() as t:
            dll = os.path.join(t, "builder.dll")
            with open(dll, "wb") as f:
                f.write(dll_like(bare, builder_container("SortKeys", fssm_blob(1))))
            out = os.path.join(t, "bvh")
            self.assertEqual(tool("pe_shader_containers.py", dll, "--extract", out).returncode, 0)
            r = tool("fssm_unpack.py", out, "--carve-index", os.path.join(out, "index.json"))
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertEqual(sorted(os.listdir(os.path.join(out, "fssm"))), ["SortKeys.fssm"])
            self.assertTrue(os.path.isfile(os.path.join(out, "NoProgram.dxbc")))


class BvhStage(unittest.TestCase):
    """Which tools bvh_shaders() runs and what it reports. Each stand-in writes what the real tool writes last."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = os.path.join(self.tmp.name, "work")
        self.game = os.path.join(self.tmp.name, "game")
        os.makedirs(self.game)
        self.dll = os.path.join(self.game, make_game.BVH_DLL)
        self.programs = {"SortKeys": fssm_blob(1), "WriteNodes": fssm_blob(2), "CopyNodes": fssm_blob(3)}
        self.untranslated = set()        # the programs the stand-in translator has no rule for
        self.write_dll()
        self.unpacked = os.path.join(self.work, "shaders", "bvh")
        self.spirv = os.path.join(self.work, "shaders", "spirv", "bvh")
        self.calls, self.logs = [], []

    def write_dll(self):
        with open(self.dll, "wb") as f:
            f.write(dll_like(*(builder_container(name, blob) for name, blob in self.programs.items())))

    def fake_translate(self, *args):
        """rdna2spirv.py: a module and an index entry for every listing it has a rule for."""
        out = args[args.index("--out") + 1]
        os.makedirs(out, exist_ok=True)
        index = {}
        for listing in sorted(os.listdir(args[0])):
            name = listing[:-2]
            if not listing.endswith(".s") or name in self.untranslated:
                continue
            with open(os.path.join(os.path.dirname(args[0]), "fssm", name + ".fssm"), "rb") as f:
                index[name] = {"stage": "cs", "fssm_sha256": hashlib.sha256(f.read()).hexdigest()}
            with open(os.path.join(out, name + ".spv"), "wb") as f:
                f.write(b"module of " + name.encode())
        with open(os.path.join(out, "index.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(index, f)

    def fake_py(self, tool_name, *args):
        self.calls.append((tool_name, args))
        if tool_name == "pe_shader_containers.py":
            os.makedirs(self.unpacked, exist_ok=True)
            for name in self.programs:
                open(os.path.join(self.unpacked, name + ".dxbc"), "wb").close()
            with open(os.path.join(self.unpacked, "index.json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump([{"name": n, "file": n + ".dxbc", "stage": "cs"} for n in self.programs], f)
        elif tool_name == "fssm_unpack.py":
            os.makedirs(os.path.join(self.unpacked, "fssm"), exist_ok=True)
            for name, blob in self.programs.items():
                with open(os.path.join(self.unpacked, "fssm", name + ".fssm"), "wb") as f:
                    f.write(blob)
            with open(os.path.join(self.unpacked, "fssm_index.json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump([{"name": n} for n in self.programs], f)
        elif tool_name == "shader_disasm.py":
            os.makedirs(os.path.join(self.unpacked, "disasm"), exist_ok=True)
            for name in self.programs:
                with open(os.path.join(self.unpacked, "disasm", name + ".s"), "w", encoding="utf-8",
                          newline="\n") as f:
                    f.write("s_endpgm\n")
        elif tool_name == "rdna2spirv.py":
            self.fake_translate(*args)

    def run_step(self, signature="llvm-mc 23", emulate=False, fallbacks=(), py=None):
        self.calls, self.logs = [], []
        with mock.patch.object(make_game, "py", py or self.fake_py), \
                mock.patch.object(make_game, "log", self.logs.append):
            make_game.bvh_shaders(self.game, self.work, signature, 3, emulate=emulate, fallbacks=fallbacks)
        return [name for name, _ in self.calls]

    def test_runs_once_then_only_reports(self):
        self.assertEqual(self.run_step(), TOOLS)
        (_, extract), (_, unpack), (_, disasm), (_, translate) = self.calls
        self.assertEqual(extract, (self.dll, "--extract", self.unpacked))
        self.assertEqual(unpack, (self.unpacked, "--carve-index", os.path.join(self.unpacked, "index.json")))
        self.assertEqual(disasm, (os.path.join(self.unpacked, "fssm"), "--unpack-index",
                                  os.path.join(self.unpacked, "fssm_index.json"), "--out",
                                  os.path.join(self.unpacked, "disasm"), "--mcpu", "gfx1013", "--jobs", 3))
        # The listings sit beside index.json and fssm/, where rdna2spirv.py looks for the stage and the FSSM blob.
        self.assertEqual(translate, (os.path.join(self.unpacked, "disasm"), "--out", self.spirv, "--no-asm",
                                     "--max-shared-memory", 32768))
        self.assertTrue(os.path.isfile(os.path.join(self.spirv, "bvh_build.json")))
        self.assertIn(f"shaders: 3 of 3 {make_game.BVH_DLL} programs translated", self.logs)
        self.assertEqual(self.run_step(), [])
        self.assertEqual(self.logs, [f"shaders: 3 of 3 {make_game.BVH_DLL} programs translated"])

    def test_the_real_tools_fill_the_folder_and_a_second_run_leaves_no_duplicates(self):
        """Extraction, unpacking and disassembly by the repository's tools (llvm-mc stood in for), twice: the
        second run must replace the first one's containers, where pe_shader_containers.py alone would add
        <module>_NN.dxbc beside them and those would be translated as more programs."""
        mc = os.path.join(self.tmp.name, "fake_mc.py")
        with open(mc, "w", encoding="utf-8", newline="\n") as f:
            f.write("print('\\ts_endpgm')\n")

        def run_tool(tool_name, *args):
            self.calls.append((tool_name, args))
            if tool_name == "rdna2spirv.py":
                return self.fake_translate(*args)
            make_game.run([sys.executable, "-I", os.path.join(make_game.TOOLS, tool_name), *args],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        with mock.patch.dict(os.environ, {"LLVM_MC": f'"{sys.executable}" "{mc}"'}):
            for signature in ("llvm-mc 23", "llvm-mc 24"):
                self.assertEqual(self.run_step(signature=signature, py=run_tool), TOOLS)
                names = sorted(self.programs)
                self.assertEqual(sorted(f for f in os.listdir(self.unpacked) if f.endswith(".dxbc")),
                                 [n + ".dxbc" for n in names])
                self.assertEqual(sorted(os.listdir(os.path.join(self.unpacked, "fssm"))),
                                 [n + ".fssm" for n in names])
                self.assertEqual(sorted(os.listdir(os.path.join(self.unpacked, "disasm"))),
                                 [n + ".s" for n in names])
                with open(os.path.join(self.spirv, "index.json"), encoding="utf-8") as f:
                    index = json.load(f)
                self.assertEqual({n: index[n]["fssm_sha256"] for n in index},
                                 {n: hashlib.sha256(blob).hexdigest() for n, blob in self.programs.items()})
                self.assertIn(f"shaders: 3 of 3 {make_game.BVH_DLL} programs translated", self.logs)
            self.assertEqual(self.run_step(signature="llvm-mc 24", py=run_tool), [])

    def test_no_dll_no_step(self):
        os.remove(self.dll)
        self.assertEqual(self.run_step(), [])
        self.assertFalse(os.path.exists(self.spirv))
        self.assertFalse(os.path.exists(self.unpacked))

    def test_a_dll_that_is_gone_takes_the_old_index_with_it(self):
        """A work folder reused for a package without the DLL: the runtime loads <shaders>/bvh/index.json whenever
        it exists, so the index of the earlier run must not stay, and the DLL coming back runs the step again."""
        self.assertEqual(self.run_step(), TOOLS)
        index, settings = os.path.join(self.spirv, "index.json"), os.path.join(self.spirv, "bvh_build.json")
        self.assertTrue(os.path.isfile(index) and os.path.isfile(settings))
        os.remove(self.dll)
        self.assertEqual(self.run_step(), [])
        self.assertFalse(os.path.exists(index))
        self.assertFalse(os.path.exists(settings))
        self.assertEqual(self.logs, [f"shaders: no {make_game.BVH_DLL} in the game folder, so no BVH builder "
                                     "programs to translate"])
        self.assertEqual(self.run_step(), [])                      # and again, with nothing left to remove
        self.write_dll()
        self.assertEqual(self.run_step(), TOOLS)
        self.assertTrue(os.path.isfile(index) and os.path.isfile(settings))

    def test_each_input_change_runs_it_again(self):
        self.run_step()
        self.assertEqual(self.run_step(signature="llvm-mc 24"), TOOLS)
        self.assertEqual(self.run_step(signature="llvm-mc 24", emulate=True), TOOLS)
        self.assertIn("--wave64-emulate", self.calls[3][1])
        flags = ("--no-float-controls",)
        self.assertEqual(self.run_step(signature="llvm-mc 24", emulate=True, fallbacks=flags), TOOLS)
        self.assertIn("--no-float-controls", self.calls[3][1])
        self.programs["CopyNodes"] = fssm_blob(4)
        self.write_dll()
        self.assertEqual(self.run_step(signature="llvm-mc 24", emulate=True, fallbacks=flags), TOOLS)
        self.assertEqual(self.run_step(signature="llvm-mc 24", emulate=True, fallbacks=flags), [])
        with mock.patch.object(make_game, "translation_sources_digest", return_value="another translator"):
            self.assertEqual(self.run_step(signature="llvm-mc 24", emulate=True, fallbacks=flags), TOOLS)
        with mock.patch.object(make_game, "BVH_MAX_SHARED_MEMORY", 65536):
            self.assertEqual(self.run_step(signature="llvm-mc 24", emulate=True, fallbacks=flags), TOOLS)
            self.assertEqual(self.calls[3][1][4:6], ("--max-shared-memory", 65536))

    def test_an_untranslated_program_is_named_on_every_run_and_the_step_still_finishes(self):
        """The translator has no rule for one program: its name and the counts are logged by the run that
        translates and by every run that reuses the result. The others are published; nothing is skipped quietly."""
        self.untranslated = {"WriteNodes"}
        for expected in (TOOLS, [], []):
            self.assertEqual(self.run_step(), expected)
            self.assertIn(f"shaders: 2 of 3 {make_game.BVH_DLL} programs translated", self.logs)
            named = [line for line in self.logs if "NOT translated: WriteNodes." in line]
            self.assertEqual(len(named), 1, self.logs)
            self.assertIn("E_INVALIDARG", named[0])
            self.assertNotIn("SortKeys", named[0])
        with open(os.path.join(self.spirv, "index.json"), encoding="utf-8") as f:
            self.assertEqual(sorted(json.load(f)), ["CopyNodes", "SortKeys"])
        self.assertFalse(os.path.exists(os.path.join(self.spirv, "WriteNodes.spv")))
        # Once the translator has the rule, the changed translator runs the step again and the line is gone.
        self.untranslated = set()
        with mock.patch.object(make_game, "translation_sources_digest", return_value="a translator with the rule"):
            self.assertEqual(self.run_step(), TOOLS)
            self.assertEqual([line for line in self.logs if "NOT translated" in line], [])
            self.assertIn(f"shaders: 3 of 3 {make_game.BVH_DLL} programs translated", self.logs)

    def test_an_entry_the_runtime_cannot_find_counts_as_not_translated(self):
        """rdna2spirv.py keeps an entry whose metadata failed as {"error": ...}: it has no fssm_sha256, so the
        runtime skips it, and the step reports the program instead of counting its module."""
        real = self.fake_translate

        def no_key(*args):
            real(*args)
            path = os.path.join(self.spirv, "index.json")
            with open(path, encoding="utf-8") as f:
                index = json.load(f)
            index["SortKeys"] = {"error": "ValueError: synthetic"}
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                json.dump(index, f)
        self.fake_translate = no_key
        self.assertEqual(self.run_step(), TOOLS)
        self.assertIn(f"shaders: 2 of 3 {make_game.BVH_DLL} programs translated", self.logs)
        self.assertEqual(len([line for line in self.logs if "NOT translated: SortKeys." in line]), 1)

    def test_nothing_translated_fails_and_keeps_no_index(self):
        self.untranslated = set(self.programs)
        with self.assertRaises(SystemExit), mock.patch("sys.stderr", new_callable=io.StringIO) as err:
            self.run_step()
        self.assertIn(f"none of the {make_game.BVH_DLL} programs translated", err.getvalue())
        self.assertFalse(os.path.exists(os.path.join(self.spirv, "index.json")))   # nothing for the runtime to load
        self.assertFalse(os.path.exists(os.path.join(self.spirv, "bvh_build.json")))
        self.untranslated = set()
        self.assertEqual(self.run_step(), TOOLS)

    def test_missing_cached_module_or_index_is_rebuilt(self):
        self.assertEqual(self.run_step(), TOOLS)
        module = os.path.join(self.spirv, "SortKeys.spv")
        os.remove(module)
        self.assertEqual(self.run_step(), TOOLS)
        with open(module, "rb") as f:
            self.assertEqual(f.read(), b"module of SortKeys")
        self.assertEqual(self.run_step(), [])
        os.remove(os.path.join(self.unpacked, "index.json"))     # the list of programs the report counts against
        self.assertEqual(self.run_step(), TOOLS)
        self.assertEqual(self.run_step(), [])

    def test_a_stopped_translation_leaves_no_marker_and_no_old_modules(self):
        self.run_step()
        open(os.path.join(self.spirv, "notes.txt"), "w").close()

        def stop(tool_name, *args):
            if tool_name == "rdna2spirv.py":
                raise KeyboardInterrupt
            self.fake_py(tool_name, *args)
        with self.assertRaises(KeyboardInterrupt):
            self.run_step(signature="llvm-mc 24", py=stop)
        self.assertFalse(os.path.exists(os.path.join(self.spirv, "index.json")))
        self.assertFalse(os.path.exists(os.path.join(self.spirv, "bvh_build.json")))
        self.assertFalse(os.path.exists(os.path.join(self.spirv, "SortKeys.spv")))   # from the run before
        self.assertTrue(os.path.isfile(os.path.join(self.spirv, "notes.txt")))
        self.assertEqual(self.run_step(signature="llvm-mc 24"), TOOLS)
        self.assertEqual(self.run_step(signature="llvm-mc 24"), [])


class ShadersStageWithTheDlls(unittest.TestCase):
    """The whole shaders() stage with both DLLs in the game folder: the builder's step comes last, and neither it
    nor the FSPX step makes the other or the game's shaders run again."""

    def test_the_builder_runs_after_fspx_and_alone_when_only_its_dll_changes(self):
        from test_fspx_shaders_stage import container as fspx_container, dll_like as fspx_dll, program
        with tempfile.TemporaryDirectory() as t:
            work, game = os.path.join(t, "work"), os.path.join(t, "game")
            os.makedirs(game)
            with open(os.path.join(game, make_game.FSPX_DLL), "wb") as f:
                f.write(fspx_dll(fspx_container(program())))
            dll = os.path.join(game, make_game.BVH_DLL)
            with open(dll, "wb") as f:
                f.write(dll_like(builder_container("SortKeys", fssm_blob(1))))
            ran = []
            dxbc = os.path.join(work, "shaders", "dxbc")
            bvh = os.path.join(work, "shaders", "bvh")

            def write(text, *parts):
                os.makedirs(os.path.dirname(os.path.join(*parts)), exist_ok=True)
                with open(os.path.join(*parts), "w", encoding="utf-8", newline="\n") as f:
                    f.write(text)

            def fake_py(tool_name, *args):
                ran.append(tool_name)
                builder = args[0].startswith(bvh)
                if tool_name == "shader_carve.py":
                    write('[{"file": "shader.dxbc"}]', dxbc, "index.json")
                elif tool_name == "pe_shader_containers.py":
                    write('[{"name": "SortKeys", "file": "SortKeys.dxbc", "stage": "cs"}]', bvh, "index.json")
                elif tool_name == "fssm_unpack.py":
                    write('[{"name": "%s"}]' % ("SortKeys" if builder else "shader"), bvh if builder else dxbc,
                          "fssm_index.json")
                elif tool_name == "shader_disasm.py":
                    # make_game checks that every unpacked shader of the corpus got a non-empty listing
                    write("s_endpgm\n", bvh if builder else dxbc, "disasm",
                          ("SortKeys" if builder else "shader") + ".s")
                elif tool_name == "rdna2spirv.py":
                    out = args[args.index("--out") + 1]
                    # the FSPX step keeps its output only when the index names a program whose module is there
                    name = "SortKeys" if builder else "Upscale"
                    write('{"%s": {"stage": "cs", "fssm_sha256": "%s"}}' % (name, "a" * 64) if builder else
                          '{"%s": {"stage": "cs"}}' % name, out, "index.json")
                    write("", out, name + ".spv")

            def fake_translate(disasm, spirv, jobs, emulate=False, fallbacks=()):
                ran.append("translate")
                write("", spirv, "index.json")

            def stage():
                ran.clear()
                with mock.patch.object(make_game, "py", fake_py), \
                        mock.patch.object(make_game, "need", lambda *a: None), \
                        mock.patch.object(make_game, "translate", fake_translate), \
                        mock.patch.object(make_game, "llvm_mc_signature", return_value="llvm-mc 23"), \
                        mock.patch.object(make_game, "log", lambda *a: None):
                    make_game.shaders(game, work, 2)
                return list(ran)

            self.assertEqual(stage(), ["shader_carve.py", "fssm_unpack.py", "shader_disasm.py", "translate",
                                       "fspx_unpack.py", "rdna2spirv.py", *TOOLS])
            self.assertEqual(stage(), [])
            with open(dll, "wb") as f:
                f.write(dll_like(builder_container("SortKeys", fssm_blob(2))))
            self.assertEqual(stage(), TOOLS)
            with open(os.path.join(game, make_game.FSPX_DLL), "wb") as f:
                f.write(fspx_dll(fspx_container(program() + b"\0\0\x80\xbf")))
            self.assertEqual(stage(), ["fspx_unpack.py", "rdna2spirv.py"])
            for sub in ((), ("fspx",), ("bvh",)):
                self.assertTrue(os.path.isfile(os.path.join(work, "shaders", "spirv", *sub, "index.json")))


if __name__ == "__main__":
    unittest.main()
