"""Expanded stdlib-only unit tests for tools/spirv_metal_set0.py.

Covers bindless capacity parsing from vulkan_bindless.h and custom headers,
metal_ids() layout math, rewrite() across all 26 bindings, non-set-0 /
push-constant / non-runtime-array / pre-sized / raw-ID / error branches,
synthetic SPIR-V binary word header/magic/endianness/bounds handling,
out-of-place vs in-place CLI execution via mocked spirv-dis/spirv-as,
--print-layout, invalid argument handling, and __main__ entry guard.

Standard library only; no spirv-dis/spirv-as/spirv-val binary, no GPU,
and no game data.

    python -I tests/test_spirv_metal_set0_extra.py -v
"""
import contextlib
import io
import os
import re
import runpy
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS_DIR = os.path.join(ROOT, "tools")
TOOL_PATH = os.path.join(TOOLS_DIR, "spirv_metal_set0.py")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import spirv_metal_set0 as m  # noqa: E402


SPIRV_MAGIC = 0x07230203
SPIRV_VERSION_1_0 = 0x00010000
SPIRV_VERSION_1_5 = 0x00010500

OP_MEMORY_MODEL = 14
OP_CAPABILITY = 17
OP_TYPE_VOID = 19
OP_TYPE_INT = 21
OP_TYPE_FLOAT = 22
OP_TYPE_IMAGE = 25
OP_TYPE_SAMPLER = 26
OP_TYPE_ARRAY = 28
OP_TYPE_RUNTIME_ARRAY = 29
OP_TYPE_STRUCT = 30
OP_TYPE_POINTER = 32
OP_CONSTANT = 43
OP_VARIABLE = 59
OP_DECORATE = 71

CAP_SHADER = 1
CAP_RUNTIME_DESCRIPTOR_ARRAY = 5301
ADDR_LOGICAL = 0
MEM_GLSL450 = 1

DEC_BLOCK = 2
DEC_BINDING = 33
DEC_DESCRIPTOR_SET = 34

SC_UNIFORM_CONSTANT = 0
SC_PUSH_CONSTANT = 9
SC_STORAGE_BUFFER = 12

STORAGE_CLASS_BY_NAME = {
    "UniformConstant": SC_UNIFORM_CONSTANT,
    "PushConstant": SC_PUSH_CONSTANT,
    "StorageBuffer": SC_STORAGE_BUFFER,
}
STORAGE_CLASS_BY_CODE = {v: k for k, v in STORAGE_CLASS_BY_NAME.items()}

DIM_BY_CODE = {0: "1D", 1: "2D", 2: "3D"}
DIM_BY_NAME = {v: k for k, v in DIM_BY_CODE.items()}

FMT_BY_CODE = {0: "Unknown", 33: "R32ui"}
FMT_BY_NAME = {v: k for k, v in FMT_BY_CODE.items()}


def _encode_inst(opcode: int, *operands: int) -> list[int]:
    """Encode one SPIR-V instruction into 32-bit binary words."""
    wc = 1 + len(operands)
    return [(wc << 16) | opcode, *operands]


def _pack_spv(
    inst_words: list[int],
    *,
    version: int = SPIRV_VERSION_1_5,
    generator: int = 0,
    bound: int | None = None,
    schema: int = 0,
    endian: str = "<",
) -> bytes:
    """Pack a 5-word SPIR-V header and instruction stream into binary bytes."""
    if bound is None:
        max_id = 0
        pos = 0
        while pos < len(inst_words):
            wc = inst_words[pos] >> 16
            op = inst_words[pos] & 0xFFFF
            if wc < 1 or pos + wc > len(inst_words):
                break
            ops = inst_words[pos + 1 : pos + wc]
            if op in (OP_TYPE_VOID, OP_TYPE_INT, OP_TYPE_FLOAT, OP_TYPE_IMAGE, OP_TYPE_SAMPLER,
                      OP_TYPE_ARRAY, OP_TYPE_RUNTIME_ARRAY, OP_TYPE_STRUCT, OP_TYPE_POINTER):
                max_id = max(max_id, ops[0])
            elif op in (OP_CONSTANT, OP_VARIABLE):
                max_id = max(max_id, ops[1])
            elif op == OP_DECORATE:
                max_id = max(max_id, ops[0])
            pos += wc
        bound = max_id + 1
    words = [SPIRV_MAGIC, version, generator, bound, schema, *inst_words]
    return struct.pack(f"{endian}{len(words)}I", *words)


def _decode_spv(data: bytes) -> tuple[tuple[int, int, int, int, int], list[tuple[int, list[int]]]]:
    """Validate SPIR-V binary header/endianness/word-bounds and decode instructions."""
    if len(data) < 20 or len(data) % 4 != 0:
        raise ValueError(f"invalid SPIR-V byte length: {len(data)}")
    words = list(struct.unpack(f"<{len(data) // 4}I", data))
    magic, version, generator, bound, schema = words[:5]
    if magic == 0x03022307:
        raise ValueError("wrong endianness (big-endian SPIR-V magic)")
    if magic != SPIRV_MAGIC:
        raise ValueError(f"bad SPIR-V magic: {magic:#010x}")
    if bound < 1:
        raise ValueError(f"invalid SPIR-V ID bound: {bound}")
    if schema != 0:
        raise ValueError(f"unsupported non-zero schema: {schema}")
    insts = []
    pos = 5
    while pos < len(words):
        wc = words[pos] >> 16
        op = words[pos] & 0xFFFF
        if wc < 1 or pos + wc > len(words):
            raise ValueError(f"truncated or zero-length instruction at word {pos}")
        insts.append((op, words[pos + 1 : pos + wc]))
        pos += wc
    return (magic, version, generator, bound, schema), insts


def _synthetic_spv_to_disasm(data: bytes) -> str:
    """Decode synthetic SPIR-V binary words into spirv-dis-style assembly text."""
    header, insts = _decode_spv(data)
    _, version, generator, bound, schema = header
    major = (version >> 16) & 0xFF
    minor = (version >> 8) & 0xFF
    lines = [
        "; SPIR-V",
        f"; Version: {major}.{minor}",
        f"; Generator: {generator}",
        f"; Bound: {bound}",
        f"; Schema: {schema}",
    ]
    indent = " " * 15
    def lhs(res_id: int) -> str:
        return f"%{res_id}".rjust(12)

    for op, ops in insts:
        if op == OP_CAPABILITY:
            name = {CAP_SHADER: "Shader", CAP_RUNTIME_DESCRIPTOR_ARRAY: "RuntimeDescriptorArray"}[ops[0]]
            lines.append(f"{indent}OpCapability {name}")
        elif op == OP_MEMORY_MODEL:
            lines.append(f"{indent}OpMemoryModel Logical GLSL450")
        elif op == OP_DECORATE:
            if ops[1] == DEC_DESCRIPTOR_SET:
                lines.append(f"{indent}OpDecorate %{ops[0]} DescriptorSet {ops[2]}")
            elif ops[1] == DEC_BINDING:
                lines.append(f"{indent}OpDecorate %{ops[0]} Binding {ops[2]}")
            elif ops[1] == DEC_BLOCK:
                lines.append(f"{indent}OpDecorate %{ops[0]} Block")
            else:
                raise ValueError(f"unsupported decoration {ops[1]}")
        elif op == OP_TYPE_VOID:
            lines.append(f"{lhs(ops[0])} = OpTypeVoid")
        elif op == OP_TYPE_INT:
            lines.append(f"{lhs(ops[0])} = OpTypeInt {ops[1]} {ops[2]}")
        elif op == OP_TYPE_FLOAT:
            lines.append(f"{lhs(ops[0])} = OpTypeFloat {ops[1]}")
        elif op == OP_TYPE_IMAGE:
            dim = DIM_BY_CODE[ops[2]]
            fmt = FMT_BY_CODE[ops[7]]
            lines.append(
                f"{lhs(ops[0])} = OpTypeImage %{ops[1]} {dim} {ops[3]} {ops[4]} {ops[5]} {ops[6]} {fmt}"
            )
        elif op == OP_TYPE_SAMPLER:
            lines.append(f"{lhs(ops[0])} = OpTypeSampler")
        elif op == OP_TYPE_ARRAY:
            lines.append(f"{lhs(ops[0])} = OpTypeArray %{ops[1]} %{ops[2]}")
        elif op == OP_TYPE_RUNTIME_ARRAY:
            lines.append(f"{lhs(ops[0])} = OpTypeRuntimeArray %{ops[1]}")
        elif op == OP_TYPE_STRUCT:
            members = " ".join(f"%{m_id}" for m_id in ops[1:])
            lines.append(f"{lhs(ops[0])} = OpTypeStruct {members}".rstrip())
        elif op == OP_TYPE_POINTER:
            sc = STORAGE_CLASS_BY_CODE[ops[1]]
            lines.append(f"{lhs(ops[0])} = OpTypePointer {sc} %{ops[2]}")
        elif op == OP_CONSTANT:
            lines.append(f"{lhs(ops[1])} = OpConstant %{ops[0]} {ops[2]}")
        elif op == OP_VARIABLE:
            sc = STORAGE_CLASS_BY_CODE[ops[2]]
            lines.append(f"{lhs(ops[1])} = OpVariable %{ops[0]} {sc}")
        else:
            raise ValueError(f"unsupported opcode {op}")
    return "\n".join(lines) + "\n"


def _synthetic_disasm_to_spv(text: str, target_env: str) -> bytes:
    """Assemble synthetic spirv-dis text (including rewritten %msl_* IDs) into SPIR-V words."""
    id_map: dict[str, int] = {}
    next_id = 1

    def resolve(tok: str) -> int:
        nonlocal next_id
        if tok in id_map:
            return id_map[tok]
        if tok[1:].isdigit():
            val = int(tok[1:])
            id_map[tok] = val
            next_id = max(next_id, val + 1)
            return val
        val = next_id
        next_id += 1
        id_map[tok] = val
        return val

    # Pre-register numeric IDs so newly synthesized %msl_* names never collide.
    for tok in re.findall(r"%\S+", text):
        if tok[1:].isdigit():
            resolve(tok)

    inst_words: list[int] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith("OpCapability "):
            cap = {"Shader": CAP_SHADER, "RuntimeDescriptorArray": CAP_RUNTIME_DESCRIPTOR_ARRAY}[line.split()[1]]
            inst_words.extend(_encode_inst(OP_CAPABILITY, cap))
        elif line == "OpMemoryModel Logical GLSL450":
            inst_words.extend(_encode_inst(OP_MEMORY_MODEL, ADDR_LOGICAL, MEM_GLSL450))
        elif line.startswith("OpDecorate "):
            parts = line.split()
            tid = resolve(parts[1])
            if parts[2] == "DescriptorSet":
                inst_words.extend(_encode_inst(OP_DECORATE, tid, DEC_DESCRIPTOR_SET, int(parts[3])))
            elif parts[2] == "Binding":
                inst_words.extend(_encode_inst(OP_DECORATE, tid, DEC_BINDING, int(parts[3])))
            elif parts[2] == "Block":
                inst_words.extend(_encode_inst(OP_DECORATE, tid, DEC_BLOCK))
            else:
                raise ValueError(f"unsupported OpDecorate: {line}")
        else:
            lhs, rhs = [p.strip() for p in line.split("=", 1)]
            rid = resolve(lhs)
            parts = rhs.split()
            opname = parts[0]
            if opname == "OpTypeVoid":
                inst_words.extend(_encode_inst(OP_TYPE_VOID, rid))
            elif opname == "OpTypeInt":
                inst_words.extend(_encode_inst(OP_TYPE_INT, rid, int(parts[1]), int(parts[2])))
            elif opname == "OpTypeFloat":
                inst_words.extend(_encode_inst(OP_TYPE_FLOAT, rid, int(parts[1])))
            elif opname == "OpTypeImage":
                inst_words.extend(_encode_inst(
                    OP_TYPE_IMAGE,
                    rid,
                    resolve(parts[1]),
                    DIM_BY_NAME[parts[2]],
                    int(parts[3]),
                    int(parts[4]),
                    int(parts[5]),
                    int(parts[6]),
                    FMT_BY_NAME[parts[7]],
                ))
            elif opname == "OpTypeSampler":
                inst_words.extend(_encode_inst(OP_TYPE_SAMPLER, rid))
            elif opname == "OpTypeArray":
                inst_words.extend(_encode_inst(OP_TYPE_ARRAY, rid, resolve(parts[1]), resolve(parts[2])))
            elif opname == "OpTypeRuntimeArray":
                inst_words.extend(_encode_inst(OP_TYPE_RUNTIME_ARRAY, rid, resolve(parts[1])))
            elif opname == "OpTypeStruct":
                inst_words.extend(_encode_inst(OP_TYPE_STRUCT, rid, *[resolve(p) for p in parts[1:]]))
            elif opname == "OpTypePointer":
                inst_words.extend(_encode_inst(OP_TYPE_POINTER, rid, STORAGE_CLASS_BY_NAME[parts[1]], resolve(parts[2])))
            elif opname == "OpConstant":
                inst_words.extend(_encode_inst(OP_CONSTANT, resolve(parts[1]), rid, int(parts[2])))
            elif opname == "OpVariable":
                inst_words.extend(_encode_inst(OP_VARIABLE, resolve(parts[1]), rid, STORAGE_CLASS_BY_NAME[parts[2]]))
            else:
                raise ValueError(f"unsupported op: {line}")

    ver = SPIRV_VERSION_1_5 if target_env == "spv1.5" else SPIRV_VERSION_1_0
    return _pack_spv(inst_words, version=ver, bound=next_id)


def _build_mixed_bindless_spv_words() -> list[int]:
    """Build synthetic SPIR-V instruction words with set-0, set-1, and PushConstant resources."""
    # IDs:
    # 1=uint, 2=float, 3=img2d, 4=smp, 5=rt_img, 6=rt_smp, 7=ptr_img, 8=ptr_smp
    # 9=rt_map, 10=struct_map, 11=ptr_map, 12=ptr_pc_u32
    # 13=images (set 0, b0), 14=samplers (set 0, b1), 15=desc_map (set 0, b2)
    # 16=push_consts, 17=set1_var (set 1, b0)
    words: list[int] = []
    words.extend(_encode_inst(OP_CAPABILITY, CAP_SHADER))
    words.extend(_encode_inst(OP_CAPABILITY, CAP_RUNTIME_DESCRIPTOR_ARRAY))
    words.extend(_encode_inst(OP_MEMORY_MODEL, ADDR_LOGICAL, MEM_GLSL450))
    words.extend(_encode_inst(OP_DECORATE, 13, DEC_DESCRIPTOR_SET, 0))
    words.extend(_encode_inst(OP_DECORATE, 13, DEC_BINDING, 0))
    words.extend(_encode_inst(OP_DECORATE, 14, DEC_DESCRIPTOR_SET, 0))
    words.extend(_encode_inst(OP_DECORATE, 14, DEC_BINDING, 1))
    words.extend(_encode_inst(OP_DECORATE, 15, DEC_DESCRIPTOR_SET, 0))
    words.extend(_encode_inst(OP_DECORATE, 15, DEC_BINDING, 2))
    words.extend(_encode_inst(OP_DECORATE, 10, DEC_BLOCK))
    words.extend(_encode_inst(OP_DECORATE, 17, DEC_DESCRIPTOR_SET, 1))
    words.extend(_encode_inst(OP_DECORATE, 17, DEC_BINDING, 0))
    words.extend(_encode_inst(OP_TYPE_INT, 1, 32, 0))
    words.extend(_encode_inst(OP_TYPE_FLOAT, 2, 32))
    words.extend(_encode_inst(OP_TYPE_IMAGE, 3, 2, 1, 0, 0, 0, 1, 0))
    words.extend(_encode_inst(OP_TYPE_SAMPLER, 4))
    words.extend(_encode_inst(OP_TYPE_RUNTIME_ARRAY, 5, 3))
    words.extend(_encode_inst(OP_TYPE_RUNTIME_ARRAY, 6, 4))
    words.extend(_encode_inst(OP_TYPE_POINTER, 7, SC_UNIFORM_CONSTANT, 5))
    words.extend(_encode_inst(OP_TYPE_POINTER, 8, SC_UNIFORM_CONSTANT, 6))
    words.extend(_encode_inst(OP_TYPE_RUNTIME_ARRAY, 9, 1))
    words.extend(_encode_inst(OP_TYPE_STRUCT, 10, 9))
    words.extend(_encode_inst(OP_TYPE_POINTER, 11, SC_STORAGE_BUFFER, 10))
    words.extend(_encode_inst(OP_TYPE_POINTER, 12, SC_PUSH_CONSTANT, 1))
    words.extend(_encode_inst(OP_VARIABLE, 7, 13, SC_UNIFORM_CONSTANT))
    words.extend(_encode_inst(OP_VARIABLE, 8, 14, SC_UNIFORM_CONSTANT))
    words.extend(_encode_inst(OP_VARIABLE, 11, 15, SC_STORAGE_BUFFER))
    words.extend(_encode_inst(OP_VARIABLE, 12, 16, SC_PUSH_CONSTANT))
    words.extend(_encode_inst(OP_VARIABLE, 7, 17, SC_UNIFORM_CONSTANT))
    return words


def _make_fake_spirv_tools(dis_name="spirv-dis", asm_name="spirv-as"):
    """Return a subprocess.run replacement backed by the synthetic SPIR-V binary codec."""
    calls: list[list[str]] = []

    def fake_run(cmd, check=False, capture_output=False, text=False, input=None, errors=None):
        calls.append(list(cmd))
        exe = cmd[0]
        if exe == dis_name:
            in_path = cmd[1]
            try:
                with open(in_path, "rb") as f:
                    data = f.read()
                disasm = _synthetic_spv_to_disasm(data)
            except (OSError, ValueError) as exc:
                if check:
                    raise subprocess.CalledProcessError(1, cmd, output="", stderr=str(exc)) from exc
                return subprocess.CompletedProcess(cmd, 1, "", str(exc))
            return subprocess.CompletedProcess(cmd, 0, disasm, "")
        if exe == asm_name:
            # Expect: [asm, "--target-env", target, "-", "-o", out_path]
            target_env = cmd[cmd.index("--target-env") + 1]
            out_path = cmd[cmd.index("-o") + 1]
            try:
                out_bytes = _synthetic_disasm_to_spv(input, target_env)
                with open(out_path, "wb") as f:
                    f.write(out_bytes)
            except (OSError, ValueError) as exc:
                if check:
                    raise subprocess.CalledProcessError(1, cmd, output="", stderr=str(exc)) from exc
                return subprocess.CompletedProcess(cmd, 1, "", str(exc))
            return subprocess.CompletedProcess(cmd, 0, "", "")
        raise AssertionError(f"unexpected subprocess command: {cmd!r}")

    return fake_run, calls


class TestCapacities(unittest.TestCase):
    def test_default_bindless_header_all_26_entries_and_cross_check(self):
        counts = m.capacities()
        expected = [4096, 1024, 1, 1024, 1024] + [256] * 21
        self.assertEqual(counts, expected)
        self.assertEqual(len(counts), 26)
        self.assertEqual(sum(counts), 12545)

        with open(m.BINDLESS_H, encoding="utf-8") as f:
            hdr = f.read()
        self.assertIn("inline constexpr std::uint32_t binding_count = 26;", hdr)
        self.assertIn("inline constexpr std::array<Binding, 26> bindings =", hdr)

    def test_custom_header_parsing_variants(self):
        cases = [
            ("empty_table", "std::array<Binding, 0> bindings = {{}};\n", []),
            (
                "single_line",
                "std::array<Binding, 1> bindings = {{{VK_DESCRIPTOR_TYPE_SAMPLER, 7}}};\n",
                [7],
            ),
            (
                "multiline_mixed_types_and_comments",
                """
                // unrelated header prefix
                inline constexpr std::array<Binding,  4>   bindings   = {{
                    {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 128},
                    {VK_DESCRIPTOR_TYPE_SAMPLER,   32},
                    {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
                    {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 64},
                }};
                """,
                [128, 32, 1, 64],
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            for name, body, expected in cases:
                with self.subTest(case=name):
                    p = os.path.join(tmp, f"{name}.h")
                    with open(p, "w", encoding="utf-8") as f:
                        f.write(body)
                    self.assertEqual(m.capacities(p), expected)

    def test_missing_file_raises_file_not_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "missing_bindless.h")
            with self.assertRaises(FileNotFoundError):
                m.capacities(missing)

    def test_missing_or_mismatched_bindings_table_raises_system_exit(self):
        bad_headers = [
            ("empty_file", "", "no `bindings` table"),
            ("wrong_type_name", "std::array<int, 1> bindings = {{ {VK_DESCRIPTOR_TYPE_SAMPLER, 4} }};", "no `bindings` table"),
            (
                "too_few_entries",
                "std::array<Binding, 3> bindings = {{ {VK_DESCRIPTOR_TYPE_SAMPLER, 1}, {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 2} }};",
                "`bindings` declares 3 entries, found 2",
            ),
            (
                "too_many_entries",
                "std::array<Binding, 1> bindings = {{ {VK_DESCRIPTOR_TYPE_SAMPLER, 1}, {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 2} }};",
                "`bindings` declares 1 entries, found 2",
            ),
            (
                "unmatched_entry_syntax",
                "std::array<Binding, 1> bindings = {{ {NOT_A_VK_DESCRIPTOR, 1} }};",
                "`bindings` declares 1 entries, found 0",
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            for name, body, msg_part in bad_headers:
                with self.subTest(case=name):
                    p = os.path.join(tmp, f"{name}.h")
                    with open(p, "w", encoding="utf-8") as f:
                        f.write(body)
                    with self.assertRaises(SystemExit) as ctx:
                        m.capacities(p)
                    self.assertIn(p, str(ctx.exception))
                    self.assertIn(msg_part, str(ctx.exception))


class TestMetalIds(unittest.TestCase):
    def test_empty_single_and_zero_capacities(self):
        self.assertEqual(m.metal_ids([]), [])
        self.assertEqual(m.metal_ids([42]), [0])
        self.assertEqual(m.metal_ids([10, 0, 5, 0, 1]), [0, 10, 10, 15, 15])

    def test_vulkan_bindless_ids_contiguous_and_non_overlapping(self):
        counts = m.capacities()
        ids = m.metal_ids(counts)
        self.assertEqual(len(ids), 26)
        self.assertEqual(ids[0], 0)
        self.assertEqual(ids[:6], [0, 4096, 5120, 5121, 6145, 7169])
        self.assertEqual(ids[8], 7937)
        self.assertEqual(ids[20], 11009)
        self.assertEqual(ids[25], 12289)
        for b in range(len(counts) - 1):
            self.assertEqual(ids[b + 1], ids[b] + counts[b])
        self.assertEqual(ids[-1] + counts[-1], sum(counts))

        seen_total = 0
        for b in range(len(counts)):
            if b > 0:
                self.assertGreater(ids[b], ids[b - 1] + counts[b - 1] - 1)
            seen_total += counts[b]
        self.assertEqual(seen_total, 12545)


class TestRewrite(unittest.TestCase):
    def test_all_26_vulkan_bindless_bindings(self):
        counts = m.capacities()
        ids = m.metal_ids(counts)
        lines = [
            "               OpCapability Shader",
            "               OpCapability RuntimeDescriptorArray",
            "               OpMemoryModel Logical GLSL450",
        ]
        for b in range(26):
            lines.append(f"               OpDecorate %v{b} DescriptorSet 0")
            lines.append(f"               OpDecorate %v{b} Binding {b}")
        lines.extend([
            "       %uint = OpTypeInt 32 0",
            "      %float = OpTypeFloat 32",
            "      %img2d = OpTypeImage %float 2D 0 0 0 1 Unknown",
            "        %smp = OpTypeSampler",
            "     %rt_map = OpTypeRuntimeArray %uint",
            "        %map = OpTypeStruct %rt_map",
            "    %ptr_map = OpTypePointer StorageBuffer %map",
        ])
        for b in range(26):
            if b == 2:
                continue
            elem = "%smp" if b == 1 else "%img2d"
            lines.append(f"      %rt_b{b} = OpTypeRuntimeArray {elem}")
            lines.append(f"     %ptr_b{b} = OpTypePointer UniformConstant %rt_b{b}")
        for b in range(26):
            if b == 2:
                lines.append("         %v2 = OpVariable %ptr_map StorageBuffer")
            else:
                lines.append(f"         %v{b} = OpVariable %ptr_b{b} UniformConstant")

        out = m.rewrite("\n".join(lines) + "\n", counts)
        out_lines = out.splitlines()

        for b in range(26):
            self.assertIn(f"               OpDecorate %v{b} Binding {ids[b]}", out_lines)
            if b == 2:
                self.assertIn("         %v2 = OpVariable %ptr_map StorageBuffer", out_lines)
                self.assertNotIn("%msl_cap_b2 ", out)
                self.assertNotIn("%msl_arr_b2 ", out)
                self.assertNotIn("%msl_ptr_b2 ", out)
            else:
                elem = "%smp" if b == 1 else "%img2d"
                cap_line = f"%msl_cap_b{b} = OpConstant %uint {counts[b]}"
                arr_line = f"%msl_arr_b{b} = OpTypeArray {elem} %msl_cap_b{b}"
                ptr_line = f"%msl_ptr_b{b} = OpTypePointer UniformConstant %msl_arr_b{b}"
                orig_ptr_line = f"     %ptr_b{b} = OpTypePointer UniformConstant %rt_b{b}"
                idx = out_lines.index(orig_ptr_line)
                self.assertEqual(out_lines[idx + 1 : idx + 4], [cap_line, arr_line, ptr_line])
                self.assertIn(f"         %v{b} = OpVariable %msl_ptr_b{b} UniformConstant", out_lines)

    def test_push_constants_and_other_descriptor_sets_untouched(self):
        module = """\
               OpDecorate %set0_img DescriptorSet 0
               OpDecorate %set0_img Binding 0
               OpDecorate %set0_smp DescriptorSet 0
               OpDecorate %set0_smp Binding 1
               OpDecorate %set1_in DescriptorSet 1
               OpDecorate %set1_in Binding 0
               OpDecorate %set1_high DescriptorSet 1
               OpDecorate %set1_high Binding 99
               OpDecorate %set2_uc DescriptorSet 2
               OpDecorate %set2_uc Binding 0
       %uint = OpTypeInt 32 0
      %float = OpTypeFloat 32
      %img2d = OpTypeImage %float 2D 0 0 0 1 Unknown
        %smp = OpTypeSampler
     %rt_img = OpTypeRuntimeArray %img2d
     %rt_smp = OpTypeRuntimeArray %smp
    %ptr_img = OpTypePointer UniformConstant %rt_img
    %ptr_smp = OpTypePointer UniformConstant %rt_smp
     %ptr_pc = OpTypePointer PushConstant %uint
   %ptr_s1hi = OpTypePointer UniformConstant %rt_img
   %ptr_s2uc = OpTypePointer UniformConstant %rt_img
   %set0_img = OpVariable %ptr_img UniformConstant
   %set0_smp = OpVariable %ptr_smp UniformConstant
  %user_data = OpVariable %ptr_pc PushConstant
    %set1_in = OpVariable %ptr_img StorageBuffer
  %set1_high = OpVariable %ptr_s1hi UniformConstant
    %set2_uc = OpVariable %ptr_s2uc UniformConstant
"""
        out = m.rewrite(module, [8, 4])
        self.assertIn("OpDecorate %set0_img Binding 0", out)
        self.assertIn("OpDecorate %set0_smp Binding 8", out)
        # Set 1 (including binding 99 >= len(counts)), Set 2, and PushConstant stay untouched.
        self.assertIn("OpDecorate %set1_in DescriptorSet 1", out)
        self.assertIn("OpDecorate %set1_in Binding 0", out)
        self.assertIn("OpDecorate %set1_high DescriptorSet 1", out)
        self.assertIn("OpDecorate %set1_high Binding 99", out)
        self.assertIn("OpDecorate %set2_uc DescriptorSet 2", out)
        self.assertIn("OpDecorate %set2_uc Binding 0", out)
        self.assertIn("%user_data = OpVariable %ptr_pc PushConstant", out)
        self.assertIn("%set1_in = OpVariable %ptr_img StorageBuffer", out)
        self.assertIn("%set1_high = OpVariable %ptr_s1hi UniformConstant", out)
        self.assertIn("%set2_uc = OpVariable %ptr_s2uc UniformConstant", out)

    def test_non_runtime_array_set0_resources_rebind_without_retyping_or_uint32(self):
        # No OpTypeInt 32 0 is present, yet none of these set-0 bindings use OpTypeRuntimeArray
        # under OpTypePointer UniformConstant, so rewriting succeeds without raising.
        module = """\
               OpDecorate %single DescriptorSet 0
               OpDecorate %single Binding 0
               OpDecorate %fixed DescriptorSet 0
               OpDecorate %fixed Binding 1
               OpDecorate %ubo DescriptorSet 0
               OpDecorate %ubo Binding 2
               OpDecorate %mismatch DescriptorSet 0
               OpDecorate %mismatch Binding 3
               OpDecorate %dangling DescriptorSet 0
               OpDecorate %dangling Binding 4
      %float = OpTypeFloat 32
      %img2d = OpTypeImage %float 2D 0 0 0 1 Unknown
       %sint = OpTypeInt 32 1
        %c16 = OpConstant %sint 16
  %fixed_arr = OpTypeArray %img2d %c16
    %ubo_blk = OpTypeStruct %float
 %ptr_single = OpTypePointer UniformConstant %img2d
  %ptr_fixed = OpTypePointer UniformConstant %fixed_arr
    %ptr_ubo = OpTypePointer Uniform %ubo_blk
 %ptr_mis_sb = OpTypePointer StorageBuffer %fixed_arr
%ptr_dangling = OpTypePointer UniformConstant %missing_pointee
     %single = OpVariable %ptr_single UniformConstant
      %fixed = OpVariable %ptr_fixed UniformConstant
        %ubo = OpVariable %ptr_ubo Uniform
   %mismatch = OpVariable %ptr_mis_sb UniformConstant
   %dangling = OpVariable %ptr_dangling UniformConstant
"""
        counts = [10, 20, 1, 5, 2]
        out = m.rewrite(module, counts)
        self.assertIn("OpDecorate %single Binding 0", out)
        self.assertIn("OpDecorate %fixed Binding 10", out)
        self.assertIn("OpDecorate %ubo Binding 30", out)
        self.assertIn("OpDecorate %mismatch Binding 31", out)
        self.assertIn("OpDecorate %dangling Binding 36", out)
        self.assertIn("%single = OpVariable %ptr_single UniformConstant", out)
        self.assertIn("%fixed = OpVariable %ptr_fixed UniformConstant", out)
        self.assertIn("%ubo = OpVariable %ptr_ubo Uniform", out)
        self.assertIn("%mismatch = OpVariable %ptr_mis_sb UniformConstant", out)
        self.assertIn("%dangling = OpVariable %ptr_dangling UniformConstant", out)
        self.assertNotIn("%msl_", out)

    def test_partial_decorations_and_missing_var_def(self):
        module = """\
OpDecorate %only_set DescriptorSet 0
OpDecorate %only_bind Binding 1
OpDecorate %orphan DescriptorSet 0
OpDecorate %orphan Binding 1
%u = OpTypeInt 32 0
"""
        out = m.rewrite(module, [5, 7])
        self.assertIn("OpDecorate %only_set DescriptorSet 0", out)
        self.assertIn("OpDecorate %only_bind Binding 1", out)
        self.assertIn("OpDecorate %orphan Binding 5", out)
        self.assertNotIn("%msl_", out)

    def test_raw_numeric_ids_custom_uint_and_crlf_and_shared_pointer(self):
        module = (
            "; Header comment\r\n"
            "OpName %10 \"images\"\r\n"
            "OpDecorate %10 DescriptorSet 0\r\n"
            "OpDecorate %10 Binding 1\r\n"
            "OpDecorate %11 DescriptorSet 0\r\n"
            "OpDecorate %11 Binding 0\r\n"
            "%1 = OpTypeVoid\r\n"
            "%2 = OpTypeInt 32 1\r\n"
            "%3 = OpTypeInt 64 0\r\n"
            "%4 = OpTypeInt 32 0\r\n"
            "%5 = OpTypeFloat 32\r\n"
            "%6 = OpTypeImage %5 2D 0 0 0 1 Unknown\r\n"
            "%7 = OpTypeRuntimeArray %6\r\n"
            "%8 = OpTypePointer UniformConstant %7\r\n"
            "%9 = OpTypePointer Function %5\r\n"
            "%10 = OpVariable %8 UniformConstant\r\n"
            "%11 = OpVariable %8 UniformConstant\r\n"
        )
        out = m.rewrite(module, [16, 8])
        self.assertNotIn("\r\n", out)
        self.assertTrue(out.endswith("\n"))
        self.assertIn("OpDecorate %11 Binding 0", out)
        self.assertIn("OpDecorate %10 Binding 16", out)
        # First OpTypeInt 32 0 is %4 (not %2 signed or %3 64-bit).
        self.assertIn("%msl_cap_b0 = OpConstant %4 16", out)
        self.assertIn("%msl_cap_b1 = OpConstant %4 8", out)
        self.assertIn("%11 = OpVariable %msl_ptr_b0 UniformConstant", out)
        self.assertIn("%10 = OpVariable %msl_ptr_b1 UniformConstant", out)
        # Sorted binding order (b0 then b1) inserted immediately after %8, not after %9.
        lines = out.splitlines()
        idx8 = lines.index("%8 = OpTypePointer UniformConstant %7")
        self.assertEqual(lines[idx8 + 1 : idx8 + 7], [
            "%msl_cap_b0 = OpConstant %4 16",
            "%msl_arr_b0 = OpTypeArray %6 %msl_cap_b0",
            "%msl_ptr_b0 = OpTypePointer UniformConstant %msl_arr_b0",
            "%msl_cap_b1 = OpConstant %4 8",
            "%msl_arr_b1 = OpTypeArray %6 %msl_cap_b1",
            "%msl_ptr_b1 = OpTypePointer UniformConstant %msl_arr_b1",
        ])
        idx9 = lines.index("%9 = OpTypePointer Function %5")
        self.assertEqual(lines[idx9 + 1], "%10 = OpVariable %msl_ptr_b1 UniformConstant")

    def test_empty_and_no_set0_modules(self):
        self.assertEqual(m.rewrite("", [4]), "\n")
        plain = "OpCapability Shader\nOpMemoryModel Logical GLSL450\n"
        self.assertEqual(m.rewrite(plain, [4]), plain)

    def test_out_of_bounds_set0_binding_raises_system_exit(self):
        for b, counts in ((2, [10, 20]), (26, m.capacities()), (100, m.capacities())):
            with self.subTest(binding=b, count_len=len(counts)):
                text = f"OpDecorate %bad DescriptorSet 0\nOpDecorate %bad Binding {b}\n"
                with self.assertRaises(SystemExit) as ctx:
                    m.rewrite(text, counts)
                self.assertIn("%bad", str(ctx.exception))
                self.assertIn(f"binding {b}", str(ctx.exception))

    def test_missing_uint32_type_raises_when_runtime_array_rewritten(self):
        base = """\
OpDecorate %images DescriptorSet 0
OpDecorate %images Binding 0
{int_decl}
%float = OpTypeFloat 32
%img2d = OpTypeImage %float 2D 0 0 0 1 Unknown
%rt_img = OpTypeRuntimeArray %img2d
%ptr_img = OpTypePointer UniformConstant %rt_img
%images = OpVariable %ptr_img UniformConstant
"""
        for label, int_decl in (
            ("no_int", ""),
            ("signed_32", "%s32 = OpTypeInt 32 1"),
            ("unsigned_64", "%u64 = OpTypeInt 64 0"),
        ):
            with self.subTest(case=label):
                with self.assertRaises(SystemExit) as ctx:
                    m.rewrite(base.format(int_decl=int_decl), [4096])
                self.assertIn("no 32-bit uint type", str(ctx.exception))


class TestMainAndCli(unittest.TestCase):
    def _call_main(self, argv):
        out, err = io.StringIO(), io.StringIO()
        # tools/spirv_metal_set0.py:31 uses open(path).read() without a context manager, which emits
        # ResourceWarning under unittest's default filter on redirected stderr; isolate tool stderr here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = m.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_print_layout_outputs_26_formatted_ranges_without_subprocess(self):
        with mock.patch.object(m.subprocess, "run") as run_mock:
            code, out, err = self._call_main(["spirv_metal_set0.py", "--print-layout"])
        run_mock.assert_not_called()
        self.assertEqual(code, 0)
        self.assertEqual(err, "")
        lines = out.splitlines()
        self.assertEqual(len(lines), 26)
        self.assertEqual(lines[0], "binding  0:  4096 descriptors at IDs 0..4095")
        self.assertEqual(lines[1], "binding  1:  1024 descriptors at IDs 4096..5119")
        self.assertEqual(lines[2], "binding  2:     1 descriptors at IDs 5120..5120")
        self.assertEqual(lines[3], "binding  3:  1024 descriptors at IDs 5121..6144")
        self.assertEqual(lines[4], "binding  4:  1024 descriptors at IDs 6145..7168")
        self.assertEqual(lines[25], "binding 25:   256 descriptors at IDs 12289..12544")

    def test_print_layout_uses_capacities_table(self):
        with mock.patch.object(m, "capacities", return_value=[3, 1, 2]):
            code, out, err = self._call_main(["spirv_metal_set0.py", "--print-layout"])
        self.assertEqual(code, 0)
        self.assertEqual(err, "")
        self.assertEqual(out.splitlines(), [
            "binding  0:     3 descriptors at IDs 0..2",
            "binding  1:     1 descriptors at IDs 3..3",
            "binding  2:     2 descriptors at IDs 4..5",
        ])

    def test_invalid_argc_returns_2_and_prints_usage_to_stderr(self):
        bad_argvs = [
            [],
            ["spirv_metal_set0.py"],
            ["spirv_metal_set0.py", "in.spv"],
            ["spirv_metal_set0.py", "--unknown"],
            ["spirv_metal_set0.py", "--print-layout", "in.spv", "out.spv"],
            ["spirv_metal_set0.py", "in.spv", "out.spv", "extra"],
        ]
        for argv in bad_argvs:
            with self.subTest(argv=argv), mock.patch.object(m.subprocess, "run") as run_mock:
                code, out, err = self._call_main(argv)
                run_mock.assert_not_called()
                self.assertEqual(code, 2)
                self.assertEqual(out, "")
                self.assertIn("Usage: spirv_metal_set0.py in.spv out.spv", err)
                self.assertIn("--print-layout", err)

    def test_print_layout_with_extra_arg_falls_through_to_two_arg_mode(self):
        with mock.patch.object(
            m.subprocess,
            "run",
            side_effect=[
                subprocess.CompletedProcess([], 0, "OpCapability Shader\n", ""),
                subprocess.CompletedProcess([], 0, "", ""),
            ],
        ) as run_mock:
            code, out, err = self._call_main(["spirv_metal_set0.py", "--print-layout", "out.spv"])
        self.assertEqual((code, out, err), (0, "", ""))
        self.assertEqual(run_mock.call_args_list[0].args[0], ["spirv-dis", "--print-layout"])
        self.assertEqual(
            run_mock.call_args_list[1].args[0],
            ["spirv-as", "--target-env", "vulkan1.2", "-", "-o", "out.spv"],
        )

    def test_cli_subprocess_isolated_print_layout_and_bad_args(self):
        r_layout = subprocess.run(
            [sys.executable, "-I", TOOL_PATH, "--print-layout"],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        self.assertEqual(r_layout.returncode, 0, r_layout.stderr)
        self.assertEqual(r_layout.stderr, "")
        self.assertEqual(len(r_layout.stdout.splitlines()), 26)
        self.assertEqual(r_layout.stdout.splitlines()[0], "binding  0:  4096 descriptors at IDs 0..4095")

        for args in ([], ["only_one.spv"], ["a.spv", "b.spv", "c.spv"]):
            with self.subTest(args=args):
                r = subprocess.run(
                    [sys.executable, "-I", TOOL_PATH, *args],
                    capture_output=True,
                    text=True,
                    cwd=ROOT,
                )
                self.assertEqual(r.returncode, 2)
                self.assertEqual(r.stdout, "")
                self.assertIn("Usage: spirv_metal_set0.py in.spv out.spv", r.stderr)
                self.assertNotIn("Traceback", r.stderr)

    def test_synthetic_spv_out_of_place_and_in_place_binary_round_trip(self):
        in_bytes = _pack_spv(_build_mixed_bindless_spv_words(), version=SPIRV_VERSION_1_5)
        fake_run, calls = _make_fake_spirv_tools()

        with tempfile.TemporaryDirectory() as tmp:
            unicode_dir = os.path.join(tmp, "dir-\u00e9-\u2603")
            os.makedirs(unicode_dir)
            in_path = os.path.join(unicode_dir, "in.spv")
            out_path = os.path.join(unicode_dir, "out.spv")
            with open(in_path, "wb") as f:
                f.write(in_bytes)

            with mock.patch.dict(os.environ, {}, clear=False), \
                    mock.patch.object(m.subprocess, "run", side_effect=fake_run):
                os.environ.pop("SPIRV_DIS", None)
                os.environ.pop("SPIRV_AS", None)
                code, out, err = self._call_main(["spirv_metal_set0.py", in_path, out_path])

            self.assertEqual(code, 0)
            self.assertEqual(out, "")
            self.assertEqual(err, "")
            self.assertEqual(calls, [
                ["spirv-dis", in_path],
                ["spirv-as", "--target-env", "spv1.5", "-", "-o", out_path],
            ])

            # Original input file is preserved untouched for out-of-place conversion.
            with open(in_path, "rb") as f:
                self.assertEqual(f.read(), in_bytes)

            with open(out_path, "rb") as f:
                out_bytes = f.read()
            header, insts = _decode_spv(out_bytes)
            self.assertEqual(header[0], SPIRV_MAGIC)
            self.assertEqual(header[1], SPIRV_VERSION_1_5)
            self.assertEqual(header[4], 0)

            decs = [ops for op, ops in insts if op == OP_DECORATE]
            bindings_by_id = {ops[0]: ops[2] for ops in decs if ops[1] == DEC_BINDING}
            sets_by_id = {ops[0]: ops[2] for ops in decs if ops[1] == DEC_DESCRIPTOR_SET}
            self.assertEqual(sets_by_id, {13: 0, 14: 0, 15: 0, 17: 1})
            # Set-0 bindings rewritten per vulkan_bindless.h; set-1 binding 0 stays 0.
            self.assertEqual(bindings_by_id, {13: 0, 14: 4096, 15: 5120, 17: 0})

            consts = {ops[1]: ops[2] for op, ops in insts if op == OP_CONSTANT}
            arrays = {ops[0]: (ops[1], consts[ops[2]]) for op, ops in insts if op == OP_TYPE_ARRAY}
            ptrs = {ops[0]: (ops[1], ops[2]) for op, ops in insts if op == OP_TYPE_POINTER}
            vars_by_id = {ops[1]: (ops[0], ops[2]) for op, ops in insts if op == OP_VARIABLE}

            # %13 (images) and %14 (samplers) are retyped to sized arrays [4096] and [1024];
            # %15 (desc_map StorageBuffer), %16 (PushConstant), and %17 (set 1) keep original types.
            self.assertEqual(arrays[ptrs[vars_by_id[13][0]][1]], (3, 4096))
            self.assertEqual(arrays[ptrs[vars_by_id[14][0]][1]], (4, 1024))
            self.assertEqual(vars_by_id[15], (11, SC_STORAGE_BUFFER))
            self.assertEqual(vars_by_id[16], (12, SC_PUSH_CONSTANT))
            self.assertEqual(vars_by_id[17], (7, SC_UNIFORM_CONSTANT))

            # In-place conversion: argv[1] == argv[2].
            inplace_path = os.path.join(unicode_dir, "inplace.spv")
            with open(inplace_path, "wb") as f:
                f.write(in_bytes)
            calls.clear()
            with mock.patch.object(m.subprocess, "run", side_effect=fake_run):
                code, out, err = self._call_main(["spirv_metal_set0.py", inplace_path, inplace_path])
            self.assertEqual(code, 0)
            self.assertEqual(out, "")
            self.assertEqual(err, "")
            self.assertEqual(calls, [
                ["spirv-dis", inplace_path],
                ["spirv-as", "--target-env", "spv1.5", "-", "-o", inplace_path],
            ])
            with open(inplace_path, "rb") as f:
                self.assertEqual(f.read(), out_bytes)

    def test_target_env_extraction_and_env_var_overrides(self):
        cases = [
            ("; Version: 1.0\n", "spv1.0"),
            ("; SPIR-V\n; Version: 1.5\n", "spv1.5"),
            ("; Version: 1.6\n", "spv1.6"),
            ("; SPIR-V without version comment\n", "vulkan1.2"),
        ]
        for header_text, expected_target in cases:
            with self.subTest(expected_target=expected_target):
                disasm = header_text + "OpCapability Shader\n"
                seen_cmds = []

                def run_stub(cmd, check=False, capture_output=False, text=False, input=None, errors=None):
                    seen_cmds.append((list(cmd), check, capture_output, text, input))
                    if len(seen_cmds) == 1:
                        return subprocess.CompletedProcess(cmd, 0, disasm, "")
                    return subprocess.CompletedProcess(cmd, 0, "", "")

                with mock.patch.dict(os.environ, {"SPIRV_DIS": "/opt/bin/dis", "SPIRV_AS": "/opt/bin/as"}), \
                        mock.patch.object(m.subprocess, "run", side_effect=run_stub):
                    code, out, err = self._call_main(["spirv_metal_set0.py", "in.spv", "out.spv"])

                self.assertEqual((code, out, err), (0, "", ""))
                self.assertEqual(seen_cmds[0][0], ["/opt/bin/dis", "in.spv"])
                self.assertTrue(seen_cmds[0][1])
                self.assertTrue(seen_cmds[0][2])
                self.assertTrue(seen_cmds[0][3])
                self.assertEqual(seen_cmds[1][0], ["/opt/bin/as", "--target-env", expected_target, "-", "-o", "out.spv"])
                self.assertTrue(seen_cmds[1][1])
                self.assertTrue(seen_cmds[1][3])
                self.assertEqual(seen_cmds[1][4], disasm)

    def test_invalid_spirv_binary_words_and_missing_or_directory_paths(self):
        fake_run, calls = _make_fake_spirv_tools()
        valid_words = _build_mixed_bindless_spv_words()

        bad_inputs = [
            ("empty_file", b""),
            ("truncated_header_8B", struct.pack("<2I", SPIRV_MAGIC, SPIRV_VERSION_1_5)),
            ("unaligned_21B", _pack_spv([], bound=1) + b"\x00"),
            ("big_endian_magic", struct.pack(">5I", SPIRV_MAGIC, SPIRV_VERSION_1_5, 0, 2, 0)),
            ("wrong_magic", struct.pack("<5I", 0xDEADBEEF, SPIRV_VERSION_1_5, 0, 2, 0)),
            ("non_zero_schema", _pack_spv([], bound=1, schema=1)),
            ("zero_bound", _pack_spv([], bound=0)),
            ("zero_word_count_inst", _pack_spv([OP_CAPABILITY], bound=2)),
            ("truncated_inst_bounds", _pack_spv(valid_words[:-1], bound=18)),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            for label, blob in bad_inputs:
                with self.subTest(case=label):
                    calls.clear()
                    in_path = os.path.join(tmp, f"{label}.spv")
                    out_path = os.path.join(tmp, f"{label}.out.spv")
                    with open(in_path, "wb") as f:
                        f.write(blob)
                    with mock.patch.object(m.subprocess, "run", side_effect=fake_run):
                        with self.assertRaises(subprocess.CalledProcessError):
                            m.main(["spirv_metal_set0.py", in_path, out_path])
                    self.assertEqual(calls, [["spirv-dis", in_path]])
                    self.assertFalse(os.path.exists(out_path))

            # Missing input file and directory passed as input file both fail at spirv-dis.
            missing_in = os.path.join(tmp, "does_not_exist.spv")
            dir_in = os.path.join(tmp, "a_directory")
            os.makedirs(dir_in)
            for label, bad_path in (("missing_file", missing_in), ("directory_input", dir_in)):
                with self.subTest(case=label):
                    calls.clear()
                    out_path = os.path.join(tmp, f"{label}.out.spv")
                    with mock.patch.object(m.subprocess, "run", side_effect=fake_run):
                        with self.assertRaises(subprocess.CalledProcessError):
                            m.main(["spirv_metal_set0.py", bad_path, out_path])
                    self.assertEqual(calls, [["spirv-dis", bad_path]])
                    self.assertFalse(os.path.exists(out_path))

    def test_spirv_as_failure_missing_executable_and_rewrite_abort(self):
        with tempfile.TemporaryDirectory() as tmp:
            in_path = os.path.join(tmp, "in.spv")
            out_path = os.path.join(tmp, "out.spv")
            with open(in_path, "wb") as f:
                f.write(_pack_spv(_build_mixed_bindless_spv_words()))

            # 1. Missing spirv-dis binary raises FileNotFoundError before calling spirv-as.
            with mock.patch.object(m.subprocess, "run", side_effect=FileNotFoundError("no spirv-dis")) as run_mock:
                with self.assertRaises(FileNotFoundError):
                    m.main(["spirv_metal_set0.py", in_path, out_path])
                self.assertEqual(run_mock.call_count, 1)
            self.assertFalse(os.path.exists(out_path))

            # 2. rewrite() SystemExit aborts before spirv-as is invoked.
            bad_spv = _pack_spv([
                *_encode_inst(OP_DECORATE, 2, DEC_DESCRIPTOR_SET, 0),
                *_encode_inst(OP_DECORATE, 2, DEC_BINDING, 99),
            ], bound=3)
            with open(in_path, "wb") as f:
                f.write(bad_spv)
            fake_run, calls = _make_fake_spirv_tools()
            with mock.patch.object(m.subprocess, "run", side_effect=fake_run):
                with self.assertRaises(SystemExit) as ctx:
                    m.main(["spirv_metal_set0.py", in_path, out_path])
            self.assertIn("binding 99", str(ctx.exception))
            self.assertEqual(calls, [["spirv-dis", in_path]])
            self.assertFalse(os.path.exists(out_path))

            # 3. spirv-as failure raises CalledProcessError.
            dis_ok = subprocess.CompletedProcess(["spirv-dis", in_path], 0, "; Version: 1.5\nOpCapability Shader\n", "")
            asm_fail = subprocess.CalledProcessError(1, ["spirv-as"], stderr="bad asm")
            with mock.patch.object(m.subprocess, "run", side_effect=[dis_ok, asm_fail]) as run_mock:
                with self.assertRaises(subprocess.CalledProcessError):
                    m.main(["spirv_metal_set0.py", in_path, out_path])
                self.assertEqual(run_mock.call_count, 2)

    def test_main_guard_handles_stdout_reconfigure_errors_and_exit_codes(self):
        class _Stream(io.StringIO):
            def __init__(self, exc):
                super().__init__()
                self._exc = exc

            def reconfigure(self, **_kwargs):
                raise self._exc

        for exc in (AttributeError("no reconfigure"), ValueError("bad"), OSError("io")):
            with self.subTest(exc=type(exc).__name__):
                out_stream = _Stream(exc)
                err_stream = io.StringIO()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ResourceWarning)
                    with mock.patch.object(sys, "argv", ["spirv_metal_set0.py", "--print-layout"]), \
                            mock.patch.object(sys, "stdout", out_stream), \
                            mock.patch.object(sys, "stderr", err_stream):
                        with self.assertRaises(SystemExit) as ctx:
                            runpy.run_path(TOOL_PATH, run_name="__main__")
                self.assertEqual(ctx.exception.code, 0)
                self.assertEqual(err_stream.getvalue(), "")
                self.assertEqual(len(out_stream.getvalue().splitlines()), 26)

        err_stream = io.StringIO()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            with mock.patch.object(sys, "argv", ["spirv_metal_set0.py"]), \
                    mock.patch.object(sys, "stdout", io.StringIO()), \
                    mock.patch.object(sys, "stderr", err_stream):
                with self.assertRaises(SystemExit) as ctx:
                    runpy.run_path(TOOL_PATH, run_name="__main__")
        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("Usage: spirv_metal_set0.py in.spv out.spv", err_stream.getvalue())


if __name__ == "__main__":
    unittest.main()
