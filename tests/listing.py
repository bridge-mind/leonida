"""Hand-written ``.s`` listings for translator tests that need neither llvm-mc nor spirv-tools."""
import struct

import rdna2spirv

EMPTY_DXBC = b"DXBC" + bytes(16) + struct.pack("<III", 1, 32, 0)
EPILOGUE = "v_mov_b32 v8, v3\nexp mrt0 v8, v8, v8, v8 done vm\ns_endpgm"


def translate_listing(tmp_path, body, name="ps_listing"):
    """Translate pixel-shader ``body`` (assembly text, one instruction per line) and return the SPIR-V assembly.

    The first line of a listing is its header (shader_ir.parse_file skips it), so one is written here. Pixel shaders
    must not read v0/v1 (barycentrics) before writing them. v3 is the register the epilogue exports.
    """
    (tmp_path / "disasm").mkdir(exist_ok=True)
    (tmp_path / f"{name}.dxbc").write_bytes(EMPTY_DXBC)
    lines = [f"; {name} code 0x0..0x100"] + [ln.strip() for ln in (body + "\n" + EPILOGUE).splitlines() if ln.strip()]
    path = tmp_path / "disasm" / f"{name}.s"
    path.write_text(lines[0] + "\n" + "".join("\t" + ln + "\n" for ln in lines[1:]))
    return rdna2spirv.translate(str(path))
