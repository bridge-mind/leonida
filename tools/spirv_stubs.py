"""Generate minimal SPIR-V partner stubs for graphics pipeline compilation checks.

    python3 -I tools/spirv_stubs.py <spirv dir>

Emits binary SPIR-V 1.0 directly with the standard library (no external spirv-as needed) into
<spirv dir>/stubs/. Stubs are deduplicated by their sorted location signature:

    vs_empty.spv, vs_0_1.spv, ...   Vertex shader writing BuiltIn Position and vec4(0) to each Location
    ps_empty.spv, ps_0_1.spv, ...   Fragment shader reading each Location as vec4 and writing Location 0

Every inter-stage varying emitted by tools/rdna2spirv.py is a 32-bit vec4 float, so a single stub per
unique location set matches every shader with that interface.
"""
import json
import os
import shutil
import struct
import sys
import time
import uuid

SPIRV_MAGIC = 0x07230203
SPIRV_VERSION_1_0 = 0x00010000

# Opcodes used by the minimal stubs.
OP_CAPABILITY = 17
OP_MEMORY_MODEL = 14
OP_ENTRY_POINT = 15
OP_EXECUTION_MODE = 16
OP_TYPE_VOID = 19
OP_TYPE_FLOAT = 22
OP_TYPE_VECTOR = 23
OP_TYPE_POINTER = 32
OP_TYPE_FUNCTION = 33
OP_CONSTANT = 43
OP_CONSTANT_COMPOSITE = 44
OP_FUNCTION = 54
OP_FUNCTION_END = 56
OP_VARIABLE = 59
OP_LOAD = 61
OP_STORE = 62
OP_DECORATE = 71
OP_FADD = 129
OP_LABEL = 248
OP_RETURN = 253

CAP_SHADER = 1
ADDR_LOGICAL = 0
MEM_GLSL450 = 1
EXEC_VERTEX = 0
EXEC_FRAGMENT = 4
MODE_ORIGIN_UPPER_LEFT = 7
STORAGE_INPUT = 1
STORAGE_OUTPUT = 3
DEC_BUILTIN = 11
DEC_FLAT = 14
DEC_LOCATION = 30
BUILTIN_POSITION = 0


def _inst(opcode, *words):
    return [((len(words) + 1) << 16) | opcode, *words]


def _encode_str(text):
    raw = text.encode("utf-8") + b"\x00"
    pad = (-len(raw)) % 4
    raw += b"\x00" * pad
    return list(struct.unpack(f"<{len(raw) // 4}I", raw))


def stub_name(kind, locations):
    """Canonical filename for a stub of kind ('vs' or 'ps') matching sorted unique `locations`."""
    locs = sorted({int(x) for x in locations})
    suffix = "_".join(str(x) for x in locs) if locs else "empty"
    return f"{kind}_{suffix}.spv"


def build_vs_stub(out_locations=()):
    """Minimal Vertex SPIR-V 1.0 module writing BuiltIn Position=(0,0,0,1) and vec4(0) to each Location."""
    locs = sorted({int(x) for x in out_locations})
    nid = 1

    def alloc():
        nonlocal nid
        cur = nid
        nid += 1
        return cur

    id_void = alloc()
    id_fn = alloc()
    id_float = alloc()
    id_v4 = alloc()
    id_ptr_out = alloc()
    id_f0 = alloc()
    id_f1 = alloc()
    id_pos_val = alloc()
    id_zero_v4 = alloc()
    id_out_pos = alloc()
    out_vars = [(loc, alloc()) for loc in locs]
    id_main = alloc()
    id_entry = alloc()

    iface = [id_out_pos] + [vid for _, vid in out_vars]
    words = [SPIRV_MAGIC, SPIRV_VERSION_1_0, 0, nid, 0]
    words += _inst(OP_CAPABILITY, CAP_SHADER)
    words += _inst(OP_MEMORY_MODEL, ADDR_LOGICAL, MEM_GLSL450)
    words += _inst(OP_ENTRY_POINT, EXEC_VERTEX, id_main, *_encode_str("main"), *iface)
    words += _inst(OP_DECORATE, id_out_pos, DEC_BUILTIN, BUILTIN_POSITION)
    for loc, vid in out_vars:
        words += _inst(OP_DECORATE, vid, DEC_LOCATION, loc)

    words += _inst(OP_TYPE_VOID, id_void)
    words += _inst(OP_TYPE_FUNCTION, id_fn, id_void)
    words += _inst(OP_TYPE_FLOAT, id_float, 32)
    words += _inst(OP_TYPE_VECTOR, id_v4, id_float, 4)
    words += _inst(OP_TYPE_POINTER, id_ptr_out, STORAGE_OUTPUT, id_v4)
    words += _inst(OP_CONSTANT, id_float, id_f0, 0x00000000)
    words += _inst(OP_CONSTANT, id_float, id_f1, 0x3F800000)
    words += _inst(OP_CONSTANT_COMPOSITE, id_v4, id_pos_val, id_f0, id_f0, id_f0, id_f1)
    words += _inst(OP_CONSTANT_COMPOSITE, id_v4, id_zero_v4, id_f0, id_f0, id_f0, id_f0)
    words += _inst(OP_VARIABLE, id_ptr_out, id_out_pos, STORAGE_OUTPUT)
    for _, vid in out_vars:
        words += _inst(OP_VARIABLE, id_ptr_out, vid, STORAGE_OUTPUT)

    words += _inst(OP_FUNCTION, id_void, id_main, 0, id_fn)
    words += _inst(OP_LABEL, id_entry)
    words += _inst(OP_STORE, id_out_pos, id_pos_val)
    for _, vid in out_vars:
        words += _inst(OP_STORE, vid, id_zero_v4)
    words += _inst(OP_RETURN)
    words += _inst(OP_FUNCTION_END)
    return struct.pack(f"<{len(words)}I", *words)


def build_ps_stub(in_locations=(), flat_locations=()):
    """Minimal Fragment SPIR-V 1.0 module reading each Location as vec4 and writing Location 0."""
    locs = sorted({int(x) for x in in_locations})
    flat = {int(x) for x in flat_locations}
    nid = 1

    def alloc():
        nonlocal nid
        cur = nid
        nid += 1
        return cur

    id_void = alloc()
    id_fn = alloc()
    id_float = alloc()
    id_v4 = alloc()
    id_ptr_out = alloc()
    id_ptr_in = alloc() if locs else None
    id_f0 = alloc()
    id_zero_v4 = alloc()
    id_out0 = alloc()
    in_vars = [(loc, alloc()) for loc in locs]
    id_main = alloc()
    id_entry = alloc()
    load_add = [(vid, alloc(), alloc()) for _, vid in in_vars]

    iface = [id_out0] + [vid for _, vid in in_vars]
    words = [SPIRV_MAGIC, SPIRV_VERSION_1_0, 0, nid, 0]
    words += _inst(OP_CAPABILITY, CAP_SHADER)
    words += _inst(OP_MEMORY_MODEL, ADDR_LOGICAL, MEM_GLSL450)
    words += _inst(OP_ENTRY_POINT, EXEC_FRAGMENT, id_main, *_encode_str("main"), *iface)
    words += _inst(OP_EXECUTION_MODE, id_main, MODE_ORIGIN_UPPER_LEFT)
    words += _inst(OP_DECORATE, id_out0, DEC_LOCATION, 0)
    for loc, vid in in_vars:
        words += _inst(OP_DECORATE, vid, DEC_LOCATION, loc)
        if loc in flat:
            words += _inst(OP_DECORATE, vid, DEC_FLAT)

    words += _inst(OP_TYPE_VOID, id_void)
    words += _inst(OP_TYPE_FUNCTION, id_fn, id_void)
    words += _inst(OP_TYPE_FLOAT, id_float, 32)
    words += _inst(OP_TYPE_VECTOR, id_v4, id_float, 4)
    words += _inst(OP_TYPE_POINTER, id_ptr_out, STORAGE_OUTPUT, id_v4)
    if id_ptr_in is not None:
        words += _inst(OP_TYPE_POINTER, id_ptr_in, STORAGE_INPUT, id_v4)
    words += _inst(OP_CONSTANT, id_float, id_f0, 0x00000000)
    words += _inst(OP_CONSTANT_COMPOSITE, id_v4, id_zero_v4, id_f0, id_f0, id_f0, id_f0)
    words += _inst(OP_VARIABLE, id_ptr_out, id_out0, STORAGE_OUTPUT)
    for _, vid in in_vars:
        words += _inst(OP_VARIABLE, id_ptr_in, vid, STORAGE_INPUT)

    words += _inst(OP_FUNCTION, id_void, id_main, 0, id_fn)
    words += _inst(OP_LABEL, id_entry)
    acc = id_zero_v4
    for vid, id_ld, id_sum in load_add:
        words += _inst(OP_LOAD, id_v4, id_ld, vid)
        words += _inst(OP_FADD, id_v4, id_sum, acc, id_ld)
        acc = id_sum
    words += _inst(OP_STORE, id_out0, acc)
    words += _inst(OP_RETURN)
    words += _inst(OP_FUNCTION_END)
    return struct.pack(f"<{len(words)}I", *words)


def _interface_locations(meta, field, legacy_field=None):
    if legacy_field is not None and meta.get(field) is None:
        field = legacy_field
    locations = meta.get(field, [])
    if not isinstance(locations, (list, tuple)) or any(type(loc) is not int for loc in locations):
        raise ValueError(f"{field} must be a list of integer locations")
    return tuple(sorted(set(locations)))


def required_stubs_for_meta(meta):
    """Return list of (kind, tuple(sorted_locations)) partner stubs needed for one index.json entry."""
    stage = meta.get("stage", "")
    if stage == "ps":
        return [("vs", _interface_locations(meta, "inputs"))]
    if stage == "vs":
        return [("ps", _interface_locations(meta, "outputs"))]
    if stage == "ms":
        return [("ps", _interface_locations(meta, "params"))]
    if stage == "gs":
        return [
            ("vs", _interface_locations(meta, "gs_input_locations", "gs_inputs")),
            ("ps", _interface_locations(meta, "gs_output_locations", "gs_outputs")),
        ]
    return []


def _replace_stub(source, target):
    # Windows can briefly deny the rename while another writer publishes the same stub.
    for delay in (0.01, 0.02, 0.04) if os.name == "nt" else ():
        try:
            os.replace(source, target)
            return
        except PermissionError as exc:
            if getattr(exc, "winerror", None) not in (5, 32, 33):
                raise
            time.sleep(delay)
    os.replace(source, target)


def generate_for_index(spirv_dir, index):
    """Write all deduplicated partner stubs needed by `index` into <spirv_dir>/stubs/. Returns filenames written."""
    needed = set()
    for name, meta in index.items():
        if isinstance(meta, dict):
            try:
                for spec in required_stubs_for_meta(meta):
                    needed.add(spec)
            except (TypeError, ValueError) as e:
                raise ValueError(f"{name}: invalid graphics interface locations: {e}") from e
    # Validate the entire requested set before writing any stubs: negative or
    # oversized locations cannot be encoded as SPIR-V unsigned literals.
    for kind, locs in needed:
        if any(not 0 <= loc <= 0xFFFFFFFF for loc in locs):
            raise ValueError(f"invalid Location in {kind} partner stub: {locs}")
    if not needed:
        return []
    stubs_dir = os.path.join(spirv_dir, "stubs")
    os.makedirs(stubs_dir, exist_ok=True)
    written = []
    for kind, locs in sorted(needed):
        fname = stub_name(kind, locs)
        data = build_vs_stub(locs) if kind == "vs" else build_ps_stub(locs)
        target = os.path.join(stubs_dir, fname)
        # A sibling file inherits the output directory's permissions.
        staged = os.path.join(stubs_dir, f".stub-{uuid.uuid4().hex}.tmp")
        stream = open(staged, "xb")
        try:
            with stream as f:
                f.write(data)
            if os.name == "posix":
                try:
                    shutil.copymode(target, staged)
                except FileNotFoundError:
                    pass
            _replace_stub(staged, target)
        except BaseException:
            try:
                os.remove(staged)
            except OSError:
                pass
            raise
        written.append(fname)
    return written


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv in (["-h"], ["--help"]):
        print(__doc__)
        return 0
    if len(argv) != 1:
        sys.exit(__doc__)
    spirv_dir = argv[0]
    idx_path = os.path.join(spirv_dir, "index.json")
    if not os.path.isfile(idx_path):
        sys.exit(f"missing {idx_path}")
    try:
        with open(idx_path, encoding="utf-8-sig") as f:
            index = json.load(f)
    except (OSError, ValueError) as e:
        sys.exit(f"cannot read {idx_path}: {e}")
    if not isinstance(index, dict):
        sys.exit(f"cannot read {idx_path}: expected a JSON object of shader -> metadata entries")
    try:
        written = generate_for_index(spirv_dir, index)
    except (OSError, ValueError, struct.error) as e:
        sys.exit(f"cannot generate stubs from {idx_path}: {e}")
    print(f"wrote {len(written)} stubs to {os.path.join(spirv_dir, 'stubs')}")
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
