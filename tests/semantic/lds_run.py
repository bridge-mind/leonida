"""Shared helper for the LDS tests: translate a hand-written compute listing with tools/rdna2spirv.py and run it
lane by lane in spirv_mini_interp (standard library only; no GPU, no game files).

The listing sees v0 = the lane's thread id x and s[0:3] = a 4 KiB output buffer. `store(reg, slot)` writes a VGPR
to out[lane * 8 + slot] (32 bytes per lane); `out(mem, lane, slot)` reads it back.

spirv_mini_interp keeps LDS per invocation (run_workgroup does not share a Workgroup variable between lanes), so a
test that needs one lane's atomics to see another's cannot be written; `run_chain` runs one lane through a sequence
of DS operations on a few dwords instead and returns every value it returned and every dword it left.
"""
import struct

import support
import spirv_mini_interp as si

M32 = 0xFFFFFFFF
# RSRC2 0x40088: 4 user SGPRs (the output V#), workgroup id in s4, LDS_SIZE = 8 x 512 bytes = 4 KiB.
RSRC2_4K = 0x40088
VSHARP = [0, 0, 0x1000, 0x31016FAC]
SLOT_BYTES = 32


def fssm(rsrc2=RSRC2_4K, size=(32, 1, 1)):
    blob = bytearray(0x148)
    struct.pack_into("<4I", blob, 0x138, rsrc2, *size)
    return bytes(blob)


def prologue():
    """v30 = lane * 32 (this lane's output slot base)."""
    return ["v_lshlrev_b32_e32 v30, 5, v0"]


def store(reg, slot):
    return [f"v_add_nc_u32_e32 v31, {slot * 4}, v30", f"buffer_store_dword {reg}, v31, s[0:3], 0 offen"]


def translate(body, rsrc2=RSRC2_4K, size=(32, 1, 1), name="cs_lds_real"):
    return support.translate_lines(name, list(body) + ["s_endpgm"], fssm=fssm(rsrc2, size))


def run(body, lanes=32, rsrc2=RSRC2_4K, size=(32, 1, 1)):
    """Translate and run `lanes` lanes in lockstep; return {byte address: dword} of the output buffer."""
    spv = translate(prologue() + list(body), rsrc2, size)
    return si.run_workgroup(spv, lanes, 32,
                            inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
                            push_constants=VSHARP + [0] * 28, memory={})


def out(mem, lane, slot):
    return mem.get(lane * SLOT_BYTES + slot * 4)


def run_chain(ops, init):
    """One lane. `init` {byte address: dword} is written with ds_write_b32; then each op (mnemonic, byte address,
    byte offset, data dwords) runs with v1 = address and data in v3, v4 (a mnemonic with `_rtn_` returns the old
    value in v5, preset to 0xdead). Returns ([v5 after each op], {effective byte address: dword read back})."""
    body = []
    for a, v in init.items():
        body += [f"v_mov_b32_e32 v1, {a:#x}", f"v_mov_b32_e32 v2, {v:#x}", "ds_write_b32 v1, v2", "s_waitcnt lgkmcnt(0)"]
    n = 0
    for name, addr, off, data in ops:
        body += [f"v_mov_b32_e32 v1, {addr:#x}", "v_mov_b32_e32 v5, 0xdead"]
        body += [f"v_mov_b32_e32 v{3 + k}, {d:#x}" for k, d in enumerate(data)]
        regs = (["v5"] if "_rtn_" in name else []) + ["v1"] + [f"v{3 + k}" for k in range(len(data))]
        body += [f"{name} {', '.join(regs)}" + (f" offset:{off}" if off else ""), "s_waitcnt lgkmcnt(0)",
                 "v_mov_b32_e32 v31, %d" % (4 * n), "buffer_store_dword v5, v31, s[0:3], 0 offen"]
        n += 1
    finals = sorted({a + o for _, a, o, _ in ops})
    for j, a in enumerate(finals):
        body += [f"v_mov_b32_e32 v1, {a:#x}", "ds_read_b32 v6, v1", "s_waitcnt lgkmcnt(0)",
                 "v_mov_b32_e32 v31, %d" % (4 * (n + j)), "buffer_store_dword v6, v31, s[0:3], 0 offen"]
    mem = run(body, lanes=1)
    return [mem.get(4 * k) for k in range(n)], {a: mem.get(4 * (n + j)) for j, a in enumerate(finals)}
