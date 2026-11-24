"""The per-basic-block descriptor lookup cache of tools/rdna2spirv.py (standard library only: no GPU, LLVM or SPIRV-Tools).

    python -I tests/test_desc_cache.py

Every image instruction needs the bindless index of its T# (8 SGPRs) and S# (4 SGPRs): a call of %desc_lookup_fn. A
second instruction on the same descriptors in the same basic block reuses the first one's result as long as no SGPR
of the descriptor was written in between. These tests translate small pixel shaders (the .s text is made here, with
encodings of the right length so that branch targets work out) and count the calls in the SPIR-V.
"""
import gc
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(ROOT, "tools"))
import rdna2spirv  # noqa: E402

CALL = "OpFunctionCall %uint %desc_lookup_fn"
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
SAMPLE = "image_sample v[4:7], v[0:1], s[8:15], s[16:19] dmask:0xf dim:SQ_RSRC_IMG_2D"
LOAD = "image_load v[4:7], v[0:1], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D unorm"
EXP = "exp mrt0 v4, v5, v6, v7 done vm"


def words(line):
    """Encoding length in dwords of one line of the shaders below (only the length matters to the translator)."""
    op = line.split()[0]
    if op.startswith(("image_", "exp", "buffer_")):
        return 2
    return 1


def translate(lines):
    """SPIR-V assembly text of a pixel shader made of these instruction lines."""
    out, pc = ["; ps_desc_cache  code 0x0..0x0 (synthetic), gfx1013, 0 invalid words"], 0
    for ln in lines:
        n = words(ln)
        enc = ", ".join(f"0x{(pc + i) & 0xFF:02x}" for i in range(4 * n))
        if ln.startswith("s_endpgm"):
            enc = "0x00, 0x00, 0x81, 0xbf"
        out.append(f"\t{ln} ; encoding: [{enc}]")
        pc += 4 * n
    with tempfile.TemporaryDirectory() as tmp:
        os.mkdir(os.path.join(tmp, "disasm"))
        with open(os.path.join(tmp, "disasm", "ps_desc_cache.s"), "w", encoding="utf-8") as f:
            f.write("\n".join(out) + "\n")
        with open(os.path.join(tmp, "ps_desc_cache.dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            result = rdna2spirv.translate(os.path.join(tmp, "disasm", "ps_desc_cache.s"))
            gc.collect()
        return result


def calls(*lines):
    return translate(list(lines) + [EXP, "s_endpgm"]).count(CALL)


def test_one_lookup_per_descriptor():
    # Three samples on one T#/S#: one lookup for the T#, one for the S#.
    assert calls(SAMPLE, SAMPLE, SAMPLE) == 2
    # A different T# is another lookup; the S# is shared.
    assert calls(SAMPLE, "image_sample v[4:7], v[0:1], s[24:31], s[16:19] dmask:0xf dim:SQ_RSRC_IMG_2D") == 3
    # image_load looks its T# up once (it used to look it up a second time for the typed read).
    assert calls(LOAD) == 1
    assert calls(LOAD, LOAD, LOAD) == 1


def test_sgpr_write_inside_the_descriptor_makes_a_new_lookup():
    # Any SGPR of the T# (s8..s15): a scalar move, a 64-bit move, a v_readfirstlane, a load.
    for write in ("s_mov_b32 s9, 5", "s_mov_b32 s8, 5", "s_mov_b32 s15, 5", "s_mov_b64 s[14:15], 0",
                  "s_mov_b64 s[6:7], 0x1234", "v_readfirstlane_b32 s12, v0", "s_load_dwordx2 s[10:11], s[0:1], 0x0"):
        n = calls(SAMPLE, write, SAMPLE)
        # 2 calls first, then only the T# again (the S#, s16..s19, is untouched). s[6:7] is outside both.
        want = 2 if write.startswith("s_mov_b64 s[6:7]") else 3
        assert n == want, (write, n, want)
    # An SGPR of the S# only: the S# is looked up again, the T# is not.
    assert calls(SAMPLE, "s_mov_b32 s17, 5", SAMPLE) == 3
    assert calls(SAMPLE, "s_mov_b32 s19, 5", SAMPLE) == 3
    # Both written: both again.
    assert calls(SAMPLE, "s_mov_b32 s9, 5", "s_mov_b32 s16, 5", SAMPLE) == 4
    # Neighbours of the ranges: s7 and s20 are in neither, so there is no new lookup.
    assert calls(SAMPLE, "s_mov_b32 s7, 5", "s_mov_b32 s20, 5", SAMPLE) == 2
    # After the write the new lookup is cached again.
    assert calls(SAMPLE, "s_mov_b32 s9, 5", SAMPLE, SAMPLE) == 3


def test_cache_is_per_basic_block():
    # Block 0 (two samples), block 1 (two samples), block 2 (one): each block looks the descriptors up once.
    n = calls(SAMPLE, SAMPLE, "s_cmp_eq_u32 s0, 0", "s_cbranch_scc1 5", SAMPLE, SAMPLE, "s_branch 0", SAMPLE)
    # (branch offsets count dwords from the next instruction: scc1 skips block 1's two samples and the s_branch)
    assert n == 6, n
    # The same code without the branch would be one block: two lookups.
    assert calls(SAMPLE, SAMPLE, "s_cmp_eq_u32 s0, 0", SAMPLE, SAMPLE, SAMPLE) == 2


def test_lookups_around_image_load_constructs():
    # An image_load opens selection constructs for the typed read. Lookups made after it still hit the cache made
    # before it (the id dominates), and the construct itself added no lookup.
    assert calls(LOAD, SAMPLE, LOAD, SAMPLE) == 2



def test_no_caching_inside_construct():
    # A lookup emitted inside an instruction's own selection construct must not be cached: its SSA id does not
    # dominate code after the merge label, so reusing it there is invalid SPIR-V.
    g = rdna2spirv.Gen("ps")
    t_desc = {"k": "sgpr", "r": 8, "n": 8}
    then_lbl, merge_lbl = "%t_then", "%t_merge"

    assert g.desc_cache_ok()
    g.emit(f"OpSelectionMerge {merge_lbl} None")
    g.emit(f"OpBranchConditional %true {then_lbl} {merge_lbl}")
    g.emit(f"{then_lbl} = OpLabel")
    assert not g.desc_cache_ok()
    idx_inner = g.desc_lookup_sgpr(t_desc, 8)
    assert g.desc_cache == {}, "lookup inside selection construct must not populate desc_cache"
    g.emit(f"OpBranch {merge_lbl}")
    g.emit(f"{merge_lbl} = OpLabel")

    # Back at top level of the block: a fresh lookup must be emitted, cached, and reused by a third lookup.
    assert g.desc_cache_ok()
    idx_outer = g.desc_lookup_sgpr(t_desc, 8)
    assert idx_outer != idx_inner, "top-level lookup reused non-dominating id from inside construct"
    assert g.desc_cache == {(8, 8): idx_outer}
    assert g.desc_lookup_sgpr(t_desc, 8) == idx_outer
    g.store_reg("sgpr", 20, idx_outer)  # use the post-merge id so dominance is checked by spirv-val

    asm = rdna2spirv.assemble(g)
    assert asm.count(CALL) == 2, asm.count(CALL)
    if shutil.which("spirv-as") and shutil.which("spirv-val"):
        with tempfile.TemporaryDirectory() as tmp:
            sa = os.path.join(tmp, "dom.spvasm")
            sv = os.path.join(tmp, "dom.spv")
            with open(sa, "w", encoding="utf-8") as f:
                f.write(asm)
            r1 = subprocess.run(["spirv-as", "--target-env", "vulkan1.2", sa, "-o", sv], capture_output=True, text=True)
            assert r1.returncode == 0, f"spirv-as failed:\n{r1.stderr}"
            r2 = subprocess.run(["spirv-val", "--target-env", "vulkan1.2", sv], capture_output=True, text=True)
            assert r2.returncode == 0, f"spirv-val failed:\n{r2.stderr}"


def test_desc_probe_defers_index_load_to_hit_block():
    # In desc_probe (%desc_lookup_fn), the binding index load from %desc_map at offset 9 must be inside
    # %dl_hit, not before OpBranchConditional in %dl_body. This avoids up to 8 useless storage-buffer reads
    # on non-matching probes and misses.
    asm = translate([SAMPLE, EXP, "s_endpgm"])
    lines = [l.strip() for l in asm.splitlines()]
    fn_idx = lines.index("%desc_lookup_fn = OpFunction %uint None %fn_desc_lookup")
    end_idx = lines.index("OpFunctionEnd", fn_idx)
    fn_lines = lines[fn_idx:end_idx]

    body_idx = fn_lines.index("%dl_body = OpLabel")
    hit_idx = fn_lines.index("%dl_hit = OpLabel")
    next_idx = fn_lines.index("%dl_next = OpLabel")

    pre_hit = fn_lines[body_idx:hit_idx]
    assert not any("%c9" in l for l in pre_hit), "index offset %c9 loaded before conditional branch in %dl_body"

    hit_block = fn_lines[hit_idx:next_idx]
    has_c9 = any("%c9" in l and "OpIAdd %uint" in l for l in hit_block)
    has_load = any("OpLoad %uint" in l for l in hit_block)
    has_store = any("OpStore %dl_idx" in l for l in hit_block)
    assert has_c9 and has_load and has_store, f"dl_hit missing deferred load: {hit_block}"


def test_lookup_result_is_broadcast_before_use():
    # Mesa ANV 26.2.7 (Alder Lake Xe-LP) segfaults compiling a sampler index that comes straight out of the probe
    # loop when one sampler serves a float and an integer image (cs_sample_lz_int_d16, ps_ssharp). Each lookup's
    # result goes through OpGroupNonUniformBroadcastFirst, and only the broadcast id indexes the descriptor arrays.
    asm = translate([SAMPLE, LOAD, EXP, "s_endpgm"])
    lines = [ln.strip() for ln in asm.splitlines()]
    results = {ln.split()[0] for ln in lines if CALL in ln}
    bcast = {}
    for ln in lines:
        if "OpGroupNonUniformBroadcastFirst" in ln:
            bcast[ln.split()[0]] = ln.split()[-1]
        elif "OpGroupNonUniformShuffle" in ln:
            # a wave32 pixel shader in a 64-lane subgroup takes its own half's first lane with a shuffle (SHADER-731,
            # #4225; docs/design.md "Execution model")
            bcast[ln.split()[0]] = ln.split()[-2]
    assert results and sorted(bcast.values()) == sorted(results), (results, bcast)
    assert "OpCapability GroupNonUniform" in lines
    assert "OpCapability GroupNonUniformBallot" in lines or "OpCapability GroupNonUniformShuffle" in lines
    for ln in lines:
        if ln.split()[1:2] == ["="] and "OpAccessChain" in ln and ("%images" in ln or "%samplers" in ln):
            assert not results & set(ln.split()), ln


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("ok")
