"""Run the test shaders that build.py made on this machine's GPU and compare with what each one must give.

    python -I runtime/test/shaders/check.py <out dir of build.py> [dir with the test programs]

The test programs are vk_compute_test, vk_store_test, vk_harness, vk_texture_test, vk_bindless_test and vk_sampler_test
(runtime/build by default). With
the Visual Studio generator they are in runtime/build/RelWithDebInfo, Release and so on: that subfolder is found by
itself (the newest complete one if there are several) and its name is printed.
Each line is one shader: ok, or what came back instead. The wave32 test ps_bitmask_perm checks every pixel, including
the second 32-lane half when a GPU runs fragment shaders in 64-wide subgroups.
The two whole-wave mask checks are skipped when the device cannot provide a full 64-wide compute subgroup.
The cs_w64x_* / ms_w64x_* cross-half tests (lanes 32-63 against lanes 0-31, see their .asm headers) are skipped there
too. With LEONIDA_W64_EMULATE=1 they run without that requirement, on a device whose subgroups are narrower (NVIDIA),
to check a translator that emulates wave64 on them. scripts/run_gpu_tests.py builds the mesh fixtures and
fs_passthru.spv; here the mesh checks run when those files are in <out dir>/spirv.
ps_tex is skipped when the device lacks a feature the bindless set or the shader needs, ps_ssharp when it lacks one
vk_sampler_test needs.
ps_texoff (texel offsets, on vk_texture_test's texture) is skipped on a device without VK_KHR_maintenance8.
"""
import collections
import json
import math
import os
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
EXE = ".exe" if os.name == "nt" else ""
UNSUPPORTED = 77                                       # a test tool exits with this when the device lacks a feature


def barycentric_fallback_override():
    v = (os.environ.get("LEONIDA_BARY_FALLBACK") or "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return None


def barycentric_fallback():
    override = barycentric_fallback_override()
    if override is not None:
        return override
    try:
        tools = os.path.abspath(os.path.join(HERE, "..", "..", "..", "tools"))
        r = subprocess.run([sys.executable, "-I", os.path.join(tools, "vk_subgroup_info.py")],
                           capture_output=True, text=True, timeout=60)
        answer = json.loads(r.stdout.strip().splitlines()[-1])
        return answer.get("barycentric_fallback") is True
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return False

# The programs every run needs. Multi-config CMake generators (Visual Studio, Xcode, Ninja Multi-Config) put each of
# them in build/<Config> instead of build/ (TEST-38).
TEST_PROGRAMS = ("vk_compute_test", "vk_store_test", "vk_harness")
CONFIG_DIRS = ("RelWithDebInfo", "Release", "Debug", "MinSizeRel")


def complete_build_dirs(build, programs):
    """`build` and its configuration subfolders, in that order, that hold all of `programs` (file names with extension)."""
    return [d for d in (build, *(os.path.join(build, c) for c in CONFIG_DIRS))
            if all(os.path.isfile(os.path.join(d, n)) for n in programs)]


def resolve_build_dir(build, programs):
    """The folder that holds all of `programs`.

    Makefile and Ninja builds leave them in `build`. With the Visual Studio generator they land in build/RelWithDebInfo,
    build/Release and so on. `build` itself wins when it has them all; otherwise the configuration folder that has them
    all and was built last. If none has them, `build` is returned, so a "not found" message names the folder given."""
    found = complete_build_dirs(build, programs)
    if not found or found[0] == build:
        return build
    return max(found, key=lambda d: max(os.path.getmtime(os.path.join(d, n)) for n in programs))


def pick_build_dir(build, programs):
    """resolve_build_dir, and say which folder was picked when it is a configuration subfolder, so a report can name
    the build it ran and a stale pick is visible."""
    chosen = resolve_build_dir(build, programs)
    if chosen != build:
        names = [os.path.basename(d) for d in complete_build_dirs(build, programs)]
        print(f"test programs: {chosen}" + (f" (newest of {', '.join(names)})" if len(names) > 1 else ""))
    return chosen


def not_found_message(build, missing):
    return (f"Test programs not found in {build}: {', '.join(missing)}\n"
            f"(its {', '.join(CONFIG_DIRS)} subfolders were searched too)\n"
            "Build them with cmake -S runtime -B runtime/build, then cmake --build runtime/build.\n"
            "CMake only builds them when it finds Vulkan. If it printed \"Vulkan SDK not found\", install the "
            "Vulkan headers (vulkan-headers on Arch) and configure again.")

COMPUTE = [("cs_lds", "4", "mirror64"), ("cs_exec64", "1", "exec64"), ("cs_bvh", "1", "bvh"),
           ("cs_workgroup_yz", "2x3x4", "workgroup_yz")]
COMPUTE += [("cs_bvh32", "1", "bvh")]   # TEST-184: image_bvh_intersect_ray, the 32-bit node pointer; same scene and words
CS_STORE = [0x400080ff, 0xe00003ff, 0x3c004000, 0xeeeeeeee, 0x40003c00, 0xbc003800, 0x40ff8000, 0x3ff00000]
CS_SMEM = [0x5eed0024, 0x5eed0028, 0x5eed002c, 0x5eed0030, 0x5eed0010, 0x5eed0014, 0x5eed0018, 0x5eed001c]
# -0, and NaN checks that need SignedZeroInfNanPreserve (cs_signed_zero_nan.asm); 0 in out[4..7] means it's missing
SIGNED_ZERO_NAN = [0x80000000, 0x80000000, 0x80008000, 0x00008000, 1, 1, 1, 0x00007c00]
WAVE32_DUMPS = ("cs_store", "cs_signed_zero_nan", "cs_smem_soffset_imm", "cs_wqm_w32", "cs_w32_lane_wrap", "cs_movrelsd_2", "cs_w32_permute",
                "cs_barrier_divergent", "cs_barrier_early_end", "cs_barrier_loop", "cs_w32_lds_u8_d16", "cs_bpermute32",
                "cs_w32_lds_subdword", "cs_w32_lds_wide",
                "cs_w32_partial_scmp", "cs_exec_const", "cs_w32_mask_imm", "cs_w32_int_ext",
                "cs_w32_salu64", "cs_w32_misc_ext", "cs_w32_new_ops", "cs_w32_sincos", "cs_lds_b64_cmpst", "cs_w32_andn2_exec",
                "cs_mask_vgpr_exec", "cs_w32_buffer_sub16", "cs_w32_buffer_store_sub", "cs_w32_fcmpswap", "cs_w32_permlane_fi")
# Wave32 lane-mask dumps: their EXEC/VCC masks and lane selects cover 32 lanes, so a device whose widest subgroup is
# narrower (Mesa llvmpipe: 8) can't produce their words. They are skipped there, as ps_bitmask_perm is. They run with
# 32 lanes requested (WAVE32_LANE_ENV), as run_gpu_tests.py does: a device with a range like 8..32 (Intel ANV) may
# otherwise pick a narrower size and return different words (#774).
WAVE32_LANE_ENV = {"VKC_SUBGROUP_SIZE": "32"}
WAVE32_LANE_DUMPS = ("cs_wqm_w32", "cs_w32_lane_wrap", "cs_w32_permute", "cs_bpermute32", "cs_w32_partial_scmp", "cs_exec_const",
                     "cs_w32_mask_imm", "cs_lds_b64_cmpst", "cs_w32_andn2_exec", "cs_mask_vgpr_exec", "cs_w32_permlane_fi")
CS_DPP = [("cs_dpp", [1, 0, 3, 2, 5, 4, 7, 6]),
          ("cs_dpp_row", [0, 0, 1, 2, 3, 4, 5, 6]),
          ("cs_dpp_mask", [1, 0, 3, 2, 0xeeeeeeee, 0xeeeeeeee, 0xeeeeeeee, 0xeeeeeeee]),
          ("cs_dpp8", [7, 6, 5, 4, 3, 2, 1, 0]),
          ("cs_dpp_ops", [100, 10, 11, 12, 13, 14, 15, 16]),
          ("cs_dpp_bound", [10, 10, 11, 12, 13, 14, 15, 16]),
          ("cs_dpp_bank", [0xffffffff, 0xfffffffe, 0xfffffffd, 0xfffffffc, 7, 7, 7, 7]),
          ("cs_dpp_fi", [1, 2, 3, 0x7777, 5, 6, 7, 0x7777])]
WAVE32_DUMPS += tuple(n for n, _ in CS_DPP)
WAVE32_LANE_DUMPS += tuple(n for n, _ in CS_DPP)
CS_DUMPS = [*CS_DPP, ("cs_store", CS_STORE), ("cs_smem_soffset_imm", CS_SMEM),
            # SHADER-52: wave32 lane select is & 31; lane 40 wraps to 8
            ("cs_w32_lane_wrap", [25, 16, 777, 777]),
            # v_movrelsd_2_b32: VGPR[D + M0[25:16]] = VGPR[S0 + M0[9:0]]
            ("cs_movrelsd_2", [11, 33, 33, 55, 1717, 77, 1515, 1717]),
            # SHADER-90: ds_permute_b32, the forward (scatter) permute; unwritten lane 0, EXEC gates the destination
            ("cs_w32_permute", [218, 29, 8, 211, 0, 15, 99, 15]),
            # integer ops with CPU-only checks until now (derivations in cs_w32_int_ext.asm)
            ("cs_w32_int_ext", [1, 27, 1, 0x33332222, 0xC0000000, 0x4000, 18, 0x20304050]),
            # 64-bit scalar data ops and v_mad_u64_u32 (derivations in cs_w32_salu64.asm)
            # more ops with CPU-only checks until now (derivations in cs_w32_misc_ext.asm)
            # ops merged on 2026-11-22 with CPU-only checks (derivations in cs_w32_new_ops.asm)
            # plain 8/16-bit MUBUF stores, one crossing a dword (cs_w32_buffer_store_sub.asm)
            ("cs_w32_buffer_store_sub", [0x1122AB44, 0xCDEF7788, 0x34AABBCC, 0xDDDDDD12, 0x0102035A, 0x05061234, 0xAB, 0xFFFFCDEF]),
            # 8/16-bit MUBUF d16 loads and stores (cs_w32_buffer_sub16.asm)
            ("cs_w32_buffer_sub16", [0xBEEF00AA, 0x00AA1234, 0xBEEFFFAA, 0xFFAA1234, 0xBEEF8899, 0x88991234, 0x1122CD44, 0xABCD7788]),
            # ds_cmpst_f32 and buffer_atomic_fcmpswap: float compare (cs_w32_fcmpswap.asm)
            ("cs_w32_fcmpswap", [0, 0x3F800000, 0x7FC00000, 0x3F800000, 0x80000000, 0x40400000, 0x3F800000, 0x7FC00000]),
            ("cs_w32_new_ops", [0x55, 0x8001, 0xC0000000, 0x80014000, 0x4000FFFF, 0x77, 0x99, 0x1234]),
            ("cs_w32_misc_ext", [0x0FFF0FFF, 0xE2C4A688, 0x0007FFFF, 0x7FFF8000, 0xFFFE0002, 0xFFFF0005, 0x509, 0xCAFEF00D]),
            ("cs_w32_salu64", [0x1E6A2C48, 0x1234, 0xFFFFFFFF, 0x100, 47, 48, 0xFFFF8000, 0xFFFFFFFE]),
            # SHADER-85: 8-bit LDS reads into one half of a VGPR, the other half kept
            ("cs_w32_lds_u8_d16", [0xBEEF00FF, 0x00FF1234, 0xBEEFFF80, 0xFF801234, 0xBEEF007F, 0x007F1234, 0xBEEF0080, 0xFFFF1234]),
            # 8/16-bit LDS writes (incl. _d16_hi) and sign/zero-extending reads
            ("cs_w32_lds_subdword", [0x11B2D444, 0xA1B2C3D4, 0xFFFFFFB2, 0x000000B2, 0x00000011, 0xFFFFA1B2, 0x0000A1B2, 0xFFFFC3D4]),
            # 64/96/128-bit LDS accesses and ds_read2st64_b32
            ("cs_w32_lds_wide", [0xCAFEF00D, 0xFEEDFACE, 0x600DD00D, 0x00000000, 0x8BADF00D, 0xDEADC0DE, 0x12345678, 0x9ABCDEF0]),
            # ds_cmpst_b32 and the b64 2-address forms (cs_lds_b64_cmpst.asm header)
            ("cs_lds_b64_cmpst", [0x11110001, 0x22220002, 0x33330003, 0x44440004,
                                  0xCCCC000C, 0x44440004, 0xBBBB000B, 0x3579BADC]),
            # constant EXEC: each lane takes its own bit (Intel got 16-bit constants wrong for lanes 16-31)
            ("cs_exec_const", [16, 8, 2, 16, 0, 16]),
            # a constant lane mask (saveexec, wrexec, s_and/s_not/s_mov_b64 into EXEC or VCC) gives lane L its bit L,
            # not bit 0 for every lane; word = set lanes | lowest lane << 8 (derivations in each .asm header)
            ("cs_w32_mask_imm", [2056, 20, 2, 6152, 2050, 4112, 258, 32]),
            ("cs_w64_mask_imm", [2056, 1084, 828, 2052, 319, 2104, 516, 2064]),
            # SHADER-99: -0 and NaN survive translation (SignedZeroInfNanPreserve); words 0, 2, 3 on Apple: see below
            ("cs_signed_zero_nan", SIGNED_ZERO_NAN),
            # s_andn2_saveexec_b32 / s_andn2_wrexec_b32: EXEC = S0 & ~EXEC, D old / new EXEC, SCC (cs_w32_andn2_exec.asm)
            ("cs_w32_andn2_exec", [1, 8, 16, 2064, 2068, 3, 20, 1544]),
            ("cs_wave64_ballot", [32, 16, 24, 40, 8, 48]),
            ("cs_wm_half_ops", [32, 16, 16, 0, 16]), ("cs_wm_half_cmp", [1, 0, 1, 1]),
            ("cs_wm_bfm64", [16, 8, 8, 40]), ("cs_wm_scc_rfl", [55, 48]),
            # mask-only wave64: s_bcnt1_i32_b64 on ordinary SGPRs, no WAVE64_RE EXEC markers (#210 / SHADER-28)
            ("cs_w64_mask_detect", [48]),
            ("cs_wqm_w64", [20, 4, 12, 8, 1, 0, 15728895, 20]), ("cs_wqm_w32", [12, 1, 4, 0, 8, 15728880, 4, 32]),
            ("cs_bpermute32", [0, 50, 162, 99, 36, 218, 183, 15]),
            ("cs_w32_partial_scmp", [0, 1, 1, 0, 1, 1, 1, 1]),   # a 20-thread wave32: s_cmp mask, -1
            # SHADER-88: v_sin/v_cos reduce the angle in turns (a regression guard, tolerance 1e-3)
            ("cs_w32_sincos", [1, 1, 1, 1, 1, 1]),
            # SHADER-74: v_permlane16/x16 with the source lane off in EXEC (FI / BOUND_CTRL)
            ("cs_w32_permlane_fi", [1000, 0, 4, 1, 1000, 0, 52, 49]),
            ("cs_wqm_w64_half", [11, 14, 0, 1, 6, 8, 12, 2]),
            # SHADER-91: two s_barrier sites in wave-uniform branches meet like one barrier; wave 0 reads wave 3
            ("cs_barrier_divergent", [127, 126, 125, 124, 123, 122, 121, 120]),
            # SHADER-91: waves that end without an s_barrier don't hold the barrier of the ones still running
            ("cs_barrier_early_end", [127, 126, 125, 124, 123, 122, 121, 120]),
            # SHADER-91: a barrier in a loop with a per-wave trip count; the k-th one waits for the waves still running
            ("cs_barrier_loop", [128, 96, 64, 32, 128, 96, 64, 0xEEEEEEEE]),
            # SHADER-21 part 2: a mask copied to a VGPR with v_mov_b32 respects EXEC (inactive lanes keep the old VGPR)
            ("cs_mask_vgpr_exec", [0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
                                   0xDEADBEEF, 0xDEADBEEF, 0xDEADBEEF, 0xDEADBEEF])]
# num_records bounds check (cs_bounds.asm): one thread, so no 64-wide subgroup requirement.
# The SMEM and raw MUBUF rules measured on an RX 6800 (derivations in each .asm header): one thread too.
# Never run cs_smem_sload_neg with a translator that zero-extends the s_load immediate: it reads 4 GiB past the heap.
BOUNDS = [("cs_bounds", [0x12345678, 0x12345678, 0x00000000, 0xEEEEEEEE]),
          ("cs_smem_bounds", [0, 0x5eed0010, 0x5eed000c, 0, 0x5eed000c, 0x5eed0010, 0, 0]),
          ("cs_smem_stride_align", [0x5eed0010, 0, 0, 0x5eed0000, 0x5eed0000, 0x5eed0000, 0x5eed0000, 0x5eed0004]),
          # SHADER-328: a constant SGPR offset gives what the run-time one gives (no 32-bit wrap)
          ("cs_smem_sconst_off", [0, 0x5eed0018, 0x5eed0018, 0x5eed0014]),
          ("cs_smem_sload_neg", [0x5eed0004, 0x5eed0000, 0x5eed0004, 0x5eed0004, 0x5eed0008, 0x5eed000c,
                                 0xeeeeeeee, 0xeeeeeeee]),
          ("cs_buf_tail", [0, 0x5eed000c, 0x5eed0008, 0x5eed000c, 0, 0, 0x5eed0010, 0]),
          # V# OOB_SELECT 0..3 (literal and run-time V#s), the RDNA2 rules of docs/buffer_bounds.md, derived in the .asm header
          ("cs_oob_select", [0x5eed000c, 0, 0x5eed0014, 0, 0x5eed0018, 0xeeeeeeee, 0, 0]),
          # the end of an access decides (a start check, Mesa's GFX10 column, reads more); x4 loads and stores per dword, typed all or nothing
          ("cs_oob_tail", [0, 0, 0x5eed0010, 0, 0xbeef0000, 0xeeeeeeee, 0, 0x5eed0010]),
          # swizzled MUBUF addressing (measured natively on gfx1036) and ADD_TID_ENABLE on lanes 0 and 1, derived in the .asm header
          ("cs_buf_swizzle", [0x5eed0024, 0x5eed0044, 0x5eed004c, 0x5eed0008, 0x5eed000c, 0, 0xbeef0000, 0xbeef0001])]
CMPX_CLASS_WORDS = [0xEEEEEEEE] * 3 + [0x22] + [0x11] * 4
# cs_fmt_1010102.asm: descriptor loads, immediate-format load, raw input words and the two packed stores.
FMT_1010102_WORDS = [0, 0x3F800000, 0, 0x3F800000, 0x3F800000, 0, 0x3F800000, 0,
                     0, 0x3F800000, 0, 0x3F800000, 0xC00FFC00, 0x003FF003, 0xE00003FF, 0xFFE00003]
# cs_fmt_int_narrow.asm: 8- and 16-bit UINT/SINT typed buffer loads and stores (layout in the .asm header). The
# words are the same program's output on gfx1036 run natively (tools/hw_oracle, wave32 and wave64 identical).
FMT_INT_NARROW_WORDS = [
    0x00000080, 0x00000080, 0x00000080, 0x00000080, 0xFFFFFF80, 0xFFFFFF80, 0xFFFFFF80, 0xFFFFFF80,
    0x0000FF80, 0x0000FF80, 0x0000FF80, 0x0000FF80, 0xFFFFFF80, 0xFFFFFF80, 0xFFFFFF80, 0xFFFFFF80,
    0x00000080, 0x000000FF, 0x00000080, 0x000000FF, 0xFFFFFF80, 0xFFFFFFFF, 0xFFFFFF80, 0xFFFFFFFF,
    0xFFFFFF80, 0x00007F01, 0xFFFFFF80, 0x00007F01, 0xFFFFFF80, 0xFFFFFFFF, 0x00000001, 0x0000007F,
    0xFFFFFF80, 0x00007F01, 0xFFFFFFFF, 0xFFFF8000, 0xFFFFFF80, 0x00000000, 0x00000001, 0x00000001,
    0x0000FF80, 0x00000000, 0x00000001, 0x00000001, 0xFF80FF80, 0xFF80FF80, 0xFFFFFF80, 0x007F0001,
    0x7F01FF80, 0x8000FFFF, 0xEEEEEEEE, 0xEEEEEEEE, 0xEEEEEEEE, 0xEEEEEEEE, 0xEEEEEEEE, 0xEEEEEEEE,
    0xEEEEEEEE, 0xEEEEEEEE, 0xEEEEEEEE, 0xEEEEEEEE, 0xEEEEEEEE, 0xEEEEEEEE, 0xEEEEEEEE, 0xEEEEEEEE,
    0xEEEEEE7F, 0xEEEEEEEE, 0xEEEEEE7F, 0xEEEEEEEE, 0xEEEE7FFF, 0xEEEEEEEE, 0xEEEE7FFF, 0xEEEEEEEE,
    0xEEEE807F, 0xEEEEEEEE, 0xEEEE7F7F, 0xEEEEEEEE, 0x7FFF7FFF, 0xEEEEEEEE, 0x80807F7F, 0xEEEEEEEE,
    0x7FFF7FFF, 0x80008000, 0xEEEEEEEE, 0xEEEEEEEE, 0xEEEEEEEE, 0xEEEEEEEE, 0xEEEEEEEE, 0xEEEEEEEE,
    0xEEEEEEEE, 0xEEEEEEEE, 0xEEEEEEEE, 0xEEEEEEEE, 0xEEEEEEEE, 0xEEEEEEEE, 0xEEEEEEEE, 0xEEEEEEEE,
    0xEEEEEEEE, 0xEEEEEEEE, 0xEEEEEEEE, 0xEEEEEEEE, 0xEEEEEEEE, 0xEEEEEEEE, 0xEEEEEEEE, 0xEEEEEEEE,
    0xEEEEEEEE, 0xEEEEEEEE, 0xEEEEEEEE, 0xEEEEEEEE, 0xEEEEEEEE, 0xEEEEEEEE, 0xEEEEEEEE, 0xEEEEEEEE,
    0xEEEEEEEE, 0xEEEEEEEE, 0xEEEEEEEE, 0xEEEEEEEE, 0xEEEEEEEE, 0xEEEEEEEE, 0xEEEEEEEE, 0xEEEEEEEE,
    0xEEEEEEEE, 0xEEEEEEEE, 0xEEEEEEEE, 0xEEEEEEEE, 0xEEEEEEEE, 0xEEEEEEEE, 0xEEEEEEEE, 0xEEEEEEEE,
    0x7F01FF80, 0x8000FFFF, 0x12345678, 0x9ABCDEF0, 0xEEEEEEEE, 0xEEEEEEEE, 0xEEEEEEEE, 0xEEEEEEEE,
]
# cs_fmt_sint32.asm: hand-derived full-width and D16 load results, inputs, stores, and untouched words.
FMT_SINT32_WORDS = [
    0x80000000, 0x7FFFFFFF, 0x80000000, 1,
    0x80000000, 0x7FFFFFFF, 0xFFFFFFFD, 0x80000000,
    0x80000000, 0x7FFFFFFF, 0xFFFFFFFD, 17,
    0x80000000, 0x7FFFFFFF, 0, 0,
    0x80000000, 0x7FFFFFFF, 0xFFFFFFFD, 0,
    0x80000000, 0x7FFFFFFF, 0xFFFFFFFD, 17,
    0x7FFF8000, 0x0011FFFD, 0x7FFF8000, 0x0011FFFD,
    0xFFFFFFFD, 0xFFFFFFFD, 0xEEEEEEEE, 0xEEEEEEEE,
    0x80000000, 0x7FFFFFFF, 0xFFFFFFFD, 17,
    0x80000000, 0x7FFFFFFF, 0xEEEEEEEE, 0xEEEEEEEE,
    0x80000000, 0x7FFFFFFF, 0xFFFFFFFD, 0xEEEEEEEE,
    0x80000000, 0x7FFFFFFF, 0xFFFFFFFD, 17,
    0xFFFFFFFD, 0xFFFF8000, 0xFFFFFFFD, *[0xEEEEEEEE] * 13,
]
# cs_fmt_all.asm: 32 typed loads of the SCALED, SINT, small-float and 10:10:10:2 formats (29 through a V# FORMAT, 3 through
# tbuffer's own), their 16-byte inputs, and 20 typed stores; the words are tools/buffer_format_fuzz.py's MODEL, which the
# Radeon 880M (gfx1150) and a real RDNA2 (gfx1036, @AwkGrep256, #2022) reproduced for every format here. Layout and rules in
# the .asm header.
FMT_ALL_WORDS = [
    0x000000ff, 0x000000ff, 0x000000ff, 0x000000ff, 0xffffff80, 0xffffff80, 0xffffff80, 0xffffff80, 0x43480000,
    0x43480000, 0x43480000, 0x43480000, 0xc2c80000, 0xc2c80000, 0xc2c80000, 0xc2c80000, 0xffff8000, 0xffff8000,
    0xffff8000, 0xffff8000, 0x476a6000, 0x476a6000, 0x476a6000, 0x476a6000, 0xc47a0000, 0xc47a0000, 0xc47a0000,
    0xc47a0000, 0x0000007f, 0xffffff80, 0x0000007f, 0xffffff80, 0x3f800000, 0x437f0000, 0x3f800000, 0x437f0000,
    0x00000001, 0xffffffff, 0xffffff80, 0x0000007f, 0x00000000, 0x3f800000, 0x43000000, 0x437f0000, 0x00000000,
    0x3f800000, 0xc3000000, 0xbf800000, 0x00007fff, 0xffff8000, 0x00007fff, 0xffff8000, 0x3f800000, 0x477fff00,
    0x3f800000, 0x477fff00, 0xc7000000, 0x46fffe00, 0x3f800000, 0xbf800000, 0x3f800000, 0x40000000, 0x3f000000,
    0x3f800000, 0x7f800000, 0x7f820000, 0x36000000, 0x7f800000, 0x477e0000, 0x387c0000, 0x7ffc0000, 0x477e0000,
    0x3f800000, 0x40000000, 0x477e0000, 0x3f800000, 0x3f800000, 0xbf800000, 0x3f800000, 0xbf800000, 0xfffffffe,
    0xfffffe00, 0x000001ff, 0xffffffff, 0xbf800000, 0x3f800000, 0xbf800000, 0x3f800000, 0x447fc000, 0x00000000,
    0x44000000, 0x40400000, 0xc4000000, 0x43ff8000, 0xbf800000, 0xc0000000, 0xfffffe00, 0x000001ff, 0xffffffff,
    0xfffffffe, 0x3f800000, 0x40c00000, 0x7ffe0000, 0x00000000, 0x00000001, 0xffffffff, 0xffffff80, 0x0000007f,
    0x00007fff, 0xffff8000, 0x00000000, 0x00000000, 0x00000000, 0x3f800000, 0x3f800000, 0x00000000, 0xbf800000,
    0x3f800000, 0xbf800000, 0xbf800000, 0x000007ff, 0x00000001, 0x000003ff, 0x000007ff, 0xffffffff, 0xfffffc00,
    0xffffffff, 0xffffffff, 0x000000ff, 0x00000000, 0x00000000, 0x00000000, 0x00000080, 0x00000000, 0x00000000,
    0x00000000, 0x000000c8, 0x00000000, 0x00000000, 0x00000000, 0x0000009c, 0x00000000, 0x00000000, 0x00000000,
    0x00008000, 0x00000000, 0x00000000, 0x00000000, 0x0000ea60, 0x00000000, 0x00000000, 0x00000000, 0x0000fc18,
    0x00000000, 0x00000000, 0x00000000, 0x0000807f, 0x00000000, 0x00000000, 0x00000000, 0x0000ff01, 0x00000000,
    0x00000000, 0x00000000, 0x7f80ff01, 0x00000000, 0x00000000, 0x00000000, 0xff800100, 0x00000000, 0x00000000,
    0x00000000, 0xff800100, 0x00000000, 0x00000000, 0x00000000, 0x80007fff, 0x00000000, 0x00000000, 0x00000000,
    0xffff0001, 0x00000000, 0x00000000, 0x00000000, 0x7fff8000, 0xffff0001, 0x00000000, 0x00000000, 0x702003c0,
    0x00000000, 0x00000000, 0x00000000, 0x007e0fc0, 0x00000000, 0x00000000, 0x00000000, 0xffc1ffbf, 0x00000000,
    0x00000000, 0x00000000, 0xf7f001e0, 0x00000000, 0x00000000, 0x00000000, 0x805ff801, 0x00000000, 0x00000000,
    0x00000000, 0xffdff802, 0x00000000, 0x00000000, 0x00000000, 0x6017fe00, 0x00000000, 0x00000000, 0x00000000,
    0xe00003ff, 0x00000000, 0x00000000, 0x00000000, 0xbff7fe00, 0x00000000, 0x00000000, 0x00000000, 0xbff7fe00,
    0x00000000, 0x00000000, 0x00000000, 0xfff181e0, 0x00000000, 0x00000000, 0x00000000, 0x7f80ff01, 0x00000000,
    0x00000000, 0x00000000, 0x80007fff, 0x00000000, 0x00000000, 0x00000000, 0xfffff800, 0x00000000, 0x00000000,
    0x00000000, 0x800ffe00, 0x00000000, 0x00000000, 0x00000000, 0xffc00fff, 0x00000000, 0x00000000, 0x00000000,
    0xfff003ff, 0x00000000, 0x00000000, 0x00000000, 0xeeeeee7f, 0xeeeeeeee, 0xeeeeeeee, 0xeeeeeeee, 0xeeeeee80,
    0xeeeeeeee, 0xeeeeeeee, 0xeeeeeeee, 0xeeeeeeff, 0xeeeeeeee, 0xeeeeeeee, 0xeeeeeeee, 0xeeeeeeff, 0xeeeeeeee,
    0xeeeeeeee, 0xeeeeeeee, 0xeeee7fff, 0xeeeeeeee, 0xeeeeeeee, 0xeeeeeeee, 0x80007fff, 0xeeeeeeee, 0xeeeeeeee,
    0xeeeeeeee, 0xeeeebfff, 0xeeeeeeee, 0xeeeeeeee, 0xeeeeeeee, 0x7fd00401, 0xeeeeeeee, 0xeeeeeeee, 0xeeeeeeee,
    0xffe017ff, 0xeeeeeeee, 0xeeeeeeee, 0xeeeeeeee, 0x37ff7bff, 0x80007c01, 0xeeeeeeee, 0xeeeeeeee, 0x00004ccd,
    0xeeeeeeee, 0xeeeeeeee, 0xeeeeeeee, 0xc040817f, 0xeeeeeeee, 0xeeeeeeee, 0xeeeeeeee, 0x702003c0, 0xeeeeeeee,
    0xeeeeeeee, 0xeeeeeeee, 0xf7fe0000, 0xeeeeeeee, 0xeeeeeeee, 0xeeeeeeee, 0xf83efde0, 0xeeeeeeee, 0xeeeeeeee,
    0xeeeeeeee, 0x265ff805, 0xeeeeeeee, 0xeeeeeeee, 0xeeeeeeee, 0xd00803ff, 0xeeeeeeee, 0xeeeeeeee, 0xeeeeeeee,
    0x405801ff, 0xeeeeeeee, 0xeeeeeeee, 0xeeeeeeee, 0x027ff017, 0xeeeeeeee, 0xeeeeeeee, 0xeeeeeeee, 0x7fff8000,
    0xffff7fff, 0xeeeeeeee, 0xeeeeeeee,
]
# cs_fmt_xy_xyz.asm (TEST-185): 12 xy/xyz loads, then the 9 store regions read back, as a Radeon 680M (gfx1035) gives
# them natively, with the updated UINT saturation case checked on gfx1036. Layout and rules in the .asm header.
FMT_XY_XYZ_WORDS = [
    0x3F800000, 0x40000000, 0x5A5A0008, 0x5A5A000C, 0x3F800000, 0x40000000, 0x40400000, 0x5A5A001C,
    0x3F800000, 0x40000000, 0x5A5A0028, 0x5A5A002C, 0x3F800000, 0x40000000, 0x40400000, 0x5A5A003C,
    0x3F800000, 0x40000000, 0x3F800000, 0x5A5A004C, 0x3F800000, 0x3F800000, 0x5A5A0058, 0x5A5A005C,
    0x3F800000, 0x40000000, 0x5A5A0068, 0x5A5A006C, 0x3F800000, 0x40000000, 0x40400000, 0x5A5A007C,
    0x00000000, 0x00000000, 0x5A5A0088, 0x5A5A008C, 0x00000000, 0x00000000, 0x00000000, 0x5A5A009C,
    0x3F800000, 0x40000000, 0x5A5A00A8, 0x5A5A00AC, 0x00000000, 0x00000000, 0x00000000, 0x5A5A00BC,
    0x3F800000, 0x40000000, 0x00000000, 0x00000000, 0x3F800000, 0x40000000, 0x40400000, 0x00000000,
    0x40003C00, 0x00000000, 0xA5A50268, 0xA5A5026C, 0x40003C00, 0x00004200, 0xA5A50278, 0xA5A5027C,
    0x3F800000, 0xA5A50284, 0xA5A50288, 0xA5A5028C, 0x0000FFFF, 0xA5A50294, 0xA5A50298, 0xA5A5029C,
    0x0003FF01, 0xA5A502A4, 0xA5A502A8, 0xA5A502AC, 0xA5A502B0, 0xA5A502B4, 0xA5A502B8, 0xA5A502BC,
    0xA5A502C0, 0xA5A502C4, 0xA5A502C8, 0xA5A502CC]
# cs_tbuffer_store.asm: 13 tbuffer_store_format_* regions read back, as a Radeon 680M (gfx1035) gives them
# natively. Layout and rules in the .asm header.
TBUFFER_STORE_WORDS = [
    0x3F800000, 0x40000000, 0x40400000, 0x40800000, 0x40003C00, 0x00000000, 0xA5A50218, 0xA5A5021C,
    0x00030201, 0xA5A50224, 0xA5A50228, 0xA5A5022C, 0xA5A5FF30, 0xA5A50234, 0xA5A50238, 0xA5A5023C,
    0x40000240, 0xA5A50244, 0xA5A50248, 0xA5A5024C, 0xD00003FF, 0xA5A50254, 0xA5A50258, 0xA5A5025C,
    0x3F800000, 0x40000000, 0xA5A50268, 0xA5A5026C, 0x40003C00, 0xA5A50274, 0xA5A50278, 0xA5A5027C,
    0xFF0080FF, 0xA5A50284, 0xA5A50288, 0xA5A5028C, 0xA5A50290, 0xA5A50294, 0xA5A50298, 0xA5A5029C,
    0xA5A502FF, 0xA5A502A4, 0xA5A502A8, 0xA5A502AC, 0x3F800000, 0xA5A502B4, 0xA5A502B8, 0xA5A502BC,
    0xA5A502C0, 0xA5A502C4, 0xA5A502C8, 0xA5A502CC]
# cs_fmt_d16_xyz.asm: d16_xyz / d16_xy typed-buffer loads (out[0..10]), the two d16_xyz stores (out[16..18] and
# out[20..22]) and the inputs at 0x100 (out[64..72]); everything else is the untouched fill. Derivation in the header.
FMT_D16_XYZ_WORDS = [
    0xABCD1234, 0xA5A57FFF, 0xFFFF0005, 0x1234FFFF, 0xC0003C00, 0xC0003C00, 0xBEEF3555, 0x00003C00,
    0xCAFE3C00, 0xFFFF0005, 0x0BADFFFF, *[0xEEEEEEEE] * 5,
    0x00000007, 0x00008001, 0x0000FFFE, 0xEEEEEEEE, 0x3F800000, 0xC0000000, 0x3F000000, 0xEEEEEEEE,
    *[0xEEEEEEEE] * 40,
    0xABCD1234, 0x55557FFF, 0xC0003C00, 0x7BFF3555, 5, 0x10000, 0xFFFFFFFF, 0, 0x80FF00FF, *[0xEEEEEEEE] * 7]
# Negative global_load/global_store offsets (cs_global_neg_offset.asm): 40 groups make a 0x2800-byte heap that holds
# both the right addresses and the 4096-lower ones; only group 0 thread 0 runs, so no wave64 requirement.
GLOBAL_OFFSET = [("cs_global_neg_offset", "40", [0x5EED1FFC, 0x8BAD1800, 0x5EED1FF0, 0x5EED1FF4,
                                                 0x0000009E, 0xFFFF8BAD, 0x57023FFC, 0xEEEEEEEE]),
                 # global/buffer atomics, 8/16-bit global stores and d16 loads with CPU-only checks until now
                 ("cs_global_mem_ext", "40", [0x5, 0xC, 0x977, 0xFFF00000, 0xEDCB5678, 0xEF332211, 0x887766BE, 0xFF801234])]
# cs_heap_init.asm: vk_compute_test's VKC_HEAP_INIT copies this file (zeros, then four words at 0x100) to the start
# of the heap; the shader copies the four words to 0, and out[4..7] are the file's zeros, not the 0xEE fill.
HEAP_INIT_WORDS = [0x1BADB002, 0xCAFEF00D, 0x80000001, 0x0000FFFF]
HEAP_INIT_FILE = bytes(0x100) + struct.pack("<4I", *HEAP_INIT_WORDS)
HEAP_INIT_EXPECTED = HEAP_INIT_WORDS + [0] * 4


def heap_init_env():
    """vk_compute_test's environment for cs_heap_init: the init file (written to the temp folder) and a buffer big
    enough to hold it (68 words for its 0x110 bytes; one group's default is 64)."""
    path = os.path.join(tempfile.gettempdir(), "leonida_cs_heap_init.bin")
    with open(path, "wb") as f:
        f.write(HEAP_INIT_FILE)
    return {"VKC_HEAP_INIT": path, "VKC_OUT_WORDS": str(len(HEAP_INIT_FILE) // 4)}


# cs_user_sgprs.asm: vk_compute_test puts these words in s4..s7 and the shader stores them back.
USER_SGPRS = [0x4F000000, 0xDEADBEEF, 7, 0xFFFFFFFF]
USER_SGPRS_ENV = {"VKC_SGPRS": ",".join(hex(w) for w in USER_SGPRS)}
USER_SGPRS_WORDS = USER_SGPRS + [0xEEEEEEEE] * 4
# cs_fma_fused.asm: single-rounded f32 multiply-adds (and one f16). The operands arrive in s4..s7 so no driver can fold
# them; FMA_FUSED are the exact single-rounded words and FMA_SPLIT are the exact split-rounded words (derivations in
# the .asm header; out[7] is f16 round-to-odd, never GLSL Fma, so it is always 0x7351). Under --fma khr
# (LEONIDA_FMA=khr, shaderFmaFloat32 in index.json) every word must match FMA_FUSED; under default --fma glsl a
# spec-compliant driver (e.g. Mesa RADV 26.2.6) may legally split GLSL.std.450 Fma.
FMA_FUSED_ENV = {"VKC_SGPRS": "0x3f800800,0x3f800001,0x3f7ffffe,0xbf800002"}
FMA_FUSED = [0x33800000, 0xa8800000, 0x3a000400, 0x33800000, 0x33800000, 0x33800000, 0x28800000, 0x00007351]
FMA_SPLIT = [0x00000000, 0x00000000, 0x3a000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00007351]
# cs_cvt_f32_sat.asm (#300): s4 is a binary32 input, out[0] = v_cvt_u32_f32, out[1] = v_cvt_i32_f32 (truncate,
# saturate, NaN -> 0). One dispatch per input through VKC_SGPRS; the inputs of tests/semantic/test_float_to_int.py.
CVT_F32_SAT_CASES = (
    (0x00000000, 0, 0), (0x80000000, 0, 0), (0x00000001, 0, 0), (0x3FC00000, 1, 1), (0xBFC00000, 0, 0xFFFFFFFF),
    (0x4EFFFFFF, 0x7FFFFF80, 0x7FFFFF80), (0x4F000000, 0x80000000, 0x7FFFFFFF), (0x4F7FFFFF, 0xFFFFFF00, 0x7FFFFFFF),
    (0x4F800000, 0xFFFFFFFF, 0x7FFFFFFF), (0xCEFFFFFF, 0, 0x80000080), (0xCF000000, 0, 0x80000000),
    (0xCF000001, 0, 0x80000000), (0x7F800000, 0xFFFFFFFF, 0x7FFFFFFF), (0xFF800000, 0, 0x80000000),
    (0x7F800001, 0, 0), (0xFFC00000, 0, 0))
# v_cvt_pk_u8_f32 (cs_cvt_pk_u8.asm), measured natively on RDNA2 (gfx1036): halves round to even, NaN gives 0.
CVT_PK_U8_WORDS = [0x1122330C, 0x11220E44, 0x11003344, 0xFE223344, 0x1122330D, 0x11220044, 0x11FF3344, 0x00223344]
# v_pk_mul_lo_u16 with a constant operand (cs_pk_mul_lo_const.asm), measured natively on RDNA2 (gfx1036); lanes 0 and 1.
PK_MUL_LO_WORDS = [0x00009E88, 0x03675433, 0x00230000, 0x00077F6C] * 2
# Wave64-only dump tests (mostly the x = cross-half checks): exact results need 64 lanes, or an emulation of them
# on narrower subgroups. cs_dpp_w64 is not cross-half, but its lane id needs both halves.
W64_EMULATED = os.environ.get("LEONIDA_W64_EMULATE") == "1"
# LEONIDA_BARY_FALLBACK=1/0 overrides the device probe for raw barycentrics. Unset, Mesa ANV selects the fallback.
BARY_FALLBACK = barycentric_fallback()
# LEONIDA_F16_INTERP_FALLBACK=1 (as build.py reads it): *_f16 interps were translated from decoded-half inputs, with no
# PerVertexKHR inputs at all, so the translator refuses a computed I/J pair (ps_ij_computed) there.
F16_HALVES = os.environ.get("LEONIDA_F16_INTERP_FALLBACK") == "1"
W64X_DUMPS = [("cs_dpp_w64", [1, 0, 3, 2, 5, 4, 7, 6]),
              ("cs_w64x_readlane", [121, 121, 16, 97, 16, 100, 100, 121]),
              ("cs_w64x_writelane", [4660, 77, 99, 2748, 0, 0, 4660, 99]),
              ("cs_w64x_readfirstlane", [140, 140, 163, 132, 100, 105, 133, 48]),
              ("cs_w64x_bpermute", [36, 316, 253, 218, 435, 225, 0, 442]),
              # SHADER-90: ds_permute_b32 on each wave32 half; unwritten lane 0, EXEC gates the destination
              ("cs_w64x_permute", [218, 442, 435, 239, 0, 99, 239, 232]),
              ("cs_w64x_branch", [0, 0, 0, 0, 5, 5, 9, 9]),
              ("cs_w64x_mask_reuse", [8, 40, 8, 40, 2, 20, 10, 64]),
              ("cs_w64x_mask_reuse2", [10, 20, 8, 8, 5, 2, 132, 131]),
              ("cs_w64x_chain_barrier", [564, 563, 500, 627]),                   # 128-thread group
              ("cs_w64x_ballot", [2053, 2061, 2069, 2077, 2085, 2093, 2101, 2109]),
              ("cs_w64x_wg128_branches", [121, 121, 343, 343, 14, 31, 19812, 22728]),   # 128-thread group
              ("cs_w64x_lds_raw", [132, 137, 100, 108, 131, 101, 149, 118]),
              ("cs_w64x_lds_war", [1032, 1063, 1000, 1031, 2032, 2063, 2000, 2031]),
              ("cs_w64x_sbarrier", [605, 605, 686, 686, 3032, 3013, 3032, 3031]),         # 128-thread group
              ("cs_w64x_load_ballot_load", [43, 74, 75, 42, 42, 18, 41, 32]),
              # s_andn2_saveexec_b64 / s_andn2_wrexec_b64: EXEC = S0 & ~EXEC over both halves, D old / new EXEC, SCC
              ("cs_w64x_andn2_exec", [1, 16, 32, 4144, 2052, 3, 8, 1544]),
              # s_and/s_or_saveexec_b64 (VCC and SGPR-pair sources), s_andn2_saveexec/wrexec_b64 (SGPR pair): D, EXEC, SCC
              ("cs_w64x_saveexec_b64", [6160, 12304, 5120, 2088, 6168, 8192, 7176, 1837171])]
# A 40-thread group (one partial wave) cannot use --require-wave64 (full subgroups), so it asks for subgroup size 64.
W64X_PARTIAL = [("cs_w64x_partial", [40, 40, 528, 0, 0, 532, 539, 524820]),
                ("cs_w64x_partial_scmp", [0, 1, 0, 1, 1, 1, 1, 0])]
PIXEL_BITS = {"ps_clip_color": (0x3e800000, 0x3f000000, 0x3f400000, 0x3f800000),
              # exp mrt0 done (no vm) with every lane on, EXEC off afterwards: written (.asm header, measured on gfx1036)
              "ps_exp_color_mask": (0x40000000, 0x40800000, 0x40000000, 0x40800000),
              "ps_legacy_mul": (0, 0, 0, 0x40c00000),
              "ps_mul_legacy_denorm": (0, 0, 0, 0x80000000),
              "ps_mul_legacy_denorm_mods": (0, 0, 0, 0x40400000),
              "ps_ops": (0x48804300, 5, 7, 0xbffffffc), "ps_ops2": (0xabcd4400, 0x46001234, 0xbffffffa, 7),
              "ps_ops3": (0xbffffffa, 0x44660066, 0x44aabbcc, 16), "ps_ops4": (0x34000010, 0x00120800, 0xbeef0000, 0x12343c00),
              "ps_ops5": (0x12340010, 0xbeef0800, 0xaaaaffff, 0xdeadbc00),
              "ps_ops6": (0xf800ffff, 0x01000010, 0x00800800, 0x8000fff0),
              "ps_ops7": (0x00ff8077, 0xff0000ff, 0x89abcdef, 0x00070003),
              "ps_ops8": (0x80000000, 0xbfffffff, 0x000f100e, 0x0f0f0f1f),
              "ps_f32_mods": (0xc0256a01, 0x3f800000, 0x41e00000, 0x3fc00000),
              "ps_scc": (1, 1, 3, 0x80000000), "ps_sdwa": (0x000000f1, 0xbffefff2, 0x3c003344, 0xbff70000),
              # 16-bit SDWA results with dst_sel:DWORD zero the high half (.asm header, measured on gfx1036)
              "ps_sdwa_dword_hi": (0x00004000, 0x00000003, 0x00004700, 0x0000c000),
              "ps_vcc_x2": (0x11223344, 0x55667788, 0, 0),
              # v_cmpx_{eq,gt,le,lt,ngt,nle}_f32, f_i32, t_u32, {eq,gt,le}_i64, {ge,lt,ne}_u64 (.asm header)
              "ps_vcmpx_forms": (0xf05b2a48, 0xf05e8a12, 0xb174be18, 0x15550000),
              # v_cmp_{eq,ne,lt,le,gt,ge,f,t}_{u32,i32}, e32 and e64, constants and literals (.asm header)
              "ps_vcmp_int32": (0x963e52e8, 0x984dcee8, 0xb9458ee8, 0xb73612e8),
              "ps_med3_mods": (0x3f800000, 0x40400000, 0x40c00000, 0x3f400000),
              "ps_scc_minmax": (0x00030205, 0x00030207, 0x00030209, 0x00030203),
              "ps_scc_logic32": (0x0fff0fff, 0x000f000f, 0xf00ff00f, 0x36),
              "ps_scc_logic64": (0x0fff0fff, 0xff000000, 0xff0000ff, 0x16),
              "ps_fma_literal": (0x41b80000, 0x41800000, 0x40e00000, 0x41a40000),
              "ps_f64_mods": (0x40040000, 0x3ff00000, 0x3fe40000, 0x3ff00000),
              "ps_f64_round": (0xc12af665, 0x3fe7ffff, 0xbfffffff, 0xc12af666),
              "ps_outmods_f16": (0xaaaa3c00, 0xaaaa3e00, 0x7c000000, 0x3c000000),
              # f16 op_sel, partial writes, inline constants: words measured on RDNA2 hardware (.asm headers)
              "ps_f16_med3_opsel": (0xaaaa4100, 0xaaaa4200, 0xaaaa3e00, 0x40005555),
              "ps_f16_med3_mods": (0x41005555, 0x3c005555, 0x41003800, 0xaaaaba00),
              "ps_f16_vop2": (0xaaaa4000, 0xaaaac000, 0xaaaa3a00, 0x42003f00),
              "ps_f16_vop2b": (0xaaaa3e00, 0xaaaab800, 0xaaaa3c00, 0x42003400),
              "ps_f16_vop1": (0xaaaa4000, 0xaaaabc00, 0xaaaac000, 0xaaaa8000),
              "ps_f16_vop1b": (0xaaaa3400, 0xaaaa4000, 0xaaaab400, 0xaaaa3800),
              "ps_f16_const": (0xaaaa3d00, 0x38005555, 0xaaaa3800, 0x3f000000),
              "ps_f16_const2": (0x42003d00, 0x3d004200, 0xaaaabc00, 3),
              "ps_f16_pack": (0x34003c00, 0x38003400, 0x34000000, 0x44004200),
              # v_pack_b32_f16 neg/abs on the selected half: a sign flip / clear before packing (SHADER-260)
              "ps_f16_pack_mods": (0x4400b400, 0xc4003400, 0x4400c200, 0xb400b400),
              # SDWA src_sel on a float inline constant in an f16 op (measured on gfx1036)
              "ps_sdwa_f16_const": (0xaaaa3800, 0xaaaa1c00, 0xaaaa3e00, 0xaaaa73f0),
              "ps_sff_imm": (2, 1, 20, 16),
              "ps_fnan_med3": (0x40a00000, 0xc0200000, 0xc0200000, 0x40600000),
              "ps_fnan_fract": (0x3f7fffff, 0x00003bff, 0xc0200000, 0xc0200000),
              # v_med3 with a NaN in S2 or in two operands, by the same measured rule (derivations in the .asm headers)
              "ps_fnan_med3_s2": (0x40a00000, 0x40a00000, 0xc0200000, 0x40e00000),
              "ps_f16_fnan_med3": (0xaaaa4500, 0xaaaa4500, 0xaaaac100, 0xaaaa4700),
              # TEST-175: docs/design.md rules that were checked on the CPU only (derivations in the .asm headers)
              "ps_cmp_nlg": (0x40000037, 0x40000007, 0x40000001, 0x3f800000),
              # far branches: s_getpc/s_add|s_sub/s_addc|s_subb/s_setpc forward, backward loops, conditional (.asm header)
              "ps_far_branch": (0x00000007, 0x00000013, 0x00000051, 0x0000000b),
              # 16-bit float v_cmp, all 16 predicates, e32 and e64 with swapped operands (.asm headers)
              "ps_vcmp_f16": (0x48a12080, 0x4ce9b1a2, 0x4ba72c98, 0x4fefbdba),
              "ps_vcmp_f16_e64": (0x44c11100, 0x4cf171c1, 0x47c71d18, 0x4ff77dd9),
              # 16-bit float v_cmpx, all 16 predicates, EXEC result with VCC kept (.asm headers)
              "ps_vcmpx_f16": (0x68a12080, 0x6ce9b1a2, 0x6ba72c98, 0x6fefbdba),
              "ps_vcmpx_f16_e64": (0x64c11100, 0x6cf171c1, 0x67c71d18, 0x6ff77dd9),
              "ps_cvt_off": (0xbf000000, 0x3ee00000, 0xbe800000, 0x3f800000),
              "ps_fmac_f16_mods": (0xaaaa3c00, 0xaaaa3a00, 0xaaaa0000, 0xaaaa4400),
              "ps_vop3_const_mods": (0xc0000000, 0xbf800000, 0x3f800000, 0xaaaabc00),
              "ps_cndmask_sign_mods": (0xff800001, 0x7f800001, 0x80000001, 0x7f800001),
              "ps_interp_f32_mods": (0x3f800000, 0x00000000, 0x3f400000, 0x3f800000),
              "ps_interp_mov_mods": (0x3f800000, 0x3fc00000, 0x40000000, 0x3f400000),
              # v_mad_u32_u16 / v_mad_i32_i16 and v_subrev_f16, which had CPU checks only (derivations in the .asm headers)
              "ps_valu16_mad": (0x00000002, 0x00030002, 0x00008000, 0x00000004),
              # op_sel on S2 / D and clamp, measured on gfx1036 (EGEN-HW-MAD-U32-U16)
              "ps_valu16_mad_sel": (0x00008003, 0xdb880000, 0xffffffff, 0x80000000),
              "ps_valu16_subrev": (0xaaaabc00, 0xaaaac000, 0xaaaa3c00, 0x42003800),
              # f32 VOPC ordered/unordered with a NaN, 64-bit integer VOPC (derivations in the .asm headers)
              "ps_vcmp_f32_nan": (0x0000376c, 0x0000fd5b, 0x0000d36c, 0x00000fe7),
              "ps_vcmp_64": (0x0000b3c4, 0x000091e6, 0x00006787, 0x00007c93),
              # 32-bit integer VALU ops on the VINT_ENV heap: 24-bit sign extension, shifts of 32+, signed vs unsigned
              # min/max (derivations in the .asm headers)
              "ps_vint_logic3": (0xfffffffe, 0xbbb05070, 0x1d3b5977, 0x123f5f7e),
              "ps_vint_bits": (0xfffffffe, 0x40000000, 0x1e6a2c48, 0x000001d5),
              "ps_vint_mul24": (0x13345678, 0xfffffffd, 0x00007fff, 0x0000ffff),
              "ps_vint_minmax": (0x7ffffffe, 0x80000001, 0x80000001, 0x7ffffffe),
              "ps_vint_minmax3": (0xf0f0f0f0, 0xab800000, 0xf0f0f0f0, 0xab800000),
              # v_readlane_b32 across both halves of a wave64 (the .asm header derives the words)
              "ps_w64x_readlane": (0x40000079, 0x40000010, 0x40000064, 0x400000be)}
# Pixel shaders that are exact only when the pixel stage really ran 64 lanes wide. vk_harness says so when it chained
# a required subgroup size onto the fragment stage ("fragment subgroup size: N", the whole line; a device that could
# not take the size gets "fragment subgroup size: N not available" or no line). A device that cannot
# require 64 there (NVIDIA: 32 only; llvmpipe: 8) has no frame to judge: the test is skipped, not failed.
WIDE_PIXEL_TESTS = {"ps_w64x_readlane": 64}
# h0..h7 of the ps_vint_* fixtures: 24-bit fields with bit 23 set and garbage above, 33 (a shift of 32+), a
# negative i32 that is a large u32, and plain bit patterns.
VINT_ENV = {"VKH_SGPRS": "0x1000,0,32,0",
            "VKH_HEAP": "0x1000=0xab800000,0x1004=0x12fffffe,0x1008=0x00000021,0x100c=0x80000001,"
                        "0x1010=0x7ffffffe,0x1014=0xf0f0f0f0,0x1018=0x12345678,0x101c=0xffffffff"}
# Pixel shaders that read memory: vk_harness puts this V# in s[0:3] and seeds the heap it points at, as
# tests/semantic/test_value_semantics.py does for the interpreter.
PIXEL_ENV = {"ps_vcc_x2": {"VKH_SGPRS": "0x1000,0,16,0", "VKH_HEAP": "0x1000=0x11223344,0x1004=0x55667788"},
             # the 14 v_cmpx forms: f32 and 64-bit operands from memory (derivations in the .asm header)
             "ps_vcmpx_forms": {"VKH_SGPRS": "0x1000,0,96,0",
                                "VKH_HEAP": "0x1000=0x7fc10001,0x1004=0x3f800000,0x1008=0x40000000,0x100c=0x80000000,"
                                            "0x1010=0,0x1014=0xff800000,0x1018=0,0x101c=0,0x1020=0x80000000,0x1024=0,"
                                            "0x1028=0x1,0x102c=0,0x1030=0x7,0x1034=0x80000000,0x1038=0x7,0x103c=0,"
                                            "0x1040=0x1,0x1044=0x5,0x1048=0x80000000,0x104c=0x5,0x1050=0x7,0x1054=0x5,"
                                            "0x1058=0x7,0x105c=0x5"},
             # 32-bit integer compares: operands from memory (derivations in the .asm header)
             "ps_vcmp_int32": {"VKH_SGPRS": "0x1000,0,32,0",
                               "VKH_HEAP": "0x1000=0x1,0x1004=0,0x1008=0x80000000,0x100c=0x12345678,0x1010=0x3,"
                                           "0x1014=0x5,0x1018=0,0x101c=0"},
             # float NaN / fract rows measured on RDNA2 hardware (see the .asm headers)
             "ps_fnan_med3": {"VKH_SGPRS": "0x1000,0,32,0",
                              "VKH_HEAP": "0x1000=0x7fc10001,0x1004=0x40e00000,0x1008=0x40a00000,"
                                          "0x100c=0x7fc10001,0x1010=0x40600000,0x1014=0xc0200000"},
             "ps_fnan_fract": {"VKH_SGPRS": "0x1000,0,16,0",
                               "VKH_HEAP": "0x1000=0xaedbe6ff,0x1004=0x00008800,0x1008=0x7fc10001,0x100c=0xc0200000"},
             "ps_fnan_med3_s2": {"VKH_SGPRS": "0x1000,0,16,0",
                                 "VKH_HEAP": "0x1000=0x40a00000,0x1004=0x40e00000,0x1008=0x7fc10001,0x100c=0xc0200000"},
             "ps_f16_fnan_med3": {"VKH_SGPRS": "0x1000,0,16,0",
                                  "VKH_HEAP": "0x1000=0x00004500,0x1004=0x00004700,0x1008=0x00007e01,0x100c=0x0000c100"},
             # f32 VOPC with a NaN, 64-bit integer VOPC: inputs from memory (derivations in the .asm headers)
             "ps_vcmp_f32_nan": {"VKH_SGPRS": "0x1000,0,16,0",
                                 "VKH_HEAP": "0x1000=0x7fc10001,0x1004=0x3f800000,0x1008=0x40000000"},
             "ps_vcmp_64": {"VKH_SGPRS": "0x1000,0,64,0",
                            "VKH_HEAP": "0x1000=0x80000000,0x1004=0,0x1008=1,0x100c=0,0x1010=0xffffffff,"
                                        "0x1014=0xffffffff,0x1018=1,0x101c=0,0x1020=1,0x1024=0,0x1028=0x80000000,"
                                        "0x102c=0,0x1030=7,0x1034=5,0x1038=7,0x103c=5"},
             # the ps_vint_* integer fixtures share one heap, h0..h7 (see their .asm headers)
             "ps_vint_logic3": VINT_ENV,
             "ps_vint_bits": VINT_ENV,
             "ps_vint_mul24": VINT_ENV,
             "ps_vint_minmax": VINT_ENV,
             "ps_vint_minmax3": VINT_ENV}
# ps_gather_wrap.asm (SHADER-36): image_gather4_l_o under each S# address mode, run by vk_texture_test (the 8x8 gradient
# texture); the expected channels are derived in the .asm header.
TEXTURE_CONSTANTS = {
    "ps_gather_wrap": (5090733.0, 14942181.0, 258623.0, 9702309.0),
    "ps_gather_border": (32776.0, 34935.0, 34680.0, 28679.0),  # SHADER-348: border modes 6 and 7, derived in the .asm
    "ps_gather4_c_l": (6.0, 9.0, 17459.0, 9.0),  # image_gather4_c_l / _c_l_o / _cl / _c_b_cl_o d16
    "ps_array_resinfo": (8.0, 8.0, 12.0, 1.0),  # binding 3, index 1: two 6-face cubes, one mip
}
CLIP_TESTS = ("vs_clip", "vs_clip_param4", "vs_clip_undeclared")
NAN_PAYLOAD_TEST = "ps_nan_payload"
# One definition: a later assignment used to drop ps_texoff (and any entry added by an earlier assignment) from
# the standalone run. Keep every pixel fixture in this tuple.
PIXEL_TESTS = sorted((*PIXEL_BITS, "ps_bary", "ps_persp_bary", "ps_flat", "ps_bitmask_perm", "ps_texoff",
                      "ps_texoff_implicit", "ps_texoff_forms",
                      "ps_uniform_branch", "ps_wqm", "ps_scc_mask32", "ps_interp_f16_high", "ps_ij_computed",
                      "ps_mandelbrot", NAN_PAYLOAD_TEST))
# Pixel fixtures drawn with another vertex shader than vs_ids.
PIXEL_VERTEX = {"ps_interp_f16_high": "vs_f16_pack", "ps_persp_bary": "vs_persp",
                "ps_interp_f32_mods": "vs_interp_consts", "ps_interp_mov_mods": "vs_interp_consts"}
# vs_persp's triangle (see vs_persp.asm): screen positions, clip w and attr0.xy of its three vertices.
PERSP_SCREEN = ((5.0, 8.0), (57.0, 16.0), (21.0, 55.0))
PERSP_W = (1.0, 2.0, 4.0)
PERSP_ATTR = ((0.25, -0.5), (2.25, 0.75), (-1.0, 3.0))
TEXOFF_ARGS = ("2", "92", "0", "0", "0")              # ps_texoff: T# pointer in s2, then the shader's literal S#
TEXTURE_MIP_TEST = "ps_texture_mip0"
SIGNED_MIP_TEST = "ps_sample_implicit_sint"
UINT_MIP_TEST = "ps_sample_implicit_uint"
GATHER_LOD_TEST = "ps_gather_sampler_lod"
TEXTURE_MIP_DEPENDENCIES = ("vs_ids", "ps_texture_sample")
# Each pair of draws expects mip 0, then mip 1. S# word 1 is minLOD[11:0], maxLOD[23:12], unsigned 4.8.
MIP_TEST_PASSES = {
    TEXTURE_MIP_TEST: ((TEXTURE_MIP_TEST, "c00000"), ("ps_texture_sample", "c00000")),
    GATHER_LOD_TEST: ((GATHER_LOD_TEST, "40000"), (GATHER_LOD_TEST, "1000c0")),
    SIGNED_MIP_TEST: ((SIGNED_MIP_TEST, "0"), (SIGNED_MIP_TEST, "100000")),
    UINT_MIP_TEST: ((UINT_MIP_TEST, "0"), (UINT_MIP_TEST, "100000")),
}
# The integer fixtures sample the same two-mip image through the signed or unsigned view the harness binds for them.
MIP_TEST_MODES = {SIGNED_MIP_TEST: "--two-mip-sint", UINT_MIP_TEST: "--two-mip-uint"}

# TEST-184: vk_store_test "words" fills the r32 uint image with 100 + i, the r32 sint image with -2i (texel
# i = 8y + x) and the r32 uint volume with 200 + i (texel i = 16z + 4y + x), and prints the three after the dispatch.
# Per fixture: the words it changes, {word index: value}, in the uint image, the sint image and the volume, as
# derived in the .asm header. Every other word must still hold its fill.
STORE_IMAGES = ("uint", "sint", "volume")
STORE_FILL = (lambda i: 100 + i, lambda i: -2 * i & 0xFFFFFFFF, lambda i: 200 + i)
STORE_WORDS = {
    # 2D image atomics: the texel afterwards (uint image) and the data register afterwards (sint image) of cases
    # 0..15, then the texel 64 threads each added 1 to
    "cs_imgatomic": (dict(enumerate([0x54, 0xFFFFFF9D, 0x06, 0xF067, 0x97, 0xFFFFFFF0, 0x6A, 0x6B,
                                     0xC8, 0x6D, 0x05, 0xFFFFFFF0, 0x70, 0xC0FFEE, 0x72, 0x7A, 0xB4])),
                     dict(enumerate([*range(100, 115), 7])), {}),
    # 3D image atomics: the data register afterwards (sint image) and the six texels afterwards (volume)
    "cs_imgatomic3d": ({}, dict(enumerate([0x101, 0xDB, 0xF4, 0xCE, 0xE6, 0xFF])),
                       {57: 0x111, 19: 3, 44: 0xF4, 6: 0xD00D, 30: 0xE6, 39: 0x10}),
    # image_get_resinfo on a 16x4 image with five levels: (width, height, depth, levels) of levels 0, 1 and 3, then
    # width and height of level 2 and an untouched register, height and levels of level 1, the levels alone, and
    # level 5, past the last (2D depth 0 and the past-the-end zeros measured on RDNA2, SHADER-388)
    "cs_resinfo": ({**dict(enumerate([16, 4, 0, 5, 8, 2, 0, 5, 2, 1, 0, 5, 4, 1, 0xABCD])), 16: 2, 17: 5, 18: 5,
                    20: 0, 21: 0, 22: 0, 23: 5},
                   {}, {}),
}
# TEST-184: image_get_lod on vk_texture_test's textures. Per fixture, its draws: S# word 1 (MIN_LOD [11:0], MAX_LOD
# [23:12], 4.8 fixed point), whether the two-mip texture is drawn, and mrt0, as derived in the .asm header:
# ps_get_lod = (clamped LOD, calculated LOD, calculated LOD at 4u, clamped LOD at 4u) on the two-mip texture,
# ps_get_lod_array = (clamped, calculated LOD at 32u, clamped, calculated LOD) on the one-level 2D array.
# A device may keep as few as four fraction bits of LOD (Vulkan's smallest mipmapPrecisionBits), hence the tolerance.
GET_LOD_TESTS = {
    "ps_get_lod": (("c00000", True, (1.0, 1.0, 3.0, 1.0)), ("40000", True, (0.0, 1.0, 3.0, 0.0))),
    "ps_get_lod_array": (("c00000", False, (0.0, 2.0, 0.0, -3.0)),),
}
GET_LOD_TOLERANCE = 1 / 16


# Windows error codes for a program the system refuses to start, and who refused it.
BLOCKED_BY = {4551: "Windows Application Control (Smart App Control or an App Control policy)",
              1260: "a Windows software restriction policy",
              225: "Microsoft Defender, which reported it as a threat"}


def start_failure(exe, err):
    """One result line for a test program that could not be started, instead of a traceback that ends the run.
    Smart App Control blocks unsigned programs it has no reputation for, so a fresh build can be refused."""
    name = os.path.basename(str(exe))
    who = BLOCKED_BY.get(getattr(err, "winerror", None))
    if who is None:
        return f"could not start {name}: {err}"
    return (f"could not start {name}: blocked by {who}. A freshly built test program has no reputation yet: "
            "allow it, or turn Smart App Control off (Windows Security > App & browser control), and run again")


# A test program that hangs (a driver deadlock, e.g. a barrier not every invocation reaches) must not stall the whole
# suite: each run gets a time limit, is killed when it runs out, and counts as a failure. LEONIDA_GPU_TEST_TIMEOUT
# sets the limit in seconds; 0 turns it off.
DEFAULT_GPU_TEST_TIMEOUT = 120.0


def gpu_test_timeout():
    """Seconds one GPU test program may run (LEONIDA_GPU_TEST_TIMEOUT), or None for no limit."""
    text = (os.environ.get("LEONIDA_GPU_TEST_TIMEOUT") or "").strip()
    if not text:
        return DEFAULT_GPU_TEST_TIMEOUT
    error = f"LEONIDA_GPU_TEST_TIMEOUT must be a finite number of seconds (0 = no limit), not {text!r}"
    try:
        limit = float(text)
    except ValueError:
        sys.exit(error)
    if not math.isfinite(limit):
        sys.exit(error)
    return None if limit <= 0 else limit


def timed_out(name, limit):
    return (f"timed out after {limit:g} s: {os.path.basename(str(name))} was killed (a GPU hang?); "
            "LEONIDA_GPU_TEST_TIMEOUT sets the limit")


def run(*cmd, env=None, cwd=None):
    limit = gpu_test_timeout()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=limit, cwd=cwd)
    except subprocess.TimeoutExpired:
        return 1, timed_out(cmd[0], limit)
    except OSError as e:
        if isinstance(e, FileNotFoundError):
            tools = os.path.abspath(os.path.dirname(cmd[0]))
            missing = [t + EXE for t in TEST_PROGRAMS if not os.path.isfile(os.path.join(tools, t + EXE))]
            if missing:
                sys.exit(not_found_message(tools, missing))
        return 1, start_failure(cmd[0], e)
    return r.returncode, r.stdout + r.stderr


def run_readback(path, launch, *args, **kwargs):
    """Require this invocation to write its framebuffer before comparing it."""
    path = Path(path)
    try:
        path.unlink(missing_ok=True)
    except OSError as err:
        return 1, f"could not remove the previous readback {path.name}: {err}"
    code, output = launch(*args, **kwargs)
    if code == 0 and not path.is_file():
        return 1, f"{output.rstrip()}\nmissing readback: {path.name}".lstrip()
    return code, output


def compute_unsupported(code, output):
    """A Skip quoting vk_compute_test's "unsupported: ..." line when it exited UNSUPPORTED because the module needs a
    feature the device lacks (Int64, Float64, OpFmaKHR), in any run mode; None otherwise. Its --require-wave64
    refusal prints no such line and keeps the callers' own handling, and a 77 without a reason stays a failure."""
    m = re.search(r"^unsupported: .*", output, re.MULTILINE) if code == UNSUPPORTED else None
    return Skip(f"SKIP ({m.group(0)})") if m else None


def failure(code, output, lines):
    details = "\n".join(output.strip().splitlines()[-lines:])
    return f"exit {code}: {details or 'no test output'}"


def dump_test(tools, spirv, name, want, words, env=None):
    """Run one group with a heap of `words` words and compare the first len(want) words of the whole dump. The
    console dump stops at word 7."""
    with tempfile.TemporaryDirectory(prefix="leonida-fmt-") as tmp:
        dump = Path(tmp) / "out.bin"
        # VKC_OUT_WORDS sets a minimum. Pin this dispatch to exactly `words` words.
        child = dict(os.environ if env is None else env, VKC_DUMP=str(dump), VKC_OUT_WORDS=str(words))
        code, out = run(os.path.join(tools, "vk_compute_test" + EXE), os.path.join(spirv, name + ".spv"),
                        "1", "dump", env=child)
        if skip := compute_unsupported(code, out):
            return skip
        if code:
            return failure(code, out, 3)
        if not dump.is_file():
            return "no VKC_DUMP file written"
        data = dump.read_bytes()
    if len(data) != 4 * words:
        return f"expected {4 * words} dump bytes, got {len(data)}"
    for i, (expected, got) in enumerate(zip(want, struct.unpack(f"<{words}I", data))):
        if got != expected:
            return f"word {i}: expected {expected:#010x}, got {got:#010x}"
    return ""


def fmt_1010102_test(tools, spirv, env=None):
    """Check both packed stores and the untouched tail of a single group's 64 words."""
    return dump_test(tools, spirv, "cs_fmt_1010102", FMT_1010102_WORDS + [0xEEEEEEEE] * 48, 64, env)


def fmt_int_narrow_test(tools, spirv, env=None):
    return dump_test(tools, spirv, "cs_fmt_int_narrow", FMT_INT_NARROW_WORDS, len(FMT_INT_NARROW_WORDS), env)


def fmt_sint32_test(tools, spirv, env=None):
    return dump_test(tools, spirv, "cs_fmt_sint32", FMT_SINT32_WORDS, len(FMT_SINT32_WORDS), env)


def fmt_all_test(tools, spirv, env=None):
    """cs_fmt_all: 28 loads, the inputs, 15 stores, and the untouched words between (316 words in all). Wave32, one
    thread: the translator recorded 32 for it, so the run is pinned to a 32-wide subgroup like compute_test does."""
    env = dict(os.environ if env is None else env)
    env.setdefault("VKC_SUBGROUP_SIZE", str(subgroup_size(spirv, "cs_fmt_all") or 32))
    return dump_test(tools, spirv, "cs_fmt_all", FMT_ALL_WORDS, len(FMT_ALL_WORDS), env)


def fmt_xy_xyz_test(tools, spirv, env=None):
    """TEST-185: the 84 result words and the untouched words after them, up to the inputs at 0x200. The heap needs
    0x2d0 bytes, so it is 192 words."""
    return dump_test(tools, spirv, "cs_fmt_xy_xyz", FMT_XY_XYZ_WORDS + [0xEEEEEEEE] * 44, 192, env)


def tbuffer_store_test(tools, spirv, env=None):
    """The 52 region words tbuffer_store_format_* leaves and the untouched words after them, up to the regions at
    0x200. The heap needs 0x2d0 bytes, so it is 192 words."""
    return dump_test(tools, spirv, "cs_tbuffer_store", TBUFFER_STORE_WORDS + [0xEEEEEEEE] * 76, 192, env)


def fmt_d16_xyz_test(tools, spirv, env=None):
    """The d16_xyz/d16_xy loads and d16_xyz stores of cs_fmt_d16_xyz.asm: the whole 80-word heap."""
    return dump_test(tools, spirv, "cs_fmt_d16_xyz", FMT_D16_XYZ_WORDS, len(FMT_D16_XYZ_WORDS), env)


class Skip(str):
    """A result that is neither a pass nor a failure: the device can't run the test."""


def check_mesh_v5(data):
    if len(data) != 24 * 16:
        return f"expected 24 RGBA32F pixels (384 bytes), got {len(data)} bytes"
    pixels = iter(struct.iter_unpack("<4f", data))
    errors = []
    for z in range(4):
        for y in range(3):
            for x in range(2):
                pixel = next(pixels)
                # v5 = linear group index * NumThreads + lane (#209); the shader records lanes 0, 1 and 31 of 32
                g = x + 2*y + 6*z
                want = (32 * g, 32 * g + 1, 32 * g + 31, 1)
                if pixel != want:
                    errors.append(f"({x},{y},{z}): got {pixel}, expected {want}")
    return f"{len(errors)}/24 mismatches; first {errors[0]}" if errors else ""


# LEONIDA_MESH_COMPUTE=1 (for run_gpu_tests.py, which builds the mesh fixtures): the mesh shaders were translated
# with rdna2spirv.py --mesh-compute, and vk_harness dispatches each as compute and draws its slots with the vertex
# shader tools/mesh_compute_vs.py wrote (<name>.mesh_vs.spv), so they run without VK_EXT_mesh_shader.
MESH_COMPUTE = os.environ.get("LEONIDA_MESH_COMPUTE") == "1"


def mesh_env(spirv, name, base=None):
    """vk_harness environment for mesh shader `name`: VKH_MESH=1, or under LEONIDA_MESH_COMPUTE=1 the
    --mesh-compute variables (vertex shader, slot bytes and primitive count from index.json's mesh_compute)."""
    env = dict(base if base is not None else os.environ)
    if MESH_COMPUTE:
        vs = Path(spirv) / f"{name}.mesh_vs.spv"
        try:
            with open(Path(spirv) / "index.json", encoding="utf-8") as f:
                lay = json.load(f).get(name, {}).get("mesh_compute")
        except (OSError, ValueError):
            lay = None
        if lay and vs.is_file():
            env.update(VKH_MESH_COMPUTE=str(vs), VKH_MESH_SLOT=str(lay["slot_bytes"]),
                       VKH_MESH_PRIMS=str(lay["max_primitives"]))
            return env
    env["VKH_MESH"] = "1"
    return env


def mesh_v5_test(tools, spirv, raw):
    shader = Path(spirv) / "ms_v5_workgroup.spv"
    if not shader.is_file():
        return Skip("SKIP (build the mesh fixture with scripts/run_gpu_tests.py)")
    fs = Path(spirv) / "fs_passthru.spv"
    if not fs.is_file():
        return "missing fs_passthru.spv"
    code, out = run_readback(raw, run, os.path.join(tools, "vk_harness" + EXE), shader, fs, raw,
                             "--mesh-group-readback", "2", "3", "4", env=mesh_env(spirv, "ms_v5_workgroup"))
    if code == UNSUPPORTED:
        return Skip("SKIP (mesh shading unavailable)")
    if code:
        return failure(code, out, 3)
    return check_mesh_v5(raw.read_bytes())


def partial_wave_test(tools, spirv, name, expected_words):
    """A group smaller than a wave: run with subgroup size 64 requested (no full subgroups); None if the device
    has no 64-wide subgroup. With LEONIDA_W64_EMULATE=1 it asks for 32 instead, the width the emulation is for, so
    a device that defaults to 64 (AMD) runs the emulated path at 32 too (#166)."""
    shader = os.path.join(spirv, name + ".spv")
    if not os.path.isfile(shader):
        return f"missing {name}.spv"
    old = os.environ.get("VKC_SUBGROUP_SIZE")
    os.environ["VKC_SUBGROUP_SIZE"] = "32" if W64_EMULATED else "64"
    try:
        code, out = run(os.path.join(tools, "vk_compute_test" + EXE), shader, "1", "dump")
    finally:
        if old is None:
            os.environ.pop("VKC_SUBGROUP_SIZE", None)
        else:
            os.environ["VKC_SUBGROUP_SIZE"] = old
    if code != 0:
        return failure(code, out, 2)
    if not W64_EMULATED and not any(int(x) >= 64 for x in re.findall(r"subgroup \d+\.\.(\d+)", out)):
        return None
    m = re.search(r"^out\[0\.\.7\] = ((?:[0-9]+ ){7}[0-9]+)$", out, re.MULTILINE)
    if m and [int(x) for x in m.group(1).split()] == expected_words:
        return ""
    return failure(code, out, 2)


def narrow_subgroup_skip(out):
    """A Skip when vk_compute_test's "device: ... (subgroup MIN..MAX)" line says MAX is below 32 lanes, else None."""
    m = re.search(r"subgroup \d+\.\.(\d+)", out)
    if m and int(m.group(1)) < 32:
        return Skip(f"SKIP (subgroup {m.group(1)} is narrower than 32 lanes)")
    return None


def unpinned_subgroup_skip(out):
    """A Skip when vk_compute_test says the VKC_SUBGROUP_SIZE it was given can't be pinned on this device, else None.
    Output without the line (an older vk_compute_test) is judged as before. "compute subgroup size: 32 (observed: ...)"
    (MoltenVK: not requirable, but its probe saw 32 lanes) is judged, and "observed wrong" comes with exit 1, a
    failure."""
    m = re.search(r"^compute subgroup size: (\d+) not available$", out, re.MULTILINE)
    if m:
        return Skip(f"SKIP (the device can't run this shader at {m.group(1)} lanes on request)")
    return None


def wave32_lane_skip(out):
    """The skip rule for WAVE32_LANE_DUMPS, shared with run_gpu_tests.py."""
    return narrow_subgroup_skip(out) or unpinned_subgroup_skip(out)


def bitmask_perm_narrow_skip(out, px):
    """A Skip when ps_bitmask_perm ran narrower than 32 lanes on a device whose subgroup range permits it (#618).
    Requires every lane 0..width-1 of a power-of-two width in [min, 32), so a broken v_mbcnt still fails."""
    m = re.search(r"subgroup (\d+)\.\.(\d+)", out)
    if not m or len(px) != 64 * 64:
        return None
    lo, hi = int(m.group(1)), int(m.group(2))
    if lo >= 32:
        return None
    lanes = {p[0] for p in px}
    width = max(lanes) + 1
    # "pippi" reads lane ^ 9. Where that lane runs (every lane at 16 wide, none at 8) the word is defined and has
    # to be right: a narrow run excuses only the sources that aren't there.
    if any(a != lane ^ 9 for lane, a, _, _ in px if lane ^ 9 < width):
        return None
    if lo <= width < 32 and width & (width - 1) == 0 and lanes == set(range(width)):
        return Skip(f"SKIP (fragment subgroup ran at {width} lanes on {lo}..{hi} device; needs 32)")
    return None


def vcmpx_forms_narrow_skip(out):
    """A Skip when ps_vcmpx_forms cannot be judged at 32-lane EXEC (TEST-205 / #618).

    The shader packs v_cmpx EXEC results into constant mrt0 words and does not write lane IDs, so unlike
    bitmask_perm_narrow_skip the width cannot be read back from the frame. On devices with minSubgroupSize < 32,
    requiredSubgroupSizeStages typically omits FRAGMENT (#1318), so vk_harness cannot pin the stage and the
    driver runs it narrower (Intel UHD 770: 4096/4096 mismatches, last word looking like a lane mask)."""
    m = re.search(r"subgroup (\d+)\.\.(\d+)", out)
    if not m:
        return Skip("SKIP (vk_harness reported no subgroup width)")
    lo, hi = int(m.group(1)), int(m.group(2))
    if hi < 32:
        return Skip(f"SKIP (subgroup {hi} is narrower than 32 lanes)")
    if lo < 32:
        return Skip(f"SKIP (fragment subgroup may run narrower than 32 on {lo}..{hi} device; needs 32)")
    return None


# Dump words a driver gets wrong with or without SignedZeroInfNanPreserve, by the first word of the device name on
# vk_compute_test's "device: ..." line. Only these words go unjudged there. Shared with run_gpu_tests.py.
# - Apple GPUs (MoltenVK) return +0 for cs_signed_zero_nan's -0 products, words 0, 2 and 3 (v_mul_f32, v_pk_mul_f16,
#   v_mul_f16; #1203, SHADER-182).
# - Mesa llvmpipe returns 0 in word 5, the NaN check of 0 * inf (an inline-constant 0), while its other NaN checks give
#   1 (#664, Mesa with LLVM 21.1.8). The CI runs the GPU tests on llvmpipe.
UNJUDGED_WORDS = {"Apple": {"cs_signed_zero_nan": (0, 2, 3)},
                  "llvmpipe": {"cs_signed_zero_nan": (5,)}}


def judged_words(name, out, words):
    """The expected words with None where nothing is judged: UNJUDGED_WORDS for the device that vk_compute_test's
    "device: ..." line names, else every word as given."""
    m = re.search(r"^device: (\S+)", out, re.MULTILINE)
    unjudged = UNJUDGED_WORDS.get(m.group(1), {}).get(name, ()) if m else ()
    return [None if i in unjudged else w for i, w in enumerate(words)]


def words_match(got, want):
    """The dump's words against want, which may be shorter; a None in want is not judged."""
    return len(got) >= len(want) and all(w is None or g == w for g, w in zip(got, want))


def compute_test(tools, spirv, name, groups, mode, *, require_wave64=False, expected_words=None, needs_32_lanes=False,
                 env=None):
    """Return an empty string for a pass, None for unsupported wave64, a Skip for a wave32 lane-mask test on a device
    narrower than 32 lanes (needs_32_lanes) or a module the device lacks a feature for, or a failure diagnostic. `env` adds variables for vk_compute_test
    (VKC_SGPRS)."""
    shader = os.path.join(spirv, name + ".spv")
    args = []
    if require_wave64:
        if not os.path.isfile(shader):
            return f"missing {name}.spv"
        args.append("--require-wave64")
    if needs_32_lanes:
        env = {**WAVE32_LANE_ENV, **(env or {})}
    # Run at the wave size the translator recorded, as run_gpu_tests.py does: a wave32 test left to the device's
    # default runs 64-wide on AMD (subgroups 32..64), and cs_barrier_loop's four waves of 32 became two.
    sg = None if "VKC_SUBGROUP_SIZE" in (env or {}) else subgroup_size(spirv, name)
    if sg:
        env = {**(env or {}), "VKC_SUBGROUP_SIZE": str(sg)}
    extra = {"env": dict(os.environ, **env)} if env else {}
    code, out = run(os.path.join(tools, "vk_compute_test" + EXE), shader, groups, mode, *args, **extra)
    if skip := compute_unsupported(code, out):
        return skip
    if require_wave64 and code == 77:  # vk_compute_test: the device cannot run a 64-wide compute subgroup
        return None
    if needs_32_lanes:
        if code == UNSUPPORTED:
            return Skip(f"SKIP ({out.strip() or 'unsupported device'})")
        if code == 0 and (skip := wave32_lane_skip(out)):
            return skip
    if sg and (skip := unpinned_subgroup_skip(out)):
        return skip
    if code != 0:
        return failure(code, out, 2)
    if require_wave64 and "compute subgroup size: 64" not in out.splitlines():
        return "64-wide subgroup was not confirmed (rebuild vk_compute_test)"
    if mode == "workgroup_yz":
        if "2x3x4 workgroup Y/Z: 0 mismatches" in out.splitlines():
            return ""
    elif expected_words is not None:
        m = re.search(r"^out\[0\.\.7\] = ((?:[0-9]+ ){7}[0-9]+)$", out, re.MULTILINE)
        if m and words_match([int(x) for x in m.group(1).split()], judged_words(name, out, expected_words)):
            return ""
    else:
        m = re.search(r"(\d+) mismatches", out)
        if m and m.group(1) == "0":
            return ""
    return failure(code, out, 2)


def shader_features(spirv, name):
    """The Vulkan feature names recorded for a shader in index.json, or an empty set."""
    try:
        with open(os.path.join(spirv, "index.json"), encoding="utf-8") as f:
            feats = json.load(f).get(name, {}).get("features") or ()
            return set(feats)
    except (OSError, ValueError, TypeError):
        return set()


def shader_fma_mode(spirv, name):
    """How the translator emitted fused multiply-adds for a shader ('glsl', 'khr' or 'exact'), or '' if unrecorded.

    Older indexes have no field: callers must fall back to the capability and LEONIDA_FMA as before.
    """
    try:
        with open(os.path.join(spirv, "index.json"), encoding="utf-8") as f:
            mode = json.load(f).get(name, {}).get("fma") or ""
            return mode if mode in ("glsl", "khr", "exact") else ""
    except (OSError, ValueError, TypeError):
        return ""


def judge_fma_fused(words, khr, exact=False):
    """Judge cs_fma_fused's 8 words: (ok, detail). Under --fma khr every word must be exact fused; under --fma glsl
    each of the 7 f32 words may be either fused or exact split (Vulkan allows splitting GLSL.std.450 Fma), while the
    f16 round-to-odd word out[7] must always be exact 0x00007351."""
    if not words or len(words) < 8:
        return False, "missing out[0..7] dump"
    if khr or exact:
        label = "OpFmaKHR" if khr else "--fma exact"
        for i, (got, want) in enumerate(zip(words, FMA_FUSED)):
            if got != want:
                return False, (f"word {i}: got {got:#010x}, want fused {want:#010x} "
                               "(single-rounded FMA: OpFmaKHR or --fma exact)")
        return True, "8/8 fused exact (single-rounded FMA: OpFmaKHR or --fma exact)"
    if words[7] != FMA_FUSED[7]:
        return False, f"word 7 (f16): got {words[7]:#010x}, want {FMA_FUSED[7]:#010x}"
    n_split = 0
    for i in range(7):
        got, fused, split = words[i], FMA_FUSED[i], FMA_SPLIT[i]
        if got == fused:
            continue
        if got == split:
            n_split += 1
            continue
        return False, f"word {i}: got {got:#010x}, want fused {fused:#010x} or split {split:#010x}"
    if n_split == 0:
        return True, "8/8 fused exact (GLSL Fma)"
    return True, f"{7 - n_split}/7 f32 fused, {n_split}/7 split (GLSL Fma; LEONIDA_FMA=khr for OpFmaKHR), f16 exact"


def fma_fused_test(tools, spirv):
    """Run cs_fma_fused and judge against FMA_FUSED (strict under --fma khr and --fma exact, fused-or-split f32 under
    --fma glsl)."""
    shader = os.path.join(spirv, "cs_fma_fused.spv")
    # --fma exact also guarantees single rounding (built from NoContraction ops), but records no KHR capability.
    # Without the index field a fresh shell cannot tell it from the default GLSL Fma (TEST-1147).
    mode = shader_fma_mode(spirv, "cs_fma_fused")
    if mode:
        # A recorded mode is authoritative: a stale LEONIDA_FMA from the build shell must not override it.
        strict = mode in ("khr", "exact")
    else:
        # An older index has no mode field: capability and environment are the only remaining signals.
        strict = ("shaderFmaFloat32" in shader_features(spirv, "cs_fma_fused")
                  or os.environ.get("LEONIDA_FMA") in ("khr", "exact"))
    code, out = run(os.path.join(tools, "vk_compute_test" + EXE), shader, "1", "dump",
                    env=dict(os.environ, **FMA_FUSED_ENV))
    if code == UNSUPPORTED:
        m = re.search(r"unsupported: .*", out)
        return Skip(f"SKIP ({m.group(0) if m else 'unsupported device'})")
    if code != 0:
        return failure(code, out, 2)
    m = re.search(r"^out\[0\.\.7\] = ((?:[0-9]+ ){7}[0-9]+)$", out, re.MULTILINE)
    if not m:
        return failure(code, out, 2)
    words = [int(x) for x in m.group(1).split()]
    ok, detail = judge_fma_fused(words, khr=strict)
    return "" if ok else detail


def cvt_f32_sat_test(tools, spirv):
    """cs_cvt_f32_sat once per input of CVT_F32_SAT_CASES: an empty string when every pair of words is exact."""
    wrong = []
    for bits, unsigned, signed in CVT_F32_SAT_CASES:
        problem = compute_test(tools, spirv, "cs_cvt_f32_sat", "1", "dump", expected_words=[unsigned, signed],
                               env={"VKC_SGPRS": f"{bits:#010x}"})
        if problem is None or isinstance(problem, Skip):
            return problem   # the device can't run it as recorded (wave size not pinnable): the same for every input
        if problem:
            wrong.append(f"{bits:#010x}: {problem}")
    return f"{len(wrong)} of {len(CVT_F32_SAT_CASES)} inputs wrong, first {wrong[0]}" if wrong else ""


def mesh_w64x_test(tools, spirv, name):
    """ms_w64x_ballot: ms_tri's picture, blue 0.5 only if the ballot/mbcnt compaction was exact across both halves."""
    shader, fs = os.path.join(spirv, name + ".spv"), os.path.join(spirv, "fs_passthru.spv")
    if not (os.path.isfile(shader) and os.path.isfile(fs)):
        return Skip("SKIP (mesh shader: build it with scripts/run_gpu_tests.py, build.py does not)")
    env = mesh_env(spirv, name)
    if not W64_EMULATED:
        env["VKH_SUBGROUP_SIZE"] = "64"
    with tempfile.TemporaryDirectory(prefix="leonida-mesh-") as tmp:
        raw = Path(tmp) / "pixels.raw"
        harness = os.path.join(tools, "vk_harness" + EXE)
        limit = gpu_test_timeout()
        try:
            r = subprocess.run([harness, shader, fs, raw], capture_output=True, text=True, env=env, timeout=limit)
        except subprocess.TimeoutExpired:
            return timed_out(harness, limit)
        except OSError as e:
            return start_failure(harness, e)
        if r.returncode == UNSUPPORTED:   # vk_harness: the device has no VK_EXT_mesh_shader
            m = re.search(r"unsupported: .*", r.stderr)
            return Skip(f"SKIP ({m.group(0) if m else 'unsupported device'})")
        if r.returncode or not raw.is_file():
            return failure(r.returncode, r.stdout + r.stderr, 2)
        data = raw.read_bytes()
    if len(data) != 64 * 64 * 16:
        return "vk_harness returned an unexpected pixel count"
    px = list(struct.iter_unpack("<4f", data))
    if not W64_EMULATED and 64 not in [int(x) for x in re.findall(r"subgroup \d+\.\.(\d+)", r.stderr)]:
        return None   # the device has no 64-wide subgroups
    wrong = sum(abs(p[0] - ((i % 64) + .5) / 64) > 1e-3 or abs(p[1] - ((i // 64) + .5) / 64) > 1e-3
                or abs(p[2] - .5) > 1e-3 or abs(p[3] - 1) > 1e-3 for i, p in enumerate(px))
    return "" if not wrong else f"{wrong} of {len(px)} pixels differ (blue {px[0][2]:.3g} at pixel 0, want 0.5)"


def subgroup_size(spirv, name):
    """The wave size the translator recorded for a shader (index.json), or None."""
    try:
        with open(os.path.join(spirv, "index.json"), encoding="utf-8") as f:
            return json.load(f).get(name, {}).get("subgroup_size")
    except (OSError, ValueError):
        return None


def bindless_test(tools, spirv, pixel="ps_tex"):
    """ps_tex: a texture found through the descriptor hash table and the bindless set (vk_bindless_test).
    ps_imgatomic: image atomics in a pixel shader through the set's r32ui storage arrays, checked per texel."""
    shaders = [os.path.join(spirv, name + ".spv") for name in ("vs_ids", pixel)]
    for path in shaders:
        if not os.path.isfile(path):
            return f"missing {os.path.basename(path)}"
    code, out = run(os.path.join(tools, "vk_bindless_test" + EXE), *shaders)
    if code == UNSUPPORTED:
        m = re.search(r"unsupported: .*", out)
        return Skip(f"SKIP ({m.group(0) if m else 'unsupported device'})")
    if pixel == "ps_imgatomic":
        m = re.search(r"ps_imgatomic: (\d+) of \d+ texels wrong.*; (\d+) of \d+ render-target pixels wrong", out)
        return "" if code == 0 and m and m.group(1) == "0" and m.group(2) == "0" else failure(code, out, 3)
    m = re.search(r"(\d+) draws read back, (\d+) bad pixels", out)
    return "" if code == 0 and m and m.group(2) == "0" else failure(code, out, 3)


# Samples with a16 coordinates after whole-VGPR arguments (gradients, a bias, a clamp). On the 8x8 gradient every
# pixel must be the texel its own (u, v) reads. ps_texture_grad_cl_a16 draws on the two-mip texture instead, where its
# clamp picks mip 0 left of x = 32 and mip 1 from there on.
GRADIENT_TESTS = ("ps_texture_grad_a16", "ps_texture_bias_a16", "ps_texture_grad_cl_a16", "ps_texture_cl", "ps_texture_b_cl")
# ps_texture_cl / ps_texture_b_cl (image_sample_cl / _b_cl) draw on the two-mip texture too: the left half reads mip 0 (red,
# the top-left quadrant), the right half mip 1 (cyan) where the clamp is 1.
CLAMP_TESTS = ("ps_texture_cl", "ps_texture_b_cl")


def table_fetch_test(tools, spirv):
    tool = os.path.abspath(os.path.join(tools, "vk_texture_test" + EXE))
    with tempfile.TemporaryDirectory(prefix="table-fetch-") as tmp:
        for mode in ("fetch-sint", "fetch-uint", "fetch-miss"):
            raw = os.path.join(tmp, mode + ".raw")
            code, output = run_readback(raw, run, tool, mode, raw, cwd=spirv)
            if code == UNSUPPORTED:
                return Skip(output.strip())
            if code or f"{mode}: 4096 exact pixels" not in output:
                return failure(code, output, 3)
    return ""


def texture_gradient_a16_test(tools: str, spirv: str, name: str = "ps_texture_grad_a16") -> str:
    """Check every pixel of one of the GRADIENT_TESTS samples."""
    vs = os.path.join(spirv, "vs_ids.spv")
    fs = os.path.join(spirv, name + ".spv")
    two_mip = name == "ps_texture_grad_cl_a16" or name in CLAMP_TESTS
    with tempfile.TemporaryDirectory(prefix="leonida-texture-") as tmp:
        raw = Path(tmp) / "pixels.raw"
        code, out = run(os.path.join(tools, "vk_texture_test" + EXE), vs, fs, raw,
                        "0", "92", "c00000", "14500000", "0", *(["--two-mip"] if two_mip else []))
        if code == UNSUPPORTED:   # vk_texture_test: e.g. no shaderResourceMinLod for the d_cl clamp
            m = re.search(r"unsupported: .*", out)
            return Skip(f"SKIP ({m.group(0) if m else 'unsupported device'})")
        if code or not raw.is_file():
            return failure(code, out, 3)
        data = raw.read_bytes()
    if len(data) != 64 * 64 * 16:
        return "vk_texture_test returned an unexpected pixel count"
    wrong = 0
    first = ""
    for index, pixel in enumerate(struct.iter_unpack("<4f", data)):
        x, y = index % 64, index // 64
        if name in CLAMP_TESTS:   # mip 1 (cyan) where the clamp is 1, else mip 0's top-left quadrant (red)
            want = (0.0, 1.0, 1.0, 1.0) if x >= 32 else (1.0, 0.0, 0.0, 1.0)
        elif two_mip:   # mip 1 (cyan) where the clamp is 1, else mip 0's red and blue quadrants
            want = (0.0, 1.0, 1.0, 1.0) if x >= 32 else (1.0, 0.0, 0.0, 1.0) if y < 32 else (0.0, 0.0, 1.0, 1.0)
        else:
            want = (32 * (x // 8) / 255, 32 * (y // 8) / 255, 128 / 255, 1.0)
        if any(not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-6)
               for actual, expected in zip(pixel, want)):
            if not wrong:
                first = f"at ({x},{y}): got {pixel}, expected {want}"
            wrong += 1
    return f"{wrong} of 4096 pixels differ; {first}" if wrong else ""


def ssharp_words(mip_filter, min_lod=0.0, max_lod=15.0, bias=0.0):
    """An S# with point min/mag filters, clamp-to-last-texel addressing and the given mip fields. Positions as
    Mesa's gfx10-rsrc.json has them: WORD0 CLAMP_X/Y/Z [8:0], WORD1 MIN_LOD [11:0] and MAX_LOD [23:12] (4.8 fixed),
    WORD2 LOD_BIAS [13:0] (signed 5.8 fixed) and MIP_FILTER [27:26] (0 none, 1 point, 2 linear)."""
    clamp = 2 | 2 << 3 | 2 << 6
    return (clamp, round(min_lod * 256) | round(max_lod * 256) << 12,
            (round(bias * 256) & 0x3fff) | mip_filter << 26, 0)


# (name, S#, (BASE_LEVEL, LAST_LEVEL) of the T#). Expected levels, explicit LOD 0 / implicit LOD 1:
SAMPLER_CASES = [("point", ssharp_words(1), (0, 1)),                       # 0 / 1
                 ("point max_lod 0", ssharp_words(1, max_lod=0), (0, 1)),  # 0 / 0
                 ("point min_lod 1", ssharp_words(1, min_lod=1), (0, 1)),  # 1 / 1: MIN_LOD clamps an explicit LOD too
                 ("point bias +1", ssharp_words(1, bias=1), (0, 1)),       # 1 / 1: the S# bias applies to an explicit LOD
                 ("point bias -1", ssharp_words(1, bias=-1), (0, 1)),      # 0 / 0
                 ("none", ssharp_words(0), (0, 1)),                        # 0 / 0: MIP_FILTER_NONE reads the base level
                 ("none min_lod 1", ssharp_words(0, min_lod=1), (0, 1)),   # 0 / 0: and ignores the LOD clamp
                 ("point base 1", ssharp_words(1), (1, 1))]                # 1 / 1: the T#'s BASE_LEVEL starts the view
SAMPLER_SIZE = 64          # the target; the texture is twice as wide, so implicit sampling sees LOD 1
GREY = 128 / 255


def sampler_level(words, view, lod):
    """The texture level one sample reads, from the S# fields and the T# view alone (not from gfx10_descriptors.h).
    MIP_FILTER_NONE reads the view's base level whatever the LOD. Otherwise the LOD plus LOD_BIAS is clamped to
    [MIN_LOD, MAX_LOD], a point mip filter takes the nearest level (ceil(lod + 0.5) - 1, as Vulkan's
    VK_SAMPLER_MIPMAP_MODE_NEAREST does) and the result stays inside the view."""
    base, last = view
    if (words[2] >> 26) & 3 == 0:
        return base
    bias = ((words[2] & 0x3fff) ^ 0x2000) - 0x2000
    lo, hi = (words[1] & 0xfff) / 256, (words[1] >> 12 & 0xfff) / 256
    level = math.ceil(min(max(lod + bias / 256, lo), hi) + 0.5) - 1
    return base + min(max(level, 0), last - base)


def sampler_texel(level, x, y):
    """(r, g) of the texel under pixel (x, y): level 0 is red, green, blue and yellow quadrants, level 1 grey."""
    if level:
        return (GREY, GREY)
    half = SAMPLER_SIZE // 2
    return ((1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (1.0, 1.0))[2 * (y >= half) + (x >= half)]


def sampler_mismatch(data, words, view):
    """The first pixel whose (lz.r, lz.g, implicit.r, implicit.g) is not what the S# and T# select, or ""."""
    if len(data) != SAMPLER_SIZE ** 2 * 16:
        return f"expected {SAMPLER_SIZE ** 2} RGBA32F pixels, got {len(data)} bytes"
    explicit, implicit = sampler_level(words, view, 0), sampler_level(words, view, 1)
    for i, got in enumerate(struct.iter_unpack("<4f", data)):
        y, x = divmod(i, SAMPLER_SIZE)
        want = sampler_texel(explicit, x, y) + sampler_texel(implicit, x, y)
        if any(not math.isclose(g, w, rel_tol=0, abs_tol=1e-6) for g, w in zip(got, want)):
            return f"({x},{y}): {tuple(round(g, 4) for g in got)} != {tuple(round(w, 4) for w in want)}"
    return ""


def sampler_test(tools, spirv):
    """ps_ssharp: every SAMPLER_CASES S#/T# pair drawn with vk_sampler_test and judged per pixel by sampler_level.
    A grey level-1 texel differs from every level-0 texel in both channels, so a wrong level fails at each pixel."""
    shaders = [os.path.join(spirv, name + ".spv") for name in ("vs_ids", "ps_ssharp")]
    for path in shaders:
        if not os.path.isfile(path):
            return f"missing {os.path.basename(path)}"
    with tempfile.TemporaryDirectory(prefix="leonida-sampler-") as tmp:
        for name, words, view in SAMPLER_CASES:
            raw = Path(tmp) / "pixels.raw"
            raw.unlink(missing_ok=True)
            code, out = run(os.path.join(tools, "vk_sampler_test" + EXE), *shaders, raw,
                            *(f"{w:x}" for w in words), *(str(v) for v in view))
            if code == UNSUPPORTED:
                m = re.search(r"unsupported: .*", out)
                return Skip(f"SKIP ({m.group(0) if m else 'unsupported device'})")
            if code or not raw.is_file():
                return f"{name}: {failure(code, out, 3)}"
            problem = sampler_mismatch(raw.read_bytes(), words, view)
            if problem:
                return f"{name}: {problem}"
    return ""


def bary_persp_problem(px):
    """ps_persp_bary drawn with vs_persp: "" when every pixel matches the perspective-correct CPU reference, else what
    differs. At a pixel centre with screen-space weights l0, l1, l2, the weights are p_k = (l_k / w_k) / sum(l_m / w_m);
    mrt0 = (attr0.xy interpolated with p, p1, p2), and pixels outside the triangle keep the clear value (-1). Coverage
    comes from edge functions at doubled coordinates, which are exact integers: no centre lies on an edge, so every
    pixel is judged whatever the rasterization rule. The affine weights (l1, l2) must differ from (p1, p2) by more than
    100 times the tolerance on every covered pixel, so a NoPerspective regression can't pass by accident."""
    tol = 1e-5
    (x0, y0), (x1, y1), (x2, y2) = (tuple(int(c) for c in v) for v in PERSP_SCREEN)
    area2 = 2 * ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))   # twice the area, in doubled coordinates
    covered, wrong, first, gap = 0, 0, "", math.inf
    for i, pixel in enumerate(px):
        x, y = 2 * (i % 64) + 1, 2 * (i // 64) + 1                  # the pixel centre, doubled
        e1 = (x - 2 * x0) * (y2 - y0) - (x2 - x0) * (y - 2 * y0)
        e2 = (x1 - x0) * (y - 2 * y0) - (x - 2 * x0) * (y1 - y0)
        e = (area2 - e1 - e2, e1, e2)
        if 0 in e:
            return f"test geometry: pixel centre ({i % 64},{i // 64}) lies on an edge"
        if min(e) < 0:
            want = (-1.0, -1.0, -1.0, -1.0)
        else:
            l = [ek / area2 for ek in e]
            p = [lk / wk for lk, wk in zip(l, PERSP_W)]
            p = [pk / sum(p) for pk in p]
            want = (sum(pk * a[0] for pk, a in zip(p, PERSP_ATTR)), sum(pk * a[1] for pk, a in zip(p, PERSP_ATTR)),
                    p[1], p[2])
            gap = min(gap, max(abs(p[1] - l[1]), abs(p[2] - l[2])))
            covered += 1
        if not all(math.isfinite(v) and abs(v - w) <= tol for v, w in zip(pixel, want)):
            if not wrong:
                got, exp = (tuple(round(v, 6) for v in t) for t in (pixel, want))
                first = f"at ({i % 64},{i // 64}): got {got}, want {exp}"
            wrong += 1
    if gap <= 100 * tol:
        return f"test geometry: perspective and affine weights only {gap:.3g} apart"
    return f"{wrong} pixels off the perspective-correct reference ({covered} covered); {first}" if wrong else ""


def pixels(tools, spirv, name, fmt, vertex="vs_ids"):
    # Draw at the shader's own wave size, as run_gpu_tests.py does (#146): a wave32 pixel shader run as wave64
    # (RADV's default on RDNA1) reads lanes it was never written for.
    sg = subgroup_size(spirv, name)
    env = dict(os.environ, VKH_SUBGROUP_SIZE=str(sg)) if sg else None
    if name in PIXEL_ENV:
        env = dict(env or os.environ, **PIXEL_ENV[name])
    gs = os.path.join(spirv, name + ".bary_gs.spv")
    if BARY_FALLBACK and os.path.isfile(gs):
        env = dict(env or os.environ, VKH_GS=gs, VKH_GS_HELPER="1")
    with tempfile.TemporaryDirectory(prefix="leonida-pixels-") as tmp:
        raw = Path(tmp) / "pixels.raw"
        code, out = run(os.path.join(tools, "vk_harness" + EXE), os.path.join(spirv, vertex + ".spv"),
                        os.path.join(spirv, name + ".spv"), raw, env=env)
        if code == UNSUPPORTED:   # vk_harness: the shader needs a device feature this GPU lacks
            return Skip(""), out
        if code or not raw.is_file():
            return None, out
        data = raw.read_bytes()
    if len(data) != 64 * 64 * 16:  # default harness viewport, four 32-bit components per pixel
        return None, out
    return list(struct.iter_unpack(fmt, data)), out


def texture_constant_test(tools, spirv, name):
    """Compare every pixel returned through vk_texture_test with the fixture reference."""
    expected = TEXTURE_CONSTANTS[name]
    shader = os.path.join(spirv, name + ".spv")
    if not os.path.isfile(shader):
        return f"missing {name}.spv"
    with tempfile.TemporaryDirectory(prefix="leonida-gather-") as tmp:
        raw = Path(tmp) / "gather.raw"
        code, out = run(os.path.join(tools, "vk_texture_test" + EXE), os.path.join(spirv, "vs_ids.spv"), shader, raw,
                        "0", "92", "c00000", "14500000", "0")
        data = raw.read_bytes() if code == 0 and raw.is_file() else b""
    if len(data) != 64 * 64 * 16:
        return failure(code, out, 3)
    wrong = sum(p != expected for p in struct.iter_unpack("<4f", data))
    return "" if not wrong else f"{wrong} of 4096 pixels differ from {expected}"


def texoff_words(px, py, name="ps_texoff"):
    """ps_texoff's mrt0 words at pixel (px, py), from the rule in docs/design.md (derivation in ps_texoff.asm).
    ps_texoff_forms (image_sample_o, image_sample_b_o, image_gather4_l_o) puts the per-pixel sample in .x and .y and
    the same two gathers in .z and .w (derivation in ps_texoff_forms.asm)."""
    bx, by = px >> 3, py >> 3                          # per-pixel offset (bx - 4, by - 4) from texel (4, 4)
    x0, y0 = max(bx - 1, 0), max(by - 1, 0)            # gather footprint 3..4 moved by the offset, clamped at 0
    sample = bx | by << 4
    return (sample, sample if name == "ps_texoff_forms" else 0x0761, x0 | bx << 4 | bx << 8 | x0 << 12,
            by | by << 4 | y0 << 8 | y0 << 12)


# LEONIDA_TEXOFF_FALLBACK=1 (set for build.py too): image_sample*_o was translated with --texel-offset-fallback, so
# ps_texoff runs without VK_KHR_maintenance8 and every offset is defined, whatever the device's texel offset limits.
TEXOFF_FALLBACK = os.environ.get("LEONIDA_TEXOFF_FALLBACK") == "1"


def texoff_problem(data, output, all_offsets=None, name="ps_texoff"):
    """Judge ps_texoff's 64x64 RGBA32 dump: (failure text or "", what was covered). The +16 / -32 sample (bits 8-15
    of .y) is only defined, and so only judged, where vk_texture_test printed texel offset limits that cover it."""
    m = re.search(r"texel offset limits: (-?\d+)\.\.(-?\d+)", output)
    wide = bool(m) and int(m.group(1)) <= -32 and int(m.group(2)) >= 16
    if TEXOFF_FALLBACK if all_offsets is None else all_offsets:
        wide = True                    # the offset is a coordinate shift: no device limit applies
    if len(data) != 64 * 64 * 16:
        return "vk_texture_test returned an unexpected pixel count", ""
    if name == "ps_texoff_forms":
        wide = True                    # every offset there is in -4..3, inside Vulkan's guaranteed -8..7
    keep = (0xFFFFFFFF, 0xFFFFFFFF if wide else 0xFFFF00FF, 0xFFFFFFFF, 0xFFFFFFFF)
    px = list(struct.iter_unpack("<4I", data))
    bad = [(i, got) for i, got in enumerate(px)
           if any((g ^ w) & k for g, w, k in zip(got, texoff_words(i % 64, i // 64, name), keep))]
    limits = f"{m.group(1)}..{m.group(2)}" if m else "unknown"
    covered = ("offsets +16/-32 judged" + (" (texel offset fallback)" if wide and not (
        bool(m) and int(m.group(1)) <= -32 and int(m.group(2)) >= 16) else "")) if wide else f"offsets +16/-32 not judged (device texel offset limits {limits})"
    if name == "ps_texoff_forms":
        covered = "offsets -4..3 on image_sample_o, image_sample_b_o and image_gather4_l_o"
    if not bad:
        return "", covered
    i, got = bad[0]
    return (f"{len(bad)} of {len(px)} pixels differ; first at ({i % 64},{i // 64}): got {[hex(x) for x in got]}, "
            f"want {[hex(x) for x in texoff_words(i % 64, i // 64, name)]} ({covered})"), covered


def texoff_env(base=None):
    """vk_texture_test environment for ps_texoff: it needs VK_KHR_maintenance8 unless translated with the fallback."""
    env = dict(base if base is not None else os.environ)
    if not TEXOFF_FALLBACK:
        env["VKT_NEED_MAINTENANCE8"] = "1"
    return env


def texoff_test(tools, spirv, name):
    """ps_texoff on vk_texture_test (it binds the texture, sampler and descriptor table); vs_ids supplies attr0."""
    with tempfile.TemporaryDirectory(prefix="leonida-pixels-") as tmp:
        raw = Path(tmp) / "pixels.raw"
        code, out = run(os.path.join(tools, "vk_texture_test" + EXE), os.path.join(spirv, "vs_ids.spv"),
                        os.path.join(spirv, name + ".spv"), raw, *TEXOFF_ARGS,
                        env=texoff_env())
        if code == UNSUPPORTED:   # vk_texture_test: no VK_KHR_maintenance8 (a non-constant Offset outside gather)
            m = re.search(r"unsupported: .*", out)
            return Skip(f"SKIP ({m.group(0) if m else 'unsupported device'})")
        if code or not raw.is_file():
            return failure(code, out, 2)
        return texoff_problem(raw.read_bytes(), out, name=name)[0]


def store_words(name):
    """The 64 words STORE_WORDS[name] expects in each of STORE_IMAGES: its changes over the tool's fill."""
    return tuple([changed.get(i, fill(i)) for i in range(64)] for changed, fill in zip(STORE_WORDS[name], STORE_FILL))


def store_words_problem(output, name):
    """Judge what vk_store_test printed in "words" mode: "" when every image holds store_words(name)."""
    for image, want in zip(STORE_IMAGES, store_words(name)):
        m = re.search(rf"^{image} words:((?: 0x[0-9a-f]{{8}}){{64}})$", output, re.MULTILINE)
        if not m:
            return f"vk_store_test printed no {image} words"
        got = [int(word, 16) for word in m.group(1).split()]
        bad = [i for i in range(64) if got[i] != want[i]]
        if bad:
            i = bad[0]
            return (f"{len(bad)} of 64 {image} words differ; first word {i}: got {got[i]:#010x}, "
                    f"expected {want[i]:#010x}")
    return ""


def store_words_test(tools, spirv, name):
    """A STORE_WORDS fixture through vk_store_test's "words" mode (it binds the images and prints them)."""
    shader = os.path.join(spirv, name + ".spv")
    if not os.path.isfile(shader):
        return f"missing {name}.spv"
    code, out = run(os.path.join(tools, "vk_store_test" + EXE), shader, "words")
    if code == UNSUPPORTED:   # vk_store_test: the device lacks a feature its images need
        m = re.search(r"unsupported: .*", out)
        return Skip(f"SKIP ({m.group(0) if m else 'unsupported device'})")
    return store_words_problem(out, name) if code == 0 else failure(code, out, 3)


def get_lod_problem(data, want):
    """Judge one image_get_lod frame: "" when all 4096 RGBA32F pixels are within GET_LOD_TOLERANCE of `want`."""
    if len(data) != 64 * 64 * 16:
        return f"expected 4096 RGBA32F pixels, got {len(data)} bytes"
    bad, first = 0, ""
    for i, pixel in enumerate(struct.iter_unpack("<4f", data)):
        if not all(abs(got - expected) <= GET_LOD_TOLERANCE for got, expected in zip(pixel, want)):   # a NaN too
            if not bad:
                first = f"first at ({i % 64},{i // 64}): got {pixel}"
            bad += 1
    return f"{bad} of 4096 pixels differ from {want}; {first}" if bad else ""


def get_lod_test(tools, spirv, name):
    """A GET_LOD_TESTS fixture on vk_texture_test (it binds the textures and the sampler), once per draw."""
    tool = os.path.join(tools, "vk_texture_test" + EXE)
    shaders = [os.path.join(spirv, shader + ".spv") for shader in ("vs_ids", name)]
    for path in (tool, *shaders):
        if not os.path.isfile(path):
            return f"missing {os.path.basename(path)}"
    with tempfile.TemporaryDirectory(prefix="leonida-lod-") as tmp:
        for limits, two_mip, want in GET_LOD_TESTS[name]:
            raw = Path(tmp) / f"lod_{limits}.raw"
            code, out = run(tool, *shaders, raw, "0", "92", limits, "14500000", "0",
                            *(["--two-mip"] if two_mip else []))
            if code == UNSUPPORTED:   # vk_texture_test: e.g. fewer push constant bytes than the texture tests need
                m = re.search(r"unsupported: .*", out)
                return Skip(f"SKIP ({m.group(0) if m else 'unsupported device'})")
            if code or not raw.is_file():
                return f"S# word 1 {limits}: {failure(code, out, 3)}"
            problem = get_lod_problem(raw.read_bytes(), want)
            if problem:
                return f"S# word 1 {limits}: {problem}"
    return ""


def texture_mip_mismatch(data: bytes, mip: int, *, transpose_uv: bool = False) -> str:
    width = 64
    if len(data) != width**2 * 16:
        return f"expected {width**2} RGBA32F pixels, got {len(data)} bytes"
    one = 0x3f800000
    quadrants = ((one, 0, 0, one), (0, one, 0, one),
                 (0, 0, one, one), (one, one, 0, 0))
    for i, actual in enumerate(struct.iter_unpack("<4I", data)):
        y, x = divmod(i, width)
        # The normalized UV at each pixel center lies within the same quadrant of mip 0.
        u, v = (y, x) if transpose_uv else (x, y)
        expected = (0, one, one, one) if mip == 1 else quadrants[2 * (v >= width // 2) + (u >= width // 2)]
        if actual != expected:
            return f"({x},{y}): RGBA32F bits {actual} != {expected}"
    return ""


def texture_mip_test(tools: str, spirv: str, test: str = TEXTURE_MIP_TEST) -> str:
    tool = Path(tools) / ("vk_texture_test" + EXE)
    if not tool.is_file():
        return f"missing {tool}"
    shaders = Path(spirv)
    vertex = shaders / "vs_ids.spv"
    with tempfile.TemporaryDirectory(prefix="leonida-mip-") as tmp:
        for mip, (name, limits) in enumerate(MIP_TEST_PASSES[test]):
            fragment = shaders / (name + ".spv")
            for shader in (vertex, fragment):
                if not shader.is_file():
                    return f"missing {shader}"
            raw = Path(tmp) / f"mip{mip}.raw"
            mode = MIP_TEST_MODES.get(test, "--two-mip")
            code, output = run(tool, vertex, fragment, raw, "0", "92", limits, "14500000", "0", mode)
            if code == UNSUPPORTED:
                return Skip(f"SKIP (texture mip test: {output.strip() or 'unsupported device'})")
            if code or not raw.is_file():
                return f"{name}: {failure(code, output, 4)}"
            data = raw.read_bytes()
            mismatch = texture_mip_mismatch(data, mip)
            if mismatch:
                return f"{name}: {mismatch}"
            if not texture_mip_mismatch(data, 1 - mip).startswith("(0,0):"):
                return f"{name}: wrong mip control missed the expected first pixel"
            if mip == 0 and not texture_mip_mismatch(data, mip, transpose_uv=True).startswith("(32,0):"):
                return f"{name}: transposed UV control missed the expected first pixel"
    return ""


def ij_computed_problem(px):
    """ps_ij_computed.asm: (snapped 0.5, zeroed pair 0.0, v/2 from (i, j/2), 0.5 control) per pixel; "" when right.
    Shared with scripts/run_gpu_tests.py."""
    w = 64
    if len(px) != w * w:
        return f"{len(px)} pixels, expected {w * w}"
    wrong, err = 0, 0.0
    for i, (a, b, c, d) in enumerate(px):
        v_over_two = struct.unpack("<f", struct.pack("<I", c))[0]
        if not math.isfinite(v_over_two):
            return f"non-finite computed I/J result at ({i % w},{i // w}): {v_over_two}"
        wrong += a != 0x3F000000 or b != 0 or d != 0x3F000000
        err = max(err, abs(v_over_two - (i // w + 0.5) / (2 * w)))
    return "" if not wrong and err < 1e-6 else f"{wrong} pixels without the computed-pair words, v/2 off by {err:.3g}"


def interp_f16_high_problem(px):
    """ps_interp_f16_high (drawn with vs_f16_pack or ms_f16_pack): low halves of mrt0.xyz = f16 of u, v, u at the
    pixel centre, read from both halves of packed channels, and mrt0.w = the f32 v. "" when every pixel is close."""
    if len(px) != 64 * 64:
        return f"{len(px)} pixels, expected 4096"
    w = 64
    half = lambda b: struct.unpack("<e", struct.pack("<H", b & 0xFFFF))[0]      # noqa: E731
    f32 = lambda b: struct.unpack("<f", struct.pack("<I", b))[0]                 # noqa: E731
    err = 0.0
    for i, (a, b, c, d) in enumerate(px):
        values = half(a), half(b), half(c), f32(d)
        x, y = i % w, i // w
        if not all(math.isfinite(v) for v in values):
            return f"non-finite interpolation result at ({x},{y}): {values}"
        u, v = (x + 0.5) / w, (y + 0.5) / w
        err = max(err, abs(values[0] - u), abs(values[1] - v), abs(values[2] - u), abs(values[3] - v))
    return "" if err < 2e-3 else f"f16 halves off by {err:.3g}"


def base_ids_test(tools, spirv):
    """vs_base_ids with ps_flat, drawn with firstVertex 16 and firstInstance 7: the VS input VGPRs are zero-based on
    the hardware, so every pixel must still see vertex 0 of instance 0 (0x101, 0x102), not 0x7111."""
    for n in ("vs_base_ids", "ps_flat"):
        if not os.path.isfile(os.path.join(spirv, n + ".spv")):
            return f"missing {n}.spv"
    env = dict(os.environ, VKH_DRAW="16,7")
    with tempfile.TemporaryDirectory(prefix="leonida-pixels-") as tmp:
        raw = Path(tmp) / "pixels.raw"
        code, out = run(os.path.join(tools, "vk_harness" + EXE), os.path.join(spirv, "vs_base_ids.spv"),
                        os.path.join(spirv, "ps_flat.spv"), raw, env=env)
        if code == UNSUPPORTED:
            m = re.search(r"unsupported: .*", out)
            return Skip(f"SKIP ({m.group(0) if m else 'unsupported device'})")
        if code or not raw.is_file():
            return out if out.startswith("could not start") else "vk_harness failed or returned no pixels"
        px = list(struct.iter_unpack("<2I2f", raw.read_bytes()))
    wrong = [p for p in px if (p[0], p[1]) != (0x101, 0x102)]
    return "" if px and not wrong else f"{len(wrong)} of {len(px)} pixels differ, e.g. {hex(wrong[0][0]) if wrong else '-'}"


def wide_pixel_skip(name, out):
    """A Skip when `name` needs a 64-wide pixel stage (WIDE_PIXEL_TESTS) and vk_harness's output does not say that size
    was required of the fragment stage, else None."""
    want = WIDE_PIXEL_TESTS.get(name)
    if not want or re.search(rf"^fragment subgroup size: {want}$", out, re.MULTILINE):
        return None
    m = re.search(r"subgroup (\d+)\.\.(\d+)", out)
    seen = f": subgroup {m.group(1)}..{m.group(2)}" if m else ""
    return Skip(f"SKIP (this device cannot require a {want}-wide fragment subgroup{seen})")


def f32(x):
    """x rounded to the nearest float32. One double add or multiply of two float32 values, then this, is the exact
    IEEE float32 result, so a model built from it matches the GPU bit for bit."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def mandelbrot_counts(w):
    """ps_mandelbrot.asm's iteration counts over a w x w frame, in the shader's order of float32 operations."""
    counts = []
    for y in range(w):
        cy = f32(f32(3.0 * f32((y + 0.5) / w)) - 1.5)
        for x in range(w):
            cx = f32(f32(3.0 * f32((x + 0.5) / w)) - 2.25)
            zx = zy = 0.0
            n = 0
            while n < 64:
                zx2, zy2 = f32(zx * zx), f32(zy * zy)
                if not 4.0 >= f32(zx2 + zy2):
                    break
                xy = f32(zx * zy)
                zx, zy = f32(f32(zx2 - zy2) + cx), f32(f32(xy + xy) + cy)
                n += 1
            counts.append(n)
    return counts


def mandelbrot_problem(px):
    """ps_mandelbrot: channel 3 is the lane's iteration count. A lane that leaves EXEC too early or late, or a loop
    that ends on the wrong branch, changes counts; the colour channels are only for looking at the frame."""
    w = int(len(px) ** 0.5)
    if w * w != len(px):
        return f"{len(px)} pixels is not a square frame"
    want = mandelbrot_counts(w)
    wrong = [(i, p[3], e) for i, (p, e) in enumerate(zip(px, want)) if p[3] != e]
    if not wrong:
        return ""
    i, got, e = wrong[0]
    return f"{len(wrong)} of {len(px)} iteration counts differ, e.g. pixel ({i % w}, {i // w}) {got} instead of {e}"


def barycentric_problem(px: list[tuple[float, float, float, float]], abs_tol: float) -> str:
    """Check ps_bary over the 64-by-64 frame covered by the vs_ids triangle."""
    if len(px) != 64 * 64:
        return f"expected 4096 pixels, got {len(px)}"
    for index, pixel in enumerate(px):
        a, b, _, _ = pixel
        if not all(math.isfinite(v) for v in pixel):
            return "barycentrics contain non-finite values"
        y, x = divmod(index, 64)
        # vs_ids exports (0,0), (2,0), (0,2) at the vertices. Check that gradient
        # independently so a failure that zeros both interpolation paths is visible.
        u, v = (x + 0.5) / 64, (y + 0.5) / 64
        expected = (u, v, a / 2, b / 2)
        if any(abs(got - want) > abs_tol for got, want in zip(pixel, expected)):
            return f"at ({x},{y}): got {pixel}, expected {expected}"
    return ""


def nan_payload_problem(px, out):
    """Check the exact words, except for the measured KosmicKrisp RGBA32F canonicalization (#3125)."""
    if len(px) != 64 * 64:
        return f"expected 4096 pixels, got {len(px)}"
    want = (0x7fc12345, 0xffc54321, 0x3f800000, 0x3f000000)
    canonicalizes = "driver: KosmicKrisp" in out.splitlines()
    for i, p in enumerate(px):
        if any(actual != expected and not (canonicalizes and c < 2 and actual == 0x7fc00000)
               for c, (actual, expected) in enumerate(zip(p, want))):
            return f"pixel ({i % 64},{i // 64}) has unexpected words: {[hex(x) for x in p]}"
    return ""


def pixel_test(tools, spirv, name):
    path = os.path.join(spirv, name + ".spv")
    if name == "ps_ij_computed" and (BARY_FALLBACK or F16_HALVES):
        # Before the .spv check: without PerVertexKHR inputs the translator refuses a computed I/J pair (no .spv).
        return Skip("SKIP (computed I/J pairs need PerVertexKHR inputs: no --bary-fallback / --f16-interp-fallback)")
    if not os.path.isfile(path):
        return f"missing {name}.spv"
    if name in ("ps_texoff", "ps_texoff_implicit", "ps_texoff_forms"):
        return texoff_test(tools, spirv, name)
    if name == "ps_interp_f16_high" and BARY_FALLBACK and not F16_HALVES:   # with F16_HALVES it reads decoded halves
        return Skip("SKIP (per-vertex interpolants need VK_KHR_fragment_shader_barycentric)")
    fmt = "<4f" if name in ("ps_bary", "ps_persp_bary") else "<2I2f" if name in ("ps_flat", "ps_uniform_branch", "ps_scc_mask32") else "<4I"
    px, out = pixels(tools, spirv, name, fmt, PIXEL_VERTEX.get(name, "vs_ids"))
    if isinstance(px, Skip):   # exit 77: the device cannot run this shader (e.g. no barycentrics)
        m = re.search(r"unsupported: .*", out)
        return Skip(f"SKIP ({m.group(0) if m else 'unsupported device'})")
    if name == "ps_bitmask_perm":
        # Decide from the reported width before the pixel count: a narrow device's frame can be short, and with no
        # width line the lanes 0-31 this shader writes are not known to exist.
        m = re.search(r"subgroup \d+\.\.(\d+)", out)
        if not m:
            return Skip("SKIP (vk_harness reported no subgroup width)")
        if int(m.group(1)) < 32:
            return Skip(f"SKIP (subgroup {m.group(1)} is narrower than 32 lanes)")
    if skip := wide_pixel_skip(name, out):
        return skip
    if name == "ps_vcmpx_forms":
        if skip := vcmpx_forms_narrow_skip(out):
            return skip
    if not px:
        return out if out.startswith("could not start") else "vk_harness failed or returned no pixels"
    n = len(px)
    if name == NAN_PAYLOAD_TEST:
        return nan_payload_problem(px, out)
    if name in PIXEL_BITS:
        wrong = sum(p != PIXEL_BITS[name] for p in px)
        return "" if not wrong else f"{wrong} of {n} pixels differ, e.g. {[hex(x) for x in next(p for p in px if p != PIXEL_BITS[name])]}"
    if name == "ps_bary":
        return barycentric_problem(px, 1e-6)
    if name == "ps_persp_bary":
        return bary_persp_problem(px)
    if name == "ps_flat":
        if any(not math.isfinite(v) for p in px for v in p[2:]):
            return "UVs contain non-finite values"
        w = int(n ** 0.5)
        wrong = sum((a, b) != (0x101, 0x102) for a, b, _, _ in px)
        err = max(max(abs(u - (i % w + 0.5) / w), abs(v - (i // w + 0.5) / w)) for i, (_, _, u, v) in enumerate(px))
        return "" if not wrong and err < 1e-6 else f"{wrong} pixels without the first vertex's value, uv off by {err:.3g}"
    if name == "ps_interp_f16_high":
        return interp_f16_high_problem(px)
    if name == "ps_ij_computed":
        return ij_computed_problem(px)
    if name == "ps_bitmask_perm":
        if skip := bitmask_perm_narrow_skip(out, px):
            return skip
        wrong = sum(not (a == lane ^ 9 and b == ((lane & 5) | 16) ^ 1) for lane, a, b, _ in px)
        return "" if px and not wrong else f"{wrong} of {len(px)} pixels wrong"
    if name == "ps_wqm":                               # (wqm, own, wqm, own) per pixel, ps_wqm.asm
        w = int(n ** 0.5)
        wrong = sum(p != (int(i % w % 4 < 2 and i // w % 4 >= 2), int(i % w % 4 == 1 and i // w % 4 == 2)) * 2 for i, p in
                    enumerate(px))
        return "" if not wrong else f"{wrong} of {n} pixels differ from the whole-quad expectation"
    if name in ("ps_uniform_branch", "ps_scc_mask32"):  # (s5, own VCC bit): 8 wherever the bit is set, never 7 there
        c = collections.Counter((a, b) for a, b, _, _ in px)
        fine = c[(8, 1)] and c[(8, 0)] and c[(7, 0)] and set(c) <= {(8, 1), (8, 0), (7, 0)}
        return "" if fine else f"(s5, bit) counts {dict(c)}"
    if name == "ps_mandelbrot":
        return mandelbrot_problem(px)
    return "no check written for this shader"


def clip_test(tools, spirv, name):
    with tempfile.TemporaryDirectory(prefix="leonida-clip-") as tmp:
        raw = Path(tmp) / "pixels.raw"
        for clip_x, clip_y in ((False, False), (True, False), (False, True), (True, True)):
            code, output = run(os.path.join(tools, "vk_harness" + EXE), os.path.join(spirv, name + ".spv"),
                               os.path.join(spirv, "ps_clip_color.spv"), raw, "64", "64",
                               "0" if clip_x else "4", "0" if clip_y else "4")
            if code == UNSUPPORTED:
                return Skip("SKIP (shaderClipDistance unavailable)")
            if code:
                return failure(code, output, 2)
            if not raw.is_file():
                return "missing framebuffer"
            data = raw.read_bytes()
            raw.unlink()
            if len(data) != 64 * 64 * 16:
                return "incomplete framebuffer"
            clear = (0xbf800000,) * 4
            wrong = 0
            for i, pixel in enumerate(struct.iter_unpack("<4I", data)):
                inside = (not clip_x or i % 64 < 32) and (not clip_y or i // 64 < 16)
                drawn = inside or name == "vs_clip_undeclared"
                wrong += pixel != (PIXEL_BITS["ps_clip_color"] if drawn else clear)
            if wrong:
                return f"x={clip_x} y={clip_y}: {wrong} of 4096 pixels differ"
    return ""


def main():
    if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
        print(__doc__)
        return 0
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    spirv = os.path.join(sys.argv[1], "spirv")
    tools = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "..", "build")
    tools = pick_build_dir(os.path.abspath(tools), tuple(t + EXE for t in TEST_PROGRAMS))
    results = []
    for name, groups, expect in COMPUTE:
        results.append((name, compute_test(tools, spirv, name, groups, expect)))
    results.append(("cs_wavemask", compute_test(tools, spirv, "cs_wavemask", "1", "wavemask",
                                                require_wave64=not W64_EMULATED)))
    for name, want in CS_DUMPS:   # all but cs_store need a 64-wide subgroup unless wave64 is emulated
        results.append((name, compute_test(tools, spirv, name, "1", "dump",
                                           require_wave64=name not in WAVE32_DUMPS and not W64_EMULATED,
                                           expected_words=want, needs_32_lanes=name in WAVE32_LANE_DUMPS)))
    for name, want in BOUNDS:
        results.append((name, compute_test(tools, spirv, name, "1", "dump", expected_words=want)))
    results.append(("cs_cmpx_class_f32", compute_test(tools, spirv, "cs_cmpx_class_f32", "1", "dump",
                                                      expected_words=CMPX_CLASS_WORDS)))
    results.append(("cs_fmt_1010102", fmt_1010102_test(tools, spirv)))
    results.append(("cs_fmt_sint32", fmt_sint32_test(tools, spirv)))
    results.append(("cs_fmt_int_narrow", fmt_int_narrow_test(tools, spirv)))
    results.append(("cs_fmt_xy_xyz", fmt_xy_xyz_test(tools, spirv)))
    results.append(("cs_fmt_all", fmt_all_test(tools, spirv)))
    results.append(("cs_tbuffer_store", tbuffer_store_test(tools, spirv)))
    results.append(("cs_fmt_d16_xyz", fmt_d16_xyz_test(tools, spirv)))
    for name, groups, want in GLOBAL_OFFSET:
        results.append((name, compute_test(tools, spirv, name, groups, "dump", expected_words=want)))
    results.append(("cs_user_sgprs", compute_test(tools, spirv, "cs_user_sgprs", "1", "dump",
                                                  expected_words=USER_SGPRS_WORDS, env=USER_SGPRS_ENV)))
    results.append(("cs_heap_init", compute_test(tools, spirv, "cs_heap_init", "1", "dump",
                                                 expected_words=HEAP_INIT_EXPECTED, env=heap_init_env())))
    results.append(("cs_cvt_f32_sat", cvt_f32_sat_test(tools, spirv)))
    results.append(("cs_fma_fused", fma_fused_test(tools, spirv)))
    results.append(("cs_cvt_pk_u8", compute_test(tools, spirv, "cs_cvt_pk_u8", "1", "dump",
                                                 expected_words=CVT_PK_U8_WORDS)))
    results.append(("cs_pk_mul_lo_const", compute_test(tools, spirv, "cs_pk_mul_lo_const", "1", "dump",
                                                       expected_words=PK_MUL_LO_WORDS)))
    for name, want in W64X_DUMPS:   # without --require-wave64 when LEONIDA_W64_EMULATE=1
        results.append((name, compute_test(tools, spirv, name, "1", "dump", require_wave64=not W64_EMULATED,
                                           expected_words=want)))
    for name, want in W64X_PARTIAL:
        results.append((name, partial_wave_test(tools, spirv, name, want)))
    results.append(("ms_w64x_ballot", mesh_w64x_test(tools, spirv, "ms_w64x_ballot")))
    results.append(("ms_wave_info_s3", mesh_w64x_test(tools, spirv, "ms_wave_info_s3")))
    with tempfile.TemporaryDirectory(prefix="leonida-mesh-") as tmp:
        results.append(("ms_v5_workgroup", mesh_v5_test(tools, spirv, Path(tmp) / "groups.raw")))
    for name, args in (("cs_imgstore", []), ("cs_imgload", ["load"]), ("cs_imgload_d16", ["load-d16"]),
                       ("cs_sample_lz_int_d16", ["load-d16"])):
        code, out = run(os.path.join(tools, "vk_store_test" + EXE), os.path.join(spirv, name + ".spv"), *args)
        if code == UNSUPPORTED:   # vk_store_test: the device lacks a feature this test needs
            m = re.search(r"unsupported: .*", out)
            results.append((name, Skip(f"SKIP ({m.group(0) if m else 'unsupported device'})")))
            continue
        bad = re.findall(r"(\d+) bad", out)
        results.append((name, "" if code == 0 and len(bad) == 3 and set(bad) == {"0"} else failure(code, out, 3)))
    for name in STORE_WORDS:   # TEST-184: image atomics and image_get_resinfo
        results.append((name, store_words_test(tools, spirv, name)))
    for name in GET_LOD_TESTS:   # TEST-184: image_get_lod, 2D and 2D array
        results.append((name, get_lod_test(tools, spirv, name)))
    for name in CLIP_TESTS:
        results.append((name, clip_test(tools, spirv, name)))
    for name in PIXEL_TESTS:
        results.append((name, pixel_test(tools, spirv, name)))
    for name in TEXTURE_CONSTANTS:
        results.append((name, texture_constant_test(tools, spirv, name)))
    results.append(("vs_base_ids", base_ids_test(tools, spirv)))
    results.append(("ps_tex", bindless_test(tools, spirv)))
    results.append(("ps_ssharp", sampler_test(tools, spirv)))
    results.append(("ps_imgatomic", bindless_test(tools, spirv, "ps_imgatomic")))
    for name in MIP_TEST_PASSES:
        results.append((name, texture_mip_test(tools, spirv, name)))
    results.append(("ps_table_fetch", table_fetch_test(tools, spirv)))
    for name in GRADIENT_TESTS:
        results.append((name, texture_gradient_a16_test(tools, spirv, name)))
    for name, problem in results:
        if problem is None:
            problem = "SKIP (full 64-wide compute subgroups unavailable)"
        print(f"{name:20} {'ok' if not problem else problem}")
    skipped = sum(p is None or isinstance(p, Skip) for _, p in results)
    failed = sum(bool(p) for _, p in results) - sum(isinstance(p, Skip) for _, p in results)
    print(f"{len(results) - failed - skipped} of {len(results)} pass, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
