; Legacy (LDS-ring) geometry shader idioms: s_bfm_b64/s_cselect EXEC prologue, input vertex offsets from v0/v1 (SDWA),
; LDS reads of SV_Position (reg 0) and TEXCOORD0 (reg 1), output slots of 2 registers (8 dwords), EMIT x3, CUT, DONE.
; Needs a DXBC with ISG1/OSG1 = {SV_Position reg 0, TEXCOORD0 reg 1}. Run: VKH_GS=gs_tri.spv vk_harness vs_attrs.spv
; fs_loc1.spv out.raw (fs_loc1 = fs_passthru reading location 1) -> every pixel = ((x+0.5)/64, (y+0.5)/64, 0, 0).
s_bfe_i32 s0, s3, 0x80008
s_bfm_b64 exec, s0, 0
s_cmp_eq_u32 s0, 64
s_cselect_b64 exec, -1, exec
v_lshlrev_b32_e32 v3, 2, v0
v_and_b32_e32 v10, 0x3fffc, v3
s_mov_b32 s1, 2
v_lshlrev_b32_sdwa v8, s1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
v_lshlrev_b32_e32 v1, 2, v1
v_and_b32_e32 v6, 0x3fffc, v1
v_mov_b32_e32 v12, 0
ds_read2_b32 v[20:21], v10 offset1:1
ds_read2_b32 v[22:23], v10 offset0:2 offset1:3
ds_read2_b32 v[24:25], v10 offset0:4 offset1:5
s_waitcnt lgkmcnt(0)
ds_write2_b32 v12, v20, v21 offset0:0 offset1:1
ds_write2_b32 v12, v22, v23 offset0:2 offset1:3
ds_write2_b32 v12, v24, v25 offset0:4 offset1:5
s_sendmsg sendmsg(MSG_GS, GS_OP_EMIT, 0)
ds_read2_b32 v[20:21], v8 offset1:1
ds_read2_b32 v[22:23], v8 offset0:2 offset1:3
ds_read2_b32 v[24:25], v8 offset0:4 offset1:5
s_waitcnt lgkmcnt(0)
ds_write2_b32 v12, v20, v21 offset0:8 offset1:9
ds_write2_b32 v12, v22, v23 offset0:10 offset1:11
ds_write2_b32 v12, v24, v25 offset0:12 offset1:13
s_sendmsg sendmsg(MSG_GS, GS_OP_EMIT, 0)
ds_read2_b32 v[20:21], v6 offset1:1
ds_read2_b32 v[22:23], v6 offset0:2 offset1:3
ds_read2_b32 v[24:25], v6 offset0:4 offset1:5
s_waitcnt lgkmcnt(0)
ds_write2_b32 v12, v20, v21 offset0:16 offset1:17
ds_write2_b32 v12, v22, v23 offset0:18 offset1:19
ds_write2_b32 v12, v24, v25 offset0:20 offset1:21
s_sendmsg sendmsg(MSG_GS, GS_OP_EMIT, 0)
s_sendmsg sendmsg(MSG_GS, GS_OP_CUT, 0)
s_sendmsg sendmsg(MSG_GS_DONE, GS_OP_NOP)
s_endpgm
