; TEST-510: GFX10_FORMAT_8_UINT (code 5) loads on the GPU (docs/design.md, "Sub-dword MUBUF loads and
; GFX10_FORMAT_8_UINT"; CPU-checked by tests/semantic/test_buffer_load_sub.py). Thread 0 seeds heap bytes 0x40..0x43
; with 7f 80 ff 01. Every V# has base 0x40, stride 0, NUM_RECORDS 4 and OOB_SELECT 3 (RAW). Words, in out[0..7]:
;   0..3 tbuffer_load_format_x format:[BUF_FMT_8_UINT] at 0..3, V# format 20 (32_UINT): the instruction's own format
;        is used, one byte zero-extended: 0x7f 0x80 0xff 0x01
;   4    the same at 4: 4 + 1 > 4 -> 0
;   5    the same at 1 with V# format 0 (INVALID): an MTBUF load of an invalid V# reads 0 even in range
;   6, 7 buffer_load_format_xyzw at 2 with V# format 5 (8_UINT) and DST_SEL (X, 0, 1, 1): x = 0xff and w = integer 1
;        (not 1.0f, 0x3f800000)
; One thread, no wave64 requirement.
; Run: vk_compute_test cs_buf_fmt8.spv 1 dump -> out[0..7] = 0x0000007f 0x00000080 0x000000ff 0x00000001 0x00000000
; 0x00000000 0x000000ff 0x00000001 (127 128 255 1 0 0 255 1).
v_cmpx_eq_u32_e32 0, v0
v_mov_b32 v1, 0x01ff807f
buffer_store_dword v1, off, s[0:3], 0 offset:64
s_waitcnt vmcnt(0)
s_mov_b32 s4, 0x40
s_mov_b32 s5, 0
s_mov_b32 s6, 4
s_mov_b32 s7, 0x31014fac
s_mov_b32 s8, 0x40
s_mov_b32 s9, 0
s_mov_b32 s10, 4
s_mov_b32 s11, 0x31000fac
s_mov_b32 s12, 0x40
s_mov_b32 s13, 0
s_mov_b32 s14, 4
s_mov_b32 s15, 0x31005244
v_mov_b32 v20, 0
v_mov_b32 v21, 1
v_mov_b32 v22, 2
v_mov_b32 v23, 3
v_mov_b32 v24, 4
tbuffer_load_format_x v10, v20, s[4:7], 0 format:[BUF_FMT_8_UINT] offen
tbuffer_load_format_x v11, v21, s[4:7], 0 format:[BUF_FMT_8_UINT] offen
tbuffer_load_format_x v12, v22, s[4:7], 0 format:[BUF_FMT_8_UINT] offen
tbuffer_load_format_x v13, v23, s[4:7], 0 format:[BUF_FMT_8_UINT] offen
tbuffer_load_format_x v14, v24, s[4:7], 0 format:[BUF_FMT_8_UINT] offen
tbuffer_load_format_x v15, v21, s[8:11], 0 format:[BUF_FMT_8_UINT] offen
buffer_load_format_xyzw v[30:33], v22, s[12:15], 0 offen
s_waitcnt vmcnt(0)
v_mov_b32 v16, v30
v_mov_b32 v17, v33
buffer_store_dwordx4 v[10:13], off, s[0:3], 0
buffer_store_dwordx4 v[14:17], off, s[0:3], 0 offset:16
s_endpgm
