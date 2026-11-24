; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; SHADER-257: GFX10 32-bit signed typed buffers. Lane 0 writes [-2147483648, 2147483647, -3, 17] at 0x80.
; MUBUF loads at out[0..11] test integer DST_SEL=1 for format 63 (X,Y,X,1) and raw words for 73/76.
; MTBUF loads at out[12..23] use a bound 32_FLOAT descriptor and take width and identity selectors from the
; instruction (XY00, XYZ0, XYZW). D16 loads at out[24..27] check signed saturation, including -3, for both forms.
; Single-channel SINT MUBUF/MTBUF loads place -3 in out[28..29].
; MUBUF stores at 0x90/0xa0/0xb0 write the low 2/3/4 original words. The D16 store at 0xc0 sign-extends
; 0xfffd and 0x8000 to dwords. The single-channel store at 0xc8 writes -3. Unwritten words retain 0xeeeeeeee.
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v1, 0x80000000
v_mov_b32_e32 v2, 0x7fffffff
v_mov_b32_e32 v3, -3
v_mov_b32_e32 v4, 17
v_mov_b32_e32 v5, 0x80
buffer_store_dwordx4 v[1:4], v5, s[0:3], 0 offen
s_mov_b32 s4, s0
s_mov_b32 s5, s1
s_mov_b32 s6, s2

; MUBUF load via format 63.
s_mov_b32 s7, 0x3103f32c
buffer_load_format_xyzw v[8:11], v5, s[4:7], 0 offen
v_mov_b32_e32 v6, 0x0
buffer_store_dwordx4 v[8:11], v6, s[0:3], 0 offen

; MUBUF load via format 73.
s_mov_b32 s7, 0x31049fac
buffer_load_format_xyzw v[8:11], v5, s[4:7], 0 offen
v_mov_b32_e32 v6, 0x10
buffer_store_dwordx4 v[8:11], v6, s[0:3], 0 offen

; MUBUF load via format 76.
s_mov_b32 s7, 0x3104cfac
buffer_load_format_xyzw v[8:11], v5, s[4:7], 0 offen
v_mov_b32_e32 v6, 0x20
buffer_store_dwordx4 v[8:11], v6, s[0:3], 0 offen

; A bound FLOAT descriptor with a different size and selectors from each instruction.
s_mov_b32 s7, 0x31016000
tbuffer_load_format_xyzw v[8:11], v5, s[4:7], 0 format:[BUF_FMT_32_32_SINT] offen
v_mov_b32_e32 v6, 0x30
buffer_store_dwordx4 v[8:11], v6, s[0:3], 0 offen
tbuffer_load_format_xyzw v[8:11], v5, s[4:7], 0 format:[BUF_FMT_32_32_32_SINT] offen
v_mov_b32_e32 v6, 0x40
buffer_store_dwordx4 v[8:11], v6, s[0:3], 0 offen
tbuffer_load_format_xyzw v[8:11], v5, s[4:7], 0 format:[BUF_FMT_32_32_32_32_SINT] offen
v_mov_b32_e32 v6, 0x50
buffer_store_dwordx4 v[8:11], v6, s[0:3], 0 offen

; Packs (0x8000, 0x7fff) and (0xfffd, 0x0011).
s_mov_b32 s7, 0x3104cfac
buffer_load_format_d16_xyzw v[8:9], v5, s[4:7], 0 offen
v_mov_b32_e32 v6, 0x60
buffer_store_dwordx2 v[8:9], v6, s[0:3], 0 offen
s_mov_b32 s7, 0x31016000
tbuffer_load_format_d16_xyzw v[8:9], v5, s[4:7], 0 format:[BUF_FMT_32_32_32_32_SINT] offen
v_mov_b32_e32 v6, 0x68
buffer_store_dwordx2 v[8:9], v6, s[0:3], 0 offen

; Single-component 32_SINT at 0x88, with a bound FLOAT descriptor for the MTBUF case.
s_mov_b32 s7, 0x31015004
buffer_load_format_x v8, v5, s[4:7], 0 offen offset:8
s_mov_b32 s7, 0x31016000
tbuffer_load_format_x v9, v5, s[4:7], 0 format:[BUF_FMT_32_SINT] offen offset:8
v_mov_b32_e32 v6, 0x70
buffer_store_dwordx2 v[8:9], v6, s[0:3], 0 offen

; MUBUF store via format 63 (DST_SEL XY01, standard 2-component RG).
s_mov_b32 s7, 0x3103f22c
v_mov_b32_e32 v6, 0x90
buffer_store_format_xy v[1:2], v6, s[4:7], 0 offen

; MUBUF store via format 73.
s_mov_b32 s7, 0x31049fac
v_mov_b32_e32 v6, 0xa0
buffer_store_format_xyz v[1:3], v6, s[4:7], 0 offen

; MUBUF store via format 76.
s_mov_b32 s7, 0x3104cfac
v_mov_b32_e32 v6, 0xb0
buffer_store_format_xyzw v[1:4], v6, s[4:7], 0 offen

; A signed packed store must extend both halves.
s_mov_b32 s7, 0x3103f22c
v_mov_b32_e32 v6, 0xc0
v_mov_b32_e32 v7, 0x8000fffd
buffer_store_format_d16_xy v7, v6, s[4:7], 0 offen
s_mov_b32 s7, 0x31015004
v_mov_b32_e32 v6, 0xc8
buffer_store_format_x v3, v6, s[4:7], 0 offen
s_endpgm
