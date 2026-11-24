; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; The d16 typed-buffer forms no other test runs: buffer_load_format_d16_xyz, tbuffer_load_format_d16_xy,
; tbuffer_load_format_d16_xyz and buffer_store_format_d16_xyz. One thread (lane 0) does everything.
; RDNA2 ISA (MUBUF/MTBUF): a d16 load converts each component to 16 bits (f16 for float and normalized formats, the
; integer saturated to 16 bits for UINT/SINT) and packs two per VGPR, X in bits 15:0 and Y in 31:16 of the first, Z
; in 15:0 of the second. docs/design.md (d16 and f16 format narrowing): an odd final component (d16_x, d16_xyz)
; writes only the low 16 bits of its VGPR and the high half keeps its bits; unsigned 32-bit channels clamp to 65535.
; A d16 store takes the components from the same halves and encodes them in the V# format (an f16 widened to the
; f32 of a 32_FLOAT channel, a 16-bit integer zero-extended into a 32_UINT channel).
; MUBUF uses the V# format (word3 bits 18:12) and DST_SEL (bits 11:0, here X Y Z W); MTBUF uses the instruction's
; format and identity selectors with a bound V# (32_FLOAT). V# word3 = 0x31000fac | format << 12.
; Inputs, written first by lane 0:
;   0x100: 0xabcd1234 0x55557fff   16_16_16_16 X=0x1234 Y=0xabcd Z=0x7fff W=0x5555
;   0x108: 0xc0003c00 0x7bff3555   f16 X=1.0 (0x3c00) Y=-2.0 (0xc000) Z=0x3555 (0.33325) W=0x7bff
;   0x110: 5 0x10000 0xffffffff 0  32-bit X Y Z (and a zero pad)
;   0x120: 0x80ff00ff              8_8_8_8 X=0xff Y=0x00 Z=0xff W=0x80
; Loads (the second VGPR of each _xyz is seeded first; its high half must survive):
;   L1 buffer_load_format_d16_xyz v[8:9] @0x100, V# 16_16_16_16_UINT (69): X, Y, Z fit in 16 bits.
;      v8 = 0xabcd1234; v9 = 0xa5a5 << 16 | 0x7fff = 0xa5a57fff (seed 0xa5a55a5a)
;   L2 buffer_load_format_d16_xyz v[10:11] @0x110, V# 32_32_32_UINT (72): 5, min(0x10000, 65535) = 0xffff,
;      min(0xffffffff, 65535) = 0xffff.  v10 = 0xffff0005; v11 = 0x1234ffff (seed 0x12345678)
;   L3 tbuffer_load_format_d16_xy v12 @0x108, BUF_FMT_16_16_FLOAT: the f16 bits as they are. v12 = 0xc0003c00
;   L4 tbuffer_load_format_d16_xyz v[13:14] @0x108, BUF_FMT_16_16_16_16_FLOAT: v13 = 0xc0003c00;
;      v14 = 0xbeef3555 (seed 0xbeef7777)
;   L5 tbuffer_load_format_d16_xyz v[15:16] @0x120, BUF_FMT_8_8_8_8_UNORM: 255/255 = 1.0 -> 0x3c00, 0/255 = 0,
;      255/255 = 1.0 -> 0x3c00 (exact, no rounding).  v15 = 0x00003c00; v16 = 0xcafe3c00 (seed 0xcafe1111)
;   L6 tbuffer_load_format_d16_xyz v[17:18] @0x110, BUF_FMT_32_32_32_UINT: as L2. v17 = 0xffff0005;
;      v18 = 0x0badffff (seed 0x0bad0bad)
;   v8..v18 go to out[0..10] (0x00..0x28); out[11..15] are untouched (0xeeeeeeee).
; Stores (the fourth word of each 16-byte region stays untouched):
;   S1 buffer_store_format_d16_xyz v[20:21] @0x40, V# 32_32_32_UINT (72): v20 = 0x80010007, v21 = 0xdeadfffe
;      -> X = 0x0007, Y = 0x8001, Z = 0xfffe (v21's high half is not a component), zero-extended:
;      out[16..19] = 0x00000007 0x00008001 0x0000fffe 0xeeeeeeee
;   S2 buffer_store_format_d16_xyz v[22:23] @0x50, V# 32_32_32_FLOAT (74): v22 = 0xc0003c00, v23 = 0x12343800
;      -> 1.0, -2.0, 0.5 (0x3800) as f32: out[20..23] = 0x3f800000 0xc0000000 0x3f000000 0xeeeeeeee
;   out[24..63] are untouched except the inputs at out[64..72] (0x100..0x120); the dump is 80 words.
; Run: VKC_OUT_WORDS=80 VKC_DUMP=<file> vk_compute_test cs_fmt_d16_xyz.spv 1 dump
v_cmpx_eq_u32_e32 0, v0
s_mov_b32 s4, s0
s_mov_b32 s5, s1
s_mov_b32 s6, s2
v_mov_b32_e32 v1, 0xabcd1234
v_mov_b32_e32 v2, 0x55557fff
v_mov_b32_e32 v3, 0xc0003c00
v_mov_b32_e32 v4, 0x7bff3555
v_mov_b32_e32 v5, 0x100
buffer_store_dwordx4 v[1:4], v5, s[0:3], 0 offen
v_mov_b32_e32 v1, 5
v_mov_b32_e32 v2, 0x10000
v_mov_b32_e32 v3, -1
v_mov_b32_e32 v4, 0
v_mov_b32_e32 v5, 0x110
buffer_store_dwordx4 v[1:4], v5, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x80ff00ff
v_mov_b32_e32 v5, 0x120
buffer_store_dword v1, v5, s[0:3], 0 offen
s_waitcnt vmcnt(0)
; L1
s_mov_b32 s7, 0x31045fac
v_mov_b32_e32 v5, 0x100
v_mov_b32_e32 v9, 0xa5a55a5a
buffer_load_format_d16_xyz v[8:9], v5, s[4:7], 0 offen
; L2
s_mov_b32 s7, 0x31048fac
v_mov_b32_e32 v5, 0x110
v_mov_b32_e32 v11, 0x12345678
buffer_load_format_d16_xyz v[10:11], v5, s[4:7], 0 offen
; L3..L6: MTBUF with a bound 32_FLOAT V#
s_mov_b32 s7, 0x31016fac
v_mov_b32_e32 v5, 0x108
tbuffer_load_format_d16_xy v12, v5, s[4:7], 0 format:[BUF_FMT_16_16_FLOAT] offen
v_mov_b32_e32 v14, 0xbeef7777
tbuffer_load_format_d16_xyz v[13:14], v5, s[4:7], 0 format:[BUF_FMT_16_16_16_16_FLOAT] offen
v_mov_b32_e32 v5, 0x120
v_mov_b32_e32 v16, 0xcafe1111
tbuffer_load_format_d16_xyz v[15:16], v5, s[4:7], 0 format:[BUF_FMT_8_8_8_8_UNORM] offen
v_mov_b32_e32 v5, 0x110
v_mov_b32_e32 v18, 0x0bad0bad
tbuffer_load_format_d16_xyz v[17:18], v5, s[4:7], 0 format:[BUF_FMT_32_32_32_UINT] offen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v6, 0
buffer_store_dwordx4 v[8:11], v6, s[0:3], 0 offen
v_mov_b32_e32 v6, 0x10
buffer_store_dwordx4 v[12:15], v6, s[0:3], 0 offen
v_mov_b32_e32 v6, 0x20
buffer_store_dwordx3 v[16:18], v6, s[0:3], 0 offen
; S1
s_mov_b32 s7, 0x31048fac
v_mov_b32_e32 v20, 0x80010007
v_mov_b32_e32 v21, 0xdeadfffe
v_mov_b32_e32 v6, 0x40
buffer_store_format_d16_xyz v[20:21], v6, s[4:7], 0 offen
; S2
s_mov_b32 s7, 0x3104afac
v_mov_b32_e32 v22, 0xc0003c00
v_mov_b32_e32 v23, 0x12343800
v_mov_b32_e32 v6, 0x50
buffer_store_format_d16_xyz v[22:23], v6, s[4:7], 0 offen
s_endpgm
