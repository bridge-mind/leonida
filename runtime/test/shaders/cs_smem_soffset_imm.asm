; SMEM loads that carry a register offset AND an immediate (llvm-mc: "sOFF offset:IMM"): address = base + soffset + imm.
; Measured on an AMD Radeon RX 6800 (gfx1030) by running s_buffer_load_dword/x2/x4/x8 as a raw compute kernel:
; soffset 8 + imm 4 reads the dword at +12. s_load_dword shares the SMEM encoding; applying the rule to it is an inference.
; Thread 0 seeds heap bytes 0x10..0x30 with 0x5eed0000 + byte offset, then loads with s8 = 0x10:
;   s_buffer_load_dword   s9,        s[0:3], s8 offset:0x14 -> byte 0x24        (imm dropped: 0x10, s8 dropped: 0x14)
;   s_load_dword          s10,       s[4:5], s8 offset:0x18 -> byte 0x28        (imm dropped: 0x10, s8 dropped: 0x18)
;   s_buffer_load_dwordx2 s[12:13],  s[0:3], s8 offset:0x1c -> bytes 0x2c, 0x30 (imm dropped: 0x10, 0x14; s8 dropped: 0x1c, 0x20)
; and writes s9, s10, s12, s13 to out[0..3]; out[4..7] keep the seeds of bytes 0x10..0x1c.
; All offsets are multiples of 4 and inside the buffer, so alignment and num_records play no part.
; Run: vk_compute_test cs_smem_soffset_imm.spv 1 dump -> out[0..7] = 0x5eed0024 0x5eed0028 0x5eed002c 0x5eed0030
; 0x5eed0010 0x5eed0014 0x5eed0018 0x5eed001c (1592590372 1592590376 1592590380 1592590384 1592590352 1592590356
; 1592590360 1592590364). Same FSSM header as cs_lds.
v_cmpx_eq_u32_e32 0, v0
v_mov_b32 v1, 0x5eed0010
v_mov_b32 v2, 0x10
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32 v1, 0x5eed0014
v_mov_b32 v2, 0x14
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32 v1, 0x5eed0018
v_mov_b32 v2, 0x18
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32 v1, 0x5eed001c
v_mov_b32 v2, 0x1c
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32 v1, 0x5eed0020
v_mov_b32 v2, 0x20
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32 v1, 0x5eed0024
v_mov_b32 v2, 0x24
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32 v1, 0x5eed0028
v_mov_b32 v2, 0x28
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32 v1, 0x5eed002c
v_mov_b32 v2, 0x2c
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32 v1, 0x5eed0030
v_mov_b32 v2, 0x30
buffer_store_dword v1, v2, s[0:3], 0 offen
s_waitcnt vmcnt(0)
s_mov_b32 s4, 0
s_mov_b32 s5, 0
s_mov_b32 s8, 0x10
s_buffer_load_dword s9, s[0:3], s8 offset:0x14
s_load_dword s10, s[4:5], s8 offset:0x18
s_buffer_load_dwordx2 s[12:13], s[0:3], s8 offset:0x1c
s_waitcnt lgkmcnt(0)
v_mov_b32 v1, s9
v_mov_b32 v2, 0
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32 v1, s10
v_mov_b32 v2, 4
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32 v1, s12
v_mov_b32 v2, 8
buffer_store_dword v1, v2, s[0:3], 0 offen
v_mov_b32 v1, s13
v_mov_b32 v2, 12
buffer_store_dword v1, v2, s[0:3], 0 offen
s_endpgm
