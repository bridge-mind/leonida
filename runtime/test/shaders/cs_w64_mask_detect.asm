; Wave64 detection from a 64-bit lane-mask op alone (#210 part 2 / SHADER-28).
; mattr: +wavefrontsize64,-wavefrontsize32
; No exec_hi / mbcnt_hi / *_saveexec_b64 / *_wrexec_b64 and no _b64 op on exec/vcc: only s_bcnt1_i32_b64
; on an ordinary SGPR pair. Lanes where (lane & 5) != 0: 48 of 64. A wave32 mis-detect counts 24.
; Run: VKC_SUBGROUP_SIZE=64 vk_compute_test cs_w64_mask_detect.spv 1 dump --require-wave64
;   -> out[0] = 48
; With LEONIDA_W64_EMULATE=1 the same on a 32-wide subgroup.
; Every lane stores the count at lane * 4, so the 64 lanes fill the 256 bytes vk_compute_test gives one group. (At
; lane * 64, lanes 4-63 wrote past the buffer through its device address: the V# allows any offset.)
v_and_b32_e32 v5, 5, v0
v_cmp_ne_u32_e64 s[20:21], 0, v5
s_bcnt1_i32_b64 s30, s[20:21]
v_mov_b32 v1, s30
v_lshlrev_b32_e32 v13, 2, v0
buffer_store_dword v1, v13, s[0:3], 0 offen
s_endpgm
