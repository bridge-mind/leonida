; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; Wave64 ballot + v_mbcnt prefix-count stream compaction across both halves. Selected lanes: (lane & 7) == 5, that
; is lanes 5, 13, 21, 29 (low half) and 37, 45, 53, 61 (upper half). The ballot is v_cmp_*_e64 into s[4:5]; s_bcnt1
; of it = 8 (a 32-wide subgroup sees one half: 4). The slot of a selected lane = v_mbcnt_lo(s4) + v_mbcnt_hi(s5) =
; selected lanes below it = lane >> 3 (upper lanes must count the 4 low-half lanes too). Each selected lane stores
; lane + (bcnt << 8) = lane + 0x800 into out[slot]:
;   out[0..7] = 0x805 0x80d 0x815 0x81d 0x825 0x82d 0x835 0x83d = 2053 2061 2069 2077 2085 2093 2101 2109
; Run: vk_compute_test cs_w64x_ballot.spv 1 dump --require-wave64
; mattr: +wavefrontsize64,-wavefrontsize32
v_mbcnt_lo_u32_b32 v1, -1, 0
v_mbcnt_hi_u32_b32 v1, -1, v1
v_and_b32_e32 v2, 7, v1
v_cmp_eq_u32_e64 s[4:5], 5, v2
s_bcnt1_i32_b64 s6, s[4:5]
v_mbcnt_lo_u32_b32 v3, s4, 0
v_mbcnt_hi_u32_b32 v3, s5, v3
v_lshlrev_b32_e32 v4, 2, v3
s_lshl_b32 s7, s6, 8
v_add_nc_u32_e32 v5, s7, v1
s_mov_b64 exec, s[4:5]
buffer_store_dword v5, v4, s[0:3], 0 offen
s_endpgm
