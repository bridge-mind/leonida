; group: 64,1,1
; rsrc2: 0x808c
; fssm-at: 0x150
; fssm-data: 0x4e554f43 0x3d524554
; UAV counters (D3D12 CreateUnorderedAccessView with pCounterResource): AppendStructuredBuffer and
; ConsumeStructuredBuffer as hand-written RDNA2, the way docs/d3d12_vulkan.md section 3.4 describes the contract:
; the counter is a second, raw 4-byte V# in the upper 16 bytes of the view's descriptor slot, and append and consume
; are buffer atomics on it. Run: d3d12_buffer_uav_test (a compute PSO found by the SHA-256 of its synthetic FSSM header).
; The branch skips the 10 dwords of the append path (checked with llvm-mc -show-encoding).
; User data, from the root signature: [0] four 32-bit constants, [1] a descriptor table of six UAVs, TGID_X in s6:
;   s0 = mode: 0 append, 1 consume   s[4:5] = console VA of the table's first slot (32 bytes a slot)
;   slot 0 (+0x00): V#, structured buffer, stride 8, 256 records; the counter V# in +0x10 (raw, 4 bytes)
;   slot 1 (+0x20): V#, raw buffer, 128 records of 8 bytes (written by consume)
; D3D12 build: runtime/CMakeLists.txt builds this with LEONIDA_USER_DATA_SHIFT=4, so in the PSO every SGPR named
; here is 4 higher (the console compute layout starts the root arguments at s4, D3D-448). The table pointer
; then reads s[8:9]: s8 is the table's 32-bit VA and s9 a user dword the runtime leaves zero.
; g = 64 * group + lane
;   append:  i = counter++ (buffer_atomic_add, glc returns the old value); record i = { g, g * 3 + 1 }
;   consume: i = --counter (the old value of buffer_atomic_sub, minus one); out[g] = record i
s_load_dwordx8 s[8:15], s[4:5], 0x0
s_load_dwordx4 s[16:19], s[4:5], 0x20
s_lshl_b32 s7, s6, 6
s_waitcnt lgkmcnt(0)
v_add_nc_u32_e32 v1, s7, v0
v_mov_b32_e32 v2, 1
s_cmp_eq_u32 s0, 0
s_cbranch_scc0 10
buffer_atomic_add v2, off, s[12:15], 0 glc
s_waitcnt vmcnt(0)
v_mul_lo_u32 v6, v1, 3
v_add_nc_u32_e32 v6, 1, v6
v_mov_b32_e32 v5, v1
buffer_store_dwordx2 v[5:6], v2, s[8:11], 0 idxen
s_endpgm
; consume:
buffer_atomic_sub v2, off, s[12:15], 0 glc
s_waitcnt vmcnt(0)
v_add_nc_u32_e32 v2, -1, v2
buffer_load_dwordx2 v[5:6], v2, s[8:11], 0 idxen
s_waitcnt vmcnt(0)
v_lshlrev_b32_e32 v7, 3, v1
buffer_store_dwordx2 v[5:6], v7, s[16:19], 0 offen
s_endpgm
