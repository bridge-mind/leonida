; M1.5 vertex shader: vertex fetch through a V# the application wrote into a descriptor-table slot, the way the
; translated game VS that fetch without the driver's fetch program do. Run: d3d12_translated_draw_test (the VS through
; a D3D12 graphics PSO found by the SHA-256 of its FSSM program).
;
; User data, from the root signature [0] descriptor table, [1] four 32-bit constants (docs/d3d12_vulkan.md 3.5):
;   s[0:1] = console VA of the table's first slot: the 16 bytes of a V# (base VA, stride 32, 9 records)
;   s2, s3 = the draw's clip-space offset (x, y) as floats;   s4, s5 = used by the pixel shader (s4: bias)
; v0 = VertexIndex. The draw has BaseVertex 0 and BaseInstance 0, so v0 is the same with and without the translator's
; zero-based ids (PR 436): both give 0, 1, 2, ... for DrawInstanced(n, 1, 0, 0). A non-zero StartVertexLocation would
; make the two differ (hardware: zero-based, base in an SGPR), so the test never uses one.
; Vertex record, 32 bytes: position xyzw at 0, colour rgba at 16 (both float).
;   pos0   = (position.x + s2, position.y + s3, position.z, position.w)
;   param0 = colour
; fssm-at: 0x10
; fssm-data: 0x35314d56 0x20535620
s_load_dwordx4 s[8:11], s[0:1], 0x0
s_waitcnt lgkmcnt(0)
buffer_load_dwordx4 v[4:7], v0, s[8:11], 0 idxen
buffer_load_dwordx4 v[8:11], v0, s[8:11], 0 idxen offset:16
s_waitcnt vmcnt(0)
v_add_f32_e32 v4, s2, v4
v_add_f32_e32 v5, s3, v5
exp pos0 v4, v5, v6, v7 done
exp param0 v8, v9, v10, v11
s_endpgm
