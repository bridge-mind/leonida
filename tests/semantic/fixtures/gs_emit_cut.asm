; Two separate triangles in clip space. Vertices 0,1,2 and 3,4,5 must form
; two strips. Joining them also rasterizes triangles spanning the gap.
; The output signature is SV_Position only (four dwords per vertex).
v_mov_b32 v0, 0
v_mov_b32 v1, 1.0
v_mov_b32 v2, -1.0
v_mov_b32 v3, -0.5
v_mov_b32 v4, 0.5
ds_write2_b32 v0, v2, v3 offset0:0 offset1:1
ds_write2_b32 v0, v0, v1 offset0:2 offset1:3
s_waitcnt lgkmcnt(0)
s_sendmsg sendmsg(MSG_GS, GS_OP_EMIT, 0)
ds_write2_b32 v0, v3, v3 offset0:4 offset1:5
ds_write2_b32 v0, v0, v1 offset0:6 offset1:7
s_waitcnt lgkmcnt(0)
s_sendmsg sendmsg(MSG_GS, GS_OP_EMIT, 0)
ds_write2_b32 v0, v2, v4 offset0:8 offset1:9
ds_write2_b32 v0, v0, v1 offset0:10 offset1:11
s_waitcnt lgkmcnt(0)
s_sendmsg sendmsg(MSG_GS, GS_OP_EMIT_CUT, 0)
ds_write2_b32 v0, v4, v3 offset0:12 offset1:13
ds_write2_b32 v0, v0, v1 offset0:14 offset1:15
s_waitcnt lgkmcnt(0)
s_sendmsg sendmsg(MSG_GS, GS_OP_EMIT, 0)
ds_write2_b32 v0, v1, v3 offset0:16 offset1:17
ds_write2_b32 v0, v0, v1 offset0:18 offset1:19
s_waitcnt lgkmcnt(0)
s_sendmsg sendmsg(MSG_GS, GS_OP_EMIT, 0)
ds_write2_b32 v0, v1, v4 offset0:20 offset1:21
ds_write2_b32 v0, v0, v1 offset0:22 offset1:23
s_waitcnt lgkmcnt(0)
s_sendmsg sendmsg(MSG_GS, GS_OP_EMIT_CUT, 0)
s_sendmsg sendmsg(MSG_GS_DONE, GS_OP_NOP)
s_endpgm
