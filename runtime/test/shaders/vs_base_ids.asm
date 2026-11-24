; Vertex shader for the base-vertex / base-instance test. On AMD hardware the VS input VGPRs are zero-based: v0 is
; the vertex id without the draw's first vertex (Mesa: ac.vertex_id = load_vertex_id_zero_base, the base comes in
; the base_vertex SGPR), and v3 the instance id without its first instance (ac.instance_id = load_instance_id, the
; base in start_instance). Vulkan's VertexIndex / InstanceIndex include firstVertex / firstInstance.
; Same full-screen triangle as vs_ids (from v0's low bits), param0 = (u, v, 0, 1), and param1.x =
; 0x101 + v0 + 0x1000 * v3 as raw bits, so ps_flat (attr1.x from the provoking vertex) shows both ids.
; Run: VKH_DRAW=16,7 vk_harness vs_base_ids.spv ps_flat.spv out.raw
;   -> mrt0 bits = 0x101, 0x102 on every pixel (vertex 0 of instance 0); 0x7111 means firstVertex/firstInstance
;      leaked into v0 / v3. With VKH_DRAW unset (0,0) both read 0x101, as vs_ids.
v_lshlrev_b32_e32 v8, 12, v3
v_lshlrev_b32_e32 v1, 1, v0
v_and_b32_e32 v1, 2, v1
v_and_b32_e32 v2, 2, v0
v_cvt_f32_u32_e32 v1, v1
v_cvt_f32_u32_e32 v2, v2
v_fma_f32 v3, v1, 2.0, -1.0
v_fma_f32 v4, v2, 2.0, -1.0
v_mov_b32_e32 v5, 0
v_mov_b32_e32 v6, 1.0
exp pos0 v3, v4, v5, v6 done
exp param0 v1, v2, v5, v6
v_add3_u32 v7, v0, v8, 0x101
exp param1 v7, v7, v7, v7
s_endpgm
