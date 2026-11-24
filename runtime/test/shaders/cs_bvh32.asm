; image_bvh_intersect_ray, the form with a 32-bit node pointer (cs_bvh.asm runs image_bvh64_intersect_ray). Its address
; is 11 dwords: node, ray extent (t max), origin xyz, direction xyz, inverse direction xyz. The 64-bit form has the
; node's high dword after the low one, so read with that layout every operand here would be one register off.
; Same scene and ray as cs_bvh.asm: vk_compute_test writes an AMD-format BVH into the heap (box32 root at 0x100 ->
; triangle z=10 node 0x40, triangle z=5 node 0x48, a box the ray misses, a box with a NaN min.x). Ray origin
; (0.25, 0.25, 0), direction +z, t max 100. Run: vk_compute_test cs_bvh32.spv 1 bvh
; Expected, from the "Ray tracing" row of docs/design.md (one traversal step; the box test as measured on RDNA2,
; SHADER-650, and RADV's software triangle test):
;   box node 0x25 (address 0x100 >> 3 | type 5): along +z the ray enters a box that contains its x and y at the
;   box's near z, so child 0x40 (z = 10) at 10 and child 0x48 (z = 5) at 5. The third box lies at x in [5, 6], which
;   the ray never reaches: 0xffffffff. The fourth (0x68) has min.x = NaN; the hardware skips a NaN plane, so the box
;   is x <= 1, y and z in [0, 1], which the ray is in at t = 0. box_sort_enable (descriptor bit 63, set in s5) orders
;   the children by entry distance clamped to 0: out[0..3] = 0x68 0x48 0x40 0xffffffff (native gfx1030 gives the same).
;   triangle node 0x40 (address 0x200, vertices (0,0,10) (1,0,10) (0,1,10)): the ray runs along z, so the vertices
;   relative to the origin project onto xy unsheared: A = (-0.25, -0.25), B = (0.75, -0.25), C = (-0.25, 0.75). The
;   edge functions are u = Cx By - Cy Bx = -0.5, v = Ax Cy - Ay Cx = -0.25 and w = Bx Ay - By Ax = -0.25, all of one
;   sign (a hit); det = u + v + w = -1 and T = 10 (u + v + w) = -10. The result, as RDNA2 returns it, is
;   k (T, det, v, w) with k = |d.z|^3 = 1, so (-10, -1, -0.25, -0.25): out[4..7] = 0xc1200000 0xbf800000 0xbe800000
;   0xbe800000 (the same words natively on gfx1030, tests/semantic/test_image_bvh_tri_hw.py's cs_bvh row). The shader's
;   division by the second word gives t = 10 and the barycentrics 0.25, 0.25.
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v2, 0x25
v_mov_b32_e32 v3, 0x42c80000
v_mov_b32_e32 v4, 0.25
v_mov_b32_e32 v5, 0.25
v_mov_b32_e32 v6, 0
v_mov_b32_e32 v7, 0
v_mov_b32_e32 v8, 0
v_mov_b32_e32 v9, 1.0
v_mov_b32_e32 v10, 0x7f800000
v_mov_b32_e32 v11, 0x7f800000
v_mov_b32_e32 v12, 1.0
s_mov_b32 s4, 0
s_mov_b32 s5, 0x80000000
s_mov_b32 s6, -1
s_mov_b32 s7, 0x81000000
image_bvh_intersect_ray v[20:23], [v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12], s[4:7]
v_mov_b32_e32 v30, 0
buffer_store_dwordx4 v[20:23], v30, s[0:3], 0 offen
v_mov_b32_e32 v2, 0x40
image_bvh_intersect_ray v[24:27], [v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12], s[4:7]
v_mov_b32_e32 v30, 16
buffer_store_dwordx4 v[24:27], v30, s[0:3], 0 offen
s_endpgm
