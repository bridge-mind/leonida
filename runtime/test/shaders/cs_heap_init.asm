; group: 1,1,1
; rsrc2: 0x8090
; vk_compute_test's VKC_HEAP_INIT copies a file to the start of the heap before the dispatch. Both runners write a
; 0x110-byte file (zeros, then the words 0x1badb002, 0xcafef00d, 0x80000001, 0x0000ffff at 0x100) and pass
; VKC_OUT_WORDS=68 so it fits. One thread loads the four words through the output V# and stores them at 0, so
; out[0..3] are the seeded words and out[4..7] the file's zeros at 0x10..0x1f (not the 0xEE fill): the whole file was
; copied. Without VKC_HEAP_INIT every word reads 0xEEEEEEEE.
; Run: VKC_OUT_WORDS=68 VKC_HEAP_INIT=init.bin vk_compute_test cs_heap_init.spv 1 dump
; out[0..7] = 1BADB002 CAFEF00D 80000001 0000FFFF 00000000 00000000 00000000 00000000 (hex)
v_mov_b32_e32 v1, 0
buffer_load_dwordx4 v[2:5], v1, s[0:3], 0 offen offset:256
s_waitcnt vmcnt(0)
buffer_store_dwordx4 v[2:5], v1, s[0:3], 0 offen
s_endpgm
