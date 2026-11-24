; For an active lane, v_cmp writes its constant predicate into the named mask.
; v_cmpx changes only EXEC. A true predicate cannot reactivate an inactive lane.
v_cmp_f_i64_e32 vcc_lo, v[0:1], v[2:3]
v_cmp_t_i64_e64 s10, v[0:1], v[2:3]
v_cmp_f_u64_e64 s10, v[0:1], v[2:3]
v_cmp_t_u64_e32 vcc_lo, v[0:1], v[2:3]
v_cmp_f_f64_e32 vcc_lo, v[0:1], v[2:3]
v_cmp_tru_f64_e64 s10, v[0:1], v[2:3]
v_cmpx_f_i64_e32 v[0:1], v[2:3]
v_cmpx_t_i64_e64 v[0:1], v[2:3]
v_cmpx_f_u64_e64 v[0:1], v[2:3]
v_cmpx_t_u64_e32 v[0:1], v[2:3]
v_cmpx_f_f64_e32 v[0:1], v[2:3]
v_cmpx_tru_f64_e64 v[0:1], v[2:3]
