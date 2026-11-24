"""RDNA2 null reads zero in either 64-bit integer VALU compare source."""
import itertools
import struct
import unittest

import support
import spirv_mini_interp as si

CONDITIONS = ("eq", "ne", "lt", "le", "gt", "ge")
VALUES = (0, 1, 1 << 32, 1 << 63, (1 << 64) - 1)


def reference(cond, ty, x, y):
    if ty == "i64":
        x, y = (v - (1 << 64) if v >> 63 else v for v in (x, y))
    return {"eq": x == y, "ne": x != y, "lt": x < y, "le": x <= y,
            "gt": x > y, "ge": x >= y}[cond]


def run_compare(cond, ty, form, cmpx, sources, value=0, inactive=False):
    op = f"v_cmp{'x' if cmpx else ''}_{cond}_{ty}_{form}"
    operands = ", ".join(sources)
    if not cmpx:
        operands = "vcc_lo, " + operands
    lines = [f"v_mov_b32 v2, {value & 0xFFFFFFFF:#x}", f"v_mov_b32 v3, {value >> 32:#x}",
             "s_mov_b32 vcc_lo, -1"]
    if inactive:
        lines.append("s_mov_b32 exec_lo, 0")
    lines += [f"{op} {operands}", "v_cndmask_b32 v0, 0, 1, vcc_lo",
              "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]
    return si.run_fragment(support.translate_lines("ps_vcmp64_null", lines)).get("%out_mrt0")


class CompareNullSources(unittest.TestCase):
    def check_sources(self, cmpx):
        for cond, ty, (form, slot), value in itertools.product(
                CONDITIONS, ("u64", "i64"), (("e32", 0), ("e64", 0), ("e64", 1)), VALUES):
            with self.subTest(cond=cond, ty=ty, form=form, slot=slot, value=hex(value)):
                sources = ["v[2:3]", "v[2:3]"]
                sources[slot] = "null"
                x, y = (0, value) if slot == 0 else (value, 0)
                expected = reference(cond, ty, x, y)
                result = run_compare(cond, ty, form, cmpx, sources, value)
                if cmpx:
                    self.assertEqual(result, (1, 1, 1, 1) if expected else None)
                else:
                    self.assertEqual(result, (int(expected),) * 4)

    def test_vcmp_sources(self):
        self.check_sources(False)

    def test_vcmpx_sources(self):
        self.check_sources(True)

    def test_vcmpx_mixed_lanes_preserve_vcc(self):
        lanes = 32
        vcc = 0x96A53C69
        fssm = bytearray(0x148)
        struct.pack_into("<4I", fssm, 0x138, 0x8088, lanes, 1, 1)
        for ty, slot in itertools.product(("u64", "i64"), (0, 1)):
            with self.subTest(ty=ty, slot=slot):
                cond = ("lt" if slot == 0 else "gt") if ty == "u64" else ("gt" if slot == 0 else "lt")
                sources = ["v[2:3]", "v[2:3]"]
                sources[slot] = "null"
                lines = ["v_and_b32 v2, 1, v0",
                         "v_mov_b32 v3, 0" if ty == "u64" else "v_lshlrev_b32 v3, 31, v2",
                         f"s_mov_b32 vcc_lo, {vcc:#x}",
                         f"v_cmpx_{cond}_{ty}_e64 {', '.join(sources)}",
                         "s_mov_b32 s8, exec_lo", "s_mov_b32 s9, vcc_lo",
                         "s_mov_b32 exec_lo, -1",
                         "v_cndmask_b32_e64 v4, 0, 1, s8", "v_cndmask_b32_e64 v5, 0, 1, s9",
                         "v_lshlrev_b32 v6, 3, v0",
                         "buffer_store_dword v4, v6, s[0:3], 0 offen",
                         "buffer_store_dword v5, v6, s[0:3], 4 offen", "s_endpgm"]
                spv = support.translate_lines("cs_vcmp64_null_masks", lines, fssm=bytes(fssm))
                got = si.run_workgroup(
                    spv, lanes, 32,
                    inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
                    push_constants=[0, 0, 0x1000, 0x31016FAC] + [0] * 28, memory={})
                values = [(i & 1) if ty == "u64" else ((1 << 63) + 1 if i & 1 else 0)
                          for i in range(lanes)]
                operands = [(0, value) if slot == 0 else (value, 0) for value in values]
                mask = sum(1 << i for i, (x, y) in enumerate(operands) if reference(cond, ty, x, y))
                self.assertNotIn(mask, (0, (1 << lanes) - 1, vcc))
                expected = {8 * i + offset: value for i in range(lanes)
                            for offset, value in ((0, (mask >> i) & 1), (4, (vcc >> i) & 1))}
                self.assertEqual(got, expected)

    def test_both_null(self):
        for cond, ty, cmpx in itertools.product(CONDITIONS, ("u64", "i64"), (False, True)):
            with self.subTest(cond=cond, ty=ty, cmpx=cmpx):
                expected = reference(cond, ty, 0, 0)
                result = run_compare(cond, ty, "e64", cmpx, ["null", "null"])
                self.assertEqual(result, ((1,) * 4 if expected else None) if cmpx else (int(expected),) * 4)

    def test_inactive_lane_stays_inactive(self):
        for cond, ty, cmpx in itertools.product(CONDITIONS, ("u64", "i64"), (False, True)):
            with self.subTest(cond=cond, ty=ty, cmpx=cmpx):
                self.assertIsNone(run_compare(cond, ty, "e64", cmpx, ["null", "v[2:3]"], inactive=True))

    def test_non_pair_register_stays_unsupported(self):
        for cond, cmpx in itertools.product(CONDITIONS, (False, True)):
            with self.subTest(cond=cond, cmpx=cmpx):
                with self.assertRaises(support.rdna2spirv.Unsupported):
                    run_compare(cond, "u64", "e64", cmpx, ["null", "v2"])


if __name__ == "__main__":
    unittest.main()
