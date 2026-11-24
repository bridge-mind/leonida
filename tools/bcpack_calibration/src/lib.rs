// BCPack modes 4 through 7. A cell carries a left-aligned value and the number
// of bits established by an archive observation or an earlier decoded block.
use std::slice;

mod blocks;
mod model;

#[repr(C)]
#[derive(Copy, Clone, Default)]
struct Cell {
    value: u8,
    precision: u8,
}
type History = [Cell; 56];

fn spread(n: u8) -> usize {
    (0..4).map(|k| ((n as usize >> k) & 1) << (2 * k)).sum()
}
fn zig(n: u8) -> i32 {
    (n as i32 >> 1) ^ -((n as i32) & 1)
}
fn add(c: Cell, b: u8, delta: i32) -> Cell {
    let shift = 8 - b;
    let v = (((c.value as i32 >> shift) + delta) & ((1 << b) - 1)) as u8;
    if c.precision >= b {
        return Cell {
            value: v << shift,
            precision: 8,
        };
    }
    let range = (1u16 << (b - c.precision)) - 1;
    let end = v as u16 + range;
    if end >= (1u16 << b) {
        return Cell::default();
    }
    let last = (end as u8) << shift;
    let first = v << shift;
    let precision = (first ^ last).leading_zeros() as u8;
    let mask = if precision == 0 {
        0
    } else {
        255u8 << (8 - precision)
    };
    Cell {
        value: first & mask,
        precision,
    }
}
fn part(code: u8, parity: u8, bits: u8) -> u8 {
    (0..bits)
        .map(|k| ((code >> (2 * k + parity)) & 1) << k)
        .sum()
}
fn put(out: &mut u128, at: &mut u32, value: u8, bits: u8) -> bool {
    if bits == 0 {
        return value == 0;
    }
    if *at + bits as u32 > 128 {
        return false;
    }
    *out |= (value as u128 & ((1u128 << bits) - 1)) << *at;
    *at += bits as u32;
    (value as u16) < (1u16 << bits)
}

#[no_mangle]
pub unsafe extern "C" fn rebuild(
    input: *const u8,
    n: usize,
    table: *const i16,
    seed: *const u8,
    anchor: *const u16,
    bytes: *mut u8,
    ready: *mut u8,
) {
    let sy = slice::from_raw_parts(input, n * 32);
    let labels = slice::from_raw_parts(table, 1024 * 4);
    let seeds = slice::from_raw_parts(seed as *const History, 256);
    let anchors = slice::from_raw_parts(anchor, 128);
    let out = slice::from_raw_parts_mut(bytes, n * 16);
    let known = slice::from_raw_parts_mut(ready, n);
    let mut history = [[Cell::default(); 56]; 256];
    let zero = [Cell {
        value: 0,
        precision: 8,
    }; 56];
    for i in 0..n {
        let s = &sy[i * 32..i * 32 + 32];
        let label = &labels[4 * (32 * s[15] as usize + s[31] as usize)..];
        let mode = label[0];
        let mut cur = [Cell::default(); 56];
        known[i] = 0;
        if !(0..=7).contains(&mode) || !(0..=3).contains(&label[2]) || !(0..64).contains(&label[1])
        {
            history[i % 256] = cur;
            continue;
        }
        if (label[2] != 0 && !matches!(mode, 4 | 5)) || (mode == 0 && label[1] >= 16) {
            history[i % 256] = cur;
            continue;
        }
        let rot = label[2] as usize;
        let partition = label[1] as usize;
        let di = spread(s[1]) + 2 * spread(s[17]);
        let de = ((di as i32 + zig((spread(s[0]) + 2 * spread(s[16])) as u8)) & 255) as usize;
        let reference = |d: usize| -> &History {
            if d == 0 {
                &zero
            } else if d > i {
                &seeds[d - i]
            } else {
                &history[(i - d) % 256]
            }
        };
        let e = reference(de);
        let w = reference(di);
        let (mut bits, planes): ([u8; 4], [u8; 2]) = match mode {
            0 => ([5, 4, 4, 0], [0, 3]),
            1 => ([7, 6, 6, 0], [0, 3]),
            2 => ([5, 5, 5, 0], [0, 2]),
            3 => ([8, 7, 7, 0], [0, 2]),
            4 => ([5, 5, 5, 6], [2, 3]),
            5 => ([7, 7, 7, 8], [2, 2]),
            6 => ([8, 7, 7, 7], [0, 4]),
            _ => ([6, 5, 5, 5], [0, 2]),
        };
        if rot > 0 {
            bits.swap(rot - 1, 3)
        }
        let count = [6, 4, 6, 4, 2, 2, 2, 4][mode as usize];
        let mut ep = [[0u8; 4]; 6];
        let mut ok = true;
        for j in 0..count {
            for c in 0..4 {
                let z = j * 4 + c;
                let b = bits[c];
                if b == 0 {
                    cur[z] = zero[0];
                    continue;
                }
                let lane = [[2, 3, 4, 5], [6, 7, 8, 9], [10, 5, 9, 0]][j / 2][c];
                cur[z] = add(e[z], b, zig(s[(j % 2) * 16 + lane]));
                ok &= cur[z].precision >= b;
                ep[j][c] = cur[z].value >> (8 - b);
            }
            if rot > 0 {
                ep[j].swap(rot - 1, 3)
            }
        }
        if count == 2 {
            cur.copy_within(0..8, 8)
        }
        if count < 6 {
            cur.copy_within(0..8, 16)
        }
        let mut indices = [[0u8; 16]; 2];
        for (plane, &b) in planes.iter().enumerate() {
            if b == 0 {
                for cell in &mut cur[24..40] {
                    *cell = zero[0]
                }
                continue;
            }
            let lanes = if plane == 0 {
                [6, 7, 8, 10]
            } else {
                [11, 12, 13, 14]
            };
            for (row, &lane) in lanes.iter().enumerate() {
                for half in 0..2 {
                    let code = s[half * 16 + lane];
                    for k in 0..2 {
                        let parity = if row == 3 { k } else { 1 - k };
                        let p = row * 4 + half * 2 + k;
                        let at = 24 + 16 * plane + p;
                        cur[at] = add(w[at], b, zig(part(code, parity as u8, b)));
                        ok &= cur[at].precision >= b;
                        indices[plane][p] = cur[at].value >> (8 - b);
                    }
                }
            }
        }
        history[i % 256] = cur;
        if !ok || label[1] < 0 {
            continue;
        }
        let (mut data, mut at) = (1u128 << mode, (mode + 1) as u32);
        if count > 2 {
            put(
                &mut data,
                &mut at,
                partition as u8,
                if mode == 0 { 4 } else { 6 },
            );
        }
        if mode == 4 || mode == 5 {
            put(&mut data, &mut at, rot as u8, 2);
        }
        if mode == 4 {
            put(&mut data, &mut at, label[3] as u8, 1);
        }
        if rot > 0 {
            bits.swap(rot - 1, 3)
        }
        let has_p = matches!(mode, 0 | 1 | 3 | 6 | 7);
        for c in 0..4 {
            for e in &ep[..count] {
                let drop = (has_p && c == 0) as u8;
                put(&mut data, &mut at, e[c] >> drop, bits[c] - drop);
            }
        }
        if has_p {
            for j in 0..count {
                if mode == 1 && j % 2 == 1 {
                    ok &= (ep[j][0] & 1) == (ep[j - 1][0] & 1);
                    continue;
                }
                put(&mut data, &mut at, ep[j][0] & 1, 1);
            }
        }
        let mask = if count > 2 {
            anchors[(count / 2 - 2) * 64 + partition]
        } else {
            1
        };
        for wire in 0..2 {
            let plane = if mode == 5 { 1 - wire } else { wire };
            let b = planes[plane];
            if b > 0 {
                for p in 0..16 {
                    ok &= put(
                        &mut data,
                        &mut at,
                        indices[plane][p],
                        b - ((mask >> p) & 1) as u8,
                    );
                }
            }
        }
        if ok && at == 128 {
            known[i] = 1;
            out[i * 16..i * 16 + 16].copy_from_slice(&data.to_le_bytes());
        }
    }
}

// Python owns the byte input and the target-by-lanes uint8 output buffer.
#[no_mangle]
pub unsafe extern "C" fn unpack(
    data: *const u8,
    size: usize,
    widths: *const u8,
    lanes: usize,
    target: usize,
    output: *mut u8,
) -> isize {
    let stream = slice::from_raw_parts(data, size);
    let bits = slice::from_raw_parts(widths, lanes);
    if bits.iter().any(|&b| b > 7) {
        return -1;
    }
    let Ok(decoded) = bcpack_rans::rans::decode_slice(stream, bits, target) else {
        return -1;
    };
    let out = slice::from_raw_parts_mut(output, target * lanes);
    for lane in 0..lanes {
        for (i, &v) in decoded.lane(lane).iter().enumerate() {
            out[i * lanes + lane] = v as u8;
        }
    }
    decoded.consumed as isize
}

// The lane-array model of tools/bcpack.py: same streams accepted, same symbols.
// `tables` is the flat i32 table of `model::Tables::new`; `output` receives
// `lanes * target` u16 symbols, lane-major. Returns the bytes used, or -1.
#[no_mangle]
pub unsafe extern "C" fn unpack_model(
    data: *const u8,
    size: usize,
    widths: *const u8,
    lanes: usize,
    target: usize,
    tables: *const i32,
    tables_len: usize,
    output: *mut u16,
) -> isize {
    let stream = slice::from_raw_parts(data, size);
    let bits = slice::from_raw_parts(widths, lanes);
    let Some(t) = model::Tables::new(slice::from_raw_parts(tables, tables_len)) else {
        return -1;
    };
    let Some((symbols, used)) = model::decode(stream, bits, target, &t) else {
        return -1;
    };
    if stream[used..].iter().any(|&b| b != 0) {
        return -1;
    }
    slice::from_raw_parts_mut(output, lanes * target).copy_from_slice(&symbols);
    used as isize
}

// Lanes and the strict block rebuild of codec 1 (BC1), 3 (BC3), 4 (BC4) or 5
// (BC5) in one call. `lut` is BC1_QUADRANT, the flattened BC3_ALPHA_PAIR then
// BC3_COLOUR_QUADRANT, the flattened BC4_PAIR or the flattened BC5_PAIR,
// `builtin` the confirmed built-in history (rows of 6, 8, 2 or 4 values, -1 where absent). Returns the bytes
// used, or -1: then Python decodes the chunk itself.
#[no_mangle]
pub unsafe extern "C" fn decode_blocks(
    data: *const u8,
    size: usize,
    codec: u32,
    widths: *const u8,
    lanes: usize,
    target: usize,
    tables: *const i32,
    tables_len: usize,
    lut: *const u8,
    lut_len: usize,
    builtin: *const i32,
    builtin_len: usize,
    output: *mut u8,
    output_len: usize,
) -> isize {
    if lanes != 12 {
        return -1;
    }
    let stream = slice::from_raw_parts(data, size);
    let bits = slice::from_raw_parts(widths, lanes);
    let Some(t) = model::Tables::new(slice::from_raw_parts(tables, tables_len)) else {
        return -1;
    };
    let Some((symbols, used)) = model::decode(stream, bits, target, &t) else {
        return -1;
    };
    if stream[used..].iter().any(|&b| b != 0) {
        return -1;
    }
    let lut = slice::from_raw_parts(lut, lut_len);
    let builtin = slice::from_raw_parts(builtin, builtin_len);
    let out = slice::from_raw_parts_mut(output, output_len);
    let done = match codec {
        1 => blocks::bc1(&symbols, target, lut, builtin, out),
        3 => blocks::bc3(&symbols, target, lut, builtin, out),
        4 => blocks::bc4(&symbols, target, lut, builtin, out),
        5 => blocks::bc5(&symbols, target, lut, builtin, out),
        _ => None,
    };
    if done.is_some() {
        used as isize
    } else {
        -1
    }
}
