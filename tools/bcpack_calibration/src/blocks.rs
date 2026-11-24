// Strict BC1, BC3, BC4 and BC5 block rebuild, ported from `_bc1_blocks`,
// `_bc3_blocks`, `_bc4_blocks` and `_bc5_blocks` in tools/bcpack.py. Python passes the lookup table and the confirmed built-in
// history it would use. Anything the strict path would not decode on its own
// (a non-zero lane 1, a history entry that is not confirmed) returns None:
// Python then runs the whole chunk again, so its errors and its provisional
// entries are unchanged.

fn dist(v: u16) -> i64 {
    let v = v as i64;
    if v & 1 != 0 {
        256 - ((v + 1) >> 1)
    } else {
        v >> 1
    }
}

fn unzig(v: u16) -> i64 {
    let v = v as i64;
    if v & 1 != 0 {
        -((v + 1) >> 1)
    } else {
        v >> 1
    }
}

/// The base of block `i`: the zero state, an earlier block, or a confirmed built-in entry.
fn base<const K: usize>(
    i: usize,
    d: i64,
    zero: [i64; K],
    hist: &[[i64; K]],
    builtin: &[i32],
) -> Option<[i64; K]> {
    if d == 0 {
        return Some(zero);
    }
    if i as i64 - d >= 0 {
        return Some(hist[i - d as usize]);
    }
    let entry = (256 + i as i64 - d) as usize;
    let row = builtin.get(entry * K..entry * K + K)?;
    if row[0] < 0 {
        return None;
    }
    let mut b = [0i64; K];
    for k in 0..K {
        b[k] = row[k] as i64;
    }
    Some(b)
}

/// `lane(l)[i]` is symbol i of lane l (lane-major, `n` symbols per lane).
pub fn bc1(sym: &[u16], n: usize, quad: &[u8], builtin: &[i32], out: &mut [u8]) -> Option<()> {
    let lane = |l: usize| &sym[l * n..(l + 1) * n];
    if quad.len() != 256 || builtin.len() != 256 * 6 || out.len() != n * 8 || lane(1).iter().any(|&s| s != 0) {
        return None;
    }
    const MODS: [i64; 6] = [32, 64, 32, 32, 64, 32];
    let mut hist = vec![[0i64; 6]; n];
    for i in 0..n {
        let b = base(i, dist(lane(0)[i]), [0; 6], &hist, builtin)?;
        let mut e = [0i64; 6];
        for k in 0..6 {
            e[k] = (b[k] + unzig(lane(2 + k)[i])).rem_euclid(MODS[k]);
        }
        hist[i] = e;
        let c0 = ((e[0] << 11) | (e[1] << 5) | e[2]) as u16;
        let c1 = ((e[3] << 11) | (e[4] << 5) | e[5]) as u16;
        let q = |l: usize| *quad.get(lane(l)[i] as usize).unwrap_or(&0);
        if (8..12).any(|l| lane(l)[i] > 255) {
            return None;
        }
        let (a, bb, c, dd) = (q(8), q(9), q(10), q(11));
        let o = &mut out[i * 8..i * 8 + 8];
        o[0..2].copy_from_slice(&c0.to_le_bytes());
        o[2..4].copy_from_slice(&c1.to_le_bytes());
        o[4] = (a & 15) | ((bb & 15) << 4);
        o[5] = (a >> 4) | ((bb >> 4) << 4);
        o[6] = (c & 15) | ((dd & 15) << 4);
        o[7] = (c >> 4) | ((dd >> 4) << 4);
    }
    Some(())
}

/// `pairs` is `BC4_PAIR` flattened: 64 rows of two 3-bit indices.
pub fn bc4(sym: &[u16], n: usize, pairs: &[u8], builtin: &[i32], out: &mut [u8]) -> Option<()> {
    let lane = |l: usize| &sym[l * n..(l + 1) * n];
    if pairs.len() != 128 || builtin.len() != 256 * 2 || out.len() != n * 8 || lane(1).iter().any(|&s| s != 0) {
        return None;
    }
    let mut hist = vec![[0i64; 2]; n];
    for i in 0..n {
        let b = base(i, dist(lane(0)[i]), [128; 2], &hist, builtin)?;
        let a0 = (b[0] + unzig(lane(2)[i])) & 255;
        let a1 = (b[1] + unzig(lane(3)[i])) & 255;
        hist[i] = [a0, a1];
        let mut bits = 0u64;
        for r in 0..4 {
            let (l, rr) = (lane(4 + 2 * r)[i] as usize, lane(5 + 2 * r)[i] as usize);
            if l >= 64 || rr >= 64 {
                return None;
            }
            let v = pairs[2 * l] as u64
                | (pairs[2 * l + 1] as u64) << 3
                | (pairs[2 * rr] as u64) << 6
                | (pairs[2 * rr + 1] as u64) << 9;
            bits |= v << (12 * r);
        }
        let o = &mut out[i * 8..i * 8 + 8];
        o[0] = a0 as u8;
        o[1] = a1 as u8;
        o[2..8].copy_from_slice(&bits.to_le_bytes()[..6]);
    }
    Some(())
}

/// The 8-bit history distance of the two-element codecs: lane 0 carries the low
/// four bits of the distance code in element `a` and the high four in `b`,
/// interleaved bit by bit, then zigzag decoded modulo 256 (`_inter8`).
fn dist8(a: u16, b: u16) -> i64 {
    let mut code = 0u16;
    for k in 0..4 {
        code |= (((a >> k) & 1) << (2 * k)) | (((b >> k) & 1) << (2 * k + 1));
    }
    unzig(code) & 255
}

/// Three index bits of each element of a pair, from a symbol of `pairs` (64 rows of two 3-bit values).
fn pair_bits(pairs: &[u8], left: u16, right: u16) -> Option<u64> {
    let (l, r) = (left as usize, right as usize);
    if l >= 64 || r >= 64 {
        return None;
    }
    Some(
        pairs[2 * l] as u64
            | (pairs[2 * l + 1] as u64) << 3
            | (pairs[2 * r] as u64) << 6
            | (pairs[2 * r + 1] as u64) << 9,
    )
}

/// `lut` is `BC3_ALPHA_PAIR` flattened (64 rows of two 3-bit indices) followed by `BC3_COLOUR_QUADRANT` (256 bytes).
/// BC3 and BC5 carry two symbols of every lane per block, so `n` is twice the block count.
pub fn bc3(sym: &[u16], n: usize, lut: &[u8], builtin: &[i32], out: &mut [u8]) -> Option<()> {
    let lane = |l: usize| &sym[l * n..(l + 1) * n];
    if n % 2 != 0 || lut.len() != 128 + 256 || builtin.len() != 256 * 8 || out.len() != n * 8 || lane(1).iter().any(|&s| s != 0) {
        return None;
    }
    const MODS: [i64; 8] = [256, 256, 32, 64, 32, 32, 64, 32];
    let (pairs, quads) = lut.split_at(128);
    let mut hist = vec![[0i64; 8]; n / 2];
    for i in 0..n / 2 {
        let (a, b) = (2 * i, 2 * i + 1);
        let base = base(i, dist8(lane(0)[a], lane(0)[b]), [128, 128, 0, 0, 0, 0, 0, 0], &hist, builtin)?;
        let dz = [
            lane(2)[a], lane(2)[b], lane(7)[a], lane(8)[a], lane(9)[a], lane(7)[b], lane(8)[b], lane(9)[b],
        ];
        let mut e = [0i64; 8];
        for k in 0..8 {
            e[k] = (base[k] + unzig(dz[k])).rem_euclid(MODS[k]);
        }
        hist[i] = e;
        let mut bits = 0u64;
        for r in 0..4 {
            bits |= pair_bits(pairs, lane(3 + r)[a], lane(3 + r)[b])? << (12 * r);
        }
        if (10..12).any(|l| lane(l)[a] > 255 || lane(l)[b] > 255) {
            return None;
        }
        let q = |l: usize, at: usize| quads[lane(l)[at] as usize];
        let (qa, qb, qc, qd) = (q(10, a), q(10, b), q(11, a), q(11, b));
        let c0 = ((e[2] << 11) | (e[3] << 5) | e[4]) as u16;
        let c1 = ((e[5] << 11) | (e[6] << 5) | e[7]) as u16;
        let o = &mut out[i * 16..i * 16 + 16];
        o[0] = e[0] as u8;
        o[1] = e[1] as u8;
        o[2..8].copy_from_slice(&bits.to_le_bytes()[..6]);
        o[8..10].copy_from_slice(&c0.to_le_bytes());
        o[10..12].copy_from_slice(&c1.to_le_bytes());
        o[12] = (qa & 15) | ((qb & 15) << 4);
        o[13] = (qa >> 4) | ((qb >> 4) << 4);
        o[14] = (qc & 15) | ((qd & 15) << 4);
        o[15] = (qc >> 4) | ((qd >> 4) << 4);
    }
    Some(())
}

/// `pairs` is `BC5_PAIR` flattened: 64 rows of two 3-bit indices.
pub fn bc5(sym: &[u16], n: usize, pairs: &[u8], builtin: &[i32], out: &mut [u8]) -> Option<()> {
    let lane = |l: usize| &sym[l * n..(l + 1) * n];
    if n % 2 != 0 || pairs.len() != 128 || builtin.len() != 256 * 4 || out.len() != n * 8 || lane(1).iter().any(|&s| s != 0) {
        return None;
    }
    let mut hist = vec![[0i64; 4]; n / 2];
    for i in 0..n / 2 {
        let (a, b) = (2 * i, 2 * i + 1);
        let base = base(i, dist8(lane(0)[a], lane(0)[b]), [128; 4], &hist, builtin)?;
        let e = [
            (base[0] + unzig(lane(2)[a])) & 255,
            (base[1] + unzig(lane(2)[b])) & 255,
            (base[2] + unzig(lane(3)[a])) & 255,
            (base[3] + unzig(lane(3)[b])) & 255,
        ];
        hist[i] = e;
        let o = &mut out[i * 16..i * 16 + 16];
        for ch in 0..2 {
            let mut bits = 0u64;
            for r in 0..4 {
                bits |= pair_bits(pairs, lane(4 + 4 * ch + r)[a], lane(4 + 4 * ch + r)[b])? << (12 * r);
            }
            o[8 * ch] = e[2 * ch] as u8;
            o[8 * ch + 1] = e[2 * ch + 1] as u8;
            o[8 * ch + 2..8 * ch + 8].copy_from_slice(&bits.to_le_bytes()[..6]);
        }
    }
    Some(())
}
