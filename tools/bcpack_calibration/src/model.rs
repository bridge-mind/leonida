// A line-by-line port of `_decode_lanes_raw` in tools/bcpack.py (the lane-array
// model of docs/textures.md, with its virtual zero bytes), so Python and Rust
// accept the same streams. Python passes its own probability tables in. The
// state is an i128 because the Python model works on unbounded integers: a
// fragment that lands below zero is shifted left and compared, never wrapped.
// Any refusal returns None; Python then decodes the stream again and raises its
// own message.
const RANS_L: i128 = 1 << 24;
const MAX_WIDTH: usize = 10;
const PREFETCH_BUDGET: u32 = 8;
const MAX_VIRTUAL_ZEROS: u32 = 8;
const HDR0: u8 = 0;
const HDR1: u8 = 1;
const HDR2: u8 = 2;
const PROC: u8 = 3;

/// `freq[w][q]` (bands 0..=w) and its running `base`, or None where Python has no table.
pub struct Tables {
    models: Vec<Option<([i128; MAX_WIDTH + 1], [i128; MAX_WIDTH + 1])>>,
}

impl Tables {
    /// `flat[(w * 16 + q) * 11 + p]` for w in 0..=10; a table whose first entry is -1 is absent.
    pub fn new(flat: &[i32]) -> Option<Tables> {
        if flat.len() != (MAX_WIDTH + 1) * 16 * (MAX_WIDTH + 1) {
            return None;
        }
        let mut models = Vec::with_capacity((MAX_WIDTH + 1) * 16);
        for w in 0..=MAX_WIDTH {
            for q in 0..16 {
                let row = &flat[(w * 16 + q) * (MAX_WIDTH + 1)..][..MAX_WIDTH + 1];
                if row[0] < 0 {
                    models.push(None);
                    continue;
                }
                let mut freq = [0i128; MAX_WIDTH + 1];
                let mut base = [0i128; MAX_WIDTH + 1];
                let mut acc = 0i128;
                for p in 0..=w {
                    freq[p] = row[p] as i128;
                    base[p] = acc;
                    acc += freq[p] * if p == 0 { 1 } else { 1 << (p - 1) };
                }
                if acc != 65536 {
                    return None;
                }
                models.push(Some((freq, base)));
            }
        }
        Some(Tables { models })
    }

    fn get(&self, w: usize, q: usize) -> Option<usize> {
        self.models[w * 16 + q].as_ref().map(|_| w * 16 + q)
    }

    fn lookup(&self, m: usize, cf: i128) -> Option<i128> {
        let (freq, base) = self.models[m].as_ref().unwrap();
        let mut p = m / 16;
        while p > 0 && cf < base[p] {
            p -= 1;
        }
        if p == 0 {
            return Some(0);
        }
        if freq[p] == 0 {
            return None;
        }
        Some((1i128 << (p - 1)) + (cf - base[p]).div_euclid(freq[p]))
    }

    fn advance(&self, m: usize, s: i128, x: i128) -> i128 {
        let (freq, base) = self.models[m].as_ref().unwrap();
        let p = (128 - s.leading_zeros()) as usize;
        let mut f = freq[p];
        let mut c = base[p] + f * (s - if p == 0 { 0 } else { 1 << (p - 1) });
        if c == 0 {
            f -= 1;
            c = 1;
        }
        f * (x >> 16) + (x & 0xFFFF) - c
    }
}

/// `(symbols lane-major, bytes used)`, or None where the Python model raises.
pub fn decode(raw: &[u8], widths: &[u8], nsym: usize, t: &Tables) -> Option<(Vec<u16>, usize)> {
    let nl = widths.len();
    let mut ctrl = vec![HDR0; nl];
    let mut out_rem = vec![nsym; nl];
    let mut w = vec![0usize; nl];
    let mut q = vec![0usize; nl];
    let mut h3 = vec![false; nl];
    let mut flush = vec![false; nl];
    let mut done = vec![nsym == 0; nl];
    let mut full = vec![false; nl];
    let mut held = vec![false; nl];
    let mut vz = vec![0u32; nl];
    let mut inbuf = vec![0i128; nl];
    let mut sym = vec![0i128; nl];
    let mut in_rem = vec![0i128; nl];
    let mut p0 = vec![0i128; nl];
    let mut p1 = vec![0i128; nl];
    let mut model = vec![usize::MAX; nl];
    let mut slotfull = vec![false; nl];
    let mut slotval = vec![0i128; nl];
    let mut stall = vec![false; nl];
    let mut out: Vec<Vec<u16>> = vec![Vec::with_capacity(nsym); nl];
    let mut pos = 0usize;
    // The Python model has no cycle limit. A stream on which no lane changes
    // for this many cycles is handed back to Python unchanged.
    let mut idle = 0u32;

    loop {
        if (0..nl).all(|i| done[i] && !slotfull[i]) {
            let mut flat = Vec::with_capacity(nl * nsym);
            for lane in &out {
                if lane.len() != nsym {
                    return None;
                }
                flat.extend_from_slice(lane);
            }
            return Some((flat, pos));
        }
        let before = (pos, out.iter().map(Vec::len).sum::<usize>());
        let mut moved = false;

        let mut left = PREFETCH_BUDGET;
        for i in 0..nl {
            if done[i] {
                stall[i] = true;
                continue;
            }
            if held[i] && slotfull[i] {
                stall[i] = true;
                continue;
            }
            stall[i] = false;
            if ctrl[i] == PROC {
                p0[i] = if held[i] { t.advance(model[i], sym[i], p1[i]) } else { p1[i] };
            }
            if !full[i] && (ctrl[i] != PROC || in_rem[i] > 0) {
                if left == 0 {
                    stall[i] = true;
                    continue;
                }
                if pos >= raw.len() {
                    return None;
                }
                inbuf[i] = raw[pos] as i128;
                pos += 1;
                full[i] = true;
                left -= 1;
            }
            if held[i] {
                slotfull[i] = true;
                slotval[i] = sym[i];
                held[i] = false;
                moved = true;
            }
        }

        for i in 0..nl {
            if done[i] || stall[i] {
                continue;
            }
            match ctrl[i] {
                HDR0 => {
                    if !full[i] {
                        continue;
                    }
                    let h = inbuf[i];
                    q[i] = (h & 15) as usize;
                    flush[i] = (h >> 4) & 1 != 0;
                    h3[i] = (h >> 5) & 1 != 0;
                    let wi = widths[i] as i128 - (h >> 6);
                    if !(1..=MAX_WIDTH as i128).contains(&wi) {
                        // Python checks the width at HDR1; it fails there either way.
                        w[i] = usize::MAX;
                    } else {
                        w[i] = wi as usize;
                    }
                    full[i] = false;
                    ctrl[i] = HDR1;
                    vz[i] = 0;
                    moved = true;
                }
                HDR1 => {
                    if !full[i] {
                        continue;
                    }
                    full[i] = false;
                    if w[i] == usize::MAX {
                        return None;
                    }
                    model[i] = t.get(w[i], q[i])?;
                    if h3[i] {
                        in_rem[i] = inbuf[i];
                        ctrl[i] = HDR2;
                    } else {
                        in_rem[i] = inbuf[i] + 1;
                        ctrl[i] = PROC;
                    }
                    moved = true;
                }
                HDR2 => {
                    if !full[i] {
                        continue;
                    }
                    in_rem[i] = ((inbuf[i] << 8) | (in_rem[i] & 0xFF)) + 257;
                    ctrl[i] = PROC;
                    full[i] = false;
                    moved = true;
                }
                _ => {
                    let mut st = p0[i];
                    if st < 0 {
                        // Only a corrupt stream gets here; leave it to Python.
                        return None;
                    }
                    if st < RANS_L {
                        if in_rem[i] > 0 {
                            if full[i] {
                                st = (st << 8) | inbuf[i];
                                in_rem[i] -= 1;
                                full[i] = false;
                                if in_rem[i] == 0 && flush[i] {
                                    while st < RANS_L && st != 0 && vz[i] < MAX_VIRTUAL_ZEROS {
                                        st <<= 8;
                                        vz[i] += 1;
                                    }
                                }
                            }
                        } else if flush[i] {
                            if st == 0 || vz[i] >= MAX_VIRTUAL_ZEROS {
                                return None;
                            }
                            while st < RANS_L && st != 0 && vz[i] < MAX_VIRTUAL_ZEROS {
                                st <<= 8;
                                vz[i] += 1;
                            }
                        } else {
                            p1[i] = st;
                            if out_rem[i] > 0 {
                                ctrl[i] = HDR0;
                            } else {
                                done[i] = true;
                            }
                            moved = true;
                            continue;
                        }
                    }
                    let nxt = st > RANS_L && out_rem[i] > 0;
                    if nxt {
                        let s = t.lookup(model[i], st & 0xFFFF)?;
                        sym[i] = s;
                        out_rem[i] -= 1;
                        if s >> w[i] != 0 {
                            return None;
                        }
                    }
                    if nxt != held[i] || p1[i] != st {
                        moved = true;
                    }
                    held[i] = nxt;
                    p1[i] = st;
                    if in_rem[i] == 0 {
                        if flush[i] {
                            if st == RANS_L {
                                if out_rem[i] > 0 {
                                    ctrl[i] = HDR0;
                                    p0[i] = 0;
                                    p1[i] = 0;
                                } else {
                                    done[i] = true;
                                }
                                moved = true;
                            } else if st > RANS_L && !nxt {
                                return None;
                            }
                        } else if out_rem[i] == 0 && !nxt {
                            done[i] = true;
                            moved = true;
                        }
                    } else if st == RANS_L || (out_rem[i] == 0 && !nxt && st > RANS_L) {
                        return None;
                    }
                }
            }
        }

        let mut ready = true;
        let mut anyfull = false;
        for i in 0..nl {
            if slotfull[i] {
                anyfull = true;
            } else if !(done[i] && !held[i]) {
                ready = false;
                break;
            }
        }
        if ready && anyfull {
            for i in 0..nl {
                if slotfull[i] {
                    out[i].push(slotval[i] as u16);
                    slotfull[i] = false;
                }
            }
        }
        for i in 0..nl {
            if done[i] && held[i] && !slotfull[i] {
                slotfull[i] = true;
                slotval[i] = sym[i];
                held[i] = false;
                moved = true;
            }
        }
        if (pos, out.iter().map(Vec::len).sum::<usize>()) != before || moved {
            idle = 0;
        } else {
            idle += 1;
            if idle > 1 << 16 {
                return None;
            }
        }
    }
}
