//! Multi-lane MDU rANS decoder (entropy layer).
//!
//! Each lane is a cycle-level port of the `decoder` module in Microsoft's
//! MIT-licensed `silicon-rans` reference Verilog (`mdu_minirans.v`): two phases
//! per clock, a one-byte input buffer, a one-symbol output buffer, the
//! HDR0/HDR1/HDR2/PROCESSING/DRAINING control states, the flush bit, the 2-bit
//! alphabet narrowing, 8/16-bit fragment lengths (+1 / +257) and the legal end
//! states `L`, `L >> 8`, `L >> 16`.
//!
//! Two parts are not in the published Verilog and were established on retail
//! data (exact convergence on every lane of 15,000 sampled chunks, and every
//! structural variant fails, see `docs/BCPACK_SPEC.md`):
//!
//! * feeder: in phase 0 of each clock, lanes that request a byte are served
//!   from the single stream in lane order, at most [`FEED_WINDOW`] bytes per
//!   clock; a lane that is not served stalls for that clock;
//! * symbol vector: one slot per lane; a lane may hand over a symbol only while
//!   its slot is empty, and all slots are released together once every lane
//!   that is still active holds a symbol (lockstep).
//!
//! The byte schedule therefore depends only on the lane states, which is what
//! lets [`crate::encode`] build valid interleaved streams for tests.

use crate::tables::{BASE_TABLE, FREQ_TABLE, RF_TABLE, RN_TABLE, RS_TABLE};
use std::fmt;
use std::sync::OnceLock;

/// rANS lower bound (`MDU_RANS_LOWER_LIMIT`).
pub const L: u32 = 1 << 24;
/// Bytes the feeder hands to the lanes per clock (`MDU_RANS_FEED_WINDOW`).
pub const FEED_WINDOW: usize = 8;
/// Maximum lanes per stream supported by this implementation.
pub const MAX_LANES: usize = 16;

/// True for the three legal end states of a lane.
#[inline]
pub fn is_legal_end_state(x: u32) -> bool {
    x == L || x == L >> 8 || x == L >> 16
}

/// One static model (alphabet index `eab` = bits - 1, model `q`).
#[derive(Clone, Debug)]
pub struct Model {
    pub eab: u8,
    pub q: u8,
    base: [u16; 10],
    freq: [u16; 10],
    rf: [u32; 10],
    rs: [u8; 10],
    rn: [u16; 10],
    nseg: u8,
    /// Segment thresholds for the branch-free inverse: `base[k]` for enabled
    /// segments 1..nseg, `0x10000` (never reached) for disabled ones.
    thr: [u32; 10],
    /// Full inverse table (cf -> symbol), built on first use.
    lut: OnceLock<Box<[u16]>>,
}

impl Model {
    fn build(eab: usize, q: usize) -> Model {
        let mut thr = [0x10000u32; 10];
        for k in 1..eab + 2 {
            thr[k] = BASE_TABLE[eab][q][k] as u32;
        }
        Model {
            thr,
            eab: eab as u8,
            q: q as u8,
            base: BASE_TABLE[eab][q],
            freq: FREQ_TABLE[eab][q],
            rf: RF_TABLE[eab][q],
            rs: RS_TABLE[eab][q],
            rn: RN_TABLE[eab][q],
            nseg: (eab + 2) as u8,
            lut: OnceLock::new(),
        }
    }

    /// The 65536-entry inverse table. Filled from the forward intervals
    /// `[cf, cf + f)` of every symbol (cheap), plus `cf = 0` from the exact
    /// path. A unit test checks it equals `inverse_exact` for every cf.
    pub fn lut(&self) -> &[u16] {
        self.lut.get_or_init(|| {
            let mut t = vec![0u16; 65536].into_boxed_slice();
            for s in 0..(1u32 << (self.eab + 1)) {
                let (f, cf) = self.forward(s as u16);
                let end = (cf + f).min(65536);
                for v in &mut t[cf as usize..end as usize] {
                    *v = s as u16;
                }
            }
            t[0] = self.inverse_exact(0);
            t
        })
    }

    /// Symbol for the low 16 state bits, exactly as the Verilog computes it:
    /// comparator chain over enabled segments, then the reciprocal multiply.
    #[inline]
    pub fn inverse_exact(&self, cf: u16) -> u16 {
        let cf = cf as u32;
        let mut seg = 0usize;
        let mut k = self.nseg as usize - 1;
        while k > 0 {
            if cf >= self.base[k] as u32 {
                seg = k;
                break;
            }
            k -= 1;
        }
        let base_x = if seg == 0 { 0 } else { 1u32 << (seg - 1) };
        let dist = (cf
            .wrapping_sub(self.base[seg] as u32)
            .wrapping_add(self.rn[seg] as u32))
            & 0xFFFF;
        let add_mul = dist.wrapping_mul(self.rf[seg]);
        let steps = ((add_mul >> 15) >> self.rs[seg]) & 0x1FF;
        ((base_x + steps) & 0x1FF) as u16
    }

    /// Branch-free inverse: the enabled segment bases are non-decreasing, so
    /// the priority encoder equals the count of thresholds `cf` reaches. A unit
    /// test checks this equals [`Model::inverse_exact`] for every model and cf.
    #[inline(always)]
    pub fn inverse_fast(&self, cf: u16) -> u16 {
        let cf = cf as u32;
        let mut seg = 0usize;
        for k in 1..10 {
            seg += (cf >= self.thr[k]) as usize;
        }
        let base_x = (1u32 << seg) >> 1;
        let dist = (cf
            .wrapping_sub(self.base[seg] as u32)
            .wrapping_add(self.rn[seg] as u32))
            & 0xFFFF;
        let steps = ((dist.wrapping_mul(self.rf[seg]) >> 15) >> self.rs[seg]) & 0x1FF;
        ((base_x + steps) & 0x1FF) as u16
    }

    /// Inverse through the table. Same result as [`Model::inverse_exact`].
    #[inline]
    pub fn inverse(&self, cf: u16) -> u16 {
        self.lut()[cf as usize]
    }

    /// `(freq, cumulative freq)` of `sym`. Slot 0 is reserved: a symbol whose
    /// cumulative frequency would be 0 gets `(f - 1, 1)`.
    #[inline]
    pub fn forward(&self, sym: u16) -> (u32, u32) {
        let s = sym as u32;
        let seg = (32 - s.leading_zeros()) as usize;
        let st = if seg == 0 { 0 } else { 1u32 << (seg - 1) };
        let fa = self.freq[seg] as u32;
        let cfa = (self.base[seg] as u32).wrapping_add(fa.wrapping_mul(s - st)) & 0xFFFF;
        if cfa == 0 {
            (fa.wrapping_sub(1) & 0xFFFF, 1)
        } else {
            (fa, cfa)
        }
    }

    /// State after decoding `sym` from state `x`.
    #[inline]
    pub fn advance(&self, x: u32, sym: u16) -> u32 {
        let (f, cf) = self.forward(sym);
        f.wrapping_mul(x >> 16)
            .wrapping_add(x & 0xFFFF)
            .wrapping_sub(cf)
    }
}

static MODELS: OnceLock<Vec<Model>> = OnceLock::new();

/// The 9 x 16 models, built once.
pub fn model(eab: u8, q: u8) -> &'static Model {
    let all = MODELS.get_or_init(|| {
        let mut v = Vec::with_capacity(144);
        for e in 0..9 {
            for q in 0..16 {
                v.push(Model::build(e, q));
            }
        }
        v
    });
    &all[eab as usize * 16 + q as usize]
}

/// Why a stream failed to decode.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DecodeError {
    /// A fragment header selects an alphabet index above 8 (undefined in the Verilog).
    BadHeader { lane: usize, eab: u8 },
    /// No lane made progress for 64 clocks: the stream ran out or is not this format.
    NoProgress { consumed: usize },
    /// The cycle budget ran out.
    CycleLimit,
    /// Every lane finished but these lanes ended outside the legal end states.
    IllegalEndState { lanes: Vec<usize> },
    /// Bytes after the last consumed byte are not all zero padding.
    TrailingData { consumed: usize, len: usize },
    /// Invalid arguments (lane count, block count).
    Invalid(String),
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecodeError::BadHeader { lane, eab } => {
                write!(
                    f,
                    "lane {lane}: fragment header selects alphabet index {eab} (> 8)"
                )
            }
            DecodeError::NoProgress { consumed } => {
                write!(
                    f,
                    "no progress after {consumed} bytes (truncated stream or not BCPack BC1)"
                )
            }
            DecodeError::CycleLimit => write!(f, "cycle limit reached"),
            DecodeError::IllegalEndState { lanes } => {
                write!(f, "lanes {lanes:?} ended in an illegal rANS state")
            }
            DecodeError::TrailingData { consumed, len } => {
                write!(
                    f,
                    "stream ended after {consumed} of {len} bytes but the rest is not zero padding"
                )
            }
            DecodeError::Invalid(s) => write!(f, "{s}"),
        }
    }
}

impl std::error::Error for DecodeError {}

/// Where lane bytes come from.
pub trait ByteSource {
    /// Next byte for `lane`, or `None` if the stream is exhausted.
    fn next(&mut self, lane: usize) -> Option<u8>;
}

/// The normal case: one shared stream.
pub struct SliceSource<'a> {
    pub data: &'a [u8],
    pub pos: usize,
}

impl ByteSource for SliceSource<'_> {
    #[inline]
    fn next(&mut self, _lane: usize) -> Option<u8> {
        let b = *self.data.get(self.pos)?;
        self.pos += 1;
        Some(b)
    }
}

const IDLE: u8 = 0;
const HDR0: u8 = 1;
const HDR1: u8 = 2;
const HDR2: u8 = 3;
const PROC: u8 = 4;
const DRAIN: u8 = 5;

#[derive(Clone)]
struct Lane {
    ab: u8,
    target: u32,
    state: u8,
    ib: u8,
    ib_full: bool,
    sym: u16,
    sym_full: bool,
    in_rem: u32,
    out_rem: u32,
    p0: u32,
    p1: u32,
    hdr3: bool,
    flush: bool,
    q: u8,
    eab: u8,
    model: &'static Model,
    done: bool,
    go: bool,
    stall: bool,
    nsyms: u32,
    nbytes: u32,
    nfrags: u32,
    final_state: u32,
}

impl Lane {
    #[inline]
    fn need_byte(&self) -> bool {
        !self.ib_full
            && match self.state {
                HDR0 | HDR1 | HDR2 => true,
                PROC => self.in_rem > 0,
                _ => false,
            }
    }
}

/// Per-lane statistics of a successful decode.
#[derive(Clone, Debug, Default)]
pub struct LaneStats {
    pub bytes: u32,
    pub fragments: u32,
    pub final_state: u32,
}

/// Result of [`decode_lanes`]: `symbols[lane * target + i]` (lane major).
#[derive(Clone, Debug)]
pub struct LaneOutput {
    pub lanes: usize,
    pub target: usize,
    pub symbols: Vec<u16>,
    pub consumed: usize,
    pub cycles: u64,
    pub stats: Vec<LaneStats>,
}

impl LaneOutput {
    /// Symbols of one lane.
    pub fn lane(&self, i: usize) -> &[u16] {
        &self.symbols[i * self.target..(i + 1) * self.target]
    }
}

/// Decode `alphabet_bits.len()` lanes of `target` symbols each from a byte source.
///
/// `alphabet_bits[i]` is lane i's symbol width minus 1 (the Verilog input).
/// `stream_len` bounds the cycle budget the same way as the reference model
/// (`64 * (lanes * target + stream_len) + 1000`).
pub fn decode_lanes<S: ByteSource>(
    src: &mut S,
    alphabet_bits: &[u8],
    target: usize,
    stream_len: usize,
) -> Result<LaneOutput, DecodeError> {
    let n = alphabet_bits.len();
    if n == 0 || n > MAX_LANES {
        return Err(DecodeError::Invalid(format!(
            "lane count {n} not in 1..={MAX_LANES}"
        )));
    }
    if target == 0 || target > u32::MAX as usize / 2 {
        return Err(DecodeError::Invalid(format!("bad symbol count {target}")));
    }
    let m0 = model(0, 0);
    let mut lanes: Vec<Lane> = alphabet_bits
        .iter()
        .map(|&ab| Lane {
            ab,
            target: target as u32,
            state: IDLE,
            ib: 0,
            ib_full: false,
            sym: 0,
            sym_full: false,
            in_rem: 0,
            out_rem: 0,
            p0: 0,
            p1: 0,
            hdr3: false,
            flush: false,
            q: 0,
            eab: 0,
            model: m0,
            done: false,
            go: true,
            stall: false,
            nsyms: 0,
            nbytes: 0,
            nfrags: 0,
            final_state: 0,
        })
        .collect();
    let mut symbols = vec![0u16; n * target];
    let mut slots: u32 = 0;
    let mut pos: usize = 0;
    let mut exhausted = false;
    let mut cycle: u64 = 0;
    let max_cycles: u64 = 64 * (n as u64 * target as u64 + stream_len as u64) + 1000;
    let mut total_syms: u64 = 0;
    let mut last_prog = (usize::MAX, u64::MAX, u32::MAX);
    let mut last_change: u64 = 0;

    while cycle < max_cycles {
        // ---------------- phase 0: forward update, feed, hand-off ----------------
        let mut served = 0usize;
        for (i, ln) in lanes.iter_mut().enumerate() {
            let bit = 1u32 << i;
            let blocked_out = ln.sym_full && (slots & bit) != 0;
            let need = ln.need_byte();
            let mut valid = false;
            let mut byte = 0u8;
            if need && !blocked_out && served < FEED_WINDOW && !exhausted {
                match src.next(i) {
                    Some(b) => {
                        valid = true;
                        byte = b;
                    }
                    None => exhausted = true,
                }
            }
            ln.stall = (need && !valid) || blocked_out;
            if ln.stall {
                continue;
            }
            if ln.state == PROC || ln.state == DRAIN {
                ln.p0 = if ln.sym_full {
                    ln.model.advance(ln.p1, ln.sym)
                } else {
                    ln.p1
                };
            }
            if valid {
                ln.ib = byte;
                ln.ib_full = true;
                pos += 1;
                served += 1;
                ln.nbytes += 1;
            }
            if ln.sym_full {
                slots |= bit;
                // nsyms < target: a symbol is only produced while out_rem > 0.
                debug_assert!((ln.nsyms as usize) < target);
                unsafe { *symbols.get_unchecked_mut(i * target + ln.nsyms as usize) = ln.sym };
                ln.nsyms += 1;
                total_syms += 1;
                ln.sym_full = false;
            }
        }
        // ---------------- phase 1: header FSM, renorm, decode ----------------
        // (the lockstep bookkeeping is folded into the same pass: slots do not
        // change in phase 1)
        let mut active = 0u32;
        let mut finished = true;
        let mut state_sum = 0u32;
        for (i, ln) in lanes.iter_mut().enumerate() {
            if !ln.stall {
                phase1(i, ln)?;
            }
            let bit = 1u32 << i;
            if !(ln.done && !ln.sym_full && (slots & bit) == 0) {
                active |= bit;
            }
            finished &= ln.done && !ln.sym_full;
            state_sum += ln.state as u32;
        }
        // ---------------- lockstep symbol vector ----------------
        if slots != 0 && (slots & active) == active {
            slots = 0;
        }
        cycle += 1;
        if finished {
            let bad: Vec<usize> = (0..n)
                .filter(|&i| !is_legal_end_state(lanes[i].final_state))
                .collect();
            if !bad.is_empty() {
                return Err(DecodeError::IllegalEndState { lanes: bad });
            }
            let stats = lanes
                .iter()
                .map(|l| LaneStats {
                    bytes: l.nbytes,
                    fragments: l.nfrags,
                    final_state: l.final_state,
                })
                .collect();
            return Ok(LaneOutput {
                lanes: n,
                target,
                symbols,
                consumed: pos,
                cycles: cycle,
                stats,
            });
        }
        let prog = (pos, total_syms, state_sum);
        if prog != last_prog {
            last_prog = prog;
            last_change = cycle;
        } else if cycle - last_change > 64 {
            return Err(DecodeError::NoProgress { consumed: pos });
        }
    }
    Err(DecodeError::CycleLimit)
}

/// Phase 1 of one lane (header FSM, renormalisation, symbol decode).
#[inline(always)]
fn phase1(i: usize, ln: &mut Lane) -> Result<(), DecodeError> {
    {
        {
            match ln.state {
                IDLE => {
                    if ln.go {
                        ln.go = false;
                        ln.state = HDR0;
                        ln.out_rem = ln.target;
                        ln.p0 = 0;
                        ln.p1 = 0;
                    }
                }
                HDR0 => {
                    if ln.ib_full {
                        let b = ln.ib;
                        ln.ib_full = false;
                        ln.in_rem = 0;
                        ln.state = HDR1;
                        ln.q = b & 15;
                        ln.hdr3 = b & 0x20 != 0;
                        ln.flush = b & 0x10 != 0;
                        let e = (ln.ab as i32 - (b >> 6) as i32) & 15;
                        ln.eab = if e < 10 { e as u8 } else { 9 };
                        ln.nfrags += 1;
                        if ln.eab > 8 {
                            return Err(DecodeError::BadHeader {
                                lane: i,
                                eab: ln.eab,
                            });
                        }
                    }
                }
                HDR1 => {
                    if ln.ib_full {
                        ln.ib_full = false;
                        ln.model = model(ln.eab, ln.q);
                        if ln.hdr3 {
                            ln.in_rem = ln.ib as u32;
                            ln.state = HDR2;
                        } else {
                            ln.in_rem = ln.ib as u32 + 1;
                            ln.state = PROC;
                        }
                    }
                }
                HDR2 => {
                    if ln.ib_full {
                        ln.ib_full = false;
                        ln.in_rem = (((ln.ib as u32) << 8) | (ln.in_rem & 0xFF)) + 257;
                        ln.state = PROC;
                    }
                }
                st => {
                    // PROC or DRAIN
                    let will = st == PROC && ln.p0 < L && ln.in_rem > 0 && ln.ib_full;
                    let x = if will {
                        ((ln.p0 & 0xFF_FFFF) << 8) | ln.ib as u32
                    } else {
                        ln.p0
                    };
                    let nsf = x > L && ln.out_rem > 0;
                    if nsf {
                        ln.out_rem -= 1;
                        ln.sym = ln.model.inverse(x as u16);
                    }
                    let new_out = ln.out_rem;
                    ln.p1 = x;
                    ln.sym_full = nsf;
                    if st == PROC {
                        if will {
                            ln.in_rem -= 1;
                            ln.ib_full = false;
                        }
                        if ln.in_rem == 0 {
                            if new_out > 0 {
                                if x == L {
                                    ln.state = HDR0;
                                    if ln.flush {
                                        ln.p1 = 0;
                                    }
                                } else {
                                    ln.state = DRAIN;
                                }
                            } else {
                                ln.done = true;
                                ln.state = IDLE;
                                ln.final_state = if nsf { ln.model.advance(x, ln.sym) } else { x };
                            }
                        }
                    } else if x <= L {
                        if new_out > 0 {
                            ln.state = HDR0;
                            if ln.flush {
                                ln.p0 = 0;
                                ln.p1 = 0;
                            }
                        } else {
                            ln.state = IDLE;
                            ln.done = true;
                            ln.final_state = if nsf { ln.model.advance(x, ln.sym) } else { x };
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// Fast decoder for the shared-stream case. Same clocked model as
/// [`decode_lanes`] (the tests check both agree on synthetic and real
/// streams), written with few data-dependent branches: the lanes spend almost
/// every clock in PROCESSING, so that path is computed with selects and the
/// rare state changes go through [`phase1`].
pub fn decode_lanes_fast(
    data: &[u8],
    alphabet_bits: &[u8],
    target: usize,
) -> Result<LaneOutput, DecodeError> {
    let n = alphabet_bits.len();
    if n == 0 || n > MAX_LANES {
        return Err(DecodeError::Invalid(format!(
            "lane count {n} not in 1..={MAX_LANES}"
        )));
    }
    if target == 0 || target > u32::MAX as usize / 2 {
        return Err(DecodeError::Invalid(format!("bad symbol count {target}")));
    }
    if data.is_empty() {
        return Err(DecodeError::NoProgress { consumed: 0 });
    }
    let m0 = model(0, 0);
    let lut0: &'static [u16] = m0.lut();
    let mut lanes: Vec<Lane> = alphabet_bits
        .iter()
        .map(|&ab| Lane {
            ab,
            target: target as u32,
            state: IDLE,
            ib: 0,
            ib_full: false,
            sym: 0,
            sym_full: false,
            in_rem: 0,
            out_rem: 0,
            p0: 0,
            p1: 0,
            hdr3: false,
            flush: false,
            q: 0,
            eab: 0,
            model: m0,
            done: false,
            go: true,
            stall: false,
            nsyms: 0,
            nbytes: 0,
            nfrags: 0,
            final_state: 0,
        })
        .collect();
    let mut luts: Vec<&'static [u16]> = vec![lut0; n];
    // One spare slot per lane: the symbol store below writes unconditionally.
    let stride = target + 1;
    let mut store = vec![0u16; n * stride];
    let len = data.len();
    let mut slots: u32 = 0;
    let mut pos: usize = 0;
    let mut cycle: u64 = 0;
    let max_cycles: u64 = 64 * (n as u64 * target as u64 + len as u64) + 1000;
    let mut total_syms: u64 = 0;
    let mut last_prog = (usize::MAX, u64::MAX, u32::MAX);
    let mut last_change: u64 = 0;

    while cycle < max_cycles {
        // ---------------- phase 0 ----------------
        let mut served = 0usize;
        for (i, ln) in lanes.iter_mut().enumerate() {
            let bit = 1u32 << i;
            let st = ln.state;
            let blocked_out = ln.sym_full & (slots & bit != 0);
            let need =
                !ln.ib_full & (((st >= HDR0) & (st <= HDR2)) | ((st == PROC) & (ln.in_rem > 0)));
            let grant = need & !blocked_out & (served < FEED_WINDOW) & (pos < len);
            let stall = (need & !grant) | blocked_out;
            ln.stall = stall;
            let go = !stall;
            if st >= PROC {
                let adv = ln.model.advance(ln.p1, ln.sym);
                let np0 = if ln.sym_full { adv } else { ln.p1 };
                ln.p0 = if go { np0 } else { ln.p0 };
            }
            let b = data[pos.min(len - 1)];
            ln.ib = if grant { b } else { ln.ib };
            ln.ib_full |= grant;
            pos += grant as usize;
            served += grant as usize;
            ln.nbytes += grant as u32;
            let push = go & ln.sym_full;
            // nsyms <= target, and slot `target` is the spare.
            unsafe { *store.get_unchecked_mut(i * stride + ln.nsyms as usize) = ln.sym };
            ln.nsyms += push as u32;
            total_syms += push as u64;
            slots |= (push as u32) << i;
            ln.sym_full &= !push;
        }
        // ---------------- phase 1 ----------------
        let mut active = 0u32;
        let mut finished = true;
        let mut state_sum = 0u32;
        for (i, ln) in lanes.iter_mut().enumerate() {
            if !ln.stall {
                let st = ln.state;
                if st >= PROC {
                    let is_proc = st == PROC;
                    let will = is_proc & (ln.p0 < L) & (ln.in_rem > 0) & ln.ib_full;
                    let x = if will {
                        (ln.p0 << 8) | ln.ib as u32
                    } else {
                        ln.p0
                    };
                    let nsf = (x > L) & (ln.out_rem > 0);
                    ln.sym = ln.model.inverse_fast(x as u16);
                    ln.out_rem -= nsf as u32;
                    ln.p1 = x;
                    ln.sym_full = nsf;
                    ln.in_rem -= will as u32;
                    ln.ib_full &= !will;
                    let change = if is_proc { ln.in_rem == 0 } else { x <= L };
                    if change {
                        transition(ln, x, nsf);
                    }
                } else {
                    phase1(i, ln)?;
                    if ln.state == PROC || ln.state == HDR2 {
                        luts[i] = ln.model.lut();
                    }
                }
            }
            let bit = 1u32 << i;
            if !(ln.done & !ln.sym_full & (slots & bit == 0)) {
                active |= bit;
            }
            finished &= ln.done & !ln.sym_full;
            state_sum += ln.state as u32;
        }
        if slots != 0 && (slots & active) == active {
            slots = 0;
        }
        cycle += 1;
        if finished {
            let bad: Vec<usize> = (0..n)
                .filter(|&i| !is_legal_end_state(lanes[i].final_state))
                .collect();
            if !bad.is_empty() {
                return Err(DecodeError::IllegalEndState { lanes: bad });
            }
            let mut symbols = vec![0u16; n * target];
            for i in 0..n {
                symbols[i * target..(i + 1) * target]
                    .copy_from_slice(&store[i * stride..i * stride + target]);
            }
            let stats = lanes
                .iter()
                .map(|l| LaneStats {
                    bytes: l.nbytes,
                    fragments: l.nfrags,
                    final_state: l.final_state,
                })
                .collect();
            return Ok(LaneOutput {
                lanes: n,
                target,
                symbols,
                consumed: pos,
                cycles: cycle,
                stats,
            });
        }
        let prog = (pos, total_syms, state_sum);
        if prog != last_prog {
            last_prog = prog;
            last_change = cycle;
        } else if cycle - last_change > 64 {
            return Err(DecodeError::NoProgress { consumed: pos });
        }
    }
    Err(DecodeError::CycleLimit)
}

/// End of a fragment's input (PROCESSING) or of draining: same rules as [`phase1`].
#[cold]
fn transition(ln: &mut Lane, x: u32, nsf: bool) {
    let new_out = ln.out_rem;
    if ln.state == PROC {
        if new_out > 0 {
            if x == L {
                ln.state = HDR0;
                if ln.flush {
                    ln.p1 = 0;
                }
            } else {
                ln.state = DRAIN;
            }
        } else {
            ln.done = true;
            ln.state = IDLE;
            ln.final_state = if nsf { ln.model.advance(x, ln.sym) } else { x };
        }
    } else if new_out > 0 {
        ln.state = HDR0;
        if ln.flush {
            ln.p0 = 0;
            ln.p1 = 0;
        }
    } else {
        ln.state = IDLE;
        ln.done = true;
        ln.final_state = if nsf { ln.model.advance(x, ln.sym) } else { x };
    }
}

/// Decode lanes from a byte slice and check that everything after the last
/// consumed byte is zero padding.
pub fn decode_slice(
    data: &[u8],
    alphabet_bits: &[u8],
    target: usize,
) -> Result<LaneOutput, DecodeError> {
    let out = decode_lanes_fast(data, alphabet_bits, target)?;
    if data[out.consumed..].iter().any(|&b| b != 0) {
        return Err(DecodeError::TrailingData {
            consumed: out.consumed,
            len: data.len(),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_matches_exact_division_for_every_model() {
        // The reciprocal path must invert forward() for every cf, and the
        // table must agree with the exact path everywhere (cf = 0 included).
        for eab in 0..9u8 {
            for q in 0..16u8 {
                let m = model(eab, q);
                assert_eq!(m.inverse(0), m.inverse_exact(0));
                assert_eq!(m.inverse_fast(0), m.inverse_exact(0));
                for cf in 1..=0xFFFFu32 {
                    let s = m.inverse(cf as u16);
                    assert_eq!(s, m.inverse_exact(cf as u16), "eab {eab} q {q} cf {cf}");
                    assert_eq!(
                        s,
                        m.inverse_fast(cf as u16),
                        "fast: eab {eab} q {q} cf {cf}"
                    );
                    let (f, c) = m.forward(s);
                    assert!(c <= cf && cf < c + f, "eab {eab} q {q} cf {cf} sym {s}");
                }
            }
        }
    }
}
