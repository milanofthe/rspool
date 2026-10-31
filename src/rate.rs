//! What an op of some work costs, and how wide a section of it runs best,
//! both measured on the runs.

use std::sync::atomic::{AtomicU32, Ordering::Relaxed};

use crate::constants::{
    RATE_SAMPLE_EVERY, RATE_WEIGHT_DOWN, RATE_WEIGHT_UP, TUNE_ARMS, TUNE_BINS, TUNE_BIN_BASE_NS,
    TUNE_BIN_FACTOR, TUNE_EXPLORE, TUNE_WEIGHT_DOWN, TUNE_WEIGHT_UP,
};
use crate::{Piece, Place};

/// What an op of some kind of work costs, and how long a section of it took
/// on each width it ran on.
///
/// The cost of an op is a prior until serial runs are measured (one in
/// [`RATE_SAMPLE_EVERY`]), then their running mean; a section's work is its
/// ops at that cost. A serial run inside which a section went wide (a call
/// whose own work has wide sections) does not teach it.
///
/// How wide a section runs is chosen from what its runs took: per width
/// (an arm: 1, 2, 4, ... threads up to what is free here) and per size of
/// the work (bins a factor [`TUNE_BIN_FACTOR`] apart from
/// [`TUNE_BIN_BASE_NS`]) the measured time per op. A section takes the arm
/// that was fastest for work of its size, an arm not yet measured counted at
/// what the pool's section cost predicts for it. Now and then (one in
/// [`TUNE_EXPLORE`]) it takes a neighbouring arm instead, measured, so that
/// a change in the code (a body specialized, compiled) or in the machine is
/// followed. Whatever a wide run pays beyond the work over its threads (the
/// output its pieces hand from core to core, the memory bandwidth they
/// share, the scratch they take), the measurement holds; nothing of it has
/// to be modelled.
#[derive(Debug)]
pub struct Rate {
    /// ns per op of serial runs, as `f32` bits.
    ns: AtomicU32,
    /// ns per op a run of a size bin took on an arm, as `f32` bits; zero
    /// where none was measured.
    cells: [[AtomicU32; TUNE_ARMS]; TUNE_BINS],
}

thread_local! {
    /// This thread's dice for the sampling and the exploration: a draw per
    /// decision, so no counter is shared between the threads that run a
    /// rate's sections (a cache line every one of them would write), and no
    /// rate's runs fall into step with another's.
    static DICE: std::cell::Cell<u32> = const { std::cell::Cell::new(0x9E37_79B9) };
    /// The run this thread is about to make is to be measured: for which
    /// rate (its address), size bin and arm ([`Work::go`]).
    static NEXT: std::cell::Cell<Option<(usize, usize, usize)>> = const { std::cell::Cell::new(None) };
}

/// Whether this thread's next draw comes up one in `n`: per-thread dice,
/// so no counter is shared between threads, for a library's own
/// explorations and samples.
pub fn draw(n: u32) -> bool {
    one_in(n)
}

/// Whether this thread's next draw comes up one in `n`.
fn one_in(n: u32) -> bool {
    DICE.with(|d| {
        let mut x = d.get();
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        d.set(x);
        x.is_multiple_of(n.max(1))
    })
}

/// The running mean `old` moved toward `seen`: down at `down`, up at `up`
/// (interruptions only ever add time to a run), `seen` taken whole where
/// nothing was known.
fn mean(old: f32, seen: f32, down: f32, up: f32) -> f32 {
    if old <= 0.0 {
        return seen;
    }
    let w = if seen < old { down } else { up };
    old + w * (seen - old)
}

/// `cell` moved toward `seen` ([`mean`]).
fn update(cell: &AtomicU32, seen: f32, down: f32, up: f32) {
    let mut bits = cell.load(Relaxed);
    loop {
        let new = mean(f32::from_bits(bits), seen, down, up).to_bits();
        match cell.compare_exchange_weak(bits, new, Relaxed, Relaxed) {
            Ok(_) => return,
            Err(now) => bits = now,
        }
    }
}

/// The size bin of `work` ns.
fn bin(work: f64) -> usize {
    if !(work > TUNE_BIN_BASE_NS) {
        return 0;
    }
    ((work / TUNE_BIN_BASE_NS).ln() / TUNE_BIN_FACTOR.ln()) as usize
}

/// The threads of arm `a` where `cap` are free: `2^a`, at most `cap`.
fn arm_width(a: usize, cap: usize) -> usize {
    (1usize << a.min(usize::BITS as usize - 1)).min(cap).max(1)
}

impl Rate {
    /// A rate of `prior` ns per op until runs are measured.
    pub const fn new(prior: f32) -> Self {
        Rate {
            ns: AtomicU32::new(prior.to_bits()),
            cells: [const { [const { AtomicU32::new(0) }; TUNE_ARMS] }; TUNE_BINS],
        }
    }

    /// ns per op.
    pub fn ns_per_op(&self) -> f64 {
        f32::from_bits(self.ns.load(Relaxed)) as f64
    }

    /// Whether to measure a serial run of `ops` ops: one in
    /// [`RATE_SAMPLE_EVERY`], drawn.
    pub fn sample(&self, ops: f64) -> bool {
        ops > 0.0 && one_in(RATE_SAMPLE_EVERY)
    }

    /// A serial run of `ops` ops took `ns` ns: the running mean follows a
    /// faster run at once ([`RATE_WEIGHT_DOWN`]), a slower one only as they
    /// keep coming ([`RATE_WEIGHT_UP`]).
    pub fn learn(&self, ns: f64, ops: f64) {
        if ns > 0.0 && ops > 0.0 {
            update(
                &self.ns,
                (ns / ops) as f32,
                RATE_WEIGHT_DOWN,
                RATE_WEIGHT_UP,
            );
        }
    }

    /// `f`, a serial run of `ops` ops, measured if sampled.
    pub fn time<R>(&self, ops: f64, f: impl FnOnce() -> R) -> R {
        let timing = self.begin(ops, false);
        let r = f();
        if let Some(t) = timing {
            self.end(t, ops);
        }
        r
    }

    /// `f`, a serial run of `ops` ops, measured.
    pub fn measure<R>(&self, ops: f64, f: impl FnOnce() -> R) -> R {
        let timing = self.begin(ops, true);
        let r = f();
        if let Some(t) = timing {
            self.end(t, ops);
        }
        r
    }

    /// A serial run of `ops` ops starts: timed if sampled, or `always`;
    /// [`end`](Self::end) learns from it. For a run that is no closure.
    pub fn begin(&self, ops: f64, always: bool) -> Option<Timing> {
        (always || self.sample(ops)).then(|| Timing {
            wide: crate::wide_count(),
            start: crate::clock::Instant::now(),
        })
    }

    /// The run `t` of `ops` ops ([`begin`](Self::begin)) ended; it teaches
    /// nothing if a section went wide inside it.
    pub fn end(&self, t: Timing, ops: f64) {
        let ns = t.start.elapsed().as_nanos() as f64;
        if crate::wide_count() == t.wide {
            self.learn(ns, ops);
        }
    }

    /// The measured ns per op of arm `a` for size bin `b`, if any.
    fn cell(&self, b: usize, a: usize) -> Option<f64> {
        let v =
            f32::from_bits(self.cells[b.min(TUNE_BINS - 1)][a.min(TUNE_ARMS - 1)].load(Relaxed));
        (v > 0.0).then_some(v as f64)
    }

    /// A run of `ops` ops of size bin `b` on arm `a` took `ns` ns.
    fn record(&self, b: usize, a: usize, ns: f64, ops: f64) {
        if ns > 0.0 && ops > 0.0 {
            let cell = &self.cells[b.min(TUNE_BINS - 1)][a.min(TUNE_ARMS - 1)];
            update(cell, (ns / ops) as f32, TUNE_WEIGHT_DOWN, TUNE_WEIGHT_UP);
        }
    }

    /// The address that tells this rate in [`NEXT`].
    fn key(&self) -> usize {
        self as *const Rate as usize
    }
}

/// A serial run being measured ([`Rate::begin`]): how many sections this
/// thread had sent wide, and when it started.
#[derive(Debug)]
pub struct Timing {
    wide: u64,
    start: crate::clock::Instant,
}

/// A section's work: `ops` at a learned rate, which learns from the
/// section's runs ([`of`](Self::of)), or an estimate in ns ([`ns`](
/// Self::ns)).
#[derive(Clone, Copy, Debug)]
pub struct Work<'r> {
    ops: f64,
    rate: Option<&'r Rate>,
    /// The work in ns where it is not `ops` at the rate ([`estimated`](
    /// Self::estimated)).
    ns: Option<f64>,
}

/// How a section runs ([`Work::go`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Go {
    /// Serially on the calling thread.
    Here,
    /// On this many threads.
    Wide(usize),
}

impl<'r> Work<'r> {
    /// `ns` ns of work: decided on the pool's section cost, never measured.
    pub fn ns(ns: f64) -> Self {
        Work {
            ops: ns,
            rate: None,
            ns: None,
        }
    }

    /// `ops` ops at `rate`, which learns from the section's runs.
    pub fn of(ops: f64, rate: &'r Rate) -> Self {
        Work {
            ops,
            rate: Some(rate),
            ns: None,
        }
    }

    /// `ns` ns of work in `ops` ops, estimated by the caller (the calls of a
    /// stage, each at its own rate), deciding and timed with `rate`'s
    /// measurements. Its serial runs teach the ns per op of their parts at
    /// their own rates, the caller's to measure ([`serial_timed`](
    /// Self::serial_timed)).
    pub fn estimated(ns: f64, ops: f64, rate: &'r Rate) -> Self {
        Work {
            ops,
            rate: Some(rate),
            ns: Some(ns),
        }
    }

    /// The work in ns.
    pub fn time(&self) -> f64 {
        match (self.ns, self.rate) {
            (Some(ns), _) => ns,
            (None, Some(r)) => self.ops * r.ns_per_op(),
            (None, None) => self.ops,
        }
    }

    /// How the section runs in `pieces` pieces at the place `P`: on the
    /// arm that was fastest for work of its size there (see [`Rate`]), an
    /// arm not measured yet counted at its predicted time; now and then on a
    /// neighbouring arm instead. Work below what any section is worth on
    /// the place runs serially without looking further.
    pub fn go<P: Place>(&self, pieces: usize) -> Go {
        let w = self.time();
        let cap = P::cap().min(pieces);
        let Some(rate) = self.rate else {
            return match P::width(w, pieces) {
                k if k >= 2 => Go::Wide(k),
                _ => Go::Here,
            };
        };
        if cap < 2 || !(w > 0.0) {
            return Go::Here;
        }
        let d = P::dispatch();
        if w < d.break_even() {
            return Go::Here;
        }
        let b = bin(w);
        let p = pieces as f64;
        let predicted = |k: usize| {
            if k < 2 {
                w
            } else {
                w * pieces.div_ceil(k) as f64 / p + d.cost(k)
            }
        };
        // the arms up to the free threads, the last one exactly them
        let mut arms = [0usize; TUNE_ARMS];
        let mut n = 0;
        while n < TUNE_ARMS {
            let k = arm_width(n, cap);
            arms[n] = k;
            n += 1;
            if k >= cap {
                break;
            }
        }
        let est = |a: usize| match rate.cell(b, a) {
            Some(per_op) => per_op * self.ops,
            None => predicted(arms[a]),
        };
        let mut best = 0;
        for a in 1..n {
            if est(a) < est(best) {
                best = a;
            }
        }
        let mut chosen = best;
        let explore = n > 1 && one_in(TUNE_EXPLORE);
        if explore {
            let down = best > 0 && (best + 1 >= n || one_in(2));
            chosen = if down { best - 1 } else { best + 1 };
        }
        if explore || rate.cell(b, chosen).is_none() || rate.sample(self.ops) {
            NEXT.with(|c| c.set(Some((rate.key(), b, chosen))));
        }
        match arms[chosen] {
            k if k >= 2 => Go::Wide(k),
            _ => Go::Here,
        }
    }

    /// The measurement [`go`](Self::go) asked of the run it decided, if it
    /// was for this work's rate: its size bin and arm.
    fn take_next(&self) -> Option<(usize, usize)> {
        let rate = self.rate?;
        let next = NEXT.with(|c| c.take())?;
        (next.0 == rate.key()).then_some((next.1, next.2))
    }

    /// `f`, the section run on this thread as [`go`](Self::go) decided:
    /// measured when asked for, or sampled; the time teaches the rate's ns
    /// per op and its serial arm.
    pub fn serial<R>(&self, f: impl FnOnce() -> R) -> R {
        let Some(rate) = self.rate else {
            return f();
        };
        let next = self.take_next();
        if next.is_none() && !rate.sample(self.ops) {
            return f();
        }
        let wide = crate::wide_count();
        let t = crate::clock::Instant::now();
        let r = f();
        let ns = t.elapsed().as_nanos() as f64;
        if crate::wide_count() == wide {
            if self.ns.is_none() {
                rate.learn(ns, self.ops);
            }
            rate.record(next.map_or(bin(self.time()), |n| n.0), 0, ns, self.ops);
        }
        r
    }

    /// Whether the serial run [`go`](Self::go) decided is to be measured as
    /// a whole, for a caller that runs it itself (in parts at their own
    /// rates, [`estimated`](Self::estimated)): then [`record_serial`](
    /// Self::record_serial) with its time.
    pub fn serial_timed(&self) -> Option<SerialRun> {
        let next = self.take_next()?;
        Some(SerialRun {
            bin: next.0,
            wide: crate::wide_count(),
            start: crate::clock::Instant::now(),
        })
    }

    /// The serial run `t` ([`serial_timed`](Self::serial_timed)) ended.
    pub fn record_serial(&self, t: SerialRun) {
        let ns = t.start.elapsed().as_nanos() as f64;
        if let Some(rate) = self.rate.filter(|_| crate::wide_count() == t.wide) {
            rate.record(t.bin, 0, ns, self.ops);
        }
    }

    /// The section as `n` pieces on `k` threads at the place `P`, measured
    /// when [`go`](Self::go) asked for it ([`watch`](Self::watch)).
    pub fn wide<P: Place>(&self, n: usize, k: usize, f: &Piece<'_>) {
        self.watch::<P, _>(n, k, || P::run_width(n, k, f))
    }

    /// `f`, the section run wide on `k` threads at `P`: measured when
    /// [`go`](Self::go) asked for it, or sampled; the time teaches the
    /// rate's arm for `k` threads.
    pub fn watch<P: Place, R>(&self, pieces: usize, k: usize, f: impl FnOnce() -> R) -> R {
        let _ = pieces;
        let Some(rate) = self.rate else {
            return f();
        };
        let next = self.take_next();
        let b = next.map_or(bin(self.time()), |n| n.0);
        if next.is_none() && !rate.sample(self.ops) {
            return f();
        }
        let t = crate::clock::Instant::now();
        let r = f();
        let a = next.map_or(arm_of(k), |n| n.1);
        rate.record(b, a, t.elapsed().as_nanos() as f64, self.ops);
        r
    }

    /// The section, `n` pieces of this work: serially or on as many threads
    /// of `P` as it is worth, the rate learning from it.
    pub fn run<P: Place>(&self, n: usize, f: &Piece<'_>) {
        match self.go::<P>(n) {
            Go::Wide(k) => self.wide::<P>(n, k, f),
            Go::Here => self.serial(|| (0..n).for_each(f)),
        }
    }
}

/// A serial run measured as a whole by its caller ([`Work::serial_timed`]).
#[derive(Debug)]
pub struct SerialRun {
    bin: usize,
    wide: u64,
    start: crate::clock::Instant,
}

/// The arm of a run on `k` threads: the least `a` with `2^a >= k`.
fn arm_of(k: usize) -> usize {
    (k.max(1).next_power_of_two().trailing_zeros() as usize).min(TUNE_ARMS - 1)
}
