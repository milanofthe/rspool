//! What an op of some work costs, how wide a section of it runs best, and
//! which of a library's ways to run some work is fastest: all measured on
//! the runs.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU8, Ordering::Relaxed};

use crate::constants::{
    RATE_SAMPLE_EVERY, RATE_WEIGHT_DOWN, RATE_WEIGHT_UP, TUNE_ARMS, TUNE_BINS, TUNE_BIN_BASE_NS,
    TUNE_BIN_FACTOR, TUNE_EXPLORE, TUNE_WEIGHT_DOWN, TUNE_WEIGHT_UP,
};
use crate::{Dispatch, Piece, Place};

/// What an op of some kind of work costs, and how long a section of it took
/// on each width it ran on.
///
/// The cost of an op is a prior until serial runs are measured (one in
/// [`RATE_SAMPLE_EVERY`]), then their running mean; a section's work is its
/// ops at that cost. A serial run inside which a section went wide (a call
/// whose own work has wide sections) does not teach it.
///
/// How wide a section runs is a [`Choice`] among widths (1, 2, 4, ...
/// threads up to what is free here), an arm not yet measured counted at what
/// the pool's section cost predicts for it. Whatever a wide run pays beyond
/// the work over its threads (the output its pieces hand from core to core,
/// the memory bandwidth they share, the scratch they take), the measurement
/// holds; nothing of it has to be modelled.
#[derive(Debug)]
pub struct Rate {
    /// ns per op of serial runs, as `f32` bits.
    ns: AtomicU32,
    /// Whether a serial run was measured: `ns` is no longer the prior.
    learned: AtomicBool,
    /// The widths, arm `a` on `2^a` threads (the last one on all free).
    widths: Choice,
}

/// Which of up to [`TUNE_ARMS`] ways to run some work is fastest, from what
/// its runs took: per way (an option) and per size of the work (bins a
/// factor [`TUNE_BIN_FACTOR`] apart from [`TUNE_BIN_BASE_NS`]) the measured
/// time per op. A run takes the option that was fastest for work of its
/// size. An option not yet measured counts at the time the caller expects
/// for it ([`Prior`]); past option 0, the reference way (serially, at the
/// rate its runs measured), times what the runs of the nearest measured
/// option past it took over what was expected of them: a model that was
/// off by a factor on one way is taken to be off by as much on the next,
/// so a library does not try every way its model overrates one after the
/// other. Now and then (one
/// in [`TUNE_EXPLORE`]) a run takes, measured, the option nearest to the
/// fastest whose least time could beat it, so that a change in the code or
/// in the machine is followed, and never one that could not; besides, one
/// in [`RATE_SAMPLE_EVERY`] runs is measured. The first run of a way is a
/// warm-up (cold caches, first-touched scratch): it counts at no more than
/// what was expected, and the way's next run replaces it.
///
/// The per-region choice the literature tunes online (feedback-driven
/// threading, Suleman et al. 2008; Otter, SCALO): for the width of a section
/// ([`Rate`]), and for whatever else a library chooses between, how a solver
/// replays a block of its factorization, say.
#[derive(Debug)]
pub struct Choice {
    /// ns per op a run of a size bin took with an option, as `f32` bits;
    /// zero where none was measured.
    cells: [[AtomicU32; TUNE_ARMS]; TUNE_BINS],
    /// The runs measured per cell, up to two: the first warms the way up
    /// (its code, its caches, its scratch) and is replaced by the second.
    runs: [[AtomicU8; TUNE_ARMS]; TUNE_BINS],
}

/// What a caller predicts for an option of a [`Choice`] its runs have not
/// measured: the time it expects, and the least the option can take (a
/// bound: an option whose least is above the fastest's time is never
/// explored; zero where nothing bounds it). An option not expected to be
/// run at all (expected infinite) is only ever explored. For a schedule of work `T1` on `k` threads with a longest
/// dependency chain `Tinf`, Brent's bounds: `T1 / k + Tinf` expected (any
/// greedy schedule), `max(T1 / k, Tinf)` least.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Prior {
    /// The time expected, in ns.
    pub expected: f64,
    /// The least time possible, in ns.
    pub least: f64,
}

impl Prior {
    /// A prediction that is its own bound: `ns` expected and at least.
    pub fn exactly(ns: f64) -> Self {
        Prior {
            expected: ns,
            least: ns,
        }
    }
}

/// What [`Choice::pick`] chose: the option, and whether (for which size
/// bin) its run is to be measured.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pick {
    /// The option chosen.
    pub option: usize,
    bin: usize,
    measure: bool,
}

impl Pick {
    /// Whether the run is to be measured ([`Choice::record`]).
    pub fn measured(&self) -> bool {
        self.measure
    }
}

impl Choice {
    /// No option measured yet.
    pub const fn new() -> Self {
        Choice {
            cells: [const { [const { AtomicU32::new(0) }; TUNE_ARMS] }; TUNE_BINS],
            runs: [const { [const { AtomicU8::new(0) }; TUNE_ARMS] }; TUNE_BINS],
        }
    }

    /// Among the options `0..n` (at most [`TUNE_ARMS`]) for `ops` ops of
    /// `work` ns: of those available here (`predicted(option)` is some),
    /// the one whose measured time, or where none is measured the expected
    /// one corrected by the nearest measured option's, is least; now and
    /// then the nearest available one whose least could beat it. An option
    /// keeps its number wherever it is offered, so what was measured for it
    /// is its own; option 0 where none is available.
    pub fn pick(
        &self,
        work: f64,
        ops: f64,
        n: usize,
        predicted: impl Fn(usize) -> Option<Prior>,
    ) -> Pick {
        let n = n.clamp(1, TUNE_ARMS);
        let b = bin(work);
        let mut prior = [None; TUNE_ARMS];
        let mut seen = [None; TUNE_ARMS];
        for o in 0..n {
            prior[o] = predicted(o);
            if prior[o].is_some() {
                seen[o] = self.cell(b, o).map(|per_op| per_op * ops);
            }
        }
        // what the measured option nearest to `o` past the reference took
        // over what was expected of it (the slower on a tie); one where
        // none is measured, and for the reference itself
        let error = |o: usize| {
            let off = |q: usize| match (seen[q], prior[q]) {
                (Some(t), Some(p)) if q > 0 && p.expected > 0.0 && p.expected.is_finite() => {
                    Some(t / p.expected)
                }
                _ => None,
            };
            if o == 0 {
                return 1.0;
            }
            (1..n)
                .find_map(|d| {
                    let below = o.checked_sub(d).and_then(off);
                    let above = (o + d < n).then(|| off(o + d)).flatten();
                    match (below, above) {
                        (Some(x), Some(y)) => Some(x.max(y)),
                        (x, y) => x.or(y),
                    }
                })
                .unwrap_or(1.0)
        };
        let mut time = [f64::INFINITY; TUNE_ARMS];
        let mut best = None;
        for o in 0..n {
            if let Some(p) = prior[o] {
                let expected = || p.expected * error(o);
                time[o] = match seen[o] {
                    Some(t) if self.warming(b, o) => t.min(expected()),
                    Some(t) => t,
                    None => expected(),
                };
                if best.is_none_or(|q: usize| time[o] < time[q]) {
                    best = Some(o);
                }
            }
        }
        let Some(best) = best else {
            return Pick {
                option: 0,
                bin: b,
                measure: false,
            };
        };
        // exploring: the nearest option that could beat the fastest, below
        // or above it alike
        let could = |o: usize| o != best && prior[o].is_some_and(|p| p.least < time[best]);
        let explore = one_in(TUNE_EXPLORE).then(|| {
            let down = one_in(2);
            (1..n).find_map(|d| {
                let below = best.checked_sub(d).filter(|&o| could(o));
                let above = Some(best + d).filter(|&o| o < n && could(o));
                if down {
                    below.or(above)
                } else {
                    above.or(below)
                }
            })
        });
        let option = explore.flatten().unwrap_or(best);
        let measure = option != best
            || seen[option].is_none()
            || self.warming(b, option)
            || (ops > 0.0 && one_in(RATE_SAMPLE_EVERY));
        Pick {
            option,
            bin: b,
            measure,
        }
    }

    /// The run of `pick`, `ops` ops, took `ns` ns.
    pub fn record(&self, pick: Pick, ns: f64, ops: f64) {
        if ns > 0.0 && ops > 0.0 {
            let (b, o) = (pick.bin.min(TUNE_BINS - 1), pick.option.min(TUNE_ARMS - 1));
            let seen = (ns / ops) as f32;
            let runs = self.runs[b][o]
                .fetch_update(Relaxed, Relaxed, |r| Some(r.saturating_add(1).min(2)));
            match runs.unwrap_or(2) {
                // the first run, and the one that replaces it
                0 | 1 => self.cells[b][o].store(seen.to_bits(), Relaxed),
                _ => update(&self.cells[b][o], seen, TUNE_WEIGHT_DOWN, TUNE_WEIGHT_UP),
            }
        }
    }

    /// Whether option `o` of size bin `b` has only its warm-up run measured.
    fn warming(&self, b: usize, o: usize) -> bool {
        self.runs[b.min(TUNE_BINS - 1)][o.min(TUNE_ARMS - 1)].load(Relaxed) == 1
    }

    /// A pick of `option` for work of `work` ns, measured: for a run made
    /// without one.
    fn at(work: f64, option: usize) -> Pick {
        Pick {
            option,
            bin: bin(work),
            measure: true,
        }
    }

    /// The measured ns per op of `option` for size bin `b`, if any.
    fn cell(&self, b: usize, option: usize) -> Option<f64> {
        let v = f32::from_bits(
            self.cells[b.min(TUNE_BINS - 1)][option.min(TUNE_ARMS - 1)].load(Relaxed),
        );
        (v > 0.0).then_some(v as f64)
    }
}

impl Default for Choice {
    fn default() -> Self {
        Choice::new()
    }
}

thread_local! {
    /// This thread's dice for the sampling and the exploration: a draw per
    /// decision, so no counter is shared between the threads that run a
    /// rate's sections (a cache line every one of them would write), and no
    /// rate's runs fall into step with another's.
    static DICE: std::cell::Cell<u32> = const { std::cell::Cell::new(0x9E37_79B9) };
    /// The run this thread is about to make is to be measured: for which
    /// rate (its address), and as which pick ([`Work::go`]).
    static NEXT: std::cell::Cell<Option<(usize, Pick)>> = const { std::cell::Cell::new(None) };
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
            learned: AtomicBool::new(false),
            widths: Choice::new(),
        }
    }

    /// ns per op.
    pub fn ns_per_op(&self) -> f64 {
        f32::from_bits(self.ns.load(Relaxed)) as f64
    }

    /// Whether a serial run was measured: until then the cost of an op is
    /// the prior.
    pub fn learned(&self) -> bool {
        self.learned.load(Relaxed)
    }

    /// Whether to measure a serial run of `ops` ops: one in
    /// [`RATE_SAMPLE_EVERY`], drawn.
    pub fn sample(&self, ops: f64) -> bool {
        ops > 0.0 && one_in(RATE_SAMPLE_EVERY)
    }

    /// A serial run of `ops` ops took `ns` ns: the first replaces the
    /// prior, then the running mean follows a faster run at once
    /// ([`RATE_WEIGHT_DOWN`]), a slower one only as they keep coming
    /// ([`RATE_WEIGHT_UP`]).
    pub fn learn(&self, ns: f64, ops: f64) {
        if ns > 0.0 && ops > 0.0 {
            if !self.learned.swap(true, Relaxed) {
                self.ns.store(((ns / ops) as f32).to_bits(), Relaxed);
                return;
            }
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
        // the section cost borrowed, not cloned: a decision is made on every
        // thread at once, and a shared count taken and given back per
        // decision cost them tens of ns each per thread there was
        P::with_dispatch(|d| self.decide(rate, d, w, pieces, cap))
    }

    /// [`go`](Self::go) for work worth a decision: the width among those up
    /// to `cap` its runs of its size were fastest on, on a place whose
    /// sections cost `d`.
    fn decide(&self, rate: &Rate, d: &Dispatch, w: f64, pieces: usize, cap: usize) -> Go {
        if w < d.break_even() {
            return Go::Here;
        }
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
        let pick = rate
            .widths
            .pick(w, self.ops, n, |a| Some(Prior::exactly(predicted(arms[a]))));
        if pick.measure {
            NEXT.with(|c| c.set(Some((rate.key(), pick))));
        }
        match arms[pick.option] {
            k if k >= 2 => Go::Wide(k),
            _ => Go::Here,
        }
    }

    /// The measurement [`go`](Self::go) asked of the run it decided, if it
    /// was for this work's rate.
    fn take_next(&self) -> Option<Pick> {
        let rate = self.rate?;
        let next = NEXT.with(|c| c.take())?;
        (next.0 == rate.key()).then_some(next.1)
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
            let pick = next.map_or(Choice::at(self.time(), 0), |n| Pick { option: 0, ..n });
            rate.widths.record(pick, ns, self.ops);
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
            pick: Pick { option: 0, ..next },
            wide: crate::wide_count(),
            start: crate::clock::Instant::now(),
        })
    }

    /// The serial run `t` ([`serial_timed`](Self::serial_timed)) ended.
    pub fn record_serial(&self, t: SerialRun) {
        let ns = t.start.elapsed().as_nanos() as f64;
        if let Some(rate) = self.rate.filter(|_| crate::wide_count() == t.wide) {
            rate.widths.record(t.pick, ns, self.ops);
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
        if next.is_none() && !rate.sample(self.ops) {
            return f();
        }
        let t = crate::clock::Instant::now();
        let r = f();
        let pick = next.unwrap_or_else(|| Choice::at(self.time(), arm_of(k)));
        rate.widths
            .record(pick, t.elapsed().as_nanos() as f64, self.ops);
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
    pick: Pick,
    wide: u64,
    start: crate::clock::Instant,
}

/// The arm of a run on `k` threads: the least `a` with `2^a >= k`.
fn arm_of(k: usize) -> usize {
    (k.max(1).next_power_of_two().trailing_zeros() as usize).min(TUNE_ARMS - 1)
}
