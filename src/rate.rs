//! What an op of some work costs, learned from the runs, and how a section
//! of such work runs.

use std::sync::atomic::{AtomicU32, Ordering::Relaxed};

use crate::constants::{
    RATE_EXPLORE, RATE_PROBES, RATE_SAMPLE_EVERY, RATE_STALE, RATE_WEIGHT_DOWN, RATE_WEIGHT_UP,
    SURCHARGE_WEIGHT_DOWN, SURCHARGE_WEIGHT_UP,
};
use crate::{Piece, Place};

/// What an op of some work costs, in ns: a prior until runs are measured,
/// then their running mean. A section's work is its ops at its kind's rate
/// (a body's ops, a replay's multiply-adds, a pass's entries).
///
/// Only serial runs teach it, never wide ones, whose time mixes the work
/// with what the hand-off cost: the first [`RATE_PROBES`] sections of a
/// rate run serially for it ([`probing`](Self::probing)); after them one
/// serial run in [`RATE_SAMPLE_EVERY`] is measured; a section that goes wide
/// runs serially now and then to be measured ([`explore`](Self::explore));
/// and a wide run far faster than its rate predicts has the rate probe anew
/// ([`Work::watch`]). A serial run inside which a section went wide (a call
/// whose own work has wide sections) teaches nothing.
///
/// A rate also learns its surcharge: what a wide run of its work takes
/// beyond its work over the threads and the pool's section cost, the output
/// its pieces hand from core to core and the scratch they take, which the
/// pool's calibration (pieces of a few cache lines) cannot know. The timed
/// wide runs teach it, and a section decides on it on top of the section
/// cost; one it keeps serial that would go wide without it goes wide now and
/// then, timed, so that a surcharge that fell is seen to fall.
#[derive(Debug)]
pub struct Rate {
    /// ns per op, as `f32` bits.
    ns: AtomicU32,
    /// Serial probes run.
    probes: AtomicU32,
    /// ns a wide run takes beyond its prediction, as `f32` bits.
    surcharge: AtomicU32,
}

thread_local! {
    /// The next wide run this thread watches is a retry: timed whatever the
    /// sampling draws ([`Work::go`]).
    static RETRY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// This thread's dice for the sampling and the exploration: a draw per
    /// decision, so no counter is shared between the threads that run a
    /// rate's sections (a cache line every one of them would write), and no
    /// rate's runs fall into step with another's.
    static DICE: std::cell::Cell<u32> = const { std::cell::Cell::new(0x9E37_79B9) };
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

impl Rate {
    /// A rate of `prior` ns per op until runs are measured.
    pub const fn new(prior: f32) -> Self {
        Rate {
            ns: AtomicU32::new(prior.to_bits()),
            probes: AtomicU32::new(0),
            surcharge: AtomicU32::new(0),
        }
    }

    /// ns per op.
    pub fn ns_per_op(&self) -> f64 {
        f32::from_bits(self.ns.load(Relaxed)) as f64
    }

    /// Whether a run of `ops` ops is to run serially, to be measured: the
    /// rate has not had its probes yet.
    pub fn probing(&self, ops: f64) -> bool {
        ops > 0.0 && self.probes.load(Relaxed) < RATE_PROBES
    }

    /// Whether a run of `ops` ops that would go `k` wide is to run serially
    /// instead, measured: one in [`RATE_EXPLORE`] times `k - 1`, drawn.
    pub fn explore(&self, ops: f64, k: usize) -> bool {
        let every = RATE_EXPLORE.saturating_mul(k.saturating_sub(1).max(1) as u32);
        ops > 0.0 && one_in(every)
    }

    /// What a wide run of this work takes beyond its prediction, in ns.
    pub fn surcharge(&self) -> f64 {
        f32::from_bits(self.surcharge.load(Relaxed)) as f64
    }

    /// A timed wide run took `excess` ns beyond its prediction (negative
    /// where it took less): the surcharge follows up at once
    /// ([`SURCHARGE_WEIGHT_UP`]), down more slowly.
    pub fn learn_surcharge(&self, excess: f64) {
        let seen = excess.max(0.0) as f32;
        let mut bits = self.surcharge.load(Relaxed);
        loop {
            let old = f32::from_bits(bits);
            let w = if seen > old {
                SURCHARGE_WEIGHT_UP
            } else {
                SURCHARGE_WEIGHT_DOWN
            };
            let new = (old + w * (seen - old)).to_bits();
            match self
                .surcharge
                .compare_exchange_weak(bits, new, Relaxed, Relaxed)
            {
                Ok(_) => return,
                Err(now) => bits = now,
            }
        }
    }

    /// Whether a run of `ops` ops its surcharge keeps serial, though it would
    /// go `k` wide without it, is to go wide instead, timed: one in
    /// [`RATE_EXPLORE`] times `k - 1`, drawn.
    pub fn retry(&self, ops: f64, k: usize) -> bool {
        self.explore(ops, k)
    }

    /// The rate is stale: its next runs probe again (warm, all counted).
    pub fn reprobe(&self) {
        self.probes.store(1, Relaxed);
    }

    /// Whether to measure a run of `ops` ops: one in [`RATE_SAMPLE_EVERY`],
    /// drawn.
    pub fn sample(&self, ops: f64) -> bool {
        ops > 0.0 && one_in(RATE_SAMPLE_EVERY)
    }

    /// A serial run of `ops` ops took `ns` ns: the running mean follows a
    /// faster run at once ([`RATE_WEIGHT_DOWN`]), a slower one only as they
    /// keep coming ([`RATE_WEIGHT_UP`]).
    pub fn learn(&self, ns: f64, ops: f64) {
        if !(ns > 0.0 && ops > 0.0) {
            return;
        }
        let seen = (ns / ops) as f32;
        let mut bits = self.ns.load(Relaxed);
        loop {
            let old = f32::from_bits(bits);
            let w = if seen < old {
                RATE_WEIGHT_DOWN
            } else {
                RATE_WEIGHT_UP
            };
            let new = (old + w * (seen - old)).to_bits();
            match self.ns.compare_exchange_weak(bits, new, Relaxed, Relaxed) {
                Ok(_) => return,
                Err(now) => bits = now,
            }
        }
    }

    /// `f`, a serial run of `ops` ops, measured if it is a probe or sampled.
    /// The first probe warms the caches and is not counted; the first
    /// counted one replaces the prior.
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

    /// A serial run of `ops` ops starts: timed if it is a probe or sampled
    /// ([`time`](Self::time)), or `always`; [`end`](Self::end) learns from
    /// it. For a run that is no closure.
    pub fn begin(&self, ops: f64, always: bool) -> Option<Timing> {
        let probe = if always {
            None
        } else {
            let probe = self.probing(ops).then(|| self.probes.fetch_add(1, Relaxed));
            if probe.is_none() && !self.sample(ops) {
                return None;
            }
            probe
        };
        Some(Timing {
            probe,
            wide: crate::wide_count(),
            start: crate::clock::Instant::now(),
        })
    }

    /// The run `t` of `ops` ops ([`begin`](Self::begin)) ended; it teaches
    /// nothing if a section went wide inside it.
    pub fn end(&self, t: Timing, ops: f64) {
        let ns = t.start.elapsed().as_nanos() as f64;
        if crate::wide_count() != t.wide {
            return;
        }
        match t.probe {
            Some(0) => {}
            Some(1) if ns > 0.0 && ops > 0.0 => {
                self.ns.store(((ns / ops) as f32).to_bits(), Relaxed)
            }
            _ => self.learn(ns, ops),
        }
    }
}

/// A serial run being measured ([`Rate::begin`]): which probe it is, if one,
/// how many sections this thread had sent wide, and when it started.
#[derive(Debug)]
pub struct Timing {
    probe: Option<u32>,
    wide: u64,
    start: crate::clock::Instant,
}

/// A section's work: `ops` at a learned rate, which learns from the
/// section's serial runs ([`of`](Self::of)), or an estimate in ns
/// ([`ns`](Self::ns)).
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
    /// Serially on the calling thread, measured as a probe or sampled.
    Here,
    /// Serially on the calling thread, measured: it would have gone wide.
    Explore,
    /// On this many threads, the run watched for a stale rate.
    Wide(usize),
}

impl<'r> Work<'r> {
    /// `ns` ns of work: decided on, never measured. For a section whose
    /// work is estimated at a rate other runs learn.
    pub fn ns(ns: f64) -> Self {
        Work {
            ops: ns,
            rate: None,
            ns: None,
        }
    }

    /// `ops` ops at `rate`, which learns from the section's serial runs.
    pub fn of(ops: f64, rate: &'r Rate) -> Self {
        Work {
            ops,
            rate: Some(rate),
            ns: None,
        }
    }

    /// `ns` ns of work in `ops` ops, estimated by the caller (the calls of a
    /// stage, each at its own rate), deciding with `rate`'s probes,
    /// surcharge, exploration and retries. Its serial runs are the caller's
    /// to measure, each part at its own rate.
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

    /// How the section runs in `pieces` pieces at the place `P`: serially
    /// while its rate probes, else on as many threads as it is worth there
    /// with the rate's surcharge on top ([`Place::width_after`]), now and then
    /// serially instead to keep the rate the code's ([`Rate::explore`]), and
    /// now and then wide where only the surcharge keeps it serial
    /// ([`Rate::retry`]; that run is timed).
    pub fn go<P: Place>(&self, pieces: usize) -> Go {
        let Some(rate) = self.rate else {
            return match P::width(self.time(), pieces) {
                k if k >= 2 => Go::Wide(k),
                _ => Go::Here,
            };
        };
        if rate.probing(self.ops) {
            return Go::Here;
        }
        let extra = rate.surcharge();
        let k = P::width_after(self.time(), pieces, extra);
        if k < 2 {
            if extra > 0.0 {
                let k0 = P::width(self.time(), pieces);
                if k0 >= 2 && rate.retry(self.ops, k0) {
                    RETRY.with(|r| r.set(true));
                    return Go::Wide(k0);
                }
            }
            return Go::Here;
        }
        if rate.explore(self.ops, k) {
            return Go::Explore;
        }
        Go::Wide(k)
    }

    /// `f`, the section run on this thread as `go` has it: measured for the
    /// rate as a probe or sampled, or (exploring) every time.
    pub fn serial<R>(&self, go: Go, f: impl FnOnce() -> R) -> R {
        match (self.rate, go) {
            (Some(r), Go::Explore) => r.measure(self.ops, f),
            (Some(r), _) => r.time(self.ops, f),
            (None, _) => f(),
        }
    }

    /// The section as `n` pieces on `k` threads at the place `P`, watched
    /// ([`watch`](Self::watch)).
    pub fn wide<P: Place>(&self, n: usize, k: usize, f: &Piece<'_>) {
        self.watch::<P, _>(n, k, || P::run_width(n, k, f))
    }

    /// `f`, the section run wide in `pieces` pieces on `k` threads at `P`:
    /// timed when sampled (always when [`go`](Self::go) made it a retry);
    /// what it took beyond its prediction teaches the rate's surcharge, and
    /// far faster than its rate predicts ([`RATE_STALE`]) it has the rate
    /// probe anew.
    pub fn watch<P: Place, R>(&self, pieces: usize, k: usize, f: impl FnOnce() -> R) -> R {
        let retry = RETRY.with(|r| r.replace(false));
        let Some(rate) = self.rate.filter(|r| retry || r.sample(self.ops)) else {
            return f();
        };
        let t = crate::clock::Instant::now();
        let r = f();
        let spent = t.elapsed().as_nanos() as f64;
        if spent > 0.0 {
            let predicted = predicted::<P>(self.time(), pieces, k);
            rate.learn_surcharge(spent - predicted);
            if spent < RATE_STALE * (predicted + rate.surcharge()) {
                rate.reprobe();
            }
        }
        r
    }

    /// The section, `n` pieces of this work: serially or on as many threads
    /// of `P` as it is worth, the rate learning from it.
    pub fn run<P: Place>(&self, n: usize, f: &Piece<'_>) {
        match self.go::<P>(n) {
            Go::Wide(k) => self.wide::<P>(n, k, f),
            go => self.serial(go, || (0..n).for_each(f)),
        }
    }
}

/// Whether a run of `work` ns in `pieces` pieces on `k` threads of `P` that
/// took `spent` ns took less than [`RATE_STALE`] of the time predicted for
/// it.
pub(crate) fn stale<P: Place>(spent: f64, work: f64, pieces: usize, k: usize) -> bool {
    spent > 0.0 && spent < RATE_STALE * predicted::<P>(work, pieces, k)
}

/// What `work` ns in `pieces` pieces on `k` threads of `P` should take: its
/// share of the pieces and the pool's section cost.
pub(crate) fn predicted<P: Place>(work: f64, pieces: usize, k: usize) -> f64 {
    let (p, k) = (pieces.max(1), k.clamp(1, pieces.max(1)));
    work * (p.div_ceil(k) as f64 / p as f64) + P::dispatch().cost(k)
}
