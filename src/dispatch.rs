//! What a section costs on a pool, and so how many of its threads a section
//! of some work is worth.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::Arc;

use crate::constants::{
    ASSUMED_HELPER_NS, ASSUMED_SECTION_NS, CALIBRATION_LINES, CALIBRATION_PIECE_NS,
    CALIBRATION_ROUNDS, GAIN,
};
use crate::Piece;

/// What a section costs on a pool: `cost(k)` ns for a section on `k`
/// threads, the calling one among them. A section of `work` ns of serial
/// work in `p` equal pieces takes about `work * ceil(p / k) / p + cost(k)`
/// on `k` threads, and goes as wide as that is shortest ([`width`](
/// Self::width)).
#[derive(Clone, Debug)]
pub struct Dispatch {
    /// `cost[k]` for `k` threads; `cost[0]` and `cost[1]` are zero.
    cost: Arc<[f64]>,
    /// The least work any section goes wide with ([`break_even`](
    /// Self::break_even)).
    least: f64,
}

impl Dispatch {
    /// The costs `cost[k]` of a section on `k` threads, for `k` up to
    /// `cost.len() - 1` (`cost[0]` and `cost[1]` are not read).
    pub fn new(cost: Vec<f64>) -> Self {
        let mut cost = cost;
        if cost.len() < 2 {
            cost.resize(2, 0.0);
        }
        for c in cost.iter_mut().take(2) {
            *c = 0.0;
        }
        // Over many pieces a section on `k` threads takes `work / k +
        // cost[k]`, which saves `GAIN` of `work` from this work on.
        let least = (2..cost.len())
            .filter_map(|k| {
                let saved = 1.0 - GAIN - 1.0 / k as f64;
                (saved > 0.0).then(|| cost[k].max(0.0) / saved)
            })
            .fold(f64::INFINITY, f64::min);
        Dispatch {
            cost: cost.into(),
            least,
        }
    }

    /// Sections cost nothing on `threads` threads: every section of two
    /// pieces or more goes wide (tests, and hosts that know better).
    pub fn free(threads: usize) -> Self {
        Dispatch::new(vec![0.0; threads.max(1) + 1])
    }

    /// No section goes wide.
    pub fn never() -> Self {
        Dispatch::new(vec![0.0; 2])
    }

    /// What a pool of `threads` threads that measures nothing costs:
    /// [`ASSUMED_SECTION_NS`] plus [`ASSUMED_HELPER_NS`] per helper.
    pub fn assumed(threads: usize) -> Self {
        let cost = (0..=threads.max(1))
            .map(|k| ASSUMED_SECTION_NS + ASSUMED_HELPER_NS * k.saturating_sub(1) as f64)
            .collect();
        Dispatch::new(cost)
    }

    /// The most threads a section is costed for.
    pub fn threads(&self) -> usize {
        self.cost.len().saturating_sub(1).max(1)
    }

    /// What a section costs on `k` threads, in ns.
    pub fn cost(&self, k: usize) -> f64 {
        if k < 2 {
            return 0.0;
        }
        self.cost.get(k).copied().unwrap_or(f64::INFINITY)
    }

    /// The least work, in ns, with which a section of any number of pieces
    /// goes wide: below it every section runs serially.
    pub fn break_even(&self) -> f64 {
        self.least
    }

    /// The least work, in ns, with which a section of `pieces` equal pieces
    /// goes wide (infinite where none does).
    pub fn least(&self, pieces: usize) -> f64 {
        let p = pieces.max(1) as f64;
        (2..=pieces.min(self.threads()))
            .filter_map(|k| {
                let saved = 1.0 - GAIN - pieces.div_ceil(k) as f64 / p;
                (saved > 0.0).then(|| self.cost[k].max(0.0) / saved)
            })
            .fold(f64::INFINITY, f64::min)
    }

    /// The threads a section of `work` ns in `pieces` pieces runs on: the
    /// count it takes the shortest on, if that saves at least [`GAIN`] of
    /// the serial time; one (serially) otherwise.
    pub fn width(&self, work: f64, pieces: usize) -> usize {
        self.width_after(work, pieces, 0.0)
    }

    /// [`width`](Self::width) for a section that pays `extra` ns more on
    /// two threads or more (a hand-off into the pool it runs on).
    pub fn width_after(&self, work: f64, pieces: usize, extra: f64) -> usize {
        if !(work > 0.0) || pieces < 2 || work < self.least {
            return 1;
        }
        if work.is_infinite() {
            // worth any section: as wide as the pieces and the threads go
            return pieces.min(self.threads());
        }
        let p = pieces as f64;
        let mut best = (1, work);
        for k in 2..=pieces.min(self.threads()) {
            let t = work * pieces.div_ceil(k) as f64 / p + self.cost[k] + extra;
            if t < best.1 {
                best = (k, t);
            }
        }
        if best.1 <= (1.0 - GAIN) * work {
            best.0
        } else {
            1
        }
    }

    /// What a section costs on a pool of `threads` threads, measured:
    /// `run(k, piece)` runs `piece(i)` for `i` in `0..k` on `k` threads.
    /// Per thread count the median of [`CALIBRATION_ROUNDS`] sections of as
    /// many pieces, each writing [`CALIBRATION_LINES`] cache lines (the
    /// hand-over of a piece's output a real section pays) and then busy for
    /// [`CALIBRATION_PIECE_NS`], less what one piece costs alone; never less
    /// than with fewer threads. A piece lasts long enough for the pool's
    /// other threads to take theirs: on a work-stealing pool whose caller
    /// runs the pieces nobody took, instant pieces would all run on the
    /// caller and the section would look free. Run after as many sections on
    /// all threads, so it finds the workers awake as a solver's sections do.
    pub fn measure(threads: usize, run: &dyn Fn(usize, &Piece<'_>)) -> Dispatch {
        if threads < 2 {
            return Dispatch::never();
        }
        #[repr(align(64))]
        struct Line(AtomicU64);
        let lines: Vec<Line> = (0..threads * CALIBRATION_LINES)
            .map(|_| Line(AtomicU64::new(0)))
            .collect();
        let busy = std::time::Duration::from_nanos(CALIBRATION_PIECE_NS);
        let piece = |i: usize| {
            let t = crate::clock::Instant::now();
            for l in &lines[i * CALIBRATION_LINES..(i + 1) * CALIBRATION_LINES] {
                l.0.fetch_add(1, Relaxed);
            }
            // an inert clock (no OS clock) reads zero: no busy wait there
            while cfg!(not(target_arch = "wasm32")) && t.elapsed() < busy {
                std::hint::spin_loop();
            }
        };
        let median = |f: &dyn Fn()| {
            let mut t: Vec<f64> = (0..CALIBRATION_ROUNDS)
                .map(|_| {
                    let s = crate::clock::Instant::now();
                    f();
                    s.elapsed().as_nanos() as f64
                })
                .collect();
            t.sort_by(f64::total_cmp);
            t[t.len() / 2]
        };
        for _ in 0..CALIBRATION_ROUNDS {
            run(threads, &piece);
        }
        let one = median(&|| piece(0));
        let mut cost = vec![0.0; threads + 1];
        for k in 2..=threads {
            let c = median(&|| run(k, &piece)) - one;
            cost[k] = c.max(cost[k - 1]);
        }
        Dispatch::new(cost)
    }
}
