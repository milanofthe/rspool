//! Every number rspool's behavior depends on. Time only: a section computes
//! the same on any number of threads, so none of these changes a result.
//!
//! No value here is a fact about one machine: what a section costs is
//! measured on the pool that runs it ([`Dispatch::measure`](
//! crate::Dispatch::measure)), and what an op costs is learned from the runs
//! ([`Rate`](crate::Rate)). The values below are margins, budgets of the
//! measurement, and priors for pools that do not measure themselves.

/// A section goes wide only if that saves at least this fraction of its
/// serial time ([`Dispatch::width`](crate::Dispatch::width)): margin for
/// the scatter of the measured costs and the learned rates, so that a
/// section near break-even stays serial instead of flipping between the two.
pub const GAIN: f64 = 0.2;

/// What a section costs a pool that measures nothing
/// ([`Dispatch::assumed`](crate::Dispatch::assumed)), in ns: this, plus
/// [`ASSUMED_HELPER_NS`] per thread beyond the calling one. Machine: the
/// wake and hand-off of a desktop CPU's pool, a prior for pools that do not
/// say better through [`Pool::dispatch`](crate::Pool::dispatch).
pub const ASSUMED_SECTION_NS: f64 = 2_000.0;
/// See [`ASSUMED_SECTION_NS`] (ns per helper thread).
pub const ASSUMED_HELPER_NS: f64 = 1_000.0;

/// Sections timed per thread count when a pool's section cost is measured
/// ([`Dispatch::measure`](crate::Dispatch::measure), the median taken),
/// after as many to wake the workers: a stable median, about a millisecond
/// per thread count.
pub const CALIBRATION_ROUNDS: usize = 64;
/// How long each piece of a calibration section is busy, in ns: long
/// enough for an awake pool's threads to take their pieces before the
/// caller has run them all (a work-stealing pool runs a piece nobody took
/// on the caller, and instant pieces would make every section look free),
/// and about the least work a section is ever worth.
pub const CALIBRATION_PIECE_NS: u64 = 5_000;
/// Cache lines each piece of a calibration section writes: the calibration
/// pays the hand-over of a piece's output a real section pays, not only the
/// queue's.
pub const CALIBRATION_LINES: usize = 8;

/// A [`Rate`](crate::Rate) measures one in this many of its serial runs:
/// two clock reads, a few ns a run on average even for the smallest
/// sections, whose rate decides as much as a large one's.
pub const RATE_SAMPLE_EVERY: u32 = 16;
/// The first runs of a [`Rate`](crate::Rate)'s sections run serially and
/// are measured, the first of them (cold caches) not counted: the rate a
/// section's width is decided on comes from runs on one thread, never from
/// wide ones, whose time mixes the work with what the hand-off cost; a rate
/// learned from them would keep its sections wide or serial whatever the
/// work.
pub const RATE_PROBES: u32 = 4;
/// A wide run (one in [`RATE_SAMPLE_EVERY`] is timed) that takes less than
/// this fraction of the time its rate predicts shows the rate stale (the
/// code got cheaper since: a body specialized or compiled, warm caches):
/// the rate probes anew.
pub const RATE_STALE: f64 = 0.5;
/// A section that would run on `k` threads runs serially, measured, once
/// in this many times `k - 1` runs: its rate stays the code's as it is now,
/// for about a percent of the section's time.
pub const RATE_EXPLORE: u32 = 100;
/// The weight of a new measurement in a rate's running mean toward less
/// time: interruptions only ever add time to a run, so the mean follows a
/// faster run at once and a slower one only as they keep coming
/// ([`RATE_WEIGHT_UP`]).
pub const RATE_WEIGHT_DOWN: f32 = 0.5;
/// See [`RATE_WEIGHT_DOWN`].
pub const RATE_WEIGHT_UP: f32 = 0.1;

/// Pieces of work per thread a section of many items is cut into
/// ([`piece_len`](crate::piece_len)): about four balance items of unequal
/// cost, and a piece still carries enough work to pay for its hand-off.
pub const PIECES_PER_WORKER: usize = 4;

/// How long [`Workers`](crate::Workers) keep looking for work before they
/// sleep: longer than the serial stretch between a solver's sections, so
/// they reach awake workers, and short enough that an idle pool costs
/// nothing.
pub const SPIN: std::time::Duration = std::time::Duration::from_micros(300);
/// A thread waiting for work yields its core once every this many empty
/// looks, and spins between them: rarely enough to answer fast, often
/// enough to let another thread on the core run.
pub const SPIN_YIELD_EVERY: u32 = 64;
