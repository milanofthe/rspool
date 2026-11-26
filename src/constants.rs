//! Every number rspool's behavior depends on. Time only: a section computes
//! the same on any number of threads, so none of these changes a result.
//!
//! No value here is a fact about one machine: what a section costs is
//! measured on the pool that runs it ([`Dispatch::measure`](
//! crate::Dispatch::measure)), and what an op costs is the caller's
//! estimate ([`Rate`](crate::Rate)). The values below are margins, budgets
//! of the measurement, and priors for pools that do not measure themselves.

/// A section goes wide only if that saves at least this fraction of its
/// serial time ([`Dispatch::width`](crate::Dispatch::width)): margin for
/// the scatter of the measured costs and the callers' estimates, so that a
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
/// after as many to wake the workers. More rounds do not make it steadier:
/// on 12 Linux cores, 20 pools of 12 workers each, the median cost of a
/// section on all of them was 16.0, 18.8, 17.6 and 16.0 us for 8, 16, 32
/// and 64 rounds, from pool to pool a factor of two to four apart at any
/// of them, while building a pool took 1.2, 2.3, 4.1 and 7.3 ms.
pub const CALIBRATION_ROUNDS: usize = 16;
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

/// How long [`Workers`](crate::Workers) keep looking for work before they
/// sleep: longer than the serial stretch between a solver's sections, so
/// they reach awake workers, and short enough that an idle pool costs
/// nothing.
pub const SPIN: std::time::Duration = std::time::Duration::from_micros(300);
/// A thread waiting for work yields its core once every this many empty
/// looks, and spins between them: rarely enough to answer fast, often
/// enough to let another thread on the core run.
pub const SPIN_YIELD_EVERY: u32 = 64;
