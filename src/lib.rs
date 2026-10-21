//! One set of worker threads several libraries share, and sections that go
//! as wide as their work is worth on them.
//!
//! A host lends a [`Pool`] ([`install`]); every section the code it calls
//! opens runs on it, whatever library that code is in: a model's
//! evaluations, a sparse factorization between them, a frequency sweep.
//! One set of threads, kept awake across the short serial stretches between
//! sections ([`Workers`]), instead of pools competing for the cores.
//!
//! A section runs on as many threads as its work is worth, one included:
//! its work in ns (a count of ops at a [`Rate`] learned from the runs)
//! against what a section costs the pool on `k` threads (its [`Dispatch`],
//! which [`Workers`] measure when they start). A section not worth a second
//! thread runs on the calling thread without touching the pool, so small
//! work stays serial and large work goes wide on any machine, with no
//! threshold tuned to one.
//!
//! A section on `k` threads leaves the sections its pieces open the pool's
//! threads over `k` ([`width`]): inside a section as wide as the pool,
//! everything runs serially on its piece's thread; inside a narrower one,
//! the idle threads go to the pieces' own sections.
//!
//! Every result is the same however a section runs: no piece's arithmetic
//! may depend on how its pieces are scheduled or how many threads run them.
//!
//! Where nothing is lent, sections run serially ([`width`], [`run`] and
//! the rest of the crate root), or on the current rayon pool for code that
//! asks for that ([`rayon`], feature `rayon`).

use std::cell::{Cell, RefCell};
use std::sync::Arc;

mod clock;
pub mod constants;
mod dispatch;
mod pool;
mod rate;
#[cfg(feature = "rayon")]
pub mod rayon;
mod workers;

pub use clock::Instant;
pub use constants::{GAIN, PIECES_PER_WORKER, SPIN};
pub use dispatch::Dispatch;
pub use pool::{Piece, Pool, Serial, Spawn, Task};
pub use rate::{Go, Rate, Timing, Work};
pub use workers::Workers;

/// `m` locked: a lock is poisoned only by a panic its section carries on.
pub(crate) fn lock<T>(m: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// A pool lent to the code a host calls, and what a section costs on it.
#[derive(Clone)]
pub struct Lent {
    pub pool: Arc<dyn Pool>,
    /// The pool's [`Pool::dispatch`], or another for a host that wants every
    /// section wide ([`Dispatch::free`]) or none ([`Dispatch::never`]).
    pub dispatch: Dispatch,
}

impl Lent {
    /// `pool` with what a section costs on it.
    pub fn new(pool: Arc<dyn Pool>) -> Self {
        let dispatch = pool.dispatch();
        Lent { pool, dispatch }
    }
}

thread_local! {
    static CURRENT: RefCell<Option<Lent>> = const { RefCell::new(None) };
    /// The threads the sections this thread opens may take: its piece's
    /// share of an enclosing section (`0`: no section encloses it, the
    /// whole pool).
    static SHARE: Cell<usize> = const { Cell::new(0) };
    /// Sections this thread sent wide: a measured run that sent one teaches
    /// its rate nothing ([`Rate::end`]).
    pub(crate) static WIDE: Cell<u64> = const { Cell::new(0) };
    /// Pieces of sections this thread is running.
    static IN_PIECE: Cell<u32> = const { Cell::new(0) };
}

/// Run `f` with `pool` lent: the sections `f` opens run on it, as wide as
/// its [`Pool::dispatch`] makes them worth. The previous one comes back when
/// `f` returns (or unwinds).
pub fn install<R>(pool: Arc<dyn Pool>, f: impl FnOnce() -> R) -> R {
    install_lent(Lent::new(pool), f)
}

/// [`install`] with the section cost given.
pub fn install_lent<R>(lent: Lent, f: impl FnOnce() -> R) -> R {
    struct Restore(Option<Lent>, usize);
    impl Drop for Restore {
        fn drop(&mut self) {
            let p = self.0.take();
            CURRENT.with(|c| *c.borrow_mut() = p);
            SHARE.with(|s| s.set(self.1));
        }
    }
    let _restore = Restore(
        CURRENT.with(|c| c.replace(Some(lent))),
        SHARE.with(|s| s.replace(0)),
    );
    f()
}

/// What is lent on this thread.
pub fn current() -> Option<Lent> {
    CURRENT.with(|c| c.borrow().clone())
}

/// The sections this thread sent wide so far.
pub(crate) fn wide_count() -> u64 {
    WIDE.with(|w| w.get())
}

/// Whether this thread is running a piece of a section.
pub fn in_piece() -> bool {
    IN_PIECE.with(|p| p.get()) > 0
}

/// The threads a section opened here may take of a pool of `threads`.
pub(crate) fn share(threads: usize) -> usize {
    match SHARE.with(|s| s.get()) {
        0 => threads,
        s => s.min(threads),
    }
}

/// `f(i)` as a piece of a section on `k` of `cap` threads: on this thread
/// with `lent` lent (where it runs on another thread than the section's),
/// the sections it opens left `cap / k` threads.
pub(crate) fn piece<R>(lent: Option<&Lent>, inner: usize, f: impl FnOnce() -> R) -> R {
    struct Leave(usize);
    impl Drop for Leave {
        fn drop(&mut self) {
            IN_PIECE.with(|p| p.set(p.get() - 1));
            SHARE.with(|s| s.set(self.0));
        }
    }
    IN_PIECE.with(|p| p.set(p.get() + 1));
    let _leave = Leave(SHARE.with(|s| s.replace(inner)));
    match lent {
        Some(l) if CURRENT.with(|c| c.borrow().is_none()) => install_inner(l, f),
        _ => f(),
    }
}

/// `f`, a piece of a section a library runs itself (a side of a `join`, a
/// task of a `scope` on the lent pool or on rayon): counted as a piece on
/// this thread, `lent` lent here if nothing is (it may run on a worker
/// that is not the section's), and the sections it opens left `share`
/// threads (`None`: what this thread has).
pub fn as_piece<R>(lent: Option<&Lent>, share: Option<usize>, f: impl FnOnce() -> R) -> R {
    let inner = share.unwrap_or_else(|| SHARE.with(|s| s.get()));
    piece(lent, inner, f)
}

/// The threads the sections opened here may take of a pool of `threads`:
/// all of them, or this thread's share of an enclosing section.
pub fn share_of(threads: usize) -> usize {
    share(threads)
}

/// `f` with `lent` lent on this thread, the share already set.
fn install_inner<R>(lent: &Lent, f: impl FnOnce() -> R) -> R {
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            CURRENT.with(|c| *c.borrow_mut() = None);
        }
    }
    CURRENT.with(|c| *c.borrow_mut() = Some(lent.clone()));
    let _restore = Restore;
    f()
}

/// Where sections run: the lent pool or this thread alone ([`Here`]), or
/// the lent pool or the current rayon pool (`rayon::Rayon`).
pub trait Place {
    /// The threads of the place.
    fn threads() -> usize;
    /// What a section costs there.
    fn dispatch() -> Dispatch;
    /// The threads `work` ns of work in `pieces` pieces runs on there.
    fn width(work: f64, pieces: usize) -> usize;
    /// `f(i)` for `i` in `0..n` on `k` threads there (the calling one among
    /// them), serially for `k < 2`.
    fn run_width(n: usize, k: usize, f: &Piece<'_>);
}

/// The lent pool, or this thread alone where none is.
pub struct Here;

impl Place for Here {
    fn threads() -> usize {
        CURRENT.with(|c| c.borrow().as_ref().map_or(1, |l| l.pool.threads().max(1)))
    }

    fn dispatch() -> Dispatch {
        CURRENT.with(|c| {
            c.borrow()
                .as_ref()
                .map_or_else(Dispatch::never, |l| l.dispatch.clone())
        })
    }

    fn width(work: f64, pieces: usize) -> usize {
        if pieces < 2 || !(work > 0.0) {
            return 1;
        }
        CURRENT.with(|c| {
            c.borrow().as_ref().map_or(1, |l| {
                let cap = share(l.pool.threads());
                l.dispatch.width(work, pieces).min(cap).max(1)
            })
        })
    }

    fn run_width(n: usize, k: usize, f: &Piece<'_>) {
        match current() {
            Some(l) if n >= 2 && k >= 2 => lent_run(&l, n, k, f),
            _ => (0..n).for_each(f),
        }
    }
}

/// `f(i)` for `i` in `0..n` on `k` threads of the lent pool `l`.
pub(crate) fn lent_run(l: &Lent, n: usize, k: usize, f: &Piece<'_>) {
    let cap = share(l.pool.threads());
    let k = k.min(n).min(cap).max(1);
    if k < 2 {
        return (0..n).for_each(f);
    }
    let inner = (cap / k).max(1);
    WIDE.with(|w| w.set(w.get() + 1));
    l.pool.run_width(n, k, &|i| piece(Some(l), inner, || f(i)));
}

/// The threads of the lent pool, one where none is.
pub fn threads() -> usize {
    Here::threads()
}

/// What a section costs on the lent pool ([`Dispatch::never`] where none
/// is).
pub fn dispatch() -> Dispatch {
    Here::dispatch()
}

/// The threads `work` ns of work in `pieces` pieces runs on here: as many
/// of the lent pool's as it is worth, within this thread's share of an
/// enclosing section; one (serially) where nothing is lent.
pub fn width(work: f64, pieces: usize) -> usize {
    Here::width(work, pieces)
}

/// Run `f(i)` for `i` in `0..n` on `k` threads of the lent pool (the
/// calling one among them); serially for `k < 2` or where nothing is lent.
pub fn run_width(n: usize, k: usize, f: &Piece<'_>) {
    Here::run_width(n, k, f)
}

/// Run `n` pieces of `work` ns of work together: on as many threads as it
/// is worth here ([`width`]), else serially on this thread.
pub fn run(n: usize, work: f64, f: &Piece<'_>) {
    run_width(n, width(work, n), f)
}

/// Run `n` pieces of `ops` ops at `rate` together, on as many threads as
/// they are worth here; the rate learns from the serial runs (see
/// [`Rate`]).
pub fn run_rated(n: usize, ops: f64, rate: &Rate, f: &Piece<'_>) {
    Work::of(ops, rate).run::<Here>(n, f)
}

/// [`run_width`] of `n` pieces of `work` ns on `k` threads, timed if
/// `timed`: whether it took less than [`RATE_STALE`](constants::RATE_STALE)
/// of the time predicted for it (the rates its work was estimated at are
/// stale).
pub fn run_watched(n: usize, k: usize, work: f64, timed: bool, f: &Piece<'_>) -> bool {
    if !timed {
        run_width(n, k, f);
        return false;
    }
    let t = clock::Instant::now();
    run_width(n, k, f);
    rate::stale::<Here>(t.elapsed().as_nanos() as f64, work, n, k)
}

/// Items per piece for `n` items on `k` threads: about
/// [`PIECES_PER_WORKER`] pieces per thread, all in one piece on one.
pub fn piece_len(n: usize, k: usize) -> usize {
    if k < 2 {
        return n.max(1);
    }
    (n / (k * PIECES_PER_WORKER)).max(1)
}
