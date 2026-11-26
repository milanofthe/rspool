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
//! its work in ns as the caller estimates it (a count of ops at a [`Rate`],
//! or ns, a [`Work`]) against what a section costs the pool on `k` threads
//! (its [`Dispatch`], which [`Workers`] measure when they start). A section
//! not worth a second thread runs on the calling thread without touching
//! the pool, so small work stays serial and large work goes wide on any
//! machine, with no threshold tuned to one.
//!
//! A loop over `n` items ([`ranges_init`], and on it [`ranges`],
//! [`chunks_mut`], [`map_init`], [`map`]) is such a section whose threads
//! claim ranges of the items until none is left, each with a state of its
//! own: the width and the length of the ranges both follow from the
//! caller's estimate of the work ([`Dispatch::ranges`]). A section that
//! writes several slices by range takes them as [`Disjoint`].
//!
//! A section on `k` threads leaves the sections its pieces open the pool's
//! threads over `k` ([`width`]): inside a section as wide as the pool,
//! everything runs serially on its piece's thread; inside a narrower one,
//! the idle threads go to the pieces' own sections.
//!
//! Every result is the same however a section runs: no piece's arithmetic
//! may depend on how its pieces are scheduled or how many threads run them.
//!
//! Where nothing is lent, sections run serially ([`width`], [`Here`] and
//! the rest of the crate root), or on the current rayon pool for code that
//! asks for that ([`rayon`], feature `rayon`).
//!
//! A task that runs outside the call that issued it (a section's helper, a
//! side of a `join` or a task of a `scope` taken from a queue, a rayon piece
//! on another thread) runs with the issuing thread's current [`rsprobe`]
//! node entered, so its scopes and counts land under that node and its time
//! in the node's `busy`. Every section counts at the current node, at level
//! DEBUG: `sections`, `width` (the threads it ran on, one for a serial one)
//! and `serial`. A wide section on [`Workers`] adds `wait`, the time the
//! calling thread waits for its helpers after its own pieces.

use std::cell::{Cell, RefCell};
use std::sync::Arc;

mod clock;
pub mod constants;
mod dispatch;
mod pool;
mod ranges;
mod rate;
#[cfg(feature = "rayon")]
pub mod rayon;
mod workers;

pub use clock::Instant;
pub use constants::{GAIN, SPIN};
pub use dispatch::Dispatch;
pub use pool::{Piece, Pool, Serial, Spawn, Task};
pub use ranges::{chunks_mut, map, map_init, ranges, ranges_init, Disjoint};
pub use rate::{Go, Rate, Work};
pub use workers::Workers;

pub(crate) fn lock_ignoring_poison<T>(m: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
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
    pub(crate) static CURRENT: RefCell<Option<Lent>> = const { RefCell::new(None) };
    static SHARE: Cell<usize> = const { Cell::new(WHOLE_POOL) };
    static PIECES_RUNNING: Cell<u32> = const { Cell::new(0) };
}

pub(crate) const WHOLE_POOL: usize = 0;

#[derive(Clone, Copy)]
pub(crate) struct Origin(rsprobe::Handle);

impl Origin {
    pub(crate) fn here() -> Origin {
        Origin(rsprobe::current())
    }

    pub(crate) fn run<R>(self, f: impl FnOnce() -> R) -> R {
        let _node = self.0.enter();
        f()
    }
}

pub(crate) fn section(n: usize, k: usize, f: &Piece<'_>, wide: impl FnOnce()) {
    if k < 2 {
        (0..n).for_each(f);
    } else {
        wide();
    }
    counted(k);
}

pub(crate) fn counted(k: usize) {
    rsprobe::add!(DEBUG, "sections", 1);
    rsprobe::add!(DEBUG, "width", k);
    if k < 2 {
        rsprobe::add!(DEBUG, "serial", 1);
    }
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
        SHARE.with(|s| s.replace(WHOLE_POOL)),
    );
    f()
}

/// What is lent on this thread.
pub fn current() -> Option<Lent> {
    CURRENT.with(|c| c.borrow().clone())
}

/// Whether a pool is lent on this thread (nothing taken or cloned).
pub fn is_lent() -> bool {
    CURRENT.with(|c| c.borrow().is_some())
}

/// Whether this thread is running a piece of a section.
pub fn in_piece() -> bool {
    PIECES_RUNNING.with(|p| p.get()) > 0
}

pub(crate) fn share(threads: usize) -> usize {
    match SHARE.with(|s| s.get()) {
        WHOLE_POOL => threads,
        s => s.min(threads),
    }
}

pub(crate) fn piece<R>(lent: Option<&Lent>, inner: usize, f: impl FnOnce() -> R) -> R {
    struct Leave(usize);
    impl Drop for Leave {
        fn drop(&mut self) {
            PIECES_RUNNING.with(|p| p.set(p.get() - 1));
            SHARE.with(|s| s.set(self.0));
        }
    }
    PIECES_RUNNING.with(|p| p.set(p.get() + 1));
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
    /// `f` of what a section costs there, borrowed.
    fn with_dispatch<R>(f: impl FnOnce(&Dispatch) -> R) -> R;
    /// The threads a section opened here may take there: all of the
    /// place's, or this thread's share of an enclosing section.
    fn cap() -> usize;
    /// The threads `work` ns of work in `pieces` pieces runs on there, by the
    /// section cost alone.
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

    fn with_dispatch<R>(f: impl FnOnce(&Dispatch) -> R) -> R {
        CURRENT.with(|c| match c.borrow().as_ref() {
            Some(l) => f(&l.dispatch),
            None => f(Dispatch::never_ref()),
        })
    }

    fn cap() -> usize {
        CURRENT.with(|c| c.borrow().as_ref().map_or(1, |l| share(l.pool.threads())))
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
            _ => section(n, 1, f, || ()),
        }
    }
}

pub(crate) fn lent_run(l: &Lent, n: usize, k: usize, f: &Piece<'_>) {
    let cap = share(l.pool.threads());
    let k = k.min(n).min(cap).max(1);
    let inner = (cap / k).max(1);
    section(n, k, f, || {
        l.pool.run_width(n, k, &|i| piece(Some(l), inner, || f(i)))
    });
}

/// The threads of the lent pool this thread may use: all of them, or in a
/// piece of a section its share of them; one where none is lent. What a
/// library sizes its own work by: a piece of a sweep whose every thread
/// takes points has one, as if it ran alone.
pub fn threads() -> usize {
    share(Here::threads()).max(1)
}

/// What a section costs on the lent pool ([`Dispatch::never`] where none
/// is).
pub fn dispatch() -> Dispatch {
    Here::with_dispatch(Dispatch::clone)
}

/// The threads `work` ns of work in `pieces` pieces runs on here: as many
/// of the lent pool's as it is worth, within this thread's share of an
/// enclosing section; one (serially) where nothing is lent.
pub fn width(work: f64, pieces: usize) -> usize {
    Here::width(work, pieces)
}

/// Run `n` pieces of `ops` ops at `rate` together, on as many threads as
/// they are worth here.
pub fn run_rated(n: usize, ops: f64, rate: &Rate, f: &Piece<'_>) {
    Work::of(ops, rate).run::<Here>(n, f)
}
