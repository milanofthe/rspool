//! Sections on the current rayon pool where no pool is lent: for code that
//! runs on rayon by itself (a library's own thread policy) and on a host's
//! workers when it lends them. What a section costs a rayon pool is
//! measured once per process and pool size, on the pool's first use.

use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use crate::{current, lent_run, lock, share, Dispatch, Lent, Piece, Place, Rate, Work};

/// The lent pool, or the current rayon pool where none is.
pub struct Rayon;

thread_local! {
    /// The section cost of the rayon pool this thread last worked in, by
    /// its thread count and whether this thread is one of its workers.
    static RAYON: RefCell<Option<(usize, bool, Dispatch)>> = const { RefCell::new(None) };
}

/// The rayon section costs measured in this process, by thread count and
/// whether measured from a worker.
static MEASURED: std::sync::Mutex<Vec<(usize, bool, Dispatch)>> = std::sync::Mutex::new(Vec::new());

/// The section costs made from a formula, kept once made, so a gate that
/// needs one allocates nothing.
static KEPT: std::sync::Mutex<Vec<(bool, usize, Dispatch)>> = std::sync::Mutex::new(Vec::new());

/// [`Dispatch::free`] (`free`) or [`Dispatch::assumed`] on `threads`
/// threads, made once.
pub fn kept(free: bool, threads: usize) -> Dispatch {
    let mut k = lock(&KEPT);
    if let Some(e) = k.iter().find(|e| e.0 == free && e.1 == threads) {
        return e.2.clone();
    }
    let d = if free {
        Dispatch::free(threads)
    } else {
        Dispatch::assumed(threads)
    };
    k.push((free, threads, d.clone()));
    d
}

impl Place for Rayon {
    fn threads() -> usize {
        match current() {
            Some(l) => l.pool.threads().max(1),
            None => ::rayon::current_num_threads().max(1),
        }
    }

    fn dispatch() -> Dispatch {
        if let Some(l) = current() {
            return l.dispatch;
        }
        let threads = ::rayon::current_num_threads();
        if threads < 2 {
            return kept(true, 1);
        }
        with_rayon(threads, Dispatch::clone)
    }

    fn width(work: f64, pieces: usize) -> usize {
        if pieces < 2 || !(work > 0.0) {
            return 1;
        }
        if let Some(l) = current() {
            let cap = share(l.pool.threads());
            return l.dispatch.width(work, pieces).min(cap).max(1);
        }
        let threads = ::rayon::current_num_threads();
        if threads < 2 {
            return 1;
        }
        let cap = share(threads);
        with_rayon(threads, |d| d.width(work, pieces))
            .min(cap)
            .max(1)
    }

    fn run_width(n: usize, k: usize, f: &Piece<'_>) {
        if n < 2 || k < 2 {
            return (0..n).for_each(f);
        }
        match current() {
            Some(l) => lent_run(&l, n, k, f),
            None => rayon_run(n, k, f),
        }
    }
}

/// `f` of the section cost of the current rayon pool of `threads` threads.
fn with_rayon<R>(threads: usize, f: impl FnOnce(&Dispatch) -> R) -> R {
    let inside = ::rayon::current_thread_index().is_some();
    let cached =
        RAYON.with(|c| matches!(&*c.borrow(), Some((t, i, _)) if *t == threads && *i == inside));
    if !cached {
        match measured(threads, inside) {
            Some(d) => RAYON.with(|c| *c.borrow_mut() = Some((threads, inside, d))),
            None => return f(&kept(false, threads)),
        }
    }
    RAYON.with(|c| match &*c.borrow() {
        Some((_, _, d)) => f(d),
        None => unreachable!("cached above"),
    })
}

/// The section cost of a rayon pool of `threads` threads, from inside it or
/// not: measured once per process, here and now when nobody did yet. `None`
/// inside a piece of a section, where a measurement would time the
/// section's own work.
fn measured(threads: usize, inside: bool) -> Option<Dispatch> {
    let find = |m: &[(usize, bool, Dispatch)]| {
        m.iter()
            .find(|(t, i, _)| *t == threads && *i == inside)
            .map(|e| e.2.clone())
    };
    if let Some(d) = find(&lock(&MEASURED)) {
        return Some(d);
    }
    if crate::in_piece() {
        return None;
    }
    // not under the lock: a thread waiting in a rayon section runs other
    // work meanwhile, which may ask here too
    let d = crate::piece(None, 1, || {
        Dispatch::measure(threads, &|k, p| rayon_pieces(k, k, 1, p))
    });
    let mut m = lock(&MEASURED);
    if let Some(d) = find(&m) {
        return Some(d);
    }
    m.push((threads, inside, d.clone()));
    Some(d)
}

/// Measure what a section costs on the current rayon pool now, if nobody
/// did yet: for a pool just built, whose workers nothing else keeps busy.
pub fn prepare() {
    let threads = ::rayon::current_num_threads();
    if current().is_none() && threads >= 2 {
        with_rayon(threads, |_| ());
    }
}

/// The threads `work` ns in `pieces` pieces runs on in a rayon pool of
/// `threads` threads the caller is about to enter: as [`width`] where that
/// pool is the lent one or the current rayon pool, else as a rayon pool of
/// as many threads measured when one was last used from its workers
/// (assumed before), the section into it from here paid on top.
pub fn width_on(threads: usize, work: f64, pieces: usize) -> usize {
    if threads < 2 || pieces < 2 || !(work > 0.0) {
        return 1;
    }
    if current().is_some() || ::rayon::current_num_threads() == threads {
        return width(work, pieces).min(threads);
    }
    let measured = lock(&MEASURED)
        .iter()
        .find(|(t, inside, _)| *t == threads && *inside)
        .map(|e| e.2.clone());
    let enter = dispatch().cost(2);
    measured
        .unwrap_or_else(|| kept(false, threads))
        .width_after(work, pieces, enter)
        .min(threads)
}

/// `f(i)` for `i` in `0..n` on `k` threads of the current rayon pool.
fn rayon_run(n: usize, k: usize, f: &Piece<'_>) {
    let cap = share(::rayon::current_num_threads());
    let k = k.min(n).min(cap).max(1);
    if k < 2 {
        return (0..n).for_each(f);
    }
    crate::WIDE.with(|w| w.set(w.get() + 1));
    rayon_pieces(n, k, (cap / k).max(1), f)
}

/// The pieces of a rayon section, each with its share `inner`.
fn rayon_pieces(n: usize, k: usize, inner: usize, f: &Piece<'_>) {
    use ::rayon::prelude::*;
    let lent: Option<&Lent> = None;
    if k >= n {
        (0..n)
            .into_par_iter()
            .with_max_len(1)
            .for_each(|i| crate::piece(lent, inner, || f(i)));
    } else {
        let next = AtomicUsize::new(0);
        (0..k).into_par_iter().with_max_len(1).for_each(|_| {
            crate::piece(lent, inner, || loop {
                let i = next.fetch_add(1, Relaxed);
                if i >= n {
                    break;
                }
                f(i);
            })
        });
    }
}

/// The threads here: the lent pool's, or the current rayon pool's.
pub fn threads() -> usize {
    Rayon::threads()
}

/// What a section costs here: on the lent pool, or on the current rayon
/// pool (measured on its first use).
pub fn dispatch() -> Dispatch {
    Rayon::dispatch()
}

/// The threads `work` ns of work in `pieces` pieces runs on here.
pub fn width(work: f64, pieces: usize) -> usize {
    Rayon::width(work, pieces)
}

/// Run `f(i)` for `i` in `0..n` on `k` threads here, serially for `k < 2`.
pub fn run_width(n: usize, k: usize, f: &Piece<'_>) {
    Rayon::run_width(n, k, f)
}

/// Run `n` pieces of `work` ns together, on as many threads as it is worth
/// here.
pub fn run(n: usize, work: f64, f: &Piece<'_>) {
    run_width(n, width(work, n), f)
}

/// Run `n` pieces of `ops` ops at `rate`, the rate learning from the runs.
pub fn run_rated(n: usize, ops: f64, rate: &Rate, f: &Piece<'_>) {
    Work::of(ops, rate).run::<Rayon>(n, f)
}

/// [`run_width`] timed if `timed`: whether it took less than
/// [`RATE_STALE`](crate::constants::RATE_STALE) of its predicted time.
pub fn run_watched(n: usize, k: usize, work: f64, timed: bool, f: &Piece<'_>) -> bool {
    if !timed {
        run_width(n, k, f);
        return false;
    }
    let t = crate::clock::Instant::now();
    run_width(n, k, f);
    crate::rate::stale::<Rayon>(t.elapsed().as_nanos() as f64, work, n, k)
}
