//! Sections on the current rayon pool where no pool is lent: for code that
//! runs on rayon by itself (a library's own thread policy) and on a host's
//! workers when it lends them. What a section costs a rayon pool is
//! measured once per process and pool size, on the pool's first use.

use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use crate::{
    current, lent_run, lock_ignoring_poison, section, share, Dispatch, Lent, Origin, Piece, Place,
    WHOLE_POOL,
};

/// The lent pool, or the current rayon pool where none is.
pub struct Rayon;

thread_local! {
    static RAYON: RefCell<Option<(usize, bool, Dispatch)>> = const { RefCell::new(None) };
}

static MEASURED: std::sync::Mutex<Vec<(usize, bool, Dispatch)>> = std::sync::Mutex::new(Vec::new());

static KEPT: std::sync::Mutex<Vec<(bool, usize, Dispatch)>> = std::sync::Mutex::new(Vec::new());

/// [`Dispatch::free`] (`free`) or [`Dispatch::assumed`] on `threads`
/// threads, made once.
pub fn kept(free: bool, threads: usize) -> Dispatch {
    let mut k = lock_ignoring_poison(&KEPT);
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
        crate::CURRENT
            .with(|c| c.borrow().as_ref().map(|l| l.pool.threads()))
            .unwrap_or_else(::rayon::current_num_threads)
            .max(1)
    }

    fn with_dispatch<R>(f: impl FnOnce(&Dispatch) -> R) -> R {
        crate::CURRENT.with(|c| {
            let current = c.borrow();
            if let Some(l) = current.as_ref() {
                return f(&l.dispatch);
            }
            drop(current);
            let threads = ::rayon::current_num_threads();
            if threads < 2 {
                return f(Dispatch::never_ref());
            }
            with_rayon(threads, f)
        })
    }

    fn cap() -> usize {
        let lent = crate::CURRENT.with(|c| c.borrow().as_ref().map(|l| l.pool.threads()));
        share(lent.unwrap_or_else(::rayon::current_num_threads)).max(1)
    }

    fn width(work: f64, pieces: usize) -> usize {
        if pieces < 2 || !(work > 0.0) {
            return 1;
        }
        let lent = crate::CURRENT.with(|c| {
            c.borrow().as_ref().map(|l| {
                let cap = share(l.pool.threads());
                l.dispatch.width(work, pieces).min(cap).max(1)
            })
        });
        if let Some(k) = lent {
            return k;
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
            return section(n, 1, f, || ());
        }
        match current() {
            Some(l) => lent_run(&l, n, k, f),
            None => rayon_run(n, k, f),
        }
    }
}

fn with_rayon<R>(threads: usize, f: impl FnOnce(&Dispatch) -> R) -> R {
    let inside = ::rayon::current_thread_index().is_some();
    let cached =
        RAYON.with(|c| matches!(&*c.borrow(), Some((t, i, _)) if *t == threads && *i == inside));
    if !cached {
        if let Some(d) = measured(threads, inside) {
            RAYON.with(|c| *c.borrow_mut() = Some((threads, inside, d)));
        }
    }
    RAYON.with(|c| match &*c.borrow() {
        Some((t, i, d)) if *t == threads && *i == inside => f(d),
        _ => f(&kept(false, threads)),
    })
}

fn measured(threads: usize, inside: bool) -> Option<Dispatch> {
    if let Some(d) = found(threads, inside) {
        return Some(d);
    }
    if crate::in_piece() {
        return None;
    }
    let measured_unlocked = crate::piece(None, 1, || {
        Dispatch::measure(threads, &|k, p| rayon_pieces(k, k, 1, p))
    });
    let mut m = lock_ignoring_poison(&MEASURED);
    if let Some(d) = in_list(&m, threads, inside) {
        return Some(d);
    }
    m.push((threads, inside, measured_unlocked.clone()));
    Some(measured_unlocked)
}

fn found(threads: usize, inside: bool) -> Option<Dispatch> {
    in_list(&lock_ignoring_poison(&MEASURED), threads, inside)
}

fn in_list(m: &[(usize, bool, Dispatch)], threads: usize, inside: bool) -> Option<Dispatch> {
    m.iter()
        .find(|(t, i, _)| *t == threads && *i == inside)
        .map(|e| e.2.clone())
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
/// (assumed before), entering it paid on top: a piece handed to one of its
/// workers, what a section of two costs there. Nothing is measured: a
/// measurement runs sections on a pool, which a caller on its way into
/// another must not (a worker of a foreign pool would take that pool's own
/// work while it waits).
pub fn width_on(threads: usize, work: f64, pieces: usize) -> usize {
    if threads < 2 || pieces < 2 || !(work > 0.0) {
        return 1;
    }
    if current().is_some() || ::rayon::current_num_threads() == threads {
        return width(work, pieces).min(threads);
    }
    let there = found(threads, true).unwrap_or_else(|| kept(false, threads));
    there.width_after(work, pieces, there.cost(2)).min(threads)
}

fn rayon_run(n: usize, k: usize, f: &Piece<'_>) {
    let cap = share(::rayon::current_num_threads());
    let k = k.min(n).min(cap).max(1);
    section(n, k, f, || rayon_pieces(n, k, WHOLE_POOL, f))
}

fn rayon_pieces(n: usize, k: usize, inner: usize, f: &Piece<'_>) {
    use ::rayon::prelude::*;
    let lent: Option<&Lent> = None;
    let origin = Origin::here();
    if k >= n {
        (0..n)
            .into_par_iter()
            .with_max_len(1)
            .for_each(|i| origin.run(|| crate::piece(lent, inner, || f(i))));
    } else {
        let next = AtomicUsize::new(0);
        (0..k).into_par_iter().with_max_len(1).for_each(|_| {
            origin.run(|| {
                crate::piece(lent, inner, || loop {
                    let i = next.fetch_add(1, Relaxed);
                    if i >= n {
                        break;
                    }
                    f(i);
                })
            })
        });
    }
}

/// What a section costs on the current rayon pool, whatever is lent: for
/// code that runs on rayon by itself even where a pool is lent.
pub fn pool_dispatch() -> Dispatch {
    let threads = ::rayon::current_num_threads();
    if threads < 2 {
        return kept(true, 1);
    }
    with_rayon(threads, Dispatch::clone)
}

/// The threads this thread may use here: the lent pool's (in a piece of a
/// section its share of them, see [`crate::threads`]), or the current
/// rayon pool's.
pub fn threads() -> usize {
    share(Rayon::threads()).max(1)
}

/// What a section costs here: on the lent pool, or on the current rayon
/// pool (measured on its first use).
pub fn dispatch() -> Dispatch {
    Rayon::with_dispatch(Dispatch::clone)
}

/// The threads `work` ns of work in `pieces` pieces runs on here.
pub fn width(work: f64, pieces: usize) -> usize {
    Rayon::width(work, pieces)
}

/// Run `f(i)` for `i` in `0..n` on `k` threads here, serially for `k < 2`.
pub fn run_width(n: usize, k: usize, f: &Piece<'_>) {
    Rayon::run_width(n, k, f)
}
