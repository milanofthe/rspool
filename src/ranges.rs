//! The loop over the items `0..n`: the threads of a section claim ranges of
//! them from a shared cursor until none is left, each with a state of its
//! own made at its first range. How wide the section goes and how long its
//! ranges are both follow from the caller's estimate of the work
//! ([`Dispatch::width`](crate::Dispatch::width),
//! [`Dispatch::ranges`](crate::Dispatch::ranges)); the caller knows no
//! thread count and no piece size. Claimed as the threads free up, unequal
//! items balance by themselves.

use std::marker::PhantomData;
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use crate::{Place, Work};

/// `f(state, range)` over ranges that cover `0..n` once, `work` all of
/// them, at the place `P`: serially as one range where the work is not
/// worth a second thread, else on as many threads as it is worth, each
/// claiming the next range as it frees up. A thread's `state` is made by
/// `init` when it claims its first range and goes with it to all of its
/// ranges, so there is one per thread that claimed any. A range's result
/// must not depend on what its state saw before (scratch, or a copy of a
/// shared origin): which thread takes which range is not fixed.
///
/// An item is the smallest unit worth running alone: a range may hold a
/// single one. A caller whose code works on groups (the lanes of a SIMD
/// pass, the entries of a block) passes the groups as items, not their
/// members.
pub fn ranges_init<P: Place, S>(
    n: usize,
    work: Work<'_>,
    init: impl Fn() -> S + Sync,
    f: impl Fn(&mut S, Range<usize>) + Sync,
) {
    if n == 0 {
        return;
    }
    let time = work.time();
    let k = P::width(time, n);
    if k < 2 {
        f(&mut init(), 0..n);
        crate::counted(1);
        return;
    }
    let len = n.div_ceil(P::with_dispatch(|d| d.ranges(time, n, k)));
    let next = AtomicUsize::new(0);
    P::run_width(k, k, &|_| {
        let mut state = None;
        loop {
            let start = next.fetch_add(len, Relaxed);
            if start >= n {
                return;
            }
            f(state.get_or_insert_with(&init), start..(start + len).min(n));
        }
    });
}

/// [`ranges_init`] without a state.
pub fn ranges<P: Place>(n: usize, work: Work<'_>, f: impl Fn(Range<usize>) + Sync) {
    ranges_init::<P, ()>(n, work, || (), |_, r| f(r))
}

/// `f(i, part)` over consecutive parts of `v` that cover it once, `v` taken
/// as items of `item` entries (the last one shorter): `part` holds whole
/// items, the first of them item `i`; `work` all of them ([`ranges`]).
pub fn chunks_mut<P: Place, T: Send>(
    v: &mut [T],
    item: usize,
    work: Work<'_>,
    f: impl Fn(usize, &mut [T]) + Sync,
) {
    let (item, len) = (item.max(1), v.len());
    let v = Disjoint::new(v);
    ranges::<P>(len.div_ceil(item), work, |r| {
        // SAFETY: the ranges of a section are disjoint.
        let part = unsafe { v.range(r.start * item..(r.end * item).min(len)) };
        f(r.start, part)
    })
}

/// `f(state, i)` for `i` in `0..n`, in order, `work` all of them; the
/// states as in [`ranges_init`].
pub fn map_init<P: Place, S, R: Send>(
    n: usize,
    work: Work<'_>,
    init: impl Fn() -> S + Sync,
    f: impl Fn(&mut S, usize) -> R + Sync,
) -> Vec<R> {
    let mut out = Vec::with_capacity(n);
    let slots = Disjoint::new(&mut out.spare_capacity_mut()[..n]);
    ranges_init::<P, S>(n, work, init, |s, r| {
        // SAFETY: the ranges of a section are disjoint.
        let part = unsafe { slots.range(r.clone()) };
        for (slot, i) in part.iter_mut().zip(r) {
            slot.write(f(s, i));
        }
    });
    // SAFETY: every one of the `n` slots is written above (a panic unwinds
    // past this, the results written leak).
    unsafe { out.set_len(n) };
    out
}

/// [`map_init`] without a state.
pub fn map<P: Place, R: Send>(n: usize, work: Work<'_>, f: impl Fn(usize) -> R + Sync) -> Vec<R> {
    map_init::<P, (), R>(n, work, || (), |_, i| f(i))
}

/// A slice the ranges of a section write, each its own entries: for a
/// section that writes several slices at once ([`chunks_mut`] covers one).
/// The one place disjoint writes of a section are unsafe.
pub struct Disjoint<'a, T> {
    ptr: *mut T,
    len: usize,
    _v: PhantomData<&'a mut [T]>,
}

// SAFETY: the threads of a section reach only the entries they claimed
// (see `range`), which they may move between them.
unsafe impl<T: Send> Send for Disjoint<'_, T> {}
// SAFETY: as above.
unsafe impl<T: Send> Sync for Disjoint<'_, T> {}

impl<'a, T> Disjoint<'a, T> {
    /// `v`, borrowed for as long as this lives.
    pub fn new(v: &'a mut [T]) -> Self {
        Disjoint {
            ptr: v.as_mut_ptr(),
            len: v.len(),
            _v: PhantomData,
        }
    }

    /// The entries `r` (in bounds, or a panic).
    ///
    /// # Safety
    /// No other slice of entries of `r` is in use while this one is.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn range(&self, r: Range<usize>) -> &mut [T] {
        assert!(
            r.start <= r.end && r.end <= self.len,
            "{r:?} of {}",
            self.len
        );
        // SAFETY: in bounds; exclusive by the caller's contract.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.add(r.start), r.len()) }
    }
}
