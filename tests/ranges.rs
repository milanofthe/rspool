//! The loop over items: every index once, one state per thread that claimed
//! a range, serial where the cost model says so, the cut the rule gives,
//! nesting as every section nests, and panics reaching the caller.

use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::sync::{Arc, Mutex};
use std::thread::ThreadId;

use rspool::{Dispatch, Here, Lent, Pool, Work, Workers};

/// `f` with `threads` workers lent at the section cost `dispatch`.
fn lent(threads: usize, dispatch: Dispatch, f: impl FnOnce()) {
    let pool: Arc<dyn Pool> = Arc::new(Workers::new(threads).unwrap());
    rspool::install_lent(Lent { pool, dispatch }, f)
}

/// Busy for about `ns`, so the other threads of a section get to claim.
fn busy(ns: u64) {
    let t = std::time::Instant::now();
    while t.elapsed().as_nanos() < ns as u128 {
        std::hint::spin_loop();
    }
}

/// 1 us per section plus 1 us per helper on up to `threads` threads.
fn costed(threads: usize) -> Dispatch {
    Dispatch::new((0..=threads).map(|k| 1e3 * k as f64).collect())
}

#[test]
fn every_index_is_visited_once() {
    for dispatch in [Dispatch::free(4), costed(4), Dispatch::never()] {
        lent(4, dispatch, || {
            for n in [0, 1, 2, 7, 1000] {
                for work in [1.0, 1e5, 1e8, f64::INFINITY] {
                    let hits: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
                    rspool::ranges::<Here>(n, Work::ns(work), |r| {
                        r.for_each(|i| {
                            hits[i].fetch_add(1, Relaxed);
                        })
                    });
                    assert!(hits.iter().all(|h| h.load(Relaxed) == 1), "{n} {work}");
                    let v = rspool::map::<Here, _>(n, Work::ns(work), |i| 3 * i);
                    assert!(v.iter().enumerate().all(|(i, &x)| x == 3 * i), "{n}");
                    for item in [1, 3] {
                        let mut v = vec![usize::MAX; n];
                        rspool::chunks_mut::<Here, _>(&mut v, item, Work::ns(work), |i, part| {
                            for (j, x) in part.iter_mut().enumerate() {
                                *x = i * item + j;
                            }
                        });
                        assert!(v.iter().enumerate().all(|(i, &x)| x == i), "{n} {item}");
                    }
                }
            }
        });
    }
}

#[test]
fn every_thread_that_claims_has_one_state() {
    lent(4, Dispatch::free(4), || {
        for _ in 0..20 {
            // the thread each state was made on, and the thread of each range
            let made: Mutex<Vec<ThreadId>> = Mutex::new(Vec::new());
            let used: Mutex<Vec<(usize, ThreadId)>> = Mutex::new(Vec::new());
            let out = rspool::map_init::<Here, _, _>(
                400,
                Work::ns(f64::INFINITY),
                || {
                    let mut m = made.lock().unwrap();
                    m.push(std::thread::current().id());
                    m.len() - 1
                },
                |state, i| {
                    busy(2_000);
                    used.lock()
                        .unwrap()
                        .push((*state, std::thread::current().id()));
                    i
                },
            );
            assert!(out.iter().enumerate().all(|(i, &x)| x == i));
            let (made, used) = (made.into_inner().unwrap(), used.into_inner().unwrap());
            // a state on the thread that made it alone, every state used,
            // and no thread with two
            assert!(used.iter().all(|&(s, t)| made[s] == t));
            assert!((0..made.len()).all(|s| used.iter().any(|u| u.0 == s)));
            let threads: std::collections::HashSet<_> = made.iter().collect();
            assert_eq!(threads.len(), made.len(), "{made:?}");
            assert!(made.len() <= 4);
        }
    });
}

#[test]
fn work_not_worth_a_thread_runs_serially_as_one_range() {
    let one = |n: usize| {
        let here = std::thread::current().id();
        let seen: Mutex<Vec<std::ops::Range<usize>>> = Mutex::new(Vec::new());
        let states = AtomicUsize::new(0);
        rspool::ranges_init::<Here, _>(
            n,
            Work::ns(10.0),
            || states.fetch_add(1, Relaxed),
            |_, r| {
                assert_eq!(std::thread::current().id(), here);
                seen.lock().unwrap().push(r);
            },
        );
        assert_eq!(seen.into_inner().unwrap(), vec![0..n]);
        assert_eq!(states.into_inner(), 1);
    };
    // below the break-even of the workers, and with nothing lent
    lent(4, costed(4), || one(1000));
    one(1000);
}

#[test]
fn the_cut_follows_the_work_against_the_section_cost() {
    let d = costed(8);
    // one range per thread at the least, one per item at the most
    assert_eq!(d.ranges(1.0, 1000, 8), 8);
    assert_eq!(d.ranges(f64::INFINITY, 1000, 8), 1000);
    assert_eq!(Dispatch::free(8).ranges(1e6, 1000, 8), 1000);
    assert_eq!(d.ranges(1e12, 50, 8), 50);
    assert_eq!(d.ranges(1e6, 1000, 1), 1);
    assert_eq!(d.ranges(1e6, 0, 8), 0);
    // between: k sqrt(work / cost(k)), the cost 8 us on eight threads
    assert_eq!(d.ranges(8e5, 100_000, 8), 80);
    let m: Vec<usize> = [1e4, 1e5, 1e6, 1e7]
        .iter()
        .map(|&w| d.ranges(w, 100_000, 8))
        .collect();
    assert!(m.windows(2).all(|m| m[0] <= m[1]), "{m:?}");
}

#[test]
fn a_loop_in_a_piece_of_a_pool_wide_section_runs_on_its_thread() {
    lent(4, Dispatch::free(4), || {
        rspool::ranges::<Here>(4, Work::ns(f64::INFINITY), |_| {
            let here = std::thread::current().id();
            rspool::ranges::<Here>(100, Work::ns(f64::INFINITY), |_| {
                assert_eq!(std::thread::current().id(), here);
            });
        });
    });
}

#[test]
fn a_panic_in_a_range_reaches_the_caller() {
    lent(3, Dispatch::free(3), || {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            rspool::ranges::<Here>(64, Work::ns(f64::INFINITY), |r| {
                assert!(!r.contains(&37), "range with 37")
            });
        }));
        assert!(r.is_err());
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            rspool::map::<Here, usize>(64, Work::ns(f64::INFINITY), |i| {
                assert!(i != 37, "item 37");
                i
            })
        }));
        assert!(r.is_err());
        // the pool goes on
        let n = AtomicUsize::new(0);
        rspool::ranges::<Here>(64, Work::ns(f64::INFINITY), |r| {
            n.fetch_add(r.len(), Relaxed);
        });
        assert_eq!(n.into_inner(), 64);
    });
}

#[cfg(feature = "rayon")]
#[test]
fn rayon_loops_visit_every_index_once() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    pool.install(|| {
        let hits: Vec<AtomicUsize> = (0..1000).map(|_| AtomicUsize::new(0)).collect();
        let states = AtomicUsize::new(0);
        rspool::ranges_init::<rspool::rayon::Rayon, _>(
            1000,
            Work::ns(f64::INFINITY),
            || states.fetch_add(1, Relaxed),
            |_, r| {
                r.for_each(|i| {
                    hits[i].fetch_add(1, Relaxed);
                })
            },
        );
        assert!(hits.iter().all(|h| h.load(Relaxed) == 1));
        assert!((1..=4).contains(&states.into_inner()));
    });
}
