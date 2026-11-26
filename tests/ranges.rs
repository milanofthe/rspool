//! The loop over items: every index once, one state per thread that claimed
//! a range, serial where the cost model says so, the cut the rule gives,
//! nesting as every section nests, and panics reaching the caller.

use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::sync::{Arc, Mutex};
use std::thread::ThreadId;

use rspool::{Dispatch, Here, Lent, Pool, Work, Workers};

fn lent(threads: usize, dispatch: Dispatch, f: impl FnOnce()) {
    let pool: Arc<dyn Pool> = Arc::new(Workers::new(threads).unwrap());
    rspool::install_lent(Lent { pool, dispatch }, f)
}

fn busy(ns: u64) {
    let t = std::time::Instant::now();
    while t.elapsed().as_nanos() < ns as u128 {
        std::hint::spin_loop();
    }
}

fn one_us_per_thread(threads: usize) -> Dispatch {
    Dispatch::new((0..=threads).map(|k| 1e3 * k as f64).collect())
}

#[test]
fn every_index_is_visited_once() {
    for dispatch in [Dispatch::free(4), one_us_per_thread(4), Dispatch::never()] {
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
            let thread_of_state: Mutex<Vec<ThreadId>> = Mutex::new(Vec::new());
            let state_and_thread_of_range: Mutex<Vec<(usize, ThreadId)>> = Mutex::new(Vec::new());
            let out = rspool::map_init::<Here, _, _>(
                400,
                Work::ns(f64::INFINITY),
                || {
                    let mut m = thread_of_state.lock().unwrap();
                    m.push(std::thread::current().id());
                    m.len() - 1
                },
                |state, i| {
                    busy(2_000);
                    state_and_thread_of_range
                        .lock()
                        .unwrap()
                        .push((*state, std::thread::current().id()));
                    i
                },
            );
            assert!(out.iter().enumerate().all(|(i, &x)| x == i));
            let made = thread_of_state.into_inner().unwrap();
            let used = state_and_thread_of_range.into_inner().unwrap();
            assert!(
                used.iter().all(|&(s, t)| made[s] == t),
                "a state on the thread that made it alone"
            );
            assert!(
                (0..made.len()).all(|s| used.iter().any(|u| u.0 == s)),
                "every state used"
            );
            let threads: std::collections::HashSet<_> = made.iter().collect();
            assert_eq!(
                threads.len(),
                made.len(),
                "no thread with two states: {made:?}"
            );
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
    lent(4, one_us_per_thread(4), || one(1000));
    one(1000);
}

#[test]
fn the_cut_follows_the_work_against_the_section_cost() {
    let d = one_us_per_thread(8);
    assert_eq!(d.ranges(1.0, 1000, 8), 8);
    assert_eq!(d.ranges(f64::INFINITY, 1000, 8), 1000);
    assert_eq!(Dispatch::free(8).ranges(1e6, 1000, 8), 1000);
    assert_eq!(d.ranges(1e12, 50, 8), 50);
    assert_eq!(d.ranges(1e6, 1000, 1), 1);
    assert_eq!(d.ranges(1e6, 0, 8), 0);
    let (k, work) = (8, 8e5);
    let k_sqrt_work_over_cost = k * (work / d.cost(k)).sqrt() as usize;
    assert_eq!(d.ranges(work, 100_000, k), k_sqrt_work_over_cost);
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
        let n = AtomicUsize::new(0);
        rspool::ranges::<Here>(64, Work::ns(f64::INFINITY), |r| {
            n.fetch_add(r.len(), Relaxed);
        });
        assert_eq!(n.into_inner(), 64, "the pool runs after a panic");
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

struct Counted<'a> {
    i: usize,
    drops: &'a [AtomicUsize],
}

impl Drop for Counted<'_> {
    fn drop(&mut self) {
        self.drops[self.i].fetch_add(1, Relaxed);
    }
}

fn counted(drops: &[AtomicUsize]) -> Vec<Counted<'_>> {
    (0..drops.len()).map(|i| Counted { i, drops }).collect()
}

#[test]
fn map_owned_moves_every_item_into_f_once_in_order() {
    let one = |work: f64| {
        for n in [0, 1, 2, 7, 1000] {
            let drops: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
            let calls: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
            let out = rspool::map_owned::<Here, _, _>(counted(&drops), Work::ns(work), |c| {
                calls[c.i].fetch_add(1, Relaxed);
                3 * c.i
            });
            assert!(out.iter().enumerate().all(|(i, &x)| x == 3 * i), "{n}");
            assert!(calls.iter().all(|c| c.load(Relaxed) == 1), "{n}");
            assert!(drops.iter().all(|d| d.load(Relaxed) == 1), "{n}");
        }
    };
    one(f64::INFINITY);
    lent(4, Dispatch::free(4), || {
        one(1.0);
        one(f64::INFINITY);
    });
}

#[test]
fn map_owned_hands_back_results_that_drop_once() {
    lent(4, Dispatch::free(4), || {
        let drops: Vec<AtomicUsize> = (0..500).map(|_| AtomicUsize::new(0)).collect();
        let kept: Vec<AtomicUsize> = (0..500).map(|_| AtomicUsize::new(0)).collect();
        let out = rspool::map_owned::<Here, _, _>(counted(&drops), Work::ns(f64::INFINITY), |c| {
            Counted {
                i: c.i,
                drops: &kept,
            }
        });
        assert!(out.iter().enumerate().all(|(i, c)| c.i == i));
        assert!(kept.iter().all(|d| d.load(Relaxed) == 0));
        drop(out);
        assert!(kept.iter().all(|d| d.load(Relaxed) == 1));
        assert!(drops.iter().all(|d| d.load(Relaxed) == 1));
    });
}

#[test]
fn a_panic_in_map_owned_reaches_the_caller_and_drops_no_item_twice() {
    let one = |work: f64| {
        let drops: Vec<AtomicUsize> = (0..64).map(|_| AtomicUsize::new(0)).collect();
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            rspool::map_owned::<Here, _, _>(counted(&drops), Work::ns(work), |c| {
                assert!(c.i != 37, "item 37");
                c.i
            })
        }));
        assert!(r.is_err());
        assert!(drops.iter().all(|d| d.load(Relaxed) <= 1));
        assert_eq!(drops[37].load(Relaxed), 1, "the item of the panic drops");
    };
    one(f64::INFINITY);
    lent(3, Dispatch::free(3), || one(f64::INFINITY));
}

#[cfg(feature = "rayon")]
#[test]
fn rayon_map_owned_moves_every_item_once_in_order() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    pool.install(|| {
        let drops: Vec<AtomicUsize> = (0..1000).map(|_| AtomicUsize::new(0)).collect();
        let out = rspool::map_owned::<rspool::rayon::Rayon, _, _>(
            counted(&drops),
            Work::ns(f64::INFINITY),
            |c| c.i,
        );
        assert!(out.iter().enumerate().all(|(i, &x)| x == i));
        assert!(drops.iter().all(|d| d.load(Relaxed) == 1));
    });
}
