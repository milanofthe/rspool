//! Sections: how wide they go, the share a piece leaves its own sections,
//! and that nothing leaks out of a lent pool.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use rspool::{Dispatch, Here, Lent, Place, Pool, Rate, Serial, Workers};

#[test]
fn a_section_goes_as_wide_as_its_work_is_worth() {
    // free: every section of two pieces or more on as many threads as pieces
    assert_eq!(Dispatch::free(8).width(1.0, 3), 3);
    assert_eq!(Dispatch::free(8).width(1.0, 100), 8);
    assert_eq!(Dispatch::free(8).width(1.0, 1), 1);
    assert_eq!(Dispatch::never().width(1e12, 100), 1);
    // 1 us per section plus 1 us per helper: a microsecond of work stays
    // serial, a millisecond goes to all eight threads, and in between some
    let d = Dispatch::new((0..=8).map(|k| 1e3 * k as f64).collect());
    assert_eq!(d.width(1e3, 64), 1);
    assert_eq!(d.width(1e6, 64), 8);
    let mid = d.width(2e4, 64);
    assert!(mid > 1 && mid < 8, "{mid}");
    assert!(d.break_even() > 1e3 && d.break_even() < 2e4);
    // nothing to gain: serial; infinite work: as wide as it goes
    assert_eq!(d.width(f64::NAN, 64), 1);
    assert_eq!(d.width(f64::INFINITY, 64), 8);
}

#[test]
fn workers_measure_what_a_section_costs_them() {
    let w = Workers::new(4);
    let d = w.dispatch();
    assert_eq!(d.threads(), 4);
    let costs: Vec<f64> = (2..=4).map(|k| d.cost(k)).collect();
    assert!(
        costs.iter().all(|c| c.is_finite() && *c >= 0.0),
        "{costs:?}"
    );
    assert!(costs.windows(2).all(|c| c[0] <= c[1]), "{costs:?}");
    assert_eq!(Workers::new(1).dispatch().width(1e12, 100), 1);
}

#[test]
fn nothing_is_lent_outside_install() {
    assert!(rspool::current().is_none());
    assert_eq!(rspool::threads(), 1);
    assert_eq!(rspool::width(f64::INFINITY, 100), 1);
    let w: Arc<dyn Pool> = Arc::new(Workers::new(3));
    rspool::install(w, || {
        assert!(rspool::current().is_some());
        assert_eq!(rspool::threads(), 3);
    });
    assert!(rspool::current().is_none());
}

#[test]
fn every_piece_runs_once_however_wide() {
    let w: Arc<dyn Pool> = Arc::new(Workers::new(4));
    let lent = Lent {
        dispatch: Dispatch::free(4),
        pool: w,
    };
    rspool::install_lent(lent, || {
        for k in 1..=5 {
            let hits: Vec<AtomicUsize> = (0..97).map(|_| AtomicUsize::new(0)).collect();
            Here::run_width(97, k, &|i| {
                hits[i].fetch_add(1, Ordering::Relaxed);
            });
            assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1), "{k}");
        }
    });
}

#[test]
fn a_piece_leaves_its_sections_its_share_of_the_pool() {
    let w: Arc<dyn Pool> = Arc::new(Workers::new(8));
    let lent = Lent {
        dispatch: Dispatch::free(8),
        pool: w,
    };
    rspool::install_lent(lent, || {
        assert_eq!(rspool::width(f64::INFINITY, 100), 8);
        // a section as wide as the pool: its pieces' sections run serially,
        // and a piece has one thread to size its work by
        let inner = AtomicUsize::new(usize::MAX);
        let threads = AtomicUsize::new(0);
        Here::run_width(8, 8, &|_| {
            inner.fetch_min(rspool::width(f64::INFINITY, 100), Ordering::Relaxed);
            threads.fetch_max(rspool::threads(), Ordering::Relaxed);
        });
        assert_eq!(inner.into_inner(), 1);
        assert_eq!(threads.into_inner(), 1);
        // two wide: each piece's sections get four threads
        let inner = AtomicUsize::new(0);
        Here::run_width(2, 2, &|_| {
            inner.fetch_max(rspool::width(f64::INFINITY, 100), Ordering::Relaxed);
        });
        assert_eq!(inner.into_inner(), 4);
        // and the share comes back after the section
        assert_eq!(rspool::width(f64::INFINITY, 100), 8);
    });
}

#[test]
fn a_serial_pool_keeps_everything_on_its_thread() {
    let lent = Lent::new(Arc::new(Serial));
    rspool::install_lent(lent, || {
        assert_eq!(rspool::width(f64::INFINITY, 100), 1);
        let here = std::thread::current().id();
        Here::run_width(16, 16, &|_| assert_eq!(std::thread::current().id(), here));
    });
}

#[test]
fn rated_sections_compute_every_piece_once() {
    let w: Arc<dyn Pool> = Arc::new(Workers::new(4));
    rspool::install(w, || {
        let r = Rate::new(1.0);
        for _ in 0..50 {
            let hits: Vec<AtomicUsize> = (0..1000).map(|_| AtomicUsize::new(0)).collect();
            rspool::run_rated(1000, 1e6, &r, &|i| {
                hits[i].fetch_add(1, Ordering::Relaxed);
            });
            assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1));
        }
    });
}

#[cfg(feature = "rayon")]
#[test]
fn rayon_sections_measure_the_pool_and_run_every_piece() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    pool.install(|| {
        rspool::rayon::prepare();
        let d = rspool::rayon::dispatch();
        assert_eq!(d.threads(), 4);
        let hits: Vec<AtomicUsize> = (0..97).map(|_| AtomicUsize::new(0)).collect();
        rspool::rayon::run_width(97, 4, &|i| {
            hits[i].fetch_add(1, Ordering::Relaxed);
        });
        assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1));
    });
}

#[cfg(feature = "rayon")]
#[test]
fn a_rayon_piece_leaves_its_sections_the_whole_pool() {
    // rayon runs nested pieces on free threads only: a piece of a section
    // two wide still opens its sections on all four
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    pool.install(|| {
        let inner = AtomicUsize::new(0);
        rspool::rayon::run_width(2, 2, &|_| {
            inner.fetch_max(rspool::rayon::width(f64::INFINITY, 100), Ordering::Relaxed);
        });
        assert_eq!(inner.into_inner(), 4);
    });
}

#[cfg(feature = "rayon")]
#[test]
fn a_caller_on_one_thread_enters_a_wider_pool_for_work_worth_it() {
    // from a pool of one thread, as a solve under a policy of its own
    // threads is asked from: a second of work is worth the wider pool, a
    // nanosecond is not
    let one = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    one.install(|| {
        assert!(rspool::rayon::width_on(4, 1e9, 64) > 1);
        assert_eq!(rspool::rayon::width_on(4, 1.0, 64), 1);
    });
}

#[test]
fn a_section_takes_the_width_its_estimate_is_worth() {
    use rspool::{Go, Work};
    let w: Arc<dyn Pool> = Arc::new(Workers::new(4));
    let lent = Lent {
        dispatch: Dispatch::new((0..=4).map(|k| 1e3 * k as f64).collect()),
        pool: w,
    };
    rspool::install_lent(lent, || {
        // ops at the rate against the section cost, as the cost model says
        let r = Rate::new(2.0);
        for ops in [1e2, 1e3, 1e4, 1e5, 1e6] {
            let k = rspool::dispatch().width(ops * 2.0, 64);
            let go = if k >= 2 { Go::Wide(k) } else { Go::Here };
            assert_eq!(Work::of(ops, &r).go::<Here>(64), go, "{ops}");
            assert_eq!(Work::ns(ops * 2.0).go::<Here>(64), go, "{ops}");
        }
        assert_eq!(Work::of(1e6, &r).go::<Here>(64), Go::Wide(4));
        assert_eq!(Work::of(1e2, &r).go::<Here>(64), Go::Here);
    });
}

#[test]
fn small_work_stays_serial() {
    use rspool::{Go, Work};
    let w: Arc<dyn Pool> = Arc::new(Workers::new(4));
    rspool::install(w.clone(), || {
        let r = Rate::new(1.0);
        // below what any section is worth on these workers
        let tiny = Work::of(rspool::dispatch().break_even() * 0.5, &r);
        assert!((0..1000).all(|_| tiny.go::<Here>(64) == Go::Here));
    });
}
