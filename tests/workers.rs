//! The workers: every piece once, awake or woken; a section gets its width;
//! sections nest; threads outside the pool share it; panics reach the
//! section that waits.

#[test]
fn workers_run_every_piece_once_awake_or_woken() {
    use rspool::{Pool, Workers};
    use std::sync::atomic::{AtomicUsize, Ordering};
    let spin = std::time::Duration::from_micros(50);
    let longer_than_spin = std::time::Duration::from_millis(2);
    let w = Workers::with_spin(4, spin).unwrap();
    let hits: Vec<AtomicUsize> = (0..500).map(|_| AtomicUsize::new(0)).collect();
    for round in 0..200 {
        w.run(500, &|i| {
            hits[i].fetch_add(1, Ordering::Relaxed);
        });
        if round % 50 == 0 {
            std::thread::sleep(longer_than_spin);
        }
    }
    assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 200));
}

#[test]
fn a_panic_in_a_piece_reaches_the_caller() {
    use rspool::{Pool, Workers};
    let w = Workers::new(3).unwrap();
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        w.run(64, &|i| assert!(i != 37, "piece 37"));
    }));
    assert!(r.is_err());
    let n = std::sync::atomic::AtomicUsize::new(0);
    w.run(64, &|_| {
        n.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    });
    assert_eq!(n.into_inner(), 64, "the pool runs after a panic");
}

#[test]
fn sections_nest_on_the_workers() {
    use rspool::{Pool, Spawn, Workers};
    use std::sync::atomic::{AtomicUsize, Ordering};
    let w = Workers::new(4).unwrap();
    let hits: Vec<AtomicUsize> = (0..16 * 32 * 2).map(|_| AtomicUsize::new(0)).collect();
    let tasks = AtomicUsize::new(0);
    for _ in 0..20 {
        w.run(16, &|i| {
            w.run(32, &|j| {
                let k = (i * 32 + j) * 2;
                let (h0, h1) = (&hits[k], &hits[k + 1]);
                w.join(
                    &mut || {
                        h0.fetch_add(1, Ordering::Relaxed);
                    },
                    &mut || {
                        h1.fetch_add(1, Ordering::Relaxed);
                    },
                );
            });
            w.scope(&mut |s: &dyn Spawn<'_>| {
                for _ in 0..3 {
                    let tasks = &tasks;
                    s.spawn(Box::new(move |s: &dyn Spawn<'_>| {
                        tasks.fetch_add(1, Ordering::Relaxed);
                        s.spawn(Box::new(move |_: &dyn Spawn<'_>| {
                            tasks.fetch_add(1, Ordering::Relaxed);
                        }));
                    }));
                }
            });
        });
    }
    assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 20));
    assert_eq!(tasks.into_inner(), 20 * 16 * 6);
}

#[test]
fn threads_outside_the_pool_share_it() {
    use rspool::{Pool, Workers};
    use std::sync::atomic::{AtomicUsize, Ordering};
    let w = Workers::new(3).unwrap();
    let n = AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..4 {
            s.spawn(|| {
                for _ in 0..200 {
                    w.run(17, &|_| {
                        w.run(5, &|_| {
                            n.fetch_add(1, Ordering::Relaxed);
                        });
                    });
                }
            });
        }
    });
    assert_eq!(n.into_inner(), 4 * 200 * 17 * 5);
}

#[test]
fn a_panic_in_a_task_or_a_side_reaches_its_section() {
    use rspool::{Pool, Spawn, Workers};
    let w = Workers::new(3).unwrap();
    let side = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        w.join(&mut || {}, &mut || panic!("side"));
    }));
    assert!(side.is_err());
    let task = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        w.scope(&mut |s: &dyn Spawn<'_>| {
            s.spawn(Box::new(|_: &dyn Spawn<'_>| panic!("task")));
        });
    }));
    assert!(task.is_err());
    let n = std::sync::atomic::AtomicUsize::new(0);
    w.run(8, &|_| {
        n.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    });
    assert_eq!(n.into_inner(), 8);
}

#[test]
fn a_section_runs_on_its_width_whether_the_workers_spin_or_sleep() {
    use rspool::{Pool, Workers};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{Duration, Instant};
    for spin in [Duration::ZERO, Duration::from_millis(50)] {
        let w = Workers::with_spin(8, spin).unwrap();
        for round in 0..100 {
            let width = 2 + round % 3;
            let here = AtomicUsize::new(0);
            let together = AtomicUsize::new(0);
            w.run_width(width, width, &|_| {
                here.fetch_add(1, Ordering::SeqCst);
                let t = Instant::now();
                while here.load(Ordering::SeqCst) < width && t.elapsed() < Duration::from_secs(5) {
                    std::hint::spin_loop();
                }
                if here.load(Ordering::SeqCst) == width {
                    together.fetch_add(1, Ordering::SeqCst);
                }
            });
            assert_eq!(together.into_inner(), width, "spin {spin:?}, round {round}");
        }
    }
}
