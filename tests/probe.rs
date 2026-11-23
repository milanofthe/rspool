//! Profiling: a task run outside the call that issued it carries the
//! issuing thread's rsprobe node, and every section counts itself there.
//! The records need `--features rsprobe/on`; without it they are empty, and
//! what is checked is that nothing panics.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use rspool::{Pool, Serial, Spawn, Workers};
use rsprobe::{add, scope, Context, Level, Record};

/// Whether rsprobe records in this build.
fn on() -> bool {
    let ctx = Context::new();
    {
        let _c = ctx.enter();
        let _s = scope!("on");
    }
    ctx.record().find(&["on"]).is_some()
}

/// Busy for `d`, or until `until` holds.
fn hold(d: Duration, until: &dyn Fn() -> bool) {
    let t = Instant::now();
    while t.elapsed() < d && !until() {
        std::hint::spin_loop();
    }
}

const PIECE: Duration = Duration::from_micros(100);
/// How long a side waits for the other to be taken by another thread.
const SIDE: Duration = Duration::from_millis(200);

/// A side of the join: one scope site for both.
fn side(items: u64, f: impl FnOnce()) {
    let _s = scope!("side");
    add!("items", items);
    f()
}

/// A task of the scope: one scope site for all, a task spawned inside a task
/// hanging under it.
fn task(items: u64, f: impl FnOnce()) {
    let _t = scope!("task");
    add!("items", items);
    hold(PIECE, &|| false);
    f()
}

/// A run whose pieces run again, a join and a scope whose tasks spawn
/// tasks, all scopes under `section`.
fn work(p: &dyn Pool) {
    let _s = scope!("section");
    p.run(16, &|i| {
        let _p = scope!("piece");
        add!("items", i + 1);
        hold(PIECE, &|| false);
        p.run(4, &|_| {
            let _q = scope!("inner");
            add!("items", 1);
        });
    });
    let started = AtomicBool::new(false);
    p.join(
        &mut || side(1, || hold(SIDE, &|| started.load(Ordering::SeqCst))),
        &mut || side(2, || started.store(true, Ordering::SeqCst)),
    );
    p.scope(&mut |s: &dyn Spawn<'_>| {
        for t in 0..8 {
            s.spawn(Box::new(move |s: &dyn Spawn<'_>| {
                task(t, || s.spawn(Box::new(|_: &dyn Spawn<'_>| task(1, || ()))));
            }));
        }
    });
}

/// Per node its path, calls and `items`, sorted.
fn exact(r: &Record) -> Vec<(String, u64, u64)> {
    let name = |i: usize| {
        let s = r.nodes[i].site?;
        r.sites.iter().find(|x| x.0 == s).map(|x| x.1.name)
    };
    let path = |mut i: usize| {
        let mut p = Vec::new();
        while let Some(n) = name(i) {
            p.push(n);
            i = r.nodes[i].parent.unwrap_or(0) as usize;
        }
        p.reverse();
        p.join("/")
    };
    let mut v: Vec<_> = (0..r.nodes.len())
        .map(|i| (path(i), sum(r, i, "calls"), sum(r, i, "items")))
        .collect();
    v.sort();
    v
}

/// `m` at node `i`, summed over the sites that add it.
fn sum(r: &Record, i: usize, m: &str) -> u64 {
    r.nodes[i]
        .values
        .iter()
        .filter(|v| v.0 == m)
        .map(|v| v.1)
        .sum()
}

fn get(r: &Record, path: &[&str], m: &str) -> u64 {
    r.find(path).map_or(0, |i| sum(r, i as usize, m))
}

fn recorded(f: impl FnOnce()) -> Record {
    let ctx = Context::builder().threshold(Level::DEBUG).build();
    {
        let _c = ctx.enter();
        f();
    }
    ctx.record()
}

#[test]
fn tasks_of_a_run_a_join_and_a_scope_land_under_the_caller() {
    let w = Workers::new(4);
    let wide = recorded(|| work(&w));
    let serial = recorded(|| work(&Serial));
    if !on() {
        assert_eq!(exact(&wide).len(), 1);
        return;
    }
    // the same tree and counts as the serial run, nested sections included
    assert_eq!(exact(&wide), exact(&serial));
    assert_eq!(get(&wide, &["section", "piece"], "calls"), 16);
    assert_eq!(get(&wide, &["section", "piece", "inner"], "calls"), 64);
    assert_eq!(get(&wide, &["section", "side"], "items"), 3);
    assert_eq!(get(&wide, &["section", "task"], "calls"), 8);
    assert_eq!(get(&wide, &["section", "task", "task"], "calls"), 8);
    // more than the issuing thread recorded: the join holds its first side
    // until another thread has run the second
    assert!(wide.threads > 1);
}

#[test]
fn every_section_counts_its_width_at_the_current_node() {
    let pool: Arc<dyn Pool> = Arc::new(Workers::new(4));
    let lent = rspool::Lent {
        pool,
        dispatch: rspool::Dispatch::free(4),
    };
    let r = recorded(|| {
        rspool::install_lent(lent, || {
            let _s = scope!("section");
            rspool::run_width(16, 4, &|_| ());
            rspool::run_width(16, 1, &|_| ());
            // two pieces of a pool of four leave each piece two threads
            rspool::run_width(2, 2, &|_| {
                let _p = scope!("piece");
                rspool::run_width(8, 2, &|_| ());
            });
        })
    });
    if !on() {
        return;
    }
    assert_eq!(get(&r, &["section"], "sections"), 3);
    assert_eq!(get(&r, &["section"], "width"), 4 + 1 + 2);
    assert_eq!(get(&r, &["section"], "serial"), 1);
    assert_eq!(get(&r, &["section", "piece"], "sections"), 2);
    assert_eq!(get(&r, &["section", "piece"], "width"), 4);
    assert_eq!(get(&r, &["section", "piece"], "serial"), 0);
    // below the threshold nothing is counted
    let info = Context::new();
    {
        let _c = info.enter();
        rspool::run_width(16, 1, &|_| ());
    }
    assert_eq!(get(&info.record(), &[], "sections"), 0);
}

/// Whether node `i` has measure `m`, and its kind (`true` exact).
fn kind(r: &Record, path: &[&str], m: &str) -> Option<bool> {
    let i = r.find(path)? as usize;
    r.nodes[i].values.iter().find(|v| v.0 == m)?;
    r.measures.iter().find(|x| x.0 == m).map(|x| x.1)
}

#[test]
fn a_rated_section_records_what_its_width_was_chosen_on() {
    let pool: Arc<dyn Pool> = Arc::new(Workers::new(4));
    let lent = rspool::Lent {
        pool,
        dispatch: rspool::Dispatch::free(4),
    };
    // 16 pieces of 5 us estimated at 80 ns in all: every narrower width
    // could beat the measured one, so some sections explore
    let (sections, rate) = (640, rspool::Rate::new(1.0));
    let r = recorded(|| {
        rspool::install_lent(lent, || {
            let _s = scope!("section");
            for _ in 0..sections {
                let work = rspool::Work::estimated(80.0, 80.0, &rate);
                work.run::<rspool::Here>(16, &|_| hold(Duration::from_micros(5), &|| false));
            }
        })
    });
    if !on() {
        return;
    }
    let at = ["section"];
    assert_eq!(get(&r, &at, "sections"), sections);
    for m in ["predicted", "measured", "exploring", "wait"] {
        assert_eq!(kind(&r, &at, m), Some(false), "{m}");
    }
    assert_eq!(kind(&r, &at, "explored"), Some(true));
    // the estimate of every measured section, 80 ns, at least the first two
    let predicted = get(&r, &at, "predicted");
    assert!(
        predicted >= 160 && predicted.is_multiple_of(80),
        "{predicted}"
    );
    assert!(get(&r, &at, "measured") >= get(&r, &at, "exploring"));
    let explored = get(&r, &at, "explored");
    assert!(explored > 0 && explored < sections / 8, "{explored}");
    assert!(get(&r, &at, "exploring") > 0);
}

#[test]
fn without_a_current_context_nothing_is_recorded() {
    let ctx = Context::builder().threshold(Level::DEBUG).build();
    let w = Workers::new(4);
    work(&w);
    rspool::install(Arc::new(Workers::new(4)), || {
        rspool::run_width(16, 4, &|_| ());
        rspool::run(16, f64::INFINITY, &|_| ());
    });
    let r = ctx.record();
    assert_eq!(r.nodes.len(), 1);
    assert!(
        r.nodes[0].values.iter().all(|v| v.1 == 0),
        "{:?}",
        r.nodes[0]
    );
}

#[cfg(feature = "rayon")]
#[test]
fn rayon_pieces_land_under_the_caller() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let r = recorded(|| {
        let caller = rsprobe::current();
        pool.install(|| {
            let _c = caller.enter();
            let _s = scope!("section");
            rspool::rayon::run_width(16, 4, &|i| {
                let _p = scope!("piece");
                add!("items", i + 1);
                hold(PIECE, &|| false);
            });
        })
    });
    if !on() {
        return;
    }
    assert_eq!(get(&r, &["section", "piece"], "calls"), 16);
    assert_eq!(get(&r, &["section", "piece"], "items"), 16 * 17 / 2);
    assert_eq!(get(&r, &["section"], "width"), 4);
    assert!(get(&r, &["section", "piece"], "busy") > 0);
}
