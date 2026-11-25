//! Nested sections on one pool, three effects each in isolation, on rspool's
//! own workers or on a rayon pool of as many threads (the same workload,
//! nested `par_iter`s and `join`s, rayon's stealing):
//!
//!     cargo run --release --example nested --features rayon -- \
//!         case=steal runner=workers threads=8 [key=value ...]
//!
//! - `case=steal`: `branches` branches side by side, one piece each of an
//!   outer section. Branch 0 opens short sections (`short` pieces of
//!   `short_us`), every other branch sections of `long` pieces of
//!   `long_us`. A thread of branch 0 that waits for its section and runs a
//!   long piece of another branch meanwhile returns late: branch 0's section
//!   times (median, p99, max) and the share of its time in sections that
//!   took more than half a long piece.
//! - `case=tail`: an outer section of `slices` pieces, piece `i` a chain of
//!   `steps * (i + 1) / slices` steps (they finish at different times), a
//!   step an inner section of `nodes` pieces of `node_us` and then
//!   `judge_us` on the piece's thread. The wall time against the work over
//!   the threads (`ideal`), and the threads busy on average.
//! - `case=admit`: an outer loop of `jobs` pieces, piece `i` holding
//!   `need(i)` units of a resource of `budget` units while it runs (an
//!   inner section of `nodes` pieces of `node_us`, `rounds` times). With
//!   nothing to admit it by, as many run as the width gives them: the peak
//!   of units held against the budget, and the wall time.
//!
//! One line per run on stdout. Every piece busy-waits its time, so a thread
//! is either working or idle; the work is the same on both runners.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

fn busy(d: Duration) {
    let t = Instant::now();
    while t.elapsed() < d {
        std::hint::spin_loop();
    }
}

fn us(x: u64) -> Duration {
    Duration::from_micros(x)
}

/// A section of `n` pieces of `f`: on rspool as wide as its share, or on
/// the current rayon pool.
fn section(rayon: bool, n: usize, f: &(dyn Fn(usize) + Sync)) {
    if rayon {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            (0..n).into_par_iter().with_max_len(1).for_each(f);
            return;
        }
        #[cfg(not(feature = "rayon"))]
        panic!("runner=rayon needs the feature rayon");
    }
    rspool::run_width(n, rspool::threads(), f)
}

/// `f` with the pool of `threads` threads: rspool's workers lent, or a
/// rayon pool installed.
fn on_pool(rayon: bool, threads: usize, f: impl FnOnce() + Send) {
    if rayon {
        #[cfg(feature = "rayon")]
        {
            let p = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("a rayon pool");
            p.install(f);
            return;
        }
        #[cfg(not(feature = "rayon"))]
        panic!("runner=rayon needs the feature rayon");
    }
    let w: Arc<dyn rspool::Pool> = Arc::new(rspool::Workers::new(threads));
    rspool::install(w, f)
}

fn pct(v: &mut [f64], p: f64) -> f64 {
    v.sort_by(f64::total_cmp);
    v[((v.len() - 1) as f64 * p) as usize]
}

fn main() {
    let args: Vec<(String, String)> = std::env::args()
        .skip(1)
        .map(|a| {
            let (k, v) = a.split_once('=').expect("key=value");
            (k.to_string(), v.to_string())
        })
        .collect();
    let get = |k: &str, d: &str| {
        args.iter()
            .rev()
            .find(|(a, _)| a == k)
            .map_or(d.to_string(), |(_, v)| v.clone())
    };
    let num = |k: &str, d: &str| -> u64 { get(k, d).parse().expect("a number") };
    let (case, runner) = (get("case", "steal"), get("runner", "workers"));
    let rayon = runner == "rayon";
    let threads = num("threads", "8") as usize;
    let busy_ns = AtomicU64::new(0);
    let timed = |d: Duration| {
        busy(d);
        busy_ns.fetch_add(d.as_nanos() as u64, Relaxed);
    };
    let t0 = Instant::now();
    match case.as_str() {
        "steal" => {
            let branches = num("branches", "2") as usize;
            let (short, short_us) = (num("short", "16") as usize, num("short_us", "20"));
            let (long, long_us) = (num("long", "8") as usize, num("long_us", "5000"));
            let secs = num("secs", "2");
            let times = Mutex::new(Vec::new());
            on_pool(rayon, threads, || {
                section(rayon, branches, &|b| {
                    let start = Instant::now();
                    while start.elapsed() < Duration::from_secs(secs) {
                        if b == 0 {
                            let t = Instant::now();
                            section(rayon, short, &|_| busy(us(short_us)));
                            times.lock().unwrap().push(t.elapsed().as_secs_f64() * 1e3);
                        } else {
                            section(rayon, long, &|_| busy(us(long_us)));
                        }
                    }
                })
            });
            let mut t = times.into_inner().unwrap();
            let late = long_us as f64 * 0.5e-3;
            let total: f64 = t.iter().sum();
            let lost: f64 = t.iter().filter(|&&x| x > late).sum();
            let n = t.len();
            println!(
                "steal {runner} threads {threads} branches {branches}: branch 0 sections {n}, \
                 median {:.3} ms, p99 {:.3} ms, max {:.2} ms, {:.1} % of its time in sections \
                 over {late:.1} ms (alone {:.3} ms)",
                pct(&mut t, 0.5),
                pct(&mut t, 0.99),
                pct(&mut t, 1.0),
                100.0 * lost / total,
                (short as f64 / threads as f64).ceil() * short_us as f64 * 1e-3,
            );
        }
        "tail" => {
            let (slices, steps) = (num("slices", "12") as usize, num("steps", "40") as usize);
            let (nodes, node_us) = (num("nodes", "8") as usize, num("node_us", "200"));
            let judge_us = num("judge_us", "100");
            on_pool(rayon, threads, || {
                section(rayon, slices, &|i| {
                    for _ in 0..steps * (i + 1) / slices {
                        section(rayon, nodes, &|_| timed(us(node_us)));
                        timed(us(judge_us));
                    }
                })
            });
            let wall = t0.elapsed().as_secs_f64();
            let work = busy_ns.load(Relaxed) as f64 * 1e-9;
            // the longest chain bounds it too: the last slice's steps one after the other
            let chain = steps as f64 * (judge_us + node_us) as f64 * 1e-6;
            println!(
                "tail {runner} threads {threads} slices {slices}: wall {wall:.3} s, work/threads \
                 {:.3} s, longest chain {chain:.3} s, threads busy {:.2}",
                work / threads as f64,
                work / wall,
            );
        }
        "admit" => {
            let (jobs, budget) = (num("jobs", "24") as usize, num("budget", "8") as usize);
            let (nodes, node_us, rounds) = (
                num("nodes", "8") as usize,
                num("node_us", "200"),
                num("rounds", "50") as usize,
            );
            // piece i holds 1, 2 or 3 units
            let need = |i: usize| 1 + i % 3;
            let (held, peak) = (AtomicUsize::new(0), AtomicUsize::new(0));
            on_pool(rayon, threads, || {
                section(rayon, jobs, &|i| {
                    let h = held.fetch_add(need(i), Relaxed) + need(i);
                    peak.fetch_max(h, Relaxed);
                    for _ in 0..rounds {
                        section(rayon, nodes, &|_| timed(us(node_us)));
                    }
                    held.fetch_sub(need(i), Relaxed);
                })
            });
            let wall = t0.elapsed().as_secs_f64();
            let work = busy_ns.load(Relaxed) as f64 * 1e-9;
            println!(
                "admit {runner} threads {threads} jobs {jobs}: peak held {} of a budget of \
                 {budget} units, wall {wall:.3} s, work/threads {:.3} s, threads busy {:.2}",
                peak.load(Relaxed),
                work / threads as f64,
                work / wall,
            );
        }
        _ => panic!("case=steal|tail|admit"),
    }
}
