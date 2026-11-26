//! Sections of a fixed busy time on a pool of a given width, the work given
//! as ops at a rate that is right or a factor off; the widths the sections
//! take follow from that estimate against the pool's section cost. Writes the run's
//! rsprobe record (level DEBUG) to the file named by `RSPROBE_RECORD`.
//!
//!     cargo run --release --example width --features rsprobe/on -- \
//!         THREADS FACTOR [SECTIONS PIECES SECTION_US GAP_US]
//!
//! FACTOR is the rate over the true cost of an op (1: right, 100: a
//! hundred times too high); GAP_US the serial work between sections.

use std::hint::spin_loop;
use std::sync::Arc;
use std::time::{Duration, Instant};

const OPS_PER_PIECE: f64 = 1000.0;

fn busy(d: Duration) {
    let t = Instant::now();
    while t.elapsed() < d {
        spin_loop();
    }
}

fn main() {
    let a: Vec<f64> = std::env::args()
        .skip(1)
        .map(|a| {
            a.parse()
                .expect("THREADS FACTOR [SECTIONS PIECES SECTION_US GAP_US]")
        })
        .collect();
    let arg = |i: usize, default: f64| a.get(i).copied().unwrap_or(default);
    let (threads, factor) = (arg(0, 4.0) as usize, arg(1, 1.0));
    let (sections, pieces) = (arg(2, 2000.0) as usize, arg(3, 12.0) as usize);
    let (section_us, gap_us) = (arg(4, 400.0), arg(5, 100.0));
    let piece = Duration::from_nanos((section_us * 1e3 / pieces as f64) as u64);
    let gap = Duration::from_nanos((gap_us * 1e3) as u64);
    let ops = OPS_PER_PIECE * pieces as f64;
    let rate = rspool::Rate::new((factor * section_us * 1e3 / ops) as f32);

    let ctx = rsprobe::Context::builder()
        .threshold(rsprobe::Level::DEBUG)
        .build();
    for (k, v) in [
        ("threads", threads as f64),
        ("factor", factor),
        ("pieces", pieces as f64),
    ] {
        ctx.param(k, v);
    }
    ctx.param("section_us", section_us);
    ctx.param("gap_us", gap_us);
    let pool: Arc<dyn rspool::Pool> =
        Arc::new(rspool::Workers::new(threads).expect("the worker threads"));
    let t = Instant::now();
    {
        let _on = ctx.enter();
        rspool::install(pool, || {
            let _s = rsprobe::scope!("sections");
            for _ in 0..sections {
                rspool::run_rated(pieces, ops, &rate, &|_| busy(piece));
                busy(gap);
            }
        });
    }
    let wall = t.elapsed().as_secs_f64();
    let record = ctx.record();
    let node = record
        .find(&["sections"])
        .map(|i| &record.nodes[i as usize]);
    let (n, width) = node.map_or((0, 0), |n| (n.get("sections"), n.get("width")));
    eprintln!(
        "threads {threads} factor {factor}: {wall:.3} s, {:.1} us per section, mean width {:.2} of {n}",
        wall * 1e6 / sections as f64,
        width as f64 / n.max(1) as f64,
    );
    match std::env::var("RSPROBE_RECORD") {
        Ok(path) => std::fs::write(path, record.to_string()).expect("the record written"),
        Err(_) => println!("{record}"),
    }
}
