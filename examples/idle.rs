//! Serial work on the calling thread alternating with sections of a fixed
//! width, on a pool of a given size: the time of the serial work and of the
//! sections apart, for pools wider than their sections. Each round runs the
//! serial work (scope `serial`), then one section (scope `section`); the
//! record (level DEBUG) goes to the file named by `RSPROBE_RECORD`, one
//! line per case to stderr.
//!
//!     cargo run --release --example idle --features rayon,rsprobe/on -- \
//!         runner=workers threads=32 width=8 [key=value ...]
//!
//! Keys (defaults in brackets):
//! - `runner`: `workers` (rspool's own pool) or `rayon` (a rayon pool,
//!   sections through `rspool::rayon`) [workers]
//! - `threads`, `width`: the pool's threads and the section's [8, 8]
//! - `pieces`, `piece`: pieces per section and the ops of each [12, 20000]
//! - `serial`: `lu:N`, a dense factorization of order N; `mv:N`, a dense
//!   matrix-vector product of order N; or `ops:K`, K ops [lu:256]
//! - `feed`: `1`: the section's pieces write the input of the next serial
//!   work (the rows of its matrix); `0`: the serial work writes it [0]
//! - `rounds` [400]
//! - `spin`: how long the workers of `workers` look for work before they
//!   sleep, in us [rspool::SPIN]
//! - `pin` (Linux): `none`; `caller`, the calling thread on the first
//!   allowed CPU; `away`, that and the workers on the allowed CPUs other
//!   than that CPU's core [none]
//!
//! An op is one step of eight independent multiply-add chains, a fixed
//! amount of arithmetic; the single-thread time of the serial work and of
//! a piece is measured before the pool starts and printed with the case.
//! Per case: median and 10/90 percentiles over the rounds of the serial
//! work and the section, the median time from a section's start to the
//! first piece on another thread (`start`) and to the first piece on the
//! calling thread, after it queued the section's helpers (`issue`), how often the calling
//! thread was on another core or L3 than in the round before (`moved`), and
//! the share of pieces run on another L3 than the calling thread's
//! (`remote`).

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// `ops` steps of eight independent multiply-add chains.
fn ops(ops: u64) -> f64 {
    let mut x = [1.0f64, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7];
    for _ in 0..ops {
        for v in x.iter_mut() {
            *v = *v * 0.999_999_9 + 1e-7;
        }
    }
    std::hint::black_box(x.iter().sum())
}

/// The serial work of a round and its input.
struct Serial {
    kind: Kind,
    n: usize,
    /// The input as written.
    a: Vec<f64>,
    /// The input the work runs on: written by the section's pieces where
    /// the section feeds the serial work, else from `a` by the serial work.
    w: Vec<f64>,
    x: Vec<f64>,
}

enum Kind {
    /// In-place LU without pivoting of a diagonally dominant matrix.
    Lu,
    /// A dense matrix-vector product.
    Mv,
    /// `ops` ops.
    Ops(u64),
}

/// The input of the serial work, written by the pieces in disjoint rows.
#[derive(Clone, Copy)]
struct Rows(*mut f64);
unsafe impl Sync for Rows {}
unsafe impl Send for Rows {}

/// Rows `i * n / of..(i + 1) * n / of` of `rows` written from `a`, of order
/// `n`.
///
/// # Safety
/// `rows` holds `n * n` values, and no other thread touches these rows
/// meanwhile.
unsafe fn write_rows(rows: Rows, a: &[f64], n: usize, i: usize, of: usize) {
    let (r0, r1) = (i * n / of, (i + 1) * n / of);
    let dst = unsafe { std::slice::from_raw_parts_mut(rows.0.add(r0 * n), (r1 - r0) * n) };
    dst.copy_from_slice(&a[r0 * n..r1 * n]);
}

impl Serial {
    fn parse(s: &str) -> Serial {
        let (kind, n) = match s.split_once(':') {
            Some(("lu", n)) => (Kind::Lu, n.parse().expect("lu:N")),
            Some(("mv", n)) => (Kind::Mv, n.parse().expect("mv:N")),
            Some(("ops", k)) => (Kind::Ops(k.parse().expect("ops:K")), 1),
            _ => panic!("serial=lu:N, mv:N or ops:K"),
        };
        let a: Vec<f64> = (0..n * n)
            .map(|k| {
                let (i, j) = (k / n, k % n);
                if i == j {
                    n as f64
                } else {
                    1.0 / (1.0 + (i + 2 * j) as f64)
                }
            })
            .collect();
        Serial {
            kind,
            n,
            w: a.clone(),
            a,
            x: vec![1.0; n],
        }
    }

    fn run(&mut self, fed: bool) -> f64 {
        let n = self.n;
        match self.kind {
            Kind::Ops(k) => ops(k),
            Kind::Lu => {
                let w = &mut self.w;
                if !fed {
                    w.copy_from_slice(&self.a);
                }
                for k in 0..n {
                    let (top, rest) = w.split_at_mut((k + 1) * n);
                    let row = &top[k * n..];
                    let p = 1.0 / row[k];
                    for r in rest.chunks_exact_mut(n) {
                        let l = r[k] * p;
                        r[k] = l;
                        for (x, y) in r[k + 1..].iter_mut().zip(&row[k + 1..]) {
                            *x -= l * y;
                        }
                    }
                }
                std::hint::black_box(w[n * n - 1])
            }
            Kind::Mv => {
                let y: f64 = self
                    .w
                    .chunks_exact(n)
                    .map(|r| r.iter().zip(&self.x).map(|(a, b)| a * b).sum::<f64>())
                    .sum();
                std::hint::black_box(y)
            }
        }
    }
}

#[cfg(target_os = "linux")]
mod cpu {
    //! Affinity and the current CPU through glibc.
    extern "C" {
        fn sched_setaffinity(pid: i32, size: usize, mask: *const u8) -> i32;
        fn sched_getaffinity(pid: i32, size: usize, mask: *mut u8) -> i32;
        fn sched_getcpu() -> i32;
    }
    const BYTES: usize = 128;

    /// The CPUs this thread may run on.
    pub fn allowed() -> Vec<usize> {
        let mut m = [0u8; BYTES];
        assert_eq!(unsafe { sched_getaffinity(0, BYTES, m.as_mut_ptr()) }, 0);
        (0..BYTES * 8)
            .filter(|c| m[c / 8] & (1 << (c % 8)) != 0)
            .collect()
    }

    /// This thread on `cpus` only.
    pub fn pin(cpus: &[usize]) {
        let mut m = [0u8; BYTES];
        for c in cpus {
            m[c / 8] |= 1 << (c % 8);
        }
        assert_eq!(unsafe { sched_setaffinity(0, BYTES, m.as_ptr()) }, 0);
    }

    pub fn current() -> usize {
        unsafe { sched_getcpu() as usize }
    }

    fn list(path: &str) -> Vec<usize> {
        let s = std::fs::read_to_string(path).unwrap_or_default();
        s.trim()
            .split(',')
            .filter(|r| !r.is_empty())
            .flat_map(|r| match r.split_once('-') {
                Some((a, b)) => (a.parse().unwrap()..=b.parse().unwrap()).collect::<Vec<_>>(),
                None => vec![r.parse().unwrap()],
            })
            .collect()
    }

    /// The hardware threads of `cpu`'s core.
    pub fn core(cpu: usize) -> Vec<usize> {
        list(&format!(
            "/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
        ))
    }

    /// The first CPU sharing `cpu`'s L3: the L3's name.
    pub fn l3(cpu: usize) -> usize {
        list(&format!(
            "/sys/devices/system/cpu/cpu{cpu}/cache/index3/shared_cpu_list"
        ))
        .first()
        .copied()
        .unwrap_or(0)
    }
}

#[cfg(not(target_os = "linux"))]
mod cpu {
    pub fn allowed() -> Vec<usize> {
        Vec::new()
    }
    pub fn pin(_: &[usize]) {}
    pub fn current() -> usize {
        0
    }
    pub fn core(cpu: usize) -> Vec<usize> {
        vec![cpu]
    }
    pub fn l3(_: usize) -> usize {
        0
    }
}

/// Where the caller and the workers run (`pin=`).
#[derive(Clone)]
struct Pin {
    caller: Option<Vec<usize>>,
    workers: Option<Vec<usize>>,
}

impl Pin {
    fn parse(s: &str) -> Pin {
        let all = cpu::allowed();
        let first = all.first().copied().unwrap_or(0);
        let core = cpu::core(first);
        let away: Vec<usize> = all.iter().copied().filter(|c| !core.contains(c)).collect();
        match s {
            "none" => Pin {
                caller: None,
                workers: None,
            },
            "caller" => Pin {
                caller: Some(vec![first]),
                workers: None,
            },
            "away" => Pin {
                caller: Some(vec![first]),
                workers: Some(away),
            },
            _ => panic!("pin=none|caller|away"),
        }
    }
    fn workers(&self) {
        if let Some(c) = &self.workers {
            cpu::pin(c);
        }
    }
    fn caller(&self) {
        if let Some(c) = &self.caller {
            cpu::pin(c);
        }
    }
}

fn quantiles(v: &mut [f64]) -> [f64; 3] {
    v.sort_by(f64::total_cmp);
    let q = |p: f64| v[((v.len() - 1) as f64 * p).round() as usize];
    [q(0.5), q(0.1), q(0.9)]
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
    let runner = get("runner", "workers");
    let threads = num("threads", "8") as usize;
    let width = num("width", "8") as usize;
    let pieces = num("pieces", "12") as usize;
    let piece = num("piece", "20000");
    let rounds = num("rounds", "400") as usize;
    let spin = Duration::from_micros(num("spin", &rspool::SPIN.as_micros().to_string()));
    let pin_s = get("pin", "none");
    let pin = Pin::parse(&pin_s);
    let serial_s = get("serial", "lu:256");
    let mut serial = Serial::parse(&serial_s);
    let feed = num("feed", "0") == 1;

    // single-thread references, before any pool runs
    let reference = |f: &mut dyn FnMut()| {
        let mut t: Vec<f64> = (0..50)
            .map(|_| {
                let s = Instant::now();
                f();
                s.elapsed().as_secs_f64()
            })
            .collect();
        quantiles(&mut t)[0]
    };
    let serial_ref = reference(&mut || {
        serial.run(false);
    });
    let piece_ref = reference(&mut || {
        ops(piece);
    });

    let ctx = rsprobe::Context::builder()
        .threshold(rsprobe::Level::DEBUG)
        .build();
    for (k, v) in [
        ("threads", threads as f64),
        ("width", width as f64),
        ("pieces", pieces as f64),
        ("piece", piece as f64),
        ("spin_us", spin.as_micros() as f64),
        ("feed", f64::from(u8::from(feed))),
    ] {
        ctx.param(k, v);
    }

    let mut t_serial = Vec::with_capacity(rounds);
    let mut t_section = Vec::with_capacity(rounds);
    let mut t_start = Vec::with_capacity(rounds);
    let mut t_own = Vec::with_capacity(rounds);
    let (mut moved_core, mut moved_l3) = (0, 0);
    let (remote, pieces_run) = (AtomicU64::new(0), AtomicU64::new(0));
    let cores: Vec<usize> = (0..1024)
        .map(|c| cpu::core(c).first().copied().unwrap_or(c))
        .collect();
    let l3s: Vec<usize> = (0..1024).map(cpu::l3).collect();
    let mut body = || {
        pin.caller();
        let me = std::thread::current().id();
        let mut last = cpu::current();
        for r in 0..rounds {
            let s = Instant::now();
            {
                let _s = rsprobe::scope!("serial");
                serial.run(feed);
            }
            let serial_s = s.elapsed().as_secs_f64();
            let first = AtomicU64::new(u64::MAX);
            let own = AtomicU64::new(u64::MAX);
            let s = Instant::now();
            {
                let _s = rsprobe::scope!("section");
                let here = l3s[cpu::current()];
                let rows = Rows(serial.w.as_mut_ptr());
                let (a, n) = (&serial.a, serial.n);
                let warm = r >= rounds / 10;
                let f = |i: usize| {
                    let at = s.elapsed().as_nanos() as u64;
                    if std::thread::current().id() != me {
                        first.fetch_min(at, Relaxed);
                    } else {
                        own.fetch_min(at, Relaxed);
                    }
                    if warm {
                        pieces_run.fetch_add(1, Relaxed);
                        remote.fetch_add(u64::from(l3s[cpu::current()] != here), Relaxed);
                    }
                    if feed {
                        // SAFETY: piece i alone writes its rows, the serial
                        // work runs after the section
                        unsafe { write_rows(rows, a, n, i, pieces) };
                    }
                    ops(piece);
                };
                if runner == "rayon" {
                    #[cfg(feature = "rayon")]
                    rspool::rayon::run_width(pieces, width, &f);
                } else {
                    <rspool::Here as rspool::Place>::run_width(pieces, width, &f);
                }
            }
            let section_s = s.elapsed().as_secs_f64();
            let now = cpu::current();
            // the first rounds wake the pool
            if r >= rounds / 10 {
                t_serial.push(serial_s);
                t_section.push(section_s);
                let f = first.load(Relaxed);
                if f != u64::MAX {
                    t_start.push(f as f64 * 1e-9);
                }
                let o = own.load(Relaxed);
                if o != u64::MAX {
                    t_own.push(o as f64 * 1e-9);
                }
                moved_core += usize::from(cores[now] != cores[last]);
                moved_l3 += usize::from(l3s[now] != l3s[last]);
            }
            last = now;
        }
    };

    let t = Instant::now();
    match runner.as_str() {
        "workers" => {
            // threads inherit the affinity of the thread that starts them
            let all = cpu::allowed();
            pin.workers();
            let pool: Arc<dyn rspool::Pool> = Arc::new(rspool::Workers::with_spin(threads, spin));
            if !all.is_empty() {
                cpu::pin(&all);
            }
            let _on = ctx.enter();
            rspool::install(pool, &mut body);
        }
        #[cfg(feature = "rayon")]
        "rayon" => {
            let p = pin.clone();
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .start_handler(move |_| p.workers())
                .build()
                .expect("a rayon pool");
            pool.install(|| {
                let _on = ctx.enter();
                body()
            });
        }
        _ => panic!("runner=workers|rayon (rayon with the feature)"),
    }
    let wall = t.elapsed().as_secs_f64();
    let n = t_serial.len().max(1);
    let [sm, s10, s90] = quantiles(&mut t_serial);
    let [cm, c10, c90] = quantiles(&mut t_section);
    let median = |v: &mut Vec<f64>| {
        if v.is_empty() {
            f64::NAN
        } else {
            quantiles(v)[0]
        }
    };
    let (start, issue) = (median(&mut t_start), median(&mut t_own));
    eprintln!(
        "{runner} threads {threads} width {width} spin {} pin {pin_s} serial {serial_s} feed {}: serial {:.1} us [{:.1}..{:.1}] (alone {:.1}), section {:.1} us [{:.1}..{:.1}] (piece alone {:.1}), start {:.2} us, issue {:.2} us, remote pieces {:.1}%, moved core {moved_core}/{n} l3 {moved_l3}/{n}, wall {wall:.2} s",
        spin.as_micros(),
        u8::from(feed),
        sm * 1e6,
        s10 * 1e6,
        s90 * 1e6,
        serial_ref * 1e6,
        cm * 1e6,
        c10 * 1e6,
        c90 * 1e6,
        piece_ref * 1e6,
        start * 1e6,
        issue * 1e6,
        100.0 * remote.load(Relaxed) as f64 / pieces_run.load(Relaxed).max(1) as f64,
    );
    if let Ok(path) = std::env::var("RSPROBE_RECORD") {
        std::fs::write(path, ctx.record().to_string()).expect("the record written");
    }
}
