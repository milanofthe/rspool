//! A pool of threads that steal work, kept awake across the short serial
//! stretches between sections.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::SeqCst};
use std::sync::{Arc, Condvar, Mutex};

use crate::constants::{SPIN, SPIN_YIELD_EVERY};
use crate::{lock_ignoring_poison, Dispatch, Origin, Pool, Spawn};

/// A pool of threads that steal work: `threads - 1` workers and the thread
/// that starts a section, which takes part in it. A section runs on as
/// many of them as it asks for ([`Pool::run_width`]); what that costs is
/// measured once, when they start ([`Pool::dispatch`]). Every worker keeps a
/// queue of its own, the threads outside the pool share one; a thread
/// looking for work takes the newest of its own queue, else the oldest of
/// the outside one or another worker's. A thread waiting for its section to
/// finish keeps taking work meanwhile, so sections nest on the pool: what a
/// piece of work opens, the waiting threads run. After the work runs out
/// the workers keep looking for [`SPIN`] (as OpenMP's active wait), then
/// sleep until work comes; so the stages of a solve, a few tens of
/// microseconds apart, reach awake workers, and an idle pool costs nothing.
/// A sleeping worker is woken only for work the workers looking for it do
/// not cover, so a pool wider than its sections keeps about as many
/// workers awake as the sections take.
pub struct Workers {
    shared: Arc<Shared>,
    threads: usize,
    handles: Vec<std::thread::JoinHandle<()>>,
    dispatch: Dispatch,
}

#[derive(Clone, Copy)]
struct Job {
    data: *const (),
    run: unsafe fn(*const ()),
}

// SAFETY: a job's data is the frame or the box of the section that queued it, which outlives the job.
unsafe impl Send for Job {}

impl Job {
    unsafe fn execute(self) {
        unsafe { (self.run)(self.data) }
    }
}

struct Shared {
    queues: Vec<Mutex<VecDeque<Job>>>,
    queued: AtomicUsize,
    shutdown: AtomicBool,
    spin: std::time::Duration,
    spinning: AtomicUsize,
    sleepers: AtomicUsize,
    sleep: Mutex<()>,
    wake: Condvar,
}

const OUTSIDE_QUEUE: usize = 0;

thread_local! {
    static WORKER: std::cell::Cell<Option<(*const Shared, usize)>> =
        const { std::cell::Cell::new(None) };
}

impl Shared {
    fn own(&self) -> usize {
        match WORKER.with(|w| w.get()) {
            Some((p, k)) if std::ptr::eq(p, self) => k,
            _ => OUTSIDE_QUEUE,
        }
    }

    fn push(&self, job: Job) {
        lock_ignoring_poison(&self.queues[self.own()]).push_back(job);
        let queued = self.queued.fetch_add(1, SeqCst).wrapping_add(1);
        let uncovered_by_spinning = self.spinning.load(SeqCst) < queued;
        if uncovered_by_spinning && self.sleepers.load(SeqCst) > 0 {
            let _g = lock_ignoring_poison(&self.sleep);
            self.wake.notify_one();
        }
    }

    fn retract(&self, job: Job) -> bool {
        let mut q = lock_ignoring_poison(&self.queues[self.own()]);
        let at = q.iter().rposition(|j| std::ptr::eq(j.data, job.data));
        let found = at.and_then(|i| q.remove(i)).is_some();
        drop(q);
        if found {
            self.queued.fetch_sub(1, SeqCst);
        }
        found
    }

    fn find(&self) -> Option<Job> {
        if self.queued.load(SeqCst) == 0 {
            return None;
        }
        let own = self.own();
        let n = self.queues.len();
        let mine = lock_ignoring_poison(&self.queues[own]).pop_back();
        let job = mine.or_else(|| {
            (1..n).find_map(|d| lock_ignoring_poison(&self.queues[(own + d) % n]).pop_front())
        })?;
        self.queued.fetch_sub(1, SeqCst);
        Some(job)
    }

    fn wait_until(&self, done: impl Fn() -> bool) {
        let mut idle = 0u32;
        while !done() {
            match self.find() {
                // SAFETY: a queued job's section waits for it.
                Some(job) => unsafe { job.execute() },
                None => {
                    idle += 1;
                    if idle.is_multiple_of(SPIN_YIELD_EVERY) {
                        std::thread::yield_now();
                    } else {
                        std::hint::spin_loop();
                    }
                }
            }
        }
    }

    fn work(&self, k: usize) {
        WORKER.with(|w| w.set(Some((self as *const Shared, k))));
        let mut since = std::time::Instant::now();
        let mut idle = 0u32;
        self.spinning.fetch_add(1, SeqCst);
        loop {
            if self.shutdown.load(SeqCst) {
                return;
            }
            if let Some(job) = self.find() {
                self.spinning.fetch_sub(1, SeqCst);
                // SAFETY: a queued job's section waits for it.
                unsafe { job.execute() };
                self.spinning.fetch_add(1, SeqCst);
                since = std::time::Instant::now();
                continue;
            }
            if since.elapsed() < self.spin {
                idle += 1;
                if idle.is_multiple_of(SPIN_YIELD_EVERY) {
                    std::thread::yield_now();
                } else {
                    std::hint::spin_loop();
                }
                continue;
            }
            let mut g = lock_ignoring_poison(&self.sleep);
            self.spinning.fetch_sub(1, SeqCst);
            self.sleepers.fetch_add(1, SeqCst);
            while self.queued.load(SeqCst) == 0 && !self.shutdown.load(SeqCst) {
                g = self.wake.wait(g).unwrap_or_else(|e| e.into_inner());
            }
            self.sleepers.fetch_sub(1, SeqCst);
            self.spinning.fetch_add(1, SeqCst);
            drop(g);
            since = std::time::Instant::now();
        }
    }
}

type Panic = Mutex<Option<Box<dyn std::any::Any + Send>>>;

fn catch(slot: &Panic, f: impl FnOnce()) {
    if let Err(p) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        lock_ignoring_poison(slot).get_or_insert(p);
    }
}

fn resume(slot: Panic) {
    if let Some(p) = slot.into_inner().unwrap_or_else(|e| e.into_inner()) {
        std::panic::resume_unwind(p);
    }
}

struct RunJob<'a> {
    f: &'a (dyn Fn(usize) + Sync),
    n: usize,
    next: AtomicUsize,
    done: AtomicUsize,
    helpers_out: AtomicUsize,
    panic: Panic,
    origin: Origin,
}

impl RunJob<'_> {
    fn take(&self) {
        loop {
            let i = self.next.fetch_add(1, SeqCst);
            if i >= self.n {
                return;
            }
            catch(&self.panic, || (self.f)(i));
            self.done.fetch_add(1, SeqCst);
        }
    }

    unsafe fn help(data: *const ()) {
        // SAFETY: `data` is a `RunJob` whose `helpers_out` counts this helper, so it lives until this helper's decrement, the last touch.
        let job = unsafe { &*(data as *const RunJob<'_>) };
        job.origin.run(|| job.take());
        job.helpers_out.fetch_sub(1, SeqCst);
    }
}

struct SideJob<'a> {
    f: Mutex<Option<&'a mut (dyn FnMut() + Send)>>,
    done: AtomicBool,
    panic: Panic,
    origin: Origin,
}

impl SideJob<'_> {
    fn call_once(&self) {
        if let Some(f) = lock_ignoring_poison(&self.f).take() {
            catch(&self.panic, f);
        }
    }

    unsafe fn run(data: *const ()) {
        // SAFETY: `data` is a `SideJob` that its `join` waits for, so it lives until `done` is set, the last touch.
        let job = unsafe { &*(data as *const SideJob<'_>) };
        job.origin.run(|| job.call_once());
        job.done.store(true, SeqCst);
    }
}

struct ScopeJob<'p> {
    shared: &'p Shared,
    open: AtomicUsize,
    panic: Panic,
}

struct TaskJob<'s> {
    scope: *const ScopeJob<'s>,
    task: Box<dyn FnOnce(&dyn Spawn<'s>) + Send + 's>,
    origin: Origin,
}

impl<'s> Spawn<'s> for ScopeJob<'s> {
    fn spawn(&self, task: Box<dyn FnOnce(&dyn Spawn<'s>) + Send + 's>) {
        self.open.fetch_add(1, SeqCst);
        let boxed = Box::new(TaskJob {
            scope: self as *const ScopeJob<'s>,
            task,
            origin: Origin::here(),
        });
        self.shared.push(Job {
            data: Box::into_raw(boxed) as *const (),
            run: TaskJob::run,
        });
    }
}

impl TaskJob<'_> {
    unsafe fn run(data: *const ()) {
        // SAFETY: `data` is a boxed `TaskJob` from `spawn`, taken from its queue once.
        let TaskJob {
            scope,
            task,
            origin,
        } = *unsafe { Box::from_raw(data as *mut TaskJob<'_>) };
        // SAFETY: the scope waits for this task, so it lives until `open` is decremented, the last touch.
        let scope = unsafe { &*scope };
        origin.run(|| catch(&scope.panic, || task(scope)));
        scope.open.fetch_sub(1, SeqCst);
    }
}

impl Workers {
    /// A pool of `threads` threads (the caller one of them), waiting
    /// [`SPIN`] for work before they sleep.
    ///
    /// # Errors
    /// The error of [`std::thread::Builder::spawn`] if a worker thread cannot
    /// be started. The workers started before it are shut down and joined.
    pub fn new(threads: usize) -> std::io::Result<Self> {
        Self::with_spin(threads, SPIN)
    }

    /// [`new`](Self::new) with the workers waiting `spin` for work.
    ///
    /// # Errors
    /// As [`new`](Self::new).
    pub fn with_spin(threads: usize, spin: std::time::Duration) -> std::io::Result<Self> {
        let threads = threads.max(1);
        let shared = Arc::new(Shared {
            queues: (0..threads).map(|_| Mutex::new(VecDeque::new())).collect(),
            queued: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
            spin,
            spinning: AtomicUsize::new(0),
            sleepers: AtomicUsize::new(0),
            sleep: Mutex::new(()),
            wake: Condvar::new(),
        });
        let mut w = Workers {
            shared,
            threads,
            handles: Vec::with_capacity(threads - 1),
            dispatch: Dispatch::never(),
        };
        for k in 1..threads {
            let sh = w.shared.clone();
            let handle = std::thread::Builder::new()
                .name(format!("rspool-worker-{k}"))
                .spawn(move || sh.work(k))?;
            w.handles.push(handle);
        }
        let measured = Dispatch::measure(threads, &|k, piece| w.run_width(k, k, piece));
        w.dispatch = measured;
        Ok(w)
    }
}

impl Pool for Workers {
    fn threads(&self) -> usize {
        self.threads
    }

    fn run(&self, n: usize, f: &(dyn Fn(usize) + Sync)) {
        self.run_width(n, self.threads, f)
    }

    fn dispatch(&self) -> Dispatch {
        self.dispatch.clone()
    }

    fn run_width(&self, n: usize, width: usize, f: &(dyn Fn(usize) + Sync)) {
        let width = width.min(self.threads);
        if width < 2 || n < 2 {
            (0..n).for_each(f);
            return;
        }
        let sh = &*self.shared;
        let job = RunJob {
            f,
            n,
            next: AtomicUsize::new(0),
            done: AtomicUsize::new(0),
            helpers_out: AtomicUsize::new(0),
            panic: Mutex::new(None),
            origin: Origin::here(),
        };
        let helper = Job {
            data: &job as *const RunJob<'_> as *const (),
            run: RunJob::help,
        };
        let helpers = (n - 1).min(width - 1);
        job.helpers_out.store(helpers, SeqCst);
        for _ in 0..helpers {
            sh.push(helper);
        }
        job.take();
        let mut wait_start_if_recorded = None;
        rsprobe::time!(DEBUG, "wait", {
            wait_start_if_recorded = Some(crate::clock::Instant::now());
            0
        });
        debug_assert!(job.next.load(SeqCst) >= n, "a piece is left for a helper");
        while job.helpers_out.load(SeqCst) > 0 && sh.retract(helper) {
            job.helpers_out.fetch_sub(1, SeqCst);
        }
        sh.wait_until(|| job.done.load(SeqCst) == n && job.helpers_out.load(SeqCst) == 0);
        if let Some(t) = wait_start_if_recorded {
            rsprobe::time!(DEBUG, "wait", t.elapsed().as_nanos());
        }
        resume(job.panic);
    }

    fn join(&self, a: &mut (dyn FnMut() + Send), b: &mut (dyn FnMut() + Send)) {
        if self.threads < 2 {
            a();
            b();
            return;
        }
        let sh = &*self.shared;
        let side = SideJob {
            f: Mutex::new(Some(b)),
            done: AtomicBool::new(false),
            panic: Mutex::new(None),
            origin: Origin::here(),
        };
        let job = Job {
            data: &side as *const SideJob<'_> as *const (),
            run: SideJob::run,
        };
        sh.push(job);
        let first: Panic = Mutex::new(None);
        catch(&first, a);
        if sh.retract(job) {
            side.call_once();
        } else {
            sh.wait_until(|| side.done.load(SeqCst));
        }
        resume(first);
        resume(side.panic);
    }

    fn scope<'s>(&self, body: &mut (dyn FnMut(&dyn Spawn<'s>) + Send)) {
        let scope = ScopeJob {
            shared: &self.shared,
            open: AtomicUsize::new(0),
            panic: Mutex::new(None),
        };
        // SAFETY: the tasks borrow nothing that outlives the scope beyond `'s`, and the scope returns only once every task ran.
        let spawn: &ScopeJob<'s> =
            unsafe { std::mem::transmute::<&ScopeJob<'_>, &ScopeJob<'s>>(&scope) };
        catch(&scope.panic, || body(spawn));
        self.shared.wait_until(|| scope.open.load(SeqCst) == 0);
        resume(scope.panic);
    }
}

impl Drop for Workers {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, SeqCst);
        {
            let _g = lock_ignoring_poison(&self.shared.sleep);
            self.shared.wake.notify_all();
        }
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}
