//! A pool of threads that steal work, kept awake across the short serial
//! stretches between sections.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::SeqCst};
use std::sync::{Arc, Condvar, Mutex};

use crate::constants::{SPIN, SPIN_YIELD_EVERY};
use crate::{lock, Dispatch, Pool, Spawn};

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
pub struct Workers {
    shared: Arc<Shared>,
    threads: usize,
    handles: Vec<std::thread::JoinHandle<()>>,
    /// What a section costs on them, measured when they start.
    dispatch: Dispatch,
}

/// A piece of work in a queue: what it runs on and how.
#[derive(Clone, Copy)]
struct Job {
    data: *const (),
    run: unsafe fn(*const ()),
}

// SAFETY: a job's data is the frame or the box of the section that queued
// it, which outlives the job (see the sections).
unsafe impl Send for Job {}

impl Job {
    /// # Safety
    /// Its section is waiting for it.
    unsafe fn execute(self) {
        unsafe { (self.run)(self.data) }
    }
}

/// What the workers and the threads starting sections share.
struct Shared {
    /// Queue `0` is the outside threads', queue `k` worker `k`'s.
    queues: Vec<Mutex<VecDeque<Job>>>,
    /// Jobs in all the queues.
    queued: AtomicUsize,
    shutdown: AtomicBool,
    spin: std::time::Duration,
    /// Sleeping workers, waiting on `wake` for work.
    sleepers: AtomicUsize,
    sleep: Mutex<()>,
    wake: Condvar,
}

thread_local! {
    /// The pool this thread works for and its queue there (`None` outside
    /// any pool).
    static WORKER: std::cell::Cell<Option<(*const Shared, usize)>> =
        const { std::cell::Cell::new(None) };
}

impl Shared {
    /// This thread's queue in this pool: its own for a worker, else the
    /// outside threads'.
    fn own(&self) -> usize {
        match WORKER.with(|w| w.get()) {
            Some((p, k)) if std::ptr::eq(p, self) => k,
            _ => 0,
        }
    }

    fn push(&self, job: Job) {
        lock(&self.queues[self.own()]).push_back(job);
        self.queued.fetch_add(1, SeqCst);
        // Queued, then look: a worker going to sleep either is seen here or
        // sees the job (both sequentially consistent); it waits holding the
        // lock from its look on, so the notice reaches it.
        if self.sleepers.load(SeqCst) > 0 {
            let _g = lock(&self.sleep);
            self.wake.notify_one();
        }
    }

    /// Take `job` back from this thread's queue if nobody took it yet.
    fn retract(&self, job: Job) -> bool {
        let mut q = lock(&self.queues[self.own()]);
        let at = q.iter().rposition(|j| std::ptr::eq(j.data, job.data));
        let found = at.and_then(|i| q.remove(i)).is_some();
        drop(q);
        if found {
            self.queued.fetch_sub(1, SeqCst);
        }
        found
    }

    /// A queued job: the newest of this thread's queue, else the oldest of
    /// another.
    fn find(&self) -> Option<Job> {
        if self.queued.load(SeqCst) == 0 {
            return None;
        }
        let own = self.own();
        let n = self.queues.len();
        let mine = lock(&self.queues[own]).pop_back();
        let job =
            mine.or_else(|| (1..n).find_map(|d| lock(&self.queues[(own + d) % n]).pop_front()))?;
        self.queued.fetch_sub(1, SeqCst);
        Some(job)
    }

    /// Run queued work until `done` holds.
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

    /// A worker's life: run work while there is some, wait for more awake
    /// for `spin`, then asleep.
    fn work(&self, k: usize) {
        WORKER.with(|w| w.set(Some((self as *const Shared, k))));
        let mut since = std::time::Instant::now();
        let mut idle = 0u32;
        loop {
            if self.shutdown.load(SeqCst) {
                return;
            }
            if let Some(job) = self.find() {
                // SAFETY: a queued job's section waits for it.
                unsafe { job.execute() };
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
            // Announce, then look (see `push`).
            let mut g = lock(&self.sleep);
            self.sleepers.fetch_add(1, SeqCst);
            while self.queued.load(SeqCst) == 0 && !self.shutdown.load(SeqCst) {
                g = self.wake.wait(g).unwrap_or_else(|e| e.into_inner());
            }
            self.sleepers.fetch_sub(1, SeqCst);
            drop(g);
            since = std::time::Instant::now();
        }
    }
}

/// A panic out of a job, carried to the thread that waits for it.
type Panic = Mutex<Option<Box<dyn std::any::Any + Send>>>;

fn catch(slot: &Panic, f: impl FnOnce()) {
    if let Err(p) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        lock(slot).get_or_insert(p);
    }
}

fn resume(slot: Panic) {
    if let Some(p) = slot.into_inner().unwrap_or_else(|e| e.into_inner()) {
        std::panic::resume_unwind(p);
    }
}

/// A `run` in progress, on the frame of the thread that started it: its
/// pieces, taken in order, and its helpers queued for other threads, each
/// taking pieces too.
struct RunJob<'a> {
    f: &'a (dyn Fn(usize) + Sync),
    n: usize,
    next: AtomicUsize,
    done: AtomicUsize,
    /// Helpers queued and not yet finished or taken back.
    out: AtomicUsize,
    panic: Panic,
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

    /// # Safety
    /// `data` is a `RunJob` whose `out` counts this helper.
    unsafe fn help(data: *const ()) {
        let job = unsafe { &*(data as *const RunJob<'_>) };
        job.take();
        // The last touch: the starting thread may return from here on.
        job.out.fetch_sub(1, SeqCst);
    }
}

/// One side of a `join`, on the frame of the thread that started it.
struct SideJob<'a> {
    f: Mutex<Option<&'a mut (dyn FnMut() + Send)>>,
    done: AtomicBool,
    panic: Panic,
}

impl SideJob<'_> {
    /// # Safety
    /// `data` is a `SideJob` its `join` waits for.
    unsafe fn run(data: *const ()) {
        let job = unsafe { &*(data as *const SideJob<'_>) };
        if let Some(f) = lock(&job.f).take() {
            catch(&job.panic, f);
        }
        // The last touch: the joining thread may return from here on.
        job.done.store(true, SeqCst);
    }
}

/// A `scope` in progress: its tasks not yet finished.
struct ScopeJob<'p> {
    shared: &'p Shared,
    open: AtomicUsize,
    panic: Panic,
}

/// A task of a scope, boxed, with its scope.
struct TaskJob<'s> {
    scope: *const ScopeJob<'s>,
    task: Box<dyn FnOnce(&dyn Spawn<'s>) + Send + 's>,
}

impl<'s> Spawn<'s> for ScopeJob<'s> {
    fn spawn(&self, task: Box<dyn FnOnce(&dyn Spawn<'s>) + Send + 's>) {
        self.open.fetch_add(1, SeqCst);
        let boxed = Box::new(TaskJob {
            scope: self as *const ScopeJob<'s>,
            task,
        });
        self.shared.push(Job {
            data: Box::into_raw(boxed) as *const (),
            run: TaskJob::run,
        });
    }
}

impl TaskJob<'_> {
    /// # Safety
    /// `data` is a boxed `TaskJob` whose scope waits for it.
    unsafe fn run(data: *const ()) {
        let TaskJob { scope, task } = *unsafe { Box::from_raw(data as *mut TaskJob<'_>) };
        let scope = unsafe { &*scope };
        catch(&scope.panic, || task(scope));
        // The last touch: the scope may return from here on.
        scope.open.fetch_sub(1, SeqCst);
    }
}

impl Workers {
    /// A pool of `threads` threads (the caller one of them), waiting
    /// [`SPIN`] for work before they sleep.
    pub fn new(threads: usize) -> Self {
        Self::with_spin(threads, SPIN)
    }

    /// [`new`](Self::new) with the workers waiting `spin` for work.
    pub fn with_spin(threads: usize, spin: std::time::Duration) -> Self {
        let threads = threads.max(1);
        let shared = Arc::new(Shared {
            queues: (0..threads).map(|_| Mutex::new(VecDeque::new())).collect(),
            queued: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
            spin,
            sleepers: AtomicUsize::new(0),
            sleep: Mutex::new(()),
            wake: Condvar::new(),
        });
        let handles = (1..threads)
            .map(|k| {
                let sh = shared.clone();
                std::thread::Builder::new()
                    .name(format!("rspool-worker-{k}"))
                    .spawn(move || sh.work(k))
                    .expect("a worker thread")
            })
            .collect();
        let mut w = Workers {
            shared,
            threads,
            handles,
            dispatch: Dispatch::never(),
        };
        let measured = Dispatch::measure(threads, &|k, piece| w.run_width(k, k, piece));
        w.dispatch = measured;
        w
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
            out: AtomicUsize::new(0),
            panic: Mutex::new(None),
        };
        let helper = Job {
            data: &job as *const RunJob<'_> as *const (),
            run: RunJob::help,
        };
        let helpers = (n - 1).min(width - 1);
        job.out.store(helpers, SeqCst);
        for _ in 0..helpers {
            sh.push(helper);
        }
        job.take();
        // Every piece is taken: the helpers nobody started come back.
        while job.out.load(SeqCst) > 0 && sh.retract(helper) {
            job.out.fetch_sub(1, SeqCst);
        }
        sh.wait_until(|| job.done.load(SeqCst) == n && job.out.load(SeqCst) == 0);
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
        };
        let job = Job {
            data: &side as *const SideJob<'_> as *const (),
            run: SideJob::run,
        };
        sh.push(job);
        let first: Panic = Mutex::new(None);
        catch(&first, a);
        if sh.retract(job) {
            // SAFETY: taken back, so this thread alone runs it.
            unsafe { job.execute() };
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
        // SAFETY: the tasks borrow nothing that outlives the scope beyond
        // `'s`, and the scope returns only once every task ran.
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
            let _g = lock(&self.shared.sleep);
            self.shared.wake.notify_all();
        }
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}
