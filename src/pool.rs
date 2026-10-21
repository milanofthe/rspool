//! What a set of worker threads offers: sections of pieces, two sides, a
//! scope of tasks, and what a section costs on it.

use crate::Dispatch;

/// The pieces of a section: `f(i)` for every `i` of it.
pub type Piece<'a> = dyn Fn(usize) + Sync + 'a;

/// Worker threads to run independent pieces of work on.
pub trait Pool: Send + Sync {
    /// Number of threads, the calling one among them.
    fn threads(&self) -> usize;
    /// Run `f(i)` for every `i` in `0..n`, returning when all ran. A piece
    /// may open sections of its own.
    fn run(&self, n: usize, f: &Piece<'_>);
    /// [`run`](Self::run) on at most `width` threads, the calling one among
    /// them. The default runs on all.
    fn run_width(&self, n: usize, width: usize, f: &Piece<'_>) {
        let _ = width;
        self.run(n, f)
    }
    /// What a section costs on this pool. The default assumes
    /// ([`Dispatch::assumed`]); a pool that measures itself says so.
    fn dispatch(&self) -> Dispatch {
        Dispatch::assumed(self.threads())
    }
    /// Run `f` where handing work to the pool is cheapest: on one of its
    /// workers, for a pool whose hand-off from outside costs more (a thread
    /// put to sleep and woken). The default runs it here.
    fn enter(&self, f: &mut (dyn FnMut() + Send)) {
        f()
    }
    /// Run `a` and `b`, side by side where the pool can, returning when
    /// both ran. The default runs one after the other.
    fn join(&self, a: &mut (dyn FnMut() + Send), b: &mut (dyn FnMut() + Send)) {
        a();
        b();
    }
    /// Run `body` with a [`Spawn`] its tasks start tasks through, returning
    /// once every task ran. The default runs the tasks one after the other
    /// after `body`.
    fn scope<'s>(&self, body: &mut (dyn FnMut(&dyn Spawn<'s>) + Send)) {
        let queue = Queue(std::sync::Mutex::new(std::collections::VecDeque::new()));
        body(&queue);
        loop {
            let task = crate::lock(&queue.0).pop_front();
            match task {
                Some(task) => task(&queue),
                None => return,
            }
        }
    }
}

/// A task of a scope (see [`Pool::scope`]).
pub type Task<'s> = Box<dyn FnOnce(&dyn Spawn<'s>) + Send + 's>;

/// What a scope's tasks are started through (see [`Pool::scope`]).
pub trait Spawn<'s>: Sync {
    /// Start `task`; its scope returns once it ran.
    fn spawn(&self, task: Task<'s>);
}

/// The tasks of a scope, run one after the other.
struct Queue<'s>(std::sync::Mutex<std::collections::VecDeque<Task<'s>>>);

impl<'s> Spawn<'s> for Queue<'s> {
    fn spawn(&self, task: Task<'s>) {
        crate::lock(&self.0).push_back(task);
    }
}

/// A pool of the calling thread alone: every section runs in order where it
/// is opened. Lent where the sections inside some work are to stay on its
/// thread whatever the code inside asks for.
#[derive(Clone, Copy, Debug, Default)]
pub struct Serial;

impl Pool for Serial {
    fn threads(&self) -> usize {
        1
    }
    fn run(&self, n: usize, f: &Piece<'_>) {
        (0..n).for_each(f)
    }
    fn dispatch(&self) -> Dispatch {
        Dispatch::never()
    }
}
