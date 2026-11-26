//! What a set of worker threads offers: sections of pieces, two sides, a
//! scope of tasks, and what a section costs on it.

use crate::{Dispatch, Origin};

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
    /// Run `f` once where handing work to the pool is cheapest, returning
    /// when it ran: on one of its workers, for a pool whose hand-off from
    /// outside costs more (a thread put to sleep and woken). A panic of `f`
    /// reaches the caller. The default runs it here.
    fn enter(&self, f: &mut (dyn FnMut() + Send)) {
        f()
    }
    /// Run `a` and `b` once each, side by side where the pool can,
    /// returning when both ran. A panic of either reaches the caller (the
    /// other one need not run then). The default runs one after the other.
    fn join(&self, a: &mut (dyn FnMut() + Send), b: &mut (dyn FnMut() + Send)) {
        a();
        b();
    }
    /// Run `body` once with a [`Spawn`] its tasks start tasks through,
    /// returning once `body` and every task ran. A panic of `body` or of a
    /// task reaches the caller (tasks not started yet need not run then).
    /// The default runs the tasks one after the other after `body`.
    fn scope<'s>(&self, body: &mut (dyn FnMut(&dyn Spawn<'s>) + Send)) {
        queued(|s| body(s))
    }
}

pub(crate) fn queued<'s, R>(body: impl FnOnce(&dyn Spawn<'s>) -> R) -> R {
    let queue = Queue(std::sync::Mutex::new(std::collections::VecDeque::new()));
    let r = body(&queue);
    loop {
        let task = crate::lock_ignoring_poison(&queue.0).pop_front();
        match task {
            Some((origin, task)) => origin.run(|| task(&queue)),
            None => return r,
        }
    }
}

pub(crate) struct Unfinished;

pub(crate) enum Once<F, R> {
    Ready(F),
    Done(std::thread::Result<R>),
}

impl<F, R> Once<F, R> {
    pub(crate) fn run(&mut self, call: impl FnOnce(F) -> R) {
        let unfinished = Once::Done(Err(Box::new(Unfinished)));
        *self = match std::mem::replace(self, unfinished) {
            Once::Ready(f) => Once::Done(std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                || call(f),
            ))),
            done => done,
        };
    }

    pub(crate) fn finish(self, call: impl FnOnce(F) -> R) -> R {
        match self {
            Once::Ready(f) => call(f),
            Once::Done(Ok(r)) => r,
            Once::Done(Err(p)) => std::panic::resume_unwind(p),
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

struct Queue<'s>(std::sync::Mutex<std::collections::VecDeque<(Origin, Task<'s>)>>);

impl<'s> Spawn<'s> for Queue<'s> {
    fn spawn(&self, task: Task<'s>) {
        crate::lock_ignoring_poison(&self.0).push_back((Origin::here(), task));
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
