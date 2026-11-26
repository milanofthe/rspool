//! Results of the closures a lent pool runs: what they return comes back
//! from `enter`, `join` and `scope`, a closure the pool does not run runs
//! on the caller, and a panic of a closure reaches the caller.

use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::sync::Arc;

use rspool::{Dispatch, Lent, Piece, Pool, Serial, Spawn, Workers};

struct Idle;

impl Pool for Idle {
    fn threads(&self) -> usize {
        2
    }
    fn run(&self, n: usize, f: &Piece<'_>) {
        (0..n).for_each(f)
    }
    fn enter(&self, _: &mut (dyn FnMut() + Send)) {}
    fn join(&self, _: &mut (dyn FnMut() + Send), _: &mut (dyn FnMut() + Send)) {}
    fn scope<'s>(&self, _: &mut (dyn FnMut(&dyn Spawn<'s>) + Send)) {}
}

fn spawn_tree<'s>(s: &dyn Spawn<'s>, ran: &'s AtomicUsize) {
    for _ in 0..8 {
        s.spawn(Box::new(move |s: &dyn Spawn<'s>| {
            ran.fetch_add(1, Relaxed);
            s.spawn(Box::new(move |_: &dyn Spawn<'s>| {
                ran.fetch_add(1, Relaxed);
            }));
        }));
    }
}

fn returns_every_result(lent: &Lent) {
    assert_eq!(lent.enter(|| 7), 7);
    let (a, b) = lent.join(|| vec![1, 2], || String::from("b"));
    assert_eq!((a, b.as_str()), (vec![1, 2], "b"));
    let ran = AtomicUsize::new(0);
    let r = lent.scope(|s| {
        spawn_tree(s, &ran);
        "body"
    });
    assert_eq!(r, "body");
    assert_eq!(
        ran.into_inner(),
        16,
        "every task ran before the scope returned"
    );
}

#[test]
fn enter_join_and_scope_return_what_their_closures_return() {
    returns_every_result(&Lent::new(Arc::new(Workers::new(4).unwrap())));
    returns_every_result(&Lent::new(Arc::new(Workers::new(1).unwrap())));
    returns_every_result(&Lent::new(Arc::new(Serial)));
}

#[test]
fn a_closure_the_pool_does_not_run_runs_on_the_caller() {
    let lent = Lent {
        pool: Arc::new(Idle),
        dispatch: Dispatch::never(),
    };
    returns_every_result(&lent);
    let here = std::thread::current().id();
    let (a, b) = lent.join(
        || std::thread::current().id(),
        || std::thread::current().id(),
    );
    assert_eq!((a, b), (here, here));
    assert_eq!(lent.enter(|| std::thread::current().id()), here);
    assert_eq!(lent.scope(|_| std::thread::current().id()), here);
}

fn message(p: Box<dyn std::any::Any + Send>) -> String {
    match p.downcast::<&str>() {
        Ok(s) => s.to_string(),
        Err(p) => p.downcast::<String>().map(|s| *s).unwrap_or_default(),
    }
}

fn caught<R>(f: impl FnOnce() -> R) -> String {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(_) => String::new(),
        Err(p) => message(p),
    }
}

#[test]
fn a_panic_of_a_closure_reaches_the_caller() {
    for pool in [
        Arc::new(Workers::new(3).unwrap()) as Arc<dyn Pool>,
        Arc::new(Serial),
        Arc::new(Idle),
    ] {
        let lent = Lent::new(pool);
        assert_eq!(caught(|| lent.enter(|| panic!("enter"))), "enter");
        let other_side = AtomicUsize::new(0);
        let side = caught(|| {
            lent.join(
                || -> usize { panic!("a") },
                || other_side.fetch_add(1, Relaxed),
            )
        });
        assert_eq!(side, "a");
        assert_eq!(other_side.into_inner(), 1, "the other side ran");
        assert_eq!(caught(|| lent.join(|| 1, || panic!("b"))), "b");
        assert_eq!(caught(|| lent.scope(|_| panic!("body"))), "body");
        assert_eq!(lent.join(|| 1, || 2), (1, 2), "the pool runs after a panic");
    }
}
