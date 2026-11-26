//! A section is decided in a library's inner loop, so the decision and a
//! serial run must not allocate. The count is the calling thread's: the
//! workers allocate on their own threads.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Arc;

use rspool::{Here, Pool, Rate, Work, Workers};

thread_local!(static ALLOCS: Cell<usize> = const { Cell::new(0) });

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
        unsafe { System.realloc(p, l, new) }
    }
}

#[global_allocator]
static A: Counting = Counting;

fn allocs(mut f: impl FnMut()) -> usize {
    f();
    let before = ALLOCS.with(Cell::get);
    for _ in 0..100 {
        f();
    }
    ALLOCS.with(Cell::get) - before
}

#[test]
fn deciding_a_section_allocates_nothing() {
    let r = Rate::new(1.0);
    let decide = |ops: f64| {
        let work = Work::of(ops, &r);
        std::hint::black_box(work.go::<Here>(64));
    };
    assert_eq!(allocs(|| decide(1e3)), 0, "without a pool");
    let w: Arc<dyn Pool> = Arc::new(Workers::new(4).unwrap());
    rspool::install(w, || {
        assert_eq!(allocs(|| decide(1.0)), 0, "small work");
        assert_eq!(allocs(|| decide(1e3)), 0, "decided work");
    });
}

#[cfg(feature = "rayon")]
#[test]
fn deciding_a_section_in_a_piece_of_an_unmeasured_rayon_pool_allocates_nothing() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    pool.install(|| {
        rspool::rayon::run_width(2, 2, &|_| {
            let decide = || {
                std::hint::black_box(rspool::rayon::width(1e6, 64));
            };
            assert_eq!(allocs(decide), 0);
        });
    });
}
