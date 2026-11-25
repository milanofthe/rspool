//! A section's work as its caller estimates it, and how wide that makes it
//! run.

use crate::{Piece, Place};

/// What an op of some kind of work costs, in ns: the caller's constant
/// estimate. Nothing is measured or learned at run time; a section's work is
/// its ops at this cost ([`Work::of`]).
#[derive(Debug)]
pub struct Rate {
    ns: f32,
}

impl Rate {
    /// `ns` ns per op.
    pub const fn new(ns: f32) -> Self {
        Rate { ns }
    }

    /// ns per op.
    pub fn ns_per_op(&self) -> f64 {
        self.ns as f64
    }
}

/// A section's work: `ops` at a [`Rate`] ([`of`](Self::of)), or an estimate
/// in ns ([`ns`](Self::ns)).
#[derive(Clone, Copy, Debug)]
pub struct Work<'r> {
    ops: f64,
    rate: Option<&'r Rate>,
}

/// How a section runs ([`Work::go`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Go {
    /// Serially on the calling thread.
    Here,
    /// On this many threads.
    Wide(usize),
}

impl<'r> Work<'r> {
    /// `ns` ns of work.
    pub fn ns(ns: f64) -> Self {
        Work {
            ops: ns,
            rate: None,
        }
    }

    /// `ops` ops at `rate`.
    pub fn of(ops: f64, rate: &'r Rate) -> Self {
        Work {
            ops,
            rate: Some(rate),
        }
    }

    /// The work in ns.
    pub fn time(&self) -> f64 {
        match self.rate {
            Some(r) => self.ops * r.ns_per_op(),
            None => self.ops,
        }
    }

    /// How the section runs in `pieces` pieces at the place `P`: on the
    /// width its work is worth against what a section costs there
    /// ([`Place::width`]), serially below two threads.
    pub fn go<P: Place>(&self, pieces: usize) -> Go {
        match P::width(self.time(), pieces) {
            k if k >= 2 => Go::Wide(k),
            _ => Go::Here,
        }
    }

    /// The section, `n` pieces of this work: serially or on as many threads
    /// of `P` as it is worth.
    pub fn run<P: Place>(&self, n: usize, f: &Piece<'_>) {
        P::run_width(n, P::width(self.time(), n), f)
    }
}
