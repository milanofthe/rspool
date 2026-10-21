//! A monotonic clock that is inert where there is none: on
//! wasm32-unknown-unknown `std::time::Instant::now()` panics, so there every
//! instant reads zero time. Rates learn nothing there, and sections decide on
//! their priors; nothing traps.

#[cfg(not(target_arch = "wasm32"))]
pub use std::time::Instant;

/// An instant of a target without a clock: every duration is zero.
#[cfg(target_arch = "wasm32")]
#[derive(Clone, Copy, Debug)]
pub struct Instant;

#[cfg(target_arch = "wasm32")]
impl Instant {
    pub fn now() -> Self {
        Instant
    }
    pub fn elapsed(&self) -> std::time::Duration {
        std::time::Duration::ZERO
    }
}
