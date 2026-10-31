# rspool

One set of worker threads that several libraries share, and parallel sections that go exactly as wide as their work is worth on them.

A host lends a pool once; every parallel section in the code it calls, whatever library it lives in, runs on that pool. How many threads a section takes is decided per section and per machine, from two measurements instead of thresholds:

- **what a section costs** the pool on `k` threads, measured when the pool starts (`Dispatch`);
- **what an op of the work costs**, learned from sampled serial runs of it (`Rate`), and **how long its sections took on each width**, per size of the work: a section takes the width that was fastest, a width not yet measured counted at what the section cost predicts, and now and then a neighbouring width, measured, so that a change in the code or the machine is followed. Whatever going wide costs beyond the pool's section cost (output handed between cores, shared memory bandwidth, scratch) is in the measurement; nothing of it is modelled.

A section that is not worth a second thread runs on the calling thread without touching the pool. This is the per-region choice of the thread count the literature tunes online (feedback-driven threading, Otter), for every section of every library sharing the pool. A section on `k` threads leaves its pieces' own sections their share of the pool, so nesting never oversubscribes it. Results never depend on how a section was scheduled.

Without a lent pool, sections run serially, or on the current rayon pool for code that asks for it (feature `rayon`). On targets without a clock (wasm32), nothing is timed and sections decide on their priors.

## License

MIT
