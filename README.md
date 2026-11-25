# rspool

One set of worker threads that several libraries share, and parallel sections that go exactly as wide as their work is worth on them.

A host lends a pool once; every parallel section in the code it calls, whatever library it lives in, runs on that pool. How many threads a section takes is decided per section and per machine: the caller's estimate of its work (ops at a constant cost per op, `Rate`, or ns, `Work`) against **what a section costs** the pool on `k` threads, measured when the pool starts (`Dispatch`). A section goes as wide as that model makes it shortest, if that saves a fixed fraction of its serial time (`GAIN`).

A section that is not worth a second thread runs on the calling thread without touching the pool. A section on `k` threads leaves its pieces' own sections their share of the pool, so nesting never oversubscribes it. Results never depend on how a section was scheduled.

Without a lent pool, sections run serially, or on the current rayon pool for code that asks for it (feature `rayon`). On targets without a clock (wasm32), nothing is timed.

A task carries its issuer's [rsprobe](https://github.com/milanofthe/rsprobe) scope, and every section counts its width there at level DEBUG, and on `Workers` the caller's wait for its helpers.
rspool depends on rsprobe without features; a binary that wants a record enables `rsprobe/on` itself.

## License

MIT
