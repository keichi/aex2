# AEX2

Array data transfer over wide-area networks. Rust, with a gRPC control plane and
a raw-TCP data plane. The design lives in `SPEC.md`; milestones are in its §13.

## Comments

- Write every comment in English, including in tests and in `.toml` / `.yml`.
- Keep them short. One line where one line will do.
- Never cite `SPEC.md` section numbers from a comment — they go stale. Say the
  reason in the comment itself. Prose files such as `README.md` may cite them.
- Explain why, not what. Skip comments that restate the code.

## Git

Commit one feature at a time. A commit should be a single self-contained change
that builds and passes its tests on its own; do not bundle unrelated work.

## Checks

```console
$ cargo fmt --all
$ cargo clippy --all-targets --all-features -- -D warnings
$ cargo test --all-features && cargo test --release --all-features
$ cargo check --all-targets
```

Run both profiles: overflow checks make debug and release behave differently.
`--all-features` needs libhdf5 >= 1.14; the last line keeps the default build
compiling without it.

## Layout

Rust tests are `#[cfg(test)] mod tests` inline in the file they cover.
`tests/rust/` is reserved for integration tests once a server and client exist.
