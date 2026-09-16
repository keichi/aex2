//! Backend implementations.
//!
//! Only `.npy` for now. The shared `ArrayFile` / `ArrayDataset` traits arrive
//! with selection resolution in M2, alongside their first real caller.

pub mod npy;
