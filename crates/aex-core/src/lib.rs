//! Core layer of AEX2.
//!
//! Holds the server-side logic — backends, selection resolution, reductions —
//! independent of tokio and tonic, so its unit tests stay fast.
//!
//! As of M1 that means element types ([`dtype`]), error classification
//! ([`error`]), the backend traits ([`backend`]) and the `.npy` backend
//! ([`backends::npy`]).

pub mod backend;
pub mod backends;
pub mod dtype;
pub mod error;

pub use backend::{ArrayDataset, ArrayFile, Item};
pub use backends::npy::{NpyDataset, NpyFile};
pub use dtype::{DType, ALL_DTYPES};
pub use error::{AexError, ErrorClass, Result};
