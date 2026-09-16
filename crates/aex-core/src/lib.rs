//! Core layer of AEX2.
//!
//! Holds the server-side logic — backends, selection resolution, reductions —
//! independent of tokio and tonic, so its unit tests stay fast.
//!
//! As of M0 that means element types ([`dtype`]), error classification
//! ([`error`]), and the `.npy` backend ([`backends::npy`]).

pub mod backends;
pub mod dtype;
pub mod error;

pub use backends::npy::NpyFile;
pub use dtype::{DType, ALL_DTYPES};
pub use error::{AexError, ErrorClass, Result};
