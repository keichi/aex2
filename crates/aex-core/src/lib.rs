//! Core layer of AEX2.
//!
//! Holds the server-side logic — backends, selection resolution, reductions —
//! independent of tokio and tonic, so its unit tests stay fast.

pub mod dtype;
pub mod error;

pub use dtype::{DType, ALL_DTYPES};
pub use error::{AexError, ErrorClass, Result};
