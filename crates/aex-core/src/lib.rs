//! Core layer of AEX2.
//!
//! Holds the server-side logic — backends, selection resolution, reductions —
//! independent of tokio and tonic, so its unit tests stay fast.

pub mod error;

pub use error::{AexError, ErrorClass, Result};
