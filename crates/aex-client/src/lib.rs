//! The AEX2 Rust client.
//!
//! Usable on its own, and the layer the Python bindings sit on.
//! Keeping it separate from the bindings also lets a benchmark measure the
//! transfer without Python in the picture.
//!
//! It speaks both planes: the control plane for sessions, files, metadata and
//! the resolution of a selection, and the data plane for the bytes themselves.

pub mod client;
pub mod config;
pub mod error;
pub mod pool;
pub mod transfer;

pub use aex_core::{DType, Index, QualitySpec};
pub use client::{Client, DatasetInfo, FileHandle, Item, SessionInfo, PROTOCOL_VERSION};
pub use config::ClientConfig;
pub use error::{ClientError, Result};
pub use transfer::{ArrayData, Element, Plan, TransferResult, TypedArray};
