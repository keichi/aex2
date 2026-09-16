//! The AEX2 Rust client.
//!
//! Usable on its own, and the layer the Python bindings will sit on in M3.
//! Keeping it separate from the bindings also lets a benchmark measure the
//! transfer without Python in the picture.
//!
//! As of M1 it speaks the control plane: sessions, files and metadata. Reading
//! data needs the data plane, which arrives in M2.

pub mod client;
pub mod config;
pub mod error;

pub use client::{Client, DatasetInfo, FileHandle, Item, SessionInfo, PROTOCOL_VERSION};
pub use config::ClientConfig;
pub use error::{ClientError, Result};
