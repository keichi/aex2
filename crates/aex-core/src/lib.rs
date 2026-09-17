//! Core layer of AEX2.
//!
//! Holds the server-side logic — backends, selection resolution, reductions —
//! independent of tokio and tonic, so its unit tests stay fast.
//!
//! The central concept is the **logical byte stream**: the result of a
//! selection, flattened in C order. [`selection`] resolves a selection into
//! one, [`backend`] reads ranges of it, and every offset on the wire is a
//! position in it.

pub mod backend;
pub mod backends;
pub mod dtype;
pub mod error;
pub mod quality;
pub mod selection;

pub use backend::{ArrayDataset, ArrayFile, Item};
#[cfg(feature = "hdf5")]
pub use backends::hdf5::{Hdf5Dataset, Hdf5File};
pub use backends::npy::{NpyDataset, NpyFile};
pub use backends::null::{NullDataset, NullFile};
pub use dtype::{DType, ALL_DTYPES};
pub use error::{AexError, ErrorClass, Result};
pub use quality::{Codec, Encoding, QualitySpec};
pub use selection::{AxisSel, Index, LayoutKind, SelectionLayout};
