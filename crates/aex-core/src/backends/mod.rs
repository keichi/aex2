//! Backend implementations.
//!
//! `.npy` and, with the `hdf5` feature, HDF5 are the real formats. [`null`] is
//! not a format at all: it is a dataset with no storage behind it, for
//! measuring what the transfer costs when reading the data costs nothing.
//!
//! The traits they implement live in [`crate::backend`].

#[cfg(feature = "hdf5")]
pub mod decode_cache;
#[cfg(feature = "hdf5")]
pub mod hdf5;
pub mod npy;
pub mod null;
