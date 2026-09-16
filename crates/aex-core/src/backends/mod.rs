//! Backend implementations.
//!
//! `.npy` is the one real format so far. [`null`] is not a format at all: it is
//! a dataset with no storage behind it, for measuring what the transfer costs
//! when reading the data costs nothing.
//!
//! The traits they implement live in [`crate::backend`].

pub mod npy;
pub mod null;
