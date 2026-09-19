//! Backend implementations.
//!
//! `.npy` and, with the `hdf5` feature, HDF5 are the real formats. [`null`] is
//! not a format at all: it is a dataset with no storage behind it, for
//! measuring what the transfer costs when reading the data costs nothing.
//!
//! The traits they implement live in [`crate::backend`].

/// Ask the kernel to start reading `[at, at + len)` of `file`.
///
/// Advisory in both directions: the kernel may ignore it, and a failure only
/// means the pages are fetched on demand, which is what would have happened.
#[cfg(target_os = "linux")]
pub(crate) fn will_need(file: &std::fs::File, at: u64, len: u64) {
    use std::os::fd::AsRawFd;
    // SAFETY: the fd is open for the duration, and the call only advises.
    unsafe {
        libc::posix_fadvise(
            file.as_raw_fd(),
            at as libc::off_t,
            len as libc::off_t,
            libc::POSIX_FADV_WILLNEED,
        );
    }
}

#[cfg(not(target_os = "linux"))]
pub(crate) fn will_need(_file: &std::fs::File, _at: u64, _len: u64) {}

#[cfg(feature = "hdf5")]
pub mod decode_cache;
#[cfg(feature = "hdf5")]
pub mod hdf5;
pub mod npy;
pub mod null;
