//! What a backend has to provide.
//!
//! The traits are sized for four formats even though only `.npy` is
//! implemented: HDF5, netCDF4 and Zarr all present a hierarchy of groups and
//! datasets, and all of them can serve a range of a logical byte stream.
//!
//! A dataset answers two questions: what a selection resolves to, and what the
//! bytes of a range of that selection are. Everything the data plane needs is
//! one of those two, which is what keeps a new backend a small piece of work.

use std::sync::Arc;

use crate::dtype::DType;
use crate::error::Result;
use crate::quality::QualitySpec;
use crate::selection::{Index, SelectionLayout};

/// An open file, seen as a hierarchy.
///
/// Every method takes `&self`: one file is shared by all the connections of a
/// session, and by every connection thread of the data plane.
pub trait ArrayFile: Send + Sync {
    /// Whether anything lives at `path`.
    fn contains(&self, path: &str) -> bool;

    /// The item at `path`.
    fn get_item(&self, path: &str) -> Result<Item>;

    /// The children of the group at `path`, in a stable order.
    fn list_children(&self, path: &str) -> Result<Vec<(String, Item)>>;
}

/// What lives at a path in a file.
#[derive(Clone)]
pub enum Item {
    Dataset(Arc<dyn ArrayDataset>),
    /// A container. It carries no data of its own, which is why this is a unit
    /// variant: nothing yet distinguishes one group from another. A format with
    /// group attributes would turn it into a trait object like `Dataset`.
    Group,
}

impl std::fmt::Debug for Item {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Item::Dataset(d) => f
                .debug_struct("Dataset")
                .field("dtype", &d.dtype())
                .field("shape", &d.shape())
                .finish(),
            Item::Group => f.write_str("Group"),
        }
    }
}

/// One array within a file.
pub trait ArrayDataset: Send + Sync {
    fn dtype(&self) -> DType;

    /// Length of each axis; empty for a scalar array.
    fn shape(&self) -> &[u64];

    /// Number of dimensions.
    fn ndim(&self) -> usize {
        self.shape().len()
    }

    /// Resolve a selection into a logical byte stream.
    ///
    /// The default is right for any backend whose logical byte stream is the
    /// array flattened in C order; only `read_range` has to know how the bytes
    /// are actually stored.
    fn layout(&self, indices: &[Index], quality: &QualitySpec) -> Result<SelectionLayout> {
        SelectionLayout::resolve(self.shape(), self.dtype(), indices, quality)
    }

    /// Write `[offset, offset + dst.len())` of the layout's logical byte stream
    /// into `dst`.
    ///
    /// The caller owns the buffer: the data plane keeps one per connection and
    /// reuses it, so a transfer allocates nothing. Whether the backend serves
    /// this with a `pread`, a decompression or a remote read is its own
    /// business, and stays invisible to the send path.
    ///
    /// `&self` rather than `&mut self` because every connection thread reads
    /// the same dataset at once. A backend needing state does so with interior
    /// mutability.
    fn read_range(&self, layout: &SelectionLayout, offset: u64, dst: &mut [u8]) -> Result<()>;
}

/// Strip the leading and trailing slashes of a path.
///
/// Clients send `/group/data`, `group/data` and `group/data/` for the same
/// item, so every backend has to agree on how to reduce them to one form.
pub fn normalize_path(path: &str) -> &str {
    path.trim_matches('/')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_forms_reduce_to_one() {
        for path in ["a/b", "/a/b", "a/b/", "/a/b/", "//a/b//"] {
            assert_eq!(normalize_path(path), "a/b");
        }
        // Every spelling of the root becomes the empty path.
        for path in ["", "/", "//"] {
            assert_eq!(normalize_path(path), "");
        }
    }
}
