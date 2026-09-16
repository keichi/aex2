//! A dataset with nothing behind it.
//!
//! Every measurement so far has had storage in the path: a file, a page cache,
//! a `pread`. This one has none. It answers a read from a small repeating
//! pattern that stays in cache, so what is left to measure is the transfer
//! itself — the framing, the socket, and the copies either side of it.
//!
//! That makes it directly comparable to what `iperf3` does, which also sends
//! the same small buffer over and over. The difference between this and the
//! `.npy` backend is what reading the data costs; the difference between this
//! and `iperf3` is what the protocol costs.
//!
//! The pattern's length is part of the specification, which makes it an
//! instrument for a second question: a pattern that fits in cache costs almost
//! nothing to read, and one much larger than cache has to be fetched from
//! memory like any real data. The difference between the two is what reading
//! costs, with no filesystem anywhere near it.
//!
//! It serves data that was never stored anywhere, so a server only offers it
//! when explicitly told to. It is a measuring instrument, not a format.

use std::sync::Arc;

use crate::backend::{normalize_path, ArrayDataset, ArrayFile, Item};
use crate::dtype::DType;
use crate::error::{AexError, Result};
use crate::selection::{LayoutKind, SelectionLayout};

/// Name of the one dataset, matching what the other backends call theirs.
pub const DATASET_NAME: &str = "array";

/// Length of the repeating pattern unless the specification says otherwise.
///
/// Small enough to sit in cache, so that reading it costs as close to nothing
/// as a real read can.
const DEFAULT_PERIOD: u64 = 64 * 1024;

/// The pattern repeats on a multiple of 256 whatever its length, so that the
/// byte at any position in the logical stream is the low byte of that position
/// and a caller can check what arrived without holding a copy of it.
const PERIOD_MULTIPLE: u64 = 256;

/// A dataset that answers from a pattern rather than from storage.
#[derive(Debug)]
pub struct NullDataset {
    dtype: DType,
    shape: Vec<u64>,
    data_len: u64,
    /// `pattern[i] == i as u8`.
    pattern: Vec<u8>,
}

impl NullDataset {
    /// Build one from a specification: `<dtype>:<dim>[x<dim>...][@<period>]`.
    ///
    /// For instance `float32:1000x200`, or `uint8:4294967296` for a stream of a
    /// given number of bytes. `@<period>` sets the length of the pattern the
    /// reads come from: leave it out and they come from cache, set it far above
    /// the cache and they come from memory like any real data would.
    pub fn from_spec(spec: &str) -> Result<Self> {
        let bad = |what: &str| {
            AexError::BadSelection(format!(
                "{spec:?} is not a synthetic dataset: {what}. Write it as \
                 <dtype>:<dim>[x<dim>...][@<period>], such as float32:1000x200"
            ))
        };

        let (spec, period) = match spec.split_once('@') {
            Some((spec, period)) => (
                spec,
                period
                    .parse::<u64>()
                    .map_err(|_| bad("the period is not a number"))?,
            ),
            None => (spec, DEFAULT_PERIOD),
        };
        if period == 0 || period % PERIOD_MULTIPLE != 0 {
            return Err(bad("the period must be a positive multiple of 256"));
        }

        let (dtype, shape) = spec.split_once(':').ok_or_else(|| bad("no ':'"))?;
        let dtype = dtype_from_name(dtype).ok_or_else(|| bad("unknown dtype"))?;
        let shape = shape
            .split('x')
            .map(|dim| {
                dim.parse::<u64>()
                    .map_err(|_| bad("a dimension is not a number"))
            })
            .collect::<Result<Vec<u64>>>()?;

        Self::with_period(dtype, shape, period)
    }

    pub fn new(dtype: DType, shape: Vec<u64>) -> Result<Self> {
        Self::with_period(dtype, shape, DEFAULT_PERIOD)
    }

    /// Build one whose reads come from a pattern of the given length.
    pub fn with_period(dtype: DType, shape: Vec<u64>, period: u64) -> Result<Self> {
        let elements = shape.iter().try_fold(1u64, |acc, &n| acc.checked_mul(n));
        let data_len = elements
            .and_then(|elements| elements.checked_mul(dtype.itemsize()))
            .ok_or_else(|| {
                AexError::BadSelection(format!(
                    "a synthetic {dtype} dataset of {shape:?} is larger than the byte range"
                ))
            })?;

        let period = usize::try_from(period).map_err(|_| {
            AexError::BadSelection(format!(
                "a pattern of {period} bytes does not fit in memory"
            ))
        })?;

        Ok(NullDataset {
            dtype,
            shape,
            data_len,
            pattern: (0..period).map(|i| i as u8).collect(),
        })
    }

    /// Length of the logical byte stream.
    pub fn data_len(&self) -> u64 {
        self.data_len
    }

    /// Length of the pattern the reads come from.
    pub fn period(&self) -> usize {
        self.pattern.len()
    }

    /// Fill `dst` with what stands at `at` in the stream.
    fn fill(&self, at: u64, dst: &mut [u8]) {
        let period = self.pattern.len();
        let mut written = 0;
        while written < dst.len() {
            // Where in the pattern the next stretch begins.
            let start = ((at + written as u64) % period as u64) as usize;
            let run = (period - start).min(dst.len() - written);
            dst[written..written + run].copy_from_slice(&self.pattern[start..start + run]);
            written += run;
        }
    }
}

impl ArrayDataset for NullDataset {
    fn dtype(&self) -> DType {
        self.dtype
    }

    fn shape(&self) -> &[u64] {
        &self.shape
    }

    fn read_range(&self, layout: &SelectionLayout, offset: u64, dst: &mut [u8]) -> Result<()> {
        layout.check_range(offset, dst.len() as u64)?;
        match layout.kind {
            LayoutKind::Contiguous { src_offset, .. } => {
                self.fill(src_offset + offset, dst);
                Ok(())
            }
        }
    }
}

/// A synthetic dataset presented as a hierarchy, like a one-array file.
#[derive(Debug, Clone)]
pub struct NullFile {
    dataset: Arc<NullDataset>,
}

impl NullFile {
    /// Build one from a specification; see [`NullDataset::from_spec`].
    pub fn from_spec(spec: &str) -> Result<Self> {
        Ok(NullFile {
            dataset: Arc::new(NullDataset::from_spec(spec)?),
        })
    }

    pub fn dataset(&self) -> &Arc<NullDataset> {
        &self.dataset
    }
}

impl ArrayFile for NullFile {
    fn contains(&self, path: &str) -> bool {
        matches!(normalize_path(path), "" | DATASET_NAME)
    }

    fn get_item(&self, path: &str) -> Result<Item> {
        match normalize_path(path) {
            "" => Ok(Item::Group),
            DATASET_NAME => Ok(Item::Dataset(self.dataset.clone())),
            other => Err(AexError::NotFound(other.to_string())),
        }
    }

    fn list_children(&self, path: &str) -> Result<Vec<(String, Item)>> {
        match normalize_path(path) {
            "" => Ok(vec![(
                DATASET_NAME.to_string(),
                Item::Dataset(self.dataset.clone()),
            )]),
            DATASET_NAME => Err(AexError::NotAGroup(DATASET_NAME.to_string())),
            other => Err(AexError::NotFound(other.to_string())),
        }
    }
}

/// An element type by name.
///
/// Spelled out rather than taken as a numpy `descr`, where `i8` means eight
/// bytes and not eight bits — a trap not worth leaving in a command line.
fn dtype_from_name(name: &str) -> Option<DType> {
    let dtype = match name {
        "int8" => DType::Int8,
        "int16" => DType::Int16,
        "int32" => DType::Int32,
        "int64" => DType::Int64,
        "uint8" => DType::Uint8,
        "uint16" => DType::Uint16,
        "uint32" => DType::Uint32,
        "uint64" => DType::Uint64,
        "float16" => DType::Float16,
        "float32" => DType::Float32,
        "float64" => DType::Float64,
        "complex64" => DType::Complex64,
        "complex128" => DType::Complex128,
        "bool" => DType::Bool,
        _ => return None,
    };
    Some(dtype)
}

#[cfg(test)]
mod tests {
    use crate::quality::QualitySpec;
    use crate::selection::Index;

    use super::*;

    fn whole(dataset: &NullDataset) -> SelectionLayout {
        dataset
            .layout(&[], &QualitySpec::exact())
            .expect("the whole dataset")
    }

    #[test]
    fn a_spec_gives_a_dataset_of_that_shape_and_type() {
        let dataset = NullDataset::from_spec("float32:1000x200").expect("spec");
        assert_eq!(dataset.dtype(), DType::Float32);
        assert_eq!(dataset.shape(), &[1000, 200]);
        assert_eq!(dataset.data_len(), 1000 * 200 * 4);

        let dataset = NullDataset::from_spec("uint8:4096").expect("spec");
        assert_eq!(dataset.dtype(), DType::Uint8);
        assert_eq!(dataset.data_len(), 4096);
    }

    #[test]
    fn a_spec_that_makes_no_sense_says_how_to_write_one() {
        for spec in [
            "",
            "float32",
            "notatype:10",
            "float32:ten",
            "float32:10xten",
        ] {
            let err = NullDataset::from_spec(spec).unwrap_err();
            assert!(err.to_string().contains("<dtype>:<dim>"), "{spec:?}: {err}");
        }
        // i8 is eight bits here, and would have been eight bytes in a numpy
        // descr; the names avoid having to know which convention is in play.
        assert_eq!(
            NullDataset::from_spec("int8:4").unwrap().dtype(),
            DType::Int8
        );
        assert!(NullDataset::from_spec("i8:4").is_err());
    }

    #[test]
    fn a_shape_too_large_for_the_byte_range_is_refused() {
        assert!(NullDataset::new(DType::Complex128, vec![u64::MAX / 8]).is_err());
        assert!(NullDataset::new(DType::Uint8, vec![u64::MAX, 2]).is_err());
    }

    #[test]
    fn every_byte_says_where_it_came_from() {
        // The byte at position p is p as u8, whatever range is asked for and
        // wherever in the pattern that range happens to start. A caller can
        // check what arrived without holding a copy of the whole thing.
        let dataset = NullDataset::from_spec("uint8:1048576").expect("spec");
        let layout = whole(&dataset);

        for &(offset, len) in &[
            (0u64, 16usize),
            (1, 3),
            (DEFAULT_PERIOD - 8, 16),
            (DEFAULT_PERIOD, 8),
            (3 * DEFAULT_PERIOD + 17, 5000),
            (1048576 - 4, 4),
        ] {
            let mut got = vec![0u8; len];
            dataset.read_range(&layout, offset, &mut got).expect("read");
            let expected: Vec<u8> = (offset..offset + len as u64).map(|p| p as u8).collect();
            assert_eq!(got, expected, "at {offset} for {len}");
        }
    }

    #[test]
    fn the_pattern_length_can_be_set_and_does_not_change_what_arrives() {
        // A pattern far larger than cache makes a read cost what a real one
        // costs. What it produces has to stay the same, or the two are not
        // measuring the same transfer.
        let small = NullDataset::from_spec("uint8:1048576").expect("spec");
        let large = NullDataset::from_spec("uint8:1048576@16777216").expect("spec");
        assert_eq!(small.period(), 64 * 1024);
        assert_eq!(large.period(), 16 * 1024 * 1024);

        let layout = whole(&small);
        let mut from_small = vec![0u8; 100_000];
        let mut from_large = vec![0u8; 100_000];
        small
            .read_range(&layout, 777, &mut from_small)
            .expect("read");
        large
            .read_range(&layout, 777, &mut from_large)
            .expect("read");
        assert_eq!(from_small, from_large);

        // A period that would break `p as u8` is refused rather than quietly
        // serving something the caller cannot check.
        assert!(NullDataset::from_spec("uint8:1024@100").is_err());
        assert!(NullDataset::from_spec("uint8:1024@0").is_err());
        assert!(NullDataset::from_spec("uint8:1024@abc").is_err());
    }

    #[test]
    fn a_selection_reads_from_where_it_starts_in_the_source() {
        let dataset = NullDataset::from_spec("uint8:1000").expect("spec");
        let layout = dataset
            .layout(&[Index::range(100, 200)], &QualitySpec::exact())
            .expect("layout");
        assert_eq!(layout.total_bytes, 100);

        let mut got = vec![0u8; 100];
        dataset.read_range(&layout, 0, &mut got).expect("read");
        // The selection starts at element 100, so that is where the pattern
        // has to be sampled from, not at 0.
        let expected: Vec<u8> = (100u64..200).map(|p| p as u8).collect();
        assert_eq!(got, expected);
    }

    #[test]
    fn reads_past_the_end_are_refused() {
        let dataset = NullDataset::from_spec("uint8:64").expect("spec");
        let layout = whole(&dataset);
        dataset.read_range(&layout, 0, &mut [0u8; 64]).expect("all");
        assert!(dataset.read_range(&layout, 1, &mut [0u8; 64]).is_err());
        assert!(dataset
            .read_range(&layout, u64::MAX, &mut [0u8; 8])
            .is_err());
    }

    #[test]
    fn the_hierarchy_looks_like_any_other_backend() {
        let file = NullFile::from_spec("float32:8").expect("spec");
        assert!(file.contains("/"));
        assert!(file.contains("array"));
        assert!(!file.contains("other"));
        assert!(matches!(file.get_item("/"), Ok(Item::Group)));
        assert!(matches!(file.get_item("array"), Ok(Item::Dataset(_))));
        assert!(file.get_item("other").is_err());
        assert_eq!(file.list_children("/").expect("children").len(), 1);
        assert!(file.list_children("array").is_err());
    }
}
