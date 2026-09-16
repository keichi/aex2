//! The `.npy` backend.
//!
//! Header parsing goes through [`npyz`]; the data is read with `pread`. An npy
//! header is a restricted Python dict literal, and hand-parsing it tends to
//! produce files numpy can write but we cannot read.
//!
//! We never `mmap`: it would turn a truncation mid-transfer into SIGBUS, and
//! page faults cannot keep the I/O queue deep on a cold cache.
//!
//! A `.npy` holds one array, which [`NpyFile`] presents as the hierarchy v1
//! used: a root group with a single dataset named `array`.

use std::fs::File;
use std::io::Seek;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::Arc;

use crate::backend::{normalize_path, ArrayDataset, ArrayFile, Item};
use crate::dtype::DType;
use crate::error::{AexError, Result};

/// Name of the one dataset in a `.npy`, as v1 exposed it.
pub const DATASET_NAME: &str = "array";

/// A `.npy` file opened for reading.
///
/// The data is `data_len` bytes of the array flattened in C order: the simplest
/// form of a logical byte stream, which is the coordinate system
/// [`NpyDataset::read_range`] takes its offsets in.
///
/// `read_range` needs only `&self` because `pread` carries no shared file
/// position, so many connection threads can read one file at once.
#[derive(Debug)]
pub struct NpyDataset {
    /// Read with `pread`, never mapped.
    file: File,
    /// First byte after the header; the base for every `pread`.
    data_offset: u64,
    /// File length as of open.
    file_len: u64,
    /// Product of the shape, computed without overflow at open.
    num_elements: u64,
    /// Length of the logical byte stream: elements times item size.
    data_len: u64,
    dtype: DType,
    shape: Vec<u64>,
}

impl NpyDataset {
    /// Open a `.npy`, parse its header and check it against the file.
    ///
    /// Anything AEX cannot serve — fortran order, big-endian, structured dtypes,
    /// object arrays — is rejected here. The declared shape and dtype are
    /// checked against the file length so a corrupt file cannot lead to a
    /// `pread` past the end.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let mut file = File::open(path)?;

        // from_reader read_exacts just the header and never reads ahead, so the
        // position right afterwards is the start of the data.
        let header = npyz::NpyHeader::from_reader(&mut file)?;
        let data_offset = file.stream_position()?;

        if header.order() != npyz::Order::C {
            return Err(AexError::UnsupportedNpy(
                "fortran_order arrays are not supported; fragment computation assumes C order"
                    .to_string(),
            ));
        }
        if header.uses_pickled_array() {
            return Err(AexError::UnsupportedDType(
                "object arrays (pickled) are not supported".to_string(),
            ));
        }

        let dtype = match header.dtype() {
            npyz::DType::Plain(ts) => DType::from_type_str(&ts)?,
            other => {
                return Err(AexError::UnsupportedDType(format!(
                    "structured dtype {} is not supported",
                    other.descr()
                )))
            }
        };

        let shape = header.shape().to_vec();
        let num_elements = checked_num_elements(&shape)?;
        let data_len = num_elements.checked_mul(dtype.itemsize()).ok_or_else(|| {
            AexError::MalformedNpy(format!(
                "shape {shape:?} of {dtype} overflows the addressable byte range"
            ))
        })?;

        let file_len = file.metadata()?.len();
        let declared_end = data_offset.checked_add(data_len).ok_or_else(|| {
            AexError::MalformedNpy(format!(
                "header declares {data_len} bytes of data past offset {data_offset}, which overflows"
            ))
        })?;
        if file_len < declared_end {
            return Err(AexError::MalformedNpy(format!(
                "header declares shape {shape:?} of {dtype} ({data_len} bytes after a \
                 {data_offset} byte header) but the file is only {file_len} bytes"
            )));
        }

        Ok(NpyDataset {
            file,
            data_offset,
            file_len,
            num_elements,
            data_len,
            dtype,
            shape,
        })
    }

    /// The element type.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Length of each axis.
    pub fn shape(&self) -> &[u64] {
        &self.shape
    }

    /// Number of dimensions; 0 for a scalar array.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Product of the shape; 1 for a scalar array.
    pub fn num_elements(&self) -> u64 {
        self.num_elements
    }

    /// Length of the logical byte stream.
    pub fn data_len(&self) -> u64 {
        self.data_len
    }

    /// Offset in the file where the data begins.
    pub fn data_offset(&self) -> u64 {
        self.data_offset
    }

    /// File length as of open.
    pub fn file_len(&self) -> u64 {
        self.file_len
    }

    /// Read `[offset, offset + dst.len())` of the logical byte stream into `dst`.
    ///
    /// The caller owns the buffer by design: the data plane allocates one buffer
    /// per connection and reuses it, so a transfer allocates nothing.
    pub fn read_range(&self, offset: u64, dst: &mut [u8]) -> Result<()> {
        let len = dst.len() as u64;
        let out_of_range = || AexError::OutOfRange {
            offset,
            len,
            total: self.data_len,
        };
        let end = offset.checked_add(len).ok_or_else(out_of_range)?;
        if end > self.data_len {
            return Err(out_of_range());
        }

        // pread may return short, so loop until the buffer is full.
        let mut filled: usize = 0;
        while filled < dst.len() {
            let at = self.data_offset + offset + filled as u64;
            let n = self.file.read_at(&mut dst[filled..], at)?;
            if n == 0 {
                return Err(AexError::MalformedNpy(format!(
                    "unexpected end of file at byte {at}; the file shrank after it was opened"
                )));
            }
            filled += n;
        }
        Ok(())
    }
}

impl ArrayDataset for NpyDataset {
    fn dtype(&self) -> DType {
        self.dtype
    }

    fn shape(&self) -> &[u64] {
        &self.shape
    }
}

/// A `.npy` presented as a hierarchy: a root group holding one dataset.
///
/// The dataset is behind an `Arc` because a transfer plan outlives the request
/// that produced it and holds the dataset it reads from.
#[derive(Debug, Clone)]
pub struct NpyFile {
    dataset: Arc<NpyDataset>,
}

impl NpyFile {
    /// Open a `.npy`. Rejects anything AEX cannot serve; see [`NpyDataset::open`].
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Ok(NpyFile {
            dataset: Arc::new(NpyDataset::open(path)?),
        })
    }

    /// The single array in this file.
    pub fn dataset(&self) -> &Arc<NpyDataset> {
        &self.dataset
    }
}

impl ArrayFile for NpyFile {
    fn contains(&self, path: &str) -> bool {
        matches!(normalize_path(path), "" | DATASET_NAME)
    }

    fn get_item(&self, path: &str) -> Result<Item> {
        match normalize_path(path) {
            "" => Ok(Item::Group),
            DATASET_NAME => Ok(Item::Dataset(self.dataset.clone())),
            other => Err(AexError::NotFound(format!(
                "{other:?}: a .npy holds a single array, reachable as {DATASET_NAME:?}"
            ))),
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

/// Product of the shape, without overflowing.
///
/// An npy header can declare any shape, and something like `(2^40, 2^40)`
/// overflows u64. Multiplying unchecked would make every later bounds check
/// meaningless.
fn checked_num_elements(shape: &[u64]) -> Result<u64> {
    shape.iter().try_fold(1u64, |acc, &n| {
        acc.checked_mul(n).ok_or_else(|| {
            AexError::MalformedNpy(format!(
                "shape {shape:?} has more elements than u64 can count"
            ))
        })
    })
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use tempfile::TempDir;

    use super::*;

    /// Builds `.npy` bytes for tests.
    ///
    /// Building fixtures here rather than committing numpy-generated files lets
    /// us produce what numpy never writes — fortran order, big-endian,
    /// structured dtypes, truncation — which is what the rejection paths need.
    /// Agreement with real numpy output is covered by the differential tests.
    struct NpyBuilder {
        /// Python literal placed into the header dict verbatim: usually a quoted
        /// type string like `'<f4'`, but a structured dtype list also works.
        descr_literal: String,
        shape: Vec<u64>,
        fortran_order: bool,
        version: (u8, u8),
        payload: Vec<u8>,
    }

    impl NpyBuilder {
        fn new(descr: &str, shape: &[u64]) -> Self {
            NpyBuilder {
                descr_literal: format!("'{descr}'"),
                shape: shape.to_vec(),
                fortran_order: false,
                version: (1, 0),
                payload: Vec::new(),
            }
        }

        /// Replace the whole `descr` literal, for structured dtype tests.
        fn descr_literal(mut self, literal: &str) -> Self {
            self.descr_literal = literal.to_string();
            self
        }

        fn fortran_order(mut self) -> Self {
            self.fortran_order = true;
            self
        }

        fn version(mut self, major: u8, minor: u8) -> Self {
            self.version = (major, minor);
            self
        }

        fn payload(mut self, payload: Vec<u8>) -> Self {
            self.payload = payload;
            self
        }

        /// Fill with deterministic, position-dependent bytes matching the shape.
        fn filled_payload(self, itemsize: u64) -> Self {
            let n = self.shape.iter().product::<u64>() * itemsize;
            let payload = (0..n).map(|i| (i % 251) as u8).collect();
            self.payload(payload)
        }

        fn build(&self) -> Vec<u8> {
            // Match numpy's formatting: `()` for 0-d, `(3,)` for 1-d.
            let shape_text = match self.shape.as_slice() {
                [] => "()".to_string(),
                [n] => format!("({n},)"),
                dims => {
                    let inner: Vec<String> = dims.iter().map(|n| n.to_string()).collect();
                    format!("({})", inner.join(", "))
                }
            };
            let dict = format!(
                "{{'descr': {}, 'fortran_order': {}, 'shape': {}, }}",
                self.descr_literal,
                if self.fortran_order { "True" } else { "False" },
                shape_text,
            );

            let len_field_size = if self.version.0 == 1 { 2 } else { 4 };
            let prefix_len = 6 + 2 + len_field_size;
            // Pad with spaces to a 64-byte boundary and end with a newline.
            let unpadded = prefix_len + dict.len() + 1;
            let padding = (64 - unpadded % 64) % 64;
            let header_text_len = dict.len() + padding + 1;

            let mut out = Vec::new();
            out.extend_from_slice(b"\x93NUMPY");
            out.push(self.version.0);
            out.push(self.version.1);
            if self.version.0 == 1 {
                out.extend_from_slice(&(header_text_len as u16).to_le_bytes());
            } else {
                out.extend_from_slice(&(header_text_len as u32).to_le_bytes());
            }
            out.extend_from_slice(dict.as_bytes());
            out.extend(std::iter::repeat_n(b' ', padding));
            out.push(b'\n');
            debug_assert_eq!(out.len() % 64, 0);
            out.extend_from_slice(&self.payload);
            out
        }
    }

    /// Write bytes to a temporary file, returning the dir so it outlives the test.
    fn write_npy(bytes: &[u8]) -> (TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("test.npy");
        let mut f = File::create(&path).expect("create");
        f.write_all(bytes).expect("write");
        f.sync_all().expect("sync");
        (dir, path)
    }

    fn open_bytes(bytes: &[u8]) -> Result<NpyDataset> {
        let (_dir, path) = write_npy(bytes);
        NpyDataset::open(path)
    }

    #[test]
    fn reads_shape_and_dtype_of_a_2d_float32_array() {
        let bytes = NpyBuilder::new("<f4", &[1000, 200])
            .filled_payload(4)
            .build();
        let npy = open_bytes(&bytes).expect("open");

        assert_eq!(npy.dtype(), DType::Float32);
        assert_eq!(npy.shape(), &[1000, 200]);
        assert_eq!(npy.ndim(), 2);
        assert_eq!(npy.num_elements(), 200_000);
        assert_eq!(npy.data_len(), 800_000);
        // A v1.0 header ends on a 64-byte boundary.
        assert_eq!(npy.data_offset() % 64, 0);
        assert_eq!(npy.file_len(), npy.data_offset() + npy.data_len());
    }

    #[test]
    fn reads_metadata_for_every_supported_dtype() {
        for dtype in crate::dtype::ALL_DTYPES {
            let bytes = NpyBuilder::new(dtype.descr(), &[7])
                .filled_payload(dtype.itemsize())
                .build();
            let npy = open_bytes(&bytes).unwrap_or_else(|e| panic!("{}: {e}", dtype.descr()));
            assert_eq!(npy.dtype(), dtype);
            assert_eq!(npy.shape(), &[7]);
            assert_eq!(npy.data_len(), 7 * dtype.itemsize());
        }
    }

    #[test]
    fn reads_a_zero_dimensional_array() {
        let bytes = NpyBuilder::new("<f8", &[]).filled_payload(8).build();
        let npy = open_bytes(&bytes).expect("open");
        assert_eq!(npy.shape(), &[] as &[u64]);
        assert_eq!(npy.ndim(), 0);
        assert_eq!(npy.num_elements(), 1);
        assert_eq!(npy.data_len(), 8);
    }

    #[test]
    fn reads_an_empty_array() {
        let bytes = NpyBuilder::new("<i4", &[0, 5]).filled_payload(4).build();
        let npy = open_bytes(&bytes).expect("open");
        assert_eq!(npy.shape(), &[0, 5]);
        assert_eq!(npy.num_elements(), 0);
        assert_eq!(npy.data_len(), 0);
        // Reading an empty range succeeds.
        npy.read_range(0, &mut []).expect("empty read");
    }

    #[test]
    fn parses_a_version_2_header() {
        let bytes = NpyBuilder::new("<u2", &[4, 4])
            .version(2, 0)
            .filled_payload(2)
            .build();
        let npy = open_bytes(&bytes).expect("open");
        assert_eq!(npy.dtype(), DType::Uint16);
        assert_eq!(npy.shape(), &[4, 4]);
        // v2.0 has a 4-byte length field, so data_offset may differ from v1.0.
        assert_eq!(npy.data_offset() % 64, 0);
        assert_eq!(npy.data_len(), 32);
    }

    #[test]
    fn read_range_returns_the_exact_bytes_at_every_offset() {
        let builder = NpyBuilder::new("<f4", &[16, 8]).filled_payload(4);
        let expected = builder.payload.clone();
        let npy = open_bytes(&builder.build()).expect("open");
        assert_eq!(npy.data_len(), expected.len() as u64);

        // The whole stream.
        let mut whole = vec![0u8; expected.len()];
        npy.read_range(0, &mut whole).expect("read whole");
        assert_eq!(whole, expected);

        // Sweep partial ranges. Parallel chunk receive depends on a logical
        // offset being usable directly as a position in the output buffer.
        for &(offset, len) in &[(0u64, 1usize), (1, 3), (7, 32), (100, 412), (511, 1)] {
            let mut buf = vec![0u8; len];
            npy.read_range(offset, &mut buf).expect("read part");
            assert_eq!(
                buf,
                &expected[offset as usize..offset as usize + len],
                "mismatch at offset {offset} len {len}"
            );
        }

        // Up to the exact end.
        let tail_len = 13;
        let tail_off = npy.data_len() - tail_len as u64;
        let mut tail = vec![0u8; tail_len];
        npy.read_range(tail_off, &mut tail).expect("read tail");
        assert_eq!(tail, &expected[tail_off as usize..]);
    }

    #[test]
    fn read_range_rejects_ranges_past_the_end() {
        let bytes = NpyBuilder::new("<i8", &[10]).filled_payload(8).build();
        let npy = open_bytes(&bytes).expect("open");
        assert_eq!(npy.data_len(), 80);

        let mut buf = vec![0u8; 8];
        // One byte past the end.
        let err = npy.read_range(73, &mut buf).unwrap_err();
        assert!(matches!(err, AexError::OutOfRange { .. }), "{err}");
        // Entirely out of range.
        assert!(npy.read_range(80, &mut buf).is_err());
        // Offset plus length overflows u64.
        assert!(npy.read_range(u64::MAX, &mut buf).is_err());
        // Exactly at the boundary is fine.
        npy.read_range(72, &mut buf).expect("exact tail");
    }

    #[test]
    fn rejects_fortran_order() {
        let bytes = NpyBuilder::new("<f4", &[4, 4])
            .fortran_order()
            .filled_payload(4)
            .build();
        let err = open_bytes(&bytes).unwrap_err();
        assert!(matches!(err, AexError::UnsupportedNpy(_)), "{err}");
    }

    #[test]
    fn rejects_big_endian_arrays() {
        let bytes = NpyBuilder::new(">f8", &[4]).filled_payload(8).build();
        let err = open_bytes(&bytes).unwrap_err();
        assert!(matches!(err, AexError::UnsupportedDType(_)), "{err}");
    }

    #[test]
    fn rejects_structured_dtypes() {
        let bytes = NpyBuilder::new("<f4", &[4])
            .descr_literal("[('a', '<i4'), ('b', '<f8')]")
            .payload(vec![0u8; 48])
            .build();
        let err = open_bytes(&bytes).unwrap_err();
        assert!(matches!(err, AexError::UnsupportedDType(_)), "{err}");
    }

    #[test]
    fn rejects_object_arrays() {
        let bytes = NpyBuilder::new("|O", &[4]).payload(vec![0u8; 64]).build();
        let err = open_bytes(&bytes).unwrap_err();
        assert!(matches!(err, AexError::UnsupportedDType(_)), "{err}");
    }

    #[test]
    fn rejects_a_truncated_file() {
        // The header declares 400 bytes of data but only 100 follow.
        let bytes = NpyBuilder::new("<f4", &[100])
            .payload(vec![0u8; 100])
            .build();
        let err = open_bytes(&bytes).unwrap_err();
        assert!(matches!(err, AexError::MalformedNpy(_)), "{err}");
    }

    #[test]
    fn element_count_overflow_is_detected() {
        // This header is not put through open() because npyz 0.9.1 computes the
        // shape product with an unchecked product() and panics in debug builds
        // before reaching our code. In release its value merely wraps, and
        // checked_num_elements recomputes from the shape and rejects the file.
        let huge = 1u64 << 40;
        let err = checked_num_elements(&[huge, huge, huge]).unwrap_err();
        assert!(matches!(err, AexError::MalformedNpy(_)), "{err}");

        assert_eq!(checked_num_elements(&[]).unwrap(), 1);
        assert_eq!(checked_num_elements(&[1000, 200]).unwrap(), 200_000);
        assert_eq!(checked_num_elements(&[0, 5]).unwrap(), 0);
    }

    #[test]
    fn rejects_a_shape_whose_byte_length_overflows() {
        // The element count fits in u64, but times 16 bytes it does not.
        let bytes = NpyBuilder::new("<c16", &[u64::MAX / 8]).build();
        let err = open_bytes(&bytes).unwrap_err();
        assert!(matches!(err, AexError::MalformedNpy(_)), "{err}");
    }

    #[test]
    fn rejects_a_file_that_is_not_npy() {
        let err = open_bytes(b"not an npy file at all").unwrap_err();
        assert!(matches!(err, AexError::Io(_)), "{err}");
        // Retrying will not help, so this must not be classified Transient.
        assert_eq!(err.class(), crate::error::ErrorClass::Permanent);
    }

    #[test]
    fn open_reports_a_missing_file_as_io_error() {
        let err = NpyDataset::open("/nonexistent/aex2/does-not-exist.npy").unwrap_err();
        assert!(matches!(err, AexError::Io(_)), "{err}");
        assert_eq!(err.class(), crate::error::ErrorClass::Request);
    }

    #[test]
    fn the_hierarchy_holds_one_dataset_named_array() {
        let bytes = NpyBuilder::new("<f4", &[3, 4]).filled_payload(4).build();
        let (_dir, path) = write_npy(&bytes);
        let file = NpyFile::open(&path).expect("open");

        // v1 exposed the array under both spellings of the path.
        for name in [DATASET_NAME, "/array", "/array/"] {
            assert!(file.contains(name), "{name} must exist");
            let Item::Dataset(dataset) = file.get_item(name).expect(name) else {
                panic!("{name} must be a dataset");
            };
            assert_eq!(dataset.dtype(), DType::Float32);
            assert_eq!(dataset.shape(), &[3, 4]);
            assert_eq!(dataset.ndim(), 2);
        }

        for root in ["", "/"] {
            assert!(file.contains(root));
            assert!(matches!(file.get_item(root), Ok(Item::Group)));
            let children = file.list_children(root).expect("list root");
            assert_eq!(children.len(), 1);
            assert_eq!(children[0].0, DATASET_NAME);
            assert!(matches!(children[0].1, Item::Dataset(_)));
        }
    }

    #[test]
    fn the_hierarchy_rejects_other_paths() {
        let bytes = NpyBuilder::new("<i2", &[2]).filled_payload(2).build();
        let (_dir, path) = write_npy(&bytes);
        let file = NpyFile::open(&path).expect("open");

        assert!(!file.contains("data"));
        let err = file.get_item("data").unwrap_err();
        assert!(matches!(err, AexError::NotFound(_)), "{err}");
        // The message has to say where the array actually is: "data" is what
        // every other backend would have called it.
        assert!(err.to_string().contains(DATASET_NAME), "{err}");
        assert!(file.list_children("data").is_err());

        // A dataset has no children.
        let err = file.list_children(DATASET_NAME).unwrap_err();
        assert!(matches!(err, AexError::NotAGroup(_)), "{err}");
    }

    #[test]
    fn one_open_file_is_shared_by_every_request() {
        // A transfer plan outlives the request that made it, so the dataset it
        // reads from has to be clonable out of the file.
        let bytes = NpyBuilder::new("<f8", &[8]).filled_payload(8).build();
        let (_dir, path) = write_npy(&bytes);
        let file = NpyFile::open(&path).expect("open");

        let Item::Dataset(first) = file.get_item(DATASET_NAME).unwrap() else {
            panic!("expected a dataset");
        };
        let Item::Dataset(second) = file.get_item(DATASET_NAME).unwrap() else {
            panic!("expected a dataset");
        };
        assert!(
            Arc::ptr_eq(&first, &second),
            "each request must see one file"
        );
    }

    #[test]
    fn concurrent_reads_from_one_file_agree() {
        // Many connection threads read one dataset at once, which works because
        // pread carries no shared file position and read_range takes &self.
        let builder = NpyBuilder::new("<f4", &[256, 64]).filled_payload(4);
        let expected = builder.payload.clone();
        let (_dir, path) = write_npy(&builder.build());
        let npy = NpyDataset::open(path).expect("open");

        std::thread::scope(|scope| {
            for t in 0..8u64 {
                let npy = &npy;
                let expected = &expected;
                scope.spawn(move || {
                    let chunk = npy.data_len() / 8;
                    let offset = t * chunk;
                    for _ in 0..50 {
                        let mut buf = vec![0u8; chunk as usize];
                        npy.read_range(offset, &mut buf).expect("read");
                        assert_eq!(
                            buf,
                            &expected[offset as usize..(offset + chunk) as usize],
                            "thread {t} read the wrong bytes"
                        );
                    }
                });
            }
        });
    }
}
