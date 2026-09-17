//! The HDF5 backend, which also serves netCDF-4 (HDF5 underneath).
//!
//! libhdf5 serializes every call behind one process-wide lock, so reading data
//! through it would put all data connections in single file. It is only asked
//! for metadata: where a dataset's bytes are, and what they mean. The bytes
//! themselves are `pread` from a descriptor of our own, as with `.npy`.
//!
//! Like the `.npy` backend, this assumes nobody writes to a file while it is
//! served: the offsets are taken once, when a dataset is opened.

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::{Arc, Once};

use hdf5::dataset::{FillValue, Layout};
use hdf5::datatype::ByteOrder;
use hdf5::types::{CompoundType, FloatSize, IntSize, TypeDescriptor};
use hdf5::LocationType;

use crate::backend::{normalize_path, ArrayDataset, ArrayFile, Item};
use crate::dtype::DType;
use crate::error::{AexError, Result};
use crate::selection::SelectionLayout;

/// An HDF5 file opened for reading.
pub struct Hdf5File {
    /// For metadata only; every call on it takes libhdf5's global lock.
    file: hdf5::File,
    /// For data. Shared by every dataset opened from this file.
    raw: Arc<File>,
}

impl Hdf5File {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        disable_external_links();
        // Opened first so a missing or unreadable file is reported as such,
        // not as a libhdf5 failure.
        let raw = Arc::new(File::open(path)?);
        let file = hdf5::File::open(path).map_err(|e| {
            AexError::UnsupportedHdf5(format!(
                "{} is not an HDF5 file (netCDF-4 is served, netCDF-3 is not): {e}",
                path.display()
            ))
        })?;
        Ok(Hdf5File { file, raw })
    }

    fn item(&self, path: &str) -> Result<Item> {
        if path.is_empty() {
            return Ok(Item::Group);
        }
        let info = self
            .file
            .loc_info_by_name(path)
            .map_err(|_| AexError::NotFound(path.to_string()))?;
        match info.loc_type {
            LocationType::Group => Ok(Item::Group),
            LocationType::Dataset => {
                let dataset = self.file.dataset(path)?;
                Ok(Item::Dataset(Arc::new(Hdf5Dataset::open(
                    &dataset,
                    self.raw.clone(),
                )?)))
            }
            other => Err(AexError::NotFound(format!(
                "{path:?} is a {other:?}, not a group or a dataset"
            ))),
        }
    }
}

impl std::fmt::Debug for Hdf5File {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Hdf5File")
            .field("file", &self.file.filename())
            .finish()
    }
}

impl ArrayFile for Hdf5File {
    fn contains(&self, path: &str) -> bool {
        let path = normalize_path(path);
        path.is_empty() || self.file.loc_info_by_name(path).is_ok()
    }

    fn get_item(&self, path: &str) -> Result<Item> {
        self.item(normalize_path(path))
    }

    fn list_children(&self, path: &str) -> Result<Vec<(String, Item)>> {
        let path = normalize_path(path);
        let group = match self.item(path)? {
            Item::Group if path.is_empty() => self.file.as_group()?,
            Item::Group => self.file.group(path)?,
            Item::Dataset(_) => return Err(AexError::NotAGroup(path.to_string())),
        };
        let mut names = group.member_names()?;
        // libhdf5 lists in whatever order the group's index keeps.
        names.sort();
        let mut children = Vec::new();
        for name in names {
            let child = if path.is_empty() {
                name.clone()
            } else {
                format!("{path}/{name}")
            };
            // A dataset we cannot serve, a dangling link or a named datatype
            // would otherwise hide every sibling that we can.
            if let Ok(item) = self.item(&child) {
                children.push((name, item));
            }
        }
        Ok(children)
    }
}

/// Where a dataset's bytes are.
#[derive(Debug, Clone, Copy)]
enum Storage {
    /// One run of the file starting here.
    Contiguous(u64),
    /// Never written: every element reads as the fill value.
    Unallocated,
}

/// One dataset of an HDF5 file.
#[derive(Debug)]
pub struct Hdf5Dataset {
    raw: Arc<File>,
    dtype: DType,
    shape: Vec<u64>,
    storage: Storage,
    /// One element's worth of bytes, as stored.
    fill: Vec<u8>,
}

impl Hdf5Dataset {
    fn open(dataset: &hdf5::Dataset, raw: Arc<File>) -> Result<Self> {
        let name = dataset.name();
        let dtype = dtype_of(&dataset.dtype()?)
            .map_err(|e| AexError::UnsupportedDType(format!("{name}: {e}")))?;
        if dataset.space()?.is_null() {
            return Err(AexError::UnsupportedHdf5(format!(
                "{name} has a null dataspace and holds no array"
            )));
        }
        let shape: Vec<u64> = dataset.shape().iter().map(|&n| n as u64).collect();
        let data_len = shape
            .iter()
            .try_fold(dtype.itemsize(), |acc, &n| acc.checked_mul(n))
            .ok_or_else(|| {
                AexError::MalformedHdf5(format!("{name}: shape {shape:?} overflows u64"))
            })?;

        let dcpl = dataset.dcpl()?;
        if !dcpl.external().is_empty() {
            return Err(AexError::UnsupportedHdf5(format!(
                "{name} keeps its data in external files"
            )));
        }
        let storage = match dcpl.layout() {
            Layout::Contiguous => match dataset.offset() {
                Some(offset) => Storage::Contiguous(offset),
                None => Storage::Unallocated,
            },
            other => {
                return Err(AexError::UnsupportedHdf5(format!(
                    "{name} has {other:?} layout; only contiguous datasets are served"
                )))
            }
        };

        if let Storage::Contiguous(offset) = storage {
            let file_len = raw.metadata()?.len();
            if offset
                .checked_add(data_len)
                .is_none_or(|end| end > file_len)
            {
                return Err(AexError::MalformedHdf5(format!(
                    "{name} claims {data_len} bytes at offset {offset}, \
                     but the file is only {file_len} bytes"
                )));
            }
        }

        let fill = fill_value(dataset, &dcpl, dtype)?;
        Ok(Hdf5Dataset {
            raw,
            dtype,
            shape,
            storage,
            fill,
        })
    }
}

impl ArrayDataset for Hdf5Dataset {
    fn dtype(&self) -> DType {
        self.dtype
    }

    fn shape(&self) -> &[u64] {
        &self.shape
    }

    fn read_range(&self, layout: &SelectionLayout, offset: u64, dst: &mut [u8]) -> Result<()> {
        match self.storage {
            Storage::Contiguous(base) => layout.read_with(offset, dst, |at, buf| {
                Ok(self.raw.read_exact_at(buf, base + at)?)
            }),
            Storage::Unallocated => layout.read_with(offset, dst, |at, buf| {
                fill_from(&self.fill, at, buf);
                Ok(())
            }),
        }
    }
}

/// Fill `dst` with the bytes at `[at, at + dst.len())` of an array made of
/// nothing but `pattern`.
fn fill_from(pattern: &[u8], at: u64, dst: &mut [u8]) {
    let n = pattern.len();
    let start = (at % n as u64) as usize;
    for (i, b) in dst.iter_mut().enumerate() {
        *b = pattern[(start + i) % n];
    }
}

/// The AEX type of an HDF5 datatype, or why there is none.
fn dtype_of(datatype: &hdf5::Datatype) -> std::result::Result<DType, String> {
    let desc = datatype.to_descriptor().map_err(|e| e.to_string())?;
    // A single byte has no order, and libhdf5 reports whatever it was created with.
    if desc.size() > 1 && datatype.byte_order() != ByteOrder::LittleEndian {
        return Err(format!(
            "{desc} is stored {:?}; only little-endian data is served",
            datatype.byte_order()
        ));
    }
    Ok(match desc {
        TypeDescriptor::Integer(size) => match size {
            IntSize::U1 => DType::Int8,
            IntSize::U2 => DType::Int16,
            IntSize::U4 => DType::Int32,
            IntSize::U8 => DType::Int64,
        },
        TypeDescriptor::Unsigned(size) => match size {
            IntSize::U1 => DType::Uint8,
            IntSize::U2 => DType::Uint16,
            IntSize::U4 => DType::Uint32,
            IntSize::U8 => DType::Uint64,
        },
        TypeDescriptor::Float(size) => match size {
            FloatSize::U2 => DType::Float16,
            FloatSize::U4 => DType::Float32,
            FloatSize::U8 => DType::Float64,
        },
        TypeDescriptor::Boolean => DType::Bool,
        TypeDescriptor::Compound(ref c) if complex_float(c) == Some(FloatSize::U4) => {
            DType::Complex64
        }
        TypeDescriptor::Compound(ref c) if complex_float(c) == Some(FloatSize::U8) => {
            DType::Complex128
        }
        other => return Err(format!("type {other} is not supported")),
    })
}

/// The part type of a compound laid out the way h5py stores a complex number.
fn complex_float(c: &CompoundType) -> Option<FloatSize> {
    let [r, i] = c.fields.as_slice() else {
        return None;
    };
    let TypeDescriptor::Float(size) = r.ty else {
        return None;
    };
    let half = c.size / 2;
    let laid_out = r.name == "r" && i.name == "i" && r.offset == 0 && i.offset == half;
    (laid_out && i.ty == r.ty && size as usize == half).then_some(size)
}

/// One element of the fill value, as stored.
fn fill_value(
    dataset: &hdf5::Dataset,
    dcpl: &hdf5::plist::DatasetCreate,
    dtype: DType,
) -> Result<Vec<u8>> {
    let mut fill = vec![0u8; dtype.itemsize() as usize];
    // Undefined is what libhdf5 reads back as zeros too.
    if dcpl.fill_value_defined() == FillValue::Undefined {
        return Ok(fill);
    }
    let datatype = dataset.dtype()?;
    // Asked for in the dataset's own type, so the bytes are what storage would hold.
    let status = hdf5::sync::sync(|| unsafe {
        hdf5_sys::h5p::H5Pget_fill_value(dcpl.id(), datatype.id(), fill.as_mut_ptr().cast())
    });
    if status < 0 {
        return Err(AexError::MalformedHdf5(format!(
            "{}: cannot read the fill value",
            dataset.name()
        )));
    }
    Ok(fill)
}

/// Make libhdf5 refuse to follow external links, process-wide.
///
/// An external link names any file on the host, so following one would serve
/// a file outside the data roots.
fn disable_external_links() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let status = hdf5::sync::sync(|| unsafe {
            hdf5_sys::h5l::H5Lunregister(hdf5_sys::h5l::H5L_type_t::H5L_TYPE_EXTERNAL)
        });
        // Serving with external links enabled is not an option.
        assert!(status >= 0, "cannot disable HDF5 external links");
    });
}

#[cfg(test)]
mod tests {
    use hdf5::types::VarLenUnicode;
    use num_complex::{Complex32, Complex64};

    use super::*;
    use crate::quality::QualitySpec;
    use crate::selection::{Index, LayoutKind};

    fn read_all(dataset: &dyn ArrayDataset, indices: &[Index]) -> (SelectionLayout, Vec<u8>) {
        let layout = dataset.layout(indices, &QualitySpec::default()).unwrap();
        let mut out = vec![0u8; layout.total_bytes as usize];
        dataset.read_range(&layout, 0, &mut out).unwrap();
        (layout, out)
    }

    fn dataset(file: &Hdf5File, path: &str) -> Arc<dyn ArrayDataset> {
        match file.get_item(path).unwrap() {
            Item::Dataset(d) => d,
            Item::Group => panic!("{path} is a group"),
        }
    }

    fn bytes_of<T: Copy>(values: &[T]) -> Vec<u8> {
        let len = std::mem::size_of_val(values);
        // SAFETY: plain numeric types, read as bytes.
        unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), len) }.to_vec()
    }

    /// Write `values` as a contiguous dataset of `shape` and return its bytes.
    fn write<T: hdf5::H5Type + Copy>(
        group: &hdf5::Group,
        name: &str,
        values: &[T],
        shape: &[usize],
    ) -> Vec<u8> {
        group
            .new_dataset::<T>()
            .shape(shape)
            .create(name)
            .unwrap()
            .write_raw(values)
            .unwrap();
        bytes_of(values)
    }

    #[test]
    fn every_dtype_reads_back_as_written() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.h5");
        let h5 = hdf5::File::create(&path).unwrap();
        let n: usize = 24;
        let mut expected = Vec::new();
        macro_rules! case {
            ($name:literal, $t:ty, $dtype:expr, $f:expr) => {{
                let values: Vec<$t> = (0..n).map($f).collect();
                expected.push(($name, $dtype, write(&h5, $name, &values, &[4, 6])));
            }};
        }
        case!("i1", i8, DType::Int8, |i| i as i8 - 12);
        case!("i2", i16, DType::Int16, |i| i as i16 * -300);
        case!("i4", i32, DType::Int32, |i| i as i32 * -70000);
        case!("i8", i64, DType::Int64, |i| i as i64 * -(1 << 40));
        case!("u1", u8, DType::Uint8, |i| i as u8);
        case!("u2", u16, DType::Uint16, |i| i as u16 * 300);
        case!("u4", u32, DType::Uint32, |i| i as u32 * 70000);
        case!("u8", u64, DType::Uint64, |i| i as u64 * (1 << 40));
        case!("f2", half::f16, DType::Float16, |i| half::f16::from_f32(
            i as f32 / 4.0
        ));
        case!("f4", f32, DType::Float32, |i| i as f32 / 3.0);
        case!("f8", f64, DType::Float64, |i| i as f64 / 7.0);
        case!("c8", Complex32, DType::Complex64, |i| Complex32::new(
            i as f32,
            -(i as f32)
        ));
        case!("c16", Complex64, DType::Complex128, |i| Complex64::new(
            i as f64 / 3.0,
            1.0
        ));
        case!("b1", bool, DType::Bool, |i| i % 3 == 0);
        drop(h5);

        let file = Hdf5File::open(&path).unwrap();
        for (name, dtype, bytes) in expected {
            let d = dataset(&file, name);
            assert_eq!(d.dtype(), dtype, "{name}");
            assert_eq!(d.shape(), &[4, 6], "{name}");
            let (layout, out) = read_all(&*d, &[]);
            assert!(matches!(layout.kind, LayoutKind::Contiguous { .. }));
            assert_eq!(out, bytes, "{name}");
        }
    }

    #[test]
    fn selections_and_odd_offsets_match_the_source() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.h5");
        let h5 = hdf5::File::create(&path).unwrap();
        let values: Vec<i32> = (0..10 * 7).collect();
        write(&h5, "a", &values, &[10, 7]);
        drop(h5);

        let file = Hdf5File::open(&path).unwrap();
        let d = dataset(&file, "a");

        // Row 3: one run of the file.
        let (layout, out) = read_all(&*d, &[Index::Single(3)]);
        assert!(matches!(layout.kind, LayoutKind::Contiguous { .. }));
        assert_eq!(out, bytes_of(&values[21..28]));

        // Every other row of column 2: gathered.
        let step = Index::Slice {
            start: None,
            stop: None,
            step: Some(2),
        };
        let (layout, out) = read_all(&*d, &[step, Index::Single(2)]);
        assert!(matches!(layout.kind, LayoutKind::Gathered(_)));
        let expected: Vec<i32> = (0..10).step_by(2).map(|r| values[r * 7 + 2]).collect();
        assert_eq!(out, bytes_of(&expected));

        // A range that starts and ends mid-element.
        let layout = d.layout(&[], &QualitySpec::default()).unwrap();
        let mut out = vec![0u8; 9];
        d.read_range(&layout, 5, &mut out).unwrap();
        assert_eq!(out, bytes_of(&values)[5..14]);
    }

    #[test]
    fn scalars_empty_arrays_and_unwritten_data() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.h5");
        let h5 = hdf5::File::create(&path).unwrap();
        h5.new_dataset::<f64>()
            .create("scalar")
            .unwrap()
            .write_scalar(&2.5f64)
            .unwrap();
        h5.new_dataset::<u16>()
            .shape([0, 5])
            .no_chunk()
            .create("empty")
            .unwrap();
        h5.new_dataset::<i32>()
            .shape([3, 2])
            .fill_value(-7i32)
            .create("filled")
            .unwrap();
        h5.new_dataset::<u8>().shape([4]).create("zeros").unwrap();
        drop(h5);

        let file = Hdf5File::open(&path).unwrap();
        let (_, out) = read_all(&*dataset(&file, "scalar"), &[]);
        assert_eq!(out, 2.5f64.to_le_bytes());

        let d = dataset(&file, "empty");
        assert_eq!(d.shape(), &[0, 5]);
        assert!(read_all(&*d, &[]).1.is_empty());

        let d = dataset(&file, "filled");
        let (_, out) = read_all(&*d, &[]);
        assert_eq!(out, bytes_of(&[-7i32; 6]));
        let layout = d.layout(&[], &QualitySpec::default()).unwrap();
        let mut out = vec![0u8; 3];
        d.read_range(&layout, 6, &mut out).unwrap();
        // The last two bytes of one -7 and the first of the next.
        assert_eq!(out, [0xff, 0xff, 0xf9]);

        assert_eq!(read_all(&*dataset(&file, "zeros"), &[]).1, [0u8; 4]);
    }

    #[test]
    fn groups_are_walked_and_listed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.h5");
        let h5 = hdf5::File::create(&path).unwrap();
        let inner = h5
            .create_group("outer")
            .unwrap()
            .create_group("inner")
            .unwrap();
        write(&inner, "b", &[1u8, 2], &[2]);
        write(&inner, "a", &[3u8], &[1]);
        h5.new_dataset::<VarLenUnicode>()
            .shape([1])
            .create("text")
            .unwrap();
        h5.link_soft("/outer/inner/a", "alias").unwrap();
        h5.link_soft("/nowhere", "dangling").unwrap();
        drop(h5);

        let file = Hdf5File::open(&path).unwrap();
        for p in ["", "/", "outer", "/outer/inner/", "outer/inner/a", "alias"] {
            assert!(file.contains(p), "{p}");
        }
        assert!(!file.contains("outer/missing"));
        assert!(matches!(file.get_item("/outer").unwrap(), Item::Group));
        assert_eq!(read_all(&*dataset(&file, "/alias"), &[]).1, [3]);

        let names = |p: &str| -> Vec<String> {
            file.list_children(p)
                .unwrap()
                .into_iter()
                .map(|(n, _)| n)
                .collect()
        };
        assert_eq!(names("outer/inner"), ["a", "b"]);
        // The string dataset and the dangling link are left out, not fatal.
        assert_eq!(names("/"), ["alias", "outer"]);

        assert!(matches!(
            file.get_item("outer/missing").unwrap_err(),
            AexError::NotFound(_)
        ));
        assert!(matches!(
            file.list_children("alias").unwrap_err(),
            AexError::NotAGroup(_)
        ));
        assert!(matches!(
            file.get_item("text").unwrap_err(),
            AexError::UnsupportedDType(_)
        ));
    }

    #[test]
    fn external_links_are_not_followed() {
        // Opening any file disables external links for the whole process, so
        // the file holding one is written by a child that has opened nothing.
        const DIR_VAR: &str = "AEX_TEST_WRITE_EXTERNAL_LINK";
        if let Some(dir) = std::env::var_os(DIR_VAR) {
            let dir = Path::new(&dir);
            let secret = dir.join("secret.h5");
            let h5 = hdf5::File::create(&secret).unwrap();
            write(&h5, "data", &[42u8], &[1]);
            let h5 = hdf5::File::create(dir.join("t.h5")).unwrap();
            h5.link_external(secret.to_str().unwrap(), "/data", "leak")
                .unwrap();
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "backends::hdf5::tests::external_links_are_not_followed",
            ])
            .env(DIR_VAR, dir.path())
            .status()
            .unwrap();
        assert!(status.success());

        let file = Hdf5File::open(dir.path().join("t.h5")).unwrap();
        assert!(!file.contains("leak"));
        assert!(matches!(
            file.get_item("leak").unwrap_err(),
            AexError::NotFound(_)
        ));
        assert!(file.list_children("").unwrap().is_empty());
    }

    #[test]
    fn unservable_datasets_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.h5");
        let h5 = hdf5::File::create(&path).unwrap();
        // hdf5-metno only writes native types, so big-endian goes through the C API.
        let be_id = hdf5::sync::sync(|| *hdf5_sys::h5t::H5T_STD_I32BE);
        let space = hdf5::Dataspace::try_new(&[3][..]).unwrap();
        let name = std::ffi::CString::new("big").unwrap();
        hdf5::sync::sync(|| unsafe {
            let id = hdf5_sys::h5d::H5Dcreate2(
                h5.id(),
                name.as_ptr(),
                be_id,
                space.id(),
                hdf5_sys::h5p::H5P_DEFAULT,
                hdf5_sys::h5p::H5P_DEFAULT,
                hdf5_sys::h5p::H5P_DEFAULT,
            );
            assert!(id >= 0);
            hdf5_sys::h5d::H5Dclose(id);
        });
        h5.new_dataset::<u8>()
            .shape([4])
            .chunk([2])
            .create("chunked")
            .unwrap();
        h5.new_dataset_builder()
            .with_data(&[1u8, 2])
            .layout(Layout::Compact)
            .create("compact")
            .unwrap();
        drop(h5);

        let file = Hdf5File::open(&path).unwrap();
        let err = |p: &str| file.get_item(p).unwrap_err();
        assert!(matches!(err("big"), AexError::UnsupportedDType(_)));
        assert!(matches!(err("chunked"), AexError::UnsupportedHdf5(_)));
        assert!(matches!(err("compact"), AexError::UnsupportedHdf5(_)));
        assert_eq!(err("compact").class(), crate::ErrorClass::Request);
    }

    #[test]
    fn a_file_that_is_not_hdf5_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.nc");
        std::fs::write(&path, b"CDF\x01 not HDF5 at all").unwrap();
        assert!(matches!(
            Hdf5File::open(&path).unwrap_err(),
            AexError::UnsupportedHdf5(_)
        ));
        let missing = Hdf5File::open(dir.path().join("missing.h5")).unwrap_err();
        assert_eq!(missing.class(), crate::ErrorClass::Request);
    }

    #[test]
    fn many_threads_read_at_once() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.h5");
        let h5 = hdf5::File::create(&path).unwrap();
        let values: Vec<u64> = (0..1 << 16).collect();
        let bytes = write(&h5, "a", &values, &[values.len()]);
        drop(h5);

        let file = Hdf5File::open(&path).unwrap();
        let d = dataset(&file, "a");
        let layout = d.layout(&[], &QualitySpec::default()).unwrap();
        let piece = 4096 + 3;
        std::thread::scope(|s| {
            for t in 0..8u64 {
                let (d, layout, bytes) = (&d, &layout, &bytes);
                s.spawn(move || {
                    for k in 0..(bytes.len() as u64 / piece) {
                        let at = (k * 7919 + t * 104729) % (bytes.len() as u64 - piece);
                        let mut out = vec![0u8; piece as usize];
                        d.read_range(layout, at, &mut out).unwrap();
                        assert_eq!(out, bytes[at as usize..(at + piece) as usize]);
                    }
                });
            }
        });
    }

    #[test]
    fn fill_patterns_wrap_mid_element() {
        let mut out = [0u8; 5];
        fill_from(&[1, 2, 3], 4, &mut out);
        assert_eq!(out, [2, 3, 1, 2, 3]);
    }
}
