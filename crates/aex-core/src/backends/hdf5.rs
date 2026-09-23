//! The HDF5 backend, which also serves netCDF-4 (HDF5 underneath).
//!
//! libhdf5 serializes every call behind one process-wide lock, so reading data
//! through it would put all data connections in single file. It is only asked
//! for metadata: where a dataset's bytes are, and what they mean. The bytes
//! themselves are `pread` from a descriptor of our own, as with `.npy`.
//!
//! Chunked datasets are decoded here too: the chunk index comes from libhdf5,
//! the compressed bytes from `pread`, and the filters are undone by us, with
//! the result kept in a [`DecodeCache`].
//!
//! Like the `.npy` backend, this assumes nobody writes to a file while it is
//! served: the offsets are taken once, when a dataset is opened.

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::{Arc, Mutex, Once, OnceLock};

use hdf5::dataset::{FillValue, Layout};
use hdf5::datatype::ByteOrder;
use hdf5::filters::Filter;
use hdf5::types::{CompoundType, FloatSize, IntSize, TypeDescriptor, VarLenAscii, VarLenUnicode};
use hdf5::LocationType;

use crate::backend::{normalize_path, ArrayDataset, ArrayFile, AttrValue, Item, MAX_ATTR_BYTES};
use crate::backends::chunks::{fill_from, ChunkGrid, MAX_CHUNKS};
use crate::backends::decode_cache::DecodeCache;
use crate::dtype::DType;
use crate::error::{AexError, Result};
use crate::quality::QualitySpec;
use crate::selection::{Index, SelectionLayout};

/// An HDF5 file opened for reading.
pub struct Hdf5File {
    /// For metadata only; every call on it takes libhdf5's global lock.
    file: hdf5::File,
    /// For data. Shared by every dataset opened from this file.
    raw: Arc<File>,
    cache: Arc<DecodeCache>,
    /// Opened once per path, so a chunk index is built once and the decoded
    /// chunks of one plan serve the next.
    datasets: Mutex<HashMap<String, Arc<Hdf5Dataset>>>,
}

impl Hdf5File {
    /// Open `path`, decoding chunks into `cache`.
    pub fn open(path: impl AsRef<Path>, cache: Arc<DecodeCache>) -> Result<Self> {
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
        Ok(Hdf5File {
            file,
            raw,
            cache,
            datasets: Mutex::default(),
        })
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
            LocationType::Dataset => Ok(Item::Dataset(self.dataset(path)?)),
            other => Err(AexError::NotFound(format!(
                "{path:?} is a {other:?}, not a group or a dataset"
            ))),
        }
    }

    fn dataset(&self, path: &str) -> Result<Arc<Hdf5Dataset>> {
        let mut datasets = self.datasets.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(dataset) = datasets.get(path) {
            return Ok(dataset.clone());
        }
        let dataset = Arc::new(Hdf5Dataset::open(
            self.file.dataset(path)?,
            self.raw.clone(),
            self.cache.clone(),
        )?);
        datasets.insert(path.to_string(), dataset.clone());
        Ok(dataset)
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

    fn attrs(&self, path: &str) -> Result<Vec<(String, AttrValue)>> {
        let path = normalize_path(path);
        // The root group is where netCDF keeps the global attributes.
        if path.is_empty() {
            return read_attrs(&self.file);
        }
        let info = self
            .file
            .loc_info_by_name(path)
            .map_err(|_| AexError::NotFound(path.to_string()))?;
        match info.loc_type {
            LocationType::Group => {
                let group = self.file.group(path)?;
                read_attrs(&group)
            }
            LocationType::Dataset => {
                let dataset = self.file.dataset(path)?;
                read_attrs(&dataset)
            }
            other => Err(AexError::NotFound(format!(
                "{path:?} is a {other:?}, not a group or a dataset"
            ))),
        }
    }
}

/// Every attribute of an object that AEX can represent.
///
/// libhdf5 iterates the name index in increasing order, so the result is
/// already sorted; unlike `member_names`, this needs no sort of its own.
fn read_attrs(location: &hdf5::Location) -> Result<Vec<(String, AttrValue)>> {
    let mut attrs = Vec::new();
    for name in location.attr_names()? {
        // A type with no wire form leaves the attribute out, the way an
        // unservable child is left out of a listing.
        if let Ok(attr) = location.attr(&name) {
            if let Some(value) = attr_value(&attr) {
                attrs.push((name, value));
            }
        }
    }
    Ok(attrs)
}

/// One attribute, or `None` if AEX has no way to carry it.
fn attr_value(attr: &hdf5::Attribute) -> Option<AttrValue> {
    let datatype = attr.dtype().ok()?;
    let shape: Vec<u64> = attr.shape().iter().map(|&n| n as u64).collect();
    // A scalar dataspace has no axes and holds one element.
    let elements: u64 = shape.iter().product();
    match datatype.to_descriptor().ok()? {
        // One string per attribute: an array of them has no wire form.
        TypeDescriptor::VarLenUnicode if elements == 1 => {
            let mut read = attr.read_raw::<VarLenUnicode>().ok()?;
            Some(AttrValue::Text(read.pop()?.as_str().to_owned()))
        }
        TypeDescriptor::VarLenAscii if elements == 1 => {
            let mut read = attr.read_raw::<VarLenAscii>().ok()?;
            Some(AttrValue::Text(read.pop()?.as_str().to_owned()))
        }
        // How netCDF-4 writes text, so this is the common case rather than the
        // odd one. The bytes are taken as they are stored: libhdf5 refuses to
        // convert between character sets, and `FixedAscii` wants its length at
        // compile time, so neither route to a variable-length string exists.
        TypeDescriptor::FixedAscii(len) | TypeDescriptor::FixedUnicode(len) if elements == 1 => {
            let raw = read_raw(attr, &datatype, len as u64)?;
            let end = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
            Some(AttrValue::Text(
                std::str::from_utf8(&raw[..end]).ok()?.to_owned(),
            ))
        }
        // Compound, enum, references, opaque and arrays of strings land here
        // and are refused by `dtype_of`.
        _ => {
            let dtype = dtype_of(&datatype).ok()?;
            let data = read_raw(attr, &datatype, dtype.itemsize().checked_mul(elements)?)?;
            Some(AttrValue::Array { dtype, shape, data })
        }
    }
}

/// `len` bytes of an attribute, asked for in its own type.
///
/// No conversion, so the bytes are what storage holds. `dtype_of` has already
/// refused anything but little-endian numbers, which is what the wire wants.
fn read_raw(attr: &hdf5::Attribute, datatype: &hdf5::Datatype, len: u64) -> Option<Vec<u8>> {
    if len > MAX_ATTR_BYTES {
        return None;
    }
    let mut buf = vec![0u8; usize::try_from(len).ok()?];
    let status = hdf5::sync::sync(|| unsafe {
        hdf5_sys::h5a::H5Aread(attr.id(), datatype.id(), buf.as_mut_ptr().cast())
    });
    (status >= 0).then_some(buf)
}

/// Where a dataset's bytes are.
#[derive(Debug)]
enum Storage {
    /// One run of the file starting here.
    Contiguous(u64),
    /// Never written: every element reads as the fill value.
    Unallocated,
    Chunked(Box<Chunked>),
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
    fn open(dataset: hdf5::Dataset, raw: Arc<File>, cache: Arc<DecodeCache>) -> Result<Self> {
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
            Layout::Chunked => {
                let chunk_shape = dcpl
                    .chunk()
                    .ok_or_else(|| AexError::MalformedHdf5(format!("{name} has no chunk shape")))?;
                let filters = dataset.filters();
                if let Some(filter) = filters.iter().find(|f| {
                    !matches!(f, Filter::Deflate(_) | Filter::Shuffle | Filter::Fletcher32)
                }) {
                    return Err(AexError::UnsupportedHdf5(format!(
                        "{name} uses the {filter:?} filter; only deflate, shuffle and \
                         fletcher32 are decoded"
                    )));
                }
                let chunk_shape: Vec<u64> = chunk_shape.iter().map(|&n| n as u64).collect();
                Storage::Chunked(Box::new(Chunked::new(
                    dataset.clone(),
                    &name,
                    &shape,
                    dtype,
                    chunk_shape,
                    filters,
                    cache,
                )?))
            }
            other => {
                return Err(AexError::UnsupportedHdf5(format!(
                    "{name} has {other:?} layout, which is not served"
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

        let fill = fill_value(&dataset, &dcpl, dtype)?;
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

    /// Builds the chunk index here, on the control plane, so the data plane
    /// never waits on libhdf5 and a bad index fails the request that met it.
    fn layout(&self, indices: &[Index], quality: &QualitySpec) -> Result<SelectionLayout> {
        if let Storage::Chunked(chunked) = &self.storage {
            chunked.index(&self.raw)?;
        }
        SelectionLayout::resolve(&self.shape, self.dtype, indices, quality)
    }

    fn read_range(&self, layout: &SelectionLayout, offset: u64, dst: &mut [u8]) -> Result<()> {
        match &self.storage {
            Storage::Contiguous(base) => layout.read_with(offset, dst, |at, buf| {
                Ok(self.raw.read_exact_at(buf, base + at)?)
            }),
            Storage::Unallocated => layout.read_with(offset, dst, |at, buf| {
                fill_from(&self.fill, at, buf);
                Ok(())
            }),
            Storage::Chunked(chunked) => {
                layout.read_with(offset, dst, |at, buf| chunked.read(self, at, buf))
            }
        }
    }

    fn will_need(&self, layout: &SelectionLayout, offset: u64, len: u64) {
        // Only an uncompressed contiguous dataset maps a logical range to one
        // file range; a chunked one is served through the decode cache.
        if let Storage::Contiguous(base) = &self.storage {
            if let Some((at, len)) = layout.source_run(offset, len) {
                super::will_need(&self.raw, base + at, len);
            }
        }
    }

    fn decoded_chunk_bytes(&self) -> Option<u64> {
        match &self.storage {
            Storage::Chunked(chunked) if !chunked.filters.is_empty() => {
                Some(chunked.grid.chunk_bytes())
            }
            _ => None,
        }
    }
}

/// Where one chunk is stored.
#[derive(Debug, Clone, Copy)]
struct ChunkEntry {
    addr: u64,
    size: u64,
    /// Bit `i` set: filter `i` of the pipeline was skipped for this chunk.
    filter_mask: u32,
}

impl ChunkEntry {
    /// A chunk never written, which reads as the fill value.
    const MISSING: ChunkEntry = ChunkEntry {
        addr: u64::MAX,
        size: 0,
        filter_mask: 0,
    };
}

/// A chunked dataset: a regular grid of chunks, each stored on its own.
#[derive(Debug)]
struct Chunked {
    /// Where the index comes from.
    dataset: hdf5::Dataset,
    grid: ChunkGrid,
    /// In the order they were applied when writing.
    filters: Vec<Filter>,
    /// By chunk number in C order; built on first use.
    index: OnceLock<Vec<ChunkEntry>>,
    cache: Arc<DecodeCache>,
    /// Tells this dataset's chunks apart from others' in the shared cache.
    cache_key: u64,
}

impl Chunked {
    fn new(
        dataset: hdf5::Dataset,
        name: &str,
        shape: &[u64],
        dtype: DType,
        chunk_shape: Vec<u64>,
        filters: Vec<Filter>,
        cache: Arc<DecodeCache>,
    ) -> Result<Self> {
        let malformed = || {
            AexError::MalformedHdf5(format!(
                "{name}: chunks of {chunk_shape:?} do not fit an array of {shape:?}"
            ))
        };
        let grid = ChunkGrid::new(shape, &chunk_shape, dtype.itemsize()).ok_or_else(malformed)?;
        let chunks = grid.chunks();
        if chunks > MAX_CHUNKS {
            return Err(AexError::UnsupportedHdf5(format!(
                "{name} has {chunks} chunks; at most {MAX_CHUNKS} are served"
            )));
        }
        Ok(Chunked {
            dataset,
            grid,
            filters,
            index: OnceLock::new(),
            cache_key: cache.new_key(),
            cache,
        })
    }

    /// The chunk index, reading it from libhdf5 the first time.
    fn index(&self, raw: &File) -> Result<&[ChunkEntry]> {
        if let Some(index) = self.index.get() {
            return Ok(index);
        }
        let name = self.dataset.name();
        let file_len = raw.metadata()?.len();
        let mut index = vec![ChunkEntry::MISSING; self.grid.chunks() as usize];
        let mut bad = None;
        self.dataset.chunks_visit(|chunk| {
            let coords = chunk
                .offset
                .iter()
                .zip(self.grid.chunk_shape())
                .map(|(o, c)| o / c);
            // A chunk left behind by shrinking the dataset is not part of it.
            let Some(number) = self.grid.chunk_of(coords) else {
                return 0;
            };
            let entry = ChunkEntry {
                addr: chunk.addr,
                size: chunk.size,
                filter_mask: chunk.filter_mask,
            };
            let unfiltered = self.filters.is_empty() && entry.size != self.grid.chunk_bytes();
            if unfiltered
                || entry
                    .addr
                    .checked_add(entry.size)
                    .is_none_or(|end| end > file_len)
            {
                bad = Some(entry);
                return -1;
            }
            index[number] = entry;
            0
        })?;
        if let Some(entry) = bad {
            return Err(AexError::MalformedHdf5(format!(
                "{name}: a chunk of {} bytes at {} does not fit the file ({file_len} bytes) \
                 or the chunk shape",
                entry.size, entry.addr
            )));
        }
        Ok(self.index.get_or_init(|| index))
    }

    /// Read `[at, at + dst.len())` of the dataset's C-order bytes.
    fn read(&self, dataset: &Hdf5Dataset, at: u64, dst: &mut [u8]) -> Result<()> {
        let index = self.index(&dataset.raw)?;
        self.grid.walk(at, dst, |chunk, start, out| {
            let entry = index[chunk];
            if entry.addr == ChunkEntry::MISSING.addr {
                fill_from(&dataset.fill, start % self.grid.itemsize(), out);
            } else if self.filters.is_empty() {
                dataset.raw.read_exact_at(out, entry.addr + start)?;
            } else {
                let decoded = self
                    .cache
                    .get_or_decode((self.cache_key, chunk as u64), |spare| {
                        self.decode(&dataset.raw, entry, spare)
                    })?;
                let start = start as usize;
                out.copy_from_slice(&decoded[start..start + out.len()]);
            }
            Ok(())
        })
    }

    /// Read one chunk and undo its filters.
    ///
    /// `spare` is the buffer the cache lent. A filter that produces a new
    /// buffer writes into it and then swaps, so the last one to run leaves its
    /// output there and that is what the cache keeps.
    fn decode(&self, raw: &File, entry: ChunkEntry, mut spare: Vec<u8>) -> Result<Vec<u8>> {
        let malformed =
            |what: &str| AexError::MalformedHdf5(format!("chunk at byte {}: {what}", entry.addr));
        let mut bytes = vec![0u8; entry.size as usize];
        raw.read_exact_at(&mut bytes, entry.addr)?;
        for (i, filter) in self.filters.iter().enumerate().rev() {
            if entry.filter_mask & (1 << i) != 0 {
                continue;
            }
            match filter {
                Filter::Fletcher32 => {
                    bytes = strip_fletcher32(bytes)
                        .ok_or_else(|| malformed("fletcher32 checksum mismatch"))?;
                }
                Filter::Deflate(_) => {
                    inflate(&bytes, self.grid.chunk_bytes(), &mut spare)
                        .ok_or_else(|| malformed("deflate stream is corrupt"))?;
                    std::mem::swap(&mut bytes, &mut spare);
                }
                Filter::Shuffle => {
                    unshuffle(&bytes, self.grid.itemsize() as usize, &mut spare);
                    std::mem::swap(&mut bytes, &mut spare);
                }
                other => unreachable!("{other:?} was rejected when the dataset was opened"),
            }
        }
        if bytes.len() as u64 != self.grid.chunk_bytes() {
            return Err(malformed(&format!(
                "decodes to {} bytes instead of {}",
                bytes.len(),
                self.grid.chunk_bytes()
            )));
        }
        Ok(bytes)
    }
}

/// Undo zlib into `out`, refusing to produce more than `limit` bytes.
fn inflate(bytes: &[u8], limit: u64, out: &mut Vec<u8>) -> Option<()> {
    out.clear();
    out.reserve(limit as usize);
    flate2::read::ZlibDecoder::new(bytes)
        .take(limit + 1)
        .read_to_end(out)
        .ok()?;
    Some(())
}

/// Undo HDF5's shuffle into `out`: byte `b` of every element was moved to
/// block `b`. Whatever the permutation leaves untouched is the input itself,
/// so `out` starts as a copy of it.
fn unshuffle(bytes: &[u8], itemsize: usize, out: &mut Vec<u8>) {
    out.clear();
    out.extend_from_slice(bytes);
    if itemsize > 1 {
        let elements = bytes.len() / itemsize;
        for b in 0..itemsize {
            let block = &bytes[b * elements..(b + 1) * elements];
            for (e, &byte) in block.iter().enumerate() {
                out[e * itemsize + b] = byte;
            }
        }
    }
}

/// Check and drop the trailing checksum HDF5's fletcher32 filter appends.
fn strip_fletcher32(mut bytes: Vec<u8>) -> Option<Vec<u8>> {
    let body = bytes.len().checked_sub(4)?;
    let stored = u32::from_le_bytes(bytes[body..].try_into().unwrap());
    let sum = fletcher32(&bytes[..body]);
    // Before 1.6.3, libhdf5 wrote each half of the sum byte-swapped on
    // little-endian machines, and it still accepts both.
    let swapped = ((sum & 0x00ff_00ff) << 8) | ((sum >> 8) & 0x00ff_00ff);
    if stored != sum && stored != swapped {
        return None;
    }
    bytes.truncate(body);
    Some(bytes)
}

/// libhdf5's Fletcher-32: big-endian 16-bit words, an odd byte padded.
fn fletcher32(data: &[u8]) -> u32 {
    let (mut a, mut b) = (0u32, 0u32);
    // 360 words keep the sums inside u32 between reductions.
    for block in data.chunks(720) {
        for word in block.chunks(2) {
            a += u32::from(word[0]) << 8 | u32::from(*word.get(1).unwrap_or(&0));
            b += a;
        }
        a = (a & 0xffff) + (a >> 16);
        b = (b & 0xffff) + (b >> 16);
    }
    a = (a & 0xffff) + (a >> 16);
    b = (b & 0xffff) + (b >> 16);
    (b << 16) | a
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
    use hdf5::types::{FixedAscii, VarLenUnicode};
    use num_complex::{Complex32, Complex64};

    use super::*;
    use crate::quality::QualitySpec;
    use crate::selection::{Index, LayoutKind};

    /// Open a fixture, retrying a refused file lock.
    ///
    /// libhdf5 takes a file lock when it opens a file. On macOS, with these
    /// tests running in parallel, the lock is now and then still refused a
    /// fraction of a millisecond after the writer closed its file — no other
    /// process or descriptor holds it by then, and the next attempt succeeds.
    /// Nothing outside the tests waits on it, so the retry lives here.
    fn open(path: impl AsRef<Path>) -> Result<Hdf5File> {
        open_with(path, Arc::new(DecodeCache::new(1 << 20)))
    }

    fn open_with(path: impl AsRef<Path>, cache: Arc<DecodeCache>) -> Result<Hdf5File> {
        let path = path.as_ref();
        for _ in 0..20 {
            match Hdf5File::open(path, cache.clone()) {
                Err(AexError::UnsupportedHdf5(e)) if e.contains("unable to lock file") => {
                    std::thread::sleep(std::time::Duration::from_millis(25));
                }
                other => return other,
            }
        }
        Hdf5File::open(path, cache)
    }

    fn read_all(dataset: &dyn ArrayDataset, indices: &[Index]) -> (SelectionLayout, Vec<u8>) {
        let layout = dataset.layout(indices, &QualitySpec::default()).unwrap();
        let mut out = vec![0u8; layout.total_bytes as usize];
        dataset.read_range(&layout, 0, &mut out).unwrap();
        (layout, out)
    }

    fn dataset(file: &dyn ArrayFile, path: &str) -> Arc<dyn ArrayDataset> {
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

        let file = open(&path).unwrap();
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

        let file = open(&path).unwrap();
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

        let file = open(&path).unwrap();
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

        let file = open(&path).unwrap();
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

        let file = open(dir.path().join("t.h5")).unwrap();
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
            .nbit()
            .create("nbit")
            .unwrap();
        h5.new_dataset_builder()
            .with_data(&[1u8, 2])
            .layout(Layout::Compact)
            .create("compact")
            .unwrap();
        drop(h5);

        let file = open(&path).unwrap();
        let err = |p: &str| file.get_item(p).unwrap_err();
        assert!(matches!(err("big"), AexError::UnsupportedDType(_)));
        assert!(matches!(err("nbit"), AexError::UnsupportedHdf5(_)));
        assert!(matches!(err("compact"), AexError::UnsupportedHdf5(_)));
        assert_eq!(err("compact").class(), crate::ErrorClass::Request);
    }

    #[test]
    fn a_file_that_is_not_hdf5_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.nc");
        std::fs::write(&path, b"CDF\x01 not HDF5 at all").unwrap();
        assert!(matches!(
            open(&path).unwrap_err(),
            AexError::UnsupportedHdf5(_)
        ));
        let missing = open(dir.path().join("missing.h5")).unwrap_err();
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

        let file = open(&path).unwrap();
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

    /// Every C-order byte range of `d` that a transfer could ask for, checked
    /// against `bytes`.
    fn check_ranges(d: &dyn ArrayDataset, bytes: &[u8]) {
        let layout = d.layout(&[], &QualitySpec::default()).unwrap();
        for piece in [1, 3, 7, 64, bytes.len()] {
            let mut at = 0;
            while at < bytes.len() {
                let len = piece.min(bytes.len() - at);
                let mut out = vec![0u8; len];
                d.read_range(&layout, at as u64, &mut out).unwrap();
                assert_eq!(out, bytes[at..at + len], "piece {piece} at {at}");
                at += len;
            }
        }
    }

    #[test]
    fn chunked_datasets_read_back_under_every_filter() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.h5");
        let h5 = hdf5::File::create(&path).unwrap();
        // Shapes whose chunks leave edges, cover a trailing axis exactly, or both.
        let shapes: [(&[usize], &[usize]); 5] = [
            (&[10, 7], &[3, 4]),
            (&[10, 7], &[3, 7]),
            (&[4, 5, 6], &[3, 5, 6]),
            (&[4, 5, 6], &[4, 2, 6]),
            (&[9], &[4]),
        ];
        let mut cases = Vec::new();
        for (s, (shape, chunk)) in shapes.iter().enumerate() {
            let n: usize = shape.iter().product();
            let values: Vec<u32> = (0..n as u32)
                .map(|i| i.wrapping_mul(2_654_435_761))
                .collect();
            for (f, filters) in ["none", "deflate", "shuffle+deflate", "fletcher32", "all"]
                .iter()
                .enumerate()
            {
                let name = format!("d{s}_{f}");
                let mut builder = h5.new_dataset::<u32>().shape(*shape).chunk(*chunk);
                builder = match *filters {
                    "none" => builder,
                    "deflate" => builder.deflate(6),
                    "shuffle+deflate" => builder.shuffle().deflate(1),
                    "fletcher32" => builder.fletcher32(),
                    _ => builder.shuffle().deflate(9).fletcher32(),
                };
                builder
                    .create(name.as_str())
                    .unwrap()
                    .write_raw(&values)
                    .unwrap();
                cases.push((name, filters.to_string(), bytes_of(&values)));
            }
        }
        drop(h5);

        let file = open(&path).unwrap();
        for (name, filters, bytes) in cases {
            let d = dataset(&file, &name);
            let (_, out) = read_all(&*d, &[]);
            assert_eq!(out, bytes, "{name} {filters}");
            check_ranges(&*d, &bytes);
            assert_eq!(
                d.decoded_chunk_bytes().is_some(),
                filters != "none",
                "{name}"
            );
        }

        // A gathered selection across chunk boundaries.
        let d = dataset(&file, "d0_2");
        let step = Index::Slice {
            start: Some(1),
            stop: None,
            step: Some(3),
        };
        let (layout, out) = read_all(&*d, &[step, Index::Fancy(vec![6, 0, 3])]);
        assert!(matches!(layout.kind, LayoutKind::Gathered(_)));
        let values: Vec<u32> = (0..70u32).map(|i| i.wrapping_mul(2_654_435_761)).collect();
        let expected: Vec<u32> = [1, 4, 7]
            .iter()
            .flat_map(|r| [6, 0, 3].map(|c| values[r * 7 + c]))
            .collect();
        assert_eq!(out, bytes_of(&expected));
    }

    #[test]
    fn unwritten_chunks_read_as_the_fill_value() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.h5");
        let h5 = hdf5::File::create(&path).unwrap();
        for (name, deflate) in [("plain", false), ("deflated", true)] {
            let mut builder = h5
                .new_dataset::<i16>()
                .shape([10])
                .chunk([4])
                .fill_value(-2i16);
            if deflate {
                builder = builder.deflate(1);
            }
            builder
                .create(name)
                .unwrap()
                .write_slice(&[5i16, 6, 7][..], 4..7)
                .unwrap();
        }
        drop(h5);

        let file = open(&path).unwrap();
        let expected = bytes_of(&[-2i16, -2, -2, -2, 5, 6, 7, -2, -2, -2]);
        for name in ["plain", "deflated"] {
            let d = dataset(&file, name);
            assert_eq!(read_all(&*d, &[]).1, expected, "{name}");
            check_ranges(&*d, &expected);
        }
    }

    #[test]
    fn a_corrupt_chunk_is_reported() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.h5");
        let h5 = hdf5::File::create(&path).unwrap();
        let values: Vec<u64> = (0..64).collect();
        let d = h5
            .new_dataset::<u64>()
            .shape([64])
            .chunk([16])
            .fletcher32()
            .create("d")
            .unwrap();
        d.write_raw(&values).unwrap();
        let chunk = d.chunk_info(1).unwrap();
        drop((d, h5));

        let raw = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        raw.write_all_at(&[0xff], chunk.addr + 3).unwrap();
        drop(raw);

        let file = open(&path).unwrap();
        let d = dataset(&file, "d");
        let layout = d.layout(&[], &QualitySpec::default()).unwrap();
        let mut out = vec![0u8; 8 * 16];
        // Chunk 0 is intact, chunk 1 is not.
        d.read_range(&layout, 0, &mut out).unwrap();
        assert_eq!(out, bytes_of(&values[..16]));
        let err = d.read_range(&layout, 8 * 16, &mut out).unwrap_err();
        assert!(matches!(err, AexError::MalformedHdf5(_)), "{err}");
        assert_eq!(err.class(), crate::ErrorClass::Permanent);
    }

    #[test]
    fn many_threads_share_a_small_cache() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.h5");
        let h5 = hdf5::File::create(&path).unwrap();
        let values: Vec<f64> = (0..1 << 15).map(|i| (i as f64).sin()).collect();
        h5.new_dataset::<f64>()
            .shape([values.len()])
            .chunk([1000])
            .shuffle()
            .deflate(4)
            .create("d")
            .unwrap()
            .write_raw(&values)
            .unwrap();
        drop(h5);
        let bytes = bytes_of(&values);

        // Room for three chunks among eight threads: constant eviction.
        let file = Hdf5File::open(&path, Arc::new(DecodeCache::new(3 * 8000))).unwrap();
        let d = dataset(&file, "d");
        let layout = d.layout(&[], &QualitySpec::default()).unwrap();
        let piece = 3001u64;
        std::thread::scope(|s| {
            for t in 0..8u64 {
                let (d, layout, bytes) = (&d, &layout, &bytes);
                s.spawn(move || {
                    for k in 0..40 {
                        let at = (k * 7919 + t * 104_729) % (bytes.len() as u64 - piece);
                        let mut out = vec![0u8; piece as usize];
                        d.read_range(layout, at, &mut out).unwrap();
                        assert_eq!(out, bytes[at as usize..(at + piece) as usize]);
                    }
                });
            }
        });
    }

    #[test]
    fn a_zarr_array_sharing_the_cache_does_not_serve_its_chunks() {
        let dir = tempfile::tempdir().unwrap();
        let store = dir.path().join("s.zarr");
        std::fs::create_dir_all(store.join("a/c")).unwrap();
        std::fs::write(
            store.join("zarr.json"),
            r#"{"zarr_format": 3, "node_type": "group"}"#,
        )
        .unwrap();
        std::fs::write(
            store.join("a/zarr.json"),
            r#"{"zarr_format": 3, "node_type": "array", "data_type": "uint32",
                "shape": [4], "chunk_grid": {"name": "regular",
                "configuration": {"chunk_shape": [4]}},
                "chunk_key_encoding": {"name": "default"}, "fill_value": 0,
                "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}]}"#,
        )
        .unwrap();
        std::fs::write(store.join("a/c/0"), bytes_of(&[7u32; 4])).unwrap();

        let path = dir.path().join("t.h5");
        let values = [1u32, 2, 3, 4];
        hdf5::File::create(&path)
            .unwrap()
            .new_dataset::<u32>()
            .shape([4])
            .chunk([4])
            .deflate(1)
            .create("d")
            .unwrap()
            .write_raw(&values)
            .unwrap();

        // Both formats' first chunk used to be key (0, 0) in one cache.
        let cache = Arc::new(DecodeCache::new(1 << 20));
        let zarr = crate::ZarrFile::open(&store, cache.clone()).unwrap();
        let file = open_with(&path, cache).unwrap();
        read_all(&*dataset(&zarr, "a"), &[]);
        assert_eq!(read_all(&*dataset(&file, "d"), &[]).1, bytes_of(&values));
    }

    #[test]
    fn a_dataset_is_opened_once_per_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.h5");
        let h5 = hdf5::File::create(&path).unwrap();
        write(&h5, "a", &[1u8], &[1]);
        drop(h5);
        let file = open(&path).unwrap();
        assert!(Arc::ptr_eq(&dataset(&file, "a"), &dataset(&file, "/a/")));
    }

    #[test]
    fn filters_undo_what_libhdf5_does() {
        // Two words and an odd byte.
        assert_eq!(fletcher32(&[0x01, 0x02, 0x03, 0x04, 0x05]), 0x0e0e_0906);
        let mut out = Vec::new();
        unshuffle(&[1, 3, 5, 2, 4, 6], 2, &mut out);
        assert_eq!(out, [1, 2, 3, 4, 5, 6]);
        // A trailing partial element is left in place, and the buffer is
        // reused rather than grown.
        unshuffle(&[1, 3, 2, 4, 9], 2, &mut out);
        assert_eq!(out, [1, 2, 3, 4, 9]);
        assert!(inflate(b"not zlib", 10, &mut out).is_none());
    }

    /// Write a scalar attribute of type `T` on `location`.
    fn attr<T: hdf5::H5Type>(location: &hdf5::Location, name: &str, value: T) {
        location
            .new_attr::<T>()
            .create(name)
            .unwrap()
            .write_scalar(&value)
            .unwrap();
    }

    fn text(value: &str) -> VarLenUnicode {
        value.parse().unwrap()
    }

    #[test]
    fn attributes_keep_their_text_and_their_dtype() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.h5");
        let h5 = hdf5::File::create(&path).unwrap();
        let ds = h5.new_dataset::<i16>().shape([4]).create("ds").unwrap();
        // Named out of alphabetical order, to pin that the order comes back sorted.
        attr(&ds, "units", text("K"));
        // How netCDF-4 writes text: fixed-length ASCII, shorter than its type.
        attr(
            &ds,
            "long_name",
            FixedAscii::<16>::from_ascii("temperature").unwrap(),
        );
        attr(&ds, "_FillValue", -3i16);
        ds.new_attr::<f64>()
            .shape([2])
            .create("valid_range")
            .unwrap()
            .write_raw(&[0.0f64, 1.0])
            .unwrap();
        drop(h5);

        let file = open(&path).unwrap();
        let attrs = file.attrs("/ds").unwrap();
        assert_eq!(
            attrs.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>(),
            ["_FillValue", "long_name", "units", "valid_range"]
        );
        let value = |name: &str| attrs.iter().find(|(n, _)| n == name).unwrap().1.clone();
        assert_eq!(value("units"), AttrValue::Text("K".into()));
        // The padding of the fixed-length type does not come with it.
        assert_eq!(value("long_name"), AttrValue::Text("temperature".into()));
        // A _FillValue keeps the dtype of its dataset rather than widening.
        assert_eq!(
            value("_FillValue"),
            AttrValue::Array {
                dtype: DType::Int16,
                shape: vec![],
                data: bytes_of(&[-3i16]),
            }
        );
        assert_eq!(
            value("valid_range"),
            AttrValue::Array {
                dtype: DType::Float64,
                shape: vec![2],
                data: bytes_of(&[0.0f64, 1.0]),
            }
        );
    }

    #[test]
    fn groups_and_the_root_carry_attributes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.h5");
        let h5 = hdf5::File::create(&path).unwrap();
        // netCDF keeps its global attributes on the root group.
        attr(&h5, "Conventions", text("CF-1.8"));
        let g = h5.create_group("g").unwrap();
        attr(&g, "title", text("a group"));
        drop(h5);

        let file = open(&path).unwrap();
        for root in ["", "/", "//"] {
            assert_eq!(
                file.attrs(root).unwrap(),
                [("Conventions".to_string(), AttrValue::Text("CF-1.8".into()))]
            );
        }
        assert_eq!(
            file.attrs("/g").unwrap(),
            [("title".to_string(), AttrValue::Text("a group".into()))]
        );
        assert!(matches!(file.attrs("/nope"), Err(AexError::NotFound(_))));
    }

    #[test]
    fn an_attribute_with_no_wire_form_is_left_out() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.h5");
        let h5 = hdf5::File::create(&path).unwrap();
        let ds = h5.new_dataset::<i16>().shape([4]).create("ds").unwrap();
        attr(&ds, "units", text("K"));
        // An array of strings: one attribute cannot carry several of them.
        ds.new_attr::<VarLenUnicode>()
            .shape([2])
            .create("flag_meanings")
            .unwrap()
            .write_raw(&[text("a"), text("b")])
            .unwrap();
        // A compound type has no dtype of ours, unless it is a complex number.
        attr(&ds, "pair", Pair { a: 1, b: 2.0 });
        // Over the per-attribute cap. An attribute this big does not fit a
        // default object header, and libhdf5 1.14 refuses to create one at
        // all, so the cap goes unexercised wherever that is the case rather
        // than failing a test over the writer's limits.
        let huge = MAX_ATTR_BYTES as usize + 1;
        if let Ok(attr) = ds.new_attr::<u8>().shape([huge]).create("huge") {
            attr.write_raw(&vec![0u8; huge]).unwrap();
        }
        drop(h5);

        let file = open(&path).unwrap();
        // The siblings we can carry survive the ones we cannot.
        assert_eq!(
            file.attrs("/ds").unwrap(),
            [("units".to_string(), AttrValue::Text("K".into()))]
        );
    }

    #[derive(hdf5::H5Type, Clone, Copy)]
    #[repr(C)]
    struct Pair {
        a: i32,
        b: f64,
    }

    #[test]
    fn fill_patterns_wrap_mid_element() {
        let mut out = [0u8; 5];
        fill_from(&[1, 2, 3], 4, &mut out);
        assert_eq!(out, [2, 3, 1, 2, 3]);
    }
}
