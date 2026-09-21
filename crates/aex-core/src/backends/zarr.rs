//! The Zarr v3 backend.
//!
//! A store is a directory tree, one `zarr.json` per node, and one file per
//! chunk. Nothing but the metadata is parsed by anyone else: the chunk bytes
//! are read and decoded here, and kept in a [`DecodeCache`] so that however a
//! client splits the logical byte stream, a chunk is decoded once.
//!
//! Unlike HDF5 there is no index to build: a chunk's name follows from its
//! coordinates, so nothing has to be read before the first transfer.
//!
//! The codecs a chunk went through are listed in the order they were applied,
//! so decoding runs the bytes-to-bytes ones backwards.
//!
//! A sharded array puts many chunks in one file with an index of where each
//! one sits. That is the same problem twice over — a shard is a chunk of the
//! array, an inner chunk is a chunk of the shard — so the walk is the same
//! one, called inside itself.
//!
//! Every path read here — a node's metadata as much as a chunk — goes through
//! [`StoreRoot::under`], which resolves symlinks and refuses anything that
//! lands outside the store. The server decides which directory may be served;
//! this decides that nothing outside it is ever opened.
//!
//! Like the other backends, this assumes nobody writes to the store while it
//! is served. That assumption is weaker here than for a file, since stores are
//! routinely appended to: a chunk rewritten under a cached decode is served as
//! it was when the transfer started.

use std::collections::HashMap;
use std::fmt::Write;
use std::fs::File;
use std::io::Read;
use std::os::unix::fs::FileExt;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use serde::Deserialize;
use serde_json::{Map, Value};

use crate::backend::{normalize_path, ArrayDataset, ArrayFile, AttrValue, Item, MAX_ATTR_BYTES};
use crate::backends::chunks::{fill_from, ChunkGrid, MAX_CHUNKS};
use crate::backends::decode_cache::DecodeCache;
use crate::dtype::DType;
use crate::error::{AexError, Result};
use crate::quality::QualitySpec;
use crate::selection::{Index, SelectionLayout};

/// The metadata document of every node, group or array alike.
const METADATA: &str = "zarr.json";

/// A store directory, and the rule that nothing outside it is ever opened.
#[derive(Debug)]
struct StoreRoot {
    path: PathBuf,
}

impl StoreRoot {
    /// The absolute path of a store-relative name, or `None` if nothing is
    /// there.
    ///
    /// The name comes from a client (an item path) or from the store's own
    /// metadata (a chunk key); both are checked the same way, and by the same
    /// rule the data roots are checked with: resolve the symlinks, then
    /// compare.
    // ponytail: canonicalize then open, the same race PathPolicy already
    // accepts. Opening from a store-root fd would close it, at 2-5 syscalls
    // per chunk.
    fn under(&self, relative: &str) -> Result<Option<PathBuf>> {
        let name = Path::new(relative);
        if name
            .components()
            .any(|c| !matches!(c, Component::Normal(_)))
        {
            return Err(AexError::MalformedZarr(format!(
                "{relative:?} is not a name inside a store"
            )));
        }
        match std::fs::canonicalize(self.path.join(name)) {
            Ok(path) if path.starts_with(&self.path) => Ok(Some(path)),
            // Name what was asked for; where the link pointed is not the
            // client's business.
            Ok(_) => Err(AexError::MalformedZarr(format!(
                "{relative:?} leaves the store"
            ))),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }
}

/// A Zarr store opened for reading.
#[derive(Debug)]
pub struct ZarrFile {
    root: Arc<StoreRoot>,
    cache: Arc<DecodeCache>,
    /// Opened once per path, so one plan's decoded chunks serve the next.
    arrays: Mutex<HashMap<String, Arc<ZarrArray>>>,
}

impl ZarrFile {
    /// Open the store rooted at `path`.
    pub fn open(path: impl AsRef<Path>, cache: Arc<DecodeCache>) -> Result<Self> {
        let root = Arc::new(StoreRoot {
            path: std::fs::canonicalize(path)?,
        });
        let file = ZarrFile {
            root,
            cache,
            arrays: Mutex::default(),
        };
        // Refusing here means a directory that is not a store fails at open
        // rather than on the first item asked for.
        if file.node("")?.is_none() {
            return Err(AexError::UnsupportedZarr(format!(
                "no {METADATA} at the root: not a Zarr store"
            )));
        }
        Ok(file)
    }

    /// The node at a store-relative path, or `None` if there is none.
    fn node(&self, path: &str) -> Result<Option<Node>> {
        let key = if path.is_empty() {
            METADATA.to_string()
        } else {
            format!("{path}/{METADATA}")
        };
        let Some(file) = self.root.under(&key)? else {
            return Ok(None);
        };
        let text = std::fs::read_to_string(&file)?;
        let node: Node = serde_json::from_str(&text)
            .map_err(|e| AexError::MalformedZarr(format!("{key}: {e}")))?;
        node.check(&key)?;
        Ok(Some(node))
    }

    /// The array at a store-relative path, opening it the first time.
    fn array(&self, path: &str, meta: ArrayMeta) -> Result<Arc<ZarrArray>> {
        let mut arrays = self.arrays.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(array) = arrays.get(path) {
            return Ok(array.clone());
        }
        let array = Arc::new(ZarrArray::new(
            self.root.clone(),
            path,
            meta,
            self.cache.clone(),
        )?);
        arrays.insert(path.to_string(), array.clone());
        Ok(array)
    }

    fn item(&self, path: &str, node: Node) -> Result<Item> {
        match node {
            Node::Group(_) => Ok(Item::Group),
            Node::Array(meta) => Ok(Item::Dataset(self.array(path, *meta)?)),
        }
    }
}

impl ArrayFile for ZarrFile {
    fn contains(&self, path: &str) -> bool {
        matches!(self.node(normalize_path(path)), Ok(Some(_)))
    }

    fn get_item(&self, path: &str) -> Result<Item> {
        let path = normalize_path(path);
        match self.node(path)? {
            Some(node) => self.item(path, node),
            None => Err(AexError::NotFound(path.to_string())),
        }
    }

    fn list_children(&self, path: &str) -> Result<Vec<(String, Item)>> {
        let path = normalize_path(path);
        match self.node(path)? {
            Some(Node::Group(_)) => {}
            Some(Node::Array(_)) => return Err(AexError::NotAGroup(path.to_string())),
            None => return Err(AexError::NotFound(path.to_string())),
        }
        let Some(dir) = self.root.under(path)? else {
            return Err(AexError::NotFound(path.to_string()));
        };

        let mut children = Vec::new();
        for entry in std::fs::read_dir(&dir)? {
            let Ok(name) = entry?.file_name().into_string() else {
                continue;
            };
            let child = if path.is_empty() {
                name.clone()
            } else {
                format!("{path}/{name}")
            };
            // A child that cannot be served is left out rather than hiding its
            // siblings; asking for it directly still says why.
            match self.node(&child) {
                Ok(Some(node)) => {
                    if let Ok(item) = self.item(&child, node) {
                        children.push((name, item));
                    }
                }
                Ok(None) | Err(_) => continue,
            }
        }
        children.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(children)
    }

    fn attrs(&self, path: &str) -> Result<Vec<(String, AttrValue)>> {
        let path = normalize_path(path);
        let Some(node) = self.node(path)? else {
            return Err(AexError::NotFound(path.to_string()));
        };
        let mut attrs: Vec<(String, AttrValue)> = node
            .attributes()
            .iter()
            .filter_map(|(name, value)| Some((name.clone(), attr_value(value)?)))
            .collect();
        attrs.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(attrs)
    }
}

/// One array of a store.
#[derive(Debug)]
pub struct ZarrArray {
    root: Arc<StoreRoot>,
    /// This array's own path in the store; empty if the store is the array.
    prefix: String,
    dtype: DType,
    shape: Vec<u64>,
    key: KeyEncoding,
    store: Store,
    /// One element's worth of bytes, as stored.
    fill: Vec<u8>,
    cache: Arc<DecodeCache>,
    /// Tells this array's chunks apart from others' in the shared cache.
    cache_key: u64,
}

/// How an array's chunks are laid out in files.
#[derive(Debug)]
enum Store {
    /// One file per chunk, named after its coordinates.
    Chunked {
        grid: ChunkGrid,
        /// In the order they were applied when writing.
        codecs: Vec<ChunkCodec>,
    },
    Sharded(Box<Sharded>),
}

/// Many chunks to a file, with an index saying where each one is.
#[derive(Debug)]
struct Sharded {
    /// The array, in chunks of the shard shape.
    shards: ChunkGrid,
    /// One shard, in chunks of the inner chunk shape. Shards are all the same
    /// size, edge ones included, so one grid describes every shard.
    inner: ChunkGrid,
    /// The inner chunks' codecs, in the order they were applied.
    codecs: Vec<ChunkCodec>,
    /// Bytes the index occupies: sixteen per inner chunk, plus its checksum.
    index_bytes: u64,
    index_at_start: bool,
    index_crc32c: bool,
    /// By `shard * inner.chunks() + inner number`; built on first use.
    index: OnceLock<Vec<ShardEntry>>,
}

/// Where one inner chunk lies in its shard file.
#[derive(Debug, Clone, Copy)]
struct ShardEntry {
    offset: u64,
    nbytes: u64,
}

impl ShardEntry {
    /// What the index holds for a chunk that was never written.
    const MISSING: ShardEntry = ShardEntry {
        offset: u64::MAX,
        nbytes: u64::MAX,
    };

    fn is_missing(&self) -> bool {
        self.offset == u64::MAX && self.nbytes == u64::MAX
    }
}

/// A codec that turns a chunk's bytes into other bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChunkCodec {
    Gzip,
    Zstd,
    /// A checksum rather than a transformation: four little-endian bytes of
    /// CRC-32C at the end of what it wrapped.
    Crc32c,
}

/// How a chunk's coordinates become its file name.
#[derive(Debug)]
struct KeyEncoding {
    /// The `default` encoding prefixes `c`; the `v2` encoding does not.
    prefix: bool,
    separator: char,
}

impl ZarrArray {
    fn new(
        root: Arc<StoreRoot>,
        prefix: &str,
        meta: ArrayMeta,
        cache: Arc<DecodeCache>,
    ) -> Result<Self> {
        static NEXT_KEY: AtomicU64 = AtomicU64::new(0);

        let name = meta.data_type.as_str().ok_or_else(|| {
            AexError::UnsupportedZarr(format!("{prefix}: only the core data types are served"))
        })?;
        let dtype = dtype_of(name).ok_or_else(|| {
            AexError::UnsupportedDType(format!("{prefix}: Zarr data type {name:?}"))
        })?;

        if !meta.storage_transformers.is_empty() {
            return Err(AexError::UnsupportedZarr(format!(
                "{prefix} uses a storage transformer"
            )));
        }
        if meta.chunk_grid.name != "regular" {
            return Err(AexError::UnsupportedZarr(format!(
                "{prefix} has a {:?} chunk grid; only regular grids are served",
                meta.chunk_grid.name
            )));
        }
        let chunk_shape: Vec<u64> = serde_json::from_value(
            meta.chunk_grid
                .configuration
                .get("chunk_shape")
                .cloned()
                .unwrap_or(Value::Null),
        )
        .map_err(|e| AexError::MalformedZarr(format!("{prefix}: chunk_shape: {e}")))?;

        let grid =
            ChunkGrid::new(&meta.shape, &chunk_shape, dtype.itemsize()).ok_or_else(|| {
                AexError::MalformedZarr(format!(
                    "{prefix}: chunks of {chunk_shape:?} do not fit an array of {:?}",
                    meta.shape
                ))
            })?;

        // A sharded array names its shards with the outer grid, and the codec
        // says how a shard is divided further.
        let store = match sharding_of(&meta.codecs) {
            Some(named) => Store::Sharded(Box::new(Sharded::new(prefix, grid, dtype, named)?)),
            None => Store::Chunked {
                grid,
                codecs: parse_codecs(prefix, &meta.codecs)?,
            },
        };

        Ok(ZarrArray {
            root,
            prefix: prefix.to_string(),
            dtype,
            shape: meta.shape,
            key: key_encoding(prefix, &meta.chunk_key_encoding)?,
            store,
            fill: fill_bytes(dtype, &meta.fill_value)
                .map_err(|e| AexError::MalformedZarr(format!("{prefix}: {e}")))?,
            cache,
            cache_key: NEXT_KEY.fetch_add(1, Ordering::Relaxed),
        })
    }

    /// The store-relative name of the file holding `n` of `grid`.
    fn key_of(&self, grid: &ChunkGrid, n: u64, coords: &mut Vec<u64>) -> String {
        grid.coords_of(n, coords);
        let mut key = String::with_capacity(self.prefix.len() + 4 * coords.len() + 2);
        if !self.prefix.is_empty() {
            key.push_str(&self.prefix);
            key.push('/');
        }
        if self.key.prefix {
            key.push('c');
            for c in coords.iter() {
                key.push(self.key.separator);
                let _ = write!(key, "{c}");
            }
        } else if coords.is_empty() {
            // A scalar array has one chunk and no coordinates to name it with.
            key.push('0');
        } else {
            for (i, c) in coords.iter().enumerate() {
                if i > 0 {
                    key.push(self.key.separator);
                }
                let _ = write!(key, "{c}");
            }
        }
        key
    }

    /// Read `[at, at + dst.len())` of the array's C-order bytes.
    fn read_at(&self, at: u64, dst: &mut [u8]) -> Result<()> {
        match &self.store {
            Store::Chunked { grid, codecs } => {
                let mut coords = Vec::new();
                grid.walk(at, dst, |chunk, start, out| {
                    let key = self.key_of(grid, chunk as u64, &mut coords);
                    let decoded =
                        self.cache
                            .get_or_decode((self.cache_key, chunk as u64), || {
                                let Some(path) = self.root.under(&key)? else {
                                    return Ok(self.fill_chunk(grid.chunk_bytes()));
                                };
                                let bytes = std::fs::read(&path)?;
                                undo(&key, codecs, bytes, grid.chunk_bytes())
                            })?;
                    let start = start as usize;
                    out.copy_from_slice(&decoded[start..start + out.len()]);
                    Ok(())
                })
            }
            Store::Sharded(sharded) => {
                let index = sharded.index(self)?;
                let per_shard = sharded.inner.chunks();
                sharded.shards.walk(at, dst, |shard, in_shard, out| {
                    // The outer walk hands back an offset into an array of the
                    // shard shape, which is what the inner grid walks.
                    let key = self.key_of(&sharded.shards, shard as u64, &mut Vec::new());
                    sharded.inner.walk(in_shard, out, |chunk, start, out| {
                        let n = shard as u64 * per_shard + chunk as u64;
                        let decoded = self.cache.get_or_decode((self.cache_key, n), || {
                            let entry = index[n as usize];
                            if entry.is_missing() {
                                return Ok(self.fill_chunk(sharded.inner.chunk_bytes()));
                            }
                            let Some(path) = self.root.under(&key)? else {
                                return Ok(self.fill_chunk(sharded.inner.chunk_bytes()));
                            };
                            let mut bytes = vec![0u8; entry.nbytes as usize];
                            File::open(&path)?.read_exact_at(&mut bytes, entry.offset)?;
                            undo(&key, &sharded.codecs, bytes, sharded.inner.chunk_bytes())
                        })?;
                        let start = start as usize;
                        out.copy_from_slice(&decoded[start..start + out.len()]);
                        Ok(())
                    })
                })
            }
        }
    }

    /// A whole chunk of the fill value, for one that was never written.
    // ponytail: the fill chunk is cached like any other, so a sparse array can
    // evict real chunks. Keep one shared all-fill chunk if that shows up.
    fn fill_chunk(&self, chunk_bytes: u64) -> Vec<u8> {
        let mut chunk = vec![0u8; chunk_bytes as usize];
        fill_from(&self.fill, 0, &mut chunk);
        chunk
    }
}

/// Undo a chunk's codecs, which are listed in the order they were applied.
fn undo(key: &str, codecs: &[ChunkCodec], mut bytes: Vec<u8>, chunk_bytes: u64) -> Result<Vec<u8>> {
    let malformed = |what: String| AexError::MalformedZarr(format!("chunk {key}: {what}"));
    for codec in codecs.iter().rev() {
        bytes = match codec {
            ChunkCodec::Gzip => ungzip(&bytes, chunk_bytes).ok_or_else(|| {
                malformed("the gzip stream does not expand to a chunk".to_string())
            })?,
            ChunkCodec::Zstd => zstd::bulk::decompress(&bytes, chunk_bytes as usize)
                .map_err(|e| malformed(format!("zstd: {e}")))?,
            ChunkCodec::Crc32c => {
                strip_crc32c(bytes).ok_or_else(|| malformed("CRC-32C mismatch".to_string()))?
            }
        };
    }
    if bytes.len() as u64 != chunk_bytes {
        return Err(malformed(format!(
            "holds {} bytes instead of {chunk_bytes}",
            bytes.len()
        )));
    }
    Ok(bytes)
}

impl Sharded {
    fn new(prefix: &str, shards: ChunkGrid, dtype: DType, named: &Named) -> Result<Self> {
        let unsupported = |what: String| AexError::UnsupportedZarr(format!("{prefix}: {what}"));
        let config = &named.configuration;
        let chunk_shape: Vec<u64> =
            serde_json::from_value(config.get("chunk_shape").cloned().unwrap_or(Value::Null))
                .map_err(|e| {
                    AexError::MalformedZarr(format!("{prefix}: shard chunk_shape: {e}"))
                })?;
        let inner = ChunkGrid::new(shards.chunk_shape(), &chunk_shape, dtype.itemsize())
            .ok_or_else(|| {
                AexError::MalformedZarr(format!(
                    "{prefix}: inner chunks of {chunk_shape:?} do not fit a shard of {:?}",
                    shards.chunk_shape()
                ))
            })?;

        // A shard is divided evenly: one index entry per inner chunk, and no
        // padding between the last chunk of an axis and the shard's edge.
        if shards
            .chunk_shape()
            .iter()
            .zip(&chunk_shape)
            .any(|(shard, chunk)| shard % chunk != 0)
        {
            return Err(AexError::MalformedZarr(format!(
                "{prefix}: inner chunks of {chunk_shape:?} do not divide a shard of {:?}",
                shards.chunk_shape()
            )));
        }

        let codecs: Vec<Named> =
            serde_json::from_value(config.get("codecs").cloned().unwrap_or(Value::Null))
                .map_err(|e| AexError::MalformedZarr(format!("{prefix}: shard codecs: {e}")))?;
        // A sharding codec inside a shard lands here and is refused: the flat
        // cache key would stop telling two levels of inner chunk apart.
        let codecs = parse_codecs(prefix, &codecs)?;

        // The index has to sit at a known place, so only codecs that keep its
        // size are allowed there.
        let index_codecs: Vec<Named> =
            serde_json::from_value(config.get("index_codecs").cloned().unwrap_or(Value::Null))
                .map_err(|e| AexError::MalformedZarr(format!("{prefix}: index_codecs: {e}")))?;
        let mut index_crc32c = false;
        for codec in &index_codecs {
            match codec.name.as_str() {
                "bytes" => {}
                "crc32c" => index_crc32c = true,
                other => {
                    return Err(unsupported(format!(
                        "the shard index uses the {other:?} codec, which would move it"
                    )))
                }
            }
        }
        let index_at_start = match config.get("index_location") {
            None => false,
            Some(Value::String(at)) if at == "end" => false,
            Some(Value::String(at)) if at == "start" => true,
            Some(other) => {
                return Err(AexError::MalformedZarr(format!(
                    "{prefix}: index_location {other}"
                )))
            }
        };

        let entries = shards
            .chunks()
            .checked_mul(inner.chunks())
            .ok_or_else(|| AexError::MalformedZarr(format!("{prefix}: too many inner chunks")))?;
        if entries > MAX_CHUNKS {
            return Err(unsupported(format!(
                "{entries} inner chunks; at most {MAX_CHUNKS} are served"
            )));
        }
        let index_bytes = inner.chunks() * 16 + if index_crc32c { 4 } else { 0 };

        Ok(Sharded {
            shards,
            inner,
            codecs,
            index_bytes,
            index_at_start,
            index_crc32c,
            index: OnceLock::new(),
        })
    }

    /// Where every inner chunk lies, reading each shard's index the first time.
    fn index(&self, array: &ZarrArray) -> Result<&[ShardEntry]> {
        if let Some(index) = self.index.get() {
            return Ok(index);
        }
        let per_shard = self.inner.chunks() as usize;
        let mut index = vec![ShardEntry::MISSING; self.shards.chunks() as usize * per_shard];
        let mut coords = Vec::new();
        for shard in 0..self.shards.chunks() {
            let key = array.key_of(&self.shards, shard, &mut coords);
            let malformed = |what: String| AexError::MalformedZarr(format!("shard {key}: {what}"));
            // A shard nobody wrote leaves all of its chunks missing.
            let Some(path) = array.root.under(&key)? else {
                continue;
            };
            let file = File::open(&path)?;
            let len = file.metadata()?.len();
            if len < self.index_bytes {
                return Err(malformed(format!("{len} bytes cannot hold an index")));
            }
            let at = if self.index_at_start {
                0
            } else {
                len - self.index_bytes
            };
            let mut raw = vec![0u8; self.index_bytes as usize];
            file.read_exact_at(&mut raw, at)?;
            if self.index_crc32c {
                raw = strip_crc32c(raw)
                    .ok_or_else(|| malformed("index CRC-32C mismatch".to_string()))?;
            }
            for i in 0..per_shard {
                let pair = &raw[i * 16..(i + 1) * 16];
                let entry = ShardEntry {
                    offset: u64::from_le_bytes(pair[..8].try_into().expect("eight bytes")),
                    nbytes: u64::from_le_bytes(pair[8..].try_into().expect("eight bytes")),
                };
                if entry.is_missing() {
                    continue;
                }
                if entry
                    .offset
                    .checked_add(entry.nbytes)
                    .is_none_or(|end| end > len)
                {
                    return Err(malformed(format!(
                        "an inner chunk of {} bytes at {} does not fit {len} bytes",
                        entry.nbytes, entry.offset
                    )));
                }
                index[shard as usize * per_shard + i] = entry;
            }
        }
        Ok(self.index.get_or_init(|| index))
    }
}

impl ArrayDataset for ZarrArray {
    fn dtype(&self) -> DType {
        self.dtype
    }

    fn shape(&self) -> &[u64] {
        &self.shape
    }

    /// Builds the shard indexes here, on the control plane, so the data plane
    /// never waits on metadata and a broken index fails the request that met
    /// it rather than a transfer already under way.
    fn layout(&self, indices: &[Index], quality: &QualitySpec) -> Result<SelectionLayout> {
        if let Store::Sharded(sharded) = &self.store {
            sharded.index(self)?;
        }
        SelectionLayout::resolve(&self.shape, self.dtype, indices, quality)
    }

    fn read_range(&self, layout: &SelectionLayout, offset: u64, dst: &mut [u8]) -> Result<()> {
        layout.read_with(offset, dst, |at, buf| self.read_at(at, buf))
    }

    fn decoded_chunk_bytes(&self) -> Option<u64> {
        // Every chunk is cached whole, compressed or not, so the server should
        // size the cache for one per stream either way. A shard is deliberately
        // large and is never cached whole, so it is the inner chunk that counts.
        Some(match &self.store {
            Store::Chunked { grid, .. } => grid.chunk_bytes(),
            Store::Sharded(sharded) => sharded.inner.chunk_bytes(),
        })
    }
}

/// A node's metadata, as far as this backend reads it.
#[derive(Debug, Deserialize)]
#[serde(tag = "node_type", rename_all = "lowercase")]
enum Node {
    Group(GroupMeta),
    Array(Box<ArrayMeta>),
}

impl Node {
    fn attributes(&self) -> &Map<String, Value> {
        match self {
            Node::Group(meta) => &meta.attributes,
            Node::Array(meta) => &meta.attributes,
        }
    }

    /// The parts that are the same for both kinds of node.
    fn check(&self, key: &str) -> Result<()> {
        let format = match self {
            Node::Group(meta) => meta.zarr_format,
            Node::Array(meta) => meta.zarr_format,
        };
        if format != 3 {
            return Err(AexError::UnsupportedZarr(format!(
                "{key} is Zarr version {format}; only version 3 is served"
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
struct GroupMeta {
    zarr_format: u32,
    #[serde(default)]
    attributes: Map<String, Value>,
}

#[derive(Debug, Deserialize)]
struct ArrayMeta {
    zarr_format: u32,
    shape: Vec<u64>,
    /// A string for a core type; an object for an extension, which is refused.
    data_type: Value,
    chunk_grid: Named,
    chunk_key_encoding: Option<Named>,
    #[serde(default)]
    fill_value: Value,
    #[serde(default)]
    codecs: Vec<Named>,
    #[serde(default)]
    storage_transformers: Vec<Value>,
    #[serde(default)]
    attributes: Map<String, Value>,
}

/// The shape every codec, grid and key encoding is written in.
#[derive(Debug, Deserialize)]
struct Named {
    name: String,
    #[serde(default)]
    configuration: Map<String, Value>,
}

/// One attribute as the wire carries it, or `None` if it has no wire form.
///
/// JSON says nothing about element types, so the only honest reading is the
/// one numpy would give: integers stay integers, anything else numeric becomes
/// a double. What cannot be carried at all is left out, the way an unservable
/// child is left out of a listing.
fn attr_value(value: &Value) -> Option<AttrValue> {
    if let Value::String(text) = value {
        return (text.len() as u64 <= MAX_ATTR_BYTES).then(|| AttrValue::Text(text.clone()));
    }

    let mut shape = Vec::new();
    let mut leaves = Vec::new();
    let mut leaf_depth = None;
    if !flatten(value, 0, &mut shape, &mut leaves, &mut leaf_depth) {
        return None;
    }
    // A leaf below the deepest axis means the array was ragged.
    if leaf_depth.is_some_and(|depth| depth != shape.len()) {
        return None;
    }

    let dtype = if leaves.iter().all(|v| v.is_boolean()) && !leaves.is_empty() {
        DType::Bool
    } else if leaves.iter().all(|v| v.as_i64().is_some()) && !leaves.is_empty() {
        DType::Int64
    } else if leaves.iter().all(|v| v.is_number()) {
        // An empty array lands here, and numpy calls that one double too.
        DType::Float64
    } else {
        return None;
    };
    if leaves.len() as u64 * dtype.itemsize() > MAX_ATTR_BYTES {
        return None;
    }

    let mut data = Vec::with_capacity(leaves.len() * dtype.itemsize() as usize);
    for leaf in leaves {
        match dtype {
            DType::Bool => data.push(u8::from(leaf.as_bool()?)),
            DType::Int64 => data.extend(leaf.as_i64()?.to_le_bytes()),
            _ => data.extend(leaf.as_f64()?.to_le_bytes()),
        }
    }
    Some(AttrValue::Array { dtype, shape, data })
}

/// The shape and leaves of a rectangular JSON array, or `false` if it is not
/// one. A number or a boolean on its own is a scalar: no axes, one leaf.
fn flatten<'a>(
    value: &'a Value,
    depth: usize,
    shape: &mut Vec<u64>,
    leaves: &mut Vec<&'a Value>,
    leaf_depth: &mut Option<usize>,
) -> bool {
    match value {
        Value::Array(items) => {
            if depth == shape.len() {
                shape.push(items.len() as u64);
            } else if shape[depth] != items.len() as u64 {
                return false;
            }
            items
                .iter()
                .all(|item| flatten(item, depth + 1, shape, leaves, leaf_depth))
        }
        Value::Bool(_) | Value::Number(_) => {
            if *leaf_depth.get_or_insert(depth) != depth {
                return false;
            }
            leaves.push(value);
            true
        }
        _ => false,
    }
}

/// The AEX type of a Zarr core data type name.
fn dtype_of(name: &str) -> Option<DType> {
    Some(match name {
        "bool" => DType::Bool,
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
        _ => return None,
    })
}

fn key_encoding(prefix: &str, named: &Option<Named>) -> Result<KeyEncoding> {
    let Some(named) = named else {
        return Ok(KeyEncoding {
            prefix: true,
            separator: '/',
        });
    };
    let (with_prefix, default) = match named.name.as_str() {
        "default" => (true, '/'),
        "v2" => (false, '.'),
        other => {
            return Err(AexError::UnsupportedZarr(format!(
                "{prefix} names its chunks with the {other:?} encoding"
            )))
        }
    };
    let separator = match named.configuration.get("separator") {
        None => default,
        Some(Value::String(s)) if s == "/" => '/',
        Some(Value::String(s)) if s == "." => '.',
        Some(other) => {
            return Err(AexError::MalformedZarr(format!(
                "{prefix}: chunk key separator {other}"
            )))
        }
    };
    Ok(KeyEncoding {
        prefix: with_prefix,
        separator,
    })
}

/// The sharding codec of a chain, if that is what the chain is.
///
/// It has to be the whole chain: anything wrapping it would have to be undone
/// before the index could be found.
fn sharding_of(codecs: &[Named]) -> Option<&Named> {
    match codecs {
        [only] if only.name == "sharding_indexed" => Some(only),
        _ => None,
    }
}

/// The bytes-to-bytes codecs of a chain, refusing one this build cannot undo.
///
/// `bytes` and an identity `transpose` change nothing about the stored bytes,
/// so they are checked and then forgotten.
fn parse_codecs(prefix: &str, codecs: &[Named]) -> Result<Vec<ChunkCodec>> {
    let mut chain = Vec::new();
    for codec in codecs {
        match codec.name.as_str() {
            // Every target is little-endian and so is the wire, so big-endian
            // data would have to be swapped on the way out. Refuse it instead,
            // as the other backends do.
            "bytes" => match codec.configuration.get("endian") {
                None => {}
                Some(Value::String(s)) if s == "little" => {}
                Some(other) => {
                    return Err(AexError::UnsupportedZarr(format!(
                        "{prefix} stores {other} elements; only little-endian data is served"
                    )))
                }
            },
            "transpose" => {
                let order: Vec<u64> = serde_json::from_value(
                    codec
                        .configuration
                        .get("order")
                        .cloned()
                        .unwrap_or(Value::Null),
                )
                .map_err(|e| AexError::MalformedZarr(format!("{prefix}: transpose: {e}")))?;
                if order.iter().enumerate().any(|(i, &o)| o != i as u64) {
                    return Err(AexError::UnsupportedZarr(format!(
                        "{prefix} is stored transposed as {order:?}; only C order is served"
                    )));
                }
            }
            "gzip" => chain.push(ChunkCodec::Gzip),
            "zstd" => chain.push(ChunkCodec::Zstd),
            "crc32c" => chain.push(ChunkCodec::Crc32c),
            other => {
                return Err(AexError::UnsupportedZarr(format!(
                    "{prefix} uses the {other:?} codec, which is not decoded"
                )))
            }
        }
    }
    Ok(chain)
}

/// Undo gzip, refusing to produce more than `limit` bytes.
fn ungzip(bytes: &[u8], limit: u64) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(limit as usize);
    flate2::read::GzDecoder::new(bytes)
        .take(limit + 1)
        .read_to_end(&mut out)
        .ok()?;
    Some(out)
}

/// Drop the trailing CRC-32C, if it is the one the rest of the bytes have.
fn strip_crc32c(mut bytes: Vec<u8>) -> Option<Vec<u8>> {
    let at = bytes.len().checked_sub(4)?;
    let stored = u32::from_le_bytes(bytes[at..].try_into().expect("four bytes"));
    bytes.truncate(at);
    (crc32c::crc32c(&bytes) == stored).then_some(bytes)
}

/// One element of `dtype` holding the JSON fill value, as it would be stored.
///
/// A fill value that cannot be read as the array's type is an error rather
/// than a zero: the wrong one is silent corruption of every unwritten chunk.
fn fill_bytes(dtype: DType, value: &Value) -> Result<Vec<u8>> {
    let malformed = || AexError::MalformedZarr(format!("fill_value {value} is not a {dtype}"));
    if value.is_null() {
        return Ok(vec![0u8; dtype.itemsize() as usize]);
    }
    match dtype {
        DType::Bool => match value.as_bool() {
            Some(set) => Ok(vec![u8::from(set)]),
            None => Err(malformed()),
        },
        DType::Complex64 | DType::Complex128 => {
            let parts = value
                .as_array()
                .filter(|parts| parts.len() == 2)
                .ok_or_else(malformed)?;
            let part = if dtype == DType::Complex64 {
                DType::Float32
            } else {
                DType::Float64
            };
            let mut out = fill_bytes(part, &parts[0])?;
            out.extend(fill_bytes(part, &parts[1])?);
            Ok(out)
        }
        DType::Float16 | DType::Float32 | DType::Float64 => {
            let size = dtype.itemsize() as usize;
            // The raw bits, written big-endian the way the spec spells them.
            if let Some(hex) = value.as_str().and_then(|s| s.strip_prefix("0x")) {
                if hex.len() != size * 2 {
                    return Err(malformed());
                }
                let bits = u128::from_str_radix(hex, 16).map_err(|_| malformed())?;
                return Ok(bits.to_le_bytes()[..size].to_vec());
            }
            let number = json_float(value).ok_or_else(malformed)?;
            Ok(match dtype {
                DType::Float16 => half::f16::from_f64(number).to_le_bytes().to_vec(),
                DType::Float32 => (number as f32).to_le_bytes().to_vec(),
                _ => number.to_le_bytes().to_vec(),
            })
        }
        _ => {
            let number = value
                .as_u64()
                .map(i128::from)
                .or_else(|| value.as_i64().map(i128::from))
                .ok_or_else(malformed)?;
            int_bytes(dtype, number).ok_or_else(malformed)
        }
    }
}

/// A JSON number, or one of the three floats JSON cannot write as one.
fn json_float(value: &Value) -> Option<f64> {
    match value {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => match s.as_str() {
            "NaN" => Some(f64::NAN),
            "Infinity" => Some(f64::INFINITY),
            "-Infinity" => Some(f64::NEG_INFINITY),
            _ => None,
        },
        _ => None,
    }
}

/// `number` as `dtype`, or `None` if it does not fit.
fn int_bytes(dtype: DType, number: i128) -> Option<Vec<u8>> {
    Some(match dtype {
        DType::Int8 => i8::try_from(number).ok()?.to_le_bytes().to_vec(),
        DType::Int16 => i16::try_from(number).ok()?.to_le_bytes().to_vec(),
        DType::Int32 => i32::try_from(number).ok()?.to_le_bytes().to_vec(),
        DType::Int64 => i64::try_from(number).ok()?.to_le_bytes().to_vec(),
        DType::Uint8 => u8::try_from(number).ok()?.to_le_bytes().to_vec(),
        DType::Uint16 => u16::try_from(number).ok()?.to_le_bytes().to_vec(),
        DType::Uint32 => u32::try_from(number).ok()?.to_le_bytes().to_vec(),
        DType::Uint64 => u64::try_from(number).ok()?.to_le_bytes().to_vec(),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use std::io::Write as _;

    use super::*;
    use crate::error::ErrorClass;
    use crate::quality::QualitySpec;
    use crate::selection::Index;

    /// A store written as raw text.
    ///
    /// The metadata is not serialized from a struct on purpose: a store this
    /// backend must refuse is exactly what a serializer would refuse to write,
    /// and those are the paths worth testing.
    struct StoreBuilder {
        dir: tempfile::TempDir,
    }

    impl StoreBuilder {
        fn new() -> Self {
            let store = StoreBuilder {
                dir: tempfile::tempdir().expect("tempdir"),
            };
            store.node("", r#"{"zarr_format": 3, "node_type": "group"}"#);
            store
        }

        fn path(&self) -> &Path {
            self.dir.path()
        }

        fn write(&self, relative: &str, bytes: &[u8]) {
            let path = self.dir.path().join(relative);
            std::fs::create_dir_all(path.parent().expect("a name has a parent")).expect("mkdir");
            let mut file = std::fs::File::create(&path).expect("create");
            file.write_all(bytes).expect("write");
        }

        fn node(&self, at: &str, json: &str) {
            let key = if at.is_empty() {
                METADATA.to_string()
            } else {
                format!("{at}/{METADATA}")
            };
            self.write(&key, json.as_bytes());
        }

        /// An array whose metadata is the usual one, with `overrides` applied.
        fn array(&self, at: &str, dtype: &str, shape: &[u64], chunk: &[u64], overrides: &str) {
            let fill = if dtype.starts_with("complex") {
                "[0.0, 0.0]"
            } else if dtype == "bool" {
                "false"
            } else {
                "0"
            };
            let mut meta: Map<String, Value> = serde_json::from_str(&format!(
                r#"{{"zarr_format": 3, "node_type": "array",
                     "data_type": "{dtype}", "shape": {shape:?},
                     "chunk_grid": {{"name": "regular",
                                     "configuration": {{"chunk_shape": {chunk:?}}}}},
                     "chunk_key_encoding": {{"name": "default"}},
                     "fill_value": {fill},
                     "codecs": [{{"name": "bytes",
                                  "configuration": {{"endian": "little"}}}}]}}"#
            ))
            .expect("metadata");
            if !overrides.is_empty() {
                let overrides: Map<String, Value> =
                    serde_json::from_str(overrides).expect("overrides");
                meta.extend(overrides);
            }
            self.node(at, &Value::Object(meta).to_string());
        }

        fn open(&self) -> Result<ZarrFile> {
            ZarrFile::open(self.dir.path(), Arc::new(DecodeCache::new(1 << 20)))
        }
    }

    /// Write `values` as `shape` in chunks of `chunk`, and return the array's
    /// own C-order bytes.
    fn write_chunks<T: Copy>(
        store: &StoreBuilder,
        at: &str,
        shape: &[u64],
        chunk: &[u64],
        values: &[T],
    ) -> Vec<u8> {
        let itemsize = std::mem::size_of::<T>();
        let flat = bytes_of(values);
        let grid: Vec<u64> = shape
            .iter()
            .zip(chunk)
            .map(|(n, c)| n.div_ceil(*c))
            .collect();
        let per_chunk: usize = chunk.iter().product::<u64>() as usize;
        let chunks: usize = grid.iter().product::<u64>() as usize;
        let mut bytes = vec![vec![0u8; per_chunk * itemsize]; chunks];
        let ndim = shape.len();
        for i in 0..values.len() {
            let mut rest = i as u64;
            let mut pos = vec![0u64; ndim];
            for axis in (0..ndim).rev() {
                pos[axis] = rest % shape[axis];
                rest /= shape[axis];
            }
            let (mut c, mut within) = (0u64, 0u64);
            for axis in 0..ndim {
                c = c * grid[axis] + pos[axis] / chunk[axis];
                within = within * chunk[axis] + pos[axis] % chunk[axis];
            }
            let at = within as usize * itemsize;
            bytes[c as usize][at..at + itemsize]
                .copy_from_slice(&flat[i * itemsize..(i + 1) * itemsize]);
        }
        for (c, chunk_bytes) in bytes.iter().enumerate() {
            let mut coords = Vec::new();
            let mut rest = c as u64;
            for &d in grid.iter().rev() {
                coords.push(rest % d);
                rest /= d;
            }
            coords.reverse();
            let name: Vec<String> = coords.iter().map(|c| c.to_string()).collect();
            let key = if at.is_empty() {
                format!("c/{}", name.join("/"))
            } else {
                format!("{at}/c/{}", name.join("/"))
            };
            store.write(&key, chunk_bytes);
        }
        flat
    }

    fn bytes_of<T: Copy>(values: &[T]) -> Vec<u8> {
        let len = std::mem::size_of_val(values);
        // SAFETY: plain numeric types, read as bytes.
        unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), len) }.to_vec()
    }

    fn array_at(file: &ZarrFile, path: &str) -> Arc<dyn ArrayDataset> {
        match file.get_item(path).expect("item") {
            Item::Dataset(array) => array,
            Item::Group => panic!("{path} is a group"),
        }
    }

    fn read_all(array: &dyn ArrayDataset, indices: &[Index]) -> Vec<u8> {
        let layout = array
            .layout(indices, &QualitySpec::default())
            .expect("layout");
        let mut out = vec![0u8; layout.total_bytes as usize];
        array.read_range(&layout, 0, &mut out).expect("read");
        out
    }

    /// Read the whole stream in pieces of several sizes, so a run that starts
    /// inside a chunk or inside an element is exercised too.
    fn check_ranges(array: &dyn ArrayDataset, expected: &[u8]) {
        let layout = array.layout(&[], &QualitySpec::default()).expect("layout");
        for piece in [1usize, 3, 7, 64, expected.len().max(1)] {
            let mut out = vec![0u8; expected.len()];
            let mut at = 0;
            while at < out.len() {
                let len = piece.min(out.len() - at);
                array
                    .read_range(&layout, at as u64, &mut out[at..at + len])
                    .expect("read");
                at += len;
            }
            assert_eq!(out, expected, "{piece} bytes at a time");
        }
    }

    #[test]
    fn every_dtype_reads_back_as_written() {
        let store = StoreBuilder::new();
        let n = 24usize;
        let mut expected = Vec::new();
        macro_rules! case {
            ($name:literal, $t:ty, $dtype:expr, $f:expr) => {{
                let values: Vec<$t> = (0..n).map($f).collect();
                store.array($name, $name, &[4, 6], &[3, 4], "");
                let bytes = write_chunks(&store, $name, &[4, 6], &[3, 4], &values);
                expected.push(($name, $dtype, bytes));
            }};
        }
        case!("int8", i8, DType::Int8, |i| i as i8 - 12);
        case!("int16", i16, DType::Int16, |i| i as i16 * -300);
        case!("int32", i32, DType::Int32, |i| i as i32 * -70_000);
        case!("int64", i64, DType::Int64, |i| i as i64 * -5_000_000_000);
        case!("uint8", u8, DType::Uint8, |i| i as u8);
        case!("uint16", u16, DType::Uint16, |i| i as u16 * 300);
        case!("uint32", u32, DType::Uint32, |i| i as u32 * 70_000);
        case!("uint64", u64, DType::Uint64, |i| i as u64 * 5_000_000_000);
        case!("float16", half::f16, DType::Float16, |i| {
            half::f16::from_f32(i as f32 / 4.0)
        });
        case!("float32", f32, DType::Float32, |i| i as f32 / 4.0);
        case!("float64", f64, DType::Float64, |i| i as f64 / 4.0);
        case!("complex64", [f32; 2], DType::Complex64, |i| [
            i as f32,
            -(i as f32)
        ]);
        case!("complex128", [f64; 2], DType::Complex128, |i| [
            i as f64,
            -(i as f64)
        ]);
        case!("bool", u8, DType::Bool, |i| u8::from(i % 3 == 0));

        let file = store.open().expect("open");
        for (name, dtype, bytes) in &expected {
            let array = array_at(&file, name);
            assert_eq!(array.dtype(), *dtype, "{name}");
            assert_eq!(array.shape(), [4, 6], "{name}");
            assert_eq!(&read_all(array.as_ref(), &[]), bytes, "{name}");
            check_ranges(array.as_ref(), bytes);
        }
    }

    #[test]
    fn edge_chunks_are_padded() {
        // None of these shapes is a multiple of its chunk shape, so every
        // chunk on an edge carries bytes the array does not own.
        let cases: [(&[u64], &[u64]); 4] = [
            (&[8], &[3]),
            (&[5, 7], &[2, 7]),
            (&[5, 7], &[2, 3]),
            (&[4, 3, 5], &[3, 2, 2]),
        ];
        for (shape, chunk) in cases {
            let store = StoreBuilder::new();
            let n: u64 = shape.iter().product();
            let values: Vec<i32> = (0..n as i32).collect();
            store.array("a", "int32", shape, chunk, "");
            let bytes = write_chunks(&store, "a", shape, chunk, &values);
            let file = store.open().expect("open");
            let array = array_at(&file, "a");
            check_ranges(array.as_ref(), &bytes);
        }
    }

    #[test]
    fn selections_and_odd_offsets_match_the_source() {
        let store = StoreBuilder::new();
        let (shape, chunk) = ([6u64, 7], [4u64, 3]);
        let values: Vec<i32> = (0..42).collect();
        store.array("a", "int32", &shape, &chunk, "");
        write_chunks(&store, "a", &shape, &chunk, &values);
        let file = store.open().expect("open");
        let array = array_at(&file, "a");

        let take = |rows: &[usize], cols: &[usize]| -> Vec<u8> {
            let mut out = Vec::new();
            for r in rows {
                for c in cols {
                    out.extend(values[r * 7 + c].to_le_bytes());
                }
            }
            out
        };

        // A stride that lands in every chunk of a row of the grid.
        assert_eq!(
            read_all(
                array.as_ref(),
                &[
                    Index::full(),
                    Index::Slice {
                        start: Some(1),
                        stop: Some(7),
                        step: Some(2),
                    },
                ],
            ),
            take(&(0..6).collect::<Vec<_>>(), &[1, 3, 5])
        );
        // Rows picked out of order, so the runs are not in source order.
        assert_eq!(
            read_all(array.as_ref(), &[Index::Fancy(vec![5, 0, 4])]),
            take(&[5, 0, 4], &(0..7).collect::<Vec<_>>())
        );
        // One element, deep inside a chunk.
        assert_eq!(
            read_all(array.as_ref(), &[Index::Single(3), Index::Single(5)]),
            take(&[3], &[5])
        );
    }

    #[test]
    fn missing_chunks_read_as_the_fill_value() {
        let store = StoreBuilder::new();
        store.node(
            "a",
            r#"{"zarr_format": 3, "node_type": "array", "data_type": "int32",
                "shape": [4, 4],
                "chunk_grid": {"name": "regular",
                               "configuration": {"chunk_shape": [2, 2]}},
                "chunk_key_encoding": {"name": "default"},
                "fill_value": -7,
                "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}]}"#,
        );
        // Only the first chunk is written; the other three are absent.
        let written: Vec<i32> = vec![1, 2, 3, 4];
        store.write("a/c/0/0", &bytes_of(&written));

        let file = store.open().expect("open");
        let array = array_at(&file, "a");
        let mut expected = Vec::new();
        for row in 0..4 {
            for col in 0..4 {
                let v = if row < 2 && col < 2 {
                    written[row * 2 + col]
                } else {
                    -7
                };
                expected.extend(v.to_le_bytes());
            }
        }
        assert_eq!(read_all(array.as_ref(), &[]), expected);
        check_ranges(array.as_ref(), &expected);
    }

    #[test]
    fn fill_values_parse_in_every_json_form() {
        let cases: [(DType, &str, Vec<u8>); 12] = [
            (DType::Bool, "true", vec![1]),
            (DType::Bool, "false", vec![0]),
            (DType::Int32, "-7", (-7i32).to_le_bytes().to_vec()),
            (DType::Uint64, "18446744073709551615", vec![0xff; 8]),
            (DType::Float32, "1.5", 1.5f32.to_le_bytes().to_vec()),
            (
                DType::Float64,
                "\"Infinity\"",
                f64::INFINITY.to_le_bytes().to_vec(),
            ),
            (
                DType::Float64,
                "\"-Infinity\"",
                f64::NEG_INFINITY.to_le_bytes().to_vec(),
            ),
            (
                DType::Float32,
                "\"0x7fc00000\"",
                0x7fc0_0000u32.to_le_bytes().to_vec(),
            ),
            (
                DType::Float16,
                "\"0x3c00\"",
                half::f16::ONE.to_le_bytes().to_vec(),
            ),
            (
                DType::Complex64,
                "[1.0, -2.0]",
                [1.0f32.to_le_bytes(), (-2.0f32).to_le_bytes()].concat(),
            ),
            // Absent, and so zero: the same rule HDF5 follows.
            (DType::Int16, "null", vec![0, 0]),
            (DType::Float64, "\"NaN\"", f64::NAN.to_le_bytes().to_vec()),
        ];
        for (dtype, json, expected) in cases {
            let value: Value = serde_json::from_str(json).expect("json");
            assert_eq!(fill_bytes(dtype, &value).expect(json), expected, "{json}");
        }

        // A value that is not the array's type is an error, never a zero.
        for (dtype, json) in [
            (DType::Int8, "300"),
            (DType::Uint8, "-1"),
            (DType::Bool, "1"),
            (DType::Float32, "\"0x7fc0\""),
            (DType::Float32, "\"nan\""),
            (DType::Complex64, "[1.0]"),
            (DType::Int32, "\"7\""),
        ] {
            let value: Value = serde_json::from_str(json).expect("json");
            let err = fill_bytes(dtype, &value).expect_err(json);
            assert!(matches!(err, AexError::MalformedZarr(_)), "{json}: {err}");
        }
    }

    #[test]
    fn chunk_keys_follow_the_encoding() {
        let cases: [(&str, &[u64], &str, &str); 5] = [
            (r#"{"name": "default"}"#, &[2, 3], "", "c/1/2"),
            (
                r#"{"name": "default", "configuration": {"separator": "."}}"#,
                &[2, 3],
                "",
                "c.1.2",
            ),
            (r#"{"name": "v2"}"#, &[2, 3], "", "1.2"),
            (
                r#"{"name": "v2", "configuration": {"separator": "/"}}"#,
                &[2, 3],
                "",
                "1/2",
            ),
            // A scalar has one chunk and no coordinates to name it with.
            (r#"{"name": "v2"}"#, &[], "g", "g/0"),
        ];
        for (encoding, shape, prefix, expected) in cases {
            let chunk: Vec<u64> = shape.iter().map(|_| 1).collect();
            let grid = ChunkGrid::new(shape, &chunk, 4).expect("grid");
            let last = grid.chunks() - 1;
            let store = Store::Chunked {
                grid,
                codecs: Vec::new(),
            };
            let array = ZarrArray {
                root: Arc::new(StoreRoot {
                    path: PathBuf::from("/"),
                }),
                prefix: prefix.to_string(),
                dtype: DType::Int32,
                shape: shape.to_vec(),
                key: key_encoding(
                    "a",
                    &Some(serde_json::from_str(encoding).expect("encoding")),
                )
                .expect("encoding"),
                store,
                fill: vec![0; 4],
                cache: Arc::new(DecodeCache::new(0)),
                cache_key: 0,
            };
            let Store::Chunked { grid, .. } = &array.store else {
                unreachable!("built as a chunked store");
            };
            assert_eq!(array.key_of(grid, last, &mut Vec::new()), expected);
        }
    }

    #[test]
    fn scalar_and_empty_arrays() {
        let store = StoreBuilder::new();
        store.node(
            "scalar",
            r#"{"zarr_format": 3, "node_type": "array", "data_type": "int32",
                "shape": [],
                "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": []}},
                "chunk_key_encoding": {"name": "default"},
                "fill_value": 5,
                "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}]}"#,
        );
        store.write("scalar/c", &42i32.to_le_bytes());
        store.array("empty", "int32", &[0, 4], &[2, 2], "");
        store.array("unwritten", "int32", &[], &[], "");

        let file = store.open().expect("open");
        let scalar = array_at(&file, "scalar");
        assert_eq!(scalar.shape(), [0u64; 0]);
        assert_eq!(read_all(scalar.as_ref(), &[]), 42i32.to_le_bytes());
        check_ranges(scalar.as_ref(), &42i32.to_le_bytes());

        let empty = array_at(&file, "empty");
        assert_eq!(read_all(empty.as_ref(), &[]), Vec::<u8>::new());

        // A scalar with no chunk file reads as its fill value.
        let unwritten = array_at(&file, "unwritten");
        assert_eq!(read_all(unwritten.as_ref(), &[]), 0i32.to_le_bytes());
    }

    #[test]
    fn unservable_arrays_are_rejected() {
        // A store AEX cannot serve is the requester's problem; a store that
        // disagrees with itself will not serve however often it is asked.
        let cases: [(&str, ErrorClass, &str); 14] = [
            ("version 2", ErrorClass::Request, r#"{"zarr_format": 2}"#),
            ("version 4", ErrorClass::Request, r#"{"zarr_format": 4}"#),
            (
                "unknown data type",
                ErrorClass::Request,
                r#"{"data_type": "string"}"#,
            ),
            (
                "an extension data type",
                ErrorClass::Request,
                r#"{"data_type": {"name": "r16"}}"#,
            ),
            (
                "an irregular grid",
                ErrorClass::Request,
                r#"{"chunk_grid": {"name": "rectangular"}}"#,
            ),
            (
                "a rank mismatch",
                ErrorClass::Permanent,
                r#"{"chunk_grid": {"name": "regular",
                                   "configuration": {"chunk_shape": [2]}}}"#,
            ),
            (
                "a zero chunk axis",
                ErrorClass::Permanent,
                r#"{"chunk_grid": {"name": "regular",
                                   "configuration": {"chunk_shape": [2, 0]}}}"#,
            ),
            (
                "big-endian data",
                ErrorClass::Request,
                r#"{"codecs": [{"name": "bytes", "configuration": {"endian": "big"}}]}"#,
            ),
            (
                "a transposed array",
                ErrorClass::Request,
                r#"{"codecs": [{"name": "transpose", "configuration": {"order": [1, 0]}},
                               {"name": "bytes"}]}"#,
            ),
            (
                "an undecoded codec",
                ErrorClass::Request,
                r#"{"codecs": [{"name": "bytes"}, {"name": "blosc"}]}"#,
            ),
            (
                "a shard inside a shard",
                ErrorClass::Request,
                r#"{"codecs": [{"name": "sharding_indexed", "configuration": {
                        "chunk_shape": [2, 2],
                        "codecs": [{"name": "sharding_indexed"}],
                        "index_codecs": [{"name": "bytes"}]}}]}"#,
            ),
            (
                "a shard index that moves",
                ErrorClass::Request,
                r#"{"codecs": [{"name": "sharding_indexed", "configuration": {
                        "chunk_shape": [2, 2],
                        "codecs": [{"name": "bytes"}],
                        "index_codecs": [{"name": "bytes"}, {"name": "gzip"}]}}]}"#,
            ),
            (
                "inner chunks that do not fit a shard",
                ErrorClass::Permanent,
                r#"{"codecs": [{"name": "sharding_indexed", "configuration": {
                        "chunk_shape": [3, 3],
                        "codecs": [{"name": "bytes"}],
                        "index_codecs": [{"name": "bytes"}]}}]}"#,
            ),
            (
                "a storage transformer",
                ErrorClass::Request,
                r#"{"storage_transformers": [{}]}"#,
            ),
        ];
        for (what, class, overrides) in cases {
            let store = StoreBuilder::new();
            store.array("a", "int32", &[4, 4], &[2, 2], overrides);
            let file = store.open().expect("open");
            let err = file.get_item("a").expect_err(what);
            assert_eq!(err.class(), class, "{what}: {err}");
        }
    }

    #[test]
    fn a_chunk_of_the_wrong_size_is_reported() {
        let store = StoreBuilder::new();
        store.array("a", "int32", &[4], &[2], "");
        store.write("a/c/0", &[1u8, 2, 3]);
        store.write("a/c/1", &bytes_of(&[3i32, 4]));

        let file = store.open().expect("open");
        let array = array_at(&file, "a");
        let layout = array.layout(&[], &QualitySpec::default()).expect("layout");
        let mut out = vec![0u8; layout.total_bytes as usize];
        let err = array.read_range(&layout, 0, &mut out).expect_err("short");
        assert!(matches!(err, AexError::MalformedZarr(_)), "{err}");
        assert_eq!(err.class(), ErrorClass::Permanent);

        // The intact chunk beside it still reads.
        let mut tail = vec![0u8; 8];
        array.read_range(&layout, 8, &mut tail).expect("read");
        assert_eq!(tail, bytes_of(&[3i32, 4]));
    }

    /// `bytes` through the codec chain, the way a writer would apply it.
    fn encode(chain: &[ChunkCodec], mut bytes: Vec<u8>) -> Vec<u8> {
        use std::io::Write as _;
        for codec in chain {
            bytes = match codec {
                ChunkCodec::Gzip => {
                    let mut out =
                        flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
                    out.write_all(&bytes).expect("gzip");
                    out.finish().expect("gzip")
                }
                ChunkCodec::Zstd => zstd::bulk::compress(&bytes, 3).expect("zstd"),
                ChunkCodec::Crc32c => {
                    let sum = crc32c::crc32c(&bytes);
                    bytes.extend(sum.to_le_bytes());
                    bytes
                }
            };
        }
        bytes
    }

    #[test]
    fn compressed_chunks_read_back_under_every_chain() {
        let chains: [&[ChunkCodec]; 6] = [
            &[],
            &[ChunkCodec::Gzip],
            &[ChunkCodec::Zstd],
            &[ChunkCodec::Crc32c],
            &[ChunkCodec::Zstd, ChunkCodec::Crc32c],
            &[ChunkCodec::Gzip, ChunkCodec::Crc32c],
        ];
        for chain in chains {
            let names: Vec<String> = chain
                .iter()
                .map(|c| match c {
                    ChunkCodec::Gzip => r#"{"name": "gzip"}"#.to_string(),
                    ChunkCodec::Zstd => r#"{"name": "zstd"}"#.to_string(),
                    ChunkCodec::Crc32c => r#"{"name": "crc32c"}"#.to_string(),
                })
                .collect();
            let store = StoreBuilder::new();
            // A shape the chunks do not cover evenly, so edge chunks are
            // padded before they are compressed.
            let (shape, chunk) = ([5u64, 7], [2u64, 3]);
            store.array(
                "a",
                "int32",
                &shape,
                &chunk,
                &format!(
                    r#"{{"codecs": [{{"name": "bytes",
                                      "configuration": {{"endian": "little"}}}}{}]}}"#,
                    names.iter().map(|n| format!(", {n}")).collect::<String>()
                ),
            );
            let values: Vec<i32> = (0..35).collect();
            let flat = write_chunks(&store, "a", &shape, &chunk, &values);
            // Rewrite each chunk through the chain.
            for row in 0..3 {
                for col in 0..3 {
                    let key = format!("a/c/{row}/{col}");
                    let raw = std::fs::read(store.path().join(&key)).expect("read");
                    store.write(&key, &encode(chain, raw));
                }
            }

            let file = store.open().expect("open");
            let array = array_at(&file, "a");
            assert_eq!(read_all(array.as_ref(), &[]), flat, "{names:?}");
            check_ranges(array.as_ref(), &flat);
            assert_eq!(array.decoded_chunk_bytes(), Some(2 * 3 * 4));
        }
    }

    #[test]
    fn a_corrupt_chunk_is_reported() {
        // Each case damages the one chunk that holds the whole array.
        let cases: [(&str, Vec<u8>); 5] = [
            (r#"{"name": "gzip"}"#, b"not gzip at all".to_vec()),
            (r#"{"name": "zstd"}"#, b"not zstd at all".to_vec()),
            (
                r#"{"name": "crc32c"}"#,
                // The right length, the wrong checksum.
                [bytes_of(&[1i32, 2]), vec![0, 0, 0, 0]].concat(),
            ),
            (
                r#"{"name": "gzip"}"#,
                // Valid gzip, but not of a whole chunk.
                encode(&[ChunkCodec::Gzip], vec![0u8; 4]),
            ),
            (r#"{"name": "crc32c"}"#, vec![1, 2]),
        ];
        for (codec, chunk) in cases {
            let store = StoreBuilder::new();
            store.array(
                "a",
                "int32",
                &[2],
                &[2],
                &format!(r#"{{"codecs": [{{"name": "bytes"}}, {codec}]}}"#),
            );
            store.write("a/c/0", &chunk);
            let file = store.open().expect("open");
            let array = array_at(&file, "a");
            let layout = array.layout(&[], &QualitySpec::default()).expect("layout");
            let mut out = vec![0u8; layout.total_bytes as usize];
            let err = array.read_range(&layout, 0, &mut out).expect_err(codec);
            assert!(matches!(err, AexError::MalformedZarr(_)), "{codec}: {err}");
            assert_eq!(err.class(), ErrorClass::Permanent, "{codec}");
        }
    }

    #[test]
    fn a_decompression_bomb_does_not_expand() {
        let store = StoreBuilder::new();
        store.array(
            "a",
            "int32",
            &[2],
            &[2],
            r#"{"codecs": [{"name": "bytes"}, {"name": "gzip"}]}"#,
        );
        // Sixteen megabytes of zeros in a few kilobytes: the reader must stop
        // at one chunk, not at the end of the stream.
        store.write("a/c/0", &encode(&[ChunkCodec::Gzip], vec![0u8; 1 << 24]));

        let file = store.open().expect("open");
        let array = array_at(&file, "a");
        let layout = array.layout(&[], &QualitySpec::default()).expect("layout");
        let mut out = vec![0u8; layout.total_bytes as usize];
        let err = array.read_range(&layout, 0, &mut out).expect_err("bomb");
        assert!(matches!(err, AexError::MalformedZarr(_)), "{err}");
    }

    /// The metadata of a sharded array of `shape`, shards of `shard`, inner
    /// chunks of `inner`.
    fn shard_meta(shape: &[u64], shard: &[u64], inner: &[u64], at_start: bool) -> String {
        let location = if at_start { "start" } else { "end" };
        format!(
            r#"{{"chunk_grid": {{"name": "regular",
                                 "configuration": {{"chunk_shape": {shard:?}}}}},
                 "shape": {shape:?},
                 "codecs": [{{"name": "sharding_indexed", "configuration": {{
                     "chunk_shape": {inner:?},
                     "codecs": [{{"name": "bytes", "configuration": {{"endian": "little"}}}}],
                     "index_codecs": [{{"name": "bytes",
                                        "configuration": {{"endian": "little"}}}},
                                      {{"name": "crc32c"}}],
                     "index_location": "{location}"}}}}]}}"#
        )
    }

    /// One shard file holding `chunks` in order; `None` is an unwritten chunk.
    fn shard_file(chunks: &[Option<Vec<u8>>], at_start: bool) -> Vec<u8> {
        let mut index = Vec::new();
        let mut body = Vec::new();
        // With the index at the start, the chunks begin after it.
        let base = if at_start {
            chunks.len() as u64 * 16 + 4
        } else {
            0
        };
        for chunk in chunks {
            match chunk {
                Some(bytes) => {
                    index.extend((base + body.len() as u64).to_le_bytes());
                    index.extend((bytes.len() as u64).to_le_bytes());
                    body.extend(bytes);
                }
                None => {
                    index.extend(u64::MAX.to_le_bytes());
                    index.extend(u64::MAX.to_le_bytes());
                }
            }
        }
        index.extend(crc32c::crc32c(&index).to_le_bytes());
        if at_start {
            index.extend(body);
            index
        } else {
            body.extend(index);
            body
        }
    }

    #[test]
    fn sharded_arrays_read_back_from_either_index_location() {
        for at_start in [false, true] {
            let store = StoreBuilder::new();
            // Two shards along each axis, four inner chunks in each shard, and
            // a shape the shards do not cover evenly.
            let (shape, shard, inner) = ([5u64, 7], [4u64, 4], [2u64, 2]);
            store.array(
                "a",
                "int32",
                &shape,
                &shard,
                &shard_meta(&shape, &shard, &inner, at_start),
            );

            let values: Vec<i32> = (0..35).collect();
            let flat: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            // Lay the array out by hand: shard grid 2x2, inner grid 2x2.
            for (sr, sc) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                let mut chunks = Vec::new();
                for (ir, ic) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                    let mut chunk = Vec::new();
                    let mut written = false;
                    for r in 0..2 {
                        for c in 0..2 {
                            let row = sr * 4 + ir * 2 + r;
                            let col = sc * 4 + ic * 2 + c;
                            let v = if row < 5 && col < 7 {
                                written = true;
                                values[row * 7 + col]
                            } else {
                                0
                            };
                            chunk.extend(v.to_le_bytes());
                        }
                    }
                    // A chunk wholly outside the array was never written.
                    chunks.push(written.then_some(chunk));
                }
                store.write(&format!("a/c/{sr}/{sc}"), &shard_file(&chunks, at_start));
            }

            let file = store.open().expect("open");
            let array = array_at(&file, "a");
            assert_eq!(read_all(array.as_ref(), &[]), flat, "at_start={at_start}");
            check_ranges(array.as_ref(), &flat);
            // The inner chunk is what lands in the cache, not the shard.
            assert_eq!(array.decoded_chunk_bytes(), Some(2 * 2 * 4));
        }
    }

    #[test]
    fn an_unwritten_shard_reads_as_the_fill_value() {
        let store = StoreBuilder::new();
        let (shape, shard, inner) = ([4u64], [2u64], [2u64]);
        let mut meta = shard_meta(&shape, &shard, &inner, false);
        meta.insert_str(1, r#""fill_value": -7, "#);
        store.array("a", "int32", &shape, &shard, &meta);
        // Only the first shard exists; the second is absent entirely.
        store.write("a/c/0", &shard_file(&[Some(bytes_of(&[1i32, 2]))], false));

        let file = store.open().expect("open");
        let array = array_at(&file, "a");
        let expected = bytes_of(&[1i32, 2, -7, -7]);
        assert_eq!(read_all(array.as_ref(), &[]), expected);
        check_ranges(array.as_ref(), &expected);
    }

    #[test]
    fn a_broken_shard_is_reported_before_the_transfer() {
        let whole = shard_file(&[Some(bytes_of(&[1i32, 2]))], false);
        let cases: [(&str, Vec<u8>); 3] = [
            ("too short to hold an index", vec![0u8; 8]),
            ("a damaged index checksum", {
                let mut bytes = whole.clone();
                let at = bytes.len() - 1;
                bytes[at] ^= 0xff;
                bytes
            }),
            ("an inner chunk off the end", {
                let mut bytes = whole.clone();
                // The index sits at the end; overwrite its first length.
                let at = bytes.len() - 4 - 8;
                bytes[at..at + 8].copy_from_slice(&u64::MAX.wrapping_sub(1).to_le_bytes());
                let index = bytes.len() - 4 - 16;
                let sum = crc32c::crc32c(&bytes[index..bytes.len() - 4]);
                let end = bytes.len() - 4;
                bytes[end..].copy_from_slice(&sum.to_le_bytes());
                bytes
            }),
        ];
        for (what, shard) in cases {
            let store = StoreBuilder::new();
            let (shape, shard_shape, inner) = ([2u64], [2u64], [2u64]);
            store.array(
                "a",
                "int32",
                &shape,
                &shard_shape,
                &shard_meta(&shape, &shard_shape, &inner, false),
            );
            store.write("a/c/0", &shard);

            let file = store.open().expect("open");
            let array = array_at(&file, "a");
            // The index is built by layout(), on the control plane.
            let err = array.layout(&[], &QualitySpec::default()).expect_err(what);
            assert!(matches!(err, AexError::MalformedZarr(_)), "{what}: {err}");
        }
    }

    #[test]
    fn a_store_cannot_be_escaped() {
        let store = StoreBuilder::new();
        let outside = tempfile::tempdir().expect("tempdir");
        std::fs::write(outside.path().join("secret"), b"secret").expect("write");
        std::fs::create_dir_all(outside.path().join("a/c")).expect("mkdir");
        std::fs::write(outside.path().join("a/c/0"), bytes_of(&[9i32, 9])).expect("write");
        std::fs::write(outside.path().join("a").join(METADATA), b"{}").expect("write");
        std::os::unix::fs::symlink(outside.path(), store.path().join("away")).expect("symlink");

        store.array("a", "int32", &[4], &[2], "");
        store.write("a/c/0", &bytes_of(&[1i32, 2]));
        store.write("a/c/1", &bytes_of(&[3i32, 4]));
        // A link that stays inside the store is fine: the rule is containment.
        std::os::unix::fs::symlink(store.path().join("a"), store.path().join("same"))
            .expect("symlink");

        let file = store.open().expect("open");

        // A name that walks out of the store never reaches the filesystem.
        assert!(!file.contains("../../etc/passwd"));
        let err = file.get_item("../secret").expect_err("traversal");
        assert!(matches!(err, AexError::MalformedZarr(_)), "{err}");

        // A link out of the store is an error, and does not name its target.
        let err = file.get_item("away/a").expect_err("symlink");
        let message = err.to_string();
        assert!(matches!(err, AexError::MalformedZarr(_)), "{message}");
        assert!(
            !message.contains(outside.path().to_str().expect("utf-8")),
            "{message}"
        );

        // The link that stays inside resolves and reads.
        let same = array_at(&file, "same");
        assert_eq!(read_all(same.as_ref(), &[]), bytes_of(&[1i32, 2, 3, 4]));
    }

    #[test]
    fn a_chunk_outside_the_store_is_refused_rather_than_filled() {
        let store = StoreBuilder::new();
        let outside = tempfile::tempdir().expect("tempdir");
        std::fs::write(outside.path().join("chunk"), bytes_of(&[9i32, 9])).expect("write");
        store.array("a", "int32", &[2], &[2], "");
        std::fs::create_dir_all(store.path().join("a/c")).expect("mkdir");
        std::os::unix::fs::symlink(outside.path().join("chunk"), store.path().join("a/c/0"))
            .expect("symlink");

        let file = store.open().expect("open");
        let array = array_at(&file, "a");
        let layout = array.layout(&[], &QualitySpec::default()).expect("layout");
        let mut out = vec![0u8; layout.total_bytes as usize];
        let err = array.read_range(&layout, 0, &mut out).expect_err("escape");
        assert!(matches!(err, AexError::MalformedZarr(_)), "{err}");
    }

    #[test]
    fn groups_are_walked_and_listed() {
        let store = StoreBuilder::new();
        store.node("g1", r#"{"zarr_format": 3, "node_type": "group"}"#);
        store.node("g1/g2", r#"{"zarr_format": 3, "node_type": "group"}"#);
        store.array("g1/g2/deep", "int32", &[2], &[2], "");
        store.array("g1/b", "float32", &[2], &[2], "");
        store.array("g1/a", "float64", &[2], &[2], "");
        // A sibling that cannot be served is left out, not fatal.
        store.array("g1/broken", "string", &[2], &[2], "");
        // A directory with no metadata is not a node at all.
        std::fs::create_dir_all(store.path().join("g1/loose")).expect("mkdir");

        let file = store.open().expect("open");
        assert!(file.contains("g1/g2/deep"));
        assert!(file.contains("/g1/g2/deep/"));
        assert!(!file.contains("g1/loose"));
        assert!(matches!(file.get_item("g1").expect("group"), Item::Group));

        let names: Vec<String> = file
            .list_children("g1")
            .expect("children")
            .into_iter()
            .map(|(name, _)| name)
            .collect();
        assert_eq!(names, ["a", "b", "g2"]);

        // Asking for the unservable child directly still says why.
        assert_eq!(
            file.get_item("g1/broken").expect_err("broken").class(),
            ErrorClass::Request
        );
        let err = file.list_children("g1/a").expect_err("not a group");
        assert!(matches!(err, AexError::NotAGroup(_)), "{err}");
        let err = file.get_item("g1/absent").expect_err("absent");
        assert!(matches!(err, AexError::NotFound(_)), "{err}");
    }

    #[test]
    fn attributes_keep_their_shape_and_their_dtype() {
        let cases: [(&str, AttrValue); 7] = [
            (r#""K""#, AttrValue::Text("K".to_string())),
            (
                "true",
                AttrValue::Array {
                    dtype: DType::Bool,
                    shape: vec![],
                    data: vec![1],
                },
            ),
            (
                "-3",
                AttrValue::Array {
                    dtype: DType::Int64,
                    shape: vec![],
                    data: (-3i64).to_le_bytes().to_vec(),
                },
            ),
            (
                "1.5",
                AttrValue::Array {
                    dtype: DType::Float64,
                    shape: vec![],
                    data: 1.5f64.to_le_bytes().to_vec(),
                },
            ),
            (
                "[0, 1]",
                AttrValue::Array {
                    dtype: DType::Int64,
                    shape: vec![2],
                    data: [0i64.to_le_bytes(), 1i64.to_le_bytes()].concat(),
                },
            ),
            // One double among integers makes the whole array double, as numpy
            // would have it.
            (
                "[0, 1.5]",
                AttrValue::Array {
                    dtype: DType::Float64,
                    shape: vec![2],
                    data: [0.0f64.to_le_bytes(), 1.5f64.to_le_bytes()].concat(),
                },
            ),
            (
                "[[1, 2], [3, 4]]",
                AttrValue::Array {
                    dtype: DType::Int64,
                    shape: vec![2, 2],
                    data: (1..=4i64).flat_map(|n| n.to_le_bytes()).collect(),
                },
            ),
        ];
        for (json, expected) in cases {
            let value: Value = serde_json::from_str(json).expect("json");
            assert_eq!(attr_value(&value), Some(expected), "{json}");
        }
    }

    #[test]
    fn an_attribute_with_no_wire_form_is_left_out() {
        let cases = [
            // One attribute cannot carry several strings.
            r#"["a", "b"]"#,
            r#"{"nested": 1}"#,
            "null",
            // Ragged and mixed arrays have no shape to send.
            "[[1, 2], [3]]",
            "[1, [2]]",
            "[1, \"a\"]",
            "[true, 1]",
        ];
        for json in cases {
            let value: Value = serde_json::from_str(json).expect("json");
            assert_eq!(attr_value(&value), None, "{json}");
        }

        // Too big to leave room for its siblings in one listing.
        let huge = Value::Array(vec![Value::from(0); (MAX_ATTR_BYTES / 8) as usize + 1]);
        assert_eq!(attr_value(&huge), None);
        let long = Value::String("x".repeat(MAX_ATTR_BYTES as usize + 1));
        assert_eq!(attr_value(&long), None);
    }

    #[test]
    fn attributes_travel_with_every_kind_of_node() {
        let store = StoreBuilder::new();
        store.node(
            "",
            r#"{"zarr_format": 3, "node_type": "group",
                "attributes": {"Conventions": "CF-1.8"}}"#,
        );
        store.node(
            "g1",
            r#"{"zarr_format": 3, "node_type": "group",
                "attributes": {"title": "a group"}}"#,
        );
        store.array(
            "g1/ds1",
            "float32",
            &[2],
            &[2],
            r#"{"attributes": {"units": "K", "valid_range": [0.0, 1.0],
                               "_FillValue": -3, "flag_meanings": ["a", "b"]}}"#,
        );

        let file = store.open().expect("open");
        assert_eq!(
            file.attrs("").expect("root"),
            [(
                "Conventions".to_string(),
                AttrValue::Text("CF-1.8".to_string())
            )]
        );
        assert_eq!(
            file.attrs("/g1/").expect("group"),
            [("title".to_string(), AttrValue::Text("a group".to_string()))]
        );

        // Sorted by name, and the array of strings is gone.
        let attrs = file.attrs("g1/ds1").expect("array");
        let names: Vec<&str> = attrs.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["_FillValue", "units", "valid_range"]);

        // A node with no attributes has none, and a missing one is an error.
        assert!(file.attrs("g1").expect("group").len() == 1);
        assert!(matches!(
            file.attrs("absent").expect_err("absent"),
            AexError::NotFound(_)
        ));
    }

    #[test]
    fn a_directory_that_is_not_a_store_is_refused() {
        let dir = tempfile::tempdir().expect("tempdir");
        let err = ZarrFile::open(dir.path(), Arc::new(DecodeCache::new(0))).expect_err("empty");
        assert_eq!(err.class(), ErrorClass::Request, "{err}");
    }

    #[test]
    fn an_array_is_opened_once_per_store() {
        let store = StoreBuilder::new();
        store.array("a", "int32", &[2], &[2], "");
        let file = store.open().expect("open");
        let (first, second) = (array_at(&file, "a"), array_at(&file, "/a/"));
        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn many_threads_share_a_small_cache() {
        let store = StoreBuilder::new();
        let (shape, chunk) = ([64u64, 64], [8u64, 8]);
        let values: Vec<i32> = (0..4096).collect();
        store.array("a", "int32", &shape, &chunk, "");
        let expected = write_chunks(&store, "a", &shape, &chunk, &values);

        // The same numbers again, as one shard per 32x32 block of 8x8 chunks.
        let (shard, inner) = ([32u64, 32], [8u64, 8]);
        store.array(
            "s",
            "int32",
            &shape,
            &shard,
            &shard_meta(&shape, &shard, &inner, false),
        );
        for sr in 0..2u64 {
            for sc in 0..2u64 {
                let mut chunks = Vec::new();
                for ir in 0..4u64 {
                    for ic in 0..4u64 {
                        let mut chunk = Vec::new();
                        for r in 0..8 {
                            for c in 0..8 {
                                let row = sr * 32 + ir * 8 + r;
                                let col = sc * 32 + ic * 8 + c;
                                chunk.extend(values[(row * 64 + col) as usize].to_le_bytes());
                            }
                        }
                        chunks.push(Some(chunk));
                    }
                }
                store.write(&format!("s/c/{sr}/{sc}"), &shard_file(&chunks, false));
            }
        }

        // Room for three chunks and eight readers, so they evict each other.
        let file =
            ZarrFile::open(store.path(), Arc::new(DecodeCache::new(3 * 8 * 8 * 4))).expect("open");
        for name in ["a", "s"] {
            let array = array_at(&file, name);
            std::thread::scope(|scope| {
                for _ in 0..8 {
                    scope.spawn(|| {
                        for _ in 0..4 {
                            assert_eq!(read_all(array.as_ref(), &[]), expected, "{name}");
                        }
                    });
                }
            });
        }
    }
}
