//! Decompressed storage chunks, shared by every connection.
//!
//! A compressed chunk can only be decoded whole, and clients split the logical
//! byte stream however they like, so several connections routinely want parts
//! of the same chunk. Keeping the decoded chunk means it is decoded once no
//! matter how the transfer was split.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use crate::error::Result;

/// Identifies one storage chunk: which dataset, and which chunk of it.
pub type ChunkKey = (u64, u64);

/// A byte-bounded LRU of decoded chunks.
///
/// One lock for the whole cache: it is held only to look up or insert, never
/// while decoding or copying out, so it is short next to either.
pub struct DecodeCache {
    capacity: u64,
    lru: Mutex<Lru>,
}

#[derive(Default)]
struct Lru {
    /// Least recently used first.
    // ponytail: linear scan; entries are MiB-sized so there are few. An index
    // map would pay off only with many small chunks.
    entries: VecDeque<(ChunkKey, Arc<[u8]>)>,
    bytes: u64,
}

impl DecodeCache {
    /// A cache holding at most `capacity` decoded bytes. Zero disables it.
    pub fn new(capacity: u64) -> Self {
        DecodeCache {
            capacity,
            lru: Mutex::default(),
        }
    }

    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    /// The decoded chunk at `key`, decoding it with `decode` on a miss.
    // ponytail: two threads missing the same chunk at once both decode it.
    // Wait on an in-flight decode instead if measurements show it matters.
    pub fn get_or_decode(
        &self,
        key: ChunkKey,
        decode: impl FnOnce() -> Result<Vec<u8>>,
    ) -> Result<Arc<[u8]>> {
        if let Some(hit) = self.get(key) {
            return Ok(hit);
        }
        let chunk: Arc<[u8]> = decode()?.into();
        self.insert(key, chunk.clone());
        Ok(chunk)
    }

    fn get(&self, key: ChunkKey) -> Option<Arc<[u8]>> {
        let mut lru = self.lru.lock().unwrap_or_else(|e| e.into_inner());
        let at = lru.entries.iter().position(|(k, _)| *k == key)?;
        let entry = lru.entries.remove(at)?;
        let chunk = entry.1.clone();
        lru.entries.push_back(entry);
        Some(chunk)
    }

    fn insert(&self, key: ChunkKey, chunk: Arc<[u8]>) {
        let len = chunk.len() as u64;
        if self.capacity == 0 || len > self.capacity {
            return;
        }
        let mut lru = self.lru.lock().unwrap_or_else(|e| e.into_inner());
        // Another thread decoded it too.
        if lru.entries.iter().any(|(k, _)| *k == key) {
            return;
        }
        lru.entries.push_back((key, chunk));
        lru.bytes += len;
        while lru.bytes > self.capacity {
            let (_, evicted) = lru.entries.pop_front().expect("bytes > 0 means entries");
            lru.bytes -= evicted.len() as u64;
        }
    }

    #[cfg(test)]
    fn keys(&self) -> Vec<ChunkKey> {
        let lru = self.lru.lock().unwrap();
        lru.entries.iter().map(|(k, _)| *k).collect()
    }
}

impl std::fmt::Debug for DecodeCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecodeCache")
            .field("capacity", &self.capacity)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;
    use crate::error::AexError;

    fn decode_to(len: usize, calls: &Cell<u32>) -> impl FnOnce() -> Result<Vec<u8>> + '_ {
        move || {
            calls.set(calls.get() + 1);
            Ok(vec![0; len])
        }
    }

    #[test]
    fn a_chunk_is_decoded_once() {
        let cache = DecodeCache::new(100);
        let calls = Cell::new(0);
        for _ in 0..3 {
            let chunk = cache.get_or_decode((1, 7), decode_to(10, &calls)).unwrap();
            assert_eq!(chunk.len(), 10);
        }
        assert_eq!(calls.get(), 1);
        // Another dataset's chunk 7 is another chunk.
        cache.get_or_decode((2, 7), decode_to(10, &calls)).unwrap();
        assert_eq!(calls.get(), 2);
    }

    #[test]
    fn the_least_recently_used_goes_first() {
        let cache = DecodeCache::new(30);
        let calls = Cell::new(0);
        for chunk in 0..3 {
            cache
                .get_or_decode((0, chunk), decode_to(10, &calls))
                .unwrap();
        }
        // Touch chunk 0, so chunk 1 is now the oldest.
        cache.get_or_decode((0, 0), decode_to(10, &calls)).unwrap();
        cache.get_or_decode((0, 3), decode_to(10, &calls)).unwrap();
        assert_eq!(cache.keys(), [(0, 2), (0, 0), (0, 3)]);
        assert_eq!(calls.get(), 4);
    }

    #[test]
    fn what_does_not_fit_is_not_kept() {
        let calls = Cell::new(0);
        let cache = DecodeCache::new(5);
        cache.get_or_decode((0, 0), decode_to(10, &calls)).unwrap();
        cache.get_or_decode((0, 0), decode_to(10, &calls)).unwrap();
        assert_eq!(calls.get(), 2);
        assert!(cache.keys().is_empty());

        let disabled = DecodeCache::new(0);
        disabled
            .get_or_decode((0, 0), decode_to(0, &calls))
            .unwrap();
        assert!(disabled.keys().is_empty());
    }

    #[test]
    fn a_failed_decode_is_not_cached() {
        let cache = DecodeCache::new(100);
        let err = cache
            .get_or_decode((0, 0), || Err(AexError::MalformedHdf5("bad".into())))
            .unwrap_err();
        assert!(matches!(err, AexError::MalformedHdf5(_)));
        assert!(cache.keys().is_empty());
    }
}
