//! Decompressed storage chunks, shared by every connection.
//!
//! A compressed chunk can only be decoded whole, and clients split the logical
//! byte stream however they like, so several connections routinely want parts
//! of the same chunk. Keeping the decoded chunk means it is decoded once no
//! matter how the transfer was split.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::error::Result;

/// Identifies one storage chunk: which dataset, and which chunk of it.
pub type ChunkKey = (u64, u64);

/// A byte-bounded LRU of decoded chunks.
///
/// One lock for the whole cache: it is held only to look up or insert, never
/// while decoding or copying out, so it is short next to either.
///
/// Chunks are held as `Arc<Vec<u8>>` rather than `Arc<[u8]>`: the latter
/// cannot take a `Vec`'s buffer, so every chunk would be allocated and copied
/// a second time on its way in.
pub struct DecodeCache {
    capacity: u64,
    lru: Mutex<Lru>,
    counts: Counts,
}

/// What the cache did, since the server started.
///
/// Misses are decodes, so comparing them with the chunks a transfer delivered
/// says whether the cache is holding a chunk long enough to be read out, or
/// evicting it under the reader that is still walking it.
#[derive(Debug, Default)]
pub struct Counts {
    hits: AtomicU64,
    misses: AtomicU64,
    /// Chunks two threads decoded at once. Both decodes happened; one is lost.
    races: AtomicU64,
}

/// A reading of [`Counts`], for logging or for a test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub races: u64,
}

#[derive(Default)]
struct Lru {
    /// Least recently used first.
    // ponytail: linear scan; entries are MiB-sized so there are few. An index
    // map would pay off only with many small chunks.
    entries: VecDeque<(ChunkKey, Arc<Vec<u8>>)>,
    bytes: u64,
}

impl DecodeCache {
    /// A cache holding at most `capacity` decoded bytes. Zero disables it.
    pub fn new(capacity: u64) -> Self {
        DecodeCache {
            capacity,
            lru: Mutex::default(),
            counts: Counts::default(),
        }
    }

    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    /// What the cache has done so far. Counters never reset, so a measurement
    /// takes the difference across the transfer it cares about.
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            hits: self.counts.hits.load(Ordering::Relaxed),
            misses: self.counts.misses.load(Ordering::Relaxed),
            races: self.counts.races.load(Ordering::Relaxed),
        }
    }

    /// The decoded chunk at `key`, decoding it with `decode` on a miss.
    // ponytail: two threads missing the same chunk at once both decode it.
    // Wait on an in-flight decode instead if measurements show it matters.
    pub fn get_or_decode(
        &self,
        key: ChunkKey,
        decode: impl FnOnce() -> Result<Vec<u8>>,
    ) -> Result<Arc<Vec<u8>>> {
        if let Some(hit) = self.get(key) {
            self.counts.hits.fetch_add(1, Ordering::Relaxed);
            return Ok(hit);
        }
        self.counts.misses.fetch_add(1, Ordering::Relaxed);
        let chunk = Arc::new(decode()?);
        self.insert(key, chunk.clone());
        Ok(chunk)
    }

    fn get(&self, key: ChunkKey) -> Option<Arc<Vec<u8>>> {
        let mut lru = self.lru.lock().unwrap_or_else(|e| e.into_inner());
        let at = lru.entries.iter().position(|(k, _)| *k == key)?;
        let entry = lru.entries.remove(at)?;
        let chunk = entry.1.clone();
        lru.entries.push_back(entry);
        Some(chunk)
    }

    fn insert(&self, key: ChunkKey, chunk: Arc<Vec<u8>>) {
        let len = chunk.len() as u64;
        if self.capacity == 0 || len > self.capacity {
            return;
        }
        let mut lru = self.lru.lock().unwrap_or_else(|e| e.into_inner());
        // Another thread decoded it too.
        if lru.entries.iter().any(|(k, _)| *k == key) {
            self.counts.races.fetch_add(1, Ordering::Relaxed);
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
    fn the_counters_say_how_often_a_decode_was_saved() {
        let cache = DecodeCache::new(100);
        let calls = Cell::new(0);
        for _ in 0..4 {
            cache.get_or_decode((1, 7), decode_to(10, &calls)).unwrap();
        }
        assert_eq!(
            cache.stats(),
            CacheStats {
                hits: 3,
                misses: 1,
                races: 0
            }
        );

        // A chunk that does not fit is decoded every time and never held, so
        // every look is a miss rather than a silent hit.
        let big = DecodeCache::new(4);
        for _ in 0..3 {
            big.get_or_decode((1, 7), decode_to(10, &calls)).unwrap();
        }
        assert_eq!(big.stats().hits, 0);
        assert_eq!(big.stats().misses, 3);
    }

    #[test]
    fn a_chunk_two_threads_decoded_at_once_is_counted() {
        let cache = DecodeCache::new(100);
        let calls = Cell::new(0);
        // Decoding from inside the decode closure is how two threads racing on
        // the same key looks to the cache: the second insert finds it there.
        let inner = &cache;
        cache
            .get_or_decode((1, 7), || {
                inner.get_or_decode((1, 7), decode_to(10, &calls))?;
                Ok(vec![0; 10])
            })
            .unwrap();
        assert_eq!(cache.stats().races, 1);
        assert_eq!(cache.stats().misses, 2, "both decodes happened");
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
