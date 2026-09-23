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

/// Buffers kept back for the next decode.
// ponytail: a fixed bound the pool never approaches -- a miss takes one and an
// eviction hands one back, so it sits near one per connection. Tie it to the
// stream limit if a configuration ever gets near it.
const MAX_SPARES: usize = 32;

/// A byte-bounded LRU of decoded chunks.
///
/// One lock for the whole cache: it is held only to look up or insert, never
/// while decoding or copying out, so it is short next to either.
///
/// Chunks are held as `Arc<Vec<u8>>` rather than `Arc<[u8]>`: the latter
/// cannot take a `Vec`'s buffer, so every chunk would be allocated and copied
/// a second time on its way in.
///
/// An evicted chunk's buffer is kept and lent to the next decode. A chunk is
/// four megabytes, which is over the threshold where the allocator asks the
/// kernel, so letting them go means paying for the mapping and for zeroing
/// its pages again on the way back.
pub struct DecodeCache {
    capacity: u64,
    lru: Mutex<Lru>,
    counts: Counts,
    /// Hands out dataset keys, so datasets of different formats sharing this
    /// cache never read each other's chunks.
    next_key: AtomicU64,
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
    /// Buffers of evicted chunks, waiting to be lent out again.
    spare: Vec<Vec<u8>>,
}

impl DecodeCache {
    /// A cache holding at most `capacity` decoded bytes. Zero disables it.
    pub fn new(capacity: u64) -> Self {
        DecodeCache {
            capacity,
            lru: Mutex::default(),
            counts: Counts::default(),
            next_key: AtomicU64::new(0),
        }
    }

    /// A key no other dataset using this cache has.
    pub fn new_key(&self) -> u64 {
        self.next_key.fetch_add(1, Ordering::Relaxed)
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
    ///
    /// `decode` is handed a buffer to write the chunk into, which is an
    /// evicted chunk's if one is waiting. It is free to return a different
    /// one; the lent buffer is then simply dropped.
    // ponytail: two threads missing the same chunk at once both decode it.
    // Measured: with a chunk the size of one fetch it never happens, with a
    // chunk sixteen times larger it is most of the decoding. Wait on an
    // in-flight decode when that case is worth serving.
    fn get_or_decode(
        &self,
        key: ChunkKey,
        decode: impl FnOnce(Vec<u8>) -> Result<Vec<u8>>,
    ) -> Result<Arc<Vec<u8>>> {
        if let Some(hit) = self.get(key) {
            self.counts.hits.fetch_add(1, Ordering::Relaxed);
            return Ok(hit);
        }
        self.counts.misses.fetch_add(1, Ordering::Relaxed);
        let chunk = Arc::new(decode(self.take_spare())?);
        self.insert(key, chunk.clone());
        Ok(chunk)
    }

    /// Copy the decoded chunk at `key` from `start` into `out`, decoding it
    /// with `decode` on a miss.
    pub fn read_into(
        &self,
        key: ChunkKey,
        start: u64,
        out: &mut [u8],
        decode: impl FnOnce(Vec<u8>) -> Result<Vec<u8>>,
    ) -> Result<()> {
        let decoded = self.get_or_decode(key, decode)?;
        let start = start as usize;
        out.copy_from_slice(&decoded[start..start + out.len()]);
        Ok(())
    }

    /// A buffer for the next decode: an evicted chunk's, or a new one.
    fn take_spare(&self) -> Vec<u8> {
        let mut lru = self.lru.lock().unwrap_or_else(|e| e.into_inner());
        lru.spare.pop().unwrap_or_default()
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
            // Only when nobody is still reading it. If someone is, their own
            // handle keeps it alive and it is freed when they are done.
            if lru.spare.len() < MAX_SPARES {
                if let Some(buffer) = Arc::into_inner(evicted) {
                    lru.spare.push(buffer);
                }
            }
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

    fn decode_to(len: usize, calls: &Cell<u32>) -> impl FnOnce(Vec<u8>) -> Result<Vec<u8>> + '_ {
        move |mut chunk| {
            calls.set(calls.get() + 1);
            chunk.clear();
            chunk.resize(len, 0);
            Ok(chunk)
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
            .get_or_decode((1, 7), |chunk| {
                inner.get_or_decode((1, 7), decode_to(10, &calls))?;
                Ok(chunk)
            })
            .unwrap();
        assert_eq!(cache.stats().races, 1);
        assert_eq!(cache.stats().misses, 2, "both decodes happened");
    }

    #[test]
    fn an_evicted_chunk_lends_its_buffer_to_the_next_decode() {
        // Room for two chunks, so the third evicts the first. A buffer is
        // taken before the insert that frees one, so the fourth is the decode
        // that gets it.
        let cache = DecodeCache::new(20);
        let calls = Cell::new(0);
        for chunk in 0..3 {
            cache
                .get_or_decode((0, chunk), decode_to(10, &calls))
                .unwrap();
        }

        let lent = Cell::new(0usize);
        cache
            .get_or_decode((0, 3), |mut chunk| {
                lent.set(chunk.capacity());
                chunk.clear();
                chunk.resize(10, 0);
                Ok(chunk)
            })
            .unwrap();
        assert_eq!(lent.get(), 10, "the buffer came back with its capacity");
    }

    #[test]
    fn a_buffer_still_being_read_is_not_lent_out() {
        let cache = DecodeCache::new(20);
        let calls = Cell::new(0);
        // Hold chunk 0 the way a reader in the middle of a copy does, then
        // push it out of the cache.
        let held = cache.get_or_decode((0, 0), decode_to(10, &calls)).unwrap();
        for chunk in 1..3 {
            cache
                .get_or_decode((0, chunk), decode_to(10, &calls))
                .unwrap();
        }

        let lent = Cell::new(usize::MAX);
        cache
            .get_or_decode((0, 3), |mut chunk| {
                lent.set(chunk.capacity());
                chunk.resize(10, 0);
                Ok(chunk)
            })
            .unwrap();
        assert_eq!(lent.get(), 0, "a fresh buffer, since chunk 0 is still held");
        assert_eq!(*held, vec![0u8; 10], "and the reader's chunk is intact");
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
            .get_or_decode((0, 0), |_| Err(AexError::MalformedHdf5("bad".into())))
            .unwrap_err();
        assert!(matches!(err, AexError::MalformedHdf5(_)));
        assert!(cache.keys().is_empty());
    }
}
