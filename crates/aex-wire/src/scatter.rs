//! The receive buffer several connections write into at once.
//!
//! A transfer is one output array and many chunks, and the chunks arrive on
//! whichever connection is free. Every chunk knows its own position in the
//! logical byte stream, which is the same as its position in the array, so a
//! connection thread can read straight from the socket into the right part of
//! the array with nothing in between. That is the whole point of the fixed
//! header and the logical offsets: one copy, kernel to numpy.
//!
//! Handing several threads mutable slices of one buffer is what needs care
//! here, so it is confined to this file. [`ScatterBuffer::claim`] hands out a
//! range only if nothing else holds an overlapping one, and the claim is
//! released when the slice is dropped. The check costs one lock per chunk —
//! per several hundred kilobytes at least — which is nothing next to letting a
//! chunk allocation bug corrupt a result in silence.

use std::marker::PhantomData;
use std::ops::{Deref, DerefMut, Range};
use std::sync::Mutex;

use crate::error::{Result, WireError};

/// An output buffer that hands out disjoint mutable ranges.
///
/// The buffer it borrows outlives every slice taken from it, which is what
/// keeps the receive path sound: the array is owned by the caller — from M3, a
/// numpy array held alive by the binding — for as long as any connection might
/// still write into it.
pub struct ScatterBuffer<'a> {
    ptr: *mut u8,
    len: u64,
    /// Ranges currently handed out, as a list because there are at most as many
    /// as there are connections.
    claimed: Mutex<Vec<Range<u64>>>,
    /// Ties this to the buffer it borrows without keeping a reference that
    /// would conflict with the pointers handed out.
    owner: PhantomData<&'a mut [u8]>,
}

// The pointer is only ever dereferenced through a claim, and claims are
// disjoint, so sharing the buffer between threads hands each of them a
// different part of it.
unsafe impl Send for ScatterBuffer<'_> {}
unsafe impl Sync for ScatterBuffer<'_> {}

impl std::fmt::Debug for ScatterBuffer<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScatterBuffer")
            .field("len", &self.len)
            .field("claimed", &lock(&self.claimed).len())
            .finish()
    }
}

impl<'a> ScatterBuffer<'a> {
    pub fn new(buffer: &'a mut [u8]) -> Self {
        ScatterBuffer {
            ptr: buffer.as_mut_ptr(),
            len: buffer.len() as u64,
            claimed: Mutex::new(Vec::new()),
            owner: PhantomData,
        }
    }

    /// Length of the whole buffer, which is the transfer's `total_bytes`.
    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Take `[offset, offset + len)` for exclusive use.
    ///
    /// Fails if the range runs past the end of the buffer or overlaps one
    /// already taken. Both mean the caller's chunk allocation is wrong, and
    /// both are better as an error than as a wrong answer.
    pub fn claim(&self, offset: u64, len: u64) -> Result<ScatterSlice<'_>> {
        let end = offset.checked_add(len).filter(|end| *end <= self.len);
        let Some(end) = end else {
            return Err(WireError::OutOfRange {
                offset,
                len,
                capacity: self.len,
            });
        };

        let mut claimed = lock(&self.claimed);
        // An empty range cannot overlap anything, and two of them at the same
        // offset are not a conflict either.
        if len > 0
            && claimed
                .iter()
                .any(|held| held.start < end && offset < held.end)
        {
            return Err(WireError::Overlap { offset, len });
        }
        claimed.push(offset..end);
        drop(claimed);

        Ok(ScatterSlice {
            claimed: &self.claimed,
            // Within the allocation by the bounds check above.
            ptr: unsafe { self.ptr.add(offset as usize) },
            len: len as usize,
            offset,
        })
    }
}

/// One claimed range, usable as a `&mut [u8]`.
///
/// Dropping it releases the claim, so a chunk that failed can be fetched again
/// into the same place.
pub struct ScatterSlice<'s> {
    claimed: &'s Mutex<Vec<Range<u64>>>,
    ptr: *mut u8,
    len: usize,
    offset: u64,
}

// The range is exclusive for as long as the slice exists, so it can be moved to
// the connection thread that fills it.
unsafe impl Send for ScatterSlice<'_> {}

impl ScatterSlice<'_> {
    /// Where this range starts in the logical byte stream.
    pub fn offset(&self) -> u64 {
        self.offset
    }
}

impl Deref for ScatterSlice<'_> {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        // Claimed, so no other slice covers these bytes.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl DerefMut for ScatterSlice<'_> {
    fn deref_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for ScatterSlice<'_> {
    fn drop(&mut self) {
        let mut claimed = lock(self.claimed);
        if let Some(index) = claimed.iter().position(|held| held.start == self.offset) {
            claimed.swap_remove(index);
        }
    }
}

impl std::fmt::Debug for ScatterSlice<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScatterSlice")
            .field("offset", &self.offset)
            .field("len", &self.len)
            .finish()
    }
}

/// Take the lock, ignoring poisoning.
///
/// A panic while a claim was being recorded leaves the list itself consistent —
/// it is only ever pushed to or removed from — and refusing to hand out any
/// further range would turn one failed chunk into a stuck transfer.
fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_claim_writes_through_to_the_buffer() {
        let mut buffer = vec![0u8; 16];
        {
            let scatter = ScatterBuffer::new(&mut buffer);
            assert_eq!(scatter.len(), 16);
            assert!(!scatter.is_empty());

            let mut slice = scatter.claim(4, 4).expect("claim");
            assert_eq!(slice.offset(), 4);
            assert_eq!(slice.len(), 4);
            slice.copy_from_slice(b"abcd");
        }
        assert_eq!(&buffer, b"\0\0\0\0abcd\0\0\0\0\0\0\0\0");
    }

    #[test]
    fn disjoint_claims_are_all_held_at_once() {
        let mut buffer = vec![0u8; 12];
        {
            let scatter = ScatterBuffer::new(&mut buffer);
            let mut first = scatter.claim(0, 4).expect("first");
            let mut second = scatter.claim(8, 4).expect("second");
            let mut middle = scatter.claim(4, 4).expect("middle");
            first.fill(1);
            middle.fill(2);
            second.fill(3);
        }
        assert_eq!(buffer, [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]);
    }

    #[test]
    fn an_overlapping_claim_is_refused() {
        let mut buffer = vec![0u8; 16];
        let scatter = ScatterBuffer::new(&mut buffer);
        let held = scatter.claim(4, 8).expect("claim");

        // Every way two ranges can overlap.
        for (offset, len) in [(4, 8), (0, 5), (11, 5), (6, 2), (0, 16)] {
            let err = scatter.claim(offset, len).unwrap_err();
            assert!(
                matches!(err, WireError::Overlap { .. }),
                "[{offset}, {}) should have overlapped: {err}",
                offset + len
            );
        }
        // Touching at the edges does not overlap.
        scatter.claim(0, 4).expect("just before");
        scatter.claim(12, 4).expect("just after");

        // Releasing one frees its range, which is how a failed chunk is retried.
        drop(held);
        scatter.claim(4, 8).expect("claim again");
    }

    #[test]
    fn a_claim_past_the_end_is_refused() {
        let mut buffer = vec![0u8; 16];
        let scatter = ScatterBuffer::new(&mut buffer);

        scatter.claim(0, 16).expect("the whole buffer");
        for (offset, len) in [(0, 17), (16, 1), (12, 8), (u64::MAX, 1)] {
            let err = scatter.claim(offset, len).expect_err("out of range");
            assert!(matches!(err, WireError::OutOfRange { .. }), "{err}");
        }
        // An empty range at the very end is inside it.
        scatter.claim(16, 0).expect("empty at the end");
    }

    #[test]
    fn an_empty_buffer_can_only_be_claimed_empty() {
        let scatter = ScatterBuffer::new(&mut []);
        assert!(scatter.is_empty());
        scatter.claim(0, 0).expect("empty");
        assert!(scatter.claim(0, 1).is_err());
    }

    #[test]
    fn several_threads_fill_one_buffer_between_them() {
        // The receive path in miniature: each thread takes chunks off a shared
        // queue and writes them where they belong. A run must not depend on
        // which thread got which chunk.
        const CHUNK: u64 = 64;
        const CHUNKS: u64 = 64;
        let mut buffer = vec![0u8; (CHUNK * CHUNKS) as usize];
        let next = std::sync::atomic::AtomicU64::new(0);
        {
            let scatter = ScatterBuffer::new(&mut buffer);
            std::thread::scope(|scope| {
                for _ in 0..8 {
                    scope.spawn(|| loop {
                        let chunk = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if chunk >= CHUNKS {
                            break;
                        }
                        let mut slice = scatter
                            .claim(chunk * CHUNK, CHUNK)
                            .expect("disjoint chunks");
                        slice.fill(chunk as u8);
                    });
                }
            });
        }
        for chunk in 0..CHUNKS {
            let start = (chunk * CHUNK) as usize;
            assert!(
                buffer[start..start + CHUNK as usize]
                    .iter()
                    .all(|&b| b == chunk as u8),
                "chunk {chunk} was not written whole"
            );
        }
    }
}
