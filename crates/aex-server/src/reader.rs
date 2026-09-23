//! The read half of a data connection.
//!
//! `pread` blocks and so does `write`, and running them one after the other
//! leaves the disk idle while the socket works and the socket idle while the
//! disk works. For data that does not fit in memory — which is what this whole
//! system is for — that difference is most of the throughput.
//!
//! The overlap is bought from the kernel rather than from a second thread: the
//! range is cut into pieces, and before blocking on one the storage is told to
//! start fetching the window after it. The disk then works while the socket
//! does, with one thread, one buffer and no handoff.
//!
//! A reader thread was tried instead and measured worse: two threads passing a
//! buffer cost more than they save except on one connection, and even there the
//! same two threads spent on two connections go faster.
//!
//! The pieces go out as separate `DATA` frames, which the protocol allows for
//! exactly this reason: what a fetch asks for and what one frame carries were
//! never required to be the same thing.

use aex_core::{AexError, SelectionLayout};

use crate::transfer::TransferEntry;

/// How far ahead of the piece being read the storage is asked to fetch.
///
/// A cold read is as deep as what is in flight, and one thread can only have
/// one read in flight by itself. Measured on a virtio disk: the throughput
/// stops climbing past about this much, and the hint costs a syscall per piece
/// whatever its size.
const READ_AHEAD_BYTES: u64 = 4 << 20;

/// How a connection reads what it is asked to send.
pub struct FetchReader {
    /// Allocated once and reused, so a transfer in progress never asks the
    /// allocator for anything. It keeps its full size whatever a piece holds,
    /// so that nothing ever has to be zeroed to make it fit.
    buffer: Vec<u8>,
    /// How far into the logical stream the storage has been asked to read.
    hinted: u64,
}

impl FetchReader {
    /// A reader whose pieces are at most `piece_bytes`.
    pub fn new(piece_bytes: usize) -> Self {
        FetchReader {
            buffer: vec![0u8; piece_bytes],
            hinted: 0,
        }
    }

    /// Read `[offset, offset + len)` of a transfer in pieces, handing each to
    /// `send` with its offset.
    ///
    /// A failed read stops the range and comes back as `Ok(Err(_))`, since
    /// the connection survives it; a failed `send` comes back as it is.
    pub fn serve<E>(
        &mut self,
        entry: &TransferEntry,
        mut offset: u64,
        mut len: u64,
        mut send: impl FnMut(u64, &[u8]) -> Result<(), E>,
    ) -> Result<Result<(), AexError>, E> {
        let layout = entry.layout();
        // A fetch that carries on from where the last window reached keeps it;
        // one that jumps somewhere else starts again. Fetches are what the
        // client chose to cut the stream into, and the storage does not care
        // where one ends.
        let carries_on = offset <= self.hinted && self.hinted - offset <= READ_AHEAD_BYTES;
        if !carries_on {
            self.hinted = offset;
        }

        while len > 0 {
            let piece = len.min(self.buffer.len() as u64);
            let piece = row_aligned(layout, offset, piece) as usize;

            // Ask for what comes after this piece before blocking on this one,
            // so that a cold read is already under way by the time it is
            // wanted. The window runs past the end of this fetch, into the rest
            // of the selection: stopping at the fetch boundary leaves the
            // storage idle for the whole of the last piece's send.
            let window = offset
                .saturating_add(READ_AHEAD_BYTES)
                .min(layout.total_bytes);
            if window > self.hinted {
                let from = self.hinted.max(offset);
                entry.dataset().will_need(layout, from, window - from);
                self.hinted = window;
            }
            let bytes = &mut self.buffer[..piece];
            if let Err(e) = entry.dataset().read_range(layout, offset, bytes) {
                return Ok(Err(e));
            }
            send(offset, bytes)?;

            offset += piece as u64;
            len -= piece as u64;
        }
        Ok(Ok(()))
    }
}

/// Cut a piece so that it ends where a row of the output array does.
///
/// A block that holds whole rows is a slab of the output array, and a codec
/// that can predict across rows gets several times the ratio it gets from a
/// flat run of elements. Nothing depends on this for correctness: a block that
/// is not whole rows is simply described as one long row instead.
///
/// A fetch that itself starts mid-row costs one short piece and is then back in
/// step for the rest of its range.
fn row_aligned(layout: &SelectionLayout, offset: u64, piece: u64) -> u64 {
    if layout.quality.eps().is_none() {
        return piece;
    }
    let aligned = match layout.row_bytes() {
        Some(row) if (offset + piece) % row < piece => piece - (offset + piece) % row,
        // A row longer than one piece. Cut on an element instead, and the block
        // goes as a row of its own.
        _ => piece - piece % layout.dtype.itemsize(),
    };
    // Whatever happens, a piece has to move the range forward.
    if aligned == 0 {
        piece
    } else {
        aligned
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;

    use aex_core::{ArrayDataset, DType, QualitySpec, SelectionLayout};

    use super::*;

    fn bounded(shape: &[u64], dtype: DType) -> SelectionLayout {
        let quality = QualitySpec {
            encoding: aex_core::Encoding::ErrorBound,
            abs_error_bound: Some(1e-3),
            ..QualitySpec::default()
        };
        SelectionLayout::resolve(shape, dtype, &[], &quality).expect("layout")
    }

    #[test]
    fn an_exact_transfer_is_cut_wherever_the_buffer_ends() {
        let layout =
            SelectionLayout::resolve(&[64, 300], DType::Float32, &[], &QualitySpec::exact())
                .expect("layout");
        // 1200 bytes a row, and nothing lines up with 4096.
        assert_eq!(row_aligned(&layout, 0, 4096), 4096);
        assert_eq!(row_aligned(&layout, 4096, 4096), 4096);
    }

    #[test]
    fn an_encoded_transfer_is_cut_where_a_row_ends() {
        // 300 float32 a row: 1200 bytes.
        let layout = bounded(&[64, 300], DType::Float32);
        // Three whole rows out of the 3.41 that fit.
        assert_eq!(row_aligned(&layout, 0, 4096), 3600);
        // Already on a row, and still three.
        assert_eq!(row_aligned(&layout, 3600, 4096), 3600);
        // Nothing to trim.
        assert_eq!(row_aligned(&layout, 0, 2400), 2400);
    }

    #[test]
    fn a_fetch_that_starts_mid_row_is_back_in_step_after_one_piece() {
        let layout = bounded(&[64, 300], DType::Float32);
        // Starting 400 bytes into a row, the first piece ends at the third row
        // boundary, and every piece after it is whole rows.
        let first = row_aligned(&layout, 400, 4096);
        assert_eq!(first, 3200);
        assert_eq!((400 + first) % 1200, 0);
        assert_eq!(row_aligned(&layout, 400 + first, 4096), 3600);
    }

    #[test]
    fn a_row_longer_than_the_buffer_is_cut_on_an_element() {
        // 4 MiB a row against a 4096 byte buffer: no boundary is reachable.
        let layout = bounded(&[4, 1 << 20], DType::Float32);
        assert_eq!(row_aligned(&layout, 0, 4094), 4092);
        assert_eq!(row_aligned(&layout, 0, 4096), 4096);

        // And a piece with no whole element in it still moves the range on,
        // rather than looping forever on nothing.
        let wide = bounded(&[4, 1 << 20], DType::Float64);
        assert_eq!(row_aligned(&wide, 0, 3), 3);
    }

    /// A dataset that reads back its own offsets, and counts its reads.
    struct Counting {
        shape: Vec<u64>,
        reads: AtomicU64,
        fail_at: Option<u64>,
        /// The end of the furthest range read-ahead was asked for.
        hinted: AtomicU64,
    }

    impl ArrayDataset for Counting {
        fn dtype(&self) -> DType {
            DType::Uint8
        }
        fn shape(&self) -> &[u64] {
            &self.shape
        }
        fn read_range(
            &self,
            _layout: &SelectionLayout,
            offset: u64,
            dst: &mut [u8],
        ) -> aex_core::Result<()> {
            self.reads.fetch_add(1, Ordering::Relaxed);
            if self.fail_at == Some(offset) {
                return Err(AexError::UnsupportedNpy("no".to_string()));
            }
            for (i, byte) in dst.iter_mut().enumerate() {
                *byte = (offset + i as u64) as u8;
            }
            Ok(())
        }

        fn will_need(&self, _layout: &SelectionLayout, offset: u64, len: u64) {
            self.hinted.fetch_max(offset + len, Ordering::Relaxed);
        }
    }

    fn entry_of(len: u64, fail_at: Option<u64>) -> (Arc<TransferEntry>, Arc<Counting>) {
        let dataset = Arc::new(Counting {
            shape: vec![len],
            reads: AtomicU64::new(0),
            fail_at,
            hinted: AtomicU64::new(0),
        });
        let layout =
            SelectionLayout::resolve(&[len], DType::Uint8, &[], &QualitySpec::exact()).unwrap();
        let registry = crate::transfer::TransferRegistry::new(Arc::new(
            crate::config::ServerConfig::default(),
        ));
        let entry = registry
            .insert([0u8; 16], dataset.clone(), layout)
            .expect("insert");
        (entry, dataset)
    }

    /// Serve a range, returning what came back in order.
    fn drain(
        reader: &mut FetchReader,
        entry: &TransferEntry,
        offset: u64,
        len: u64,
    ) -> (Vec<(u64, Vec<u8>)>, Option<AexError>) {
        let mut pieces = Vec::new();
        let Ok(served) = reader.serve(entry, offset, len, |at, bytes| {
            pieces.push((at, bytes.to_vec()));
            Ok::<_, std::convert::Infallible>(())
        });
        (pieces, served.err())
    }

    #[test]
    fn a_piece_asks_for_the_pieces_after_it_before_it_is_read() {
        let len = 4 * READ_AHEAD_BYTES;
        let (entry, dataset) = entry_of(len, None);
        let mut reader = FetchReader::new(4096);

        let mut first = true;
        let Ok(served) = reader.serve(&entry, 0, len, |_, _| {
            if first {
                assert_eq!(
                    dataset.hinted.load(Ordering::Relaxed),
                    READ_AHEAD_BYTES,
                    "the first piece read should have asked for a window past itself"
                );
                assert_eq!(dataset.reads.load(Ordering::Relaxed), 1, "one piece read");
                first = false;
            }
            Ok::<_, std::convert::Infallible>(())
        });
        served.expect("served");
        // The window slides rather than being asked for again from the start.
        assert_eq!(dataset.hinted.load(Ordering::Relaxed), len);
    }

    #[test]
    fn a_range_comes_back_in_pieces_that_cover_it_exactly() {
        let (entry, _) = entry_of(1000, None);
        let (pieces, failure) = drain(&mut FetchReader::new(256), &entry, 0, 1000);
        assert!(failure.is_none());
        assert_eq!(pieces.len(), 4, "1000 bytes in 256 byte pieces");

        let mut next = 0;
        for (offset, bytes) in &pieces {
            assert_eq!(*offset, next, "a gap or an overlap at {offset}");
            // Every byte says where it came from, so a piece read from the
            // wrong place is a mismatch rather than a plausible number.
            let expected: Vec<u8> = (*offset..offset + bytes.len() as u64)
                .map(|i| i as u8)
                .collect();
            assert_eq!(*bytes, expected);
            next += bytes.len() as u64;
        }
        assert_eq!(next, 1000);
    }

    #[test]
    fn a_range_from_an_offset_starts_and_ends_where_it_was_asked_to() {
        let (entry, _) = entry_of(600, None);
        let (pieces, failure) = drain(&mut FetchReader::new(256), &entry, 100, 400);
        assert!(failure.is_none());
        assert_eq!(
            pieces.iter().map(|(_, b)| b.len()).sum::<usize>(),
            400,
            "the pieces cover the range"
        );
        assert_eq!(pieces[0].0, 100, "and start where they were asked to");
    }

    #[test]
    fn a_read_that_fails_stops_the_range_and_says_so() {
        let (entry, dataset) = entry_of(1000, Some(512));
        let mut reader = FetchReader::new(256);
        let (pieces, failure) = drain(&mut reader, &entry, 0, 1000);
        assert!(failure.is_some(), "the failure has to reach the caller");
        assert_eq!(pieces.len(), 2, "nothing past the piece that failed");
        // The reader stopped rather than carrying on through the range.
        assert_eq!(dataset.reads.load(Ordering::Relaxed), 3);

        // And the next range is served as usual.
        let (pieces, failure) = drain(&mut reader, &entry, 0, 512);
        assert!(failure.is_none());
        assert_eq!(pieces.len(), 2);
    }

    #[test]
    fn a_send_that_fails_stops_the_range() {
        let (entry, dataset) = entry_of(1000, None);
        let served = FetchReader::new(256).serve(&entry, 0, 1000, |_, _| Err("gone"));
        assert!(matches!(served, Err("gone")));
        assert_eq!(dataset.reads.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn the_reader_serves_one_range_after_another() {
        let (entry, _) = entry_of(1000, None);
        let mut reader = FetchReader::new(512);
        for offset in [0u64, 200, 400] {
            let (pieces, failure) = drain(&mut reader, &entry, offset, 100);
            assert!(failure.is_none());
            assert_eq!(pieces.len(), 1);
            assert_eq!(pieces[0].0, offset);
        }
    }
}
