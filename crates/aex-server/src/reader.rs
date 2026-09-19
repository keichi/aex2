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

use std::cell::RefCell;
use std::sync::Arc;

use aex_core::AexError;

use crate::transfer::TransferEntry;

/// A range of one transfer, for the reader to produce.
///
/// The offset and the length are walked forward as pieces come out of it.
struct ReadRequest {
    entry: Arc<TransferEntry>,
    offset: u64,
    len: u64,
}

/// What comes back from the reader.
pub enum Piece {
    /// The first `len` bytes of `bytes` are the logical byte stream from
    /// `offset`. The buffer keeps its full size whatever the piece holds, so
    /// that nothing ever has to be zeroed to make it fit.
    Data {
        offset: u64,
        bytes: Vec<u8>,
        len: usize,
    },
    /// The request is complete.
    Done,
    /// The read failed. Nothing further arrives for this request.
    Failed(AexError),
}

/// Reading on the caller's thread, one piece at a time.
struct Inline {
    /// Taken by `next_piece` and put back by `recycle`.
    buffer: Option<Vec<u8>>,
    /// What is left of the range being served.
    pending: Option<ReadRequest>,
    /// How far into the logical stream the storage has been asked to read.
    hinted: u64,
}

/// How far ahead of the piece being read the storage is asked to fetch.
///
/// A cold read is as deep as what is in flight, and one thread can only have
/// one read in flight by itself. Measured on a virtio disk: the throughput
/// stops climbing past about this much, and the hint costs a syscall per piece
/// whatever its size.
const READ_AHEAD_BYTES: u64 = 4 << 20;

/// How a connection reads what it is asked to send.
pub struct ReadPipeline {
    inline: RefCell<Inline>,
    piece_bytes: usize,
}

impl ReadPipeline {
    /// Start a reader with one buffer of `piece_bytes`.
    ///
    /// Allocated once and reused, so a transfer in progress never asks the
    /// allocator for anything.
    pub fn start(piece_bytes: usize) -> Self {
        ReadPipeline {
            inline: RefCell::new(Inline {
                buffer: Some(vec![0u8; piece_bytes]),
                pending: None,
                hinted: 0,
            }),
            piece_bytes,
        }
    }

    /// How much of a fetch one piece carries.
    pub fn piece_bytes(&self) -> usize {
        self.piece_bytes
    }

    /// Ask for a range. The pieces of it come back from [`Self::next_piece`].
    pub fn request(&self, entry: Arc<TransferEntry>, offset: u64, len: u64) {
        let mut inline = self.inline.borrow_mut();
        inline.hinted = offset;
        inline.pending = Some(ReadRequest { entry, offset, len });
    }

    /// The next piece of the range that was asked for.
    pub fn next_piece(&self) -> Piece {
        self.inline.borrow_mut().next(self.piece_bytes)
    }

    /// Give a buffer back once it has been written out.
    pub fn recycle(&self, bytes: Vec<u8>) {
        self.inline.borrow_mut().buffer = Some(bytes);
    }
}

impl Inline {
    fn next(&mut self, piece_bytes: usize) -> Piece {
        let Some(request) = self.pending.as_mut() else {
            return Piece::Done;
        };
        if request.len == 0 {
            self.pending = None;
            return Piece::Done;
        }

        // Only missing if the caller dropped a buffer rather than recycling it,
        // which happens on the error path and costs one allocation.
        let mut bytes = self.buffer.take().unwrap_or_else(|| vec![0u8; piece_bytes]);
        let piece = request.len.min(bytes.len() as u64) as usize;
        let at = request.offset;

        // Ask for what comes after this piece before blocking on this one, so
        // that a cold read is already under way by the time it is wanted.
        let window = at.saturating_add(READ_AHEAD_BYTES).min(at + request.len);
        if window > self.hinted {
            let from = self.hinted.max(at);
            request
                .entry
                .dataset()
                .will_need(request.entry.layout(), from, window - from);
            self.hinted = window;
        }
        if let Err(e) =
            request
                .entry
                .dataset()
                .read_range(request.entry.layout(), at, &mut bytes[..piece])
        {
            self.buffer = Some(bytes);
            self.pending = None;
            return Piece::Failed(e);
        }

        request.offset += piece as u64;
        request.len -= piece as u64;
        Piece::Data {
            offset: at,
            bytes,
            len: piece,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use aex_core::{ArrayDataset, DType, QualitySpec, SelectionLayout};

    use super::*;

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

    /// Drain a request, returning what came back in order.
    fn drain(pipeline: &ReadPipeline) -> (Vec<(u64, Vec<u8>)>, Option<AexError>) {
        let mut data = Vec::new();
        loop {
            match pipeline.next_piece() {
                Piece::Data { offset, bytes, len } => {
                    data.push((offset, bytes[..len].to_vec()));
                    pipeline.recycle(bytes);
                }
                Piece::Done => return (data, None),
                Piece::Failed(e) => return (data, Some(e)),
            }
        }
    }

    #[test]
    fn reading_inline_asks_for_the_pieces_after_the_one_it_is_reading() {
        let len = 4 * READ_AHEAD_BYTES;
        let (entry, dataset) = entry_of(len, None);
        let pipeline = ReadPipeline::start(4096);
        pipeline.request(entry, 0, len);

        let Piece::Data { bytes, .. } = pipeline.next_piece() else {
            panic!("the first piece is data");
        };
        assert_eq!(
            dataset.hinted.load(Ordering::Relaxed),
            READ_AHEAD_BYTES,
            "the first piece read should have asked for a window past itself"
        );
        assert_eq!(dataset.reads.load(Ordering::Relaxed), 1, "one piece read");
        pipeline.recycle(bytes);

        // The window slides rather than being asked for again from the start.
        while let Piece::Data { bytes, .. } = pipeline.next_piece() {
            pipeline.recycle(bytes);
        }
        assert_eq!(dataset.hinted.load(Ordering::Relaxed), len);
    }

    #[test]
    fn a_request_comes_back_in_pieces_that_cover_it_exactly() {
        let (entry, _) = entry_of(1000, None);
        let pipeline = ReadPipeline::start(256);
        pipeline.request(entry, 0, 1000);

        let (pieces, failure) = drain(&pipeline);
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
    fn a_request_from_an_offset_starts_and_ends_where_it_was_asked_to() {
        let (entry, _) = entry_of(600, None);
        let pipeline = ReadPipeline::start(256);
        pipeline.request(entry, 100, 400);

        let (pieces, failure) = drain(&pipeline);
        assert!(failure.is_none());
        assert_eq!(
            pieces.iter().map(|(_, b)| b.len()).sum::<usize>(),
            400,
            "the pieces cover the range"
        );
        assert_eq!(pieces[0].0, 100, "and start where they were asked to");
    }

    #[test]
    fn a_read_that_fails_stops_the_request_and_says_so() {
        let (entry, dataset) = entry_of(1000, Some(512));
        let pipeline = ReadPipeline::start(256);
        pipeline.request(entry, 0, 1000);

        let (pieces, failure) = drain(&pipeline);
        assert!(failure.is_some(), "the failure has to reach the caller");
        assert_eq!(pieces.len(), 2, "nothing past the piece that failed");
        // The reader stopped rather than carrying on through the range.
        assert_eq!(dataset.reads.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn failed_reads_do_not_use_up_the_buffers() {
        let (entry, _) = entry_of(1000, Some(0));
        let pipeline = ReadPipeline::start(256);
        // More failures than there are buffers; a leak would hang here.
        for _ in 0..5 {
            pipeline.request(entry.clone(), 0, 100);
            assert!(drain(&pipeline).1.is_some());
        }
        pipeline.request(entry, 256, 600);
        let (pieces, failure) = drain(&pipeline);
        assert!(failure.is_none());
        assert_eq!(pieces.len(), 3);
    }

    #[test]
    fn the_reader_serves_one_request_after_another() {
        let (entry, _) = entry_of(1000, None);
        let pipeline = ReadPipeline::start(512);
        for offset in [0u64, 200, 400] {
            pipeline.request(entry.clone(), offset, 100);
            let (pieces, failure) = drain(&pipeline);
            assert!(failure.is_none());
            assert_eq!(pieces.len(), 1);
            assert_eq!(pieces[0].0, offset);
        }
    }

    #[test]
    fn the_piece_size_is_what_it_was_started_with() {
        assert_eq!(ReadPipeline::start(64).piece_bytes(), 64);
    }
}
