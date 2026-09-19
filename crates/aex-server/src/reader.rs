//! The read half of a data connection.
//!
//! `pread` blocks and so does `write`, and running them one after the other
//! leaves the disk idle while the socket works and the socket idle while the
//! disk works. For data that does not fit in memory — which is what this whole
//! system is for — that difference is most of the throughput.
//!
//! So each connection gets a reader thread. It takes what the connection was
//! asked for, cuts it into pieces, fills a buffer with each, and hands them over
//! as they come; the connection thread writes each piece out and hands the
//! buffer straight back. A fixed set of buffers circulates between the two, so
//! a transfer in progress allocates nothing and neither side can run away from
//! the other.
//!
//! The pieces go out as separate `DATA` frames, which the protocol allows for
//! exactly this reason: what a fetch asks for and what one frame carries were
//! never required to be the same thing.
//!
//! Whether this pays depends on which side is slower. When the link is slower
//! than the storage — which is the case this system is built for — overlapping
//! them is most of the throughput. When the storage is the slower side by a
//! wide margin, as it is over loopback, there is nothing much to overlap and
//! the handoff is pure cost. `read_buffers = 1` says so: the reading then
//! happens on the connection's own thread, with no second thread at all.

use std::cell::RefCell;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::thread::JoinHandle;

use aex_core::AexError;

use crate::error::{Result, ServerError};
use crate::transfer::TransferEntry;

/// A range of one transfer, for the reader to produce.
///
/// Inline, the offset and the length are walked forward as pieces come out of
/// it; on the reader thread they are only ever read.
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

/// Where the reading happens.
enum Mode {
    /// On the connection's own thread. No handoff, and no overlap.
    Inline(RefCell<Inline>),
    /// On a thread of its own, overlapping with the writing.
    Threaded(Threaded),
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

/// The reader thread of one connection, and the channels it shares with it.
struct Threaded {
    /// `None` once shut down, so that the reader sees its work end.
    work: Option<Sender<ReadRequest>>,
    pieces: Receiver<Piece>,
    /// Buffers the connection thread has finished writing.
    free: Option<Sender<Vec<u8>>>,
    thread: Option<JoinHandle<()>>,
}

/// How a connection reads what it is asked to send.
pub struct ReadPipeline {
    mode: Mode,
    piece_bytes: usize,
}

impl ReadPipeline {
    /// Start a reader with `buffers` buffers of `piece_bytes` each.
    ///
    /// One buffer means no second thread: the reading happens inline, and the
    /// two never overlap. Two or more starts the reader.
    pub fn start(buffers: u32, piece_bytes: usize) -> Result<Self> {
        let mode = if buffers <= 1 {
            Mode::Inline(RefCell::new(Inline {
                buffer: Some(vec![0u8; piece_bytes]),
                pending: None,
                hinted: 0,
            }))
        } else {
            Mode::Threaded(Threaded::start(buffers, piece_bytes)?)
        };
        Ok(ReadPipeline { mode, piece_bytes })
    }

    /// How much of a fetch one piece carries.
    pub fn piece_bytes(&self) -> usize {
        self.piece_bytes
    }

    /// Ask for a range. The pieces of it come back from [`Self::next_piece`].
    pub fn request(&self, entry: Arc<TransferEntry>, offset: u64, len: u64) -> Result<()> {
        let request = ReadRequest { entry, offset, len };
        match &self.mode {
            Mode::Inline(inline) => {
                let mut inline = inline.borrow_mut();
                inline.hinted = request.offset;
                inline.pending = Some(request);
                Ok(())
            }
            Mode::Threaded(threaded) => threaded
                .work
                .as_ref()
                .and_then(|work| work.send(request).ok())
                .ok_or_else(stopped),
        }
    }

    /// The next piece of the range that was asked for.
    pub fn next_piece(&self) -> Result<Piece> {
        match &self.mode {
            Mode::Inline(inline) => Ok(inline.borrow_mut().next(self.piece_bytes)),
            Mode::Threaded(threaded) => threaded.pieces.recv().map_err(|_| stopped()),
        }
    }

    /// Give a buffer back once it has been written out.
    pub fn recycle(&self, bytes: Vec<u8>) {
        match &self.mode {
            Mode::Inline(inline) => inline.borrow_mut().buffer = Some(bytes),
            // A closed pool means the reader has already stopped, and the
            // buffer is of no use to anyone; dropping it is the whole of the
            // cleanup.
            Mode::Threaded(threaded) => {
                if let Some(free) = &threaded.free {
                    let _ = free.send(bytes);
                }
            }
        }
    }
}

fn stopped() -> ServerError {
    ServerError::Protocol("the reader thread has stopped".to_string())
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

impl Threaded {
    fn start(buffers: u32, piece_bytes: usize) -> Result<Self> {
        let (work_tx, work_rx) = channel::<ReadRequest>();
        let (piece_tx, piece_rx) = channel::<Piece>();
        let (free_tx, free_rx) = channel::<Vec<u8>>();

        // Allocated once, up front: the point of the pool is that a transfer in
        // progress never has to ask the allocator for anything.
        for _ in 0..buffers {
            free_tx
                .send(vec![0u8; piece_bytes])
                .map_err(|_| ServerError::Protocol("the read pool closed at once".to_string()))?;
        }

        let thread = std::thread::Builder::new()
            .name("aex-data-read".to_string())
            .spawn(move || read_loop(&work_rx, &piece_tx, &free_rx))?;

        Ok(Threaded {
            work: Some(work_tx),
            pieces: piece_rx,
            free: Some(free_tx),
            thread: Some(thread),
        })
    }
}

impl Drop for Threaded {
    fn drop(&mut self) {
        // Both senders have to go before the reader can see that there is
        // nothing more coming, whichever of the two it is waiting on.
        self.work.take();
        self.free.take();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn read_loop(work: &Receiver<ReadRequest>, pieces: &Sender<Piece>, free: &Receiver<Vec<u8>>) {
    // The buffer of a failed read never went out, so it is kept here for the
    // next one; dropping it would shrink the pool until the reader starves.
    let mut spare: Option<Vec<u8>> = None;
    while let Ok(request) = work.recv() {
        let mut offset = request.offset;
        let mut remaining = request.len;

        let outcome = loop {
            if remaining == 0 {
                break Piece::Done;
            }
            // Blocks until the connection thread has written one out, which is
            // what keeps the reader from running ahead without bound.
            let mut bytes = match spare.take() {
                Some(bytes) => bytes,
                None => match free.recv() {
                    Ok(bytes) => bytes,
                    Err(_) => return,
                },
            };

            // The buffer keeps its full length; only this much of it is read
            // into. Shrinking it and growing it back would zero the difference
            // before every fetch, for bytes about to be overwritten anyway.
            let piece = remaining.min(bytes.len() as u64) as usize;
            if let Err(e) = request.entry.dataset().read_range(
                request.entry.layout(),
                offset,
                &mut bytes[..piece],
            ) {
                spare = Some(bytes);
                break Piece::Failed(e);
            }

            let at = offset;
            offset += piece as u64;
            remaining -= piece as u64;
            if pieces
                .send(Piece::Data {
                    offset: at,
                    bytes,
                    len: piece,
                })
                .is_err()
            {
                return;
            }
        };

        if pieces.send(outcome).is_err() {
            return;
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
            match pipeline.next_piece().expect("the reader is alive") {
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
        let pipeline = ReadPipeline::start(1, 4096).expect("start");
        pipeline.request(entry, 0, len).expect("request");

        let Piece::Data { bytes, .. } = pipeline.next_piece().expect("a piece") else {
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
        while let Piece::Data { bytes, .. } = pipeline.next_piece().expect("a piece") {
            pipeline.recycle(bytes);
        }
        assert_eq!(dataset.hinted.load(Ordering::Relaxed), len);
    }

    /// Read-ahead is for the connection's own thread; the reader thread has its
    /// own buffers ahead of the writer and would only ask twice.
    #[test]
    fn the_reader_thread_does_not_ask_for_read_ahead() {
        let (entry, dataset) = entry_of(8192, None);
        let pipeline = ReadPipeline::start(3, 4096).expect("start");
        pipeline.request(entry, 0, 8192).expect("request");
        let (pieces, failure) = drain(&pipeline);

        assert!(failure.is_none());
        assert_eq!(pieces.len(), 2);
        assert_eq!(dataset.hinted.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn a_request_comes_back_in_pieces_that_cover_it_exactly() {
        let (entry, _) = entry_of(1000, None);
        let pipeline = ReadPipeline::start(3, 256).expect("start");
        pipeline.request(entry, 0, 1000).expect("request");

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
    fn one_buffer_reads_inline_and_still_covers_the_range() {
        // No second thread at all: the escape hatch for a deployment whose
        // storage is so much slower than its link that overlapping the two
        // buys less than handing the buffers between threads costs.
        let (entry, _) = entry_of(600, None);
        let pipeline = ReadPipeline::start(1, 256).expect("start");
        pipeline.request(entry, 100, 400).expect("request");

        let (pieces, failure) = drain(&pipeline);
        assert!(failure.is_none());
        assert_eq!(
            pieces.iter().map(|(_, b)| b.len()).sum::<usize>(),
            400,
            "the pieces cover the range whatever the pool size"
        );
        assert_eq!(pieces[0].0, 100, "and start where they were asked to");
    }

    #[test]
    fn a_read_that_fails_stops_the_request_and_says_so() {
        let (entry, dataset) = entry_of(1000, Some(512));
        let pipeline = ReadPipeline::start(2, 256).expect("start");
        pipeline.request(entry, 0, 1000).expect("request");

        let (pieces, failure) = drain(&pipeline);
        assert!(failure.is_some(), "the failure has to reach the caller");
        assert_eq!(pieces.len(), 2, "nothing past the piece that failed");
        // The reader stopped rather than carrying on through the range.
        assert_eq!(dataset.reads.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn failed_reads_do_not_use_up_the_buffers() {
        let (entry, _) = entry_of(1000, Some(0));
        let pipeline = ReadPipeline::start(2, 256).expect("start");
        // More failures than there are buffers; a leak would hang here.
        for _ in 0..5 {
            pipeline.request(entry.clone(), 0, 100).expect("request");
            assert!(drain(&pipeline).1.is_some());
        }
        pipeline.request(entry, 256, 600).expect("request");
        let (pieces, failure) = drain(&pipeline);
        assert!(failure.is_none());
        assert_eq!(pieces.len(), 3);
    }

    #[test]
    fn the_reader_serves_one_request_after_another() {
        let (entry, _) = entry_of(1000, None);
        let pipeline = ReadPipeline::start(2, 512).expect("start");
        for offset in [0u64, 200, 400] {
            pipeline
                .request(entry.clone(), offset, 100)
                .expect("request");
            let (pieces, failure) = drain(&pipeline);
            assert!(failure.is_none());
            assert_eq!(pieces.len(), 1);
            assert_eq!(pieces[0].0, offset);
        }
    }

    #[test]
    fn dropping_the_pipeline_stops_its_thread() {
        let pipeline = ReadPipeline::start(2, 64).expect("start");
        assert_eq!(pipeline.piece_bytes(), 64);
        // Nothing to assert beyond the fact that this returns: Drop joins the
        // reader, so a thread that did not notice would hang the test.
        drop(pipeline);
    }
}
