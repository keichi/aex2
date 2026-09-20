//! What a transfer is, and what comes back from one.
//!
//! The plan the control plane returns is the whole description of a transfer:
//! a length, and either the data itself for something small or the identifiers
//! the data plane needs. From there the client decides on its own how to cut
//! the logical byte stream into chunks — the server neither knows nor cares,
//! because the information that decides a good chunk size (how many
//! connections, what the round trip and the bandwidth are) is all on this side.

use std::time::Duration;

use aex_core::{DType, QualitySpec};
use aex_proto::convert::quality_from_proto;
use aex_wire::{Ticket, TICKET_LEN};

use crate::error::{ClientError, Result};

/// A resolved selection, ready to fetch.
///
/// The dtype and shape are the server's answer, so a caller sizes its buffer
/// from them rather than working the selection out a second time.
#[derive(Clone)]
pub struct Plan {
    /// 0 when the data came back inline and there is nothing to fetch.
    pub(crate) request_id: u32,
    pub(crate) ticket: Ticket,
    pub dtype: DType,
    pub shape: Vec<u64>,
    pub total_bytes: u64,
    /// What the server did to the elements, which may be less than asked.
    pub applied_quality: QualitySpec,
    pub(crate) requested_quality: QualitySpec,
    pub(crate) inline_data: Vec<u8>,
}

impl std::fmt::Debug for Plan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The ticket is a capability and the inline data can be large.
        f.debug_struct("Plan")
            .field("request_id", &self.request_id)
            .field("dtype", &self.dtype)
            .field("shape", &self.shape)
            .field("total_bytes", &self.total_bytes)
            .field("inline", &self.is_inline())
            .finish()
    }
}

impl Plan {
    /// Read a plan off the wire.
    pub(crate) fn from_proto(
        plan: aex_proto::TransferPlan,
        requested_quality: &QualitySpec,
    ) -> Result<Self> {
        let dtype =
            DType::from_i32(plan.dtype).map_err(|e| ClientError::Protocol(e.to_string()))?;
        let shape = plan
            .shape
            .iter()
            .map(|&n| {
                u64::try_from(n)
                    .map_err(|_| ClientError::Protocol(format!("negative axis length {n}")))
            })
            .collect::<Result<Vec<u64>>>()?;

        // A zero request_id is how the server says the data is attached; only
        // then is there no ticket.
        let ticket = if plan.request_id == 0 {
            [0u8; TICKET_LEN]
        } else {
            plan.ticket.clone().try_into().map_err(|_| {
                ClientError::Protocol(format!(
                    "a transfer plan carries a {} byte ticket, not {TICKET_LEN}",
                    plan.ticket.len()
                ))
            })?
        };

        let plan = Plan {
            request_id: plan.request_id,
            ticket,
            dtype,
            shape,
            total_bytes: plan.total_bytes,
            applied_quality: quality_from_proto(plan.applied_quality.as_ref()),
            requested_quality: requested_quality.clone(),
            inline_data: plan.inline_data,
        };
        plan.check()?;
        Ok(plan)
    }

    /// Whether the data came back with the plan.
    pub fn is_inline(&self) -> bool {
        self.request_id == 0
    }

    /// Reject a plan that does not describe itself consistently.
    fn check(&self) -> Result<()> {
        let elements: u64 = self.shape.iter().copied().product();
        let expected = elements.saturating_mul(self.dtype.itemsize());
        if expected != self.total_bytes {
            return Err(ClientError::Protocol(format!(
                "a plan for {:?} of {} says it is {} bytes, not {expected}",
                self.shape, self.dtype, self.total_bytes
            )));
        }
        if self.is_inline() && self.inline_data.len() as u64 != self.total_bytes {
            return Err(ClientError::Protocol(format!(
                "a plan carrying its data inline holds {} of {} bytes",
                self.inline_data.len(),
                self.total_bytes
            )));
        }
        Ok(())
    }

    /// Cut the logical byte stream into the chunks to fetch.
    ///
    /// Each chunk is `(offset, len)` in the logical byte stream, which is also
    /// its position in the output buffer, so a chunk can be received on any
    /// connection and still land in the right place.
    pub(crate) fn chunks(&self, chunk_bytes: u64) -> impl Iterator<Item = (u64, u64)> + '_ {
        let chunk_bytes = chunk_bytes.max(1);
        (0..self.total_bytes)
            .step_by(chunk_bytes as usize)
            .map(move |offset| (offset, chunk_bytes.min(self.total_bytes - offset)))
    }
}

/// How a transfer went.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransferResult {
    pub bytes: u64,
    /// Payload bytes actually read off the data plane. Equal to `bytes` for a
    /// lossless transfer, less for an encoded one, and 0 for an inline one.
    pub wire_bytes: u64,
    pub elapsed: Duration,
    /// 0 for a transfer answered inline.
    pub chunks: u32,
    /// Connections used. 0 for a transfer answered inline, which never touches
    /// the data plane at all.
    pub streams: u32,
    /// Fetches that had to be repeated.
    pub retries: u32,
    /// Whether the data came back with the plan, in one round trip.
    pub inline: bool,
}

impl TransferResult {
    /// Logical bytes per byte sent. 1.0 when nothing was encoded, and 0.0
    /// when there is nothing to divide.
    pub fn compression_ratio(&self) -> f64 {
        if self.wire_bytes == 0 {
            return 0.0;
        }
        self.bytes as f64 / self.wire_bytes as f64
    }

    /// Throughput in mebibytes per second, for benchmarks.
    pub fn throughput_mib_per_sec(&self) -> f64 {
        let seconds = self.elapsed.as_secs_f64();
        if seconds <= 0.0 {
            return 0.0;
        }
        self.bytes as f64 / seconds / (1024.0 * 1024.0)
    }
}

/// What a client has transferred since it connected.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ClientStats {
    pub bytes: u64,
    /// Time spent filling buffers, summed over transfers.
    pub elapsed: Duration,
    pub chunks: u64,
    pub retries: u64,
    /// Data connections held.
    pub streams: u32,
    /// Fastest control plane call seen, which bounds the round trip from above.
    pub rtt: Duration,
}

impl ClientStats {
    pub(crate) fn add(&mut self, transfer: &TransferResult) {
        self.bytes += transfer.bytes;
        self.elapsed += transfer.elapsed;
        self.chunks += u64::from(transfer.chunks);
        self.retries += u64::from(transfer.retries);
    }

    pub fn throughput_mib_per_sec(&self) -> f64 {
        let seconds = self.elapsed.as_secs_f64();
        if seconds <= 0.0 {
            return 0.0;
        }
        self.bytes as f64 / seconds / (1024.0 * 1024.0)
    }
}

/// An array fetched from a server, as raw bytes.
///
/// C order, little-endian: the logical byte stream exactly as it travelled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArrayData {
    pub dtype: DType,
    pub shape: Vec<u64>,
    pub bytes: Vec<u8>,
    pub transfer: TransferResult,
}

/// An array fetched from a server, as elements.
#[derive(Debug, Clone, PartialEq)]
pub struct TypedArray<T> {
    pub shape: Vec<u64>,
    /// C order, as numpy would have it.
    pub data: Vec<T>,
    pub transfer: TransferResult,
}

mod sealed {
    pub trait Sealed {}
}

/// A Rust type a selection can be read into directly.
///
/// Sealed on purpose. Reading into a `Vec<T>` writes whatever the server sent
/// over the elements, which is only sound for a type where every bit pattern is
/// a value. `bool` is the counterexample — a byte other than 0 or 1 in a Rust
/// `bool` is undefined behaviour — and `float16` and the complex types have no
/// std type to be. Those go through [`ArrayData`], and through numpy once the
/// Python bindings exist.
pub trait Element: sealed::Sealed + Copy + Default {
    const DTYPE: DType;
}

macro_rules! element {
    ($($ty:ty => $dtype:ident),* $(,)?) => {
        $(
            impl sealed::Sealed for $ty {}
            impl Element for $ty {
                const DTYPE: DType = DType::$dtype;
            }
        )*
    };
}

element! {
    i8 => Int8, i16 => Int16, i32 => Int32, i64 => Int64,
    u8 => Uint8, u16 => Uint16, u32 => Uint32, u64 => Uint64,
    f32 => Float32, f64 => Float64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn proto_plan(total_bytes: u64, shape: Vec<i64>) -> aex_proto::TransferPlan {
        aex_proto::TransferPlan {
            request_id: 7,
            ticket: vec![9u8; TICKET_LEN],
            dtype: DType::Float32.as_i32(),
            shape,
            total_bytes,
            ..aex_proto::TransferPlan::default()
        }
    }

    #[test]
    fn a_plan_arrives_with_its_shape_and_ticket() {
        let plan =
            Plan::from_proto(proto_plan(4000, vec![10, 100]), &QualitySpec::exact()).expect("plan");
        assert_eq!(plan.request_id, 7);
        assert_eq!(plan.ticket, [9u8; TICKET_LEN]);
        assert_eq!(plan.shape, vec![10, 100]);
        assert_eq!(plan.total_bytes, 4000);
        assert!(!plan.is_inline());
    }

    #[test]
    fn a_plan_says_what_quality_was_applied() {
        let requested = QualitySpec {
            encoding: aex_core::Encoding::Subsample,
            subsample_step: vec![2, 2],
            ..QualitySpec::exact()
        };
        let mut proto = proto_plan(4000, vec![10, 100]);
        proto.applied_quality = Some(aex_proto::convert::quality_to_proto(&QualitySpec::exact()));
        let plan = Plan::from_proto(proto, &requested).expect("plan");
        assert!(plan.applied_quality.is_exact());
        assert_eq!(plan.requested_quality, requested);
    }

    #[test]
    fn a_small_selection_arrives_with_its_data() {
        let plan = Plan::from_proto(
            aex_proto::TransferPlan {
                request_id: 0,
                ticket: Vec::new(),
                dtype: DType::Int32.as_i32(),
                shape: vec![4],
                total_bytes: 16,
                inline_data: vec![1u8; 16],
                ..aex_proto::TransferPlan::default()
            },
            &QualitySpec::exact(),
        )
        .expect("plan");
        assert!(plan.is_inline());
        assert_eq!(plan.inline_data, vec![1u8; 16]);
        // No ticket is issued for one of these, and none is needed.
        assert_eq!(plan.ticket, [0u8; TICKET_LEN]);
    }

    #[test]
    fn a_plan_that_contradicts_itself_is_refused() {
        // The length has to follow from the shape and the dtype; if it does not,
        // the client would allocate one size and be sent another.
        let err =
            Plan::from_proto(proto_plan(4001, vec![10, 100]), &QualitySpec::exact()).unwrap_err();
        assert!(matches!(err, ClientError::Protocol(_)), "{err}");

        // A ticket of the wrong length means the two sides disagree about the
        // handshake, not that this one transfer is unlucky.
        let mut short = proto_plan(4000, vec![10, 100]);
        short.ticket = vec![1, 2, 3];
        assert!(Plan::from_proto(short, &QualitySpec::exact()).is_err());

        // Inline data that is not all there.
        let truncated = aex_proto::TransferPlan {
            request_id: 0,
            ticket: Vec::new(),
            dtype: DType::Uint8.as_i32(),
            shape: vec![8],
            total_bytes: 8,
            inline_data: vec![0u8; 4],
            ..aex_proto::TransferPlan::default()
        };
        assert!(Plan::from_proto(truncated, &QualitySpec::exact()).is_err());

        // A dtype from a newer server.
        let mut unknown = proto_plan(4000, vec![10, 100]);
        unknown.dtype = 99;
        assert!(Plan::from_proto(unknown, &QualitySpec::exact()).is_err());
    }

    #[test]
    fn chunks_cover_the_stream_exactly_once() {
        for total in [0u64, 1, 255, 256, 257, 1024] {
            for chunk_bytes in [1u64, 7, 256, 4096] {
                let plan = Plan::from_proto(
                    proto_plan(total * 4, vec![total as i64]),
                    &QualitySpec::exact(),
                )
                .unwrap();
                let chunks: Vec<_> = plan.chunks(chunk_bytes).collect();

                let mut next = 0;
                for (offset, len) in &chunks {
                    assert_eq!(*offset, next, "a gap or an overlap at {offset}");
                    assert!(*len > 0 && *len <= chunk_bytes);
                    next += len;
                }
                assert_eq!(next, plan.total_bytes, "{total} bytes in {chunk_bytes}");
                // Only the last chunk may be short.
                if chunks.len() > 1 {
                    assert!(chunks[..chunks.len() - 1]
                        .iter()
                        .all(|(_, len)| *len == chunk_bytes));
                }
            }
        }
    }

    #[test]
    fn an_empty_selection_has_nothing_to_fetch() {
        let plan = Plan::from_proto(proto_plan(0, vec![0]), &QualitySpec::exact()).unwrap();
        assert_eq!(plan.chunks(4096).count(), 0);
    }

    #[test]
    fn throughput_is_reported_from_the_bytes_and_the_time() {
        let result = TransferResult {
            bytes: 1024 * 1024,
            wire_bytes: 1024 * 1024,
            elapsed: Duration::from_millis(500),
            chunks: 1,
            streams: 1,
            retries: 0,
            inline: false,
        };
        assert!((result.throughput_mib_per_sec() - 2.0).abs() < 1e-9);

        // A transfer too fast to measure reports nothing rather than infinity.
        let instant = TransferResult {
            elapsed: Duration::ZERO,
            ..result
        };
        assert_eq!(instant.throughput_mib_per_sec(), 0.0);
    }

    #[test]
    fn stats_add_up_transfers() {
        let transfer = TransferResult {
            bytes: 1024 * 1024,
            wire_bytes: 512 * 1024,
            elapsed: Duration::from_millis(250),
            chunks: 3,
            streams: 2,
            retries: 1,
            inline: false,
        };
        let mut stats = ClientStats::default();
        assert_eq!(stats.throughput_mib_per_sec(), 0.0);
        stats.add(&transfer);
        stats.add(&transfer);
        assert_eq!(stats.bytes, 2 * 1024 * 1024);
        assert_eq!(stats.chunks, 6);
        assert_eq!(stats.retries, 2);
        assert!((stats.throughput_mib_per_sec() - 4.0).abs() < 1e-9);
    }

    #[test]
    fn every_element_type_knows_its_dtype() {
        assert_eq!(<i8 as Element>::DTYPE, DType::Int8);
        assert_eq!(<u64 as Element>::DTYPE, DType::Uint64);
        assert_eq!(<f32 as Element>::DTYPE, DType::Float32);
        assert_eq!(<f64 as Element>::DTYPE, DType::Float64);
        // Reading into a Vec<T> writes the server's bytes over the elements, so
        // the size on this side has to be the size on the wire.
        assert_eq!(
            std::mem::size_of::<f64>() as u64,
            <f64 as Element>::DTYPE.itemsize()
        );
        assert_eq!(
            std::mem::size_of::<i16>() as u64,
            <i16 as Element>::DTYPE.itemsize()
        );
    }
}
