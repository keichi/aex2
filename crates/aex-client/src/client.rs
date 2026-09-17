//! The client.
//!
//! The API is blocking, with a tokio runtime kept inside. Callers are analysis
//! code and, later, Python: neither wants to own a runtime, and the Python
//! bindings will release the GIL around exactly these blocking calls.
//!
//! Reading a selection costs one round trip when it is small enough to come
//! back with its plan, and two when it is not. The second is the data plane,
//! where the bytes go from the kernel into the caller's buffer without passing
//! through protobuf or through a copy of this client's making — which is why
//! [`Client::read_selection_into`] takes the buffer rather than returning one.

use std::net::IpAddr;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use aex_core::{DType, Index};
use aex_proto::aex_control_client::AexControlClient;
use aex_proto::convert::{check_fancy_limit, indices_to_proto, quality_to_proto};
use aex_proto::{
    CloseFileRequest, ConnectRequest, DisconnectRequest, GetItemRequest, ListChildrenRequest,
    OpenFileRequest, PrepareSelectionRequest,
};
use tokio::runtime::Runtime;
use tonic::transport::{Channel, Endpoint};

use crate::config::ClientConfig;
use crate::error::{ClientError, Result};
use crate::pool::{ConnSettings, DataPool, FetchSpec};
use crate::transfer::{ArrayData, ClientStats, Element, Plan, TransferResult, TypedArray};

/// The data plane frame version this client speaks.
pub const PROTOCOL_VERSION: u32 = 1;

/// A file opened on the server.
///
/// Only meaningful within the session that opened it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FileHandle(u64);

impl FileHandle {
    /// The handle as it travels on the wire.
    pub fn as_u64(self) -> u64 {
        self.0
    }

    /// A handle that went out through [`FileHandle::as_u64`], such as one kept
    /// by the Python layer.
    pub fn from_u64(handle: u64) -> Self {
        FileHandle(handle)
    }
}

/// What lives at a path in a file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Item {
    Dataset(DatasetInfo),
    Group,
}

/// An array's metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatasetInfo {
    pub dtype: DType,
    pub shape: Vec<u64>,
}

impl DatasetInfo {
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Number of elements; 1 for a scalar array.
    pub fn num_elements(&self) -> u64 {
        self.shape.iter().copied().product()
    }
}

/// What the server granted this session.
#[derive(Clone)]
pub struct SessionInfo {
    pub id: Vec<u8>,
    /// Authenticates this client's data connections. Never log it.
    pub token: Vec<u8>,
    /// Host and port of the data plane, with the host already resolved: the
    /// server may answer with an empty one, meaning "wherever you reached me".
    pub data_endpoint: (String, u16),
    pub granted_streams: u32,
    pub protocol_version: u32,
    pub default_chunk_bytes: u64,
    pub max_fetch_bytes: u64,
    pub supported_codecs: u32,
    pub supported_encodings: u32,
    /// Ceiling on the expanded indices of one fancy selection, so that an
    /// oversized one is refused here rather than after a round trip.
    pub max_fancy_indices: u64,
}

impl std::fmt::Debug for SessionInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The token is a capability, so it is redacted rather than printed:
        // debug output ends up in logs and bug reports.
        f.debug_struct("SessionInfo")
            .field("id", &HexBytes(&self.id))
            .field("token", &"<redacted>")
            .field("data_endpoint", &self.data_endpoint)
            .field("granted_streams", &self.granted_streams)
            .field("protocol_version", &self.protocol_version)
            .field("default_chunk_bytes", &self.default_chunk_bytes)
            .field("max_fetch_bytes", &self.max_fetch_bytes)
            .field("supported_codecs", &self.supported_codecs)
            .field("supported_encodings", &self.supported_encodings)
            .field("max_fancy_indices", &self.max_fancy_indices)
            .finish()
    }
}

/// Renders bytes as hex in debug output.
struct HexBytes<'a>(&'a [u8]);

impl std::fmt::Debug for HexBytes<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for byte in self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

/// A connected client.
///
/// Dropping one does not tell the server: a disconnect on a dead connection
/// would block, and the session expires on its own idle timeout anyway. Call
/// [`Client::disconnect`] to release it now.
pub struct Client {
    // Declared before the runtime: the channel's tasks have to be dropped
    // while the runtime that owns them is still alive.
    control: AexControlClient<Channel>,
    runtime: Arc<Runtime>,
    pool: DataPool,
    session: SessionInfo,
    config: ClientConfig,
    stats: Mutex<ClientStats>,
}

impl std::fmt::Debug for Client {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Client")
            .field("session", &self.session)
            .field("pool", &self.pool)
            .field("config", &self.config)
            .finish()
    }
}

impl Client {
    /// Connect to a control plane and open a session.
    ///
    /// `url` is a gRPC endpoint such as `http://127.0.0.1:50051`.
    pub fn connect(url: &str, config: ClientConfig) -> Result<Self> {
        let runtime = Arc::new(
            tokio::runtime::Builder::new_multi_thread()
                // One worker is plenty for the control plane, and it keeps the
                // connection serviced between calls, which a current-thread
                // runtime would not.
                .worker_threads(1)
                .enable_all()
                .build()?,
        );

        let endpoint = Endpoint::from_shared(url.to_string())
            .map_err(|e| ClientError::BadRequest(format!("bad control plane url {url:?}: {e}")))?
            .connect_timeout(config.connect_timeout)
            .tcp_nodelay(true);
        let control_host = endpoint
            .uri()
            .host()
            .ok_or_else(|| ClientError::BadRequest(format!("url {url:?} has no host")))?
            .to_string();

        let channel = runtime.block_on(endpoint.connect())?;
        let mut control = AexControlClient::new(channel)
            .max_decoding_message_size(config.max_message_bytes)
            .max_encoding_message_size(config.max_message_bytes);

        let started = Instant::now();
        let reply = runtime
            .block_on(control.connect(ConnectRequest {
                protocol_version: PROTOCOL_VERSION,
                desired_streams: config.streams,
                client_name: config.client_name.clone(),
            }))?
            .into_inner();
        let rtt = started.elapsed();

        if reply.protocol_version != PROTOCOL_VERSION {
            return Err(ClientError::Protocol(format!(
                "server speaks data plane version {}, this client speaks {PROTOCOL_VERSION}",
                reply.protocol_version
            )));
        }
        let endpoint = reply.endpoints.first().ok_or_else(|| {
            ClientError::Protocol("server granted a session but no data endpoint".to_string())
        })?;
        let port = u16::try_from(endpoint.port).map_err(|_| {
            ClientError::Protocol(format!(
                "data endpoint port {} is not a port",
                endpoint.port
            ))
        })?;
        let (host, port) = match &config.data_endpoint {
            Some(pinned) => parse_endpoint(pinned)?,
            None if endpoint.host.is_empty() => (strip_brackets(&control_host).to_string(), port),
            None => (endpoint.host.clone(), port),
        };

        let session = SessionInfo {
            id: reply.session_id,
            token: reply.session_token,
            data_endpoint: (host, port),
            granted_streams: reply.granted_streams,
            protocol_version: reply.protocol_version,
            default_chunk_bytes: reply.default_chunk_bytes,
            max_fetch_bytes: reply.max_fetch_bytes,
            supported_codecs: reply.supported_codecs,
            supported_encodings: reply.supported_encodings,
            // Not advertised: both sides hold the same default, and a server
            // that lowers it still refuses what is over its own limit.
            max_fancy_indices: aex_proto::convert::DEFAULT_MAX_FANCY_INDICES,
        };

        // Opened now rather than at the first transfer, so that a data plane
        // which cannot be reached is reported here, and so that the first large
        // read does not pay for a handshake.
        let pool = DataPool::connect(
            ConnSettings {
                host: session.data_endpoint.0.clone(),
                port: session.data_endpoint.1,
                session_id: as_16_bytes(&session.id, "session id")?,
                session_token: as_16_bytes(&session.token, "session token")?,
                nodelay: config.tcp_nodelay,
                rcvbuf: config.rcvbuf,
                connect_timeout: config.connect_timeout,
            },
            session.granted_streams,
        )?;

        let stats = Mutex::new(ClientStats {
            streams: pool.streams(),
            rtt,
            ..ClientStats::default()
        });
        Ok(Client {
            control,
            runtime,
            pool,
            session,
            config,
            stats,
        })
    }

    pub fn session(&self) -> &SessionInfo {
        &self.session
    }

    pub fn config(&self) -> &ClientConfig {
        &self.config
    }

    /// Totals over every transfer this client has made.
    pub fn stats(&self) -> ClientStats {
        self.stats.lock().unwrap_or_else(|p| p.into_inner()).clone()
    }

    /// Open a file, letting the server infer the format from the name.
    pub fn open(&self, path: &str) -> Result<FileHandle> {
        self.open_as(path, "")
    }

    /// Open a file in a named format, such as `npy`.
    pub fn open_as(&self, path: &str, format: &str) -> Result<FileHandle> {
        let reply = self.call(|mut control| async move {
            control
                .open_file(OpenFileRequest {
                    session_id: self.session.id.clone(),
                    path: path.to_string(),
                    format: format.to_string(),
                })
                .await
        })?;
        Ok(FileHandle(reply.handle))
    }

    /// Close a file. The server forgets the handle.
    pub fn close(&self, handle: FileHandle) -> Result<()> {
        self.call(|mut control| async move {
            control
                .close_file(CloseFileRequest {
                    session_id: self.session.id.clone(),
                    handle: handle.0,
                })
                .await
        })?;
        Ok(())
    }

    /// The item at `name`; `/` is the root group.
    pub fn get_item(&self, handle: FileHandle, name: &str) -> Result<Item> {
        let reply = self.call(|mut control| async move {
            control
                .get_item(GetItemRequest {
                    session_id: self.session.id.clone(),
                    handle: handle.0,
                    name: name.to_string(),
                })
                .await
        })?;
        Ok(item_from_proto(&reply)?.1)
    }

    /// The children of the group at `name`, with their names.
    pub fn list_children(&self, handle: FileHandle, name: &str) -> Result<Vec<(String, Item)>> {
        let reply = self.call(|mut control| async move {
            control
                .list_children(ListChildrenRequest {
                    session_id: self.session.id.clone(),
                    handle: handle.0,
                    name: name.to_string(),
                })
                .await
        })?;
        reply.items.iter().map(item_from_proto).collect()
    }

    /// Read a selection into a buffer the caller owns.
    ///
    /// `dst` has to be exactly the length of the selection, which the caller
    /// learns from the metadata or from a previous read. Taking the buffer
    /// rather than returning one is the point: it is what lets the bytes go
    /// from the kernel into a numpy array with nothing in between.
    pub fn read_selection_into(
        &self,
        handle: FileHandle,
        name: &str,
        indices: &[Index],
        dst: &mut [u8],
    ) -> Result<TransferResult> {
        let plan = self.prepare(handle, name, indices)?;
        self.fill(&plan, handle, name, indices, dst)
    }

    /// Read a selection, allocating for it.
    ///
    /// The bytes are the logical byte stream: C order, little-endian, exactly
    /// as they travelled.
    pub fn read_selection(
        &self,
        handle: FileHandle,
        name: &str,
        indices: &[Index],
    ) -> Result<ArrayData> {
        let plan = self.prepare(handle, name, indices)?;
        let (dtype, shape) = (plan.dtype, plan.shape.clone());
        let mut bytes = vec![0u8; plan.total_bytes as usize];
        let transfer = self.fill(&plan, handle, name, indices, &mut bytes)?;
        Ok(ArrayData {
            dtype,
            shape,
            bytes,
            transfer,
        })
    }

    /// Read a selection as elements, checking that the dtype is the one asked
    /// for.
    ///
    /// Only for the types where every bit pattern is a value; see [`Element`].
    pub fn read_selection_as<T: Element>(
        &self,
        handle: FileHandle,
        name: &str,
        indices: &[Index],
    ) -> Result<TypedArray<T>> {
        let plan = self.prepare(handle, name, indices)?;
        if plan.dtype != T::DTYPE {
            return Err(ClientError::BadRequest(format!(
                "the selection is {} and was asked for as {}",
                plan.dtype,
                T::DTYPE
            )));
        }

        let shape = plan.shape.clone();
        let total_bytes = plan.total_bytes as usize;
        let mut data = vec![T::default(); total_bytes / std::mem::size_of::<T>()];
        // Element is sealed to the types where writing arbitrary bytes over an
        // element is defined, and every target is little-endian, so the wire
        // form and the in-memory form are the same bytes.
        let dst =
            unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, total_bytes) };
        let transfer = self.fill(&plan, handle, name, indices, dst)?;
        Ok(TypedArray {
            shape,
            data,
            transfer,
        })
    }

    /// Ask the server to resolve a selection, in one round trip.
    ///
    /// The plan says what the result will be, so the caller can allocate for
    /// it before [`Client::fill`].
    pub fn prepare(&self, handle: FileHandle, name: &str, indices: &[Index]) -> Result<Plan> {
        // Checked here so that a selection too large to travel does not cost a
        // round trip to be told so.
        check_fancy_limit(indices, self.session.max_fancy_indices)
            .map_err(|e| ClientError::BadRequest(e.to_string()))?;

        let reply = self.call(|mut control| async move {
            control
                .prepare_selection(PrepareSelectionRequest {
                    session_id: self.session.id.clone(),
                    handle: handle.0,
                    name: name.to_string(),
                    indices: indices_to_proto(indices),
                    // Lossless and uncompressed: nothing else is implemented on
                    // either side yet, and the plan says what was applied.
                    requested_quality: Some(quality_to_proto(&aex_core::QualitySpec::exact())),
                    requested_codec: aex_core::Codec::Raw.as_u32(),
                })
                .await
        })?;
        Plan::from_proto(reply)
    }

    /// Fill `dst` from a plan, preparing again once if the plan has gone.
    ///
    /// `dst` has to be exactly `plan.total_bytes` long. The selection is taken
    /// again because a plan can be evicted while the client is still working
    /// through its chunks, and then the client just asks for another one;
    /// twice in a row would mean something other than eviction.
    pub fn fill(
        &self,
        plan: &Plan,
        handle: FileHandle,
        name: &str,
        indices: &[Index],
        dst: &mut [u8],
    ) -> Result<TransferResult> {
        let result = self.fill_once(plan, handle, name, indices, dst)?;
        self.stats
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .add(&result);
        Ok(result)
    }

    fn fill_once(
        &self,
        plan: &Plan,
        handle: FileHandle,
        name: &str,
        indices: &[Index],
        dst: &mut [u8],
    ) -> Result<TransferResult> {
        if plan.total_bytes != dst.len() as u64 {
            return Err(ClientError::BadRequest(format!(
                "the selection is {} bytes and the buffer is {}",
                plan.total_bytes,
                dst.len()
            )));
        }
        let started = Instant::now();

        if plan.is_inline() {
            dst.copy_from_slice(&plan.inline_data);
            return Ok(TransferResult {
                bytes: plan.total_bytes,
                elapsed: started.elapsed(),
                chunks: 0,
                streams: 0,
                retries: 0,
                inline: true,
            });
        }

        let chunk_bytes = self.chunk_bytes(plan.total_bytes);
        let chunks = plan.chunks(chunk_bytes).count() as u32;
        let mut plan = plan.clone();
        let mut retries = 0;
        loop {
            let spec = FetchSpec {
                request_id: plan.request_id,
                ticket: &plan.ticket,
                credit: self.config.credit,
                max_retries: self.config.max_retries,
            };
            match self.pool.fetch(spec, plan.chunks(chunk_bytes), dst) {
                Ok(fetched) => {
                    return Ok(TransferResult {
                        bytes: plan.total_bytes,
                        elapsed: started.elapsed(),
                        chunks,
                        streams: fetched.streams,
                        retries: retries + fetched.retries,
                        inline: false,
                    })
                }
                Err(e) if e.needs_reprepare() && retries == 0 => {
                    let fresh = self.prepare(handle, name, indices)?;
                    if fresh.total_bytes != plan.total_bytes {
                        // The file changed underneath the selection; the buffer
                        // no longer fits what is being sent.
                        return Err(ClientError::BadRequest(format!(
                            "the selection was {} bytes and is now {}",
                            plan.total_bytes, fresh.total_bytes
                        )));
                    }
                    plan = fresh;
                    retries += 1;
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// How much of the logical byte stream one fetch asks for.
    ///
    /// The client decides this, not the server: what makes a good chunk size —
    /// how many connections there are, what the round trip and the bandwidth
    /// are — is all known on this side. The server only states a ceiling.
    fn chunk_bytes(&self, total_bytes: u64) -> u64 {
        let ceiling = self.session.max_fetch_bytes.max(1);
        if self.config.chunk_bytes != 0 {
            // A benchmark sweeping this wants exactly what it asked for.
            return self.config.chunk_bytes.clamp(1, ceiling);
        }
        split_for_streams(
            self.session.default_chunk_bytes.clamp(1, ceiling),
            total_bytes,
            self.pool.streams(),
        )
    }

    /// Release the session and everything it holds.
    pub fn disconnect(self) -> Result<()> {
        let session_id = self.session.id.clone();
        self.call(|mut control| async move {
            control.disconnect(DisconnectRequest { session_id }).await
        })?;
        Ok(())
    }

    /// Run one RPC to completion.
    fn call<F, T>(&self, rpc: impl FnOnce(AexControlClient<Channel>) -> F) -> Result<T>
    where
        F: std::future::Future<Output = std::result::Result<tonic::Response<T>, tonic::Status>>,
    {
        // The generated client wants &mut self; cloning it is cheap and shares
        // the one connection.
        let started = Instant::now();
        let response = self.runtime.block_on(rpc(self.control.clone()))?;
        let took = started.elapsed();
        let mut stats = self.stats.lock().unwrap_or_else(|p| p.into_inner());
        stats.rtt = stats.rtt.min(took);
        Ok(response.into_inner())
    }
}

/// Chunks are not cut smaller than this to keep extra connections busy: past
/// here the headers and syscalls start to cost more than the parallelism gains.
const MIN_CHUNK_BYTES: u64 = 256 * 1024;

/// Shrink `chunk_bytes` so that a transfer has a chunk for every connection,
/// down to [`MIN_CHUNK_BYTES`].
fn split_for_streams(chunk_bytes: u64, total_bytes: u64, streams: u32) -> u64 {
    let streams = u64::from(streams.max(1));
    if total_bytes.div_ceil(chunk_bytes) >= streams {
        return chunk_bytes;
    }
    total_bytes
        .div_ceil(streams)
        .max(MIN_CHUNK_BYTES)
        .min(chunk_bytes)
}

/// Convert one item off the wire.
fn item_from_proto(item: &aex_proto::Item) -> Result<(String, Item)> {
    let data = item
        .data
        .as_ref()
        .ok_or_else(|| ClientError::Protocol(format!("item {:?} is neither kind", item.name)))?;

    let converted = match data {
        aex_proto::item::Data::Group(_) => Item::Group,
        aex_proto::item::Data::Dataset(dataset) => {
            let dtype =
                DType::from_i32(dataset.dtype).map_err(|e| ClientError::Protocol(e.to_string()))?;
            let shape = dataset
                .shape
                .iter()
                .map(|&n| {
                    u64::try_from(n).map_err(|_| {
                        ClientError::Protocol(format!("negative axis length {n} in a shape"))
                    })
                })
                .collect::<Result<Vec<u64>>>()?;
            if dataset.ndim as usize != shape.len() {
                return Err(ClientError::Protocol(format!(
                    "dataset {:?} says it has {} dimensions but its shape has {}",
                    item.name,
                    dataset.ndim,
                    shape.len()
                )));
            }
            Item::Dataset(DatasetInfo { dtype, shape })
        }
    };
    Ok((item.name.clone(), converted))
}

/// Read a 16-byte identifier out of what the server sent.
fn as_16_bytes(bytes: &[u8], what: &str) -> Result<[u8; 16]> {
    bytes
        .try_into()
        .map_err(|_| ClientError::Protocol(format!("{what} is {} bytes, expected 16", bytes.len())))
}

/// Split `host:port`, accepting `[v6]:port` too.
fn parse_endpoint(endpoint: &str) -> Result<(String, u16)> {
    endpoint
        .rsplit_once(':')
        .and_then(|(host, port)| Some((strip_brackets(host).to_string(), port.parse().ok()?)))
        .filter(|(host, _)| !host.is_empty())
        .ok_or_else(|| {
            ClientError::BadRequest(format!("data endpoint {endpoint:?} is not host:port"))
        })
}

/// Strip the brackets a URI puts around an IPv6 host.
///
/// The data plane connects with a socket address, not a URI, so it needs the
/// address as the resolver spells it.
fn strip_brackets(host: &str) -> &str {
    host.strip_prefix('[')
        .and_then(|h| h.strip_suffix(']'))
        .filter(|h| h.parse::<IpAddr>().is_ok())
        .unwrap_or(host)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dataset(dtype: DType, shape: Vec<i64>) -> aex_proto::Item {
        aex_proto::Item {
            name: "array".to_string(),
            data: Some(aex_proto::item::Data::Dataset(aex_proto::Dataset {
                dtype: dtype.as_i32(),
                ndim: shape.len() as i32,
                shape,
            })),
        }
    }

    #[test]
    fn a_dataset_arrives_with_its_metadata() {
        let (name, item) = item_from_proto(&dataset(DType::Float32, vec![1000, 200])).unwrap();
        assert_eq!(name, "array");
        let Item::Dataset(info) = item else {
            panic!("expected a dataset");
        };
        assert_eq!(info.dtype, DType::Float32);
        assert_eq!(info.shape, vec![1000, 200]);
        assert_eq!(info.ndim(), 2);
        assert_eq!(info.num_elements(), 200_000);
    }

    #[test]
    fn a_scalar_array_arrives_with_an_empty_shape() {
        let (_, item) = item_from_proto(&dataset(DType::Int64, vec![])).unwrap();
        let Item::Dataset(info) = item else {
            panic!("expected a dataset");
        };
        assert_eq!(info.ndim(), 0);
        // An empty product is one element, as numpy counts it.
        assert_eq!(info.num_elements(), 1);
    }

    #[test]
    fn a_group_arrives_as_a_group() {
        let item = aex_proto::Item {
            name: "/".to_string(),
            data: Some(aex_proto::item::Data::Group(aex_proto::Group {})),
        };
        assert_eq!(
            item_from_proto(&item).unwrap(),
            ("/".to_string(), Item::Group)
        );
    }

    #[test]
    fn a_reply_that_makes_no_sense_is_a_protocol_error() {
        // Neither kind set: an older or broken server.
        let empty = aex_proto::Item {
            name: "array".to_string(),
            data: None,
        };
        assert!(matches!(
            item_from_proto(&empty),
            Err(ClientError::Protocol(_))
        ));

        // A dtype this client does not know.
        let mut unknown = dataset(DType::Float32, vec![4]);
        if let Some(aex_proto::item::Data::Dataset(d)) = unknown.data.as_mut() {
            d.dtype = 99;
        }
        assert!(matches!(
            item_from_proto(&unknown),
            Err(ClientError::Protocol(_))
        ));

        // A negative axis length.
        assert!(matches!(
            item_from_proto(&dataset(DType::Int8, vec![-1])),
            Err(ClientError::Protocol(_))
        ));

        // ndim disagreeing with the shape means the two were built separately.
        let mut inconsistent = dataset(DType::Int8, vec![2, 3]);
        if let Some(aex_proto::item::Data::Dataset(d)) = inconsistent.data.as_mut() {
            d.ndim = 3;
        }
        assert!(matches!(
            item_from_proto(&inconsistent),
            Err(ClientError::Protocol(_))
        ));
    }

    #[test]
    fn debug_output_keeps_the_token_out_of_the_logs() {
        let session = SessionInfo {
            id: vec![0xab; 16],
            token: vec![0xcd; 16],
            data_endpoint: ("127.0.0.1".to_string(), 50052),
            granted_streams: 4,
            protocol_version: PROTOCOL_VERSION,
            default_chunk_bytes: 4 << 20,
            max_fetch_bytes: 16 << 20,
            supported_codecs: 1,
            supported_encodings: 1,
            max_fancy_indices: 262_144,
        };
        let rendered = format!("{session:?}");
        assert!(rendered.contains("abababab"), "{rendered}");
        assert!(!rendered.contains("cdcdcdcd"), "{rendered}");
        assert!(!rendered.contains("205"), "{rendered}");
    }

    #[test]
    fn an_ipv6_host_loses_its_uri_brackets() {
        assert_eq!(strip_brackets("[::1]"), "::1");
        assert_eq!(strip_brackets("127.0.0.1"), "127.0.0.1");
        assert_eq!(strip_brackets("example.org"), "example.org");
        // Brackets around something that is not an address are left alone.
        assert_eq!(strip_brackets("[host]"), "[host]");
    }

    #[test]
    fn a_small_transfer_is_split_to_keep_every_connection_busy() {
        const MIB: u64 = 1 << 20;
        // Plenty of chunks already.
        assert_eq!(split_for_streams(4 * MIB, 64 * MIB, 4), 4 * MIB);
        // Two chunks for four connections becomes four.
        assert_eq!(split_for_streams(4 * MIB, 8 * MIB, 4), 2 * MIB);
        assert_eq!(split_for_streams(4 * MIB, 8 * MIB + 1, 4), 2 * MIB + 1);
        // Not below the floor, even if some connections then sit idle.
        assert_eq!(split_for_streams(4 * MIB, MIB, 16), MIN_CHUNK_BYTES);
        // Never above what was asked for.
        assert_eq!(split_for_streams(128 * 1024, 200 * 1024, 4), 128 * 1024);
    }

    #[test]
    fn a_pinned_data_endpoint_is_host_and_port() {
        let parsed = |s| parse_endpoint(s).ok();
        assert_eq!(parsed("tunnel:6000"), Some(("tunnel".to_string(), 6000)));
        assert_eq!(parsed("[::1]:6000"), Some(("::1".to_string(), 6000)));
        assert_eq!(parsed("tunnel"), None);
        assert_eq!(parsed(":6000"), None);
        assert_eq!(parsed("tunnel:port"), None);
    }
}
