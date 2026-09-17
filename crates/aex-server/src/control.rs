//! The gRPC control plane.
//!
//! It decides *what* to send: sessions, files, metadata, and the resolution of
//! a selection into a transfer plan. The bulk data never passes through here.
//!
//! A selection small enough to fit the inline limit is answered with its data
//! attached instead of a plan. Without that path, a small interactive read
//! would cost two round trips where v1 needed one, and making a small read
//! slower in order to make a large one faster is the wrong trade for a system
//! whose main complaint about v1 is latency.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use aex_core::{ArrayDataset, ArrayFile, Codec, Item, NpyFile, NullFile, SelectionLayout};
use aex_proto::aex_control_server::AexControl;
use aex_proto::convert::{indices_from_proto, quality_from_proto, quality_to_proto};
use aex_proto::transfer_plan_or_error;
use aex_proto::{
    ApplyFunctionReply, ApplyFunctionRequest, CloseFileReply, CloseFileRequest, ConnectReply,
    ConnectRequest, DataEndpoint, Dataset, DisconnectReply, DisconnectRequest, GetItemRequest,
    Group, ItemList, ListChildrenRequest, OpenFileReply, OpenFileRequest, PlanError,
    PrepareSelectionRequest, PrepareSelectionsRequest, TransferPlan, TransferPlanList,
    TransferPlanOrError,
};
use tonic::{Request, Response, Status};

use crate::config::{ServerConfig, PROTOCOL_VERSION, SUPPORTED_CODECS, SUPPORTED_ENCODINGS};
use crate::error::{Result, ServerError};
use crate::paths::PathPolicy;
use crate::session::SessionRegistry;
use crate::transfer::TransferRegistry;

const NPY_FORMAT: &str = "npy";

/// Names a client or a file extension may use for HDF5. netCDF-4 is HDF5
/// underneath, so it is served by the same backend.
const HDF5_FORMATS: [&str; 5] = ["hdf5", "h5", "he5", "nc", "netcdf4"];

/// Not a format: a dataset with no storage behind it, for measuring what the
/// transfer costs when reading the data costs nothing. Served only when the
/// server was started with it enabled.
const NULL_FORMAT: &str = "null";

pub struct ControlService {
    sessions: Arc<SessionRegistry>,
    transfers: Arc<TransferRegistry>,
    paths: Arc<PathPolicy>,
    config: Arc<ServerConfig>,
    /// The port the data plane really bound, which is not the configured one
    /// when that was 0.
    data_port: u16,
    /// Shared by every HDF5 file this server opens.
    #[cfg(feature = "hdf5")]
    decode_cache: Arc<aex_core::DecodeCache>,
    /// Whether the cache has been reported too small, so it is said once.
    warned_small_cache: std::sync::atomic::AtomicBool,
}

impl ControlService {
    pub fn new(
        sessions: Arc<SessionRegistry>,
        transfers: Arc<TransferRegistry>,
        paths: Arc<PathPolicy>,
        config: Arc<ServerConfig>,
        data_port: u16,
    ) -> Self {
        ControlService {
            sessions,
            transfers,
            paths,
            #[cfg(feature = "hdf5")]
            decode_cache: Arc::new(aex_core::DecodeCache::new(
                config.transfer.decode_cache_bytes,
            )),
            config,
            data_port,
            warned_small_cache: Default::default(),
        }
    }

    fn open(&self, request: &OpenFileRequest) -> Result<u64> {
        let session = self.sessions.get(&request.session_id)?;

        // The synthetic backend has no file, so it is settled before any path
        // is resolved: there is nothing on disk for a root to contain.
        if request.format.eq_ignore_ascii_case(NULL_FORMAT) {
            if !self.config.enable_null_backend {
                return Err(ServerError::BadRequest(format!(
                    "this server does not offer the {NULL_FORMAT:?} backend; it is for \
                     measuring the transfer path and has to be enabled deliberately"
                )));
            }
            let file: Arc<dyn ArrayFile> = Arc::new(NullFile::from_spec(&request.path)?);
            let handle = session.files().insert(file);
            tracing::debug!(
                session = %hex(session.id()),
                handle,
                spec = %request.path,
                "opened a synthetic dataset"
            );
            return Ok(handle);
        }

        let path = self.paths.resolve(&request.path)?;
        let format = if request.format.is_empty() {
            format_from_extension(&path)?
        } else {
            request.format.to_ascii_lowercase()
        };
        let file: Arc<dyn ArrayFile> = if format == NPY_FORMAT {
            Arc::new(NpyFile::open(&path)?)
        } else if HDF5_FORMATS.contains(&format.as_str()) {
            self.open_hdf5(&path)?
        } else {
            return Err(ServerError::BadRequest(format!(
                "format {format:?} is not supported; this server serves {NPY_FORMAT:?} and {:?}",
                HDF5_FORMATS[0]
            )));
        };
        let handle = session.files().insert(file);
        tracing::debug!(
            session = %hex(session.id()),
            handle,
            path = %path.display(),
            "opened file"
        );
        Ok(handle)
    }

    fn file_of(&self, session_id: &[u8], handle: u64) -> Result<Arc<dyn ArrayFile>> {
        let session = self.sessions.get(session_id)?;
        session.files().get(handle)
    }

    /// Resolve a selection into a plan, or into the data itself when it is
    /// small enough to travel inline and `inline_budget` still has room for it.
    fn prepare(
        &self,
        session_id: &[u8],
        request: &PrepareSelectionRequest,
        inline_budget: &mut u64,
    ) -> Result<TransferPlan> {
        let session = self.sessions.get(session_id)?;
        let file = session.files().get(request.handle)?;

        let Item::Dataset(dataset) = file.get_item(&request.name).map_err(ServerError::from)?
        else {
            return Err(ServerError::BadRequest(format!(
                "{:?} is a group; only a dataset can be transferred",
                request.name
            )));
        };

        let indices = indices_from_proto(&request.indices, self.config.limits.max_fancy_indices)?;
        // What the client asked for, and what this server can actually do. The
        // difference goes back in the plan rather than being an error, so that
        // a newer client still gets its data.
        let requested = quality_from_proto(request.requested_quality.as_ref());
        let applied = requested.applied();
        let codec = Codec::from_u32(request.requested_codec)
            .filter(|codec| codec.is_supported())
            .unwrap_or(Codec::Raw);

        let layout = dataset.layout(&indices, &applied)?;
        self.check_decode_cache(&*dataset, session.granted_streams());

        if layout.total_bytes <= self.config.transfer.inline_limit_bytes
            && layout.total_bytes <= *inline_budget
        {
            *inline_budget -= layout.total_bytes;
            let mut inline_data = vec![0u8; layout.total_bytes as usize];
            dataset.read_range(&layout, 0, &mut inline_data)?;
            tracing::debug!(
                session = %hex(session.id()),
                bytes = layout.total_bytes,
                "answered a selection inline"
            );
            // No request_id and no ticket: there is nothing left to fetch, and
            // a zero request_id is how the client knows that.
            return plan_reply(&layout, codec, &applied, 0, Vec::new(), 0, inline_data);
        }

        let entry = self
            .transfers
            .insert(*session.id(), dataset.clone(), layout)?;
        let expires = unix_ms_from_now(self.transfers.ttl_secs());
        tracing::debug!(
            session = %hex(session.id()),
            request_id = entry.request_id(),
            bytes = entry.layout().total_bytes,
            "prepared a transfer"
        );

        plan_reply(
            entry.layout(),
            codec,
            &applied,
            entry.request_id(),
            entry.ticket().to_vec(),
            expires,
            Vec::new(),
        )
    }
}

#[tonic::async_trait]
impl AexControl for ControlService {
    async fn connect(
        &self,
        request: Request<ConnectRequest>,
    ) -> std::result::Result<Response<ConnectReply>, Status> {
        let request = request.into_inner();
        if request.protocol_version != PROTOCOL_VERSION {
            // Refuse up front: a version mismatch would otherwise show up as a
            // malformed frame on the data plane, far from its cause.
            return Err(ServerError::BadRequest(format!(
                "client speaks data plane version {}, this server speaks {PROTOCOL_VERSION}",
                request.protocol_version
            ))
            .into());
        }

        let session = self
            .sessions
            .create(&request.client_name, request.desired_streams)?;
        tracing::info!(
            session = %hex(session.id()),
            client = %request.client_name,
            streams = session.granted_streams(),
            "session opened"
        );

        Ok(Response::new(ConnectReply {
            session_id: session.id().to_vec(),
            session_token: session.token().to_vec(),
            // One endpoint, and an empty host so the client reuses the address
            // it already reached the control plane on. The server cannot know
            // how the client addresses it through a NAT or a container.
            endpoints: vec![DataEndpoint {
                host: self.config.data_advertise_host.clone(),
                port: self.data_port as u32,
            }],
            granted_streams: session.granted_streams(),
            protocol_version: PROTOCOL_VERSION,
            default_chunk_bytes: self.config.transfer.default_chunk_bytes,
            supported_codecs: SUPPORTED_CODECS,
            supported_encodings: SUPPORTED_ENCODINGS,
            max_fetch_bytes: self.config.transfer.max_fetch_bytes,
        }))
    }

    async fn disconnect(
        &self,
        request: Request<DisconnectRequest>,
    ) -> std::result::Result<Response<DisconnectReply>, Status> {
        let request = request.into_inner();
        let session_id = self.sessions.remove(&request.session_id)?;
        let dropped = self.transfers.remove_session(&session_id);
        tracing::info!(
            session = %hex(&session_id),
            transfers = dropped,
            "session closed"
        );
        Ok(Response::new(DisconnectReply {}))
    }

    async fn open_file(
        &self,
        request: Request<OpenFileRequest>,
    ) -> std::result::Result<Response<OpenFileReply>, Status> {
        let handle = self.open(&request.into_inner())?;
        Ok(Response::new(OpenFileReply { handle }))
    }

    async fn close_file(
        &self,
        request: Request<CloseFileRequest>,
    ) -> std::result::Result<Response<CloseFileReply>, Status> {
        let request = request.into_inner();
        let session = self.sessions.get(&request.session_id)?;
        session.files().remove(request.handle)?;
        Ok(Response::new(CloseFileReply {}))
    }

    async fn get_item(
        &self,
        request: Request<GetItemRequest>,
    ) -> std::result::Result<Response<aex_proto::Item>, Status> {
        let request = request.into_inner();
        let file = self.file_of(&request.session_id, request.handle)?;
        let item = file.get_item(&request.name).map_err(ServerError::from)?;
        Ok(Response::new(item_to_proto(&request.name, &item)?))
    }

    async fn list_children(
        &self,
        request: Request<ListChildrenRequest>,
    ) -> std::result::Result<Response<ItemList>, Status> {
        let request = request.into_inner();
        let file = self.file_of(&request.session_id, request.handle)?;
        let children = file
            .list_children(&request.name)
            .map_err(ServerError::from)?;

        let items = children
            .iter()
            .map(|(name, item)| item_to_proto(name, item))
            .collect::<Result<Vec<_>>>()?;
        Ok(Response::new(ItemList { items }))
    }

    async fn prepare_selection(
        &self,
        request: Request<PrepareSelectionRequest>,
    ) -> std::result::Result<Response<TransferPlan>, Status> {
        let request = request.into_inner();
        let mut budget = u64::MAX;
        Ok(Response::new(self.prepare(
            &request.session_id,
            &request,
            &mut budget,
        )?))
    }

    async fn prepare_selections(
        &self,
        request: Request<PrepareSelectionsRequest>,
    ) -> std::result::Result<Response<TransferPlanList>, Status> {
        let request = request.into_inner();
        // A dead session fails the call rather than every element.
        self.sessions.get(&request.session_id)?;
        // Half the message limit, so that many small selections still fit one
        // reply with their plans; the rest go over the data plane.
        let mut budget = self.config.limits.grpc_max_message_bytes as u64 / 2;
        let results = request
            .requests
            .iter()
            .map(|element| {
                let result = match self.prepare(&request.session_id, element, &mut budget) {
                    Ok(plan) => transfer_plan_or_error::Result::Plan(plan),
                    Err(e) => transfer_plan_or_error::Result::Error(PlanError {
                        klass: e.class() as i32,
                        message: e.to_string(),
                    }),
                };
                TransferPlanOrError {
                    result: Some(result),
                }
            })
            .collect();
        Ok(Response::new(TransferPlanList { results }))
    }

    async fn apply_function(
        &self,
        _request: Request<ApplyFunctionRequest>,
    ) -> std::result::Result<Response<ApplyFunctionReply>, Status> {
        Err(not_implemented_yet(
            "ApplyFunction",
            "server-side reductions",
        ))
    }
}

/// Says what the RPC is waiting on, so that a client hitting one during
/// development is not left wondering whether it misdialled.
fn not_implemented_yet(rpc: &str, what: &str) -> Status {
    Status::unimplemented(format!(
        "{rpc} is part of {what}, which this server does not serve yet"
    ))
}

/// Assemble the reply, whichever way the selection is being answered.
fn plan_reply(
    layout: &SelectionLayout,
    codec: Codec,
    applied: &aex_core::QualitySpec,
    request_id: u32,
    ticket: Vec<u8>,
    expires_unix_ms: u64,
    inline_data: Vec<u8>,
) -> Result<TransferPlan> {
    Ok(TransferPlan {
        request_id,
        ticket,
        dtype: layout.dtype.as_i32(),
        shape: shape_to_proto(&layout.out_shape)?,
        total_bytes: layout.total_bytes,
        codec: codec.as_u32(),
        applied_quality: Some(quality_to_proto(applied)),
        expires_unix_ms,
        inline_data,
    })
}

/// Wall-clock milliseconds `secs` from now.
///
/// Only the client reads this, and only to tell a user how long it has. The
/// server times its own plans on a monotonic clock.
fn unix_ms_from_now(secs: u64) -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|since| since.as_millis() as u64)
        .unwrap_or(0)
        .saturating_add(secs.saturating_mul(1000))
}

impl ControlService {
    #[cfg(feature = "hdf5")]
    fn open_hdf5(&self, path: &std::path::Path) -> Result<Arc<dyn ArrayFile>> {
        Ok(Arc::new(aex_core::Hdf5File::open(
            path,
            self.decode_cache.clone(),
        )?))
    }

    #[cfg(not(feature = "hdf5"))]
    fn open_hdf5(&self, _path: &std::path::Path) -> Result<Arc<dyn ArrayFile>> {
        Err(ServerError::BadRequest(
            "this server was built without the hdf5 feature and cannot serve HDF5 or netCDF-4"
                .to_string(),
        ))
    }

    /// Warn once if streams reading one dataset would evict each other's
    /// chunks, which makes every chunk decode many times over.
    fn check_decode_cache(&self, dataset: &dyn ArrayDataset, streams: u32) {
        let Some(chunk_bytes) = dataset.decoded_chunk_bytes() else {
            return;
        };
        let wanted = chunk_bytes.saturating_mul(u64::from(streams));
        let capacity = self.config.transfer.decode_cache_bytes;
        if wanted > capacity
            && !self
                .warned_small_cache
                .swap(true, std::sync::atomic::Ordering::Relaxed)
        {
            tracing::warn!(
                chunk_bytes,
                streams,
                decode_cache_bytes = capacity,
                "transfer.decode_cache_bytes cannot hold one decoded chunk per stream; \
                 chunks will be decoded repeatedly"
            );
        }
    }
}

/// The backend a file extension asks for.
fn format_from_extension(path: &std::path::Path) -> Result<String> {
    match path.extension().and_then(|e| e.to_str()) {
        Some(ext) => Ok(ext.to_ascii_lowercase()),
        None => Err(ServerError::BadRequest(format!(
            "cannot tell the format of {} from its name; pass one explicitly",
            path.display()
        ))),
    }
}

fn item_to_proto(name: &str, item: &Item) -> Result<aex_proto::Item> {
    let data = match item {
        Item::Dataset(dataset) => aex_proto::item::Data::Dataset(dataset_to_proto(&**dataset)?),
        Item::Group => aex_proto::item::Data::Group(Group {}),
    };
    Ok(aex_proto::Item {
        name: name.to_string(),
        data: Some(data),
    })
}

fn dataset_to_proto(dataset: &dyn ArrayDataset) -> Result<Dataset> {
    let shape = shape_to_proto(dataset.shape())?;
    Ok(Dataset {
        dtype: dataset.dtype().as_i32(),
        ndim: shape.len() as i32,
        shape,
    })
}

/// The wire carries shapes as int64, as numpy does. A header can declare a
/// longer axis than that; such a file is not one we can describe, let alone
/// serve.
fn shape_to_proto(shape: &[u64]) -> Result<Vec<i64>> {
    shape
        .iter()
        .map(|&n| {
            i64::try_from(n).map_err(|_| {
                ServerError::Core(aex_core::AexError::MalformedNpy(format!(
                    "axis of {n} elements does not fit the int64 shape on the wire"
                )))
            })
        })
        .collect()
}

/// Hex for logs. Session ids are opaque, so they are shown as bytes.
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use aex_core::DType;

    use super::*;

    /// A dataset with the given metadata and nothing behind it.
    struct FakeDataset {
        dtype: DType,
        shape: Vec<u64>,
    }

    impl ArrayDataset for FakeDataset {
        fn dtype(&self) -> DType {
            self.dtype
        }
        fn shape(&self) -> &[u64] {
            &self.shape
        }
        fn read_range(
            &self,
            _layout: &aex_core::SelectionLayout,
            _offset: u64,
            _dst: &mut [u8],
        ) -> aex_core::Result<()> {
            // These tests only convert metadata; nothing reads from one.
            unimplemented!("a fake dataset has no bytes")
        }
    }

    #[test]
    fn a_dataset_crosses_the_wire_with_its_metadata() {
        let dataset = FakeDataset {
            dtype: DType::Float32,
            shape: vec![1000, 200],
        };
        let item = item_to_proto("array", &Item::Dataset(Arc::new(dataset))).expect("convert");

        assert_eq!(item.name, "array");
        let Some(aex_proto::item::Data::Dataset(dataset)) = item.data else {
            panic!("expected a dataset");
        };
        assert_eq!(dataset.dtype, DType::Float32.as_i32());
        assert_eq!(dataset.ndim, 2);
        assert_eq!(dataset.shape, vec![1000, 200]);
    }

    #[test]
    fn a_scalar_array_keeps_its_empty_shape() {
        let dataset = FakeDataset {
            dtype: DType::Int64,
            shape: vec![],
        };
        let item = item_to_proto("array", &Item::Dataset(Arc::new(dataset))).unwrap();
        let Some(aex_proto::item::Data::Dataset(dataset)) = item.data else {
            panic!("expected a dataset");
        };
        assert_eq!(dataset.ndim, 0);
        assert!(dataset.shape.is_empty());
    }

    #[test]
    fn a_group_crosses_the_wire_as_a_group() {
        let item = item_to_proto("/", &Item::Group).expect("convert");
        assert_eq!(item.name, "/");
        assert!(matches!(item.data, Some(aex_proto::item::Data::Group(_))));
    }

    #[test]
    fn an_axis_too_long_for_the_wire_is_rejected() {
        // Reachable only through a corrupt header: an empty array can declare
        // any axis length, since the product is zero either way.
        let dataset = FakeDataset {
            dtype: DType::Uint8,
            shape: vec![u64::MAX, 0],
        };
        let err = item_to_proto("array", &Item::Dataset(Arc::new(dataset))).unwrap_err();
        assert_eq!(err.class(), aex_core::ErrorClass::Permanent);
    }

    #[test]
    fn formats_come_from_the_extension_case_insensitively() {
        assert_eq!(
            format_from_extension(std::path::Path::new("/data/a.npy")).unwrap(),
            "npy"
        );
        assert_eq!(
            format_from_extension(std::path::Path::new("/data/A.NPY")).unwrap(),
            "npy"
        );
        assert!(format_from_extension(std::path::Path::new("/data/noext")).is_err());
    }

    #[test]
    fn hex_renders_a_session_id() {
        assert_eq!(hex(&[0x00, 0x0f, 0xff]), "000fff");
    }
}
