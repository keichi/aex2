//! The gRPC control plane.
//!
//! A selection under the inline limit is answered with its data attached, so a
//! small interactive read costs one round trip, not two.

use std::collections::HashMap;
use std::sync::Arc;

use aex_core::{
    ArrayDataset, ArrayFile, AttrValue, Axis, Codec, Function, Item, NpyFile, NullFile, ReduceArgs,
    SelectionLayout,
};
use aex_proto::aex_control_server::AexControl;
use aex_proto::convert::{indices_from_proto, quality_from_proto, quality_to_proto};
use aex_proto::function_argument::Value;
use aex_proto::transfer_plan_or_error;
use aex_proto::{
    ApplyFunctionReply, ApplyFunctionRequest, CloseFileReply, CloseFileRequest, ConnectReply,
    ConnectRequest, DataEndpoint, Dataset, DisconnectReply, DisconnectRequest, FunctionArgument,
    GetItemRequest, Group, ItemList, ListChildrenRequest, OpenFileReply, OpenFileRequest,
    PlanError, PrepareSelectionRequest, PrepareSelectionsRequest, TransferPlan, TransferPlanList,
    TransferPlanOrError,
};
use tonic::{Request, Response, Status};

use crate::config::{ServerConfig, PROTOCOL_VERSION};
use crate::error::{Result, ServerError};
use crate::paths::PathPolicy;
use crate::session::SessionRegistry;
use crate::transfer::TransferRegistry;

const NPY_FORMAT: &str = "npy";

/// Names a client or a file extension may use for HDF5. netCDF-4 is HDF5
/// underneath, so it is served by the same backend.
const HDF5_FORMATS: [&str; 5] = ["hdf5", "h5", "he5", "nc", "netcdf4"];

/// A Zarr store, which is a directory rather than a file.
const ZARR_FORMAT: &str = "zarr";

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
    /// Shared by every compressed file this server opens.
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

        // A store is a directory, so which rule resolves the path depends on
        // the format, and the format has to be settled first.
        let asked = if request.format.is_empty() {
            format_from_extension(std::path::Path::new(&request.path)).ok()
        } else {
            Some(request.format.to_ascii_lowercase())
        };
        if asked.as_deref() == Some(ZARR_FORMAT) {
            let root = self.paths.resolve_store(&request.path)?;
            let file: Arc<dyn ArrayFile> =
                Arc::new(aex_core::ZarrFile::open(&root, self.decode_cache.clone())?);
            let handle = session.files().insert(file);
            tracing::debug!(
                session = %hex(session.id()),
                handle,
                path = %root.display(),
                "opened store"
            );
            return Ok(handle);
        }

        let path = self.paths.resolve(&request.path)?;
        let format = match asked {
            Some(format) => format,
            None => format_from_extension(&path)?,
        };
        let file: Arc<dyn ArrayFile> = if format == NPY_FORMAT {
            Arc::new(NpyFile::open(&path)?)
        } else if HDF5_FORMATS.contains(&format.as_str()) {
            self.open_hdf5(&path)?
        } else {
            return Err(ServerError::BadRequest(format!(
                "format {format:?} is not supported; this server serves {NPY_FORMAT:?}, \
                 {ZARR_FORMAT:?} and {:?}",
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
        let mut requested = quality_from_proto(request.requested_quality.as_ref());
        // The codec has its own request field, outside QualitySpec.
        requested.codec = Codec::from_u32(request.requested_codec);
        let applied = requested.applied(dataset.dtype());
        // Which is RAW unless a codec this build has was named, and the same
        // answer the send path reads back out of the layout.
        let codec = applied.codec();

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
            return plan_reply(&layout, codec, &applied, 0, Vec::new(), inline_data);
        }

        let entry = self
            .transfers
            .insert(*session.id(), dataset.clone(), layout)?;
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
            // Empty host by default: the client reuses the control plane's address.
            endpoints: vec![DataEndpoint {
                host: self.config.data_advertise_host.clone(),
                port: self.data_port as u32,
            }],
            granted_streams: session.granted_streams(),
            protocol_version: PROTOCOL_VERSION,
            default_chunk_bytes: self.config.transfer.default_chunk_bytes,
            supported_codecs: aex_core::quality::supported_codecs(),
            supported_encodings: aex_core::quality::supported_encodings(),
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
        let attrs = file.attrs(&request.name).map_err(ServerError::from)?;
        Ok(Response::new(item_to_proto(&request.name, &item, attrs)?))
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

        // The backend names children relative to their parent, so the path an
        // attribute lookup takes has to be put back together here.
        let parent = request.name.trim_end_matches('/');
        let items = children
            .iter()
            .map(|(name, item)| {
                let attrs = file
                    .attrs(&format!("{parent}/{name}"))
                    .map_err(ServerError::from)?;
                item_to_proto(name, item, attrs)
            })
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
        request: Request<ApplyFunctionRequest>,
    ) -> std::result::Result<Response<ApplyFunctionReply>, Status> {
        let request = request.into_inner();
        let session = self.sessions.get(&request.session_id)?;
        let file = session.files().get(request.handle)?;
        let Item::Dataset(dataset) = file.get_item(&request.name).map_err(ServerError::from)?
        else {
            return Err(ServerError::BadRequest(format!(
                "{:?} is a group; only a dataset can be reduced",
                request.name
            ))
            .into());
        };
        let function = Function::from_name(&request.function_name).ok_or_else(|| {
            ServerError::BadRequest(format!(
                "{:?} is not a function this server computes",
                request.function_name
            ))
        })?;
        let args = reduce_args(&request.kwargs)?;
        let indices = indices_from_proto(&request.indices, self.config.limits.max_fancy_indices)
            .map_err(ServerError::from)?;
        let layout = dataset
            .layout(&indices, &aex_core::QualitySpec::exact())
            .map_err(ServerError::from)?;
        let limit = self.config.transfer.inline_limit_bytes;

        // Reading the whole selection takes a while; keep it off the threads
        // that answer the other calls.
        let reduced = tokio::task::spawn_blocking(move || {
            aex_core::reduce(&*dataset, &layout, function, &args, limit)
        })
        .await
        .map_err(|e| Status::internal(format!("the reduction did not finish: {e}")))?
        .map_err(ServerError::from)?;

        Ok(Response::new(ApplyFunctionReply {
            dtype: reduced.dtype.as_i32(),
            shape: shape_to_proto(&reduced.shape)?,
            data: reduced.data,
        }))
    }
}

/// numpy's keyword arguments to a reduction, as far as this server takes them.
fn reduce_args(kwargs: &HashMap<String, FunctionArgument>) -> Result<ReduceArgs> {
    let mut args = ReduceArgs::default();
    for (name, value) in kwargs {
        let bad = || ServerError::BadRequest(format!("{name}={value:?} is not supported"));
        let value = value.value.as_ref().ok_or_else(bad)?;
        match (name.as_str(), value) {
            ("axis", Value::NoneValue(_)) => args.axis = Axis::All,
            ("axis", Value::IntValue(a)) => args.axis = Axis::One(*a),
            ("axis", Value::TupleInt(t)) => args.axis = Axis::Many(t.values.clone()),
            ("keepdims", Value::BoolValue(k)) => args.keepdims = *k,
            ("ddof", Value::IntValue(d)) => args.ddof = *d,
            ("ddof", Value::FloatValue(d)) if d.fract() == 0.0 => args.ddof = *d as i64,
            _ => return Err(bad()),
        }
    }
    Ok(args)
}

/// Assemble the reply, whichever way the selection is being answered.
fn plan_reply(
    layout: &SelectionLayout,
    codec: Codec,
    applied: &aex_core::QualitySpec,
    request_id: u32,
    ticket: Vec<u8>,
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
        inline_data,
    })
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

        // Misses are decodes. Against the chunks a transfer delivers, they say
        // whether the cache held a chunk long enough to be read out or evicted
        // it under a reader still walking it. Cumulative, so a measurement
        // takes the difference across the transfer it cares about.
        let stats = self.decode_cache.stats();
        tracing::debug!(
            hits = stats.hits,
            misses = stats.misses,
            races = stats.races,
            chunk_bytes,
            streams,
            "decode cache"
        );
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

fn item_to_proto(
    name: &str,
    item: &Item,
    attrs: Vec<(String, AttrValue)>,
) -> Result<aex_proto::Item> {
    let data = match item {
        Item::Dataset(dataset) => aex_proto::item::Data::Dataset(dataset_to_proto(&**dataset)?),
        Item::Group => aex_proto::item::Data::Group(Group {}),
    };
    Ok(aex_proto::Item {
        name: name.to_string(),
        data: Some(data),
        attrs: attrs
            .into_iter()
            .map(|(name, value)| attr_to_proto(name, value))
            .collect::<Result<Vec<_>>>()?,
    })
}

fn attr_to_proto(name: String, value: AttrValue) -> Result<aex_proto::Attribute> {
    let value = match value {
        AttrValue::Text(text) => aex_proto::attribute::Value::Text(text),
        AttrValue::Array { dtype, shape, data } => {
            aex_proto::attribute::Value::Array(aex_proto::AttrArray {
                dtype: dtype.as_i32(),
                shape: shape_to_proto(&shape)?,
                data,
            })
        }
    };
    Ok(aex_proto::Attribute {
        name,
        value: Some(value),
    })
}

fn dataset_to_proto(dataset: &dyn ArrayDataset) -> Result<Dataset> {
    let shape = shape_to_proto(dataset.shape())?;
    Ok(Dataset {
        dtype: dataset.dtype().as_i32(),
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
        let item =
            item_to_proto("array", &Item::Dataset(Arc::new(dataset)), Vec::new()).expect("convert");

        assert_eq!(item.name, "array");
        let Some(aex_proto::item::Data::Dataset(dataset)) = item.data else {
            panic!("expected a dataset");
        };
        assert_eq!(dataset.dtype, DType::Float32.as_i32());
        assert_eq!(dataset.shape, vec![1000, 200]);
    }

    #[test]
    fn a_scalar_array_keeps_its_empty_shape() {
        let dataset = FakeDataset {
            dtype: DType::Int64,
            shape: vec![],
        };
        let item = item_to_proto("array", &Item::Dataset(Arc::new(dataset)), Vec::new()).unwrap();
        let Some(aex_proto::item::Data::Dataset(dataset)) = item.data else {
            panic!("expected a dataset");
        };
        assert!(dataset.shape.is_empty());
    }

    #[test]
    fn a_group_crosses_the_wire_as_a_group() {
        let item = item_to_proto("/", &Item::Group, Vec::new()).expect("convert");
        assert_eq!(item.name, "/");
        assert!(matches!(item.data, Some(aex_proto::item::Data::Group(_))));
    }

    #[test]
    fn attributes_cross_the_wire_in_order_and_by_kind() {
        let attrs = vec![
            (
                "_FillValue".to_string(),
                AttrValue::Array {
                    dtype: DType::Int16,
                    shape: vec![],
                    data: vec![0xfd, 0xff],
                },
            ),
            ("units".to_string(), AttrValue::Text("K".into())),
        ];
        let item = item_to_proto("/ds", &Item::Group, attrs).expect("convert");

        assert_eq!(
            item.attrs
                .iter()
                .map(|a| a.name.as_str())
                .collect::<Vec<_>>(),
            ["_FillValue", "units"]
        );
        let Some(aex_proto::attribute::Value::Array(array)) = &item.attrs[0].value else {
            panic!("expected an array");
        };
        assert_eq!(array.dtype, DType::Int16.as_i32());
        assert!(array.shape.is_empty());
        assert_eq!(array.data, vec![0xfd, 0xff]);
        assert_eq!(
            item.attrs[1].value,
            Some(aex_proto::attribute::Value::Text("K".into()))
        );
    }

    #[test]
    fn an_axis_too_long_for_the_wire_is_rejected() {
        // Reachable only through a corrupt header: an empty array can declare
        // any axis length, since the product is zero either way.
        let dataset = FakeDataset {
            dtype: DType::Uint8,
            shape: vec![u64::MAX, 0],
        };
        let err =
            item_to_proto("array", &Item::Dataset(Arc::new(dataset)), Vec::new()).unwrap_err();
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
    fn reduction_arguments_come_off_the_wire() {
        let arg = |value| FunctionArgument { value: Some(value) };
        let kwargs = HashMap::from([
            (
                "axis".to_string(),
                arg(Value::TupleInt(aex_proto::IntTuple {
                    values: vec![0, -1],
                })),
            ),
            ("keepdims".to_string(), arg(Value::BoolValue(true))),
            ("ddof".to_string(), arg(Value::FloatValue(1.0))),
        ]);
        let args = reduce_args(&kwargs).unwrap();
        assert_eq!(args.axis, Axis::Many(vec![0, -1]));
        assert!(args.keepdims);
        assert_eq!(args.ddof, 1);

        for (name, value) in [
            ("out", Value::NoneValue(true)),
            ("ddof", Value::FloatValue(0.5)),
            ("keepdims", Value::IntValue(1)),
        ] {
            let kwargs = HashMap::from([(name.to_string(), arg(value))]);
            let err = reduce_args(&kwargs).unwrap_err();
            assert_eq!(err.class(), aex_core::ErrorClass::Request, "{name}");
        }
    }

    #[test]
    fn hex_renders_a_session_id() {
        assert_eq!(hex(&[0x00, 0x0f, 0xff]), "000fff");
    }
}
