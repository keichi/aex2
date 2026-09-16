//! The gRPC control plane.
//!
//! Sessions, files and metadata are served here. The transfer and compute RPCs
//! are part of the settled protocol but arrive with the data plane in M2, and
//! report themselves as unimplemented until then.

use std::sync::Arc;

use aex_core::{ArrayDataset, ArrayFile, Item, NpyFile};
use aex_proto::aex_control_server::AexControl;
use aex_proto::{
    ApplyFunctionReply, ApplyFunctionRequest, CloseFileReply, CloseFileRequest, ConnectReply,
    ConnectRequest, DataEndpoint, Dataset, DisconnectReply, DisconnectRequest, GetItemRequest,
    Group, ItemList, ListChildrenRequest, OpenFileReply, OpenFileRequest, PrepareSelectionRequest,
    PrepareSelectionsRequest, TransferPlan, TransferPlanList,
};
use tonic::{Request, Response, Status};

use crate::config::{ServerConfig, PROTOCOL_VERSION, SUPPORTED_CODECS, SUPPORTED_ENCODINGS};
use crate::error::{Result, ServerError};
use crate::paths::PathPolicy;
use crate::session::SessionRegistry;

/// The one format this release serves.
const NPY_FORMAT: &str = "npy";

pub struct ControlService {
    sessions: Arc<SessionRegistry>,
    paths: Arc<PathPolicy>,
    config: Arc<ServerConfig>,
}

impl ControlService {
    pub fn new(
        sessions: Arc<SessionRegistry>,
        paths: Arc<PathPolicy>,
        config: Arc<ServerConfig>,
    ) -> Self {
        ControlService {
            sessions,
            paths,
            config,
        }
    }

    fn open(&self, request: &OpenFileRequest) -> Result<u64> {
        let session = self.sessions.get(&request.session_id)?;
        let path = self.paths.resolve(&request.path)?;

        let format = if request.format.is_empty() {
            format_from_extension(&path)?
        } else {
            request.format.to_ascii_lowercase()
        };
        if format != NPY_FORMAT {
            return Err(ServerError::BadRequest(format!(
                "format {format:?} is not supported; this server serves {NPY_FORMAT:?} only"
            )));
        }

        let file: Arc<dyn ArrayFile> = Arc::new(NpyFile::open(&path)?);
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
            // it already reached the control plane on. The data plane itself
            // starts listening in M2; the port it will listen on is fixed now.
            endpoints: vec![DataEndpoint {
                host: self.config.data_advertise_host.clone(),
                port: self.config.data_addr.port() as u32,
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
        self.sessions.remove(&request.session_id)?;
        tracing::info!(session = %hex(&request.session_id), "session closed");
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
        _request: Request<PrepareSelectionRequest>,
    ) -> std::result::Result<Response<TransferPlan>, Status> {
        Err(unimplemented_in_m1("PrepareSelection"))
    }

    async fn prepare_selections(
        &self,
        _request: Request<PrepareSelectionsRequest>,
    ) -> std::result::Result<Response<TransferPlanList>, Status> {
        Err(unimplemented_in_m1("PrepareSelections"))
    }

    async fn apply_function(
        &self,
        _request: Request<ApplyFunctionRequest>,
    ) -> std::result::Result<Response<ApplyFunctionReply>, Status> {
        Err(unimplemented_in_m1("ApplyFunction"))
    }
}

/// Says which milestone the RPC is waiting on, so that a client hitting one
/// during development is not left wondering whether it misdialled.
fn unimplemented_in_m1(rpc: &str) -> Status {
    Status::unimplemented(format!(
        "{rpc} needs the data plane, which this server does not serve yet"
    ))
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
    let shape = dataset
        .shape()
        .iter()
        .map(|&n| {
            // The wire carries shapes as int64, as numpy does. A header can
            // declare a longer axis than that; such a file is not one we can
            // describe, let alone serve.
            i64::try_from(n).map_err(|_| {
                ServerError::Core(aex_core::AexError::MalformedNpy(format!(
                    "axis of {n} elements does not fit the int64 shape on the wire"
                )))
            })
        })
        .collect::<Result<Vec<i64>>>()?;

    Ok(Dataset {
        dtype: dataset.dtype().as_i32(),
        ndim: shape.len() as i32,
        shape,
    })
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
