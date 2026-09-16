//! The control plane client.
//!
//! The API is blocking, with a tokio runtime kept inside. Callers are analysis
//! code and, from M3, Python: neither wants to own a runtime, and the Python
//! bindings will release the GIL around exactly these blocking calls.

use std::net::IpAddr;
use std::sync::Arc;

use aex_core::DType;
use aex_proto::aex_control_client::AexControlClient;
use aex_proto::{
    CloseFileRequest, ConnectRequest, DisconnectRequest, GetItemRequest, ListChildrenRequest,
    OpenFileRequest,
};
use tokio::runtime::Runtime;
use tonic::transport::{Channel, Endpoint};

use crate::config::ClientConfig;
use crate::error::{ClientError, Result};

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
    session: SessionInfo,
    config: ClientConfig,
}

impl std::fmt::Debug for Client {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Client")
            .field("session", &self.session)
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

        let reply = runtime
            .block_on(control.connect(ConnectRequest {
                protocol_version: PROTOCOL_VERSION,
                desired_streams: config.streams,
                client_name: config.client_name.clone(),
            }))?
            .into_inner();

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
        let host = if endpoint.host.is_empty() {
            strip_brackets(&control_host).to_string()
        } else {
            endpoint.host.clone()
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
        };

        Ok(Client {
            control,
            runtime,
            session,
            config,
        })
    }

    pub fn session(&self) -> &SessionInfo {
        &self.session
    }

    pub fn config(&self) -> &ClientConfig {
        &self.config
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
        let response = self.runtime.block_on(rpc(self.control.clone()))?;
        Ok(response.into_inner())
    }
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
}
