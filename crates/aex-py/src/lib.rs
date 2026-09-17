//! The `aex._aex` extension module.
//!
//! A thin layer over `aex-client`: key parsing, proxies and numpy dispatch live
//! in Python. What stays here is what Python cannot do safely — handing a numpy
//! array's memory to the transfer — and releasing the GIL around every call
//! that waits on the network.

use std::sync::{Arc, RwLock};

use aex_client::{ClientConfig, ClientError, DType, ErrorClass, FileHandle, Index, Item};
use half::f16;
use numpy::{
    Complex32, Complex64, Element, PyArrayDescrMethods, PyArrayDyn, PyArrayMethods,
    PyReadonlyArray1, PyUntypedArray, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PySlice, PyTuple};

/// Raise a client error as the `aex.errors` exception for its class.
fn to_py(err: ClientError) -> PyErr {
    let (kind, class) = match &err {
        ClientError::Server {
            code: aex_client::Code::NotFound,
            class,
            ..
        } => ("AexNotFoundError", *class),
        ClientError::Server { class, .. } | ClientError::Data { class, .. } => {
            (exception_for(*class), *class)
        }
        ClientError::Transport(_) | ClientError::Io(_) => {
            ("AexConnectionError", ErrorClass::Transient)
        }
        ClientError::Protocol(_) => ("AexProtocolError", ErrorClass::Protocol),
        ClientError::BadRequest(_) => ("AexValueError", ErrorClass::Request),
        _ => ("AexError", ErrorClass::Permanent),
    };
    aex_error(kind, class, err.to_string())
}

fn exception_for(class: ErrorClass) -> &'static str {
    match class {
        ErrorClass::Protocol => "AexProtocolError",
        ErrorClass::Auth => "AexConnectionError",
        ErrorClass::Request => "AexValueError",
        _ => "AexTransferError",
    }
}

fn aex_error(kind: &str, class: ErrorClass, message: String) -> PyErr {
    Python::attach(|py| {
        let made = py
            .import("aex.errors")
            .and_then(|m| m.getattr(kind))
            .and_then(|cls| cls.call1((message, class_name(class))));
        match made {
            Ok(exc) => PyErr::from_value(exc),
            Err(e) => e,
        }
    })
}

fn value_error(message: String) -> PyErr {
    aex_error("AexValueError", ErrorClass::Request, message)
}

fn class_name(class: ErrorClass) -> &'static str {
    match class {
        ErrorClass::Ok => "OK",
        ErrorClass::Protocol => "PROTOCOL",
        ErrorClass::Auth => "AUTH",
        ErrorClass::Plan => "PLAN",
        ErrorClass::Request => "REQUEST",
        ErrorClass::Transient => "TRANSIENT",
        ErrorClass::Permanent => "PERMANENT",
    }
}

/// A selection resolved by the server.
#[pyclass(frozen, module = "aex._aex")]
struct Plan(aex_client::Plan);

#[pymethods]
impl Plan {
    /// Shape of the result.
    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.0.shape)
    }

    /// numpy `descr` of the result, such as `<f4`.
    #[getter]
    fn dtype(&self) -> &'static str {
        self.0.dtype.descr()
    }

    #[getter]
    fn total_bytes(&self) -> u64 {
        self.0.total_bytes
    }

    /// Whether the data came back with the plan.
    #[getter]
    fn is_inline(&self) -> bool {
        self.0.is_inline()
    }

    fn __repr__(&self) -> String {
        format!(
            "<Plan shape {:?}, dtype {}, {} bytes>",
            self.0.shape,
            self.0.dtype.descr(),
            self.0.total_bytes
        )
    }
}

/// A session with a server.
///
/// Frozen and shared: calls run with the GIL released, so two Python threads
/// may be inside the same client at once.
#[pyclass(frozen, module = "aex._aex")]
struct Client {
    inner: RwLock<Option<Arc<aex_client::Client>>>,
}

impl Client {
    fn get(&self) -> PyResult<Arc<aex_client::Client>> {
        self.inner
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
            .ok_or_else(|| {
                aex_error(
                    "AexConnectionError",
                    ErrorClass::Auth,
                    "the client is closed".to_string(),
                )
            })
    }

    /// Run `f` on the client with the GIL released.
    fn call<T: Send>(
        &self,
        py: Python<'_>,
        f: impl FnOnce(&aex_client::Client) -> aex_client::Result<T> + Send,
    ) -> PyResult<T> {
        let client = self.get()?;
        py.detach(|| f(&client)).map_err(to_py)
    }
}

#[pymethods]
impl Client {
    /// Connect to `url`, a gRPC endpoint such as `http://host:50051`.
    ///
    /// Settings come from the `AEX_*` environment variables.
    #[new]
    fn new(py: Python<'_>, url: String) -> PyResult<Self> {
        let client = py
            .detach(|| aex_client::Client::connect(&url, ClientConfig::from_env()))
            .map_err(to_py)?;
        Ok(Client {
            inner: RwLock::new(Some(Arc::new(client))),
        })
    }

    fn open(&self, py: Python<'_>, path: String) -> PyResult<u64> {
        self.call(py, |c| c.open(&path)).map(|h| h.as_u64())
    }

    fn close_file(&self, py: Python<'_>, handle: u64) -> PyResult<()> {
        self.call(py, |c| c.close(FileHandle::from_u64(handle)))
    }

    /// `(dtype, shape)` for a dataset, `None` for a group.
    fn get_item(
        &self,
        py: Python<'_>,
        handle: u64,
        name: String,
    ) -> PyResult<Option<(&'static str, Vec<u64>)>> {
        let item = self.call(py, |c| c.get_item(FileHandle::from_u64(handle), &name))?;
        Ok(item_to_py(item))
    }

    /// `(name, item)` for each child, with items as in `get_item`.
    #[allow(clippy::type_complexity)]
    fn list_children(
        &self,
        py: Python<'_>,
        handle: u64,
        name: String,
    ) -> PyResult<Vec<(String, Option<(&'static str, Vec<u64>)>)>> {
        let children = self.call(py, |c| c.list_children(FileHandle::from_u64(handle), &name))?;
        Ok(children
            .into_iter()
            .map(|(name, item)| (name, item_to_py(item)))
            .collect())
    }

    /// Resolve a selection. `key` is a tuple of int, slice, `...`, `None` and
    /// 1-d int64 arrays, as `aex.array_proxy` prepares it.
    fn prepare(
        &self,
        py: Python<'_>,
        handle: u64,
        name: String,
        key: &Bound<'_, PyTuple>,
    ) -> PyResult<Plan> {
        let indices = indices_from_py(key)?;
        self.call(py, |c| {
            c.prepare(FileHandle::from_u64(handle), &name, &indices)
        })
        .map(Plan)
    }

    /// Transfer a plan into `out`, which must match it exactly.
    ///
    /// On failure the contents of `out` are undefined.
    fn fill(
        &self,
        py: Python<'_>,
        plan: &Plan,
        handle: u64,
        name: String,
        key: &Bound<'_, PyTuple>,
        out: &Bound<'_, PyUntypedArray>,
    ) -> PyResult<()> {
        let indices = indices_from_py(key)?;
        let plan = &plan.0;
        let client = self.get()?;
        check_buffer(plan, out)?;

        macro_rules! fill_as {
            ($($dtype:ident => $ty:ty),* $(,)?) => {
                match plan.dtype {
                    $(DType::$dtype => fill_typed::<$ty>(py, &client, plan, handle, &name, &indices, out),)*
                }
            };
        }
        fill_as!(
            Bool => bool, Int8 => i8, Int16 => i16, Int32 => i32, Int64 => i64,
            Uint8 => u8, Uint16 => u16, Uint32 => u32, Uint64 => u64,
            Float16 => f16, Float32 => f32, Float64 => f64,
            Complex64 => Complex32, Complex128 => Complex64,
        )
    }

    /// End the session. Later calls fail.
    fn disconnect(&self, py: Python<'_>) -> PyResult<()> {
        let taken = self.inner.write().unwrap_or_else(|e| e.into_inner()).take();
        // If another thread is still mid-call the session is left to expire.
        match taken.and_then(|c| Arc::try_unwrap(c).ok()) {
            Some(client) => py.detach(|| client.disconnect()).map_err(to_py),
            None => Ok(()),
        }
    }
}

fn item_to_py(item: Item) -> Option<(&'static str, Vec<u64>)> {
    match item {
        Item::Dataset(info) => Some((info.dtype.descr(), info.shape)),
        Item::Group => None,
    }
}

fn indices_from_py(key: &Bound<'_, PyTuple>) -> PyResult<Vec<Index>> {
    let py = key.py();
    key.iter()
        .map(|entry| {
            if entry.is_none() {
                Ok(Index::NewAxis)
            } else if entry.is(py.Ellipsis()) {
                Ok(Index::Ellipsis)
            } else if let Ok(slice) = entry.cast::<PySlice>() {
                let bound =
                    |attr: &str| -> PyResult<Option<i64>> { slice.getattr(attr)?.extract() };
                Ok(Index::Slice {
                    start: bound("start")?,
                    stop: bound("stop")?,
                    step: bound("step")?,
                })
            } else if let Ok(array) = entry.extract::<PyReadonlyArray1<'_, i64>>() {
                Ok(Index::Fancy(array.as_array().to_vec()))
            } else if let Ok(single) = entry.extract::<i64>() {
                Ok(Index::Single(single))
            } else {
                Err(PyTypeError::new_err(format!(
                    "cannot send {} as an index",
                    entry.get_type().name()?
                )))
            }
        })
        .collect()
}

/// Refuse a buffer the transfer cannot write straight into.
///
/// No temporary is put in between: a copy would defeat the point of passing a
/// buffer at all.
fn check_buffer(plan: &aex_client::Plan, out: &Bound<'_, PyUntypedArray>) -> PyResult<()> {
    if !out.is_c_contiguous() {
        return Err(value_error("the output array is not C-contiguous".into()));
    }
    let descr = out.dtype();
    if descr.byteorder() == b'>' {
        return Err(value_error("the output array is big-endian".into()));
    }
    let nbytes = out.len() * descr.itemsize();
    if nbytes as u64 != plan.total_bytes {
        return Err(value_error(format!(
            "the output array is {nbytes} bytes and the selection is {}",
            plan.total_bytes
        )));
    }
    Ok(())
}

fn fill_typed<T: Element>(
    py: Python<'_>,
    client: &aex_client::Client,
    plan: &aex_client::Plan,
    handle: u64,
    name: &str,
    indices: &[Index],
    out: &Bound<'_, PyUntypedArray>,
) -> PyResult<()> {
    let array = out.cast::<PyArrayDyn<T>>().map_err(|_| {
        value_error(format!(
            "the output array is not of dtype {}",
            plan.dtype.descr()
        ))
    })?;
    // The borrow is what keeps another view from writing at the same time. It
    // is held here, outside the closure, for as long as the GIL is released.
    let mut guard = array
        .try_readwrite()
        .map_err(|e| value_error(format!("the output array cannot be written: {e}")))?;
    let elements = guard
        .as_slice_mut()
        .map_err(|e| value_error(format!("the output array is not contiguous: {e}")))?;
    // Every AEX dtype is plain little-endian bytes, the byte order was checked
    // above, and the length is the element count times the item size.
    let bytes = unsafe {
        std::slice::from_raw_parts_mut(
            elements.as_mut_ptr() as *mut u8,
            std::mem::size_of_val(elements),
        )
    };
    let handle = FileHandle::from_u64(handle);
    py.detach(|| client.fill(plan, handle, name, indices, bytes))
        .map(|_| ())
        .map_err(to_py)
}

#[pymodule]
fn _aex(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Client>()?;
    m.add_class::<Plan>()?;
    Ok(())
}
