//! The `aex._aex` extension module.
//!
//! A thin layer over `aex-client`: key parsing, proxies and numpy dispatch live
//! in Python. What stays here is what Python cannot do safely — handing a numpy
//! array's memory to the transfer — and releasing the GIL around every call
//! that waits on the network.

use std::sync::{Arc, RwLock};

use aex_client::{
    resolve_selection, AexError, ClientConfig, ClientError, Codec, DType, Encoding, ErrorClass,
    FileHandle, FunctionArg, Index, Item, QualitySpec, Selection,
};
use half::f16;
use numpy::{
    Complex32, Complex64, Element, PyArrayDescrMethods, PyArrayDyn, PyArrayMethods,
    PyReadonlyArray1, PyUntypedArray, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyFloat, PySlice, PyTuple};

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

/// Raise a core error as the `aex.errors` exception for its class.
fn core_error(err: AexError) -> PyErr {
    let class = err.class();
    aex_error(exception_for(class), class, err.to_string())
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

    /// What the server did to the elements, as a dict naming the encoding.
    #[getter]
    fn applied_quality<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        quality_to_py(py, &self.0.applied_quality)
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
    #[pyo3(signature = (handle, name, key, quality=None))]
    fn prepare(
        &self,
        py: Python<'_>,
        handle: u64,
        name: String,
        key: &Bound<'_, PyTuple>,
        quality: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Plan> {
        let indices = indices_from_py(key)?;
        let quality = quality_from_py(quality)?;
        self.call(py, |c| {
            c.prepare_selection(&Selection {
                quality: &quality,
                ..Selection::exact(FileHandle::from_u64(handle), &name, &indices)
            })
        })
        .map(Plan)
    }

    /// Resolve several selections of one dataset in one round trip. Each
    /// element is a `Plan` or the exception it failed with.
    fn prepare_many<'py>(
        &self,
        py: Python<'py>,
        handle: u64,
        name: String,
        keys: Vec<Bound<'py, PyTuple>>,
    ) -> PyResult<Vec<Bound<'py, PyAny>>> {
        let indices = keys
            .iter()
            .map(indices_from_py)
            .collect::<PyResult<Vec<_>>>()?;
        let handle = FileHandle::from_u64(handle);
        let results = self.call(py, |c| c.prepare_many(&selections(handle, &name, &indices)))?;
        results
            .into_iter()
            .map(|result| match result {
                Ok(plan) => Ok(Bound::new(py, Plan(plan))?.into_any()),
                Err(e) => Ok(to_py(e).into_value(py).into_bound(py).into_any()),
            })
            .collect()
    }

    /// Transfer a plan into `out`, which must match it exactly.
    ///
    /// On failure the contents of `out` are undefined.
    fn fill(
        &self,
        py: Python<'_>,
        plan: Bound<'_, Plan>,
        handle: u64,
        name: String,
        key: Bound<'_, PyTuple>,
        out: Bound<'_, PyUntypedArray>,
    ) -> PyResult<()> {
        self.fill_many(py, vec![plan], handle, name, vec![key], vec![out])
    }

    /// Transfer plans of one dataset into their buffers as one batch.
    ///
    /// On failure the contents of every buffer are undefined.
    fn fill_many(
        &self,
        py: Python<'_>,
        plans: Vec<Bound<'_, Plan>>,
        handle: u64,
        name: String,
        keys: Vec<Bound<'_, PyTuple>>,
        outs: Vec<Bound<'_, PyUntypedArray>>,
    ) -> PyResult<()> {
        let indices = keys
            .iter()
            .map(indices_from_py)
            .collect::<PyResult<Vec<_>>>()?;
        let plans: Vec<aex_client::Plan> = plans.iter().map(|p| p.get().0.clone()).collect();
        if plans.len() != outs.len() || plans.len() != indices.len() {
            return Err(value_error(
                "plans, keys and output arrays do not pair up".into(),
            ));
        }
        let Some(dtype) = plans.first().map(|p| p.dtype) else {
            return Ok(());
        };
        for (plan, out) in plans.iter().zip(&outs) {
            if plan.dtype != dtype {
                return Err(value_error("a batch has to be of one dtype".into()));
            }
            check_buffer(plan, out)?;
        }
        let client = self.get()?;
        let batch = Batch {
            client: &client,
            plans: &plans,
            selections: &selections(FileHandle::from_u64(handle), &name, &indices),
            outs: &outs,
        };

        macro_rules! fill_as {
            ($($dtype:ident => $ty:ty),* $(,)?) => {
                match dtype {
                    $(DType::$dtype => batch.fill::<$ty>(py),)*
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

    /// Reduce a selection on the server. Returns `(dtype, shape, bytes)`.
    ///
    /// `kwargs` values are None, bool, int, float or a tuple of ints.
    #[allow(clippy::type_complexity)]
    fn apply_function<'py>(
        &self,
        py: Python<'py>,
        handle: u64,
        name: String,
        key: &Bound<'_, PyTuple>,
        function: String,
        kwargs: &Bound<'_, PyDict>,
    ) -> PyResult<(&'static str, Vec<u64>, Bound<'py, PyBytes>)> {
        let indices = indices_from_py(key)?;
        let kwargs = kwargs
            .iter()
            .map(|(k, v)| Ok((k.extract::<String>()?, function_arg(&v)?)))
            .collect::<PyResult<Vec<_>>>()?;
        let reduced = self.call(py, |c| {
            let kwargs: Vec<_> = kwargs
                .iter()
                .map(|(k, v)| (k.as_str(), v.clone()))
                .collect();
            c.apply_function(
                &Selection::exact(FileHandle::from_u64(handle), &name, &indices),
                &function,
                &kwargs,
            )
        })?;
        Ok((
            reduced.dtype.descr(),
            reduced.shape,
            PyBytes::new(py, &reduced.data),
        ))
    }

    /// Totals over every transfer, for benchmarks and tuning.
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let stats = self.get()?.stats();
        let dict = PyDict::new(py);
        dict.set_item("bytes", stats.bytes)?;
        dict.set_item("elapsed", stats.elapsed.as_secs_f64())?;
        dict.set_item("throughput_mibps", stats.throughput_mib_per_sec())?;
        dict.set_item("streams", stats.streams)?;
        dict.set_item("chunks", stats.chunks)?;
        dict.set_item("retries", stats.retries)?;
        dict.set_item("rtt_ms", stats.rtt.as_secs_f64() * 1e3)?;
        Ok(dict)
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

fn function_arg(value: &Bound<'_, PyAny>) -> PyResult<FunctionArg> {
    // bool before int: Python's bool is an int.
    Ok(if value.is_none() {
        FunctionArg::None
    } else if let Ok(b) = value.cast::<PyBool>() {
        FunctionArg::Bool(b.is_true())
    } else if let Ok(i) = value.extract::<i64>() {
        FunctionArg::Int(i)
    } else if let Ok(f) = value.cast::<PyFloat>() {
        FunctionArg::Float(f.value())
    } else if let Ok(t) = value.cast::<PyTuple>() {
        FunctionArg::Ints(t.extract()?)
    } else {
        return Err(PyTypeError::new_err(format!(
            "cannot send {} as a function argument",
            value.get_type().name()?
        )));
    })
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

fn selections<'a>(
    handle: FileHandle,
    name: &'a str,
    indices: &'a [Vec<Index>],
) -> Vec<Selection<'a>> {
    indices
        .iter()
        .map(|indices| Selection::exact(handle, name, indices))
        .collect()
}

/// A quality request as `aex.array_proxy` spells it: a dict holding at most
/// one of `dtype` or `abs_error` / `rel_error`, and an optional `codec`.
/// `None` is lossless and uncompressed.
fn quality_from_py(quality: Option<&Bound<'_, PyDict>>) -> PyResult<QualitySpec> {
    let mut spec = QualitySpec::exact();
    let Some(quality) = quality else {
        return Ok(spec);
    };
    if let Some(dtype) = quality.get_item("dtype")? {
        spec.encoding = Encoding::DtypeCast;
        spec.cast_dtype = Some(
            DType::from_descr(&dtype.extract::<String>()?)
                .map_err(|e| value_error(e.to_string()))?,
        );
    } else {
        spec.abs_error_bound = quality
            .get_item("abs_error")?
            .map(|v| v.extract())
            .transpose()?;
        spec.rel_error_bound = quality
            .get_item("rel_error")?
            .map(|v| v.extract())
            .transpose()?;
        if spec.abs_error_bound.is_some() || spec.rel_error_bound.is_some() {
            spec.encoding = Encoding::ErrorBound;
        }
    }
    // Outside the branch: a lossless codec goes with no quality at all, which
    // is how GZIP is asked for.
    spec.codec = quality
        .get_item("codec")?
        .map(|v| codec_from_name(&v.extract::<String>()?))
        .transpose()?;
    Ok(spec)
}

/// The codecs a caller may name. A name this build has no codec for is still
/// accepted: the server decides whether it can honour it, and says so in the
/// plan.
fn codec_from_name(name: &str) -> PyResult<Codec> {
    match name {
        "gzip" => Ok(Codec::Gzip),
        "sz" => Ok(Codec::Sz),
        "zfp" => Ok(Codec::Zfp),
        other => Err(value_error(format!(
            "codec must be 'gzip', 'sz' or 'zfp', not {other:?}"
        ))),
    }
}

fn codec_name(codec: Codec) -> &'static str {
    match codec {
        Codec::Raw => "raw",
        Codec::Gzip => "gzip",
        Codec::Sz => "sz",
        Codec::Zfp => "zfp",
    }
}

/// The inverse of `quality_from_py`, with the encoding named.
fn quality_to_py<'py>(py: Python<'py>, spec: &QualitySpec) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    let encoding = match spec.encoding {
        Encoding::Exact => "exact",
        Encoding::DtypeCast => "dtype_cast",
        Encoding::ErrorBound => "error_bound",
    };
    dict.set_item("encoding", encoding)?;
    match spec.encoding {
        Encoding::Exact => {}
        Encoding::DtypeCast => dict.set_item("dtype", spec.cast_dtype.map(DType::descr))?,
        Encoding::ErrorBound => {
            dict.set_item("abs_error", spec.abs_error_bound)?;
            dict.set_item("rel_error", spec.rel_error_bound)?;
        }
    }
    // Always, because an exact transfer can be compressed too.
    dict.set_item("codec", codec_name(spec.codec()))?;
    Ok(dict)
}

/// A batch whose buffers have been checked against their plans.
struct Batch<'a, 'py> {
    client: &'a aex_client::Client,
    plans: &'a [aex_client::Plan],
    selections: &'a [Selection<'a>],
    outs: &'a [Bound<'py, PyUntypedArray>],
}

impl Batch<'_, '_> {
    fn fill<T: Element>(&self, py: Python<'_>) -> PyResult<()> {
        // The borrows are what keep another view from writing at the same time,
        // and what refuse one array passed twice. They are held here, outside
        // the closure, for as long as the GIL is released.
        let mut guards = Vec::with_capacity(self.outs.len());
        for out in self.outs {
            let array = out.cast::<PyArrayDyn<T>>().map_err(|_| {
                value_error(format!(
                    "the output array is not of dtype {}",
                    self.plans[0].dtype.descr()
                ))
            })?;
            guards.push(
                array
                    .try_readwrite()
                    .map_err(|e| value_error(format!("the output array cannot be written: {e}")))?,
            );
        }
        let mut dsts = Vec::with_capacity(guards.len());
        for guard in &mut guards {
            let elements = guard
                .as_slice_mut()
                .map_err(|e| value_error(format!("the output array is not contiguous: {e}")))?;
            // Every AEX dtype is plain little-endian bytes, the byte order was
            // checked, and the length is the element count times the item size.
            dsts.push(unsafe {
                std::slice::from_raw_parts_mut(
                    elements.as_mut_ptr() as *mut u8,
                    std::mem::size_of_val(elements),
                )
            });
        }
        py.detach(|| {
            self.client
                .fill_many(self.plans, self.selections, &mut dsts)
        })
        .map(|_| ())
        .map_err(to_py)
    }
}

/// Resolve `key` against an array of `shape` and `dtype`. Returns
/// `(dtype, shape)` of what the selection would yield.
///
/// The server resolves a selection with this same function, so nothing here
/// has to agree with it by hand; a client and a server that disagreed could
/// not share a session anyway, since the protocol version has to match.
#[pyfunction]
fn resolve(
    shape: Vec<u64>,
    dtype: &str,
    key: &Bound<'_, PyTuple>,
) -> PyResult<(&'static str, Vec<u64>)> {
    let dtype = DType::from_descr(dtype).map_err(core_error)?;
    let indices = indices_from_py(key)?;
    let resolved = resolve_selection(&shape, &indices).map_err(core_error)?;
    Ok((dtype.descr(), resolved.out_shape))
}

#[pymodule]
fn _aex(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Client>()?;
    m.add_class::<Plan>()?;
    m.add_function(wrap_pyfunction!(resolve, m)?)?;
    Ok(())
}
