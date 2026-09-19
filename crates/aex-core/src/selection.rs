//! Selections, and the byte range they resolve to.
//!
//! A selection is resolved in two steps. [`resolve`] normalises it against a
//! shape — expanding `Ellipsis`, folding negative indices, filling in axes the
//! client did not mention — and yields one [`AxisSel`] per source axis plus the
//! output shape. [`SelectionLayout::resolve`] then decides how the selected
//! elements lie in the source, which is what the send path needs.
//!
//! The coordinate system throughout is the **logical byte stream**: the result
//! flattened in C order into `total_bytes` bytes. Every offset a client sends
//! and every offset a frame carries is a position in it, so a chunk can be
//! received on any connection, in any order, and still land in the right place.
//!
//! A selection that is one run of bytes in the source is served with a single
//! read. Anything else is gathered in output order, reading nearby runs as one
//! span: a read is a syscall, and a strided selection would otherwise cost one
//! per element.

use crate::dtype::DType;
use crate::error::{AexError, Result};
use crate::quality::QualitySpec;

/// One element of a selection, as numpy spells it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Index {
    /// `arr[5]`. Drops the axis.
    Single(i64),
    /// `arr[10:20:2]`. A missing bound means "as far as it goes".
    Slice {
        start: Option<i64>,
        stop: Option<i64>,
        step: Option<i64>,
    },
    /// `arr[[1, 5, 10]]`. Also how a boolean mask arrives, already expanded.
    Fancy(Vec<i64>),
    /// `arr[...]`. Stands for every axis the selection does not mention.
    Ellipsis,
    /// `arr[None]`. Inserts a length-1 axis without consuming a source axis.
    NewAxis,
}

impl Index {
    /// `arr[:]` along one axis.
    pub fn full() -> Self {
        Index::Slice {
            start: None,
            stop: None,
            step: None,
        }
    }

    /// `arr[start:stop]`.
    pub fn range(start: i64, stop: i64) -> Self {
        Index::Slice {
            start: Some(start),
            stop: Some(stop),
            step: None,
        }
    }

    /// Whether this consumes a source axis. `Ellipsis` and `NewAxis` do not.
    fn consumes_axis(&self) -> bool {
        !matches!(self, Index::Ellipsis | Index::NewAxis)
    }
}

/// What one source axis contributes, after normalisation.
///
/// Every index here is a valid position on its axis, so nothing downstream has
/// to bounds-check again.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AxisSel {
    /// One index. The axis does not appear in the output.
    Point(u64),
    /// `len` indices from `start`, `step` apart. `step` is never 0, and when
    /// `len` is 0 or 1 it carries no meaning.
    Range { start: u64, len: u64, step: i64 },
    /// Arbitrary indices, in the order the client gave them.
    Fancy(Vec<u64>),
}

impl AxisSel {
    /// A whole axis of `dim` elements.
    fn whole(dim: u64) -> Self {
        AxisSel::Range {
            start: 0,
            len: dim,
            step: 1,
        }
    }

    /// How many indices it selects.
    pub fn len(&self) -> u64 {
        match self {
            AxisSel::Point(_) => 1,
            AxisSel::Range { len, .. } => *len,
            AxisSel::Fancy(indices) => indices.len() as u64,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The length this axis contributes to the output shape, if it survives.
    fn out_len(&self) -> Option<u64> {
        match self {
            AxisSel::Point(_) => None,
            other => Some(other.len()),
        }
    }

    /// The one run of consecutive ascending indices this selects, if it is one.
    ///
    /// Returned as `(start, len)`. A single index is a run of one, and so is a
    /// slice or a fancy list that happens to name consecutive indices: what
    /// matters for the layout is the shape of the byte range, not how the
    /// client spelled it.
    fn as_run(&self) -> Option<(u64, u64)> {
        match self {
            AxisSel::Point(index) => Some((*index, 1)),
            AxisSel::Range { start, len, step } => match len {
                0 => Some((0, 0)),
                1 => Some((*start, 1)),
                _ if *step == 1 => Some((*start, *len)),
                _ => None,
            },
            AxisSel::Fancy(indices) => match indices.split_first() {
                None => Some((0, 0)),
                Some((first, rest)) => rest
                    .iter()
                    .enumerate()
                    .all(|(i, index)| first.checked_add(i as u64 + 1) == Some(*index))
                    .then_some((*first, indices.len() as u64)),
            },
        }
    }
}

/// The axes numpy walks in step rather than independently.
///
/// An index array turns the whole selection into an advanced one: every plain
/// integer sitting alongside it joins the same group, the group is broadcast
/// together, and it contributes **one** dimension to the result rather than one
/// per axis. `arr[[0, 2], [1, 3]]` picks two elements, not four.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Advanced {
    /// Source axes in the group, in axis order.
    pub axes: Vec<usize>,
    /// Length of the dimension they contribute between them.
    pub len: u64,
    /// Whether they sit next to each other in the selection. When they do not,
    /// numpy moves their dimension to the front of the result, because there is
    /// no one place among the axes they were taken from that it belongs.
    pub adjacent: bool,
}

/// A selection normalised against a shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedSelection {
    /// One entry per source axis, in axis order.
    pub axes: Vec<AxisSel>,
    /// The shape of the result, with the axes `NewAxis` inserted.
    pub out_shape: Vec<u64>,
    /// The advanced group, when the selection contains an index array.
    pub advanced: Option<Advanced>,
}

impl ResolvedSelection {
    /// Number of elements selected.
    ///
    /// Taken from the output shape rather than from the axes, because the axes
    /// of an advanced group are walked in step: multiplying their lengths would
    /// count the elements `arr[[0, 2], [1, 3]]` does *not* select.
    pub fn num_elements(&self) -> u64 {
        self.out_shape.iter().product()
    }
}

/// One position in a selection, after the ellipsis has been expanded.
enum Entry {
    /// Consumes a source axis.
    Axis {
        sel: AxisSel,
        /// Whether this is part of the advanced group.
        advanced: bool,
    },
    /// Inserts a length-1 axis and consumes nothing.
    New,
}

/// Normalise `indices` against `shape`.
///
/// Rejects what numpy rejects — an index off the end, more indices than the
/// array has axes, a second `Ellipsis`, a zero step, index arrays that will not
/// broadcast — with the same meaning, since the client is a numpy user either
/// way.
pub fn resolve(shape: &[u64], indices: &[Index]) -> Result<ResolvedSelection> {
    let entries = expand(shape, indices)?;

    let mut axes = Vec::with_capacity(shape.len());
    let mut group = Vec::new();
    let mut positions = Vec::new();
    let mut lengths = Vec::new();
    for (position, entry) in entries.iter().enumerate() {
        if let Entry::Axis { sel, advanced } = entry {
            if *advanced {
                group.push(axes.len());
                positions.push(position);
                if let AxisSel::Fancy(list) = sel {
                    // A plain integer is a scalar and broadcasts against
                    // anything, so only the arrays constrain the length.
                    lengths.push(list.len() as u64);
                }
            }
            axes.push(sel.clone());
        }
    }

    let advanced = if group.is_empty() {
        None
    } else {
        Some(Advanced {
            axes: group,
            len: broadcast_len(&lengths)?,
            adjacent: positions.windows(2).all(|pair| pair[1] == pair[0] + 1),
        })
    };

    Ok(ResolvedSelection {
        out_shape: out_shape(&entries, advanced.as_ref(), positions.first().copied()),
        axes,
        advanced,
    })
}

/// Expand the ellipsis and the axes the selection leaves out.
fn expand(shape: &[u64], indices: &[Index]) -> Result<Vec<Entry>> {
    let ndim = shape.len();
    let mut consuming = 0usize;
    let mut has_ellipsis = false;
    // An index array makes the whole selection advanced, which is what pulls
    // the plain integers into the group with it.
    let mut has_array = false;
    for index in indices {
        match index {
            Index::Ellipsis => {
                if has_ellipsis {
                    return Err(AexError::BadSelection(
                        "a selection may contain at most one ellipsis".to_string(),
                    ));
                }
                has_ellipsis = true;
            }
            other => {
                if matches!(other, Index::Fancy(_)) {
                    has_array = true;
                }
                if other.consumes_axis() {
                    consuming += 1;
                }
            }
        }
    }
    if consuming > ndim {
        return Err(AexError::BadSelection(format!(
            "too many indices for a {ndim}-dimensional array: the selection names {consuming}"
        )));
    }

    // Axes the selection leaves out. They sit where the ellipsis is, or at the
    // end when there is none.
    let implied = ndim - consuming;

    let mut entries = Vec::with_capacity(indices.len() + implied);
    let mut axis = 0usize;
    let whole = |axis: usize| Entry::Axis {
        sel: AxisSel::whole(shape[axis]),
        advanced: false,
    };
    for index in indices {
        match index {
            Index::NewAxis => entries.push(Entry::New),
            Index::Ellipsis => {
                for _ in 0..implied {
                    entries.push(whole(axis));
                    axis += 1;
                }
            }
            other => {
                entries.push(Entry::Axis {
                    sel: resolve_axis(other, shape[axis], axis)?,
                    advanced: has_array && matches!(other, Index::Single(_) | Index::Fancy(_)),
                });
                axis += 1;
            }
        }
    }
    if !has_ellipsis {
        for _ in 0..implied {
            entries.push(whole(axis));
            axis += 1;
        }
    }
    debug_assert_eq!(axis, ndim);

    Ok(entries)
}

/// The shape of the result.
///
/// The advanced group contributes its one dimension where its first axis was,
/// unless a slice or a new axis came between its members, in which case numpy
/// puts it at the front instead.
fn out_shape(entries: &[Entry], advanced: Option<&Advanced>, first: Option<usize>) -> Vec<u64> {
    let mut shape = Vec::with_capacity(entries.len() + 1);
    if let Some(advanced) = advanced {
        if !advanced.adjacent {
            shape.push(advanced.len);
        }
    }
    for (position, entry) in entries.iter().enumerate() {
        match entry {
            Entry::New => shape.push(1),
            Entry::Axis { advanced: true, .. } => {
                if let Some(advanced) = advanced {
                    if advanced.adjacent && Some(position) == first {
                        shape.push(advanced.len);
                    }
                }
            }
            Entry::Axis {
                sel,
                advanced: false,
            } => {
                if let Some(len) = sel.out_len() {
                    shape.push(len);
                }
            }
        }
    }
    shape
}

/// Broadcast the index arrays of an advanced group against each other.
///
/// One-dimensional throughout, since that is all an index can be on the wire,
/// so this is the whole of numpy's broadcasting here: a length of 1 stretches
/// to meet anything, and everything else has to agree.
fn broadcast_len(lengths: &[u64]) -> Result<u64> {
    let mut result = 1u64;
    for &len in lengths {
        if len == 1 {
            continue;
        }
        if result != 1 && result != len {
            return Err(AexError::BadSelection(format!(
                "index arrays of lengths {lengths:?} cannot be broadcast together"
            )));
        }
        result = len;
    }
    Ok(result)
}

/// Normalise one index against one axis./// Normalise one index against one axis.
fn resolve_axis(index: &Index, dim: u64, axis: usize) -> Result<AxisSel> {
    let dim_i64 = i64::try_from(dim).map_err(|_| {
        AexError::BadSelection(format!(
            "axis {axis} has {dim} elements, more than an index can address"
        ))
    })?;

    match index {
        Index::Single(i) => Ok(AxisSel::Point(normalize_index(*i, dim_i64, axis)?)),
        Index::Fancy(list) => {
            let indices = list
                .iter()
                .map(|&i| normalize_index(i, dim_i64, axis))
                .collect::<Result<Vec<u64>>>()?;
            Ok(AxisSel::Fancy(indices))
        }
        Index::Slice { start, stop, step } => resolve_slice(*start, *stop, *step, dim_i64),
        // Handled by the caller: neither consumes an axis.
        Index::Ellipsis | Index::NewAxis => unreachable!("{index:?} does not consume an axis"),
    }
}

/// Fold a possibly negative index and check it against the axis.
fn normalize_index(i: i64, dim: i64, axis: usize) -> Result<u64> {
    let folded = if i < 0 { i.saturating_add(dim) } else { i };
    if folded < 0 || folded >= dim {
        return Err(AexError::BadSelection(format!(
            "index {i} is out of bounds for axis {axis} with {dim} elements"
        )));
    }
    Ok(folded as u64)
}

/// Resolve a slice the way Python's `slice.indices` does.
///
/// Out-of-range bounds clamp rather than fail — `arr[:1000]` on ten elements is
/// ten elements, not an error — which is the one place slices and single
/// indices disagree.
fn resolve_slice(
    start: Option<i64>,
    stop: Option<i64>,
    step: Option<i64>,
    dim: i64,
) -> Result<AxisSel> {
    let step = step.unwrap_or(1);
    if step == 0 {
        return Err(AexError::BadSelection("slice step cannot be 0".to_string()));
    }

    // Walking backwards can legitimately end at -1, one before the first
    // element, so the two directions clamp to different ranges.
    let (lower, upper) = if step > 0 { (0, dim) } else { (-1, dim - 1) };
    let clamp = |bound: i64| {
        let folded = if bound < 0 {
            bound.saturating_add(dim)
        } else {
            bound
        };
        folded.clamp(lower, upper)
    };

    let start = match start {
        Some(bound) => clamp(bound),
        None if step > 0 => 0,
        None => dim - 1,
    };
    let stop = match stop {
        Some(bound) => clamp(bound),
        None if step > 0 => dim,
        None => -1,
    };

    // Ceiling division of the distance by the stride, in the direction of
    // travel. Done in i128 because negating i64::MIN, or adding a step near the
    // maximum to the distance, would overflow on the way to the same answer.
    let distance = if step > 0 {
        stop as i128 - start as i128
    } else {
        start as i128 - stop as i128
    };
    let len = if distance > 0 {
        1 + (distance - 1) / step.unsigned_abs() as i128
    } else {
        0
    };

    if len == 0 {
        // Nothing is selected, so the start would only be misleading.
        return Ok(AxisSel::Range {
            start: 0,
            len: 0,
            step,
        });
    }
    Ok(AxisSel::Range {
        start: start as u64,
        len: len as u64,
        step,
    })
}

/// How the selected bytes lie in the source.
///
/// The variants exist so that a `FETCH` at an arbitrary logical offset can find
/// its first fragment without walking the fragment sequence: without that, each
/// of N connections asking for an arbitrary offset would cost an O(n) scan and
/// parallel transfer would gain nothing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayoutKind {
    /// One run of bytes in the source. The fastest path.
    Contiguous { src_offset: u64, len: u64 },
    /// Anything else, read in output order.
    Gathered(Gathered),
}

/// A selection resolved into a logical byte stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectionLayout {
    /// Shape of the result, as it will be reported to the client.
    pub out_shape: Vec<u64>,
    /// Element type actually transferred.
    pub dtype: DType,
    /// Length of the logical byte stream.
    pub total_bytes: u64,
    pub kind: LayoutKind,
}

impl SelectionLayout {
    /// Resolve `indices` against an array of `shape` and `dtype`.
    ///
    /// `quality` has to be one the server can produce: the fallback to `EXACT`
    /// belongs to the control plane, which reports what it applied, and doing
    /// it again here would hide a shape that no longer matches the plan.
    pub fn resolve(
        shape: &[u64],
        dtype: DType,
        indices: &[Index],
        quality: &QualitySpec,
    ) -> Result<Self> {
        if !quality.is_exact() {
            return Err(AexError::UnsupportedSelection(format!(
                "{:?} encoding is not implemented; only EXACT is",
                quality.encoding
            )));
        }

        let resolved = resolve(shape, indices)?;
        let num_elements = resolved.num_elements();
        let total_bytes = num_elements.checked_mul(dtype.itemsize()).ok_or_else(|| {
            AexError::BadSelection(format!(
                "a selection of {num_elements} {dtype} elements is larger than the byte range"
            ))
        })?;

        let run = if walks_independently(&resolved) {
            contiguous_run(shape, &resolved.axes)
        } else {
            None
        };
        let kind = match run {
            Some((offset_elements, len_elements)) => {
                debug_assert_eq!(len_elements, num_elements);
                LayoutKind::Contiguous {
                    src_offset: offset_elements * dtype.itemsize(),
                    len: total_bytes,
                }
            }
            None => LayoutKind::Gathered(Gathered::new(shape, &resolved)?),
        };

        Ok(SelectionLayout {
            out_shape: resolved.out_shape,
            dtype,
            total_bytes,
            kind,
        })
    }

    /// Read `[offset, offset + dst.len())` of the logical byte stream.
    ///
    /// `read_src` reads the source array's own bytes at a byte offset; this
    /// decides which of them go where, so a backend only has to know storage.
    pub fn read_with(
        &self,
        offset: u64,
        dst: &mut [u8],
        mut read_src: impl FnMut(u64, &mut [u8]) -> Result<()>,
    ) -> Result<()> {
        self.check_range(offset, dst.len() as u64)?;
        match &self.kind {
            LayoutKind::Contiguous { src_offset, .. } => read_src(src_offset + offset, dst),
            LayoutKind::Gathered(gathered) => {
                gathered.read(self.dtype.itemsize(), offset, dst, read_src)
            }
        }
    }

    /// Where `[offset, offset + len)` of the logical stream lies in the source,
    /// when the selection is one run.
    ///
    /// `None` for a gathered selection, where one logical range is many source
    /// ranges and the caller has to walk them, and `None` for a range with
    /// nothing in it. The range is clipped to the stream, because the callers
    /// of this are asking in order to read ahead.
    pub fn source_run(&self, offset: u64, len: u64) -> Option<(u64, u64)> {
        match &self.kind {
            LayoutKind::Contiguous { src_offset, .. } => {
                let len = len.min(self.total_bytes.checked_sub(offset)?);
                (len > 0).then_some((src_offset + offset, len))
            }
            LayoutKind::Gathered(_) => None,
        }
    }

    /// Reject a range that falls outside the logical byte stream.
    pub fn check_range(&self, offset: u64, len: u64) -> Result<()> {
        let end = offset.checked_add(len);
        if end.is_none_or(|end| end > self.total_bytes) {
            return Err(AexError::OutOfRange {
                offset,
                len,
                total: self.total_bytes,
            });
        }
        Ok(())
    }
}

/// Whether walking the axes independently visits what numpy selects.
///
/// The axes of an advanced group are walked in step, and [`contiguous_run`]
/// walks them independently. The two agree only when at most one of them takes
/// more than one index, and when the group's dimension has not been moved to
/// the front.
fn walks_independently(resolved: &ResolvedSelection) -> bool {
    let Some(advanced) = &resolved.advanced else {
        return true;
    };
    let walked = advanced
        .axes
        .iter()
        .filter(|&&axis| resolved.axes[axis].len() > 1)
        .count();
    advanced.adjacent && walked <= 1
}

/// A selection that is not one run, as a walk over its output dimensions.
///
/// The source element behind an output position is `base` plus one term per
/// dimension, so any position can be reached without walking up to it. Output
/// dimensions of length 1 add a constant and are folded into `base`, which is
/// also where `NewAxis` and plain integers go.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gathered {
    /// Source element of the first output element.
    base: u64,
    /// Output dimensions longer than 1, in output order.
    dims: Vec<Dim>,
}

/// What one output dimension adds to the source element as its index moves.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Dim {
    /// `k * step` source elements. `step` may be negative.
    Stride { len: u64, step: i64 },
    /// `offsets[k]` source elements: one index array, or several walked in
    /// step, already multiplied out.
    List(Vec<u64>),
}

impl Dim {
    fn len(&self) -> u64 {
        match self {
            Dim::Stride { len, .. } => *len,
            Dim::List(offsets) => offsets.len() as u64,
        }
    }

    /// The term for index `k`. Wrapping, because a negative stride only lands
    /// on a valid element once the other terms are added.
    fn term(&self, k: u64) -> u64 {
        match self {
            Dim::Stride { step, .. } => k.wrapping_mul(*step as u64),
            Dim::List(offsets) => offsets[k as usize],
        }
    }
}

impl Gathered {
    fn new(shape: &[u64], resolved: &ResolvedSelection) -> Result<Self> {
        let too_large = || {
            AexError::BadSelection(format!(
                "an array of shape {shape:?} has more elements than an offset can address"
            ))
        };
        // Source elements per step along each axis, in C order.
        let mut strides = vec![1u64; shape.len()];
        for axis in (0..shape.len().saturating_sub(1)).rev() {
            strides[axis] = strides[axis + 1]
                .checked_mul(shape[axis + 1])
                .ok_or_else(too_large)?;
        }

        let group = resolved.advanced.as_ref();
        let mut base = 0u64;
        let mut dims = Vec::new();
        let mut group_at = None;
        for (axis, sel) in resolved.axes.iter().enumerate() {
            let stride = strides[axis];
            if group.is_some_and(|g| g.axes.contains(&axis)) {
                group_at.get_or_insert(dims.len());
                continue;
            }
            match sel {
                AxisSel::Point(index) => base += index * stride,
                AxisSel::Range { start, len, step } => {
                    base += start * stride;
                    if *len > 1 {
                        let step = i64::try_from(stride)
                            .ok()
                            .and_then(|stride| stride.checked_mul(*step))
                            .ok_or_else(too_large)?;
                        dims.push(Dim::Stride { len: *len, step });
                    }
                }
                AxisSel::Fancy(_) => unreachable!("an index array is always in the group"),
            }
        }

        if let Some(group) = group {
            // Position k takes the k-th index of every array, and the one index
            // of a scalar or a length-1 array.
            let mut offsets = vec![0u64; group.len as usize];
            for &axis in &group.axes {
                let stride = strides[axis];
                match &resolved.axes[axis] {
                    AxisSel::Point(index) => base += index * stride,
                    AxisSel::Fancy(list) if list.len() == 1 => base += list[0] * stride,
                    AxisSel::Fancy(list) => {
                        for (offset, index) in offsets.iter_mut().zip(list) {
                            *offset += index * stride;
                        }
                    }
                    AxisSel::Range { .. } => unreachable!("a slice is never in the group"),
                }
            }
            if group.len > 1 {
                let at = if group.adjacent {
                    group_at.unwrap_or(0)
                } else {
                    0
                };
                dims.insert(at, Dim::List(offsets));
            }
        }

        Ok(Gathered { base, dims })
    }

    /// Call `f(source element, count)` for each run of consecutive source
    /// elements behind output elements `[first, first + count)`, in order.
    fn for_each_run(
        &self,
        first: u64,
        count: u64,
        mut f: impl FnMut(u64, u64) -> Result<()>,
    ) -> Result<()> {
        let ndim = self.dims.len();

        let mut index = vec![0u64; ndim];
        let mut rest = first;
        for (i, dim) in self.dims.iter().enumerate().rev() {
            index[i] = rest % dim.len();
            rest /= dim.len();
        }

        // prefix[i + 1] is the source element with the dimensions past i at 0.
        let mut prefix = vec![self.base; ndim + 1];
        let refresh = |prefix: &mut [u64], index: &[u64], from: usize| {
            for i in from..ndim {
                prefix[i + 1] = prefix[i].wrapping_add(self.dims[i].term(index[i]));
            }
        };
        refresh(&mut prefix, &index, 0);

        let mut run: Option<(u64, u64)> = None;
        for _ in 0..count {
            let element = prefix[ndim];
            run = match run {
                Some((start, len)) if start + len == element => Some((start, len + 1)),
                Some((start, len)) => {
                    f(start, len)?;
                    Some((element, 1))
                }
                None => Some((element, 1)),
            };

            // Advance like an odometer, recomputing only what changed.
            let mut i = ndim;
            while i > 0 {
                i -= 1;
                index[i] += 1;
                if index[i] < self.dims[i].len() {
                    break;
                }
                index[i] = 0;
            }
            refresh(&mut prefix, &index, i);
        }
        match run {
            Some((start, len)) => f(start, len),
            None => Ok(()),
        }
    }

    /// Fill `dst` with `[offset, offset + dst.len())` of the logical stream.
    fn read(
        &self,
        itemsize: u64,
        offset: u64,
        dst: &mut [u8],
        mut read_src: impl FnMut(u64, &mut [u8]) -> Result<()>,
    ) -> Result<()> {
        let end = offset + dst.len() as u64;
        let first = offset / itemsize;
        let count = end.div_ceil(itemsize) - first;

        // A chunk need not start or end on an element boundary. Then whole
        // elements are read aside and the requested bytes cut out of them.
        let skip = (offset - first * itemsize) as usize;
        let aligned = skip == 0 && end % itemsize == 0;
        let mut aside = Vec::new();
        let out: &mut [u8] = if aligned {
            &mut *dst
        } else {
            aside.resize((count * itemsize) as usize, 0);
            &mut aside
        };

        // Runs close together in the source are read as one span, then cut
        // out of it by walking the same runs again.
        // ponytail: the walk is per element (~600 MiB/s for every other f32);
        // copying a strided innermost dimension in one loop is the upgrade.
        let mut span: Option<Span> = None;
        let mut next = first;
        self.for_each_run(first, count, |element, len| {
            let joined = span
                .as_mut()
                .is_some_and(|w| w.join(element, len, itemsize));
            if !joined {
                if let Some(w) = span.take() {
                    self.read_span(&w, first, itemsize, out, &mut read_src)?;
                }
                span = Some(Span::new(element, len, next));
            }
            next += len;
            Ok(())
        })?;
        if let Some(w) = span {
            self.read_span(&w, first, itemsize, out, &mut read_src)?;
        }

        if !aligned {
            let len = dst.len();
            dst.copy_from_slice(&aside[skip..skip + len]);
        }
        Ok(())
    }
}

/// Runs further apart than this are read separately: the bytes between them
/// would cost more to read than the syscall they save.
const SPAN_GAP_BYTES: u64 = 4096;

/// Ceiling on one span read, which is also the scratch buffer it needs.
const SPAN_MAX_BYTES: u64 = 1 << 20;

thread_local! {
    // Kept per thread so that a transfer in progress does not allocate.
    static SCRATCH: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Source elements `[lo, hi)` holding output elements `[out_first, +out_count)`.
struct Span {
    lo: u64,
    hi: u64,
    out_first: u64,
    out_count: u64,
    runs: u32,
}

impl Span {
    fn new(element: u64, len: u64, out_first: u64) -> Self {
        Span {
            lo: element,
            hi: element + len,
            out_first,
            out_count: len,
            runs: 1,
        }
    }

    /// Take in the next run if it lies close enough, in either direction.
    fn join(&mut self, element: u64, len: u64, itemsize: u64) -> bool {
        let gap = SPAN_GAP_BYTES / itemsize;
        let (lo, hi) = (self.lo.min(element), self.hi.max(element + len));
        let near = element + len + gap >= self.lo && element <= self.hi + gap;
        if !near || (hi - lo) * itemsize > SPAN_MAX_BYTES {
            return false;
        }
        (self.lo, self.hi) = (lo, hi);
        self.out_count += len;
        self.runs += 1;
        true
    }
}

impl Gathered {
    /// Read the elements of `span` into `out`, whose first element is output
    /// element `out_base`.
    fn read_span(
        &self,
        span: &Span,
        out_base: u64,
        itemsize: u64,
        out: &mut [u8],
        read_src: &mut impl FnMut(u64, &mut [u8]) -> Result<()>,
    ) -> Result<()> {
        let at = ((span.out_first - out_base) * itemsize) as usize;
        let bytes = (span.out_count * itemsize) as usize;
        if span.runs == 1 {
            return read_src(span.lo * itemsize, &mut out[at..at + bytes]);
        }

        SCRATCH.with(|scratch| {
            let mut scratch = scratch.borrow_mut();
            let len = ((span.hi - span.lo) * itemsize) as usize;
            if scratch.len() < len {
                scratch.resize(len, 0);
            }
            read_src(span.lo * itemsize, &mut scratch[..len])?;

            let mut pos = at;
            self.for_each_run(span.out_first, span.out_count, |element, count| {
                let from = ((element - span.lo) * itemsize) as usize;
                let n = (count * itemsize) as usize;
                out[pos..pos + n].copy_from_slice(&scratch[from..from + n]);
                pos += n;
                Ok(())
            })
        })
    }
}

/// The one run of source elements a selection covers, if it is one.
///
/// Returns `(offset, len)` in elements. A selection is one run when every axis
/// takes consecutive indices and, past the first axis that takes more than one,
/// every axis takes all of its own: anything else leaves gaps between the rows.
fn contiguous_run(shape: &[u64], axes: &[AxisSel]) -> Option<(u64, u64)> {
    debug_assert_eq!(shape.len(), axes.len());

    let runs: Vec<(u64, u64)> = axes.iter().map(AxisSel::as_run).collect::<Option<_>>()?;

    // An empty selection is a run of nothing, wherever it nominally starts.
    if runs.iter().any(|(_, len)| *len == 0) {
        return Some((0, 0));
    }

    if let Some(first_multi) = runs.iter().position(|(_, len)| *len > 1) {
        for (axis, &(start, len)) in runs.iter().enumerate().skip(first_multi + 1) {
            if start != 0 || len != shape[axis] {
                return None;
            }
        }
    }

    // C order, so each axis contributes its start times everything below it.
    // Computed in u128 because the intermediate products are only bounded by
    // the element count once the whole shape has been folded in.
    let mut offset = 0u128;
    let mut len = 1u128;
    for (axis, &(start, run_len)) in runs.iter().enumerate() {
        offset = offset * shape[axis] as u128 + start as u128;
        len *= run_len as u128;
    }
    Some((u64::try_from(offset).ok()?, u64::try_from(len).ok()?))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `resolve`, panicking on failure, for the many cases that must succeed.
    fn axes(shape: &[u64], indices: &[Index]) -> Vec<AxisSel> {
        resolve(shape, indices)
            .expect("selection must resolve")
            .axes
    }

    fn out_shape(shape: &[u64], indices: &[Index]) -> Vec<u64> {
        resolve(shape, indices)
            .expect("selection must resolve")
            .out_shape
    }

    /// The contiguous byte range of a selection, in elements.
    fn run(shape: &[u64], indices: &[Index]) -> Option<(u64, u64)> {
        contiguous_run(shape, &axes(shape, indices))
    }

    fn layout(shape: &[u64], dtype: DType, indices: &[Index]) -> Result<SelectionLayout> {
        SelectionLayout::resolve(shape, dtype, indices, &QualitySpec::exact())
    }

    #[test]
    fn an_empty_selection_is_the_whole_array() {
        let resolved = resolve(&[4, 3], &[]).expect("resolve");
        assert_eq!(resolved.axes, vec![AxisSel::whole(4), AxisSel::whole(3)]);
        assert_eq!(resolved.out_shape, vec![4, 3]);
        assert_eq!(resolved.num_elements(), 12);
    }

    #[test]
    fn axes_the_selection_leaves_out_are_taken_whole() {
        // arr[5] on a 3-d array means arr[5, :, :].
        assert_eq!(
            axes(&[10, 4, 3], &[Index::Single(5)]),
            vec![AxisSel::Point(5), AxisSel::whole(4), AxisSel::whole(3)]
        );
        assert_eq!(out_shape(&[10, 4, 3], &[Index::Single(5)]), vec![4, 3]);
    }

    #[test]
    fn an_ellipsis_stands_for_the_axes_in_between() {
        let indices = [Index::Ellipsis, Index::Single(0)];
        assert_eq!(
            axes(&[2, 3, 4], &indices),
            vec![AxisSel::whole(2), AxisSel::whole(3), AxisSel::Point(0)]
        );
        assert_eq!(out_shape(&[2, 3, 4], &indices), vec![2, 3]);

        // An ellipsis that stands for nothing is still allowed.
        let indices = [Index::Single(1), Index::Ellipsis, Index::Single(2)];
        assert_eq!(
            axes(&[4, 5], &indices),
            vec![AxisSel::Point(1), AxisSel::Point(2)]
        );
        assert!(out_shape(&[4, 5], &indices).is_empty());
    }

    #[test]
    fn a_new_axis_adds_a_length_one_axis_without_consuming_one() {
        let indices = [Index::full(), Index::NewAxis];
        assert_eq!(axes(&[6], &indices), vec![AxisSel::whole(6)]);
        assert_eq!(out_shape(&[6], &indices), vec![6, 1]);

        // It can also come first, or be the whole selection.
        assert_eq!(
            out_shape(&[6], &[Index::NewAxis, Index::full()]),
            vec![1, 6]
        );
        assert_eq!(out_shape(&[6], &[Index::NewAxis]), vec![1, 6]);
    }

    #[test]
    fn negative_indices_count_from_the_end() {
        assert_eq!(axes(&[10], &[Index::Single(-1)]), vec![AxisSel::Point(9)]);
        assert_eq!(axes(&[10], &[Index::Single(-10)]), vec![AxisSel::Point(0)]);
        assert_eq!(
            axes(&[10], &[Index::Fancy(vec![-1, -2, 3])]),
            vec![AxisSel::Fancy(vec![9, 8, 3])]
        );
    }

    #[test]
    fn slices_resolve_the_way_python_does() {
        let slice =
            |start, stop, step| resolve_slice(start, stop, step, 10).expect("slice must resolve");
        let range = |start: u64, len: u64, step: i64| AxisSel::Range { start, len, step };

        assert_eq!(slice(None, None, None), range(0, 10, 1));
        assert_eq!(slice(Some(2), Some(5), None), range(2, 3, 1));
        assert_eq!(slice(Some(2), Some(9), Some(3)), range(2, 3, 3));
        // Bounds past the end clamp rather than fail.
        assert_eq!(slice(Some(5), Some(1000), None), range(5, 5, 1));
        assert_eq!(slice(Some(-100), Some(3), None), range(0, 3, 1));
        // Negative bounds count from the end.
        assert_eq!(slice(Some(-3), None, None), range(7, 3, 1));
        assert_eq!(slice(None, Some(-8), None), range(0, 2, 1));
        // Backwards.
        assert_eq!(slice(None, None, Some(-1)), range(9, 10, -1));
        assert_eq!(slice(Some(8), Some(2), Some(-2)), range(8, 3, -2));
        // Empty in every way it can be empty.
        for empty in [
            slice(Some(5), Some(5), None),
            slice(Some(8), Some(2), None),
            slice(Some(2), Some(8), Some(-1)),
            slice(Some(100), None, None),
        ] {
            assert_eq!(empty.len(), 0, "{empty:?}");
        }
    }

    #[test]
    fn a_zero_length_axis_selects_nothing() {
        let resolved = resolve(&[0, 5], &[]).expect("resolve");
        assert_eq!(resolved.out_shape, vec![0, 5]);
        assert_eq!(resolved.num_elements(), 0);
        // Any slice of an empty axis is empty, as numpy has it.
        assert_eq!(
            axes(&[0], &[Index::range(0, 10)]),
            vec![AxisSel::Range {
                start: 0,
                len: 0,
                step: 1
            }]
        );
        // But a single index has nothing to name.
        assert!(resolve(&[0], &[Index::Single(0)]).is_err());
    }

    #[test]
    fn selections_numpy_rejects_are_rejected() {
        let bad = |shape: &[u64], indices: &[Index]| {
            let err = resolve(shape, indices).expect_err("must be rejected");
            assert!(matches!(err, AexError::BadSelection(_)), "{err}");
            assert_eq!(err.class(), crate::ErrorClass::Request);
        };

        bad(&[10], &[Index::Single(10)]);
        bad(&[10], &[Index::Single(-11)]);
        bad(&[10], &[Index::Fancy(vec![0, 10])]);
        bad(&[4, 3], &[Index::full(), Index::full(), Index::full()]);
        bad(&[4, 3], &[Index::Ellipsis, Index::Ellipsis]);
        bad(
            &[10],
            &[Index::Slice {
                start: None,
                stop: None,
                step: Some(0),
            }],
        );
    }

    #[test]
    fn a_whole_array_is_one_run() {
        assert_eq!(run(&[10], &[]), Some((0, 10)));
        assert_eq!(run(&[10, 4], &[]), Some((0, 40)));
        assert_eq!(run(&[10, 4], &[Index::full()]), Some((0, 40)));
        assert_eq!(run(&[], &[]), Some((0, 1)));
    }

    #[test]
    fn leading_axes_taken_one_at_a_time_stay_one_run() {
        // arr[5] on (10, 4, 3) is row 5: 12 elements from element 60.
        assert_eq!(run(&[10, 4, 3], &[Index::Single(5)]), Some((60, 12)));
        // arr[5, 2] is 3 elements from 60 + 2*3.
        assert_eq!(
            run(&[10, 4, 3], &[Index::Single(5), Index::Single(2)]),
            Some((66, 3))
        );
        // Down to a single element.
        assert_eq!(
            run(
                &[10, 4, 3],
                &[Index::Single(5), Index::Single(2), Index::Single(1)]
            ),
            Some((67, 1))
        );
    }

    #[test]
    fn a_step_one_slice_of_the_leading_axis_is_one_run() {
        assert_eq!(run(&[10, 4], &[Index::range(2, 5)]), Some((8, 12)));
        // Preceded by single indices, it still is.
        assert_eq!(
            run(&[6, 10, 4], &[Index::Single(1), Index::range(2, 5)]),
            Some((48, 12))
        );
        // A slice of the last axis is one run only because nothing follows it.
        assert_eq!(
            run(&[6, 10], &[Index::Single(1), Index::range(2, 5)]),
            Some((12, 3))
        );
    }

    #[test]
    fn a_selection_that_leaves_gaps_is_not_one_run() {
        // arr[:, 0:2] on (4, 3) skips a column of every row.
        assert_eq!(run(&[4, 3], &[Index::full(), Index::range(0, 2)]), None);
        // A stride leaves gaps of its own.
        let every_other = Index::Slice {
            start: None,
            stop: None,
            step: Some(2),
        };
        assert_eq!(run(&[10], std::slice::from_ref(&every_other)), None);
        assert_eq!(run(&[4, 10], &[Index::full(), every_other]), None);
        // So does a reversal.
        let reversed = Index::Slice {
            start: None,
            stop: None,
            step: Some(-1),
        };
        assert_eq!(run(&[10], &[reversed]), None);
        // And a fancy list that jumps.
        assert_eq!(run(&[10], &[Index::Fancy(vec![1, 5, 9])]), None);
        assert_eq!(run(&[10], &[Index::Fancy(vec![3, 2, 1])]), None);
    }

    #[test]
    fn an_axis_selecting_one_index_does_not_break_the_run() {
        // A slice of one, or a fancy list of one, is a run just as a single
        // index is; only the output shape differs.
        assert_eq!(run(&[4, 3], &[Index::full(), Index::range(1, 2)]), None);
        assert_eq!(run(&[4, 3], &[Index::range(1, 2)]), Some((3, 3)));
        assert_eq!(run(&[4, 3], &[Index::Fancy(vec![1])]), Some((3, 3)));
        // Consecutive fancy indices name a run like a slice does.
        assert_eq!(run(&[10], &[Index::Fancy(vec![4, 5, 6])]), Some((4, 3)));
        // An axis of one element is whole however it is selected.
        assert_eq!(
            run(&[4, 1], &[Index::range(1, 3), Index::Single(0)]),
            Some((1, 2))
        );
    }

    #[test]
    fn an_empty_selection_is_one_run_of_nothing() {
        assert_eq!(run(&[10, 4], &[Index::range(5, 5)]), Some((0, 0)));
        assert_eq!(run(&[0, 4], &[]), Some((0, 0)));
        // Even where the rest of the selection would not have been a run.
        assert_eq!(
            run(&[10, 4], &[Index::range(5, 5), Index::range(0, 2)]),
            Some((0, 0))
        );
    }

    #[test]
    fn a_layout_carries_the_shape_dtype_and_byte_range() {
        let layout = layout(&[1000, 200], DType::Float32, &[Index::range(10, 20)]).expect("layout");
        assert_eq!(layout.out_shape, vec![10, 200]);
        assert_eq!(layout.dtype, DType::Float32);
        assert_eq!(layout.total_bytes, 10 * 200 * 4);
        assert_eq!(
            layout.kind,
            LayoutKind::Contiguous {
                src_offset: 10 * 200 * 4,
                len: 10 * 200 * 4,
            }
        );
    }

    #[test]
    fn a_scalar_selection_is_one_element() {
        let layout = layout(
            &[10, 4],
            DType::Float64,
            &[Index::Single(3), Index::Single(1)],
        )
        .expect("layout");
        assert!(layout.out_shape.is_empty());
        assert_eq!(layout.total_bytes, 8);
        assert_eq!(
            layout.kind,
            LayoutKind::Contiguous {
                src_offset: 13 * 8,
                len: 8
            }
        );
    }

    #[test]
    fn a_contiguous_layout_says_where_a_range_comes_from() {
        // Row 10 of a (100, 200) int32 array: one run, 200 * 4 bytes in.
        let layout = layout(&[100, 200], DType::Int32, &[Index::Single(10)]).expect("layout");
        let base = 10 * 200 * 4;
        assert_eq!(layout.source_run(0, 40), Some((base, 40)));
        assert_eq!(layout.source_run(40, 40), Some((base + 40, 40)));
        // Read-ahead asks past the end, and gets what is there.
        assert_eq!(layout.source_run(600, 1000), Some((base + 600, 200)));
        assert_eq!(layout.source_run(800, 8), None);
    }

    #[test]
    fn a_gathered_layout_says_a_range_is_not_one_run() {
        let layout = layout(
            &[100, 200],
            DType::Int32,
            &[Index::full(), Index::range(0, 100)],
        )
        .expect("layout");
        assert!(matches!(layout.kind, LayoutKind::Gathered(_)));
        assert_eq!(layout.source_run(0, 40), None);
    }

    #[test]
    fn an_empty_layout_has_no_bytes() {
        let layout = layout(&[10], DType::Int32, &[Index::range(4, 4)]).expect("layout");
        assert_eq!(layout.out_shape, vec![0]);
        assert_eq!(layout.total_bytes, 0);
        assert_eq!(
            layout.kind,
            LayoutKind::Contiguous {
                src_offset: 0,
                len: 0
            }
        );
        // A range of nothing is the only one in bounds.
        layout.check_range(0, 0).expect("empty range");
        assert!(layout.check_range(0, 1).is_err());
    }

    #[test]
    fn an_encoding_the_layout_cannot_apply_is_refused() {
        // The control plane resolves the fallback before it gets here, so a
        // quality this code cannot honour means the two disagree.
        let quality = QualitySpec {
            encoding: crate::quality::Encoding::Subsample,
            ..QualitySpec::default()
        };
        let err = SelectionLayout::resolve(&[10], DType::Int8, &[], &quality).unwrap_err();
        assert!(matches!(err, AexError::UnsupportedSelection(_)), "{err}");
    }

    #[test]
    fn ranges_outside_the_logical_stream_are_rejected() {
        let layout = layout(&[10], DType::Int32, &[]).expect("layout");
        assert_eq!(layout.total_bytes, 40);

        layout.check_range(0, 40).expect("the whole stream");
        layout
            .check_range(40, 0)
            .expect("an empty range at the end");
        layout.check_range(8, 16).expect("the middle");
        assert!(layout.check_range(0, 41).is_err());
        assert!(layout.check_range(40, 1).is_err());
        // Offset plus length overflowing must not wrap into a valid range.
        assert!(layout.check_range(u64::MAX, 8).is_err());
    }

    #[test]
    fn advanced_indexing_matches_numpy() {
        // An index array makes the whole selection advanced: the integers
        // beside it join the group, the group broadcasts to one dimension, and
        // that dimension goes where the group was — or at the front, if a slice
        // came between its members. Every line here was checked against numpy.
        let cases: Vec<(&[u64], Vec<Index>, Vec<u64>)> = vec![
            // One array, on its own: the axes it does not name are untouched.
            (&[4, 5, 6], vec![Index::Fancy(vec![0, 1])], vec![2, 5, 6]),
            (
                &[4, 5, 6],
                vec![Index::full(), Index::Fancy(vec![0, 1])],
                vec![4, 2, 6],
            ),
            // Two arrays side by side pick pairwise, not as a grid.
            (
                &[4, 5, 6],
                vec![Index::Fancy(vec![0, 1]), Index::Fancy(vec![2, 3])],
                vec![2, 6],
            ),
            (
                &[4, 5, 6],
                vec![Index::Fancy(vec![0]), Index::Fancy(vec![1])],
                vec![1, 6],
            ),
            (
                &[4, 5, 6],
                vec![
                    Index::Fancy(vec![0, 1, 2]),
                    Index::Fancy(vec![0, 1, 2]),
                    Index::Fancy(vec![0, 1, 2]),
                ],
                vec![3],
            ),
            // A length of one stretches to meet the others.
            (
                &[4, 5, 6],
                vec![Index::Fancy(vec![0]), Index::Fancy(vec![1, 2])],
                vec![2, 6],
            ),
            (
                &[4, 5, 6],
                vec![Index::Fancy(vec![0, 1]), Index::Fancy(vec![2])],
                vec![2, 6],
            ),
            // An integer beside an array is part of the same group, so it adds
            // no dimension of its own and does not break the group up.
            (
                &[4, 5, 6],
                vec![Index::Fancy(vec![0, 1]), Index::Single(2)],
                vec![2, 6],
            ),
            (
                &[4, 5, 6],
                vec![Index::Single(2), Index::Fancy(vec![0, 1])],
                vec![2, 6],
            ),
            (
                &[4, 5, 6],
                vec![Index::Single(0), Index::Fancy(vec![1, 2]), Index::Single(3)],
                vec![2],
            ),
            (
                &[4, 5, 6],
                vec![Index::full(), Index::Fancy(vec![0, 1]), Index::Single(2)],
                vec![4, 2],
            ),
            // Separated by a slice: the dimension moves to the front.
            (
                &[4, 5, 6],
                vec![
                    Index::Fancy(vec![0, 1]),
                    Index::full(),
                    Index::Fancy(vec![2, 3]),
                ],
                vec![2, 5],
            ),
            (
                &[4, 5, 6],
                vec![Index::Single(0), Index::full(), Index::Fancy(vec![1, 2])],
                vec![2, 5],
            ),
            (
                &[4, 5, 6],
                vec![Index::Fancy(vec![1, 2]), Index::full(), Index::Single(0)],
                vec![2, 5],
            ),
            // Adjacent, so it stays where it was.
            (
                &[4, 5, 6],
                vec![Index::full(), Index::Single(0), Index::Fancy(vec![1, 2])],
                vec![4, 2],
            ),
            // An ellipsis that stands for an axis separates; one that stands
            // for nothing does not.
            (
                &[4, 5, 6],
                vec![
                    Index::Fancy(vec![0, 1]),
                    Index::Ellipsis,
                    Index::Fancy(vec![2, 3]),
                ],
                vec![2, 5],
            ),
            (
                &[4, 5, 6, 7],
                vec![
                    Index::Fancy(vec![0, 1]),
                    Index::Ellipsis,
                    Index::Fancy(vec![2, 3]),
                ],
                vec![2, 5, 6],
            ),
            (
                &[4, 5, 6],
                vec![Index::Ellipsis, Index::Fancy(vec![0, 1])],
                vec![4, 5, 2],
            ),
            (
                &[4, 5, 6],
                vec![Index::Fancy(vec![0, 1]), Index::Ellipsis],
                vec![2, 5, 6],
            ),
            // A new axis separates too, and keeps its own place.
            (
                &[4, 5, 6],
                vec![
                    Index::Fancy(vec![0, 1]),
                    Index::NewAxis,
                    Index::Fancy(vec![2, 3]),
                ],
                vec![2, 1, 6],
            ),
            (
                &[4, 5, 6],
                vec![Index::Fancy(vec![0, 1]), Index::NewAxis],
                vec![2, 1, 5, 6],
            ),
            (
                &[4, 5, 6],
                vec![Index::NewAxis, Index::Fancy(vec![0, 1])],
                vec![1, 2, 5, 6],
            ),
            // An empty array selects nothing, and broadcasts against a scalar.
            (&[4, 5, 6], vec![Index::Fancy(vec![])], vec![0, 5, 6]),
            (
                &[4, 5, 6],
                vec![Index::Fancy(vec![]), Index::Fancy(vec![1])],
                vec![0, 6],
            ),
            // No array at all, so the integers stay ordinary and drop their axes.
            (
                &[4, 5, 6],
                vec![Index::Single(0), Index::full(), Index::Single(1)],
                vec![5],
            ),
        ];

        for (shape, indices, expected) in cases {
            let resolved = resolve(shape, &indices)
                .unwrap_or_else(|e| panic!("{indices:?} on {shape:?}: {e}"));
            assert_eq!(resolved.out_shape, expected, "{indices:?} on {shape:?}");
            // The element count has to follow the shape, not the axes: the axes
            // of a group are walked in step.
            assert_eq!(
                resolved.num_elements(),
                expected.iter().product::<u64>(),
                "{indices:?}"
            );
        }
    }

    #[test]
    fn index_arrays_that_cannot_broadcast_are_rejected() {
        // numpy raises IndexError here, and so must this.
        let err = resolve(
            &[4, 5, 6],
            &[Index::Fancy(vec![0, 1]), Index::Fancy(vec![2, 3, 4])],
        )
        .unwrap_err();
        assert!(matches!(err, AexError::BadSelection(_)), "{err}");
        assert_eq!(err.class(), crate::ErrorClass::Request);

        // Lengths of one stretch, so these do broadcast.
        resolve(&[4, 5], &[Index::Fancy(vec![0]), Index::Fancy(vec![1, 3])]).expect("broadcasts");
    }

    /// Read `[offset, offset + len)` of a layout over a source whose elements
    /// are their own flat index, as little-endian u32.
    fn read_indices(layout: &SelectionLayout, offset: u64, len: u64) -> Vec<u8> {
        let mut dst = vec![0u8; len as usize];
        layout
            .read_with(offset, &mut dst, |at, buf| {
                for (i, byte) in buf.iter_mut().enumerate() {
                    let pos = at + i as u64;
                    *byte = ((pos / 4) as u32).to_le_bytes()[(pos % 4) as usize];
                }
                Ok(())
            })
            .expect("read");
        dst
    }

    #[test]
    fn a_gathered_selection_matches_numpy() {
        // Every expectation here is what numpy returns for np.arange(n) of the
        // same shape.
        // Source shape, selection, result shape, result.
        type Case = (&'static [u64], Vec<Index>, Vec<u64>, Vec<u32>);
        let cases: Vec<Case> = vec![
            (
                &[4, 6],
                vec![Index::range(1, 3), Index::range(2, 5)],
                vec![2, 3],
                vec![8, 9, 10, 14, 15, 16],
            ),
            (
                &[4, 6],
                vec![Index::Slice {
                    start: None,
                    stop: None,
                    step: Some(2),
                }],
                vec![2, 6],
                vec![0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17],
            ),
            (
                &[4, 6],
                vec![
                    Index::full(),
                    Index::Slice {
                        start: None,
                        stop: None,
                        step: Some(-2),
                    },
                ],
                vec![4, 3],
                vec![5, 3, 1, 11, 9, 7, 17, 15, 13, 23, 21, 19],
            ),
            (
                &[4, 6],
                vec![Index::Fancy(vec![3, 0, 3])],
                vec![3, 6],
                vec![
                    18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 18, 19, 20, 21, 22, 23,
                ],
            ),
            (
                &[4, 6],
                vec![Index::full(), Index::Single(4)],
                vec![4],
                vec![4, 10, 16, 22],
            ),
            (
                &[4, 6],
                vec![Index::Fancy(vec![0, 2]), Index::Fancy(vec![1, 5])],
                vec![2],
                vec![1, 17],
            ),
            (
                &[4, 6],
                vec![Index::Fancy(vec![3]), Index::Fancy(vec![0, 1, 2])],
                vec![3],
                vec![18, 19, 20],
            ),
            (
                &[4, 6],
                vec![Index::NewAxis, Index::Fancy(vec![2, 1]), Index::NewAxis],
                vec![1, 2, 1, 6],
                vec![12, 13, 14, 15, 16, 17, 6, 7, 8, 9, 10, 11],
            ),
            (
                &[3, 4, 5],
                vec![
                    Index::Fancy(vec![0, 2]),
                    Index::full(),
                    Index::Fancy(vec![4, 1]),
                ],
                vec![2, 4],
                vec![4, 9, 14, 19, 41, 46, 51, 56],
            ),
            (
                &[3, 4, 5],
                vec![
                    Index::Single(1),
                    Index::range(1, 3),
                    Index::Fancy(vec![0, 3]),
                ],
                vec![2, 2],
                vec![25, 30, 28, 33],
            ),
            (
                &[3, 4, 5],
                vec![Index::Ellipsis, Index::Fancy(vec![1, 0])],
                vec![3, 4, 2],
                vec![
                    1, 0, 6, 5, 11, 10, 16, 15, 21, 20, 26, 25, 31, 30, 36, 35, 41, 40, 46, 45, 51,
                    50, 56, 55,
                ],
            ),
            (
                &[3, 4, 5],
                vec![
                    Index::Slice {
                        start: None,
                        stop: None,
                        step: Some(-1),
                    },
                    Index::Fancy(vec![1, 2]),
                    Index::Fancy(vec![0]),
                ],
                vec![3, 2],
                vec![45, 50, 25, 30, 5, 10],
            ),
        ];

        for (shape, indices, shape_out, expected) in cases {
            let layout = layout(shape, DType::Uint32, &indices)
                .unwrap_or_else(|e| panic!("{indices:?}: {e}"));
            assert_eq!(layout.out_shape, shape_out, "{indices:?}");
            let expected: Vec<u8> = expected.iter().flat_map(|v| v.to_le_bytes()).collect();
            assert_eq!(layout.total_bytes, expected.len() as u64, "{indices:?}");
            assert_eq!(
                read_indices(&layout, 0, layout.total_bytes),
                expected,
                "{indices:?}"
            );

            // Any cut of the stream, element boundary or not, reads the same.
            for chunk in [1, 3, 4, 7] {
                let mut joined = Vec::new();
                for offset in (0..layout.total_bytes).step_by(chunk) {
                    let len = (chunk as u64).min(layout.total_bytes - offset);
                    joined.extend(read_indices(&layout, offset, len));
                }
                assert_eq!(joined, expected, "{indices:?} in chunks of {chunk}");
            }
        }
    }

    #[test]
    fn a_selection_that_is_one_run_is_still_read_as_one() {
        // One array and one integer side by side walk the same either way.
        let served = layout(
            &[4, 5],
            DType::Int8,
            &[Index::Single(1), Index::Fancy(vec![0, 1])],
        )
        .expect("layout");
        assert_eq!(served.out_shape, vec![2]);
        assert_eq!(
            served.kind,
            LayoutKind::Contiguous {
                src_offset: 5,
                len: 2
            }
        );

        // So does a group that only ever names one element.
        let served = layout(
            &[4, 5],
            DType::Int8,
            &[Index::Fancy(vec![0]), Index::Fancy(vec![1])],
        )
        .expect("both arrays hold one index");
        assert_eq!(served.out_shape, vec![1], "numpy gives (1,), not (1, 1)");
        assert!(matches!(served.kind, LayoutKind::Contiguous { .. }));
    }

    /// The `(offset, len)` of each source read behind reading all of a layout.
    fn reads_of(layout: &SelectionLayout) -> Vec<(u64, usize)> {
        let mut reads = Vec::new();
        let mut dst = vec![0u8; layout.total_bytes as usize];
        layout
            .read_with(0, &mut dst, |at, buf| {
                reads.push((at, buf.len()));
                Ok(())
            })
            .expect("read");
        reads
    }

    #[test]
    fn nearby_elements_are_read_together() {
        // Two rows of a 2-d slice, three bytes apart: one read covers both.
        let rows = layout(
            &[4, 6],
            DType::Uint8,
            &[Index::range(1, 3), Index::range(2, 5)],
        )
        .expect("layout");
        assert_eq!(reads_of(&rows), vec![(8, 9)]);

        // Every other element of a large array: one read per span, not per
        // element.
        let strided = layout(
            &[1 << 22],
            DType::Uint32,
            &[Index::Slice {
                start: None,
                stop: None,
                step: Some(2),
            }],
        )
        .expect("layout");
        let reads = reads_of(&strided);
        assert_eq!(reads.len(), 16, "16 MiB of source in 1 MiB spans");
        assert!(reads.iter().all(|(_, len)| *len as u64 <= SPAN_MAX_BYTES));

        // Elements far apart are not worth the bytes between them.
        let sparse =
            layout(&[1 << 20], DType::Uint8, &[Index::Fancy(vec![0, 100_000])]).expect("layout");
        assert_eq!(reads_of(&sparse), vec![(0, 1), (100_000, 1)]);
    }

    #[test]
    fn span_reads_match_numpy_on_a_large_array() {
        // Wide enough that spans are cut by both limits, in both directions.
        let shape = [300u64, 5000];
        let at = |r: u64, c: u64| (r * shape[1] + c) as u32;
        let every = |step: i64| Index::Slice {
            start: None,
            stop: None,
            step: Some(step),
        };
        let rows = [299u64, 0, 1, 150, 151, 7];
        type Case = (Vec<Index>, Vec<u32>);
        let cases: Vec<Case> = vec![
            (
                vec![every(2), every(3)],
                (0..300)
                    .step_by(2)
                    .flat_map(|r| (0..5000).step_by(3).map(move |c| at(r, c)))
                    .collect(),
            ),
            (
                vec![Index::full(), every(-7)],
                (0..300)
                    .flat_map(|r| (0..5000).rev().step_by(7).map(move |c| at(r, c)))
                    .collect(),
            ),
            (
                vec![
                    Index::Fancy(rows.iter().map(|&r| r as i64).collect()),
                    every(2),
                ],
                rows.iter()
                    .flat_map(|&r| (0..5000).step_by(2).map(move |c| at(r, c)))
                    .collect(),
            ),
        ];
        for (indices, expected) in cases {
            let layout = layout(&shape, DType::Uint32, &indices).expect("layout");
            let expected: Vec<u8> = expected.iter().flat_map(|v| v.to_le_bytes()).collect();
            for chunk in [layout.total_bytes, 1 << 20, 65_537] {
                let mut joined = Vec::new();
                for offset in (0..layout.total_bytes).step_by(chunk as usize) {
                    let len = chunk.min(layout.total_bytes - offset);
                    joined.extend(read_indices(&layout, offset, len));
                }
                assert!(joined == expected, "{indices:?} in chunks of {chunk}");
            }
        }
    }

    #[test]
    fn a_selection_matches_what_numpy_would_have_taken() {
        // The source is the flat index of each element, so the expected result
        // of a selection can be written out by hand and compared to the run.
        let shape = [4u64, 3];
        let cases: Vec<(Vec<Index>, Vec<u64>)> = vec![
            (vec![], (0..12).collect()),
            (vec![Index::Single(0)], vec![0, 1, 2]),
            (vec![Index::Single(-1)], vec![9, 10, 11]),
            (vec![Index::range(1, 3)], (3..9).collect()),
            (vec![Index::Single(2), Index::Single(1)], vec![7]),
            (vec![Index::Ellipsis, Index::NewAxis], (0..12).collect()),
            (vec![Index::Fancy(vec![1, 2])], (3..9).collect()),
        ];
        for (indices, expected) in cases {
            let (offset, len) = run(&shape, &indices).unwrap_or_else(|| {
                panic!("{indices:?} should have been one run");
            });
            let selected: Vec<u64> = (offset..offset + len).collect();
            assert_eq!(selected, expected, "{indices:?}");
        }
    }
}
