//! numpy reductions over a selection, computed where the data is.
//!
//! The selection is read in blocks and folded into one accumulator per output
//! element, so memory stays bounded whatever the selection's size. Results
//! follow numpy's dtype and NaN rules; floating sums are not bit-identical,
//! since numpy adds pairwise and a stream cannot.

use half::f16;

use crate::backend::ArrayDataset;
use crate::dtype::DType;
use crate::error::{AexError, Result};
use crate::selection::SelectionLayout;

/// A reduction the server can compute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Function {
    Sum,
    Prod,
    Mean,
    Max,
    Min,
    Std,
    Var,
    All,
    Any,
    Argmax,
    Argmin,
    NanSum,
    NanMean,
    NanMax,
    NanMin,
}

impl Function {
    /// The numpy name, such as `nanmax`.
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "sum" => Function::Sum,
            "prod" => Function::Prod,
            "mean" => Function::Mean,
            "max" | "amax" => Function::Max,
            "min" | "amin" => Function::Min,
            "std" => Function::Std,
            "var" => Function::Var,
            "all" => Function::All,
            "any" => Function::Any,
            "argmax" => Function::Argmax,
            "argmin" => Function::Argmin,
            "nansum" => Function::NanSum,
            "nanmean" => Function::NanMean,
            "nanmax" => Function::NanMax,
            "nanmin" => Function::NanMin,
            _ => return None,
        })
    }

    /// The dtype numpy gives the result.
    pub fn result_dtype(self, input: DType) -> DType {
        use DType::*;
        let integral = matches!(
            input,
            Bool | Int8 | Int16 | Int32 | Int64 | Uint8 | Uint16 | Uint32 | Uint64
        );
        match self {
            Function::Sum | Function::Prod | Function::NanSum => match input {
                Bool | Int8 | Int16 | Int32 | Int64 => Int64,
                Uint8 | Uint16 | Uint32 | Uint64 => Uint64,
                other => other,
            },
            Function::Mean | Function::NanMean if integral => Float64,
            Function::Mean | Function::NanMean => input,
            Function::Std | Function::Var => match input {
                Float16 | Float32 | Float64 => input,
                Complex64 => Float32,
                _ => Float64,
            },
            Function::Max | Function::Min | Function::NanMax | Function::NanMin => input,
            Function::All | Function::Any => Bool,
            Function::Argmax | Function::Argmin => Int64,
        }
    }

    /// Whether an empty reduction has no answer.
    fn needs_an_element(self) -> bool {
        matches!(
            self,
            Function::Max
                | Function::Min
                | Function::NanMax
                | Function::NanMin
                | Function::Argmax
                | Function::Argmin
        )
    }
}

/// numpy's `axis` argument.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum Axis {
    /// `axis=None`.
    #[default]
    All,
    /// `axis=1`.
    One(i64),
    /// `axis=(0, 2)`.
    Many(Vec<i64>),
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ReduceArgs {
    pub axis: Axis,
    pub keepdims: bool,
    /// For `std` and `var`.
    pub ddof: i64,
}

/// A reduction's result, as C-order little-endian bytes.
#[derive(Debug, Clone, PartialEq)]
pub struct Reduced {
    pub dtype: DType,
    pub shape: Vec<u64>,
    pub data: Vec<u8>,
}

/// How much of the selection is read at once.
const BLOCK_BYTES: usize = 4 << 20;

/// Reduce the selection `layout` of `dataset`, refusing a result over
/// `max_bytes`.
pub fn reduce(
    dataset: &dyn ArrayDataset,
    layout: &SelectionLayout,
    function: Function,
    args: &ReduceArgs,
    max_bytes: u64,
) -> Result<Reduced> {
    let shape = &layout.out_shape;
    let reduced = reduced_axes(shape.len(), &args.axis, function)?;

    let mut out_shape = Vec::new();
    let mut stride = vec![0u64; shape.len()];
    let mut cells = 1u64;
    for d in (0..shape.len()).rev() {
        if !reduced[d] {
            stride[d] = cells;
            cells *= shape[d];
        }
    }
    for (d, &len) in shape.iter().enumerate() {
        if !reduced[d] {
            out_shape.push(len);
        } else if args.keepdims {
            out_shape.push(1);
        }
    }
    let per_cell: u64 = (0..shape.len())
        .filter(|&d| reduced[d])
        .map(|d| shape[d])
        .product();

    let dtype = function.result_dtype(layout.dtype);
    let bytes = cells.saturating_mul(dtype.itemsize());
    if bytes > max_bytes {
        return Err(AexError::BadFunction(format!(
            "the result would be {bytes} bytes, over this server's limit of {max_bytes}"
        )));
    }
    if per_cell == 0 && cells > 0 && function.needs_an_element() {
        return Err(AexError::BadFunction(format!(
            "zero-size array to reduction operation {function:?} which has no identity"
        )));
    }

    let walk = Walk {
        shape: shape.clone(),
        stride,
        // argmax reduces over at most one axis, and reports the position along
        // it; over all of them, the position in the flattened selection.
        pos_axis: match args.axis {
            Axis::One(_) => reduced.iter().position(|&r| r),
            _ => None,
        },
        cells: cells as usize,
        ddof: args.ddof,
        out: dtype,
    };

    macro_rules! by_dtype {
        ($($dtype:ident => $ty:ty),* $(,)?) => {
            match layout.dtype {
                $(DType::$dtype => run::<$ty>(dataset, layout, function, &walk),)*
            }
        };
    }
    let data = by_dtype!(
        Bool => bool, Int8 => i8, Int16 => i16, Int32 => i32, Int64 => i64,
        Uint8 => u8, Uint16 => u16, Uint32 => u32, Uint64 => u64,
        Float16 => f16, Float32 => f32, Float64 => f64,
        Complex64 => C32, Complex128 => C64,
    )?;
    Ok(Reduced {
        dtype,
        shape: out_shape,
        data,
    })
}

/// Which axes `axis` reduces, checked as numpy checks them.
fn reduced_axes(ndim: usize, axis: &Axis, function: Function) -> Result<Vec<bool>> {
    let axes = match axis {
        Axis::All => return Ok(vec![true; ndim]),
        Axis::One(a) => std::slice::from_ref(a),
        Axis::Many(_) if matches!(function, Function::Argmax | Function::Argmin) => {
            return Err(AexError::BadFunction(
                "argmax and argmin take a single axis, not a tuple".to_string(),
            ))
        }
        Axis::Many(axes) => axes,
    };
    let mut reduced = vec![false; ndim];
    for &a in axes {
        let d = if a < 0 { a + ndim as i64 } else { a };
        if d < 0 || d >= ndim as i64 {
            return Err(AexError::BadFunction(format!(
                "axis {a} is out of bounds for array of dimension {ndim}"
            )));
        }
        if std::mem::replace(&mut reduced[d as usize], true) {
            return Err(AexError::BadFunction(format!("repeated axis {a}")));
        }
    }
    Ok(reduced)
}

/// How elements map to output cells.
struct Walk {
    shape: Vec<u64>,
    /// Per axis, how far a step moves the output cell; 0 on a reduced axis.
    stride: Vec<u64>,
    pos_axis: Option<usize>,
    cells: usize,
    ddof: i64,
    out: DType,
}

fn run<T: Elem>(
    dataset: &dyn ArrayDataset,
    layout: &SelectionLayout,
    function: Function,
    walk: &Walk,
) -> Result<Vec<u8>> {
    let cells = walk.cells;
    match function {
        Function::Sum | Function::Prod | Function::NanSum => {
            let acc = Sum::<T>::new(cells, function);
            Ok(fold(dataset, layout, walk, acc)?.finish(walk.out))
        }
        Function::Mean | Function::NanMean => {
            let acc = Sum::<T>::new(cells, function);
            Ok(fold(dataset, layout, walk, acc)?.finish_mean(walk.out))
        }
        Function::Var | Function::Std => {
            let acc = Var::new(cells);
            Ok(fold::<T, _>(dataset, layout, walk, acc)?.finish(walk, function == Function::Std))
        }
        Function::Max | Function::Min | Function::NanMax | Function::NanMin => {
            let acc = Extreme::<T>::new(cells, function);
            Ok(fold(dataset, layout, walk, acc)?.finish())
        }
        Function::Argmax | Function::Argmin => {
            let acc = Arg::<T>::new(cells, function == Function::Argmax);
            Ok(fold(dataset, layout, walk, acc)?.finish())
        }
        Function::All | Function::Any => {
            let acc = Logic::new(cells, function == Function::All);
            let Logic(done, _) = fold::<T, _>(dataset, layout, walk, acc)?;
            Ok(done.into_iter().map(u8::from).collect())
        }
    }
}

/// Feed every element of the selection to `acc`.
fn fold<T: Elem, A: Acc<T>>(
    dataset: &dyn ArrayDataset,
    layout: &SelectionLayout,
    walk: &Walk,
    mut acc: A,
) -> Result<A> {
    // ponytail: one thread per reduction; split the stream across threads if
    // server-side reductions turn out to be the bottleneck.
    let block = BLOCK_BYTES / T::SIZE * T::SIZE;
    let mut buf = vec![0u8; block.min(layout.total_bytes as usize)];
    let ndim = walk.shape.len();
    let mut idx = vec![0u64; ndim];
    let (mut cell, mut flat) = (0u64, 0u64);
    let mut offset = 0u64;
    while offset < layout.total_bytes {
        let len = (layout.total_bytes - offset).min(block as u64) as usize;
        dataset.read_range(layout, offset, &mut buf[..len])?;
        for bytes in buf[..len].chunks_exact(T::SIZE) {
            let pos = walk.pos_axis.map_or(flat, |a| idx[a]);
            acc.push(cell as usize, pos, T::read(bytes));
            flat += 1;
            // Advance the index like an odometer, carrying the cell with it.
            for d in (0..ndim).rev() {
                idx[d] += 1;
                cell += walk.stride[d];
                if idx[d] < walk.shape[d] {
                    break;
                }
                cell -= walk.stride[d] * walk.shape[d];
                idx[d] = 0;
            }
        }
        offset += len as u64;
    }
    Ok(acc)
}

trait Acc<T> {
    fn push(&mut self, cell: usize, pos: u64, x: T);
}

/// Sums, products and means. Integer sums wrap as numpy's do, and two's
/// complement makes signed and unsigned the same bits; means add floats, as
/// numpy's do, so that they cannot wrap.
struct Sum<T> {
    ints: Vec<u64>,
    floats: Vec<(f64, f64)>,
    counts: Vec<u64>,
    float: bool,
    prod: bool,
    skip_nan: bool,
    _elem: std::marker::PhantomData<T>,
}

impl<T: Elem> Sum<T> {
    fn new(cells: usize, function: Function) -> Self {
        let prod = function == Function::Prod;
        let float = T::FLOAT || matches!(function, Function::Mean | Function::NanMean);
        let one = u64::from(prod);
        Sum {
            ints: vec![one; if float { 0 } else { cells }],
            floats: vec![(one as f64, 0.0); if float { cells } else { 0 }],
            counts: vec![0; cells],
            float,
            prod,
            skip_nan: matches!(function, Function::NanSum | Function::NanMean),
            _elem: std::marker::PhantomData,
        }
    }

    fn finish(self, out: DType) -> Vec<u8> {
        let mut data = Vec::new();
        for (re, im) in &self.floats {
            write_float(&mut data, out, *re, *im);
        }
        for v in &self.ints {
            data.extend_from_slice(&v.to_le_bytes());
        }
        data
    }

    fn finish_mean(self, out: DType) -> Vec<u8> {
        let mut data = Vec::new();
        for (&(re, im), &n) in self.floats.iter().zip(&self.counts) {
            write_float(&mut data, out, re / n as f64, im / n as f64);
        }
        data
    }
}

impl<T: Elem> Acc<T> for Sum<T> {
    fn push(&mut self, cell: usize, _pos: u64, x: T) {
        if self.float {
            if self.skip_nan && x.is_nan() {
                return;
            }
            let (re, im) = x.parts();
            let acc = &mut self.floats[cell];
            *acc = if self.prod {
                (acc.0 * re - acc.1 * im, acc.0 * im + acc.1 * re)
            } else {
                (acc.0 + re, acc.1 + im)
            };
        } else {
            let acc = &mut self.ints[cell];
            let v = x.int_bits();
            *acc = if self.prod {
                acc.wrapping_mul(v)
            } else {
                acc.wrapping_add(v)
            };
        }
        self.counts[cell] += 1;
    }
}

/// Variance by Welford's update, which stays accurate where the variance is
/// small next to the mean.
struct Var(Vec<(u64, f64, f64, f64)>);

impl Var {
    fn new(cells: usize) -> Self {
        Var(vec![(0, 0.0, 0.0, 0.0); cells])
    }

    fn finish(self, walk: &Walk, sqrt: bool) -> Vec<u8> {
        let mut data = Vec::new();
        for (n, _, _, m2) in self.0 {
            let dof = (n as i64 - walk.ddof).max(0) as f64;
            let var = m2 / dof;
            let v = if sqrt { var.sqrt() } else { var };
            write_float(&mut data, walk.out, v, 0.0);
        }
        data
    }
}

impl<T: Elem> Acc<T> for Var {
    fn push(&mut self, cell: usize, _pos: u64, x: T) {
        let (re, im) = x.parts();
        let (n, mr, mi, m2) = &mut self.0[cell];
        *n += 1;
        let (dr, di) = (re - *mr, im - *mi);
        *mr += dr / *n as f64;
        *mi += di / *n as f64;
        *m2 += dr * (re - *mr) + di * (im - *mi);
    }
}

/// max, min and their NaN-ignoring forms, with numpy's `maximum` and `fmax`.
struct Extreme<T> {
    best: Vec<Option<T>>,
    max: bool,
    skip_nan: bool,
}

impl<T: Elem> Extreme<T> {
    fn new(cells: usize, function: Function) -> Self {
        Extreme {
            best: vec![None; cells],
            max: matches!(function, Function::Max | Function::NanMax),
            skip_nan: matches!(function, Function::NanMax | Function::NanMin),
        }
    }

    fn finish(self) -> Vec<u8> {
        let mut data = Vec::new();
        for best in self.best {
            best.expect("every cell has an element").write(&mut data);
        }
        data
    }
}

impl<T: Elem> Acc<T> for Extreme<T> {
    fn push(&mut self, cell: usize, _pos: u64, x: T) {
        let slot = &mut self.best[cell];
        let Some(acc) = *slot else {
            *slot = Some(x);
            return;
        };
        let keep = if self.max { acc.ge(x) } else { acc.le(x) };
        let keep = if self.skip_nan {
            x.is_nan() || keep
        } else {
            acc.is_nan() || keep
        };
        if !keep {
            *slot = Some(x);
        }
    }
}

/// argmax and argmin: the first extreme, or the first NaN.
struct Arg<T> {
    best: Vec<Option<(T, u64, bool)>>,
    max: bool,
}

impl<T: Elem> Arg<T> {
    fn new(cells: usize, max: bool) -> Self {
        Arg {
            best: vec![None; cells],
            max,
        }
    }

    fn finish(self) -> Vec<u8> {
        let mut data = Vec::new();
        for best in self.best {
            let (_, pos, _) = best.expect("every cell has an element");
            data.extend_from_slice(&(pos as i64).to_le_bytes());
        }
        data
    }
}

impl<T: Elem> Acc<T> for Arg<T> {
    fn push(&mut self, cell: usize, pos: u64, x: T) {
        match &mut self.best[cell] {
            slot @ None => *slot = Some((x, pos, x.is_nan())),
            Some((_, _, true)) => {}
            Some((best, at, done)) => {
                let better = if self.max {
                    x.arg_gt(*best)
                } else {
                    x.arg_lt(*best)
                };
                if better {
                    (*best, *at, *done) = (x, pos, x.is_nan());
                }
            }
        }
    }
}

struct Logic(Vec<bool>, bool);

impl Logic {
    fn new(cells: usize, all: bool) -> Self {
        Logic(vec![all; cells], all)
    }
}

impl<T: Elem> Acc<T> for Logic {
    fn push(&mut self, cell: usize, _pos: u64, x: T) {
        if self.1 {
            self.0[cell] &= x.truthy();
        } else {
            self.0[cell] |= x.truthy();
        }
    }
}

fn write_float(data: &mut Vec<u8>, dtype: DType, re: f64, im: f64) {
    match dtype {
        DType::Float16 => data.extend_from_slice(&f16::from_f64(re).to_le_bytes()),
        DType::Float32 => data.extend_from_slice(&(re as f32).to_le_bytes()),
        DType::Float64 => data.extend_from_slice(&re.to_le_bytes()),
        DType::Complex64 => {
            data.extend_from_slice(&(re as f32).to_le_bytes());
            data.extend_from_slice(&(im as f32).to_le_bytes());
        }
        DType::Complex128 => {
            data.extend_from_slice(&re.to_le_bytes());
            data.extend_from_slice(&im.to_le_bytes());
        }
        other => unreachable!("{other} is not a floating result"),
    }
}

/// A complex element; numpy's layout, real part first.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Complex<F>(F, F);
type C32 = Complex<f32>;
type C64 = Complex<f64>;

/// An element as the reductions see it.
trait Elem: Copy {
    const SIZE: usize;
    const FLOAT: bool;
    fn read(bytes: &[u8]) -> Self;
    fn write(self, data: &mut Vec<u8>);
    /// Real and imaginary parts.
    fn parts(self) -> (f64, f64);
    /// Sign-extended bits, for the integer types.
    fn int_bits(self) -> u64;
    fn is_nan(self) -> bool;
    fn truthy(self) -> bool;
    /// numpy's `>=` as `maximum` uses it; lexicographic for complex.
    fn ge(self, other: Self) -> bool;
    fn le(self, other: Self) -> bool;
    /// Whether argmax should move to `self` from `best`.
    fn arg_gt(self, best: Self) -> bool;
    fn arg_lt(self, best: Self) -> bool;
}

macro_rules! int_elem {
    ($($ty:ty),* $(,)?) => {$(
        impl Elem for $ty {
            const SIZE: usize = std::mem::size_of::<$ty>();
            const FLOAT: bool = false;
            fn read(bytes: &[u8]) -> Self {
                <$ty>::from_le_bytes(bytes.try_into().unwrap())
            }
            fn write(self, data: &mut Vec<u8>) {
                data.extend_from_slice(&self.to_le_bytes());
            }
            fn parts(self) -> (f64, f64) {
                (self as f64, 0.0)
            }
            fn int_bits(self) -> u64 {
                self as i64 as u64
            }
            fn is_nan(self) -> bool {
                false
            }
            fn truthy(self) -> bool {
                self != 0
            }
            fn ge(self, other: Self) -> bool {
                self >= other
            }
            fn le(self, other: Self) -> bool {
                self <= other
            }
            fn arg_gt(self, best: Self) -> bool {
                self > best
            }
            fn arg_lt(self, best: Self) -> bool {
                self < best
            }
        }
    )*};
}

// The unsigned ones go through i64 too: `as i64 as u64` is the identity for
// them up to u64, whose bits it keeps.
int_elem!(i8, i16, i32, i64, u8, u16, u32, u64);

impl Elem for bool {
    const SIZE: usize = 1;
    const FLOAT: bool = false;
    fn read(bytes: &[u8]) -> Self {
        bytes[0] != 0
    }
    fn write(self, data: &mut Vec<u8>) {
        data.push(u8::from(self));
    }
    fn parts(self) -> (f64, f64) {
        (f64::from(u8::from(self)), 0.0)
    }
    fn int_bits(self) -> u64 {
        u64::from(self)
    }
    fn is_nan(self) -> bool {
        false
    }
    fn truthy(self) -> bool {
        self
    }
    fn ge(self, other: Self) -> bool {
        self >= other
    }
    fn le(self, other: Self) -> bool {
        self <= other
    }
    fn arg_gt(self, best: Self) -> bool {
        self & !best
    }
    fn arg_lt(self, best: Self) -> bool {
        !self & best
    }
}

macro_rules! float_elem {
    ($($ty:ty),*) => {$(
        impl Elem for $ty {
            const SIZE: usize = std::mem::size_of::<$ty>();
            const FLOAT: bool = true;
            fn read(bytes: &[u8]) -> Self {
                <$ty>::from_le_bytes(bytes.try_into().unwrap())
            }
            fn write(self, data: &mut Vec<u8>) {
                data.extend_from_slice(&self.to_le_bytes());
            }
            fn parts(self) -> (f64, f64) {
                (f64::from(self), 0.0)
            }
            fn int_bits(self) -> u64 {
                unreachable!("floats are summed as floats")
            }
            fn is_nan(self) -> bool {
                <$ty>::is_nan(self)
            }
            fn truthy(self) -> bool {
                self != <$ty>::from(0u8)
            }
            fn ge(self, other: Self) -> bool {
                self >= other
            }
            fn le(self, other: Self) -> bool {
                self <= other
            }
            // Written as numpy writes them, so that NaN counts as an extreme.
            #[allow(clippy::neg_cmp_op_on_partial_ord)]
            fn arg_gt(self, best: Self) -> bool {
                !(self <= best)
            }
            #[allow(clippy::neg_cmp_op_on_partial_ord)]
            fn arg_lt(self, best: Self) -> bool {
                !(self >= best)
            }
        }
    )*};
}

float_elem!(f16, f32, f64);

macro_rules! complex_elem {
    ($($ty:ty => $f:ty),*) => {$(
        impl Elem for Complex<$f> {
            const SIZE: usize = std::mem::size_of::<$ty>();
            const FLOAT: bool = true;
            fn read(bytes: &[u8]) -> Self {
                let half = bytes.len() / 2;
                Complex(
                    <$f>::from_le_bytes(bytes[..half].try_into().unwrap()),
                    <$f>::from_le_bytes(bytes[half..].try_into().unwrap()),
                )
            }
            fn write(self, data: &mut Vec<u8>) {
                data.extend_from_slice(&self.0.to_le_bytes());
                data.extend_from_slice(&self.1.to_le_bytes());
            }
            fn parts(self) -> (f64, f64) {
                (f64::from(self.0), f64::from(self.1))
            }
            fn int_bits(self) -> u64 {
                unreachable!("complex values are summed as floats")
            }
            fn is_nan(self) -> bool {
                self.0.is_nan() || self.1.is_nan()
            }
            fn truthy(self) -> bool {
                self.0 != 0.0 || self.1 != 0.0
            }
            fn ge(self, o: Self) -> bool {
                (self.0 > o.0 && !self.1.is_nan() && !o.1.is_nan())
                    || (self.0 == o.0 && self.1 >= o.1)
            }
            fn le(self, o: Self) -> bool {
                (self.0 < o.0 && !self.1.is_nan() && !o.1.is_nan())
                    || (self.0 == o.0 && self.1 <= o.1)
            }
            fn arg_gt(self, b: Self) -> bool {
                self.0 > b.0 || (self.0 == b.0 && self.1 > b.1) || self.is_nan()
            }
            fn arg_lt(self, b: Self) -> bool {
                self.0 < b.0 || (self.0 == b.0 && self.1 < b.1) || self.is_nan()
            }
        }
    )*};
}

complex_elem!(C32 => f32, C64 => f64);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quality::QualitySpec;
    use crate::selection::Index;

    /// An array held in memory.
    struct Memory {
        dtype: DType,
        shape: Vec<u64>,
        bytes: Vec<u8>,
    }

    impl ArrayDataset for Memory {
        fn dtype(&self) -> DType {
            self.dtype
        }
        fn shape(&self) -> &[u64] {
            &self.shape
        }
        fn read_range(&self, layout: &SelectionLayout, offset: u64, dst: &mut [u8]) -> Result<()> {
            layout.read_with(offset, dst, |at, buf| {
                buf.copy_from_slice(&self.bytes[at as usize..at as usize + buf.len()]);
                Ok(())
            })
        }
    }

    fn f64s(shape: &[u64], values: &[f64]) -> Memory {
        Memory {
            dtype: DType::Float64,
            shape: shape.to_vec(),
            bytes: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        }
    }

    fn i32s(shape: &[u64], values: &[i32]) -> Memory {
        Memory {
            dtype: DType::Int32,
            shape: shape.to_vec(),
            bytes: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        }
    }

    fn apply(data: &Memory, indices: &[Index], name: &str, args: ReduceArgs) -> Result<Reduced> {
        let layout = data.layout(indices, &QualitySpec::exact())?;
        reduce(
            data,
            &layout,
            Function::from_name(name).unwrap(),
            &args,
            1 << 16,
        )
    }

    fn axis(axis: Axis) -> ReduceArgs {
        ReduceArgs {
            axis,
            ..ReduceArgs::default()
        }
    }

    fn as_f64(r: &Reduced) -> Vec<f64> {
        assert_eq!(r.dtype, DType::Float64);
        r.data
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    fn as_i64(r: &Reduced) -> Vec<i64> {
        assert_eq!(r.dtype, DType::Int64);
        r.data
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn sums_follow_the_axes() {
        // [[0, 1, 2], [3, 4, 5]]
        let data = i32s(&[2, 3], &[0, 1, 2, 3, 4, 5]);
        let all = apply(&data, &[], "sum", ReduceArgs::default()).unwrap();
        assert_eq!((as_i64(&all), all.shape.clone()), (vec![15], vec![]));

        let rows = apply(&data, &[], "sum", axis(Axis::One(-1))).unwrap();
        assert_eq!((as_i64(&rows), rows.shape.clone()), (vec![3, 12], vec![2]));

        let cols = apply(
            &data,
            &[],
            "sum",
            ReduceArgs {
                axis: Axis::One(0),
                keepdims: true,
                ddof: 0,
            },
        )
        .unwrap();
        assert_eq!(
            (as_i64(&cols), cols.shape.clone()),
            (vec![3, 5, 7], vec![1, 3])
        );

        let none = apply(&data, &[], "sum", axis(Axis::Many(vec![]))).unwrap();
        assert_eq!(as_i64(&none), vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn a_reduction_covers_only_the_selection() {
        let data = i32s(&[2, 3], &[0, 1, 2, 3, 4, 5]);
        let r = apply(
            &data,
            &[Index::Single(1), Index::range(0, 2)],
            "prod",
            ReduceArgs::default(),
        );
        assert_eq!(as_i64(&r.unwrap()), vec![12]);
    }

    #[test]
    fn integer_sums_wrap() {
        let data = Memory {
            dtype: DType::Uint64,
            shape: vec![2],
            bytes: [u64::MAX, 2].iter().flat_map(|v| v.to_le_bytes()).collect(),
        };
        let r = apply(&data, &[], "sum", ReduceArgs::default()).unwrap();
        assert_eq!(r.dtype, DType::Uint64);
        assert_eq!(r.data, 1u64.to_le_bytes());
    }

    #[test]
    fn integer_means_do_not_wrap() {
        let data = Memory {
            dtype: DType::Uint64,
            shape: vec![2],
            bytes: [u64::MAX, u64::MAX]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
        };
        let r = apply(&data, &[], "mean", ReduceArgs::default()).unwrap();
        assert_eq!(as_f64(&r), vec![u64::MAX as f64]);
    }

    #[test]
    fn nan_propagates_unless_asked_not_to() {
        let data = f64s(&[4], &[1.0, f64::NAN, 3.0, 2.0]);
        let get = |name| as_f64(&apply(&data, &[], name, ReduceArgs::default()).unwrap())[0];
        assert!(get("max").is_nan());
        assert!(get("sum").is_nan());
        assert_eq!(get("nanmax"), 3.0);
        assert_eq!(get("nanmin"), 1.0);
        assert_eq!(get("nansum"), 6.0);
        assert_eq!(get("nanmean"), 2.0);
        let arg = |name| as_i64(&apply(&data, &[], name, ReduceArgs::default()).unwrap())[0];
        assert_eq!(arg("argmax"), 1);
        assert_eq!(arg("argmin"), 1);

        let all_nan = f64s(&[2], &[f64::NAN, f64::NAN]);
        let r = apply(&all_nan, &[], "nanmax", ReduceArgs::default()).unwrap();
        assert!(as_f64(&r)[0].is_nan());
    }

    #[test]
    fn argmax_reports_the_position_along_its_axis() {
        let data = i32s(&[2, 3], &[5, 1, 5, 0, 7, 7]);
        let r = apply(&data, &[], "argmax", axis(Axis::One(1))).unwrap();
        assert_eq!(as_i64(&r), vec![0, 1]);
        let r = apply(&data, &[], "argmin", ReduceArgs::default()).unwrap();
        assert_eq!(as_i64(&r), vec![3]);
        assert!(apply(&data, &[], "argmax", axis(Axis::Many(vec![0]))).is_err());
    }

    #[test]
    fn variance_takes_ddof() {
        let data = f64s(&[4], &[1.0, 2.0, 3.0, 4.0]);
        let var = |ddof| {
            let args = ReduceArgs {
                ddof,
                ..ReduceArgs::default()
            };
            as_f64(&apply(&data, &[], "var", args).unwrap())[0]
        };
        assert!((var(0) - 1.25).abs() < 1e-12);
        assert!((var(1) - 5.0 / 3.0).abs() < 1e-12);
        assert!(var(4).is_infinite());
        let std = as_f64(&apply(&data, &[], "std", ReduceArgs::default()).unwrap())[0];
        assert!((std - 1.25f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn complex_values_order_lexicographically() {
        let values = [(1.0f64, 5.0f64), (2.0, -1.0), (2.0, 0.0)];
        let data = Memory {
            dtype: DType::Complex128,
            shape: vec![3],
            bytes: values
                .iter()
                .flat_map(|(re, im)| re.to_le_bytes().into_iter().chain(im.to_le_bytes()))
                .collect(),
        };
        let max = apply(&data, &[], "max", ReduceArgs::default()).unwrap();
        assert_eq!(max.dtype, DType::Complex128);
        assert_eq!(max.data[..8], 2.0f64.to_le_bytes());
        assert_eq!(max.data[8..], 0.0f64.to_le_bytes());
        let arg = apply(&data, &[], "argmin", ReduceArgs::default()).unwrap();
        assert_eq!(as_i64(&arg), vec![0]);
        let var = apply(&data, &[], "var", ReduceArgs::default()).unwrap();
        // |z - mean|^2 averaged, with mean = (5/3, 4/3).
        let expected = [(1.0, 5.0), (2.0, -1.0), (2.0, 0.0)]
            .iter()
            .map(|(r, i): &(f64, f64)| (r - 5.0 / 3.0).powi(2) + (i - 4.0 / 3.0).powi(2))
            .sum::<f64>()
            / 3.0;
        assert!((as_f64(&var)[0] - expected).abs() < 1e-12);
    }

    #[test]
    fn empty_reductions_answer_as_numpy_does() {
        let empty = i32s(&[0, 3], &[]);
        assert_eq!(
            as_i64(&apply(&empty, &[], "sum", ReduceArgs::default()).unwrap()),
            vec![0]
        );
        let all = apply(&empty, &[], "all", ReduceArgs::default()).unwrap();
        assert_eq!((all.dtype, all.data), (DType::Bool, vec![1]));
        assert!(apply(&empty, &[], "max", ReduceArgs::default()).is_err());
        assert!(apply(&empty, &[], "argmax", axis(Axis::One(0))).is_err());
        // No cells, so nothing is missing an element.
        let r = apply(&empty, &[], "max", axis(Axis::One(1))).unwrap();
        assert_eq!((r.shape, r.data.len()), (vec![0], 0));
        let mean = as_f64(&apply(&empty, &[], "mean", ReduceArgs::default()).unwrap());
        assert!(mean[0].is_nan());
    }

    #[test]
    fn bad_axes_and_large_results_are_refused() {
        let data = i32s(&[2, 3], &[0; 6]);
        for bad in [Axis::One(2), Axis::One(-3), Axis::Many(vec![0, -2])] {
            let err = apply(&data, &[], "sum", axis(bad)).unwrap_err();
            assert_eq!(err.class(), crate::ErrorClass::Request);
        }
        let layout = data.layout(&[], &QualitySpec::exact()).unwrap();
        let err = reduce(&data, &layout, Function::Sum, &axis(Axis::One(0)), 16).unwrap_err();
        assert!(err.to_string().contains("24 bytes"), "{err}");
    }

    #[test]
    fn blocks_do_not_disturb_the_walk() {
        // More than one block of i32, reduced along the last axis.
        let rows = 3u64;
        let cols = (BLOCK_BYTES as u64 / 4) / 2 + 7;
        let values: Vec<i32> = (0..rows * cols).map(|i| (i % 5) as i32).collect();
        let data = i32s(&[rows, cols], &values);
        let r = apply(&data, &[], "sum", axis(Axis::One(1))).unwrap();
        let expected: Vec<i64> = values
            .chunks(cols as usize)
            .map(|row| row.iter().map(|&v| i64::from(v)).sum())
            .collect();
        assert_eq!(as_i64(&r), expected);
    }

    #[test]
    fn every_dtype_reduces() {
        for dtype in crate::ALL_DTYPES {
            let data = Memory {
                dtype,
                shape: vec![2],
                bytes: vec![1; 2 * dtype.itemsize() as usize],
            };
            for name in ["sum", "mean", "max", "std", "all", "argmin", "nanmax"] {
                let r = apply(&data, &[], name, ReduceArgs::default()).unwrap();
                let function = Function::from_name(name).unwrap();
                assert_eq!(r.dtype, function.result_dtype(dtype));
                assert_eq!(r.data.len() as u64, r.dtype.itemsize(), "{name} of {dtype}");
            }
        }
    }
}
