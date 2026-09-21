//! A regular grid of chunks over a C-order array.
//!
//! HDF5 and Zarr both store edge chunks padded to the full chunk shape, so
//! turning a position in the array's byte stream into (which chunk, where in
//! it, how far the run carries) is the same work for both. Only where the
//! chunk's bytes come from differs, and that is the callback's business.
//!
//! Zarr's sharding needs the same walk twice over: a shard is a chunk of the
//! array, and an inner chunk is a chunk of the shard. Keeping the walk in one
//! place is what makes that a nested call rather than a third copy of the
//! index arithmetic.

use crate::error::Result;

/// Chunks of one shape laid over an array of another.
#[derive(Debug)]
pub(crate) struct ChunkGrid {
    shape: Vec<u64>,
    chunk_shape: Vec<u64>,
    /// Chunks along each axis.
    grid: Vec<u64>,
    itemsize: u64,
    /// Bytes of a whole chunk, decoded. Edge chunks are stored whole too.
    chunk_bytes: u64,
    /// Every axis after this one is covered by one chunk with no padding, so
    /// a run of elements carries on across them within a chunk.
    run_axis: usize,
}

// The HDF5 backend is optional, so a build without it has no caller for some
// of these. Sharding, which wants the rest, will close the gap.
#[cfg_attr(not(feature = "hdf5"), allow(dead_code))]
impl ChunkGrid {
    /// `None` if the ranks disagree, a chunk axis is zero, or a chunk is too
    /// big to address. The caller words the error: what is malformed in one
    /// format is unsupported in another.
    pub(crate) fn new(shape: &[u64], chunk_shape: &[u64], itemsize: u64) -> Option<Self> {
        if chunk_shape.len() != shape.len() || chunk_shape.contains(&0) {
            return None;
        }
        let grid: Vec<u64> = shape
            .iter()
            .zip(chunk_shape)
            .map(|(n, c)| n.div_ceil(*c))
            .collect();
        grid.iter().try_fold(1u64, |acc, &n| acc.checked_mul(n))?;
        let chunk_bytes = chunk_shape
            .iter()
            .try_fold(itemsize, |acc, &n| acc.checked_mul(n))
            .filter(|&n| usize::try_from(n).is_ok())?;
        // A scalar has no axis to run along; `walk` never reads this.
        let mut run_axis = shape.len().saturating_sub(1);
        while run_axis > 0 && chunk_shape[run_axis] == shape[run_axis] {
            run_axis -= 1;
        }
        Some(ChunkGrid {
            shape: shape.to_vec(),
            chunk_shape: chunk_shape.to_vec(),
            grid,
            itemsize,
            chunk_bytes,
            run_axis,
        })
    }

    /// How many chunks the grid holds. Zero if any axis is empty.
    pub(crate) fn chunks(&self) -> u64 {
        self.grid.iter().product()
    }

    pub(crate) fn chunk_bytes(&self) -> u64 {
        self.chunk_bytes
    }

    pub(crate) fn chunk_shape(&self) -> &[u64] {
        &self.chunk_shape
    }

    pub(crate) fn itemsize(&self) -> u64 {
        self.itemsize
    }

    /// The C-order number of the chunk at `coords`, if it lies inside.
    pub(crate) fn chunk_of(&self, coords: impl Iterator<Item = u64>) -> Option<usize> {
        linear(coords, &self.grid)
    }

    /// The per-axis coordinates of chunk `n`, for building a chunk key.
    pub(crate) fn coords_of(&self, n: u64, out: &mut Vec<u64>) {
        out.clear();
        out.resize(self.grid.len(), 0);
        let mut rest = n;
        for (axis, &d) in self.grid.iter().enumerate().rev() {
            out[axis] = rest % d;
            rest /= d;
        }
    }

    /// Walk `[at, at + dst.len())` of the array's C-order bytes, calling `read`
    /// once per run that stays inside one chunk.
    ///
    /// `read` is given the chunk's number, the byte offset within it, and the
    /// piece of `dst` to fill. The offset within a whole element, for a run
    /// that starts mid-element, is `offset % itemsize`.
    pub(crate) fn walk(
        &self,
        mut at: u64,
        mut dst: &mut [u8],
        mut read: impl FnMut(usize, u64, &mut [u8]) -> Result<()>,
    ) -> Result<()> {
        // A scalar array is one chunk holding one element, so the offset into
        // the array is already the offset into the chunk.
        if self.shape.is_empty() {
            return read(0, at, dst);
        }
        let ndim = self.shape.len();
        let mut pos = vec![0u64; ndim];
        while !dst.is_empty() {
            let element = at / self.itemsize;
            let skip = at % self.itemsize;
            let mut rest = element;
            for axis in (0..ndim).rev() {
                pos[axis] = rest % self.shape[axis];
                rest /= self.shape[axis];
            }

            let chunk = linear(
                pos.iter().zip(&self.chunk_shape).map(|(p, c)| p / c),
                &self.grid,
            )
            .expect("an in-range element is in some chunk");
            let within = linear_unchecked(
                pos.iter().zip(&self.chunk_shape).map(|(p, c)| p % c),
                &self.chunk_shape,
            );

            // Elements from here to the end of this chunk along the run axis,
            // less those of the trailing axes already behind us.
            let j = self.run_axis;
            let trailing: u64 = self.shape[j + 1..].iter().product();
            let behind = linear_unchecked(pos[j + 1..].iter().copied(), &self.shape[j + 1..]);
            let along =
                (self.chunk_shape[j] - pos[j] % self.chunk_shape[j]).min(self.shape[j] - pos[j]);
            let run = along * trailing - behind;

            let start = within * self.itemsize + skip;
            let len = (run * self.itemsize - skip).min(dst.len() as u64) as usize;
            let (out, tail) = dst.split_at_mut(len);
            read(chunk, start, out)?;
            at += len as u64;
            dst = tail;
        }
        Ok(())
    }
}

/// The C-order number of `coords` in a grid of `dims`, if it lies inside.
fn linear(coords: impl Iterator<Item = u64>, dims: &[u64]) -> Option<usize> {
    let mut n = 0u64;
    for (c, d) in coords.zip(dims) {
        if c >= *d {
            return None;
        }
        n = n * d + c;
    }
    Some(n as usize)
}

/// [`linear`] for coordinates known to be inside.
fn linear_unchecked(coords: impl Iterator<Item = u64>, dims: &[u64]) -> u64 {
    coords.zip(dims).fold(0, |n, (c, d)| n * d + c)
}

/// Fill `dst` with the bytes at `[at, at + dst.len())` of an array made of
/// nothing but `pattern`.
pub(crate) fn fill_from(pattern: &[u8], at: u64, dst: &mut [u8]) {
    let n = pattern.len();
    let start = (at % n as u64) as usize;
    for (i, b) in dst.iter_mut().enumerate() {
        *b = pattern[(start + i) % n];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Read the whole grid one piece at a time, serving each chunk from a
    /// buffer holding the chunk's padded bytes.
    fn read_all(grid: &ChunkGrid, chunks: &[Vec<u8>], total: u64, piece: usize) -> Vec<u8> {
        let mut out = vec![0u8; total as usize];
        let mut at = 0u64;
        while (at as usize) < out.len() {
            let len = piece.min(out.len() - at as usize);
            grid.walk(
                at,
                &mut out[at as usize..at as usize + len],
                |c, start, dst| {
                    let start = start as usize;
                    dst.copy_from_slice(&chunks[c][start..start + dst.len()]);
                    Ok(())
                },
            )
            .expect("walk");
            at += len as u64;
        }
        out
    }

    /// The padded chunks of `values` laid out as `shape` in chunks of
    /// `chunk_shape`, and the array's own bytes.
    fn lay_out(shape: &[u64], chunk_shape: &[u64], values: &[u32]) -> (Vec<Vec<u8>>, Vec<u8>) {
        let grid: Vec<u64> = shape
            .iter()
            .zip(chunk_shape)
            .map(|(n, c)| n.div_ceil(*c))
            .collect();
        let per_chunk: u64 = chunk_shape.iter().product();
        let chunk_count: u64 = grid.iter().product();
        let mut chunks = vec![vec![0u8; per_chunk as usize * 4]; chunk_count as usize];
        let ndim = shape.len();
        for (i, v) in values.iter().enumerate() {
            let mut rest = i as u64;
            let mut pos = vec![0u64; ndim];
            for axis in (0..ndim).rev() {
                pos[axis] = rest % shape[axis];
                rest /= shape[axis];
            }
            let c = linear(pos.iter().zip(chunk_shape).map(|(p, s)| p / s), &grid).unwrap();
            let within =
                linear_unchecked(pos.iter().zip(chunk_shape).map(|(p, s)| p % s), chunk_shape);
            let at = within as usize * 4;
            chunks[c][at..at + 4].copy_from_slice(&v.to_le_bytes());
        }
        let flat = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        (chunks, flat)
    }

    #[test]
    fn a_walk_reassembles_the_array_however_it_is_split() {
        // Shapes that are not a multiple of the chunk shape, so edge chunks
        // carry padding the walk must step over.
        let cases: [(&[u64], &[u64]); 5] = [
            (&[8], &[3]),
            (&[5, 7], &[2, 7]),
            (&[5, 7], &[2, 3]),
            (&[4, 3, 5], &[2, 3, 5]),
            (&[4, 3, 5], &[3, 2, 2]),
        ];
        for (shape, chunk_shape) in cases {
            let n: u64 = shape.iter().product();
            let values: Vec<u32> = (0..n as u32).collect();
            let (chunks, flat) = lay_out(shape, chunk_shape, &values);
            let grid = ChunkGrid::new(shape, chunk_shape, 4).expect("grid");
            for piece in [1, 3, 7, 64, flat.len()] {
                assert_eq!(
                    read_all(&grid, &chunks, flat.len() as u64, piece),
                    flat,
                    "{shape:?} in {chunk_shape:?}, {piece} bytes at a time"
                );
            }
        }
    }

    #[test]
    fn a_scalar_is_one_chunk() {
        let grid = ChunkGrid::new(&[], &[], 4).expect("grid");
        assert_eq!(grid.chunks(), 1);
        assert_eq!(grid.chunk_bytes(), 4);

        let chunk = 0x11223344u32.to_le_bytes();
        let mut out = [0u8; 4];
        grid.walk(0, &mut out, |c, start, dst| {
            assert_eq!(c, 0);
            dst.copy_from_slice(&chunk[start as usize..start as usize + dst.len()]);
            Ok(())
        })
        .expect("walk");
        assert_eq!(out, chunk);

        // A piece starting mid-element still lands at the right offset.
        let mut tail = [0u8; 2];
        grid.walk(2, &mut tail, |_, start, dst| {
            assert_eq!(start, 2);
            dst.copy_from_slice(&chunk[start as usize..start as usize + dst.len()]);
            Ok(())
        })
        .expect("walk");
        assert_eq!(tail, chunk[2..]);
    }

    #[test]
    fn an_empty_axis_has_no_chunks_to_walk() {
        let grid = ChunkGrid::new(&[0, 4], &[2, 2], 4).expect("grid");
        assert_eq!(grid.chunks(), 0);
        // Nothing is asked for, so nothing is read.
        grid.walk(0, &mut [], |_, _, _| panic!("no chunk holds anything"))
            .expect("walk");
    }

    #[test]
    fn a_grid_that_does_not_fit_is_refused() {
        assert!(ChunkGrid::new(&[4, 4], &[4], 4).is_none(), "rank mismatch");
        assert!(ChunkGrid::new(&[4, 4], &[4, 0], 4).is_none(), "zero axis");
        assert!(
            ChunkGrid::new(&[u64::MAX, u64::MAX], &[1, 1], 1).is_none(),
            "more chunks than fit a u64"
        );
        assert!(
            ChunkGrid::new(&[u64::MAX], &[u64::MAX], 8).is_none(),
            "chunk too big to address"
        );
    }

    #[test]
    fn chunk_numbers_and_coordinates_agree() {
        let grid = ChunkGrid::new(&[5, 7], &[2, 3], 4).expect("grid");
        assert_eq!(grid.chunks(), 3 * 3);
        let mut coords = Vec::new();
        for n in 0..grid.chunks() {
            grid.coords_of(n, &mut coords);
            assert_eq!(grid.chunk_of(coords.iter().copied()), Some(n as usize));
        }
        assert_eq!(grid.chunk_of([3, 0].into_iter()), None, "outside the grid");
    }

    #[test]
    fn a_fill_pattern_repeats_from_any_offset() {
        let pattern = [1u8, 2, 3, 4];
        let mut out = [0u8; 6];
        fill_from(&pattern, 2, &mut out);
        assert_eq!(out, [3, 4, 1, 2, 3, 4]);
        // The offset is taken modulo the pattern, so a whole element in is the
        // same as the start of it.
        fill_from(&pattern, 4, &mut out);
        assert_eq!(out, [1, 2, 3, 4, 1, 2]);
    }
}
