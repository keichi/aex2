//! Element types and their numpy `descr` equivalents.
//!
//! Discriminants match the proto `DataType` enum so that converting to and from
//! generated protobuf code is just a tag swap.

use npyz::{Endianness, TypeChar, TypeStr};

use crate::error::{AexError, Result};

/// The 14 element types AEX can transfer. Values match `DataType` on the wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum DType {
    Int8 = 0,
    Int16 = 1,
    Int32 = 2,
    Int64 = 3,
    Uint8 = 4,
    Uint16 = 5,
    Uint32 = 6,
    Uint64 = 7,
    Float16 = 8,
    Float32 = 9,
    Float64 = 10,
    Complex64 = 11,
    Complex128 = 12,
    Bool = 13,
}

/// Every element type, for sweeps and enumeration.
pub const ALL_DTYPES: [DType; 14] = [
    DType::Int8,
    DType::Int16,
    DType::Int32,
    DType::Int64,
    DType::Uint8,
    DType::Uint16,
    DType::Uint32,
    DType::Uint64,
    DType::Float16,
    DType::Float32,
    DType::Float64,
    DType::Complex64,
    DType::Complex128,
    DType::Bool,
];

impl DType {
    /// Size of one element in bytes.
    pub const fn itemsize(self) -> u64 {
        match self {
            DType::Int8 | DType::Uint8 | DType::Bool => 1,
            DType::Int16 | DType::Uint16 | DType::Float16 => 2,
            DType::Int32 | DType::Uint32 | DType::Float32 => 4,
            DType::Int64 | DType::Uint64 | DType::Float64 | DType::Complex64 => 8,
            DType::Complex128 => 16,
        }
    }

    /// The numpy `descr` string, little-endian.
    ///
    /// Single-byte types use `|`, matching what numpy writes into a `.npy`.
    pub const fn descr(self) -> &'static str {
        match self {
            DType::Bool => "|b1",
            DType::Int8 => "|i1",
            DType::Int16 => "<i2",
            DType::Int32 => "<i4",
            DType::Int64 => "<i8",
            DType::Uint8 => "|u1",
            DType::Uint16 => "<u2",
            DType::Uint32 => "<u4",
            DType::Uint64 => "<u8",
            DType::Float16 => "<f2",
            DType::Float32 => "<f4",
            DType::Float64 => "<f8",
            DType::Complex64 => "<c8",
            DType::Complex128 => "<c16",
        }
    }

    /// To the wire `DataType` value.
    pub const fn as_i32(self) -> i32 {
        self as i32
    }

    /// From the wire `DataType` value. Unknown values are an error.
    pub fn from_i32(v: i32) -> Result<Self> {
        ALL_DTYPES
            .iter()
            .copied()
            .find(|d| d.as_i32() == v)
            .ok_or_else(|| AexError::UnsupportedDType(format!("unknown DataType value {v}")))
    }

    /// From a numpy `descr` string.
    ///
    /// Structured dtypes never reach here: callers check for
    /// [`npyz::DType::Plain`] first. Only the `<`, `>` and `|` prefixes are
    /// accepted, since numpy normalises `=` away when writing a `.npy`.
    pub fn from_descr(descr: &str) -> Result<Self> {
        let ts: TypeStr = descr.parse().map_err(|e| {
            AexError::UnsupportedDType(format!("cannot parse descr {descr:?}: {e}"))
        })?;
        Self::from_type_str(&ts)
    }

    /// From an npyz type string.
    ///
    /// Types AEX does not handle (`m`, `M`, `S`, `U`, `O`, `V`) are rejected, as
    /// are multi-byte big-endian ones: every target is little-endian, so byte
    /// swapping is not worth its cost.
    pub fn from_type_str(ts: &TypeStr) -> Result<Self> {
        let size = ts.num_bytes().ok_or_else(|| {
            AexError::UnsupportedDType(format!("dtype {ts} has no fixed item size"))
        })?;

        if size > 1 && ts.endianness() == Endianness::Big {
            return Err(AexError::UnsupportedDType(format!(
                "big-endian dtype {ts} is not supported"
            )));
        }

        let unsupported = || AexError::UnsupportedDType(format!("dtype {ts} is not supported"));

        let dtype = match (ts.type_char(), size) {
            (TypeChar::Bool, 1) => DType::Bool,
            (TypeChar::Int, 1) => DType::Int8,
            (TypeChar::Int, 2) => DType::Int16,
            (TypeChar::Int, 4) => DType::Int32,
            (TypeChar::Int, 8) => DType::Int64,
            (TypeChar::Uint, 1) => DType::Uint8,
            (TypeChar::Uint, 2) => DType::Uint16,
            (TypeChar::Uint, 4) => DType::Uint32,
            (TypeChar::Uint, 8) => DType::Uint64,
            (TypeChar::Float, 2) => DType::Float16,
            (TypeChar::Float, 4) => DType::Float32,
            (TypeChar::Float, 8) => DType::Float64,
            (TypeChar::Complex, 8) => DType::Complex64,
            (TypeChar::Complex, 16) => DType::Complex128,
            _ => return Err(unsupported()),
        };
        Ok(dtype)
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.descr())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descr_roundtrips_for_every_dtype() {
        for dtype in ALL_DTYPES {
            let parsed = DType::from_descr(dtype.descr()).expect("descr must parse");
            assert_eq!(parsed, dtype, "descr {} did not round-trip", dtype.descr());
        }
    }

    #[test]
    fn descr_itemsize_matches_the_digits_in_the_type_string() {
        for dtype in ALL_DTYPES {
            let ts: TypeStr = dtype.descr().parse().unwrap();
            assert_eq!(ts.num_bytes(), Some(dtype.itemsize() as usize));
        }
    }

    #[test]
    fn proto_discriminants_roundtrip() {
        for dtype in ALL_DTYPES {
            assert_eq!(DType::from_i32(dtype.as_i32()).unwrap(), dtype);
        }
        // Pin the wire values.
        assert_eq!(DType::Int8.as_i32(), 0);
        assert_eq!(DType::Float64.as_i32(), 10);
        assert_eq!(DType::Bool.as_i32(), 13);
        assert!(DType::from_i32(14).is_err());
        assert!(DType::from_i32(-1).is_err());
    }

    #[test]
    fn endianness_prefixes_are_accepted() {
        assert_eq!(DType::from_descr("<f4").unwrap(), DType::Float32);
        assert_eq!(DType::from_descr("|i1").unwrap(), DType::Int8);
        // Single-byte types are the same under any prefix.
        assert_eq!(DType::from_descr("<u1").unwrap(), DType::Uint8);
        assert_eq!(DType::from_descr(">u1").unwrap(), DType::Uint8);
    }

    #[test]
    fn big_endian_multibyte_is_rejected() {
        let err = DType::from_descr(">f8").unwrap_err();
        assert!(matches!(err, AexError::UnsupportedDType(_)), "{err}");
        assert!(DType::from_descr(">i4").is_err());
        assert!(DType::from_descr(">c16").is_err());
    }

    #[test]
    fn non_numeric_type_chars_are_rejected() {
        // datetime64, timedelta64, bytes, unicode, object, raw
        for descr in ["<M8[s]", "<m8[ns]", "|S7", "<U3", "|O", "|V12"] {
            assert!(
                DType::from_descr(descr).is_err(),
                "{descr} must not be accepted"
            );
        }
    }

    #[test]
    fn unsupported_widths_are_rejected() {
        // numpy can write float128 / longdouble; AEX does not handle them.
        assert!(DType::from_descr("<f16").is_err());
        assert!(DType::from_descr("<c32").is_err());
    }
}
