//! How the bytes of a transfer are encoded: adaptive quality and the codec.
//!
//! Both are negotiated rather than demanded. A client asks for what it wants; a
//! server that cannot produce it falls back to the lossless default and reports
//! what it actually applied, so that a newer client and an older server never
//! fail to understand each other.
//!
//! What a build can produce depends on its features: [`Encoding::Exact`] and
//! [`Codec::Raw`] always, [`Encoding::ErrorBound`] and [`Codec::Sz`] with `sz`.
//! The rest of the values exist because they travel on the wire in a
//! fixed-width field, and a decoder has to name what it is refusing.

/// What was done to the elements before they were put on the wire.
///
/// Values match the proto `Encoding` enum and the `encoding` byte of a frame
/// header.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum Encoding {
    /// Lossless. The only one this release produces.
    #[default]
    Exact = 0,
    /// Elements narrowed to another dtype, e.g. float64 to float32.
    DtypeCast = 1,
    /// Every n-th element along an axis.
    Subsample = 2,
    /// Lossy, within a stated error bound.
    ErrorBound = 3,
}

impl Encoding {
    pub const fn as_u8(self) -> u8 {
        self as u8
    }

    pub const fn as_i32(self) -> i32 {
        self as i32
    }

    /// From the wire. `None` for a value this build does not know, which the
    /// caller reports in the terms of its own protocol.
    pub const fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Encoding::Exact),
            1 => Some(Encoding::DtypeCast),
            2 => Some(Encoding::Subsample),
            3 => Some(Encoding::ErrorBound),
            _ => None,
        }
    }

    /// From the proto `Encoding` value.
    pub const fn from_i32(v: i32) -> Option<Self> {
        if v < 0 || v > u8::MAX as i32 {
            return None;
        }
        Self::from_u8(v as u8)
    }

    /// Whether this build can produce it.
    pub const fn is_supported(self) -> bool {
        match self {
            Encoding::Exact => true,
            // Needs a codec to carry it.
            Encoding::ErrorBound => cfg!(feature = "sz"),
            Encoding::DtypeCast | Encoding::Subsample => false,
        }
    }
}

/// Every encoding, for enumeration.
pub const ALL_ENCODINGS: [Encoding; 4] = [
    Encoding::Exact,
    Encoding::DtypeCast,
    Encoding::Subsample,
    Encoding::ErrorBound,
];

/// How the payload is compressed on the wire.
///
/// Values match `TransferPlan.codec` and the `codec` byte of a frame header.
/// Only [`Codec::Raw`] keeps `wire_len == logical_len`, and only then can the
/// receiver read straight into the output array.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum Codec {
    #[default]
    Raw = 0,
    Lz4 = 1,
    Zstd = 2,
    /// Error-bounded and lossy: only ever paired with [`Encoding::ErrorBound`].
    Sz = 3,
}

impl Codec {
    pub const fn as_u8(self) -> u8 {
        self as u8
    }

    pub const fn as_u32(self) -> u32 {
        self as u32
    }

    /// From the wire. `None` for a value this build does not know.
    pub const fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Codec::Raw),
            1 => Some(Codec::Lz4),
            2 => Some(Codec::Zstd),
            3 => Some(Codec::Sz),
            _ => None,
        }
    }

    /// From the proto `requested_codec` value.
    pub const fn from_u32(v: u32) -> Option<Self> {
        if v > u8::MAX as u32 {
            return None;
        }
        Self::from_u8(v as u8)
    }

    /// Whether this build can produce it.
    pub const fn is_supported(self) -> bool {
        match self {
            Codec::Raw => true,
            Codec::Sz => cfg!(feature = "sz"),
            Codec::Lz4 | Codec::Zstd => false,
        }
    }
}

/// Every codec, for enumeration.
pub const ALL_CODECS: [Codec; 4] = [Codec::Raw, Codec::Lz4, Codec::Zstd, Codec::Sz];

/// The codecs this build can produce, as the wire's bitmask: bit n for codec n.
pub fn supported_codecs() -> u32 {
    ALL_CODECS
        .iter()
        .filter(|codec| codec.is_supported())
        .fold(0, |mask, codec| mask | 1 << codec.as_u32())
}

/// The encodings this build can produce, as the wire's bitmask.
pub fn supported_encodings() -> u32 {
    ALL_ENCODINGS
        .iter()
        .filter(|encoding| encoding.is_supported())
        .fold(0, |mask, encoding| mask | 1 << encoding.as_u8())
}

/// What a client asks for, or what a server applied.
///
/// The fields not belonging to `encoding` are carried along rather than
/// validated: a request for `Subsample` that falls back to `Exact` must not be
/// refused because its unused `abs_error_bound` made no sense.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct QualitySpec {
    pub encoding: Encoding,
    /// For [`Encoding::DtypeCast`].
    pub cast_dtype: Option<crate::dtype::DType>,
    /// For [`Encoding::Subsample`], one per axis.
    pub subsample_step: Vec<i64>,
    /// For [`Encoding::ErrorBound`].
    pub abs_error_bound: Option<f64>,
    pub rel_error_bound: Option<f64>,
}

impl QualitySpec {
    /// Lossless, which is what every transfer in this release gets.
    pub const fn exact() -> Self {
        QualitySpec {
            encoding: Encoding::Exact,
            cast_dtype: None,
            subsample_step: Vec::new(),
            abs_error_bound: None,
            rel_error_bound: None,
        }
    }

    pub fn is_exact(&self) -> bool {
        self.encoding == Encoding::Exact
    }

    /// What a server can actually produce for this request, on `dtype`.
    ///
    /// Anything this build cannot honour becomes `Exact` rather than an error:
    /// the client learns what happened from the plan it gets back and decides
    /// for itself whether the data is still worth having.
    pub fn applied(&self, dtype: crate::dtype::DType) -> QualitySpec {
        if self.can_apply(dtype) {
            self.clone()
        } else {
            QualitySpec::exact()
        }
    }

    fn can_apply(&self, dtype: crate::dtype::DType) -> bool {
        use crate::dtype::DType;
        match self.encoding {
            Encoding::Exact => true,
            Encoding::ErrorBound => {
                Encoding::ErrorBound.is_supported()
                    && matches!(dtype, DType::Float32 | DType::Float64)
                    // A bound relative to the value range would have to mean
                    // the range of the whole selection, and a block only ever
                    // sees its own, so asking for one is not honoured at all.
                    && self.rel_error_bound.is_none()
                    && self
                        .abs_error_bound
                        .is_some_and(|bound| bound.is_finite() && bound > 0.0)
            }
            Encoding::DtypeCast | Encoding::Subsample => false,
        }
    }

    /// The codec that carries this quality.
    pub fn codec(&self) -> Codec {
        match self.encoding {
            Encoding::ErrorBound => Codec::Sz,
            _ => Codec::Raw,
        }
    }

    /// The error bound, when there is one.
    pub fn eps(&self) -> Option<f64> {
        (self.encoding == Encoding::ErrorBound)
            .then_some(self.abs_error_bound)
            .flatten()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;

    #[test]
    fn encodings_and_codecs_survive_a_wire_roundtrip() {
        for encoding in [
            Encoding::Exact,
            Encoding::DtypeCast,
            Encoding::Subsample,
            Encoding::ErrorBound,
        ] {
            assert_eq!(Encoding::from_u8(encoding.as_u8()), Some(encoding));
            assert_eq!(Encoding::from_i32(encoding.as_i32()), Some(encoding));
        }
        for codec in [Codec::Raw, Codec::Lz4, Codec::Zstd, Codec::Sz] {
            assert_eq!(Codec::from_u8(codec.as_u8()), Some(codec));
            assert_eq!(Codec::from_u32(codec.as_u32()), Some(codec));
        }
        // Pin the wire values.
        assert_eq!(Encoding::Exact.as_u8(), 0);
        assert_eq!(Encoding::ErrorBound.as_u8(), 3);
        assert_eq!(Codec::Raw.as_u8(), 0);
        assert_eq!(Codec::Zstd.as_u8(), 2);
        assert_eq!(Codec::Sz.as_u8(), 3);
    }

    #[test]
    fn unknown_wire_values_are_reported_rather_than_guessed() {
        assert_eq!(Encoding::from_u8(4), None);
        assert_eq!(Encoding::from_u8(255), None);
        assert_eq!(Encoding::from_i32(-1), None);
        assert_eq!(Encoding::from_i32(1 << 20), None);
        assert_eq!(Codec::from_u8(4), None);
        assert_eq!(Codec::from_u32(1 << 20), None);
    }

    #[test]
    fn what_this_build_can_produce_is_what_it_advertises() {
        assert!(Encoding::Exact.is_supported());
        assert!(Codec::Raw.is_supported());
        assert!(!Encoding::DtypeCast.is_supported());
        assert!(!Encoding::Subsample.is_supported());
        assert!(!Codec::Lz4.is_supported());
        assert!(!Codec::Zstd.is_supported());
        assert_eq!(Encoding::ErrorBound.is_supported(), cfg!(feature = "sz"));
        assert_eq!(Codec::Sz.is_supported(), cfg!(feature = "sz"));

        // Bit n of the mask is codec n, which is what the client reads.
        assert_eq!(supported_codecs() & 1, 1);
        assert_eq!(supported_encodings() & 1, 1);
        assert_eq!(supported_codecs() >> 3 & 1, cfg!(feature = "sz") as u32);
        assert_eq!(supported_encodings() >> 3 & 1, cfg!(feature = "sz") as u32);
    }

    #[test]
    fn an_encoding_this_server_lacks_falls_back_to_exact() {
        let requested = QualitySpec {
            encoding: Encoding::Subsample,
            subsample_step: vec![2, 2],
            ..QualitySpec::default()
        };
        let applied = requested.applied(DType::Float32);
        assert!(applied.is_exact());
        // The fallback drops the settings that belonged to the encoding it
        // could not apply, so the plan reports exactly what was done.
        assert!(applied.subsample_step.is_empty());

        // A request it can honour comes back untouched.
        let exact = QualitySpec::exact();
        assert_eq!(exact.applied(DType::Float32), exact);
    }

    fn error_bound(abs: Option<f64>, rel: Option<f64>) -> QualitySpec {
        QualitySpec {
            encoding: Encoding::ErrorBound,
            abs_error_bound: abs,
            rel_error_bound: rel,
            ..QualitySpec::default()
        }
    }

    #[test]
    fn an_error_bound_is_applied_only_where_it_means_something() {
        let asked = error_bound(Some(1e-3), None);
        let honoured = cfg!(feature = "sz");
        assert_eq!(!asked.applied(DType::Float32).is_exact(), honoured);
        assert_eq!(!asked.applied(DType::Float64).is_exact(), honoured);

        // An integer has no error to bound, so the request is dropped rather
        // than turned into something the client did not ask for.
        assert!(asked.applied(DType::Int32).is_exact());
        assert!(asked.applied(DType::Complex64).is_exact());
        assert!(asked.applied(DType::Float16).is_exact());

        // A bound that is not a bound.
        assert!(error_bound(None, None).applied(DType::Float32).is_exact());
        assert!(error_bound(Some(0.0), None)
            .applied(DType::Float32)
            .is_exact());
        assert!(error_bound(Some(-1.0), None)
            .applied(DType::Float32)
            .is_exact());
        assert!(error_bound(Some(f64::NAN), None)
            .applied(DType::Float32)
            .is_exact());

        // A bound relative to the value range would have to mean the whole
        // selection's range, which no single block can see.
        assert!(error_bound(None, Some(1e-3))
            .applied(DType::Float32)
            .is_exact());
        assert!(error_bound(Some(1e-3), Some(1e-3))
            .applied(DType::Float32)
            .is_exact());
    }

    #[test]
    fn a_quality_names_the_codec_that_carries_it() {
        assert_eq!(QualitySpec::exact().codec(), Codec::Raw);
        assert_eq!(QualitySpec::exact().eps(), None);
        assert_eq!(error_bound(Some(1e-3), None).codec(), Codec::Sz);
        assert_eq!(error_bound(Some(1e-3), None).eps(), Some(1e-3));
    }
}
