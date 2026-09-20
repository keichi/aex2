//! How the bytes of a transfer are encoded: adaptive quality and the codec.
//!
//! Both are negotiated rather than demanded. A client asks for what it wants; a
//! server that cannot produce it falls back to the lossless default and reports
//! what it actually applied, so that a newer client and an older server never
//! fail to understand each other.
//!
//! What a build can produce depends on its features: [`Encoding::Exact`],
//! [`Encoding::DtypeCast`], [`Codec::Raw`] and [`Codec::Gzip`] always,
//! [`Codec::Sz`] with `sz`, [`Codec::Zfp`] with `zfp`, and
//! [`Encoding::ErrorBound`] with either.

/// What was done to the elements before they were put on the wire.
///
/// Values match the proto `Encoding` enum and the `encoding` byte of a frame
/// header.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum Encoding {
    /// Lossless, and what everything falls back to.
    #[default]
    Exact = 0,
    /// Elements narrowed to another dtype: float64 to float32, or float32 to
    /// float16, and nothing else.
    DtypeCast = 1,
    /// Lossy, within a stated error bound.
    ErrorBound = 2,
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
            2 => Some(Encoding::ErrorBound),
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
            Encoding::ErrorBound => cfg!(feature = "sz") || cfg!(feature = "zfp"),
            // Narrowing a float is arithmetic this build always has.
            Encoding::DtypeCast => true,
        }
    }
}

/// Every encoding, for the capability bitmask below.
const ALL_ENCODINGS: [Encoding; 3] = [Encoding::Exact, Encoding::DtypeCast, Encoding::ErrorBound];

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
    /// Error-bounded and lossy: only ever paired with [`Encoding::ErrorBound`].
    Sz = 1,
    /// The other error-bounded one. Faster and blockier than [`Codec::Sz`],
    /// and it does not survive NaN or infinity.
    Zfp = 2,
    /// Lossless deflate. The one codec that pairs with [`Encoding::Exact`].
    ///
    /// It is here to be measured against, not to be reached for: on real
    /// float32 fields it returns 1.1x to 1.4x, where an error-bounded codec
    /// returns more than an order of magnitude at an error nobody can see.
    /// Never the default for anything.
    Gzip = 3,
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
            1 => Some(Codec::Sz),
            2 => Some(Codec::Zfp),
            3 => Some(Codec::Gzip),
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

    /// Whether this is one of the lossy, error-bounded codecs.
    pub const fn is_error_bounded(self) -> bool {
        matches!(self, Codec::Sz | Codec::Zfp)
    }

    /// Whether this build can produce it.
    pub const fn is_supported(self) -> bool {
        match self {
            Codec::Raw | Codec::Gzip => true,
            Codec::Sz => cfg!(feature = "sz"),
            Codec::Zfp => cfg!(feature = "zfp"),
        }
    }
}

/// Every codec, for the capability bitmask below.
const ALL_CODECS: [Codec; 4] = [Codec::Raw, Codec::Sz, Codec::Zfp, Codec::Gzip];

/// The error-bounded codec a build reaches for when the client names none.
///
/// SZ3 first because it is the one that was measured first, and because it
/// honours an error bound as given where zfp rounds it down to a power of two.
const DEFAULT_LOSSY: Codec = if cfg!(feature = "sz") {
    Codec::Sz
} else {
    Codec::Zfp
};

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
/// validated: a request for `DtypeCast` that falls back to `Exact` must not be
/// refused because its unused `abs_error_bound` made no sense.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct QualitySpec {
    pub encoding: Encoding,
    /// For [`Encoding::DtypeCast`].
    pub cast_dtype: Option<crate::dtype::DType>,
    /// For [`Encoding::ErrorBound`].
    pub abs_error_bound: Option<f64>,
    pub rel_error_bound: Option<f64>,
    /// Which error-bounded codec should carry it. `None` takes the build's
    /// own, which is what a client with no preference asks for.
    ///
    /// Unlike the rest, this one is not a field of the proto `QualitySpec`: it
    /// travels as `PrepareSelectionRequest.requested_codec` going out and as
    /// `TransferPlan.codec` coming back, which is where the wire already had
    /// room for it.
    pub codec: Option<Codec>,
}

impl QualitySpec {
    /// Lossless, which is what every transfer in this release gets.
    pub const fn exact() -> Self {
        QualitySpec {
            encoding: Encoding::Exact,
            cast_dtype: None,
            abs_error_bound: None,
            rel_error_bound: None,
            codec: None,
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
                // A codec this build cannot produce is not quietly swapped for
                // one it can: the client asked for a particular one, and
                // falling back to EXACT is what tells it so.
                self.codec().is_supported()
                    && matches!(dtype, DType::Float32 | DType::Float64)
                    // A bound relative to the value range would have to mean
                    // the range of the whole selection, and a block only ever
                    // sees its own, so asking for one is not honoured at all.
                    && self.rel_error_bound.is_none()
                    && self
                        .abs_error_bound
                        .is_some_and(|bound| bound.is_finite() && bound > 0.0)
            }
            // Only these two. Both halve the element and keep it a float, so
            // the wire length is exactly half and the values still mean what
            // they meant. A cast that changes the kind of number — a float to
            // an integer, say — is a different question, about saturation and
            // rounding and signedness, and is not answered here.
            //
            // Nothing checks that the values fit: float32 to float16 overflows
            // to infinity above 65504, the way `numpy.astype` does. Finding
            // out would mean reading the whole selection before agreeing to
            // send it, and the plan says what was applied, so a caller that
            // cares can look at what arrived.
            Encoding::DtypeCast => matches!(
                (dtype, self.cast_dtype),
                (DType::Float64, Some(DType::Float32)) | (DType::Float32, Some(DType::Float16))
            ),
        }
    }

    /// The codec that carries this quality, once `codec` has had the build's
    /// default filled in for it.
    ///
    /// Under an error bound, a codec that is not one of the error-bounded
    /// ones reads as no preference rather than as a request: a client from
    /// before there was a choice leaves the field at RAW, and RAW cannot carry
    /// a bound. Anywhere else the answer is RAW unless a lossless codec this
    /// build has was named. Falling back to RAW there needs no telling: a
    /// lossless codec changes the bytes on the wire and nothing else.
    pub fn codec(&self) -> Codec {
        match self.encoding {
            Encoding::ErrorBound => self
                .codec
                .filter(|codec| codec.is_error_bounded())
                .unwrap_or(DEFAULT_LOSSY),
            _ => self
                .codec
                .filter(|codec| !codec.is_error_bounded() && codec.is_supported())
                .unwrap_or(Codec::Raw),
        }
    }

    /// The element type that travels, given the array's own.
    ///
    /// Only a cast this quality can actually apply changes it, so an applied
    /// quality is what this wants; a request that was going to fall back
    /// would otherwise name a dtype the transfer never uses.
    pub fn wire_dtype(&self, dtype: crate::dtype::DType) -> crate::dtype::DType {
        match self.encoding {
            Encoding::DtypeCast if self.can_apply(dtype) => self.cast_dtype.unwrap_or(dtype),
            _ => dtype,
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
        for encoding in ALL_ENCODINGS {
            assert_eq!(Encoding::from_u8(encoding.as_u8()), Some(encoding));
            assert_eq!(Encoding::from_i32(encoding.as_i32()), Some(encoding));
        }
        for codec in ALL_CODECS {
            assert_eq!(Codec::from_u8(codec.as_u8()), Some(codec));
            assert_eq!(Codec::from_u32(codec.as_u32()), Some(codec));
        }
        // Pin the wire values.
        assert_eq!(Encoding::Exact.as_u8(), 0);
        assert_eq!(Encoding::DtypeCast.as_u8(), 1);
        assert_eq!(Encoding::ErrorBound.as_u8(), 2);
        assert_eq!(Codec::Raw.as_u8(), 0);
        assert_eq!(Codec::Sz.as_u8(), 1);
        assert_eq!(Codec::Zfp.as_u8(), 2);
        assert_eq!(Codec::Gzip.as_u8(), 3);
    }

    #[test]
    fn unknown_wire_values_are_reported_rather_than_guessed() {
        assert_eq!(Encoding::from_u8(3), None);
        assert_eq!(Encoding::from_u8(255), None);
        assert_eq!(Encoding::from_i32(-1), None);
        assert_eq!(Encoding::from_i32(1 << 20), None);
        assert_eq!(Codec::from_u8(4), None);
        assert_eq!(Codec::from_u32(1 << 20), None);
    }

    #[test]
    fn what_this_build_can_produce_is_what_it_advertises() {
        assert!(Encoding::Exact.is_supported());
        assert!(Encoding::DtypeCast.is_supported());
        assert!(Codec::Raw.is_supported());
        assert!(Codec::Gzip.is_supported());
        assert_eq!(Codec::Sz.is_supported(), cfg!(feature = "sz"));
        assert_eq!(Codec::Zfp.is_supported(), cfg!(feature = "zfp"));
        // Either codec carries an error bound; neither is needed for the other.
        let lossy = cfg!(feature = "sz") || cfg!(feature = "zfp");
        assert_eq!(Encoding::ErrorBound.is_supported(), lossy);

        // Bit n of the mask is codec n, which is what the client reads.
        assert_eq!(supported_codecs() & 1, 1);
        assert_eq!(supported_codecs() >> 3 & 1, 1);
        assert_eq!(supported_encodings() & 3, 3);
        assert_eq!(supported_codecs() >> 1 & 1, cfg!(feature = "sz") as u32);
        assert_eq!(supported_codecs() >> 2 & 1, cfg!(feature = "zfp") as u32);
        assert_eq!(supported_encodings() >> 2 & 1, lossy as u32);
    }

    fn cast(to: DType) -> QualitySpec {
        QualitySpec {
            encoding: Encoding::DtypeCast,
            cast_dtype: Some(to),
            ..QualitySpec::default()
        }
    }

    #[test]
    fn an_encoding_this_server_lacks_falls_back_to_exact() {
        // Widening is not one of the two narrowings, so it is refused.
        let applied = cast(DType::Float64).applied(DType::Float32);
        assert!(applied.is_exact());
        // The fallback drops the settings that belonged to the encoding it
        // could not apply, so the plan reports exactly what was done.
        assert!(applied.cast_dtype.is_none());

        // A request it can honour comes back untouched.
        let exact = QualitySpec::exact();
        assert_eq!(exact.applied(DType::Float32), exact);
    }

    #[test]
    fn only_the_two_float_narrowings_are_cast() {
        for (from, to) in [
            (DType::Float64, DType::Float32),
            (DType::Float32, DType::Float16),
        ] {
            let applied = cast(to).applied(from);
            assert_eq!(applied.encoding, Encoding::DtypeCast, "{from} to {to}");
            assert_eq!(applied.wire_dtype(from), to);
        }
        // Skipping a step, widening, changing the kind of number, and asking
        // for the type the array already has.
        for (from, to) in [
            (DType::Float64, DType::Float16),
            (DType::Float32, DType::Float64),
            (DType::Float32, DType::Int16),
            (DType::Int64, DType::Int32),
            (DType::Float32, DType::Float32),
        ] {
            assert!(cast(to).applied(from).is_exact(), "{from} to {to}");
            assert_eq!(cast(to).wire_dtype(from), from, "{from} to {to}");
        }
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
        let honoured = Encoding::ErrorBound.is_supported();
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
        assert_eq!(error_bound(Some(1e-3), None).codec(), DEFAULT_LOSSY);
        assert_eq!(error_bound(Some(1e-3), None).eps(), Some(1e-3));
    }

    #[test]
    fn the_codec_asked_for_is_the_one_that_carries_it() {
        for want in [Codec::Sz, Codec::Zfp] {
            let asked = QualitySpec {
                codec: Some(want),
                ..error_bound(Some(1e-3), None)
            };
            assert_eq!(asked.codec(), want);
            // Asking for one this build does not have falls back to the
            // lossless default rather than quietly using the other one: the
            // client asked for that codec's error behaviour, not any codec's.
            let applied = asked.applied(DType::Float32);
            assert_eq!(!applied.is_exact(), want.is_supported());
            if want.is_supported() {
                assert_eq!(applied.codec(), want);
            }
        }
    }

    #[test]
    fn a_lossless_codec_in_that_field_reads_as_no_preference() {
        // A client from before there was a choice leaves the field at RAW,
        // and RAW cannot carry a bound, so it must not be read as one.
        let asked = QualitySpec {
            codec: Some(Codec::Raw),
            ..error_bound(Some(1e-3), None)
        };
        assert_eq!(asked.codec(), DEFAULT_LOSSY);
        assert_eq!(
            !asked.applied(DType::Float32).is_exact(),
            Encoding::ErrorBound.is_supported()
        );
    }
}
