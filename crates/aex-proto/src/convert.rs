//! Between the generated types and the core ones.
//!
//! Both sides of the control plane convert the same things in opposite
//! directions, so the mapping lives once and is tested once. Two independent
//! copies of it is how a client comes to send what a server reads as something
//! else.
//!
//! Quality is the asymmetric part. A client asks for an encoding; a server that
//! cannot produce it falls back to `EXACT` and reports what it applied. So an
//! encoding this build cannot even name is read as `EXACT` rather than
//! refused — it could not have been applied either way, and the reply says so.

use aex_core::{AexError, DType, Encoding, Index, QualitySpec, Result};

use crate::{index, Fancy, Slice};

/// Expanded indices one fancy selection may have.
///
/// Both sides hold the same number so that the client can refuse an oversized
/// selection before it costs a round trip. A server configured lower still
/// refuses what is over its own limit.
pub const DEFAULT_MAX_FANCY_INDICES: u64 = 262_144;

/// Convert a selection for the wire.
pub fn indices_to_proto(indices: &[Index]) -> Vec<crate::Index> {
    indices.iter().map(index_to_proto).collect()
}

/// Convert a selection off the wire, refusing one bigger than `max_fancy`.
///
/// The limit is checked on both sides: the server has to enforce it, and the
/// client checks before sending so that an oversized selection does not cost a
/// round trip to find out.
pub fn indices_from_proto(indices: &[crate::Index], max_fancy: u64) -> Result<Vec<Index>> {
    indices
        .iter()
        .map(|index| index_from_proto(index, max_fancy))
        .collect()
}

pub fn index_to_proto(index: &Index) -> crate::Index {
    let kind = match index {
        Index::Single(i) => index::Kind::Single(*i),
        Index::Slice { start, stop, step } => index::Kind::Slice(Slice {
            start: *start,
            stop: *stop,
            step: *step,
        }),
        // A mask arrives here already expanded, and reaches the server as the
        // integer indices it became.
        Index::Fancy(list) => index::Kind::Fancy(Fancy {
            indices: list.clone(),
        }),
        Index::Ellipsis => index::Kind::Ellipsis(true),
        Index::NewAxis => index::Kind::Newaxis(true),
    };
    crate::Index { kind: Some(kind) }
}

pub fn index_from_proto(index: &crate::Index, max_fancy: u64) -> Result<Index> {
    let kind = index.kind.as_ref().ok_or_else(|| {
        AexError::BadSelection("an index of no kind at all is not a selection".to_string())
    })?;

    let converted = match kind {
        index::Kind::Single(i) => Index::Single(*i),
        index::Kind::Slice(slice) => Index::Slice {
            start: slice.start,
            stop: slice.stop,
            step: slice.step,
        },
        index::Kind::Fancy(fancy) | index::Kind::MaskTrueIndices(fancy) => {
            check_fancy_len(fancy.indices.len(), max_fancy)?;
            Index::Fancy(fancy.indices.clone())
        }
        // The bool is only how a oneof carries a field with nothing in it; what
        // it means is that this arm was chosen.
        index::Kind::Ellipsis(_) => Index::Ellipsis,
        index::Kind::Newaxis(_) => Index::NewAxis,
    };
    Ok(converted)
}

/// Refuse a selection too large to travel, before it is sent.
pub fn check_fancy_limit(indices: &[Index], max_fancy: u64) -> Result<()> {
    for index in indices {
        if let Index::Fancy(list) = index {
            check_fancy_len(list.len(), max_fancy)?;
        }
    }
    Ok(())
}

/// Refuse a fancy selection too large to travel.
///
/// The indices are `repeated int64`, so a million of them is eight megabytes
/// and over the gRPC message limit. Refusing is the honest answer until there
/// is a compact way to say the same thing.
fn check_fancy_len(len: usize, max_fancy: u64) -> Result<()> {
    if len as u64 > max_fancy {
        return Err(AexError::BadSelection(format!(
            "a fancy selection of {len} indices is over this server's limit of {max_fancy}; \
             split the selection, or take a range instead"
        )));
    }
    Ok(())
}

pub fn quality_to_proto(quality: &QualitySpec) -> crate::QualitySpec {
    crate::QualitySpec {
        encoding: quality.encoding.as_i32(),
        cast_dtype: quality.cast_dtype.map(DType::as_i32),
        abs_error_bound: quality.abs_error_bound,
        rel_error_bound: quality.rel_error_bound,
    }
}

/// Convert a quality request off the wire.
///
/// Total by design: anything this build cannot name reads as the lossless
/// default, which is what it would have fallen back to anyway.
pub fn quality_from_proto(quality: Option<&crate::QualitySpec>) -> QualitySpec {
    let Some(quality) = quality else {
        return QualitySpec::exact();
    };
    let Some(encoding) = Encoding::from_i32(quality.encoding) else {
        return QualitySpec::exact();
    };
    QualitySpec {
        encoding,
        cast_dtype: quality.cast_dtype.and_then(|d| DType::from_i32(d).ok()),
        abs_error_bound: quality.abs_error_bound,
        rel_error_bound: quality.rel_error_bound,
        // Not a field of the proto message: the codec has one of its own, next
        // to the quality rather than inside it.
        codec: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// No limit, for the cases that are not about the limit.
    const ANY: u64 = u64::MAX;

    #[test]
    fn every_kind_of_index_survives_a_roundtrip() {
        let indices = vec![
            Index::Single(5),
            Index::Single(-1),
            Index::Slice {
                start: Some(10),
                stop: Some(20),
                step: Some(2),
            },
            Index::Slice {
                start: None,
                stop: None,
                step: None,
            },
            Index::Slice {
                start: None,
                stop: None,
                step: Some(-1),
            },
            Index::Fancy(vec![1, 5, 10]),
            Index::Fancy(vec![]),
            Index::Ellipsis,
            Index::NewAxis,
        ];
        let wire = indices_to_proto(&indices);
        assert_eq!(indices_from_proto(&wire, ANY).unwrap(), indices);
    }

    #[test]
    fn a_missing_slice_bound_stays_missing() {
        // None and 0 mean different things — arr[:5] is not arr[0:5] when the
        // step is negative — so they must not collapse into each other.
        let open = Index::Slice {
            start: None,
            stop: Some(5),
            step: None,
        };
        let closed = Index::Slice {
            start: Some(0),
            stop: Some(5),
            step: None,
        };
        assert_ne!(index_to_proto(&open), index_to_proto(&closed));
        assert_eq!(index_from_proto(&index_to_proto(&open), ANY).unwrap(), open);
    }

    #[test]
    fn a_boolean_mask_arrives_as_the_indices_it_expanded_to() {
        let wire = crate::Index {
            kind: Some(index::Kind::MaskTrueIndices(Fancy {
                indices: vec![0, 3, 7],
            })),
        };
        assert_eq!(
            index_from_proto(&wire, ANY).unwrap(),
            Index::Fancy(vec![0, 3, 7])
        );
    }

    #[test]
    fn an_index_of_no_kind_is_refused() {
        // An older or broken client, or a field this build does not know.
        let err = index_from_proto(&crate::Index { kind: None }, ANY).unwrap_err();
        assert!(matches!(err, AexError::BadSelection(_)), "{err}");
    }

    #[test]
    fn a_fancy_selection_over_the_limit_is_refused_with_advice() {
        let wire = indices_to_proto(&[Index::Fancy((0..100).collect())]);
        indices_from_proto(&wire, 100).expect("exactly at the limit");

        let err = indices_from_proto(&wire, 99).unwrap_err();
        assert!(matches!(err, AexError::BadSelection(_)), "{err}");
        // The message has to say what to do instead; the limit exists because
        // the request would not fit a gRPC message, not because of the array.
        assert!(err.to_string().contains("split the selection"), "{err}");
        assert_eq!(err.class(), aex_core::ErrorClass::Request);

        // The client checks the same limit before sending, so that an
        // oversized selection does not cost a round trip to be refused.
        let indices = [Index::Ellipsis, Index::Fancy((0..100).collect())];
        check_fancy_limit(&indices, 100).expect("at the limit");
        assert!(check_fancy_limit(&indices, 99).is_err());
    }

    #[test]
    fn a_quality_request_survives_a_roundtrip() {
        let quality = QualitySpec {
            encoding: Encoding::DtypeCast,
            cast_dtype: Some(DType::Float32),
            abs_error_bound: Some(0.5),
            rel_error_bound: None,
            // The codec does not ride inside the message, so a roundtrip
            // through it cannot bring one back.
            codec: None,
        };
        assert_eq!(
            quality_from_proto(Some(&quality_to_proto(&quality))),
            quality
        );
        // An absent one is the lossless default.
        assert_eq!(quality_from_proto(None), QualitySpec::exact());
    }

    #[test]
    fn a_quality_this_build_cannot_name_reads_as_exact() {
        // A newer client asking for an encoding this server has never heard of
        // gets what it would have got anyway: EXACT, and a plan that says so.
        let wire = crate::QualitySpec {
            encoding: 99,
            ..crate::QualitySpec::default()
        };
        assert!(quality_from_proto(Some(&wire)).is_exact());

        // A dtype it cannot name is simply not requested.
        let wire = crate::QualitySpec {
            encoding: Encoding::DtypeCast.as_i32(),
            cast_dtype: Some(99),
            ..crate::QualitySpec::default()
        };
        assert_eq!(quality_from_proto(Some(&wire)).cast_dtype, None);
    }
}
