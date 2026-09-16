//! Generated control plane types, and the conversions to the core ones.
//!
//! The generated code is kept in its own crate so that running `protoc` is not
//! part of building the server or the client logic. [`convert`] sits alongside
//! it because both sides of the control plane map the same types in opposite
//! directions, and two copies of that mapping is how a client comes to send
//! what a server reads as something else.

pub mod convert;

pub use generated::*;

/// Generated code is not ours to lint, so it is kept behind a module that can
/// carry the exemption without it reaching the rest of the crate.
#[allow(clippy::all)]
mod generated {
    tonic::include_proto!("aex.v2");
}

#[cfg(test)]
mod tests {
    use aex_core::{DType, ALL_DTYPES};

    use super::DataType;

    #[test]
    fn data_type_matches_the_core_dtype() {
        // The server converts between the two with a cast, so the two enums
        // have to agree value for value.
        for dtype in ALL_DTYPES {
            let wire = DataType::try_from(dtype.as_i32())
                .unwrap_or_else(|_| panic!("{dtype} has no DataType"));
            assert_eq!(wire.as_str_name(), format!("{dtype:?}").to_uppercase());
            assert_eq!(DType::from_i32(wire as i32).unwrap(), dtype);
        }
        // Neither side has a value the other lacks.
        assert!(DataType::try_from(ALL_DTYPES.len() as i32).is_err());
    }
}
