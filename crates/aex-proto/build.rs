//! Generates the control plane client and server from `protos/aex.proto`.

use std::path::PathBuf;

fn main() {
    let proto = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../protos/aex.proto")
        .canonicalize()
        .expect("protos/aex.proto must exist");
    let proto_dir = proto
        .parent()
        .expect("proto has a parent directory")
        .to_path_buf();

    println!("cargo:rerun-if-changed={}", proto.display());

    tonic_prost_build::configure()
        .build_client(true)
        .build_server(true)
        // Without this, the generated client gets a `connect(dst)` constructor
        // that collides with our `Connect` RPC. The client builds its own
        // channel anyway, to set timeouts and message limits.
        .build_transport(false)
        .compile_protos(&[&proto], &[&proto_dir])
        .expect("failed to compile protos/aex.proto");
}
