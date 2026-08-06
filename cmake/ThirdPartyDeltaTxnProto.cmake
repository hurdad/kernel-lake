# Generates C++ protobuf + gRPC client stubs from the vendored
# proto/delta_txn.proto (see that file's own header comment) at build time,
# producing a `delta_txn_proto` static library src/delta/ links against.
#
# Deliberately does NOT vendor or build delta-txn-service itself (a
# separate, standalone Rust project -- github "delta-txn-service" repo):
# KernelLake talks to it purely as a gRPC *client* over the network, the
# same way it talks to S3/GCS/Azure/HDFS as a client of those services --
# see docs/ROADMAP.md's lakehouse roadmap, Phase 4, for why a Rust
# toolchain was deliberately kept out of this build rather than vendoring
# delta-kernel-rs (or delta-rs, what delta-txn-service itself uses)
# directly into KernelLake.

set(DELTA_TXN_PROTO_FILE "${PROJECT_SOURCE_DIR}/proto/delta_txn.proto")
set(DELTA_TXN_PROTO_GEN_DIR "${CMAKE_BINARY_DIR}/generated/delta_txn")
file(MAKE_DIRECTORY "${DELTA_TXN_PROTO_GEN_DIR}")

set(DELTA_TXN_PB_H "${DELTA_TXN_PROTO_GEN_DIR}/delta_txn.pb.h")
set(DELTA_TXN_PB_CC "${DELTA_TXN_PROTO_GEN_DIR}/delta_txn.pb.cc")
set(DELTA_TXN_GRPC_PB_H "${DELTA_TXN_PROTO_GEN_DIR}/delta_txn.grpc.pb.h")
set(DELTA_TXN_GRPC_PB_CC "${DELTA_TXN_PROTO_GEN_DIR}/delta_txn.grpc.pb.cc")

# Two separate protoc invocations (plain protobuf codegen, then the gRPC
# plugin's service-stub codegen) rather than one combined --cpp_out
# --grpc_out call: both still run once per configure/build via the same
# custom command, this just mirrors how protoc's own docs and most
# hand-written (non-FetchContent) CMake gRPC integrations structure it,
# keeping the two clearly-separate codegen passes visible as separate
# COMMAND lines instead of one long one.
add_custom_command(
  OUTPUT "${DELTA_TXN_PB_H}" "${DELTA_TXN_PB_CC}" "${DELTA_TXN_GRPC_PB_H}" "${DELTA_TXN_GRPC_PB_CC}"
  COMMAND protobuf::protoc
          --cpp_out "${DELTA_TXN_PROTO_GEN_DIR}"
          -I "${PROJECT_SOURCE_DIR}/proto"
          "${DELTA_TXN_PROTO_FILE}"
  COMMAND protobuf::protoc
          --grpc_out "${DELTA_TXN_PROTO_GEN_DIR}"
          --plugin=protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>
          -I "${PROJECT_SOURCE_DIR}/proto"
          "${DELTA_TXN_PROTO_FILE}"
  DEPENDS "${DELTA_TXN_PROTO_FILE}" protobuf::protoc gRPC::grpc_cpp_plugin
  COMMENT "Generating C++ protobuf/gRPC stubs from proto/delta_txn.proto"
  VERBATIM
)

add_library(delta_txn_proto STATIC
  "${DELTA_TXN_PB_CC}"
  "${DELTA_TXN_GRPC_PB_CC}"
)

target_include_directories(delta_txn_proto PUBLIC "${DELTA_TXN_PROTO_GEN_DIR}")
target_compile_features(delta_txn_proto PUBLIC cxx_std_20)

target_link_libraries(delta_txn_proto PUBLIC
  protobuf::libprotobuf
  gRPC::grpc++
)
