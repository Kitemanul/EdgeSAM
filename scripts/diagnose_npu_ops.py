#!/usr/bin/env python
"""
Diagnose NPU compilation failures by testing individual ONNX operator groups.

Generates minimal ONNX models (opset 11, all int64 converted to int32 —
matching the real NPU export pipeline), each isolating one group of operators
from the EdgeSAM decoder.  Compile each .onnx with your NPU toolchain;
pass/fail results pinpoint the problematic operator(s).

Operator coverage (27 operators, 8 test models):

  Test model          | Operators tested
  --------------------+------------------------------------------------------
  01_conv_transpose   | ConvTranspose, Relu
  02_bool_logic       | Equal, Not, Cast
  03_gather           | Gather
  04_tile_expand      | Tile, Expand, Unsqueeze
  05_attention        | MatMul, Softmax, Transpose, Reshape, Slice, Mul
  06_layernorm        | ReduceMean, Sub, Pow, Sqrt, Div, Mul, Add
  07_sincos_pe        | Sin, Cos, Mul, Concat
  08_gemm_tanh        | Gemm, Tanh

Usage:
    python scripts/diagnose_npu_ops.py [--output-dir ./npu_diag]

Then compile each generated .onnx file with your NPU compiler and report
which ones PASS and which ones FAIL.
"""

import argparse
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OPSET = 11
IR_VERSION = 6


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _build_model(name, nodes, inputs, outputs, initializers=None):
    """Build opset-11 model, validate, convert int64→int32, return."""
    graph = helper.make_graph(
        nodes, name, inputs, outputs, initializer=initializers or [],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", OPSET)]
    )
    model.ir_version = IR_VERSION
    onnx.checker.check_model(model)
    # Match real NPU export: convert all int64/int16 → int32
    _convert_int_to_int32(model)
    return model


def _convert_int_to_int32(model):
    """In-place int64/int16 → int32 (mirrors NPU export Fix 4)."""
    CONVERTIBLE = {TensorProto.INT64, TensorProto.INT16}

    for node in model.graph.node:
        if node.op_type == "Cast":
            for a in node.attribute:
                if a.name == "to" and a.i in CONVERTIBLE:
                    a.i = TensorProto.INT32
        if node.op_type in ("Constant", "ConstantOfShape"):
            for a in node.attribute:
                if a.name == "value" and a.t.data_type in CONVERTIBLE:
                    dt = np.int64 if a.t.data_type == TensorProto.INT64 else np.int16
                    arr = np.frombuffer(a.t.raw_data, dtype=dt)
                    a.t.raw_data = arr.astype(np.int32).tobytes()
                    a.t.data_type = TensorProto.INT32

    for init in model.graph.initializer:
        if init.data_type in CONVERTIBLE:
            dt = np.int64 if init.data_type == TensorProto.INT64 else np.int16
            arr = np.frombuffer(init.raw_data, dtype=dt)
            init.raw_data = arr.astype(np.int32).tobytes()
            init.data_type = TensorProto.INT32

    for coll in (model.graph.input, model.graph.output, model.graph.value_info):
        for vi in coll:
            if vi.type.tensor_type.elem_type in CONVERTIBLE:
                vi.type.tensor_type.elem_type = TensorProto.INT32


def _list_ops(model):
    """Return sorted unique operator names from a model."""
    return sorted(set(n.op_type for n in model.graph.node))


# ------------------------------------------------------------------
# Test model builders
# ------------------------------------------------------------------

def make_conv_transpose():
    """ConvTranspose + Relu — mask upscaling path.

    Decoder output_upscaling:
      ConvTranspose2d(256→64, k=2, s=2) → Relu
      ConvTranspose2d(64→32,  k=2, s=2) → Relu
    """
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 256, 64, 64])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 32, 256, 256])

    W1 = numpy_helper.from_array(
        np.random.randn(256, 64, 2, 2).astype(np.float32) * 0.01, "w1")
    B1 = numpy_helper.from_array(np.zeros(64, dtype=np.float32), "b1")
    W2 = numpy_helper.from_array(
        np.random.randn(64, 32, 2, 2).astype(np.float32) * 0.01, "w2")
    B2 = numpy_helper.from_array(np.zeros(32, dtype=np.float32), "b2")

    nodes = [
        helper.make_node("ConvTranspose", ["input", "w1", "b1"], ["h1"],
                         kernel_shape=[2, 2], strides=[2, 2]),
        helper.make_node("Relu", ["h1"], ["h2"]),
        helper.make_node("ConvTranspose", ["h2", "w2", "b2"], ["h3"],
                         kernel_shape=[2, 2], strides=[2, 2]),
        helper.make_node("Relu", ["h3"], ["output"]),
    ]

    return _build_model("conv_transpose_test", nodes, [X], [Y], [W1, B1, W2, B2])


def make_bool_logic():
    """Equal + Not + Cast(bool→float) — prompt label comparison.

    _embed_points does:
      point_embedding *= (point_labels != -1)
      point_embedding += embed[i] * (point_labels == i)
    """
    labels = helper.make_tensor_value_info(
        "labels", TensorProto.FLOAT, [1, 5, 256])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 5, 256])

    neg_one = numpy_helper.from_array(np.array(-1.0, dtype=np.float32), "neg_one")
    zero_f = numpy_helper.from_array(np.array(0.0, dtype=np.float32), "zero_f")

    nodes = [
        # (labels == -1)
        helper.make_node("Equal", ["labels", "neg_one"], ["eq_neg1"]),
        # !(labels == -1)
        helper.make_node("Not", ["eq_neg1"], ["neq_neg1"]),
        # cast bool → float
        helper.make_node("Cast", ["neq_neg1"], ["mask_f"], to=TensorProto.FLOAT),
        # (labels == 0)
        helper.make_node("Equal", ["labels", "zero_f"], ["eq_zero"]),
        helper.make_node("Cast", ["eq_zero"], ["mask0_f"], to=TensorProto.FLOAT),
        # combine
        helper.make_node("Add", ["mask_f", "mask0_f"], ["output"]),
    ]

    return _build_model("bool_logic_test", nodes, [labels], [Y], [neg_one, zero_f])


def make_gather():
    """Gather — embedding table lookup.

    prompt_encoder.point_embeddings[i].weight selected by label index.
    Uses int64 indices (converted to int32 by our pipeline).
    """
    table = helper.make_tensor_value_info("table", TensorProto.FLOAT, [5, 256])
    indices = helper.make_tensor_value_info("indices", TensorProto.INT64, [1, 3])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 256])

    nodes = [
        helper.make_node("Gather", ["table", "indices"], ["output"], axis=0),
    ]

    return _build_model("gather_test", nodes, [table, indices], [Y])


def make_tile_expand():
    """Tile + Expand + Unsqueeze — prompt broadcasting.

    point_labels.unsqueeze(-1).expand_as(point_embedding) and
    dense embedding spatial broadcast both use these ops.
    """
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 5])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 5, 256])

    # int64 for ONNX spec, will be converted to int32
    repeats = numpy_helper.from_array(
        np.array([1, 1, 256], dtype=np.int64), "repeats")
    shape = numpy_helper.from_array(
        np.array([1, 5, 256], dtype=np.int64), "shape")

    nodes = [
        # Unsqueeze: (1,5) → (1,5,1)   [opset 11: axes is attribute]
        helper.make_node("Unsqueeze", ["input"], ["unsq"], axes=[2]),
        # Tile: (1,5,1) → (1,5,256)
        helper.make_node("Tile", ["unsq", "repeats"], ["tiled"]),
        # Expand: (1,5,1) → (1,5,256)
        helper.make_node("Expand", ["unsq", "shape"], ["expanded"]),
        # Combine to keep both paths
        helper.make_node("Add", ["tiled", "expanded"], ["output"]),
    ]

    return _build_model("tile_expand_test", nodes, [X], [Y], [repeats, shape])


def make_attention():
    """MatMul + Softmax + Transpose + Reshape + Slice — transformer attention.

    TwoWayTransformer: multi-head attention with head reshape/transpose,
    plus Slice for extracting mask/IoU tokens from output.

    Shapes match decoder: 10 query tokens, 4096 (=64*64) key/value tokens,
    8 heads, head_dim=32, embed_dim=256.
    """
    Q_in = helper.make_tensor_value_info("q", TensorProto.FLOAT, [1, 10, 256])
    K_in = helper.make_tensor_value_info("k", TensorProto.FLOAT, [1, 4096, 256])
    V_in = helper.make_tensor_value_info("v", TensorProto.FLOAT, [1, 4096, 256])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 5, 256])

    # Reshape targets (int64 → will be int32 after conversion)
    q_shape = numpy_helper.from_array(
        np.array([1, 10, 8, 32], dtype=np.int64), "q_shape")
    kv_shape = numpy_helper.from_array(
        np.array([1, 4096, 8, 32], dtype=np.int64), "kv_shape")
    out_shape = numpy_helper.from_array(
        np.array([1, 10, 256], dtype=np.int64), "out_shape")
    scale = numpy_helper.from_array(
        np.array(1.0 / np.sqrt(32), dtype=np.float32), "scale")

    # Slice params: extract first 5 tokens (like mask tokens)
    sl_starts = numpy_helper.from_array(np.array([0], dtype=np.int64), "sl_starts")
    sl_ends = numpy_helper.from_array(np.array([5], dtype=np.int64), "sl_ends")
    sl_axes = numpy_helper.from_array(np.array([1], dtype=np.int64), "sl_axes")

    nodes = [
        # Reshape to multi-head: (B, N, 8, 32)
        helper.make_node("Reshape", ["q", "q_shape"], ["q4"]),
        helper.make_node("Reshape", ["k", "kv_shape"], ["k4"]),
        helper.make_node("Reshape", ["v", "kv_shape"], ["v4"]),
        # Transpose to (B, 8, N, 32)
        helper.make_node("Transpose", ["q4"], ["qt"], perm=[0, 2, 1, 3]),
        helper.make_node("Transpose", ["k4"], ["kt"], perm=[0, 2, 3, 1]),
        helper.make_node("Transpose", ["v4"], ["vt"], perm=[0, 2, 1, 3]),
        # Attention: softmax(Q @ K^T / sqrt(d)) @ V
        helper.make_node("MatMul", ["qt", "kt"], ["qk"]),
        helper.make_node("Mul", ["qk", "scale"], ["qk_s"]),
        helper.make_node("Softmax", ["qk_s"], ["attn"], axis=-1),
        helper.make_node("MatMul", ["attn", "vt"], ["attn_out"]),
        # Transpose back: (B, 8, N, 32) → (B, N, 8, 32)
        helper.make_node("Transpose", ["attn_out"], ["attn_t"], perm=[0, 2, 1, 3]),
        # Reshape to (B, N, 256)
        helper.make_node("Reshape", ["attn_t", "out_shape"], ["merged"]),
        # Slice: extract first 5 tokens (mask/IoU token extraction)
        helper.make_node("Slice", ["merged", "sl_starts", "sl_ends", "sl_axes"],
                         ["output"]),
    ]

    return _build_model(
        "attention_test", nodes, [Q_in, K_in, V_in], [Y],
        [q_shape, kv_shape, out_shape, scale, sl_starts, sl_ends, sl_axes])


def make_layernorm():
    """ReduceMean + Sub + Pow + Sqrt + Div — decomposed LayerNorm.

    opset 11 decomposes nn.LayerNorm into these primitive ops.
    9 LayerNorm instances in the decoder's TwoWayTransformer.
    """
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 256])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 256])

    gamma = numpy_helper.from_array(np.ones(256, dtype=np.float32), "gamma")
    beta = numpy_helper.from_array(np.zeros(256, dtype=np.float32), "beta")
    eps = numpy_helper.from_array(np.array(1e-5, dtype=np.float32), "eps")
    two = numpy_helper.from_array(np.array(2.0, dtype=np.float32), "two")

    nodes = [
        # mean = ReduceMean(x, axis=-1)
        helper.make_node("ReduceMean", ["input"], ["mean"],
                         axes=[-1], keepdims=1),
        # centered = x - mean
        helper.make_node("Sub", ["input", "mean"], ["centered"]),
        # var = ReduceMean(centered^2, axis=-1)
        helper.make_node("Pow", ["centered", "two"], ["sq"]),
        helper.make_node("ReduceMean", ["sq"], ["var"],
                         axes=[-1], keepdims=1),
        # std = sqrt(var + eps)
        helper.make_node("Add", ["var", "eps"], ["var_eps"]),
        helper.make_node("Sqrt", ["var_eps"], ["std"]),
        # normalized = centered / std
        helper.make_node("Div", ["centered", "std"], ["normed"]),
        # output = gamma * normalized + beta
        helper.make_node("Mul", ["normed", "gamma"], ["scaled"]),
        helper.make_node("Add", ["scaled", "beta"], ["output"]),
    ]

    return _build_model("layernorm_test", nodes, [X], [Y],
                        [gamma, beta, eps, two])


def make_sincos_pe():
    """Sin + Cos + Concat — positional encoding.

    pe_layer._pe_encoding: coords → sin/cos features → concat.
    """
    X = helper.make_tensor_value_info("coords", TensorProto.FLOAT, [1, 5, 128])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 5, 256])

    freq = numpy_helper.from_array(
        np.random.randn(128).astype(np.float32) * 6.2832, "freq")

    nodes = [
        helper.make_node("Mul", ["coords", "freq"], ["scaled"]),
        helper.make_node("Sin", ["scaled"], ["sin_out"]),
        helper.make_node("Cos", ["scaled"], ["cos_out"]),
        helper.make_node("Concat", ["sin_out", "cos_out"], ["output"], axis=-1),
    ]

    return _build_model("sincos_pe_test", nodes, [X], [Y], [freq])


def make_gemm_tanh():
    """Gemm + Tanh — MLP / fully-connected layers.

    TwoWayAttentionBlock MLP: Linear → GELU(tanh) → Linear.
    GELUManual uses Tanh as its core nonlinearity.
    """
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [10, 256])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [10, 256])

    W1 = numpy_helper.from_array(
        np.random.randn(256, 1024).astype(np.float32) * 0.01, "w1")
    B1 = numpy_helper.from_array(np.zeros(1024, dtype=np.float32), "b1")
    W2 = numpy_helper.from_array(
        np.random.randn(1024, 256).astype(np.float32) * 0.01, "w2")
    B2 = numpy_helper.from_array(np.zeros(256, dtype=np.float32), "b2")

    nodes = [
        helper.make_node("Gemm", ["input", "w1", "b1"], ["h1"]),
        helper.make_node("Tanh", ["h1"], ["h2"]),
        helper.make_node("Gemm", ["h2", "w2", "b2"], ["output"]),
    ]

    return _build_model("gemm_tanh_test", nodes, [X], [Y], [W1, B1, W2, B2])


# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------

TEST_MODELS = [
    ("01_conv_transpose", "ConvTranspose, Relu",
     make_conv_transpose),
    ("02_bool_logic",     "Equal, Not, Cast",
     make_bool_logic),
    ("03_gather",         "Gather",
     make_gather),
    ("04_tile_expand",    "Tile, Expand, Unsqueeze",
     make_tile_expand),
    ("05_attention",      "MatMul, Softmax, Transpose, Reshape, Slice, Mul",
     make_attention),
    ("06_layernorm",      "ReduceMean, Sub, Pow, Sqrt, Div, Add, Mul",
     make_layernorm),
    ("07_sincos_pe",      "Sin, Cos, Concat, Mul",
     make_sincos_pe),
    ("08_gemm_tanh",      "Gemm, Tanh",
     make_gemm_tanh),
]


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate minimal ONNX models to diagnose NPU operator support.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output-dir", type=str, default="./npu_diag",
        help="Directory for generated .onnx files (default: ./npu_diag)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Generating diagnostic ONNX models (opset %d, int32-only)..." % OPSET)
    print("=" * 64)

    for filename, ops_desc, builder in TEST_MODELS:
        model = builder()
        path = os.path.join(args.output_dir, filename + ".onnx")
        onnx.save(model, path)

        actual_ops = _list_ops(model)
        print("  %-24s  %s" % (filename + ".onnx", ops_desc))
        print("  %-24s  actual ops: %s" % ("", ", ".join(actual_ops)))
        print()

    print("=" * 64)
    print("Generated %d models in %s/" % (len(TEST_MODELS), args.output_dir))
    print()
    print("Next steps:")
    print("  1. Compile each .onnx with your NPU toolchain")
    print("  2. Record PASS / FAIL for each model")
    print("  3. Failed models identify the problematic operator group(s)")
    print()
    print("If a group fails, you can further isolate by editing this script")
    print("to split that group into individual single-operator models.")


if __name__ == "__main__":
    main()
