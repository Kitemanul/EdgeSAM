"""
Export EdgeSAM decoder to NPU-compatible ONNX (opset 11).

Compared to export_onnx_model.py, this script applies four NPU-compatibility
fixes to the decoder:

  1. nn.GELU -> tanh approximation (eliminates unsupported Erf op)
  2. stability_score int16 -> int32 (NPU only supports int32)
  3. onnxruntime graph simplification (constant folding eliminates Shape subgraphs)
  4. Comprehensive int64/int16 -> int32 conversion

The encoder (RepViT) is pure CNN with only FLOAT types and basic ops,
so it doesn't need most of these fixes but gets dtype conversion for safety.

Usage:
    # Export encoder
    python scripts/export_onnx_model_npu.py weights/edge_sam_3x.pth

    # Export decoder (all NPU fixes applied)
    python scripts/export_onnx_model_npu.py weights/edge_sam_3x.pth \\
        --decoder --use-stability-score

    # Custom point count (static shape)
    python scripts/export_onnx_model_npu.py weights/edge_sam_3x.pth \\
        --decoder --use-stability-score --num-points 2

    # Inspect operators and dtypes
    python scripts/export_onnx_model_npu.py weights/edge_sam_3x.pth \\
        --decoder --use-stability-score --check-ops-only
"""

import os
import torch
import torch.nn as nn
import argparse
from edge_sam import sam_model_registry
from edge_sam.utils.coreml import SamCoreMLModel


parser = argparse.ArgumentParser(
    description="Export EdgeSAM to ONNX for NPU (opset 11)."
)

parser.add_argument(
    "checkpoint", type=str, help="The path to the EdgeSAM model checkpoint."
)

parser.add_argument(
    "--use-stability-score",
    action="store_true",
    help=(
        "Replaces the model's predicted mask quality score with the stability "
        "score calculated on the low resolution masks using an offset of 1.0. "
    ),
)

parser.add_argument(
    "--decoder",
    action="store_true",
    help="If set, export decoder, otherwise export encoder",
)

parser.add_argument(
    "--num-points",
    type=int,
    default=5,
    help="Number of prompt points for decoder export (default: 5). "
         "Decoder is exported with static shapes for NPU compatibility.",
)

parser.add_argument(
    "--check-ops-only",
    action="store_true",
    help="Export, print operator/dtype summary, then delete the file.",
)


# ============================================================
# Fix 1: nn.GELU -> tanh approximation (eliminates Erf op)
# ============================================================

class GELUManual(nn.Module):
    """GELU using tanh approximation (Hendrycks 2016).

    Avoids the Erf ONNX op which is unsupported by most NPU compilers.
    GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    Max absolute error vs exact GELU: < 0.004
    """

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(
            0.7978845608028654 * (x + 0.044715 * x * x * x)
        ))


def replace_gelu(model):
    """Replace all nn.GELU modules in the model with GELUManual (tanh)."""
    count = 0
    for name, module in model.named_modules():
        for attr_name in list(module._modules.keys()):
            if isinstance(module._modules[attr_name], nn.GELU):
                module._modules[attr_name] = GELUManual()
                count += 1
    print(f"  [Fix 1] Replaced {count} nn.GELU -> GELUManual (tanh approximation)")


# ============================================================
# Fix 2: stability_score int16 -> int32
# ============================================================

def patch_stability_score():
    """Monkey-patch calculate_stability_score to use int32 instead of int16.

    The original uses torch.int16 for intermediate sums to save memory,
    but NPU compilers only support int32.
    """
    import edge_sam.utils.amg as amg_module
    import edge_sam.utils.coreml as coreml_module

    def calculate_stability_score_int32(masks, mask_threshold, threshold_offset):
        intersections = (
            (masks > (mask_threshold + threshold_offset))
            .sum(-1, dtype=torch.int32)
            .sum(-1, dtype=torch.int32)
        )
        unions = (
            (masks > (mask_threshold - threshold_offset))
            .sum(-1, dtype=torch.int32)
            .sum(-1, dtype=torch.int32)
        )
        return intersections / unions

    amg_module.calculate_stability_score = calculate_stability_score_int32
    coreml_module.calculate_stability_score = calculate_stability_score_int32
    print("  [Fix 2] Patched calculate_stability_score: int16 -> int32")


# ============================================================
# Fix 3: onnxruntime graph simplification (constant folding)
# ============================================================

def simplify_onnx(onnx_path):
    """Use onnxruntime ORT_ENABLE_BASIC to fold constants.

    The PyTorch ONNX exporter generates Shape -> Gather -> Concat -> Reshape
    subgraphs even for static shapes. These use int64 (ONNX spec requirement).
    Constant folding collapses them into simple Constant nodes, making the
    subsequent int64->int32 conversion straightforward.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("  [Fix 3] WARNING: onnxruntime not installed, skipping graph simplification")
        print("           Install with: pip install onnxruntime")
        return

    tmp_path = onnx_path + ".tmp"
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess_options.optimized_model_filepath = tmp_path
    ort.InferenceSession(onnx_path, sess_options)
    os.replace(tmp_path, onnx_path)
    print("  [Fix 3] Graph simplification (constant folding) complete")


# ============================================================
# Fix 4: Comprehensive int64/int16 -> int32 conversion
# ============================================================

def convert_all_int64_to_int32(onnx_path):
    """Convert ALL int64/int16 to int32 in every location of the ONNX graph.

    Converts in 7 locations:
      1. Cast node target types
      2. ConstantOfShape value dtypes
      3. Constant node data
      4. Initializer weights
      5. value_info type declarations
      6. Graph input types
      7. Graph output types

    WARNING: The resulting model does NOT comply with ONNX spec (which
    requires int64 for Reshape/Expand/Tile shape inputs) and will NOT
    load in onnxruntime. It is intended for NPU compilers that only
    accept int32.
    """
    try:
        import onnx
        import numpy as np
        from onnx import TensorProto, shape_inference
    except ImportError:
        print("  [Fix 4] WARNING: onnx package not installed, skipping dtype conversion")
        print("           Install with: pip install onnx")
        return

    model = onnx.load(onnx_path)
    INT64 = TensorProto.INT64
    INT16 = TensorProto.INT16
    INT32 = TensorProto.INT32
    convertible = {INT64, INT16}
    count = 0

    def _np_dtype(dt):
        return np.int64 if dt == INT64 else np.int16

    # 1. Cast nodes: change target type
    for node in model.graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i in convertible:
                    attr.i = INT32
                    count += 1

    # 2. ConstantOfShape nodes: change value dtype
    for node in model.graph.node:
        if node.op_type == "ConstantOfShape":
            for attr in node.attribute:
                if attr.name == "value" and attr.t.data_type in convertible:
                    raw = np.frombuffer(attr.t.raw_data, dtype=_np_dtype(attr.t.data_type))
                    attr.t.raw_data = raw.astype(np.int32).tobytes()
                    attr.t.data_type = INT32
                    count += 1

    # 3. Constant nodes: change data dtype
    for node in model.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value" and attr.t.data_type in convertible:
                    raw = np.frombuffer(attr.t.raw_data, dtype=_np_dtype(attr.t.data_type))
                    attr.t.raw_data = raw.astype(np.int32).tobytes()
                    attr.t.data_type = INT32
                    count += 1

    # 4. Initializers
    for init in model.graph.initializer:
        if init.data_type in convertible:
            raw = np.frombuffer(init.raw_data, dtype=_np_dtype(init.data_type))
            init.raw_data = raw.astype(np.int32).tobytes()
            init.data_type = INT32
            count += 1

    # 5. value_info type declarations
    for vi in model.graph.value_info:
        t = vi.type.tensor_type
        if t.elem_type in convertible:
            t.elem_type = INT32
            count += 1

    # 6. Graph inputs
    for inp in model.graph.input:
        t = inp.type.tensor_type
        if t.elem_type in convertible:
            t.elem_type = INT32
            count += 1

    # 7. Graph outputs
    for out in model.graph.output:
        t = out.type.tensor_type
        if t.elem_type in convertible:
            t.elem_type = INT32
            count += 1

    # Clear stale value_info and try to re-infer shapes
    del model.graph.value_info[:]
    try:
        model = shape_inference.infer_shapes(model)
    except Exception:
        pass  # Shape inference may fail on non-spec-compliant int32 model

    onnx.save(model, onnx_path)
    print(f"  [Fix 4] Converted {count} int64/int16 -> int32 occurrences")


# ============================================================
# Diagnostic: print ONNX model summary
# ============================================================

def print_onnx_summary(onnx_path):
    """Print operator types, counts, and data types found in the model."""
    try:
        import onnx
        from collections import Counter
    except ImportError:
        print("Cannot print summary: onnx package not installed")
        return

    model = onnx.load(onnx_path)
    dtype_names = {
        1: "FLOAT", 2: "UINT8", 3: "INT8", 5: "INT16",
        6: "INT32", 7: "INT64", 9: "BOOL", 10: "FLOAT16",
    }

    op_counts = Counter(node.op_type for node in model.graph.node)
    print(f"\n{'=' * 60}")
    print(f"ONNX Model Summary: {os.path.basename(onnx_path)}")
    print(f"{'=' * 60}")
    print(f"Total nodes: {len(model.graph.node)}")
    print(f"Operator types: {len(op_counts)}")
    print(f"\nOperators:")
    for op, cnt in sorted(op_counts.items()):
        print(f"  {op}: {cnt}")

    # Collect data types from all sources
    dtypes = set()
    for node in model.graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to":
                    dtypes.add(dtype_names.get(attr.i, f"?({attr.i})"))
        for attr in node.attribute:
            if attr.name == "value" and hasattr(attr, 't') and attr.t.data_type:
                dtypes.add(dtype_names.get(attr.t.data_type, f"?({attr.t.data_type})"))
    for init in model.graph.initializer:
        dtypes.add(dtype_names.get(init.data_type, f"?({init.data_type})"))

    print(f"\nData types: {', '.join(sorted(dtypes))}")

    # Inputs/outputs
    print(f"\nInputs:")
    for inp in model.graph.input:
        shape = [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]
        dt = dtype_names.get(inp.type.tensor_type.elem_type, "?")
        print(f"  {inp.name}: {dt} {shape}")
    print(f"Outputs:")
    for out in model.graph.output:
        shape = [d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim]
        dt = dtype_names.get(out.type.tensor_type.elem_type, "?")
        print(f"  {out.name}: {dt} {shape}")

    # Check for known NPU compatibility issues
    issues = []
    if "INT64" in dtypes:
        issues.append("INT64 present (most NPUs only support INT32)")
    if "INT16" in dtypes:
        issues.append("INT16 present (most NPUs only support INT32)")
    if "Erf" in op_counts:
        issues.append("Erf operator present (from nn.GELU, not supported by most NPUs)")
    if "LayerNormalization" in op_counts:
        issues.append("LayerNormalization present (not supported by some NPUs)")

    if issues:
        print(f"\nPotential NPU issues:")
        for issue in issues:
            print(f"  WARNING: {issue}")
    else:
        print(f"\nNo known NPU compatibility issues detected.")
    print(f"{'=' * 60}")


# ============================================================
# PyTorch >= 2.6 compatibility
# ============================================================

def _onnx_export(*args, **kwargs):
    """torch.onnx.export wrapper that uses legacy exporter on PyTorch >= 2.6.

    PyTorch 2.6+ defaults to dynamo-based ONNX exporter which doesn't
    support ScriptModule and has a minimum opset of 18.
    """
    parts = torch.__version__.split('.')
    major, minor = int(parts[0]), int(parts[1])
    if major > 2 or (major == 2 and minor >= 6):
        kwargs['dynamo'] = False
    torch.onnx.export(*args, **kwargs)


# ============================================================
# Export functions
# ============================================================

def export_encoder_to_onnx(sam, args):
    image_input = torch.randn(1, 3, 1024, 1024, dtype=torch.float)
    sam.forward = sam.forward_dummy_encoder

    traced_model = torch.jit.trace(sam, image_input)

    input_names = ["image"]
    output_names = ["image_embeddings"]

    onnx_path = args.checkpoint.replace('.pth', '_encoder_npu.onnx')
    _onnx_export(
        traced_model,
        image_input,
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=11,
        verbose=False
    )

    print(f"Exported encoder to {onnx_path}")
    convert_all_int64_to_int32(onnx_path)

    if args.check_ops_only:
        print_onnx_summary(onnx_path)
        os.remove(onnx_path)
        print(f"\n(Removed {onnx_path} -- check-ops-only mode)")
    else:
        print_onnx_summary(onnx_path)


def export_decoder_to_onnx(sam, args):
    print("Applying NPU compatibility fixes...")

    # Fix 1: Replace GELU with tanh approximation (eliminates Erf op)
    replace_gelu(sam)

    # Fix 2: Patch stability_score to use int32 instead of int16
    if args.use_stability_score:
        patch_stability_score()

    sam_decoder = SamCoreMLModel(
        model=sam,
        use_stability_score=args.use_stability_score
    )
    sam_decoder.eval()

    embed_dim = sam.prompt_encoder.embed_dim
    embed_size = sam.prompt_encoder.image_embedding_size
    num_points = args.num_points

    image_embeddings = torch.randn(1, embed_dim, *embed_size, dtype=torch.float)
    point_coords = torch.randint(low=0, high=1024, size=(1, num_points, 2), dtype=torch.float)
    point_labels = torch.randint(low=0, high=4, size=(1, num_points), dtype=torch.float)

    input_names = ["image_embeddings", "point_coords", "point_labels"]
    output_names = ["scores", "masks"]

    # Static shapes (no dynamic_axes) - NPU compilers need fixed shapes
    # for correct quantization parameter computation
    onnx_path = args.checkpoint.replace('.pth', '_decoder_npu.onnx')
    _onnx_export(
        sam_decoder,
        (image_embeddings, point_coords, point_labels),
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=11,
        verbose=False
    )

    print(f"Exported decoder to {onnx_path}")

    # Fix 3: Graph simplification (constant folding to eliminate Shape subgraphs)
    simplify_onnx(onnx_path)

    # Fix 4: Comprehensive int64/int16 -> int32 conversion
    convert_all_int64_to_int32(onnx_path)

    if args.check_ops_only:
        print_onnx_summary(onnx_path)
        os.remove(onnx_path)
        print(f"\n(Removed {onnx_path} -- check-ops-only mode)")
    else:
        print_onnx_summary(onnx_path)
        print(f"\nDone! NPU-compatible decoder: {onnx_path}")
        print(f"  Input shapes: image_embeddings [1,{embed_dim},{embed_size[0]},{embed_size[1]}], "
              f"point_coords [1,{num_points},2], point_labels [1,{num_points}]")


if __name__ == "__main__":
    args = parser.parse_args()
    print("Loading model...")
    sam = sam_model_registry["edge_sam"](checkpoint=args.checkpoint, upsample_mode="bilinear")
    sam.eval()

    if args.decoder:
        export_decoder_to_onnx(sam, args)
    else:
        export_encoder_to_onnx(sam, args)
