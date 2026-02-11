"""
Export EdgeSAM decoder to ONNX for NPU compilation (opset 11).

NPU compatibility fixes applied:
  1. nn.LayerNorm  → manual impl (avoids LayerNormalization op)
  2. nn.GELU       → tanh approx  (avoids Erf op)
  3. int64/int16   → int32        (NPU only supports int32)
  4. dynamic_axes  → removed      (NPU quantization needs static shapes)
  5. stability_score int16 → int32

Usage:
    # Export decoder (fixed 5 prompt points)
    python scripts/export_onnx_model_npu.py weights/edge_sam_3x.pth --decoder --use-stability-score

    # Export decoder with custom number of prompt points
    python scripts/export_onnx_model_npu.py weights/edge_sam_3x.pth --decoder --use-stability-score --num-points 2

    # Export encoder
    python scripts/export_onnx_model_npu.py weights/edge_sam_3x.pth

    # Check ops and data types in exported model
    python scripts/export_onnx_model_npu.py weights/edge_sam_3x.pth --decoder --check-ops-only
"""

import torch
import torch.nn as nn
import argparse
import numpy as np
from edge_sam import sam_model_registry
from edge_sam.utils.coreml import SamCoreMLModel
from edge_sam.utils.amg import calculate_stability_score


parser = argparse.ArgumentParser(
    description="Export EdgeSAM to ONNX for NPU (opset 11, no LayerNorm/GELU, int32 only)."
)
parser.add_argument(
    "checkpoint", type=str, help="The path to the EdgeSAM model checkpoint."
)
parser.add_argument(
    "--decoder", action="store_true",
    help="If set, export decoder, otherwise export encoder",
)
parser.add_argument(
    "--use-stability-score", action="store_true",
    help="Use stability score instead of IoU predictions for mask selection.",
)
parser.add_argument(
    "--opset", type=int, default=11,
    help="ONNX opset version (default: 11)",
)
parser.add_argument(
    "--num-points", type=int, default=5,
    help="Fixed number of prompt points for decoder export (default: 5)",
)
parser.add_argument(
    "--check-ops-only", action="store_true",
    help="Only export and print all ONNX op types and data types, do not save.",
)
parser.add_argument(
    "--output", type=str, default=None,
    help="Output ONNX file path. If not set, auto-generated from checkpoint path.",
)


# ============================================================
# NPU-compatible decoder wrapper (avoids OneHot op)
# ============================================================

class SamCoreMLModelNPU(SamCoreMLModel):
    """Override _embed_points to use Gather instead of OneHot.

    The original _embed_points compares point_labels against each label value
    in a loop: ``(point_labels == i) for i in range(4)``.  The ONNX exporter
    fuses these comparisons into an ``onnx.OneHot`` op, which many NPU
    compilers cannot legalize.

    This version builds a single embedding lookup table and uses
    ``torch.index_select`` (→ ONNX ``Gather``) to select embeddings,
    completely avoiding any OneHot pattern.
    """

    def _embed_points(self, point_coords: torch.Tensor, point_labels: torch.Tensor) -> torch.Tensor:
        point_coords = point_coords + 0.5
        point_coords = point_coords / self.img_size
        point_embedding = self.model.prompt_encoder.pe_layer._pe_encoding(point_coords)

        # Build lookup table: [num_labels, embed_dim]
        #   index 0 → not_a_point_embed  (label == -1)
        #   index 1 → point_embeddings[0] (label == 0)
        #   index 2 → point_embeddings[1] (label == 1)
        #   index 3 → point_embeddings[2] (label == 2)
        #   index 4 → point_embeddings[3] (label == 3)
        all_embeddings = torch.cat([
            self.model.prompt_encoder.not_a_point_embed.weight,
        ] + [
            pe.weight for pe in self.model.prompt_encoder.point_embeddings
        ], dim=0)  # [5, embed_dim]

        # Shift labels so -1 → 0, 0 → 1, …, 3 → 4
        indices = (point_labels + 1).to(torch.long)  # [B, N]

        # Gather embeddings (ONNX Gather op, no OneHot)
        flat_indices = indices.reshape(-1)
        selected = torch.index_select(all_embeddings, 0, flat_indices)
        selected = selected.reshape(indices.shape[0], indices.shape[1], -1)  # [B, N, embed_dim]

        # Zero out position encoding for not-a-point labels (index 0).
        # clamp(0,1) maps 0→0 and 1..4→1, producing a float mask via Clip op.
        keep_mask = indices.to(point_embedding.dtype).clamp(min=0.0, max=1.0).unsqueeze(-1)

        point_embedding = point_embedding * keep_mask + selected
        return point_embedding


# ============================================================
# Op replacements (PyTorch side, before export)
# ============================================================

class LayerNormManual(nn.Module):
    """Drop-in replacement for nn.LayerNorm using only basic arithmetic ops."""

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dims = list(range(-len(self.normalized_shape), 0))
        mean = x.mean(dim=dims, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=dims, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        if self.elementwise_affine:
            x = x * self.weight + self.bias
        return x


class GELUManual(nn.Module):
    """GELU approximation using tanh, avoiding the Erf op."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(
            0.7978845608028654 * (x + 0.044715 * x * x * x)
        ))


def replace_layernorm(model):
    """Recursively replace all nn.LayerNorm with LayerNormManual."""
    count = 0
    for name, module in model.named_children():
        if isinstance(module, nn.LayerNorm):
            manual = LayerNormManual(
                module.normalized_shape,
                eps=module.eps,
                elementwise_affine=module.elementwise_affine,
            )
            if module.elementwise_affine:
                manual.weight = module.weight
                manual.bias = module.bias
            setattr(model, name, manual)
            count += 1
        else:
            count += replace_layernorm(module)
    return count


def replace_gelu(model):
    """Recursively replace all nn.GELU with GELUManual (tanh approximation)."""
    count = 0
    for name, module in model.named_children():
        if isinstance(module, nn.GELU):
            setattr(model, name, GELUManual())
            count += 1
        else:
            count += replace_gelu(module)
    return count


def patch_stability_score():
    """Monkey-patch calculate_stability_score to use int32 instead of int16."""
    import edge_sam.utils.amg as amg

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

    amg.calculate_stability_score = calculate_stability_score_int32
    # Also patch the import in coreml module
    import edge_sam.utils.coreml as coreml_mod
    if hasattr(coreml_mod, 'calculate_stability_score'):
        coreml_mod.calculate_stability_score = calculate_stability_score_int32


# ============================================================
# ONNX post-processing: convert all int64/int16 → int32
# ============================================================

def simplify_onnx(onnx_path):
    """Use onnxruntime graph optimizer to fold constants and simplify.

    With static shapes, this folds away most Shape/Gather/Concat shape
    computations, leaving mainly float compute nodes.
    """
    import onnxruntime as ort
    import tempfile, shutil

    simplified_path = onnx_path + ".simplified"
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess_options.optimized_model_filepath = simplified_path
    # This creates the session AND saves the optimized model
    ort.InferenceSession(onnx_path, sess_options)
    shutil.move(simplified_path, onnx_path)
    print(f"  Graph simplified with onnxruntime optimizer")


def convert_onnx_int64_to_int32(onnx_path):
    """Aggressively convert ALL int64/int16 → int32 in ONNX model.

    Strategy: first simplify the graph (fold shape computations into
    constants), then convert every int64/int16 occurrence to int32
    with no exceptions. This includes shape inputs to Reshape/Expand/etc.

    The resulting model is NOT valid per ONNX spec (shape inputs should
    be int64), but NPU compilers have their own type system and only
    accept int32. No Cast bridge nodes are inserted.
    """
    import onnx
    from onnx import TensorProto, numpy_helper

    # Step 1: Simplify graph (fold constants, eliminate shape subgraph)
    simplify_onnx(onnx_path)

    model = onnx.load(onnx_path)
    INT64 = TensorProto.INT64
    INT16 = TensorProto.INT16
    INT32 = TensorProto.INT32
    converted = {"int64": 0, "int16": 0}

    def _needs_fix(t):
        return t in (INT64, INT16)

    def _kind(t):
        return "int64" if t == INT64 else "int16"

    # 1. Cast nodes: change target type
    for node in model.graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and _needs_fix(attr.i):
                    converted[_kind(attr.i)] += 1
                    attr.i = INT32

    # 2. ConstantOfShape: change output value type
    for node in model.graph.node:
        if node.op_type == "ConstantOfShape":
            for attr in node.attribute:
                if attr.name == "value" and attr.t and _needs_fix(attr.t.data_type):
                    converted[_kind(attr.t.data_type)] += 1
                    arr = numpy_helper.to_array(attr.t).astype(np.int32)
                    attr.t.CopyFrom(numpy_helper.from_array(arr))

    # 3. Constant nodes: change all int64/int16 data
    for node in model.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value" and attr.t and _needs_fix(attr.t.data_type):
                    converted[_kind(attr.t.data_type)] += 1
                    arr = numpy_helper.to_array(attr.t).astype(np.int32)
                    attr.t.CopyFrom(numpy_helper.from_array(arr))

    # 4. Initializers
    for init in model.graph.initializer:
        if _needs_fix(init.data_type):
            converted[_kind(init.data_type)] += 1
            arr = numpy_helper.to_array(init).astype(np.int32)
            new_init = numpy_helper.from_array(arr, name=init.name)
            init.CopyFrom(new_init)

    # 5. Value info, graph inputs, graph outputs
    for vi in model.graph.value_info:
        if _needs_fix(vi.type.tensor_type.elem_type):
            converted[_kind(vi.type.tensor_type.elem_type)] += 1
            vi.type.tensor_type.elem_type = INT32
    for inp in model.graph.input:
        if _needs_fix(inp.type.tensor_type.elem_type):
            converted[_kind(inp.type.tensor_type.elem_type)] += 1
            inp.type.tensor_type.elem_type = INT32
    for out in model.graph.output:
        if _needs_fix(out.type.tensor_type.elem_type):
            converted[_kind(out.type.tensor_type.elem_type)] += 1
            out.type.tensor_type.elem_type = INT32

    onnx.save(model, onnx_path)
    total = converted["int64"] + converted["int16"]
    print(f"  int64→int32: {converted['int64']}")
    print(f"  int16→int32: {converted['int16']}")
    print(f"  Total: {total} type conversions (no Cast bridges)")
    return total


# ============================================================
# ONNX inspection
# ============================================================

def check_onnx_ops(onnx_path):
    """Print all unique op types and data types in an ONNX model."""
    import onnx
    from onnx import TensorProto

    dtype_names = {v: k for k, v in TensorProto.DataType.items()}
    model = onnx.load(onnx_path)

    ops = set()
    for node in model.graph.node:
        ops.add(node.op_type)
    print(f"\nONNX model: {onnx_path}")
    print(f"Total nodes: {len(model.graph.node)}")
    print(f"Unique op types ({len(ops)}):")
    for op in sorted(ops):
        print(f"  - {op}")

    # Check for remaining int64/int16
    non_int32_types = set()
    for init in model.graph.initializer:
        if init.data_type in (TensorProto.INT64, TensorProto.INT16):
            non_int32_types.add(f"initializer:{init.name}={dtype_names.get(init.data_type, init.data_type)}")
    for node in model.graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i in (TensorProto.INT64, TensorProto.INT16):
                    non_int32_types.add(f"Cast→{dtype_names.get(attr.i, attr.i)}")
    if non_int32_types:
        print(f"\n  WARNING: remaining int64/int16 found:")
        for t in sorted(non_int32_types):
            print(f"    {t}")
    else:
        print(f"\n  OK: no int64/int16 remaining")

    return ops


# ============================================================
# Export functions
# ============================================================

def _onnx_export(*args, **kwargs):
    """Call torch.onnx.export with legacy exporter for PyTorch >= 2.6 compatibility."""
    major, minor = int(torch.__version__.split('.')[0]), int(torch.__version__.split('.')[1])
    if major > 2 or (major == 2 and minor >= 6):
        kwargs['dynamo'] = False
    torch.onnx.export(*args, **kwargs)


def export_encoder(sam, args):
    image_input = torch.randn(1, 3, 1024, 1024, dtype=torch.float)
    sam.forward = sam.forward_dummy_encoder

    out_path = args.output or args.checkpoint.replace('.pth', '_encoder_npu.onnx')
    _onnx_export(
        sam,
        image_input,
        out_path,
        input_names=["image"],
        output_names=["image_embeddings"],
        opset_version=args.opset,
        verbose=False,
    )
    print(f"Exported encoder to {out_path}")

    # Post-process: convert int64/int16 → int32
    print("\nConverting int64/int16 → int32...")
    convert_onnx_int64_to_int32(out_path)

    return out_path


def export_decoder(sam, args):
    # Patch stability score to use int32 instead of int16
    patch_stability_score()

    sam_decoder = SamCoreMLModelNPU(
        model=sam,
        use_stability_score=args.use_stability_score,
    )
    sam_decoder.eval()

    # Replace nn.LayerNorm and nn.GELU before export
    ln_count = replace_layernorm(sam_decoder)
    gelu_count = replace_gelu(sam_decoder)
    print(f"Replaced {ln_count} nn.LayerNorm -> LayerNormManual")
    print(f"Replaced {gelu_count} nn.GELU -> GELUManual (tanh approx)")

    embed_dim = sam.prompt_encoder.embed_dim
    embed_size = sam.prompt_encoder.image_embedding_size
    num_points = args.num_points

    # Fixed shapes — no dynamic axes for NPU quantization
    image_embeddings = torch.randn(1, embed_dim, *embed_size, dtype=torch.float)
    point_coords = torch.randint(low=0, high=1024, size=(1, num_points, 2), dtype=torch.float)
    point_labels = torch.randint(low=0, high=4, size=(1, num_points), dtype=torch.float)

    out_path = args.output or args.checkpoint.replace('.pth', '_decoder_npu.onnx')
    _onnx_export(
        sam_decoder,
        (image_embeddings, point_coords, point_labels),
        out_path,
        input_names=["image_embeddings", "point_coords", "point_labels"],
        output_names=["scores", "masks"],
        opset_version=args.opset,
        # No dynamic_axes — fixed shapes for NPU quantization
        verbose=False,
    )
    print(f"Exported decoder to {out_path} (fixed {num_points} points)")

    # Post-process: convert int64/int16 → int32
    print("\nConverting int64/int16 → int32...")
    convert_onnx_int64_to_int32(out_path)

    return out_path


if __name__ == "__main__":
    args = parser.parse_args()
    print("Loading model...")
    sam = sam_model_registry["edge_sam"](
        checkpoint=args.checkpoint, upsample_mode="bilinear"
    )
    sam.eval()

    if args.decoder:
        out_path = export_decoder(sam, args)
    else:
        out_path = export_encoder(sam, args)

    # Always check ops after export
    try:
        check_onnx_ops(out_path)
    except ImportError:
        print("\nInstall 'onnx' package to inspect op types: pip install onnx")

    if args.check_ops_only:
        import os
        os.remove(out_path)
        print(f"\n(--check-ops-only: removed {out_path})")
