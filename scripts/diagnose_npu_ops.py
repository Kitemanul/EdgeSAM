#!/usr/bin/env python
"""
Diagnose NPU compilation failures by exporting 3 decoder sub-modules.

Splits the EdgeSAM decoder into 3 structural parts and exports each as
an independent ONNX model with all NPU fixes applied (opset 11, GELU->tanh,
int16->int32, graph simplification, int64->int32).  Compile each with your
NPU toolchain; pass/fail pinpoints which subsystem is problematic.

  Part 1 — Prompt Encoding:
    point_coords + point_labels -> sparse_embedding
    Operators: MatMul, Sin, Cos, Concat, Equal, Not, Cast, Expand,
              Unsqueeze, Mul, Add, Sub

  Part 2 — Transformer:
    image_embeddings + sparse_embedding -> (hs, src)
    Operators: Gemm, MatMul, Softmax, Reshape, Transpose, Add, Mul,
              ReduceMean, Sub, Pow, Sqrt, Div (decomposed LayerNorm), Relu

  Part 3 — Mask Head:
    (hs, src) -> (scores, masks)
    Operators: ConvTranspose, Tanh, Gemm, Relu, MatMul, Reshape,
              ReduceMean, Pow, Sqrt, Div (LayerNorm2d), Cast, Slice

Usage:
    python scripts/diagnose_npu_ops.py weights/edge_sam_3x.pth
    python scripts/diagnose_npu_ops.py weights/edge_sam_3x.pth --output-dir ./npu_diag
    python scripts/diagnose_npu_ops.py weights/edge_sam_3x.pth --num-points 2
"""

import argparse
import os
from collections import Counter

import torch
import torch.nn as nn

from edge_sam import sam_model_registry


# ============================================================
# NPU fixes (reused from export_onnx_model_npu.py)
# ============================================================

class GELUManual(nn.Module):
    """GELU using tanh approximation (eliminates Erf op)."""

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(
            0.7978845608028654 * (x + 0.044715 * x * x * x)
        ))


def replace_gelu(model):
    count = 0
    for name, module in model.named_modules():
        for attr_name in list(module._modules.keys()):
            if isinstance(module._modules[attr_name], nn.GELU):
                module._modules[attr_name] = GELUManual()
                count += 1
    return count


def simplify_onnx(path):
    try:
        import onnxruntime as ort
    except ImportError:
        print("    WARNING: onnxruntime not installed, skipping graph simplification")
        return
    tmp = path + ".tmp"
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    opts.optimized_model_filepath = tmp
    ort.InferenceSession(path, opts)
    os.replace(tmp, path)
    print("    Graph simplification (constant folding) done")


def convert_int64_to_int32(path):
    import onnx
    import numpy as np
    from onnx import TensorProto

    model = onnx.load(path)
    CONV = {TensorProto.INT64, TensorProto.INT16}
    count = 0

    def _dt(d):
        return np.int64 if d == TensorProto.INT64 else np.int16

    for node in model.graph.node:
        if node.op_type == "Cast":
            for a in node.attribute:
                if a.name == "to" and a.i in CONV:
                    a.i = TensorProto.INT32
                    count += 1
        if node.op_type in ("Constant", "ConstantOfShape"):
            for a in node.attribute:
                if a.name == "value" and a.t.data_type in CONV:
                    raw = np.frombuffer(a.t.raw_data, dtype=_dt(a.t.data_type))
                    a.t.raw_data = raw.astype(np.int32).tobytes()
                    a.t.data_type = TensorProto.INT32
                    count += 1
    for init in model.graph.initializer:
        if init.data_type in CONV:
            raw = np.frombuffer(init.raw_data, dtype=_dt(init.data_type))
            init.raw_data = raw.astype(np.int32).tobytes()
            init.data_type = TensorProto.INT32
            count += 1
    for coll in (model.graph.value_info, model.graph.input, model.graph.output):
        for vi in coll:
            if vi.type.tensor_type.elem_type in CONV:
                vi.type.tensor_type.elem_type = TensorProto.INT32
                count += 1

    del model.graph.value_info[:]
    try:
        from onnx import shape_inference
        model = shape_inference.infer_shapes(model)
    except Exception:
        pass

    onnx.save(model, path)
    print(f"    int64/int16 -> int32: {count} conversions")


def print_summary(path):
    import onnx
    dtype_names = {
        1: "FLOAT", 2: "UINT8", 3: "INT8", 5: "INT16",
        6: "INT32", 7: "INT64", 9: "BOOL", 10: "FLOAT16",
    }
    model = onnx.load(path)
    ops = Counter(n.op_type for n in model.graph.node)
    dtypes = set()
    for init in model.graph.initializer:
        dtypes.add(dtype_names.get(init.data_type, "?"))
    for inp in model.graph.input:
        dtypes.add(dtype_names.get(inp.type.tensor_type.elem_type, "?"))

    print(f"    Nodes: {len(model.graph.node)},  Op types: {len(ops)}")
    print(f"    Ops: {', '.join(sorted(ops))}")
    print(f"    Dtypes: {', '.join(sorted(dtypes))}")

    for inp in model.graph.input:
        shape = [d.dim_value or d.dim_param
                 for d in inp.type.tensor_type.shape.dim]
        dt = dtype_names.get(inp.type.tensor_type.elem_type, "?")
        print(f"    Input  {inp.name}: {dt} {shape}")
    for out in model.graph.output:
        shape = [d.dim_value or d.dim_param
                 for d in out.type.tensor_type.shape.dim]
        dt = dtype_names.get(out.type.tensor_type.elem_type, "?")
        print(f"    Output {out.name}: {dt} {shape}")


def _onnx_export(*args, **kwargs):
    parts = torch.__version__.split('.')
    major, minor = int(parts[0]), int(parts[1])
    if major > 2 or (major == 2 and minor >= 6):
        kwargs['dynamo'] = False
    torch.onnx.export(*args, **kwargs)


# ============================================================
# Part 1: Prompt Encoding
# ============================================================

class Part1_PromptEncoding(nn.Module):
    """point_coords + point_labels -> sparse_embedding.

    Replicates SamCoreMLModel._embed_points: positional encoding (sin/cos),
    label comparison (equal/not/cast), and per-label embedding selection.
    """

    def __init__(self, sam):
        super().__init__()
        self.pe_layer = sam.prompt_encoder.pe_layer
        self.point_embeddings = sam.prompt_encoder.point_embeddings
        self.not_a_point_embed = sam.prompt_encoder.not_a_point_embed
        self.num_point_embeddings = sam.prompt_encoder.num_point_embeddings
        self.img_size = sam.image_encoder.img_size

    def forward(self, point_coords, point_labels):
        point_coords = point_coords + 0.5
        point_coords = point_coords / self.img_size
        point_embedding = self.pe_layer._pe_encoding(point_coords)
        point_labels = point_labels.unsqueeze(-1).expand_as(point_embedding)

        point_embedding = point_embedding * (point_labels != -1)
        point_embedding = point_embedding + \
            self.not_a_point_embed.weight * (point_labels == -1)

        for i in range(self.num_point_embeddings):
            point_embedding = point_embedding + \
                self.point_embeddings[i].weight * (point_labels == i)

        return point_embedding


# ============================================================
# Part 2: Transformer
# ============================================================

class Part2_Transformer(nn.Module):
    """image_embeddings + sparse_embedding -> (hs, src).

    Wraps token preparation (iou_token/mask_tokens concatenation) and the
    full TwoWayTransformer (2x attention blocks + final attention + LayerNorm).
    Dense embedding and image PE are baked in as constant buffers.
    """

    def __init__(self, sam):
        super().__init__()
        self.transformer = sam.mask_decoder.transformer
        self.iou_token = sam.mask_decoder.iou_token
        self.mask_tokens = sam.mask_decoder.mask_tokens
        # These become ONNX constants (no dependency on inputs)
        self.register_buffer(
            'dense_embedding',
            sam.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).clone()
        )
        self.register_buffer(
            'image_pe',
            sam.prompt_encoder.get_dense_pe().clone()
        )

    def forward(self, image_embeddings, sparse_embedding):
        # Token preparation (from MaskDecoder.predict_masks)
        output_tokens = torch.cat(
            [self.iou_token.weight, self.mask_tokens.weight], dim=0
        )
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_embedding.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, sparse_embedding), dim=1)

        src = image_embeddings + self.dense_embedding

        # Run the two-way transformer
        hs, src = self.transformer(src, self.image_pe, tokens)
        return hs, src


# ============================================================
# Part 3: Mask Head
# ============================================================

class Part3_MaskHead(nn.Module):
    """(hs, src) -> (scores, masks).

    Wraps output upscaling (ConvTranspose + LayerNorm2d + GELUManual),
    hypernetwork MLPs, mask generation via matrix multiply, and
    stability score computation.
    """

    def __init__(self, sam):
        super().__init__()
        md = sam.mask_decoder
        self.output_upscaling = md.output_upscaling
        self.output_hypernetworks_mlps = md.output_hypernetworks_mlps
        self.iou_prediction_head = md.iou_prediction_head
        self.num_mask_tokens = md.num_mask_tokens
        self.mask_threshold = sam.mask_threshold
        self.stability_score_offset = 1.0

    def forward(self, hs, src):
        # Extract tokens from transformer output
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1:(1 + self.num_mask_tokens), :]

        # Reshape src back to spatial: (B, 4096, 256) -> (B, 256, 64, 64)
        b = src.shape[0]
        src = src.transpose(1, 2).view(b, 256, 64, 64)

        # Upscale mask embeddings: (B, 256, 64, 64) -> (B, 32, 256, 256)
        upscaled = self.output_upscaling(src)

        # Hypernetwork MLPs: each (B, 256) -> (B, 32)
        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            )
        hyper_in = torch.stack(hyper_in_list, dim=1)

        # Generate masks: (B, num_masks, 32) @ (B, 32, H*W) -> (B, num_masks, H, W)
        b, c, h, w = upscaled.shape
        masks = (hyper_in @ upscaled.view(b, c, h * w)).view(b, -1, h, w)

        # IoU prediction (unused in stability-score mode, but still in graph)
        _iou_pred = self.iou_prediction_head(iou_token_out)

        # Stability score (int32, matching NPU export pipeline)
        intersections = (
            (masks > (self.mask_threshold + self.stability_score_offset))
            .sum(-1, dtype=torch.int32)
            .sum(-1, dtype=torch.int32)
        )
        unions = (
            (masks > (self.mask_threshold - self.stability_score_offset))
            .sum(-1, dtype=torch.int32)
            .sum(-1, dtype=torch.int32)
        )
        scores = intersections / unions

        return scores, masks


# ============================================================
# Export helper
# ============================================================

def export_part(module, dummy_inputs, input_names, output_names, path, desc):
    print(f"\n{'=' * 60}")
    print(f"  {desc}")
    print(f"{'=' * 60}")

    _onnx_export(
        module, dummy_inputs, path,
        input_names=input_names,
        output_names=output_names,
        opset_version=11,
        verbose=False,
    )
    print(f"  Exported to {path}")

    simplify_onnx(path)
    convert_int64_to_int32(path)
    print_summary(path)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Export 3 decoder sub-modules for NPU compilation diagnosis."
    )
    parser.add_argument(
        "checkpoint", type=str,
        help="Path to EdgeSAM checkpoint (.pth)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./npu_diag",
        help="Directory for generated .onnx files (default: ./npu_diag)",
    )
    parser.add_argument(
        "--num-points", type=int, default=5,
        help="Number of prompt points (default: 5, must match decoder export)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    N = args.num_points

    # Load model
    print("Loading model...")
    sam = sam_model_registry["edge_sam"](
        checkpoint=args.checkpoint, upsample_mode="bilinear"
    )
    sam.eval()

    # Apply NPU fixes to the model
    n_gelu = replace_gelu(sam)
    print(f"Replaced {n_gelu} nn.GELU -> GELUManual (tanh approximation)")

    num_mask_tokens = sam.mask_decoder.num_mask_tokens  # 4
    num_output_tokens = 1 + num_mask_tokens  # iou + masks = 5
    total_tokens = num_output_tokens + N  # 5 + N = 10

    # --- Part 1: Prompt Encoding ---
    part1 = Part1_PromptEncoding(sam).eval()
    with torch.no_grad():
        export_part(
            part1,
            (torch.randint(0, 1024, (1, N, 2), dtype=torch.float),
             torch.randint(0, 4, (1, N), dtype=torch.float)),
            ["point_coords", "point_labels"],
            ["sparse_embedding"],
            os.path.join(args.output_dir, "part1_prompt_encoding.onnx"),
            "Part 1: Prompt Encoding",
        )

    # --- Part 2: Transformer ---
    part2 = Part2_Transformer(sam).eval()
    with torch.no_grad():
        export_part(
            part2,
            (torch.randn(1, 256, 64, 64),
             torch.randn(1, N, 256)),
            ["image_embeddings", "sparse_embedding"],
            ["hs", "src"],
            os.path.join(args.output_dir, "part2_transformer.onnx"),
            "Part 2: Transformer (2x TwoWayAttentionBlock + final attn)",
        )

    # --- Part 3: Mask Head ---
    part3 = Part3_MaskHead(sam).eval()
    with torch.no_grad():
        export_part(
            part3,
            (torch.randn(1, total_tokens, 256),
             torch.randn(1, 4096, 256)),
            ["hs", "src"],
            ["scores", "masks"],
            os.path.join(args.output_dir, "part3_mask_head.onnx"),
            "Part 3: Mask Head (upscaling + MLPs + stability score)",
        )

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Generated 3 models in {args.output_dir}/")
    print()
    print("  part1_prompt_encoding.onnx  — Sin/Cos PE, label comparison")
    print("  part2_transformer.onnx      — Attention, LayerNorm, MLP")
    print("  part3_mask_head.onnx        — ConvTranspose, upscaling, scores")
    print()
    print("Compile each with your NPU toolchain and report PASS/FAIL.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
