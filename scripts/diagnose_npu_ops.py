#!/usr/bin/env python
"""
Diagnose NPU compilation failures by exporting 3 decoder sub-modules.

Splits the EdgeSAM decoder into 3 structural parts and exports each as
an independent ONNX model WITHOUT any NPU fixes.  Compile each with your
NPU toolchain; pass/fail pinpoints which subsystem is problematic.

  Part 1 — Prompt Encoding:
    point_coords + point_labels -> sparse_embedding

  Part 2 — Transformer:
    image_embeddings + sparse_embedding -> (hs, src)

  Part 3 — Mask Head:
    (hs, src) -> (scores, masks)

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
# Utilities
# ============================================================

def print_summary(path):
    """Print operator types, data types, and I/O shapes of an ONNX model."""
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
    """torch.onnx.export wrapper for PyTorch >= 2.6 compatibility."""
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
    label comparison, and per-label embedding selection.

    NPU-friendly: all label masks are computed via pure float arithmetic
    (Sub, Abs, Clamp) instead of Equal/Cast(bool), avoiding any boolean
    tensor ops that NPU compilers may not support.
    """

    def __init__(self, sam):
        super().__init__()
        self.pe_layer = sam.prompt_encoder.pe_layer
        self.point_embeddings = sam.prompt_encoder.point_embeddings
        self.not_a_point_embed = sam.prompt_encoder.not_a_point_embed
        self.num_point_embeddings = sam.prompt_encoder.num_point_embeddings
        self.img_size = sam.image_encoder.img_size

    @staticmethod
    def _float_eq(x, val):
        """NPU-safe equality mask using pure float ops (no Equal/Cast/Abs).

        For integer-valued floats: relu(1 - (x-val)^2) == 1.0 when x == val,
        0.0 otherwise.  Only emits Sub, Mul, Sub, Relu — no Abs or Clip.
        """
        diff = x - val
        return torch.relu(1.0 - diff * diff)

    def forward(self, point_coords, point_labels):
        point_coords = point_coords + 0.5
        point_coords = point_coords / self.img_size
        point_embedding = self.pe_layer._pe_encoding(point_coords)
        point_labels = point_labels.unsqueeze(-1).expand_as(point_embedding)

        # NPU-safe: pure float arithmetic masks (no Equal, Not, Cast)
        mask_neg1 = self._float_eq(point_labels, -1.0)
        mask_not_neg1 = 1.0 - mask_neg1

        point_embedding = point_embedding * mask_not_neg1
        point_embedding = point_embedding + \
            self.not_a_point_embed.weight * mask_neg1

        for i in range(self.num_point_embeddings):
            mask_i = self._float_eq(point_labels, float(i))
            point_embedding = point_embedding + \
                self.point_embeddings[i].weight * mask_i

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
# NPU-safe GELU (tanh approximation, avoids Erf op)
# ============================================================

class _GELUTanh(nn.Module):
    """GELU via tanh approximation. Avoids the Erf ONNX op."""

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(
            0.7978845608028654 * (x + 0.044715 * x * x * x)
        ))


def _replace_gelu(module):
    """Replace all nn.GELU in a module tree with _GELUTanh."""
    for attr in list(module._modules.keys()):
        if isinstance(module._modules[attr], nn.GELU):
            module._modules[attr] = _GELUTanh()
        else:
            _replace_gelu(module._modules[attr])


# ============================================================
# Part 3: Mask Head
# ============================================================

class Part3_MaskHead(nn.Module):
    """(hs, src) -> (scores, masks).

    Wraps output upscaling (ConvTranspose + LayerNorm2d + GELU),
    hypernetwork MLPs, mask generation via matrix multiply, and
    stability score computation.

    NPU fixes applied:
    - nn.GELU replaced with tanh approximation (avoids Erf op)
    - Stability score uses float arithmetic instead of bool ops
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

        # Replace nn.GELU -> tanh approximation (eliminates Erf op)
        _replace_gelu(self.output_upscaling)

    @staticmethod
    def _stability_score_npu(masks, mask_threshold, threshold_offset):
        """NPU-friendly stability score using pure float ops.

        Avoids both:
        - ReduceSum on bool (LUCI "Sum unsupported type")
        - Greater/Cast(bool) pattern (compilation failure)

        Uses sigmoid step approximation: sigmoid(k*(x - t)) ≈ step(x - t)
        with large k, producing only float Mul/Add/Sigmoid/ReduceSum ops.
        """
        k = 50.0  # steepness — large enough for near-binary output
        high = torch.sigmoid(k * (masks - (mask_threshold + threshold_offset)))
        low = torch.sigmoid(k * (masks - (mask_threshold - threshold_offset)))
        intersections = high.sum(-1).sum(-1)
        unions = low.sum(-1).sum(-1)
        return intersections / unions

    def forward(self, hs, src):
        # Extract tokens from transformer output
        # Use slice + reshape instead of integer index (Slice+Reshape instead of Gather)
        iou_token_out = hs[:, 0:1, :].reshape(1, 256)
        mask_tokens_out = hs[:, 1:(1 + self.num_mask_tokens), :]

        # Reshape src back to spatial: (B, 4096, 256) -> (B, 256, 64, 64)
        src = src.transpose(1, 2).reshape(1, 256, 64, 64)

        # Upscale mask embeddings: (B, 256, 64, 64) -> (B, 32, 256, 256)
        upscaled = self.output_upscaling(src)

        # Hypernetwork MLPs: each (B, 256) -> (B, 32)
        # Use slice + reshape instead of integer index (Slice+Reshape instead of Gather)
        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            token_i = mask_tokens_out[:, i:i+1, :].reshape(1, 256)
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](token_i)
            )
        hyper_in = torch.stack(hyper_in_list, dim=1)

        # Generate masks: (B, num_masks, 32) @ (B, 32, H*W) -> (B, num_masks, H, W)
        masks = (hyper_in @ upscaled.reshape(1, 32, 65536)).reshape(1, -1, 256, 256)

        # IoU prediction
        _iou_pred = self.iou_prediction_head(iou_token_out)

        # Stability score (NPU-safe: explicit float32 cast before sum)
        scores = self._stability_score_npu(
            masks, self.mask_threshold, self.stability_score_offset
        )

        return scores, masks


# ============================================================
# Export helper
# ============================================================

def export_part(module, dummy_inputs, input_names, output_names, path,
                desc, opset):
    print(f"\n{'=' * 60}")
    print(f"  {desc}")
    print(f"{'=' * 60}")

    _onnx_export(
        module, dummy_inputs, path,
        input_names=input_names,
        output_names=output_names,
        opset_version=opset,
        verbose=False,
    )
    print(f"  Exported to {path}")
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
        help="Number of prompt points (default: 5)",
    )
    parser.add_argument(
        "--opset", type=int, default=11,
        help="ONNX opset version (default: 11)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    N = args.num_points

    # Load model (no modifications applied)
    print("Loading model...")
    sam = sam_model_registry["edge_sam"](
        checkpoint=args.checkpoint, upsample_mode="bilinear"
    )
    sam.eval()

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
            args.opset,
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
            args.opset,
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
            args.opset,
        )

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Generated 3 models in {args.output_dir}/  (opset {args.opset})")
    print()
    print("  part1_prompt_encoding.onnx  — Sin/Cos PE, label comparison")
    print("  part2_transformer.onnx      — Attention, LayerNorm, MLP")
    print("  part3_mask_head.onnx        — ConvTranspose, upscaling, scores")
    print()
    print("Compile each with your NPU toolchain and report PASS/FAIL.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
