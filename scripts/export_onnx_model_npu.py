"""
Export EdgeSAM encoder and decoder to NPU-compatible ONNX (opset 11).

The decoder applies 5 NPU compatibility fixes to eliminate unsupported ops:
  Fix 1: Sin/Cos  → PE pre-computed on CPU (positional encoding)
  Fix 2: Equal/Cast/Abs → float arithmetic for label comparison
  Fix 3: Gather  → Slice + Reshape for tensor indexing
  Fix 4: Erf     → GELU replaced with tanh approximation
  Fix 5: bool ReduceSum → sigmoid step for stability score

Plus 2 post-processing steps on the exported ONNX file:
  Fix 6: ONNX graph simplification (constant folding)
  Fix 7: int64/int16 → int32 conversion

IMPORTANT: The decoder accepts pre-computed positional encoding
(point_embedding_pe) instead of raw point coordinates. The PE must
be computed on CPU before invoking the ONNX model:

    from scripts.export_onnx_model_npu import compute_point_pe
    pe = compute_point_pe(sam, point_coords)

Usage:
    # Export encoder
    python scripts/export_onnx_model_npu.py weights/edge_sam_3x.pth

    # Export decoder (recommended: with stability score)
    python scripts/export_onnx_model_npu.py weights/edge_sam_3x.pth \\
        --decoder --use-stability-score

    # Custom point count
    python scripts/export_onnx_model_npu.py weights/edge_sam_3x.pth \\
        --decoder --use-stability-score --num-points 2

    # Inspect operators and dtypes only
    python scripts/export_onnx_model_npu.py weights/edge_sam_3x.pth \\
        --decoder --use-stability-score --check-ops-only
"""

import argparse
import os
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

from edge_sam import sam_model_registry


# ============================================================
# CPU-side PE computation (moved out of ONNX to avoid Sin/Cos)
# ============================================================

def compute_point_pe(sam, point_coords):
    """Compute positional encoding for point coordinates on CPU.

    Must be called before running the NPU decoder ONNX model.
    Replicates the sin/cos PE from PositionEmbeddingRandom._pe_encoding.

    Args:
        sam: EdgeSAM model (needs sam.image_encoder.img_size
             and sam.prompt_encoder.pe_layer)
        point_coords: [1, N, 2] float tensor of point coordinates
                      (in model pixel space, after apply_coords)

    Returns:
        point_embedding_pe: [1, N, 256] float tensor
    """
    with torch.no_grad():
        coords = point_coords + 0.5
        coords = coords / sam.image_encoder.img_size
        pe = sam.prompt_encoder.pe_layer._pe_encoding(coords)
    return pe


# ============================================================
# NPU-safe GELU (tanh approximation, avoids Erf op)
# ============================================================

class _GELUTanh(nn.Module):
    """GELU via tanh approximation.  Avoids the Erf ONNX op."""

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(
            0.7978845608028654 * (x + 0.044715 * x * x * x)
        ))


def _replace_gelu(module):
    """Replace all nn.GELU in a module tree with _GELUTanh."""
    count = 0
    for attr in list(module._modules.keys()):
        if isinstance(module._modules[attr], nn.GELU):
            module._modules[attr] = _GELUTanh()
            count += 1
        else:
            count += _replace_gelu(module._modules[attr])
    return count


# ============================================================
# Merged NPU-Safe Decoder
# ============================================================

class NpuSafeDecoder(nn.Module):
    """EdgeSAM decoder with all NPU compatibility fixes applied.

    Merges 3 functional stages into a single module:
      Stage 1 — Label Embedding: select per-label embeddings
      Stage 2 — Transformer: two-way attention between tokens and image
      Stage 3 — Mask Head: upscale, hypernetwork MLPs, score

    Inputs:
        image_embeddings:  [1, 256, 64, 64]  — from encoder
        point_embedding_pe: [1, N, 256]       — PE pre-computed on CPU
        point_labels:      [1, N]             — float labels (-1,0,1,2,3)

    Outputs:
        scores: [1, num_mask_tokens]                — mask quality
        masks:  [1, num_mask_tokens, 256, 256]      — predicted masks
    """

    def __init__(self, sam, use_stability_score=True):
        super().__init__()

        # --- Stage 1: Label Embedding ---
        self.point_embeddings = sam.prompt_encoder.point_embeddings
        self.not_a_point_embed = sam.prompt_encoder.not_a_point_embed
        self.num_point_embeddings = sam.prompt_encoder.num_point_embeddings

        # --- Stage 2: Transformer ---
        self.transformer = sam.mask_decoder.transformer
        self.iou_token = sam.mask_decoder.iou_token
        self.mask_tokens = sam.mask_decoder.mask_tokens
        self.register_buffer(
            'dense_embedding',
            sam.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).clone()
        )
        self.register_buffer(
            'image_pe',
            sam.prompt_encoder.get_dense_pe().clone()
        )

        # --- Stage 3: Mask Head ---
        md = sam.mask_decoder
        self.output_upscaling = md.output_upscaling
        self.output_hypernetworks_mlps = md.output_hypernetworks_mlps
        self.iou_prediction_head = md.iou_prediction_head
        self.num_mask_tokens = md.num_mask_tokens
        self.use_stability_score = use_stability_score
        self.mask_threshold = sam.mask_threshold
        self.stability_score_offset = 1.0

        # Static shape constants (avoid dynamic Shape ops in ONNX)
        self._embed_dim = sam.prompt_encoder.embed_dim              # 256
        self._embed_h = sam.prompt_encoder.image_embedding_size[0]  # 64
        self._embed_w = sam.prompt_encoder.image_embedding_size[1]  # 64
        self._upscale_c = self._embed_dim // 8                      # 32
        self._upscale_spatial = (self._embed_h * 4) * (self._embed_w * 4)  # 65536

        # Fix 4: Replace GELU → tanh approximation in output_upscaling
        gelu_count = _replace_gelu(self.output_upscaling)
        if gelu_count > 0:
            print(f"  [Fix 4] Replaced {gelu_count} nn.GELU → tanh approximation")

    # --- Fix 2: NPU-safe float equality (no Equal/Cast/Abs) ---

    @staticmethod
    def _float_eq(x, val):
        """relu(1 - (x-val)^2):  1.0 when x==val, 0.0 otherwise."""
        diff = x - val
        return torch.relu(1.0 - diff * diff)

    # --- Fix 5: NPU-safe stability score (no bool ReduceSum) ---

    @staticmethod
    def _stability_score_npu(masks, mask_threshold, threshold_offset):
        """Sigmoid step approximation.  All float ops."""
        k = 50.0
        high = torch.sigmoid(k * (masks - (mask_threshold + threshold_offset)))
        low = torch.sigmoid(k * (masks - (mask_threshold - threshold_offset)))
        intersections = high.sum(-1).sum(-1)
        unions = low.sum(-1).sum(-1)
        return intersections / unions

    def forward(self, image_embeddings, point_embedding_pe, point_labels):

        # ==========  Stage 1: Label Embedding  ==========
        # Fix 2: pure float masks instead of Equal/Cast
        point_labels = point_labels.unsqueeze(-1).expand_as(point_embedding_pe)

        mask_neg1 = self._float_eq(point_labels, -1.0)
        mask_not_neg1 = 1.0 - mask_neg1

        sparse_embedding = point_embedding_pe * mask_not_neg1
        sparse_embedding = sparse_embedding + \
            self.not_a_point_embed.weight * mask_neg1

        for i in range(self.num_point_embeddings):
            mask_i = self._float_eq(point_labels, float(i))
            sparse_embedding = sparse_embedding + \
                self.point_embeddings[i].weight * mask_i

        # ==========  Stage 2: Transformer  ==========
        output_tokens = torch.cat(
            [self.iou_token.weight, self.mask_tokens.weight], dim=0
        )
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_embedding.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, sparse_embedding), dim=1)

        src = image_embeddings + self.dense_embedding
        hs, src = self.transformer(src, self.image_pe, tokens)

        # ==========  Stage 3: Mask Head  ==========
        # Fix 3: Slice+Reshape instead of Gather for indexing
        iou_token_out = hs[:, 0:1, :].reshape(1, self._embed_dim)
        mask_tokens_out = hs[:, 1:(1 + self.num_mask_tokens), :]

        # Reshape src back to spatial grid
        src = src.transpose(1, 2).reshape(
            1, self._embed_dim, self._embed_h, self._embed_w
        )

        # Upscale mask embeddings  (Fix 4: GELU already replaced)
        upscaled = self.output_upscaling(src)

        # Hypernetwork MLPs  (Fix 3: Slice+Reshape instead of Gather)
        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            token_i = mask_tokens_out[:, i:i+1, :].reshape(1, self._embed_dim)
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](token_i)
            )
        hyper_in = torch.stack(hyper_in_list, dim=1)

        # Generate masks via matrix multiply (static shapes, no Shape ops)
        masks = (hyper_in @ upscaled.reshape(
            1, self._upscale_c, self._upscale_spatial
        )).reshape(1, -1, self._embed_h * 4, self._embed_w * 4)

        # Scores
        if self.use_stability_score:
            # Fix 5: sigmoid-based stability score
            scores = self._stability_score_npu(
                masks, self.mask_threshold, self.stability_score_offset
            )
        else:
            scores = self.iou_prediction_head(iou_token_out)

        return scores, masks


# ============================================================
# ONNX post-processing
# ============================================================

def _simplify_onnx(onnx_path):
    """Use onnxruntime ORT_ENABLE_BASIC to fold constants."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("  WARNING: onnxruntime not installed, skipping graph simplification")
        return

    tmp_path = onnx_path + ".tmp"
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess_options.optimized_model_filepath = tmp_path
    ort.InferenceSession(onnx_path, sess_options)
    os.replace(tmp_path, onnx_path)
    print("  [Fix 6] Graph simplification (constant folding) complete")


def _convert_int64_to_int32(onnx_path):
    """Convert all int64/int16 → int32 for NPU compatibility."""
    try:
        import onnx
        from onnx import TensorProto, shape_inference
    except ImportError:
        print("  WARNING: onnx not installed, skipping dtype conversion")
        return

    model = onnx.load(onnx_path)
    INT64, INT16, INT32 = TensorProto.INT64, TensorProto.INT16, TensorProto.INT32
    convertible = {INT64, INT16}
    count = 0

    def _np_dt(dt):
        return np.int64 if dt == INT64 else np.int16

    # Cast nodes
    for node in model.graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i in convertible:
                    attr.i = INT32
                    count += 1

    # ConstantOfShape and Constant nodes
    for node in model.graph.node:
        if node.op_type in ("ConstantOfShape", "Constant"):
            for attr in node.attribute:
                if attr.name == "value" and attr.t.data_type in convertible:
                    raw = np.frombuffer(attr.t.raw_data, dtype=_np_dt(attr.t.data_type))
                    attr.t.raw_data = raw.astype(np.int32).tobytes()
                    attr.t.data_type = INT32
                    count += 1

    # Initializers
    for init in model.graph.initializer:
        if init.data_type in convertible:
            raw = np.frombuffer(init.raw_data, dtype=_np_dt(init.data_type))
            init.raw_data = raw.astype(np.int32).tobytes()
            init.data_type = INT32
            count += 1

    # value_info, inputs, outputs
    for vi in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        t = vi.type.tensor_type
        if t.elem_type in convertible:
            t.elem_type = INT32
            count += 1

    del model.graph.value_info[:]
    try:
        model = shape_inference.infer_shapes(model)
    except Exception:
        pass

    onnx.save(model, onnx_path)
    if count > 0:
        print(f"  [Fix 7] Converted {count} int64/int16 → int32")


# ============================================================
# ONNX summary printer
# ============================================================

def _print_onnx_summary(onnx_path):
    """Print ops, dtypes, and I/O shapes of an ONNX model."""
    try:
        import onnx
    except ImportError:
        print("Cannot print summary: onnx not installed")
        return

    dtype_names = {
        1: "FLOAT", 2: "UINT8", 3: "INT8", 5: "INT16",
        6: "INT32", 7: "INT64", 9: "BOOL", 10: "FLOAT16",
    }
    model = onnx.load(onnx_path)
    ops = Counter(n.op_type for n in model.graph.node)

    dtypes = set()
    for init in model.graph.initializer:
        dtypes.add(dtype_names.get(init.data_type, "?"))
    for inp in model.graph.input:
        dtypes.add(dtype_names.get(inp.type.tensor_type.elem_type, "?"))
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.name == "value" and hasattr(attr, 't') and attr.t.data_type:
                dtypes.add(dtype_names.get(attr.t.data_type, "?"))

    print(f"\n{'='*60}")
    print(f"  {os.path.basename(onnx_path)}")
    print(f"{'='*60}")
    print(f"  Nodes: {len(model.graph.node)},  Op types: {len(ops)}")
    print(f"  Ops: {', '.join(f'{op}({cnt})' for op, cnt in sorted(ops.items()))}")
    print(f"  Dtypes: {', '.join(sorted(dtypes))}")

    for inp in model.graph.input:
        shape = [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]
        dt = dtype_names.get(inp.type.tensor_type.elem_type, "?")
        print(f"  Input  {inp.name}: {dt} {shape}")
    for out in model.graph.output:
        shape = [d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim]
        dt = dtype_names.get(out.type.tensor_type.elem_type, "?")
        print(f"  Output {out.name}: {dt} {shape}")

    npu_bad = {"Sin", "Cos", "Erf", "Abs", "Gather"}
    found = npu_bad & set(ops.keys())
    if found:
        print(f"  WARNING: NPU-incompatible ops present: {', '.join(sorted(found))}")
    if "INT64" in dtypes:
        print(f"  WARNING: INT64 data type present")
    print(f"{'='*60}")


# ============================================================
# PyTorch >= 2.6 compatibility
# ============================================================

def _onnx_export(*args, **kwargs):
    """torch.onnx.export wrapper for PyTorch >= 2.6 (use legacy exporter)."""
    parts = torch.__version__.split('.')
    major, minor = int(parts[0]), int(parts[1])
    if major > 2 or (major == 2 and minor >= 6):
        kwargs['dynamo'] = False
    torch.onnx.export(*args, **kwargs)


# ============================================================
# Export functions
# ============================================================

def export_encoder(sam, args):
    """Export encoder (RepViT CNN) to ONNX."""
    image_input = torch.randn(1, 3, 1024, 1024, dtype=torch.float)
    sam.forward = sam.forward_dummy_encoder
    traced_model = torch.jit.trace(sam, image_input)

    onnx_path = args.checkpoint.replace('.pth', '_encoder_npu.onnx')
    _onnx_export(
        traced_model, image_input, onnx_path,
        input_names=["image"],
        output_names=["image_embeddings"],
        opset_version=11,
        verbose=False,
    )
    print(f"Exported encoder to {onnx_path}")

    _convert_int64_to_int32(onnx_path)
    _print_onnx_summary(onnx_path)

    if args.check_ops_only:
        os.remove(onnx_path)
        print(f"\n(Removed {onnx_path} — check-ops-only mode)")


def export_decoder(sam, args):
    """Export merged NPU-safe decoder to ONNX."""
    print("Building NPU-safe decoder (5 fixes applied)...")
    decoder = NpuSafeDecoder(
        sam, use_stability_score=args.use_stability_score
    )
    decoder.eval()

    embed_dim = sam.prompt_encoder.embed_dim       # 256
    embed_size = sam.prompt_encoder.image_embedding_size  # (64, 64)
    N = args.num_points

    dummy_embeddings = torch.randn(1, embed_dim, *embed_size, dtype=torch.float)
    dummy_pe = torch.randn(1, N, embed_dim, dtype=torch.float)
    dummy_labels = torch.randint(0, 4, (1, N), dtype=torch.float)

    onnx_path = args.checkpoint.replace('.pth', '_decoder_npu.onnx')
    with torch.no_grad():
        _onnx_export(
            decoder,
            (dummy_embeddings, dummy_pe, dummy_labels),
            onnx_path,
            input_names=["image_embeddings", "point_embedding_pe", "point_labels"],
            output_names=["scores", "masks"],
            opset_version=11,
            verbose=False,
        )
    print(f"Exported decoder to {onnx_path}")

    # Post-processing
    _simplify_onnx(onnx_path)
    _convert_int64_to_int32(onnx_path)
    _print_onnx_summary(onnx_path)

    if args.check_ops_only:
        os.remove(onnx_path)
        print(f"\n(Removed {onnx_path} — check-ops-only mode)")
    else:
        print(f"\nDone! NPU-compatible decoder: {onnx_path}")
        print(f"  Inputs:")
        print(f"    image_embeddings:   FLOAT [1, {embed_dim}, {embed_size[0]}, {embed_size[1]}]")
        print(f"    point_embedding_pe: FLOAT [1, {N}, {embed_dim}]  (CPU pre-computed)")
        print(f"    point_labels:       FLOAT [1, {N}]")
        print()
        print(f"  NOTE: point_embedding_pe must be pre-computed on CPU:")
        print(f"    pe = compute_point_pe(sam, point_coords)")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export EdgeSAM to NPU-compatible ONNX (opset 11)."
    )
    parser.add_argument(
        "checkpoint", type=str,
        help="Path to EdgeSAM checkpoint (.pth)",
    )
    parser.add_argument(
        "--decoder", action="store_true",
        help="Export decoder (default: export encoder)",
    )
    parser.add_argument(
        "--use-stability-score", action="store_true",
        help="Use stability score instead of IoU prediction (recommended for NPU)",
    )
    parser.add_argument(
        "--num-points", type=int, default=5,
        help="Number of prompt points (default: 5, static shape)",
    )
    parser.add_argument(
        "--check-ops-only", action="store_true",
        help="Export, print op summary, then delete the file",
    )
    args = parser.parse_args()

    print("Loading model...")
    sam = sam_model_registry["edge_sam"](
        checkpoint=args.checkpoint, upsample_mode="bilinear"
    )
    sam.eval()

    if args.decoder:
        export_decoder(sam, args)
    else:
        export_encoder(sam, args)
