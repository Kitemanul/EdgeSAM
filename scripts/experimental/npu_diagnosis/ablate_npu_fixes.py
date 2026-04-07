#!/usr/bin/env python
"""
Ablation test: toggle each NPU fix independently to find which ones are needed.

Generates multiple ONNX variants of Part 1 (Prompt Encoding) and Part 3 (Mask
Head), each with a different subset of fixes enabled.  Part 2 (Transformer)
always compiles as-is, so only one copy is exported.

The 5 fixes under test:

  Part 1:
    Fix A — Sin/Cos → PE pre-computed on CPU
    Fix B — Equal/Abs → float arithmetic  (relu(1 - diff²))

  Part 3:
    Fix C — Gather → Slice + Reshape
    Fix D — GELU (Erf) → tanh approximation
    Fix E — bool ReduceSum → sigmoid stability score

Usage:
    python scripts/ablate_npu_fixes.py weights/edge_sam_3x.pth

    # Custom output dir and point count
    python scripts/ablate_npu_fixes.py weights/edge_sam_3x.pth \\
        --output-dir ./npu_ablation --num-points 5

Output (12 ONNX files):
    part1_vanilla.onnx          — no fixes
    part1_fixA.onnx             — only Fix A
    part1_fixB.onnx             — only Fix B
    part1_fixAB.onnx            — Fix A + B (all Part 1 fixes)

    part2_transformer.onnx      — always passes, single copy

    part3_vanilla.onnx          — no fixes
    part3_fixC.onnx             — only Fix C
    part3_fixD.onnx             — only Fix D
    part3_fixE.onnx             — only Fix E
    part3_fixCD.onnx            — Fix C + D
    part3_fixCE.onnx            — Fix C + E
    part3_fixDE.onnx            — Fix D + E
    part3_fixCDE.onnx           — Fix C + D + E (all Part 3 fixes)

Compile each file with the NPU toolchain.  The minimal set of PASS
variants tells you exactly which fixes are necessary.
"""

import argparse
import copy
import os
from collections import Counter
from itertools import combinations

import numpy as np
import torch
import torch.nn as nn

from edge_sam import sam_model_registry


# ============================================================
# Utilities
# ============================================================

def _onnx_export(*args, **kwargs):
    """torch.onnx.export wrapper for PyTorch >= 2.6 (use legacy exporter)."""
    parts = torch.__version__.split('.')
    major, minor = int(parts[0]), int(parts[1])
    if major > 2 or (major == 2 and minor >= 6):
        kwargs['dynamo'] = False
    torch.onnx.export(*args, **kwargs)


def print_summary(path):
    """Print ops, dtypes, and I/O shapes of an ONNX model."""
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


def export(module, dummy_inputs, input_names, output_names, path, desc):
    """Export a module to ONNX and print summary."""
    print(f"\n  {desc}")
    print(f"  {'─' * 56}")
    with torch.no_grad():
        _onnx_export(
            module, dummy_inputs, path,
            input_names=input_names,
            output_names=output_names,
            opset_version=11,
            verbose=False,
        )
    print(f"    → {os.path.basename(path)}")
    print_summary(path)


# ============================================================
# Part 1 variants: Prompt Encoding
# ============================================================

class Part1_Vanilla(nn.Module):
    """Original prompt encoding — sin/cos PE + Equal-based label lookup."""

    def __init__(self, sam):
        super().__init__()
        self.pe_layer = sam.prompt_encoder.pe_layer
        self.point_embeddings = sam.prompt_encoder.point_embeddings
        self.not_a_point_embed = sam.prompt_encoder.not_a_point_embed
        self.img_size = sam.image_encoder.img_size

    def forward(self, point_coords, point_labels):
        # PE encoding (sin/cos)
        coords = point_coords + 0.5
        coords = coords / self.img_size
        coords = 2 * coords - 1
        coords = coords @ self.pe_layer.positional_encoding_gaussian_matrix
        coords = 2 * 3.141592653589793 * coords
        point_embedding = torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

        # Label embedding (Equal/Cast pattern)
        point_embedding[point_labels == -1] = 0.0
        point_embedding[point_labels == -1] += self.not_a_point_embed.weight
        point_embedding[point_labels == 0] += self.point_embeddings[0].weight
        point_embedding[point_labels == 1] += self.point_embeddings[1].weight
        point_embedding[point_labels == 2] += self.point_embeddings[2].weight
        point_embedding[point_labels == 3] += self.point_embeddings[3].weight
        return point_embedding


class Part1_FixA(nn.Module):
    """Fix A only: PE pre-computed on CPU.  Labels still use Equal/Cast."""

    def __init__(self, sam):
        super().__init__()
        self.point_embeddings = sam.prompt_encoder.point_embeddings
        self.not_a_point_embed = sam.prompt_encoder.not_a_point_embed

    def forward(self, point_embedding_pe, point_labels):
        point_embedding = point_embedding_pe.clone()
        point_embedding[point_labels == -1] = 0.0
        point_embedding[point_labels == -1] += self.not_a_point_embed.weight
        point_embedding[point_labels == 0] += self.point_embeddings[0].weight
        point_embedding[point_labels == 1] += self.point_embeddings[1].weight
        point_embedding[point_labels == 2] += self.point_embeddings[2].weight
        point_embedding[point_labels == 3] += self.point_embeddings[3].weight
        return point_embedding


class Part1_FixB(nn.Module):
    """Fix B only: float arithmetic labels.  PE still uses sin/cos."""

    def __init__(self, sam):
        super().__init__()
        self.pe_layer = sam.prompt_encoder.pe_layer
        self.point_embeddings = sam.prompt_encoder.point_embeddings
        self.not_a_point_embed = sam.prompt_encoder.not_a_point_embed
        self.num_point_embeddings = sam.prompt_encoder.num_point_embeddings
        self.img_size = sam.image_encoder.img_size

    @staticmethod
    def _float_eq(x, val):
        diff = x - val
        return torch.relu(1.0 - diff * diff)

    def forward(self, point_coords, point_labels):
        # PE encoding (sin/cos — NOT fixed)
        coords = point_coords + 0.5
        coords = coords / self.img_size
        coords = 2 * coords - 1
        coords = coords @ self.pe_layer.positional_encoding_gaussian_matrix
        coords = 2 * 3.141592653589793 * coords
        point_embedding_pe = torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

        # Label embedding (float arithmetic — FIXED)
        point_labels = point_labels.unsqueeze(-1).expand_as(point_embedding_pe)
        mask_neg1 = self._float_eq(point_labels, -1.0)
        mask_not_neg1 = 1.0 - mask_neg1
        sparse = point_embedding_pe * mask_not_neg1
        sparse = sparse + self.not_a_point_embed.weight * mask_neg1
        for i in range(self.num_point_embeddings):
            mask_i = self._float_eq(point_labels, float(i))
            sparse = sparse + self.point_embeddings[i].weight * mask_i
        return sparse


class Part1_FixAB(nn.Module):
    """Fix A + B: PE on CPU + float arithmetic labels."""

    def __init__(self, sam):
        super().__init__()
        self.point_embeddings = sam.prompt_encoder.point_embeddings
        self.not_a_point_embed = sam.prompt_encoder.not_a_point_embed
        self.num_point_embeddings = sam.prompt_encoder.num_point_embeddings

    @staticmethod
    def _float_eq(x, val):
        diff = x - val
        return torch.relu(1.0 - diff * diff)

    def forward(self, point_embedding_pe, point_labels):
        point_labels = point_labels.unsqueeze(-1).expand_as(point_embedding_pe)
        mask_neg1 = self._float_eq(point_labels, -1.0)
        mask_not_neg1 = 1.0 - mask_neg1
        sparse = point_embedding_pe * mask_not_neg1
        sparse = sparse + self.not_a_point_embed.weight * mask_neg1
        for i in range(self.num_point_embeddings):
            mask_i = self._float_eq(point_labels, float(i))
            sparse = sparse + self.point_embeddings[i].weight * mask_i
        return sparse


# ============================================================
# Part 2: Transformer (always passes, single variant)
# ============================================================

class Part2_Transformer(nn.Module):
    """Transformer — no fixes needed, always compiles."""

    def __init__(self, sam):
        super().__init__()
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

    def forward(self, image_embeddings, sparse_embedding):
        output_tokens = torch.cat(
            [self.iou_token.weight, self.mask_tokens.weight], dim=0
        )
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_embedding.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, sparse_embedding), dim=1)
        src = image_embeddings + self.dense_embedding
        hs, src = self.transformer(src, self.image_pe, tokens)
        return hs, src


# ============================================================
# Part 3 building blocks
# ============================================================

class _GELUTanh(nn.Module):
    """GELU via tanh approximation.  Avoids the Erf ONNX op."""

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


class Part3_MaskHead(nn.Module):
    """Mask head with individually toggleable fixes C, D, E.

    Args:
        sam: EdgeSAM model
        fix_c: Replace Gather with Slice+Reshape (integer indexing)
        fix_d: Replace GELU (Erf) with tanh approximation
        fix_e: Replace bool ReduceSum with sigmoid stability score
    """

    def __init__(self, sam, fix_c=False, fix_d=False, fix_e=False):
        super().__init__()
        md = sam.mask_decoder
        # Deep copy to avoid in-place mutation leaking between variants
        self.output_upscaling = copy.deepcopy(md.output_upscaling)
        self.output_hypernetworks_mlps = copy.deepcopy(md.output_hypernetworks_mlps)
        self.iou_prediction_head = copy.deepcopy(md.iou_prediction_head)
        self.num_mask_tokens = md.num_mask_tokens
        self.mask_threshold = sam.mask_threshold
        self.stability_score_offset = 1.0
        self.fix_c = fix_c
        self.fix_e = fix_e

        # Fix D: replace GELU
        if fix_d:
            _replace_gelu(self.output_upscaling)

        # Static shape constants (for fix_c)
        self._embed_dim = 256
        self._embed_h = 64
        self._embed_w = 64
        self._upscale_c = 32
        self._upscale_spatial = 65536

    @staticmethod
    def _stability_score_sigmoid(masks, mask_threshold, threshold_offset):
        """Fix E: sigmoid step approximation — all float ops."""
        k = 50.0
        high = torch.sigmoid(k * (masks - (mask_threshold + threshold_offset)))
        low = torch.sigmoid(k * (masks - (mask_threshold - threshold_offset)))
        intersections = high.sum(-1).sum(-1)
        unions = low.sum(-1).sum(-1)
        return intersections / unions

    @staticmethod
    def _stability_score_original(masks, mask_threshold, threshold_offset):
        """Original: bool comparison + ReduceSum with int16."""
        intersections = (
            (masks > (mask_threshold + threshold_offset))
            .sum(-1, dtype=torch.int16)
            .sum(-1, dtype=torch.int32)
        )
        unions = (
            (masks > (mask_threshold - threshold_offset))
            .sum(-1, dtype=torch.int16)
            .sum(-1, dtype=torch.int32)
        )
        return intersections / unions

    def forward(self, hs, src):
        if self.fix_c:
            # Fix C: Slice+Reshape instead of integer index (Gather)
            iou_token_out = hs[:, 0:1, :].reshape(1, self._embed_dim)
            mask_tokens_out = hs[:, 1:(1 + self.num_mask_tokens), :]
        else:
            # Original: integer indexing → Gather op
            iou_token_out = hs[:, 0, :]
            mask_tokens_out = hs[:, 1:(1 + self.num_mask_tokens), :]

        if self.fix_c:
            src = src.transpose(1, 2).reshape(
                1, self._embed_dim, self._embed_h, self._embed_w
            )
        else:
            b, _, c = src.shape
            src = src.transpose(1, 2).view(1, c, 64, 64)

        upscaled = self.output_upscaling(src)

        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            if self.fix_c:
                # Fix C: Slice+Reshape
                token_i = mask_tokens_out[:, i:i+1, :].reshape(1, self._embed_dim)
            else:
                # Original: integer index → Gather
                token_i = mask_tokens_out[:, i, :]
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](token_i)
            )
        hyper_in = torch.stack(hyper_in_list, dim=1)

        if self.fix_c:
            masks = (hyper_in @ upscaled.reshape(
                1, self._upscale_c, self._upscale_spatial
            )).reshape(1, -1, self._embed_h * 4, self._embed_w * 4)
        else:
            b, c, h, w = upscaled.shape
            masks = (hyper_in @ upscaled.view(b, c, h * w)).view(b, -1, h, w)

        _iou_pred = self.iou_prediction_head(iou_token_out)

        if self.fix_e:
            scores = self._stability_score_sigmoid(
                masks, self.mask_threshold, self.stability_score_offset
            )
        else:
            scores = self._stability_score_original(
                masks, self.mask_threshold, self.stability_score_offset
            )

        return scores, masks


# ============================================================
# CPU-side PE helper (same as in export_onnx_model_npu.py)
# ============================================================

def compute_point_pe(sam, point_coords):
    """Compute positional encoding on CPU (for Fix A variants)."""
    with torch.no_grad():
        coords = point_coords + 0.5
        coords = coords / sam.image_encoder.img_size
        pe = sam.prompt_encoder.pe_layer._pe_encoding(coords)
    return pe


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ablation: export Part 1 / Part 3 variants with each fix toggled."
    )
    parser.add_argument("checkpoint", type=str, help="Path to EdgeSAM .pth")
    parser.add_argument("--output-dir", type=str, default="./npu_ablation",
                        help="Output directory (default: ./npu_ablation)")
    parser.add_argument("--num-points", type=int, default=5,
                        help="Number of prompt points (default: 5)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    N = args.num_points
    embed_dim = 256
    num_mask_tokens = 4
    total_tokens = 1 + num_mask_tokens + N  # 5 + N = 10

    print("Loading model...")
    sam = sam_model_registry["edge_sam"](
        checkpoint=args.checkpoint, upsample_mode="bilinear"
    )
    sam.eval()

    # --- Dummy inputs ---
    dummy_coords = torch.randn(1, N, 2, dtype=torch.float)
    dummy_labels_float = torch.randint(0, 4, (1, N), dtype=torch.float)
    dummy_pe = torch.randn(1, N, embed_dim, dtype=torch.float)
    dummy_img_embed = torch.randn(1, embed_dim, 64, 64, dtype=torch.float)
    dummy_sparse = torch.randn(1, N, embed_dim, dtype=torch.float)
    dummy_hs = torch.randn(1, total_tokens, embed_dim, dtype=torch.float)
    dummy_src = torch.randn(1, 4096, embed_dim, dtype=torch.float)

    print(f"\n{'='*60}")
    print(f"  Part 1: Prompt Encoding — 4 variants")
    print(f"{'='*60}")

    # Vanilla Part 1 uses raw coords (float) and float labels
    # The original _embed_points uses int labels, but ONNX export
    # needs static types.  We pass float labels for all variants.

    # Part 1 vanilla
    p1_vanilla = Part1_Vanilla(sam).eval()
    export(p1_vanilla,
           (dummy_coords, dummy_labels_float),
           ["point_coords", "point_labels"],
           ["sparse_embedding"],
           os.path.join(args.output_dir, "part1_vanilla.onnx"),
           "Part 1 VANILLA — sin/cos PE + Equal labels (no fixes)")

    # Part 1 Fix A only
    p1_a = Part1_FixA(sam).eval()
    export(p1_a,
           (dummy_pe, dummy_labels_float),
           ["point_embedding_pe", "point_labels"],
           ["sparse_embedding"],
           os.path.join(args.output_dir, "part1_fixA.onnx"),
           "Part 1 FIX A — PE on CPU, labels still use Equal")

    # Part 1 Fix B only
    p1_b = Part1_FixB(sam).eval()
    export(p1_b,
           (dummy_coords, dummy_labels_float),
           ["point_coords", "point_labels"],
           ["sparse_embedding"],
           os.path.join(args.output_dir, "part1_fixB.onnx"),
           "Part 1 FIX B — float labels, PE still uses sin/cos")

    # Part 1 Fix A+B
    p1_ab = Part1_FixAB(sam).eval()
    export(p1_ab,
           (dummy_pe, dummy_labels_float),
           ["point_embedding_pe", "point_labels"],
           ["sparse_embedding"],
           os.path.join(args.output_dir, "part1_fixAB.onnx"),
           "Part 1 FIX A+B — PE on CPU + float labels (all fixes)")

    # --- Part 2: Transformer (single copy) ---
    print(f"\n{'='*60}")
    print(f"  Part 2: Transformer — 1 variant (always passes)")
    print(f"{'='*60}")

    p2 = Part2_Transformer(sam).eval()
    export(p2,
           (dummy_img_embed, dummy_sparse),
           ["image_embeddings", "sparse_embedding"],
           ["hs", "src"],
           os.path.join(args.output_dir, "part2_transformer.onnx"),
           "Part 2 VANILLA — no fixes needed")

    # --- Part 3: Mask Head — 8 variants (all combinations of C, D, E) ---
    print(f"\n{'='*60}")
    print(f"  Part 3: Mask Head — 8 variants")
    print(f"{'='*60}")

    fix_labels = ['C', 'D', 'E']

    # Generate all 8 combinations: (), (C,), (D,), (E,), (CD), (CE), (DE), (CDE)
    for r in range(len(fix_labels) + 1):
        for combo in combinations(fix_labels, r):
            fix_c = 'C' in combo
            fix_d = 'D' in combo
            fix_e = 'E' in combo

            if not combo:
                suffix = "vanilla"
                desc_fixes = "no fixes"
            else:
                suffix = "fix" + "".join(combo)
                desc_fixes = " + ".join(
                    {"C": "Slice+Reshape", "D": "GELU tanh", "E": "sigmoid score"}[f]
                    for f in combo
                )

            p3 = Part3_MaskHead(sam, fix_c=fix_c, fix_d=fix_d, fix_e=fix_e).eval()
            export(p3,
                   (dummy_hs, dummy_src),
                   ["hs", "src"],
                   ["scores", "masks"],
                   os.path.join(args.output_dir, f"part3_{suffix}.onnx"),
                   f"Part 3 {suffix.upper()} — {desc_fixes}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Generated 13 ONNX files in {args.output_dir}/")
    print()
    print("  Part 1 (4 variants):")
    print("    part1_vanilla.onnx   — baseline, no fixes")
    print("    part1_fixA.onnx      — Fix A: PE on CPU (removes Sin/Cos)")
    print("    part1_fixB.onnx      — Fix B: float labels (removes Equal/Abs)")
    print("    part1_fixAB.onnx     — Fix A+B: both")
    print()
    print("  Part 2 (1 variant):")
    print("    part2_transformer.onnx — always passes")
    print()
    print("  Part 3 (8 variants):")
    print("    part3_vanilla.onnx   — baseline, no fixes")
    print("    part3_fixC.onnx      — Fix C: Slice+Reshape (removes Gather)")
    print("    part3_fixD.onnx      — Fix D: GELU tanh (removes Erf)")
    print("    part3_fixE.onnx      — Fix E: sigmoid score (removes bool ReduceSum)")
    print("    part3_fixCD.onnx     — Fix C+D")
    print("    part3_fixCE.onnx     — Fix C+E")
    print("    part3_fixDE.onnx     — Fix D+E")
    print("    part3_fixCDE.onnx    — Fix C+D+E: all Part 3 fixes")
    print()
    print("  Compile each with NPU toolchain.  Record PASS/FAIL per file.")
    print("  See docs/npu_fix_ablation_guide.md for interpretation.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
