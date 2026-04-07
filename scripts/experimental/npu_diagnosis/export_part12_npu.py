#!/usr/bin/env python3
"""Export merged decoder Part1+Part2 ONNX for NPU execution.

Target hybrid pipeline:
  - Encoder: NPU (TVN)
  - Part1+Part2 (merged): NPU (TVN)
  - Part3: CPU (ONNX Runtime)
"""

from __future__ import annotations

import argparse
import os
from collections import Counter

import onnx
import torch
import torch.nn as nn

from edge_sam import sam_model_registry


def _onnx_export(*args, **kwargs):
    """torch.onnx.export wrapper for PyTorch >= 2.6 compatibility."""
    parts = torch.__version__.split(".")
    major, minor = int(parts[0]), int(parts[1])
    if major > 2 or (major == 2 and minor >= 6):
        kwargs["dynamo"] = False
    torch.onnx.export(*args, **kwargs)


def count_ops(path: str) -> Counter:
    model = onnx.load(path)
    return Counter(node.op_type for node in model.graph.node)


class Part1_PromptEncoding(nn.Module):
    """Part1: point_embedding_pe + point_labels -> sparse_embedding."""

    def __init__(self, sam):
        super().__init__()
        self.point_embeddings = sam.prompt_encoder.point_embeddings
        self.not_a_point_embed = sam.prompt_encoder.not_a_point_embed
        self.num_point_embeddings = sam.prompt_encoder.num_point_embeddings

    @staticmethod
    def _float_eq(x: torch.Tensor, val: float) -> torch.Tensor:
        # Pure-float equality mask to avoid Equal/Cast(bool).
        diff = x - val
        return torch.relu(1.0 - diff * diff)

    def forward(self, point_embedding_pe: torch.Tensor, point_labels: torch.Tensor) -> torch.Tensor:
        point_labels = point_labels.unsqueeze(-1).expand_as(point_embedding_pe)

        mask_neg1 = self._float_eq(point_labels, -1.0)
        mask_not_neg1 = 1.0 - mask_neg1

        point_embedding = point_embedding_pe * mask_not_neg1
        point_embedding = point_embedding + self.not_a_point_embed.weight * mask_neg1

        for i in range(self.num_point_embeddings):
            mask_i = self._float_eq(point_labels, float(i))
            point_embedding = point_embedding + self.point_embeddings[i].weight * mask_i

        return point_embedding


class Part2_Transformer(nn.Module):
    """Part2: image_embeddings + sparse_embedding -> (hs, src)."""

    def __init__(self, sam):
        super().__init__()
        self.transformer = sam.mask_decoder.transformer
        self.iou_token = sam.mask_decoder.iou_token
        self.mask_tokens = sam.mask_decoder.mask_tokens
        self.register_buffer(
            "dense_embedding",
            sam.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).clone(),
        )
        self.register_buffer("image_pe", sam.prompt_encoder.get_dense_pe().clone())

    def forward(self, image_embeddings: torch.Tensor, sparse_embedding: torch.Tensor):
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_embedding.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_embedding), dim=1)

        src = image_embeddings + self.dense_embedding
        hs, src = self.transformer(src, self.image_pe, tokens)
        return hs, src


class Part12_PromptAndTransformer(nn.Module):
    """Merged Part1+Part2: (PE, labels, image_embeddings) -> (hs, src)."""

    def __init__(self, sam):
        super().__init__()
        self.part1 = Part1_PromptEncoding(sam)
        self.part2 = Part2_Transformer(sam)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        point_embedding_pe: torch.Tensor,
        point_labels: torch.Tensor,
    ):
        sparse_embedding = self.part1(point_embedding_pe, point_labels)
        hs, src = self.part2(image_embeddings, sparse_embedding)
        return hs, src


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export merged decoder Part1+Part2 ONNX for NPU, leaving Part3 on CPU."
    )
    parser.add_argument("checkpoint", type=str, help="Path to EdgeSAM checkpoint (.pth)")
    parser.add_argument("--output-dir", type=str, default="./part12_npu", help="Output directory")
    parser.add_argument("--num-points", type=int, default=5, help="Static point count N")
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    sam = sam_model_registry["edge_sam"](checkpoint=args.checkpoint, upsample_mode="bilinear")
    sam.eval()

    n = args.num_points
    embed_dim = sam.prompt_encoder.embed_dim  # 256
    num_mask_tokens = sam.mask_decoder.num_mask_tokens  # 4
    total_tokens = 1 + num_mask_tokens + n

    part12 = Part12_PromptAndTransformer(sam).eval()

    print("Exporting merged Part1+Part2 for NPU...")
    path = os.path.join(args.output_dir, "part12_prompt_transformer.onnx")
    _onnx_export(
        part12,
        (
            torch.randn(1, 256, 64, 64, dtype=torch.float32),
            torch.randn(1, n, embed_dim, dtype=torch.float32),
            torch.randint(-1, 4, (1, n), dtype=torch.float32),
        ),
        path,
        input_names=["image_embeddings", "point_embedding_pe", "point_labels"],
        output_names=["hs", "src"],
        opset_version=args.opset,
        verbose=False,
    )
    ops = count_ops(path)
    print(f"  {'part12_prompt_transformer.onnx':28s} -> {path}")
    print("    ops:", ", ".join(f"{k}({v})" for k, v in sorted(ops.items())))

    print()
    print("Contracts:")
    print(f"  Part12 input : image_embeddings[1,256,64,64], point_embedding_pe[1,{n},256], point_labels[1,{n}]")
    print(f"  Part12 output: hs[1,{total_tokens},256], src[1,4096,256]")
    print("Done.")


if __name__ == "__main__":
    main()
