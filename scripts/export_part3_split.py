#!/usr/bin/env python3
"""Export Part3 (Mask Head) as 3 split ONNX models for TV/NPU load diagnosis.

Split A (operator-cluster based):
  - p3a_upscale: src -> upscaled
  - p3b_hyper: hs -> hyper_in (+ iou_token)
  - p3c_fuse_score: hyper_in + upscaled (+ iou_token) -> scores + masks
"""

from __future__ import annotations

import argparse
import copy
import os
from collections import Counter

import onnx
import torch
import torch.nn as nn

from edge_sam import sam_model_registry


class Part3A_Upscale(nn.Module):
    """Part3a: only the upscaling branch."""

    def __init__(self, sam):
        super().__init__()
        self.output_upscaling = copy.deepcopy(sam.mask_decoder.output_upscaling)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # src: [1, 4096, 256]
        src_2d = src.transpose(1, 2).reshape(1, 256, 64, 64)
        upscaled = self.output_upscaling(src_2d)  # [1, 32, 256, 256]
        return upscaled


class Part3B_Hyper(nn.Module):
    """Part3b: token->hyper branch."""

    def __init__(self, sam):
        super().__init__()
        md = sam.mask_decoder
        self.num_mask_tokens = md.num_mask_tokens
        self.output_hypernetworks_mlps = copy.deepcopy(md.output_hypernetworks_mlps)

    def forward(self, hs: torch.Tensor):
        # hs: [1, 10, 256]
        iou_token = hs[:, 0:1, :].reshape(1, 256)
        mask_tokens = hs[:, 1:(1 + self.num_mask_tokens), :]

        hyper_parts = []
        for i in range(self.num_mask_tokens):
            token_i = mask_tokens[:, i:i + 1, :].reshape(1, 256)
            token_out = self.output_hypernetworks_mlps[i](token_i)  # [1, 32]
            hyper_parts.append(token_out)
        hyper_in = torch.stack(hyper_parts, dim=1)  # [1, 4, 32]
        return hyper_in, iou_token


class Part3C_FuseScore(nn.Module):
    """Part3c: hyper/upscaled fusion + score head."""

    def __init__(self, sam, with_iou_head: bool = False):
        super().__init__()
        self.with_iou_head = with_iou_head
        if with_iou_head:
            self.iou_prediction_head = copy.deepcopy(sam.mask_decoder.iou_prediction_head)

    def forward(self, hyper_in: torch.Tensor, upscaled: torch.Tensor, iou_token: torch.Tensor):
        # hyper_in: [1, 4, 32], upscaled: [1, 32, 256, 256], iou_token: [1, 256]
        masks = (hyper_in @ upscaled.reshape(1, 32, 65536)).reshape(1, -1, 256, 256)

        if self.with_iou_head:
            scores = self.iou_prediction_head(iou_token)
        else:
            # keep score path minimal and deterministic to reduce operator variables
            scores = hyper_in[:, :, :1].reshape(1, 4)

        return scores, masks


def count_ops(onnx_path: str) -> Counter:
    m = onnx.load(onnx_path)
    return Counter(node.op_type for node in m.graph.node)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Part3 split-A ONNX models (p3a/p3b/p3c).")
    parser.add_argument("checkpoint", type=str, help="Path to EdgeSAM checkpoint (.pth)")
    parser.add_argument("--output-dir", type=str, default="./part3_split_a", help="Output directory")
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version")
    parser.add_argument("--no-constant-folding", action="store_true", help="Disable ONNX constant folding")
    parser.add_argument("--with-iou-head", action="store_true", help="Enable iou_prediction_head in p3c")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    sam = sam_model_registry["edge_sam"](checkpoint=args.checkpoint, upsample_mode="bilinear")
    sam.eval()

    p3a = Part3A_Upscale(sam)
    p3b = Part3B_Hyper(sam)
    p3c = Part3C_FuseScore(sam, with_iou_head=args.with_iou_head)

    src = torch.randn(1, 4096, 256, dtype=torch.float32)
    hs = torch.randn(1, 10, 256, dtype=torch.float32)
    hyper_in = torch.randn(1, 4, 32, dtype=torch.float32)
    upscaled = torch.randn(1, 32, 256, 256, dtype=torch.float32)
    iou_token = torch.randn(1, 256, dtype=torch.float32)

    targets = [
        ("p3a_upscale.onnx", p3a, (src,), ["src"], ["upscaled"]),
        ("p3b_hyper.onnx", p3b, (hs,), ["hs"], ["hyper_in", "iou_token"]),
        (
            "p3c_fuse_score.onnx",
            p3c,
            (hyper_in, upscaled, iou_token),
            ["hyper_in", "upscaled", "iou_token"],
            ["scores", "masks"],
        ),
    ]

    print("Exporting Part3 split-A variants...")
    for filename, model, example, input_names, output_names in targets:
        out_path = os.path.join(args.output_dir, filename)
        torch.onnx.export(
            model,
            example,
            out_path,
            input_names=input_names,
            input_names=input_names,
            output_names=output_names,
            opset_version=args.opset,
            do_constant_folding=not args.no_constant_folding,
            verbose=False,
        )
        ops = count_ops(out_path)
        top = ", ".join(f"{k}({v})" for k, v in sorted(ops.items()))
        print(f"  {filename:20s} -> {out_path}")
        print(f"    ops: {top}")

    print("Done.")


if __name__ == "__main__":
    main()
