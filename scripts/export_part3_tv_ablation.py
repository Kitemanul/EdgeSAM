#!/usr/bin/env python3
"""Export Part3 TV-load ablation ONNX variants.

This script exports multiple Part3 (Mask Head) variants for TV runtime `loadModel`
diagnosis. It focuses on operator-level ablations:

- p3_base
- p3_no_score
- p3_no_deconv
- p3_no_gemm
- p3_no_tanh
- p3_no_unsqueeze
- p3_minimal

Each exported ONNX keeps the same I/O contract:
  inputs:  hs [1,10,256], src [1,4096,256]
  outputs: scores [1,4], masks [1,4,256,256]
"""

from __future__ import annotations

import argparse
import copy
import os
from collections import Counter

import onnx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from edge_sam import sam_model_registry


class LinearAsMatMulAdd(nn.Module):
    """Linear implemented explicitly as MatMul + Add for ONNX export."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.weight = nn.Parameter(linear.weight.detach().clone(), requires_grad=False)
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.detach().clone(), requires_grad=False)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x @ self.weight.transpose(0, 1)
        if self.bias is not None:
            y = y + self.bias
        return y


class UpscaleNoDeconv(nn.Module):
    """Replace ConvTranspose-based upscaling with Resize + Conv2d blocks."""

    def __init__(self, in_ch: int = 256, mid_ch: int = 64, out_ch: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.act1(self.conv1(x))
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.act2(self.conv2(x))
        return x


class Part3Variant(nn.Module):
    def __init__(
        self,
        sam,
        *,
        no_score: bool = False,
        no_deconv: bool = False,
        no_gemm: bool = False,
        no_tanh: bool = False,
        no_unsqueeze: bool = False,
        minimal: bool = False,
    ):
        super().__init__()
        self.minimal = minimal
        self.no_score = no_score
        self.no_unsqueeze = no_unsqueeze

        md = sam.mask_decoder
        self.num_mask_tokens = md.num_mask_tokens
        self.mask_threshold = sam.mask_threshold
        self.stability_score_offset = 1.0

        if self.minimal:
            return

        self.output_upscaling = copy.deepcopy(md.output_upscaling)
        self.output_hypernetworks_mlps = copy.deepcopy(md.output_hypernetworks_mlps)
        self.iou_prediction_head = copy.deepcopy(md.iou_prediction_head)

        if no_deconv:
            self.output_upscaling = UpscaleNoDeconv(in_ch=256, mid_ch=64, out_ch=32)

        if no_gemm:
            self._replace_linear_with_matmul_add(self.output_hypernetworks_mlps)
            self._replace_linear_with_matmul_add(self.iou_prediction_head)

        if no_tanh:
            self._replace_tanh_gelu_with_relu(self.output_upscaling)
            self._replace_tanh_gelu_with_relu(self.output_hypernetworks_mlps)
            self._replace_tanh_gelu_with_relu(self.iou_prediction_head)

    @staticmethod
    def _replace_linear_with_matmul_add(module: nn.Module) -> None:
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                setattr(module, name, LinearAsMatMulAdd(child))
            else:
                Part3Variant._replace_linear_with_matmul_add(child)

    @staticmethod
    def _replace_tanh_gelu_with_relu(module: nn.Module) -> None:
        for name, child in list(module.named_children()):
            if isinstance(child, (nn.Tanh, nn.GELU)):
                setattr(module, name, nn.ReLU())
            else:
                Part3Variant._replace_tanh_gelu_with_relu(child)

    @staticmethod
    def _stability_score_npu(masks: torch.Tensor, mask_threshold: float, threshold_offset: float):
        k = 50.0
        high = torch.sigmoid(k * (masks - (mask_threshold + threshold_offset)))
        low = torch.sigmoid(k * (masks - (mask_threshold - threshold_offset)))
        intersections = high.sum(-1).sum(-1)
        unions = low.sum(-1).sum(-1)
        return intersections / unions

    def _constant_scores(self, hs: torch.Tensor) -> torch.Tensor:
        """No-score path without ReduceSum (avoid hs.sum)."""
        return hs[:, :4, :1].reshape(1, 4)

    def _minimal_outputs(self, hs: torch.Tensor, src: torch.Tensor):
        """Input-dependent minimal path to avoid invalid optimized constant graph."""
        # scores: deterministic slice from hs -> [1,4]
        scores = hs[:, :4, :1].reshape(1, 4)

        # masks: deterministic path from src -> [1,4,256,256]
        src_spatial = src.transpose(1, 2).reshape(1, 256, 64, 64)
        masks = F.interpolate(src_spatial[:, :4, :, :], size=(256, 256), mode="nearest")
        return scores, masks

    def forward(self, hs: torch.Tensor, src: torch.Tensor):
        if self.minimal:
            return self._minimal_outputs(hs, src)

        iou_token_out = hs[:, 0:1, :].reshape(1, 256)
        mask_tokens_out = hs[:, 1:(1 + self.num_mask_tokens), :]

        src = src.transpose(1, 2).reshape(1, 256, 64, 64)
        upscaled = self.output_upscaling(src)

        hyper_parts = []
        for i in range(self.num_mask_tokens):
            token_i = mask_tokens_out[:, i:i + 1, :].reshape(1, 256)
            token_out = self.output_hypernetworks_mlps[i](token_i)
            # no_unsqueeze variant avoids torch.stack path.
            if self.no_unsqueeze:
                hyper_parts.append(token_out.reshape(1, 1, 32))
            else:
                hyper_parts.append(token_out)

        if self.no_unsqueeze:
            hyper_in = torch.cat(hyper_parts, dim=1)
        else:
            hyper_in = torch.stack(hyper_parts, dim=1)

        masks = (hyper_in @ upscaled.reshape(1, 32, 65536)).reshape(1, -1, 256, 256)

        if self.no_score:
            scores = self._constant_scores(hs)
        else:
            scores = self._stability_score_npu(masks, self.mask_threshold, self.stability_score_offset)

        return scores, masks


def export_onnx(model: nn.Module, out_path: str, opset: int = 11, constant_folding: bool = True) -> None:
    model.eval()
    hs = torch.randn(1, 10, 256, dtype=torch.float32)
    src = torch.randn(1, 4096, 256, dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            model,
            (hs, src),
            out_path,
            input_names=["hs", "src"],
            output_names=["scores", "masks"],
            opset_version=opset,
            do_constant_folding=constant_folding,
            verbose=False,
        )



def convert_int64_to_int32(onnx_path: str) -> int:
    """Convert int64/int16 tensors in ONNX graph to int32 for NPU compatibility."""
    from onnx import TensorProto

    model = onnx.load(onnx_path)
    INT64, INT16, INT32 = TensorProto.INT64, TensorProto.INT16, TensorProto.INT32
    convertible = {INT64, INT16}
    changed = 0

    def _np_dt(dt):
        return np.int64 if dt == INT64 else np.int16

    for node in model.graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i in convertible:
                    attr.i = INT32
                    changed += 1

    for node in model.graph.node:
        if node.op_type in ("ConstantOfShape", "Constant"):
            for attr in node.attribute:
                if attr.name == "value" and attr.t.data_type in convertible and attr.t.raw_data:
                    raw = np.frombuffer(attr.t.raw_data, dtype=_np_dt(attr.t.data_type))
                    attr.t.raw_data = raw.astype(np.int32).tobytes()
                    attr.t.data_type = INT32
                    changed += 1

    for init in model.graph.initializer:
        if init.data_type in convertible and init.raw_data:
            raw = np.frombuffer(init.raw_data, dtype=_np_dt(init.data_type))
            init.raw_data = raw.astype(np.int32).tobytes()
            init.data_type = INT32
            changed += 1

    for vi in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        t = vi.type.tensor_type
        if t.elem_type in convertible:
            t.elem_type = INT32
            changed += 1

    onnx.save(model, onnx_path)
    return changed


def count_ops(onnx_path: str) -> Counter:
    m = onnx.load(onnx_path)
    return Counter(node.op_type for node in m.graph.node)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Part3 TV ablation ONNX variants.")
    parser.add_argument("checkpoint", type=str, help="Path to EdgeSAM checkpoint (.pth)")
    parser.add_argument("--output-dir", type=str, default="./part3_tv_ablation", help="Output directory")
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version")
    parser.add_argument("--no-constant-folding", action="store_true",
                        help="Disable ONNX constant folding during export")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    sam = sam_model_registry["edge_sam"](checkpoint=args.checkpoint, upsample_mode="bilinear")
    sam.eval()

    variants = {
        "p3_base": dict(),
        "p3_no_score": dict(no_score=True),
        "p3_no_deconv": dict(no_deconv=True),
        "p3_no_gemm": dict(no_gemm=True),
        "p3_no_tanh": dict(no_tanh=True),
        "p3_no_unsqueeze": dict(no_unsqueeze=True),
        "p3_minimal": dict(minimal=True),
    }

    print("Exporting Part3 TV ablation variants...")
    for name, cfg in variants.items():
        model = Part3Variant(sam, **cfg)
        out_path = os.path.join(args.output_dir, f"{name}.onnx")
        export_onnx(model, out_path, opset=args.opset, constant_folding=not args.no_constant_folding)
        changed = convert_int64_to_int32(out_path)
        if changed:
            print(f"    int64/int16->int32: {changed}")
        ops = count_ops(out_path)
        top = ", ".join(f"{k}({v})" for k, v in sorted(ops.items()))
        print(f"  {name:16s} -> {out_path}")
        print(f"    ops: {top}")

    print("Done.")


if __name__ == "__main__":
    main()
