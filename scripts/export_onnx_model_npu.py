"""
Export EdgeSAM decoder to ONNX for NPU compilation (opset 11).

Replaces nn.LayerNorm and nn.GELU with manual implementations using only
basic ops (mean, sqrt, mul, add, tanh) to maximize NPU compiler compatibility.

Usage:
    # Export decoder
    python scripts/export_onnx_model_npu.py weights/edge_sam_3x.pth --decoder --use-stability-score

    # Export encoder
    python scripts/export_onnx_model_npu.py weights/edge_sam_3x.pth

    # Check ops in exported model
    python scripts/export_onnx_model_npu.py weights/edge_sam_3x.pth --decoder --check-ops-only
"""

import torch
import torch.nn as nn
import argparse
from edge_sam import sam_model_registry
from edge_sam.utils.coreml import SamCoreMLModel


parser = argparse.ArgumentParser(
    description="Export EdgeSAM to ONNX for NPU (opset 11, no LayerNorm/GELU)."
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
    "--check-ops-only", action="store_true",
    help="Only export and print all ONNX op types, do not save.",
)
parser.add_argument(
    "--output", type=str, default=None,
    help="Output ONNX file path. If not set, auto-generated from checkpoint path.",
)


class LayerNormManual(nn.Module):
    """Drop-in replacement for nn.LayerNorm using only basic arithmetic ops.

    Avoids the native LayerNormalization ONNX op (opset 17+) and ensures
    decomposition into ReduceMean/Sub/Pow/Sqrt/Mul/Add which are universally
    supported.
    """

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
        # Compute mean and variance over the last len(normalized_shape) dims
        dims = list(range(-len(self.normalized_shape), 0))
        mean = x.mean(dim=dims, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=dims, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        if self.elementwise_affine:
            x = x * self.weight + self.bias
        return x


class GELUManual(nn.Module):
    """GELU approximation using tanh, avoiding the Erf op.

    Uses: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """

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


def check_onnx_ops(onnx_path):
    """Print all unique op types in an ONNX model."""
    import onnx
    model = onnx.load(onnx_path)
    ops = set()
    for node in model.graph.node:
        ops.add(node.op_type)
    print(f"\nONNX model: {onnx_path}")
    print(f"Total nodes: {len(model.graph.node)}")
    print(f"Unique op types ({len(ops)}):")
    for op in sorted(ops):
        print(f"  - {op}")
    return ops


def _onnx_export(*args, **kwargs):
    """Call torch.onnx.export with legacy exporter for PyTorch >= 2.6 compatibility."""
    import torch
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
    return out_path


def export_decoder(sam, args):
    sam_decoder = SamCoreMLModel(
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

    image_embeddings = torch.randn(1, embed_dim, *embed_size, dtype=torch.float)
    point_coords = torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float)
    point_labels = torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float)

    out_path = args.output or args.checkpoint.replace('.pth', '_decoder_npu.onnx')
    _onnx_export(
        sam_decoder,
        (image_embeddings, point_coords, point_labels),
        out_path,
        input_names=["image_embeddings", "point_coords", "point_labels"],
        output_names=["scores", "masks"],
        opset_version=args.opset,
        dynamic_axes={
            "point_coords": {1: "num_points"},
            "point_labels": {1: "num_points"},
        },
        verbose=False,
    )
    print(f"Exported decoder to {out_path}")
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
