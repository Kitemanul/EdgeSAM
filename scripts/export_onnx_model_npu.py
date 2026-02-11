"""
Export EdgeSAM to ONNX for NPU (opset 11).

The only difference from export_onnx_model.py is that the decoder export
uses SamCoreMLModelNPU, which replaces the OneHot op (unsupported by many
NPU compilers) with a Gather-based embedding lookup.

Usage:
    # Export encoder (same as original)
    python scripts/export_onnx_model_npu.py weights/edge_sam_3x.pth

    # Export decoder (OneHot replaced with Gather)
    python scripts/export_onnx_model_npu.py weights/edge_sam_3x.pth --decoder --use-stability-score
"""

import torch
import argparse
import numpy as np
from edge_sam import sam_model_registry
from edge_sam.utils.coreml import SamCoreMLModel


parser = argparse.ArgumentParser(
    description="Export EdgeSAM to ONNX for NPU (opset 11, no OneHot)."
)

parser.add_argument(
    "checkpoint", type=str, help="The path to the EdgeSAM model checkpoint."
)

parser.add_argument(
    "--gelu-approximate",
    action="store_true",
    help=(
        "Replace GELU operations with approximations using tanh. Useful "
        "for some runtimes that have slow or unimplemented erf ops, used in GELU."
    ),
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


# ============================================================
# NPU-compatible decoder wrapper (replaces OneHot with Gather)
# ============================================================

class SamCoreMLModelNPU(SamCoreMLModel):
    """Override _embed_points to avoid the OneHot op.

    The original _embed_points uses ``(point_labels == i) for i in range(4)``
    which the ONNX exporter fuses into an ``onnx.OneHot`` op.  Many NPU
    compilers cannot handle OneHot.

    This version builds a lookup table and uses ``torch.index_select``
    (ONNX Gather op) instead.
    """

    def _embed_points(self, point_coords: torch.Tensor, point_labels: torch.Tensor) -> torch.Tensor:
        point_coords = point_coords + 0.5
        point_coords = point_coords / self.img_size
        point_embedding = self.model.prompt_encoder.pe_layer._pe_encoding(point_coords)

        # Lookup table: [not_a_point, label_0, label_1, label_2, label_3]
        all_embeddings = torch.cat([
            self.model.prompt_encoder.not_a_point_embed.weight,
        ] + [
            pe.weight for pe in self.model.prompt_encoder.point_embeddings
        ], dim=0)  # [5, embed_dim]

        # Shift labels: -1 → 0, 0 → 1, …, 3 → 4
        indices = (point_labels + 1).to(torch.long)

        # Gather embeddings (no OneHot)
        flat_indices = indices.reshape(-1)
        selected = torch.index_select(all_embeddings, 0, flat_indices)
        selected = selected.reshape(indices.shape[0], indices.shape[1], -1)

        # Zero out position encoding for not-a-point (index 0)
        keep_mask = indices.to(point_embedding.dtype).clamp(min=0.0, max=1.0).unsqueeze(-1)

        point_embedding = point_embedding * keep_mask + selected
        return point_embedding


# ============================================================
# ONNX post-processing: convert int64 → int32
# ============================================================

def convert_int64_to_int32(onnx_path):
    """Convert all int64 occurrences to int32 in the ONNX model.

    NPU compilers (e.g. circle-quantizer) do not support int64.
    The main source is Cast nodes from PyTorch's index_select (requires
    torch.long / int64), but ONNX Gather works fine with int32.
    """
    import onnx
    from onnx import TensorProto, numpy_helper

    model = onnx.load(onnx_path)
    INT64 = TensorProto.INT64
    INT32 = TensorProto.INT32
    count = 0

    # Cast nodes: change target type
    for node in model.graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == INT64:
                    attr.i = INT32
                    count += 1

    # Constant nodes with int64 data
    for node in model.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value" and attr.t and attr.t.data_type == INT64:
                    arr = numpy_helper.to_array(attr.t).astype(np.int32)
                    attr.t.CopyFrom(numpy_helper.from_array(arr))
                    count += 1

    # Initializers
    for init in model.graph.initializer:
        if init.data_type == INT64:
            arr = numpy_helper.to_array(init).astype(np.int32)
            new_init = numpy_helper.from_array(arr, name=init.name)
            init.CopyFrom(new_init)
            count += 1

    # Graph inputs/outputs/value_info
    for vi in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        if vi.type.tensor_type.elem_type == INT64:
            vi.type.tensor_type.elem_type = INT32
            count += 1

    onnx.save(model, onnx_path)
    print(f"  Converted {count} int64 → int32 occurrences")


# ============================================================
# Export functions
# ============================================================

def export_encoder_to_onnx(sam, args):
    if args.gelu_approximate:
        for n, m in sam.named_modules():
            if isinstance(m, torch.nn.GELU):
                m.approximate = "tanh"

    image_input = torch.randn(1, 3, 1024, 1024, dtype=torch.float)
    sam.forward = sam.forward_dummy_encoder

    traced_model = torch.jit.trace(sam, image_input)

    input_names = ["image"]
    output_names = ["image_embeddings"]

    onnx_encoder_filename = args.checkpoint.replace('.pth', '_encoder_npu.onnx')
    torch.onnx.export(
        traced_model,
        image_input,
        onnx_encoder_filename,
        input_names=input_names,
        output_names=output_names,
        opset_version=11,
        verbose=False
    )

    print(f"Exported ONNX encoder model to {onnx_encoder_filename}")
    convert_int64_to_int32(onnx_encoder_filename)


def export_decoder_to_onnx(sam, args):
    sam_decoder = SamCoreMLModelNPU(
        model=sam,
        use_stability_score=args.use_stability_score
    )
    sam_decoder.eval()

    if args.gelu_approximate:
        for n, m in sam.named_modules():
            if isinstance(m, torch.nn.GELU):
                m.approximate = "tanh"

    embed_dim = sam.prompt_encoder.embed_dim
    embed_size = sam.prompt_encoder.image_embedding_size

    image_embeddings = torch.randn(1, embed_dim, *embed_size, dtype=torch.float)
    point_coords = torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float)
    point_labels = torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float)

    input_names = ["image_embeddings", "point_coords", "point_labels"]
    output_names = ["scores", "masks"]

    onnx_decoder_filename = args.checkpoint.replace('.pth', '_decoder_npu.onnx')
    torch.onnx.export(
        sam_decoder,
        (image_embeddings, point_coords, point_labels),
        onnx_decoder_filename,
        input_names=input_names,
        output_names=output_names,
        opset_version=11,
        dynamic_axes={
            "point_coords": {1: "num_points"},
            "point_labels": {1: "num_points"},
        },
        verbose=False
    )

    print(f"Exported ONNX decoder model to {onnx_decoder_filename}")
    convert_int64_to_int32(onnx_decoder_filename)


if __name__ == "__main__":
    args = parser.parse_args()
    print("Loading model...")
    sam = sam_model_registry["edge_sam"](checkpoint=args.checkpoint, upsample_mode="bilinear")
    sam.eval()

    if args.decoder:
        export_decoder_to_onnx(sam, args)
    else:
        export_encoder_to_onnx(sam, args)
