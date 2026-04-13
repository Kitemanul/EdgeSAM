import torch
import argparse
from edge_sam import sam_model_registry
from edge_sam.utils.coreml import SamCoreMLModel


parser = argparse.ArgumentParser(
    description="Export the EdgeSAM to ONNX models."
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

parser.add_argument(
    "--encoder-input-size",
    type=int,
    default=1024,
    help="Encoder input image size (square). E.g. 512 exports an encoder with input [1,3,512,512].",
)

parser.add_argument(
    "--decoder-embed-size",
    type=int,
    nargs=2,
    metavar=("H", "W"),
    default=None,
    help=(
        "Override decoder image embedding size (H, W). "
        "Use this when encoder output embedding size changed, e.g. --decoder-embed-size 32 32."
    ),
)


def export_encoder_to_onnx(sam, args):
    if args.gelu_approximate:
        for n, m in sam.named_modules():
            if isinstance(m, torch.nn.GELU):
                m.approximate = "tanh"

    image_input = torch.randn(
        1, 3, args.encoder_input_size, args.encoder_input_size, dtype=torch.float
    )
    sam.forward = sam.forward_dummy_encoder

    # Define the input names and output names
    input_names = ["image"]
    output_names = ["image_embeddings"]

    # Export the encoder model to ONNX format
    onnx_encoder_filename = args.checkpoint.replace('.pth', '_encoder.onnx')
    torch.onnx.export(
        sam,
        image_input,
        onnx_encoder_filename,
        input_names=input_names,
        output_names=output_names,
        dynamo=False,
        verbose=False
    )

    print(f"Exported ONNX encoder model to {onnx_encoder_filename}")


def export_decoder_to_onnx(sam, args):
    sam_decoder = SamCoreMLModel(
        model=sam,
        use_stability_score=args.use_stability_score
    )
    sam_decoder.eval()

    if args.gelu_approximate:
        for n, m in sam.named_modules():
            if isinstance(m, torch.nn.GELU):
                m.approximate = "tanh"

    embed_dim = sam.prompt_encoder.embed_dim
    embed_size = (
        tuple(args.decoder_embed_size)
        if args.decoder_embed_size is not None
        else sam.prompt_encoder.image_embedding_size
    )

    image_embeddings = torch.randn(1, embed_dim, *embed_size, dtype=torch.float)
    point_coords = torch.randint(
        low=0, high=args.encoder_input_size, size=(1, 5, 2), dtype=torch.float
    )
    point_labels = torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float)

    # Define the input names and output names
    input_names = ["image_embeddings", "point_coords", "point_labels"]
    output_names = ["scores", "masks"]

    # Export the decoder model to ONNX format
    onnx_decoder_filename = args.checkpoint.replace('.pth', '_decoder.onnx')
    torch.onnx.export(
        sam_decoder,
        (image_embeddings, point_coords, point_labels),
        onnx_decoder_filename,
        input_names=input_names,
        output_names=output_names,
        dynamo=False,
        dynamic_axes={
            "point_coords": {1: "num_points"},
            "point_labels": {1: "num_points"},
        },
        verbose=False
    )

    print(f"Exported ONNX decoder model to {onnx_decoder_filename}")


if __name__ == "__main__":
    args = parser.parse_args()
    print("Loading model...")
    sam = sam_model_registry["edge_sam"](checkpoint=args.checkpoint, upsample_mode="bilinear")
    sam.eval()

    if args.decoder:
        export_decoder_to_onnx(sam, args)
    else:
        export_encoder_to_onnx(sam, args)
