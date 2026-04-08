"""
Compare standard ONNX decoder vs NPU-optimized ONNX decoder via ONNX Runtime.

Both decoders are exported from the same PyTorch model weights, then run on
identical random inputs through onnxruntime.InferenceSession.  The output
masks and scores are compared to quantify any numerical deviation introduced
by the NPU compatibility changes.

Standard decoder (SamCoreMLModel -> ONNX):
    Inputs: image_embeddings [1,256,64,64]
            point_coords     [1,N,2]      (raw pixel coords)
            point_labels     [1,N]        (float: -1/0/1/2/3)
    PE computed inside the ONNX graph (Sin/Cos ops).
    Label lookup via Equal/Cast pattern.
    Indexing via Gather.
    Stability score via bool ReduceSum (if --use-stability-score).

NPU decoder (NpuSafeDecoder -> ONNX):
    Inputs: image_embeddings   [1,256,64,64]
            point_embedding_pe [1,N,256]   (PE pre-computed on CPU)
            point_labels       [1,N]       (float: -1/0/1/2/3)
    PE computed outside the ONNX graph (Fix 1).
    Label lookup via float arithmetic relu(1-diff^2) (Fix 2).
    Indexing via Slice+Reshape (Fix 3).
    Stability score via sigmoid approximation with k=50 (Fix 4).

Expected results:
    masks      < 1e-5  (Fix 1-3 are mathematically exact)
    IoU scores < 1e-5  (identical path when --no-stability-score)
    stability  ~ 1e-3  (sigmoid approximation, by design)
    rank order   0 mismatches  (best mask selection preserved)

Usage:
    # Export both ONNX files automatically, then compare
    python scripts/compare_onnx_accuracy.py weights/edge_sam_3x.pth

    # Use pre-exported ONNX files (skip re-export)
    python scripts/compare_onnx_accuracy.py weights/edge_sam_3x.pth \\
        --standard-onnx weights/edge_sam_3x_decoder_std.onnx \\
        --npu-onnx      weights/edge_sam_3x_decoder_npu.onnx

    # Test IoU score path (no sigmoid approximation)
    python scripts/compare_onnx_accuracy.py weights/edge_sam_3x.pth \\
        --no-stability-score

    # More trials / more points
    python scripts/compare_onnx_accuracy.py weights/edge_sam_3x.pth \\
        --num-trials 20 --num-points 5
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import onnxruntime as ort
except ImportError:
    print("ERROR: onnxruntime is required.  Install with: pip install onnxruntime")
    sys.exit(1)

import torch

from edge_sam import sam_model_registry
from edge_sam.utils.coreml import SamCoreMLModel
from scripts.export_onnx_model_npu import NpuSafeDecoder, _onnx_export


# ============================================================
# Export helpers
# ============================================================

def export_standard_onnx(sam, path, num_points, use_stability_score):
    """Export SamCoreMLModel (standard decoder) to ONNX."""
    decoder = SamCoreMLModel(model=sam, use_stability_score=use_stability_score)
    decoder.eval()

    embed_dim = sam.prompt_encoder.embed_dim
    embed_size = sam.prompt_encoder.image_embedding_size
    N = num_points

    dummy_embeddings = torch.randn(1, embed_dim, *embed_size, dtype=torch.float)
    dummy_coords = torch.randint(0, 1024, (1, N, 2), dtype=torch.float)
    dummy_labels = torch.randint(0, 4, (1, N), dtype=torch.float)

    with torch.no_grad():
        _onnx_export(
            decoder,
            (dummy_embeddings, dummy_coords, dummy_labels),
            path,
            input_names=["image_embeddings", "point_coords", "point_labels"],
            output_names=["scores", "masks"],
            opset_version=11,
            verbose=False,
        )
    print(f"  Exported standard decoder -> {path}")


def export_npu_onnx(sam, path, num_points, use_stability_score):
    """Export NpuSafeDecoder to ONNX."""
    decoder = NpuSafeDecoder(sam, use_stability_score=use_stability_score)
    decoder.eval()

    embed_dim = sam.prompt_encoder.embed_dim
    embed_size = sam.prompt_encoder.image_embedding_size
    N = num_points

    dummy_embeddings = torch.randn(1, embed_dim, *embed_size, dtype=torch.float)
    dummy_pe = torch.randn(1, N, embed_dim, dtype=torch.float)
    dummy_labels = torch.randint(0, 4, (1, N), dtype=torch.float)

    with torch.no_grad():
        _onnx_export(
            decoder,
            (dummy_embeddings, dummy_pe, dummy_labels),
            path,
            input_names=["image_embeddings", "point_embedding_pe", "point_labels"],
            output_names=["scores", "masks"],
            opset_version=11,
            verbose=False,
        )
    print(f"  Exported NPU decoder    -> {path}")


# ============================================================
# PE computation (numpy, mirrors compute_point_pe in npu script)
# ============================================================

def compute_pe_numpy(point_coords_np, gaussian_matrix_np, img_size=1024):
    """Compute point positional encoding in numpy.

    Replicates PositionEmbeddingRandom._pe_encoding + the +0.5/img_size
    offset applied in SamCoreMLModel._embed_points.

    Args:
        point_coords_np:   float32 [1, N, 2]  raw pixel coords
        gaussian_matrix_np: float32 [2, 128]  from prompt_encoder.pe_layer
        img_size:           int               model image size (1024)

    Returns:
        pe: float32 [1, N, 256]
    """
    coords = (point_coords_np + 0.5) / img_size   # [0, 1]
    coords = 2.0 * coords - 1.0                    # [-1, 1]
    coords = coords @ gaussian_matrix_np            # [1, N, 128]
    coords = 2.0 * np.pi * coords
    pe = np.concatenate([np.sin(coords), np.cos(coords)], axis=-1)  # [1, N, 256]
    return pe.astype(np.float32)


# ============================================================
# Comparison utilities
# ============================================================

def print_diff(name, a, b):
    diff = np.abs(a - b)
    print(f"    {name}:")
    print(f"      max_abs_diff  = {diff.max():.6e}")
    print(f"      mean_abs_diff = {diff.mean():.6e}")
    return float(diff.max())


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare standard vs NPU ONNX decoder via ONNX Runtime."
    )
    parser.add_argument("checkpoint", type=str,
                        help="Path to EdgeSAM .pth checkpoint")
    parser.add_argument("--standard-onnx", type=str, default=None,
                        help="Pre-exported standard ONNX decoder path "
                             "(auto-exported if omitted)")
    parser.add_argument("--npu-onnx", type=str, default=None,
                        help="Pre-exported NPU ONNX decoder path "
                             "(auto-exported if omitted)")
    parser.add_argument("--num-points", type=int, default=5,
                        help="Number of prompt points per trial (default: 5)")
    parser.add_argument("--num-trials", type=int, default=10,
                        help="Number of random input trials (default: 10)")
    parser.add_argument("--no-stability-score", action="store_true",
                        help="Use IoU prediction head instead of stability score "
                             "(removes Fix-4 approximation from the comparison)")
    args = parser.parse_args()

    use_stability_score = not args.no_stability_score
    N = args.num_points

    # ── Load PyTorch model ──────────────────────────────────────
    print(f"Loading model: {args.checkpoint}")
    sam = sam_model_registry["edge_sam"](
        checkpoint=args.checkpoint, upsample_mode="bilinear"
    )
    sam.eval()

    # ── Export ONNX files if not provided ───────────────────────
    base = args.checkpoint.replace(".pth", "")
    std_path = args.standard_onnx or f"{base}_decoder_std_cmp.onnx"
    npu_path = args.npu_onnx or f"{base}_decoder_npu_cmp.onnx"

    need_export = (not os.path.exists(std_path)) or (not os.path.exists(npu_path))
    if need_export:
        score_mode = "stability_score" if use_stability_score else "iou_score"
        print(f"\nExporting ONNX decoders (N={N}, score_mode={score_mode}) ...")

    if not os.path.exists(std_path):
        export_standard_onnx(sam, std_path, N, use_stability_score)
    else:
        print(f"  Standard ONNX already exists: {std_path}")

    if not os.path.exists(npu_path):
        export_npu_onnx(sam, npu_path, N, use_stability_score)
    else:
        print(f"  NPU ONNX already exists: {npu_path}")

    # ── Load ORT sessions ────────────────────────────────────────
    providers = ["CPUExecutionProvider"]
    std_sess = ort.InferenceSession(std_path, providers=providers)
    npu_sess = ort.InferenceSession(npu_path, providers=providers)

    print(f"\nORT sessions loaded.")
    print(f"  Standard inputs: {[i.name for i in std_sess.get_inputs()]}")
    print(f"  NPU inputs:      {[i.name for i in npu_sess.get_inputs()]}")

    # ── Extract Gaussian matrix for CPU-side PE computation ──────
    gaussian_matrix = (
        sam.prompt_encoder.pe_layer
        .positional_encoding_gaussian_matrix
        .detach().numpy()
    )  # shape: (2, 128)

    embed_dim = sam.prompt_encoder.embed_dim           # 256
    embed_size = sam.prompt_encoder.image_embedding_size  # (64, 64)
    img_size = sam.image_encoder.img_size              # 1024

    # ── Run comparison trials ────────────────────────────────────
    score_mode_str = "stability_score (Fix 4: sigmoid approx)" \
        if use_stability_score else "iou_prediction (exact)"
    print(f"\nRunning {args.num_trials} trials  |  N={N}  |  scores={score_mode_str}")
    print("=" * 64)

    mask_diffs, score_diffs, rank_mismatches = [], [], 0

    for trial in range(args.num_trials):
        np.random.seed(trial)
        torch.manual_seed(trial)

        # ── Generate random inputs ──
        image_embeddings = np.random.randn(1, embed_dim, *embed_size).astype(np.float32)
        point_coords = np.random.randint(0, 1024, (1, N, 2)).astype(np.float32)

        # Label pattern: covers -1 (padding), 0 (neg), 1 (pos)
        base_labels = [1, 1, 0, -1, -1]
        point_labels = np.array([base_labels[:N]], dtype=np.float32)

        # ── Standard ONNX: pass raw coords ──
        std_scores, std_masks = std_sess.run(None, {
            "image_embeddings": image_embeddings,
            "point_coords":     point_coords,
            "point_labels":     point_labels,
        })

        # ── NPU ONNX: pre-compute PE, then pass ──
        point_pe = compute_pe_numpy(point_coords, gaussian_matrix, img_size)
        npu_scores, npu_masks = npu_sess.run(None, {
            "image_embeddings":   image_embeddings,
            "point_embedding_pe": point_pe,
            "point_labels":       point_labels,
        })

        mask_diff = float(np.abs(std_masks - npu_masks).max())
        score_diff = float(np.abs(std_scores - npu_scores).max())
        mask_diffs.append(mask_diff)
        score_diffs.append(score_diff)

        std_best = int(np.argmax(std_scores))
        npu_best = int(np.argmax(npu_scores))
        if std_best != npu_best:
            rank_mismatches += 1

        if trial < 3:
            print(f"\nTrial {trial + 1}:")
            print_diff("masks (logits)", std_masks, npu_masks)
            print_diff("scores", std_scores, npu_scores)
            rank_str = "MATCH" if std_best == npu_best else "MISMATCH"
            print(f"    best mask index: std={std_best}, npu={npu_best}  [{rank_str}]")

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'=' * 64}")
    print(f"Summary over {args.num_trials} trials  (N={N}):")
    print(f"  masks  max_abs_diff:  max={max(mask_diffs):.3e}  "
          f"mean={np.mean(mask_diffs):.3e}")
    print(f"  scores max_abs_diff:  max={max(score_diffs):.3e}  "
          f"mean={np.mean(score_diffs):.3e}")
    print(f"  rank mismatches:      {rank_mismatches}/{args.num_trials}")

    print(f"\nDiagnosis:")

    # Mask check (Fixes 1-3 are exact, only float32 rounding expected)
    if max(mask_diffs) < 1e-4:
        print(f"  [PASS] Masks match within float32 precision  "
              f"(max={max(mask_diffs):.2e} < 1e-4).")
    else:
        print(f"  [FAIL] Unexpected mask deviation: max={max(mask_diffs):.3e}")

    # Score check
    if not use_stability_score:
        # IoU path: should also be exact
        if max(score_diffs) < 1e-4:
            print(f"  [PASS] IoU scores match within float32 precision  "
                  f"(max={max(score_diffs):.2e} < 1e-4).")
        else:
            print(f"  [FAIL] IoU score deviation: max={max(score_diffs):.3e}")
    else:
        # Stability score: sigmoid approx introduces ~1e-3 error by design
        if max(score_diffs) < 1e-4:
            print(f"  [PASS] Stability scores unexpectedly match exactly "
                  f"(max={max(score_diffs):.2e}).")
        else:
            print(f"  [EXPECTED] Stability score uses sigmoid approx (k=50), "
                  f"max diff={max(score_diffs):.3e} — this is by design.")

        if rank_mismatches == 0:
            print(f"  [PASS] Best mask selection identical across all trials  "
                  f"(0/{args.num_trials} rank mismatches).")
        else:
            print(f"  [WARN] {rank_mismatches}/{args.num_trials} trials selected "
                  f"a different best mask.")

    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()
