#!/usr/bin/env python
"""
Test that the 3-part ONNX decoder pipeline produces identical results
to the original PyTorch model (SamCoreMLModel path) for point-based segmentation.

Pipeline:  encoder (PyTorch) -> Part1 -> Part2 -> Part3 (ONNX)
Reference: encoder (PyTorch) -> SamCoreMLModel (PyTorch)

The 3-part split replicates SamCoreMLModel's behavior (no padding point),
NOT SamPredictor's behavior (which auto-adds a padding point).

Usage:
    python scripts/test_3part_pipeline.py weights/edge_sam_3x.pth truck.jpg
"""

import argparse
import os

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F

from edge_sam import sam_model_registry
from edge_sam.utils.coreml import SamCoreMLModel
from edge_sam.utils.transforms import ResizeLongestSide


def preprocess_image(image_rgb, img_size=1024):
    """Replicate SamPredictor.set_image + Sam.preprocess."""
    transform = ResizeLongestSide(img_size)
    input_image = transform.apply_image(image_rgb)
    original_size = image_rgb.shape[:2]
    input_size = input_image.shape[:2]

    input_tensor = torch.as_tensor(input_image, dtype=torch.float32)
    input_tensor = input_tensor.permute(2, 0, 1).contiguous()[None, :, :, :]

    # Normalize + pad (same as Sam.preprocess)
    pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
    pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)
    input_tensor = (input_tensor - pixel_mean) / pixel_std
    h, w = input_tensor.shape[-2:]
    input_tensor = F.pad(input_tensor, (0, img_size - w, 0, img_size - h))

    return input_tensor, original_size, input_size


def transform_coords(point_coords, original_size, img_size=1024):
    """Replicate ResizeLongestSide.apply_coords."""
    transform = ResizeLongestSide(img_size)
    return transform.apply_coords(point_coords, original_size)


def postprocess_masks(masks_np, input_size, original_size, img_size=1024):
    """Replicate Sam.postprocess_masks."""
    masks = torch.from_numpy(masks_np)
    masks = F.interpolate(masks, (img_size, img_size), mode="bilinear", align_corners=False)
    masks = masks[..., :input_size[0], :input_size[1]]
    masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
    return masks.numpy()


def run_reference(sam, image_embeddings, point_coords, point_labels):
    """Run SamCoreMLModel (PyTorch) as reference — no padding point."""
    coreml_model = SamCoreMLModel(model=sam, use_stability_score=True)
    coreml_model.eval()

    with torch.no_grad():
        scores, masks = coreml_model(
            image_embeddings,
            torch.from_numpy(point_coords).float(),
            torch.from_numpy(point_labels).float(),
        )
    return scores.numpy(), masks.numpy()


def run_3part_pipeline(onnx_dir, image_embeddings_np, point_coords, point_labels):
    """Run Part1 -> Part2 -> Part3 ONNX pipeline."""
    part1 = ort.InferenceSession(os.path.join(onnx_dir, "part1_prompt_encoding.onnx"))
    part2 = ort.InferenceSession(os.path.join(onnx_dir, "part2_transformer.onnx"))
    part3 = ort.InferenceSession(os.path.join(onnx_dir, "part3_mask_head.onnx"))

    coords_input = point_coords.astype(np.float32)
    labels_input = point_labels.astype(np.float32)

    # Part 1: Prompt Encoding
    sparse_embedding = part1.run(None, {
        "point_coords": coords_input,
        "point_labels": labels_input,
    })[0]

    # Part 2: Transformer
    hs, src = part2.run(None, {
        "image_embeddings": image_embeddings_np,
        "sparse_embedding": sparse_embedding,
    })

    # Part 3: Mask Head
    scores, masks = part3.run(None, {
        "hs": hs,
        "src": src,
    })

    return scores, masks


def main():
    parser = argparse.ArgumentParser(description="Test 3-part ONNX decoder pipeline")
    parser.add_argument("checkpoint", type=str, help="EdgeSAM checkpoint (.pth)")
    parser.add_argument("image", type=str, help="Test image path")
    parser.add_argument("--onnx-dir", type=str, default="./npu_diag",
                        help="Directory with part1/2/3 ONNX files")
    parser.add_argument("--num-points", type=int, default=5,
                        help="Pad prompts to this many points (must match ONNX export)")
    parser.add_argument("--save", type=str, default="test_3part_result.jpg",
                        help="Output visualization path")
    args = parser.parse_args()

    print("Loading model...")
    sam = sam_model_registry["edge_sam"](checkpoint=args.checkpoint, upsample_mode="bilinear")
    sam.eval()

    image_bgr = cv2.imread(args.image)
    assert image_bgr is not None, f"Cannot read image: {args.image}"
    image_rgb = image_bgr[..., ::-1]
    h, w = image_bgr.shape[:2]
    print(f"Image: {args.image} ({w}x{h})")

    # Encoder (PyTorch, shared by both paths)
    input_tensor, original_size, input_size = preprocess_image(image_rgb)
    with torch.no_grad():
        image_embeddings = sam.image_encoder(input_tensor)
    image_embeddings_np = image_embeddings.numpy()
    print(f"Encoder output: {image_embeddings.shape}")

    N = args.num_points

    # Test prompts (in original image coords, X,Y format)
    raw_prompts = [
        # Single foreground point (truck body)
        (np.array([[500, 375]]), np.array([1])),
        # Single foreground point (truck wheel)
        (np.array([[600, 500]]), np.array([1])),
        # Foreground + background
        (np.array([[500, 375], [100, 100]]), np.array([1, 0])),
        # Box prompt (top-left=label2, bottom-right=label3)
        (np.array([[200, 200], [700, 550]]), np.array([2, 3])),
    ]

    print(f"\n{'='*60}")
    print("Comparison: SamCoreMLModel (PyTorch) vs 3-Part ONNX")
    print(f"{'='*60}")

    all_pass = True
    vis_rows = []

    for idx, (raw_coords, raw_labels) in enumerate(raw_prompts):
        # Transform coords to model space
        transformed_coords = transform_coords(raw_coords.copy(), original_size)
        n = transformed_coords.shape[0]

        # Pad to N points with label=-1
        if n < N:
            pad_coords = np.zeros((N - n, 2), dtype=np.float64)
            pad_labels = np.full(N - n, -1, dtype=raw_labels.dtype)
            coords_padded = np.concatenate([transformed_coords, pad_coords], axis=0)
            labels_padded = np.concatenate([raw_labels, pad_labels], axis=0)
        else:
            coords_padded = transformed_coords
            labels_padded = raw_labels

        # Shape: (1, N, 2) and (1, N)
        coords_batch = coords_padded[None]
        labels_batch = labels_padded[None]

        # Reference (SamCoreMLModel, PyTorch)
        ref_scores, ref_masks = run_reference(sam, image_embeddings, coords_batch, labels_batch)

        # Pipeline (3-part ONNX)
        pipe_scores, pipe_masks = run_3part_pipeline(
            args.onnx_dir, image_embeddings_np, coords_batch, labels_batch
        )

        # Compare scores
        ref_s = ref_scores[0]
        pipe_s = pipe_scores[0]
        score_diff = np.max(np.abs(ref_s - pipe_s))

        ref_best = np.argmax(ref_s)
        pipe_best = np.argmax(pipe_s)
        same_best = ref_best == pipe_best

        # Compare best masks (low-res 256x256)
        ref_m = (ref_masks[0, ref_best] > 0.0).astype(bool)
        pipe_m = (pipe_masks[0, pipe_best] > 0.0).astype(bool)
        intersection = np.logical_and(ref_m, pipe_m).sum()
        union = np.logical_or(ref_m, pipe_m).sum()
        iou = intersection / union if union > 0 else 1.0

        # Also compare raw mask values
        mask_abs_diff = np.max(np.abs(ref_masks - pipe_masks))

        n_actual = np.sum(raw_labels != -1)
        prompt_desc = f"coords={raw_coords[:n_actual].tolist()}, labels={raw_labels[:n_actual].tolist()}"
        print(f"\nPrompt {idx+1}: {prompt_desc}")
        print(f"  Score max diff:    {score_diff:.6f}")
        print(f"  Mask max diff:     {mask_abs_diff:.6f}")
        print(f"  Best mask index:   ref={ref_best}, pipe={pipe_best} ({'MATCH' if same_best else 'MISMATCH'})")
        print(f"  Best mask IoU:     {iou:.6f}")
        print(f"  Ref scores:  {ref_s}")
        print(f"  Pipe scores: {pipe_s}")

        if score_diff > 0.01 or iou < 0.99:
            all_pass = False
            print(f"  STATUS: FAIL")
        else:
            print(f"  STATUS: PASS")

        # Collect for visualization
        ref_masks_full = postprocess_masks(ref_masks, input_size, original_size)
        pipe_masks_full = postprocess_masks(pipe_masks, input_size, original_size)
        vis_rows.append((raw_coords, raw_labels, ref_s, pipe_s,
                         ref_masks_full[0], pipe_masks_full[0]))

    print(f"\n{'='*60}")
    if all_pass:
        print("ALL TESTS PASSED - 3-part pipeline matches SamCoreMLModel")
    else:
        print("SOME TESTS FAILED - check details above")
    print(f"{'='*60}")

    # Visualization
    n_prompts = len(vis_rows)
    canvas = np.zeros((h * n_prompts, w * 3, 3), dtype=np.uint8)

    for i, (coords, labels, ref_s, pipe_s, ref_masks_full, pipe_masks_full) in enumerate(vis_rows):
        ref_best = np.argmax(ref_s)
        pipe_best = np.argmax(pipe_s)

        img_ref = image_bgr.copy()
        mask_r = ref_masks_full[ref_best] > 0
        img_ref[mask_r] = (img_ref[mask_r] * 0.5 + np.array([0, 128, 0]) * 0.5).astype(np.uint8)

        img_pipe = image_bgr.copy()
        mask_p = pipe_masks_full[pipe_best] > 0
        img_pipe[mask_p] = (img_pipe[mask_p] * 0.5 + np.array([0, 0, 200]) * 0.5).astype(np.uint8)

        img_diff = image_bgr.copy()
        diff = np.abs(mask_r.astype(float) - mask_p.astype(float))
        img_diff[diff > 0.5] = [0, 0, 255]

        for img in [img_ref, img_pipe, img_diff]:
            for pt, lbl in zip(coords, labels):
                if lbl == -1:
                    continue
                color = {0: (0, 0, 255), 1: (0, 255, 0), 2: (255, 0, 0), 3: (255, 0, 0)}.get(lbl, (255, 255, 0))
                cv2.circle(img, (int(pt[0]), int(pt[1])), 8, color, -1)

        cv2.putText(img_ref, f"Reference (score={ref_s[ref_best]:.4f})",
                     (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img_pipe, f"3-Part ONNX (score={pipe_s[pipe_best]:.4f})",
                     (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img_diff, "Diff (red=mismatch)",
                     (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        row = i * h
        canvas[row:row+h, 0:w] = img_ref
        canvas[row:row+h, w:2*w] = img_pipe
        canvas[row:row+h, 2*w:3*w] = img_diff

    cv2.imwrite(args.save, canvas)
    print(f"Visualization saved to {args.save}")


if __name__ == "__main__":
    main()
