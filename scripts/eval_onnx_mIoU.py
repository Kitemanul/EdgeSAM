import argparse
import json
import os
import random

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from tqdm import tqdm

import onnxruntime

from edge_sam.utils.transforms import ResizeLongestSide


IMG_SIZE = 1024
PIXEL_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)[None, :, None, None]
PIXEL_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)[None, :, None, None]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ONNX EdgeSAM models (mIoU) with point prompts on COCO-format datasets.")
    parser.add_argument("--encoder", type=str, required=True, help="Path to encoder ONNX model")
    parser.add_argument("--decoder", type=str, required=True, help="Path to decoder ONNX model")
    parser.add_argument("--data-root", type=str, required=True, help="Root directory of COCO-format dataset")
    parser.add_argument("--split", type=str, default="val", help="Dataset split (default: val)")
    parser.add_argument("--annotation", type=str, default="coco", choices=["coco", "lvis", "cocofied_lvis"],
                        help="Annotation format (default: coco)")
    parser.add_argument("--anno-file", type=str, default=None,
                        help="Direct path to annotation JSON file (overrides --annotation and --split)")
    parser.add_argument("--img-dir", type=str, default=None,
                        help="Image directory relative to data-root (default: '{split}2017' for coco, 'trainval' otherwise)")
    parser.add_argument("--num-samples", type=int, default=-1, help="Number of images to evaluate (-1 for all)")
    parser.add_argument("--num-points", type=int, default=1, help="Number of point prompts per object")
    parser.add_argument("--point-strategy", type=str, default="random", choices=["random", "center"],
                        help="Point sampling strategy: 'random' samples from GT mask, 'center' uses mask centroid")
    parser.add_argument("--mask-select", type=str, default="score", choices=["score", "area"],
                        help="Mask selection strategy when multi-mask output: 'score' or 'area'")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--max-prompts-per-image", type=int, default=-1,
                        help="Max number of object prompts per image (-1 for all)")
    return parser.parse_args()


def load_coco_data(data_root, split, annotation, num_samples, anno_file=None):
    """Load COCO-format annotations and return list of (img_info, [anno_info])."""
    if anno_file is not None:
        anno_path = anno_file
    elif annotation == "coco":
        anno_path = os.path.join(data_root, "annotations", f"instances_{split}2017.json")
    elif annotation == "cocofied_lvis":
        anno_path = os.path.join(data_root, "annotations", f"lvis_v1_{split}_cocofied.json")
    elif annotation == "lvis":
        anno_path = os.path.join(data_root, "annotations", f"lvis_v1_{split}.json")

    with open(anno_path, "r") as f:
        anno_json = json.load(f)

    imgs = {}
    for img_info in anno_json["images"]:
        imgs[img_info["id"]] = img_info

    annos = {}
    for anno_info in anno_json["annotations"]:
        if anno_info.get("iscrowd", 0):
            continue
        img_id = anno_info["image_id"]
        if img_id not in annos:
            annos[img_id] = []
        annos[img_id].append(anno_info)

    data = []
    for img_id, img_info in imgs.items():
        if img_id not in annos:
            continue
        file_name = img_info.get("file_name") or img_info["coco_url"].split("/")[-1]
        data.append((img_info, file_name, annos[img_id]))
        if 0 < num_samples <= len(data):
            break

    return data


def decode_gt_mask(segm, h, w):
    """Decode COCO segmentation annotation to binary mask."""
    if isinstance(segm, list):
        rles = mask_utils.frPyObjects(segm, h, w)
        rle = mask_utils.merge(rles)
    elif isinstance(segm["counts"], list):
        rle = mask_utils.frPyObjects(segm, h, w)
    else:
        rle = segm
    return mask_utils.decode(rle)


def sample_points_from_mask(gt_mask, num_points, strategy, rng):
    """Sample point prompts from a GT binary mask.

    Args:
        gt_mask: (H, W) binary numpy array
        num_points: number of points to sample
        strategy: 'random' or 'center'
        rng: numpy random generator

    Returns:
        coords: (num_points, 2) array in (x, y) format
        labels: (num_points,) array, all ones (positive points)
    """
    if strategy == "center":
        # Use distance transform to find the point farthest from boundary
        mask_uint8 = gt_mask.astype(np.uint8)
        dt = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
        max_idx = np.unravel_index(dt.argmax(), dt.shape)
        center = np.array([[max_idx[1], max_idx[0]]], dtype=np.float32)  # (x, y)
        if num_points == 1:
            coords = center
        else:
            # First point is center, rest are random
            ys, xs = np.where(gt_mask > 0)
            extra = num_points - 1
            if len(ys) >= extra:
                sel = rng.choice(len(ys), size=extra, replace=False)
            else:
                sel = rng.choice(len(ys), size=extra, replace=True)
            extra_coords = np.stack([xs[sel], ys[sel]], axis=1).astype(np.float32)
            coords = np.concatenate([center, extra_coords], axis=0)
    else:  # random
        ys, xs = np.where(gt_mask > 0)
        if len(ys) == 0:
            coords = np.zeros((num_points, 2), dtype=np.float32)
            labels = np.ones(num_points, dtype=np.float32)
            return coords, labels
        if len(ys) >= num_points:
            sel = rng.choice(len(ys), size=num_points, replace=False)
        else:
            sel = rng.choice(len(ys), size=num_points, replace=True)
        coords = np.stack([xs[sel], ys[sel]], axis=1).astype(np.float32)

    labels = np.ones(num_points, dtype=np.float32)
    return coords, labels


def preprocess_image(image_np, transform):
    """Preprocess image: resize, normalize, pad to 1024x1024.

    Args:
        image_np: (H, W, 3) uint8 RGB image
        transform: ResizeLongestSide instance

    Returns:
        input_image: (1, 3, 1024, 1024) float32 array
        input_size: (H', W') after resize before padding
        original_size: (H, W) original image size
    """
    original_size = image_np.shape[:2]
    resized = transform.apply_image(image_np)
    input_size = resized.shape[:2]

    # HWC -> CHW, add batch dim
    x = resized.transpose(2, 0, 1)[None, :, :, :].astype(np.float32)

    # Normalize
    x = (x - PIXEL_MEAN) / PIXEL_STD

    # Pad to 1024x1024
    h, w = x.shape[-2:]
    padh = IMG_SIZE - h
    padw = IMG_SIZE - w
    x = np.pad(x, ((0, 0), (0, 0), (0, padh), (0, padw)), mode="constant", constant_values=0)

    return x, input_size, original_size


def postprocess_mask(low_res_mask, input_size, original_size):
    """Upscale low-res mask to original image resolution.

    Args:
        low_res_mask: (N, 256, 256) array
        input_size: (H', W') pre-pad size
        original_size: (H, W) original image size

    Returns:
        mask: (N, H, W) binary mask at original resolution
    """
    n = low_res_mask.shape[0]
    masks = []
    for i in range(n):
        m = low_res_mask[i]  # (256, 256)
        # Upscale to 1024x1024
        m = cv2.resize(m, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        # Crop to input_size (remove padding)
        m = m[:input_size[0], :input_size[1]]
        # Resize to original
        m = cv2.resize(m, (original_size[1], original_size[0]), interpolation=cv2.INTER_LINEAR)
        masks.append(m)
    return np.stack(masks, axis=0)


def cal_iou(pred_mask, gt_mask):
    """Calculate IoU between predicted and GT binary masks.

    Args:
        pred_mask: (H, W) binary mask
        gt_mask: (H, W) binary mask

    Returns:
        iou: float
    """
    pred_bin = pred_mask > 0.0
    gt_bin = gt_mask > 0.5
    intersect = (pred_bin & gt_bin).sum()
    union = (pred_bin | gt_bin).sum()
    if union == 0:
        return 1.0 if intersect == 0 else 0.0
    return float(intersect) / float(union)


def select_mask(masks, scores, strategy):
    """Select one mask from multi-mask output.

    Args:
        masks: (M, H, W) array of masks
        scores: (M,) array of scores
        strategy: 'score' or 'area'

    Returns:
        selected_mask: (H, W) array
    """
    if masks.shape[0] == 1:
        return masks[0]

    if strategy == "score":
        idx = scores.argmax()
    else:  # area
        areas = (masks > 0.0).reshape(masks.shape[0], -1).sum(axis=1)
        idx = areas.argmax()
    return masks[idx]


def main():
    args = parse_args()
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # Determine image directory
    if args.img_dir is not None:
        img_dir = os.path.join(args.data_root, args.img_dir)
    else:
        img_dir = os.path.join(args.data_root, f"{args.split}2017")
        if not os.path.isdir(img_dir):
            img_dir = os.path.join(args.data_root, "trainval")

    # Load data
    print("Loading annotations...")
    data = load_coco_data(args.data_root, args.split, args.annotation, args.num_samples, args.anno_file)
    print(f"Loaded {len(data)} images")

    # Create ONNX sessions
    print("Loading ONNX models...")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
        if "CUDAExecutionProvider" in onnxruntime.get_available_providers() \
        else ["CPUExecutionProvider"]
    encoder_session = onnxruntime.InferenceSession(args.encoder, providers=providers)
    decoder_session = onnxruntime.InferenceSession(args.decoder, providers=providers)
    print(f"Using providers: {encoder_session.get_providers()}")

    transform = ResizeLongestSide(IMG_SIZE)

    all_ious = []
    num_objects = 0

    for img_info, file_name, anno_list in tqdm(data, desc="Evaluating"):
        # Load image
        img_path = os.path.join(img_dir, file_name)
        if not os.path.exists(img_path):
            continue
        image = np.array(Image.open(img_path).convert("RGB"))
        h, w = img_info["height"], img_info["width"]

        # Preprocess and encode
        input_image, input_size, original_size = preprocess_image(image, transform)
        features = encoder_session.run(None, {"image": input_image})[0]

        # Decode GT masks and sample prompts
        gt_masks = []
        for record in anno_list:
            gt_mask = decode_gt_mask(record["segmentation"], h, w)
            gt_masks.append(gt_mask)

        if args.max_prompts_per_image > 0 and len(gt_masks) > args.max_prompts_per_image:
            selected = rng.choice(len(gt_masks), size=args.max_prompts_per_image, replace=False)
            gt_masks = [gt_masks[i] for i in selected]

        # Evaluate each object
        for gt_mask in gt_masks:
            if gt_mask.sum() == 0:
                continue

            # Sample point prompts
            coords, labels = sample_points_from_mask(gt_mask, args.num_points, args.point_strategy, rng)

            # Transform coordinates to model input space
            point_coords = transform.apply_coords(coords, original_size)
            point_coords = point_coords[None, :, :].astype(np.float32)  # (1, N, 2)
            point_labels = labels[None, :].astype(np.float32)  # (1, N)

            # Decode
            scores, low_res_masks = decoder_session.run(None, {
                "image_embeddings": features,
                "point_coords": point_coords,
                "point_labels": point_labels,
            })

            # scores: (1, M), low_res_masks: (1, M, 256, 256)
            scores = scores[0]  # (M,)
            low_res_masks = low_res_masks[0]  # (M, 256, 256)

            # Postprocess masks to original resolution
            masks = postprocess_mask(low_res_masks, input_size, original_size)

            # Select best mask
            pred_mask = select_mask(masks, scores, args.mask_select)

            # Calculate IoU
            iou = cal_iou(pred_mask, gt_mask)
            all_ious.append(iou)
            num_objects += 1

    # Report results
    if len(all_ious) == 0:
        print("No valid objects found for evaluation.")
        return

    all_ious = np.array(all_ious)
    print(f"\n{'=' * 50}")
    print(f"Results:")
    print(f"  Images:           {len(data)}")
    print(f"  Objects:          {num_objects}")
    print(f"  Point strategy:   {args.point_strategy}")
    print(f"  Num points:       {args.num_points}")
    print(f"  Mask selection:   {args.mask_select}")
    print(f"  mIoU:             {all_ious.mean():.4f}")
    print(f"  Median IoU:       {np.median(all_ious):.4f}")
    print(f"  IoU std:          {all_ious.std():.4f}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
