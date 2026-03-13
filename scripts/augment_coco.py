"""
Offline COCO data augmentation script.

Reads a COCO-format dataset (compressed-RLE segmentation masks), applies the
requested augmentation methods, writes augmented images to disk, and outputs a
new COCO JSON that contains the original samples plus all augmented ones.

Supported methods (--methods):
    hflip         horizontal flip
    vflip         vertical flip
    rotation      random rotation ±ROTATION_RANGE degrees
    crop          random crop (scale in [CROP_SCALE_MIN, 1.0]), repeated CROP_REPEAT times
    color_jitter  brightness/contrast/saturation/hue jitter (image only)
    noise         additive Gaussian noise (image only)
    mosaic        4-image mosaic, produces MOSAIC_COUNT new images

Usage:
    python scripts/augment_coco.py \
        --img-dir  data/images/ \
        --ann-file data/annotations.json \
        --output-dir data_aug/ \
        --methods hflip color_jitter mosaic crop \
        --mosaic-count 500 \
        --rotation-range 15 \
        --crop-scale-min 0.5 \
        --crop-repeat 2 \
        --seed 42
"""

import argparse
import json
import os
import random
import shutil
import sys

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from torchvision.transforms import ColorJitter

# ─────────────────────────────────────────────────────────────────────────────
# RLE helpers
# ─────────────────────────────────────────────────────────────────────────────

def _decode_segm(segm, h, w):
    """Return a (H, W) uint8 binary mask for any COCO segmentation format."""
    if isinstance(segm, list):
        # polygon
        rles = mask_utils.frPyObjects(segm, h, w)
        rle = mask_utils.merge(rles)
    elif isinstance(segm, dict):
        if isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = mask_utils.frPyObjects(segm, h, w)
        else:
            # compressed RLE — the common case for this script
            rle = segm
    else:
        raise ValueError(f"Unknown segmentation format: {type(segm)}")
    return mask_utils.decode(rle)  # uint8 (H, W)


def _encode_mask(mask_arr):
    """Encode a (H, W) uint8 binary mask to a JSON-serialisable compressed RLE."""
    rle = mask_utils.encode(np.asfortranarray(mask_arr))
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle


def _rle_area(rle_dict):
    """area from a serialisable RLE dict (counts as str)."""
    rle = {**rle_dict, 'counts': rle_dict['counts'].encode('utf-8')}
    return int(mask_utils.area(rle))


def _rle_bbox(rle_dict):
    """xywh bbox from a serialisable RLE dict."""
    rle = {**rle_dict, 'counts': rle_dict['counts'].encode('utf-8')}
    x, y, w, h = mask_utils.toBbox(rle).tolist()
    return [x, y, w, h]


# ─────────────────────────────────────────────────────────────────────────────
# Per-annotation transform helpers
# ─────────────────────────────────────────────────────────────────────────────

def _transform_annos(annos, img_h, img_w, new_h, new_w,
                     mask_transform_fn, new_image_id, id_counter):
    """
    Apply mask_transform_fn to every annotation mask.
    Returns a list of new annotation dicts (empty masks are dropped).
    id_counter is a list[int] used as a mutable counter.
    """
    new_annos = []
    for ann in annos:
        mask = _decode_segm(ann['segmentation'], img_h, img_w)
        new_mask = mask_transform_fn(mask)
        if new_mask.sum() == 0:
            continue  # object was cropped/rotated out of frame
        rle = _encode_mask(new_mask.astype(np.uint8))
        new_ann = {
            'id': id_counter[0],
            'image_id': new_image_id,
            'category_id': ann['category_id'],
            'segmentation': rle,
            'area': _rle_area(rle),
            'bbox': _rle_bbox(rle),
            'iscrowd': ann.get('iscrowd', 0),
        }
        id_counter[0] += 1
        new_annos.append(new_ann)
    return new_annos


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation functions
# Each returns (new_img_bgr, mask_transform_fn, new_h, new_w)
# mask_transform_fn: (H,W) uint8 → (new_H, new_W) uint8
# ─────────────────────────────────────────────────────────────────────────────

def aug_hflip(img_bgr):
    H, W = img_bgr.shape[:2]
    new_img = cv2.flip(img_bgr, 1)
    def tfm(mask):
        return cv2.flip(mask, 1)
    return new_img, tfm, H, W


def aug_vflip(img_bgr):
    H, W = img_bgr.shape[:2]
    new_img = cv2.flip(img_bgr, 0)
    def tfm(mask):
        return cv2.flip(mask, 0)
    return new_img, tfm, H, W


def aug_rotation(img_bgr, angle_range):
    H, W = img_bgr.shape[:2]
    angle = random.uniform(-angle_range, angle_range)
    cx, cy = W / 2, H / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    new_img = cv2.warpAffine(img_bgr, M, (W, H),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)
    def tfm(mask):
        return cv2.warpAffine(mask, M, (W, H),
                              flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return new_img, tfm, H, W


def aug_crop(img_bgr, scale_min):
    H, W = img_bgr.shape[:2]
    scale = random.uniform(scale_min, 1.0)
    new_h = int(H * scale)
    new_w = int(W * scale)
    y0 = random.randint(0, H - new_h)
    x0 = random.randint(0, W - new_w)
    new_img = img_bgr[y0:y0 + new_h, x0:x0 + new_w]
    def tfm(mask):
        return mask[y0:y0 + new_h, x0:x0 + new_w]
    return new_img, tfm, new_h, new_w


def aug_color_jitter(img_bgr, jitter_transform):
    """Color jitter via torchvision (image only; mask unchanged)."""
    import torch
    # BGR → RGB tensor (C,H,W) uint8
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img_rgb).permute(2, 0, 1)
    t = jitter_transform(t)
    new_img = cv2.cvtColor(t.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)
    H, W = new_img.shape[:2]
    def tfm(mask):
        return mask
    return new_img, tfm, H, W


def aug_noise(img_bgr, sigma_range=(5, 25)):
    H, W = img_bgr.shape[:2]
    sigma = random.uniform(*sigma_range)
    noise = np.random.randn(*img_bgr.shape) * sigma
    new_img = np.clip(img_bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    def tfm(mask):
        return mask
    return new_img, tfm, H, W


# ─────────────────────────────────────────────────────────────────────────────
# Mosaic (4-image grid)
# ─────────────────────────────────────────────────────────────────────────────

def _make_mosaic(entries, target_size, img_dir):
    """
    entries: list of 4 (img_info, annos) tuples
    Returns (canvas_bgr, new_annotations_list, canvas_h, canvas_w)
    The caller assigns image_id and annotation ids.
    """
    th, tw = target_size  # full mosaic size
    hh, hw = th // 2, tw // 2  # half sizes
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)

    # Quadrant offsets (y_offset, x_offset, cell_h, cell_w)
    quads = [
        (0,  0,  hh, hw),   # top-left
        (0,  hw, hh, tw - hw),  # top-right
        (hh, 0,  th - hh, hw),  # bottom-left
        (hh, hw, th - hh, tw - hw),  # bottom-right
    ]

    new_annos_all = []
    for (img_info, annos), (qy, qx, qh, qw) in zip(entries, quads):
        file_name = img_info.get('file_name') or img_info['coco_url'].split('/')[-1]
        img_path = os.path.join(img_dir, file_name)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        orig_h, orig_w = img_bgr.shape[:2]

        # Resize tile to fit quadrant
        tile = cv2.resize(img_bgr, (qw, qh), interpolation=cv2.INTER_LINEAR)
        canvas[qy:qy + qh, qx:qx + qw] = tile

        scale_y = qh / orig_h
        scale_x = qw / orig_w

        for ann in annos:
            mask = _decode_segm(ann['segmentation'], orig_h, orig_w)
            # Resize mask to quadrant size then paste into full canvas coords
            tile_mask = cv2.resize(mask, (qw, qh), interpolation=cv2.INTER_NEAREST)
            full_mask = np.zeros((th, tw), dtype=np.uint8)
            full_mask[qy:qy + qh, qx:qx + qw] = tile_mask
            if full_mask.sum() == 0:
                continue
            new_annos_all.append((ann['category_id'], full_mask, ann.get('iscrowd', 0)))

    return canvas, new_annos_all, th, tw


# ─────────────────────────────────────────────────────────────────────────────
# ID counter helper
# ─────────────────────────────────────────────────────────────────────────────

class Counter:
    def __init__(self, start):
        self.val = start

    def next(self):
        v = self.val
        self.val += 1
        return v


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Offline COCO data augmentation')
    p.add_argument('--img-dir',  required=True, help='Directory containing source images')
    p.add_argument('--ann-file', required=True, help='COCO annotation JSON file')
    p.add_argument('--output-dir', required=True, help='Output root directory')
    p.add_argument('--methods', nargs='+',
                   choices=['hflip', 'vflip', 'rotation', 'crop',
                            'color_jitter', 'noise', 'mosaic'],
                   default=['hflip', 'vflip', 'rotation', 'crop',
                            'color_jitter', 'noise', 'mosaic'],
                   help='Augmentation methods to apply')
    p.add_argument('--rotation-range', type=float, default=15.0,
                   help='Max rotation angle in degrees (default: 15)')
    p.add_argument('--crop-scale-min', type=float, default=0.5,
                   help='Min crop scale relative to original size (default: 0.5)')
    p.add_argument('--crop-repeat', type=int, default=2,
                   help='Number of random crops per image (default: 2)')
    p.add_argument('--mosaic-count', type=int, default=500,
                   help='Number of mosaic images to generate (default: 500)')
    p.add_argument('--noise-sigma-min', type=float, default=5.0)
    p.add_argument('--noise-sigma-max', type=float, default=25.0)
    p.add_argument('--keep-originals', action='store_true', default=True,
                   help='Copy original images into output dir (default: True)')
    p.add_argument('--no-keep-originals', dest='keep_originals', action='store_false')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--jpg-quality', type=int, default=95,
                   help='JPEG quality for saved images (default: 95)')
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ── Load annotation JSON ──────────────────────────────────────────────────
    print(f'Loading annotations from {args.ann_file} ...')
    with open(args.ann_file, 'r') as f:
        coco_json = json.load(f)

    images_meta = {img['id']: img for img in coco_json['images']}

    # Group annotations by image_id (skip crowd for geometric transforms)
    annos_by_img = {}
    for ann in coco_json['annotations']:
        iid = ann['image_id']
        annos_by_img.setdefault(iid, []).append(ann)

    # Build ordered list of (img_info, annos) for images that have annotations
    data = []
    for img_info in coco_json['images']:
        iid = img_info['id']
        if iid in annos_by_img:
            data.append((img_info, annos_by_img[iid]))

    print(f'  {len(data)} images with annotations')

    # ── Prepare output directories ────────────────────────────────────────────
    out_img_dir = os.path.join(args.output_dir, 'images')
    os.makedirs(out_img_dir, exist_ok=True)

    # ── ID counters (start above existing max) ────────────────────────────────
    max_img_id  = max(img['id'] for img in coco_json['images'])
    max_ann_id  = max(ann['id'] for ann in coco_json['annotations']) if coco_json['annotations'] else 0
    img_ctr  = Counter(max_img_id + 1)
    ann_ctr  = Counter(max_ann_id + 1)

    # ── Build output COCO structures ──────────────────────────────────────────
    out_images = []
    out_annotations = []

    # Copy / reference original images
    if args.keep_originals:
        print('Copying original images ...')
        for img_info in coco_json['images']:
            fn = img_info.get('file_name') or img_info['coco_url'].split('/')[-1]
            src = os.path.join(args.img_dir, fn)
            dst = os.path.join(out_img_dir, fn)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)
        out_images.extend(coco_json['images'])
        out_annotations.extend(coco_json['annotations'])

    # ── Color jitter transform (built once) ───────────────────────────────────
    jitter_tfm = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)

    save_opts = [int(cv2.IMWRITE_JPEG_QUALITY), args.jpg_quality]

    # ── Per-image augmentations ───────────────────────────────────────────────
    GEOM_METHODS = {'hflip', 'vflip', 'rotation', 'crop', 'color_jitter', 'noise'}
    active_geom = [m for m in args.methods if m in GEOM_METHODS]

    total = len(data)
    for idx, (img_info, annos) in enumerate(data):
        fn = img_info.get('file_name') or img_info['coco_url'].split('/')[-1]
        img_path = os.path.join(args.img_dir, fn)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f'  [WARN] cannot read {img_path}, skipping', file=sys.stderr)
            continue
        orig_h = img_info['height']
        orig_w = img_info['width']

        stem, ext = os.path.splitext(fn)

        # Build method list (crop may be repeated)
        method_calls = []
        for m in active_geom:
            if m == 'crop':
                for r in range(args.crop_repeat):
                    method_calls.append(('crop', r))
            else:
                method_calls.append((m, 0))

        for method, repeat_idx in method_calls:
            # ── Apply augmentation ─────────────────────────────────────────
            if method == 'hflip':
                new_img, mask_fn, new_h, new_w = aug_hflip(img_bgr)
                suffix = 'hflip'
            elif method == 'vflip':
                new_img, mask_fn, new_h, new_w = aug_vflip(img_bgr)
                suffix = 'vflip'
            elif method == 'rotation':
                new_img, mask_fn, new_h, new_w = aug_rotation(img_bgr, args.rotation_range)
                suffix = 'rot'
            elif method == 'crop':
                new_img, mask_fn, new_h, new_w = aug_crop(img_bgr, args.crop_scale_min)
                suffix = f'crop{repeat_idx}'
            elif method == 'color_jitter':
                new_img, mask_fn, new_h, new_w = aug_color_jitter(img_bgr, jitter_tfm)
                suffix = 'cj'
            elif method == 'noise':
                new_img, mask_fn, new_h, new_w = aug_noise(
                    img_bgr, (args.noise_sigma_min, args.noise_sigma_max))
                suffix = 'noise'
            else:
                continue

            new_fn = f'{stem}_{suffix}.jpg'
            out_path = os.path.join(out_img_dir, new_fn)
            cv2.imwrite(out_path, new_img, save_opts)

            new_img_id = img_ctr.next()
            new_img_info = {
                'id':        new_img_id,
                'file_name': new_fn,
                'height':    new_h,
                'width':     new_w,
            }
            out_images.append(new_img_info)

            ann_id_list = [ann_ctr.val]  # mutable for _transform_annos
            new_annos = _transform_annos(
                annos, orig_h, orig_w, new_h, new_w,
                mask_fn, new_img_id, ann_id_list)
            ann_ctr.val = ann_id_list[0]
            out_annotations.extend(new_annos)

        if (idx + 1) % 100 == 0 or (idx + 1) == total:
            print(f'  [{idx + 1}/{total}] per-image augmentations done')

    # ── Mosaic ────────────────────────────────────────────────────────────────
    if 'mosaic' in args.methods and args.mosaic_count > 0:
        print(f'Generating {args.mosaic_count} mosaic images ...')
        # Determine mosaic canvas size from median image dimensions
        all_h = [d[0]['height'] for d in data]
        all_w = [d[0]['width']  for d in data]
        mosaic_h = int(np.median(all_h)) * 2
        mosaic_w = int(np.median(all_w)) * 2

        for k in range(args.mosaic_count):
            entries = random.choices(data, k=4)
            canvas, annos_raw, th, tw = _make_mosaic(
                entries, (mosaic_h, mosaic_w), args.img_dir)

            new_fn = f'mosaic_{k:06d}.jpg'
            cv2.imwrite(os.path.join(out_img_dir, new_fn), canvas, save_opts)

            new_img_id = img_ctr.next()
            out_images.append({
                'id':        new_img_id,
                'file_name': new_fn,
                'height':    th,
                'width':     tw,
            })

            for cat_id, full_mask, iscrowd in annos_raw:
                rle = _encode_mask(full_mask)
                out_annotations.append({
                    'id':           ann_ctr.next(),
                    'image_id':     new_img_id,
                    'category_id':  cat_id,
                    'segmentation': rle,
                    'area':         _rle_area(rle),
                    'bbox':         _rle_bbox(rle),
                    'iscrowd':      iscrowd,
                })

            if (k + 1) % 100 == 0 or (k + 1) == args.mosaic_count:
                print(f'  [{k + 1}/{args.mosaic_count}] mosaic images done')

    # ── Write output JSON ─────────────────────────────────────────────────────
    out_json = {
        'info':        coco_json.get('info', {}),
        'licenses':    coco_json.get('licenses', []),
        'categories':  coco_json['categories'],
        'images':      out_images,
        'annotations': out_annotations,
    }
    out_ann_path = os.path.join(args.output_dir, 'annotations.json')
    print(f'Writing {out_ann_path} ...')
    with open(out_ann_path, 'w') as f:
        json.dump(out_json, f)

    print(f'\nDone.')
    print(f'  Total images      : {len(out_images)}')
    print(f'  Total annotations : {len(out_annotations)}')
    print(f'  Output dir        : {args.output_dir}')


if __name__ == '__main__':
    main()
