"""
data_import.py — COCO val2017 downloader and dataset class for YOLO26 adversarial attacks.

Usage:
    python data_import.py [--data-root ./data/coco] [--subset N] [--skip-download]
"""

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import requests
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# COCO category ID → YOLO 0-indexed class ID
# COCO uses non-contiguous IDs (1–90 with gaps); YOLO uses 0–79.
# ---------------------------------------------------------------------------
COCO_TO_YOLO = {
    1: 0,  2: 1,  3: 2,  4: 3,  5: 4,  6: 5,  7: 6,  8: 7,  9: 8,  10: 9,
    11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19,
    22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27, 33: 28, 34: 29,
    35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39,
    46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49,
    56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57, 64: 58, 65: 59,
    67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69,
    80: 70, 81: 71, 82: 72, 84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79,
}

YOLO_CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

VAL_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
EXPECTED_VAL_IMAGE_COUNT = 5000


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download_file(url: str, dest_path: Path, chunk_size: int = 1024 * 64) -> Path:
    """Download a file from url to dest_path, skipping if already complete."""
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if already fully downloaded by comparing Content-Length
    if dest_path.exists():
        try:
            head = requests.head(url, timeout=10, allow_redirects=True)
            remote_size = int(head.headers.get("Content-Length", 0))
            local_size = dest_path.stat().st_size
            if remote_size > 0 and local_size == remote_size:
                print(f"  Already downloaded: {dest_path.name} ({local_size / 1e6:.0f} MB)")
                return dest_path
        except Exception:
            pass  # If head request fails, re-download

    print(f"  Downloading {dest_path.name} from {url}")
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()

    total = int(response.headers.get("Content-Length", 0))
    with open(dest_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, unit_divisor=1024, desc=dest_path.name
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            bar.update(len(chunk))

    return dest_path


def extract_zip(zip_path: Path, extract_to: Path, members_prefix: str = None) -> None:
    """Extract a zip file, optionally filtering by member prefix."""
    extract_to.mkdir(parents=True, exist_ok=True)
    print(f"  Extracting {zip_path.name} → {extract_to}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        if members_prefix:
            members = [m for m in members if m.startswith(members_prefix)]
        for member in tqdm(members, desc="Extracting", unit="file"):
            zf.extract(member, extract_to)


def download_coco_val(data_root: Path) -> None:
    """Download and extract COCO val2017 images and annotations."""
    data_root = Path(data_root)
    images_dir = data_root / "images" / "val2017"
    annotations_dir = data_root / "annotations"
    annotations_file = annotations_dir / "instances_val2017.json"

    print("\n[1/4] Checking COCO val2017 images...")
    if images_dir.exists() and len(list(images_dir.glob("*.jpg"))) == EXPECTED_VAL_IMAGE_COUNT:
        print(f"  Images already present ({EXPECTED_VAL_IMAGE_COUNT} files).")
    else:
        zip_path = data_root / "val2017.zip"
        download_file(VAL_IMAGES_URL, zip_path)
        print("\n[2/4] Extracting images...")
        extract_zip(zip_path, data_root / "images")
        zip_path.unlink(missing_ok=True)

    print("\n[3/4] Checking annotations...")
    if annotations_file.exists():
        print(f"  Annotations already present: {annotations_file.name}")
    else:
        zip_path = data_root / "annotations_trainval2017.zip"
        download_file(ANNOTATIONS_URL, zip_path)
        print("\n[4/4] Extracting annotations...")
        extract_zip(zip_path, data_root, members_prefix="annotations/instances_val2017.json")
        zip_path.unlink(missing_ok=True)


def verify_dataset(data_root: Path) -> bool:
    """Verify the dataset is complete. Returns True if OK."""
    data_root = Path(data_root)
    images_dir = data_root / "images" / "val2017"
    annotations_file = data_root / "annotations" / "instances_val2017.json"

    issues = []

    if not images_dir.exists():
        issues.append(f"Images directory not found: {images_dir}")
    else:
        count = len(list(images_dir.glob("*.jpg")))
        if count != EXPECTED_VAL_IMAGE_COUNT:
            issues.append(f"Expected {EXPECTED_VAL_IMAGE_COUNT} images, found {count}")

    if not annotations_file.exists():
        issues.append(f"Annotations file not found: {annotations_file}")
    else:
        try:
            with open(annotations_file) as f:
                data = json.load(f)
            if "images" not in data or "annotations" not in data:
                issues.append("Annotations JSON missing required keys")
        except json.JSONDecodeError as e:
            issues.append(f"Annotations JSON invalid: {e}")

    if issues:
        print("\nDataset verification FAILED:")
        for issue in issues:
            print(f"  - {issue}")
        return False

    print("\nDataset verification PASSED.")
    return True


# ---------------------------------------------------------------------------
# Letterbox (matches Ultralytics preprocessing)
# ---------------------------------------------------------------------------

def letterbox_image(img_bgr: np.ndarray, new_shape: int = 640, color: tuple = (114, 114, 114)):
    """
    Resize and pad image to new_shape x new_shape using letterboxing.
    Returns (padded_img_bgr, ratio, (dw, dh)) where dw/dh are padding amounts.
    """
    h, w = img_bgr.shape[:2]
    ratio = min(new_shape / h, new_shape / w)
    new_w, new_h = int(round(w * ratio)), int(round(h * ratio))

    img_resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    dw = (new_shape - new_w) / 2
    dh = (new_shape - new_h) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img_padded = cv2.copyMakeBorder(
        img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return img_padded, ratio, (dw, dh)


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class COCODataset(Dataset):
    """
    PyTorch Dataset for COCO val2017, formatted for YOLO26 inference.

    Each item returns:
        img_tensor  : float32 torch.Tensor (3, img_size, img_size) in [0, 1], RGB
        image_id    : int  COCO image ID
        annotations : list of dicts:
                        {'yolo_class_id': int, 'coco_category_id': int,
                         'bbox': [x, y, w, h] in original image coords}
    """

    def __init__(
        self,
        images_dir: str,
        annotations_file: str,
        img_size: int = 640,
        subset: int = None,
        coco_to_yolo: dict = None,
    ):
        self.images_dir = Path(images_dir)
        self.img_size = img_size
        self.coco_to_yolo = coco_to_yolo or COCO_TO_YOLO

        print(f"Loading COCO annotations from {annotations_file}...")
        self.coco = COCO(annotations_file)

        # Get image IDs that have at least one non-crowd annotation
        all_img_ids = self.coco.getImgIds()
        valid_ids = []
        for img_id in all_img_ids:
            ann_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=False)
            anns = self.coco.loadAnns(ann_ids)
            # Keep only images with at least one annotation with a known YOLO class
            if any(a["category_id"] in self.coco_to_yolo for a in anns):
                valid_ids.append(img_id)

        if subset is not None and subset < len(valid_ids):
            valid_ids = valid_ids[:subset]
            print(f"Using subset of {subset} images.")

        self.image_ids = valid_ids
        print(f"Dataset ready: {len(self.image_ids)} images with valid annotations.")

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        img_info = self.coco.loadImgs([image_id])[0]
        img_path = self.images_dir / img_info["file_name"]

        # Load image
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Letterbox resize to match YOLO26 input
        img_padded, ratio, (dw, dh) = letterbox_image(img_bgr, new_shape=self.img_size)

        # BGR → RGB, HWC → CHW, normalize to [0, 1]
        img_rgb = img_padded[:, :, ::-1]  # BGR to RGB
        img_chw = np.ascontiguousarray(img_rgb.transpose(2, 0, 1))
        img_tensor = torch.from_numpy(img_chw).float() / 255.0

        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=[image_id], iscrowd=False)
        raw_anns = self.coco.loadAnns(ann_ids)

        annotations = []
        for ann in raw_anns:
            cat_id = ann["category_id"]
            if cat_id not in self.coco_to_yolo:
                continue
            annotations.append({
                "yolo_class_id": self.coco_to_yolo[cat_id],
                "coco_category_id": cat_id,
                "bbox": ann["bbox"],  # [x, y, w, h] in original coords
            })

        return img_tensor, image_id, annotations

    def get_class_distribution(self) -> dict:
        """Returns {class_name: count} for all annotations in the dataset."""
        counts = {}
        for img_id in self.image_ids:
            ann_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=False)
            anns = self.coco.loadAnns(ann_ids)
            for ann in anns:
                cat_id = ann["category_id"]
                if cat_id not in self.coco_to_yolo:
                    continue
                yolo_id = self.coco_to_yolo[cat_id]
                name = YOLO_CLASS_NAMES[yolo_id]
                counts[name] = counts.get(name, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def print_dataset_summary(dataset: COCODataset) -> None:
    print("\n" + "=" * 60)
    print(f"  COCO val2017 Dataset Summary")
    print("=" * 60)
    print(f"  Total images : {len(dataset)}")

    dist = dataset.get_class_distribution()
    total_anns = sum(dist.values())
    print(f"  Total annotations : {total_anns}")
    print(f"  Classes present   : {len(dist)} / 80")
    print()
    print(f"  {'Class':<22} {'Count':>8}  {'%':>6}")
    print(f"  {'-'*22} {'-'*8}  {'-'*6}")
    for name, count in list(dist.items())[:20]:
        pct = count / total_anns * 100
        print(f"  {name:<22} {count:>8}  {pct:>5.1f}%")
    if len(dist) > 20:
        rest = sum(list(dist.values())[20:])
        print(f"  {'... (remaining classes)':<22} {rest:>8}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Download and verify COCO val2017 for YOLO26 adversarial attack experiments."
    )
    parser.add_argument(
        "--data-root", default="./data/coco",
        help="Root directory for COCO data (default: ./data/coco)"
    )
    parser.add_argument(
        "--subset", type=int, default=None,
        help="Limit dataset to N images (for quick testing)"
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip downloading; only verify and summarize existing data"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)

    print(f"COCO data root: {data_root.resolve()}")

    if not args.skip_download:
        download_coco_val(data_root)
    else:
        print("Skipping download (--skip-download).")

    ok = verify_dataset(data_root)
    if not ok:
        print("\nRun without --skip-download to fetch missing files.")
        sys.exit(1)

    dataset = COCODataset(
        images_dir=data_root / "images" / "val2017",
        annotations_file=data_root / "annotations" / "instances_val2017.json",
        subset=args.subset,
    )
    print_dataset_summary(dataset)


if __name__ == "__main__":
    main()
