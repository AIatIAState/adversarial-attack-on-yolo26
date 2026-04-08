"""
adversarial_attack.py — FGSM and PGD adversarial attacks against YOLO26.

Implements three task-loss-based attacks from Choi & Tian (arXiv 2202.04781):
  A_obj : objectness-oriented attack (most effective)
  A_cls : classification-oriented attack
  A_loc : localization-oriented attack (CIoU)

Epsilon is specified in pixel space [0, 255]. PGD uses step size α=1 pixel, T=10.

Usage:
    python adversarial_attack.py [options]

Examples:
    # Quick smoke test (5 images, FGSM only, objectness attack)
    python adversarial_attack.py --subset 5 --attack fgsm --loss obj --save-vis 5

    # Full run, all loss types, all attack methods
    python adversarial_attack.py --attack all --loss all --output-dir ./results
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from ultralytics import YOLO

from data_import import COCO_TO_YOLO, YOLO_CLASS_NAMES, COCODataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NC = 80  # number of COCO/YOLO classes
IMG_SIZE = 640
DEFAULT_EPSILON_VALUES = [2, 4, 6]   # pixel-space [0, 255]; paper uses {2,4,6,8}
DEFAULT_PGD_STEPS = 10
CONFIDENCE_THRESHOLD = 0.25
PIXEL_MAX = 255.0              # for converting pixel-space epsilon to [0,1]
PGD_STEP_SIZE = 1.0 / PIXEL_MAX  # α=1 pixel (Choi & Tian §IV-A)
LOSS_TYPES = ["obj", "cls", "loc"]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_yolo26_model(model_name: str = "yolo26n.pt", device: str = "auto"):
    """
    Load YOLO26 and return the underlying nn.Module plus class names.

    Returns:
        nn_model   : torch.nn.Module (eval mode, on device)
        class_names: dict {0: 'person', 1: 'bicycle', ...}
        device     : torch.device
    """
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    print(f"Loading {model_name} on {device}...")
    yolo = YOLO(model_name)
    nn_model = yolo.model.to(device)
    nn_model.eval()
    class_names = yolo.names  # {0: 'person', ...}
    print(f"Model loaded. Classes: {len(class_names)}")
    return nn_model, class_names, device


def get_detect_head(nn_model):
    """Return the detection head (last layer of model.model)."""
    return nn_model.model[-1]


# ---------------------------------------------------------------------------
# Gradient sanity check
# ---------------------------------------------------------------------------

def verify_gradients(nn_model, device: torch.device) -> None:
    """
    Confirm gradients flow from the attack loss back to the input tensor.
    Raises AssertionError if gradient flow is broken.
    """
    print("Running gradient flow check...")
    dummy = torch.rand(1, 3, IMG_SIZE, IMG_SIZE, device=device, requires_grad=True)

    detect_head = get_detect_head(nn_model)
    original_end2end = getattr(detect_head, "end2end", True)
    detect_head.end2end = False

    try:
        raw_output = nn_model(dummy)
        preds = raw_output[1] if isinstance(raw_output, (tuple, list)) else raw_output
        if isinstance(preds, dict) and "one2many" in preds:
            preds = preds["one2many"]

        if isinstance(preds, dict):
            cls_logits = preds["scores"]  # (1, 80, 8400) — already computed by cv3 head
        else:
            reg_max = getattr(detect_head, "reg_max", 1)
            bbox_ch = reg_max * 4
            cls_logits_parts = [feat[:, bbox_ch:, :, :].reshape(1, NC, -1) for feat in preds]
            cls_logits = torch.cat(cls_logits_parts, dim=2)

        loss = cls_logits[0, [0, 1, 2], :].sigmoid().mean()
        loss.backward()
    finally:
        detect_head.end2end = original_end2end

    assert dummy.grad is not None, "Gradient is None — gradient flow broken!"
    assert dummy.grad.abs().sum() > 0, "Gradient is all zeros — no signal!"
    print(f"Gradient check PASSED. Grad norm: {dummy.grad.norm().item():.4f}")


# ---------------------------------------------------------------------------
# CIoU helper (used by A_loc)
# ---------------------------------------------------------------------------

def _ciou_loss(pred_cxcywh: torch.Tensor, gt_xyxy: torch.Tensor) -> torch.Tensor:
    """
    Compute mean CIoU loss between predicted boxes and GT boxes.

    For each GT box, the predicted box with the highest IoU is selected and
    CIoU is computed for that pair.  Loss is averaged over GT boxes.

    Args:
        pred_cxcywh : (4, 8400) decoded predicted boxes in cx,cy,w,h pixel coords
        gt_xyxy     : (N, 4)    GT boxes in x1,y1,x2,y2 pixel coords

    Returns:
        scalar tensor — CIoU loss to be maximised via gradient ascent
    """
    cx, cy, pw, ph = pred_cxcywh[0], pred_cxcywh[1], pred_cxcywh[2], pred_cxcywh[3]
    px1 = cx - pw / 2
    py1 = cy - ph / 2
    px2 = cx + pw / 2
    py2 = cy + ph / 2

    total = pred_cxcywh.new_zeros(1)

    for gt in gt_xyxy:
        gx1, gy1, gx2, gy2 = gt[0], gt[1], gt[2], gt[3]
        gw = gx2 - gx1
        gh = gy2 - gy1
        gcx = gx1 + gw / 2
        gcy = gy1 + gh / 2

        # IoU with all 8400 predicted boxes
        inter_w = (torch.min(px2, gx2) - torch.max(px1, gx1)).clamp(0)
        inter_h = (torch.min(py2, gy2) - torch.max(py1, gy1)).clamp(0)
        inter = inter_w * inter_h
        union = pw * ph + gw * gh - inter + 1e-7
        iou = inter / union  # (8400,)

        # Select the single most-overlapping predicted box
        best = iou.argmax()

        # CIoU: rho² / c²  +  alpha * v
        rho2 = (cx[best] - gcx) ** 2 + (cy[best] - gcy) ** 2
        enc_x1 = torch.min(px1[best], gx1)
        enc_y1 = torch.min(py1[best], gy1)
        enc_x2 = torch.max(px2[best], gx2)
        enc_y2 = torch.max(py2[best], gy2)
        c2 = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + 1e-7
        v = (4 / (math.pi ** 2)) * (
            torch.atan(gw / (gh + 1e-7)) - torch.atan(pw[best] / (ph[best] + 1e-7))
        ) ** 2
        alpha = v / (1 - iou[best] + v + 1e-7)

        total = total + (1 - iou[best] + rho2 / c2 + alpha * v)

    return total / max(len(gt_xyxy), 1)


# ---------------------------------------------------------------------------
# Loss functions for attacks  (Choi & Tian, arXiv 2202.04781, Eq. 8)
# ---------------------------------------------------------------------------

def compute_attack_loss(
    nn_model,
    x: torch.Tensor,
    gt_yolo_class_ids: list,
    gt_boxes_xyxy: list = None,
    loss_type: str = "obj",
) -> torch.Tensor:
    """
    Compute the adversarial attack loss (to be maximised via gradient ascent).

    Three loss types mirror the paper's task-oriented attacks:
      "obj" — objectness loss: BCE on per-anchor max class score vs target 1;
              maximising drives confidence toward 0 (suppresses detections).
      "cls" — classification loss: BCE on class logits vs GT one-hot targets;
              maximising pushes GT class scores down and non-GT scores up.
      "loc" — localisation loss: CIoU between decoded predicted boxes and GT
              boxes; maximising disrupts spatial localisation.

    Uses the one-to-many head with end2end=False to avoid the detach() that
    would block gradient flow.
    """
    detect_head = get_detect_head(nn_model)
    original_end2end = getattr(detect_head, "end2end", True)
    detect_head.end2end = False

    try:
        raw_output = nn_model(x)

        if isinstance(raw_output, (tuple, list)) and len(raw_output) == 2:
            preds = raw_output[1]
        else:
            raise RuntimeError(f"Unexpected model output structure: {type(raw_output)}")

        if isinstance(preds, dict) and "one2many" in preds:
            preds = preds["one2many"]

        if isinstance(preds, dict):
            cls_logits = preds["scores"]  # (1, 80, 8400)
        else:
            reg_max = getattr(detect_head, "reg_max", 1)
            bbox_ch = reg_max * 4
            cls_logits_parts = [feat[:, bbox_ch:, :, :].reshape(1, NC, -1) for feat in preds]
            cls_logits = torch.cat(cls_logits_parts, dim=2)

        gt_ids = list(set(gt_yolo_class_ids))

        if loss_type == "obj":
            # A_obj: per-anchor objectness proxy = max class score across 80 classes.
            # BCE with target=1 → maximising pushes sigmoid toward 0 (suppress all).
            obj_logits = cls_logits[0].max(dim=0).values  # (8400,)
            loss = F.binary_cross_entropy_with_logits(
                obj_logits, torch.ones_like(obj_logits)
            )

        elif loss_type == "cls":
            # A_cls: GT one-hot target across all anchors.
            # BCE with GT=1 → maximising pushes GT class confidence down;
            # non-GT terms (target=0) push those scores up (confident misprediction).
            targets = torch.zeros_like(cls_logits[0])  # (80, 8400)
            for gt_id in gt_ids:
                targets[gt_id, :] = 1.0
            loss = F.binary_cross_entropy_with_logits(cls_logits[0], targets)

        elif loss_type == "loc":
            # A_loc: CIoU loss on the most-overlapping predicted box per GT box.
            if not gt_boxes_xyxy:
                # No GT boxes available — fall back to obj loss
                obj_logits = cls_logits[0].max(dim=0).values
                loss = F.binary_cross_entropy_with_logits(
                    obj_logits, torch.ones_like(obj_logits)
                )
            else:
                decoded = detect_head._get_decode_boxes(preds)  # (1, 4, 8400) cxcywh
                gt_tensor = torch.tensor(
                    gt_boxes_xyxy, dtype=torch.float32, device=x.device
                )
                loss = _ciou_loss(decoded[0], gt_tensor)

        else:
            raise ValueError(f"Unknown loss_type '{loss_type}'. Choose from: obj, cls, loc")

    finally:
        detect_head.end2end = original_end2end

    return loss


# ---------------------------------------------------------------------------
# Baseline inference
# ---------------------------------------------------------------------------

def run_inference(nn_model, img_tensor: torch.Tensor, device: torch.device) -> list:
    """
    Run YOLO26 inference (default end2end head) on a single image tensor.

    Args:
        img_tensor: (1, 3, H, W) float tensor in [0, 1]

    Returns:
        list of dicts: [{'class_id': int, 'conf': float, 'bbox': [x1,y1,x2,y2]}, ...]
    """
    img = img_tensor.to(device).float()
    with torch.no_grad():
        raw_output = nn_model(img)

    # Default end2end output: (N, max_det, 6) — [x1, y1, x2, y2, conf, class_id]
    if isinstance(raw_output, (tuple, list)):
        decoded = raw_output[0]
    else:
        decoded = raw_output

    if decoded.ndim == 3:
        dets = decoded[0]  # (max_det, 6)
    else:
        dets = decoded

    detections = []
    for det in dets:
        det = det.cpu()
        if det.shape[0] < 6:
            continue
        conf = float(det[4])
        if conf < CONFIDENCE_THRESHOLD:
            continue
        detections.append({
            "class_id": int(det[5]),
            "conf": conf,
            "bbox": det[:4].tolist(),
        })
    return detections


# ---------------------------------------------------------------------------
# FGSM attack
# ---------------------------------------------------------------------------

def fgsm_attack(
    nn_model,
    img_tensor: torch.Tensor,
    gt_yolo_class_ids: list,
    epsilon: float,
    device: torch.device,
    loss_type: str = "obj",
    gt_boxes_xyxy: list = None,
) -> torch.Tensor:
    """
    Fast Gradient Sign Method attack (single PGD step).

    epsilon is in pixel space [0, 255]; normalised to [0, 1] internally.
    Returns adversarial image tensor (1, 3, H, W) in [0, 1].
    """
    eps_01 = epsilon / PIXEL_MAX
    x = img_tensor.clone().to(device).float()
    x.requires_grad_(True)

    nn_model.zero_grad()
    loss = compute_attack_loss(nn_model, x, gt_yolo_class_ids, gt_boxes_xyxy, loss_type)
    loss.backward()

    with torch.no_grad():
        x_adv = (x + eps_01 * x.grad.sign()).clamp(0.0, 1.0)

    return x_adv.detach()


# ---------------------------------------------------------------------------
# PGD attack
# ---------------------------------------------------------------------------

def pgd_attack(
    nn_model,
    img_tensor: torch.Tensor,
    gt_yolo_class_ids: list,
    epsilon: float,
    num_steps: int = DEFAULT_PGD_STEPS,
    step_size: float = None,
    device: torch.device = None,
    loss_type: str = "obj",
    gt_boxes_xyxy: list = None,
) -> torch.Tensor:
    """
    Projected Gradient Descent (PGD) attack — iterative FGSM with projection.

    epsilon is in pixel space [0, 255]; step_size defaults to α=1 pixel (paper §IV-A).
    Returns adversarial image tensor (1, 3, H, W) in [0, 1].
    """
    eps_01 = epsilon / PIXEL_MAX
    alpha = step_size if step_size is not None else PGD_STEP_SIZE

    x_orig = img_tensor.clone().to(device).float()

    # Random start within the epsilon-ball (standard PGD initialisation)
    x_adv = x_orig.clone() + torch.empty_like(x_orig).uniform_(-eps_01, eps_01)
    x_adv = x_adv.clamp(0.0, 1.0).detach()

    for _ in range(num_steps):
        x_adv.requires_grad_(True)
        nn_model.zero_grad()

        loss = compute_attack_loss(nn_model, x_adv, gt_yolo_class_ids, gt_boxes_xyxy, loss_type)
        loss.backward()

        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            delta = (x_adv - x_orig).clamp(-eps_01, eps_01)
            x_adv = (x_orig + delta).clamp(0.0, 1.0)

    return x_adv.detach()


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate_attack(
    nn_model,
    dataset: COCODataset,
    attack_fn,
    attack_params: dict,
    device: torch.device,
    loss_type: str = "obj",
    max_images: int = None,
) -> dict:
    """
    Run an attack over the dataset and collect per-image and per-class stats.

    Returns:
        {
            'per_image': [...],
            'per_class': {class_id: {'baseline_tp': int, 'adv_tp': int, 'total': int}},
        }
    """
    per_class = defaultdict(lambda: {"baseline_tp": 0, "adv_tp": 0, "total": 0})
    per_image = []

    n = min(len(dataset), max_images) if max_images else len(dataset)

    for idx in tqdm(range(n), desc="Evaluating"):
        img_tensor, image_id, annotations = dataset[idx]
        gt_class_ids = list({ann["yolo_class_id"] for ann in annotations})
        gt_boxes_xyxy = [ann["bbox_yolo"] for ann in annotations if "bbox_yolo" in ann]

        if not gt_class_ids:
            continue

        img_batch = img_tensor.unsqueeze(0)  # (1, 3, H, W)

        # Baseline detection
        baseline_dets = run_inference(nn_model, img_batch, device)
        baseline_pred_classes = {d["class_id"] for d in baseline_dets}

        # Adversarial image
        x_adv = attack_fn(
            nn_model, img_batch, gt_class_ids, device=device,
            loss_type=loss_type, gt_boxes_xyxy=gt_boxes_xyxy,
            **attack_params
        )

        # Adversarial detection
        adv_dets = run_inference(nn_model, x_adv, device)
        adv_pred_classes = {d["class_id"] for d in adv_dets}

        # Per-class stats (true positive = GT class was detected)
        for cls_id in gt_class_ids:
            per_class[cls_id]["total"] += 1
            if cls_id in baseline_pred_classes:
                per_class[cls_id]["baseline_tp"] += 1
            if cls_id in adv_pred_classes:
                per_class[cls_id]["adv_tp"] += 1

        per_image.append({
            "image_id": image_id,
            "gt_classes": gt_class_ids,
            "baseline_detections": list(baseline_pred_classes),
            "adv_detections": list(adv_pred_classes),
            "perturbation_linf": float((x_adv - img_batch.to(device)).abs().max()),
        })

    return {"per_image": per_image, "per_class": dict(per_class)}


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def print_summary_table(
    per_class: dict,
    class_names: dict,
    attack_name: str,
    loss_type: str,
    epsilon: float,
    model_name: str,
) -> None:
    """Print a formatted per-class attack success table."""
    print(f"\n{'='*70}")
    print(f"  Attack: {attack_name.upper()}  |  Loss: A_{loss_type}  |  ε={epsilon}px  |  Model: {model_name}")
    print(f"{'='*70}")
    print(f"  {'Class':<22} {'GT Imgs':>8} {'Base Acc':>9} {'Adv Acc':>8} {'Success':>8}")
    print(f"  {'-'*22} {'-'*8} {'-'*9} {'-'*8} {'-'*8}")

    total_gt = total_base_tp = total_adv_tp = 0
    rows = []
    for cls_id, stats in per_class.items():
        t = stats["total"]
        if t == 0:
            continue
        base_acc = stats["baseline_tp"] / t
        adv_acc = stats["adv_tp"] / t
        success = base_acc - adv_acc  # fraction of previously-detected that are now missed
        rows.append((cls_id, t, base_acc, adv_acc, success))
        total_gt += t
        total_base_tp += stats["baseline_tp"]
        total_adv_tp += stats["adv_tp"]

    rows.sort(key=lambda r: -r[4])  # sort by attack success rate desc
    for cls_id, t, base_acc, adv_acc, success in rows:
        name = class_names.get(cls_id, f"class_{cls_id}")
        print(f"  {name:<22} {t:>8} {base_acc:>8.1%} {adv_acc:>8.1%} {success:>8.1%}")

    if total_gt > 0:
        ov_base = total_base_tp / total_gt
        ov_adv = total_adv_tp / total_gt
        ov_success = ov_base - ov_adv
        print(f"  {'─'*22} {'─'*8} {'─'*9} {'─'*8} {'─'*8}")
        print(f"  {'OVERALL':<22} {total_gt:>8} {ov_base:>8.1%} {ov_adv:>8.1%} {ov_success:>8.1%}")
    print(f"{'='*70}")


# ---------------------------------------------------------------------------
# Results saving
# ---------------------------------------------------------------------------

def save_results(
    results: dict,
    output_dir: Path,
    metadata: dict,
) -> None:
    """Save per-image results to JSON and per-class summary to CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    payload = {
        "metadata": metadata,
        "per_image": results["per_image"],
        "per_class": {
            str(k): v for k, v in results["per_class"].items()
        },
    }
    attack_tag = f"{metadata['attack']}_{metadata['loss_type']}_eps{metadata['epsilon']}"
    json_path = output_dir / f"attack_results_{attack_tag}.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved: {json_path}")

    # CSV
    rows = []
    for cls_id, stats in results["per_class"].items():
        t = stats["total"]
        if t == 0:
            continue
        base_acc = stats["baseline_tp"] / t
        adv_acc = stats["adv_tp"] / t
        rows.append({
            "class_id": cls_id,
            "class_name": YOLO_CLASS_NAMES[int(cls_id)] if int(cls_id) < len(YOLO_CLASS_NAMES) else f"class_{cls_id}",
            "total_gt_images": t,
            "baseline_tp": stats["baseline_tp"],
            "adv_tp": stats["adv_tp"],
            "baseline_acc": round(base_acc, 4),
            "adv_acc": round(adv_acc, 4),
            "attack_success_rate": round(base_acc - adv_acc, 4),
        })
    csv_path = output_dir / f"attack_summary_{attack_tag}.csv"
    pd.DataFrame(rows).sort_values("attack_success_rate", ascending=False).to_csv(
        csv_path, index=False
    )
    print(f"  Saved: {csv_path}")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def tensor_to_rgb(img_tensor: torch.Tensor) -> np.ndarray:
    """Convert (1, 3, H, W) or (3, H, W) tensor in [0,1] to HWC uint8 RGB."""
    t = img_tensor.detach().cpu()
    if t.ndim == 4:
        t = t[0]
    return (t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)


def save_comparison_image(
    img_tensor: torch.Tensor,
    adv_tensor: torch.Tensor,
    baseline_dets: list,
    adv_dets: list,
    gt_annotations: list,
    image_id: int,
    epsilon: float,
    attack_name: str,
    loss_type: str,
    save_dir: Path,
    class_names: dict,
) -> None:
    """Save a side-by-side comparison: clean (green boxes) vs adversarial (red boxes)."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    clean_rgb = tensor_to_rgb(img_tensor)
    adv_rgb = tensor_to_rgb(adv_tensor)

    # Amplified perturbation for visualization
    delta = (adv_tensor.cpu() - img_tensor.cpu()).squeeze(0)
    delta_vis = ((delta * 10 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, img, title, dets, color in [
        (axes[0], clean_rgb, "Clean Image (baseline detections)", baseline_dets, "lime"),
        (axes[1], adv_rgb, f"Adversarial (ε={epsilon}px, {attack_name.upper()}, A_{loss_type})", adv_dets, "red"),
        (axes[2], delta_vis, "Perturbation (×10)", [], None),
    ]:
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        for det in dets:
            x1, y1, x2, y2 = det["bbox"]
            w, h = x2 - x1, y2 - y1
            rect = patches.Rectangle(
                (x1, y1), w, h,
                linewidth=1.5, edgecolor=color, facecolor="none"
            )
            ax.add_patch(rect)
            label = f"{class_names.get(det['class_id'], det['class_id'])} {det['conf']:.2f}"
            ax.text(x1, y1 - 2, label, color=color, fontsize=7, fontweight="bold")

    plt.suptitle(f"Image ID: {image_id}", fontsize=12)
    plt.tight_layout()

    out_path = save_dir / f"{image_id}_{attack_name}_{loss_type}_eps{epsilon}.png"
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Task-loss adversarial attacks (FGSM/PGD) on YOLO26 — Choi & Tian 2022."
    )
    parser.add_argument("--model", default="yolo26n.pt",
                        help="YOLO26 model weights (default: yolo26n.pt)")
    parser.add_argument("--data-root", default="./data/coco",
                        help="Path to COCO dataset root (default: ./data/coco)")
    parser.add_argument("--subset", type=int, default=None,
                        help="Limit evaluation to N images")
    parser.add_argument("--attack", default="all", choices=["fgsm", "pgd", "all"],
                        help="Attack method (default: all)")
    parser.add_argument("--loss", default="all", choices=["obj", "cls", "loc", "all"],
                        help="Task loss to attack with: obj|cls|loc|all (default: all)")
    parser.add_argument("--epsilon", type=int, default=None,
                        help=f"Perturbation budget in pixel space [0,255] (default: {DEFAULT_EPSILON_VALUES})")
    parser.add_argument("--pgd-steps", type=int, default=DEFAULT_PGD_STEPS,
                        help=f"PGD iterations (default: {DEFAULT_PGD_STEPS})")
    parser.add_argument("--output-dir", default="./results",
                        help="Directory for results (default: ./results)")
    parser.add_argument("--save-vis", type=int, default=20,
                        help="Number of visualization images to save per run (0 = all)")
    parser.add_argument("--device", default="auto",
                        help="Device: auto, cpu, cuda (default: auto)")
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    vis_dir = output_dir / "visualizations"

    nn_model, class_names, device = load_yolo26_model(args.model, args.device)
    verify_gradients(nn_model, device)

    dataset = COCODataset(
        images_dir=data_root / "images" / "val2017",
        annotations_file=data_root / "annotations" / "instances_val2017.json",
        subset=args.subset,
    )

    if len(dataset) == 0:
        print("Dataset is empty. Run data_import.py first.")
        sys.exit(1)

    epsilons = [args.epsilon] if args.epsilon is not None else DEFAULT_EPSILON_VALUES
    attacks = ["fgsm", "pgd"] if args.attack == "all" else [args.attack]
    loss_types = LOSS_TYPES if args.loss == "all" else [args.loss]

    vis_limit = args.save_vis if args.save_vis > 0 else float("inf")

    for attack_name in attacks:
        for loss_type in loss_types:
            for epsilon in epsilons:
                print(f"\n{'='*60}")
                print(f"  {attack_name.upper()} | A_{loss_type} | ε={epsilon}px")
                print(f"{'='*60}")

                if attack_name == "fgsm":
                    attack_fn = fgsm_attack
                    attack_params = {"epsilon": epsilon}
                else:
                    attack_fn = pgd_attack
                    attack_params = {"epsilon": epsilon, "num_steps": args.pgd_steps}

                results = evaluate_attack(
                    nn_model=nn_model,
                    dataset=dataset,
                    attack_fn=attack_fn,
                    attack_params=attack_params,
                    device=device,
                    loss_type=loss_type,
                )

                metadata = {
                    "model": args.model,
                    "attack": attack_name,
                    "loss_type": loss_type,
                    "epsilon": epsilon,
                    "pgd_steps": args.pgd_steps if attack_name == "pgd" else None,
                    "num_images": len(results["per_image"]),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }

                print_summary_table(
                    results["per_class"], class_names, attack_name, loss_type, epsilon, args.model
                )
                save_results(results, output_dir, metadata)

                # Save visualizations
                vis_saved = 0
                if vis_saved < vis_limit:
                    print(f"  Saving visualizations to {vis_dir}...")
                    for idx, _ in enumerate(results["per_image"]):
                        if vis_saved >= vis_limit:
                            break

                        img_tensor, image_id, annotations = dataset[idx]
                        img_batch = img_tensor.unsqueeze(0)
                        gt_class_ids = [ann["yolo_class_id"] for ann in annotations]
                        gt_boxes_xyxy = [ann["bbox_yolo"] for ann in annotations if "bbox_yolo" in ann]

                        baseline_dets = run_inference(nn_model, img_batch, device)
                        x_adv = attack_fn(
                            nn_model, img_batch, gt_class_ids, device=device,
                            loss_type=loss_type, gt_boxes_xyxy=gt_boxes_xyxy,
                            **attack_params
                        )
                        adv_dets = run_inference(nn_model, x_adv, device)

                        save_comparison_image(
                            img_tensor=img_batch,
                            adv_tensor=x_adv,
                            baseline_dets=baseline_dets,
                            adv_dets=adv_dets,
                            gt_annotations=annotations,
                            image_id=image_id,
                            epsilon=epsilon,
                            attack_name=attack_name,
                            loss_type=loss_type,
                            save_dir=vis_dir,
                            class_names=class_names,
                        )
                        vis_saved += 1

    print(f"\nDone. Results saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
