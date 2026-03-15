
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import random

import torch
from PIL import Image

from stage3_model import build_model, load_checkpoint
from stage1_data_setup import VOC_ROOT, SPLIT_DIR, load_splits

torch.backends.cudnn.benchmark     = False
torch.backends.cudnn.deterministic = True

CHECKPOINT   = Path("checkpoints/best_model.pth")
IGNORE_INDEX = 255

def to_binary(gt_mask_np):

    binary = np.where(gt_mask_np == IGNORE_INDEX, IGNORE_INDEX,
                      np.where(gt_mask_np > 0, 1, 0))
    return binary.astype(np.uint8)


def binary_dice(pred_binary, true_binary):

    valid        = (true_binary != IGNORE_INDEX)
    pred_fg      = (pred_binary == 1) & valid
    true_fg      = (true_binary == 1) & valid
    intersection = (pred_fg & true_fg).sum()
    denom        = pred_fg.sum() + true_fg.sum()
    if denom == 0:
        return 1.0   # image has no foreground → perfect by convention
    return float(2.0 * intersection / denom)


def measure_flops(checkpoint, variant):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = build_model(variant=variant, pretrained=False, device=device)
    load_checkpoint(model, path=checkpoint, device=device)
    model.eval()

    try:
        from thop import profile, clever_format
        dummy = torch.randn(1, 3, 300, 300).to(device)

        _orig = model.forward
        model.forward = lambda x: _orig(x)[0]
        flops, params = profile(model, inputs=(dummy,), verbose=False)
        model.forward = _orig

        gflops = round(flops / 1e9, 4)
        _, p   = clever_format([flops, params], "%.4f")
        print(f"  FLOPs (raw)      : {flops:,.0f}")
        print(f"  GFLOPs per image : {gflops:.4f}  ← submit this")
        print(f"  Parameters       : {p}")
        return gflops

    except ImportError:
        print("[ERROR] thop not installed. Run:  pip install thop")
        return None

def evaluate_from_masks(pred_dir):

    pred_dir = Path(pred_dir)
    gt_dir   = VOC_ROOT / "SegmentationClass"

    _, val_ids  = load_splits(SPLIT_DIR)
    dice_scores = []
    missing     = []

    print(f"[INFO] Scoring {len(val_ids)} val images ...")

    for img_id in tqdm(val_ids, desc="  Scoring"):

        pred_path = pred_dir / f"{img_id}.jpg"
        if not pred_path.exists():
            pred_path = pred_dir / f"{img_id}.png"
        if not pred_path.exists():
            missing.append(img_id)
            continue

        gt_path = gt_dir / f"{img_id}.png"
        if not gt_path.exists():
            missing.append(img_id)
            continue

        pred_np     = np.array(Image.open(pred_path).convert("L"))
        pred_binary = (pred_np == 255).astype(np.uint8)   # 255 → 1, 0 → 0

        gt_np     = np.array(Image.open(gt_path))
        gt_binary = to_binary(gt_np)

        if pred_binary.shape != gt_binary.shape:
            pred_pil    = Image.fromarray(pred_binary)
            pred_pil    = pred_pil.resize(
                (gt_binary.shape[1], gt_binary.shape[0]), Image.NEAREST
            )
            pred_binary = np.array(pred_pil)

        dice_scores.append(binary_dice(pred_binary, gt_binary))

    if missing:
        print(f"  [WARN] {len(missing)} images missing — skipped")

    macro_dice = round(float(np.mean(dice_scores)), 4) if dice_scores else 0.0
    print(f"  Scored       : {len(dice_scores)} images")
    print(f"  Macro Dice   : {macro_dice:.4f}  ← submit this")
    return macro_dice, dice_scores




def print_leaderboard_summary(macro_dice, gflops):
    score = round(macro_dice / gflops, 4) if gflops else None
    print("\n" + "╔" + "═"*41 + "╗")
    print("║     LEADERBOARD SUBMISSION VALUES       ║")
    print("╠" + "═"*41 + "╣")
    print(f"║  Dice Score   : {macro_dice:.4f}                    ║")
    if gflops:
        print(f"║  GFLOPs       : {gflops:.4f}                    ║")
        print(f"║  Dice/GFLOPs  : {score:.4f}                    ║")
    print("╚" + "═"*41 + "╝\n")


def plot_dice_histogram(dice_scores, save_dir):
    
    macro = np.mean(dice_scores)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(dice_scores, bins=40, color="steelblue", edgecolor="white")
    ax.axvline(x=macro, color="red", linestyle="--", linewidth=1.5,
               label=f"Mean = {macro:.4f}")
    ax.set_xlabel("Binary Dice per Image")
    ax.set_ylabel("Number of Images")
    ax.set_title("Distribution of Per-Image Dice Scores")
    ax.legend()
    plt.tight_layout()
    path = Path(save_dir) / "dice_histogram.png"
    plt.savefig(path, dpi=130, bbox_inches="tight")
    print(f"[INFO] Histogram → {path}")
    plt.show()

def plot_sample_predictions(pred_dir, save_dir, n=6):

    _, val_ids = load_splits(SPLIT_DIR)
    gt_dir     = VOC_ROOT / "SegmentationClass"
    img_dir    = VOC_ROOT / "JPEGImages"
    pred_dir   = Path(pred_dir)

    # Collect samples that have prediction files
    samples = []
    for img_id in random.sample(val_ids, min(100, len(val_ids))):
        pred_path = pred_dir / f"{img_id}.jpg"
        if not pred_path.exists():
            pred_path = pred_dir / f"{img_id}.png"
        if pred_path.exists():
            samples.append((img_id, pred_path))
        if len(samples) >= n:
            break

    if not samples:
        print("[WARN] No prediction files found for preview.")
        return

    fig, axes = plt.subplots(len(samples), 3, figsize=(11, 4 * len(samples)))
    if len(samples) == 1:
        axes = [axes]

    axes[0][0].set_title("Input image",          fontsize=10, fontweight="bold")
    axes[0][1].set_title("GT (binary)",           fontsize=10, fontweight="bold")
    axes[0][2].set_title("Prediction (binary)",   fontsize=10, fontweight="bold")

    for row, (img_id, pred_path) in enumerate(samples):
        img      = Image.open(img_dir / f"{img_id}.jpg").convert("RGB")
        gt_np    = np.array(Image.open(gt_dir / f"{img_id}.png"))
        pred_np  = np.array(Image.open(pred_path).convert("L"))

        gt_binary              = to_binary(gt_np)
        gt_binary[gt_binary == IGNORE_INDEX] = 0   # treat boundary as bg for display

        axes[row][0].imshow(img)
        axes[row][1].imshow(gt_binary, cmap="gray", vmin=0, vmax=1)
        axes[row][2].imshow(pred_np,   cmap="gray", vmin=0, vmax=255)
        for col in range(3):
            axes[row][col].axis("off")

    plt.suptitle("white = foreground  |  black = background", fontsize=10)
    plt.tight_layout()
    path = Path(save_dir) / "sample_predictions.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    print(f"[INFO] Predictions → {path}")
    plt.show()


# MAIN


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Evaluate binary segmentation masks and report leaderboard numbers."
    )
    parser.add_argument("--pred_dir",   required=True,
                        help="Folder with binary masks from inference.py")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth",
                        help="Model checkpoint for FLOPs measurement")
    parser.add_argument("--variant",    default="small",
                        choices=["small", "large"])
    parser.add_argument("--save_dir",   default="checkpoints",
                        help="Where to save plots")
    args = parser.parse_args()

    print("\n[STEP 1] Measuring GFLOPs ...")
    gflops = measure_flops(args.checkpoint, args.variant)

    print("\n[STEP 2] Computing Dice score ...")
    macro_dice, dice_scores = evaluate_from_masks(args.pred_dir)

    print_leaderboard_summary(macro_dice, gflops)
    
    print("[STEP 3] Saving plots ...")
    plot_dice_histogram(dice_scores, args.save_dir)
    plot_sample_predictions(args.pred_dir, args.save_dir, n=6)

    print("[DONE]")