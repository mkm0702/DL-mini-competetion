import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from stage3_model import build_model, load_checkpoint
torch.backends.cudnn.benchmark     = False
torch.backends.cudnn.deterministic = True

CHECKPOINT  = Path("checkpoints/best_model.pth")
IMAGE_SIZE  = (300, 300)
MEAN        = (0.485, 0.456, 0.406)
STD         = (0.229, 0.224, 0.225)
VARIANT     = "small"

def preprocess(image_path):
    
    img       = Image.open(image_path).convert("RGB")
    orig_size = img.size                              # (W, H)
    resized   = img.resize(IMAGE_SIZE, Image.BILINEAR)
    tensor    = TF.to_tensor(resized)                 # (3, 300, 300)
    tensor    = TF.normalize(tensor, MEAN, STD)
    tensor    = tensor.unsqueeze(0)                   # (1, 3, 300, 300)
    return tensor, orig_size

def to_binary_mask(class_mask_np):
   
    binary = np.where(class_mask_np > 0, 255, 0).astype(np.uint8)
    return binary

def run_inference(in_dir, out_dir, checkpoint=CHECKPOINT, variant=VARIANT):
   
    in_dir  = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    valid_ext   = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_paths = sorted([
        p for p in in_dir.iterdir()
        if p.suffix.lower() in valid_ext
    ])

    if not image_paths:
        raise FileNotFoundError(
            f"No images found in {in_dir}\n"
            f"Supported extensions: {valid_ext}"
        )

    print(f"Found {len(image_paths)} images in {in_dir}")
    print(f"Output → {out_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device : {device}")

    model = build_model(variant=variant, pretrained=False, device=device)
    load_checkpoint(model, path=checkpoint, device=device)
    model.eval()
    print(f"[INFO] Checkpoint loaded from {checkpoint}")
    print("-" * 50)

    for idx, img_path in enumerate(image_paths):

        try:
            tensor, orig_size = preprocess(img_path)
            tensor = tensor.to(device)
            with torch.no_grad():
                _, class_mask = model(tensor)      
            class_np = class_mask.squeeze(0).cpu().numpy()   
           
            mask_pil  = Image.fromarray(class_np.astype(np.uint8))
            mask_pil  = mask_pil.resize(orig_size, Image.NEAREST)
            class_np  = np.array(mask_pil)

            binary_np = to_binary_mask(class_np)

            out_path = out_dir / img_path.name       
            Image.fromarray(binary_np).save(out_path)

            print(f"  [{idx+1:4d}/{len(image_paths)}] {img_path.name}")

        except Exception as e:
            print(f"  [ERROR] {img_path.name}: {e}")
            continue

    print(f"\n {len(image_paths)} masks saved to {out_dir}/")


def preview_results(in_dir, out_dir, n=4):

    import matplotlib.pyplot as plt

    in_dir  = Path(in_dir)
    out_dir = Path(out_dir)

    pairs = []
    for img_path in sorted(in_dir.iterdir()):
        mask_path = out_dir / img_path.name
        if mask_path.exists():
            pairs.append((img_path, mask_path))
        if len(pairs) >= n:
            break

    if not pairs:
        print("[WARN] No matching input/output pairs found for preview.")
        return

    fig, axes = plt.subplots(len(pairs), 2, figsize=(8, 4 * len(pairs)))
    if len(pairs) == 1:
        axes = [axes]

    axes[0][0].set_title("Input image",   fontsize=11, fontweight="bold")
    axes[0][1].set_title("Binary mask",   fontsize=11, fontweight="bold")

    for i, (img_path, mask_path) in enumerate(pairs):
        axes[i][0].imshow(Image.open(img_path).convert("RGB"))
        axes[i][1].imshow(Image.open(mask_path), cmap="gray", vmin=0, vmax=255)
        axes[i][0].set_xlabel(img_path.name, fontsize=8)
        axes[i][1].set_xlabel(mask_path.name, fontsize=8)
        for col in range(2):
            axes[i][col].axis("off")

    plt.suptitle("Inference preview — white=foreground, black=background",
                 fontsize=10)
    plt.tight_layout()
    preview_path = out_dir / "preview.png"
    plt.savefig(preview_path, dpi=120, bbox_inches="tight")
    print(f"[INFO] Preview saved → {preview_path}")
    plt.show()



def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate binary segmentation masks for test images."
    )
    parser.add_argument(
        "--in_dir",
        required=True,
        help="Path to folder containing input test images",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Path to output folder  e.g. groupN_output/",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best_model.pth",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--variant",
        default="small",
        choices=["small", "large"],
        help="Model variant: small or large",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show a visual preview of a few results after inference",
    )
    return parser.parse_args()



if __name__ == "__main__":

        # python inference.py --in_dir=/path/test/ --out_dir=/path/groupN_output/
        args = parse_args()
        run_inference(args.in_dir, args.out_dir, args.checkpoint, args.variant)
        if args.preview:
            preview_results(args.in_dir, args.out_dir, n=4)

