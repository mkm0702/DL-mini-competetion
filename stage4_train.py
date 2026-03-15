import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"   
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
torch.backends.cudnn.benchmark     = False
torch.backends.cudnn.deterministic = True

from stage2_augmentation import VOCSegDataset, load_splits, VOC_ROOT, SPLIT_DIR, IMAGE_SIZE
from stage3_model import build_model, save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader

CONFIG = {
    "model_variant"       : "small",
    "pretrained"          : True,
    "epochs"              : 50,
    "batch_size"          : 4,
    "num_workers"         : 0,      
    "pin_memory"          : False,  
    "learning_rate"       : 1e-3,
    "weight_decay"        : 1e-4,
    "grad_clip"           : 1.0,
    "early_stop_patience" : 20,
    "checkpoint_path"     : Path("checkpoints/best_model.pth"),
    "num_classes"         : 21,
    "ignore_index"        : 255,
}

def get_dataloaders(config):
    
    train_ids, val_ids = load_splits(SPLIT_DIR)

    train_ds = VOCSegDataset(VOC_ROOT, train_ids, IMAGE_SIZE, split="train")
    val_ds   = VOCSegDataset(VOC_ROOT, val_ids,   IMAGE_SIZE, split="val")

    train_loader = DataLoader(
        train_ds,
        batch_size  = config["batch_size"],
        shuffle     = True,
        num_workers = config["num_workers"],
        pin_memory  = config["pin_memory"],  
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = config["batch_size"],
        shuffle     = False,
        num_workers = config["num_workers"],
        pin_memory  = config["pin_memory"],
    )

    print(f"[INFO] Train: {len(train_ds)} images | Val: {len(val_ds)} images")
    return train_loader, val_loader

# MASK CLEANING 

def clean_mask(mask, num_classes=21, ignore_index=255):

    invalid = (mask >= num_classes) & (mask != ignore_index)
    mask[invalid] = ignore_index
    return mask

def sanity_check(train_loader, val_loader, num_classes=21, ignore_index=255):
    print("[SANITY] Checking mask label ranges ...")
    for name, loader in [("train", train_loader), ("val", val_loader)]:
        for i, (imgs, masks) in enumerate(loader):
            unique = masks.unique().tolist()
            bad = [v for v in unique
                   if int(v) not in range(num_classes) and int(v) != ignore_index]
            if bad:
                print(f"  [WARN] {name} batch {i}: unexpected labels {bad} → will be clamped")
            if i >= 3:
                break
        print(f"  [OK] {name} checked.")

    imgs, masks = next(iter(train_loader))
    assert imgs.dtype  == torch.float32
    assert masks.dtype == torch.int64
    print(f"  [OK] Image range [{imgs.min():.2f}, {imgs.max():.2f}]  "
          f"Mask range [{int(masks.min())}, {int(masks.max())}]")
    print("[SANITY] Done.\n")

# LOSS

def get_loss_fn(ignore_index=255):
    return nn.CrossEntropyLoss(ignore_index=ignore_index)

def compute_dice(pred_mask, true_mask, num_classes=21, ignore_index=255):
    dice_per_class = []
    for cls in range(num_classes):
        valid        = (true_mask != ignore_index)
        pred_cls     = (pred_mask == cls) & valid
        true_cls     = (true_mask == cls) & valid
        intersection = (pred_cls & true_cls).sum().item()
        denom        = pred_cls.sum().item() + true_cls.sum().item()
        dice_per_class.append(1.0 if denom == 0 else (2.0 * intersection) / denom)
    return float(np.mean(dice_per_class))

def train_one_epoch(model, loader, optimizer, loss_fn, device,
                    grad_clip=1.0, num_classes=21, ignore_index=255):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc="  Train", leave=False)

    for images, masks in pbar:
        images = images.to(device)
        masks  = masks.to(device)
        masks = clean_mask(masks, num_classes, ignore_index)
        optimizer.zero_grad()
        logits, _ = model(images)
        loss      = loss_fn(logits, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader)

def validate(model, loader, loss_fn, device, num_classes=21, ignore_index=255):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0

    with torch.no_grad():
        pbar = tqdm(loader, desc="  Val  ", leave=False)
        for images, masks in pbar:
            images = images.to(device)
            masks  = masks.to(device)
            masks  = clean_mask(masks, num_classes, ignore_index)

            logits, pred_mask = model(images)
            loss = loss_fn(logits, masks)
            dice = compute_dice(pred_mask, masks, num_classes, ignore_index)

            total_loss += loss.item()
            total_dice += dice
            pbar.set_postfix(dice=f"{dice:.4f}")

    return total_loss / len(loader), total_dice / len(loader)

class EarlyStopping:
    def __init__(self, patience=10):
        self.patience    = patience
        self.best_dice   = -1.0
        self.counter     = 0
        self.should_stop = False

    def step(self, val_dice):
        if val_dice > self.best_dice:
            self.best_dice = val_dice
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


def plot_curves(history, save_path=Path("checkpoints/training_curves.png")):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], label="Train loss", color="steelblue")
    ax1.plot(epochs, history["val_loss"],   label="Val loss",   color="coral")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Loss"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, history["val_dice"], label="Val Dice", color="green")
    ax2.axhline(y=max(history["val_dice"]), color="gray", linestyle="--",
                label=f"Best: {max(history['val_dice']):.4f}")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Dice")
    ax2.set_title("Validation Dice"); ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"[INFO] Curves saved → {save_path}")
    plt.show()

# MAIN

def train(config=CONFIG):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device     : {device}")
    print(f"[INFO] Model      : {config['model_variant']}")
    print(f"[INFO] Batch size : {config['batch_size']}")
    print(f"[INFO] Workers    : {config['num_workers']}")
    print(f"[INFO] Pin memory : {config['pin_memory']}")
    print(f"[INFO] Epochs     : {config['epochs']}")
    print("-" * 45)

    train_loader, val_loader = get_dataloaders(config)

    sanity_check(train_loader, val_loader,
                 config["num_classes"], config["ignore_index"])
    
    model = build_model(
        variant    = config["model_variant"],
        pretrained = config["pretrained"],
        device     = device,
    )

    loss_fn   = get_loss_fn(config["ignore_index"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"], eta_min=1e-6
    )
    early_stopper = EarlyStopping(config["early_stop_patience"])
    history       = {"train_loss": [], "val_loss": [], "val_dice": []}
    best_dice     = -1.0

    for epoch in range(1, config["epochs"] + 1):
        lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch}/{config['epochs']}   LR={lr:.2e}")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device,
            grad_clip=config["grad_clip"],
            num_classes=config["num_classes"],
            ignore_index=config["ignore_index"],
        )
        val_loss, val_dice = validate(
            model, val_loader, loss_fn, device,
            num_classes=config["num_classes"],
            ignore_index=config["ignore_index"],
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)

        print(f"  train loss : {train_loss:.4f}")
        print(f"  val   loss : {val_loss:.4f}")
        print(f"  val   Dice : {val_dice:.4f}  (best: {best_dice:.4f})")

        if val_dice > best_dice:
            best_dice = val_dice
            save_checkpoint(model, optimizer, epoch, val_dice,
                            path=config["checkpoint_path"])

        early_stopper.step(val_dice)
        if early_stopper.should_stop:
            print(f"\n[EARLY STOP] No improvement for "
                  f"{config['early_stop_patience']} epochs.")
            break

    print(f"\n[DONE] Best Val Dice: {best_dice:.4f}")
    plot_curves(history)
    return history, best_dice


if __name__ == "__main__":
    history, best_dice = train(CONFIG)
