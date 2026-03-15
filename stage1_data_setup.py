import os
import random
import shutil
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

# CONFIG
VOC_URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
DATA_ROOT = Path("data")         
VOC_ROOT  = DATA_ROOT / "VOCdevkit" / "VOC2012"
SPLIT_DIR = DATA_ROOT / "splits" 

TRAIN_RATIO = 0.80
SEED        = 42
IMAGE_SIZE  = (300, 300)

# VOC 21 classes
VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
    "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
    "train", "tvmonitor",
]
NUM_CLASSES = len(VOC_CLASSES)  # 21


# DOWNLOAD

def download_voc(data_root: Path = DATA_ROOT) -> None:

    tar_path = data_root / "VOCtrainval_11-May-2012.tar"
    voc_path = data_root / "VOCdevkit"

    if voc_path.exists():
        print(f"[INFO] Dataset already exists at {voc_path}. Skipping download.")
        return

    data_root.mkdir(parents=True, exist_ok=True)

    if not tar_path.exists():
        print(f"[INFO] Downloading VOC 2012 (~2 GB) to {tar_path} ...")
        def _progress(block, block_size, total):
            mb_done  = block * block_size / 1e6
            mb_total = total / 1e6
            print(f"\r  {mb_done:.1f} / {mb_total:.1f} MB", end="", flush=True)
        urllib.request.urlretrieve(VOC_URL, tar_path, reporthook=_progress)
        print()

    print(f"[INFO] Extracting {tar_path} ...")
    with tarfile.open(tar_path) as tf_:
        tf_.extractall(data_root)
    print(f"[INFO] Extracted to {voc_path}")



# SPLIT

def create_splits(
    voc_root: Path  = VOC_ROOT,
    split_dir: Path = SPLIT_DIR,
    train_ratio: float = TRAIN_RATIO,
    seed: int = SEED,
) -> tuple[list[str], list[str]]:
    
    split_dir.mkdir(parents=True, exist_ok=True)
    train_txt = split_dir / "train.txt"
    val_txt   = split_dir / "val.txt"

    trainval_file = voc_root / "ImageSets" / "Segmentation" / "trainval.txt"
    if not trainval_file.exists():
        raise FileNotFoundError(
            f"Could not find {trainval_file}. "
            "Make sure the dataset was downloaded correctly."
        )

    with open(trainval_file) as f:
        all_ids = [line.strip() for line in f if line.strip()]

    print(f"[INFO] Total trainval images: {len(all_ids)}")

    official_val_file = voc_root / "ImageSets" / "Segmentation" / "val.txt"
    if official_val_file.exists():
        with open(official_val_file) as f:
            forbidden = set(l.strip() for l in f if l.strip())
        before = len(all_ids)
        all_ids = [i for i in all_ids if i not in forbidden]
        print(f"[INFO] Removed {before - len(all_ids)} official val images (test set). "
              f"Remaining: {len(all_ids)}")
    else:
        print("Official val.txt not found — no images removed.")

    random.seed(seed)
    shuffled = all_ids[:]
    random.shuffle(shuffled)

    split_idx  = int(len(shuffled) * train_ratio)
    train_ids  = shuffled[:split_idx]
    val_ids    = shuffled[split_idx:]

    # Save splits
    with open(train_txt, "w") as f:
        f.write("\n".join(train_ids))
    with open(val_txt, "w") as f:
        f.write("\n".join(val_ids))

    print(f"[INFO] Train split: {len(train_ids)} images → {train_txt}")
    print(f"[INFO] Val   split: {len(val_ids)}   images → {val_txt}")

    return train_ids, val_ids


def load_splits(split_dir: Path = SPLIT_DIR) -> tuple[list[str], list[str]]:
  
    with open(split_dir / "train.txt") as f:
        train_ids = [l.strip() for l in f if l.strip()]
    with open(split_dir / "val.txt") as f:
        val_ids = [l.strip() for l in f if l.strip()]
    return train_ids, val_ids


# bVERIFICATION
    

def verify_splits(
    voc_root: Path  = VOC_ROOT,
    split_dir: Path = SPLIT_DIR,
) -> bool:
    
    official_val_file = voc_root / "ImageSets" / "Segmentation" / "val.txt"
    if not official_val_file.exists():
        print("[WARN] Official val.txt not found — skipping leakage check.")
        return True

    with open(official_val_file) as f:
        forbidden = set(l.strip() for l in f if l.strip())

    train_ids, val_ids = load_splits(split_dir)
    our_ids = set(train_ids) | set(val_ids)

    leaked = our_ids & forbidden
    if leaked:
        raise AssertionError(
            f"[ERROR] LEAKAGE DETECTED! {len(leaked)} official val images "
            f"found in our splits: {sorted(leaked)[:5]} ..."
        )

    missing_imgs  = []
    missing_masks = []
    for img_id in train_ids + val_ids:
        img_path  = voc_root / "JPEGImages"           / f"{img_id}.jpg"
        mask_path = voc_root / "SegmentationClass"    / f"{img_id}.png"
        if not img_path.exists():
            missing_imgs.append(img_id)
        if not mask_path.exists():
            missing_masks.append(img_id)

    if missing_imgs:
        raise FileNotFoundError(
            f"[ERROR] {len(missing_imgs)} images missing: {missing_imgs[:3]} ..."
        )
    if missing_masks:
        raise FileNotFoundError(
            f"[ERROR] {len(missing_masks)} masks missing: {missing_masks[:3]} ..."
        )

    print(f"No leakage detected. {len(forbidden)} official val images are safe.")
    print(f" All images and masks verified on disk.")
    return True


# DATASET CLASS

class VOCSegDataset(Dataset):
   
    MEAN = (0.485, 0.456, 0.406)  
    STD  = (0.229, 0.224, 0.225)

    def __init__(
        self,
        voc_root:   Path,
        image_ids:  list[str],
        image_size: tuple[int, int] = IMAGE_SIZE,
        augment:    bool = False,
    ):
        self.voc_root   = Path(voc_root)
        self.image_ids  = image_ids
        self.image_size = image_size
        self.augment    = augment

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        import torch

        img_id = self.image_ids[idx]

        img  = Image.open(self.voc_root / "JPEGImages"        / f"{img_id}.jpg").convert("RGB")
        mask = Image.open(self.voc_root / "SegmentationClass" / f"{img_id}.png")  # palette PNG

        img  = img.resize(self.image_size, Image.BILINEAR)
        mask = mask.resize(self.image_size, Image.NEAREST)

        if self.augment:
            if random.random() > 0.5:
                img  = TF.hflip(img)
                mask = TF.hflip(mask)

       
        img_t  = TF.to_tensor(img)                                # (3, H, W) float [0,1]
        img_t  = TF.normalize(img_t, self.MEAN, self.STD)

        mask_t = torch.from_numpy(np.array(mask)).long()          # (H, W) int64
       
        return img_t, mask_t

    def get_sample_image(self, idx: int = 0):
        
        img_id = self.image_ids[idx]
        img    = Image.open(self.voc_root / "JPEGImages"        / f"{img_id}.jpg").convert("RGB")
        mask   = Image.open(self.voc_root / "SegmentationClass" / f"{img_id}.png")
        img    = img.resize(self.image_size, Image.BILINEAR)
        mask   = mask.resize(self.image_size, Image.NEAREST)
        return img, mask


# 5. DATALOADER FACTORY

def get_dataloaders(
    voc_root:    Path = VOC_ROOT,
    split_dir:   Path = SPLIT_DIR,
    image_size:  tuple[int, int] = IMAGE_SIZE,
    batch_size:  int  = 8,
    num_workers: int  = 4,
) -> tuple[DataLoader, DataLoader]:

    train_ids, val_ids = load_splits(split_dir)

    train_ds = VOCSegDataset(voc_root, train_ids, image_size, augment=True)
    val_ds   = VOCSegDataset(voc_root, val_ids,   image_size, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"[INFO] Train loader: {len(train_ds)} samples, {len(train_loader)} batches")
    print(f"[INFO] Val   loader: {len(val_ds)}   samples, {len(val_loader)} batches")

    return train_loader, val_loader



if __name__ == "__main__":

    # Download
    download_voc()

    # Split trainval → 80/20
    create_splits()

    # Verify no leakage
    verify_splits()

    # Build loaders and print a batch shape
    import torch
    train_loader, val_loader = get_dataloaders(batch_size=4, num_workers=0)

    imgs, masks = next(iter(train_loader))
    print(f"\nImage batch shape : {imgs.shape}")  
    print(f"Mask  batch shape : {masks.shape}")  
    print(f"Mask  unique vals : {torch.unique(masks).tolist()}")

    print("\nStage 1 complete")