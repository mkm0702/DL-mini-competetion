import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

from stage1_data_setup import (
    VOC_ROOT, SPLIT_DIR, IMAGE_SIZE, NUM_CLASSES,
    load_splits,
)

def random_horizontal_flip(image, mask, p=0.5):
    if random.random() < p:
        image = TF.hflip(image)
        mask  = TF.hflip(mask)
    return image, mask

def random_scale_crop(image, mask, scale_range=(0.7, 1.3), output_size=(300, 300)):
    scale  = random.uniform(*scale_range)
    w, h   = image.size
    new_w  = int(w * scale)
    new_h  = int(h * scale)

    image = image.resize((new_w, new_h), Image.BILINEAR)
    mask  = mask.resize((new_w, new_h),  Image.NEAREST)  

    pad_w = max(0, output_size[0] - new_w)
    pad_h = max(0, output_size[1] - new_h)
    if pad_w > 0 or pad_h > 0:
        new_img  = Image.new("RGB", (new_w + pad_w, new_h + pad_h), (0, 0, 0))
        new_mask = Image.new("L",   (new_w + pad_w, new_h + pad_h), 0)
        new_img.paste(image, (0, 0))
        new_mask.paste(mask,  (0, 0))
        image, mask = new_img, new_mask

    crop_x = random.randint(0, image.size[0] - output_size[0])
    crop_y = random.randint(0, image.size[1] - output_size[1])
    image  = TF.crop(image, crop_y, crop_x, output_size[1], output_size[0])
    mask   = TF.crop(mask,  crop_y, crop_x, output_size[1], output_size[0])

    return image, mask

def random_rotation(image, mask, max_angle=15):
    angle = random.uniform(-max_angle, max_angle)
    image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
    mask  = TF.rotate(mask,  angle, interpolation=TF.InterpolationMode.NEAREST)
    return image, mask

def random_color_jitter(image, p=0.8):
    if random.random() < p:
        brightness = random.uniform(0.6, 1.4)
        contrast   = random.uniform(0.6, 1.4)
        saturation = random.uniform(0.6, 1.4)
        hue        = random.uniform(-0.1, 0.1)

        image = ImageEnhance.Brightness(image).enhance(brightness)
        image = ImageEnhance.Contrast(image).enhance(contrast)
        image = ImageEnhance.Color(image).enhance(saturation)
        image = TF.adjust_hue(image, hue)
    return image

def random_grayscale(image, p=0.1):
    if random.random() < p:
        image = TF.rgb_to_grayscale(image, num_output_channels=3)
    return image

def add_gaussian_noise(image, std_range=(5, 25)):
  
    img_array = np.array(image, dtype=np.float32)
    std       = random.uniform(*std_range)
    noise     = np.random.normal(0, std, img_array.shape).astype(np.float32)
    noisy     = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def add_salt_and_pepper(image, amount_range=(0.01, 0.05)):
   
    img_array = np.array(image).copy()
    amount    = random.uniform(*amount_range)
    n_pixels  = img_array.shape[0] * img_array.shape[1]
    n_salt    = int(n_pixels * amount / 2)
    n_pepper  = int(n_pixels * amount / 2)

    coords = [np.random.randint(0, dim, n_salt) for dim in img_array.shape[:2]]
    img_array[coords[0], coords[1]] = 255

    coords = [np.random.randint(0, dim, n_pepper) for dim in img_array.shape[:2]]
    img_array[coords[0], coords[1]] = 0

    return Image.fromarray(img_array)

def add_gaussian_blur(image, radius_range=(0.5, 2.0)):
    radius = random.uniform(*radius_range)
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def add_jpeg_compression(image, quality_range=(30, 80)):
    
    import io
    quality = random.randint(*quality_range)
    buffer  = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).copy() 

def apply_train_augmentations(image, mask):
    

    image, mask = random_horizontal_flip(image, mask, p=0.5)
    image, mask = random_scale_crop(image, mask, scale_range=(0.7, 1.3))
    image, mask = random_rotation(image, mask, max_angle=10)

    image = random_color_jitter(image, p=0.8)
    image = random_grayscale(image,    p=0.1)


    if random.random() < 0.5:
        corruption = random.choice([
            lambda img: add_gaussian_noise(img),
            lambda img: add_salt_and_pepper(img),
            lambda img: add_gaussian_blur(img),
            lambda img: add_jpeg_compression(img),
        ])
        image = corruption(image)

    return image, mask

def apply_val_augmentations(image, mask):
    
    return image, mask

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

class VOCSegDataset(Dataset):

    def __init__(self, voc_root, image_ids, image_size=IMAGE_SIZE, split="train"):
        self.voc_root   = Path(voc_root)
        self.image_ids  = image_ids
        self.image_size = image_size
        self.split      = split   

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]

       
        image = Image.open(
            self.voc_root / "JPEGImages" / f"{img_id}.jpg"
        ).convert("RGB")

        mask = Image.open(
            self.voc_root / "SegmentationClass" / f"{img_id}.png"
        )  # palette PNG — pixel values are class indices 0–20, 255

        
        image = image.resize(self.image_size, Image.BILINEAR)
        mask  = mask.resize(self.image_size,  Image.NEAREST)

       
        if self.split == "train":
            image, mask = apply_train_augmentations(image, mask)
        else:
            image, mask = apply_val_augmentations(image, mask)

        
        image_tensor = TF.to_tensor(image)                              # (3,H,W) float [0,1]
        image_tensor = TF.normalize(image_tensor, IMAGENET_MEAN, IMAGENET_STD)

        mask_tensor  = torch.from_numpy(np.array(mask)).long()          # (H,W) int64

        return image_tensor, mask_tensor

    def get_raw_sample(self, idx):
        
        img_id = self.image_ids[idx]
        image  = Image.open(self.voc_root / "JPEGImages"        / f"{img_id}.jpg").convert("RGB")
        mask   = Image.open(self.voc_root / "SegmentationClass" / f"{img_id}.png")
        image  = image.resize(self.image_size, Image.BILINEAR)
        mask   = mask.resize(self.image_size,  Image.NEAREST)
        return image, mask

def get_dataloaders(
    voc_root    = VOC_ROOT,
    split_dir   = SPLIT_DIR,
    image_size  = IMAGE_SIZE,
    batch_size  = 8,
    num_workers = 4,
):

    train_ids, val_ids = load_splits(split_dir)

    train_dataset = VOCSegDataset(voc_root, train_ids, image_size, split="train")
    val_dataset   = VOCSegDataset(voc_root, val_ids,   image_size, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
    )

    print(f"[INFO] Train: {len(train_dataset)} images | Val: {len(val_dataset)} images")
    return train_loader, val_loader

if __name__ == "__main__":

    train_loader, val_loader = get_dataloaders(batch_size=4, num_workers=0)

    imgs, masks = next(iter(train_loader))
    print(f"Image batch : {imgs.shape}  dtype={imgs.dtype}")
    print(f"Mask  batch : {masks.shape} dtype={masks.dtype}")
    print(f"Mask values : {torch.unique(masks).tolist()}")

    print("\n Stage 2 complete.")
