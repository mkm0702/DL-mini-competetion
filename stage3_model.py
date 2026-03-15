import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.models.segmentation import lraspp_mobilenet_v3_large
from pathlib import Path

NUM_CLASSES = 21
IMAGE_SIZE  = (300, 300)
CHECKPOINT  = Path("checkpoints/best_model.pth")

def probe_backbone_channels(input_size=(1, 3, 300, 300)):
   
    backbone = mobilenet_v3_small(weights=None)
    x = torch.randn(*input_size)

    print("MobileNetV3-Small layer shapes:")
    print(f"  Input : {list(x.shape)}")
    for i, layer in enumerate(backbone.features):
        x = layer(x)
        print(f"  features[{i:2d}] → {list(x.shape)}  ({layer.__class__.__name__})")

    return backbone

class MobileNetV3SmallSeg(nn.Module):
    
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()

        weights  = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)

        low_tap  = 4
        high_tap = len(backbone.features)

        low_ch, high_ch = self._get_channels(backbone, low_tap, high_tap)
        print(f"[MODEL] Low-level  tap features[:{low_tap}]  → {low_ch} channels")
        print(f"[MODEL] High-level tap features[:{high_tap}] → {high_ch} channels")

       
        self.low_encoder  = backbone.features[:low_tap]
        self.high_encoder = backbone.features[low_tap:]

        self.head = LRASPPHead(
            low_channels   = low_ch,
            high_channels  = high_ch,
            num_classes    = num_classes,
            inter_channels = 128,
        )

    @staticmethod
    def _get_channels(backbone, low_tap, high_tap):
       
        backbone.eval()
        with torch.no_grad():
            x    = torch.zeros(1, 3, 300, 300)
            low  = backbone.features[:low_tap](x)
            high = backbone.features[:high_tap](x)
        return low.shape[1], high.shape[1]

    def forward(self, x):
        input_size = x.shape[-2:]              

        low_feat   = self.low_encoder(x)        
        high_feat  = self.high_encoder(low_feat) 
        logits = self.head(low_feat, high_feat, input_size)
        mask = torch.argmax(logits, dim=1)     
        return logits, mask


# LRASPP HEAD

class LRASPPHead(nn.Module):

    def __init__(self, low_channels, high_channels, num_classes, inter_channels=128):
        super().__init__()

        # Compress high-level features
        self.high_branch = nn.Sequential(
            nn.Conv2d(high_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
        )

        # Global context gate
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(high_channels, inter_channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        # Fuse low + gated-high → class scores
        self.classifier = nn.Conv2d(
            low_channels + inter_channels, num_classes, kernel_size=1
        )

    def forward(self, low_feat, high_feat, output_size):
      
        high_out = self.high_branch(high_feat)    
        gate     = self.global_branch(high_feat)  
        gated    = high_out * gate               

        gated_up = F.interpolate(
            gated, size=low_feat.shape[-2:],
            mode="bilinear", align_corners=False,
        )   

        fused  = torch.cat([low_feat, gated_up], dim=1) 
        logits = self.classifier(fused)                

        logits = F.interpolate(
            logits, size=output_size,
            mode="bilinear", align_corners=False,
        )  
        return logits
    
class TorchvisionSegWrapper(nn.Module):
   
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()
        weights    = "DEFAULT" if pretrained else None
        self.model = lraspp_mobilenet_v3_large(
            weights=weights, num_classes=num_classes
        )

    def forward(self, x):
        out    = self.model(x)             
        logits = out["out"]
        mask   = torch.argmax(logits, dim=1)
        return logits, mask

def build_model(variant="small", pretrained=True, device="cpu"):
    
    if variant == "small":
        model = MobileNetV3SmallSeg(num_classes=NUM_CLASSES, pretrained=pretrained)
    elif variant == "large":
        model = TorchvisionSegWrapper(num_classes=NUM_CLASSES, pretrained=pretrained)
    else:
        raise ValueError(f"variant must be 'small' or 'large', got '{variant}'")

    model = model.to(device)
    print(f" Model variant='{variant}' ready on {device}")
    return model


def save_checkpoint(model, optimizer, epoch, dice_score, path=CHECKPOINT):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch":      epoch,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "dice_score": dice_score,
    }, path)
    print(f"[SAVED] {path}  (epoch {epoch}, Dice {dice_score:.4f})")


def load_checkpoint(model, path=CHECKPOINT, optimizer=None, device="cpu"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint at {path}")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer:
        optimizer.load_state_dict(ckpt["optimizer"])
    print(f"[LOADED] {path}  (epoch {ckpt['epoch']}, Dice {ckpt['dice_score']:.4f})")
    return ckpt["epoch"], ckpt["dice_score"]


def count_flops(model, input_size=(1, 3, 300, 300), device="cpu"):
    try:
        from thop import profile, clever_format
        dummy = torch.randn(*input_size).to(device)
        model.eval()
       
        _orig = model.forward
        model.forward = lambda x: _orig(x)[0]
        flops, params = profile(model, inputs=(dummy,), verbose=False)
        model.forward = _orig
        f, p = clever_format([flops, params], "%.2f")
        print(f"[FLOPS] {f}   [PARAMS] {p}")
        return flops, params
    except ImportError:
        print("[WARN] pip install thop  to count FLOPs")
        return None, None


if __name__ == "__main__":

    
    print("=" * 55)
    probe_backbone_channels()
    print("=" * 55)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[INFO] Device: {device}")

    print("\n Small model (custom MobileNetV3-Small) ──")
    model = build_model("small", pretrained=True, device=device)
    model.eval()

    dummy = torch.randn(1, 3, 300, 300).to(device)
    with torch.no_grad():
        logits, mask = model(dummy)

    print(f"logits : {logits.shape}   expected (1, 21, 300, 300)")
    print(f"mask   : {mask.shape}    expected (1, 300, 300)")
    print(f"dtype  : {mask.dtype}    expected torch.int64")
    print(f"range  : [{mask.min().item()}, {mask.max().item()}]")
    assert logits.shape == (1, 21, 300, 300), "Logits shape wrong!"
    assert mask.shape   == (1, 300, 300),     "Mask shape wrong!"
    print("[OK] All shape checks passed.")
    count_flops(model, device=device)

    # Test large model
    print("\n── Large model (torchvision MobileNetV3-Large) ──")
    model_l = build_model("large", pretrained=True, device=device)
    model_l.eval()
    with torch.no_grad():
        logits_l, mask_l = model_l(dummy)
    print(f"[CHECK] logits : {logits_l.shape}")
    print(f"[CHECK] mask   : {mask_l.shape}")
    count_flops(model_l, device=device)

    print("\n Stage 3 .")
