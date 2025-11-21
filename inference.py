import os
import yaml
from typing import Tuple

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

from model.efficientnet_unet import EfficientNetUNet
from model.efficientnet_mat import EfficientNetMAT
from utils.checkpoints import load_checkpoint


# =========================
#   CONFIG & MODEL LOADING
# =========================

def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def build_model_from_config(
    cfg: dict,
    ckpt_path: str = "output/checkpoints/best_model.pth",
    device: str | None = None,
):
    """
    Tạo model từ config + load checkpoint.
    Trả về: model, cfg, device
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg_model = cfg["model"]
    name = cfg_model.get("name", "efficientnet_unet").lower()

    if name == "efficientnet_unet":
        model = EfficientNetUNet.from_config(cfg_model)
    elif name == "efficientnet_mat":
        model = EfficientNetMAT.from_config(cfg_model)
    else:
        raise ValueError(f"Unknown model name: {name}")

    model = model.to(device)

    if ckpt_path and os.path.isfile(ckpt_path):
        print(f"[INFO] Loading checkpoint from {ckpt_path}")
        model, _, _ = load_checkpoint(
            ckpt_path, model, optimizer=None, device=device, load_opt=False
        )
    else:
        print(f"[WARN] Checkpoint {ckpt_path} không tồn tại, dùng random weight.")

    model.eval()
    return model, cfg, device


# =========================
#     PRE/POST PROCESS
# =========================

def _pil_to_tensor(img: Image.Image, size: int) -> torch.Tensor:
    """
    PIL RGB -> tensor [1,3,H,W], value [0,1]
    Resize về (size, size)
    """
    img = img.convert("RGB")
    img = img.resize((size, size), Image.LANCZOS)
    arr = np.array(img).astype("float32") / 255.0  # [H,W,3]
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    return t


def _mask_pil_to_tensor(mask: Image.Image, size: int) -> torch.Tensor:
    """
    PIL -> mask tensor [1,1,H,W], giá trị 0 hoặc 1
    """
    mask = mask.convert("L")
    mask = mask.resize((size, size), Image.NEAREST)
    arr = np.array(mask).astype("float32") / 255.0
    arr = (arr > 0.5).astype("float32")  # binary
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    return t


def _tensor_to_pil(img_tensor: torch.Tensor) -> Image.Image:
    """
    tensor [1,3,H,W] hoặc [3,H,W] -> PIL RGB, value [0,1]
    """
    if img_tensor.dim() == 4:
        img_tensor = img_tensor[0]
    img_tensor = img_tensor.clamp(0, 1)
    arr = (img_tensor.detach().cpu().permute(1, 2, 0).numpy() * 255.0).astype("uint8")
    return Image.fromarray(arr)


# =========================
#       INFERENCE API
# =========================

@torch.no_grad()
def inpaint_pil(
    model: torch.nn.Module,
    cfg: dict,
    device: str,
    image_pil: Image.Image,
    mask_pil: Image.Image,
) -> Image.Image:
    """
    Inpaint 1 ảnh PIL với mask PIL, trả về PIL inpainted.
    Sử dụng cùng pipeline như khi train:
      - masked = image * (1 - mask)
      - inp = cat([masked, mask], dim=1)
    """
    img_size = cfg["dataset"]["image_size"]
    cfg_model = cfg["model"]
    model_name = cfg_model.get("name", "efficientnet_unet").lower()

    # 1) Chuẩn bị tensor
    image_t = _pil_to_tensor(image_pil, img_size).to(device)  # [1,3,H,W]
    mask_t = _mask_pil_to_tensor(mask_pil, img_size).to(device)  # [1,1,H,W]

    masked = image_t * (1.0 - mask_t)
    inp = torch.cat([masked, mask_t], dim=1)  # [1,4,H,W]

    # 2) Forward
    if model_name == "efficientnet_mat":
        style_dim = getattr(model, "style_dim", 256)
        z = torch.randn(1, style_dim, device=device)
        pred = model(inp, mask_t, z)  # [1,3,h,w]
    else:
        pred = model(inp)  # [1,3,h,w]

    # 3) Resize về đúng size target (nếu lệch)
    if pred.shape[-2:] != image_t.shape[-2:]:
        pred = F.interpolate(
            pred,
            size=image_t.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    # 4) Tensor -> PIL
    out_pil = _tensor_to_pil(pred)
    return out_pil


# =========================
#   CLI DEMO (tùy chọn)
# =========================

if __name__ == "__main__":
    """
    Dùng kiểu:
      python inference.py \
        --image /kaggle/input/datainpainted/data/images/xxx.png \
        --mask  /kaggle/input/datainpainted/data/masks/xxx.png \
        --out   result.png
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--ckpt", type=str, default="output/checkpoints/best_model.pth")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--mask", type=str, required=True)
    parser.add_argument("--out", type=str, default="result.png")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model, cfg, device = build_model_from_config(cfg, ckpt_path=args.ckpt)

    img_pil = Image.open(args.image).convert("RGB")
    mask_pil = Image.open(args.mask)

    out_pil = inpaint_pil(model, cfg, device, img_pil, mask_pil)
    out_pil.save(args.out)
    print(f"[INFO] Saved result to {args.out}")
