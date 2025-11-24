import os
import yaml
from typing import Tuple
from tqdm.auto import tqdm
import torch.nn.functional as F

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from preprocessing.dataset import InpaintDataset                 # chỉnh lại path nếu cần
from model.efficientnet_unet import EfficientNetUNet
from model.efficientnet_mat import EfficientNetMAT
from torchvision import models

from utils.checkpoints import save_checkpoint, load_checkpoint
from utils.logger import setup_logger
from utils.lr_scheduler import lr_scheduler


# =========================
#      PSNR FUNCTION
# =========================
def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """
    pred, target: tensor [B, C, H, W], giá trị thường trong [0,1]
    max_val: giá trị max của pixel, nếu bạn để 0..255 thì max_val=255
    """
    mse = F.mse_loss(pred, target, reduction="mean")
    psnr = 10 * torch.log10(max_val ** 2 / (mse + 1e-8))
    return psnr


# =========================
#   PERCEPTUAL LOSS (VGG19)
# =========================
class VGGPerceptualLoss(nn.Module):
    """
    Dùng feature VGG19 (pretrained ImageNet) để tính perceptual loss.
    Ở đây lấy 2 tầng: relu3_3 và relu4_3 (chỉ là ví dụ hợp lý).
    """
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features

        # slice: [0..16] ~ lên tới relu3_3, [16..25] ~ lên tới relu4_3
        self.slice1 = nn.Sequential(*[vgg[i] for i in range(16)])
        self.slice2 = nn.Sequential(*[vgg[i] for i in range(16, 25)])

        # không cho VGG update khi train
        for p in self.parameters():
            p.requires_grad = False

        # chuẩn hóa theo ImageNet
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        x, y: [B,3,H,W] trong [0,1]
        trả về tổng L1 giữa feature map 2 tầng
        """
        x = self._norm(x)
        y = self._norm(y)

        fx1 = self.slice1(x)
        fy1 = self.slice1(y)
        fx2 = self.slice2(fx1)
        fy2 = self.slice2(fy1)

        loss1 = torch.mean(torch.abs(fx1 - fy1))
        loss2 = torch.mean(torch.abs(fx2 - fy2))
        return loss1 + loss2


def build_dataloaders(cfg_dataset: dict, cfg_train: dict):

    dataset = InpaintDataset.from_config(cfg_dataset)

    val_ratio = cfg_train.get("val_ratio", 0.1)
    val_ratio = min(max(val_ratio, 0.0), 0.5)  # clamp 0..0.5

    n_total = len(dataset)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val

    if n_val > 0:
        train_set, val_set = random_split(dataset, [n_train, n_val])
    else:
        train_set, val_set = dataset, None

    batch_size = cfg_train.get("batch_size", 8)
    num_workers = cfg_train.get("num_workers", 4)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = None
    if val_set is not None:
        val_loader = DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader


def build_model(cfg_model: dict, device: str):
    name = cfg_model.get("name", "efficientnet_unet").lower()

    if name == "efficientnet_unet":
        model = EfficientNetUNet.from_config(cfg_model)
    elif name == "efficientnet_mat":
        model = EfficientNetMAT.from_config(cfg_model)
    else:
        raise ValueError(f"Unknown model name: {name}")

    model = model.to(device)
    return model, name


def build_optimizer(cfg_train: dict, model: nn.Module):
    lr = cfg_train.get("lr", 1e-4)
    weight_decay = cfg_train.get("weight_decay", 0.0)
    lr = float(lr)
    weight_decay = float(weight_decay)
    optimizer_type = cfg_train.get("optimizer", "adamw").lower()

    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    return optimizer


def compute_inpaint_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    l1_loss: nn.L1Loss,
    rec_weight: float,
    hole_weight: float,
    valid_weight: float,
    perc_weight: float,
    perceptual_loss: nn.Module = None,
) -> torch.Tensor:
    """
    Tính loss tổng:
      - L1 vùng hole (mask = 1) với hệ số hole_weight
      - L1 vùng valid (mask = 0) với hệ số valid_weight
      - nhân rec_weight cho toàn bộ L1
      - cộng thêm perceptual_loss * perc_weight (nếu có)
    """
    eps = 1e-6

    # L1 per-pixel: [B,3,H,W]
    l1_map = l1_loss(pred, target)

    # mask: [B,1,H,W] -> broadcast sang 3 channel
    if mask.shape[1] == 1 and l1_map.shape[1] != 1:
        hole_mask = mask.expand(-1, l1_map.shape[1], -1, -1)
    else:
        hole_mask = mask

    valid_mask = 1.0 - hole_mask

    # mean L1 trên từng vùng
    hole_loss = (l1_map * hole_mask).sum() / (hole_mask.sum() + eps)
    valid_loss = (l1_map * valid_mask).sum() / (valid_mask.sum() + eps)

    rec_l1 = hole_weight * hole_loss + valid_weight * valid_loss
    rec_l1 = rec_weight * rec_l1

    total_loss = rec_l1

    if perc_weight > 0.0 and perceptual_loss is not None:
        perc = perceptual_loss(pred, target)
        total_loss = total_loss + perc_weight * perc

    return total_loss


def train_one_epoch(
    model: nn.Module,
    model_name: str,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    num_epochs: int,
    l1_loss: nn.L1Loss,
    rec_weight: float,
    hole_weight: float,
    valid_weight: float,
    perc_weight: float,
    perceptual_loss: nn.Module = None,
) -> Tuple[float, float]:
    """
    Trả về:
      - avg_train_loss
      - avg_train_psnr
    """
    model.train()
    total_loss = 0.0
    total_psnr = 0.0
    n_samples = 0

    pbar = tqdm(loader, desc=f"Train Epoch {epoch+1}/{num_epochs}", leave=False)
    for batch in pbar:
        image = batch["image"].to(device)     # [B,3,H,W]
        mask = batch["mask"].to(device)       # [B,1,H,W]
        target = batch["target"].to(device)   # [B,3,H,W]

        masked = image * (1 - mask)
        inp = torch.cat([masked, mask], dim=1)  # [B,4,H,W]

        optimizer.zero_grad()

        if model_name == "efficientnet_mat":
            style_dim = getattr(model, "style_dim", 256)
            z = torch.randn(image.size(0), style_dim, device=device)
            pred = model(inp, mask, z)        # [B,3,h,w]
        else:
            pred = model(inp)                 # [B,3,h,w]

        if pred.shape[-2:] != target.shape[-2:]:
            pred = F.interpolate(
                pred,
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        # ----- loss mới: hole/valid + perceptual -----
        loss = compute_inpaint_loss(
            pred=pred,
            target=target,
            mask=mask,
            l1_loss=l1_loss,
            rec_weight=rec_weight,
            hole_weight=hole_weight,
            valid_weight=valid_weight,
            perc_weight=perc_weight,
            perceptual_loss=perceptual_loss,
        )

        loss.backward()
        optimizer.step()

        # PSNR
        with torch.no_grad():
            psnr = compute_psnr(pred.detach(), target.detach(), max_val=1.0)

        bsz = image.size(0)
        total_loss += loss.item() * bsz
        total_psnr += psnr.item() * bsz
        n_samples += bsz

        pbar.set_postfix({"loss": loss.item(), "psnr": psnr.item()})

    avg_loss = total_loss / max(1, n_samples)
    avg_psnr = total_psnr / max(1, n_samples)
    return avg_loss, avg_psnr


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    model_name: str,
    loader: DataLoader,
    device: str,
    epoch: int,
    num_epochs: int,
    l1_loss: nn.L1Loss,
    rec_weight: float,
    hole_weight: float,
    valid_weight: float,
    perc_weight: float,
    perceptual_loss: nn.Module = None,
) -> Tuple[float, float]:
    """
    Trả về:
      - avg_val_loss
      - avg_val_psnr
    """
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    n_samples = 0

    pbar = tqdm(loader, desc=f"Val   Epoch {epoch+1}/{num_epochs}", leave=False)
    for batch in pbar:
        image = batch["image"].to(device)
        mask = batch["mask"].to(device)
        target = batch["target"].to(device)

        masked = image * (1 - mask)
        inp = torch.cat([masked, mask], dim=1)

        if model_name == "efficientnet_mat":
            style_dim = getattr(model, "style_dim", 256)
            z = torch.randn(image.size(0), style_dim, device=device)
            pred = model(inp, mask, z)
        else:
            pred = model(inp)

        if pred.shape[-2:] != target.shape[-2:]:
            pred = F.interpolate(
                pred,
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        # dùng cùng loss như train (nhưng không backward)
        loss = compute_inpaint_loss(
            pred=pred,
            target=target,
            mask=mask,
            l1_loss=l1_loss,
            rec_weight=rec_weight,
            hole_weight=hole_weight,
            valid_weight=valid_weight,
            perc_weight=perc_weight,
            perceptual_loss=perceptual_loss,
        )

        psnr = compute_psnr(pred, target, max_val=1.0)

        bsz = image.size(0)
        total_loss += loss.item() * bsz
        total_psnr += psnr.item() * bsz
        n_samples += bsz

        pbar.set_postfix({"loss": loss.item(), "psnr": psnr.item()})

    avg_loss = total_loss / max(1, n_samples)
    avg_psnr = total_psnr / max(1, n_samples)
    return avg_loss, avg_psnr


def main():
    # ========== 1. Đọc config ==========
    config_path = os.path.join("config", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg_dataset = cfg["dataset"]
    cfg_model = cfg["model"]
    cfg_train = cfg["train"]
    cfg_ckpt = cfg.get("checkpoint", {})
    cfg_log = cfg.get("logging", {})
    cfg_loss = cfg.get("loss", {})

    device = cfg_train.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    # ========== 2. Logger ==========
    log_dir = cfg_log.get("log_dir", "logs")
    logger = setup_logger(log_dir, log_name="train")

    # ========== 3. DataLoader ==========
    train_loader, val_loader = build_dataloaders(cfg_dataset, cfg_train)

    logger.info(f"[INFO] Train size: {len(train_loader.dataset)}")
    if val_loader is not None:
        logger.info(f"[INFO] Val size:   {len(val_loader.dataset)}")
    else:
        logger.info("[INFO] No validation set (val_ratio=0).")

    # ========== 4. Model & Optimizer & Scheduler ==========
    model, model_name = build_model(cfg_model, device)
    optimizer = build_optimizer(cfg_train, model)

    # ----- Loss setup -----
    rec_weight   = float(cfg_loss.get("rec_weight", 1.0))          # tổng rec L1
    perc_weight  = float(cfg_loss.get("perceptual_weight", 0.0))   # >0 thì bật perceptual
    hole_weight  = float(cfg_loss.get("hole_weight", 6.0))         # vùng mask
    valid_weight = float(cfg_loss.get("valid_weight", 1.0))        # vùng không mask

    # L1 per-pixel để tự tính mask
    l1_loss = nn.L1Loss(reduction="none")

    perceptual_loss = None
    if perc_weight > 0:
        perceptual_loss = VGGPerceptualLoss().to(device)
        perceptual_loss.eval()
        logger.info("[INFO] Using VGG19 perceptual loss")
    else:
        logger.info("[INFO] No perceptual loss (perceptual_weight=0)")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"[INFO] Model params: total={total_params:,}, trainable={trainable_params:,}")

    num_epochs = cfg_train.get("epochs", 50)
    lr = float(cfg_train.get("lr", 1e-4))
    min_lr = float(cfg_train.get("min_lr", 1e-6))
    warmup_epochs = int(cfg_train.get("warmup_epochs", 3))

    scheduler = lr_scheduler(
        optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=num_epochs,
        min_lr=min_lr,
        max_lr=lr,
    )

    # ========== 5. Checkpoint (resume nếu có) ==========
    save_dir = cfg_ckpt.get("save_dir", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    resume_path = cfg_ckpt.get("resume", "")

    start_epoch = 0
    best_val_loss = float("inf")

    if resume_path and os.path.isfile(resume_path):
        logger.info(f"[INFO] Resuming from checkpoint: {resume_path}")
        model, optimizer, start_epoch = load_checkpoint(
            resume_path, model, optimizer, device=device, load_opt=True
        )
    else:
        logger.info("[INFO] Training from scratch.")

    # ========== 6. Training loop ==========
    save_interval = cfg_ckpt.get("save_interval", 5)
    log_interval = cfg_log.get("log_interval", 50)  # hiện tại chưa dùng, giữ lại nếu cần

    for epoch in tqdm(range(start_epoch, num_epochs), desc="Epochs"):
        # train
        train_loss, train_psnr = train_one_epoch(
            model=model,
            model_name=model_name,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            num_epochs=num_epochs,
            l1_loss=l1_loss,
            rec_weight=rec_weight,
            hole_weight=hole_weight,
            valid_weight=valid_weight,
            perc_weight=perc_weight,
            perceptual_loss=perceptual_loss,
        )

        # val
        if val_loader is not None:
            val_loss, val_psnr = validate_one_epoch(
                model=model,
                model_name=model_name,
                loader=val_loader,
                device=device,
                epoch=epoch,
                num_epochs=num_epochs,
                l1_loss=l1_loss,
                rec_weight=rec_weight,
                hole_weight=hole_weight,
                valid_weight=valid_weight,
                perc_weight=perc_weight,
                perceptual_loss=perceptual_loss,
            )
        else:
            val_loss, val_psnr = float("nan"), float("nan")

        # step scheduler
        scheduler.step()

        # log ra file CSV-like
        log_line = (
            f"{epoch},"
            f"{train_loss:.6f},{train_psnr:.4f},-,-,"
            f"{val_loss:.6f},{val_psnr:.4f},-,-"
        )
        logger.info(log_line)

        # log ra console
        logger.info(
            f"[Epoch {epoch}/{num_epochs}] "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
            f"train_psnr={train_psnr:.2f} | val_psnr={val_psnr:.2f}"
        )

        # save checkpoint
        is_best = (val_loader is not None) and (val_loss < best_val_loss)
        if is_best:
            best_val_loss = val_loss

        if ((epoch + 1) % save_interval == 0) or is_best:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                save_dir=save_dir,
                epoch=epoch,
                is_best=is_best,
            )

    logger.info("[INFO] Training finished.")


if __name__ == "__main__":
    main()
