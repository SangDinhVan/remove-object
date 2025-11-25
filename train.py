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

from utils.checkpoints import save_checkpoint, load_checkpoint
from utils.logger import setup_logger
from utils.lr_scheduler import lr_scheduler

# NEW: import từ losses.py
from model.loss import (
    PatchDiscriminator,
    VGGPerceptualLoss,
    d_logistic_loss,
    g_nonsat_loss,
    r1_penalty,
)


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
#   RECONSTRUCTION LOSS
# =========================
def compute_inpaint_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    l1_loss: nn.L1Loss,
    rec_weight: float,
    hole_weight: float,
    valid_weight: float,
) -> torch.Tensor:
    """
    Reconstruction loss:
      - L1 vùng hole (mask = 1) với hệ số hole_weight
      - L1 vùng valid (mask = 0) với hệ số valid_weight
      - nhân rec_weight cho toàn bộ L1
    Nếu muốn đúng y paper MAT thì để rec_weight = 0 (không dùng L1).
    """
    if rec_weight <= 0:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    eps = 1e-6

    # L1 per-pixel: [B,3,H,W]
    l1_map = l1_loss(pred, target)

    # mask: [B,1,H,W] -> broadcast sang 3 channel
    if mask.shape[1] == 1 and l1_map.shape[1] != 1:
        hole_mask = mask.expand(-1, l1_map.shape[1], -1, -1)
    else:
        hole_mask = mask

    valid_mask = 1.0 - hole_mask

    hole_loss = (l1_map * hole_mask).sum() / (hole_mask.sum() + eps)
    valid_loss = (l1_map * valid_mask).sum() / (valid_mask.sum() + eps)

    rec_l1 = hole_weight * hole_loss + valid_weight * valid_loss
    rec_l1 = rec_weight * rec_l1
    return rec_l1


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


# =========================
#   TRAIN ONE EPOCH (NO GAN)
# =========================
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

        # ----- rec loss -----
        rec_loss = compute_inpaint_loss(
            pred=pred,
            target=target,
            mask=mask,
            l1_loss=l1_loss,
            rec_weight=rec_weight,
            hole_weight=hole_weight,
            valid_weight=valid_weight,
        )

        total_loss_val = rec_loss

        # perceptual (nếu có)
        if perc_weight > 0.0 and perceptual_loss is not None:
            perc = perceptual_loss(pred, target)
            total_loss_val = total_loss_val + perc_weight * perc

        total_loss_val.backward()
        optimizer.step()

        # PSNR
        with torch.no_grad():
            psnr = compute_psnr(pred.detach(), target.detach(), max_val=1.0)

        bsz = image.size(0)
        total_loss += total_loss_val.item() * bsz
        total_psnr += psnr.item() * bsz
        n_samples += bsz

        pbar.set_postfix({"loss": total_loss_val.item(), "psnr": psnr.item()})

    avg_loss = total_loss / max(1, n_samples)
    avg_psnr = total_psnr / max(1, n_samples)
    return avg_loss, avg_psnr


# =========================
#   TRAIN ONE EPOCH (GAN)
# =========================
def train_one_epoch_gan(
    G: nn.Module,
    D: nn.Module,
    model_name: str,
    loader: DataLoader,
    optimizer_G: torch.optim.Optimizer,
    optimizer_D: torch.optim.Optimizer,
    device: str,
    epoch: int,
    num_epochs: int,
    l1_loss: nn.L1Loss,
    rec_weight: float,
    hole_weight: float,
    valid_weight: float,
    perc_weight: float,
    perceptual_loss: nn.Module,
    r1_gamma: float,
) -> Tuple[float, float, float]:
    """
    Train 1 epoch theo loss của MAT:
      - D: logistic + R1
      - G: adv + lambda_P * perceptual (+ optional rec nếu rec_weight > 0)

    Trả về:
      - avg_G_loss
      - avg_D_loss
      - avg_psnr
    """
    G.train()
    D.train()

    total_G_loss = 0.0
    total_D_loss = 0.0
    total_psnr = 0.0
    n_samples = 0

    pbar = tqdm(loader, desc=f"Train(GAN) Epoch {epoch+1}/{num_epochs}", leave=False)
    for batch in pbar:
        image = batch["image"].to(device)
        mask = batch["mask"].to(device)
        target = batch["target"].to(device)

        masked = image * (1 - mask)
        inp = torch.cat([masked, mask], dim=1)

        # ======================
        # 1) UPDATE D
        # ======================
        optimizer_D.zero_grad()

        # real
        real = target.detach()
        real.requires_grad_(True)
        logits_real = D(real)

        # fake (detach G)
        if model_name == "efficientnet_mat":
            style_dim = getattr(G, "style_dim", 256)
            z = torch.randn(image.size(0), style_dim, device=device)
            fake = G(inp, mask, z).detach()
        else:
            fake = G(inp).detach()

        logits_fake = D(fake)

        loss_D_adv = d_logistic_loss(logits_real, logits_fake)
        # R1 regularization (trên real)
        r1 = r1_penalty(logits_real, real)
        loss_D = loss_D_adv + (r1_gamma / 2.0) * r1

        loss_D.backward()
        optimizer_D.step()

        # ======================
        # 2) UPDATE G
        # ======================
        optimizer_G.zero_grad()

        if model_name == "efficientnet_mat":
            style_dim = getattr(G, "style_dim", 256)
            z = torch.randn(image.size(0), style_dim, device=device)
            fake = G(inp, mask, z)
        else:
            fake = G(inp)

        if fake.shape[-2:] != target.shape[-2:]:
            fake = F.interpolate(
                fake,
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        logits_fake_for_G = D(fake)
        loss_G_adv = g_nonsat_loss(logits_fake_for_G)

        # Perceptual loss
        loss_P = perceptual_loss(fake, target) if perc_weight > 0 else torch.tensor(0.0, device=device)

        # Optional rec loss (nếu rec_weight>0)
        rec_loss = compute_inpaint_loss(
            pred=fake,
            target=target,
            mask=mask,
            l1_loss=l1_loss,
            rec_weight=rec_weight,
            hole_weight=hole_weight,
            valid_weight=valid_weight,
        )

        loss_G = loss_G_adv + perc_weight * loss_P + rec_loss

        loss_G.backward()
        optimizer_G.step()

        with torch.no_grad():
            psnr = compute_psnr(fake.detach(), target.detach(), max_val=1.0)

        bsz = image.size(0)
        total_G_loss += loss_G.item() * bsz
        total_D_loss += loss_D.item() * bsz
        total_psnr += psnr.item() * bsz
        n_samples += bsz

        pbar.set_postfix({
            "G_loss": loss_G.item(),
            "D_loss": loss_D.item(),
            "psnr": psnr.item(),
        })

    avg_G_loss = total_G_loss / max(1, n_samples)
    avg_D_loss = total_D_loss / max(1, n_samples)
    avg_psnr = total_psnr / max(1, n_samples)
    return avg_G_loss, avg_D_loss, avg_psnr


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
    Val: chỉ tính rec + perceptual, không có GAN.
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

        # rec loss
        rec_loss = compute_inpaint_loss(
            pred=pred,
            target=target,
            mask=mask,
            l1_loss=l1_loss,
            rec_weight=rec_weight,
            hole_weight=hole_weight,
            valid_weight=valid_weight,
        )

        total_loss_val = rec_loss
        if perc_weight > 0.0 and perceptual_loss is not None:
            perc = perceptual_loss(pred, target)
            total_loss_val = total_loss_val + perc_weight * perc

        psnr = compute_psnr(pred, target, max_val=1.0)

        bsz = image.size(0)
        total_loss += total_loss_val.item() * bsz
        total_psnr += psnr.item() * bsz
        n_samples += bsz

        pbar.set_postfix({"loss": total_loss_val.item(), "psnr": psnr.item()})

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
    optimizer_G = build_optimizer(cfg_train, model)

    # ----- Loss setup -----
    # Nếu muốn đúng y paper: để rec_weight = 0, perceptual_weight > 0
    rec_weight   = float(cfg_loss.get("rec_weight", 0.0))          # tổng rec L1
    perc_weight  = float(cfg_loss.get("perceptual_weight", 0.1))   # lambda_P
    hole_weight  = float(cfg_loss.get("hole_weight", 6.0))         # vùng mask
    valid_weight = float(cfg_loss.get("valid_weight", 1.0))        # vùng không mask

    use_gan   = bool(cfg_loss.get("use_gan", False))
    r1_gamma  = float(cfg_loss.get("r1_gamma", 10.0))

    # L1 per-pixel để tự tính mask
    l1_loss = nn.L1Loss(reduction="none")

    # Perceptual loss (luôn tạo nếu perc_weight>0)
    perceptual_loss = None
    if perc_weight > 0:
        perceptual_loss = VGGPerceptualLoss().to(device)
        perceptual_loss.eval()
        logger.info("[INFO] Using VGG19 perceptual loss (conv4_4, conv5_4)")
    else:
        logger.info("[INFO] No perceptual loss (perceptual_weight=0)")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"[INFO] Model params: total={total_params:,}, trainable={trainable_params:,}")

    num_epochs = cfg_train.get("epochs", 50)
    lr = float(cfg_train.get("lr", 1e-4))
    min_lr = float(cfg_train.get("min_lr", 1e-6))
    warmup_epochs = int(cfg_train.get("warmup_epochs", 3))

    scheduler_G = lr_scheduler(
        optimizer_G,
        warmup_epochs=warmup_epochs,
        total_epochs=num_epochs,
        min_lr=min_lr,
        max_lr=lr,
    )

    # ========== Discriminator (nếu dùng GAN) ==========
    if use_gan:
        D = PatchDiscriminator(in_channels=3).to(device)
        d_lr = float(cfg_loss.get("d_lr", lr))
        optimizer_D = torch.optim.Adam(
            D.parameters(),
            lr=d_lr,
            betas=(0.0, 0.9),   # như StyleGAN2
        )
        logger.info(f"[INFO] Using GAN loss (r1_gamma={r1_gamma}, lambda_P={perc_weight})")
    else:
        D = None
        optimizer_D = None
        logger.info("[INFO] Training WITHOUT GAN loss (L1 + perceptual only)")

    # ========== 5. Checkpoint (resume nếu có) ==========
    save_dir = cfg_ckpt.get("save_dir", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    resume_path = cfg_ckpt.get("resume", "")

    start_epoch = 0
    best_val_loss = float("inf")

    if resume_path and os.path.isfile(resume_path):
        logger.info(f"[INFO] Resuming from checkpoint: {resume_path}")
        model, optimizer_G, start_epoch = load_checkpoint(
            resume_path, model, optimizer_G, device=device, load_opt=True
        )
    else:
        logger.info("[INFO] Training from scratch.")

    # ========== 6. Training loop ==========
    save_interval = cfg_ckpt.get("save_interval", 5)

    for epoch in tqdm(range(start_epoch, num_epochs), desc="Epochs"):
        # train
        if use_gan:
            train_G_loss, train_D_loss, train_psnr = train_one_epoch_gan(
                G=model,
                D=D,
                model_name=model_name,
                loader=train_loader,
                optimizer_G=optimizer_G,
                optimizer_D=optimizer_D,
                device=device,
                epoch=epoch,
                num_epochs=num_epochs,
                l1_loss=l1_loss,
                rec_weight=rec_weight,
                hole_weight=hole_weight,
                valid_weight=valid_weight,
                perc_weight=perc_weight,
                perceptual_loss=perceptual_loss,
                r1_gamma=r1_gamma,
            )
            train_loss = train_G_loss
        else:
            train_loss, train_psnr = train_one_epoch(
                model=model,
                model_name=model_name,
                loader=train_loader,
                optimizer=optimizer_G,
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
            train_D_loss = float("nan")

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

        # step scheduler cho G
        scheduler_G.step()

        # log ra file CSV-like
        log_line = (
            f"{epoch},"
            f"{train_loss:.6f},{train_psnr:.4f},{train_D_loss:.6f if use_gan else float('nan')},-,"  # G_loss,D_loss,...
            f"{val_loss:.6f},{val_psnr:.4f},-,-"
        )
        logger.info(log_line)

        # log ra console
        if use_gan:
            logger.info(
                f"[Epoch {epoch}/{num_epochs}] "
                f"G_loss={train_loss:.6f} | D_loss={train_D_loss:.6f} | "
                f"val_loss={val_loss:.6f} | train_psnr={train_psnr:.2f} | val_psnr={val_psnr:.2f}"
            )
        else:
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
                optimizer=optimizer_G,
                save_dir=save_dir,
                epoch=epoch,
                is_best=is_best,
            )

    logger.info("[INFO] Training finished.")


if __name__ == "__main__":
    main()
