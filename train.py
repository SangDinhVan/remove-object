import os
import yaml
from typing import Tuple
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from preprocessing.dataset import InpaintDataset
from model.efficientnet_unet import EfficientNetUNet
from model.efficientnet_mat import EfficientNetMAT  # nếu vẫn muốn giữ
from model.mat_lama_net import MatLaMaNet, PatchDiscriminator

from utils.checkpoints import save_checkpoint, load_checkpoint
from utils.logger import setup_logger
from utils.lr_scheduler import lr_scheduler

from model.loss import (
    compute_psnr,
    VGGPerceptualLoss,
    compute_inpaint_loss,
    d_logistic_loss,
    g_nonsat_loss,
    r1_penalty,
)


# ==========================
#   DATALOADERS
# ==========================

def build_dataloaders(cfg_dataset: dict, cfg_train: dict):
    dataset = InpaintDataset.from_config(cfg_dataset)

    val_ratio = cfg_train.get("val_ratio", 0.1)
    val_ratio = min(max(val_ratio, 0.0), 0.5)

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


# ==========================
#   MODEL / OPTIMIZER BUILDERS
# ==========================

def build_model(cfg_model: dict, device: str):
    name = cfg_model.get("name", "efficientnet_unet").lower()

    if name == "efficientnet_unet":
        model = EfficientNetUNet.from_config(cfg_model)
    elif name == "efficientnet_mat":
        model = EfficientNetMAT.from_config(cfg_model)
    elif name == "mat_lama_net":
        model = MatLaMaNet.from_config(cfg_model)
    else:
        raise ValueError(f"Unknown model name: {name}")

    model = model.to(device)
    return model, name


def build_optimizer(cfg_train: dict, model: nn.Module):
    lr = float(cfg_train.get("lr", 1e-4))
    weight_decay = float(cfg_train.get("weight_decay", 0.0))
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


# ==========================
#   TRAIN ONE EPOCH (G + optional GAN)
# ==========================

def train_one_epoch(
    G: nn.Module,
    D: nn.Module,
    use_gan: bool,
    model_name: str,
    loader: DataLoader,
    opt_G: torch.optim.Optimizer,
    opt_D: torch.optim.Optimizer,
    device: str,
    epoch: int,
    num_epochs: int,
    l1_loss: nn.L1Loss,
    rec_weight: float,
    hole_weight: float,
    valid_weight: float,
    perc_weight: float,
    perceptual_loss: nn.Module,
    gan_weight: float,
    r1_gamma: float,
) -> Tuple[float, float, float, float]:
    """
    Trả về:
      - avg_G_loss
      - avg_D_loss (0 nếu không dùng GAN)
      - avg_psnr
      - avg_rec_loss (L1+perceptual)
    """
    G.train()
    if D is not None:
        D.train()

    total_G_loss = 0.0
    total_D_loss = 0.0
    total_psnr = 0.0
    total_rec = 0.0
    n_samples = 0

    pbar = tqdm(loader, desc=f"Train Epoch {epoch+1}/{num_epochs}", leave=False)
    for batch in pbar:
        image = batch["image"].to(device)   # [B,3,H,W]
        mask = batch["mask"].to(device)     # [B,1,H,W]
        target = batch["target"].to(device) # [B,3,H,W]

        masked = image * (1 - mask)
        inp = torch.cat([masked, mask], dim=1)  # [B,4,H,W]

        bsz = image.size(0)
        n_samples += bsz

        # ========== 1) Train D ==========
        loss_D_val = 0.0
        if use_gan:
            opt_D.zero_grad(set_to_none=True)

            with torch.no_grad():
                if model_name == "mat_lama_net" or model_name == "efficientnet_mat":
                    style_dim = getattr(G, "style_dim", 256)
                    z = torch.randn(bsz, style_dim, device=device)
                    fake = G(inp, mask, z)
                else:
                    fake = G(inp)

            fake_detach = fake.detach()

            real_img = target.detach().requires_grad_(True)
            real_logits = D(real_img)
            fake_logits = D(fake_detach)

            loss_D_adv = d_logistic_loss(real_logits, fake_logits)
            r1 = r1_penalty(real_logits, real_img)

            loss_D = loss_D_adv + (r1_gamma / 2.0) * r1
            loss_D.backward()
            opt_D.step()

            loss_D_val = loss_D.item()

        # ========== 2) Train G ==========
        opt_G.zero_grad(set_to_none=True)

        if model_name == "mat_lama_net" or model_name == "efficientnet_mat":
            style_dim = getattr(G, "style_dim", 256)
            z = torch.randn(bsz, style_dim, device=device)
            pred = G(inp, mask, z)
        else:
            pred = G(inp)

        if pred.shape[-2:] != target.shape[-2:]:
            pred = F.interpolate(
                pred,
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        rec_loss = compute_inpaint_loss(
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

        if use_gan:
            fake_logits_for_G = D(pred)
            loss_G_adv = g_nonsat_loss(fake_logits_for_G)
            loss_G = rec_loss + gan_weight * loss_G_adv
        else:
            loss_G = rec_loss

        loss_G.backward()
        opt_G.step()

        with torch.no_grad():
            psnr = compute_psnr(pred.detach(), target.detach(), max_val=1.0)

        total_G_loss += loss_G.item() * bsz
        total_D_loss += loss_D_val * bsz
        total_psnr += psnr.item() * bsz
        total_rec += rec_loss.item() * bsz

        pbar.set_postfix({
            "G_loss": loss_G.item(),
            "D_loss": loss_D_val,
            "psnr": psnr.item(),
        })

    avg_G_loss = total_G_loss / max(1, n_samples)
    avg_D_loss = total_D_loss / max(1, n_samples)
    avg_psnr = total_psnr / max(1, n_samples)
    avg_rec = total_rec / max(1, n_samples)
    return avg_G_loss, avg_D_loss, avg_psnr, avg_rec


# ==========================
#   VALIDATION
# ==========================

@torch.no_grad()
def validate_one_epoch(
    G: nn.Module,
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
) -> Tuple[float, float, float]:
    """
    Trả về:
      - avg_val_loss (rec)
      - avg_val_psnr
      - avg_val_rec (giống avg_val_loss, cho dễ đọc)
    """
    G.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_rec = 0.0
    n_samples = 0

    pbar = tqdm(loader, desc=f"Val   Epoch {epoch+1}/{num_epochs}", leave=False)
    for batch in pbar:
        image = batch["image"].to(device)
        mask = batch["mask"].to(device)
        target = batch["target"].to(device)

        masked = image * (1 - mask)
        inp = torch.cat([masked, mask], dim=1)

        bsz = image.size(0)
        n_samples += bsz

        if model_name == "mat_lama_net" or model_name == "efficientnet_mat":
            style_dim = getattr(G, "style_dim", 256)
            z = torch.randn(bsz, style_dim, device=device)
            pred = G(inp, mask, z)
        else:
            pred = G(inp)

        if pred.shape[-2:] != target.shape[-2:]:
            pred = F.interpolate(
                pred,
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

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

        total_loss += loss.item() * bsz
        total_psnr += psnr.item() * bsz
        total_rec += loss.item() * bsz

        pbar.set_postfix({"loss": loss.item(), "psnr": psnr.item()})

    avg_loss = total_loss / max(1, n_samples)
    avg_psnr = total_psnr / max(1, n_samples)
    avg_rec = total_rec / max(1, n_samples)
    return avg_loss, avg_psnr, avg_rec


# ==========================
#   MAIN
# ==========================

def main():
    # 1. Đọc config
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

    # 2. Logger
    log_dir = cfg_log.get("log_dir", "logs")
    logger = setup_logger(log_dir, log_name="train")

    # 3. DataLoader
    train_loader, val_loader = build_dataloaders(cfg_dataset, cfg_train)
    logger.info(f"[INFO] Train size: {len(train_loader.dataset)}")
    if val_loader is not None:
        logger.info(f"[INFO] Val size:   {len(val_loader.dataset)}")
    else:
        logger.info("[INFO] No validation set (val_ratio=0).")

    # 4. Model & Optimizers & Scheduler
    G, model_name = build_model(cfg_model, device)
    opt_G = build_optimizer(cfg_train, G)

    # ----- Loss configs -----
    rec_weight   = float(cfg_loss.get("rec_weight", 1.0))
    perc_weight  = float(cfg_loss.get("perceptual_weight", 0.0))
    hole_weight  = float(cfg_loss.get("hole_weight", 6.0))
    valid_weight = float(cfg_loss.get("valid_weight", 1.0))

    gan_weight   = float(cfg_loss.get("gan_weight", 0.1))
    r1_gamma     = float(cfg_loss.get("r1_gamma", 10.0))
    use_gan = gan_weight > 0.0

    l1_loss = nn.L1Loss(reduction="none")

    perceptual_loss = None
    if perc_weight > 0:
        perceptual_loss = VGGPerceptualLoss().to(device)
        perceptual_loss.eval()
        logger.info("[INFO] Using VGG19 perceptual loss")
    else:
        logger.info("[INFO] No perceptual loss (perceptual_weight=0)")

    total_params = sum(p.numel() for p in G.parameters())
    trainable_params = sum(p.numel() for p in G.parameters() if p.requires_grad)
    logger.info(f"[INFO] G params: total={total_params:,}, trainable={trainable_params:,}")

    num_epochs = cfg_train.get("epochs", 50)
    lr = float(cfg_train.get("lr", 1e-4))
    min_lr = float(cfg_train.get("min_lr", 1e-6))
    warmup_epochs = int(cfg_train.get("warmup_epochs", 3))

    scheduler = lr_scheduler(
        opt_G,
        warmup_epochs=warmup_epochs,
        total_epochs=num_epochs,
        min_lr=min_lr,
        max_lr=lr,
    )

    # ----- Discriminator -----
    if use_gan:
        D = PatchDiscriminator(in_ch=3).to(device)
        opt_D = torch.optim.Adam(D.parameters(), lr=cfg_train.get("lr_D", lr), betas=(0.0, 0.9))
        logger.info("[INFO] Using PatchGAN discriminator")
    else:
        D = None
        opt_D = None
        logger.info("[INFO] GAN disabled (gan_weight=0)")

    # 5. Checkpoint
    save_dir = cfg_ckpt.get("save_dir", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    resume_path = cfg_ckpt.get("resume", "")

    start_epoch = 0
    best_val_loss = float("inf")

    if resume_path and os.path.isfile(resume_path):
        logger.info(f"[INFO] Resuming from checkpoint: {resume_path}")
        G, opt_G, start_epoch = load_checkpoint(
            resume_path, G, opt_G, device=device, load_opt=True
        )
    else:
        logger.info("[INFO] Training from scratch.")

    save_interval = cfg_ckpt.get("save_interval", 5)

    # 6. Training loop
    for epoch in tqdm(range(start_epoch, num_epochs), desc="Epochs"):
        G_loss, D_loss, train_psnr, train_rec = train_one_epoch(
            G=G,
            D=D,
            use_gan=use_gan,
            model_name=model_name,
            loader=train_loader,
            opt_G=opt_G,
            opt_D=opt_D,
            device=device,
            epoch=epoch,
            num_epochs=num_epochs,
            l1_loss=l1_loss,
            rec_weight=rec_weight,
            hole_weight=hole_weight,
            valid_weight=valid_weight,
            perc_weight=perc_weight,
            perceptual_loss=perceptual_loss,
            gan_weight=gan_weight,
            r1_gamma=r1_gamma,
        )

        if val_loader is not None:
            val_loss, val_psnr, val_rec = validate_one_epoch(
                G=G,
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
            val_loss, val_psnr, val_rec = float("nan"), float("nan"), float("nan")

        scheduler.step()

        logger.info(
            f"[Epoch {epoch}/{num_epochs}] "
            f"G_loss={G_loss:.6f} | D_loss={D_loss:.6f} | "
            f"train_rec={train_rec:.6f} | train_psnr={train_psnr:.2f} | "
            f"val_rec={val_rec:.6f} | val_loss={val_loss:.6f} | val_psnr={val_psnr:.2f}"
        )

        is_best = (val_loader is not None) and (val_loss < best_val_loss)
        if is_best:
            best_val_loss = val_loss

        if ((epoch + 1) % save_interval == 0) or is_best:
            save_checkpoint(
                model=G,
                optimizer=opt_G,
                save_dir=save_dir,
                epoch=epoch,
                is_best=is_best,
            )

    logger.info("[INFO] Training finished.")


if __name__ == "__main__":
    main()
