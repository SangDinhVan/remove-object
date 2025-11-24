import os
import torch

def save_checkpoint(model, optimizer, save_dir, epoch, is_best=False):
    os.makedirs(save_dir, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict()
    }

    path = os.path.join(save_dir, f"epoch_{epoch}.pth")
    torch.save(checkpoint, path)
    print(f"[INFO] Saved checkpoint: {path}")

    if is_best:
        best_path = os.path.join(save_dir, "best_model.pth")
        torch.save(checkpoint, best_path)
        print(f"[INFO] Best model saved: {best_path}")


def load_checkpoint(ckpt_path, model, optimizer=None, device="cuda", load_opt=True):
    checkpoint = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    if optimizer is not None and load_opt and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        print("[INFO] Optimizer state loaded.")

    epoch = checkpoint.get("epoch", 0) + 1
    print(f"[INFO] Loaded checkpoint from {ckpt_path} (epoch {epoch})")

    return model, optimizer, epoch
