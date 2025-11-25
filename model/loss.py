import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ==========================
#   PSNR
# ==========================

def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    mse = F.mse_loss(pred, target, reduction="mean")
    psnr = 10 * torch.log10(max_val ** 2 / (mse + 1e-8))
    return psnr


# ==========================
#   PERCEPTUAL LOSS (VGG19)
# ==========================

class VGGPerceptualLoss(nn.Module):
    """
    Dùng feature VGG19 pretrained ImageNet để tính perceptual loss.
    """
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features

        self.slice1 = nn.Sequential(*[vgg[i] for i in range(16)])      # relu3_3
        self.slice2 = nn.Sequential(*[vgg[i] for i in range(16, 25)])  # relu4_3

        for p in self.parameters():
            p.requires_grad = False

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
        x = self._norm(x)
        y = self._norm(y)

        fx1 = self.slice1(x)
        fy1 = self.slice1(y)
        fx2 = self.slice2(fx1)
        fy2 = self.slice2(fy1)

        loss1 = torch.mean(torch.abs(fx1 - fy1))
        loss2 = torch.mean(torch.abs(fx2 - fy2))
        return loss1 + loss2


# ==========================
#   INPAINT L1 + PERCEPTUAL
# ==========================

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
    L = rec_weight * (hole_weight*L1_hole + valid_weight*L1_valid) + perc_weight * L_perc
    """
    eps = 1e-6

    l1_map = l1_loss(pred, target)  # [B,3,H,W] (reduction='none')

    if mask.shape[1] == 1 and l1_map.shape[1] != 1:
        hole_mask = mask.expand(-1, l1_map.shape[1], -1, -1)
    else:
        hole_mask = mask
    valid_mask = 1.0 - hole_mask

    hole_loss = (l1_map * hole_mask).sum() / (hole_mask.sum() + eps)
    valid_loss = (l1_map * valid_mask).sum() / (valid_mask.sum() + eps)

    rec_l1 = hole_weight * hole_loss + valid_weight * valid_loss
    rec_l1 = rec_weight * rec_l1

    total_loss = rec_l1

    if perc_weight > 0.0 and perceptual_loss is not None:
        perc = perceptual_loss(pred, target)
        total_loss = total_loss + perc_weight * perc

    return total_loss


# ==========================
#   GAN LOSSES (StyleGAN-like)
# ==========================

def d_logistic_loss(real_pred, fake_pred):
    return F.softplus(-real_pred).mean() + F.softplus(fake_pred).mean()


def g_nonsat_loss(fake_pred):
    return F.softplus(-fake_pred).mean()


def r1_penalty(real_pred, real_img):
    grad_real = torch.autograd.grad(
        outputs=real_pred.sum(),
        inputs=real_img,
        create_graph=True,
    )[0]
    grad_pen = grad_real.pow(2).reshape(grad_real.size(0), -1).sum(1).mean()
    return grad_pen
