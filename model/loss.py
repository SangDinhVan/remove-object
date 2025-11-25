import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ======================
#  Patch Discriminator
# ======================
class PatchDiscriminator(nn.Module):
    """
    PatchGAN discriminator, nhận ảnh [B,3,H,W], trả logits [B,1,H',W'].
    Không dùng sigmoid, vì loss sẽ dùng BCE-with-logits (softplus) theo non-saturating GAN.
    """
    def __init__(self, in_channels: int = 3, base_ch: int = 64):
        super().__init__()

        def conv_block(in_ch, out_ch, norm=True):
            layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1)]
            if norm:
                layers.append(nn.InstanceNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.net = nn.Sequential(
            conv_block(in_channels,    base_ch, norm=False),
            conv_block(base_ch,        base_ch * 2),
            conv_block(base_ch * 2,    base_ch * 4),
            conv_block(base_ch * 4,    base_ch * 8),
            nn.Conv2d(base_ch * 8, 1, 4, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # logits


# =========================
#   VGG19 Perceptual Loss
# =========================
class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss theo paper MAT:
    L_P = sum_i eta_i || phi_i(x_hat) - phi_i(x) ||_1,
    trong đó phi_i là feature VGG19 ở conv4_4 và conv5_4, với
    eta = [1/4, 1/2].
    """
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features

        self.vgg = vgg
        # conv4_4 = idx 25, conv5_4 = idx 34
        self.layer_ids = [25, 34]
        self.etas = [1.0 / 4.0, 1.0 / 2.0]

        for p in self.vgg.parameters():
            p.requires_grad = False

        # Chuẩn hoá theo ImageNet
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,3,H,W] trong [0,1]
        return (x - self.mean) / self.std

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        x_hat, x: [B,3,H,W] trong [0,1].
        """
        x_hat = self._preprocess(x_hat)
        x = self._preprocess(x)

        feats_hat = []
        feats = []
        h_hat, h = x_hat, x
        for i, layer in enumerate(self.vgg):
            h_hat = layer(h_hat)
            h = layer(h)
            if i in self.layer_ids:
                feats_hat.append(h_hat)
                feats.append(h)

        loss = 0.0
        for f_hat, f, eta in zip(feats_hat, feats, self.etas):
            loss = loss + eta * torch.mean(torch.abs(f_hat - f))
        return loss


# ======================
#   GAN Loss Helpers
# ======================
def d_logistic_loss(real_pred: torch.Tensor, fake_pred: torch.Tensor) -> torch.Tensor:
    """
    Non-saturating logistic loss cho Discriminator:
      L_D = E[softplus(-D(x))] + E[softplus(D(x_hat))]
    """
    loss_real = F.softplus(-real_pred).mean()
    loss_fake = F.softplus(fake_pred).mean()
    return loss_real + loss_fake


def g_nonsat_loss(fake_pred: torch.Tensor) -> torch.Tensor:
    """
    Non-saturating loss cho Generator:
      L_G = E[softplus(-D(x_hat))]
    """
    return F.softplus(-fake_pred).mean()


def r1_penalty(real_pred: torch.Tensor, real_img: torch.Tensor) -> torch.Tensor:
    """
    R1 regularization: E[ || grad_x D(x) ||^2 ].
    real_pred: logits D(real_img) [B,1,H',W']
    real_img:  [B,3,H,W] with requires_grad=True
    """
    grad_real = torch.autograd.grad(
        outputs=real_pred.sum(),
        inputs=real_img,
        create_graph=True,
    )[0]
    grad_penalty = grad_real.pow(2).reshape(grad_real.size(0), -1).sum(1).mean()
    return grad_penalty
