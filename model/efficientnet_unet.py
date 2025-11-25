import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ==========================
#   BASIC BUILDING BLOCKS
# ==========================

class ConvBlock(nn.Module):
    """
    2 x (Conv2d + BN + GELU)
    Dùng cho cả bridge và decoder.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.conv(x)


class UpBlock(nn.Module):
    """
    Upsample 2x + concat skip + ConvBlock
    """
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # căn chỉnh spatial size (nếu lệch do chia / pooling)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ==========================
#   EFFICIENTNET U-NET
# ==========================

class EfficientNetUNet(nn.Module):
    """
    Inpainting model:
    - Encoder: EfficientNet (pretrained từ timm)
    - Decoder: U-Net style (UpBlock)
    - Input:  (masked_image concat mask) -> [B, 4, H, W]
    - Output: ảnh inpainted [B, 3, H, W] trong [0,1]
    """
    def __init__(
        self,
        encoder_name: str = "efficientnet_b0",
        pretrained: bool = True,
        in_channels: int = 4,
    ):
        super().__init__()

        self.pretrained = pretrained

        # 4 kênh (RGB + mask) -> 3 kênh cho EfficientNet pretrained
        self.input_proj = nn.Conv2d(in_channels, 3, kernel_size=1)

        # Encoder EfficientNet từ timm
        # features_only=True trả về list feature maps theo out_indices
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
        )

        enc_channels = self.encoder.feature_info.channels()  # vd [16, 24, 40, 112, 1280]
        c1, c2, c3, c4, c5 = enc_channels

        # Bottleneck
        self.bridge = ConvBlock(c5, c5)

        # Decoder (UNet)
        # self.up4 = UpBlock(c5, c4, 256)
        # self.up3 = UpBlock(256, c3, 128)
        # self.up2 = UpBlock(128, c2, 64)
        # self.up1 = UpBlock(64,  c1, 32)

        # # Head ra RGB
        # self.out_conv = nn.Conv2d(32, 3, kernel_size=1)
        self.up4 = UpBlock(c5, c4, 512)
        self.up3 = UpBlock(512, c3, 256)
        self.up2 = UpBlock(256, c2, 128)
        self.up1 = UpBlock(128, c1, 64)
        self.out_conv = nn.Conv2d(64, 3, 1)
        # ============= INIT =============

        # init các module mình tự thêm (không đụng encoder pretrained)
        self._init_backbone(self.input_proj)
        self._init_backbone(self.bridge)
        self._init_backbone(self.up4)
        self._init_backbone(self.up3)
        self._init_backbone(self.up2)
        self._init_backbone(self.up1)
        self._init_backbone(self.out_conv)

        # nếu KHÔNG dùng pretrained thì mới init lại encoder
        if not self.pretrained:
            self._init_backbone(self.encoder)

        # init LSTM/attn/fc nếu về sau có thêm (hiện tại không làm gì)
        self._init_weights()

    # ---------- factory từ config dict (yaml) ----------
    @classmethod
    def from_config(cls, cfg: dict):
        """
        cfg:
          encoder_name: str
          pretrained: bool
          in_channels: int
        """
        return cls(
            encoder_name=cfg.get("encoder_name", "efficientnet_b0"),
            pretrained=cfg.get("pretrained", True),
            in_channels=cfg.get("in_channels", 4),
        )

    # ---------- init backbone (conv/bn/linear) ----------
    def _init_backbone(self, module: nn.Module):
        """
        Khởi tạo Conv/BN/Linear cho các module truyền vào.
        Không gọi trên toàn bộ self nếu đang dùng pretrained.
        """
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ---------- init cho LSTM / attn / fc (cho model sau này) ----------
    def _init_weights(self):
        """
        Dành cho các module đặc biệt như:
        - self.lstm
        - self.attn
        - self.fc

        Ở UNet hiện tại chưa dùng, nên hàm này gần như no-op.
        Làm kiểu hasattr để không gây lỗi nếu không tồn tại.
        """
        # LSTM (nếu có)
        if hasattr(self, "lstm"):
            for name, param in self.lstm.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.constant_(param, 0)

        # Attention linear (nếu có)
        if hasattr(self, "attn"):
            if hasattr(self.attn, "weight") and self.attn.weight is not None:
                nn.init.xavier_uniform_(self.attn.weight)
            if hasattr(self.attn, "bias") and self.attn.bias is not None:
                nn.init.constant_(self.attn.bias, 0)

        # Fully-connected head (nếu có)
        if hasattr(self, "fc"):
            for m in self.fc.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    # ---------- forward ----------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 4, H, W]  (masked_image concat mask)
            - masked_image = image * (1 - mask)
            - mask: [B,1,H,W]

        return: [B, 3, H, W]  (ảnh inpainted, range [0,1])
        """
        B, C_in, H, W = x.shape
        x = self.input_proj(x)  # [B,3,H,W]

        # multi-scale features từ encoder
        feats = self.encoder(x)
        f1, f2, f3, f4, f5 = feats  # từ high-res -> low-res

        # bottleneck
        x = self.bridge(f5)

        # decoder với skip connections
        x = self.up4(x, f4)
        x = self.up3(x, f3)
        x = self.up2(x, f2)
        x = self.up1(x, f1)
        if x.shape[-2:] != (H, W):
            x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        # head cuối
        out = self.out_conv(x)
        out = torch.sigmoid(out)  # vì target là ảnh [0,1]
        return out


# ==========================
#   QUICK SELF-TEST
# ==========================

if __name__ == "__main__":
    model = EfficientNetUNet(encoder_name="efficientnet_b0", pretrained=True, in_channels=4)
    x = torch.randn(2, 4, 512, 512)  # batch 2, masked_image+mask
    y = model(x)
    print("Output shape:", y.shape)  # expect: [2, 3, 512, 512]


    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total params:", f"{total:,}")
    print("Trainable params:", f"{trainable:,}")
    print()
