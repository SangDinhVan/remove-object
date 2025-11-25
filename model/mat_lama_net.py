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
    (vẫn dùng cho bridge)
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


class DilatedResBlock(nn.Module):
    """
    LaMa-lite block: dùng nhiều conv giãn (dilated) để mở receptive field.
    """
    def __init__(self, ch, dilations=(1, 2, 4, 8)):
        super().__init__()
        layers = []
        for d in dilations:
            layers += [
                nn.Conv2d(ch, ch, kernel_size=3, padding=d, dilation=d),
                nn.GELU(),
            ]
        self.net = nn.Sequential(*layers)
        self.skip = nn.Conv2d(ch, ch, kernel_size=1)

    def forward(self, x):
        return self.skip(x) + self.net(x)


class UpBlockDilated(nn.Module):
    """
    Upsample 2x + concat skip + conv1x1 + DilatedResBlock (LaMa-style decoder).
    """
    def __init__(self, in_ch, skip_ch, out_ch, dilations=(1, 2, 4, 8)):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv1x1 = nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=1)
        self.dilated_block = DilatedResBlock(out_ch, dilations=dilations)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1x1(x)
        x = self.dilated_block(x)
        return x


# ==========================
#   STYLE MANIPULATION LITE
# ==========================

class StyleManipulationModule(nn.Module):
    """
    SMM-lite: z -> (gamma, beta) để FiLM token features.
    MLP nhỏ: style_dim -> hidden -> 2*d_model
    """
    def __init__(self, style_dim, d_model, hidden_dim=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(style_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * d_model),
        )

    def forward(self, z):
        """
        z: [B, style_dim]
        return:
            gamma, beta: [B, d_model]
        """
        out = self.fc(z)
        gamma, beta = out.chunk(2, dim=-1)
        return gamma, beta


class StyleTransformerBlock(nn.Module):
    """
    Transformer block + style FiLM (MAT-lite):

    - LN
    - SMM-lite: x = x * (1 + gamma) + beta
    - Multi-head Self-Attention + residual
    - LN
    - SMM-lite
    - FFN + residual

    x_seq: [B, N, C]
    z: [B, style_dim]
    """
    def __init__(self, d_model, n_heads, d_ff, style_dim, dropout=0.0, smm_hidden=256):
        super().__init__()
        self.d_model = d_model

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

        self.smm1 = StyleManipulationModule(style_dim, d_model, hidden_dim=smm_hidden)
        self.smm2 = StyleManipulationModule(style_dim, d_model, hidden_dim=smm_hidden)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x_seq, z, attn_mask=None, key_padding_mask=None):
        # Block 1: LN -> Style FiLM -> Self-Attention
        h = self.ln1(x_seq)

        gamma1, beta1 = self.smm1(z)         # [B,C]
        gamma1 = gamma1.unsqueeze(1)         # [B,1,C]
        beta1  = beta1.unsqueeze(1)          # [B,1,C]
        h = h * (1.0 + gamma1) + beta1

        attn_out, _ = self.self_attn(
            h, h, h,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x_seq = x_seq + self.dropout(attn_out)

        # Block 2: LN -> Style FiLM -> FFN
        h2 = self.ln2(x_seq)
        gamma2, beta2 = self.smm2(z)
        gamma2 = gamma2.unsqueeze(1)
        beta2  = beta2.unsqueeze(1)
        h2 = h2 * (1.0 + gamma2) + beta2

        ffn_out = self.ffn(h2)
        x_seq = x_seq + self.dropout(ffn_out)

        return x_seq


# ==========================
#   GENERATOR: MAT + LaMa-lite
# ==========================

class MatLaMaNet(nn.Module):
    """
    MAT-lite + LaMa-lite decoder cho inpainting / remove object:

    - Encoder: ResNet/EfficientNet từ timm (features_only)
    - Bottleneck: C5 -> bottleneck_dim, mask embedding, StyleTransformerBlock stack
    - Decoder: U-Net style + dilated conv (LaMa-lite)
    - Input:
        x    = concat(masked_image, mask)  -> [B, 4, H, W]
        mask = [B, 1, H, W]
        z    = [B, style_dim]
    - Output:
        [B, 3, H, W] trong [0,1]
    """
    def __init__(
        self,
        encoder_name: str = "resnet50",
        pretrained: bool = True,
        in_channels: int = 4,
        style_dim: int = 256,
        n_heads: int = 8,
        d_ff: int = 2048,
        num_layers: int = 4,
        dropout: float = 0.0,
        bottleneck_dim: int = 512,
        smm_hidden: int = 256,
    ):
        super().__init__()
        self.pretrained = pretrained
        self.style_dim = style_dim
        self.bottleneck_dim = bottleneck_dim

        # 4 kênh (RGB + mask) -> 3 kênh
        self.input_proj = nn.Conv2d(in_channels, 3, kernel_size=1)

        # Encoder
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
        )
        enc_channels = self.encoder.feature_info.channels()
        c1, c2, c3, c4, c5 = enc_channels

        d_model = bottleneck_dim
        self.bottleneck_down = nn.Conv2d(c5, d_model, kernel_size=1)
        self.bottleneck_up   = nn.Conv2d(d_model, c5, kernel_size=1)

        self.mask_embed = nn.Conv2d(1, d_model, kernel_size=1)

        self.transformer_blocks = nn.ModuleList([
            StyleTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                style_dim=style_dim,
                dropout=dropout,
                smm_hidden=smm_hidden,
            )
            for _ in range(num_layers)
        ])

        # Bridge conv
        self.bridge = ConvBlock(c5, c5)

        # Decoder LaMa-lite
        self.up4 = UpBlockDilated(c5,  c4, 256, dilations=(1, 2, 4, 8))
        self.up3 = UpBlockDilated(256, c3, 128, dilations=(1, 2, 4, 8))
        self.up2 = UpBlockDilated(128, c2, 64,  dilations=(1, 2, 4, 8))
        self.up1 = UpBlockDilated(64,  c1, 32,  dilations=(1, 2, 4, 8))

        self.out_conv = nn.Conv2d(32, 3, kernel_size=1)

        # Init
        self._init_backbone(self.input_proj)
        self._init_backbone(self.mask_embed)
        self._init_backbone(self.bottleneck_down)
        self._init_backbone(self.bottleneck_up)
        self._init_backbone(self.bridge)
        self._init_backbone(self.up4)
        self._init_backbone(self.up3)
        self._init_backbone(self.up2)
        self._init_backbone(self.up1)
        self._init_backbone(self.out_conv)

        for blk in self.transformer_blocks:
            self._init_backbone(blk)

        if not self.pretrained:
            self._init_backbone(self.encoder)

    # ---------- from_config ----------
    @classmethod
    def from_config(cls, cfg: dict):
        mat_cfg = cfg.get("mat", {})
        return cls(
            encoder_name=cfg.get("encoder_name", "resnet50"),
            pretrained=cfg.get("pretrained", True),
            in_channels=cfg.get("in_channels", 4),
            style_dim=mat_cfg.get("style_dim", 256),
            n_heads=mat_cfg.get("n_heads", 8),
            d_ff=mat_cfg.get("d_ff", 2048),
            num_layers=mat_cfg.get("num_layers", 4),
            dropout=mat_cfg.get("dropout", 0.0),
            bottleneck_dim=mat_cfg.get("bottleneck_dim", 512),
            smm_hidden=mat_cfg.get("smm_hidden", 256),
        )

    # ---------- init backbone ----------
    def _init_backbone(self, module: nn.Module):
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

    # ---------- forward ----------
    def forward(self, x: torch.Tensor, mask: torch.Tensor, z: torch.Tensor):
        """
        x   : [B, 4, H, W]  (masked_image concat mask)
        mask: [B, 1, H, W]  (mask gốc, 1 = vùng cần inpaint)
        z   : [B, style_dim]

        return: [B, 3, H, W] trong [0,1]
        """
        B, _, H, W = x.shape

        x_in = self.input_proj(x)           # [B,3,H,W]
        feats = self.encoder(x_in)
        f1, f2, f3, f4, f5 = feats         # f5: [B,C5,Hb,Wb]
        B, C5, Hb, Wb = f5.shape

        # mask embedding
        mask_ds = F.interpolate(mask, size=(Hb, Wb), mode="nearest")
        mask_emb = self.mask_embed(mask_ds)              # [B,d_model,Hb,Wb]

        # bottleneck
        f5_down = self.bottleneck_down(f5)               # [B,d_model,Hb,Wb]
        feat = f5_down + mask_emb                        # [B,d_model,Hb,Wb]

        # flatten -> [B,N,d_model]
        x_seq = feat.flatten(2).transpose(1, 2)          # [B,N,d_model]

        # transformer
        for blk in self.transformer_blocks:
            x_seq = blk(x_seq, z)

        x_bottleneck = x_seq.transpose(1, 2).view(B, self.bottleneck_dim, Hb, Wb)
        x_trans = self.bottleneck_up(x_bottleneck)       # [B,C5,Hb,Wb]

        # bridge + decoder
        x_bridge = self.bridge(x_trans)

        x = self.up4(x_bridge, f4)
        x = self.up3(x, f3)
        x = self.up2(x, f2)
        x = self.up1(x, f1)

        if x.shape[-2:] != (H, W):
            x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)

        out = self.out_conv(x)
        out = torch.sigmoid(out)
        return out


# ==========================
#   PATCHGAN DISCRIMINATOR
# ==========================

class PatchDiscriminator(nn.Module):
    """
    PatchGAN đơn giản cho inpainting.
    Input:  [B,3,H,W]
    Output: [B,1,h',w'] (logits)
    """
    def __init__(self, in_ch=3, base_ch=64, n_layers=4):
        super().__init__()
        layers = []
        ch = base_ch
        layers += [
            nn.Conv2d(in_ch, ch, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        for i in range(1, n_layers):
            prev_ch = ch
            ch = min(512, ch * 2)
            layers += [
                nn.Conv2d(prev_ch, ch, 4, 2, 1),
                nn.BatchNorm2d(ch),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        layers += [nn.Conv2d(ch, 1, 3, 1, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ==========================
#   QUICK SELF-TEST
# ==========================

if __name__ == "__main__":
    B, H, W = 2, 512, 512
    model = MatLaMaNet(
        encoder_name="resnet50",
        pretrained=True,
        in_channels=4,
        style_dim=256,
        n_heads=8,
        d_ff=2048,
        num_layers=4,
        bottleneck_dim=512,
        smm_hidden=256,
    )

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total params:", f"{total:,}")
    print("Trainable params:", f"{trainable:,}")

    masked_plus_mask = torch.randn(B, 4, H, W)
    mask = torch.randint(0, 2, (B, 1, H, W)).float()
    z = torch.randn(B, 256)

    y = model(masked_plus_mask, mask, z)
    print("Output shape:", y.shape)
