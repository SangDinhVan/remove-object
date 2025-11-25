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
    Dùng cho bridge / decoder.
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
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ==========================
#   STYLE MANIPULATION LITE
# ==========================

class StyleManipulationModule(nn.Module):
    """
    SMM-lite: z -> (gamma, beta) để FiLM token features.
    MLP nhỏ để giảm số tham số:
        style_dim -> hidden -> 2*d_model
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
        out = self.fc(z)  # [B, 2*d_model]
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

        # 2 SMM-lite: một cho trước attention, một cho trước FFN
        self.smm1 = StyleManipulationModule(style_dim, d_model, hidden_dim=smm_hidden)
        self.smm2 = StyleManipulationModule(style_dim, d_model, hidden_dim=smm_hidden)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x_seq, z, attn_mask=None, key_padding_mask=None):
        """
        x_seq: [B, N, C]
        z: [B, style_dim]
        """
        # ---- Block 1: LN -> Style FiLM -> Self-Attention ----
        h = self.ln1(x_seq)                      # [B,N,C]

        gamma1, beta1 = self.smm1(z)             # [B,C]
        gamma1 = gamma1.unsqueeze(1)             # [B,1,C]
        beta1  = beta1.unsqueeze(1)              # [B,1,C]
        h = h * (1.0 + gamma1) + beta1           # FiLM

        attn_out, _ = self.self_attn(
            h, h, h,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )                                         # [B,N,C]
        x_seq = x_seq + self.dropout(attn_out)   # residual

        # ---- Block 2: LN -> Style FiLM -> FFN ----
        h2 = self.ln2(x_seq)

        gamma2, beta2 = self.smm2(z)             # [B,C]
        gamma2 = gamma2.unsqueeze(1)
        beta2  = beta2.unsqueeze(1)
        h2 = h2 * (1.0 + gamma2) + beta2

        ffn_out = self.ffn(h2)
        x_seq = x_seq + self.dropout(ffn_out)

        return x_seq


# ==========================
#   EFFICIENTNET / RESNET + MAT-LITE
# ==========================

class EfficientNetMAT(nn.Module):
    """
    MAT-lite cho inpainting / remove object:

    - Encoder: EfficientNet/ResNet từ timm (features_only)
    - Bottleneck: down-project C5 -> bottleneck_dim,
                  + mask embedding, + stack StyleTransformerBlock
    - Decoder: U-Net style (skip f1..f4)
    - Input:
        x    = concat(masked_image, mask)  -> [B, 4, H, W]
        mask = [B, 1, H, W]
        z    = [B, style_dim] (noise / style vector)
    - Output:
        [B, 3, H, W] trong [0,1]
    """

    def __init__(
        self,
        encoder_name: str = "efficientnet_b0",
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

        # 4 kênh (RGB + mask) -> 3 kênh cho encoder
        self.input_proj = nn.Conv2d(in_channels, 3, kernel_size=1)

        # ----- Encoder từ timm -----
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
        )
        enc_channels = self.encoder.feature_info.channels()  # [c1..c5]
        c1, c2, c3, c4, c5 = enc_channels

        d_model = self.bottleneck_dim

        # Bottleneck: giảm C5 -> d_model, rồi tăng d_model -> C5
        self.bottleneck_down = nn.Conv2d(c5, d_model, kernel_size=1)
        self.bottleneck_up   = nn.Conv2d(d_model, c5, kernel_size=1)

        # Mask embedding: 1 -> d_model
        self.mask_embed = nn.Conv2d(1, d_model, kernel_size=1)

        # Stack các StyleTransformerBlock
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

        # Bridge conv sau transformer (C5 -> C5)
        self.bridge = ConvBlock(c5, c5)

        # Decoder U-Net
        self.up4 = UpBlock(c5, c4, 256)
        self.up3 = UpBlock(256, c3, 128)
        self.up2 = UpBlock(128, c2, 64)
        self.up1 = UpBlock(64,  c1, 32)

        self.out_conv = nn.Conv2d(32, 3, kernel_size=1)

        # ============= INIT =============
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

        self._init_weights_special()

    # ---------- factory từ config dict ----------
    @classmethod
    def from_config(cls, cfg: dict):
        """
        cfg:
          encoder_name, pretrained, in_channels
          mat:
            style_dim, n_heads, d_ff, num_layers, dropout, bottleneck_dim, smm_hidden
        """
        mat_cfg = cfg.get("mat", {})
        return cls(
            encoder_name=cfg.get("encoder_name", "efficientnet_b0"),
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

    # ---------- init backbone (Conv/BN/Linear) ----------
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

    def _init_weights_special(self):
        # chừa sẵn nếu sau này bạn add LSTM/attn/FC đặc biệt
        pass

    # ---------- forward ----------
    def forward(self, x: torch.Tensor, mask: torch.Tensor, z: torch.Tensor):
        """
        x   : [B, 4, H, W]  (masked_image concat mask)
        mask: [B, 1, H, W]  (mask gốc, 1 = vùng cần inpaint)
        z   : [B, style_dim] (noise / style)

        return: [B, 3, H, W] trong [0,1]
        """
        B, _, H, W = x.shape

        # 4 -> 3 kênh cho encoder
        x_in = self.input_proj(x)  # [B,3,H,W]

        # encoder features
        feats = self.encoder(x_in)
        f1, f2, f3, f4, f5 = feats        # f5: [B,C5,Hb,Wb]
        B, C5, Hb, Wb = f5.shape

        # ----- mask embedding -----
        mask_ds = F.interpolate(mask, size=(Hb, Wb), mode="nearest")  # [B,1,Hb,Wb]
        mask_emb = self.mask_embed(mask_ds)                           # [B,d_model,Hb,Wb]

        # ----- đưa f5 về d_model -----
        f5_down = self.bottleneck_down(f5)                            # [B,d_model,Hb,Wb]

        # combine feature + mask info trong không gian d_model
        feat = f5_down + mask_emb                                     # [B,d_model,Hb,Wb]

        # flatten -> [B, N, d_model]
        x_seq = feat.flatten(2).transpose(1, 2)                       # [B,N,d_model]

        # (optional) có thể thêm key_padding_mask từ mask_ds nếu muốn
        key_padding_mask = None
        attn_mask = None

        # ----- transformer bottleneck -----
        for blk in self.transformer_blocks:
            x_seq = blk(x_seq, z, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        # reshape lại về feature map d_model
        x_trans_bottleneck = x_seq.transpose(1, 2).view(
            B, self.bottleneck_dim, Hb, Wb
        )  # [B,d_model,Hb,Wb]

        # đưa ngược về C5 để feed vào bridge + decoder
        x_trans = self.bottleneck_up(x_trans_bottleneck)              # [B,C5,Hb,Wb]

        # bridge conv
        x_bridge = self.bridge(x_trans)                               # [B,C5,Hb,Wb]

        # decoder UNet
        x = self.up4(x_bridge, f4)
        x = self.up3(x, f3)
        x = self.up2(x, f2)
        x = self.up1(x, f1)

        # resize về H,W gốc nếu lệch
        if x.shape[-2:] != (H, W):
            x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)

        out = self.out_conv(x)
        out = torch.sigmoid(out)
        return out


# ==========================
#   QUICK SELF-TEST
# ==========================

if __name__ == "__main__":
    B, H, W = 2, 512, 512
    model = EfficientNetMAT(
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

    # ==== IN RA SỐ PARAM ====
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total params:", f"{total:,}")
    print("Trainable params:", f"{trainable:,}")
    print()

    # ==== TEST FORWARD ====
    masked_plus_mask = torch.randn(B, 4, H, W)
    mask = torch.randint(0, 2, (B, 1, H, W)).float()
    z = torch.randn(B, 256)

    y = model(masked_plus_mask, mask, z)
    print("Output shape:", y.shape)  # expect [B, 3, 512, 512]
