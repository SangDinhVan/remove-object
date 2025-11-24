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
    Dùng lại cho bridge / decoder.
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
#   STYLE MANIPULATION (SMM)
# ==========================

class StyleManipulationModule(nn.Module):
    """
    SMM: lấy style vector z -> sinh gamma, beta để FiLM (scale + shift)
    vào token features của Transformer.

    z: [B, style_dim]
    gamma, beta: [B, d_model]
    """
    def __init__(self, style_dim, d_model):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(style_dim, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model * 2),
        )

    def forward(self, z):
        """
        return:
            gamma, beta: [B, d_model]
        """
        out = self.fc(z)  # [B, 2*d_model]
        gamma, beta = out.chunk(2, dim=-1)
        return gamma, beta


class StyleTransformerBlock(nn.Module):
    """
    Transformer block đã được 'điều chỉnh' bởi style (SMM):

    - LayerNorm
    - FiLM: x = x * (1 + gamma) + beta (theo style)
    - Multi-head Self-Attention
    - Residual
    - FFN (2-layer MLP với GELU) + style modulation lần nữa

    x_seq: [B, N, C]
    z: [B, style_dim]
    """
    def __init__(self, d_model, n_heads, d_ff, style_dim, dropout=0.0):
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

        # 2 SMM: một cho trước attention, một cho trước FFN
        self.smm1 = StyleManipulationModule(style_dim, d_model)
        self.smm2 = StyleManipulationModule(style_dim, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x_seq, z, attn_mask=None, key_padding_mask=None):
        """
        x_seq: [B, N, C]
        z: [B, style_dim]
        """
        B, N, C = x_seq.shape

        # ---- Block 1: LN -> Style FiLM -> Self-Attention ----
        h = self.ln1(x_seq)

        gamma1, beta1 = self.smm1(z)       # [B, C]
        gamma1 = gamma1.unsqueeze(1)       # [B,1,C]
        beta1  = beta1.unsqueeze(1)        # [B,1,C]

        h = h * (1.0 + gamma1) + beta1     # style modulation

        attn_out, _ = self.self_attn(
            h, h, h,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x_seq = x_seq + self.dropout(attn_out)

        # ---- Block 2: LN -> Style FiLM -> FFN ----
        h2 = self.ln2(x_seq)

        gamma2, beta2 = self.smm2(z)       # [B, C]
        gamma2 = gamma2.unsqueeze(1)
        beta2  = beta2.unsqueeze(1)

        h2 = h2 * (1.0 + gamma2) + beta2

        ffn_out = self.ffn(h2)
        x_seq = x_seq + self.dropout(ffn_out)

        return x_seq


# ==========================
#   EFFICIENTNET + MAT
# ==========================

class EfficientNetMAT(nn.Module):
    """
    Inpainting model dạng MAT:

    - Encoder: EfficientNet (pretrained)
    - Bottleneck: stack StyleTransformerBlock (SMM + self-attn + FFN)
      kết hợp feature f5 + mask embedding + style z
    - Decoder: U-Net style
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
    ):
        super().__init__()

        self.pretrained = pretrained
        self.style_dim = style_dim

        # 4 kênh (RGB + mask) -> 3 kênh cho EfficientNet
        self.input_proj = nn.Conv2d(in_channels, 3, kernel_size=1)

        # Encoder EfficientNet
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
        )
        enc_channels = self.encoder.feature_info.channels()
        c1, c2, c3, c4, c5 = enc_channels

        d_model = c5  # dùng C của feature f5 làm dim transformer

        # Mask embedding: downsample mask -> Hb x Wb rồi map -> C
        self.mask_embed = nn.Conv2d(1, d_model, kernel_size=1)

        # Stack các StyleTransformerBlock (T1..Tn trong hình)
        self.transformer_blocks = nn.ModuleList([
            StyleTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                style_dim=style_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Bridge conv sau transformer
        self.bridge = ConvBlock(c5, c5)

        # Decoder (UNet)
        self.up4 = UpBlock(c5, c4, 256)
        self.up3 = UpBlock(256, c3, 128)
        self.up2 = UpBlock(128, c2, 64)
        self.up1 = UpBlock(64,  c1, 32)

        self.out_conv = nn.Conv2d(32, 3, kernel_size=1)

        # ============= INIT =============

        # init module mình thêm
        self._init_backbone(self.input_proj)
        self._init_backbone(self.mask_embed)
        self._init_backbone(self.bridge)
        self._init_backbone(self.up4)
        self._init_backbone(self.up3)
        self._init_backbone(self.up2)
        self._init_backbone(self.up1)
        self._init_backbone(self.out_conv)

        for blk in self.transformer_blocks:
            self._init_backbone(blk)

        # nếu KHÔNG dùng pretrained thì mới init encoder
        if not self.pretrained:
            self._init_backbone(self.encoder)

        # init cho các module đặc biệt (LSTM/attn/fc nếu có)
        self._init_weights()

    # ---------- factory từ config dict ----------
    @classmethod
    def from_config(cls, cfg: dict):
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

    # ---------- init cho LSTM / attn / fc (dạng generic) ----------
    def _init_weights(self):
        # LSTM (nếu bạn add thêm)
        if hasattr(self, "lstm"):
            for name, param in self.lstm.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.constant_(param, 0)

        # Attention linear (nếu có self.attn)
        if hasattr(self, "attn"):
            if hasattr(self.attn, "weight") and self.attn.weight is not None:
                nn.init.xavier_uniform_(self.attn.weight)
            if hasattr(self.attn, "bias") and self.attn.bias is not None:
                nn.init.constant_(self.attn.bias, 0)

        # Fully-connected head (nếu có self.fc)
        if hasattr(self, "fc"):
            for m in self.fc.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    # ---------- forward ----------
    def forward(self, x: torch.Tensor, mask: torch.Tensor, z: torch.Tensor):
        """
        x   : [B, 4, H, W]  (masked_image concat mask)
        mask: [B, 1, H, W]  (mask gốc)
        z   : [B, style_dim] (noise / style)

        return: [B, 3, H, W]
        """
        B, _, H, W = x.shape

        # 4 -> 3 kênh cho encoder
        x_in = self.input_proj(x)  # [B,3,H,W]

        # encoder features
        feats = self.encoder(x_in)
        f1, f2, f3, f4, f5 = feats    # f5: [B,C,Hb,Wb]
        B, C, Hb, Wb = f5.shape

        # ----- mask embedding -----
        mask_ds = F.interpolate(mask, size=(Hb, Wb), mode="nearest")  # [B,1,Hb,Wb]
        mask_emb = self.mask_embed(mask_ds)                           # [B,C,Hb,Wb]

        # combine feature + mask info
        feat = f5 + mask_emb                                         # [B,C,Hb,Wb]

        # flatten -> [B, N, C]
        x_seq = feat.flatten(2).transpose(1, 2)                      # [B, N, C]

        # (optional) key_padding_mask từ mask nếu muốn:
        # hiện tại bỏ qua cho đơn giản; bạn có thể thêm theo paper.

        # ----- stack style-aware transformer blocks -----
        for blk in self.transformer_blocks:
            x_seq = blk(x_seq, z)  # [B, N, C]

        # reshape lại về feature map
        x_trans = x_seq.transpose(1, 2).view(B, C, Hb, Wb)

        # bridge conv
        x_bridge = self.bridge(x_trans)

        # decoder UNet
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
#   QUICK SELF-TEST
# ==========================

if __name__ == "__main__":
    B, H, W = 2, 512, 512
    model = EfficientNetMAT(
        encoder_name="efficientnet_b0",
        pretrained=True,
        in_channels=4,
        style_dim=256,
        n_heads=8,
        d_ff=2048,
        num_layers=4,
    )
    masked_plus_mask = torch.randn(B, 4, H, W)
    mask = torch.randint(0, 2, (B, 1, H, W)).float()
    z = torch.randn(B, 256)

    y = model(masked_plus_mask, mask, z)
    print("Output shape:", y.shape)  # expect [B, 3, 512, 512]
