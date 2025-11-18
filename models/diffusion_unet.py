# models/diffusion_unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------
# Basic U-Net Blocks
# -----------------------------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # pad if shape mismatch
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(
            x1,
            [diffX // 2, diffX - diffX // 2,
             diffY // 2, diffY - diffY // 2]
        )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.conv(x)


# -----------------------------------------------------------
# Conditional Diffusion U-Net (ATLAS-FLOW-DIFF)
# -----------------------------------------------------------

class ConditionalUNet(nn.Module):
    """
    εθ(D_t | t, T, F, E)

    Inputs:
      x      : noised DT map (B, 1, H, W)
      t      : timestep (B, 1)
      tokens : topology tokens (B, cond_dim)
      extra_cond : fusion of feature map + edge map (B, feat_dim+3, h, w)
    """
    def __init__(self, cond_dim=243, in_ch=1, out_ch=1, feat_dim=512):
        super().__init__()

        # -------------------------------------------------------
        # 1. Timestep & Token Conditioning
        # -------------------------------------------------------
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

        self.token_mlp = nn.Sequential(
            nn.Linear(cond_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

        # Multiscale FiLM style projections
        self.proj_64  = nn.Linear(128, 64)
        self.proj_128 = nn.Linear(128, 128)
        self.proj_256 = nn.Linear(128, 256)
        self.proj_512 = nn.Linear(128, 512)

        # -------------------------------------------------------
        # 2. Conditioning on backbone feature F + edge bank E
        #    Expected channel=512 (C5) + 3 edge maps
        # -------------------------------------------------------
        self.extra_proj = nn.Conv2d(feat_dim + 3, 64, kernel_size=1)

        # -------------------------------------------------------
        # 3. Core U-Net Encoder
        # -------------------------------------------------------
        self.inc = DoubleConv(in_ch + 64, 64)   # Inject cond feat into input

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        self.bot = DoubleConv(512, 512)

        # -------------------------------------------------------
        # 4. Decoder
        # -------------------------------------------------------
        self.up1 = Up(1024, 256)   # 512 bottleneck + 512 encoder
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)

        self.outc = OutConv(64, out_ch)

    # -----------------------------------------------------------
    # Forward pass
    # -----------------------------------------------------------
    def forward(self, x, t, tokens, extra_cond):
        B = x.size(0)

        # Time + Token embedding → (B, 128)
        t_embed  = self.time_mlp(t.view(B, 1))
        tok_embed = self.token_mlp(tokens)

        cond_vec = t_embed + tok_embed

        # Project multi-scale conditioning
        c64  = self.proj_64(cond_vec).view(B, 64, 1, 1)
        c128 = self.proj_128(cond_vec).view(B, 128, 1, 1)
        c256 = self.proj_256(cond_vec).view(B, 256, 1, 1)
        c512 = self.proj_512(cond_vec).view(B, 512, 1, 1)

        # -------------------------------------------------------
        # Fuse extra_cond (backbone + edge maps) into input
        # -------------------------------------------------------
        cond_proj = self.extra_proj(extra_cond)

        if cond_proj.shape[2:] != x.shape[2:]:
            cond_proj = F.interpolate(cond_proj,
                                      size=x.shape[2:],
                                      mode="bilinear",
                                      align_corners=False)

        x = torch.cat([x, cond_proj], dim=1)

        # -------------------------------------------------------
        # Encoder
        # -------------------------------------------------------
        x0 = self.inc(x)
        x1 = self.down1(x0) + c128
        x2 = self.down2(x1) + c256
        x3 = self.down3(x2) + c512

        # bottleneck
        x4 = self.bot(x3)

        # -------------------------------------------------------
        # Decoder
        # -------------------------------------------------------
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        return self.outc(x)
