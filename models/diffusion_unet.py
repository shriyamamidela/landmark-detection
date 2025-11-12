import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
#  U-Net building blocks
# ------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))
    def forward(self, x): return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY, diffX = x2.size()[2] - x1.size()[2], x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
    def forward(self, x): return self.conv(x)


# ------------------------------
#  Conditional Diffusion U-Net
# ------------------------------
class ConditionalUNet(nn.Module):
    """
    εθ(D_t | F, E, T, t)
    Predicts noise or clean DT map conditioned on
    backbone features (F), edge maps (E), topology tokens (T), and timestep (t).
    """
    def __init__(self, cond_dim=243, in_ch=4, out_ch=1, feat_dim=512):
        super().__init__()

        # timestep & token embedding
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

        # feature + edge conditioning (F + E)
        self.extra_proj = nn.Conv2d(feat_dim+3, 64, kernel_size=1)

        # projection layers for multi-scale conditioning
        self.proj_64 = nn.Linear(128, 64)
        self.proj_128 = nn.Linear(128, 128)
        self.proj_256 = nn.Linear(128, 256)
        self.proj_512 = nn.Linear(128, 512)

        # U-Net encoder / decoder
        self.inc = DoubleConv(in_ch, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.bot = DoubleConv(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.outc = OutConv(64, out_ch)

    def forward(self, x, t, tokens, extra_cond=None):
        B = x.size(0)

        # --- timestep + token conditioning ---
        t_embed = self.time_mlp(t.view(B, 1))
        tok_embed = self.token_mlp(tokens)
        cond_vec = (t_embed + tok_embed)

        # project condition to hierarchical scales
        c64  = self.proj_64(cond_vec).view(B, 64, 1, 1)
        c128 = self.proj_128(cond_vec).view(B, 128, 1, 1)
        c256 = self.proj_256(cond_vec).view(B, 256, 1, 1)
        c512 = self.proj_512(cond_vec).view(B, 512, 1, 1)

        # --- inject feature conditioning (F + E) ---
        if extra_cond is not None:
            cond_feat = self.extra_proj(extra_cond)
            if cond_feat.shape[2:] != x.shape[2:]:
                cond_feat = F.interpolate(cond_feat, size=x.shape[2:], mode="bilinear", align_corners=False)
            x = x + cond_feat

        # --- encoder ---
        x0 = self.inc(x)
        x1 = self.down1(x0) + c128
        x2 = self.down2(x1) + c256
        x3 = self.down3(x2) + c512
        x4 = self.bot(x3 + c512)

        # --- decoder ---
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.outc(x)
