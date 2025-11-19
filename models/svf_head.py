import torch
import torch.nn as nn
import torch.nn.functional as F


class SVFHead(nn.Module):
    """
    Small head that predicts stationary velocity field (svf) with two channels.
    Input: feature map (B, C, h, w) + optional conditioning (we pass only F5 & D_small in training)
    Output: (B, 2, h, w) velocities in pixels (approx), small initial magnitude
    """

    def __init__(self, in_channels=512, hidden=256, out_channels=2, out_scale=16.0):
        super().__init__()
        self.out_scale = float(out_scale)

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, kernel_size=3, padding=1)
        )

        # init last conv to small weights so initial displacements are tiny
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=1e-3)
        if self.net[-1].bias is not None:
            nn.init.constant_(self.net[-1].bias, 0.0)

    def forward(self, feats, extra_input=None):
        """
        feats: (B, C, h, w)
        extra_input: optionally concatenated before final conv (not used by default)
        """
        x = feats
        # If extra_input (eg D_small) provided and same size, concat channels
        if extra_input is not None:
            # expect extra_input (B,1,h,w) or (B,k,h,w)
            if extra_input.shape[2:] != x.shape[2:]:
                extra_resized = F.interpolate(extra_input, size=x.shape[2:], mode="bilinear", align_corners=False)
            else:
                extra_resized = extra_input
            x = torch.cat([x, extra_resized], dim=1)

        # If we concatenated, the in_channels will not match; ensure network was constructed appropriately.
        out = self.net(x)

        # scale output — this constrains initial magnitude (tweak out_scale if needed)
        return out * (self.out_scale / 255.0)
