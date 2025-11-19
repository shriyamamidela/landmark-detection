import torch
import torch.nn as nn
import torch.nn.functional as F

class SVFHead(nn.Module):
    """
    SVF head that takes:
        - backbone C5 feature map  (B,512,H,W)
        - downsampled distance transform (B,1,H,W)
    Concatenates them -> (B,513,H,W)
    Produces SVF field (B,2,H,W)
    """

    def __init__(self, in_channels=512):
        super().__init__()

        # after concatenating DT: +1 channel
        inc = in_channels + 1   # 512 + 1 = 513

        self.net = nn.Sequential(
            nn.Conv2d(inc, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 2, kernel_size=3, padding=1)
        )

    def forward(self, feat_c5, dt_small):
        """
        feat_c5:  (B,512,H,W)
        dt_small: (B,1,H,W)
        """

        # concatenate along channel dim
        x = torch.cat([feat_c5, dt_small], dim=1)   # → (B,513,H,W)

        out = self.net(x)  # → (B,2,H,W)
        return out
