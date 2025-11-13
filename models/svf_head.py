import torch
import torch.nn as nn
import torch.nn.functional as F

class SVFHead(nn.Module):
    """
    SVF head predicts a stationary velocity field v from backbone features F and DT map D.
    Output: v in pixel displacements (B,2,Hf,Wf), same spatial resolution as F.
    """
    def __init__(self, in_channels, mid_channels=128, out_channels=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + 1, mid_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)

        # init small last-layer weights so initial displacement is small
        nn.init.normal_(self.conv3.weight, mean=0.0, std=1e-3)
        if self.conv3.bias is not None:
            nn.init.zeros_(self.conv3.bias)

    def forward(self, F_feat, D_map):
        # resize D_map to F spatial if needed
        if D_map.shape[2:] != F_feat.shape[2:]:
            D_res = F.interpolate(D_map, size=F_feat.shape[2:], mode='bilinear', align_corners=False)
        else:
            D_res = D_map
        x = torch.cat([F_feat, D_res], dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        v = self.conv3(x)  # (B,2,Hf,Wf)
        return v
