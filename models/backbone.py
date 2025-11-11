# models/backbone.py
import torch
import torch.nn as nn
import torchvision.models as models
from preprocessing.utils import generate_edge_bank
import cv2
import numpy as np


class ResNetBackbone(nn.Module):
    """
    ResNet backbone that extracts multi-scale feature maps:
      C2 (1/4), C3 (1/8), C4 (1/16), C5 (1/32)
    Optionally fuses edge-bank (E) features from preprocessing.
    """

    def __init__(self, name="resnet34", pretrained=True, fuse_edges=True):
        super(ResNetBackbone, self).__init__()
        assert name in ["resnet18", "resnet34", "resnet50"], "Supported: resnet18/34/50"

        self.fuse_edges = fuse_edges

        # ------------------------------
        # Load torchvision backbone
        # ------------------------------
        if name == "resnet18":
            net = models.resnet18(pretrained=pretrained)
        elif name == "resnet34":
            net = models.resnet34(pretrained=pretrained)
        else:
            net = models.resnet50(pretrained=pretrained)

        # Keep only the feature extractor layers
        self.stem = nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool
        )
        self.layer1 = net.layer1  # 1/4 resolution (C2)
        self.layer2 = net.layer2  # 1/8 (C3)
        self.layer3 = net.layer3  # 1/16 (C4)
        self.layer4 = net.layer4  # 1/32 (C5)

        # Reduce channel dimension if we concatenate edge maps
        if self.fuse_edges:
            # input channels: original 3 + edge_bank channels (3) = 6
            self.edge_conv = nn.Conv2d(6, 3, kernel_size=3, padding=1)  # fuse RGB+edges -> 3 channels

    def forward(self, x):
        # x expected in range [0,1] float, shape (B,3,H,W)
        if self.fuse_edges:
            edge_maps = []
            for img in x:
                # move to cpu numpy uint8 BGR for consistent cv2 ops
                img_np = (img.detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
                # generate edge bank (HxWx3 uint8)
                edge = generate_edge_bank(img_np)
                # convert to float [0,1]
                edge_tensor = torch.from_numpy(edge.astype(np.float32).transpose(2, 0, 1) / 255.0).to(x.device)
                edge_maps.append(edge_tensor)
            edge_stack = torch.stack(edge_maps, dim=0)  # (B, C_edge, H, W)
            # concat along channel dimension (rgb + edges)
            x = torch.cat([x, edge_stack], dim=1)  # (B, 6, H, W)
            # reduce channels to 3 so ResNet stem conv works (it expects 3 channels)
            x = self.edge_conv(x)

        # Pass through ResNet layers
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        return {"C2": c2, "C3": c3, "C4": c4, "C5": c5}
