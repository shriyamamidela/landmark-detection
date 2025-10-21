import torch
import torch.nn as nn
import torchvision.models as models
import os

from .hrnet import HRNetBackbone


class Backbone(nn.Module):
    """
    Unified backbone interface.
    Supports: VGG, ResNet, DarkNet, HRNet.
    For feature extraction, HRNet is preferred (R4, R8, R16, R32 outputs).
    """

    def __init__(self, name: str, pretrained: bool = False, weights_root_path: str = None):
        super().__init__()
        self.name = name.lower()
        self.is_hrnet = False

        # ---------------- HRNet ----------------
        if self.name in ["hrnet_w32", "hrnet_w48"]:
            variant = "w32" if "w32" in self.name else "w48"
            self.base_model = HRNetBackbone(
                variant=variant,
                pretrained=pretrained,
                weights_path=weights_root_path,
                use_c_heads=False  # ✅ pure feature extraction only
            )
            self.is_hrnet = True
            return  # HRNet initialized; skip below

        # ---------------- VGG ----------------
        elif self.name == "vgg16":
            model = models.vgg16(pretrained=pretrained)
            self.features = model.features

        elif self.name == "vgg19":
            model = models.vgg19(pretrained=pretrained)
            self.features = model.features

        # ---------------- ResNet ----------------
        elif self.name == "resnet18":
            model = models.resnet18(pretrained=pretrained)
            self.features = nn.Sequential(
                model.conv1, model.bn1, model.relu, model.maxpool,
                model.layer1, model.layer2, model.layer3, model.layer4
            )

        elif self.name == "resnet34":
            model = models.resnet34(pretrained=pretrained)
            self.features = nn.Sequential(
                model.conv1, model.bn1, model.relu, model.maxpool,
                model.layer1, model.layer2, model.layer3, model.layer4
            )

        elif self.name == "resnet50":
            model = models.resnet50(pretrained=pretrained)
            self.features = nn.Sequential(
                model.conv1, model.bn1, model.relu, model.maxpool,
                model.layer1, model.layer2, model.layer3, model.layer4
            )

        # ---------------- DarkNet (optional minimal stubs) ----------------
        elif self.name == "darknet19":
            self.features = self._make_darknet19_layers()
        elif self.name == "darknet53":
            self.features = self._make_darknet53_layers()

        else:
            raise ValueError(f"Unsupported backbone: {self.name}")

        # ---------------- Load custom weights if provided ----------------
        if weights_root_path is not None and not self.is_hrnet:
            if os.path.isfile(weights_root_path):
                self.load_weights(weights_root_path)
            else:
                raise FileNotFoundError(f"Weights file not found: {weights_root_path}")

    # ---------------------------------------------------------------------
    # Custom lightweight DarkNet placeholders
    # ---------------------------------------------------------------------
    def _make_darknet19_layers(self):
        layers = [
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1)
        ]
        return nn.Sequential(*layers)

    def _make_darknet53_layers(self):
        layers = [
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1)
        ]
        return nn.Sequential(*layers)

    # ---------------------------------------------------------------------
    # Forward pass
    # ---------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """Run the backbone forward."""
        if self.is_hrnet:
            return self.base_model(x)
        return self.features(x)

    # ---------------------------------------------------------------------
    # Feature dictionary access
    # ---------------------------------------------------------------------
    def get_feature_dict(self):
        """Return HRNet multi-scale features if available."""
        if self.is_hrnet and hasattr(self.base_model, "get_feature_dict"):
            return self.base_model.get_feature_dict()
        return None

    # ---------------------------------------------------------------------
    # Weight utilities
    # ---------------------------------------------------------------------
    def load_weights(self, path: str):
        """Load model weights (supports HRNet and torchvision)."""
        if self.is_hrnet:
            self.base_model.load_weights(path)
        else:
            checkpoint = torch.load(path, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint)
            self.load_state_dict(state_dict, strict=False)
            print(f"✓ Loaded weights from {path}")

    def save_weights(self, path: str):
        """Save current weights."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"✓ Saved weights to {path}")

    # ---------------------------------------------------------------------
    # Utility methods
    # ---------------------------------------------------------------------
    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True

    def summary(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\nBackbone: {self.name}")
        print(f"Total params: {total_params:,}")
        print(f"Trainable: {trainable:,}")
        print(self)
