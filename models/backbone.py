import torch
import torch.nn as nn
import torchvision.models as models
import os
from .hrnet import HRNetBackbone


class Backbone(nn.Module):
    def __init__(
        self,
        name: str,
        pretrained: bool = False,
        weights_root_path: str = None
    ):
        super(Backbone, self).__init__()

        self.name = name
        self.is_hrnet = False

        if name == "vgg16":
            self.base_model = models.vgg16(pretrained=pretrained)
            self.features = self.base_model.features
            self.avgpool = self.base_model.avgpool
            self.classifier = self.base_model.classifier

        elif name == "vgg19":
            self.base_model = models.vgg19(pretrained=pretrained)
            self.features = self.base_model.features
            self.avgpool = self.base_model.avgpool
            self.classifier = self.base_model.classifier

        elif name == "resnet18":
            self.base_model = models.resnet18(pretrained=pretrained)
            self.features = nn.Sequential(
                self.base_model.conv1,
                self.base_model.bn1,
                self.base_model.relu,
                self.base_model.maxpool,
                self.base_model.layer1,
                self.base_model.layer2,
                self.base_model.layer3,
                self.base_model.layer4
            )

        elif name == "resnet34":
            self.base_model = models.resnet34(pretrained=pretrained)
            self.features = nn.Sequential(
                self.base_model.conv1,
                self.base_model.bn1,
                self.base_model.relu,
                self.base_model.maxpool,
                self.base_model.layer1,
                self.base_model.layer2,
                self.base_model.layer3,
                self.base_model.layer4
            )

        elif name == "resnet50":
            self.base_model = models.resnet50(pretrained=pretrained)
            self.features = nn.Sequential(
                self.base_model.conv1,
                self.base_model.bn1,
                self.base_model.relu,
                self.base_model.maxpool,
                self.base_model.layer1,
                self.base_model.layer2,
                self.base_model.layer3,
                self.base_model.layer4
            )

        elif name == "darknet19":
            self.features = self._make_darknet19_layers()

        elif name == "darknet53":
            self.features = self._make_darknet53_layers()

        elif name in ["hrnet_w32", "hrnet_w48"]:
            # ✅ HRNet backbone (auto-select variant)
            variant = "w32" if name == "hrnet_w32" else "w48"
            print(f"Creating HRNet backbone ({variant.upper()})...")
            self.base_model = HRNetBackbone(
                variant=variant,
                pretrained=pretrained,
                weights_path=weights_root_path
            )
            self.is_hrnet = True

            if weights_root_path and os.path.isfile(weights_root_path):
                print(f"✓ Loaded pretrained HRNet-{variant.upper()} weights from {weights_root_path}")
            else:
                print(f"⚠️ No pretrained weights found — using random initialization.")

        else:
            raise ValueError(f"'{name}' no such backbone exists.")

        # Load non-HRNet weights manually (if path provided)
        if weights_root_path is not None and not self.is_hrnet:
            if os.path.isfile(weights_root_path):
                self.load_weights(weights_root_path)
            else:
                raise ValueError(f"'{weights_root_path}' no such file.")

    # ---------------------------------------------------------------------- #
    # Simplified DarkNet placeholder layers
    # ---------------------------------------------------------------------- #
    def _make_darknet19_layers(self):
        layers = [
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1)
        ]
        return nn.Sequential(*layers)

    def _make_darknet53_layers(self):
        layers = [
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1)
        ]
        return nn.Sequential(*layers)

    # ---------------------------------------------------------------------- #
    # Forward and utilities
    # ---------------------------------------------------------------------- #
    def forward(self, x):
        if self.is_hrnet:
            return self.base_model(x)
        return self.features(x)

    def load_weights(self, path: str):
        """Generic weight loader for non-HRNet backbones."""
        if self.is_hrnet:
            self.base_model.load_weights(path)
        else:
            checkpoint = torch.load(path, map_location='cpu')
            if 'state_dict' in checkpoint:
                self.load_state_dict(checkpoint['state_dict'])
            else:
                self.load_state_dict(checkpoint)
            print(f"✓ Loaded non-HRNet backbone weights from {path}")

    def save_weights(self, path):
        torch.save(self.state_dict(), path)
        print(f"✓ Saved backbone weights to {path}")

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def summary(self):
        print(self)
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    def get_feature_dict(self):
        """Return HRNet's multi-resolution features if available."""
        if self.is_hrnet and hasattr(self.base_model, 'get_feature_dict'):
            return self.base_model.get_feature_dict()
        return None
