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
            # Custom DarkNet19 implementation for PyTorch
            self.features = self._make_darknet19_layers()
        elif name == "darknet53":
            # Custom DarkNet53 implementation for PyTorch
            self.features = self._make_darknet53_layers()
        elif name in ["hrnet_w32", "hrnet_w48"]:
            # HRNet backbone
            variant = "w32" if name == "hrnet_w32" else "w48"
            self.base_model = HRNetBackbone(variant=variant, pretrained=pretrained, weights_path=weights_root_path)
            self.is_hrnet = True
        else:
            raise ValueError(f"'{name}' no such backbone exists.")

        # Load weights if path provided (but not for HRNet, which loads in its own __init__)
        if weights_root_path is not None and not self.is_hrnet:
            if os.path.isfile(weights_root_path):
                self.load_weights(weights_root_path)
            else:
                raise ValueError(f"'{weights_root_path}' no such file.")

    def _make_darknet19_layers(self):
        # Simplified DarkNet19 implementation
        layers = []
        layers.append(nn.Conv2d(3, 32, 3, padding=1))
        layers.append(nn.BatchNorm2d(32))
        layers.append(nn.LeakyReLU(0.1))
        return nn.Sequential(*layers)

    def _make_darknet53_layers(self):
        # Simplified DarkNet53 implementation
        layers = []
        layers.append(nn.Conv2d(3, 32, 3, padding=1))
        layers.append(nn.BatchNorm2d(32))
        layers.append(nn.LeakyReLU(0.1))
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.is_hrnet:
            return self.base_model(x)
        return self.features(x)

    def get_layer_output(self, layer_name):
        """Get output from a specific layer"""
        if hasattr(self, 'base_model'):
            if "vgg" in layer_name:
                return self._get_vgg_layer_output(x, layer_name)
            elif "resnet" in layer_name:
                return self._get_resnet_layer_output(x, layer_name)
        return None

    def _get_vgg_layer_output(self, x, layer_name):
        """Extract VGG layer outputs"""
        features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if str(i) in layer_name:
                features.append(x)
        return features

    def _get_resnet_layer_output(self, x, layer_name):
        """Extract ResNet layer outputs"""
        features = []
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)
        
        x = self.base_model.layer1(x)
        features.append(x)  # C2
        
        x = self.base_model.layer2(x)
        features.append(x)  # C3
        
        x = self.base_model.layer3(x)
        features.append(x)  # C4
        
        x = self.base_model.layer4(x)
        features.append(x)  # C5
        
        return features

    def load_weights(self, path: str):
        if self.is_hrnet:
            # Delegate to HRNet backbone
            self.base_model.load_weights(path)
        else:
            checkpoint = torch.load(path, map_location='cpu')
            if 'state_dict' in checkpoint:
                self.load_state_dict(checkpoint['state_dict'])
            else:
                self.load_state_dict(checkpoint)

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

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
        """Return a dict of intermediate features if available.
        For HRNet, returns the multi-resolution dict produced in forward.
        For torchvision backbones, returns None (hooks are used elsewhere).
        """
        if self.is_hrnet and hasattr(self.base_model, 'get_feature_dict'):
            return self.base_model.get_feature_dict()
        return None