import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


def make_layer(block, inplanes, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )
    layers = [block(inplanes, planes, stride, downsample)]
    inplanes = planes * block.expansion
    for _ in range(1, blocks):
        layers.append(block(inplanes, planes))
    return nn.Sequential(*layers), inplanes


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches: int, blocks, num_blocks: List[int], num_inchannels: List[int], num_channels: List[int], fuse_method="SUM"):
        super(HighResolutionModule, self).__init__()
        self.num_branches = num_branches
        self.fuse_method = fuse_method
        self.relu = nn.ReLU(inplace=True)

        # Branches
        self.branches = nn.ModuleList()
        self.num_inchannels = list(num_inchannels)
        for i in range(num_branches):
            branch, out_c = make_layer(blocks, num_inchannels[i], num_channels[i], num_blocks[i])
            self.branches.append(branch)
            self.num_inchannels[i] = out_c

        # Fuse layers
        self.fuse_layers = nn.ModuleList()
        for i in range(num_branches):
            fuse_layer = nn.ModuleList()
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(self.num_inchannels[j], self.num_inchannels[i], kernel_size=1, bias=False),
                        nn.BatchNorm2d(self.num_inchannels[i]),
                        nn.Upsample(scale_factor=2 ** (j - i), mode='nearest')
                    ))
                elif j == i:
                    fuse_layer.append(nn.Identity())
                else:
                    downsamples = []
                    inch = self.num_inchannels[j]
                    for k in range(i - j):
                        outch = inch if k != i - j - 1 else self.num_inchannels[i]
                        downsamples.append(nn.Sequential(
                            nn.Conv2d(inch, outch, kernel_size=3, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(outch),
                            nn.ReLU(inplace=True)
                        ))
                        inch = outch
                    fuse_layer.append(nn.Sequential(*downsamples))
            self.fuse_layers.append(fuse_layer)

    def forward(self, xs: List[torch.Tensor]) -> List[torch.Tensor]:
        assert len(xs) == self.num_branches
        # Process each branch
        xs = [branch(x) for branch, x in zip(self.branches, xs)]
        # Fuse
        x_fused = []
        for i in range(self.num_branches):
            y = 0
            for j in range(self.num_branches):
                y = y + self.fuse_layers[i][j](xs[j])
            x_fused.append(self.relu(y))
        return x_fused


class HRNetBackbone(nn.Module):
    def __init__(self, variant: str = "w32", pretrained: bool = False, weights_path: Optional[str] = None):
        super(HRNetBackbone, self).__init__()
        assert variant in ("w32", "w48"), "variant must be 'w32' or 'w48'"
        
        # Stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        # Stage 1 (ResNet-like)
        block = Bottleneck
        self.layer1, c1 = make_layer(block, 64, 64, blocks=4)

        # Channels per branch
        base_c = 32 if variant == "w32" else 48
        c2 = [base_c, base_c * 2]
        c3 = [base_c, base_c * 2, base_c * 4]
        c4 = [base_c, base_c * 2, base_c * 4, base_c * 8]

        # Transition layers to stage2
        self.transition1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c1, c2[0], kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c2[0]),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(c1, c2[1], kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c2[1]),
                nn.ReLU(inplace=True)
            )
        ])

        # Stage 2
        self.stage2 = HighResolutionModule(num_branches=2, blocks=BasicBlock, num_blocks=[4, 4], num_inchannels=c2, num_channels=c2)

        # Transition to stage3
        self.transition2 = nn.ModuleList([
            nn.Identity(),
            nn.Identity(),
            nn.Sequential(
                nn.Conv2d(c2[1], c3[2], kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c3[2]),
                nn.ReLU(inplace=True)
            )
        ])

        # Stage 3
        self.stage3 = HighResolutionModule(num_branches=3, blocks=BasicBlock, num_blocks=[4, 4, 4], num_inchannels=[c2[0], c2[1], c3[2]], num_channels=c3)

        # Transition to stage4
        self.transition3 = nn.ModuleList([
            nn.Identity(),
            nn.Identity(),
            nn.Identity(),
            nn.Sequential(
                nn.Conv2d(c3[2], c4[3], kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c4[3]),
                nn.ReLU(inplace=True)
            )
        ])

        # Stage 4
        self.stage4 = HighResolutionModule(num_branches=4, blocks=BasicBlock, num_blocks=[4, 4, 4, 4], num_inchannels=[c3[0], c3[1], c3[2], c4[3]], num_channels=c4)

        # Store channel dimensions
        self.c4_channels = c4  # [32, 64, 128, 256] for w32
        
        # SIMPLIFIED APPROACH: Direct concatenation + projection at each scale
        # This is more robust and easier to debug
        
        total_c = sum(c4)  # 480 for w32, 720 for w48
        
        # C3 head: All branches to 1/8 resolution, then project to 512 channels
        self.c3_head = nn.Sequential(
            nn.Conv2d(total_c, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # C4 head: All branches to 1/16 resolution, then project to 1024 channels
        self.c4_head = nn.Sequential(
            nn.Conv2d(total_c, 1024, kernel_size=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        
        # C5 head: All branches to 1/32 resolution, then project to 2048 channels
        self.c5_head = nn.Sequential(
            nn.Conv2d(total_c, 2048, kernel_size=1, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.Conv2d(2048, 2048, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True)
        )

        self._out_channel_map = {
            "C3": 512,
            "C4": 1024,
            "C5": 2048,
        }

        self._latest_features: Dict[str, torch.Tensor] = {}

        if pretrained and weights_path is not None:
            self.load_weights(weights_path)

    def _align_and_concat(self, branches: List[torch.Tensor], target_size: Tuple[int, int]) -> torch.Tensor:
        """Align all branches to target spatial size and concatenate"""
        aligned = []
        for feat in branches:
            if feat.shape[2:] != target_size:
                # Use bilinear for upsampling, strided conv for downsampling
                if feat.shape[2] < target_size[0]:
                    # Upsample
                    feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
                else:
                    # Downsample with avg pooling
                    scale = feat.shape[2] // target_size[0]
                    feat = F.avg_pool2d(feat, kernel_size=scale, stride=scale)
            aligned.append(feat)
        return torch.cat(aligned, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        
        # Stage 1
        x = self.layer1(x)
        
        # Transition to 2 branches
        x_list = [self.transition1[0](x), self.transition1[1](x)]
        
        # Stage 2
        y_list = self.stage2(x_list)
        
        # Transition to 3 branches
        x_list = [self.transition2[0](y_list[0]), self.transition2[1](y_list[1]), self.transition2[2](y_list[1])]
        
        # Stage 3
        y_list = self.stage3(x_list)
        
        # Transition to 4 branches
        x_list = [self.transition3[0](y_list[0]), self.transition3[1](y_list[1]), self.transition3[2](y_list[2]), self.transition3[3](y_list[2])]
        
        # Stage 4 - produces [R4, R8, R16, R32]
        y_list = self.stage4(x_list)
        r4, r8, r16, r32 = y_list
        
        # Print shapes for debugging
        # print(f"R4: {r4.shape}, R8: {r8.shape}, R16: {r16.shape}, R32: {r32.shape}")
        
        # Create C3 at 1/8 resolution (same as R8)
        target_size_c3 = (r8.shape[2], r8.shape[3])
        concat_c3 = self._align_and_concat([r4, r8, r16, r32], target_size_c3)
        # print(f"DEBUG concat_c3 shape: {concat_c3.shape}")
        C3 = self.c3_head(concat_c3)
        # print(f"DEBUG C3 shape after head: {C3.shape}")
        
        # Create C4 at 1/16 resolution (same as R16)
        target_size_c4 = (r16.shape[2], r16.shape[3])
        concat_c4 = self._align_and_concat([r4, r8, r16, r32], target_size_c4)
        # print(f"DEBUG concat_c4 shape: {concat_c4.shape}")
        C4 = self.c4_head(concat_c4)
        # print(f"DEBUG C4 shape after head: {C4.shape}")
        
        # Create C5 at 1/32 resolution (same as R32)
        target_size_c5 = (r32.shape[2], r32.shape[3])
        concat_c5 = self._align_and_concat([r4, r8, r16, r32], target_size_c5)
        # print(f"DEBUG concat_c5 shape: {concat_c5.shape}")
        C5 = self.c5_head(concat_c5)
        # print(f"DEBUG C5 shape after head: {C5.shape}")

        # Store feature dict
        feature_dict = {
            "R4": r4,
            "R8": r8,
            "R16": r16,
            "R32": r32,
            "C3": C3,
            "C4": C4,
            "C5": C5,
        }
        self._latest_features = feature_dict
        
        # Return highest resolution as primary output
        return r4

    def get_feature_dict(self) -> Dict[str, torch.Tensor]:
        return self._latest_features

    def get_out_channels_for_keys(self, keys: Tuple[str, str, str] = ("C3", "C4", "C5")) -> Tuple[int, int, int]:
        return (self._out_channel_map[keys[0]], self._out_channel_map[keys[1]], self._out_channel_map[keys[2]])

    def load_weights(self, path: str):
        """Load pretrained weights, handling key mismatches for our custom heads"""
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and 'state_dict' in state:
            state_dict = state['state_dict']
        else:
            state_dict = state
        
        # Filter out keys that don't match (our custom C3/C4/C5 heads)
        model_dict = self.state_dict()
        pretrained_dict = {}
        
        for k, v in state_dict.items():
            # Remove 'module.' prefix if it exists
            if k.startswith('module.'):
                k = k[7:]
            
            # Only load weights that match our model and have same shape
            if k in model_dict and model_dict[k].shape == v.shape:
                pretrained_dict[k] = v
            else:
                print(f"Skipping layer {k} (shape mismatch or not in model)")
        
        # Update our model dict
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict, strict=False)
        
        print(f"Loaded {len(pretrained_dict)}/{len(state_dict)} pretrained layers")
        print(f"Randomly initialized: c3_head, c4_head, c5_head (custom layers)")
