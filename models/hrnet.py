import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


# ------------------------------------------------------------
# Basic building blocks
# ------------------------------------------------------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(planes * self.expansion)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
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


# ------------------------------------------------------------
# High-Resolution Module
# ------------------------------------------------------------
class HighResolutionModule(nn.Module):
    def __init__(
        self,
        num_branches: int,
        blocks,
        num_blocks: List[int],
        num_inchannels: List[int],
        num_channels: List[int],
        fuse_method: str = "SUM"
    ):
        super().__init__()
        assert num_branches == len(num_blocks) == len(num_channels) == len(num_inchannels)
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
                    # upsample j -> i
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(self.num_inchannels[j], self.num_inchannels[i], kernel_size=1, bias=False),
                        nn.BatchNorm2d(self.num_inchannels[i]),
                        nn.Upsample(scale_factor=2 ** (j - i), mode="nearest")
                    ))
                elif j == i:
                    fuse_layer.append(nn.Identity())
                else:
                    # downsample j -> i
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
        # per-branch processing
        xs = [branch(x) for branch, x in zip(self.branches, xs)]
        # fuse across branches
        x_fused = []
        for i in range(self.num_branches):
            y = 0
            for j in range(self.num_branches):
                y = y + self.fuse_layers[i][j](xs[j])
            x_fused.append(self.relu(y))
        return x_fused


# ------------------------------------------------------------
# HRNet Backbone (Feature Extraction Oriented)
# ------------------------------------------------------------
class HRNetBackbone(nn.Module):
    """
    HRNet for feature extraction.

    Key points:
    - Produces native HRNet multi-resolution outputs: R4 (1/4), R8 (1/8), R16 (1/16), R32 (1/32).
    - Optional C3/C4/C5 heads (disabled by default) that aggregate all branches at fixed scales.
    - Recommended for landmark/heatmap tasks to use R-branches (esp. R4) or fuse R4–R32 externally.
    """

    def __init__(
        self,
        variant: str = "w32",
        pretrained: bool = False,
        weights_path: Optional[str] = None,
        use_c_heads: bool = False,   # NEW: skip C3/C4/C5 for pure feature extraction
    ):
        super().__init__()
        assert variant in ("w32", "w48"), "variant must be 'w32' or 'w48'"
        self.variant = variant
        self.use_c_heads = use_c_heads

        # ---------------- Stem ----------------
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(64)

        # ---------------- Stage 1 (ResNet-like) ----------------
        block = Bottleneck
        self.layer1, c1 = make_layer(block, 64, 64, blocks=4)  # output channels: 64 * 4 = 256

        # Channels per branch
        base_c = 32 if variant == "w32" else 48
        c2 = [base_c, base_c * 2]                  # [32, 64] or [48, 96]
        c3 = [base_c, base_c * 2, base_c * 4]      # [32, 64, 128] or [48, 96, 192]
        c4 = [base_c, base_c * 2, base_c * 4, base_c * 8]  # [32, 64, 128, 256] or [48, 96, 192, 384]
        self._branch_channels = c4  # for Stage 4 ordering

        # ---------------- Transition to Stage 2 ----------------
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

        # ---------------- Stage 2 (2 branches) ----------------
        self.stage2 = HighResolutionModule(
            num_branches=2, blocks=BasicBlock, num_blocks=[4, 4],
            num_inchannels=c2, num_channels=c2
        )

        # ---------------- Transition to Stage 3 (3 branches) ----------------
        self.transition2 = nn.ModuleList([
            nn.Identity(),             # keep branch 0
            nn.Identity(),             # keep branch 1
            nn.Sequential(             # create new downsampled branch from branch1
                nn.Conv2d(c2[1], c3[2], kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c3[2]),
                nn.ReLU(inplace=True)
            )
        ])

        # ---------------- Stage 3 (3 branches) ----------------
        self.stage3 = HighResolutionModule(
            num_branches=3, blocks=BasicBlock, num_blocks=[4, 4, 4],
            num_inchannels=[c2[0], c2[1], c3[2]], num_channels=c3
        )

        # ---------------- Transition to Stage 4 (4 branches) ----------------
        self.transition3 = nn.ModuleList([
            nn.Identity(),             # keep branch 0
            nn.Identity(),             # keep branch 1
            nn.Identity(),             # keep branch 2
            nn.Sequential(             # create new downsampled branch from branch2
                nn.Conv2d(c3[2], c4[3], kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c4[3]),
                nn.ReLU(inplace=True)
            )
        ])

        # ---------------- Stage 4 (4 branches -> R4,R8,R16,R32) ----------------
        self.stage4 = HighResolutionModule(
            num_branches=4, blocks=BasicBlock, num_blocks=[4, 4, 4, 4],
            num_inchannels=[c3[0], c3[1], c3[2], c4[3]], num_channels=c4
        )

        # ---------------- Optional C-heads (FPN-like) ----------------
        total_c = sum(c4)  # 480 for w32, 720 for w48
        if self.use_c_heads:
            self.c3_head = nn.Sequential(
                nn.Conv2d(total_c, 512, kernel_size=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            )
            self.c4_head = nn.Sequential(
                nn.Conv2d(total_c, 1024, kernel_size=1, bias=False),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True),
                nn.Conv2d(1024, 1024, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True)
            )
            self.c5_head = nn.Sequential(
                nn.Conv2d(total_c, 2048, kernel_size=1, bias=False),
                nn.BatchNorm2d(2048),
                nn.ReLU(inplace=True),
                nn.Conv2d(2048, 2048, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(2048),
                nn.ReLU(inplace=True)
            )
            self._out_channel_map = {"C3": 512, "C4": 1024, "C5": 2048}
        else:
            self.c3_head = self.c4_head = self.c5_head = None
            self._out_channel_map = {}

        # latest features store
        self._latest_features: Dict[str, torch.Tensor] = {}

        # load weights if provided
        if pretrained and weights_path is not None:
            self.load_weights(weights_path)

    # ---------------- Helpers ----------------
    def _align_and_concat(self, branches: List[torch.Tensor], target_size: Tuple[int, int]) -> torch.Tensor:
        """
        Align branches to target spatial size and concatenate on channel dim.
        Uses bilinear upsample and average pooling downsample.
        """
        aligned = []
        for feat in branches:
            if feat.shape[2:] != target_size:
                if feat.shape[2] < target_size[0]:  # upsample
                    feat = F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False)
                else:  # downsample by integer ratio with avg pool
                    # robust downsample (works even if not perfectly divisible)
                    feat = F.adaptive_avg_pool2d(feat, target_size)
            aligned.append(feat)
        return torch.cat(aligned, dim=1)

    # ---------------- Forward ----------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        # Stage 1
        x = self.layer1(x)

        # To Stage 2 (2 branches)
        x_list = [self.transition1[0](x), self.transition1[1](x)]
        y_list = self.stage2(x_list)

        # To Stage 3 (3 branches)
        x_list = [
            self.transition2[0](y_list[0]),
            self.transition2[1](y_list[1]),
            self.transition2[2](y_list[1]),
        ]
        y_list = self.stage3(x_list)

        # To Stage 4 (4 branches) -> R4,R8,R16,R32
        x_list = [
            self.transition3[0](y_list[0]),
            self.transition3[1](y_list[1]),
            self.transition3[2](y_list[2]),
            self.transition3[3](y_list[2]),
        ]
        r4, r8, r16, r32 = self.stage4(x_list)

        feature_dict: Dict[str, torch.Tensor] = {
            "R4": r4,   # highest resolution (1/4)
            "R8": r8,
            "R16": r16,
            "R32": r32,
        }

        # Optional fixed-scale heads (C3/C4/C5)
        if self.use_c_heads:
            target_c3 = (r8.shape[2],  r8.shape[3])   # 1/8
            target_c4 = (r16.shape[2], r16.shape[3])  # 1/16
            target_c5 = (r32.shape[2], r32.shape[3])  # 1/32
            all_branches = [r4, r8, r16, r32]

            concat_c3 = self._align_and_concat(all_branches, target_c3)
            C3 = self.c3_head(concat_c3)

            concat_c4 = self._align_and_concat(all_branches, target_c4)
            C4 = self.c4_head(concat_c4)

            concat_c5 = self._align_and_concat(all_branches, target_c5)
            C5 = self.c5_head(concat_c5)

            feature_dict.update({"C3": C3, "C4": C4, "C5": C5})

        self._latest_features = feature_dict

        # Return highest-resolution tensor by default (useful as a primary feature output)
        return r4

    # ---------------- Accessors ----------------
    def get_feature_dict(self) -> Dict[str, torch.Tensor]:
        return self._latest_features

    def get_out_channels_for_keys(self, keys: Tuple[str, str, str] = ("C3", "C4", "C5")) -> Tuple[int, int, int]:
        # Only valid when use_c_heads=True
        assert self.use_c_heads, "C-heads are disabled; enable use_c_heads=True to query C3/C4/C5 channels."
        return tuple(self._out_channel_map[k] for k in keys)

    # ---------------- Weights loading ----------------
    def load_weights(self, path: str):
        """
        Load pretrained weights; ignores unmatched shapes/keys.
        Useful when the checkpoint doesn't contain C-heads or vice versa.
        """
        state = torch.load(path, map_location="cpu")
        state_dict = state.get("state_dict", state)

        model_dict = self.state_dict()
        pretrained_dict = {}
        skipped = 0

        for k, v in state_dict.items():
            # Remove 'module.' prefix if present
            if k.startswith("module."):
                k = k[7:]
            if k in model_dict and model_dict[k].shape == v.shape:
                pretrained_dict[k] = v
            else:
                skipped += 1

        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict, strict=False)

        print(f"Loaded {len(pretrained_dict)} layers from checkpoint. Skipped {skipped}.")
        if not self.use_c_heads:
            print("Note: C3/C4/C5 heads are disabled (use_c_heads=False).")
