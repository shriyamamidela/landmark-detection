import torch
import torch.nn as nn


class SemanticFusionBlock(nn.Module):
    """
    Semantic Fusion Block (SFB)
    - Fuses multi-resolution HRNet features (e.g., R8, R16, R32)
    - Produces pyramid outputs (P3, P4, P5)
    """

    def __init__(self, num_filters: int = 256, in_channels=(64, 128, 256), name="semantic_fusion_block"):
        super(SemanticFusionBlock, self).__init__()
        self.name = name
        c3_in, c4_in, c5_in = in_channels

        # 1x1 lateral convolutions to unify channel dimensions
        self.lateral3 = nn.Conv2d(c3_in, num_filters, kernel_size=1)
        self.lateral4 = nn.Conv2d(c4_in, num_filters, kernel_size=1)
        self.lateral5 = nn.Conv2d(c5_in, num_filters, kernel_size=1)

        # Posthoc 3x3 refinement
        self.refine3 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.refine4 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.refine5 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)

        # Normalization
        self.bn3 = nn.BatchNorm2d(num_filters)
        self.bn4 = nn.BatchNorm2d(num_filters)
        self.bn5 = nn.BatchNorm2d(num_filters)

        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, inputs):
        """
        Args:
            inputs: (R8, R16, R32) feature maps from HRNet
        Returns:
            P3, P4, P5: fused feature pyramid outputs
        """
        R8, R16, R32 = inputs

        # Step 1: lateral projections (reduce channels to num_filters)
        L8 = self.lateral3(R8)
        L16 = self.lateral4(R16)
        L32 = self.lateral5(R32)

        # Step 2: top-down fusion with cumulative feature aggregation
        # P5: coarsest level (1/32)
        P5 = self.relu(self.bn5(self.refine5(L32)))
        
        # P4: middle level (1/16) - uses upsampled L32 (not P5)
        up_p5 = self.upsample(L32)
        P4 = self.relu(self.bn4(self.refine4(L16 + up_p5)))
        
        # P3: finest level (1/8) - uses cumulative upsampled features
        up_p4 = self.upsample(L16 + up_p5)
        P3 = self.relu(self.bn3(self.refine3(L8 + up_p4)))

        return P3, P4, P5
