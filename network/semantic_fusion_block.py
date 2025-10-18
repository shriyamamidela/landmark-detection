import torch
import torch.nn as nn


class SemanticFusionBlock(nn.Module):

    def __init__(self, num_filters: int, in_channels: tuple = (512, 1024, 2048), name: str = "semantic_fusion_block"):
        super(SemanticFusionBlock, self).__init__()

        # Accept dynamic input channel sizes for C3, C4, C5
        c3_in, c4_in, c5_in = in_channels
        self.block3_lateral_conv2d = nn.Conv2d(c3_in, num_filters, kernel_size=1, padding=0)
        self.block4_lateral_conv2d = nn.Conv2d(c4_in, num_filters, kernel_size=1, padding=0)
        self.block5_lateral_conv2d = nn.Conv2d(c5_in, num_filters, kernel_size=1, padding=0)

        self.block5_lateral_upsampling = nn.Upsample(scale_factor=2, mode='nearest')
        self.block4_lateral_upsampling = nn.Upsample(scale_factor=2, mode='nearest')

        self.block3_posthoc_conv2d = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.block4_posthoc_conv2d = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.block5_posthoc_conv2d = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)

        self.block3_batchnorm = nn.BatchNorm2d(num_filters, momentum=0.99, eps=1e-5)
        self.block4_batchnorm = nn.BatchNorm2d(num_filters, momentum=0.99, eps=1e-5)
        self.block5_batchnorm = nn.BatchNorm2d(num_filters, momentum=0.99, eps=1e-5)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        C3, C4, C5 = inputs

        block5_lateral_output = self.relu(self.block5_lateral_conv2d(C5))
        block5_feature_output = self.relu(self.block5_posthoc_conv2d(block5_lateral_output))
        P5 = self.block5_batchnorm(block5_feature_output)

        block4_lateral_output = self.relu(self.block4_lateral_conv2d(C4))
        feat_a = self.block5_lateral_upsampling(block5_lateral_output)
        feat_b = block4_lateral_output
        block4_fusion_output = feat_a + feat_b
        block4_feature_output = self.relu(self.block4_posthoc_conv2d(block4_fusion_output))
        P4 = self.block4_batchnorm(block4_feature_output)

        block3_lateral_output = self.relu(self.block3_lateral_conv2d(C3))
        feat_a = self.block4_lateral_upsampling(block4_fusion_output)
        feat_b = block3_lateral_output
        block3_fusion_output = feat_a + feat_b
        block3_feature_output = self.relu(self.block3_posthoc_conv2d(block3_fusion_output))
        P3 = self.block3_batchnorm(block3_feature_output)

        return P3, P4, P5