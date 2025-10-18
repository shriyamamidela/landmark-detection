import torch
import torch.nn as nn
from config import cfg


class LandmarkDetectionNetwork(nn.Module):

    def __init__(self, input_size: int):
        super(LandmarkDetectionNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(
            in_features=input_size,
            out_features=cfg.NUM_LANDMARKS * 2,
            bias=False
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.dense(x)
        return x
