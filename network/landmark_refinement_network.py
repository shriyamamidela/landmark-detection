import torch
import torch.nn as nn
from config import cfg


class LandmarkRefinementNetwork(nn.Module):

    def __init__(self, input_channels: int):
        super(LandmarkRefinementNetwork, self).__init__()
        
        # Calculate total channels from semantic fusion block (3 feature maps * 256 filters)
        total_channels = 3 * 256  # P3, P4, P5 each have 256 channels
        
        self.heads = nn.ModuleList()
        for index in range(cfg.NUM_LANDMARKS):
            head = nn.Sequential(
                nn.Conv2d(total_channels, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Flatten(),
                nn.Linear(512 * cfg.ROI_POOL_SIZE[0] * cfg.ROI_POOL_SIZE[1], 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 2)
            )
            self.heads.append(head)

    def forward(self, x):
        # x shape: (batch_size, num_proposals, channels, height, width)
        batch_size, num_proposals = x.shape[:2]
        
        outputs = []
        for i in range(num_proposals):
            # Process each proposal separately
            proposal_features = x[:, i, :, :, :]  # (batch_size, channels, height, width)
            
            proposal_outputs = []
            for head in self.heads:
                output = head(proposal_features)  # (batch_size, 2)
                proposal_outputs.append(output)
            
            # Stack outputs for this proposal
            proposal_outputs = torch.stack(proposal_outputs, dim=1)  # (batch_size, num_landmarks, 2)
            outputs.append(proposal_outputs)
        
        # Stack all proposals
        outputs = torch.stack(outputs, dim=1)  # (batch_size, num_proposals, num_landmarks, 2)
        
        return outputs
