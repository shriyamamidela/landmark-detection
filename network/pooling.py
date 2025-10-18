import torch
import torch.nn as nn
from torchvision.ops import roi_align


class ROIAlign2D(nn.Module):

    def __init__(self, crop_size: tuple, name: str = "roi_align_2d"):
        super(ROIAlign2D, self).__init__()
        self.crop_size = crop_size
        self._name = name

    def forward(self, feature_maps, roi_proposals):
        # roi_proposals shape: (batch_size, num_proposals, 4) where 4 = [x1, y1, x2, y2]
        batch_size = roi_proposals.shape[0]
        num_proposals = roi_proposals.shape[1]
        
        # Reshape proposals for roi_align: (batch_size * num_proposals, 5) where 5 = [batch_idx, x1, y1, x2, y2]
        batch_indices = torch.arange(batch_size, device=roi_proposals.device).view(-1, 1).repeat(1, num_proposals)
        batch_indices = batch_indices.view(-1, 1)
        
        # Flatten proposals and add batch indices
        proposals_flat = roi_proposals.view(-1, 4)
        proposals_with_batch = torch.cat([batch_indices, proposals_flat], dim=1)
        
        cropped_maps = []
        for feature_map in feature_maps:
            # roi_align expects proposals in format [x1, y1, x2, y2] (not [y1, x1, y2, x2])
            # Our input is already in [x1, y1, x2, y2] format
            cropped_features = roi_align(
                feature_map,
                proposals_with_batch,
                output_size=self.crop_size,
                spatial_scale=1.0,
                sampling_ratio=-1
            )
            cropped_maps.append(cropped_features)
        
        # Concatenate along channel dimension
        cropped_maps = torch.cat(cropped_maps, dim=1)
        
        # Reshape back to (batch_size, num_proposals, channels, height, width)
        cropped_maps = cropped_maps.view(batch_size, num_proposals, -1, self.crop_size[0], self.crop_size[1])
        
        return cropped_maps
