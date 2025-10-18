import torch
import torch.nn as nn


class SuccessfulDetectionRate(nn.Module):

    def __init__(self, precision_range: int = 40):
        super(SuccessfulDetectionRate, self).__init__()
        self.precision_range = precision_range
        self.successful_detection_rate = None

    def forward(self, y_true, y_pred):
        delta_x = y_pred[:, :, 0] - y_true[:, :, 0]
        delta_y = y_pred[:, :, 1] - y_true[:, :, 1]

        errors = torch.sqrt(delta_x**2 + delta_y**2)

        self.successful_detection_rate = (errors <= self.precision_range).float().mean()
        return self.successful_detection_rate
