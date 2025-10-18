import torch
import torch.nn as nn


class MeanSquaredError(nn.Module):

    def __init__(self):
        super(MeanSquaredError, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, y_true, y_pred, mask=None):
        loss = self.mse_loss(y_pred, y_true)
        if mask is not None:
            loss = loss * mask
        return loss.mean()


class MeanRadialError(nn.Module):

    def __init__(self):
        super(MeanRadialError, self).__init__()

    def forward(self, y_true, y_pred):
        delta_x = y_pred[:, :, 0] - y_true[:, :, 0]
        delta_y = y_pred[:, :, 1] - y_true[:, :, 1]

        loss = torch.sqrt(delta_x**2 + delta_y**2).mean()
        return loss
