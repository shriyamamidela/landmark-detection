import torch
import torch.nn.functional as F


def huber_landmark_loss(pred_landmarks, gt_landmarks, delta=1.0, reduction='mean'):
    """
    pred_landmarks: [B, N, 2]
    gt_landmarks:   [B, N, 2]
    delta: threshold for Huber loss
    """
    diff = pred_landmarks - gt_landmarks          # [B, N, 2]
    abs_diff = diff.abs()

    # quadratic term for |r| <= delta
    quadratic = torch.minimum(abs_diff, torch.tensor(delta, device=diff.device))

    # Huber loss element: 0.5*q^2 + delta*(|r|-q)
    loss_elem = 0.5 * (quadratic ** 2) + delta * (abs_diff - quadratic)

    # sum over coordinate axis (x,y)
    loss_per_point = loss_elem.sum(dim=-1)  # [B, N]

    if reduction == 'mean':
        return loss_per_point.mean()
    elif reduction == 'sum':
        return loss_per_point.sum()
    else:
        return loss_per_point  # return [B, N]
