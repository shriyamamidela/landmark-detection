# network/token_encoder.py
import torch.nn as nn
import torch

class TokenEncoder(nn.Module):
    """
    Encodes concatenated arc tokens into a conditioning vector.
    Input: list or tensor of shape (B, token_dim)
    """
    def __init__(self, input_dim: int, hidden: int = 256, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, tokens: torch.Tensor):
        # tokens shape (B, input_dim)
        return self.net(tokens)
