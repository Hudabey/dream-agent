"""
Encoder: Maps observations (images) to latent representations.
DreamerV3 uses a CNN encoder for visual inputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ConvEncoder(nn.Module):
    """
    CNN Encoder for visual observations.
    Takes (B, C, H, W) images and outputs (B, embed_dim) embeddings.

    Architecture follows DreamerV3: 4 conv layers with increasing channels,
    kernel size 4, stride 2, SiLU activation.
    """

    def __init__(
        self,
        in_channels: int = 3,
        depth: int = 48,  # Base channel multiplier
        embed_dim: int = 1024,
        input_size: Tuple[int, int] = (64, 64),
    ):
        super().__init__()
        self.input_size = input_size
        self.embed_dim = embed_dim

        # DreamerV3 style: channels = depth * 2^i for layer i
        channels = [in_channels, depth, 2*depth, 4*depth, 8*depth]

        layers = []
        for i in range(4):
            layers.extend([
                nn.Conv2d(channels[i], channels[i+1], kernel_size=4, stride=2, padding=1),
                nn.SiLU(),  # DreamerV3 uses SiLU (Swish) activation
            ])

        self.conv = nn.Sequential(*layers)

        # Calculate output size after convolutions
        # Each conv with k=4, s=2, p=1 halves the spatial dims
        h, w = input_size
        for _ in range(4):
            h, w = h // 2, w // 2
        conv_out_size = channels[-1] * h * w

        # Project to embedding dimension
        self.fc = nn.Linear(conv_out_size, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor, values in [0, 1]
        Returns:
            (B, embed_dim) embedding
        """
        # Normalize to [-0.5, 0.5] as in DreamerV3
        x = x - 0.5

        # Conv features
        h = self.conv(x)
        h = h.reshape(h.size(0), -1)

        # Project to embedding
        embed = self.fc(h)
        return embed


class MultiEncoder(nn.Module):
    """
    Encoder that can handle multiple input modalities.
    For now, just wraps ConvEncoder for images.
    """

    def __init__(
        self,
        image_channels: int = 3,
        depth: int = 48,
        embed_dim: int = 1024,
        input_size: Tuple[int, int] = (64, 64),
    ):
        super().__init__()
        self.image_encoder = ConvEncoder(
            in_channels=image_channels,
            depth=depth,
            embed_dim=embed_dim,
            input_size=input_size,
        )
        self.embed_dim = embed_dim

    def forward(self, obs: dict) -> torch.Tensor:
        """
        Args:
            obs: Dictionary with 'image' key containing (B, C, H, W) tensor
        Returns:
            (B, embed_dim) embedding
        """
        return self.image_encoder(obs['image'])
