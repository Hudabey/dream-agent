"""
Decoder: Reconstructs observations from latent states.
Essential for visualizing what the agent "dreams".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ConvDecoder(nn.Module):
    """
    CNN Decoder for image reconstruction.
    Takes (B, state_dim) latent states and outputs (B, C, H, W) images.

    This allows us to visualize the agent's "dreams" - what it imagines
    will happen in the future.
    """

    def __init__(
        self,
        state_dim: int,  # deter_dim + stoch_dim * stoch_classes
        out_channels: int = 3,
        depth: int = 48,
        output_size: Tuple[int, int] = (64, 64),
    ):
        super().__init__()
        self.output_size = output_size
        self.out_channels = out_channels

        # Calculate initial spatial size (will be upsampled 4x)
        h, w = output_size
        self.init_h = h // 16
        self.init_w = w // 16
        self.init_channels = 8 * depth

        # Project state to initial feature map
        self.fc = nn.Linear(state_dim, self.init_channels * self.init_h * self.init_w)

        # Transpose convolutions (mirror of encoder)
        channels = [8*depth, 4*depth, 2*depth, depth, out_channels]

        layers = []
        for i in range(4):
            layers.extend([
                nn.ConvTranspose2d(
                    channels[i], channels[i+1],
                    kernel_size=4, stride=2, padding=1
                ),
                nn.SiLU() if i < 3 else nn.Identity(),  # No activation on final layer
            ])

        self.deconv = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (B, state_dim) or (B, T, state_dim) latent state
        Returns:
            (B, C, H, W) or (B, T, C, H, W) reconstructed image in [0, 1]
        """
        # Handle sequence dimension
        has_time = state.dim() == 3
        if has_time:
            B, T, D = state.shape
            state = state.reshape(B * T, D)

        # Project to initial feature map
        h = self.fc(state)
        h = h.reshape(-1, self.init_channels, self.init_h, self.init_w)

        # Upsample through transpose convs
        h = self.deconv(h)

        # Sigmoid to get values in [0, 1]
        out = torch.sigmoid(h)

        # Restore time dimension
        if has_time:
            out = out.reshape(B, T, self.out_channels, *self.output_size)

        return out


class MultiDecoder(nn.Module):
    """
    Decoder that can reconstruct multiple modalities.
    For now, just wraps ConvDecoder for images.
    """

    def __init__(
        self,
        state_dim: int,
        image_channels: int = 3,
        depth: int = 48,
        output_size: Tuple[int, int] = (64, 64),
    ):
        super().__init__()
        self.image_decoder = ConvDecoder(
            state_dim=state_dim,
            out_channels=image_channels,
            depth=depth,
            output_size=output_size,
        )

    def forward(self, state: torch.Tensor) -> dict:
        """
        Args:
            state: (B, state_dim) or (B, T, state_dim) latent state
        Returns:
            Dictionary with 'image' key containing reconstructed images
        """
        return {'image': self.image_decoder(state)}
