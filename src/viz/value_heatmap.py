"""
Value Heatmap: Visualize the agent's value function.

Shows what states the agent thinks are valuable (green)
versus dangerous (red).
"""

import torch
import numpy as np
import cv2
from typing import Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class ValueHeatmap:
    """
    Creates value function heatmaps overlaid on game frames.

    The heatmap shows:
    - Green: High value (good states)
    - Yellow: Medium value
    - Red: Low value (dangerous states)
    """

    def __init__(
        self,
        colormap: str = 'RdYlGn',  # Red-Yellow-Green
        alpha: float = 0.4,         # Overlay transparency
        blur_kernel: int = 5,       # Smoothing
    ):
        self.colormap = plt.get_cmap(colormap)
        self.alpha = alpha
        self.blur_kernel = blur_kernel

    def create_heatmap(
        self,
        values: np.ndarray,
        size: Tuple[int, int],
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Create a heatmap from values.

        Args:
            values: Values to visualize (can be any shape)
            size: Output (height, width)
            normalize: Normalize to [0, 1]

        Returns:
            (H, W, 3) RGB heatmap
        """
        values = np.array(values).flatten()

        if normalize and values.max() != values.min():
            values = (values - values.min()) / (values.max() - values.min())
        else:
            values = np.clip(values, 0, 1)

        # Apply colormap
        colors = self.colormap(values)[:, :3]  # RGB only

        # Reshape to 2D if possible
        side = int(np.sqrt(len(values)))
        if side * side == len(values):
            heatmap = colors.reshape(side, side, 3)
        else:
            # Just create a horizontal bar
            heatmap = colors.reshape(1, -1, 3)

        # Resize
        heatmap = cv2.resize(
            (heatmap * 255).astype(np.uint8),
            size,
            interpolation=cv2.INTER_LINEAR
        )

        # Smooth
        if self.blur_kernel > 1:
            heatmap = cv2.GaussianBlur(heatmap, (self.blur_kernel, self.blur_kernel), 0)

        return heatmap

    def overlay_on_frame(
        self,
        frame: np.ndarray,
        value: float,
        value_range: Tuple[float, float] = (-10, 100),
    ) -> np.ndarray:
        """
        Overlay value visualization on a game frame.

        Creates a border glow effect based on value.

        Args:
            frame: (H, W, 3) game frame
            value: Scalar value for this state
            value_range: (min, max) for normalization

        Returns:
            Frame with value overlay
        """
        # Normalize value
        v_min, v_max = value_range
        normalized = np.clip((value - v_min) / (v_max - v_min), 0, 1)

        # Get color from colormap
        color = self.colormap(normalized)[:3]
        color_bgr = (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))

        # Create glow effect around border
        h, w = frame.shape[:2]
        output = frame.copy()

        # Draw glowing border
        glow_size = 10
        for i in range(glow_size):
            alpha = (glow_size - i) / glow_size * 0.5
            cv2.rectangle(
                output,
                (i, i),
                (w - 1 - i, h - 1 - i),
                color_bgr,
                thickness=1,
            )
            # Blend
            mask = np.zeros_like(output)
            cv2.rectangle(mask, (i, i), (w - 1 - i, h - 1 - i), color_bgr, thickness=1)
            output = cv2.addWeighted(output, 1 - alpha * 0.3, mask, alpha * 0.3, 0)

        return output

    def create_trajectory_heatmap(
        self,
        values: torch.Tensor,
        size: Tuple[int, int] = (200, 600),
    ) -> np.ndarray:
        """
        Create heatmap showing value predictions over time.

        Args:
            values: (T,) value predictions for trajectory
            size: Output (height, width)

        Returns:
            Horizontal heatmap showing value trajectory
        """
        values = values.cpu().numpy() if isinstance(values, torch.Tensor) else values
        T = len(values)

        # Normalize
        if values.max() != values.min():
            normalized = (values - values.min()) / (values.max() - values.min())
        else:
            normalized = np.ones_like(values) * 0.5

        # Create horizontal bar
        bar = np.zeros((50, T, 3), dtype=np.float32)
        for t in range(T):
            color = self.colormap(normalized[t])[:3]
            bar[:, t] = color

        # Resize
        bar = cv2.resize(
            (bar * 255).astype(np.uint8),
            size,
            interpolation=cv2.INTER_LINEAR
        )

        # Add value labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(bar, f"V={values[0]:.1f}", (5, 30), font, 0.4, (255, 255, 255), 1)
        cv2.putText(bar, f"V={values[-1]:.1f}", (size[0] - 60, 30), font, 0.4, (255, 255, 255), 1)

        return bar


class SpatialValueMap:
    """
    Creates spatial value maps by evaluating the value function
    at different "imagined" positions.

    This is more advanced - it shows what value the agent would
    assign to being at different locations on the screen.
    """

    def __init__(self, grid_size: int = 8):
        self.grid_size = grid_size
        self.colormap = plt.get_cmap('RdYlGn')

    def compute_spatial_values(
        self,
        agent,
        base_state: dict,
        world_model,
    ) -> np.ndarray:
        """
        Compute value estimates for a grid of imagined perturbations.

        This is an approximation - we perturb the latent state
        and see how value changes.

        Returns:
            (grid_size, grid_size) array of values
        """
        with torch.no_grad():
            state_vec = world_model.get_state_vector(base_state)

            # Base value
            base_value = agent.critic.predict(state_vec).item()

            # Perturb and evaluate
            values = np.zeros((self.grid_size, self.grid_size))
            state_dim = state_vec.shape[-1]

            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    # Create perturbation (this is a simplification)
                    perturbed = state_vec.clone()

                    # Add noise in a structured way
                    noise_scale = 0.1
                    perturbation = torch.randn_like(state_vec) * noise_scale
                    perturbation *= (i - self.grid_size // 2) / self.grid_size
                    perturbation *= (j - self.grid_size // 2) / self.grid_size

                    perturbed = perturbed + perturbation

                    value = agent.critic.predict(perturbed).item()
                    values[i, j] = value

        return values

    def render(
        self,
        values: np.ndarray,
        frame: np.ndarray,
        alpha: float = 0.3,
    ) -> np.ndarray:
        """
        Render spatial value map overlaid on frame.

        Args:
            values: (grid_size, grid_size) value array
            frame: (H, W, 3) game frame
            alpha: Overlay transparency

        Returns:
            Frame with value map overlay
        """
        h, w = frame.shape[:2]

        # Normalize values
        if values.max() != values.min():
            normalized = (values - values.min()) / (values.max() - values.min())
        else:
            normalized = np.ones_like(values) * 0.5

        # Create heatmap
        heatmap = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.float32)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                heatmap[i, j] = self.colormap(normalized[i, j])[:3]

        # Resize to frame size
        heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
        heatmap = (heatmap * 255).astype(np.uint8)

        # Overlay
        output = cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)

        return output
