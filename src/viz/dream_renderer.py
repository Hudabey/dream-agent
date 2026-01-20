"""
Dream Renderer: Visualize what the agent imagines.

This is the core of the demo - showing the agent's "dreams"
side-by-side with reality.
"""

import torch
import numpy as np
import cv2
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert (C, H, W) or (B, C, H, W) tensor to numpy image(s)."""
    if tensor.dim() == 3:
        # (C, H, W) -> (H, W, C)
        img = tensor.permute(1, 2, 0).cpu().numpy()
    elif tensor.dim() == 4:
        # (B, C, H, W) -> (B, H, W, C)
        img = tensor.permute(0, 2, 3, 1).cpu().numpy()
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {tensor.dim()}D")

    # Ensure [0, 1] range
    img = np.clip(img, 0, 1)

    # Convert to uint8
    img = (img * 255).astype(np.uint8)

    return img


def add_border(img: np.ndarray, color: Tuple[int, int, int], thickness: int = 3) -> np.ndarray:
    """Add colored border to image."""
    h, w = img.shape[:2]
    bordered = img.copy()
    bordered[:thickness, :] = color
    bordered[-thickness:, :] = color
    bordered[:, :thickness] = color
    bordered[:, -thickness:] = color
    return bordered


def add_label(img: np.ndarray, text: str, position: str = 'top') -> np.ndarray:
    """Add text label to image."""
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    color = (255, 255, 255)

    # Get text size
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Create output with space for label
    label_height = text_h + 10
    if position == 'top':
        output = np.zeros((h + label_height, w, 3), dtype=np.uint8)
        output[:label_height] = (30, 30, 30)  # Dark background
        output[label_height:] = img

        # Center text
        x = (w - text_w) // 2
        y = text_h + 5
    else:  # bottom
        output = np.zeros((h + label_height, w, 3), dtype=np.uint8)
        output[:h] = img
        output[h:] = (30, 30, 30)

        x = (w - text_w) // 2
        y = h + text_h + 5

    cv2.putText(output, text, (x, y), font, font_scale, color, thickness)
    return output


class DreamRenderer:
    """
    Renders agent's dreams alongside reality.

    Creates side-by-side visualizations showing:
    - Left: Reality (actual game frames)
    - Center: Agent's dream (predicted frames)
    - Right: Difference/error visualization
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (64, 64),
        upscale_factor: int = 4,
    ):
        self.image_size = image_size
        self.upscale_factor = upscale_factor
        self.output_size = (image_size[0] * upscale_factor, image_size[1] * upscale_factor)

    def render_comparison(
        self,
        reality: torch.Tensor,
        dream: torch.Tensor,
        show_diff: bool = True,
    ) -> np.ndarray:
        """
        Render reality vs dream comparison.

        Args:
            reality: (C, H, W) real frame
            dream: (C, H, W) predicted frame
            show_diff: Whether to show difference panel

        Returns:
            Combined image as numpy array
        """
        # Convert to numpy
        reality_img = tensor_to_numpy(reality)
        dream_img = tensor_to_numpy(dream)

        # Upscale
        reality_img = cv2.resize(reality_img, self.output_size, interpolation=cv2.INTER_NEAREST)
        dream_img = cv2.resize(dream_img, self.output_size, interpolation=cv2.INTER_NEAREST)

        # Add borders
        reality_img = add_border(reality_img, (0, 255, 0), thickness=3)  # Green = reality
        dream_img = add_border(dream_img, (255, 165, 0), thickness=3)    # Orange = dream

        # Add labels
        reality_img = add_label(reality_img, "REALITY")
        dream_img = add_label(dream_img, "DREAM")

        panels = [reality_img, dream_img]

        if show_diff:
            # Compute difference
            diff = np.abs(reality_img[:-20].astype(float) - dream_img[:-20].astype(float))
            diff = (diff / diff.max() * 255).astype(np.uint8) if diff.max() > 0 else diff.astype(np.uint8)
            diff = add_border(diff, (255, 0, 0), thickness=3)  # Red = error
            diff = add_label(diff, "DIFFERENCE")
            panels.append(diff)

        # Combine horizontally
        combined = np.concatenate(panels, axis=1)

        return combined

    def render_dream_sequence(
        self,
        dreams: torch.Tensor,
        reality: Optional[torch.Tensor] = None,
        max_frames: int = 10,
    ) -> np.ndarray:
        """
        Render a sequence of dreams as a filmstrip.

        Args:
            dreams: (T, C, H, W) sequence of predicted frames
            reality: (T, C, H, W) optional real frames for comparison
            max_frames: Maximum frames to show

        Returns:
            Filmstrip image
        """
        T = min(dreams.shape[0], max_frames)

        frames = []
        for t in range(T):
            dream_img = tensor_to_numpy(dreams[t])
            dream_img = cv2.resize(dream_img, self.output_size, interpolation=cv2.INTER_NEAREST)

            # Add fade effect for later frames (uncertainty)
            alpha = 1.0 - 0.3 * (t / T)
            dream_img = (dream_img * alpha).astype(np.uint8)

            # Add frame number
            cv2.putText(dream_img, f"t+{t+1}", (5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            frames.append(dream_img)

        # Combine horizontally with small gaps
        gap = np.ones((self.output_size[1], 2, 3), dtype=np.uint8) * 50
        combined = frames[0]
        for frame in frames[1:]:
            combined = np.concatenate([combined, gap, frame], axis=1)

        return combined

    def create_video_frame(
        self,
        reality: torch.Tensor,
        dreams: torch.Tensor,
        value: float,
        predicted_rewards: torch.Tensor,
        step: int,
    ) -> np.ndarray:
        """
        Create a complete video frame for the demo.

        Layout:
        ┌─────────────┬─────────────┬─────────────┐
        │   REALITY   │    DREAM    │ VALUE MAP   │
        ├─────────────┴─────────────┴─────────────┤
        │        DREAM SEQUENCE (filmstrip)        │
        ├─────────────────────────────────────────┤
        │              METRICS                     │
        └─────────────────────────────────────────┘
        """
        # Top row: Reality, Dream (t+1), Value
        reality_img = tensor_to_numpy(reality)
        dream_img = tensor_to_numpy(dreams[0]) if dreams.shape[0] > 0 else reality_img.copy()

        reality_img = cv2.resize(reality_img, self.output_size, interpolation=cv2.INTER_NEAREST)
        dream_img = cv2.resize(dream_img, self.output_size, interpolation=cv2.INTER_NEAREST)

        reality_img = add_border(reality_img, (0, 255, 0), thickness=3)
        dream_img = add_border(dream_img, (255, 165, 0), thickness=3)

        reality_img = add_label(reality_img, "REALITY")
        dream_img = add_label(dream_img, f"DREAM (t+1)")

        # Value visualization (simple colored box for now)
        value_img = self._create_value_display(value)
        value_img = add_label(value_img, f"V(s) = {value:.1f}")

        top_row = np.concatenate([reality_img, dream_img, value_img], axis=1)

        # Middle row: Dream sequence
        filmstrip = self.render_dream_sequence(dreams, max_frames=8)

        # Make filmstrip same width as top row
        target_width = top_row.shape[1]
        filmstrip_resized = cv2.resize(filmstrip, (target_width, self.output_size[1]))
        filmstrip_resized = add_label(filmstrip_resized, "IMAGINED FUTURE")

        # Bottom row: Metrics
        metrics_img = self._create_metrics_display(
            step, value, predicted_rewards, target_width
        )

        # Combine vertically
        combined = np.concatenate([top_row, filmstrip_resized, metrics_img], axis=0)

        return combined

    def _create_value_display(self, value: float) -> np.ndarray:
        """Create a colored display showing the value."""
        img = np.zeros((self.output_size[1], self.output_size[0], 3), dtype=np.uint8)

        # Color based on value (red = low, green = high)
        # Normalize assuming value range roughly [-10, 100]
        normalized = np.clip((value + 10) / 110, 0, 1)

        # Gradient from red to green
        red = int(255 * (1 - normalized))
        green = int(255 * normalized)
        color = (0, green, red)  # BGR

        # Fill with gradient
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                # Radial gradient from center
                cx, cy = img.shape[1] // 2, img.shape[0] // 2
                dist = np.sqrt((i - cy)**2 + (j - cx)**2)
                max_dist = np.sqrt(cx**2 + cy**2)
                factor = 1 - 0.5 * (dist / max_dist)
                img[i, j] = tuple(int(c * factor) for c in color)

        img = add_border(img, (100, 100, 100), thickness=3)
        return img

    def _create_metrics_display(
        self,
        step: int,
        value: float,
        predicted_rewards: torch.Tensor,
        width: int,
    ) -> np.ndarray:
        """Create metrics display panel."""
        height = 60
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:] = (30, 30, 30)  # Dark background

        # Text
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 255, 255)

        texts = [
            f"Step: {step}",
            f"Value: {value:.2f}",
            f"Pred Reward Sum: {predicted_rewards.sum().item():.2f}",
        ]

        x = 10
        for text in texts:
            cv2.putText(img, text, (x, 35), font, 0.5, color, 1)
            x += 200

        return img
