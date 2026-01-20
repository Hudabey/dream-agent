"""
Gradio Demo App: Interactive visualization of DreamerV3.

This is the interview-ready demo that shows:
1. Agent playing Procgen games
2. What the agent "dreams" (imagined futures)
3. Value function visualization
4. Dream vs reality comparison
"""

import gradio as gr
import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple
import time

from ..agent import DreamerAgent, DreamerConfig
from ..envs import ProcgenWrapper
from ..viz import DreamRenderer, ValueHeatmap


class DemoState:
    """Manages demo state across Gradio callbacks."""

    def __init__(self):
        self.agent: Optional[DreamerAgent] = None
        self.env: Optional[ProcgenWrapper] = None
        self.renderer: Optional[DreamRenderer] = None
        self.value_viz: Optional[ValueHeatmap] = None

        self.current_obs = None
        self.prev_action = None
        self.step_count = 0
        self.episode_reward = 0
        self.is_running = False

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_checkpoint(self, checkpoint_path: str, game: str = 'coinrun'):
        """Load agent from checkpoint."""
        print(f"Loading checkpoint from {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = checkpoint.get('config', DreamerConfig())

        # Create agent
        self.agent = DreamerAgent(config).to(self.device)
        self.agent.load_state_dict(checkpoint['agent'])
        self.agent.eval()

        # Create environment
        self.env = ProcgenWrapper(
            game=game,
            num_envs=1,
            num_levels=0,  # Unlimited for demo
            distribution_mode='easy',
        )

        # Visualization
        self.renderer = DreamRenderer(upscale_factor=4)
        self.value_viz = ValueHeatmap()

        # Reset
        self.reset_episode()

        print(f"Loaded! Playing {game}")
        return f"Loaded checkpoint. Playing {game}"

    def reset_episode(self):
        """Reset to start new episode."""
        if self.env is None:
            return

        self.current_obs = self.env.reset()
        self.current_obs = {k: v.to(self.device) for k, v in self.current_obs.items()}
        self.prev_action = torch.zeros(1, dtype=torch.long, device=self.device)
        self.agent.reset(1, self.device)
        self.step_count = 0
        self.episode_reward = 0

    def step(self) -> Tuple[np.ndarray, np.ndarray, float, str]:
        """
        Take one step and return visualizations.

        Returns:
            reality_frame: Current game frame
            dream_frame: Agent's imagination
            value: Current value estimate
            info: Text info
        """
        if self.agent is None or self.env is None:
            empty = np.zeros((256, 256, 3), dtype=np.uint8)
            return empty, empty, 0.0, "Load a checkpoint first!"

        # Get action from agent
        with torch.no_grad():
            action = self.agent.act(self.current_obs, self.prev_action, deterministic=False)

            # Get value
            value = self.agent.get_value().item()

            # Imagine future
            dreams, pred_rewards, pred_values = self.agent.imagine_future(horizon=10)

        # Step environment
        next_obs, reward, done, _ = self.env.step(action)
        next_obs = {k: v.to(self.device) for k, v in next_obs.items()}

        self.step_count += 1
        self.episode_reward += reward.item()

        # Prepare visualizations
        reality_img = self.current_obs['image'][0]  # (C, H, W)
        dream_img = dreams[0, 0]  # First env, first predicted frame

        # Apply value overlay to reality
        reality_np = self.renderer._process_frame(reality_img)
        reality_np = self.value_viz.overlay_on_frame(reality_np, value)

        # Dream visualization
        dream_np = self.renderer._process_frame(dream_img)
        dream_np = cv2.resize(dream_np, (256, 256), interpolation=cv2.INTER_NEAREST)

        # Add borders
        reality_np = cv2.copyMakeBorder(reality_np, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(0, 255, 0))
        dream_np = cv2.copyMakeBorder(dream_np, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(255, 165, 0))

        # Create dream filmstrip
        filmstrip = self._create_filmstrip(dreams[0])

        # Info text
        info = f"Step: {self.step_count} | Reward: {self.episode_reward:.1f} | Value: {value:.2f}"

        # Handle episode end
        if done.item():
            info += f" | EPISODE DONE!"
            self.reset_episode()
        else:
            self.current_obs = next_obs
            self.prev_action = action

        return reality_np, dream_np, filmstrip, info

    def _process_frame(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to displayable numpy array."""
        img = tensor.permute(1, 2, 0).cpu().numpy()
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_NEAREST)
        return img

    def _create_filmstrip(self, dreams: torch.Tensor, max_frames: int = 8) -> np.ndarray:
        """Create filmstrip of dream frames."""
        T = min(dreams.shape[0], max_frames)
        frame_size = 64

        filmstrip = np.zeros((frame_size + 20, T * (frame_size + 4) - 4, 3), dtype=np.uint8)

        for t in range(T):
            frame = dreams[t].permute(1, 2, 0).cpu().numpy()
            frame = np.clip(frame, 0, 1)
            frame = (frame * 255).astype(np.uint8)

            # Fade effect
            alpha = 1.0 - 0.3 * (t / T)
            frame = (frame * alpha).astype(np.uint8)

            x_start = t * (frame_size + 4)
            filmstrip[20:20+frame_size, x_start:x_start+frame_size] = frame

            # Label
            cv2.putText(filmstrip, f"t+{t+1}", (x_start + 5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return filmstrip


# Global state
state = DemoState()


def load_model(checkpoint_file, game):
    """Gradio callback for loading model."""
    if checkpoint_file is None:
        return "Please upload a checkpoint file."
    return state.load_checkpoint(checkpoint_file.name, game)


def step_agent():
    """Gradio callback for stepping."""
    reality, dream, filmstrip, info = state.step()
    return reality, dream, filmstrip, info


def reset_env():
    """Gradio callback for reset."""
    state.reset_episode()
    if state.agent is not None:
        reality, dream, filmstrip, info = state.step()
        return reality, dream, filmstrip, "Episode reset!"
    empty = np.zeros((256, 256, 3), dtype=np.uint8)
    return empty, empty, empty, "Load a checkpoint first!"


def create_demo():
    """Create the Gradio demo interface."""

    with gr.Blocks(
        title="Dream Agent - DreamerV3 Visualization",
        theme=gr.themes.Base(
            primary_hue="purple",
            secondary_hue="blue",
        ),
        css="""
        .container { max-width: 1200px; margin: auto; }
        .header { text-align: center; margin-bottom: 20px; }
        .frame-container { display: flex; justify-content: center; gap: 20px; }
        """
    ) as demo:

        gr.Markdown("""
        # 🧠 Dream Agent - DreamerV3 Visualization

        **See what the AI imagines before it acts.**

        This demo shows a DreamerV3 agent playing Procgen games.
        - **Left (Green border)**: Reality - what's actually happening
        - **Right (Orange border)**: Dream - what the agent *thinks* will happen
        - **Filmstrip**: The next 8 imagined frames
        - **Value**: How "good" the agent thinks the current state is (green = good, red = bad)
        """)

        with gr.Row():
            with gr.Column(scale=1):
                checkpoint_input = gr.File(label="Upload Checkpoint (.pt)")
                game_dropdown = gr.Dropdown(
                    choices=['coinrun', 'starpilot', 'bigfish', 'fruitbot', 'maze'],
                    value='coinrun',
                    label="Game"
                )
                load_btn = gr.Button("🚀 Load Model", variant="primary")
                status_text = gr.Textbox(label="Status", interactive=False)

            with gr.Column(scale=2):
                with gr.Row():
                    reality_display = gr.Image(label="🎮 Reality", height=280)
                    dream_display = gr.Image(label="💭 Dream (t+1)", height=280)

        filmstrip_display = gr.Image(label="🎬 Dream Sequence (Imagined Future)", height=120)

        with gr.Row():
            info_display = gr.Textbox(label="Info", interactive=False)

        with gr.Row():
            step_btn = gr.Button("▶ Step", variant="primary")
            reset_btn = gr.Button("🔄 Reset Episode")
            auto_btn = gr.Button("⏩ Auto Play (10 steps)")

        # Event handlers
        load_btn.click(
            load_model,
            inputs=[checkpoint_input, game_dropdown],
            outputs=[status_text]
        )

        step_btn.click(
            step_agent,
            outputs=[reality_display, dream_display, filmstrip_display, info_display]
        )

        reset_btn.click(
            reset_env,
            outputs=[reality_display, dream_display, filmstrip_display, info_display]
        )

        def auto_play():
            """Run 10 steps automatically."""
            results = None
            for _ in range(10):
                results = state.step()
                time.sleep(0.1)
            return results

        auto_btn.click(
            auto_play,
            outputs=[reality_display, dream_display, filmstrip_display, info_display]
        )

        gr.Markdown("""
        ---
        ### How it works

        **DreamerV3** is a model-based reinforcement learning algorithm that:

        1. **Learns a world model** - An internal simulation of how the environment works
        2. **Dreams** - Imagines thousands of possible futures without actually acting
        3. **Learns from dreams** - Trains its policy entirely in imagination

        The **value function** (V) estimates expected future reward. When V is high (green),
        the agent thinks it's in a good position. When V is low (red), danger is near!

        ---
        *Built for Anthropic interview demonstration*
        """)

    return demo


def main():
    """Launch the demo."""
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,  # Creates public link
    )


if __name__ == "__main__":
    main()
