#!/usr/bin/env python3
"""
Simple Demo - Load checkpoint and visualize agent playing.
"""

import gradio as gr
import torch
import numpy as np
import cv2
from pathlib import Path

from src.agent import DreamerAgent, DreamerConfig
from src.envs import ProcgenWrapper


class DemoState:
    def __init__(self):
        self.agent = None
        self.env = None
        self.obs = None
        self.prev_action = None
        self.step_count = 0
        self.episode_reward = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")

    def load(self, checkpoint_path, game='coinrun'):
        print(f"Loading {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        config = ckpt.get('config', DreamerConfig(
            depth=32, embed_dim=512, deter_dim=256,
            stoch_dim=8, stoch_classes=8, hidden_dim=256
        ))

        self.agent = DreamerAgent(config).to(self.device)
        self.agent.load_state_dict(ckpt['agent'])
        self.agent.eval()

        self.env = ProcgenWrapper(game=game, num_envs=1, num_levels=0)
        self.reset()
        return f"Loaded! Playing {game}"

    def reset(self):
        self.obs = self.env.reset()
        self.obs = {k: v.to(self.device) for k, v in self.obs.items()}
        self.prev_action = torch.zeros(1, dtype=torch.long, device=self.device)
        self.agent.reset(1, self.device)
        self.step_count = 0
        self.episode_reward = 0

    def step(self):
        if self.agent is None:
            empty = np.zeros((256, 256, 3), dtype=np.uint8)
            return empty, empty, 0.0, "Load a checkpoint first!"

        with torch.no_grad():
            action = self.agent.act(self.obs, self.prev_action)
            value = self.agent.get_value().item()
            dreams, rewards, values = self.agent.imagine_future(horizon=5)

        next_obs, reward, done, _ = self.env.step(action)
        next_obs = {k: v.to(self.device) for k, v in next_obs.items()}

        self.step_count += 1
        self.episode_reward += reward.item()

        # Visualize
        reality = self.obs['image'][0].permute(1,2,0).cpu().numpy()
        reality = (np.clip(reality, 0, 1) * 255).astype(np.uint8)
        reality = cv2.resize(reality, (256, 256), interpolation=cv2.INTER_NEAREST)

        dream = dreams[0, 0].permute(1,2,0).cpu().numpy()
        dream = (np.clip(dream, 0, 1) * 255).astype(np.uint8)
        dream = cv2.resize(dream, (256, 256), interpolation=cv2.INTER_NEAREST)

        # Add borders
        reality = cv2.copyMakeBorder(reality, 3,3,3,3, cv2.BORDER_CONSTANT, value=(0,255,0))
        dream = cv2.copyMakeBorder(dream, 3,3,3,3, cv2.BORDER_CONSTANT, value=(255,165,0))

        info = f"Step: {self.step_count} | Reward: {self.episode_reward:.1f} | Value: {value:.1f}"

        if done.item():
            info += " | EPISODE END!"
            self.reset()
        else:
            self.obs = next_obs
            self.prev_action = action

        return reality, dream, value, info


state = DemoState()

def load_model(ckpt, game):
    if ckpt is None:
        return "Upload a checkpoint!"
    return state.load(ckpt.name, game)

def do_step():
    return state.step()

def do_reset():
    if state.agent:
        state.reset()
        return state.step()
    empty = np.zeros((256, 256, 3), dtype=np.uint8)
    return empty, empty, 0.0, "Load first!"


with gr.Blocks(title="Dream Agent") as demo:
    gr.Markdown("# 🧠 Dream Agent - DreamerV3 Demo")
    gr.Markdown("Upload checkpoint, then click Step to watch the agent play and dream!")

    with gr.Row():
        ckpt_input = gr.File(label="Checkpoint (.pt)")
        game_input = gr.Dropdown(['coinrun','starpilot','bigfish'], value='coinrun', label="Game")
        load_btn = gr.Button("Load")
        status = gr.Textbox(label="Status")

    with gr.Row():
        reality_img = gr.Image(label="Reality (Green)")
        dream_img = gr.Image(label="Dream (Orange)")

    with gr.Row():
        value_out = gr.Number(label="Value V(s)")
        info_out = gr.Textbox(label="Info")

    with gr.Row():
        step_btn = gr.Button("▶ Step", variant="primary")
        reset_btn = gr.Button("🔄 Reset")

    load_btn.click(load_model, [ckpt_input, game_input], [status])
    step_btn.click(do_step, [], [reality_img, dream_img, value_out, info_out])
    reset_btn.click(do_reset, [], [reality_img, dream_img, value_out, info_out])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True)  # Auto-pick port
