"""
Trainer: Orchestrates the full DreamerV3 training loop.

Training consists of:
1. Collect experience in real environment
2. Train world model on real data
3. Imagine trajectories
4. Train actor/critic on dreams
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional
from pathlib import Path
from tqdm import tqdm
import time

from ..agent import DreamerAgent, DreamerConfig
from ..envs import make_procgen_env
from .replay_buffer import ReplayBuffer


class Trainer:
    """
    DreamerV3 Trainer.

    Handles:
    - Environment interaction
    - Replay buffer management
    - World model training
    - Actor-critic training
    - Logging and checkpointing
    """

    def __init__(
        self,
        config: DreamerConfig,
        game: str = 'coinrun',
        num_envs: int = 16,
        num_levels: int = 200,
        seed: int = 0,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        log_dir: str = 'runs',
        checkpoint_dir: str = 'checkpoints',
    ):
        self.config = config
        self.device = torch.device(device)
        self.game = game

        # Create directories
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Environment
        self.env = make_procgen_env(
            game=game,
            num_envs=num_envs,
            num_levels=num_levels,
            seed=seed,
        )

        # Agent
        self.agent = DreamerAgent(config).to(self.device)

        # Optimizers
        self.world_model_opt = optim.Adam(
            self.agent.world_model.parameters(),
            lr=config.world_model_lr,
        )
        self.actor_opt = optim.Adam(
            self.agent.actor.parameters(),
            lr=config.actor_lr,
        )
        self.critic_opt = optim.Adam(
            self.agent.critic.parameters(),
            lr=config.critic_lr,
        )

        # Replay buffer
        self.buffer = ReplayBuffer(
            capacity=1000000,
            sequence_length=50,
            min_sequences=1,
        )

        # Logging
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self.global_step = 0
        self.total_env_steps = 0

    def collect_experience(self, num_steps: int = 1000):
        """
        Collect experience from environment.

        Args:
            num_steps: Number of environment steps to collect
        """
        obs = self.env.reset()
        obs = {k: v.to(self.device) for k, v in obs.items()}

        prev_action = torch.zeros(self.env.num_envs, dtype=torch.long, device=self.device)
        self.agent.reset(self.env.num_envs, self.device)

        for _ in range(num_steps):
            # Get action
            with torch.no_grad():
                action = self.agent.act(obs, prev_action)

            # Step environment
            next_obs, rewards, dones, reset_mask, infos = self.env.step(action)
            next_obs = {k: v.to(self.device) for k, v in next_obs.items()}

            # Store in buffer
            self.buffer.add(obs, action, rewards, dones, reset_mask)

            # Reset agent state where needed
            for i in torch.where(reset_mask)[0]:
                # Agent internally handles state, just track for logging
                pass

            obs = next_obs
            prev_action = action
            self.total_env_steps += self.env.num_envs

        return self.env.get_episode_stats()

    def train_world_model(self, batch_size: int = 16) -> Dict[str, float]:
        """Train world model on sampled batch."""
        if not self.buffer.is_ready():
            return {}

        # Sample batch
        obs_seq, action_seq, reward_seq, continue_seq, reset_mask = self.buffer.sample(
            batch_size, self.device
        )

        # Compute loss
        losses = self.agent.train_world_model(
            obs_seq, action_seq, reward_seq, continue_seq, reset_mask
        )

        # Update
        self.world_model_opt.zero_grad()
        losses['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(self.agent.world_model.parameters(), 100.0)
        self.world_model_opt.step()

        return {k: v.item() for k, v in losses.items()}

    def train_actor_critic(self, batch_size: int = 16) -> Dict[str, float]:
        """Train actor and critic on imagined trajectories."""
        if not self.buffer.is_ready():
            return {}

        # Sample initial states from buffer
        obs_seq, action_seq, reward_seq, continue_seq, reset_mask = self.buffer.sample(
            batch_size, self.device
        )

        # Get initial states by processing first few steps
        with torch.no_grad():
            outputs = self.agent.world_model.observe_sequence(
                obs_seq, action_seq, reset_mask
            )
            # Use states at random timesteps as starting points
            T = obs_seq['image'].shape[1]
            t = torch.randint(0, T, (1,)).item()
            initial_states = {
                'deter': outputs['states']['deter'][:, t],
                'stoch': outputs['states']['stoch'][:, t],
            }

        # Train actor-critic
        losses = self.agent.train_actor_critic(initial_states)

        # Update actor
        self.actor_opt.zero_grad()
        losses['actor_loss'].backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.agent.actor.parameters(), 100.0)
        self.actor_opt.step()

        # Update critic
        self.critic_opt.zero_grad()
        losses['critic_loss'].backward()
        torch.nn.utils.clip_grad_norm_(self.agent.critic.parameters(), 100.0)
        self.critic_opt.step()

        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}

    def train(
        self,
        total_steps: int = 1000000,
        collect_interval: int = 100,
        train_ratio: int = 64,  # Train steps per env step
        batch_size: int = 16,
        log_interval: int = 1000,
        eval_interval: int = 10000,
        save_interval: int = 50000,
    ):
        """
        Main training loop.

        Args:
            total_steps: Total environment steps
            collect_interval: Env steps between training
            train_ratio: Training iterations per env step
            batch_size: Batch size for training
            log_interval: Steps between logging
            eval_interval: Steps between evaluation
            save_interval: Steps between checkpoints
        """
        print(f"Starting training on {self.game}")
        print(f"Device: {self.device}")
        print(f"Total steps: {total_steps}")

        # Initial data collection
        print("Collecting initial experience...")
        self.collect_experience(num_steps=5000)

        pbar = tqdm(total=total_steps, desc="Training")
        last_log_step = 0

        while self.total_env_steps < total_steps:
            # Collect experience
            ep_stats = self.collect_experience(num_steps=collect_interval)

            # Train
            num_train_steps = collect_interval * train_ratio

            world_model_losses = []
            actor_critic_losses = []

            for _ in range(num_train_steps):
                # World model
                wm_loss = self.train_world_model(batch_size)
                if wm_loss:
                    world_model_losses.append(wm_loss)

                # Actor-critic (less frequent)
                if self.global_step % 2 == 0:
                    ac_loss = self.train_actor_critic(batch_size)
                    if ac_loss:
                        actor_critic_losses.append(ac_loss)

                self.global_step += 1

            # Logging
            if self.total_env_steps - last_log_step >= log_interval:
                last_log_step = self.total_env_steps

                # Episode stats
                if ep_stats.get('num_episodes', 0) > 0:
                    self.writer.add_scalar('episode/reward', ep_stats['mean_reward'], self.total_env_steps)
                    self.writer.add_scalar('episode/length', ep_stats['mean_length'], self.total_env_steps)
                    tqdm.write(f"Step {self.total_env_steps}: reward={ep_stats['mean_reward']:.2f}")

                # World model losses
                if world_model_losses:
                    avg_wm = {k: sum(d[k] for d in world_model_losses) / len(world_model_losses)
                              for k in world_model_losses[0]}
                    for k, v in avg_wm.items():
                        self.writer.add_scalar(f'world_model/{k}', v, self.total_env_steps)

                # Actor-critic losses
                if actor_critic_losses:
                    avg_ac = {k: sum(d[k] for d in actor_critic_losses) / len(actor_critic_losses)
                              for k in actor_critic_losses[0]}
                    for k, v in avg_ac.items():
                        self.writer.add_scalar(f'actor_critic/{k}', v, self.total_env_steps)

            # Save checkpoint
            if self.total_env_steps % save_interval < collect_interval:
                self.save_checkpoint()

            pbar.update(collect_interval * self.env.num_envs)

        pbar.close()
        print("Training complete!")
        self.save_checkpoint()

    def save_checkpoint(self, path: Optional[str] = None):
        """Save checkpoint."""
        if path is None:
            path = self.checkpoint_dir / f"checkpoint_{self.total_env_steps}.pt"

        torch.save({
            'agent': self.agent.get_state_dict(),
            'world_model_opt': self.world_model_opt.state_dict(),
            'actor_opt': self.actor_opt.state_dict(),
            'critic_opt': self.critic_opt.state_dict(),
            'global_step': self.global_step,
            'total_env_steps': self.total_env_steps,
            'config': self.config,
        }, path)
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.agent.load_state_dict(checkpoint['agent'])
        self.world_model_opt.load_state_dict(checkpoint['world_model_opt'])
        self.actor_opt.load_state_dict(checkpoint['actor_opt'])
        self.critic_opt.load_state_dict(checkpoint['critic_opt'])
        self.global_step = checkpoint['global_step']
        self.total_env_steps = checkpoint['total_env_steps']
        print(f"Loaded checkpoint from {path}")

    def close(self):
        """Clean up."""
        self.env.close()
        self.writer.close()
