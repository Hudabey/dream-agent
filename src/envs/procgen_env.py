"""
Procgen Environment Wrapper.

Procgen is a suite of procedurally-generated games for RL research.
Key feature: train on some levels, test on unseen levels (generalization!).

Games include: bigfish, bossfight, caveflyer, chaser, climber, coinrun,
dodgeball, fruitbot, heist, jumper, leaper, maze, miner, ninja, plunder,
starpilot
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional, List
import gymnasium as gym

try:
    import procgen
    PROCGEN_AVAILABLE = True
except ImportError:
    PROCGEN_AVAILABLE = False
    print("Warning: procgen not installed. Install with: pip install procgen")


class ProcgenWrapper:
    """
    Wrapper for Procgen environments.

    Features:
    - Handles observation preprocessing (resize, normalize)
    - Supports parallel environments for faster data collection
    - Separates train/test levels for generalization experiments
    """

    # Available Procgen games
    GAMES = [
        'bigfish', 'bossfight', 'caveflyer', 'chaser', 'climber', 'coinrun',
        'dodgeball', 'fruitbot', 'heist', 'jumper', 'leaper', 'maze',
        'miner', 'ninja', 'plunder', 'starpilot'
    ]

    def __init__(
        self,
        game: str = 'coinrun',
        num_envs: int = 1,
        num_levels: int = 200,           # Train on 200 levels
        start_level: int = 0,
        distribution_mode: str = 'easy',  # 'easy', 'hard', 'extreme'
        image_size: Tuple[int, int] = (64, 64),
        use_backgrounds: bool = True,
        restrict_themes: bool = False,
        rand_seed: int = 0,
    ):
        """
        Args:
            game: Which Procgen game to play
            num_envs: Number of parallel environments
            num_levels: Number of unique levels (0 = unlimited)
            start_level: Starting level seed
            distribution_mode: Difficulty setting
            image_size: Output image size
            use_backgrounds: Use diverse backgrounds
            restrict_themes: Restrict visual themes (easier generalization)
            rand_seed: Random seed
        """
        if not PROCGEN_AVAILABLE:
            raise ImportError("procgen not installed. Run: pip install procgen")

        assert game in self.GAMES, f"Unknown game: {game}. Available: {self.GAMES}"

        self.game = game
        self.num_envs = num_envs
        self.image_size = image_size

        # Create vectorized environment
        self.env = procgen.ProcgenEnv(
            env_name=game,
            num_envs=num_envs,
            num_levels=num_levels,
            start_level=start_level,
            distribution_mode=distribution_mode,
            use_backgrounds=use_backgrounds,
            restrict_themes=restrict_themes,
            rand_seed=rand_seed,
        )

        # Action and observation spaces
        self.action_dim = self.env.action_space.n  # 15 discrete actions
        self.observation_shape = (*image_size, 3)

    def reset(self) -> Dict[str, torch.Tensor]:
        """
        Reset all environments.

        Returns:
            obs: {'image': (num_envs, C, H, W) tensor}
        """
        obs = self.env.reset()
        # Handle dict return (procgen-mirror returns {'rgb': array})
        if isinstance(obs, dict):
            obs = obs.get('rgb', obs.get('image', list(obs.values())[0]))
        return self._process_obs(obs)

    def step(
        self,
        actions: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict]:
        """
        Take a step in all environments.

        Args:
            actions: (num_envs,) action tensor

        Returns:
            obs: {'image': (num_envs, C, H, W)}
            rewards: (num_envs,)
            dones: (num_envs,)
            infos: dict with additional info
        """
        # Convert to numpy
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()

        obs, rewards, dones, infos = self.env.step(actions)

        # Handle dict return (procgen-mirror returns {'rgb': array})
        if isinstance(obs, dict):
            obs = obs.get('rgb', obs.get('image', list(obs.values())[0]))
        obs = self._process_obs(obs)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool)

        return obs, rewards, dones, infos

    def _process_obs(self, obs: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        Process raw observations.

        Args:
            obs: (num_envs, H, W, C) uint8 numpy array

        Returns:
            {'image': (num_envs, C, H, W) float32 tensor in [0, 1]}
        """
        # Resize if needed
        if obs.shape[1:3] != self.image_size:
            import cv2
            resized = []
            for i in range(obs.shape[0]):
                img = cv2.resize(obs[i], self.image_size, interpolation=cv2.INTER_AREA)
                resized.append(img)
            obs = np.stack(resized)

        # Convert to tensor: (B, H, W, C) -> (B, C, H, W)
        obs = torch.tensor(obs, dtype=torch.float32)
        obs = obs.permute(0, 3, 1, 2)

        # Normalize to [0, 1]
        obs = obs / 255.0

        return {'image': obs}

    def render(self) -> np.ndarray:
        """Render the first environment."""
        return self.env.render()

    def close(self):
        """Close all environments."""
        self.env.close()

    @property
    def num_actions(self) -> int:
        """Number of possible actions."""
        return self.action_dim


class ProcgenVecEnv:
    """
    Vectorized Procgen environment with episode management.

    Handles automatic resets and tracks episode statistics.
    """

    def __init__(
        self,
        game: str = 'coinrun',
        num_envs: int = 16,
        num_levels: int = 200,
        start_level: int = 0,
        distribution_mode: str = 'easy',
        image_size: Tuple[int, int] = (64, 64),
    ):
        self.env = ProcgenWrapper(
            game=game,
            num_envs=num_envs,
            num_levels=num_levels,
            start_level=start_level,
            distribution_mode=distribution_mode,
            image_size=image_size,
        )

        self.num_envs = num_envs
        self.action_dim = self.env.action_dim
        self.image_size = image_size

        # Episode tracking
        self.episode_rewards = np.zeros(num_envs)
        self.episode_lengths = np.zeros(num_envs)
        self.completed_episodes = []

    def reset(self) -> Dict[str, torch.Tensor]:
        """Reset and return initial observation."""
        self.episode_rewards = np.zeros(self.num_envs)
        self.episode_lengths = np.zeros(self.num_envs)
        return self.env.reset()

    def step(
        self,
        actions: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Step with automatic reset handling.

        Returns:
            obs, rewards, dones, reset_mask, infos

            reset_mask: True where episode reset (for RSSM state reset)
        """
        obs, rewards, dones, infos = self.env.step(actions)

        # Track episode stats
        self.episode_rewards += rewards.numpy()
        self.episode_lengths += 1

        # Record completed episodes
        for i in np.where(dones.numpy())[0]:
            self.completed_episodes.append({
                'reward': self.episode_rewards[i],
                'length': self.episode_lengths[i],
            })
            self.episode_rewards[i] = 0
            self.episode_lengths[i] = 0

        # Reset mask (where new episodes started)
        reset_mask = dones.clone()

        return obs, rewards, dones, reset_mask, infos

    def get_episode_stats(self) -> Dict[str, float]:
        """Get statistics from completed episodes."""
        if not self.completed_episodes:
            return {'mean_reward': 0.0, 'mean_length': 0.0}

        rewards = [ep['reward'] for ep in self.completed_episodes]
        lengths = [ep['length'] for ep in self.completed_episodes]

        stats = {
            'mean_reward': np.mean(rewards),
            'mean_length': np.mean(lengths),
            'num_episodes': len(self.completed_episodes),
        }

        # Clear completed episodes
        self.completed_episodes = []

        return stats

    def close(self):
        """Close the environment."""
        self.env.close()


def make_procgen_env(
    game: str = 'coinrun',
    num_envs: int = 16,
    num_levels: int = 200,
    seed: int = 0,
) -> ProcgenVecEnv:
    """
    Create a Procgen environment.

    Args:
        game: Game name (coinrun, starpilot, etc.)
        num_envs: Number of parallel environments
        num_levels: Number of training levels (0 = unlimited)
        seed: Random seed

    Returns:
        ProcgenVecEnv instance
    """
    return ProcgenVecEnv(
        game=game,
        num_envs=num_envs,
        num_levels=num_levels,
        start_level=seed,
    )
