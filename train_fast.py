#!/usr/bin/env python3
"""
Fast DreamerV3 Training - Optimized for quick results.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import time
from pathlib import Path
from tqdm import tqdm

from src.agent import DreamerAgent, DreamerConfig
from src.envs import make_procgen_env
from src.training import ReplayBuffer


def main():
    # Force flush prints
    print = lambda *args, **kwargs: (builtins_print(*args, **kwargs, flush=True))
    import builtins
    builtins_print = builtins.print
    builtins.print = print

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Small, fast config
    config = DreamerConfig(
        image_channels=3,
        image_size=(64, 64),
        action_dim=15,
        depth=32,          # Smaller
        embed_dim=512,     # Smaller
        deter_dim=256,     # Smaller
        stoch_dim=8,       # Smaller
        stoch_classes=8,   # Smaller
        hidden_dim=256,    # Smaller
    )

    print("Creating agent...")
    agent = DreamerAgent(config).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in agent.parameters())
    print(f"Total parameters: {total_params:,}")

    print("Creating environment...")
    env = make_procgen_env(game='coinrun', num_envs=8, num_levels=50)

    print("Creating optimizers...")
    world_opt = optim.Adam(agent.world_model.parameters(), lr=3e-4)
    actor_opt = optim.Adam(agent.actor.parameters(), lr=1e-4)
    critic_opt = optim.Adam(agent.critic.parameters(), lr=1e-4)

    buffer = ReplayBuffer(capacity=100000, sequence_length=32, min_sequences=1)

    # Training settings
    total_steps = 50000
    collect_steps = 256
    train_steps = 8  # Much smaller!
    batch_size = 8
    save_every = 10000

    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)

    print(f"\nStarting training for {total_steps} steps...")
    print(f"Collect {collect_steps} steps, train {train_steps} times per cycle\n")

    # Initial collection
    obs = env.reset()
    obs = {k: v.to(device) for k, v in obs.items()}
    prev_action = torch.zeros(8, dtype=torch.long, device=device)
    agent.reset(8, device)

    total_env_steps = 0
    episode_rewards = []
    current_rewards = torch.zeros(8)

    pbar = tqdm(total=total_steps, desc="Training")
    start_time = time.time()

    while total_env_steps < total_steps:
        # === COLLECT ===
        for _ in range(collect_steps // 8):
            with torch.no_grad():
                action = agent.act(obs, prev_action)

            next_obs, reward, done, reset_mask, _ = env.step(action)
            next_obs = {k: v.to(device) for k, v in next_obs.items()}

            buffer.add(obs, action, reward, done, reset_mask)
            current_rewards += reward

            # Track completed episodes
            for i in torch.where(done)[0]:
                episode_rewards.append(current_rewards[i].item())
                current_rewards[i] = 0

            obs = next_obs
            prev_action = action
            total_env_steps += 8

        # === TRAIN ===
        if buffer.is_ready():
            for _ in range(train_steps):
                # World model
                batch = buffer.sample(batch_size, device)
                obs_seq, action_seq, reward_seq, continue_seq, reset_mask = batch

                losses = agent.world_model.compute_loss(
                    obs_seq, action_seq, reward_seq, continue_seq, reset_mask
                )

                world_opt.zero_grad()
                losses['total_loss'].backward()
                torch.nn.utils.clip_grad_norm_(agent.world_model.parameters(), 100)
                world_opt.step()

                # Actor-critic (every other step)
                if _ % 2 == 0:
                    with torch.no_grad():
                        outputs = agent.world_model.observe_sequence(obs_seq, action_seq, reset_mask)
                        t = torch.randint(0, obs_seq['image'].shape[1], (1,)).item()
                        init_state = {
                            'deter': outputs['states']['deter'][:, t],
                            'stoch': outputs['states']['stoch'][:, t],
                        }

                    ac_losses = agent.train_actor_critic(init_state)

                    actor_opt.zero_grad()
                    ac_losses['actor_loss'].backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 100)
                    actor_opt.step()

                    critic_opt.zero_grad()
                    ac_losses['critic_loss'].backward()
                    torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 100)
                    critic_opt.step()

        # === LOG ===
        pbar.update(collect_steps)

        if episode_rewards:
            avg_reward = sum(episode_rewards[-10:]) / len(episode_rewards[-10:])
            elapsed = time.time() - start_time
            steps_per_sec = total_env_steps / elapsed
            pbar.set_postfix({
                'reward': f'{avg_reward:.1f}',
                'eps': len(episode_rewards),
                'sps': f'{steps_per_sec:.0f}'
            })

        # === SAVE ===
        if total_env_steps % save_every < collect_steps:
            path = checkpoint_dir / f'checkpoint_{total_env_steps}.pt'
            torch.save({
                'agent': agent.get_state_dict(),
                'step': total_env_steps,
                'config': config,
            }, path)
            tqdm.write(f"Saved checkpoint: {path}")

    pbar.close()

    # Final save
    final_path = checkpoint_dir / 'checkpoint_final.pt'
    torch.save({
        'agent': agent.get_state_dict(),
        'step': total_env_steps,
        'config': config,
    }, final_path)

    print(f"\nTraining complete!")
    print(f"Total episodes: {len(episode_rewards)}")
    print(f"Final avg reward: {sum(episode_rewards[-10:])/max(len(episode_rewards[-10:]),1):.1f}")
    print(f"Checkpoint saved: {final_path}")

    env.close()


if __name__ == '__main__':
    main()
