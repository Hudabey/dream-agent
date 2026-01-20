#!/usr/bin/env python3
"""
Train DreamerV3 on Procgen.

Usage:
    python train.py --game coinrun --num_envs 16 --total_steps 1000000
"""

import argparse
import torch

from src.agent import DreamerConfig
from src.training import Trainer


def main():
    parser = argparse.ArgumentParser(description='Train DreamerV3 on Procgen')

    # Environment
    parser.add_argument('--game', type=str, default='coinrun',
                        choices=['bigfish', 'bossfight', 'caveflyer', 'chaser',
                                 'climber', 'coinrun', 'dodgeball', 'fruitbot',
                                 'heist', 'jumper', 'leaper', 'maze', 'miner',
                                 'ninja', 'plunder', 'starpilot'],
                        help='Procgen game to train on')
    parser.add_argument('--num_envs', type=int, default=16,
                        help='Number of parallel environments')
    parser.add_argument('--num_levels', type=int, default=200,
                        help='Number of training levels (0=unlimited)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')

    # Training
    parser.add_argument('--total_steps', type=int, default=1000000,
                        help='Total environment steps')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--collect_interval', type=int, default=100,
                        help='Environment steps between training')

    # Model
    parser.add_argument('--depth', type=int, default=48,
                        help='CNN depth multiplier')
    parser.add_argument('--deter_dim', type=int, default=4096,
                        help='Deterministic state dimension')
    parser.add_argument('--stoch_dim', type=int, default=32,
                        help='Stochastic state dimension')
    parser.add_argument('--hidden_dim', type=int, default=1024,
                        help='MLP hidden dimension')

    # Logging
    parser.add_argument('--log_dir', type=str, default='runs',
                        help='Tensorboard log directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--log_interval', type=int, default=1000,
                        help='Steps between logging')
    parser.add_argument('--save_interval', type=int, default=50000,
                        help='Steps between checkpoints')

    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')

    # Resume
    parser.add_argument('--resume', type=str, default=None,
                        help='Checkpoint to resume from')

    args = parser.parse_args()

    # Create config
    config = DreamerConfig(
        image_channels=3,
        image_size=(64, 64),
        action_dim=15,  # Procgen
        depth=args.depth,
        embed_dim=1024,
        deter_dim=args.deter_dim,
        stoch_dim=args.stoch_dim,
        stoch_classes=32,
        hidden_dim=args.hidden_dim,
    )

    # Create trainer
    trainer = Trainer(
        config=config,
        game=args.game,
        num_envs=args.num_envs,
        num_levels=args.num_levels,
        seed=args.seed,
        device=args.device,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    try:
        trainer.train(
            total_steps=args.total_steps,
            collect_interval=args.collect_interval,
            batch_size=args.batch_size,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
        )
    finally:
        trainer.close()


if __name__ == '__main__':
    main()
