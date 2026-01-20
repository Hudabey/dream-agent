# 🧠 Dream Agent

**DreamerV3 implementation with interpretable visualization for Procgen environments.**

> See what the AI imagines before it acts.

![Demo](dream_vs_reality.png)

## What is this?

This project implements **DreamerV3**, a state-of-the-art model-based reinforcement learning algorithm, trained on **Procgen** environments. The key feature is an interactive demo that visualizes:

1. **Reality** - What's actually happening in the game
2. **Dreams** - What the agent *imagines* will happen
3. **Value Function** - How "good" the agent thinks each state is

## Why DreamerV3?

Unlike traditional RL (like DQN or PPO), DreamerV3:

- **Learns a world model** - An internal simulation of how the environment works
- **Dreams** - Imagines thousands of possible futures without actually acting
- **Learns from dreams** - Trains its policy entirely in imagination

This is more sample-efficient and provides a window into the agent's "mind".

## Project Structure

```
dream-agent/
├── src/
│   ├── world_model/      # World model (encoder, decoder, RSSM)
│   ├── agent/            # Actor-critic and imagination
│   ├── training/         # Training loop and replay buffer
│   ├── envs/             # Procgen environment wrapper
│   ├── viz/              # Dream and value visualization
│   └── demo/             # Gradio interactive demo
├── configs/              # Training configurations
├── checkpoints/          # Saved models
├── train.py              # Training script
├── demo.py               # Demo launcher
└── requirements.txt
```

## Quick Start

### Installation

```bash
# Clone the repo
git clone https://github.com/hudabey/dream-agent.git
cd dream-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Train on CoinRun (easy mode)
python train.py --game coinrun --num_envs 16 --total_steps 1000000

# Train on StarPilot
python train.py --game starpilot --num_envs 16 --total_steps 2000000

# Resume from checkpoint
python train.py --resume checkpoints/checkpoint_500000.pt
```

### Demo

```bash
# Launch interactive demo
python demo.py --share

# Then upload a checkpoint and watch the agent dream!
```

## Key Components

### World Model

The world model learns to predict:
- **Next state** given current state and action
- **Reward** for each state
- **Episode termination**

```python
# Imagine 15 steps into the future
dreams, rewards, values = agent.imagine_future(horizon=15)
```

### RSSM (Recurrent State Space Model)

The core dynamics model uses:
- **Deterministic state (h)** - GRU hidden state for long-term memory
- **Stochastic state (z)** - Categorical latent for uncertainty

### Value Function

The critic estimates V(s) - expected future reward from state s:
- **High value (green)** - Agent thinks this is a good state
- **Low value (red)** - Danger ahead!

## Generalization

Procgen tests **generalization**:
- Train on 200 levels
- Test on *unseen* levels

This shows whether the agent truly understands the game dynamics, or just memorized solutions.

## Interview Talking Points

1. **Model-based vs Model-free**: "DreamerV3 learns a world model, so it can plan ahead. This is more sample-efficient than model-free methods."

2. **Interpretability**: "I can visualize what the agent imagines will happen. This is a form of interpretability - seeing inside the model's 'mind'."

3. **Generalization**: "Procgen tests whether the agent generalizes. Train on 200 levels, test on new ones. Does it understand the dynamics, or just memorize?"

4. **Value Function**: "The value heatmap shows which states the agent thinks are good. When it spikes, the agent sees an opportunity. When it drops, danger."

## References

- [DreamerV3 Paper](https://arxiv.org/abs/2301.04104) - Mastering Diverse Domains through World Models
- [Procgen Benchmark](https://github.com/openai/procgen) - Procedurally generated environments
- [Original Dreamer](https://arxiv.org/abs/1912.01603) - Dream to Control

## License

MIT

---

*Built for Anthropic interview demonstration*
