"""
Imagination: Roll out dream trajectories using the world model.

This is where DreamerV3 does most of its learning:
1. Start from a real state
2. Imagine future states using learned dynamics
3. Collect imagined rewards
4. Train actor/critic on dreams
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class ImaginedTrajectory:
    """Container for imagined trajectory data."""
    states: Dict[str, torch.Tensor]  # {'deter': (B, H, D), 'stoch': (B, H, S, C)}
    state_vectors: torch.Tensor      # (B, H, state_dim) - flattened states
    actions: torch.Tensor            # (B, H) action indices
    log_probs: torch.Tensor          # (B, H) action log probs
    rewards: torch.Tensor            # (B, H) predicted rewards
    continues: torch.Tensor          # (B, H) predicted continuation probs
    values: torch.Tensor             # (B, H) value estimates


def imagine_trajectory(
    world_model: nn.Module,
    actor: nn.Module,
    critic: nn.Module,
    initial_state: Dict[str, torch.Tensor],
    horizon: int = 15,
) -> ImaginedTrajectory:
    """
    Imagine a trajectory starting from a real state.

    This is the "dreaming" process:
    1. Use actor to pick actions
    2. Use world model to predict next states
    3. Use reward/continue predictors
    4. Estimate values with critic

    Args:
        world_model: The learned world model
        actor: Policy network
        critic: Value network
        initial_state: Starting state from real experience
        horizon: How many steps to imagine

    Returns:
        ImaginedTrajectory with all data needed for training
    """
    B = initial_state['deter'].size(0)
    device = initial_state['deter'].device

    # Storage
    deters = []
    stochs = []
    state_vecs = []
    actions = []
    log_probs = []
    rewards = []
    continues = []
    values = []

    state = initial_state

    with torch.no_grad():  # No gradients through imagination for efficiency
        for t in range(horizon):
            # Get flattened state for actor/critic
            state_vec = world_model.get_state_vector(state)

            # Actor picks action
            action_dist = actor(state_vec)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)

            # Critic estimates value
            value = critic.predict(state_vec)

            # Predict reward and continuation
            reward = world_model.reward_predictor.predict(state_vec)
            cont = world_model.continue_predictor.predict(state_vec)

            # Store
            deters.append(state['deter'])
            stochs.append(state['stoch'])
            state_vecs.append(state_vec)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            continues.append(cont)
            values.append(value)

            # Imagine next state
            state = world_model.imagine_step(state, action)

    # Final value for bootstrapping
    final_state_vec = world_model.get_state_vector(state)
    final_value = critic.predict(final_state_vec)
    values.append(final_value)

    # Stack into tensors
    states = {
        'deter': torch.stack(deters, dim=1),
        'stoch': torch.stack(stochs, dim=1),
    }

    return ImaginedTrajectory(
        states=states,
        state_vectors=torch.stack(state_vecs, dim=1),
        actions=torch.stack(actions, dim=1),
        log_probs=torch.stack(log_probs, dim=1),
        rewards=torch.stack(rewards, dim=1),
        continues=torch.stack(continues, dim=1),
        values=torch.stack(values, dim=1),  # (B, H+1) includes bootstrap
    )


def imagine_trajectory_with_grads(
    world_model: nn.Module,
    actor: nn.Module,
    critic: nn.Module,
    initial_state: Dict[str, torch.Tensor],
    horizon: int = 15,
) -> ImaginedTrajectory:
    """
    Imagine with gradients enabled for actor/critic training.

    Same as imagine_trajectory but allows gradient flow for policy learning.
    """
    B = initial_state['deter'].size(0)
    device = initial_state['deter'].device

    deters = []
    stochs = []
    state_vecs = []
    actions = []
    log_probs = []
    rewards = []
    continues = []
    values = []

    state = {
        'deter': initial_state['deter'].detach(),
        'stoch': initial_state['stoch'].detach(),
    }

    for t in range(horizon):
        state_vec = world_model.get_state_vector(state)

        # Actor (with gradients)
        action_dist = actor(state_vec)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)

        # Critic (with gradients)
        value = critic.predict(state_vec)

        # Predictors (detached - we don't train world model here)
        with torch.no_grad():
            reward = world_model.reward_predictor.predict(state_vec)
            cont = world_model.continue_predictor.predict(state_vec)

        deters.append(state['deter'])
        stochs.append(state['stoch'])
        state_vecs.append(state_vec)
        actions.append(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        continues.append(cont)
        values.append(value)

        # Imagine next state (detached)
        with torch.no_grad():
            state = world_model.imagine_step(state, action)
            state = {k: v.detach() for k, v in state.items()}

    # Final value
    final_state_vec = world_model.get_state_vector(state)
    final_value = critic.predict(final_state_vec)
    values.append(final_value)

    states = {
        'deter': torch.stack(deters, dim=1),
        'stoch': torch.stack(stochs, dim=1),
    }

    return ImaginedTrajectory(
        states=states,
        state_vectors=torch.stack(state_vecs, dim=1),
        actions=torch.stack(actions, dim=1),
        log_probs=torch.stack(log_probs, dim=1),
        rewards=torch.stack(rewards, dim=1),
        continues=torch.stack(continues, dim=1),
        values=torch.stack(values, dim=1),
    )
