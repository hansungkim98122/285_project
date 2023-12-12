from typing import Tuple, Optional

import gym

import numpy as np
import torch
import torch.nn as nn

from cs285.networks.mlp_policy import MLPPolicy
from cs285.networks.state_action_value_critic import StateActionCritic
import cs285.infrastructure.pytorch_util as ptu

from gym.wrappers.rescale_action import RescaleAction
from gym.wrappers.clip_action import ClipAction
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics
import TeachMyAgent.environments

def sac_config_new(
    env_name: str,
    exp_name: Optional[str] = None,
    hidden_size: int = 128,
    num_layers: int = 3,

    alpha_lr: float = 1e-3,
    init_temperature: float = 0.2,

    total_steps: int = 1000000,
    random_steps: int = 5000,
    training_starts: int = 10000,
    batch_size: int = 128,
    replay_buffer_capacity: int = 1000000,
    ep_len: Optional[int] = None,
    discount: float = 0.99,
    device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    action_range: Tuple[float, float] = (-1, 1),

    # Actor configuration
    actor_betas: Tuple[float, float] = (0.9, 0.999),
    alpha_betas: Tuple[float, float] = (0.9, 0.999),
    actor_update_frequency: Optional[int] = None,
    actor_cfg: Optional[dict] = None,
    actor_lr: float = 1e-3,
    
    #Critic configuration
    critic_cfg: Optional[dict] = None,
    critic_lr: float = 1e-3,
    critic_target_update_frequency: Optional[int] = None,
    critic_tau: Optional[float] = None,
    critic_betas: Tuple[float, float] = (0.9, 0.999),
    learnable_temperature: bool = True,

    # Actor-critic configuration
    actor_gradient_type="reinforce",  # One of "reinforce" or "reparametrize"
    # Settings for multiple critics
    target_critic_backup_type: str = "mean",  # One of "doubleq", "min", or "mean"
    # Soft actor-critic
    use_entropy_bonus: bool = True,
    temperature: float = 0.005,
    training_freq: int = 1,
):

    def make_env(render: bool = False,**kwargs):
        return RecordEpisodeStatistics(
            ClipAction(
                RescaleAction(
                    gym.make(
                        env_name, **kwargs
                    ),
                    -1,
                    1,
                )
            )
        )

    log_string = "{}_{}_{}_s{}_l{}_alr{}_clr{}_b{}_d{}".format(
        exp_name or "offpolicy_ac",
        env_name,
        actor_gradient_type,
        hidden_size,
        num_layers,
        actor_lr,
        critic_lr,
        batch_size,
        discount,
    )

    if use_entropy_bonus:
        log_string += f"_t{temperature}"


    if target_critic_backup_type != "mean":
        log_string += f"_{target_critic_backup_type}"

    return {
        "agent_kwargs": {
            "action_range": action_range,
            "device": device,
            "critic_cfg": critic_cfg,
            "actor_cfg": actor_cfg,
            "discount": discount,
            "init_temperature": init_temperature,
            "alpha_lr": alpha_lr,
            "alpha_betas": alpha_betas,
            "actor_lr": actor_lr,
            "actor_betas": actor_betas,
            "actor_update_frequency": actor_update_frequency,
            "critic_lr": critic_lr,
            "critic_betas": critic_betas,
            "critic_tau": critic_tau,
            "critic_target_update_frequency": critic_target_update_frequency,
            "learnable_temperature": learnable_temperature,
        },
        "replay_buffer_capacity": replay_buffer_capacity,
        "log_name": log_string,
        "total_steps": total_steps,
        "random_steps": random_steps,
        "training_starts": training_starts,
        "ep_len": ep_len,
        "training_freq": training_freq,
        "batch_size": batch_size,
        "make_env": make_env,
    }
