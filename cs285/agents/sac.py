import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from cs285.agents.utils import *
from cs285.agents.critic import DoubleQCritic
from cs285.agents.actor import DiagGaussianActor

# import hydra
import abc

class Agent(object):
    def reset(self):
        """For state-full agents this function performs reseting at the beginning of each episode."""
        pass

    @abc.abstractmethod
    def train(self, training=True):
        """Sets the agent in either training or evaluation mode."""

    @abc.abstractmethod
    def update(self, replay_buffer, logger, step):
        """Main function of the agent that performs learning."""

    @abc.abstractmethod
    def act(self, obs, sample=False):
        """Issues an action given an observation."""

class SACAgent(Agent):
    """SAC algorithm."""
    def __init__(self, obs_shape, action_dim, action_range, device, critic_cfg,
                 actor_cfg, discount, init_temperature, alpha_lr, alpha_betas,
                 actor_lr, actor_betas, actor_update_frequency, critic_lr,
                 critic_betas, critic_tau, critic_target_update_frequency,
                learnable_temperature):
        super().__init__()

        self.action_range = action_range
        self.device = torch.device(device)
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.learnable_temperature = learnable_temperature

        # self.critic = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic = DoubleQCritic(critic_cfg['params']['obs_dim'],critic_cfg['params']['action_dim'],critic_cfg['params']['hidden_dim'],critic_cfg['params']['hidden_depth']).to(self.device)

        self.critic_target = DoubleQCritic(critic_cfg['params']['obs_dim'],critic_cfg['params']['action_dim'],critic_cfg['params']['hidden_dim'],critic_cfg['params']['hidden_depth']).to(self.device)
        # self.critic_target = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # self.actor = hydra.utils.instantiate(actor_cfg).to(self.device)
        self.actor = DiagGaussianActor(actor_cfg['params']['obs_dim'],actor_cfg['params']['action_dim'],actor_cfg['params']['hidden_dim'],actor_cfg['params']['hidden_depth'],[-5, 2]).to(self.device)
        
        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -action_dim

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr,
                                                betas=actor_betas)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr,
                                                 betas=critic_betas)

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr,
                                                    betas=alpha_betas)
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.alpha_lr = alpha_lr
        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return to_np(action[0])

    def update_critic(self, obs, action, reward, next_obs, done):
        dist = self.actor(next_obs)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1,
                             target_Q2) - self.alpha.detach() * log_prob
        
        target_Q = reward[:,None] + (1.0 - 1.0*done[:,None]) * (self.discount * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)

        assert target_Q.shape == current_Q1.shape 

        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


        return {
            "critic_loss": critic_loss.item(),
            "q1_values": current_Q1.mean().item(),
            "q2_values": current_Q2.mean().item(),
            "target_values": target_Q.mean().item(),
        }

    def update_actor_and_alpha(self, obs):
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        # logger.log('train_actor/loss', actor_loss, step)
        # logger.log('train_actor/target_entropy', self.target_entropy, step)
        # logger.log('train_actor/entropy', -log_prob.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                          (-log_prob - self.target_entropy).detach()).mean()
            # logger.log('train_alpha/loss', alpha_loss, step)
            # logger.log('train_alpha/value', self.alpha, step)
            alpha_loss.backward()
            self.log_alpha_optimizer.step()
        
        return {"actor_loss": actor_loss.item(), "entropy": -log_prob.mean().item()}

    def update(        
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,):

        critic_info = self.update_critic(observations, actions, rewards, next_observations, dones)

        if step % self.actor_update_frequency == 0:
            actor_info = self.update_actor_and_alpha(observations)

        if step % self.critic_target_update_frequency == 0:
            soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)
            
        return {
            **actor_info,
            **critic_info,
            "actor_lr": self.actor_lr,
            "critic_lr": self.critic_lr,
        }
