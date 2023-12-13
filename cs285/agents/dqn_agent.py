from typing import Sequence, Callable, Tuple, Optional

import torch
from torch import nn

import numpy as np
import pdb
import cs285.infrastructure.pytorch_util as ptu


class DQNAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        make_critic: Callable[[Tuple[int, ...], int], nn.Module],
        make_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        make_lr_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        discount: float,
        target_update_period: int,
        use_double_q: bool = False,
        clip_grad_norm: Optional[float] = None,
    ):
        super().__init__()

        self.critic = make_critic(observation_shape, num_actions)
        self.target_critic = make_critic(observation_shape, num_actions)
        self.critic_optimizer = make_optimizer(self.critic.parameters())
        self.lr_scheduler = make_lr_schedule(self.critic_optimizer)

        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.discount = discount
        self.target_update_period = target_update_period
        self.clip_grad_norm = clip_grad_norm
        self.use_double_q = use_double_q

        self.critic_loss = nn.MSELoss()

        self.update_target_critic()

    def get_action(self, observation: np.ndarray, epsilon: float = -1) -> int:
        """
        Used for evaluation.
        """
        observation = ptu.from_numpy(np.asarray(observation))[None]

        """
        action = ...
        """
        if torch.rand(1) < epsilon:
            action = torch.randint(self.num_actions, ())
        else:
            qa_values: torch.Tensor = self.critic(observation)
            action = qa_values.argmax(dim=-1)

        return ptu.to_numpy(action).squeeze(0).item()

    def compute_critic_loss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict, dict]:
        """Update the DQN critic, and return stats for logging."""
        (batch_size,) = reward.shape

        # Compute target values
        with torch.no_grad():
            """
            next_qa_values = ...

            if self.use_double_q:
                raise NotImplementedError
            else:
                next_action = ...
            
            next_q_values = ...
            target_values = ...
            """
            next_qa_values: torch.Tensor = self.target_critic(next_obs)
            assert next_qa_values.shape == (
                batch_size,
                self.num_actions,
            ), next_qa_values.shape

            if not self.use_double_q:
                # Standard
                next_q_values, _ = next_qa_values.max(dim=-1)
            else:
                # Double-Q
                doubleq_next_qa_values: torch.Tensor = self.critic(next_obs)
                doubleq_next_action = doubleq_next_qa_values.argmax(dim=-1)
                next_q_values = torch.gather(next_qa_values, 1, doubleq_next_action.unsqueeze(1)).squeeze(1)

            assert next_q_values.shape == (batch_size,), next_q_values.shape

            target_values: torch.Tensor = reward + self.discount * next_q_values * (
                1 - done.float()
            )
            assert target_values.shape == (batch_size,), target_values.shape

        """
        qa_values = ...
        q_values = ... # Compute from the data actions; see torch.gather
        loss = ...

        """
        # Predict Q-values
        qa_values = self.critic(obs)
        assert qa_values.shape == (batch_size, self.num_actions), qa_values.shape

        # Select Q-values for the actions that were actually taken
        q_values = torch.gather(qa_values, 1, action.unsqueeze(1)).squeeze(1)
        assert q_values.shape == (batch_size,), q_values.shape

        # Compute loss
        loss: torch.Tensor = self.critic_loss(q_values, target_values)

        return (
            loss,
            {
                "critic_loss": loss.item(),
                "q_values": q_values.mean().item(),
                "target_values": target_values.mean().item(),
            },
            {
                "qa_values": qa_values,
                "q_values": q_values,
            },
        )
    
    def update_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> dict:
        loss, metrics, _ = self.compute_critic_loss(obs, action, reward, next_obs, done)

        self.critic_optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
            self.critic.parameters(), self.clip_grad_norm or float("inf")
        )
        self.critic_optimizer.step()
        metrics["grad_norm"] = grad_norm.item()
        self.lr_scheduler.step()

        return metrics

    def update_target_critic(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def update(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
        step: int,
    ) -> dict:
        """
        Update the DQN agent, including both the critic and target.
        """
        critic_stats = self.update_critic(obs, action, reward, next_obs, done)

        if step % self.target_update_period == 0:
            self.update_target_critic()

        return critic_stats
    
    def save(self, filepath: str):
        torch.save({'model_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.critic_optimizer.state_dict()}, filepath + '_dqn_model.pt')
        
    def load(self, save_file: str):
        checkpoint = torch.load(save_file, map_location='cpu')
        print('Model loaded: {}'.format(save_file))
        self.critic.load_state_dict(checkpoint['model_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])