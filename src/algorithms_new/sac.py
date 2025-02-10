import copy
import os.path as op

import numpy as np
import torch
from torch import from_numpy
from torch.optim.adam import Adam

from algorithms_new.model import PolicyNetwork, QvalueNetwork


class SAC:
    def __init__(
        self,
        env_name,
        state_dim,
        action_dim,
        gamma=0.99,
        alpha=0.2,
        actor_lr=1e-4,
        critic_lr=1e-3,
        action_bounds=(-1, 1),
        reward_scale=1,
        replay_buffer=None,
        device="cpu",
        target_network_frequency=1,
        policy_frequency=2,
        tau=0.005,
    ):
        self.env_name = env_name
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.alpha = alpha
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.action_bounds = action_bounds
        self.reward_scale = reward_scale
        self.replay_buffer = replay_buffer
        self.target_network_frequency = target_network_frequency
        self.policy_frequency = policy_frequency
        self.tau = tau

        self.device = device

        self.policy_network = PolicyNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            action_bounds=self.action_bounds,
        ).to(self.device)

        self.qnet1 = QvalueNetwork(
            state_dim=self.state_dim, action_dim=self.action_dim
        ).to(self.device)

        self.qnet2 = QvalueNetwork(
            state_dim=self.state_dim, action_dim=self.action_dim
        ).to(self.device)

        self.qnet1_target = copy.deepcopy(self.qnet1).to(self.device)

        self.qnet2_target = copy.deepcopy(self.qnet2).to(self.device)

        self.q_value_loss = torch.nn.MSELoss()

        self.q_optimizer = Adam(
            list(self.qnet1.parameters()) + list(self.qnet2.parameters()),
            lr=self.critic_lr,
        )
        self.actor_optimizer = Adam(
            list(self.policy_network.parameters()), lr=self.actor_lr
        )
        self.actor_loss_value = 0
        self.q_loss_value = 0

    def store(self, state, reward, done, action, next_state):
        state = from_numpy(state).float().to("cpu")
        reward = torch.Tensor([reward]).to("cpu")
        done = torch.Tensor([done]).to("cpu")
        action = torch.Tensor([action]).to("cpu")
        next_state = from_numpy(next_state).float().to("cpu")
        self.replay_buffer.add(state, reward, done, action, next_state)

    def train(self, global_step, device):
        if not self.replay_buffer.can_sample():
            return 0, 0, 0
        else:
            print("training...")
            # take a batch
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(
                device=device
            )
            print(len(states))

            # compute a' (target action) and its probs for s'
            target_actions, target_actions_log_probs, entropy = (
                self.policy_network.get_action(next_states)
            )

            # compute q-values from target networks for s',a'
            next_action_values = torch.min(
                self.qnet1_target(next_states, target_actions),
                self.qnet2_target(next_states, target_actions),
            )

            # compute target
            expected_action_values = rewards.unsqueeze(
                -1
            ) + self.gamma * torch.logical_not(dones.unsqueeze(-1)) * (
                next_action_values - self.alpha * target_actions_log_probs
            )

            # compute action values for a and s
            action_values1 = self.qnet1(states, actions)
            action_values2 = self.qnet2(states, actions)

            # compute the loss with the target
            action_values1_loss = self.q_value_loss(
                action_values1, expected_action_values
            )
            action_values2_loss = self.q_value_loss(
                action_values2, expected_action_values
            )

            # sum the losses
            q_loss = action_values1_loss + action_values2_loss

            # store loss
            self.q_loss_value = q_loss.item()

            # perform gradient descent
            self.q_optimizer.zero_grad()
            q_loss.backward()
            self.q_optimizer.step()

            if global_step % self.policy_frequency == 0:

                # compute a_tilde_theta
                actions_tilde_theta, log_actions_tilde_theta_values, entropy = (
                    self.policy_network.get_action(states)
                )

                # take the min action values
                actions_values = torch.min(
                    self.qnet1(states, actions_tilde_theta),
                    self.qnet2(states, actions_tilde_theta),
                )

                # compute the loss
                actor_loss = (
                    (self.alpha * log_actions_tilde_theta_values) - actions_values
                ).mean()

                # store the loss
                self.actor_loss_value = actor_loss.item()

                # compute gradient ascent because the sing in actor_loss has been inrveted
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

            # update the target networks with polyak average
            if global_step % self.target_network_frequency == 0:
                for param, target_param in zip(
                    self.qnet1.parameters(),
                    self.qnet1_target.parameters(),
                ):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data
                    )
                for param, target_param in zip(
                    self.qnet2.parameters(),
                    self.qnet2_target.parameters(),
                ):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data
                    )
        return entropy

    def select_action(self, states):
        action, _, entropy = self.policy_network.get_action(states)
        return np.array(action.detach().cpu().tolist()), entropy

    @staticmethod
    def soft_update_target_network(local_network, target_network, tau=0.005):
        for target_param, local_param in zip(
            target_network.parameters(), local_network.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1 - tau) * target_param.data
            )

    def save(self, path, folder_name, n_episodes):
        import os

        file_path = op.join(path, folder_name)
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        torch.save(
            self.policy_network.state_dict(),
            os.path.join(
                file_path, self.env_name + f"_policy_{n_episodes}_weights.pth"
            ),
        )

    def load_weights(self, path, folder_name, n_n_episodes):
        print("loading model...")
        filepath = op.join(
            path,
            "models",
            folder_name,
            self.env_name + "_policy_" + str(n_n_episodes) + "_weights.pth",
        )
        self.policy_network.load_state_dict(torch.load(filepath))
        print(f"...model {filepath} loaded.")

    def set_to_eval_mode(self):
        self.policy_network.eval()
