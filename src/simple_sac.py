import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Definizione della rete neurale per il modello dell'attore
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(state_dim[0][0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7 + state_dim[1][0] * 3, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)

    def forward(self, state):
        state1, state2 = state
        x1 = None
        x1 = F.relu(self.conv1(state1))
        x1 = F.relu(self.conv2(x1))
        x1 = F.relu(self.conv3(x1))
        # x1 = x1.view(x1.size(0), -1)  # Flatten
        if len(state1.shape) == 3:
            x1 = torch.flatten(x1, start_dim=0)
            x2 = F.relu(self.fc1(torch.cat([x1, state2], dim=0)))
        else:
            x1 = torch.flatten(x1, start_dim=1)
            x2 = F.relu(self.fc1(torch.cat([x1, state2], dim=1)))
        x2 = F.relu(self.fc2(x2))
        action = torch.tanh(self.fc3(x2))
        if len(action.shape) > 1:
            action = torch.cat(
                ((action[:, 0] * 0.5 + 0.5).unsqueeze(1), action[:, 1].unsqueeze(1)),
                axis=1,
            )
        else:
            action = torch.tensor(
                (action[0] * 0.5 + 0.5, action[1]), dtype=torch.float32
            ).to(device)

        return action


# Definizione della rete neurale per il modello del critico
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(state_dim[0][0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7 + state_dim[1][0] * 3 + action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        state1, state2 = state

        x1 = F.relu(self.conv1(state1))
        x1 = F.relu(self.conv2(x1))
        x1 = F.relu(self.conv3(x1))
        x1 = x1.view(x1.size(0), -1)  # Flatten
        x = torch.cat([x1, state2, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value


# Definizione della classe SAC
class SACAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_size=256,
        lr=3e-4,
        gamma=0.99,
        tau=0.01,
        alpha=0.5,
    ):
        self.actor = Actor(state_dim, action_dim, hidden_size).to(device)
        self.critic1 = Critic(state_dim, action_dim, hidden_size).to(device)
        self.critic2 = Critic(state_dim, action_dim, hidden_size).to(device)
        self.target_critic1 = copy.deepcopy(self.critic1).to(device)
        self.target_critic2 = copy.deepcopy(self.critic2).to(device)
        self.alpha = alpha
        self.log_alpha = torch.tensor(
            np.log(self.alpha), requires_grad=True, device=device
        )
        self.alpha_optim = optim.Adam([self.log_alpha], lr=lr)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optim = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optim = optim.Adam(self.critic2.parameters(), lr=lr)
        self.gamma = gamma
        self.tau = tau
        self.target_entropy = 0.98  # (            -action_dim        )  # Adjust the target entropy based on the action space

    def select_action(self, state):
        state1, state2, state3 = state.frames

        # Converte le liste in tensori PyTorch
        state1_image = torch.FloatTensor(state1[0]).to(device)
        state2_image = torch.FloatTensor(state2[0]).to(device)
        state3_image = torch.FloatTensor(state3[0]).to(device)

        state1_array = torch.FloatTensor(state1[1]).to(device)
        state2_array = torch.FloatTensor(state2[1]).to(device)
        state3_array = torch.FloatTensor(state3[1]).to(device)

        # Concatena le immagini RGB lungo l'asse dei canali
        state_combined_image = torch.cat(
            [state1_image, state2_image, state3_image], dim=0
        )

        # Concatena gli array di 9 elementi lungo la dimensione 0
        state_combined_array = torch.cat(
            [state1_array, state2_array, state3_array], dim=0
        )

        # Passa lo stato combinato al modello dell'attore
        action = self.actor([state_combined_image, state_combined_array])

        if self.actor.training:
            action = torch.clamp(
                action + torch.randn_like(action) * 0.1, min=-1, max=1
            )  # Aggiungi rumore durante il training per l'esplorazione

        return action.cpu().detach().numpy().flatten()

    def update(self, replay_buffer, batch_size=64, logger=None, trainstep=None):
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        state1 = state[0] / 255
        state2 = state[1]
        action = action
        reward = reward.unsqueeze(1)
        next_state1 = next_state[0] / 255
        next_state2 = next_state[1]
        done = done.unsqueeze(1)

        next_action, log_prob = self._sample_action_and_compute_log_prob(
            [next_state1, next_state2]
        )

        # Update critic networks
        target_Q1 = self.target_critic1([next_state1, next_state2], next_action)
        target_Q2 = self.target_critic2([next_state1, next_state2], next_action)
        target_Q = torch.minimum(target_Q1, target_Q2)
        target_Q = reward + ~done * self.gamma * (
            target_Q - self.alpha * log_prob.unsqueeze(1)
        )
        #        print((self.alpha * log_prob.unsqueeze(1))[:10])

        current_Q1 = self.critic1([state1, state2], action)
        current_Q2 = self.critic2([state1, state2], action)

        critic1_loss = F.mse_loss(current_Q1, target_Q)
        critic2_loss = F.mse_loss(current_Q2, target_Q)

        self.critic1_optim.zero_grad()
        critic1_loss.backward(retain_graph=True)
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        # Update actor network
        sampled_action, log_prob = self._sample_action_and_compute_log_prob(
            [state1, state2]
        )
        actor_loss = (
            self.alpha * log_prob - self.critic1([state1, state2], sampled_action)
        ).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # Update temperature parameter alpha
        alpha_loss = -(
            self.log_alpha * (log_prob - self.target_entropy).detach()
        ).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        # Update target networks with Polyak averaging
        self._update_target_networks()

        # Log delle loss
        if logger is not None and trainstep is not None:
            logger.log("train/critic1_loss", critic1_loss.item(), trainstep)
            logger.log("train/critic2_loss", critic2_loss.item(), trainstep)
            logger.log("train/actor_loss", actor_loss.item(), trainstep)
            logger.log("train/alpha_loss", alpha_loss.item(), trainstep)

    def _sample_action_and_compute_log_prob(self, state):
        action = self.actor(state)
        log_prob = torch.log(torch.clamp(1 - action**2, 1e-6, 1.0 - 1e-6)).sum(dim=1)
        return action, log_prob

    def _update_target_networks(self):
        for param, target_param in zip(
            self.critic1.parameters(), self.target_critic1.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        for param, target_param in zip(
            self.critic2.parameters(), self.target_critic2.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def set_eval_mode(self):
        self.actor.eval()

    def set_train_mode(self):
        self.actor.train()


# # Esempio di utilizzo della classe SACAgent
# state_dim = ((3, 84, 84), (9,))
# action_dim = 2
# agent = SACAgent(state_dim, action_dim)

# # Addestramento dell'agente
# for episode in range(num_episodes):
#     state = (env.reset(), np.zeros((9,)))  # Esempio di stato iniziale
#     episode_reward = 0

#     for step in range(max_steps_per_episode):
#         action = agent.select_action(state, eval_mode=False)
#         next_state, reward, done, _ = env.step(action)
#         agent.update(replay_buffer)

#         state = (next_state, np.zeros((9,)))  # Esempio di stato successivo
#         episode_reward += reward

#         if done:
#             break

#     print(f"Episode: {episode+1}, Reward: {episode_reward}")

# # Modalità di valutazione
# agent.set_eval_mode()
# # Valutazione dell'agente
# for _ in range(num_eval_episodes):
#     state = (env.reset(), np.zeros((9,)))
#     episode_reward = 0

#     for step in range(max_steps_per_episode):
#         action = agent.select_action(state, eval_mode=True)
#         next_state, reward, done, _ = env.step(action)

#         state = (next_state, np.zeros((9,)))
#         episode_reward += reward

#         if done:
#             break

#     print(f"Evaluation Episode, Reward: {episode_reward}")
