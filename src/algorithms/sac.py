from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import algorithms.modules as m
import utils


class FeaturesHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output

    def close(self):
        self.hook.remove()


class SAC(object):
    def __init__(self, obs_shape, action_shape,env_action_spaces, args):
        self.discount = args.discount
        self.critic_tau = args.critic_tau
        self.encoder_tau = args.encoder_tau
        self.actor_update_freq = args.actor_update_freq
        self.critic_target_update_freq = args.critic_target_update_freq
        self.writer = args.writer_tensorboard

        shared_cnn = m.SharedCNN(
            obs_shape[0], args.num_shared_layers, args.num_filters
        ).cuda()
        head_cnn = m.HeadCNN(
            shared_cnn.out_shape, args.num_head_layers, args.num_filters
        ).cuda()
        actor_encoder = m.Encoder(
            shared_cnn,
            head_cnn,
            m.RLProjection(head_cnn.out_shape, args.projection_dim),
        )
        critic_encoder = m.Encoder(
            shared_cnn,
            head_cnn,
            m.RLProjection(head_cnn.out_shape, args.projection_dim),
        )

        self.actor = m.Actor(
            actor_encoder,
            action_shape,
            args.hidden_dim,
            args.actor_log_std_min,
            args.actor_log_std_max,
            env_action_spaces
            
        ).cuda()

        self.critic = m.CriticState(
            critic_encoder,
            action_shape,
            args.hidden_dim,
            state_dim=int(obs_shape[0][0] / 3)* obs_shape[1][0]).cuda()  # /3 because RGB channels
        self.critic_target = deepcopy(self.critic)

        self.log_alpha = torch.tensor(np.log(args.init_temperature)).cuda()
        self.log_alpha.requires_grad = True
        self.target_entropy = -np.prod(action_shape)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=args.actor_lr, betas=(args.actor_beta, 0.999)
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=args.critic_lr,
            betas=(args.critic_beta, 0.999),
            weight_decay=args.critic_weight_decay,
        )
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=args.alpha_lr, betas=(args.alpha_beta, 0.999)
        )

        self.train()
        self.critic_target.train()
        self.hook = FeaturesHook(self.critic.encoder.head_cnn)

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def eval(self):
        self.train(False)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def _obs_to_input(self, obs):
        if isinstance(obs, utils.LazyFrames):
            _obs = np.array(obs)
            _obs = np.vstack(_obs[[0, 2, 4]])
        else:
            _obs = obs
        _obs = torch.FloatTensor(_obs[0]).cuda()
        _obs = _obs.unsqueeze(0)
        return _obs

    def select_action(self, obs):
        _obs = self._obs_to_input(obs)
        with torch.no_grad():
            mu, _, _, _ = self.actor(_obs, compute_pi=False, compute_log_pi=False)
        return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        _obs = self._obs_to_input(obs)
        with torch.no_grad():
            mu, pi, _, _ = self.actor(_obs, compute_log_pi=False)
        return pi.cpu().data.numpy().flatten()

    def update_critic(self, obs, action, reward, next_obs, not_done, L=None, step=None):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs[0])
            target_Q1, target_Q2 = self.critic_target(
                next_obs[0], policy_action, next_obs[1]
            )
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward.unsqueeze(1) + (
                not_done.unsqueeze(1) * self.discount * target_V
            )

        current_Q1, current_Q2 = self.critic(obs[0], action, obs[1])
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )
        if L is not None:
            L.log("train/critic_loss", critic_loss, step)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss

    def update_actor_and_alpha(self, obs, L=None, step=None, update_alpha=True):
        _, pi, log_pi, log_std = self.actor(obs[0], detach=True)
        actor_Q1, actor_Q2 = self.critic(obs[0], pi, obs[1], detach=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        if L is not None:
            L.log("train/actor_loss", actor_loss, step)
            entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)) + log_std.sum(
                dim=-1
            )

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if update_alpha:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()

            if L is not None:
                L.log("train/alpha_loss", alpha_loss, step)
                L.log("train/alpha_value", self.alpha, step)

            alpha_loss.backward()
            self.log_alpha_optimizer.step()

        return actor_loss, alpha_loss

    def soft_update_critic_target(self):
        utils.soft_update_params(self.critic.Q1, self.critic_target.Q1, self.critic_tau)
        utils.soft_update_params(self.critic.Q2, self.critic_target.Q2, self.critic_tau)
        utils.soft_update_params(
            self.critic.encoder, self.critic_target.encoder, self.encoder_tau
        )

    def update(self, replay_buffer, L, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample()

        critic_loss = self.update_critic(
            obs, action, reward, next_obs, not_done, L, step
        )
        self.writer.add_scalar("Loss/critic_loss", critic_loss, step)

        if step % self.actor_update_freq == 0:
            actor_loss, alpha_loss = self.update_actor_and_alpha(obs, L, step)
            self.writer.add_scalar("Loss/actor_loss", actor_loss, step)
            self.writer.add_scalar("Loss/alpha_loss", alpha_loss, step)

        if step % self.critic_target_update_freq == 0:
            self.soft_update_critic_target()
