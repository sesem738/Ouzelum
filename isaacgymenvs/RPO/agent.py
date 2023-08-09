from model import Actor
from model import Critic

import os
import copy
import torch
import numpy as np

class PPO:
    def __init__(self, obs_dim, act_dim, num_envs, device) -> None:
        
        self.obs_dim  = obs_dim
        self.act_dim = act_dim
        self.num_envs = num_envs
        self.rollout_steps = 16
        self.device = device
        self.clip_coef = 0.2 
        self.gamma = 0.99
        self.gae_lamda = 0.95
        self.norm_adv = True
        self.update_epochs = 4
        self.ent_coef = 0.0
        self.vf_coef = 2
        self.clip_vloss = False
        self.target_kl = None
        self.max_grad_norm = 1
        self.batch_size = int(self.num_envs * self.rollout_steps)
        self.minibatch_size = int(self.batch_size // 2)
        
        self.actor = Actor(self.obs_dim, self.act_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.0026, eps=1e-5)
        
        self.critic = Critic(self.obs_dim).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.0026, eps=1e-5)

    def getAction(self, state, ):
        action, logprob, entropy = self.actor(state)
        return action, logprob, entropy

    def getGAE(self, next_obs, next_done, rewards, dones, values):
        with torch.no_grad():
            next_value = self.critic(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(self.device)
            lastgaelam = 0
            for t in reversed(range(self.rollout_steps)):
                if t == self.rollout_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + self.gamma * self.gae_lamda * nextnonterminal * lastgaelam
            returns = advantages + values
        return returns, advantages

    def train(self, obs, actions, next_obs, next_done, logprobs, rewards, dones):

        values = self.critic(obs)
        returns, advantages = self.getGAE(next_obs, next_done, rewards, dones, values.reshape(16, 4096))
        
        b_obs = obs.reshape((-1,) + self.obs_dim.shape)
        b_actions = actions.reshape((-1,) + self.act_dim.shape)
        b_logprobs = logprobs.reshape(-1)
        b_values = values.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)




        clipfracs = []
        for epoch in range(self.update_epochs):
            b_inds = torch.from_numpy(np.random.permutation(self.batch_size)).to(self.device)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy = self.actor(b_obs[mb_inds], b_actions[mb_inds])
                newvalue = self.critic(b_obs[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if self.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if self.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.clip_coef,
                        self.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                actor_loss = pg_loss - self.ent_coef * entropy_loss
                critic_loss = v_loss * self.vf_coef

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

            if self.target_kl is not None:
                if approx_kl > self.target_kl:
                    break

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
        
    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)