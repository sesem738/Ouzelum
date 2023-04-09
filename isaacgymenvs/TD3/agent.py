import copy
import torch
import torch.optim as optim
import torch.nn.functional as F
from model import Actor, QNetwork


class TD3(object):
    def __init__(
            self,
            envs,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(envs).to(self.device)
        self.target_actor = Actor(envs).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=3e-4)
        
        self.qf1 = QNetwork(envs).to(self.device)
        self.qf2 = QNetwork(envs).to(self.device)
        self.qf1_target = QNetwork(envs).to(self.device)
        self.qf2_target = QNetwork(envs).to(self.device)
        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=3e-4)

        self.envs = envs
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

    def select_action(self, state):
        actions = Actor(state)
        actions += torch.normal(0, self.actor.action_scale * self.exploration_noise)
        actions = actions.clip(self.envs.single_action_space.low, self.envs.single_action_space.high)
        return actions

    def train(self, rb, batch_size):
        data = rb.sample(batch_size)
        with torch.no_grad():
            clipped_noise = (torch.randn_like(data.actions, device=self.device) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            ) * self.target_actor.action_scale

            next_state_actions = (self.target_actor(data.next_observations) + clipped_noise).clamp(
                self.envs.single_action_space.low[0], self.envs.single_action_space.high[0]
            )
            qf1_next_target = self.qf1_target(data.next_observations, next_state_actions)
            qf2_next_target = self.qf2_target(data.next_observations, next_state_actions)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
            next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * self.gamma * (min_qf_next_target).view(-1)

        qf1_a_values = self.qf1(data.observations, data.actions).view(-1)
        qf2_a_values = self.qf2(data.observations, data.actions).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        # optimize the model
        self.q_optimizer.zero_grad()
        self.qf_loss.backward()
        self.q_optimizer.step()

    
    # FIX SAVE AND LOAD FUNCTIONS

    
    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(),
                   filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(),
                   filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(
            torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(
            torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)

    