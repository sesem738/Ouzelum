import gym
import isaacgym  # noqa
import isaacgymenvs
import numpy as np
import torch

import os
import time
import random
import argparse
from agent import Agent
from distutils.util import strtobool

class RecordEpisodeStatisticsTorch(gym.Wrapper):
	def __init__(self, env, device):
		super().__init__(env)
		self.num_envs = getattr(env, "num_envs", 1)
		self.device = device
		self.episode_returns = None
		self.episode_lengths = None

	def reset(self, **kwargs):
		observations = super().reset(**kwargs)
		self.episode_returns = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
		self.episode_lengths = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
		self.returned_episode_returns = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
		self.returned_episode_lengths = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
		return observations

	def step(self, action):
		observations, rewards, dones, infos = super().step(action)
		self.episode_returns += rewards
		self.episode_lengths += 1
		self.returned_episode_returns[:] = self.episode_returns
		self.returned_episode_lengths[:] = self.episode_lengths
		self.episode_returns *= 1 - dones
		self.episode_lengths *= 1 - dones
		infos["r"] = self.returned_episode_returns
		infos["l"] = self.returned_episode_lengths
		return (
			observations,
			rewards,
			dones,
			infos,
		)
	
class ExtractObsWrapper(gym.ObservationWrapper):
	def observation(self, obs):
		return obs["obs"]
	

def parse_args():
	p = argparse.ArgumentParser()
	p.add_argument("--env", default="Hawks")          		   
	p.add_argument("--seed", default=0, type=int)
	p.add_argument("--num_envs", type=int, default=4096)
	p.add_argument("--rollout_steps", type=int, default=16)
	p.add_argument("--total_steps", type=int, default=3000000)
	p.add_argument("--headless", action="store_true")
	
	args = p.parse_args()
	return args



def main():
	args = parse_args()

	# Seeding
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	# Define Isaac Gym Environment 
	envs = isaacgymenvs.make(
		seed=args.seed,
		task=args.env,
		num_envs=args.num_envs,
		sim_device="cuda:0",
		rl_device="cuda:0",
		headless= args.headless,
		force_render = not args.headless,
	)

	envs = ExtractObsWrapper(envs)
	envs = RecordEpisodeStatisticsTorch(envs, device)
	envs.single_action_space = envs.action_space
	envs.single_observation_space = envs.observation_space
	assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

	agent = Agent()
	
	# Rollout Buffer
	obs = torch.zeros((args.rollout_steps, args.num_envs) + envs.single_observation_space.shape, dtype=torch.float).to(device)
	actions = torch.zeros((args.rollout_steps, args.num_envs) + envs.single_action_space.shape, dtype=torch.float).to(device)
	logprobs = torch.zeros((args.rollout_steps, args.num_envs), dtype=torch.float).to(device)
	rewards = torch.zeros((args.rollout_steps, args.num_envs), dtype=torch.float).to(device)
	dones = torch.zeros((args.rollout_steps, args.num_envs), dtype=torch.float).to(device)
	values = torch.zeros((args.rollout_steps, args.num_envs), dtype=torch.float).to(device)
	advantages = torch.zeros_like(rewards, dtype=torch.float).to(device)

	global_step = 0
	next_obs = envs.reset()
	next_done = torch.zeros(args.num_envs, dtype=torch.float).to(device)

	for global_step in range(1, args.total_steps):
		
		for step in range(0, args.rollout_steps):

			global_step += args.num_envs
			obs[step]=next_obs
			dones[step] = next_done
			action = agent.get_action(next_obs)
			actions[step] = action
			next_obs, rewards[step], next_done, info = envs.step(action)






if __name__ == "__main__":
	main()