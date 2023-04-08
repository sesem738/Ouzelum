import os
import time
import random
import utils
import argparse
import numpy as np
from distutils.util import strtobool


import gym
import isaacgym
import isaacgymenvs
import torch
from torch.utils.tensorboard import SummaryWriter


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
	parser.add_argument("--env", default="Ingenuity")               # Isaac Gym environment name
	parser.add_argument("--num_envs", default=2048, type=int)       # Number of Parallel Environments
	parser.add_argument("--cuda", default=True, type=bool)               # Isaac Gym environment name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used
	parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--total_timesteps", default=1e6, type=int) # Max time steps to run environment
	parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99)                 # Discount factor
	parser.add_argument("--tau", default=0.005)                     # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
	parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True) 
	args = parser.parse_args()
	
	run_name = f"{args.env}__{args.policy}__{args.seed}__{int(time.time())}"
	writer = SummaryWriter(f"runs/{run_name}")
	writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

	device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

	# Environment Step
	envs = isaacgymenvs.make(
        seed=args.seed,
        task=args.env,
        num_envs=args.num_envs,
        sim_device="cuda:0" if torch.cuda.is_available() and args.cuda else "cpu",
        rl_device="cuda:0" if torch.cuda.is_available() and args.cuda else "cpu",
        graphics_device_id=0 if torch.cuda.is_available() and args.cuda else -1,
        headless=False if torch.cuda.is_available() and args.cuda else True,
        multi_gpu=False,
        virtual_screen_capture=args.capture_video,
        force_render=False,
    )
	
	if args.capture_video:
		envs.is_vector_env = True
		print(f"record_video_step_frequency={args.record_video_step_frequency}")
		envs = gym.wrappers.RecordVideo(
			envs, f"videos/{run_name}", step_trigger=lambda step: step % args.record_video_step_frequency == 0,
			video_length=100
			)
	
	envs = utils.ExtractObsWrapper(envs)
	envs = utils.RecordEpisodeStatisticsTorch(envs, device)
	envs.single_action_space = envs.action_space
	envs.single_observation_space = envs.observation_space
	assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

	agent = Agent(envs).to(device)
	


