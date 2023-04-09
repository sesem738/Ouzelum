import os
import time
import random
import argparse
import numpy as np
from distutils.util import strtobool

import gym
import isaacgym
import isaacgymenvs
import torch
import utils
from agent import TD3


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="Ingenuity")               # Isaac Gym environment name
    parser.add_argument("--num_envs", default=2048, type=int)       # Number of Parallel Environments
    parser.add_argument("--cuda", default=True, type=bool)          # CUDA
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--discount", default=0.99)                 # Discount factor
    parser.add_argument("--tau", default=0.005)                     # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True) 
    args = parser.parse_args()

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
        force_render=True,
    )

    envs = utils.ExtractObsWrapper(envs)
    envs = utils.RecordEpisodeStatisticsTorch(envs, "cuda")
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    

    max_action = float(envs.action_space.high[0])
    kwargs = {
        "envs": envs,
        "discount": args.discount,
        "tau": args.tau,
    }
    
    kwargs["policy_noise"] = args.policy_noise * max_action
    kwargs["noise_clip"] = args.noise_clip * max_action
    
    agent = TD3(**kwargs)

    eval_episodes = 1000000000
    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = envs.reset(), False
        action = agent.select_action(state)
        state, reward, done, _ = envs.step(action)
        avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    # print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")