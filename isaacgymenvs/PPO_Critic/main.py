import argparse
import os
import random
import time

import gym
import isaacgym  # noqa
import isaacgymenvs
import numpy as np
import torch
from agent import PPO
from torch.utils.tensorboard import SummaryWriter
from utils import RecordEpisodeStatisticsTorch, ExtractObsWrapper
from isaacgymenvs.utils.POMDP import POMDPWrapper

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="Landing")          		   
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--num_envs", type=int, default=4096)
    p.add_argument("--rollout_steps", type=int, default=16)
    p.add_argument("--total_steps", type=int, default=30000000)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--POMDP", default="flicker")
    p.add_argument("--pomdp_prob", type=float, default=0.1)
    
    args = p.parse_args()
    return args

if __name__=="__main__":
    
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    writer = SummaryWriter(f"../runs/PPO_Critic_{args.POMDP}_{args.pomdp_prob}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    envs = isaacgymenvs.make(
        seed=0, 
        task=args.env, 
        num_envs=args.num_envs, 
        sim_device="cuda:0",
        rl_device="cuda:0",
        graphics_device_id=-1,
        headless=True,
        force_render=True,

    )

    # Environment Setup
    envs = ExtractObsWrapper(envs)
    envs = RecordEpisodeStatisticsTorch(envs, device)
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    
    # Partial Observability
    POMDP = POMDPWrapper(pomdp=args.POMDP, pomdp_prob=args.pomdp_prob)

    print("\n------------------------------------\n")
    print(f"PPO_Critic_{args.POMDP}_{args.pomdp_prob}")

    # Agent Setup
    agent = PPO(envs.single_observation_space, envs.single_action_space, envs.num_envs, device)

    # Storage setup
    obs = torch.zeros((args.rollout_steps, args.num_envs) + envs.single_observation_space.shape, dtype=torch.float).to(device)
    pomdps = torch.zeros((args.rollout_steps, args.num_envs) + envs.single_observation_space.shape, dtype=torch.float).to(device)
    actions = torch.zeros((args.rollout_steps, args.num_envs) + envs.single_action_space.shape, dtype=torch.float).to(device)
    logprobs = torch.zeros((args.rollout_steps, args.num_envs), dtype=torch.float).to(device)
    rewards = torch.zeros((args.rollout_steps, args.num_envs), dtype=torch.float).to(device)
    dones = torch.zeros((args.rollout_steps, args.num_envs), dtype=torch.float).to(device)
    values = torch.zeros((args.rollout_steps, args.num_envs), dtype=torch.float).to(device)
    advantages = torch.zeros_like(rewards, dtype=torch.float).to(device)

    global_step = 0
    max_reward = 0
    next_obs = envs.reset()
    pomdp = next_obs.to(device)
    next_done = torch.zeros(args.num_envs, dtype=torch.float).to(device)

    while global_step <= args.total_steps:
        for step in range(0, args.rollout_steps):
            global_step += envs.num_envs
            obs[step] = next_obs
            pomdps[step] = pomdp
            dones[step] = next_done
            
            with torch.no_grad():
                action, logprob, _, = agent.getAction(pomdp)
            actions[step] = action
            logprobs[step] = logprob
            
            next_obs, rewards[step], next_done, info = envs.step(action)
            pomdp = POMDP.observation(next_obs)

            if 0 <= step <= 2:
                for idx, d in enumerate(next_done):
                    if d:
                        episodic_return = info["r"][idx].item()
                        writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                        writer.add_scalar("charts/episodic_length", info["l"][idx], global_step)
        
        agent.train(obs, pomdps, actions, next_obs, next_done, logprobs, rewards, dones)
        mean_rewards = torch.mean(rewards)
        writer.add_scalar("average/average_reward", mean_rewards, global_step)
        print(f"Step: {global_step}, Average rewards {mean_rewards}")
        if mean_rewards > max_reward:
            max_reward = mean_rewards
            agent.save(f'./checkpoints/best_{args.POMDP}_{args.pomdp_prob}')
    
    agent.save(f"./checkpoints/PPO_Critic_{args.POMDP}_{args.pomdp_prob}")
    writer.close()