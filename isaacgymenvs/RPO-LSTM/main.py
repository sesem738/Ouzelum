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

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="Lando")          		   
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--num_envs", type=int, default=4096)
    p.add_argument("--rollout_steps", type=int, default=16)
    p.add_argument("--total_steps", type=int, default=30000000)
    p.add_argument("--headless", action="store_true")
    
    args = p.parse_args()
    return args

if __name__=="__main__":
    
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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

    # Agent Setup
    agent = PPO(envs.single_observation_space, envs.single_action_space, envs.num_envs, device)

    # Storage setup
    obs = torch.zeros((args.rollout_steps, args.num_envs) + envs.single_observation_space.shape, dtype=torch.float).to(device)
    actions = torch.zeros((args.rollout_steps, args.num_envs) + envs.single_action_space.shape, dtype=torch.float).to(device)
    logprobs = torch.zeros((args.rollout_steps, args.num_envs), dtype=torch.float).to(device)
    rewards = torch.zeros((args.rollout_steps, args.num_envs), dtype=torch.float).to(device)
    dones = torch.zeros((args.rollout_steps, args.num_envs), dtype=torch.float).to(device)
    values = torch.zeros((args.rollout_steps, args.num_envs), dtype=torch.float).to(device)
    advantages = torch.zeros_like(rewards, dtype=torch.float).to(device)

    global_step = 0
    max_reward = 0
    next_obs = envs.reset()
    next_done = torch.zeros(args.num_envs, dtype=torch.float).to(device)
    next_lstm_state = (
        torch.zeros(agent.actor.lstm.num_layers, args.num_envs, agent.actor.lstm.hidden_size).to(device),
        torch.zeros(agent.actor.lstm.num_layers, args.num_envs, agent.actor.lstm.hidden_size).to(device),
    )

    while global_step <= args.total_steps:
        initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())
        for step in range(0, args.rollout_steps):
            global_step += envs.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            
            with torch.no_grad():
                action, logprob, _, next_lstm_state = agent.getAction(next_obs, next_lstm_state, next_done)
            actions[step] = action
            logprobs[step] = logprob
            
            next_obs, rewards[step], next_done, info = envs.step(action)
        
        agent.train(obs, actions, next_obs, next_done, initial_lstm_state, logprobs, rewards, dones)
        mean_rewards = torch.mean(rewards)
        print(f"Step: {global_step}, Average rewards {mean_rewards}")
        if mean_rewards > max_reward:
            max_reward = mean_rewards
            agent.save('best_reward')
    
    print("Max Reward: ", max_reward)
    agent.save('PPO_06_10_1330')

        
