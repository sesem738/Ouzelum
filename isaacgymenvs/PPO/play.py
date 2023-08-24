import argparse

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
    p.add_argument("--env", default="Lando")          		   
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--num_envs", type=int, default=1)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    writer = SummaryWriter()

    envs = isaacgymenvs.make(
        seed=0, 
        task=args.env, 
        num_envs=args.num_envs, 
        sim_device="cuda:0",
        rl_device="cuda:0",
        graphics_device_id=0,
        headless=False,
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
    print(f"PPO_{args.POMDP}_{args.pomdp_prob}")

    # Agent Setup
    agent = PPO(envs.single_observation_space, envs.single_action_space, envs.num_envs, device)
    agent.load('checkpoints/PPO_best_flicker_0.3')

    # Storage setup
    rewards = torch.zeros((args.rollout_steps, args.num_envs), dtype=torch.float).to(device)
    
    global_step = 0
    next_obs = envs.reset()
    next_done = torch.zeros(args.num_envs, dtype=torch.float).to(device)

    while global_step <= args.total_steps:
        global_step += envs.num_envs
        obs = next_obs
        dones = next_done
        
        with torch.no_grad():
            action, logprob, _, = agent.getAction(next_obs)
        actions = action
        logprobs = logprob
        
        next_obs, rewards, next_done, info = envs.step(action)
        next_obs = POMDP.observation(next_obs)
        
        mean_rewards = torch.mean(rewards)
        writer.add_scalar("reward/play", mean_rewards, global_step)
        print(f"Step: {global_step}, Average rewards {mean_rewards}")
    

    writer.close()
        
