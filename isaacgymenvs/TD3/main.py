import os
import time
import argparse
import numpy as np
from distutils.util import strtobool


import gym
import isaacgym
import isaacgymenvs
import torch
import utils
from agent import TD3
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.buffers import ReplayBuffer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="Ingenuity")               # Isaac Gym environment name
    parser.add_argument("--num_envs", default=2048, type=int)       # Number of Parallel Environments
    parser.add_argument("--cuda", default=True, type=bool)          # CUDA
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--learning_starts", default=25e3, type=int)# Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--total_timesteps", default=1e6, type=int) # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--buffer_size", default=1e6, type=int)     # Buffer size for replay buffer
    parser.add_argument("--discount", default=0.99)                 # Discount factor
    parser.add_argument("--tau", default=0.005)                     # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True) 
    args = parser.parse_args()
    
    file_name = f"{args.policy}_{args.env}_{args.seed}"
    run_name = f"{args.env}__{args.policy}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    

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
    kwargs["policy_freq"] = args.policy_freq
    
    agent = TD3(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        agent.load(f"./models/{policy_file}")

    envs.single_observation_space.dtype = np.float32

    rb = ReplayBuffer(
        int(args.buffer_size),
        envs.single_observation_space,
        envs.single_action_space,
        "cuda",
        n_envs=int(args.num_envs),
        handle_timeout_termination=True,
    )
    
    start_time = time.time()
    
    obs = envs.reset()
    
    for global_step in range(int(args.total_timesteps)):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            actions = torch.from_numpy(actions)
        else:
            actions = agent.select_action(obs)

        # Execute the game and log data.
        next_obs, rewards, dones, infos = envs.step(actions)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = torch.clone(next_obs)
        for idx, d in enumerate(dones):
            if d:
                real_next_obs[idx] = infos[idx]["terminal_observation"]
                episodic_return = infos["r"][idx].item()
                print(f"global_step={global_step}, episodic_return={episodic_return}")
                writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                writer.add_scalar("charts/episodic_length", infos["l"][idx], global_step)
                if "consecutive_successes" in infos:  # ShadowHand and AllegroHand metric
                    writer.add_scalar(
                        "charts/consecutive_successes", infos["consecutive_successes"].item(), global_step
                    )
                break
        rb.add(obs, real_next_obs, actions, rewards, dones, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        if global_step > args.learning_starts:
            agent.train(rb, args.batch_size)

    envs.close()
    writer.close()
