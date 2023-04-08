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


if __name__ == "__main__":
    
    eval_env = isaacgymenvs.make(
        seed=0,
        task="Ingenuity",
        num_envs=2048,
        sim_device="cuda:0",
        rl_device="cuda:0",
        headless=False,
        multi_gpu=False,
        virtual_screen_capture=False,
        force_render=False,
    )

    for _ in range(10):
        state,