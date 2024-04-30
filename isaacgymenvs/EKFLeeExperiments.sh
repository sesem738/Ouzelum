
clear 
echo "Staring Noise Experiments"
python ./isaacgymenvs/train.py task=EKFLeeLanded num_envs=512 test=True headless=True max_iterations=1000 +POMDP=flicker +pomdp_prob=0.0 
python ./isaacgymenvs/train.py task=EKFLeeLanded num_envs=512 test=True headless=True +POMDP=flicker +pomdp_prob=0.3 max_iterations=1000
python ./isaacgymenvs/train.py task=EKFLeeLanded num_envs=512 test=True headless=True +POMDP=flicker +pomdp_prob=0.4 max_iterations=1000
python ./isaacgymenvs/train.py task=EKFLeeLanded num_envs=512 test=True headless=True +POMDP=flicker +pomdp_prob=0.5 max_iterations=1000

clear 
echo "Staring Noise Experiments"
python ./isaacgymenvs/train.py task=EKFLeeLanded num_envs=512 test=True headless=True +POMDP=random_noise +pomdp_prob=0.15 max_iterations=1000
python ./isaacgymenvs/train.py task=EKFLeeLanded num_envs=512 test=True headless=True +POMDP=random_noise +pomdp_prob=0.20 max_iterations=1000
python ./isaacgymenvs/train.py task=EKFLeeLanded num_envs=512 test=True headless=True +POMDP=random_noise +pomdp_prob=0.25 max_iterations=1000

clear 
echo "Staring Flicker + Noise Experiments"
python ./isaacgymenvs/train.py task=EKFLeeLanded num_envs=512 test=True headless=True +POMDP=flickering_and_random_noise +pomdp_prob=0.15 max_iterations=1000
python ./isaacgymenvs/train.py task=EKFLeeLanded num_envs=512 test=True headless=True +POMDP=flickering_and_random_noise +pomdp_prob=0.20 max_iterations=1000
python ./isaacgymenvs/train.py task=EKFLeeLanded num_envs=512 test=True headless=True +POMDP=flickering_and_random_noise +pomdp_prob=0.25 max_iterations=1000
