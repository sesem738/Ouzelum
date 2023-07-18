#!/bin/bash

clear
echo "Starting Flicker Experiments"

python train.py task=Landing train=LandingPPOLSTM headless=True experiment=Flicker_0.1 max_iterations=750
sed -i sed -i 's/flicker_prob=0.1/flicker_prob=0.2/g' tasks/landing.py
python train.py task=Landing train=LandingPPOLSTM headless=True experiment=Flicker_0.2 max_iterations=750
sed -i sed -i 's/flicker_prob=0.2/flicker_prob=0.3/g' tasks/landing.py
python train.py task=Landing train=LandingPPOLSTM headless=True experiment=Flicker_0.3 max_iterations=750
sed -i sed -i 's/flicker_prob=0.3/flicker_prob=0.4/g' tasks/landing.py
python train.py task=Landing train=LandingPPOLSTM headless=True experiment=Flicker_0.4 max_iterations=750
sed -i sed -i 's/flicker_prob=0.4/flicker_prob=0.5/g' tasks/landing.py
python train.py task=Landing train=LandingPPOLSTM headless=True experiment=Flicker_0.5 max_iterations=750

clear 
echo "Staring Noise Experiments"

sed -i "s/pomdp='flicker'/pomdp='random_noise'/g" tasks/landing.py
sed -i sed -i 's/flicker_prob=0.4/random_noise_sigma=0.05/g' tasks/landing.py
python train.py task=Landing train=LandingPPOLSTM headless=True experiment=Noise_0.05 max_iterations=750
sed -i sed -i 's/random_noise_sigma=0.05/random_noise_sigma=0.08/g' tasks/landing.py
python train.py task=Landing train=LandingPPOLSTM headless=True experiment=Noise_0.08 max_iterations=750
sed -i sed -i 's/random_noise_sigma=0.08/random_noise_sigma=0.1/g' tasks/landing.py
python train.py task=Landing train=LandingPPOLSTM headless=True experiment=Noise_0.1 max_iterations=750
sed -i sed -i 's/random_noise_sigma=0.1/random_noise_sigma=0.15/g' tasks/landing.py
python train.py task=Landing train=LandingPPOLSTM headless=True experiment=Noise_0.15 max_iterations=750
sed -i sed -i 's/random_noise_sigma=0.15/random_noise_sigma=0.2/g' tasks/landing.py
python train.py task=Landing train=LandingPPOLSTM headless=True experiment=Noise_0.2 max_iterations=750


clear 
echo "Staring Noise Experiments"

sed -i "s/pomdp='random_noise'/pomdp='random_sensor_missing'/g" tasks/landing.py
sed -i sed -i 's/random_noise_sigma=0.2/random_sensor_missing_prob=0.05/g' tasks/landing.py
python train.py task=Landing train=LandingPPOLSTM headless=True experiment=Random_0.05 max_iterations=750
sed -i sed -i 's/random_sensor_missing_prob=0.05/random_sensor_missing_prob=0.1/g' tasks/landing.py
python train.py task=Landing train=LandingPPOLSTM headless=True experiment=Random_0.1 max_iterations=750
sed -i sed -i 's/random_sensor_missing_prob=0.1/random_sensor_missing_prob=0.15/g' tasks/landing.py
python train.py task=Landing train=LandingPPOLSTM headless=True experiment=Random_0.15 max_iterations=750
sed -i sed -i 's/random_sensor_missing_prob=0.15/random_sensor_missing_prob=0.2/g' tasks/landing.py
python train.py task=Landing train=LandingPPOLSTM headless=True experiment=Random_0.2 max_iterations=750
