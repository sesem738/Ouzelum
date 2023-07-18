#!/bin/bash

clear
echo "Starting Flicker Experiments"

python train.py task=Landed test=True headless=True checkpoint=runs/Flicker_0.1/nn/Flicker_0.1.pth
sed -i sed -i 's/flicker_prob=0.1/flicker_prob=0.2/g' tasks/landed.py
python train.py task=Landed test=True headless=True checkpoint=runs/Flicker_0.2/nn/Flicker_0.2.pth
sed -i sed -i 's/flicker_prob=0.2/flicker_prob=0.3/g' tasks/landed.py
python train.py task=Landed test=True headless=True checkpoint=runs/Flicker_0.3/nn/Flicker_0.3.pth
sed -i sed -i 's/flicker_prob=0.3/flicker_prob=0.4/g' tasks/landed.py
python train.py task=Landed test=True headless=True checkpoint=runs/Flicker_0.4/nn/Flicker_0.4.pth
sed -i sed -i 's/flicker_prob=0.4/flicker_prob=0.5/g' tasks/landed.py
python train.py task=Landed test=True headless=True checkpoint=runs/Flicker_0.5/nn/Flicker_0.5.pth

clear 
echo "Staring Noise Experiments"

sed -i "s/pomdp='flicker'/pomdp='random_noise'/g" tasks/landed.py
sed -i sed -i 's/flicker_prob=0.4/random_noise_sigma=0.05/g' tasks/landed.py
python train.py task=Landed test=True headless=True checkpoint=runs/Noise_0.05/nn/Noise_0.05.pth
sed -i sed -i 's/random_noise_sigma=0.05/random_noise_sigma=0.08/g' tasks/landed.py
python train.py task=Landed test=True headless=True checkpoint=runs/Noise_0.08/nn/Noise_0.08.pth
sed -i sed -i 's/random_noise_sigma=0.08/random_noise_sigma=0.1/g' tasks/landed.py
python train.py task=Landed test=True headless=True checkpoint=runs/Noise_0.1/nn/Noise_0.1.pth
sed -i sed -i 's/random_noise_sigma=0.1/random_noise_sigma=0.15/g' tasks/landed.py
python train.py task=Landed test=True headless=True checkpoint=runs/Noise_0.15/nn/Noise_0.15.pth
sed -i sed -i 's/random_noise_sigma=0.15/random_noise_sigma=0.2/g' tasks/landed.py
python train.py task=Landed test=True headless=True checkpoint=runs/Noise_0.2/nn/Noise_0.2.pth

clear 
echo "Staring Noise Experiments"

sed -i "s/pomdp='random_noise'/pomdp='random_sensor_missing'/g" tasks/landed.py
sed -i sed -i 's/random_noise_sigma=0.2/random_sensor_missing_prob=0.05/g' tasks/landed.py
python train.py task=Landed test=True headless=True checkpoint=runs/Random_0.05/nn/Random_0.05.pth
sed -i sed -i 's/random_sensor_missing_prob=0.05/random_sensor_missing_prob=0.1/g' tasks/landed.py
python train.py task=Landed test=True headless=True checkpoint=runs/Random_0.1/nn/Random_0.1.pth
sed -i sed -i 's/random_sensor_missing_prob=0.1/random_sensor_missing_prob=0.15/g' tasks/landed.py
python train.py task=Landed test=True headless=True checkpoint=runs/Random_0.15/nn/Random_0.15.pth
sed -i sed -i 's/random_sensor_missing_prob=0.15/random_sensor_missing_prob=0.2/g' tasks/landed.py
python train.py task=Landed test=True headless=True checkpoint=runs/Random_0.2/nn/Random_0.2.pth