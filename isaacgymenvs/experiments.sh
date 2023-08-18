#!/bin/bash

clear
cd PPO
echo "Starting Flicker Experiments"
python main.py --POMDP=flicker --pomdp_prob=0.1
python main.py --POMDP=flicker --pomdp_prob=0.2
python main.py --POMDP=flicker --pomdp_prob=0.3
python main.py --POMDP=flicker --pomdp_prob=0.4
python main.py --POMDP=flicker --pomdp_prob=0.5

clear 
echo "Staring Noise Experiments"
python main.py --POMDP=random_noise --pomdp_prob=0.05
python main.py --POMDP=random_noise --pomdp_prob=0.1
python main.py --POMDP=random_noise --pomdp_prob=0.15
python main.py --POMDP=random_noise --pomdp_prob=0.2
python main.py --POMDP=random_noise --pomdp_prob=0.25
python main.py --POMDP=random_noise --pomdp_prob=0.23

clear 
echo "Staring Flicker + Noise Experiments"
python main.py --POMDP=flickering_and_random_noise --pomdp_prob=0.05
python main.py --POMDP=flickering_and_random_noise --pomdp_prob=0.1
python main.py --POMDP=flickering_and_random_noise --pomdp_prob=0.15
python main.py --POMDP=flickering_and_random_noise --pomdp_prob=0.2
python main.py --POMDP=flickering_and_random_noise --pomdp_prob=0.25

clear
cd ../
cd RPO
echo "Starting Flicker Experiments"
python main.py --POMDP=flicker --pomdp_prob=0.1
python main.py --POMDP=flicker --pomdp_prob=0.2
python main.py --POMDP=flicker --pomdp_prob=0.3
python main.py --POMDP=flicker --pomdp_prob=0.4
python main.py --POMDP=flicker --pomdp_prob=0.5

clear 
echo "Staring Noise Experiments"
python main.py --POMDP=random_noise --pomdp_prob=0.05
python main.py --POMDP=random_noise --pomdp_prob=0.1
python main.py --POMDP=random_noise --pomdp_prob=0.15
python main.py --POMDP=random_noise --pomdp_prob=0.2
python main.py --POMDP=random_noise --pomdp_prob=0.25
python main.py --POMDP=random_noise --pomdp_prob=0.23

clear 
echo "Staring Flicker + Noise Experiments"
python main.py --POMDP=flickering_and_random_noise --pomdp_prob=0.05
python main.py --POMDP=flickering_and_random_noise --pomdp_prob=0.1
python main.py --POMDP=flickering_and_random_noise --pomdp_prob=0.15
python main.py --POMDP=flickering_and_random_noise --pomdp_prob=0.2
python main.py --POMDP=flickering_and_random_noise --pomdp_prob=0.25

clear
cd ../
cd PPO_Critic
echo "Starting Flicker Experiments"
python main.py --POMDP=flicker --pomdp_prob=0.1
python main.py --POMDP=flicker --pomdp_prob=0.2
python main.py --POMDP=flicker --pomdp_prob=0.3
python main.py --POMDP=flicker --pomdp_prob=0.4
python main.py --POMDP=flicker --pomdp_prob=0.5

clear 
echo "Staring Noise Experiments"
python main.py --POMDP=random_noise --pomdp_prob=0.05
python main.py --POMDP=random_noise --pomdp_prob=0.1
python main.py --POMDP=random_noise --pomdp_prob=0.15
python main.py --POMDP=random_noise --pomdp_prob=0.2
python main.py --POMDP=random_noise --pomdp_prob=0.25
python main.py --POMDP=random_noise --pomdp_prob=0.23

clear 
echo "Staring Flicker + Noise Experiments"
python main.py --POMDP=flickering_and_random_noise --pomdp_prob=0.05
python main.py --POMDP=flickering_and_random_noise --pomdp_prob=0.1
python main.py --POMDP=flickering_and_random_noise --pomdp_prob=0.15
python main.py --POMDP=flickering_and_random_noise --pomdp_prob=0.2
python main.py --POMDP=flickering_and_random_noise --pomdp_prob=0.25

clear
cd ../
cd RPO_Critic
echo "Starting Flicker Experiments"
python main.py --POMDP=flicker --pomdp_prob=0.1
python main.py --POMDP=flicker --pomdp_prob=0.2
python main.py --POMDP=flicker --pomdp_prob=0.3
python main.py --POMDP=flicker --pomdp_prob=0.4
python main.py --POMDP=flicker --pomdp_prob=0.5

clear 
echo "Staring Noise Experiments"
python main.py --POMDP=random_noise --pomdp_prob=0.05
python main.py --POMDP=random_noise --pomdp_prob=0.1
python main.py --POMDP=random_noise --pomdp_prob=0.15
python main.py --POMDP=random_noise --pomdp_prob=0.2
python main.py --POMDP=random_noise --pomdp_prob=0.25
python main.py --POMDP=random_noise --pomdp_prob=0.23

clear 
echo "Staring Flicker + Noise Experiments"
python main.py --POMDP=flickering_and_random_noise --pomdp_prob=0.05
python main.py --POMDP=flickering_and_random_noise --pomdp_prob=0.1
python main.py --POMDP=flickering_and_random_noise --pomdp_prob=0.15
python main.py --POMDP=flickering_and_random_noise --pomdp_prob=0.2
python main.py --POMDP=flickering_and_random_noise --pomdp_prob=0.25

clear
cd ../
cd PPO-LSTM
echo "Starting Flicker Experiments"
python main.py --POMDP=flicker --pomdp_prob=0.1
python main.py --POMDP=flicker --pomdp_prob=0.2
python main.py --POMDP=flicker --pomdp_prob=0.3
python main.py --POMDP=flicker --pomdp_prob=0.4
python main.py --POMDP=flicker --pomdp_prob=0.5

clear 
echo "Staring Noise Experiments"
python main.py --POMDP=random_noise --pomdp_prob=0.05
python main.py --POMDP=random_noise --pomdp_prob=0.1
python main.py --POMDP=random_noise --pomdp_prob=0.15
python main.py --POMDP=random_noise --pomdp_prob=0.2
python main.py --POMDP=random_noise --pomdp_prob=0.25
python main.py --POMDP=random_noise --pomdp_prob=0.23

clear 
echo "Staring Flicker + Noise Experiments"
python main.py --POMDP=flickering_and_random_noise --pomdp_prob=0.05
python main.py --POMDP=flickering_and_random_noise --pomdp_prob=0.1
python main.py --POMDP=flickering_and_random_noise --pomdp_prob=0.15
python main.py --POMDP=flickering_and_random_noise --pomdp_prob=0.2
python main.py --POMDP=flickering_and_random_noise --pomdp_prob=0.25

clear
cd ../
cd RPO-LSTM
echo "Starting Flicker Experiments"
python main.py --POMDP=flicker --pomdp_prob=0.1
python main.py --POMDP=flicker --pomdp_prob=0.2
python main.py --POMDP=flicker --pomdp_prob=0.3
python main.py --POMDP=flicker --pomdp_prob=0.4
python main.py --POMDP=flicker --pomdp_prob=0.5

clear 
echo "Staring Noise Experiments"
python main.py --POMDP=random_noise --pomdp_prob=0.05
python main.py --POMDP=random_noise --pomdp_prob=0.1
python main.py --POMDP=random_noise --pomdp_prob=0.15
python main.py --POMDP=random_noise --pomdp_prob=0.2
python main.py --POMDP=random_noise --pomdp_prob=0.25
python main.py --POMDP=random_noise --pomdp_prob=0.23

clear 
echo "Staring Flicker + Noise Experiments"
python main.py --POMDP=flickering_and_random_noise --pomdp_prob=0.05
python main.py --POMDP=flickering_and_random_noise --pomdp_prob=0.1
python main.py --POMDP=flickering_and_random_noise --pomdp_prob=0.15
python main.py --POMDP=flickering_and_random_noise --pomdp_prob=0.2
python main.py --POMDP=flickering_and_random_noise --pomdp_prob=0.25

clear
cd ../
cd RPO-LSTM_Critic
echo "Starting Flicker Experiments"
python main.py --POMDP=flicker --pomdp_prob=0.1
python main.py --POMDP=flicker --pomdp_prob=0.2
python main.py --POMDP=flicker --pomdp_prob=0.3
python main.py --POMDP=flicker --pomdp_prob=0.4
python main.py --POMDP=flicker --pomdp_prob=0.5

clear 
echo "Staring Noise Experiments"
python main.py --POMDP=random_noise --pomdp_prob=0.05
python main.py --POMDP=random_noise --pomdp_prob=0.1
python main.py --POMDP=random_noise --pomdp_prob=0.15
python main.py --POMDP=random_noise --pomdp_prob=0.2
python main.py --POMDP=random_noise --pomdp_prob=0.25
python main.py --POMDP=random_noise --pomdp_prob=0.23

clear 
echo "Staring Flicker + Noise Experiments"
python main.py --POMDP=flickering_and_random_noise --pomdp_prob=0.05
python main.py --POMDP=flickering_and_random_noise --pomdp_prob=0.1
python main.py --POMDP=flickering_and_random_noise --pomdp_prob=0.15
python main.py --POMDP=flickering_and_random_noise --pomdp_prob=0.2
python main.py --POMDP=flickering_and_random_noise --pomdp_prob=0.25