import torch


class POMDPWrapper():
    def __init__(self, pomdp='flicker', flicker_prob=0.1, 
                 random_noise_sigma=0.1, randon_sensor_missing_prob=0.1):
        
        self.pomdp = pomdp
        self.flicker_prob = flicker_prob
        self.random_noise_sigma = random_noise_sigma
        self.random_sensor_missing_prob = randon_sensor_missing_prob

    def observation(self, obs):
        if self.pomdp == 'flicker':
            if torch.rand(1) <= self.flicker_prob:
                return torch.zeros(obs.shape)
            else:
                return obs
        elif self.pomdp == "random_noise":
            return (obs + torch.randn(0, self.random_noise_sigma, obs.shape))
        elif self.pomdp == "random_sensor_missing":
            obs[torch.rand(len(obs)) <= self.random_sensor_missing_prob] = 0
            return obs
        elif self.pomdp == 'flickering_and_random_noise':
            # Flickering
            if torch.rand() <= self.flicker_prob:
                new_obs = torch.zeros(obs.shape)
            else:
                new_obs = obs
            # Add random noise
            return (new_obs + torch.randn(0, self.random_noise_sigma, new_obs.shape))
        elif self.pomdp == 'random_noise_and_random_sensor_missing':
            # Random noise
            new_obs = (obs + torch.randn(0, self.random_noise_sigma, obs.shape))
            # Random sensor missing
            new_obs[torch.rand(len(new_obs)) <= self.random_sensor_missing_prob] = 0
            return new_obs
        elif self.pomdp == 'random_sensor_missing_and_random_noise':
            # Random sensor missing
            obs[torch.rand(len(obs)) <= self.random_sensor_missing_prob] = 0
            # Random noise
            return (obs + torch.randn(0, self.random_noise_sigma, obs.shape))
        else:
            raise ValueError("pomdp was not in ['remove_velocity', 'flickering', 'random_noise', 'random_sensor_missing']!")

