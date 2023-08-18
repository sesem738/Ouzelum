import torch


class POMDPWrapper():
    def __init__(self, pomdp='flicker', pomdp_prob = 0.1):
        
        self.pomdp = pomdp
        self.flicker_prob = pomdp_prob
        self.random_noise_sigma = pomdp_prob
        self.range = (1-self.random_noise_sigma, 1 + self.random_noise_sigma)

        if self.pomdp == 'flicker':
            self.prob = self.flicker_prob
        elif self.pomdp == "random_noise":
            self.prob = self.random_noise_sigma
        elif self.pomdp == "flickering_and_random_noise":
            self.flicker_prob = 0.1
            self.prob = pomdp_prob
        else:
            raise ValueError("pomdp was not in ['remove_velocity', 'flickering', 'random_noise', 'random_sensor_missing']!")


    def observation(self, obs):
        if self.pomdp == 'flicker':
            if torch.rand(1) <= self.flicker_prob:
                return torch.zeros(obs.shape).to("cuda:0")
            else:
                return obs
        elif self.pomdp == "random_noise":
            noise = torch.FloatTensor(*obs.shape).uniform_(*self.range)
            return (obs * noise.to("cuda:0"))
        elif self.pomdp == 'flickering_and_random_noise':
            # Flickering
            if torch.rand(1) <= self.flicker_prob:
                new_obs = torch.zeros(obs.shape).to("cuda:0")
            else:
                new_obs = obs.to("cuda:0")
            noise = torch.FloatTensor(*obs.shape).uniform_(*self.range)
            # Add random noise
            return (new_obs * noise.to("cuda:0"))
        else:
            raise ValueError("POMDP was not in ['flicker_random', 'flicker_duration', 'flicker_freq', 'random_noise']!")

