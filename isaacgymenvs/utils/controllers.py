import torch
import numpy as np


def map_to_pi(angle):
    angle = torch.where(angle > np.pi, angle - (2 * np.pi), angle)
    angle = torch.where(angle <= -np.pi, angle + (2 * np.pi), angle)

    out_of_bounds = (angle > np.pi) | (angle < -np.pi)
    if (out_of_bounds).any():
        raise RuntimeError(f'Angle out of bounds: {angle[out_of_bounds]}')

    return angle


def differential_drive(current_pos, target_pos, current_heading, p_gain: tuple([float, float]) = (0.5, 10),
                       ang_thresh: float = 0.005):
    # husky parameters
    wheel_base = 0.54
    wheel_radius = 0.165
    max_speed = 15

    # Position and heading difference
    dx = target_pos[:, 0] - current_pos[:, 0]
    dy = target_pos[:, 1] - current_pos[:, 1]
    dtheta = map_to_pi(torch.atan2(dy, dx) - map_to_pi(current_heading))
    dtheta[torch.where((dtheta < ang_thresh) & (dtheta > -ang_thresh))] = 0.0

    # Calculate linear and angular velocities
    linear_velocities = torch.sqrt(dx ** 2 + dy ** 2) * p_gain[0]
    angular_velocities = dtheta * p_gain[1]

    # Calculate individual wheel speeds for a differential drive robot
    left_speeds = (2 * linear_velocities + angular_velocities * wheel_base) / (2 * wheel_radius)
    right_speeds = (2 * linear_velocities - angular_velocities * wheel_base) / (2 * wheel_radius)

    # Scale speeds if necessary
    max_wheel_speed = max(abs(left_speeds), abs(right_speeds))
    if max_wheel_speed > max_speed:
        scaling_factor = max_speed / max_wheel_speed
        left_speeds *= scaling_factor
        right_speeds *= scaling_factor

    return torch.stack((right_speeds, left_speeds, right_speeds, left_speeds), dim=-1)