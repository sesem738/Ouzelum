import numpy as np
from isaacgym import gymapi

def pd_controller(env, robot_handle, current_pos, target_pos):
    Kp = 0.5  # Proportional gain
    Kd = 0.2  # Derivative gain
    max_vel = 10  # Maximum wheel velocity

    # Get the current wheel velocities
    wheel_velocities = gymapi.get_actor_dof_velocity(env, robot_handle)

    # Compute the error and the derivative error
    error = target_pos - current_pos
    derivative_error = -wheel_velocities

    # Compute the output
    output = Kp * error + Kd * derivative_error

    # Cap the output at the maximum wheel velocity
    output = np.clip(output, -max_vel, max_vel)

    # Set the wheel velocities to the output
    gymapi.set_actor_dof_velocity_targets(env, robot_handle, output)

    # Step the simulation to apply the velocity targets
    gymapi.simulate(env)

    return wheel_velocities