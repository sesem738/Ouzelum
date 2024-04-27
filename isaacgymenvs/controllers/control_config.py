class control:
    """
    Control parameters
    controller:
        lee_position_control: command_actions = [x, y, z, yaw] in environment frame scaled between -1 and 1
        lee_velocity_control: command_actions = [vx, vy, vz, yaw_rate] in vehicle frame scaled between -1 and 1
        lee_attitude_control: command_actions = [thrust, roll, pitch, yaw_rate] in vehicle frame scaled between -1 and 1
    kP: gains for position
    kV: gains for velocity
    kR: gains for attitude
    kOmega: gains for angular velocity
    """
    controller = "lee_position_control" # or "lee_velocity_control" or "lee_attitude_control"
    kP = [0.8, 0.8, 1.0] # used for lee_position_control only
    kV = [0.5, 0.5, 0.4] # used for lee_position_control, lee_velocity_control only
    kR = [3.0, 3.0, 1.0] # used for lee_position_control, lee_velocity_control and lee_attitude_control
    kOmega = [0.5, 0.5, 1.20] # used for lee_position_control, lee_velocity_control and lee_attitude_control
    scale_input = [1.0, 1.0, 1.0, 1.0] # scale the input to the controller from -1 to 1 for each dimension, scale from -np.pi to np.pi for yaw in the case of position control
