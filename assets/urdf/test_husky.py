import os
import random
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
import math

import torch

gym = gymapi.acquire_gym()

from getMotorSpeeds import *

args = gymutil.parse_arguments(
    description="Test husky",
    headless=True,
    custom_parameters=[
        {"name": "--save_images", "action": "store_true", "help": "Store Images To Disk"}])




# get default set of parameters
sim_params = gymapi.SimParams()


# set common parameters
sim_params.dt = 1 / 60
sim_params.substeps = 2
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

# set PhysX-specific parameters
# sim_params.physx.use_gpu = True
# sim_params.physx.solver_type = 1
# sim_params.physx.num_position_iterations = 6
# sim_params.physx.num_velocity_iterations = 1
# sim_params.physx.contact_offset = 0.01
# sim_params.physx.rest_offset = 0.0

# set Flex-specific parameters
# sim_params.flex.solver_type = 5
# sim_params.flex.num_outer_iterations = 4
# sim_params.flex.num_inner_iterations = 20
# sim_params.flex.relaxation = 0.8
# sim_params.flex.warm_start = 0.5

# create sim with these parameters
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

# configure the ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
plane_params.distance = 0
plane_params.static_friction = 1
plane_params.dynamic_friction = 1
plane_params.restitution = 0

# create the ground plane
gym.add_ground(sim, plane_params)

asset_root = "../../assets"
asset_file = "urdf/husky_description/urdf/husky.urdf"
# asset_file = "urdf/franka_description/robots/franka_panda.urdf"
asset = gym.load_asset(sim, asset_root, asset_file)

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = False
asset_options.flip_visual_attachments = True
asset_options.armature = 0.01

asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
num_bodies = gym.get_asset_rigid_body_count(asset)


# set up the env grid
num_envs = 1
envs_per_row = 8
env_spacing = 2.0
env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

# cache some common handles for later use
envs = []
actor_handles = []

Kp = 100.0
Kd = 100.0

# create and populate the environments
for i in range(num_envs):
    env = gym.create_env(sim, env_lower, env_upper, envs_per_row)
    envs.append(env)

    height = random.uniform(1.0, 2.5)

    pose = gymapi.Transform()
    pose.p.z = 1.0
    # pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
    # pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

    actor_handle = gym.create_actor(env, asset, pose, "Husky", i, 1)
    actor_handles.append(actor_handle)

    dof_props = gym.get_actor_dof_properties(env, actor_handle)
    dof_props['driveMode'][:] = gymapi.DOF_MODE_VEL
    dof_props['stiffness'][:] = Kp
    dof_props['damping'][:] = Kd
    dof_props['hasLimits'][:] = False
    gym.set_actor_dof_properties(env, actor_handle, dof_props)

    base_idx = gym.find_actor_rigid_body_index(env, actor_handle, "base_link", gymapi.DOMAIN_SIM)


#ADD VIEWER
cam_props = gymapi.CameraProperties()
cam_target = gymapi.Vec3(-0.8, 0.5, 0)
viewer = gym.create_viewer(sim, cam_props)

_rb_states = gym.acquire_rigid_body_state_tensor(sim)
rb_states = gymtorch.wrap_tensor(_rb_states)



force_offset = 0.2
frame_count = 0


wheel_radius = 0.098
chasis_width = 0.187795*2.0
action_scale = 1.5
action_transfrom = torch.tensor([
        [1.0/wheel_radius,                     1.0/wheel_radius], 
        [-0.5*chasis_width/wheel_radius,  0.5*chasis_width/wheel_radius]],dtype=torch.float)




while not gym.query_viewer_has_closed(viewer):

    gym.refresh_rigid_body_state_tensor(sim)

    
    pos = rb_states[base_idx, :2]

    r,l = getMotorSpeeds(pos[0], pos[1], 15, 15, 0.1)

    # print("POS :", pos)

    # print()
    

    actions = torch.rand((num_envs*2, 2), dtype=torch.float32)*random.uniform(0, 500)
    # print("Raw: ", actions)
    actions_tensor = actions.clone()
    actions_tensor[:, 0] = torch.clamp(actions_tensor[:, 0], 0.0, 1.0)
    actions_tensor[:, 1] = torch.clamp(actions_tensor[:, 1], -1.0, 1.0)
    actions_tensor = action_scale*torch.matmul(actions_tensor, action_transfrom)
    # print("Tensor ", actions_tensor)

    actions_tensor = torch.tensor([[1, 2, 3, 4]], dtype=torch.float32)
    # print("Tensor ", actions_tensor)
    
    gym.set_dof_velocity_target_tensor(sim, gymtorch.unwrap_tensor(actions_tensor))

    print(gym.get_dof_target_velocity(env, 1))

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)
    frame_count += 1

print("Done")
if not args.headless:
    gym.destroy_viewer(viewer)
gym.destroy_sim(sim)