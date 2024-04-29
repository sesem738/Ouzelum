# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import csv
import math
import torch
import numpy as np

from .base.vec_task import VecTask
from isaacgymenvs.utils.torch_jit_utils import *
from isaacgymenvs.utils.POMDP import POMDPWrapper
from isaacgymenvs.controllers.control_config import control
from isaacgymenvs.controllers.controller import Controller
from isaacgymenvs.PVFilter import PVFilter
from isaacgymenvs.ahrs_ekf import EKF

from isaacgym import gymutil, gymtorch, gymapi


class EKFLeeLanded(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["maxEpisodeLength"]
        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        # Observations:
        # 0:13 - root state
        self.cfg["env"]["numObservations"] = 13

        # Actions:
        # 0:3 - xyz force vector for lower rotor
        # 4:6 - xyz force vector for upper rotor
        self.cfg["env"]["numActions"] = 4

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # Time for kalman filters to converge
        self.ConvergenceTime = self.cfg["env"]["ConvergenceTime"] # sec 

        # Partially Observability
        self.POMDP = POMDPWrapper(pomdp=self.cfg["env"]["POMDP"], pomdp_prob=self.cfg["env"]["pomdp_prob"])
        
        dofs_per_env = self.num_dofs + 4
        
        #Num of bodies including target
        bodies_per_env = self.num_bodies + 15

        self.root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)

        self.vec_root_tensor = gymtorch.wrap_tensor(self.root_tensor).view(self.num_envs, 2, 13)
        vec_dof_tensor = gymtorch.wrap_tensor(self.dof_state_tensor).view(self.num_envs, dofs_per_env, 2)
        vec_sensor_tensor = gymtorch.wrap_tensor(self.sensor_tensor).view(self.num_envs, 6)

        self.root_states = self.vec_root_tensor[:, 0, :]
        self.root_positions = self.root_states[:, 0:3]
        self.target_root_positions = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.target_root_positions[:, 2] = 0.377
        self.root_quats = self.root_states[:, 3:7]
        self.root_linvels = self.root_states[:, 7:10]
        self.root_angvels = self.root_states[:, 10:13]

        # Maintain a pervious velocity state for calculating accelerations
        self.prev_root_linvels = torch.zeros(self.num_envs,3,device=self.device)

        self.husky_states = self.vec_root_tensor[:, 1, :]
        self.husky_positions = self.husky_states[:, 0:3]
        self.husky_quats = self.husky_states[:, 3:7]
        self.husky_linvels = self.husky_states[:, 7:10]
        self.husky_angvels = self.husky_states[:, 10:13]

        self.dof_states = vec_dof_tensor
        self.dof_positions = vec_dof_tensor[..., 0]
        self.dof_velocities = vec_dof_tensor[..., 1]

        self.sensor_forces = vec_sensor_tensor[..., 0:3]
        self.sensor_torques = vec_sensor_tensor[..., 3:6]

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        self.initial_root_states = self.root_states.clone()
        self.initial_husky_states = self.husky_states.clone()
        self.initial_dof_states = self.dof_states.clone()

        self.epi = 0
        self.Landoa = 0
        self.flag = torch.BoolTensor(self.num_envs)
        self.flag[:] = False

        max_thrust = 2000
        self.thrust_lower_limits = torch.zeros(4, device=self.device, dtype=torch.float32)
        self.thrust_upper_limits = max_thrust * torch.ones(4, device=self.device, dtype=torch.float32)

        # control tensors
        self.thrusts = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device, requires_grad=False)
        self.forces = torch.zeros((self.num_envs, bodies_per_env, 3), dtype=torch.float32, device=self.device, requires_grad=False)
        self.torques = torch.zeros((self.num_envs, bodies_per_env, 3), dtype=torch.float32, device=self.device, requires_grad=False)

        self.all_actor_indices = torch.arange(self.num_envs * 2, dtype=torch.int32, device=self.device).reshape((self.num_envs, 2))

        self.output_file = f"trajectories/{self.POMDP.pomdp}_{self.POMDP.prob}_ep_{self.epi}.csv"
        with open(self.output_file, mode="w") as file:
            writer = csv.writer(file)
            writer.writerow(['Position X', 'Position Y', 'Position Z'])

        acc_var = torch.Tensor([0.01,0.01,0.01]).to(device='cuda:0')*100

        self.ekfs = []
        for _ in range(self.num_envs):
            self.ekfs.append(EKF(frequency=1/self.dt))
        self.Q_state = np.zeros([self.num_envs,4])
        self.Q_cov = np.zeros([self.num_envs,4])
        
        self.pvfilters = []
        for _ in range(self.num_envs):
            self.pvfilters.append(PVFilter(acc_var,self.device))
        
        self.pos_sensor_freq = self.cfg["env"]["position_sensor_freq"] # Hz
        self.vel_sensor_freq = self.cfg["env"]["velocity_sensor_freq"] # Hz
        self.attach_pos_sensor = self.cfg["env"]["attach_pos_sensor"] # boolean
        self.attach_vel_sensor = self.cfg["env"]["attach_vel_sensor"] # boolean
        self.pos_trigger_count = self.pos_sensor_freq*0
        self.vel_trigger_count = self.vel_sensor_freq/2

        self.sim_step_count = 0

        control_cfg = control()
        self.controller = Controller(control_config=control_cfg,device=self.device)
        self.target_waypoints = torch.zeros(self.num_envs,3,device=self.device)

        if self.viewer:
            cam_pos = gymapi.Vec3(2.25, 2.25, 3.0)
            cam_target = gymapi.Vec3(3.5, 4.0, 1.9)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

            # need rigid body states for visualizing thrusts
            self.rb_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
            self.rb_states = gymtorch.wrap_tensor(self.rb_state_tensor).view(self.num_envs, bodies_per_env, 13)
            self.rb_positions = self.rb_states[..., 0:3]
            self.rb_quats = self.rb_states[..., 3:7]

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z

        # Gravity
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self.dt = self.sim_params.dt
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))


    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = "./assets"
        asset_file = "x500/x500.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.angular_damping = 0.0
        asset_options.max_angular_velocity = 4 * math.pi
        asset_options.slices_per_cylinder = 40
        asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_bodies = self.gym.get_asset_rigid_body_count(asset)
        self.num_dofs = self.gym.get_asset_dof_count(asset)

        asset_root = "/home/sagar/ws/etc/Ouzelum/assets"
        asset_file = "urdf/husky_description/urdf/husky.urdf"

        self.top_plate_extent = torch.tensor([[-0.3663348,-0.29796878,0.000043182],
                                              [0.36366522,0.2920312,0.0063931034]],
                                              device=self.device) * 0.5

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.flip_visual_attachments = True
        asset_options.use_mesh_materials = True

        sensor_props = gymapi.ForceSensorProperties()
        sensor_props.enable_forward_dynamics_forces = True
        sensor_props.enable_constraint_solver_forces = True
        sensor_props.use_world_frame = False
        sensor_pose = gymapi.Transform(gymapi.Vec3(0.0, 0.0, 0.0))
        self.gym.create_asset_force_sensor(asset, 0, sensor_pose, sensor_props)
        husky_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        
        default_pose = gymapi.Transform()
        default_pose.p.z = 1.0

        default_husky_pose = gymapi.Transform()
        default_husky_pose.p.z = 0.1

        self.envs = []
        self.actor_handles = []
        self.husky_handles = []
        self.force_sensors = []

        for i in range(self.num_envs):
            # create env instance
            env = self.gym.create_env(self.sim, lower, upper, num_per_row)
            actor_handle = self.gym.create_actor(env, asset, default_pose, "Drone", i, 0, 1)

            dof_props = self.gym.get_actor_dof_properties(env, actor_handle)
            dof_props['stiffness'].fill(0)
            dof_props['damping'].fill(0)
            self.gym.set_actor_dof_properties(env, actor_handle, dof_props)

            husky_handle = self.gym.create_actor(env, husky_asset, default_husky_pose, "husky", i, 0, 1)
            dof_props = self.gym.get_actor_dof_properties(env, husky_handle)
            dof_props["driveMode"] = (gymapi.DOF_MODE_VEL, gymapi.DOF_MODE_VEL, gymapi.DOF_MODE_VEL, gymapi.DOF_MODE_VEL)
            dof_props['stiffness'].fill(0)
            dof_props['damping'].fill(100)
            self.gym.set_actor_dof_properties(env, husky_handle, dof_props)
            self.gym.set_actor_dof_velocity_targets(env, husky_handle, [10.0, -20.0, 20.0, -10.0])

            self.actor_handles.append(actor_handle)
            self.husky_handles.append(husky_handle)
            self.envs.append(env)

        if self.debug_viz:
            # need env offsets for the rotors
            self.rotor_env_offsets = torch.zeros((self.num_envs, 2, 3), device=self.device)
            for i in range(self.num_envs):
                env_origin = self.gym.get_env_origin(self.envs[i])
                self.rotor_env_offsets[i, ..., 0] = env_origin.x
                self.rotor_env_offsets[i, ..., 1] = env_origin.y
                self.rotor_env_offsets[i, ..., 2] = env_origin.z


    def reset_idx(self, env_ids):

        # set rotor speeds
        self.dof_velocities[:, 0] = -1000
        self.dof_velocities[:, 1] = 1000
        self.dof_velocities[:, 2] = -1000
        self.dof_velocities[:, 3] = 1000

        num_resets = len(env_ids)

        actor_indices = self.all_actor_indices[env_ids, 0].flatten()

        self.root_states[env_ids] = self.initial_root_states[env_ids]
        self.root_states[env_ids, 0] += torch_rand_float(-1.5, 1.5, (num_resets, 1), self.device).flatten()
        self.root_states[env_ids, 1] += torch_rand_float(-1.5, 1.5, (num_resets, 1), self.device).flatten()
        self.root_states[env_ids, 2] += torch_rand_float(-0.2, 1.5, (num_resets, 1), self.device).flatten()

        self.gym.set_dof_state_tensor_indexed(self.sim, self.dof_state_tensor, gymtorch.unwrap_tensor(actor_indices), num_resets)

        # Select husky in env_ids or reset if farther than 2X environment spacing
        husky_indices = torch.tensor([-1],device=self.device,dtype=torch.int32)
        for env_id in env_ids:
            if (torch.abs(self.husky_states[env_id, 0]) > 2*(self.cfg["env"]['envSpacing'])) or \
                    (torch.abs(self.husky_states[env_id, 1]) > 2*(self.cfg["env"]['envSpacing'])):
                self.husky_states[env_id] = self.initial_husky_states[env_id]
                self.husky_states[env_id, 0] += torch_rand_float(-1.5, 1.5, (num_resets, 1), self.device).flatten()[0]
                self.husky_states[env_id, 1] += torch_rand_float(-1.5, 1.5, (num_resets, 1), self.device).flatten()[0]
                husky_indices = torch.cat([husky_indices, torch.tensor([(2*env_id.item())+1],device=self.device,dtype=torch.int32)])

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

        if len(husky_indices)>1:
            husky_indices = husky_indices[1:]
            return torch.unique(torch.cat([actor_indices,husky_indices]))
        return actor_indices

    def pre_physics_step(self, _actions):

        # resets

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        actor_indices = torch.tensor([], device=self.device, dtype=torch.int32)
        if len(reset_env_ids) > 0:
            actor_indices = self.reset_idx(reset_env_ids)
            self.epi = self.epi + len(reset_env_ids)

            print('Reseting ',len(reset_env_ids),' environmenrs')
            f = open(f'metrics/{self.POMDP.pomdp}_{self.POMDP.prob}_ep_count.txt', "w")
            f.write(str(self.epi))
            f.close()

            print('Recent landings: ',sum(self.flag[reset_env_ids]))

            if self.flag[reset_env_ids].any():
                self.Landoa = self.Landoa + sum(self.flag[reset_env_ids])
                print('Total landings : ',self.Landoa)
                self.flag[reset_env_ids] = False
                f = open(f'metrics/{self.POMDP.pomdp}_{self.POMDP.prob}.txt', "w")
                f.write(str(self.Landoa))
                f.close()

        reset_indices = torch.unique(actor_indices)
        if len(reset_indices) > 0:
            self.gym.set_actor_root_state_tensor_indexed(self.sim, self.root_tensor, gymtorch.unwrap_tensor(reset_indices), len(reset_indices))

        ########################## Filtering Stuff #################################

        # Method 1: Accels from force sensor
        # print(-self.sensor_forces/2) # mass of 2 units ???

        # Method 2: Translational non-grav accel from linear velocity
        dv = self.root_linvels - self.prev_root_linvels
        linear_accels = dv/self.dt

        # Let filters converge
        if self.sim_step_count<self.ConvergenceTime:
            self.Q_state[:] = self.root_quats.cpu()[:,[3,0,1,2]].numpy()
        
        if len(reset_env_ids) > 0:
            self.Q_state[reset_env_ids.cpu().numpy()] = self.root_quats[reset_env_ids.cpu().numpy()][:,[3,0,1,2]].cpu().numpy()

        for idx in reset_env_ids:
            state = torch.zeros(9,1,device=self.device)
            state[0:3,0] = self.root_positions[idx,:]
            state[3:6,0] = self.root_linvels[idx,:]
            state[6:9,0] = 0
            self.pvfilters[idx].set_states(state)
        
        ########################## Attitude EKF #################################
        # Accelerations for EKF
        # EKF may struggle with centripetal accel + translational accels
        # ekf_accel = linear_accels + torch.cross(self.root_angvels,self.root_linvels)
        ekf_accel = linear_accels
        ekf_accel[:,2] = ekf_accel[:,2] + 9.8
        ekf_accel = my_quat_rotate(self.root_quats,ekf_accel)

        # EKF stepping
        for idx in range(self.num_envs):
            # Angle measurement sensor
            ang_sensor_data = self.root_quats.cpu()[idx,[3,0,1,2]].numpy()

            # Prediction & correction steps
            self.Q_state[idx] = self.ekfs[idx].update(
                q=self.Q_state[idx]/np.linalg.norm(self.Q_state[idx]),
                gyr=self.root_angvels[idx].cpu().numpy(),
                acc=ekf_accel[idx].cpu().numpy(),
                ang=ang_sensor_data)
            self.Q_cov[idx,:] = self.ekfs[idx].P.diagonal()

        # self.log_ekffilter_state(self.Q_state,self.Q_cov)

        ########################## Linear KF #################################
        if self.sim_step_count<self.ConvergenceTime:
            orientation = self.root_quats
            flip_Qw = True
        else:
            orientation = torch.Tensor(self.Q_state).to(self.device)
            flip_Qw = False

        # Measurements
        pos_data = self.root_positions
        vel_data = self.root_linvels
        pos_var = torch.Tensor([1,1,1]).to(device='cuda:0')*0.0000001
        vel_var = torch.Tensor([1,1,1]).to(device='cuda:0')*0.0000001

        # self.log_plot_data(linear_accels,self.root_linvels,pos_data,orientation)
        
        state = torch.zeros(self.num_envs,9,1)
        cov = torch.zeros(self.num_envs,9,1)

        # Step LinearKF
        for idx in range(self.num_envs):
            self.pvfilters[idx].prediction_step(linear_accels[idx],orientation[idx],dt=self.dt,sim_time=self.sim_step_count*self.dt,flip_Qw=flip_Qw)

            # Update from position measurement
            if (self.attach_pos_sensor & ((self.pos_trigger_count*self.dt)>(1/self.pos_sensor_freq))):
                self.pvfilters[idx].correction_step(gps_data=pos_data[idx],gps_var=pos_var)
                self.pos_trigger_count = 0
            else:
                self.pos_trigger_count = self.pos_trigger_count +1

            # Update from velocity measurement
            if (self.attach_vel_sensor & ((self.vel_trigger_count*self.dt)>(1/self.vel_sensor_freq))):
                self.pvfilters[idx].correction_step(vel_data=vel_data[idx],vel_var=vel_var)
                self.vel_trigger_count  = 0
            else:
                self.vel_trigger_count = self.vel_trigger_count+1
            
            # Retreive state data
            state[idx] = self.pvfilters[idx].get_states()
            cov[idx,:] = torch.diagonal(self.pvfilters[idx].get_covariances()).reshape(9,1)

        state = state.reshape(self.num_envs,9).to(self.device)
        cov = cov.reshape(self.num_envs,9).to(self.device)
        
        dist = torch.sqrt(torch.square(state[:,0:3] - self.root_positions).sum(-1))
        # print('Estimate distance: ',dist)

        # self.log_pvfilter_state(state,cov)

        self.prev_root_linvels[:] = self.root_linvels

        ########################## Controller Stuff #################################

        mg = 2*(-self.sim_params.gravity.z)
        target = torch.zeros((self.num_envs,3),device=self.device)
        target_command = torch.zeros((self.num_envs,4),device=self.device)

        if self.sim_step_count<self.ConvergenceTime:
            # Set stationary as target
            target[:] = self.root_positions
            self.target_waypoints[:] = target
        else:
            target[:] = self.target_root_positions
        self.target_waypoints[:] = self.target_root_positions

        # # Get target distance
        target_vector = target - self.root_positions
        target_dist = torch.sqrt(torch.square(target_vector).sum(-1))
        # waypoint_vector = self.target_waypoints - self.root_positions
        # waypoint_dist = torch.sqrt(torch.square(waypoint_vector).sum(-1))

        # # target_dist = torch.sqrt(torch.square(self.target_root_positions - self.root_positions).sum(-1))
        # # print('Distance to target: ', target_dist)

        # # If waypoint is within reach, set new waypoint as unit vector to target
        # if self.sim_step_count>=self.ConvergenceTime:
        #     waypoint_dist_check = (waypoint_dist < 0.05) | (waypoint_dist > 0.15)
        #     if not (waypoint_dist==0).any():
        #         self.target_waypoints[waypoint_dist_check] = torch.div(
        #             target_vector[waypoint_dist_check],
        #             target_dist[waypoint_dist_check].reshape(waypoint_dist_check.sum().int(),1)
        #             )*0.1
        
        target_command[:,0:3] = self.target_waypoints
        # Step controller
        if self.sim_step_count<self.ConvergenceTime:
            output_thrusts_mass_normalized, output_torques_inertia_normalized = self.controller(self.root_states, target_command)
        else:
            Estimates = torch.zeros(self.num_envs,13,device=self.device)
            Estimates[:,3:13] = self.root_states[:,3:13]
            Estimates[:,0:3] = state[:,0:3]
            Estimates[:,7:10] = state[:,3:6]
            output_thrusts_mass_normalized, output_torques_inertia_normalized = self.controller(Estimates, target_command)
        
        # Final control effort
        self.forces[:,0,2] = mg*output_thrusts_mass_normalized
        self.torques[:,0] = output_torques_inertia_normalized

        #################################################################################
        target_dist_check = target_dist<0.25
        if (target_dist_check).any():
            if self.sim_step_count>=self.ConvergenceTime:
                self.flag[target_dist_check] = True
                print(self.flag[target_dist_check],target_dist)
            # Force landing
            self.forces[target_dist_check, :] = 0
            self.torques[target_dist_check, :] = 0

        # clear actions for reset envs
        self.thrusts[reset_env_ids] = 0.0
        self.forces[reset_env_ids] = 0.0

        # apply actions
        if self.sim_step_count>self.ConvergenceTime:
            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.forces), gymtorch.unwrap_tensor(self.torques), gymapi.LOCAL_SPACE)
        else:
            self.forces[:] = 0
            self.forces[:,0,2] = -2.09*(self.sim_params.gravity.z)
            self.torques[:,0] = 0
            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.forces), gymtorch.unwrap_tensor(self.torques), gymapi.LOCAL_SPACE)

    # def log_plot_data(self,accel,truth_vel,truth_position,orientation):
    #     # Append the current root position and time to the CSV file
    #     actor_idx = 2
    #     self.output_file = f"logsout/flight_info.csv"
    #     data = np.zeros([28])
    #     data[0] = self.sim_step_count*self.dt
    #     data[1:4] = accel[actor_idx,:].cpu()
    #     data[4:7] = truth_vel[actor_idx,:].cpu()
    #     data[7:10] = truth_position[actor_idx,:].cpu()

    #     if not (type(orientation) == torch.Tensor):
    #         orientation = torch.Tensor(orientation).to(self.device)
    #     R_body_to_nav = self.quaternion_to_matrix(orientation[:,[3,0,1,2]]).squeeze().cpu()[0]

    #     data[10:13] = my_quat_rotate(orientation, accel)[actor_idx,:].cpu() # R_body_to_nav@accel[actor_idx,:].cpu()
    #     data[13:16] = my_quat_rotate(orientation, truth_vel)[actor_idx,:].cpu() # R_body_to_nav@truth_vel[actor_idx,:].cpu()
    #     data[16:19] = my_quat_rotate(orientation, truth_position)[actor_idx,:].cpu() # R_body_to_nav@truth_position[actor_idx,:].cpu()
        
        
    #     data[19:22] = R_body_to_nav@accel[actor_idx,:].cpu()
    #     data[22:25] = R_body_to_nav@truth_vel[actor_idx,:].cpu()
    #     data[25:28] = R_body_to_nav@truth_position[actor_idx,:].cpu()
        
        
    #     with open(self.output_file, 'a+') as file:
    #         writer = csv.writer(file)
    #         writer.writerow(data)

    # def log_pvfilter_state(self,state,cov):
    #     # Append the current root position and time to the CSV file
    #     actor_idx = 2
    #     self.output_file = f"logsout/pvfilter_state.csv"
    #     data = np.zeros([19])
    #     data[0] = self.sim_step_count*self.dt
    #     data[1:10] = state[actor_idx,:].cpu()
    #     data[10:19] = cov[actor_idx,:].cpu()
        
    #     with open(self.output_file, 'a+') as file:
    #         writer = csv.writer(file)
    #         writer.writerow(data)
    
    # def log_ekffilter_state(self,state,cov):
    #     # Append the current root position and time to the CSV file
    #     actor_idx = 2
    #     self.output_file = f"logsout/ekffilter_state.csv"
    #     data = np.zeros([11])
    #     data[0] = self.sim_step_count*self.dt
    #     # data[1:5] = state[actor_idx,:]
    #     r,p,y = get_euler_xyz(torch.Tensor(state[actor_idx,[1,2,3,0]]).reshape(1,4))
    #     data[1],data[2],data[3] = r.numpy(), p.numpy(), y.numpy()
    #     data[4:8] = cov[actor_idx,:]
    #     r, p,y = get_euler_xyz(self.root_quats[actor_idx].cpu().reshape(1,4))
    #     data[8],data[9],data[10] = r.numpy(), p.numpy(), y.numpy()
        
    #     with open(self.output_file, 'a+') as file:
    #         writer = csv.writer(file)
    #         writer.writerow(data)
    
    def quaternion_to_matrix(self,quaternions):
        """
        Convert rotations given as quaternions to rotation matrices.

        Args:
            quaternions: quaternions with real part first,
                as tensor of shape (..., 4).

        Returns:
            Rotation matrices as tensor of shape (..., 3, 3).
        """
        r, i, j, k = torch.unbind(quaternions, -1)
        two_s = 2.0 / (quaternions * quaternions).sum(-1)

        o = torch.stack(
            (
                1 - two_s * (j * j + k * k),
                two_s * (i * j - k * r),
                two_s * (i * k + j * r),
                two_s * (i * j + k * r),
                1 - two_s * (i * i + k * k),
                two_s * (j * k - i * r),
                two_s * (i * k - j * r),
                two_s * (j * k + i * r),
                1 - two_s * (i * i + j * j),
            ),
            -1,
        )
        return o.reshape(quaternions.shape[:-1] + (3, 3))

    def post_physics_step(self):

        self.progress_buf += 1

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        self.target_root_positions[:,0:2] = self.husky_states[:,0:2].clone()
        self.target_root_positions[:,0] += 0.08 # Top plate X-shift

        self.sim_step_count = self.sim_step_count + 1

        self.compute_observations()
        self.compute_reward()

        # debug viz
        if self.viewer and self.debug_viz:
            # compute start and end positions for visualizing thrust lines
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            rotor_indices = torch.LongTensor([2, 4, 6, 8])
            quats = self.rb_quats[:, rotor_indices]
            dirs = -quat_axis(quats.view(self.num_envs * 4, 4), 2).view(self.num_envs, 4, 3)
            starts = self.rb_positions[:, rotor_indices] + self.rotor_env_offsets
            ends = starts + 0.1 * self.thrusts.view(self.num_envs, 4, 1) * dirs

            # submit debug line geometry
            verts = torch.stack([starts, ends], dim=2).cpu().numpy()
            colors = np.zeros((self.num_envs * 4, 3), dtype=np.float32)
            colors[..., 0] = 1.0
            self.gym.clear_lines(self.viewer)
            self.gym.add_lines(self.viewer, None, self.num_envs * 4, verts, colors)

    def compute_observations(self):
        self.obs_buf[..., 0:3] = (self.target_root_positions - self.root_positions) / 3
        self.obs_buf[..., 3:7] = self.root_quats
        self.obs_buf[..., 7:10] = self.root_linvels / 2
        self.obs_buf[..., 10:13] = self.root_angvels / math.pi
        
        self.obs_buf = self.POMDP.observation(self.obs_buf)
        self.traj = self.root_positions.cpu().numpy()
        self.traj2 = self.target_root_positions.cpu().numpy()
        self.traj3 = self.root_linvels.cpu().numpy()
        # self.traj4 = self.root_linvels.cpu().numpy()
        # self.log_root_positions()
        return self.obs_buf
    
    def log_root_positions(self):
        # Append the current root position and time to the CSV file
        self.output_file = f"trajectories/{self.POMDP.pomdp}_{self.POMDP.prob}_ep_{self.epi}.csv"
        # flag = np.array([self.POMDP.flago])
        self.array = np.concatenate((self.traj[0], self.traj2[0], self.traj3[0]))
        with open(self.output_file, mode='a') as file:
            writer = csv.writer(file)
            writer.writerow(self.array)

    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:] = compute_ingenuity_reward(
            self.root_positions,
            self.target_root_positions,
            self.root_quats,
            self.root_linvels,
            self.root_angvels,
            self.forces,
            self.reset_buf, self.progress_buf, self.max_episode_length
        )


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_ingenuity_reward(root_positions, target_root_positions, root_quats, root_linvels, root_angvels, forces, reset_buf, progress_buf, max_episode_length):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]

    # distance to target
    target_dist = torch.sqrt(torch.square(target_root_positions - root_positions).sum(-1))
    pos_reward = 1.0 / (1.0 + target_dist * target_dist)
    

    # uprightness
    ups = quat_axis(root_quats, 2)
    tiltage = torch.abs(1 - ups[..., 2])
    up_reward = 5.0 / (1.0 + tiltage * tiltage)

    # spinning
    spinnage = torch.abs(root_angvels[..., 2])
    spinnage_reward = 1.0 / (1.0 + spinnage * spinnage)
        
    # combined reward
    # uprigness and spinning only matter when close to the target
    reward = pos_reward + pos_reward * (up_reward + spinnage_reward)

    # resets due to misbehavior
    ones = torch.ones_like(reset_buf)
    die = torch.zeros_like(reset_buf)
    die = torch.where(target_dist > 8.0, ones, die)
    die = torch.where(root_positions[..., 2] < 0.3, ones, die)

    # resets due to episode length
    reset = torch.where(progress_buf >= max_episode_length - 1, ones, die)

    return reward, reset
