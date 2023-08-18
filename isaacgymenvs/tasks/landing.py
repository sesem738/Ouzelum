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
import math
import torch
import numpy as np

from .base.vec_task import VecTask
from isaacgymenvs.utils.torch_jit_utils import *

from isaacgym import gymutil, gymtorch, gymapi


class Landing(VecTask):

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

        dofs_per_env = self.num_dofs + 4
        
        #Num of bodies including target
        bodies_per_env = self.num_bodies + 15

        self.root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)

        vec_root_tensor = gymtorch.wrap_tensor(self.root_tensor).view(self.num_envs, 2, 13)
        vec_dof_tensor = gymtorch.wrap_tensor(self.dof_state_tensor).view(self.num_envs, dofs_per_env, 2)

        self.root_states = vec_root_tensor[:, 0, :]
        self.root_positions = self.root_states[:, 0:3]
        self.target_root_positions = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.target_root_positions[:, 2] = 0.377
        self.root_quats = self.root_states[:, 3:7]
        self.root_linvels = self.root_states[:, 7:10]
        self.root_angvels = self.root_states[:, 10:13]

        self.husky_states = vec_root_tensor[:, 1, :]
        self.husky_positions = self.husky_states[:, 0:3]
        self.husky_quats = self.husky_states[:, 3:7]
        self.husky_linvels = self.husky_states[:, 7:10]
        self.husky_angvels = self.husky_states[:, 10:13]

        self.dof_states = vec_dof_tensor
        self.dof_positions = vec_dof_tensor[..., 0]
        self.dof_velocities = vec_dof_tensor[..., 1]

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        self.initial_root_states = self.root_states.clone()
        self.initial_husky_states = self.husky_states.clone()
        self.initial_dof_states = self.dof_states.clone()

        max_thrust = 2000
        self.thrust_lower_limits = torch.zeros(4, device=self.device, dtype=torch.float32)
        self.thrust_upper_limits = max_thrust * torch.ones(4, device=self.device, dtype=torch.float32)

        # control tensors
        self.thrusts = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device, requires_grad=False)
        self.forces = torch.zeros((self.num_envs, bodies_per_env, 3), dtype=torch.float32, device=self.device, requires_grad=False)

        self.all_actor_indices = torch.arange(self.num_envs * 2, dtype=torch.int32, device=self.device).reshape((self.num_envs, 2))

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

        asset_root = "../../assets"
        asset_file = "x500/x500.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.angular_damping = 0.0
        asset_options.max_angular_velocity = 4 * math.pi
        asset_options.slices_per_cylinder = 40
        asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_bodies = self.gym.get_asset_rigid_body_count(asset)
        self.num_dofs = self.gym.get_asset_dof_count(asset)

        asset_root = "/home/sesem/WorldWideWeb/Ouzelum/assets"
        asset_file = "urdf/husky_description/urdf/husky.urdf"

        self.top_plate_extent = torch.tensor([[-0.3663348,-0.29796878,0.000043182],
                                              [0.36366522,0.2920312,0.0063931034]],
                                              device=self.device) * 0.5

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.flip_visual_attachments = True
        asset_options.use_mesh_materials = True
        husky_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        
        default_pose = gymapi.Transform()
        default_pose.p.z = 1.0

        default_husky_pose = gymapi.Transform()
        default_husky_pose.p.z = 0.1

        self.envs = []
        self.actor_handles = []
        self.husky_handles = []

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
            self.gym.set_actor_dof_velocity_targets(env, husky_handle, [-10.0, 20.0, -20.0, 10.0])

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

        reset_indices = torch.unique(actor_indices)
        if len(reset_indices) > 0:
            self.gym.set_actor_root_state_tensor_indexed(self.sim, self.root_tensor, gymtorch.unwrap_tensor(reset_indices), len(reset_indices))

        actions = _actions.to(self.device)
    
        thrust_action_speed_scale = 2000
        self.thrusts += self.dt * thrust_action_speed_scale * actions
        self.thrusts[:] = tensor_clamp(self.thrusts, self.thrust_lower_limits, self.thrust_upper_limits)

        self.forces[:, 1, 2] = self.thrusts[:, 0]
        self.forces[:, 2, 2] = self.thrusts[:, 1]
        self.forces[:, 3, 2] = self.thrusts[:, 2]
        self.forces[:, 4, 2] = self.thrusts[:, 3]

        # clear actions for reset envs
        self.thrusts[reset_env_ids] = 0.0
        self.forces[reset_env_ids] = 0.0

        # apply actions
        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.forces), None, gymapi.LOCAL_SPACE)

    def post_physics_step(self):

        self.progress_buf += 1

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        self.target_root_positions[:,0:2] = self.husky_states[:,0:2].clone()
        self.target_root_positions[:,0] += 0.08 # Top plate X-shift

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
        return self.obs_buf

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
