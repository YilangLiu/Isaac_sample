# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
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
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgSample


class Go2ParkourCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.40] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }
        joint_angles_range_low = {
            'FL_hip_joint': -1.0472,   # [rad]
            'RL_hip_joint': -1.0472,   # [rad]
            'FR_hip_joint': -1.0472 ,  # [rad]
            'RR_hip_joint': -1.0472,   # [rad]

            'FL_thigh_joint': -1.5708,     # [rad]
            'RL_thigh_joint': -0.5236,   # [rad]
            'FR_thigh_joint': -1.5708,     # [rad]
            'RR_thigh_joint': -0.5236,   # [rad]

            'FL_calf_joint': -2.7227,   # [rad]
            'RL_calf_joint': -2.7227,    # [rad]
            'FR_calf_joint': -2.7227,  # [rad]
            'RR_calf_joint': -2.7227,    # [rad]
        }
        joint_angles_range_high = {
            'FL_hip_joint': 1.0472,   # [rad]
            'RL_hip_joint': 1.0472,   # [rad]
            'FR_hip_joint': 1.0472 ,  # [rad]
            'RR_hip_joint': 1.0472,   # [rad]

            'FL_thigh_joint': 3.4907,     # [rad]
            'RL_thigh_joint': 4.5379,   # [rad]
            'FR_thigh_joint': 3.4907,     # [rad]
            'RR_thigh_joint': 4.5379,   # [rad]

            'FL_calf_joint': -0.83776,   # [rad]
            'RL_calf_joint': -0.83776,    # [rad]
            'FR_calf_joint': -0.8376,  # [rad]
            'RR_calf_joint': -0.83776,    # [rad]
        }
    # From Go1
    class init_state_slope( LeggedRobotCfg.init_state ):
        pos = [0.56, 0.0, 0.24] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.03,   # [rad]
            'RL_hip_joint': 0.03,   # [rad]
            'FR_hip_joint': -0.03,  # [rad]
            'RR_hip_joint': -0.03,   # [rad]

            'FL_thigh_joint': 1.0,     # [rad]
            'RL_thigh_joint': 1.9,   # [rad]1.8
            'FR_thigh_joint': 1.0,     # [rad]
            'RR_thigh_joint': 1.9,   # [rad]

            'FL_calf_joint': -2.2,   # [rad]
            'RL_calf_joint': -0.9,    # [rad]
            'FR_calf_joint': -2.2,  # [rad]
            'RR_calf_joint': -0.9,    # [rad]
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 40.} # {'joint': 40.}  # [N*m/rad]
        damping = {'joint': 1} # {'joint': 1}     # [N*m*s/rad]
        action_scale = 1.0 # 0.25
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        foot_name = "foot"

        # From Go1
        penalize_contacts_on = ["thigh", "calf"]
        # penalize_contacts_on = ["thigh", "calf", "base"]

        terminate_after_contacts_on = ["base"]#, "thigh", "calf"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        height_target = 0.35
        gait = "trot"
        vel_tar = [0.1, 0.0, 0.0]
        ang_vel_tar = [0.0, 0.0, 0.0]
        class scales(LeggedRobotCfg.rewards.scales):
            # tracking rewards
            tracking_goal_vel = 0.0
            tracking_yaw = 0.0
            # regularization rewards
            lin_vel_z = 0.0
            ang_vel_xy = 0.0
            orientation = 0.0
            dof_acc = 0.0
            collision = 0.
            action_rate = 0.
            delta_torques = 0.
            torques = 0.
            hip_pos = 0.
            dof_error = -0.1
            feet_stumble = 0.0
            feet_edge = 0.0

            dial_gaits = 0.0 # 0.1
            dial_upright = 0.5
            dial_height = 1.0
            dial_yaw = 0.3
            dial_vel = 1.0
            dial_ang_vel = 1.0
            dial_air_time = 0.
    
    class depth( LeggedRobotCfg.depth):
        
        position = dict(
            mean = [0.24, -0.0175, 0.12],
            std = [0.01, 0.0025, 0.03],
        )
        # position = (0.24, -0.0175, 0.07)
        rotation = dict(
            lower = [-0.1, 0.37, -0.1],
            upper = [0.1, 0.43, 0.1],
        )
        # 广角
        horizontal_fov = [86, 90]
        # Depth image noise(not very useful, set to 0 in the original code)
        dis_noise = 0.1
        # The closet distance that the depth camera can see
        near_plane = 0.2 # [m]

        angle = [0, 1]
    
    class noise(LeggedRobotCfg.noise):
        add_noise = True

    class terrain( LeggedRobotCfg.terrain):
        mesh_type = 'trimesh'  # Explicitly set for clarity
        num_rows = 2 # 10
        num_cols = 10 # 40 seems like robot will have intercollision check over the 
        height = [0.02, 0.02]
        terrain_dict = {"smooth slope": 0., 
                        "rough slope up": 0.0,
                        "rough slope down": 0.0,
                        "rough stairs up": 0.0, 
                        "rough stairs down": 0.0, 
                        "discrete": 0., 
                        "stepping stones": 0.0,
                        "gaps": 0., 
                        "smooth flat": 0,
                        "pit": 0.0,
                        "wall": 0.0,
                        "platform": 0.,
                        "large stairs up": 0.,
                        "large stairs down": 0.,
                        "parkour": 0.0,
                        "parkour_hurdle": 0.0,
                        "parkour_flat": 1.0, # 0
                        "parkour_step": 0.0,
                        "parkour_gap": 0.0, # 0.2 
                        "demo": 0.0
                        }
        terrain_proportions = list(terrain_dict.values())
        curriculum = False
        max_difficulty = False

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        push_robots = False
        push_interval_s = 6
        randomize_base_mass = False
        randomize_base_com = False
        randomize_motor = False
        action_delay_view = 0

    class env(LeggedRobotCfg.env):
        num_envs = 1
        episode_length_s = 60 # episode length in seconds 
        num_privileged_obs = 37 # base position, base orientation, base_lin_vel, base_ang_vel, dof_pos, dof_vel  

    class shared_memory(LeggedRobotCfg.shared_memory):
        names = ["action_shm", "plan_time_shm", "time_shm", "last_actions_shm", 
                 "last_dof_vel_shm", "last_torques_shm", "last_root_vel_shm", "feet_air_time_shm", 
                 "reset_buf_shm", "obs_history_buf_shm", "contact_buf_shm", "action_history_buf_shm",
                 "cur_goal_idx_shm", "reach_goal_timer_shm", "terrain_levels_shm", "commands_shm", 
                 "time_out_buf_shm", "privileged_obs_buf_shm", "episode_length_buf_shm"]

class Go2ParkourCfgSample( LeggedRobotCfgSample ):
    class planner( LeggedRobotCfgSample.planner ):
        num_samples = 500
        sample_noise = 0.01
        horizon = 16
        num_knots = 6

    class rollout_env( LeggedRobotCfgSample.rollout_env ):
        dt = 0.01
        substeps = 2