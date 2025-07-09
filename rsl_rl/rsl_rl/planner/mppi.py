from legged_gym.envs import * 
from legged_gym.utils import get_args, task_registry, bytes_to_torch, torch_to_bytes
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgSample
from scipy.interpolate import make_interp_spline
import torch 
import numpy as np 
import matplotlib.pyplot as plt
from abc import ABC
from torch.distributions.multivariate_normal import MultivariateNormal
from rsl_rl.storage import RolloutStorage
from multiprocessing import shared_memory
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline
import time

class MPPI_Isaac(ABC):
    def __init__(self, env_cfg: LeggedRobotCfg, planner_cfg: LeggedRobotCfgSample, args, extras):
        self.env_cfg = env_cfg
        self.sample_method = planner_cfg.planner.sampling_method
        self.num_samples_per_env = planner_cfg.planner.num_samples
        self.num_knots = planner_cfg.planner.num_knots
        self.num_agent_envs =  env_cfg.env.num_envs
        self.num_planner_envs = self.num_agent_envs * self.num_samples_per_env # Important
        self.device = planner_cfg.planner.device
        self.T = planner_cfg.planner.horizon
        self.nu = env_cfg.env.num_actions
        self.rollout_envs: LeggedRobot
        args.headless = True
        env_cfg.env.is_planner = True
        # env_cfg.env.is_planner = False
        env_cfg.env.num_envs = self.num_planner_envs # Initialize envs = num_planner_envs * samples_per_env
        args.use_camera = False # disable camera for planner
        self.rollout_envs, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
        self.rollout_envs._init_shared_memory(create_new=False, ready_event=extras["ready_event"])
        # self.rollout_envs._init_shared_memory(create_new=True, ready_event=None)
        self.rollout_envs.num_samples_per_env = self.num_samples_per_env
        self.U_nom = torch.zeros((self.num_agent_envs, self.num_knots, self.nu), device=self.device) + self.rollout_envs.default_dof_pos.reshape((1, 1, self.nu))

        self.noise_sigma = torch.eye(self.nu, device=self.device) * planner_cfg.planner.sample_noise
        self.noise_mu = torch.zeros(self.nu, device=self.device)
        self.noise_dist = MultivariateNormal(self.noise_mu, self.noise_sigma)
        self.num_dof = self.rollout_envs.num_dof
        self.num_privileged_state = self.rollout_envs.num_privileged_state
        self.storage = None
        self.transition = RolloutStorage.Transition()
        self.init_storage(
            self.num_planner_envs,
            self.T,
            [self.rollout_envs.num_obs],
            [self.rollout_envs.num_privileged_state],
            [self.rollout_envs.num_actions]
        )
        self.ctrl_dt = env_cfg.control.decimation * env_cfg.sim.dt 
        self.ctrl_steps = torch.linspace(0, self.T * self.ctrl_dt, self.T, dtype=torch.float32, device=self.device)
        self.knots_steps = torch.linspace(0, self.T * self.ctrl_dt, self.num_knots, dtype=torch.float32, device=self.device)
        # self._init_shared_memory()


    def init_storage(self, num_envs, num_horizon_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(num_envs, num_horizon_per_env, actor_obs_shape,  critic_obs_shape, action_shape, self.device)

    def control_interpolations(self, x: torch.Tensor, y: torch.Tensor, interp_array: torch.Tensor):
        # torch_array (num_samples, num_knots, num_actions)
        coeffs = natural_cubic_spline_coeffs(x, y)
        spline = NaturalCubicSpline(coeffs)
        return spline.evaluate(interp_array) 

    def update(self, shift_time):
        """
        Get the optimized action of all environments.

        Args:
            actions (torch.Tensor): A 3d Tensor from the agent environments capturing the current action plan 
            for all agent environments. Shape [num_agent_envs, T, nu]
            
            priv_state (torch.Tensor): A 2d Tensor from the agent environment including the XYZ base position, 
            base orientation in quaternion, base_lin_velocities, base_ang_velocities, joint positions in radian, 
            joint velocities in radian/second. Shape [num_agent_envs, 37]

        Returns:
            actions (Tuple[torch.Tensor, torch.Tensor]):
                The optimized action plan for the agent environments to apply. Shape [num_agent_envs, T, nu]
        """
        # convert shift time
        shift_time = torch.tensor(shift_time, dtype=torch.float32, device=self.device)
        
        # Shift actions forward in sync mode 
        self.U_nom = self.control_interpolations(self.knots_steps, self.U_nom, self.knots_steps + shift_time)

        noise = self.sample_noise() 
        
        # Repeat the current action plan from the agent environments since we need to sample
        actions_sampled = torch.repeat_interleave(self.U_nom, self.num_samples_per_env, dim=0)

        # # Then overwirte the last action to be zeros
        # actions_sampled[:, -1, :] = torch.zeros((self.num_planner_envs, self.nu))  
    
        # Add noise to the sampled action plan
        U_sampled = actions_sampled + noise

        # Upsample the sampled actions to match the actual control sequences 
        U_sampled = self.control_interpolations(self.knots_steps, U_sampled, self.ctrl_steps)

        # reset the current planner environment that matches the agent environments
        self.rollout_envs.set_rollout_env_idx()
        
        # apply rollout 
        now = time.time()
        rewards = self.rollout(U_sampled).view(-1, self.num_samples_per_env)
        print("Rollout time: ", time.time() - now)

        # For every consecutive sampled environments, find the idxes that maximize the cumulative reward
        best_idxs = torch.argmax(rewards, dim=1) + torch.arange(0, self.num_planner_envs, 
                                                                self.num_samples_per_env, 
                                                                device=self.device)
        # Clear the storage to for next iteration
        self.storage.clear()

        return U_sampled[best_idxs] 

    def rollout(self, actions):
        """
        Rollout the envs given the actions sampled. Notice the planner environment will not reset automatically. 
        In other words, we need to record the reset buff and stop adding rewards after environments are reset 
        Additionally, the paralleled environments are stacked as a blcok

        Args:
            actions (torch.Tensor): sampled action plan. Shape [num_planner_envs, T, nu]
        
        Returns:
            rewards (torch.Tensor): the cumulative reward after rolling out the system dynamics. shape [num_planner_envs]
        """

        curr_reset_env_ids = torch.zeros(self.num_planner_envs, device= self.device, dtype=torch.int64)
        for i in range(self.T):
            obs, privileged_obs, rewards, dones, infos = self.rollout_envs.step(actions[:, i, :])
            obs, rewards, dones = obs.to(self.device), rewards.to(self.device), dones.to(self.device)
            curr_reset_env_ids |= dones.to(torch.int64)
            self.transition.observations = obs
            self.transition.critic_observations = privileged_obs
            self.transition.actions = actions[:, i, :]
            self.transition.rewards = rewards.clone()
            self.transition.dones = curr_reset_env_ids
            self.transition.values = torch.zeros((self.num_planner_envs, 1), device = self.device)
            self.storage.add_transitions(self.transition)
            self.transition.clear()
        self.storage.compute_returns(last_values=torch.zeros((self.num_planner_envs, 1), device = self.device),
                                    gamma=1,
                                    lam=1)
        
        # Return the cumulative sum of rewards at step 0 
        return self.storage.returns[0].to(self.device)
    
    def sample_noise(self):
        """
        Sample Gaussian Noise with mean zeros and variances are specified by the config file. Notice 
        here num_envs = env_cfg.env.num_envs * self.num_samples_per_env

        Returns:
            noises (torch.Tensor): the sampled Gaussian noise with shape [num_planner_envs, num_knots, self.nu]. 
        """
        return self.noise_dist.sample((self.num_planner_envs, self.num_knots))
    