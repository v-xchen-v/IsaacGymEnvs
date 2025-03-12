from isaacgym import gymapi
from isaacgym import gymutil, gymtorch, gymapi
from isaacgymenvs.tasks.base.vec_task import VecTask
import torch
import numpy as np
import os


class RealmanTouchThrownBall(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg
        
        # Specific necessary parameters before initialization
        self.cfg["env"]["numObservations"] = 17# 7+7+3  # Modify based on your needs
        self.cfg["env"]["numActions"] = 7        # Modify based on your needs
        self.max_episode_length = 500

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)
        # Task-specific parameters
        # self.num_envs = self.cfg["env"]["numEnvs"]
        # self.max_episode_length = self.cfg["env"]["maxEpisodeLength"]
    
        # Create environments
        # self.create_sim() # do not call this here, it is called in the parent class
        
        # # Set observation and action spaces
        # self.num_observations = 
        # self.num_actions = 
        
        self.obs_buf = torch.zeros((self.num_envs, self.num_observations), device=self.device)
        self.rewards = torch.zeros(self.num_envs, device=self.device)
        self.reset_buf = torch.ones(self.num_envs, device=self.device).long()
        
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        
    def create_sim(self):
        # set the up axis to be z-up given that assets are y-up by default
        self.up_axis = self.cfg["sim"]["up_axis"]

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        # Create ground plane
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))
    
    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        # set the normal force to be z dimension
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0) if self.up_axis == 'z' else gymapi.Vec3(0.0, 1.0, 0.0)
        self.gym.add_ground(self.sim, plane_params)
        

    def _create_envs(self, num_envs, spacing, num_per_row):
        # Set up environment spacing
        spacing = 2.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        asset_file = "urdf/realman/rm_75_6f_description.urdf"
        
        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        # self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        
        pose = gymapi.Transform()
        if self.up_axis == 'z':
            pose.p.z = 0.0
            # asset is rotated z-up by default, no additional rotations needed
            pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        else:
            pose.p.y = 0.0
            pose.r = gymapi.Quat(-np.sqrt(2)/2, 0.0, 0.0, np.sqrt(2)/2)

        self.robot_handles = []
        self.ball_handles = []
        self.envs = []
        # # Create environments
        for i in range(self.num_envs):
        #     # Add environment setup code here
            env_ptr = self.gym.create_env(
                self.sim, env_lower, env_upper, num_per_row
            )
            robot_handle = self.gym.create_actor(env_ptr, robot_asset, pose, "realman", i, 1, 0)

            # TODO: adjust the actuator properties here.
            dof_props = self.gym.get_actor_dof_properties(env_ptr, robot_handle)
            dof_props['driveMode'][:] = gymapi.DOF_MODE_POS
            dof_props['stiffness'][:] = 400.0
            dof_props['damping'][:] = 80.0
            self.gym.set_actor_dof_properties(env_ptr, robot_handle, dof_props)

            # Add a sphere as a ball, 
            ball_radius = 0.05
            ball_asset = self.gym.create_sphere(self.sim, ball_radius)
            ball_pose = gymapi.Transform()
            ball_pose.p = gymapi.Vec3(-1.5, 0.0, ball_radius)
            ball_handle = self.gym.create_actor(env_ptr, ball_asset, ball_pose, "ball", i, 1, 0)
            
            
            self.envs.append(env_ptr)
            self.robot_handles.append(robot_handle)

            
    def reset_idx(self, env_ids):
        # Reset logic for specified environments
        pass
        
    def compute_observations(self):
        # Compute observations for all environments
        return self.obs_buf
        
    def compute_reward(self):
        # Compute rewards for all environments
        pass
        
    def pre_physics_step(self, actions):
        # Apply actions before physics simulation step
        pass
        
    def post_physics_step(self):
        # Process after physics simulation step
        self.compute_observations()
        self.compute_reward()