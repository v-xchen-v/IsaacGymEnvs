from isaacgym import gymapi
from isaacgym import gymutil, gymtorch, gymapi
from isaacgymenvs.tasks.base.vec_task import VecTask
import torch
import numpy as np
import os
from isaacgymenvs.utils.torch_jit_utils import scale, to_torch, tensor_clamp

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
        
        # init rigid state here
        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(_rb_states)
        self.ball_pos = torch.zeros((self.num_envs, 3), device=self.device)
        
        # record init states of robot and sphere for reset later
        self.gym.refresh_actor_root_state_tensor(self.sim)
        _initial_root_states = self.gym.acquire_actor_root_state_tensor(
            self.sim
        )  # [num_rigid_bodies, 13]
        self.initial_root_states = gymtorch.wrap_tensor(
            _initial_root_states
        ).clone()
        
        # record init DoF states for reset later
        _global_dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        self.global_dof_states = gymtorch.wrap_tensor(_global_dof_states).clone()
        
        # get DoF limits
        
        self.dof_record = []
        
        
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
        robot_dof_props = self.gym.get_asset_dof_properties(robot_asset)
        self.robot_lower_limits = robot_dof_props['lower']
        self.robot_upper_limits = robot_dof_props['upper']

        # Get DOF names
        dof_names =self.gym.get_asset_dof_names(robot_asset)

        # Print DOF names
        print("List of DOF names:", dof_names)

        # List index and corresponding DOF name
        actuated_dof_indices = []
        for i, name in enumerate(dof_names):
            print(f"Index {i}: {name}")
            actuated_dof_indices.append(i)
            
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
        self.ball_indices = []
        self.robot_indices = []
        # # Create environments
        for i in range(self.num_envs):
        #     # Add environment setup code here
            env_ptr = self.gym.create_env(
                self.sim, env_lower, env_upper, num_per_row
            )
            robot_handle = self.gym.create_actor(env_ptr, robot_asset, pose, "realman", i, 0, 0)
            robot_index = self.gym.get_actor_index(env_ptr, robot_handle, gymapi.DOMAIN_SIM)
            self.robot_indices.append(robot_index)
            
            # TODO: adjust the actuator properties here.
            dof_props = self.gym.get_actor_dof_properties(env_ptr, robot_handle)
            dof_props['driveMode'][:] = gymapi.DOF_MODE_POS
            dof_props['stiffness'][:] = 2000.0
            dof_props['damping'][:] = 40.0
            self.gym.set_actor_dof_properties(env_ptr, robot_handle, dof_props)

            # Add a sphere as a ball, 
            ball_radius = 0.05
            ball_asset = self.gym.create_sphere(self.sim, ball_radius)
            ball_pose = gymapi.Transform()
            # TODO: adjust the ball position here.
            ball_pose.p = gymapi.Vec3(-2.0, 0.0, ball_radius)
            ball_handle = self.gym.create_actor(env_ptr, ball_asset, ball_pose, "ball", i, 0, 0)
            ball_index = self.gym.get_actor_index(env_ptr, ball_handle, gymapi.DOMAIN_SIM)
            self.ball_indices.append(ball_index)
            
            self.envs.append(env_ptr)
            self.robot_handles.append(robot_handle)
            self.ball_handles.append(ball_handle)

        self.bodies_per_env = int(self.gym.get_sim_rigid_body_count(self.sim)/self.num_envs)
        
        self.ball_indices = to_torch(self.ball_indices, dtype=torch.int32, device=self.device)
        self.robot_indices = to_torch(self.robot_indices, dtype=torch.int32, device=self.device)
        pass
                # gym.get_actor_rigid_body_count(env, actor)
        # self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(forces), None, gymapi.ENV_SPACE)
        
    def reset_idx(self, env_ids):
        # Reset logic for specified environments
        env_ids_int32 = env_ids.to(torch.int32)

        ###########################
        ####### Reset Robot #######
        ###########################
        # Reset the ball's rigid position
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.initial_root_states),
            gymtorch.unwrap_tensor(self.ball_indices[env_ids_int32]),
            len(env_ids_int32),
        )
        
        # Reset the DoF of robots
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.global_dof_states),
            gymtorch.unwrap_tensor(self.robot_indices[env_ids_int32]), 
            len(env_ids_int32),
        )
        
        ##########################
        ###### Initialize Ball ######
        ##########################
        # Add a force to the ball
        forces = torch.zeros((self.num_envs, self.bodies_per_env, 3), device=self.device, dtype=torch.float)
        ball_rigid_idx = self.gym.find_actor_rigid_body_index(self.envs[0], self.ball_handles[0], "ball", gymapi.DOMAIN_SIM)
        forces[env_ids, ball_rigid_idx, 0] = 150 # x-direction
        forces[env_ids, ball_rigid_idx, 2] = 70 # z-direction
        forces[env_ids, ball_rigid_idx, 1] = 30 # y-direction
        forces = forces.reshape(self.num_envs*self.bodies_per_env, 3)
        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(forces), None, gymapi.ENV_SPACE)

        
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        
        # recording dof
        np.save("dof_poses.npy", np.array(self.dof_record))
        
    def compute_observations(self):
        # Compute observations for all environments
        
        # Fetch latest info
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        # TODO: get the ball position
        ball_rigid_idx = self.gym.find_actor_rigid_body_index(self.envs[0], self.ball_handles[0], "ball", gymapi.DOMAIN_SIM)
        vec_rb_states = self.rb_states.view(self.num_envs, self.bodies_per_env, 13)
        self.ball_pos = vec_rb_states[:, ball_rigid_idx, 0:3]
        # relative the env root pos

        # TODO: get ee position
        ee_rigid_idx = self.gym.find_actor_rigid_body_index(self.envs[0], self.robot_handles[0], "Link7", gymapi.DOMAIN_SIM)
        vec_rb_states = self.rb_states.view(self.num_envs, self.bodies_per_env, 13)
        self.ee_pos = vec_rb_states[:, ee_rigid_idx, 0:3]
        
        # TODO: record DoF 
        self.gym.refresh_dof_state_tensor(self.sim)
        self.num_dofs = 7
        vec_dof_states = self.dof_state.view(self.num_envs, self.num_dofs, 2)
        dof_positions = vec_dof_states[:, :, 0]
        self.dof_record.append(dof_positions.cpu().numpy().copy())  # Save current pose
        return self.obs_buf
        
    def compute_reward(self):
        # Compute rewards for all environments
        self.rew_buf[:], self.reset_buf[:] = compute_touchball_reward(
            reset_buf=self.reset_buf, 
            progress_buf=self.progress_buf, 
            max_episode_length=self.max_episode_length, 
            ball_pos=self.ball_pos, 
            ee_pos=self.ee_pos
        )
        
    def pre_physics_step(self, actions):
        # Apply actions before physics simulation step
        actions_tensor = actions.clone().to(self.device)
        
        # TODO: scale action from [-1, 1] to [dof_lower_limits, dof_upper_limits]
        actions_tensor = scale(actions_tensor, 
                               to_torch(self.robot_lower_limits, device=self.device), 
                               to_torch(self.robot_upper_limits, device=self.device))
        # self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(self.cur_targets[:, self.actuated_dof_indices],
        #                                                             self.hand_dof_lower_limits[self.actuated_dof_indices], self.hand_dof_upper_limits[self.actuated_dof_indices])
        # actions_tensor = torch.ones(self.num_envs * 7, device=self.device, dtype=torch.float)
        
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(actions_tensor))
        pass

    def post_physics_step(self):
        # Process after physics simulation step
        self.progress_buf += 1
        
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
        
        self.compute_observations()
        self.compute_reward()
        
# @torch.jit.script
def compute_touchball_reward(reset_buf, progress_buf, max_episode_length, ball_pos, ee_pos):
    reset = torch.zeros_like(reset_buf)
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)
    # TODO: reset when the sphere out of the workspace
    # reset = torch.where(ball_pos[:, 2] < 0.0, torch.ones_like(reset_buf), reset)
    # reset = torch.where(ball_pos[:, 2] > 1.0, torch.ones_like(reset_buf), reset)
    # reset = torch.where(ball_pos[:, 0] < -1.0, torch.ones_like(reset_buf), reset)
    reset = torch.where(ball_pos[:, 0] > 1.0, torch.ones_like(reset_buf), reset)
    # reset = torch.where(ball_pos[:, 1] < -1.0, torch.ones_like(reset_buf), reset)
    reset = torch.where(ball_pos[:, 1] > 1.0, torch.ones_like(reset_buf), reset)
    
    reward = torch.zeros_like(reset_buf, dtype=torch.float)
    # get distance between sphere and end effector
    distance = torch.norm(ball_pos - ee_pos, dim=1)
    
    # determine if the sphere is intercepted by the end effector
    intercepted = distance < 0.1
    intercepted_reward = torch.where(intercepted, torch.tensor(5.0), torch.tensor(-1.0))
    
    # reset if intercepted
    reset = torch.where(intercepted, torch.ones_like(reset_buf), reset)    
    total_reward = intercepted_reward  + distance*-1.0
    reward = total_reward
    return reward, reset
