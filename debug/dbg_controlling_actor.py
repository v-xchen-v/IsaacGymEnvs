# import libraries
from isaacgym import gymapi, gymtorch
import torch

# initialize isaac gym
gym = gymapi.acquire_gym()
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

# create the environments
envs = []
num_envs = 4
for i in range(num_envs):
    env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 1), int(num_envs ** 0.5))
    envs.append(env)

    # load assets
    asset_root = "path/to/assets"
    robot_asset_file = "urdf/robot.urdf"
    robot_asset = gym.load_asset(sim, asset_root, robot_asset_file)

    # load actors
    actor_handle=gym.create_actor(env, robot_asset, gymapi.Transform(), "robot", i, 0)
    
    # define actor dof properties
    dof_props = gym.get_asset_dof_properties(robot_asset)
    dof_props['driveMode'].fill(gymapi.DOF_MODE_POS)
    dof_props['stiffness'].fill(400.0)
    dof_props['damping'].fill(40.0)
    gym.set_actor_dof_properties(env, actor_handle, dof_props)
        


# Flag to toggle between applying actions per environment or all environments
APPLY_PER_ENV = True  # Set to False to apply actions to all environments at once

viewer = gym.create_viewer(sim, gymapi.CameraProperties())
while not gym.query_viewer_has_closed(viewer):
    # step the simulation
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    # Refresh root state tensors
    gym.refresh_actor_root_state_tensor(sim)
    
    if APPLY_PER_ENV:
        # Apply actions per environment
        for env in envs:
            for i in range(gym.get_actor_count(env)):
                actor_handle = gym.get_actor_handle(env, i)
                num_dofs = gym.get_actor_dof_count(env, actor_handle)
                action_tensor = torch.zeros((num_dofs,), dtype=torch.float32, device="cpu")
                action_tensor[0] = 0.1  # Example: Setting position of the first joint
                gym.set_actor_dof_position_targets(env, actor_handle, action_tensor.numpy())
    else:
        # Apply actions to all environments
        action_tensor = torch.zeros((num_envs, 12), dtype=torch.float32, device="cpu")
        action_tensor[:, 0] = 0.1  # Example: Setting position of the first joint
        gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(action_tensor.numpy()))

    # visualize the simulation
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    
    # wait for the viewer to process events
    gym.sync_frame_time(sim)

# Clean up
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)