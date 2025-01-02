from isaacgym import gymapi
from isaacgym import gymutil

# Initialize gym and create simulation
gym = gymapi.acquire_gym()
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0, -9.8, 0)
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

# Create an environment
env = gym.create_env(sim, gymapi.Vec3(-5, -5, -5), gymapi.Vec3(5, 5, 5), 1)

# Add a ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 1, 0)
gym.add_ground(sim, plane_params)

# Create a sphere asset
radius = 0.1  # Sphere radius
sphere_asset_options = gymapi.AssetOptions()
sphere_asset = gym.create_sphere(sim, radius, sphere_asset_options)

# Add the sphere actor to the environment
initial_pose = gymapi.Transform()
initial_pose.p = gymapi.Vec3(0, 0.5, 0)  # Initial position of the sphere
sphere_actor = gym.create_actor(env, sphere_asset, initial_pose, "Sphere", 0, 0)

# Get the rigid body handle for the sphere
rigid_body_handle = gym.get_actor_rigid_body_handle(env, sphere_actor, 0)

# Apply a force to "throw" the sphere
force = gymapi.Vec3(50, 500, 0)  # Force vector in Newtons
force_position = gymapi.Vec3(0, 0.5, 0)  # Position where the force is applied


# viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(0, 3, 0), gymapi.Vec3(0, 0, 0))

# Simulate and apply force
for step in range(300):
    if step == 100:  # Apply force at the 10th simulation step
        # Ref: https://docs.robotsfan.com/isaacgym/api/python/gym_py.html?highlight=apply_body_force_at_pos#isaacgym.gymapi.Gym.apply_body_force_at_pos
        # gym.apply_body_force_at_pos(env, rigid_body_handle, force, force_position, gymapi.LOCAL_SPACE)
        
        # Ref: https://docs.robotsfan.com/isaacgym/api/python/gym_py.html?highlight=apply_body_forces#isaacgym.gymapi.Gym.apply_body_forces
        gym.apply_body_forces(env, rigid_body_handle, force, None, gymapi.LOCAL_SPACE)
    
    # Step simulation
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)

# Cleanup
gym.destroy_sim(sim)
