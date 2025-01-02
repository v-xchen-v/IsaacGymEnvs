from isaacgym import gymapi

# Initialize the simulation
gym = gymapi.acquire_gym()
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX)

# Create environment
env = gym.create_env(sim, gymapi.Vec3(-5, -5, -5), gymapi.Vec3(5, 5, 5), 1)

# Create ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 1, 0)
gym.add_ground(sim, plane_params)

# Create a sphere to throw
sphere_asset = gym.create_sphere(sim, 0.1, None)
sphere_pose = gymapi.Transform()
sphere_pose.p = gymapi.Vec3(0, 1, 0)  # Starting position
sphere_actor = gym.create_actor(env, sphere_asset, sphere_pose, "Sphere", 0, 0)

# Apply an initial velocity to "throw" the sphere
velocity = gymapi.Vec3(5, 10, 0)  # Adjust for desired direction and magnitude
# Ref: https://docs.robotsfan.com/isaacgym/api/python/gym_py.html?highlight=velocity#isaacgym.gymapi.Gym.set_rigid_linear_velocity
gym.set_rigid_linear_velocity(env, sphere_actor, velocity)

viewer = gym.create_viewer(sim, gymapi.CameraProperties())

# Step the simulation
for i in range(1000):
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

# Cleanup
gym.destroy_sim(sim)
