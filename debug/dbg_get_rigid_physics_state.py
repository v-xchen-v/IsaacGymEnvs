from isaacgym import gymapi
import numpy as np

# Initialize Gym
gym = gymapi.acquire_gym()

# Create a simulation environment
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, gymapi.SimParams())

# Load assets
asset = gym.load_asset(sim, "./assets/urdf", "ball.urdf")

# Add an actor to the simulation
env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 1), 1)
actor_handle = gym.create_actor(env, asset, gymapi.Transform(), "actor", 0, 1)

# Get rigid body states for tensor
rb_states = gym.get_actor_rigid_body_states(env, actor_handle, gymapi.STATE_ALL)

# Access specific properties(e.g., position, velocity)
positions = rb_states["pose"]["p"] # x, y, z position
orientations = rb_states["pose"]["r"] # x, y, z, w orientation
linear_velocities = rb_states["vel"]["linear"] # x, y, z linear velocity
angular_velocities = rb_states["vel"]["angular"] # x, y, z angular velocity


for step in range(100):
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.refresh_actor_root_state_tensor(sim)

    # Get rigid body states for tensor
    rb_states = gym.get_actor_rigid_body_states(env, actor_handle, gymapi.STATE_ALL)
    
    # Access specific properties(e.g., position, velocity)
    positions = rb_states["pose"]["p"] # x, y, z position
    orientations = rb_states["pose"]["r"] # x, y, z, w orientation
    linear_velocities = rb_states["vel"]["linear"] # x, y, z linear velocity
    angular_velocities = rb_states["vel"]["angular"] # x, y, z angular velocity

    print("Step: ", step)
    print("Positions: ", positions)
    print("Velocities: ", linear_velocities)