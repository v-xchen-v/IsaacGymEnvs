from isaacgym import gymapi
import numpy as np
import time
import os

# Initialize simulation and sphere
gym = gymapi.acquire_gym()
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0, -9.8, 0)
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
env = gym.create_env(sim, gymapi.Vec3(-1, -1, -1), gymapi.Vec3(1, 1, 1), 1)
sphere_asset = gym.create_sphere(sim, 0.1, None)
sphere_pose = gymapi.Transform()
sphere_pose.p = gymapi.Vec3(0, 1, 0)  # Starting position
sphere_handle = gym.create_actor(env, sphere_asset, sphere_pose, "sphere", 0, 0)

# File to log data
log_file = "debug/sphere_pose_log.csv"
if os.path.exists(log_file):
    os.remove(log_file)
with open(log_file, 'w') as f:
    f.write("timestamp,px,py,pz,qx,qy,qz,qw\n")

# Simulation loop
# 100 steps
for step in range(100):
# while True:
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # Get current time
    timestamp = time.time()

    # Get sphere pose
    # Ref: https://docs.robotsfan.com/isaacgym/api/python/gym_py.html?highlight=transform#isaacgym.gymapi.Gym.get_rigid_transform
    sphere_transform = gym.get_rigid_transform(env, sphere_handle)
    position = sphere_transform.p
    orientation = sphere_transform.r

    # Log pose and timestamp
    with open(log_file, 'a') as f:
        f.write(f"{timestamp},{position.x},{position.y},{position.z},{orientation.x},{orientation.y},{orientation.z},{orientation.w}\n")

    # Step simulation
    gym.step_graphics(sim)
    gym.sync_frame_time(sim)
