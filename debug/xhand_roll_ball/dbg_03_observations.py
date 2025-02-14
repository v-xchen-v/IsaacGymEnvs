'''Debug script to setup a scene with xhand and sphere in hand.
Set initial pose of xhand and cube to make sure
1. xhand is holding the sphere at initial pose, not dropping it.''' 
import os
from isaacgym import gymapi
from scipy.spatial.transform import Rotation as R
import time
import numpy as np
import cv2
# from isaacgym.torch_utils import (
#     to_torch,
#     quat_mul,
#     quat_conjugate,
#     tensor_clamp,
#     scale,
#     unscale,
#     quat_apply,
#     torch_rand_float,
#     quat_from_angle_axis,
# )

# Initialize Gym
gym = gymapi.acquire_gym()

# Simulation Configuration
sim_params = gymapi.SimParams()
sim_params.dt = 1.0 / 100.0
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.substeps = 1
# PhysX-specific parameters
# Ref: isaacgymenvs/tasks/factory/factory_schema_config_base.py
# sim_params.physx.use_gpu = True
# sim_params.use_gpu_pipeline = True
sim_params.physx.contact_offset = 0.02
sim_params.physx.rest_offset = 0.001
sim_params.physx.bounce_threshold_velocity = 0.2
sim_params.physx.num_position_iterations = 8
sim_params.physx.num_velocity_iterations = 0
sim_params.physx.use_gpu = True

# Create Simulator
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
if sim is None:
    raise Exception("Failed to create simulator")

# Load Assets
asset_root = "./assets"
robot_asset_file = "urdf/xhand/xhand_right.urdf"
# robot_asset_file = "urdf/franka_description/robots/franka_panda.urdf"
object_asset_file = "urdf/ball_to_roll.urdf"

# Robot asset
robot_asset_options = gymapi.AssetOptions()
robot_asset_options.fix_base_link = True
# robot_asset_options.disable_gravity = True
# robot_asset_options.collapse_fixed_joints = True
# robot_asset_options.flip_visual_attachments = True
robot_asset = gym.load_asset(sim, asset_root, robot_asset_file, robot_asset_options)
if robot_asset is None:
    raise Exception("Failed to load robot asset")

num_dofs = gym.get_asset_dof_count(robot_asset)
dof_indices = [i for i in range(num_dofs)]

# Object asset
object_asset = gym.load_asset(sim, asset_root, object_asset_file)
if object_asset is None:
    raise Exception("Failed to load object asset")

# Create Environment
num_envs = 4
envs = []
env_spacing = 2.0  # Space between environments
lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

goal_obj_quat = np.zeros((num_envs, 4), dtype=np.float32)

for i in range(num_envs):
    env = gym.create_env(sim, lower, upper, int(num_envs ** 0.5))
    envs.append(env)

    # Add robot to environment
    robot_pose = gymapi.Transform()
    robot_pose.p = gymapi.Vec3(0, -0.1, 0.5)  # Position (x, y, z)
    qx, qy, qz, qw = R.from_euler('xyz', [90, 0, 0], degrees=True).as_quat() # scalar-last by default
    # robot_pose.r = gymapi.Quat(0, 0, 0, 1)  # Rotation (x, y, z, w)
    robot_pose.r = gymapi.Quat(qx, qy, qz, qw)  # Rotation (x, y, z, w)
    robot_handle = gym.create_actor(env, robot_asset, robot_pose, "robot", i, 0)

    # define actor dof properties
    dof_props = gym.get_asset_dof_properties(robot_asset)
    
    # TODO: ?why if do not set driveMode and the parameters, the robot will not move?
    dof_props['driveMode'].fill(gymapi.DOF_MODE_POS)
    dof_props['stiffness'].fill(3.0)
    dof_props['damping'].fill(0.1)
    dof_props['friction'].fill(0.01)
    dof_props['armature'].fill(0.001)
    gym.set_actor_dof_properties(env, robot_handle, dof_props)
    
    # get actor dof properties
    num_dofs = gym.get_actor_dof_count(env, robot_handle)
    dof_lower_limits = dof_props['lower']
    dof_upper_limits = dof_props['upper']
    
    # Add object to environment
    object_pose = gymapi.Transform()
    object_pose.p = gymapi.Vec3(0, 0, 0.58)  # Position (x, y, z)
    object_handle = gym.create_actor(env, object_asset, object_pose, "object", i, 0)
    
    shape_props = gym.get_actor_rigid_shape_properties(env, object_handle)
    # set_actor_rigid_shape_properties enables setting shape properties for rigid body
    # Properties include friction, rolling_friction, torsion_friction, restitution etc.
    shape_props[0].friction = 1.0
    # shape_props[0].rolling_friction = 0.2
    # shape_props[0].torsion_friction = 0.2
    gym.set_actor_rigid_shape_properties(env, object_handle, shape_props)
            
    # Add plane ground to environment
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane_params)
    
    # # pretty color
    # gym.set_rigid_body_color(
    #     env, object_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.9))
    
control_cfg = {
    "use_relative_control": True,
    "relative_control_scale": 0.1,
}

# ---------------- observation function --------------- #
def _observation_dof_position(env, robot_handle):
    """
    Get the DOF positions of the robot.
    """
    dof_states = gym.get_actor_dof_states(env, robot_handle, gymapi.STATE_POS)
    dof_pos = np.array([state["pos"] for state in dof_states])
    return dof_pos

def _observation_dof_velocity(env, robot_handle):
    """
    Get the DOF velocities of the robot.
    """
    dof_states = gym.get_actor_dof_states(env, robot_handle, gymapi.STATE_VEL)
    dof_vel = np.array([state["vel"] for state in dof_states])
    return dof_vel

def _observation_obj_position(env, object_handle):
    """
    Get the position of the object.
    """
    obj_transform = gym.get_rigid_transform(env, object_handle)
    obj_pos = np.array([obj_transform.p.x, obj_transform.p.y, obj_transform.p.z])
    return obj_pos

def _observation_obj_quat(env, object_handle):
    """
    Get the orientation of the object.
    """
    obj_transform = gym.get_rigid_transform(env, object_handle)
    obj_quat = np.array([obj_transform.r.x, obj_transform.r.y, obj_transform.r.z, obj_transform.r.w])
    return obj_quat

def _observation_obj_linvel(env, object_handle):
    """
    Get the linear velocity of the object.
    """
    rb_states = gym.get_actor_rigid_body_states(env, object_handle, gymapi.STATE_ALL)
    obj_linvel = rb_states["vel"]["linear"] # x, y, z linear velocity
    return np.array([obj_linvel['x'], obj_linvel['y'], obj_linvel['z']])

def _observation_obj_angvel(env, object_handle):
    """
    Get the angular velocity of the object.
    """
    rb_states = gym.get_actor_rigid_body_states(env, object_handle, gymapi.STATE_ALL)
    obj_angvel = rb_states["vel"]["angular"] # x, y, z angular velocity
    return np.array([obj_angvel['x'], obj_angvel['y'], obj_angvel['z']])

def _observation_obj_goal_quat(env, object_handle):
    """
    Get the goal orientation of the object.
    """
    return goal_obj_quat

def _observation_obj_goal_quat_diff(env, object_handle):
    """
    Get the difference between the current orientation and the goal orientation of the object.
    """
    obj_quat = _observation_obj_quat(env, object_handle)
    goal_obj_quat = _observation_obj_goal_quat(env, object_handle)
    def quat_diff(q1, q2):
        """
        Compute the relative quaternion (q_diff) that rotates q1 to q2.
        
        Args:
            q1: np.array, shape (4,) - Quaternion (w, x, y, z)
            q2: np.array, shape (4,) - Quaternion (w, x, y, z)

        Returns:
            q_diff: np.array, shape (4,) - Relative quaternion
        """
        # Convert to scipy Rotation object
        r1 = R.from_quat(q1[[1, 2, 3, 0]])  # Convert to (x, y, z, w) format
        r2 = R.from_quat(q2[[1, 2, 3, 0]])  # Convert to (x, y, z, w) format

        # Compute relative rotation
        r_diff = r2 * r1.inv()  # Equivalent to q2 * inv(q1)
        
        # Convert back to (w, x, y, z) format
        q_diff = r_diff.as_quat()
        return np.hstack([q_diff[3], q_diff[:3]])  # Convert back to (w, x, y, z)
    obj_quat_diff = quat_diff(obj_quat, goal_obj_quat)
    return obj_quat_diff

# Run the Simulation
camera_props = gymapi.CameraProperties()
camera_props.width = 1280
camera_props.height = 720
viewer = gym.create_viewer(sim,camera_props)
view_step = 0   
step = 0
previous_targets = np.zeros(num_dofs, dtype=np.float32)
targets = np.zeros(num_dofs, dtype=np.float32)

actions = np.zeros(num_dofs, dtype=np.float32)

# Start recording video
video_dir = "./video"
if not os.path.exists(video_dir):
    os.makedirs(video_dir)
video_filename = os.path.join(video_dir, "xhand_roll_ball_02_dof_control.avi")
# gym.start_recording_video(sim, video_filename)
frames = []
running = True
RECORD_VIDEO_ON = False

num_step_per_episode = 500

def get_observations(env, robot_handle, object_handle):
    dof_pos = _observation_dof_position(env, robot_handle)
    dof_vel = _observation_dof_velocity(env, robot_handle)
    obj_pos = _observation_obj_position(env, object_handle)
    obj_quat = _observation_obj_quat(env, object_handle)
    obj_linvel = _observation_obj_linvel(env, object_handle)
    obj_angvel = _observation_obj_angvel(env, object_handle)
    obj_goal_quat = _observation_obj_goal_quat(env, object_handle)
    obj_goal_quat_diff = _observation_obj_goal_quat_diff(env, object_handle)
    return dof_pos, dof_vel, obj_pos, obj_quat, obj_linvel, obj_angvel, obj_goal_quat, obj_goal_quat_diff
    
while running and not gym.query_viewer_has_closed(viewer): # while True:
    print(f'\nStep: {step}')  
    if step == 0:
        # give hand a initial dof angles, to bend fingers a bit to avoid ball falling off
        # 0.05 radiaus for each finger
        for env in envs:
            dof_state_array = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)

            # Assign values
            for i in range(num_dofs):
                dof_state_array[i]['pos'] = 0.05  # Assign position
            gym.set_actor_dof_states(env, robot_handle, dof_state_array, gymapi.STATE_ALL)
            
    
    # Apply actions per 100 steps per environment
    if step % 100 == 0:

        for env in envs:
            # Get DOF states
            dof_states = gym.get_actor_dof_states(env, robot_handle, gymapi.STATE_ALL)

            # Separate joint positions and velocities
            joint_positions = [state["pos"] for state in dof_states]
            print(f'Joint positions: {joint_positions}')
            
            if control_cfg["use_relative_control"]:
                # random action in range of -delta to delta
                delta_limit= dof_upper_limits - dof_lower_limits
                delta_actions = np.random.uniform(-delta_limit, delta_limit)
                targets = previous_targets + control_cfg["relative_control_scale"] * delta_actions
            else:
                # random action in range of dof limits
                actions = np.random.uniform(dof_lower_limits, dof_upper_limits, num_dofs)
                targets = actions

            # Clip targets to joint limits
            targets = np.clip(targets, dof_lower_limits, dof_upper_limits)
            print(f'Targets: {targets}')
            
            # Apply targets
            gym.set_actor_dof_position_targets(env, robot_handle, targets.astype('f'))
            previous_targets = targets
            
    if step % num_step_per_episode == 0:
        # Reset the the object goal rotation
        # Got a random number from -1 to 1, then multiply by np.pi and apply the rotation to the object quat
        object_pose = gym.get_rigid_transform(env, object_handle)
        # object_pose.r = gymapi.Quat(0, 0, 0, 1)
        random_angle_x = np.random.uniform(-1, 1) * np.pi
        random_angle_y = np.random.uniform(-1, 1) * np.pi
        random_rot_quat = R.from_rotvec([random_angle_x, random_angle_y, 0]).as_quat()
        current_rot_quat = object_pose.r
        new_rot_quat = (R.from_quat([current_rot_quat.x, current_rot_quat.y, current_rot_quat.z, current_rot_quat.w]).as_matrix() @ R.from_quat(random_rot_quat).as_matrix())
        new_rot_quat = R.from_matrix(new_rot_quat).as_quat()
        new_rot_quat = gymapi.Quat(new_rot_quat[0], new_rot_quat[1], new_rot_quat[2], new_rot_quat[3])
        object_pose.r = new_rot_quat
        # gym.set_rigid_transform(env, object_handle, object_pose)
        goal_obj_quat = np.array([object_pose.r.x, object_pose.r.y, object_pose.r.z, object_pose.r.w])
        
    obs = get_observations(env, robot_handle, object_handle)
    obs_dict = {
        'dof_pos': obs[0],
        'dof_vel': obs[1],
        'obj_pos': obs[2],
        'obj_quat': obs[3],
        'obj_linvel': obs[4],
        'obj_angvel': obs[5],
        'obj_goal_quat': obs[6],
        'obj_goal_quat_diff': obs[7]
    }
    print(f'Observations: {obs_dict}')
        
        # gym.set_rigid_transform(env, object_handle, object_pose)
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    
    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    # gym.sync_frame_time(sim)

        
    step += 1
    
    # Check keyboard events
    for evt in gym.query_viewer_action_events(viewer):
        if evt.action == 'quit' and evt.value > 0: 
            running = False
        
    if RECORD_VIDEO_ON:
        image_buffer = gym.write_viewer_image_to_file(viewer, 'temp.png')
        np_image = cv2.imread('temp.png')
        frames.append(np_image)
    

# Stop recording video
def stop_recording_video():
    # Save video
    out = cv2.VideoWriter(
        video_filename, 
        cv2.VideoWriter_fourcc(*'mjpg'), 
        30, 
        (camera_props.width, camera_props.height)
    )
    for frame in frames:
        out.write(frame)
    out.release()
    os.remove('temp.png')
    print(f"Video saved at {video_filename}")
if RECORD_VIDEO_ON:
    stop_recording_video()
# Clean up
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
