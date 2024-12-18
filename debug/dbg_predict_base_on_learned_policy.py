from isaacgym import gymapi, gymutil
import torch

# Initialize Isaac Gym
gym = gymapi.acquire_gym()
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
gym_instance = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

def load_environment(gym_instance):
    gym = gymapi.acquire_gym()
    # asset_root = "path/to/assets"
    # asset_file = "urdf/robot.urdf"
    asset_root = "."
    asset_file = "isaacgymenvs/balance_bot.xml"
    asset = gym.load_asset(gym_instance, asset_root, asset_file)
    env = gym.create_env(gym_instance, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 1), 1)
    actor_handle = gym.create_actor(env, asset, gymapi.Transform(), "robot", 0, 0)
    return env, actor_handle
    
def render_preview(env, gym_instance):
    gym = gymapi.acquire_gym()
    gym.simulate(gym_instance)
    gym.fetch_results(gym_instance, True)
    gym.render_all_camera_sensors(gym_instance)
    gym.step_graphics(gym_instance)
    
def reset_env(env):
    gym = gymapi.acquire_gym()
    gym.reset_env(env)
    
# Load environment and policy
env, actor_handle = load_environment(gym_instance)  # User-defined function
# policy = torch.load("path/to/policy.pt")  # Load a trained policy
# policy = torch.load("isaacgymenvs/runs/BallBalance_09-21-06-05/nn/BallBalance.pth")  # Load a trained policy

# Preview Parameters
preview_horizon = 10  # Number of steps to preview
dt = 0.02  # Time step

def get_env_state(env, actor_handle):
    # mock data of shape [1, 24]
    return torch.rand((1, 24))
     
import torch.nn as nn
import torch

class ActorCriticModel(nn.Module):
    def __init__(self, input_dim, output_dim, mlp_units, activation="elu", fixed_sigma=True):
        super(ActorCriticModel, self).__init__()
        
        # Define activation function
        activation_fn = {
            "elu": nn.ELU,
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "none": None,
        }.get(activation.lower(), nn.ReLU)

        # Build shared MLP layers
        layers = []
        in_dim = input_dim
        for unit in mlp_units:
            layers.append(nn.Linear(in_dim, unit))
            if activation_fn:
                layers.append(activation_fn())
            in_dim = unit

        # Actor and Critic heads
        self.shared_mlp = nn.Sequential(*layers)
        self.actor_mu = nn.Linear(mlp_units[-1], output_dim)
        self.actor_sigma = nn.Parameter(torch.zeros(output_dim)) if fixed_sigma else nn.Linear(mlp_units[-1], output_dim)
        self.critic = nn.Linear(mlp_units[-1], 1)

    def forward(self, x):
        features = self.shared_mlp(x)
        mu = self.actor_mu(features)
        sigma = self.actor_sigma if isinstance(self.actor_sigma, torch.Tensor) else self.actor_sigma(features)
        value = self.critic(features)
        return mu, sigma, value
    
from omegaconf import OmegaConf

# Load the configuration file
config = OmegaConf.load("isaacgymenvs/cfg/train/BallBalancePPO.yaml")

# Access resolved values
def load_model_from_config(config_path):
    config = OmegaConf.load(config_path)
    input_dim = 10  # Replace with actual observation dimension
    output_dim = 4  # Replace with action dimension
    mlp_units = config.params.network.mlp.units
    activation = config.params.network.mlp.activation
    fixed_sigma = config.params.network.space.continuous.fixed_sigma

    model = ActorCriticModel(input_dim, output_dim, mlp_units, activation, fixed_sigma)
    model.load_state_dict(torch.load("isaacgymenvs/runs/BallBalance_09-21-06-05/nn/BallBalance.pth"))
    model.eval()
    return model

policy = load_model_from_config("isaacgymenvs/cfg/train/BallBalancePPO.yaml")

# Main preview loop
while True:
    state =get_env_state(env, actor_handle)  # Get the current environment state
    for _ in range(preview_horizon):
        action = policy(state)  # Predict the action
        env.apply_action(action)  # Apply action to the environment
        state = get_env_state(env, actor_handle)  # Update state
        render_preview(env, gym_instance)  # Render user-defined function
    reset_env(env)  # Reset the environment to the original state
