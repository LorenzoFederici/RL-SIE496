import gymnasium as gym
import envs
from stable_baselines3 import DQN, A2C, PPO
import ale_py
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from utils.initialization import config_loader
gym.register_envs(ale_py)

# Environment name
env_name = 'ALE/Breakout-v5'

# Algorithm name
algorithm = 'ppo'

# Load configuration from YAML file
policy_name, policy_kwargs, alg_params, n_train_steps = config_loader(env_name, algorithm)

# Create environment
env = make_atari_env(env_name, n_envs=8)
env = VecFrameStack(env, n_stack=4) # Stack 4 frames

# Instantiate the agent
log = "trained_models/" + env_name + "/"
model = PPO(
        policy = policy_name,
        policy_kwargs=policy_kwargs,
        env = env, 
        tensorboard_log = log,
        **alg_params)

# Save random agent
model.save(log + "model-random")

# Train the agent and display a progress bar
model.learn(total_timesteps=n_train_steps, 
            progress_bar=True)

# Save the agent
model.save(log + "model-ppo")
