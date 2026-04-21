import gymnasium as gym
from stable_baselines3 import DQN, A2C, PPO
from utils.initialization import config_loader

# Environment name
env_name = 'GridWorld-v0'
# env_name = 'CartPole-v1'
# env_name = 'Pendulum-v1'
# env_name = 'LunarLander-v3'
# env_name = 'Pusher-v5'

# Algorithm name
algorithm = 'ppo'

# Load configuration from YAML file
policy_name, policy_kwargs, alg_params, n_train_steps = config_loader(env_name, algorithm)

# Create environment
env = gym.make(env_name)

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
