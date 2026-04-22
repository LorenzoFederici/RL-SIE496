import gymnasium as gym
import envs
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.evaluation import evaluate_policy
import imageio.v2 as imageio
from utils.initialization import config_loader

# Envronment name
env_name = 'GridWorld-v0'
# env_name = 'CartPole-v1'
# env_name = 'LunarLander-v3'
# env_name = 'Pusher-v5'

# Saved model path
model_path = "trained_models/" + env_name + "/"
model_name = "model-ppo"

# Load environment configuration from YAML file
_, _, _, _, env_configs = config_loader(env_name, "ppo")

# Rendering mode
render_mode = "rgb_array"  # Use "human" to render the environment to the screen, use "rgb_array" if you want to save a GIF instead
if render_mode == "rgb_array":
    gif_path = model_path + f"render-{model_name}.gif"

# Create environment
env_kwargs = {**env_configs, "render_mode": render_mode}
env = gym.make(env_name, **env_kwargs)

# Load the trained agent
model = PPO.load(model_path + model_name)

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Reset the environment to generate the first observation
n_steps = 100
observation, info = env.reset()
if render_mode == "human":
    env.render()
elif render_mode == "rgb_array":
    frames = []
    first_frame = env.render()
    if first_frame is not None:
        frames.append(first_frame)

for _ in range(n_steps):
    # Print progress
    print(f"Step {_+1}/{n_steps}", end="\r")

    # The trained policy predicts the action, given the observation
    action, _ = model.predict(observation, deterministic=True)

    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)

    # Render the environment
    if render_mode == "human":
        env.render()
    elif render_mode == "rgb_array":
        frame = env.render()
        if frame is not None:
            frames.append(frame)

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()
        if render_mode == "human":
            env.render()
        elif render_mode == "rgb_array":
            frame = env.render()
            if frame is not None:
                frames.append(frame)

env.close()

if render_mode == "rgb_array":
    if frames:
        imageio.mimsave(gif_path, frames, fps=5)
        print(f"Saved render GIF to: {gif_path}")
    else:
        print("No frames captured. GIF was not created.")