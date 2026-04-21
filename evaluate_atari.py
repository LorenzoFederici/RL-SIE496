import gymnasium as gym
from stable_baselines3 import DQN, A2C, PPO
import ale_py
import imageio.v2 as imageio
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
gym.register_envs(ale_py)

# Envronment name
env_name = 'ALE/Breakout-v5'

# Saved model path
model_path = "trained_models/" + env_name + "/"
model_name = "model-ppo"

# Number of frame stacks for Atari environments
n_stack = 4

# Rendering mode
render_mode = "human"  # Use "human" to render the environment to the screen, use "rgb_array" if you want to save a GIF instead
if render_mode == "rgb_array":
    gif_path = model_path + f"render-{model_name}.gif"

# Create environment
env = make_atari_env(env_name, seed=50, env_kwargs={"render_mode": "human"})
env = VecFrameStack(env, n_stack=n_stack) # Stack 4 frames

# Load the trained agent
model = PPO.load(model_path + model_name)

# Reset the environment to generate the first observation
observation = env.reset()
done = False
frames = []
if render_mode == "human":
    env.render()
elif render_mode == "rgb_array":
    frames = []
    first_frame = env.render()
    if first_frame is not None:
        frames.append(first_frame)

while not done:
    # Print progress
    print(f"Step {_+1}", end="\r")

    # The trained policy predicts the action, given the observation
    action, _ = model.predict(observation, deterministic=True)

    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode is done
    observation, reward, done, info = env.step(action)

    done = bool(done[0]) if hasattr(done, "__len__") else bool(done)

    # Render the environment
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