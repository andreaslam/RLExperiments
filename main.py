import gymnasium as gym
from tqdm import tqdm
import os
from table import TDTabularAgent
from settings import TrainingSettings
import matplotlib.pyplot as plt
import statistics

# configure gymnasium setup
IS_RENDER = False
GAME = "CartPole-v1"

Q_TABLE_FOLDER = "agents_data"
if not os.path.exists(Q_TABLE_FOLDER):
    os.makedirs(Q_TABLE_FOLDER)
    print("Directory created successfully!")
else:
    print("Directory already exists!")

# Create the environment
if IS_RENDER:
    env = gym.make(GAME, render_mode="human")
else:
    env = gym.make(GAME)

observation, info = env.reset()
observation_space = observation.shape
action_space = env.action_space.n

# Training settings
TOTAL_TRAINING_STEPS = 100000
GAMMA_DISCOUNT_FACTOR = 0.9

q_table_path = f"{Q_TABLE_FOLDER}/q_table_{GAME}.pkl"

settings = TrainingSettings(
    initial_learning_rate=1,
    initial_epsilon_greedy_factor=0.8,
    gamma_discount_factor=0.99,
)

agent = TDTabularAgent(observation_space, action_space, env, settings)

# Check if Q-table exists
if os.path.isfile(q_table_path):
    print("loading existing Q Table")
    if os.path.getsize(q_table_path) > 0:
        agent.load(q_table_path)
else:
    print("Initialising new Q Table")

# Track episode returns
episode_returns = []
current_return = 0.0
discount_factor_tracker = 1.0
episode_lengths = []
episode_length = 0

for time_step in tqdm(range(TOTAL_TRAINING_STEPS), desc="Training Agent"):

    action = agent.get_action(observation)
    old_obs = observation

    observation, reward, terminated, truncated, info = env.step(action)

    # Penalize falling
    if terminated:
        reward = -1000

    agent.update_estimate(old_obs, action, reward, observation)

    current_return += reward * discount_factor_tracker
    discount_factor_tracker *= GAMMA_DISCOUNT_FACTOR
    episode_length += 1

    if IS_RENDER:
        env.render()

    # End of episode
    if terminated or truncated:
        episode_returns.append(current_return)
        episode_lengths.append(episode_length)

        current_return = 0.0
        discount_factor_tracker = 1.0
        episode_length = 0
        observation, info = env.reset()

env.close()

# Save final Q-table
agent.save(q_table_path)

# Moving average (like first script)
if len(episode_returns) >= 100:
    moving_avg = [
        statistics.mean(episode_returns[i : i + 100])
        for i in range(len(episode_returns) - 99)
    ]
    plt.plot(moving_avg)
    plt.title("Moving Average of Episode Returns (Window=100)")
    plt.xlabel("Episode")
    plt.ylabel("Avg Return")
    plt.savefig("moving_average_returns.png")
    plt.show()
    plt.close()
else:
    print("Not enough episodes to compute moving average.")
