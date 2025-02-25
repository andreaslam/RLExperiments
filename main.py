import gymnasium as gym
import pickle
from tqdm import tqdm
import os
from table import TDTabularAgent
from settings import TrainingSettings
from plotter import SimulationReturnPlotter

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
TOTAL_TRAINING_STEPS = 1000000
GAMMA_DISCOUNT_FACTOR = 0.99

q_table_path = f"{Q_TABLE_FOLDER}/q_table_{GAME}.pkl"

settings = TrainingSettings(
    initial_learning_rate=0.95, initial_epsilon_greedy_factor=0.05
)

agent = TDTabularAgent(observation_space, action_space, env, settings)

# Check if Q-table exists
if os.path.isfile(q_table_path):
    print("loading existing Q Table")
    if os.path.getsize(q_table_path) > 0:
        agent.load(q_table_path)
else:
    print("Initialising new Q Table")
    q_table = {}

plotter = SimulationReturnPlotter()

# For each episode, track discounted returns
simulation_return = 0.0
discount_factor_tracker = 1.0  # gamma^t within the current episode

for time_step in tqdm(range(TOTAL_TRAINING_STEPS), desc="updating q tables"):
    action = agent.get_action(observation)
    old_obs = observation

    observation, reward, terminated, truncated, info = env.step(action)

    # Update the Q-value for old_obs, chosen action, and received reward
    agent.update_estimate(old_obs, action, reward, observation)

    # Update the discounted return
    simulation_return += reward * discount_factor_tracker
    discount_factor_tracker *= GAMMA_DISCOUNT_FACTOR

    if IS_RENDER:
        env.render()

    # If episode ended, reset
    if terminated or truncated:
        observation, info = env.reset()
        plotter.register_datapoint(simulation_return, "TDAgent")

        # Reset for the next episode
        simulation_return = 0.0
        discount_factor_tracker = 1.0

env.close()

# Save final Q-table
agent.save(q_table_path)

# Plot returns
plotter.plot()
