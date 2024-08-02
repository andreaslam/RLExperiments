import gymnasium as gym
import pickle
from tqdm import tqdm
import os
from table import Agent
import numpy as np
from plotter import SimulationReturnPlotter

# configure gymnasium setup
env = gym.make("CartPole-v1", render_mode="human")

observation, info = env.reset()
observation_space = observation.shape
action_space = env.action_space.n

# training loop

Q_TABLE_PATH = "agents/q_table_CartPole-v1.pkl"
GAMMA_DISCOUNT_FACTOR = 0.99
num_games = int(input("enter the number of games to play: "))

# check if Q-table exists
if os.path.isfile(Q_TABLE_PATH):
    with open(Q_TABLE_PATH, "rb") as f:
        q_table = pickle.load(f)
else:
    raise FileNotFoundError(f"{Q_TABLE_PATH} does not exist!")


agent = Agent(q_table, action_space, gamma_discount_factor=GAMMA_DISCOUNT_FACTOR)

plotter = SimulationReturnPlotter()

simulation_return = 0.0

time_step = 0

games_played = 0

while games_played < num_games:
    action = agent.get_action(tuple(np.array([np.round(x) for x in observation])))

    observation_prev = observation
    observation, reward, terminated, truncated, info = env.step(action)

    agent.update_q_estimate(
        tuple(np.array([np.round(x) for x in observation_prev])),
        action,
        reward,
        tuple(np.array([np.round(x) for x in observation])),
    )

    simulation_return += reward * (time_step**GAMMA_DISCOUNT_FACTOR)

    env.render()

    if terminated or truncated:
        observation, info = env.reset()
        print(
            f"game {games_played}: discounted reward (factor: {GAMMA_DISCOUNT_FACTOR}):, {simulation_return}"
        )

        plotter.register_datapoint(simulation_return, f"TDAgent {Q_TABLE_PATH}")

        simulation_return = 0.0
        time_step = 0
        games_played += 1
    time_step += 1


env.close()

plotter.plot()
