import gymnasium as gym
import pickle
from tqdm import tqdm
import os
from table import TDTabularAgent
from settings import TrainingSettings
import numpy as np


GAME = "CartPole-v1"

# configure gymnasium setup
env = gym.make(GAME, render_mode="human")

observation, info = env.reset()
observation_space = observation.shape
action_space = env.action_space.n

# training loop

Q_TABLE_FOLDER = "agents_data"

Q_TABLE_PATH = f"{Q_TABLE_FOLDER}/q_table_{GAME}.pkl"
GAMMA_DISCOUNT_FACTOR = 0.99

while True:
    try:
        num_games = int(input("enter the number of games to play: "))
        if num_games > 0:
            break
    except Exception:
        pass

# check if Q-table exists
if os.path.isfile(Q_TABLE_PATH):
    with open(Q_TABLE_PATH, "rb") as f:
        q_table = pickle.load(f)
else:
    raise FileNotFoundError(f"{Q_TABLE_PATH} does not exist!")


settings = TrainingSettings()
agent = TDTabularAgent(observation_space, action_space, env, settings)

simulation_return = 0.0

time_step = 0

games_played = 0

while games_played < num_games:
    action = agent.get_action(tuple(agent.quantise_to_linspace(observation)), True)

    observation_prev = observation
    observation, reward, terminated, truncated, info = env.step(action)

    simulation_return += reward * (time_step**GAMMA_DISCOUNT_FACTOR)

    env.render()

    if terminated or truncated:
        observation, info = env.reset()
        print(
            f"game {games_played}: discounted reward (factor: {GAMMA_DISCOUNT_FACTOR}):, {simulation_return}"
        )

        simulation_return = 0.0
        time_step = 0
        games_played += 1
    time_step += 1


env.close()
