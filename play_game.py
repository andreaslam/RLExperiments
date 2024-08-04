import gymnasium as gym
import pickle
from tqdm import tqdm
import os
from table import TDTabularAgent
import torch
from network import LinearNetModel, NNAgent
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

Q_TABLE_PATH = f"{Q_TABLE_FOLDER}/agent_{GAME}.pt"
GAMMA_DISCOUNT_FACTOR = 0.99

while True:
    try:
        num_games = int(input("enter the number of games to play: "))
        if num_games > 0:
            break
    except Exception:
        pass


settings = TrainingSettings(num_states_in_linspace=10)
agent = NNAgent(
    observation_space,
    action_space,
    env,
    settings,
    torch.jit.script(LinearNetModel(len(observation), action_space.item(), 100, 10)),
)

# check if Q-table exists
if os.path.isfile(Q_TABLE_PATH):
    agent.load(Q_TABLE_PATH)
else:
    raise FileNotFoundError(f"{Q_TABLE_PATH} does not exist!")

simulation_return = 0.0

time_step = 0

games_played = 0

while games_played < num_games:
    action = agent.get_action(observation, True)

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
