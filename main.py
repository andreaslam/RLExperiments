import gymnasium as gym
import pickle
from tqdm import tqdm
import os
from table import Agent
import numpy as np
from plotter import SimulationReturnPlotter

# configure gymnasium setup

IS_RENDER = False
GAME = "CartPole-v1"

Q_TABLE_PATH = "agents"

if not os.path.exists(Q_TABLE_PATH):
    os.makedirs(Q_TABLE_PATH)
    print("Directory created successfully!")
else:
    print("Directory already exists!")


if IS_RENDER:
    env = gym.make(GAME, render_mode="human")
else:
    env = gym.make(GAME)


observation, info = env.reset()
observation_space = observation.shape
action_space = env.action_space.n

# training settings

TOTAL_TRAINING_STEPS = int(10000)
GAMMA_DISCOUNT_FACTOR = 0.9
EPSILON_GREEDY_FACTOR = 0.1


q_table_path = f"{Q_TABLE_PATH}/q_table_{GAME}.pkl"

# check if Q-table exists

if os.path.isfile(q_table_path):
    print("loading existing Q Table")
    if os.path.getsize(q_table_path) > 0:
        with open(q_table_path, "rb") as f:
            unpickler = pickle.Unpickler(f)
            q_table = unpickler.load()
else:
    # initialise Q-table
    print("Initialising new Q Table")
    q_table = {}

agent = Agent(
    q_table,
    action_space,
    env,
    gamma_discount_factor=GAMMA_DISCOUNT_FACTOR,
    initial_epsilon_greedy_factor=EPSILON_GREEDY_FACTOR,
    initial_learning_rate=1e-1,
)

plotter = SimulationReturnPlotter()

simulation_return = 0.0

len_sim = 0

# quantise inputs

for time_step in tqdm(range(TOTAL_TRAINING_STEPS), desc="updating q tables"):
    action = agent.get_action(tuple(agent.quantise_to_linspace(observation)))

    observation_prev = observation
    observation, reward, terminated, truncated, info = env.step(action)

    agent.update_q_estimate(
        tuple(agent.quantise_to_linspace(observation_prev)),
        action,
        reward,
        tuple(agent.quantise_to_linspace(observation)),
    )

    if IS_RENDER:
        env.render()

    if terminated or truncated:
        observation, info = env.reset()
        # print(
        #     f"discounted reward (factor: {GAMMA_DISCOUNT_FACTOR}):, {simulation_return}"
        # )
        plotter.register_datapoint(simulation_return, "TDAgent")

        simulation_return = 0.0
        with open(q_table_path, "wb") as f:
            pickle.dump(agent.table, f)

        len_sim = 0

    len_sim += 1

    simulation_return += reward * (len_sim**GAMMA_DISCOUNT_FACTOR)

env.close()

with open(q_table_path, "wb") as f:
    pickle.dump(agent.table, f)

plotter.plot()
