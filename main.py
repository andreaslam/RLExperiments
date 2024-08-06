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


if IS_RENDER:
    env = gym.make(GAME, render_mode="human")
else:
    env = gym.make(GAME)


observation, info = env.reset()
observation_space = observation.shape
action_space = env.action_space.n

# training settings

TOTAL_TRAINING_STEPS = 100000
GAMMA_DISCOUNT_FACTOR = 0.9


q_table_path = f"{Q_TABLE_FOLDER}/q_table_{GAME}.pkl"

# check if Q-table exists

settings = TrainingSettings()
agent = TDTabularAgent(observation_space, action_space, env, settings)

if os.path.isfile(q_table_path):
    print("loading existing Q Table")
    if os.path.getsize(q_table_path) > 0:
        agent.load(q_table_path)
else:
    # initialise Q-table
    print("Initialising new Q Table")
    q_table = {}


plotter = SimulationReturnPlotter()

simulation_return = 0.0

len_sim = 0

# quantise inputs

for time_step in tqdm(range(TOTAL_TRAINING_STEPS), desc="updating q tables"):
    action = agent.get_action(tuple(agent.prepare_input(observation)))

    observation_prev = observation
    observation, reward, terminated, truncated, info = env.step(action)

    agent.update_estimate_online(
        tuple(agent.prepare_input(observation_prev)),
        action,
        reward,
        tuple(agent.prepare_input(observation)),
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

        len_sim = 0

    len_sim += 1

    simulation_return += reward * (len_sim**GAMMA_DISCOUNT_FACTOR)

env.close()

agent.save(q_table_path)

plotter.plot()
