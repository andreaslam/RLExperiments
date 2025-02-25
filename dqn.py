import torch
from network import NNModel, NNAgent


class DQNAgent(NNAgent):
    def __init__(
        self,
        observation_space,
        action_space,
        env,
        training_settings,
        model,
        quantise_inputs=False,
    ):
        super().__init__(
            observation_space,
            action_space,
            env,
            training_settings,
            model,
            quantise_inputs,
        )
        self.target_model = NNModel(observation_space, action_space)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

    def update_estimate(self, old_state, action, reward, new_state):
        self.model.train()
        self.optim.zero_grad()

        old_state_tensor = self.prepare_input(old_state)
        new_state_tensor = self.prepare_input(new_state)

        old_q_values = self.model(old_state_tensor)
        new_q_values = self.target_model(new_state_tensor)

        target_q_values = old_q_values.clone()
        target_q_values[action] = reward + self.settings.gamma * torch.max(new_q_values)

        loss = self.criterion(old_q_values, target_q_values)
        loss.backward()

        self.optim.step()
        self.loss_metric.append(loss.item())

        self.steps += 1
        self.epsilon_greedy_factor = (
            self.settings.epsilon_greedy_factor * (1 - 1e-6) ** self.steps
        )

        if self.steps % 1000 == 0:
            self.update_target_model()

        return loss.item()


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

# Training settings
TOTAL_TRAINING_STEPS = 10000000
GAMMA_DISCOUNT_FACTOR = 0.99

# check if neural nets exist

model_path = f"{Q_TABLE_FOLDER}/model_{GAME}.pt"
target_model_path = f"{Q_TABLE_FOLDER}/target_model_{GAME}.pt"

settings = TrainingSettings()

model = NNModel(observation.shape[0], env.action_space.n)

if os.path.isfile(model_path):
    print("loading existing model")
    model.load_state_dict(torch.load(model_path))
else:
    print("Initialising new model")

agent = DQNAgent(observation.shape, env.action_space.n, env, settings, model)

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
        plotter.register_datapoint(simulation_return, "DQNAgent")

        # Reset for the next episode
        simulation_return = 0.0
        discount_factor_tracker = 1.0

env.close()

# Save final Q-table
agent.save(model_path)

# Plot returns
plotter.plot()
