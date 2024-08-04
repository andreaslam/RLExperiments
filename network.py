import torch
import torch.nn as nn
import torch.optim as optim
import random
from base_agent import Agent
import numpy as np


class NNAgent(Agent):
    def __init__(self, observation_space, action_space, env, training_settings, model):
        super().__init__(observation_space, action_space, env, training_settings)
        self.model = model
        # LinearNNModel(torch.from_numpy(self.observation_space).reshape(-1), action_space)
        self.optim = optim.AdamW(
            self.model.parameters(), lr=self.settings.initial_learning_rate
        )
        self.criterion = nn.MSELoss()
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.steps = 0
        self.num_optimal = 0

    def get_action(self, state, greedy=False):
        self.model.eval()
        outputs = self.model(state)

        if greedy:
            action = torch.argmax(outputs)
        else:
            if (
                random.random() < self.settings.epsilon_greedy_factor
                or np.sqrt(self.num_optimal) * self.settings.exploratory_constant
                > (1 - self.settings.epsilon_greedy_factor) * self.steps
            ):
                probability_distribution = self.softmax(outputs)
                action = np.random.choice(len(outputs), p=probability_distribution)
                noise = np.random.normal(
                    np.mean(outputs), np.std(outputs), len(outputs)
                )
                q_entry += noise
            else:
                action = np.argmax(outputs)
                self.num_optimal += 1
        return action

    def update_estimate(self, state, action, reward, next_state):
        self.model.zero_grad()

        current_q = self.model(state)

        next_q = self.model(next_state)

        self.model.train()

        best_next_action = torch.max(next_q)

        td_target = reward + self.settings.gamma_discount_factor * best_next_action

        loss = self.criterion(current_q[action], td_target)

        self.model.backward()

        self.optim.step()

        self.model.eval()

        self.steps += 1

    def save(self, file_path):
        torch.jit.save(self.model, file_path)

    def load(self, file_path):
        self.model = torch.jit.load(file_path, map_location=self.device).eval()


class LinearNetModel(nn.Module):
    def __init__(
        self, num_input_nodes, num_output_nodes, num_hidden_nodes=64, num_layers=1
    ):
        super().__init__()
        self.flatten = nn.Flatten()
        self.hidden_layers = []
        for _ in range(num_layers):
            self.hidden_layers.append(nn.Linear(num_hidden_nodes, num_hidden_nodes))
            nn.ReLU()
        self.state_action_value_head = nn.Sequential(
            nn.Linear(num_input_nodes, num_hidden_nodes),
            nn.ReLU(),
            *self.hidden_layers,
            nn.ReLU(),
            nn.Linear(num_hidden_nodes, num_output_nodes),
        )

    def forward(self, x):
        x = self.flatten(x)
        policy_outputs = self.state_action_value_head(x)
        return policy_outputs


class ResNetModel(nn.Module):
    def __init__(
        self, num_input_nodes, num_output_nodes, num_hidden_nodes=64, num_layers=1
    ):
        super().__init__()
        self.flatten = nn.Flatten()
        self.hidden_layers = []
        for _ in range(num_layers):
            self.hidden_layers.append(ResidualBlock(num_hidden_nodes))
        self.state_action_value_head = nn.Sequential(
            nn.Linear(num_input_nodes, num_hidden_nodes),
            nn.ReLU(),
            *self.hidden_layers,
            nn.Linear(num_hidden_nodes, num_output_nodes),
        )

    def forward(self, x):
        x = self.flatten(x)
        policy_outputs = self.state_action_value_head(x)
        return policy_outputs


class ResidualBlock(nn.Module):
    def __init__(self, num_hidden_nodes):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(num_hidden_nodes, num_hidden_nodes)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(num_hidden_nodes, num_hidden_nodes)

    def forward(self, x):
        residual = x
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out += residual
        out = self.relu(out)
        return out
