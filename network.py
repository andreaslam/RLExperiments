import torch
import torch.nn as nn
import torch.optim as optim
import random
from base_agent import Agent
import numpy as np


class NNAgent(Agent):
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
            observation_space, action_space, env, training_settings, quantise_inputs
        )
        self.criterion = nn.MSELoss()
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model = model.to(self.device)
        self.optim = optim.AdamW(
            self.model.parameters(), lr=self.settings.initial_learning_rate
        )
        self.steps = 0
        self.num_optimal = 0

    def get_action(self, state, greedy=False):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.prepare_input(state)).squeeze().cpu().numpy()

        if greedy:
            action = np.argmax(outputs)
        else:
            if (
                random.random() < self.settings.initial_epsilon_greedy_factor
                or np.sqrt(self.num_optimal) * self.settings.exploratory_constant
                > (1 - self.settings.initial_epsilon_greedy_factor) * self.steps
            ):
                probability_distribution = self.softmax(outputs)
                action = np.random.choice(len(outputs), p=probability_distribution)
            else:
                action = np.argmax(outputs)
                self.num_optimal += 1
        return action

    def update_estimate(self, state, action, reward, next_state):
        """
        Updates the Q-value estimate (state-action value) based on the observed reward and next state.

        Args:
            state: Current state.
            action: Action taken in the current state.
            reward: Reward received after taking the action.
            next_state: Next state observed after taking the action.

        Note: This is for [online learning updates](https://huggingface.co/learn/deep-rl-course/en/unitbonus3/offline-online) in Temporal Difference Learning and may have unstable training results due to potential high variance between update step.
        """

        current_q = self.model(self.prepare_input(state)).squeeze()
        next_q = self.model(self.prepare_input(next_state)).squeeze()

        self.model.train()
        self.optim.zero_grad()

        best_next_action = torch.max(next_q)
        td_target = reward + self.settings.gamma_discount_factor * best_next_action
        loss = self.criterion(current_q[action], td_target)
        loss.backward()
        self.optim.step()

        self.steps += 1

    def prepare_input(self, raw_observation):
        if self.quantise:
            return (
                torch.tensor(
                    self.operate_quantise_on_inputs(raw_observation), device=self.device
                )
                .reshape(-1)
                .unsqueeze(0)
            )
        return (
            torch.tensor(raw_observation, device=self.device).reshape(-1).unsqueeze(0)
        )

    def save(self, file_path):
        torch.jit.save(self.model, file_path)

    def load(self, file_path):
        self.model = torch.jit.load(file_path, map_location=self.device).eval()


class LinearNetModel(nn.Module):
    def __init__(
        self,
        num_input_nodes,
        num_output_nodes,
        num_hidden_nodes=64,
        num_layers=1,
        dropout_prob=0.5,
    ):
        super().__init__()
        self.flatten = nn.Flatten()
        self.hidden_layers = nn.ModuleList()

        for _ in range(num_layers):
            self.hidden_layers.append(
                nn.Sequential(
                    nn.Linear(num_hidden_nodes, num_hidden_nodes),
                    nn.ReLU(),
                    nn.Dropout(dropout_prob),
                )
            )

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
