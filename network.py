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
        self.criterion = nn.HuberLoss()
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model = model.to(self.device)
        self.optim = optim.AdamW(
            self.model.parameters(), lr=self.settings.learning_rate
        )

        self.loss_metric = []

        self.outputs = None  # stores current neural network evalution to be accessed when appending buffer

        self.steps = 0
        self.num_optimal = 0
        self.previous_performance = 0.0
        self.learning_rate = (
            self.settings.learning_rate
        )  # self.learning_rate is the one to be modified
        self.epsilon_greedy_factor = (
            self.settings.epsilon_greedy_factor
        )  # self.epsilon_greedy_factor is the one to be modified

    def check_state_exists(self, state):
        return self.table.get(tuple(state), None)

    def get_action(self, state, greedy=False):
        state_tensor = self.prepare_input(state)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(state_tensor).squeeze()
            self.outputs = self.outputs.cpu().numpy()

        if greedy:
            action = np.argmax(outputs)
        else:
            if (
                random.random()
                < self.epsilon_greedy_factor
            ):
                action = np.random.choice(len(outputs))
            else:
                probability_distribution = self.softmax(outputs)
                action = np.random.choice(len(outputs), p=probability_distribution)

                self.num_optimal += 1
        return action

    def update_estimate_online(self, state, action, reward, next_state):
        """
        Updates the Q-value estimate based on the observed reward and next state:

        Args:
            state: Current state.
            action: Action taken in the current state.
            reward: Reward received after taking the action.
            next_state: Next state observed after taking the action.

        Note: This is for [online learning updates](https://huggingface.co/learn/deep-rl-course/en/unitbonus3/offline-online) in Temporal Difference Learning and may have unstable training results due to potential high variance between update step.
        """
        current_q, td_target = self.prepare_training_targets(state, reward, next_state)

        self.model.train()
        self.optim.zero_grad()

        loss = self.criterion(current_q[action], td_target)
        loss.backward()
        self.loss_metric.append(float(loss))
        self.optim.step()
        if self.steps % self.settings.performance_check_interval == 0:
            self.adjust_hyperparameters()
        self.steps += 1

    def update_estimate(self, state, action, reward, new_state):
        """
        Updates the Q-value estimates based on the observed reward and next state given a batch of inputs and targets.
        Args:
            state: A batch of sampled states.
            action: A batch of actions taken in each state.
            reward: A batch of rewards received after taking each action.
            next_state: A batch of next states observed after taking each action.

        Note: This is for [offline learning](https://huggingface.co/learn/deep-rl-course/en/unitbonus3/offline-online) in Temporal Difference Learning
        """
        current_q, td_target = self.prepare_training_targets(state, reward, new_state)

        self.model.train()
        self.optim.zero_grad()

        loss = self.criterion(current_q[action], td_target)
        loss.backward()
        self.loss_metric.append(float(loss))
        self.optim.step()
        if self.steps % self.settings.performance_check_interval == 0:
            self.adjust_hyperparameters()
        self.steps += 1

    def prepare_training_targets(self, state, reward, next_state):
        state_tensor = self.prepare_input(state)
        next_state_tensor = self.prepare_input(next_state)

        current_q = self.model(state_tensor).squeeze()
        next_q = self.model(next_state_tensor).squeeze()

        best_next_action = torch.max(next_q).detach()
        td_target = reward + self.settings.gamma_discount_factor * best_next_action
        td_target = td_target.to(self.device)
        return current_q, td_target

    def prepare_input(self, raw_observation):
        if self.quantise:
            processed_input = self.operate_quantise_on_inputs(raw_observation)
        else:
            processed_input = raw_observation
        return (
            torch.tensor(processed_input, device=self.device, dtype=torch.float32)
            .reshape(-1)
            .unsqueeze(0)
        )

    def adjust_hyperparameters(self):
        self.epsilon_greedy_factor = self.settings.minimum_epsilon_greedy_factor + (
            self.settings.epsilon_greedy_factor
            - self.settings.minimum_epsilon_greedy_factor
        ) * np.exp(-1.0 * self.steps / self.settings.parameter_decay_factor)
        self.learning_rate = self.settings.learning_rate + (
            self.settings.learning_rate - self.settings.minimum_learning_rate
        ) * np.exp(-1.0 * self.steps / self.settings.parameter_decay_factor)

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
