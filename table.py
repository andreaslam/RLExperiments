import random
import numpy as np
import pickle
from base_agent import Agent


class TDTabularAgent(Agent):
    def __init__(
        self,
        observation_space,
        action_space,
        env,
        training_settings,
        quantise_inputs=True,
    ):
        super().__init__(
            observation_space, action_space, env, training_settings, quantise_inputs
        )

        self.table = {}
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
        state = self.prepare_input(state)
        q_entry = self.check_state_exists(state)
        if q_entry is None:
            q_entry = self.add_entry(state)
        if greedy:
            action = np.argmax(q_entry)
        else:
            if (
                random.random()
                < self.epsilon_greedy_factor
            ):
                probability_distribution = self.softmax(q_entry)
                action = np.random.choice(len(q_entry), p=probability_distribution)
            else:
                action = np.argmax(q_entry)
                self.num_optimal += 1
        return action

    def update_estimate(self, state, action, reward, next_state):
        current_q, td_target = self.prepare_training_targets(state, reward, next_state)

        td_delta = td_target - current_q[action]
        
        current_q[action] += self.learning_rate * td_delta
        self.steps += 1
        self.adjust_hyperparameters()

    def add_entry(self, new_state):
        """
        Adds a new state entry to the Q-table with initial random Q-values.

        Args:
            new_state: New state to add to the Q-table.

        Returns:
            list: Initial Q-values assigned to the new state.
        """

        new_entry = np.random.uniform(0,0, self.action_space)
        self.table[tuple(new_state)] = new_entry
        return new_entry

    def prepare_input(self, raw_observation):
        """
        Turn continuous observation spaces to discrete values by quantising raw observation data to the nearest quantised state.

        Args:
            raw_observation tuple(numpy.ndarray): values to be quantised

        Returns:
            tuple(numpy.ndarray): quantised values
        """
        if self.quantise:
            return tuple(self.operate_quantise_on_inputs(raw_observation))
        return tuple(raw_observation)

    def adjust_hyperparameters(self):
        self.learning_rate = 30/(30+self.steps)
        self.epsilon_greedy_factor = 1 - 20/(20+self.steps)

    def save(self, file_path):
        self.table["metadata_settings"] = self.settings
        self.table["metadata_steps"] = self.steps

        with open(file_path, "wb") as f:
            pickle.dump(self.table, f)

    def load(self, file_path):
        with open(file_path, "rb") as f:
            self.table = pickle.load(f)

        if "metadata_settings" in self.table:
            self.settings = self.table["metadata_settings"]
        if "metadata_steps" in self.table:
            self.steps = self.table["metadata_steps"]

    def prepare_training_targets(self, state, reward, next_state):
        state, next_state = self.prepare_input(state), self.prepare_input(next_state)

        current_q = self.check_state_exists(state)
        assert current_q is not None, f"No Q-value entry found for state: {state}"
        next_q = self.check_state_exists(next_state)
        if next_q is None:
            next_q = self.add_entry(next_state)
        best_next_action = np.max(next_q)

        td_target = reward + self.settings.gamma_discount_factor * best_next_action
        return current_q, td_target
