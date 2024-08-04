import random
import numpy as np
import pickle
from base_agent import Agent


class TDTabularAgent(Agent):
    def __init__(self, observation_space, action_space, env, training_settings):
        super().__init__(observation_space, action_space, env, training_settings)

        self.table = {}
        self.td_delta_metric = []

        self.linspace_range = None

        self.steps = 0
        self.delta_prev = 0.0
        self.total_weights = 0.0
        self.weights_average = 0.0
        self.weights_stdev = 0.0
        self.num_optimal = 0
        self.previous_performance = 0.0

        self.discretise_inputs()

    def check_state_exists(self, state):
        return self.table.get(state, None)

    def get_action(self, state, greedy=False):
        q_entry = self.check_state_exists(state)
        if q_entry is None:
            q_entry = self.add_entry(state)
        if greedy:
            action = np.argmax(q_entry)
        else:
            if (
                random.random() < self.settings.initial_epsilon_greedy_factor
                or np.sqrt(self.num_optimal) * self.settings.exploratory_constant
                > (1 - self.settings.initial_epsilon_greedy_factor) * self.steps
            ):
                probability_distribution = self.softmax(q_entry)
                action = np.random.choice(len(q_entry), p=probability_distribution)
                noise = np.random.normal(
                    np.mean(q_entry), np.std(q_entry), len(q_entry)
                )
                q_entry += noise
            else:
                action = np.argmax(q_entry)
                self.num_optimal += 1
        return action

    def update_estimate(self, state, action, reward, next_state):
        current_q = self.check_state_exists(state)
        assert current_q is not None, f"No Q-value entry found for state: {state}"

        next_q = self.check_state_exists(next_state)
        if next_q is None:
            next_q = self.add_entry(next_state)

        best_next_action = np.max(next_q)

        td_target = reward + self.settings.gamma_discount_factor * best_next_action
        td_delta = td_target - current_q[action]
        current_q[action] += self.settings.initial_learning_rate * td_delta

        self.steps += 1
        self.td_delta_metric.append(td_delta)

        if self.steps % self.settings.performance_check_interval == 0:
            self.adjust_parameters()

    def add_entry(self, new_state):
        """
        Adds a new state entry to the Q-table with initial random Q-values.

        Args:
            new_state: New state to add to the Q-table.

        Returns:
            list: Initial Q-values assigned to the new state.
        """

        new_entry = [
            random.uniform(
                self.weights_stdev - random.random(),
                self.weights_stdev + random.random(),
            )
            for _ in range(self.action_space)
        ]

        self.table[new_state] = new_entry

        self.calibrate_new_entries(new_entry)

        return new_entry

    def calibrate_new_entries(self, new_entry):
        """
        Calibrate the mean and standard deviation for new entries

        Args:
            new_entry: New entry to be added to the Q-Table

        """

        self.total_weights += sum(new_entry)
        self.weights_average = self.total_weights / len(self.table)
        self.weights_stdev = np.sqrt(
            max(self.total_weights / (len(self.table) - self.weights_average**2), 0)
        )

    def discretise_inputs(self):
        """

        Setting the quantisation standard for incoming observations and is stored at `TDTabularAgent.linspace_range`

        The number of states to sample is determined by `TrainingSettings.num_states_in_linspace`

        Maximum value = 100

        Minimum value = 0.001

        """

        lows = self.env.observation_space.low
        highs = self.env.observation_space.high
        num_states = self.settings.num_states_in_linspace

        linspace_ranges = []
        for low, high in zip(lows, highs):
            low = np.clip(low, self.settings.low_limit, self.settings.high_limit)
            high = np.clip(high, self.settings.low_limit, self.settings.high_limit)
            low = max(low, 0.001)
            high = min(high, 100)
            linspace_ranges.append(np.linspace(low, high, num_states))

        self.linspace_range = np.array(linspace_ranges)

    def quantise_to_linspace(self, values):
        """

        Turn continuous observation spaces to discrete values by quantising raw observation data to the nearest quantised state.

        Args:
            values tuple(numpy.ndarray): values to be quantised

        Returns:
            tuple(numpy.ndarray): quantised values

        """

        quantised_values = np.empty_like(values)
        for i, (val, linspace) in enumerate(zip(values, self.linspace_range)):
            quantised_values[i] = linspace[np.argmin(np.abs(linspace - val))]
        return quantised_values

    def decrease_parameters(self):
        """
        Increases the Learning Rate (LR) and epsilon-greedy factor (eps).
        """

        self.settings.initial_learning_rate = abs(
            max(
                self.settings.initial_learning_rate
                * self.settings.parameter_decay_factor,
                self.settings.minimum_learning_rate,
            )
            + random.uniform(-0.0001, 0.0001)
        )
        self.settings.initial_epsilon_greedy_factor = abs(
            max(
                self.settings.initial_epsilon_greedy_factor
                * self.settings.parameter_decay_factor,
                self.settings.minimum_epsilon_greedy_factor,
            )
            + random.uniform(
                -0.0001,
                0.0001,
            )
        )

    def increase_parameters(self):
        """
        Increases the Learning Rate (LR) and epsilon-greedy factor (eps).
        """

        max_lr, max_eps = 0.90, 0.90

        self.settings.initial_learning_rate = abs(
            min(
                self.settings.initial_learning_rate
                / self.settings.parameter_decay_factor,
                max_lr,
            )
            + random.uniform(-0.001, 0.001)
        )
        self.settings.initial_epsilon_greedy_factor = abs(
            min(
                self.settings.initial_epsilon_greedy_factor
                / self.settings.parameter_decay_factor,
                max_eps,
            )
            + random.uniform(-0.001, 0.001)
        )

    def adjust_parameters(self):
        """
        Adjusts the Learning Rate (LR) and epsilon-greedy factor (eps) depending on previous performance
        """

        parameter_adjustment_metric = np.mean(
            np.abs(
                self.td_delta_metric[
                    -min(
                        self.settings.performance_check_history,
                        len(self.td_delta_metric),
                    ) :
                ]
            )
        )
        percentage_change_delta = (
            parameter_adjustment_metric - self.previous_performance
        ) / max(self.previous_performance, 1)
        if percentage_change_delta < -self.settings.performance_threshold:
            self.decrease_parameters()
        else:
            self.increase_parameters()

        self.previous_performance = parameter_adjustment_metric

    def save(self, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(self.table, f)

    def load(self, file_path):
        with open(file_path, "rb") as f:
            self.table = pickle.load(f)
