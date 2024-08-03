import random
import numpy as np


class Agent:
    """
    Represents a reinforcement learning agent using Q-learning.

    Attributes:
        table (dict): Dictionary to store Q-values for each state.
        action_space (int): Number of possible actions.
        gamma_discount_factor (float): Discount factor for future rewards.
        epsilon_greedy_factor (float): Factor for exploration-exploitation trade-off.
        learning_rate (float): Rate at which the agent learns from new information.
    """

    def __init__(
        self,
        table,
        action_space,
        env,
        gamma_discount_factor=0.9,
        initial_epsilon_greedy_factor=0.35,
        initial_learning_rate=3e-1,
        num_states_in_linspace=10,
        parameter_decay_factor=0.9,
        performance_threshold=0.15,
        performance_check_interval=5,
        performance_check_history=3,
        exploratory_constant=1.2,
        low_limit=-10,
        high_limit=10,
        minimum_learning_rate=0.01,
        minimum_epsilon_greedy_factor=0.01,
    ):
        """
        Initializes the Agent with necessary parameters.

        Args:
            table (dict): Dictionary to store Q-values for each state.
            action_space (int): Number of possible actions.
            gamma_discount_factor (float): the factor that [discounts rewards](https://en.wikipedia.org/wiki/Reinforcement_learning#State-value_function) further away from the current time step, which effectively determines the "importance" of the reward r at timestep t
            epsilon_greedy_factor (float): balances the [exploration-exploitation tradeoff](https://en.wikipedia.org/wiki/Exploration-exploitation_dilemma) by allowing the agent to make a random move without regard to the Q-value
            learning_rate (float): refers to the size of the increment the agent takes to update current predictions with new predictions based on the [TD update algorithm](https://en.wikipedia.org/wiki/Temporal_difference_learning)

        """
        self.table = table
        self.gamma_discount_factor = gamma_discount_factor
        self.epsilon_greedy_factor = initial_epsilon_greedy_factor
        self.learning_rate = initial_learning_rate
        self.action_space = action_space
        self.num_states_in_linspace = num_states_in_linspace
        self.linspace_range = None
        self.env = env
        self.steps = 0
        self.delta_prev = 0.0
        self.total_weights = 0.0
        self.weights_average = 0.0
        self.weights_stdev = 0.0
        self.parameter_decay_factor = parameter_decay_factor
        self.performance_target = performance_threshold
        self.performance_check_interval = performance_check_interval
        self.td_delta_metric = []
        self.num_optimal = 0
        self.exploratory_constant = exploratory_constant
        self.minimum_learning_rate = minimum_learning_rate
        self.minimum_epsilon_greedy_factor = minimum_epsilon_greedy_factor
        self.low_limit = low_limit
        self.high_limit = high_limit
        self.performance_check_history = performance_check_history
        self.previous_performance = 0.0

        self.discretise_inputs()

    def check_state_exists(self, state):
        """
        Checks if the given state exists in the Q-table.

        Args:
            state: State to check in the Q-table.

        Returns:
            list or None: Q-values associated with the state if it exists, else None.
        """
        return self.table.get(state, None)

    def get_action(self, state):
        """
        Selects an action based on epsilon-greedy policy.

        Args:
            state: Current state for which action needs to be selected.

        Returns:
            int: Selected action index.
        """
        q_entry = self.check_state_exists(state)
        if q_entry is None:
            q_entry = self.add_entry(state)

        if (
            random.random() < self.epsilon_greedy_factor
            or np.sqrt(self.num_optimal) * self.exploratory_constant
            > (1 - self.epsilon_greedy_factor) * self.steps
        ):
            probability_distribution = self.softmax(q_entry)
            action = np.random.choice(len(q_entry), p=probability_distribution)
            noise = np.random.normal(np.mean(q_entry), np.std(q_entry), len(q_entry))
            q_entry += noise
        else:
            action = np.argmax(q_entry)
            self.num_optimal += 1
        return action

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def update_q_estimate(self, state, action, reward, next_state):
        """
        Updates the Q-value estimate based on the observed reward and next state.

        Args:
            state: Current state.
            action: Action taken in the current state.
            reward: Reward received after taking the action.
            next_state: Next state observed after taking the action.
        """

        current_q = self.check_state_exists(state)
        assert current_q is not None, f"No Q-value entry found for state: {state}"

        next_q = self.check_state_exists(next_state)
        if next_q is None:
            next_q = self.add_entry(next_state)

        best_next_action = np.max(next_q)

        td_target = reward + self.gamma_discount_factor * best_next_action
        td_delta = td_target - current_q[action]
        current_q[action] += self.learning_rate * td_delta

        self.steps += 1
        self.td_delta_metric.append(td_delta)

        if self.steps % self.performance_check_interval == 0:
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
        self.total_weights += sum(new_entry)
        self.weights_average = self.total_weights / len(self.table)
        self.weights_stdev = np.sqrt(
            max(self.total_weights / (len(self.table) - self.weights_average**2), 0)
        )

    def discretise_inputs(self):
        lows = self.env.observation_space.low
        highs = self.env.observation_space.high
        num_states = self.num_states_in_linspace

        linspace_ranges = []
        for low, high in zip(lows, highs):
            low = np.clip(low, self.low_limit, self.high_limit)
            high = np.clip(high, self.low_limit, self.high_limit)
            low = max(low, 0.001)
            high = min(high, 100)
            linspace_ranges.append(np.linspace(low, high, num_states))

        self.linspace_range = np.array(linspace_ranges)

    def quantise_to_linspace(self, values):
        return np.array(
            [
                linspace[np.argmin(np.abs(linspace - val))]
                for val, linspace in zip(values, self.linspace_range)
            ]
        )

    def decrease_parameters(self):
        self.learning_rate = abs(
            max(
                self.learning_rate * self.parameter_decay_factor,
                self.minimum_learning_rate,
            )
            + random.uniform(-0.0001, 0.0001)
        )
        self.epsilon_greedy_factor = abs(
            max(
                self.epsilon_greedy_factor * self.parameter_decay_factor,
                self.minimum_epsilon_greedy_factor,
            )
            + random.uniform(
                -0.0001,
                0.0001,
            )
        )

    def increase_parameters(self):
        max_lr, max_eps = 0.90, 0.90

        self.learning_rate = abs(
            min(self.learning_rate / self.parameter_decay_factor, max_lr)
            + random.uniform(-0.001, 0.001)
        )
        self.epsilon_greedy_factor = abs(
            min(self.epsilon_greedy_factor / self.parameter_decay_factor, max_eps)
            + random.uniform(-0.001, 0.001)
        )

    def adjust_parameters(self):
        performance = np.mean(
            np.abs(
                self.td_delta_metric[
                    -min(self.performance_check_history, len(self.td_delta_metric)) :
                ]
            )
        )
        improvement_percent = (performance - self.previous_performance) / max(
            self.previous_performance, 1
        )
        if improvement_percent < -self.performance_target:
            self.decrease_parameters()
        else:
            self.increase_parameters()

        self.previous_performance = performance
