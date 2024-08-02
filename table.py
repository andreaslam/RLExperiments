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
        gamma_discount_factor=0.99,
        initial_epsilon_greedy_factor=0.25,
        initial_learning_rate=1e-1,
        num_states_in_linspace=100,
        parameter_decay_factor=0.9,
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
        # TODO work on getting the average and stdev for preloaded tables
        self.weights_average = 0.0
        self.weights_stdev = 0.0
        self.parameter_decay_factor = parameter_decay_factor
        self.num_optimal = 0 # number of times the agent chose the greedy move
        self.discretise_inputs()

    def check_state_exists(self, state):
        """
        Checks if the given state exists in the Q-table.

        Args:
            state: State to check in the Q-table.

        Returns:
            list or None: Q-values associated with the state if it exists, else None.
        """
        if state in self.table:
            return self.table[state]
        return None

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
        
        if random.random() < self.epsilon_greedy_factor:
            action_logit = np.random.choice(q_entry, p=self.softmax(q_entry))
            action = q_entry.index(action_logit)
            self.num_optimal = 0
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

        # print(f"Updated Q-value for state {state}, action {action}: {current_q}")

        self.steps += 1

    def add_entry(self, new_state):
        """
        Adds a new state entry to the Q-table with initial random Q-values.

        Args:
            new_state: New state to add to the Q-table.

        Returns:
            list: Initial Q-values assigned to the new state.
        """

        # random.random() serves as fallback for if weights and standard deviation are 0

        new_entry = [
            random.uniform(
                self.weights_average - self.weights_stdev - random.random(),
                self.weights_average + self.weights_stdev + random.random(),
            )
            for _ in range(self.action_space)
        ]
        
        # print(self.weights_average, self.weights_stdev, new_entry)
        
        self.table[new_state] = new_entry

        self.calibrate_new_entries(new_entry)

        return new_entry

    def calibrate_new_entries(self, new_entry):
        self.total_weights += sum(new_entry)
        self.weights_average = self.total_weights / len(self.table)
        self.weights_stdev = np.sqrt(
            max(self.total_weights / (len(self.table) - self.weights_average**2), 0)
        ).item()

    def discretise_inputs(self):
        def safe_bounds(low, high):
            if low == -np.inf:
                low = -100
            if high == np.inf:
                high = 100
            return low, high

        self.linspace_range = np.array(
            [
                np.linspace(max(low, 0.001), min(high, 100), self.num_states_in_linspace)
                for low, high in [safe_bounds(low, high) for low, high in zip(self.env.observation_space.low, self.env.observation_space.high)]
            ]
        )
    
    def quantise_to_linspace(self, values):
        quantised_values = np.empty_like(values)
        for i, (val, linspace) in enumerate(zip(values, self.linspace_range)):
            quantised_values[i] = linspace[np.argmin(np.abs(linspace - val))]
        return quantised_values
