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
        parameter_decay_factor=0.97,
        performance_threshold=0.1,
        performance_check_interval=10,
        lr_increase_threshold=5,
        eps_increase_threshold=5,
        exploratory_constant=1.5,
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
        self.performance_threshold = performance_threshold
        self.performance_check_interval = performance_check_interval
        self.performance_metric = []
        self.num_optimal = 0
        self.lr_increase_count = 0
        self.lr_increase_threshold = lr_increase_threshold
        self.eps_increase_count = 0
        self.eps_increase_threshold = eps_increase_threshold
        self.exploratory_constant = exploratory_constant
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
        
        probability_distribution = self.softmax(q_entry)
        
        if random.random() < self.epsilon_greedy_factor or np.sqrt(self.num_optimal) * self.exploratory_constant > (1 - self.epsilon_greedy_factor) * self.steps:
            action_logit = np.random.choice(q_entry, p=probability_distribution)
            action = q_entry.index(action_logit)
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
        # print(reward, self.learning_rate, self.epsilon_greedy_factor)
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
        self.performance_metric.append(td_delta)

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

    def adjust_parameters(self):
        """
        Adjusts the learning rate and epsilon value based on performance.
        """
        performance = np.mean(np.abs(self.performance_metric[-self.performance_check_interval:]))
        # Decrease learning rate and epsilon factor if performance is poor
        if performance > self.performance_threshold:
            if self.lr_increase_count < self.lr_increase_threshold:
                self.learning_rate = max(self.learning_rate / self.parameter_decay_factor, 0.0001)
                self.lr_increase_count += 1
            if self.eps_increase_count < self.eps_increase_threshold:
                self.epsilon_greedy_factor = max(self.epsilon_greedy_factor / self.parameter_decay_factor, 0.01)
                self.eps_increase_count += 1
        # Increase learning rate and epsilon factor if performance is good
        else:
            if self.lr_increase_count > 0:
                self.learning_rate = min(self.learning_rate * self.parameter_decay_factor, 0.99)
                self.lr_increase_count -= 1
            if self.eps_increase_count > 0:
                self.epsilon_greedy_factor = min(self.epsilon_greedy_factor * self.parameter_decay_factor, 0.99)
                self.eps_increase_count -= 1
