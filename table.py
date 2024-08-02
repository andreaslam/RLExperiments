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
        gamma_discount_factor=0.99,
        epsilon_greedy_factor=0.01,
        learning_rate=1e-3,
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
        self.epsilon_greedy_factor = epsilon_greedy_factor
        self.learning_rate = learning_rate
        self.action_space = action_space
        self.hits = 0

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
        else:
            self.hits += 1
        if random.random() < self.epsilon_greedy_factor:
            action = random.randint(0, self.action_space - 1)
        else:
            action = np.argmax(q_entry)

        # print(f"Selected action: {action}")
        return action

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

    def add_entry(self, new_state):
        """
        Adds a new state entry to the Q-table with initial random Q-values.

        Args:
            new_state: New state to add to the Q-table.

        Returns:
            list: Initial Q-values assigned to the new state.
        """
        new_entry = [random.uniform(-1, 1) for _ in range(self.action_space)]
        self.table[new_state] = new_entry

        # print(f"Added new state entry: {new_state}, Q-values: {new_entry}")
        return new_entry
