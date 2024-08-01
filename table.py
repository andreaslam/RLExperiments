import random
import numpy as np


class Agent:
    def __init__(
        self,
        table,
        action_space,
        gamma_discount_factor=0.9,
        epsilon_greedy_factor=0.99,
        learning_rate=1e-3,
    ):
        self.table = table
        self.gamma_discount_factor = gamma_discount_factor
        self.epsilon_greedy_factor = epsilon_greedy_factor
        self.learning_rate = learning_rate
        self.action_space = action_space

    def check_state_exists(self, state):
        for entry in self.table:
            if np.array_equal(entry[0], state):
                return entry
        return None

    def get_action(self, state):
        """

        `action` refers to the index of the q_values within the q_value sublist of q_entry
         q_entry = [
            [state_0, [q_value_a, q_value_b, q_value_c, ...],
            [state_1, [q_value_d, q_value_e, q_value_f, ...],
            ... ]
            ]
        """

        q_entry = self.check_state_exists(state)

        if q_entry is None:
            q_entry = self.add_entry(state)

        if random.random() > self.epsilon_greedy_factor:
            action = random.randint(0, self.action_space - 1)
        else:
            action = np.argmax(q_entry[1])

        return action

    def update_q_estimate(self, state, action, reward, next_state):
        current_q = self.check_state_exists(state)

        assert current_q is not None

        next_q = self.check_state_exists(next_state)
        if next_q is None:
            next_q = self.add_entry(next_state)

        best_next_action = np.argmax(next_q[1])

        td_target = reward + self.gamma_discount_factor * next_q[1][best_next_action]
        td_delta = td_target - current_q[1][action]

        current_q[1][action] += self.learning_rate * td_delta

    def add_entry(self, new_state):
        new_entry = [
            new_state,
            [random.uniform(-1, 1) for _ in range(self.action_space)],
        ]
        self.table.append(new_entry)
        return new_entry
