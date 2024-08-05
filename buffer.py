import random
import torch


class ReplayBuffer:
    def __init__(self, replay_capacity, batch_size):
        assert (
            replay_capacity > batch_size
        ), "Replay capacity must be larger than batch size!"

        self.batch_size = batch_size
        self.replay_capacity = replay_capacity
        self.buffer = [None] * self.replay_capacity
        self.index = 0
        self.size = 0

    def append(self, new_position):
        """
        Adds new items to the replay buffer.

        Args:
            new_position: the game position to append.
        """
        self.buffer[self.index] = new_position
        self.size = min(self.size + 1, self.replay_capacity)
        self.index = (self.index + 1) % self.replay_capacity

    def sample(self):
        """
        Samples a number of items (determined by `ReplayBuffer.batch_size`) from the buffer.
        """
        indices = random.sample(range(self.size), self.batch_size)
        return [self.buffer[index] for index in indices]

    def __len__(self):
        return len(self.buffer)

    def __repr__(self):
        return f"ReplayBuffer(\n    self.replay_capacity={self.replay_capacity}),\n    self.batch_size={self.batch_size},\n    self.buffer={self.buffer}\n)"


class Position:

    """
    Stores data for each position, with observation
    """

    def __init__(
        self,
        position_data,
        predicted_action_value,
    ):
        self.predicted_action_value = predicted_action_value
        (
            self.observation,
            self.reward,
            self.terminated,
            self.truncated,
            self.info,
        ) = position_data

        self.target = None
