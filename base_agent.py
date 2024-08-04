import numpy as np
from abc import ABC, abstractmethod


class Agent(ABC):
    def __init__(self, observation_space, action_space, env, training_settings):
        """
        Attributes:
            observation_space: Description of raw observation from environment.
            action_space (int): Number of possible actions.
            env: Training environment for the `Agent`
            training_settings (TrainingSettings): training settings for the Reinforcement Learning Agent

        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.env = env
        self.settings = training_settings

    def softmax(self, x):
        """
        Compute softmax values for each sets of scores in x.
        """
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    @abstractmethod
    def get_action(self, state, greedy=False):
        """
        Picks an action to play in the environment given the current state.

        Args:
            state: The current state
            greedy (bool): Whether the agent picks only the optimal moves deemed by the agent

        Returns:
            action: Action to be played in the environment
        """

        ...

    @abstractmethod
    def update_estimate(self, state, action, reward, next_state):
        """
        Updates the Q-value estimate based on the observed reward and next state.

        Args:
            state: Current state.
            action: Action taken in the current state.
            reward: Reward received after taking the action.
            next_state: Next state observed after taking the action.
        """

        ...

    @abstractmethod
    def save(self, file_path):
        """
        Saves the Agent's data to a pickle file.

        Args:
            file_path (str): File path where the data should be saved.
        """
        ...

    @abstractmethod
    def load(self, file_path):
        """
        Loads the Agent's data from a pickle file.

        Args:
            file_path (str): File path from which the data should be loaded.
        """
        ...
