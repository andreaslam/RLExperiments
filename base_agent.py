import numpy as np
import warnings
from concurrent.futures import ThreadPoolExecutor
from abc import ABC, abstractmethod


class Agent(ABC):
    def __init__(
        self,
        observation_space,
        action_space,
        env,
        training_settings,
        quantise_inputs=True,
    ):
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
        self.quantise = quantise_inputs
        self.linspace_range = None
        if self.quantise:
            self.discretise_inputs()

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

    @abstractmethod
    def prepare_input(self, raw_observation):
        """
        Convert the observation obtained from the environment to a format usable for the `Agent`.
        """

    @abstractmethod
    def save(self, file_path):
        """
        Saves the `Agent`'s data to a file.

        Args:
            file_path (str): File path where the data should be saved.
        """

    @abstractmethod
    def load(self, file_path):
        """
        Loads the `Agent`'s data from a file.

        Args:
            file_path (str): File path from which the data should be loaded.
        """

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

    def operate_quantise_on_inputs(self, raw_observation):
        """
        Applies quantisation based on quantised steps set in `Agent.discretise_inputs()` to raw observation.

        Args:
            raw_observation: Raw observation to be quantised
        """

        # fallback for in case if `Agent.quantise_inputs` is modified (from false to tr)

        if self.linspace_range is None:
            warnings.warn(
                "Agent.quantise_inputs has been modified from False to True! Discretising inputs..."
            )
            self.discretise_inputs()

        quantised_values = np.empty_like(raw_observation)

        for i, linspace in enumerate(self.linspace_range):
            idx = np.abs(linspace - raw_observation[i]).argmin()
            quantised_values[i] = linspace[idx]

        return quantised_values

    @abstractmethod
    def adjust_hyperparameters(self):
        """
        Adjusts relevant hyperparameters depending on previous performance.
        """

    @abstractmethod
    def prepare_training_targets(self, state, reward, next_state):
        """
        Prepare training targets based on the TD(0) algorithm
        """
