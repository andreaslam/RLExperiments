class TrainingSettings:
    def __init__(
        self,
        gamma_discount_factor=0.9,
        initial_epsilon_greedy_factor=0.45,
        initial_learning_rate=1e-1,
        num_states_in_linspace=100,
        parameter_decay_factor=0.999,
        low_limit=-5,
        high_limit=5,
        minimum_learning_rate=0.1,
        minimum_epsilon_greedy_factor=0.1,
    ):
        """
        Settings for the Reinforcement Learning agent.

        Args:
            table (dict): Dictionary to store Q-values for each state.
            action_space (int): Number of possible actions.
            table (dict): Dictionary to store Q-values for each state.
            action_space (int): Number of possible actions.
            gamma_discount_factor (float): the factor that [discounts rewards](https://en.wikipedia.org/wiki/Reinforcement_learning#State-value_function) further away from the current time step, which effectively determines the "importance" of the reward r at timestep t
            epsilon_greedy_factor (float): balances the [exploration-exploitation tradeoff](https://en.wikipedia.org/wiki/Exploration-exploitation_dilemma) by allowing the agent to make a random move without regard to the Q-value
            learning_rate (float): refers to the size of the increment the agent takes to update current predictions with new predictions based on the [TD update algorithm](https://en.wikipedia.org/wiki/Temporal_difference_learning)
            num_states_in_linspace (int): Number of states to use for quantisation.
            parameter_decay_factor (float): Factor by which parameters decay over time.
            performance_threshold (float): Threshold for performance improvement.
            performance_check_interval (int): Interval for checking performance.
            performance_check_history (int): Number of previous steps to consider for performance.
            exploratory_constant (float): Constant for exploration-exploitation balance.
            low_limit (float): Lower limit for state quantisation.
            high_limit (float): Upper limit for state quantisation.
            minimum_learning_rate (float): Minimum learning rate allowed.
            minimum_epsilon_greedy_factor (float): Minimum epsilon-greedy factor allowed.
        """

        self.gamma_discount_factor = gamma_discount_factor
        self.epsilon_greedy_factor = initial_epsilon_greedy_factor
        self.learning_rate = initial_learning_rate
        self.num_states_in_linspace = num_states_in_linspace
        self.parameter_decay_factor = parameter_decay_factor
        self.low_limit = low_limit
        self.high_limit = high_limit
        self.minimum_epsilon_greedy_factor = minimum_epsilon_greedy_factor
        self.minimum_learning_rate = minimum_learning_rate
