# RLExperiments

## Training Loop

### Objectives of the Training Loop

#### Minimal, transferable and simple-to-maintain code where code involved use functions that abstract the details of the training code.

### Design Specifications

#### The `Agent` Class
The `Agent` class handles the storage, accessing and processing of Q-values stored. The learning algorithm is based on Temporal Difference learning (TD) of Q-values (state-action values). 

The `Agent` takes in the following parameters:

```py
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
```

- `gamma_discount_factor` (γ) is the factor that [discounts rewards](https://en.wikipedia.org/wiki/Reinforcement_learning#State-value_function) further away from the current time step, which effectively determines the "importance" of the reward r at timestep t

- `epsilon_greedy_factor` (ε) balances the [exploration-exploitation tradeoff](https://en.wikipedia.org/wiki/Exploration-exploitation_dilemma) by allowing the agent to make a random move without regard to the Q-value

- `learning_rate` refers to the size of the increment the agent takes to update current predictions with new predictions based on the [TD update algorithm](https://en.wikipedia.org/wiki/Temporal_difference_learning) 

#### The `Plotter` Class
The plotter class is a simple wrapper around `matplotlib.pyplot` and is responsible for storing and plotting datapoints. It is also able to plot different lines on the same graph, differentiated by the `label` attribute. Optional for training but helpful in tracking progress.


### Features of this codebase:

- Adaptability* of agent setup to different games
- Reusability of the Q-tables - updated tables can be loaded again to resume training and have independent Q-tables for each game.
- Option to watch the agent play a variable number of games in `play_game.py`


### Basic Usage

```py

from table import Agent
from tqdm import tqdm

GAME = "CartPole-v1"

env = gym.make(GAME)
q_table = []

observation, info = env.reset()
observation_space = observation.shape
action_space = env.action_space.n

agent = Agent(q_table, action_space, GAMMA_DISCOUNT_FACTOR, EPSILON_GREEDY_FACTOR)

TOTAL_TRAINING_STEPS = 100000
GAMMA_DISCOUNT_FACTOR = 0.99
EPSILON_GREEDY_FACTOR = 0.99

for time_step in tqdm(range(TOTAL_TRAINING_STEPS), desc="updating q tables"):
    action = agent.get_action(observation)

    observation_prev = observation
    observation, reward, terminated, truncated, info = env.step(action)

    agent.update_q_estimate(observation_prev, action, reward, observation)

    simulation_return += reward * (time_step**GAMMA_DISCOUNT_FACTOR)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

*Only applies to certain games. More specifically, deterministic games from [Farnama](https://gymnasium.farama.org/) that have a discrete action space with an observation of type `numpy.ndarray`.