# RLExperiments

## Training Loop

### Objectives of the Training Loop

#### Minimal, transferable and simple-to-maintain code where code involved use functions that abstract the details of the training code.

### Design Specifications

#### The `Agent` Class
The `Agent` class handles the storage, accessing and processing of Q-values stored. The learning algorithm is based on Temporal Difference learning (TD) of Q-values (state-action values). 

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
EPSILON_GREEDY_FACTOR = 0.01

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

*Only applies to certain games. More specifically, deterministic games from [Farama](https://gymnasium.farama.org/) that have a discrete action space with an observation of type `numpy.ndarray`.