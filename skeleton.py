from collections import defaultdict
import random
import statistics

import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

from table import Agent


env = gym.make("CartPole-v1")

discount = 0.9
turns = 1_000_000
actions = (0, 1)

# discretized state -> learned q-value
q_table = defaultdict(lambda: 0.0)

# discretized state -> number of times state has been seen
state_count = defaultdict(lambda: 0)


def greediness(count):
    """
    Given the number of times a state has been seen, return the probability of choosing
    the action with the highest q-value (as opposed to choosing a random action).

    The greediness should increase as the agent gains more experience with a state (and
    hence has less reason to explore). It should be a number between 0.0 and 1.0.
    """
    ...
    return 0.9


def learning_rate(count):
    """
    Return the learning rate as a function of the number of times a state has been seen.

    The learning rate should decrease with experience, so that what the agent has learned
    is effectively 'locked in' and not forgotten.
    """
    return 1e-3


def discretize(state, agent):
    """
    Take a state in the continuous state space and return a discrete version of that state.

    Discretization should group states into a relatively small number of "buckets", so that
    the agent can generalize from past experiences.
    """
    return (
        int(2.0 * state[0]),
        int(2.0 * state[1]),
        int(20.0 * state[2]),
        int(2.0 * state[3]),
    )


def choose_action(state, agent):
    """
    Given a state choose an action, weighing up the need to explore with the need to exploit
    what the agent has already learned for optimal behaviour.

    The agent should choose what it believes to be the 'optimal' action with probability
    `greediness(state)`, and otherwise choose a random action.
    """
    return agent.get_action(tuple(agent.quantise_to_linspace(state)))


def update_tables(state, action, new_state, reward, agent):
    """
    Update the `q_table` and `state_count` tables, based on the observed transition.
    """
    agent.update_q_estimate(
        tuple(agent.quantise_to_linspace(state)),
        action,
        reward,
        tuple(agent.quantise_to_linspace(new_state)),
    )


episode_lengths = []

episode_length = 0
state, info = env.reset()


agent = Agent(
    q_table,
    env.action_space.n,
    env,
    gamma_discount_factor=0.9,
    initial_epsilon_greedy_factor=0.35,
    initial_learning_rate=3e-1,
)

for turn in tqdm(range(turns), desc="updating q tables"):
    action = choose_action(state, agent)

    old_state = state
    state, reward, terminated, truncated, info = env.step(action)

    if terminated:
        reward = -100
    elif truncated:
        reward = +1
    else:
        reward = +1

    update_tables(old_state, action, state, reward, agent)
    # print(turn,action, state, reward,)
    episode_length += 1

    if terminated or truncated:
        episode_lengths.append(episode_length)
        # print("e_len", episode_length)
        episode_length = 0
        state, info = env.reset()


moving_average = [
    statistics.mean(episode_lengths[i : i + 100])
    for i in range(len(episode_lengths) - 100)
]
plt.plot(moving_average)
plt.show()
plt.savefig("skeleton.png")
