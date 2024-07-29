import random
import torch


class ReplayBuffer:
    def __init__(self, max_buffer_size=2048, batch_size=16):
        assert (
            max_buffer_size > batch_size
        ), "batch size must be smaller than buffer size!"
        self._predictions = []
        self._targets = []
        self.batch_size = batch_size
        self.max_buffer_size = max_buffer_size

    def import_positions(self, simulation_as_positions):
        preds, targets = simulation_as_positions

        self._predictions.append(preds)
        self._targets.append(targets)

    def clear_buffer(self):
        self._predictions = []
        self._targets = []

    def sample_batch(self):
        # zip the observations and predictions and shuffle to get randomised corresponding observation : prediction pairs
        assert (
            self._predictions == self._targets
        ), "predictions and targets must have same length!"
        zipped_predictions_targets = list(zip(self._predictions, self._targets))
        random.shuffle(zipped_predictions_targets)
        self._predictions, self._targets = zip(*zipped_predictions_targets)
        training_preds, self._predictions, training_targets, self._targets = (
            self._predictions[: self.batch_size],
            self._predictions[self.batch_size :],
            self._targets[: self.batch_size],
            self._targets[self.batch_size :],
        )

        self._predictions = list(self._predictions)
        self._targets = list(self._targets)

        training_preds = [
            tensor for sub_tuple in training_preds for tensor in sub_tuple
        ]
        training_targets = [
            tensor for sub_tuple in training_targets for tensor in sub_tuple
        ]

        return torch.stack(training_preds, dim=0), torch.stack(training_targets, dim=0)

    def get_buffer_capacity(self):
        assert (
            len(self._predictions) == len(self._targets),
            "predictions and targets should be of the same size",
        )
        return sum([len(x) for x in self._predictions])


class Simulation:
    def __init__(self, gamma_discount=0.9):
        self._positions = []
        self._discounted_returns_target = []
        self._discounted_returns_preds = []
        self.gamma_discount = gamma_discount

    def append_position(self, item):
        self._positions.append(item)

    def prepare_preds(self):
        return [position.predicted_action_value for position in self._positions]

    def prepare_targets(self):
        # Q^{pi}(s,a) = E_{s' ~ P}[r_{t} + Q(s_{t+1}, a_{t+1})]

        # Q(s_0,a_0) = Q(s_0, a_0) + Q(s_1, a_1)

        # Q(s_1, a_1) = Q(s_1, a_1) + Q(s_2, a+2)

        total_rewards = []
        cumulated_reward = 0.0
        for time_step, position in enumerate(reversed(self._positions)):
            cumulated_reward += position.reward * (self.gamma_discount**time_step)
            total_rewards.insert(0, cumulated_reward)

        targets = []

        for position, updated_reward in zip(self._positions, total_rewards):
            predicted_action = torch.max(position.predicted_action_value)
            mask = position.predicted_action_value == predicted_action
            position.predicted_action_value[mask] = updated_reward
            targets.append(position.predicted_action_value)
        return targets

    def export_positions(self):
        preds = self.prepare_preds()
        targets = self.prepare_targets()

        return (preds, targets)

    def get_positions(self):
        return self._positions


# metadata format according to https://gymnasium.farama.org/api/env/#gymnasium.Env.step
class Position:
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
