import os
import torch.nn as nn
import torch


class Trainer:
    def __init__(self, optim, criterion=nn.MSELoss()):
        folder_name = "agent_nets"
        os.makedirs(folder_name, exist_ok=True)

        self.optim = optim
        self.criterion = criterion
        self.generation = 0
        self.path_name = f"{folder_name}/agent_gen_{self.generation}.pt"

    def get_losses(self, preds, targets):
        return self.criterion(preds, targets)

    def train_step(self, sample):
        with torch.autograd.set_detect_anomaly(True):
            preds, targets = sample
            loss = self.get_losses(preds, targets)
            loss.backward()
            self.optim.step()

            self.generation += 1
            self.update_path_name()

    def update_path_name(self):
        folder_name = "agent_nets"
        self.path_name = f"{folder_name}/agent_gen_{self.generation}.pt"
