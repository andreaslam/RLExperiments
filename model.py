import torch.nn as nn


class AgentNet(nn.Module):
    def __init__(
        self, num_input_nodes, num_output_nodes, num_hidden_nodes=64, num_layers=1
    ):
        super().__init__()
        self.flatten = nn.Flatten()
        self.hidden_layers = []
        for _ in range(num_layers):
            self.hidden_layers.append(nn.Linear(num_hidden_nodes, num_hidden_nodes))
            nn.ReLU()
        self.policy_head = nn.Sequential(
            nn.Linear(num_input_nodes, num_hidden_nodes),
            nn.ReLU(),
            *self.hidden_layers,
            nn.ReLU(),
            nn.Linear(num_hidden_nodes, num_output_nodes),
        )
        self.valueHead = nn.Sequential(
            nn.Linear(num_input_nodes, num_hidden_nodes),
            *self.hidden_layers,
            nn.ReLU(),
            nn.Linear(num_hidden_nodes, 1),  # value head output is always 1
        )

    def forward(self, x):
        x = self.flatten(x)
        policy_outputs = self.policy_head(x)
        value_output = self.valueHead(x)
        return value_output, policy_outputs
