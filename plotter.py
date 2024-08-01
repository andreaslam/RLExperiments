import matplotlib.pyplot as plt


class SimulationReturnPlotter:
    def __init__(self):
        self.datapoints = {}
        self.labels = []
        self.xs = {}

    def register_datapoint(self, datapoint, label, x=None):
        if label not in self.labels:
            self.labels.append(label)
            self.datapoints[label] = []
            self.xs[label] = []

        self.datapoints[label].append(datapoint)

        if x is not None:
            self.xs[label].append(x)
        else:
            self.xs[label].append(len(self.datapoints[label]) - 1)

    def plot(self):
        for label in self.labels:
            plt.plot(self.xs[label], self.datapoints[label], label=label)
        plt.xlabel("Timestep")
        plt.ylabel("Return")
        plt.legend()
        plt.savefig("agent_returns_plot.png")
        plt.show()
