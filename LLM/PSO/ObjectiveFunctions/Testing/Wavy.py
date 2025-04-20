import numpy as np

from LLM.PSO.ObjectiveFunctions.Training.ObjectiveFunction import ObjectiveFunction


class WavyFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30, k=10):
        super().__init__(dim, num_particles)
        self.bounds = (-np.pi, np.pi)
        self.k = k

    def evaluate(self, x: np.ndarray) -> float:
        return 1 - (1 / len(x)) * np.sum([np.cos(self.k * xj) * np.exp(-(xj ** 2) / 2) for xj in x])
