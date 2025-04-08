import numpy as np

from LLM.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class TridFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (-20, 20)

    def evaluate(self, x: np.ndarray) -> float:
        return np.sum((x - 1) ** 2) - np.sum(x[1:] * x[:-1])
