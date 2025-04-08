import numpy as np

from LLM.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class ExponentialFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (-1, 1)

    def evaluate(self, x: np.ndarray) -> float:
        return -np.exp(-0.5 * np.sum(x ** 2))
