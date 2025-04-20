import numpy as np

from LLM.PSO.ObjectiveFunctions.Training.ObjectiveFunction import ObjectiveFunction


class SalomonFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (-100, 100)

    def evaluate(self, x: np.ndarray) -> float:
        sum_sq = np.sum(x ** 2)
        return -np.cos(2 * np.pi * np.sqrt(sum_sq)) + 0.1 * np.sqrt(sum_sq) + 1
