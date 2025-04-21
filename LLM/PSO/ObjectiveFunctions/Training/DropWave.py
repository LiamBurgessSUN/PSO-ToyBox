import numpy as np

from LLM.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class DropWaveFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (-5.12, 5.12)

    def evaluate(self, x: np.ndarray) -> float:
        sum_sq = np.sum(x ** 2)
        numerator = 1 + np.cos(12 * np.sqrt(sum_sq))
        denominator = 2 + 0.5 * sum_sq
        return -numerator / denominator
