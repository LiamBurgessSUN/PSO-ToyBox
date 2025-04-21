import numpy as np

from LLM.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class Ripple25Function(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (0, 1)

    def evaluate(self, x: np.ndarray) -> float:
        return np.sum([
            -np.exp(-2 * np.log(2) * ((xj - 0.1) / 0.8) ** 2) * (np.sin(5 * np.pi * xj) ** 6)
            for xj in x
        ])
