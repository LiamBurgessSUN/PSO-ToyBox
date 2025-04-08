import numpy as np

from LLM.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class EggCrateFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (-5, 5)

    def evaluate(self, x: np.ndarray) -> float:
        return np.sum(x ** 2) + 24 * np.sum(np.sin(x) ** 2)
