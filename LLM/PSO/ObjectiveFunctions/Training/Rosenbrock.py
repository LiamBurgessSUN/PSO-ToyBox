import numpy as np

from LLM.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class RosenbrockFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (-30, 30)

    def evaluate(self, x: np.ndarray) -> float:
        return np.sum([100 * (x[j + 1] - x[j] ** 2) ** 2 + (x[j] - 1) ** 2 for j in range(len(x) - 1)])
