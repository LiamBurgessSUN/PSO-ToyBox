import numpy as np

from LLM.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class QingsFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (-500, 500)

    def evaluate(self, x: np.ndarray) -> float:
        return np.sum([(x[j] ** 2 - (j + 1)) ** 2 for j in range(len(x))])
