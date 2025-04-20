import numpy as np

from LLM.PSO.ObjectiveFunctions.Training.ObjectiveFunction import ObjectiveFunction


class QuinticFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (-10, 10)

    def evaluate(self, x: np.ndarray) -> float:
        return np.sum(np.abs(x ** 5 - 3 * x ** 4 + 4 * x ** 3 + 2 * x ** 2 - 10 * x - 4))
