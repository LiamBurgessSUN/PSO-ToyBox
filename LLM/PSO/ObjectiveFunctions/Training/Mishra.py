import numpy as np

from LLM.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class Mishra1Function(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (0, 1)

    def evaluate(self, x: np.ndarray) -> float:
        sum_x = np.sum(x[:-1])
        base = 1 + self.dim - sum_x
        return base * (self.dim - sum_x)

class Mishra4Function(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (-10, 10)

    def evaluate(self, x: np.ndarray) -> float:
        sum_sq = np.sum(x ** 2)
        return np.sqrt(np.abs(np.sin(np.sqrt(sum_sq)))) + 0.01 * np.sum(x)
