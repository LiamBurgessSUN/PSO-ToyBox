import numpy as np

from LLM.PSO.ObjectiveFunctions.Training.ObjectiveFunction import ObjectiveFunction


class HolderTable1Function(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (-10, 10)

    def evaluate(self, x: np.ndarray) -> float:
        prod_cos = np.prod(np.cos(x))
        sum_sq = np.sum(x ** 2)
        return -np.abs(prod_cos * np.exp(np.abs(1 - sum_sq ** 0.5 / np.pi)))
