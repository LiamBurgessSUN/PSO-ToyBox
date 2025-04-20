import numpy as np

from LLM.PSO.ObjectiveFunctions.Training.ObjectiveFunction import ObjectiveFunction


class Schubert4Function(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (-10, 10)

    def evaluate(self, x: np.ndarray) -> float:
        return np.sum([
            np.sum([(j + 1) * np.cos((j + 1) * xj + j) for j in range(5)])
            for xj in x
        ])
