import numpy as np

from LLM.PSO.ObjectiveFunctions.Training.ObjectiveFunction import ObjectiveFunction


class SineEnvelopeFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (-100, 100)

    def evaluate(self, x: np.ndarray) -> float:
        return np.sum([
            0.5 + (np.sin(np.sqrt(x[j] ** 2 + x[j + 1] ** 2)) ** 2 - 0.5) /
            (1 + 0.001 * (x[j] ** 2 + x[j + 1] ** 2)) ** 2
            for j in range(len(x) - 1)
        ])
