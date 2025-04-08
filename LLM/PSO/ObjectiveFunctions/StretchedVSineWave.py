import numpy as np

from LLM.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class StretchedVSineWaveFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (-10, 10)

    def evaluate(self, x: np.ndarray) -> float:
        return np.sum([
            (x[j] ** 2 + x[j + 1] ** 2) ** 0.25 *
            (np.sin(50 * (x[j] ** 2 + x[j + 1] ** 2) ** 0.1) ** 2 + 0.1)
            for j in range(len(x) - 1)
        ])
