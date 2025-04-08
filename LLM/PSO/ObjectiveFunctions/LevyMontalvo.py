import numpy as np

from LLM.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class LevyMontalvo2Function(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (-5, 5)

    def evaluate(self, x: np.ndarray) -> float:
        term1 = 0.1 * np.sin(3 * np.pi * x[0]) ** 2
        term2 = np.sum((x[:-1] - 1) ** 2 * (np.sin(3 * np.pi * x[1:]) ** 2 + 1))
        term3 = (x[-1] - 1) ** 2 * (np.sin(2 * np.pi * x[-1]) ** 2 + 1)
        return term1 + 0.1 * term2 + 0.1 * term3
