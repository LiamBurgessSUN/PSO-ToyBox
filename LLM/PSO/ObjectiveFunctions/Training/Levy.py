import numpy as np

from LLM.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class Levy3Function(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (-10, 10)

    def evaluate(self, x: np.ndarray) -> float:
        y = 1 + (x - 1) / 4
        term1 = np.sin(np.pi * y[0]) ** 2
        term2 = np.sum((y[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * y[1:]) ** 2))
        term3 = (y[-1] - 1) ** 2
        return term1 + term2 + term3
