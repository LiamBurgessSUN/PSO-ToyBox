import numpy as np

from LLM.PSO.ObjectiveFunctions.Training.ObjectiveFunction import ObjectiveFunction


class MichalewiczFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30, m=10):
        super().__init__(dim, num_particles)
        self.bounds = (0, np.pi)
        self.m = m

    def evaluate(self, x: np.ndarray) -> float:
        j = np.arange(1, self.dim + 1)
        return -np.sum(np.sin(x) * (np.sin(j * x ** 2 / np.pi)) ** (2 * self.m))
