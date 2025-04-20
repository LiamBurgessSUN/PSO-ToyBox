import numpy as np

from LLM.PSO.ObjectiveFunctions.Training.ObjectiveFunction import ObjectiveFunction


class TrigonometricFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (0, np.pi)

    def evaluate(self, x: np.ndarray) -> float:
        n = len(x)
        term1 = np.sum([n - np.cos(xj) for xj in x])
        term2 = np.sum([i * (1 - np.cos(x[j]) - np.sin(x[j])) for i, j in enumerate(range(n), 1)])
        return (term1 + term2) ** 2
