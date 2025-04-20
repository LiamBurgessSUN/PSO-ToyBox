import numpy as np

from LLM.PSO.ObjectiveFunctions.Training.ObjectiveFunction import ObjectiveFunction


class Schwefel1Function(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (-100, 100)
        self.alpha = np.sqrt(np.pi)

    def evaluate(self, x: np.ndarray) -> float:
        return np.sum(x ** 2) ** self.alpha
