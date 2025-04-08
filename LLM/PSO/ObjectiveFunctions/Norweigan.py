import numpy as np

from LLM.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class NorwegianFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (-1.1, 1.1)

    def evaluate(self, x: np.ndarray) -> float:
        return np.prod(np.cos(np.pi * x ** 3) * ((99 + x) / 100))
