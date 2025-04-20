import numpy as np

from LLM.PSO.ObjectiveFunctions.Training.ObjectiveFunction import ObjectiveFunction


class BrownFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (-1, 1)

    def evaluate(self, x: np.ndarray) -> float:
        return np.sum((x[:-1]**2)**(x[1:]**2 + 1) + (x[1:]**2)**(x[:-1]**2 + 1))
