import numpy as np

from LLM.PSO.ObjectiveFunctions.Training.ObjectiveFunction import ObjectiveFunction


class VincentFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (0.25, 10)

    def evaluate(self, x: np.ndarray) -> float:
        return -np.sum(np.sin(10 * np.log(x)))
