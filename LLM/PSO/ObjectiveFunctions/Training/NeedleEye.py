import numpy as np

from LLM.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class NeedleEyeFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30, eye=0.0001):
        super().__init__(dim, num_particles)
        self.bounds = (-10, 10)
        self.eye = eye

    def evaluate(self, x: np.ndarray) -> float:
        if np.all(np.abs(x) < self.eye):
            return 1
        elif np.all(np.abs(x) <= self.eye):
            return np.sum(100 + np.abs(x))
        else:
            return 0
