import numpy as np

from LLM.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class Lanczos3Function(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (-20, 20)

    def sinc(self, x: np.ndarray) -> np.ndarray:
        return np.sinc(x / np.pi)

    def evaluate(self, x: np.ndarray) -> float:
        return np.prod(self.sinc(x) * self.sinc(x / 3))
