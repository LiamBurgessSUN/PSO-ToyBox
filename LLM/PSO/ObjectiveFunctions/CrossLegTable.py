import numpy as np

from LLM.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class CrossLegTableFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (-10, 10)

    def evaluate(self, x: np.ndarray) -> float:
        norm = np.sqrt(np.sum(x ** 2))
        sin_prod = np.abs(np.prod(np.sin(x)))
        denominator = np.abs(np.exp(np.abs(100 - norm / np.pi)) * (sin_prod + 1))
        return 1 / (denominator ** 0.1)
