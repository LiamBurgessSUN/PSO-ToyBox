import numpy as np

from LLM.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class BonyadiMichalewiczFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (-5, 5)

    def evaluate(self, x: np.ndarray) -> float:
        numerator = np.prod(x + 1)
        denominator = np.prod((x + 1) ** 2 + 1)
        return numerator / denominator
