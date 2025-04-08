import numpy as np

from LLM.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class BohachevskyFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (-15, 15)  # Correct bounds from the definition

    def evaluate(self, x: np.ndarray) -> float:
        total = 0.0
        for j in range(len(x) - 1):
            xj = x[j]
            xjp1 = x[j + 1]
            term = (
                xj ** 2 +
                2 * xjp1 ** 2 -
                0.3 * np.cos(3 * np.pi * xj) -
                0.4 * np.cos(4 * np.pi * xjp1) +
                0.7
            )
            total += term
        return total