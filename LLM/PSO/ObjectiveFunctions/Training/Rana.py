import numpy as np

from LLM.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class RanaFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (-500, 500)

    def evaluate(self, x: np.ndarray) -> float:
        sum_ = 0
        for j in range(len(x) - 1):
            xj = x[j]
            xj1 = x[j + 1]
            t1 = np.sqrt(np.abs(xj1 + xj + 1))
            t2 = np.sqrt(np.abs(xj1 - xj + 1))
            sum_ += (xj1 + 1) * np.cos(t2) * np.sin(t1) + xj * np.cos(t1) * np.sin(t2)
        return sum_
