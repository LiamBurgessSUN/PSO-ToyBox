import numpy as np

from LLM.PSO.ObjectiveFunctions.Training.ObjectiveFunction import ObjectiveFunction


class PinterFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (-10, 10)

    def evaluate(self, x: np.ndarray) -> float:
        n = len(x)
        sum1 = np.sum([(j + 1) * x[j] ** 2 for j in range(n)])
        sum2 = np.sum([20 * (j + 1) * np.sin((x[j - 1] * np.sin(x[j]) + np.sin(x[j + 1])) ** 2)
                       for j in range(n)])
        sum3 = np.sum([(j + 1) * np.log10(1 + (j + 1) * (
            (x[j - 1] ** 2 - 2 * x[j] + 3 * x[(j + 1) % n] - np.cos(x[j]) + 1) ** 2))
                       for j in range(n)])
        return sum1 + sum2 + sum3
