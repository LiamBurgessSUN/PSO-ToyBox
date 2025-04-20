import numpy as np

from LLM.PSO.ObjectiveFunctions.Training.ObjectiveFunction import ObjectiveFunction


class PathologicalFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (-100, 100)

    def evaluate(self, x: np.ndarray) -> float:
        sum_ = 0
        for j in range(len(x) - 1):
            numerator = np.sin(np.sqrt(100 * x[j] ** 2 + x[j + 1] ** 2)) ** 2 - 0.5
            denominator = 0.5 + 0.001 * (x[j] - x[j + 1]) ** 4
            sum_ += numerator / denominator
        return sum_
