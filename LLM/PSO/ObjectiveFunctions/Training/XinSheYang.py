import numpy as np

from LLM.PSO.ObjectiveFunctions.Training.ObjectiveFunction import ObjectiveFunction


class XinSheYang1Function(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (-5, 5)
        self.epsilon = np.random.uniform(0, 1, dim)

    def evaluate(self, x: np.ndarray) -> float:
        return np.sum([self.epsilon[j] * np.abs(x[j]) ** (j + 1) for j in range(len(x))])


class XinSheYang2Function(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (-2 * np.pi, 2 * np.pi)

    def evaluate(self, x: np.ndarray) -> float:
        abs_sum = np.sum(np.abs(x))
        sin_sq_sum = np.sum(np.sin(x ** 2))
        return abs_sum * np.exp(-sin_sq_sum)
