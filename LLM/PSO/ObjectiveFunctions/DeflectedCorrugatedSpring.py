import numpy as np

from LLM.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class DeflectedCorrugatedSpringFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30, alpha=5, K=5):
        super().__init__(dim, num_particles)
        self.bounds = (0, 2 * alpha)
        self.alpha = alpha
        self.K = K

    def evaluate(self, x: np.ndarray) -> float:
        alpha = self.alpha
        K = self.K
        total = 0
        for j in range(self.dim):
            shifted = x[j] - alpha
            outer_sum = np.sum((x - alpha) ** 2)
            total += (shifted ** 2 - np.cos(K * np.sqrt(outer_sum)))
        return 0.1 * total
