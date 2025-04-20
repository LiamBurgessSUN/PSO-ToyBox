import numpy as np

from LLM.PSO.ObjectiveFunctions.Training.ObjectiveFunction import ObjectiveFunction


class WeierstrassFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30, a=0.5, b=3, j_max=20):
        super().__init__(dim, num_particles)
        self.a = a
        self.b = b
        self.j_max = j_max
        self.bounds = (-0.5, 0.5)

    def evaluate(self, x: np.ndarray) -> float:
        n = len(x)
        term1 = np.sum([
            np.sum([self.a ** j * np.cos(2 * np.pi * self.b ** j * (xj + 0.5)) for j in range(self.j_max + 1)])
            for xj in x
        ])
        term2 = n * np.sum([self.a ** j * np.cos(np.pi * self.b ** j) for j in range(self.j_max + 1)])
        return term1 - term2
