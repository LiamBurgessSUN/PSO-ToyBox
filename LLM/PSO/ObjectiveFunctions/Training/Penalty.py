import numpy as np

from LLM.PSO.ObjectiveFunctions.Training.ObjectiveFunction import ObjectiveFunction


class Penalty1Function(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30, a=10, k=100, m=4):
        super().__init__(dim, num_particles)
        self.bounds = (-50, 50)
        self.a = a
        self.k = k
        self.m = m

    def u(self, xj):
        if xj > self.a:
            return self.k * (xj - self.a) ** self.m
        elif xj < -self.a:
            return self.k * (-xj - self.a) ** self.m
        return 0

    def evaluate(self, x: np.ndarray) -> float:
        y = 1 + 0.25 * (x + 1)
        term1 = 10 * np.sin(np.pi * y[0]) ** 2
        term2 = np.sum((y[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * y[1:]) ** 2))
        term3 = (y[-1] - 1) ** 2
        penalty = np.sum([self.u(xj) for xj in x])
        return (np.pi / self.dim) * (term1 + term2 + term3) + penalty

class Penalty2Function(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30, a=5, k=100, m=4):
        super().__init__(dim, num_particles)
        self.bounds = (-50, 50)
        self.a = a
        self.k = k
        self.m = m

    def u(self, xj):
        if xj > self.a:
            return self.k * (xj - self.a) ** self.m
        elif xj < -self.a:
            return self.k * (-xj - self.a) ** self.m
        return 0

    def evaluate(self, x: np.ndarray) -> float:
        term1 = 0.1 * np.sin(3 * np.pi * x[0]) ** 2
        term2 = np.sum((x[:-1] - 1) ** 2 * (1 + np.sin(3 * np.pi * x[1:]) ** 2))
        term3 = (x[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * x[-1]) ** 2)
        penalty = np.sum([self.u(xj) for xj in x])
        return 0.1 * (term1 + term2 + term3) + penalty
