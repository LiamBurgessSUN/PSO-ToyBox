import numpy as np

from LLM.PSO.ObjectiveFunctions.Training.ObjectiveFunction import ObjectiveFunction


class SinusoidalFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30, A=1, B=1, z=0):
        super().__init__(dim, num_particles)
        self.bounds = (0, 180)
        self.A = A
        self.B = B
        self.z = z

    def evaluate(self, x: np.ndarray) -> float:
        part1 = np.prod(np.sin(x - self.z))
        part2 = np.prod(np.sin(self.B * (x - self.z)))
        return -self.A * (part1 + part2)
