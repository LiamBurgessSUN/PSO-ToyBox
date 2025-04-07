# --- Rastrigin Function Implementation ---
import numpy as np

from LLM.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class RastriginFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.dim = dim
        self.num_particles = num_particles
        self.bounds = (-5.12, 5.12)  # Default bounds


    def evaluate(self, x: np.ndarray) -> float:
        return 10 * self.dim + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))