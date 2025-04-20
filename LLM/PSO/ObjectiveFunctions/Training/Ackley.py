# --- Rastrigin Function Implementation ---
import numpy as np

from LLM.PSO.ObjectiveFunctions.Training.ObjectiveFunction import ObjectiveFunction


class AckleyFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.dim = dim
        self.num_particles = num_particles
        self.bounds = (-32, 32)  # Default bounds


    def evaluate(self, x: np.ndarray) -> float:
        return -20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / self.dim)) - np.exp(np.sum(np.cos(2 * np.pi * x)) / self.dim) + 20 + np.e