# --- Rastrigin Function Implementation ---
import numpy as np

from LLM.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class AlpineFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.dim = dim
        self.num_particles = num_particles
        self.bounds = (-10, 10)  # Default bounds


    def evaluate(self, x: np.ndarray) -> float:
        return np.sum(np.abs(x * np.sin(x) + 0.1 * x))