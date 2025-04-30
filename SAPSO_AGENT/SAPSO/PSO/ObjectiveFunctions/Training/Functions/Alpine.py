# --- Rastrigin Function Implementation ---
import numpy as np

from SAPSO_AGENT.SAPSO.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class AlpineFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.dim = dim
        self.num_particles = num_particles
        self.bounds = (-10, 10)  # Default bounds


    def evaluate(self, x: np.ndarray) -> float:
        return np.sum(np.abs(x * np.sin(x) + 0.1 * x))

    # --- Method for AlpineFunction ---
    # Add to PSO-ToyBox/SAPSO_AGENT/PSO/ObjectiveFunctions/Training/Alpine.py
    def evaluate_matrix(self, x_matrix: np.ndarray) -> np.ndarray:
        """ Vectorized evaluation for Alpine N. 1 function. """
        # x_matrix shape: (num_particles, dim)
        term = np.abs(x_matrix * np.sin(x_matrix) + 0.1 * x_matrix)  # Shape: (num_particles, dim)
        fitness_values = np.sum(term, axis=1)  # Shape: (num_particles,)
        return fitness_values
