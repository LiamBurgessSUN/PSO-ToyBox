# --- Rastrigin Function Implementation ---
import numpy as np

from SAPSO_AGENT.SAPSO.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class RastriginFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.dim = dim
        self.num_particles = num_particles
        self.bounds = (-5.12, 5.12)  # Default bounds


    def evaluate(self, x: np.ndarray) -> float:
        return 10 * self.dim + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))

    # --- Method for RastriginFunction ---
    # Add to PSO-ToyBox/SAPSO_AGENT/PSO/ObjectiveFunctions/Training/Rastrgin.py (Note filename typo)
    def evaluate_matrix(self, x_matrix: np.ndarray) -> np.ndarray:
        """ Vectorized evaluation for Rastrigin function. """
        # x_matrix shape: (num_particles, dim)
        num_particles, dim = x_matrix.shape
        if dim != self.dim:
            pass  # Handle dim mismatch

        term1 = np.sum(x_matrix ** 2, axis=1)  # Shape: (num_particles,)
        term2 = 10.0 * np.sum(np.cos(2 * np.pi * x_matrix), axis=1)  # Shape: (num_particles,)

        fitness_values = 10.0 * self.dim + term1 - term2
        return fitness_values  # Shape: (num_particles,)