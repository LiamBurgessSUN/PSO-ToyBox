import numpy as np

from SAPSO_AGENT.SAPSO.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class WavyFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30, k=10):
        super().__init__(dim, num_particles)
        self.bounds = (-np.pi, np.pi)
        self.k = k

    def evaluate(self, x: np.ndarray) -> float:
        return 1 - (1 / len(x)) * np.sum([np.cos(self.k * xj) * np.exp(-(xj ** 2) / 2) for xj in x])

    # --- Method for WavyFunction ---
    # Add to PSO-ToyBox/SAPSO_AGENT/PSO/ObjectiveFunctions/Testing/Wavy.py
    def evaluate_matrix(self, x_matrix: np.ndarray) -> np.ndarray:
        """ Vectorized evaluation for Wavy function. """
        # x_matrix shape: (num_particles, dim)
        num_particles, dim = x_matrix.shape
        if dim != self.dim:
            # Handle dimension mismatch if necessary
            pass

        # Calculate term inside sum for all particles and dimensions
        term = np.cos(self.k * x_matrix) * np.exp(-(x_matrix ** 2) / 2.0)  # Shape: (num_particles, dim)

        # Sum across dimensions for each particle
        sum_term = np.sum(term, axis=1)  # Shape: (num_particles,)

        fitness_values = 1.0 - (1.0 / self.dim) * sum_term
        return fitness_values