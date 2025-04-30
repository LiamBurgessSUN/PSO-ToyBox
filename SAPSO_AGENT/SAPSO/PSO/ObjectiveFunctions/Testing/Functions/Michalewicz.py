import numpy as np

from SAPSO_AGENT.SAPSO.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class MichalewiczFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30, m=10):
        super().__init__(dim, num_particles)
        self.bounds = (0, np.pi)
        self.m = m

    def evaluate(self, x: np.ndarray) -> float:
        j = np.arange(1, self.dim + 1)
        return -np.sum(np.sin(x) * (np.sin(j * x ** 2 / np.pi)) ** (2 * self.m))

    # --- Method for MichalewiczFunction ---
    # Add to PSO-ToyBox/SAPSO_AGENT/PSO/ObjectiveFunctions/Testing/Michalewicz.py
    def evaluate_matrix(self, x_matrix: np.ndarray) -> np.ndarray:
        """ Vectorized evaluation for Michalewicz function. """
        # x_matrix shape: (num_particles, dim)
        num_particles, dim = x_matrix.shape
        if dim != self.dim:
            # Handle dimension mismatch if necessary, or assume self.dim is correct
            pass

        j = np.arange(1, self.dim + 1)  # Shape: (dim,)
        # Calculate term inside sum for all particles and dimensions
        # x_matrix: (num_particles, dim)
        # j: (dim,) - broadcasts correctly
        term = np.sin(x_matrix) * (np.sin(j * x_matrix ** 2 / np.pi)) ** (2 * self.m)  # Shape: (num_particles, dim)

        # Sum across dimensions for each particle
        fitness_values = -np.sum(term, axis=1)  # Shape: (num_particles,)
        return fitness_values