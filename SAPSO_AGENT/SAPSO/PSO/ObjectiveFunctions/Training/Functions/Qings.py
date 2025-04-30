import numpy as np

from SAPSO_AGENT.SAPSO.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class QingsFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (-500, 500)

    def evaluate(self, x: np.ndarray) -> float:
        return np.sum([(x[j] ** 2 - (j + 1)) ** 2 for j in range(len(x))])

    # --- Method for QingsFunction ---
    # Add to PSO-ToyBox/SAPSO_AGENT/PSO/ObjectiveFunctions/Training/Qings.py
    def evaluate_matrix(self, x_matrix: np.ndarray) -> np.ndarray:
        """ Vectorized evaluation for Qing's function. """
        # x_matrix shape: (num_particles, dim)
        num_particles, dim = x_matrix.shape
        if dim != self.dim:
            pass  # Handle dim mismatch

        j_indices = np.arange(1, self.dim + 1)  # Shape: (dim,)
        # Calculate term (x_j^2 - j)^2 element-wise
        term = (x_matrix ** 2 - j_indices) ** 2  # Shape: (num_particles, dim)

        # Sum across dimensions for each particle
        fitness_values = np.sum(term, axis=1)  # Shape: (num_particles,)
        return fitness_values