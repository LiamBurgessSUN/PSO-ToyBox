import numpy as np

from SAPSO_AGENT.SAPSO.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class EllipticFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (-100, 100)

    def evaluate(self, x: np.ndarray) -> float:
        weights = 1e6 ** (np.arange(self.dim) / (self.dim - 1))
        return np.sum(weights * x ** 2)

    # --- Method for EllipticFunction ---
    # Add to PSO-ToyBox/SAPSO_AGENT/PSO/ObjectiveFunctions/Training/Elliptic.py
    def evaluate_matrix(self, x_matrix: np.ndarray) -> np.ndarray:
        """ Vectorized evaluation for Elliptic function. """
        # x_matrix shape: (num_particles, dim)
        num_particles, dim = x_matrix.shape
        if dim != self.dim:
            # Handle dimension mismatch if necessary
            pass

        # Calculate weights (shape: (dim,))
        weights = 1e6 ** (np.arange(self.dim) / (self.dim - 1))

        # Calculate weighted sum of squares for each particle
        # weights will broadcast to (num_particles, dim)
        term = weights * x_matrix ** 2  # Shape: (num_particles, dim)
        fitness_values = np.sum(term, axis=1)  # Shape: (num_particles,)
        return fitness_values