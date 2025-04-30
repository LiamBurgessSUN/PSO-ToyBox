import numpy as np

from SAPSO_AGENT.SAPSO.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class TrigonometricFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (0, np.pi)

    def evaluate(self, x: np.ndarray) -> float:
        n = len(x)
        term1 = np.sum([n - np.cos(xj) for xj in x])
        term2 = np.sum([i * (1 - np.cos(x[j]) - np.sin(x[j])) for i, j in enumerate(range(n), 1)])
        return (term1 + term2) ** 2

    # --- Method for TrigonometricFunction ---
    # Add to PSO-ToyBox/SAPSO_AGENT/PSO/ObjectiveFunctions/Training/Trigonometric.py
    def evaluate_matrix(self, x_matrix: np.ndarray) -> np.ndarray:
        """ Vectorized evaluation for Trigonometric function. """
        # x_matrix shape: (num_particles, dim)
        num_particles, dim = x_matrix.shape
        if dim != self.dim:
            pass  # Handle dim mismatch
        n = self.dim

        # Term 1 part: sum_{j=1}^{n} (n - cos(xj))
        term1_inner = n - np.cos(x_matrix)  # Shape: (num_particles, dim)
        term1 = np.sum(term1_inner, axis=1)  # Shape: (num_particles,)

        # Term 2 part: sum_{j=1}^{n} i * (1 - cos(xj) - sin(xj))
        # Note: Original code uses i=1 to n for the multiplier, j=0 to n-1 for index
        i_indices = np.arange(1, n + 1)  # Shape: (dim,)
        term2_inner = i_indices * (1.0 - np.cos(x_matrix) - np.sin(x_matrix))  # Shape: (num_particles, dim)
        term2 = np.sum(term2_inner, axis=1)  # Shape: (num_particles,)

        # Combine and square
        fitness_values = (term1 + term2) ** 2
        return fitness_values  # Shape: (num_particles,)