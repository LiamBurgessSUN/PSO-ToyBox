import numpy as np

from SAPSO_AGENT.SAPSO.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class DeflectedCorrugatedSpringFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30, alpha=5, K=5):
        super().__init__(dim, num_particles)
        self.bounds = (0, 2 * alpha)
        self.alpha = alpha
        self.K = K

    def evaluate(self, x: np.ndarray) -> float:
        alpha = self.alpha
        K = self.K
        total = 0
        for j in range(self.dim):
            shifted = x[j] - alpha
            outer_sum = np.sum((x - alpha) ** 2)
            total += (shifted ** 2 - np.cos(K * np.sqrt(outer_sum)))
        return 0.1 * total

    # --- Method for DeflectedCorrugatedSpringFunction ---
    # Add to PSO-ToyBox/SAPSO_AGENT/PSO/ObjectiveFunctions/Training/DeflectedCorrugatedSpring.py
    def evaluate_matrix(self, x_matrix: np.ndarray) -> np.ndarray:
        """ Vectorized evaluation for Deflected Corrugated Spring function. """
        # x_matrix shape: (num_particles, dim)
        shifted_matrix = x_matrix - self.alpha  # Shape: (num_particles, dim)
        shifted_sq = shifted_matrix ** 2  # Shape: (num_particles, dim)

        # Sum of squares for the sqrt term, calculated once per particle
        outer_sum_sq = np.sum(shifted_sq, axis=1)  # Shape: (num_particles,)
        # Ensure non-negative for sqrt
        outer_sum_sq_sqrt = np.sqrt(np.maximum(outer_sum_sq, 0))  # Shape: (num_particles,)

        # Calculate the term inside the main sum for all particles/dimensions
        # Need to broadcast outer_sum_sq_sqrt back for the calculation
        term = shifted_sq - np.cos(self.K * outer_sum_sq_sqrt[:, np.newaxis])  # Shape: (num_particles, dim)

        # Sum across dimensions for each particle
        fitness_values = 0.1 * np.sum(term, axis=1)  # Shape: (num_particles,)
        return fitness_values