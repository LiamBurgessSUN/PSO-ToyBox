import numpy as np

from SAPSO_AGENT.SAPSO.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class Mishra1Function(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (0, 1)

    def evaluate(self, x: np.ndarray) -> float:
        sum_x = np.sum(x[:-1])
        base = 1 + self.dim - sum_x
        return base * (self.dim - sum_x)

    # --- Method for Mishra1Function ---
    # Add to PSO-ToyBox/SAPSO_AGENT/PSO/ObjectiveFunctions/Training/Mishra.py
    def evaluate_matrix(self, x_matrix: np.ndarray) -> np.ndarray:
        """ Vectorized evaluation for Mishra N. 1 function. """
        # x_matrix shape: (num_particles, dim)
        num_particles, dim = x_matrix.shape
        if dim != self.dim:
            pass  # Handle dim mismatch if needed

        sum_x_except_last = np.sum(x_matrix[:, :-1], axis=1)  # Shape: (num_particles,)
        n = self.dim  # Use self.dim as n

        base = 1.0 + n - sum_x_except_last  # Shape: (num_particles,)
        # The formula seems to be (1 + n - sum(x_i)) ^ (n - sum(x_i))
        # Let's clarify the exponent term. Assuming it's (n - sum(x_i))
        exponent = n - sum_x_except_last  # Shape: (num_particles,)

        # Handle potential issues with base <= 0 if exponent is not an integer
        # Since bounds are (0, 1), sum_x_except_last < dim, so base > 1, exponent > 0
        # Power calculation should be safe.
        fitness_values = base ** exponent  # Original file had base * exponent? Reverting to likely intended power.
        # If original file intended multiplication: fitness_values = base * exponent

        # Reverting to multiplication as per original file:
        fitness_values = base * exponent

        return fitness_values  # Shape: (num_particles,)

class Mishra4Function(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (-10, 10)

    def evaluate(self, x: np.ndarray) -> float:
        sum_sq = np.sum(x ** 2)
        return np.sqrt(np.abs(np.sin(np.sqrt(sum_sq)))) + 0.01 * np.sum(x)

    # --- Method for Mishra4Function ---
    # Add to PSO-ToyBox/SAPSO_AGENT/PSO/ObjectiveFunctions/Training/Mishra.py
    def evaluate_matrix(self, x_matrix: np.ndarray) -> np.ndarray:
        """ Vectorized evaluation for Mishra N. 4 function. """
        # x_matrix shape: (num_particles, dim)
        sum_sq = np.sum(x_matrix ** 2, axis=1)  # Shape: (num_particles,)
        sum_x = np.sum(x_matrix, axis=1)  # Shape: (num_particles,)

        # Ensure non-negative for sqrt
        term1 = np.sqrt(np.abs(np.sin(np.sqrt(np.maximum(sum_sq, 0)))))  # Shape: (num_particles,)
        term2 = 0.01 * sum_x  # Shape: (num_particles,)

        fitness_values = term1 + term2
        return fitness_values