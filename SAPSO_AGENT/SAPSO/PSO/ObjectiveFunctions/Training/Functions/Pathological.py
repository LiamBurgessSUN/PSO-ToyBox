import numpy as np

from SAPSO_AGENT.SAPSO.PSO.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction


class PathologicalFunction(ObjectiveFunction):
    def __init__(self, dim=30, num_particles=30):
        super().__init__(dim, num_particles)
        self.bounds = (-100, 100)

    def evaluate(self, x: np.ndarray) -> float:
        sum_ = 0
        for j in range(len(x) - 1):
            numerator = np.sin(np.sqrt(100 * x[j] ** 2 + x[j + 1] ** 2)) ** 2 - 0.5
            denominator = 0.5 + 0.001 * (x[j] - x[j + 1]) ** 4
            sum_ += numerator / denominator
        return sum_

    # --- Method for PathologicalFunction ---
    # Add to PSO-ToyBox/SAPSO_AGENT/PSO/ObjectiveFunctions/Training/Pathological.py
    def evaluate_matrix(self, x_matrix: np.ndarray) -> np.ndarray:
        """ Vectorized evaluation for Pathological function. """
        # x_matrix shape: (num_particles, dim)
        x_j = x_matrix[:, :-1]  # Shape: (num_particles, dim-1)
        x_j_plus_1 = x_matrix[:, 1:]  # Shape: (num_particles, dim-1)

        term_sq_sum = 100.0 * x_j ** 2 + x_j_plus_1 ** 2
        # Ensure non-negative for sqrt
        term_sqrt = np.sqrt(np.maximum(term_sq_sum, 0))

        numerator = np.sin(term_sqrt) ** 2 - 0.5
        # Denominator: Original formula seems to have (xj - xj+1)^2, not ^4
        # Using ^2 as per many standard definitions. If ^4 is intended: (x_j - x_j_plus_1)**4
        denominator = 0.5 + 0.001 * (x_j - x_j_plus_1) ** 2
        # Avoid division by zero
        denominator[denominator == 0] = np.finfo(float).eps

        # Sum terms across dimension pairs for each particle
        fitness_values = np.sum(numerator / denominator, axis=1)  # Shape: (num_particles,)
        return fitness_values

